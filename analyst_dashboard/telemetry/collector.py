"""Corporate Events Local Telemetry Collector & Persistence Simulation.

Provides in-memory or ephemeral SQLite persistence for:
1. Raw telemetry event audit stream (raw_corporate_event_telemetry).
2. Authoritative deduplicated evaluation ledger (corporate_event_evaluations).
3. Dead-letter queue for malformed/prohibited payloads (telemetry_dead_letter).

Enforces:
- Atomicity and race-safe idempotency via SQLite WAL and UNIQUE constraints.
- Non-negotiable privacy boundaries (zero prohibited fields persisted to evaluation store).
- Mathematical denominator accounting breakdown ($N_raw, N_valid, N_dedup, N_natural, N_synthetic, N_other$).
- Configurable failure injection to verify retry and self-healing recovery.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pydantic import ValidationError

from .corporate_event_contract import (
    CorporateEventTelemetryPayload,
    PROHIBITED_PRIVACY_FIELDS,
)
from .provenance import (
    ProvenanceClass,
    ProvenanceClassifier,
    ClassificationResult,
    ObservationEpochBounds,
)

logger = logging.getLogger("analyst_dashboard.telemetry.collector")


class IngestionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE_ACKNOWLEDGED = "DUPLICATE_ACKNOWLEDGED"
    REJECTED_MALFORMED = "REJECTED_MALFORMED"
    REJECTED_PRIVACY_VIOLATION = "REJECTED_PRIVACY_VIOLATION"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"


@dataclass
class IngestionResult:
    """Outcome of an ingestion attempt."""

    status: IngestionStatus
    event_id: Optional[str]
    deduplication_key: Optional[str]
    provenance_class: Optional[ProvenanceClass]
    is_denominator_eligible: bool
    rejection_reason: Optional[str] = None
    delivery_attempt_id: Optional[str] = None


@dataclass
class TelemetryAccounting:
    """Exact denominator and telemetry flow accounting."""

    n_raw: int
    n_valid: int
    n_malformed: int
    n_dedup: int
    n_duplicate: int
    n_natural: int
    n_synthetic: int
    n_other: int

    def verify_invariants(self) -> Tuple[bool, List[str]]:
        """Verify the mathematical accounting identities."""
        errors = []
        if self.n_raw != (self.n_valid + self.n_malformed):
            errors.append(
                f"Raw invariant failed: n_raw ({self.n_raw}) != n_valid ({self.n_valid}) + n_malformed ({self.n_malformed})"
            )
        if self.n_valid != (self.n_dedup + self.n_duplicate):
            errors.append(
                f"Deduplication invariant failed: n_valid ({self.n_valid}) != n_dedup ({self.n_dedup}) + n_duplicate ({self.n_duplicate})"
            )
        if self.n_dedup != (self.n_natural + self.n_synthetic + self.n_other):
            errors.append(
                f"Classification invariant failed: n_dedup ({self.n_dedup}) != n_natural ({self.n_natural}) + n_synthetic ({self.n_synthetic}) + n_other ({self.n_other})"
            )
        return len(errors) == 0, errors


class TelemetryCollector:
    """Thread-safe SQLite collector simulating edge telemetry ingestion."""

    def __init__(
        self,
        db_path: str = ":memory:",
        classifier: Optional[ProvenanceClassifier] = None,
        auto_init: bool = True,
    ):
        self.db_path = db_path
        self.classifier = classifier or ProvenanceClassifier()
        self._simulated_failure_mode: Optional[str] = None
        self._lock = threading.RLock()

        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False, timeout=10.0)
            self._shared_conn.execute("PRAGMA busy_timeout = 5000;")
            self._shared_conn.row_factory = sqlite3.Row
        else:
            self._shared_conn = None

        if auto_init:
            self._init_tables()

    def get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create the raw audit table, evaluations table, and dead letter queue."""
        with self._lock:
            conn = self.get_connection()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_corporate_event_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    delivery_attempt_id TEXT,
                    event_id TEXT,
                    evaluation_context_id TEXT,
                    deduplication_key TEXT,
                    symbol TEXT,
                    occurred_at_utc TEXT,
                    collector_received_at_utc TEXT,
                    provenance_class TEXT,
                    applied_penalty REAL,
                    risk_bucket TEXT,
                    raw_payload TEXT,
                    headers_json TEXT,
                    ingestion_status TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'utc'))
                );

                CREATE INDEX IF NOT EXISTS idx_raw_dedup_key ON raw_corporate_event_telemetry (deduplication_key);
                CREATE INDEX IF NOT EXISTS idx_raw_event_id ON raw_corporate_event_telemetry (event_id);

                CREATE TABLE IF NOT EXISTS corporate_event_evaluations (
                    deduplication_key TEXT PRIMARY KEY,
                    event_id TEXT,
                    evaluation_context_id TEXT,
                    symbol TEXT,
                    occurred_at_utc TEXT,
                    collector_received_at_utc TEXT,
                    provenance_class TEXT,
                    days_to_earnings REAL,
                    risk_bucket TEXT,
                    applied_penalty REAL,
                    release_sha TEXT,
                    ruleset_version TEXT,
                    is_natural_eligible INTEGER,
                    created_at TEXT DEFAULT (datetime('now', 'utc'))
                );

                CREATE INDEX IF NOT EXISTS idx_eval_provenance ON corporate_event_evaluations (provenance_class);

                CREATE TABLE IF NOT EXISTS telemetry_dead_letter (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_payload TEXT,
                    headers_json TEXT,
                    rejection_reason TEXT,
                    prohibited_fields TEXT,
                    received_at_utc TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'utc'))
                );
                """
            )
            conn.commit()
            if self._shared_conn is None:
                conn.close()

    def set_simulated_failure(self, failure_mode: Optional[str]) -> None:
        """Inject simulated failure for reliability and self-healing testing.
        Supported modes: 'STORAGE_WRITE_FAILURE', 'TIMEOUT'.
        """
        self._simulated_failure_mode = failure_mode

    def ingest(
        self,
        raw_payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        collector_received_at: Optional[datetime] = None,
    ) -> IngestionResult:
        """Ingest, validate, classify, deduplicate and persist a telemetry event."""
        now_utc = collector_received_at or datetime.now(timezone.utc)
        now_utc_str = now_utc.isoformat().replace("+00:00", "Z")
        headers = headers or {}
        headers_json = json.dumps(headers)

        # 0. Check simulated failure hook
        if self._simulated_failure_mode == "STORAGE_WRITE_FAILURE":
            return IngestionResult(
                status=IngestionStatus.TRANSIENT_FAILURE,
                event_id=raw_payload.get("event_id"),
                deduplication_key=None,
                provenance_class=None,
                is_denominator_eligible=False,
                rejection_reason="Simulated storage write failure",
                delivery_attempt_id=raw_payload.get("delivery_attempt_id"),
            )
        elif self._simulated_failure_mode == "TIMEOUT":
            raise TimeoutError("Simulated edge gateway timeout")

        # 1. Check for prohibited privacy fields before any parsing
        detected_prohibited = []
        if isinstance(raw_payload, dict):
            for k in raw_payload.keys():
                norm_k = k.lower().replace("-", "_")
                if norm_k in PROHIBITED_PRIVACY_FIELDS:
                    detected_prohibited.append(k)

        if detected_prohibited:
            rejection_msg = (
                f"Privacy Boundary Violation: Prohibited field(s) detected: {detected_prohibited}. "
                f"Payload strictly rejected with 0 bytes written to evaluations ledger."
            )
            with self._lock:
                conn = self.get_connection()
                try:
                    conn.execute(
                        """
                        INSERT INTO telemetry_dead_letter (
                            raw_payload, headers_json, rejection_reason, prohibited_fields, received_at_utc
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            json.dumps({k: v for k, v in raw_payload.items() if k not in detected_prohibited}),
                            headers_json,
                            rejection_msg,
                            json.dumps(detected_prohibited),
                            now_utc_str,
                        ),
                    )
                    conn.commit()
                finally:
                    if self._shared_conn is None:
                        conn.close()

            return IngestionResult(
                status=IngestionStatus.REJECTED_PRIVACY_VIOLATION,
                event_id=raw_payload.get("event_id"),
                deduplication_key=None,
                provenance_class=ProvenanceClass.MALFORMED,
                is_denominator_eligible=False,
                rejection_reason=rejection_msg,
                delivery_attempt_id=raw_payload.get("delivery_attempt_id"),
            )

        # 2. Strict Schema Validation
        try:
            payload = CorporateEventTelemetryPayload.model_validate(raw_payload)
        except ValidationError as val_err:
            rejection_msg = f"Schema validation error: {val_err}"
            with self._lock:
                conn = self.get_connection()
                try:
                    conn.execute(
                        """
                        INSERT INTO telemetry_dead_letter (
                            raw_payload, headers_json, rejection_reason, prohibited_fields, received_at_utc
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            json.dumps(raw_payload),
                            headers_json,
                            rejection_msg,
                            "[]",
                            now_utc_str,
                        ),
                    )
                    conn.commit()
                finally:
                    if self._shared_conn is None:
                        conn.close()

            return IngestionResult(
                status=IngestionStatus.REJECTED_MALFORMED,
                event_id=raw_payload.get("event_id"),
                deduplication_key=None,
                provenance_class=ProvenanceClass.MALFORMED,
                is_denominator_eligible=False,
                rejection_reason=rejection_msg,
                delivery_attempt_id=raw_payload.get("delivery_attempt_id"),
            )

        payload.collector_received_at_utc = now_utc_str
        dedup_key = payload.deduplication_key

        with self._lock:
            conn = self.get_connection()
            try:
                # 3. Deduplication Check & Prior Receipt Lookup
                existing_eval = conn.execute(
                    "SELECT deduplication_key, provenance_class, is_natural_eligible, occurred_at_utc FROM corporate_event_evaluations WHERE deduplication_key = ?",
                    (dedup_key,),
                ).fetchone()

                prior_receipt_time = None
                if existing_eval:
                    last_raw = conn.execute(
                        "SELECT collector_received_at_utc FROM raw_corporate_event_telemetry WHERE deduplication_key = ? ORDER BY id DESC LIMIT 1",
                        (dedup_key,),
                    ).fetchone()
                    if last_raw and last_raw["collector_received_at_utc"]:
                        prior_receipt_time = self.classifier.parse_iso_utc(last_raw["collector_received_at_utc"])

                # 4. Provenance Classification
                classification = self.classifier.classify(
                    payload=payload,
                    headers=headers,
                    is_known_duplicate=(existing_eval is not None),
                    prior_receipt_time=prior_receipt_time,
                    collector_received_at=now_utc,
                )

                # 5. Persistence
                if existing_eval:
                    # Log retransmission to raw stream
                    conn.execute(
                        """
                        INSERT INTO raw_corporate_event_telemetry (
                            delivery_attempt_id, event_id, evaluation_context_id, deduplication_key,
                            symbol, occurred_at_utc, collector_received_at_utc, provenance_class,
                            applied_penalty, risk_bucket, raw_payload, headers_json, ingestion_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload.delivery_attempt_id,
                            payload.event_id,
                            payload.evaluation_context_id,
                            dedup_key,
                            payload.symbol,
                            payload.occurred_at_utc,
                            now_utc_str,
                            classification.provenance_class.value,
                            payload.applied_penalty,
                            payload.risk_bucket.value,
                            json.dumps(payload.model_dump()),
                            headers_json,
                            IngestionStatus.DUPLICATE_ACKNOWLEDGED.value,
                        ),
                    )
                    conn.commit()

                    return IngestionResult(
                        status=IngestionStatus.DUPLICATE_ACKNOWLEDGED,
                        event_id=payload.event_id,
                        deduplication_key=dedup_key,
                        provenance_class=classification.provenance_class,
                        is_denominator_eligible=False,
                        rejection_reason="Duplicate evaluation acknowledged; ignored for denominator",
                        delivery_attempt_id=payload.delivery_attempt_id,
                    )

                # Unique evaluation: insert into both raw and evaluations tables
                try:
                    conn.execute(
                        """
                        INSERT INTO raw_corporate_event_telemetry (
                            delivery_attempt_id, event_id, evaluation_context_id, deduplication_key,
                            symbol, occurred_at_utc, collector_received_at_utc, provenance_class,
                            applied_penalty, risk_bucket, raw_payload, headers_json, ingestion_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload.delivery_attempt_id,
                            payload.event_id,
                            payload.evaluation_context_id,
                            dedup_key,
                            payload.symbol,
                            payload.occurred_at_utc,
                            now_utc_str,
                            classification.provenance_class.value,
                            payload.applied_penalty,
                            payload.risk_bucket.value,
                            json.dumps(payload.model_dump()),
                            headers_json,
                            IngestionStatus.ACCEPTED.value,
                        ),
                    )

                    conn.execute(
                        """
                        INSERT INTO corporate_event_evaluations (
                            deduplication_key, event_id, evaluation_context_id, symbol,
                            occurred_at_utc, collector_received_at_utc, provenance_class,
                            days_to_earnings, risk_bucket, applied_penalty, release_sha,
                            ruleset_version, is_natural_eligible
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            dedup_key,
                            payload.event_id,
                            payload.evaluation_context_id,
                            payload.symbol,
                            payload.occurred_at_utc,
                            now_utc_str,
                            classification.provenance_class.value,
                            payload.days_to_earnings,
                            payload.risk_bucket.value,
                            payload.applied_penalty,
                            payload.release_sha,
                            payload.ruleset_version,
                            1 if classification.is_natural_denominator_eligible else 0,
                        ),
                    )
                    conn.commit()

                    return IngestionResult(
                        status=IngestionStatus.ACCEPTED,
                        event_id=payload.event_id,
                        deduplication_key=dedup_key,
                        provenance_class=classification.provenance_class,
                        is_denominator_eligible=classification.is_natural_denominator_eligible,
                        delivery_attempt_id=payload.delivery_attempt_id,
                    )

                except sqlite3.IntegrityError:
                    # Race condition caught: safely acknowledge duplicate
                    conn.rollback()
                    conn.execute(
                        """
                        INSERT INTO raw_corporate_event_telemetry (
                            delivery_attempt_id, event_id, evaluation_context_id, deduplication_key,
                            symbol, occurred_at_utc, collector_received_at_utc, provenance_class,
                            applied_penalty, risk_bucket, raw_payload, headers_json, ingestion_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload.delivery_attempt_id,
                            payload.event_id,
                            payload.evaluation_context_id,
                            dedup_key,
                            payload.symbol,
                            payload.occurred_at_utc,
                            now_utc_str,
                            ProvenanceClass.DUPLICATE.value,
                            payload.applied_penalty,
                            payload.risk_bucket.value,
                            json.dumps(payload.model_dump()),
                            headers_json,
                            IngestionStatus.DUPLICATE_ACKNOWLEDGED.value,
                        ),
                    )
                    conn.commit()

                    return IngestionResult(
                        status=IngestionStatus.DUPLICATE_ACKNOWLEDGED,
                        event_id=payload.event_id,
                        deduplication_key=dedup_key,
                        provenance_class=ProvenanceClass.DUPLICATE,
                        is_denominator_eligible=False,
                        rejection_reason="Concurrent duplicate race condition caught; safely acknowledged",
                        delivery_attempt_id=payload.delivery_attempt_id,
                    )
            finally:
                if self._shared_conn is None:
                    conn.close()

    def get_natural_denominator(self) -> int:
        """Returns the exact count of unique NATURAL_PRODUCTION_ELIGIBLE evaluations."""
        with self._lock:
            conn = self.get_connection()
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT deduplication_key) as cnt
                    FROM corporate_event_evaluations
                    WHERE provenance_class = ? AND is_natural_eligible = 1
                    """,
                    (ProvenanceClass.NATURAL_PRODUCTION_ELIGIBLE.value,),
                ).fetchone()
                return int(row["cnt"]) if row else 0
            finally:
                if self._shared_conn is None:
                    conn.close()

    def get_accounting_breakdown(self) -> TelemetryAccounting:
        """Computes the full denominator accounting breakdown across all streams."""
        with self._lock:
            conn = self.get_connection()
            try:
                # Dead letter (malformed / rejected)
                row_malformed = conn.execute(
                    "SELECT COUNT(*) as cnt FROM telemetry_dead_letter"
                ).fetchone()
                n_malformed = int(row_malformed["cnt"]) if row_malformed else 0

                # Raw valid entries
                row_raw_valid = conn.execute(
                    "SELECT COUNT(*) as cnt FROM raw_corporate_event_telemetry"
                ).fetchone()
                n_valid = int(row_raw_valid["cnt"]) if row_raw_valid else 0

                n_raw = n_valid + n_malformed

                # Duplicates in raw log
                row_duplicates = conn.execute(
                    """
                    SELECT COUNT(*) as cnt FROM raw_corporate_event_telemetry
                    WHERE ingestion_status = ?
                    """,
                    (IngestionStatus.DUPLICATE_ACKNOWLEDGED.value,),
                ).fetchone()
                n_duplicate = int(row_duplicates["cnt"]) if row_duplicates else 0

                # Deduplicated evaluations
                row_dedup = conn.execute(
                    "SELECT COUNT(*) as cnt FROM corporate_event_evaluations"
                ).fetchone()
                n_dedup = int(row_dedup["cnt"]) if row_dedup else 0

                # Provenance breakdown
                row_natural = conn.execute(
                    """
                    SELECT COUNT(*) as cnt FROM corporate_event_evaluations
                    WHERE provenance_class = ?
                    """,
                    (ProvenanceClass.NATURAL_PRODUCTION_ELIGIBLE.value,),
                ).fetchone()
                n_natural = int(row_natural["cnt"]) if row_natural else 0

                row_synthetic = conn.execute(
                    """
                    SELECT COUNT(*) as cnt FROM corporate_event_evaluations
                    WHERE provenance_class IN (?, ?, ?)
                    """,
                    (
                        ProvenanceClass.SYNTHETIC_DIAGNOSTIC.value,
                        ProvenanceClass.AUTOMATED_SMOKE_TEST.value,
                        ProvenanceClass.MANUAL_QA.value,
                    ),
                ).fetchone()
                n_synthetic = int(row_synthetic["cnt"]) if row_synthetic else 0

                row_other = conn.execute(
                    """
                    SELECT COUNT(*) as cnt FROM corporate_event_evaluations
                    WHERE provenance_class NOT IN (?, ?, ?, ?)
                    """,
                    (
                        ProvenanceClass.NATURAL_PRODUCTION_ELIGIBLE.value,
                        ProvenanceClass.SYNTHETIC_DIAGNOSTIC.value,
                        ProvenanceClass.AUTOMATED_SMOKE_TEST.value,
                        ProvenanceClass.MANUAL_QA.value,
                    ),
                ).fetchone()
                n_other = int(row_other["cnt"]) if row_other else 0

                return TelemetryAccounting(
                    n_raw=n_raw,
                    n_valid=n_valid,
                    n_malformed=n_malformed,
                    n_dedup=n_dedup,
                    n_duplicate=n_duplicate,
                    n_natural=n_natural,
                    n_synthetic=n_synthetic,
                    n_other=n_other,
                )
            finally:
                if self._shared_conn is None:
                    conn.close()
