"""
Finalization Ledger & Attempt Ledger Architecture (Sections 12, 14, 15).
Implements append-only event ledger separate from the run transition ledger,
cryptographic hash-chaining anchored to PARENT_RUN_FINAL_HASH, ACID transaction atomicity,
and attempt logging.
"""

import sqlite3
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.anchor_finalization.models import (
    FinalizationEvent,
    AnchorAttemptRecord,
    FinalizationState,
    AuthorityEffectiveStatus,
    ErrorClass,
)


class FinalizationLedgerStore:
    """
    ACID-compliant append-only store for the Finalization Ledger and Attempt Ledger.
    Maintains a serializable coordinator authority for finalization sequences.
    """

    GENESIS_PREVIOUS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS finalization_events (
                    finalization_event_id TEXT PRIMARY KEY,
                    parent_run_id TEXT NOT NULL,
                    parent_run_final_hash TEXT NOT NULL,
                    finalization_sequence INTEGER NOT NULL,
                    finalization_event_type TEXT NOT NULL,
                    anchor_request_id TEXT NOT NULL,
                    anchor_attempt_id TEXT NOT NULL,
                    anchor_status_before TEXT NOT NULL,
                    anchor_status_after TEXT NOT NULL,
                    authority_effective_status_before TEXT NOT NULL,
                    authority_effective_status_after TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    previous_finalization_event_hash TEXT NOT NULL,
                    finalization_event_hash TEXT NOT NULL,
                    UNIQUE(parent_run_id, finalization_sequence)
                );

                CREATE TABLE IF NOT EXISTS anchor_attempts (
                    anchor_attempt_id TEXT PRIMARY KEY,
                    anchor_request_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    anchor_idempotency_key TEXT NOT NULL,
                    request_payload_sha256 TEXT NOT NULL,
                    request_started_at TEXT NOT NULL,
                    request_completed_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    provider_http_status INTEGER NOT NULL,
                    provider_response_sha256 TEXT NOT NULL,
                    error_class TEXT NOT NULL,
                    result TEXT NOT NULL,
                    lookup_required INTEGER NOT NULL,
                    lookup_performed INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS coordinator_state (
                    parent_run_id TEXT PRIMARY KEY,
                    parent_run_final_hash TEXT NOT NULL,
                    current_anchor_status TEXT NOT NULL,
                    current_authority_effective_status TEXT NOT NULL,
                    current_sequence INTEGER NOT NULL,
                    last_event_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    def initialize_run_coordinator(
        self,
        parent_run_id: str,
        parent_run_final_hash: str,
        initial_anchor_status: FinalizationState = FinalizationState.A1_ANCHOR_REQUIRED,
        initial_authority_effective_status: AuthorityEffectiveStatus = AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
    ) -> None:
        """Initializes coordinator state for a run if not already present."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO coordinator_state
                (parent_run_id, parent_run_final_hash, current_anchor_status,
                 current_authority_effective_status, current_sequence, last_event_hash, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    parent_run_id,
                    parent_run_final_hash,
                    initial_anchor_status.value,
                    initial_authority_effective_status.value,
                    self.GENESIS_PREVIOUS_HASH,
                    now_utc,
                ),
            )
            conn.commit()

    def record_attempt(self, attempt: AnchorAttemptRecord) -> None:
        """Appends an attempt record to the attempt ledger (Section 12)."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO anchor_attempts
                (anchor_attempt_id, anchor_request_id, attempt_number, run_id,
                 anchor_idempotency_key, request_payload_sha256, request_started_at,
                 request_completed_at, provider, provider_http_status,
                 provider_response_sha256, error_class, result, lookup_required, lookup_performed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.anchor_attempt_id,
                    attempt.anchor_request_id,
                    attempt.attempt_number,
                    attempt.run_id,
                    attempt.anchor_idempotency_key,
                    attempt.request_payload_sha256,
                    attempt.request_started_at,
                    attempt.request_completed_at,
                    attempt.provider,
                    attempt.provider_http_status,
                    attempt.provider_response_sha256,
                    attempt.error_class.value,
                    attempt.result,
                    1 if attempt.lookup_required else 0,
                    1 if attempt.lookup_performed else 0,
                ),
            )
            conn.commit()

    def append_finalization_event(
        self,
        parent_run_id: str,
        parent_run_final_hash: str,
        finalization_event_type: str,
        anchor_request_id: str,
        anchor_attempt_id: str,
        anchor_status_to: FinalizationState,
        authority_effective_status_to: AuthorityEffectiveStatus,
        recorded_at: Optional[str] = None,
    ) -> FinalizationEvent:
        """
        Atomically updates coordinator status and appends a finalization event (Section 15).
        Enforces serializable coordination, sequence monotonicity, and cryptographic hash chaining.
        """
        if recorded_at is None:
            recorded_at = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")

            # Read current coordinator state
            cursor.execute(
                """
                SELECT parent_run_final_hash, current_anchor_status,
                       current_authority_effective_status, current_sequence, last_event_hash
                FROM coordinator_state WHERE parent_run_id = ?
                """,
                (parent_run_id,),
            )
            row = cursor.fetchone()
            if not row:
                # Initialize on the fly
                cursor.execute(
                    """
                    INSERT INTO coordinator_state
                    (parent_run_id, parent_run_final_hash, current_anchor_status,
                     current_authority_effective_status, current_sequence, last_event_hash, updated_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        parent_run_id,
                        parent_run_final_hash,
                        FinalizationState.A1_ANCHOR_REQUIRED.value,
                        AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR.value,
                        self.GENESIS_PREVIOUS_HASH,
                        recorded_at,
                    ),
                )
                prev_final_hash = parent_run_final_hash
                status_before = FinalizationState.A1_ANCHOR_REQUIRED
                eff_status_before = AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR
                next_seq = 1
                prev_event_hash = self.GENESIS_PREVIOUS_HASH
            else:
                prev_final_hash, cur_status_str, cur_eff_str, cur_seq, prev_event_hash = row
                if prev_final_hash != parent_run_final_hash:
                    raise ValueError(
                        f"CRITICAL_INTEGRITY_CONFLICT: Run final hash mismatch. "
                        f"Expected {prev_final_hash}, got {parent_run_final_hash}"
                    )
                status_before = FinalizationState(cur_status_str)
                eff_status_before = AuthorityEffectiveStatus(cur_eff_str)
                next_seq = cur_seq + 1

            event_id = f"fevt_{parent_run_id}_{next_seq:04d}"

            # Compute cryptographic hash chain
            event_hash_content = (
                f"{event_id}:"
                f"{parent_run_id}:"
                f"{parent_run_final_hash}:"
                f"{next_seq}:"
                f"{finalization_event_type}:"
                f"{anchor_request_id}:"
                f"{anchor_attempt_id}:"
                f"{status_before.value}:"
                f"{anchor_status_to.value}:"
                f"{eff_status_before.value}:"
                f"{authority_effective_status_to.value}:"
                f"{recorded_at}:"
                f"{prev_event_hash}"
            )
            event_hash = hashlib.sha256(event_hash_content.encode("utf-8")).hexdigest()

            event = FinalizationEvent(
                finalization_event_id=event_id,
                parent_run_id=parent_run_id,
                parent_run_final_hash=parent_run_final_hash,
                finalization_sequence=next_seq,
                finalization_event_type=finalization_event_type,
                anchor_request_id=anchor_request_id,
                anchor_attempt_id=anchor_attempt_id,
                anchor_status_before=status_before,
                anchor_status_after=anchor_status_to,
                authority_effective_status_before=eff_status_before,
                authority_effective_status_after=authority_effective_status_to,
                recorded_at=recorded_at,
                previous_finalization_event_hash=prev_event_hash,
                finalization_event_hash=event_hash,
            )

            # Insert event
            cursor.execute(
                """
                INSERT INTO finalization_events
                (finalization_event_id, parent_run_id, parent_run_final_hash,
                 finalization_sequence, finalization_event_type, anchor_request_id,
                 anchor_attempt_id, anchor_status_before, anchor_status_after,
                 authority_effective_status_before, authority_effective_status_after,
                 recorded_at, previous_finalization_event_hash, finalization_event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.finalization_event_id,
                    event.parent_run_id,
                    event.parent_run_final_hash,
                    event.finalization_sequence,
                    event.finalization_event_type,
                    event.anchor_request_id,
                    event.anchor_attempt_id,
                    event.anchor_status_before.value,
                    event.anchor_status_after.value,
                    event.authority_effective_status_before.value,
                    event.authority_effective_status_after.value,
                    event.recorded_at,
                    event.previous_finalization_event_hash,
                    event.finalization_event_hash,
                ),
            )

            # Update coordinator state
            cursor.execute(
                """
                UPDATE coordinator_state
                SET current_anchor_status = ?,
                    current_authority_effective_status = ?,
                    current_sequence = ?,
                    last_event_hash = ?,
                    updated_at = ?
                WHERE parent_run_id = ?
                """,
                (
                    anchor_status_to.value,
                    authority_effective_status_to.value,
                    next_seq,
                    event_hash,
                    recorded_at,
                    parent_run_id,
                ),
            )

            conn.commit()
            return event

    def get_events(self, parent_run_id: str) -> List[FinalizationEvent]:
        """Fetches all finalization events ordered by sequence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT finalization_event_id, parent_run_id, parent_run_final_hash,
                       finalization_sequence, finalization_event_type, anchor_request_id,
                       anchor_attempt_id, anchor_status_before, anchor_status_after,
                       authority_effective_status_before, authority_effective_status_after,
                       recorded_at, previous_finalization_event_hash, finalization_event_hash
                FROM finalization_events
                WHERE parent_run_id = ?
                ORDER BY finalization_sequence ASC
                """,
                (parent_run_id,),
            )
            rows = cursor.fetchall()
            events = []
            for r in rows:
                events.append(
                    FinalizationEvent(
                        finalization_event_id=r[0],
                        parent_run_id=r[1],
                        parent_run_final_hash=r[2],
                        finalization_sequence=r[3],
                        finalization_event_type=r[4],
                        anchor_request_id=r[5],
                        anchor_attempt_id=r[6],
                        anchor_status_before=FinalizationState(r[7]),
                        anchor_status_after=FinalizationState(r[8]),
                        authority_effective_status_before=AuthorityEffectiveStatus(r[9]),
                        authority_effective_status_after=AuthorityEffectiveStatus(r[10]),
                        recorded_at=r[11],
                        previous_finalization_event_hash=r[12],
                        finalization_event_hash=r[13],
                    )
                )
            return events

    def get_attempts(self, run_id: str) -> List[AnchorAttemptRecord]:
        """Fetches all anchor attempt records for a run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT anchor_attempt_id, anchor_request_id, attempt_number, run_id,
                       anchor_idempotency_key, request_payload_sha256, request_started_at,
                       request_completed_at, provider, provider_http_status,
                       provider_response_sha256, error_class, result, lookup_required, lookup_performed
                FROM anchor_attempts
                WHERE run_id = ?
                ORDER BY attempt_number ASC
                """,
                (run_id,),
            )
            rows = cursor.fetchall()
            attempts = []
            for r in rows:
                attempts.append(
                    AnchorAttemptRecord(
                        anchor_attempt_id=r[0],
                        anchor_request_id=r[1],
                        attempt_number=r[2],
                        run_id=r[3],
                        anchor_idempotency_key=r[4],
                        request_payload_sha256=r[5],
                        request_started_at=r[6],
                        request_completed_at=r[7],
                        provider=r[8],
                        provider_http_status=r[9],
                        provider_response_sha256=r[10],
                        error_class=ErrorClass(r[11]),
                        result=r[12],
                        lookup_required=bool(r[13]),
                        lookup_performed=bool(r[14]),
                    )
                )
            return attempts

    def get_current_state(
        self, parent_run_id: str
    ) -> Optional[Tuple[FinalizationState, AuthorityEffectiveStatus, int, str]]:
        """Returns (anchor_status, authority_effective_status, sequence, last_event_hash)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT current_anchor_status, current_authority_effective_status,
                       current_sequence, last_event_hash
                FROM coordinator_state WHERE parent_run_id = ?
                """,
                (parent_run_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return (
                FinalizationState(row[0]),
                AuthorityEffectiveStatus(row[1]),
                row[2],
                row[3],
            )
