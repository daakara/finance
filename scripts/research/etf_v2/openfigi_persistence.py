"""
scripts/research/etf_v2/openfigi_persistence.py

Operational SQLite repository for OpenFIGI Symbology Corroboration Engine.
Physically and logically separated from canonical population store.

Invariants Enforced:
- OFIGI-INV-001: Operational evidence only; cannot modify canonical population.
- OFIGI-INV-003: One-way data flow: Canonical -> OpenFIGI -> Operational Store.
- OFIGI-INV-012: Replay and retry are idempotent; duplicate observations do not corrupt active state.
- OFIGI-INV-014: Operational evidence stored strictly in operational database; never in canonical tables.
- OFIGI-INV-016: Canonical database, backup, and snapshot remain byte-identical.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
import sqlite3
from typing import Generator, List, Optional, Sequence, Tuple

from .openfigi_config import (
    DEFAULT_OPERATIONAL_DB_PATH,
    resolve_openfigi_operational_db_path,
)
from .openfigi_models import OpenFIGIActiveMapping, OpenFIGIObservation

logger = logging.getLogger(__name__)

SCHEMA_SQL_PATH = Path(__file__).parent / "openfigi_operational_schema.sql"


class OpenFIGIPersistenceError(Exception):
    """Raised when operational persistence operations fail."""
    pass


class CanonicalStoreContaminationError(OpenFIGIPersistenceError):
    """Raised if an operational persistence component is configured with a canonical database path."""
    pass


class OpenFIGIPersistenceRepository:
    """ACID SQLite repository managing openfigi_observations and openfigi_active_mappings."""

    def __init__(self, db_path: Optional[Path | str] = None, auto_init: bool = True):
        self.db_path = resolve_openfigi_operational_db_path(db_path)

        # Strict Canonical Firewall: Refuse to initialize against canonical database paths
        resolved_str = str(self.db_path.resolve()).replace("\\", "/").lower()
        if "data/canonical" in resolved_str or "canonical_population" in resolved_str:
            raise CanonicalStoreContaminationError(
                f"OFIGI-INV-014 VIOLATION: Operational persistence cannot be initialized with "
                f"canonical database path: {self.db_path}"
            )

        if auto_init and not self.db_path.name.startswith(":memory:"):
            self.initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextlib.contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Executes the DDL schema if not already present."""
        if not SCHEMA_SQL_PATH.exists():
            raise OpenFIGIPersistenceError(f"Operational schema file not found at {SCHEMA_SQL_PATH}")
        schema_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        with self.connection() as conn:
            with conn:
                conn.executescript(schema_sql)
                # Ensure schema_version is set to 1
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM schema_version WHERE version = 1;")
                if cur.fetchone()[0] == 0:
                    conn.execute(
                        "INSERT INTO schema_version (version, applied_at) VALUES (1, CURRENT_TIMESTAMP);"
                    )

    def persist_observation_and_projection(
        self,
        observation: OpenFIGIObservation,
        projection: Optional[OpenFIGIActiveMapping] = None
    ) -> None:
        """
        Atomically appends an observation and optionally upserts the active projection.
        Guarantees that a projection update is never committed without its observation.
        """
        self.persist_batch([(observation, projection)])

    def persist_batch(
        self,
        records: Sequence[Tuple[OpenFIGIObservation, Optional[OpenFIGIActiveMapping]]]
    ) -> None:
        """Atomically persists a batch of observations and active mapping projections."""
        with self.connection() as conn:
            try:
                with conn:  # BEGIN IMMEDIATE transaction
                    for obs, proj in records:
                        # 1. Insert authoritative observation
                        conn.execute(
                            """
                            INSERT INTO openfigi_observations (
                                observation_id, execution_id, correlation_id, canonical_internal_id,
                                isin, idempotency_key, source_population_version, source_snapshot_sha256,
                                contract_version, request_position, request_filters_json, attempt_count,
                                retry_count, http_status, outcome_class, normalized_result_json,
                                provider_response_evidence_json, provider_response_digest, observed_at,
                                created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                obs.observation_id, obs.execution_id, obs.correlation_id,
                                obs.canonical_internal_id, obs.isin, obs.idempotency_key,
                                obs.source_population_version, obs.source_snapshot_sha256,
                                obs.contract_version, obs.request_position,
                                json.dumps(obs.request_filters), obs.attempt_count,
                                obs.retry_count, obs.http_status, obs.outcome_class,
                                json.dumps(obs.normalized_result),
                                json.dumps(obs.provider_response_evidence),
                                obs.provider_response_digest, obs.observed_at, obs.created_at
                            )
                        )

                        # 2. Upsert active mapping projection if provided
                        if proj is not None:
                            conn.execute(
                                """
                                INSERT INTO openfigi_active_mappings (
                                    isin, canonical_internal_id, figi, composite_figi,
                                    share_class_figi, ticker, exch_code, security_type,
                                    market_sector, name, outcome_class, last_observation_id,
                                    source_population_version, source_snapshot_sha256,
                                    contract_version, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(isin) DO UPDATE SET
                                    canonical_internal_id=excluded.canonical_internal_id,
                                    figi=excluded.figi,
                                    composite_figi=excluded.composite_figi,
                                    share_class_figi=excluded.share_class_figi,
                                    ticker=excluded.ticker,
                                    exch_code=excluded.exch_code,
                                    security_type=excluded.security_type,
                                    market_sector=excluded.market_sector,
                                    name=excluded.name,
                                    outcome_class=excluded.outcome_class,
                                    last_observation_id=excluded.last_observation_id,
                                    source_population_version=excluded.source_population_version,
                                    source_snapshot_sha256=excluded.source_snapshot_sha256,
                                    contract_version=excluded.contract_version,
                                    updated_at=excluded.updated_at
                                """,
                                (
                                    proj.isin, proj.canonical_internal_id, proj.figi,
                                    proj.composite_figi, proj.share_class_figi, proj.ticker,
                                    proj.exch_code, proj.security_type, proj.market_sector,
                                    proj.name, proj.outcome_class, proj.last_observation_id,
                                    proj.source_population_version, proj.source_snapshot_sha256,
                                    proj.contract_version, proj.updated_at
                                )
                            )
            except sqlite3.OperationalError as e:
                raise OpenFIGIPersistenceError(f"Operational persistence transaction failed: {str(e)}") from e

    def get_active_mapping(self, isin: str) -> Optional[OpenFIGIActiveMapping]:
        """Queries the current deduplicated operational projection for an ISIN."""
        clean_isin = isin.strip().upper()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM openfigi_active_mappings WHERE isin = ?;", (clean_isin,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return OpenFIGIActiveMapping(**dict(row))

    def list_observations(self, isin: Optional[str] = None) -> List[OpenFIGIObservation]:
        """Queries authoritative observation history, optionally filtered by ISIN."""
        with self.connection() as conn:
            if isin:
                clean_isin = isin.strip().upper()
                cur = conn.execute(
                    "SELECT * FROM openfigi_observations WHERE isin = ? ORDER BY request_position ASC, created_at ASC;",
                    (clean_isin,)
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM openfigi_observations ORDER BY created_at ASC;"
                )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["request_filters"] = json.loads(d.pop("request_filters_json"))
                d["normalized_result"] = json.loads(d.pop("normalized_result_json"))
                d["provider_response_evidence"] = json.loads(d.pop("provider_response_evidence_json"))
                results.append(OpenFIGIObservation(**d))
            return results

    def rebuild_active_projection(self) -> int:
        """
        Reconstructs the openfigi_active_mappings table completely from authoritative
        openfigi_observations history. Proves projection is derived state.
        """
        with self.connection() as conn:
            with conn:
                conn.execute("DELETE FROM openfigi_active_mappings;")
                cur = conn.execute(
                    """
                    SELECT * FROM openfigi_observations
                    WHERE outcome_class IN ('EXACT_OPERATIONAL_CORROBORATION', 'AMBIGUOUS_OPERATIONAL_CORROBORATION')
                    ORDER BY created_at ASC;
                    """
                )
                rebuilt_count = 0
                for row in cur.fetchall():
                    norm = json.loads(row["normalized_result_json"])
                    conn.execute(
                        """
                        INSERT INTO openfigi_active_mappings (
                            isin, canonical_internal_id, figi, composite_figi,
                            share_class_figi, ticker, exch_code, security_type,
                            market_sector, name, outcome_class, last_observation_id,
                            source_population_version, source_snapshot_sha256,
                            contract_version, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(isin) DO UPDATE SET
                            canonical_internal_id=excluded.canonical_internal_id,
                            figi=excluded.figi,
                            composite_figi=excluded.composite_figi,
                            share_class_figi=excluded.share_class_figi,
                            ticker=excluded.ticker,
                            exch_code=excluded.exch_code,
                            security_type=excluded.security_type,
                            market_sector=excluded.market_sector,
                            name=excluded.name,
                            outcome_class=excluded.outcome_class,
                            last_observation_id=excluded.last_observation_id,
                            source_population_version=excluded.source_population_version,
                            source_snapshot_sha256=excluded.source_snapshot_sha256,
                            contract_version=excluded.contract_version,
                            updated_at=excluded.updated_at;
                        """,
                        (
                            row["isin"], row["canonical_internal_id"], norm.get("figi"),
                            norm.get("composite_figi"), norm.get("share_class_figi"),
                            norm.get("ticker"), norm.get("exch_code"), norm.get("security_type"),
                            norm.get("market_sector"), norm.get("name"), row["outcome_class"],
                            row["observation_id"], row["source_population_version"],
                            row["source_snapshot_sha256"], row["contract_version"],
                            row["created_at"]
                        )
                    )
                    rebuilt_count += 1
                return rebuilt_count
