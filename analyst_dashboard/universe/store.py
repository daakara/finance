"""
analyst_dashboard/universe/store.py

Persistent SQLite Store for ARX Universe Builds & Ledgers.
Enforces:
- Schema-level immutability via SQLite BEFORE UPDATE and BEFORE DELETE triggers.
- WAL mode, busy timeout, and atomic multi-table transaction commits.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from .contracts import (
    SourcePopulationSnapshot,
    EligibilityLedgerRow,
    DataReadinessLedgerRow,
    UniverseBuildAttestation,
    ConstructionStatus,
    PublicationDecision,
    EligibilityDecision,
    DataReadinessDecision,
)

logger = logging.getLogger("arx.universe.store")

DATA_DIR = os.getenv("FINANCE_DATA_DIR", os.getenv("DATA_DIR", os.path.expanduser("~")))
os.makedirs(DATA_DIR, exist_ok=True)
UNIVERSE_DB_PATH = os.path.join(DATA_DIR, ".finance_universe_store.db")


class UniverseStore:
    """
    Persistent store for source population snapshots, universe builds, and ledgers.
    """

    def __init__(self, db_path: str = UNIVERSE_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initializes tables and triggers for immutable universe persistence."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()

                # 1. Source Population Snapshots Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS universe_source_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        source_authority TEXT NOT NULL,
                        as_of TEXT NOT NULL,
                        count INTEGER NOT NULL,
                        source_hash TEXT NOT NULL,
                        securities_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)

                # 2. Universe Builds Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS universe_builds (
                        universe_build_id TEXT PRIMARY KEY,
                        universe_id TEXT NOT NULL,
                        universe_version TEXT NOT NULL,
                        source_population_authority TEXT NOT NULL,
                        source_population_snapshot_id TEXT NOT NULL,
                        source_population_as_of TEXT NOT NULL,
                        source_population_count INTEGER NOT NULL,
                        source_population_hash TEXT NOT NULL,
                        eligibility_rule_version TEXT NOT NULL,
                        universe_definition_hash TEXT NOT NULL,
                        normalization_version TEXT NOT NULL,
                        normalization_hash TEXT NOT NULL,
                        eligibility_input_snapshot_id TEXT NOT NULL,
                        eligibility_input_hash TEXT NOT NULL,
                        eligible_count INTEGER NOT NULL,
                        ineligible_count INTEGER NOT NULL,
                        eligibility_unresolved_count INTEGER NOT NULL,
                        data_ready_count INTEGER NOT NULL,
                        data_unavailable_count INTEGER NOT NULL,
                        data_unresolved_count INTEGER NOT NULL,
                        scannable_count INTEGER NOT NULL,
                        eligible_membership_hash TEXT NOT NULL,
                        scannable_membership_hash TEXT NOT NULL,
                        per_security_decision_hash TEXT NOT NULL,
                        readiness_decision_hash TEXT NOT NULL,
                        construction_status TEXT NOT NULL,
                        publication_decision TEXT NOT NULL,
                        generated_at TEXT NOT NULL,
                        implementation_release_sha TEXT NOT NULL,
                        reconciliation_summary_json TEXT
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ubuild_pub ON universe_builds(universe_id, publication_decision)")

                # 3. Per-Security Eligibility Ledger
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS universe_eligibility_ledger (
                        universe_build_id TEXT NOT NULL,
                        security_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        exchange TEXT NOT NULL,
                        source_record_hash TEXT NOT NULL,
                        normalization_version TEXT NOT NULL,
                        normalized_security_hash TEXT NOT NULL,
                        normalization_status TEXT NOT NULL,
                        universe_version TEXT NOT NULL,
                        eligibility_rule_version TEXT NOT NULL,
                        universe_definition_hash TEXT NOT NULL,
                        eligibility_decision TEXT NOT NULL,
                        eligibility_reason_code TEXT NOT NULL,
                        eligibility_input_hash TEXT NOT NULL,
                        rule_evaluation_hash TEXT NOT NULL,
                        decision_hash TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        implementation_release_sha TEXT NOT NULL,
                        rule_results_json TEXT NOT NULL,
                        PRIMARY KEY (universe_build_id, symbol)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_el_sym ON universe_eligibility_ledger(symbol)")

                # 4. Per-Security Readiness Ledger
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS universe_readiness_ledger (
                        universe_build_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        required_input_role TEXT NOT NULL,
                        data_authority TEXT NOT NULL,
                        data_as_of TEXT,
                        freshness_status TEXT NOT NULL,
                        completeness_status TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        readiness_result TEXT NOT NULL,
                        readiness_reason_code TEXT NOT NULL,
                        readiness_hash TEXT NOT NULL,
                        candle_count INTEGER NOT NULL,
                        has_live_price INTEGER NOT NULL,
                        PRIMARY KEY (universe_build_id, symbol)
                    )
                """)

                # 5. Immutability Triggers
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_ubuild_no_update
                    BEFORE UPDATE ON universe_builds
                    BEGIN
                        SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_builds records are immutable and cannot be updated.');
                    END;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_ubuild_no_delete
                    BEFORE DELETE ON universe_builds
                    BEGIN
                        SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_builds records are immutable and cannot be deleted.');
                    END;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_eligibility_no_update
                    BEFORE UPDATE ON universe_eligibility_ledger
                    BEGIN
                        SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_eligibility_ledger records are immutable and cannot be updated.');
                    END;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_eligibility_no_delete
                    BEFORE DELETE ON universe_eligibility_ledger
                    BEGIN
                        SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_eligibility_ledger records are immutable and cannot be deleted.');
                    END;
                """)
        finally:
            conn.close()

    def save_source_snapshot(self, snapshot: SourcePopulationSnapshot) -> None:
        """Persists source population snapshot."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO universe_source_snapshots (
                        snapshot_id, source_authority, as_of, count, source_hash, securities_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.snapshot_id,
                    snapshot.source_authority,
                    snapshot.as_of,
                    snapshot.count,
                    snapshot.source_hash,
                    json.dumps([s.to_dict() for s in snapshot.securities]),
                    snapshot.as_of,
                ))
        finally:
            conn.close()

    def save_universe_build(
        self,
        attestation: UniverseBuildAttestation,
        eligibility_rows: List[EligibilityLedgerRow],
        readiness_rows: List[DataReadinessLedgerRow],
    ) -> None:
        """Persists universe build and its corresponding ledger rows in one transaction."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO universe_builds (
                        universe_build_id, universe_id, universe_version,
                        source_population_authority, source_population_snapshot_id,
                        source_population_as_of, source_population_count, source_population_hash,
                        eligibility_rule_version, universe_definition_hash,
                        normalization_version, normalization_hash,
                        eligibility_input_snapshot_id, eligibility_input_hash,
                        eligible_count, ineligible_count, eligibility_unresolved_count,
                        data_ready_count, data_unavailable_count, data_unresolved_count,
                        scannable_count, eligible_membership_hash, scannable_membership_hash,
                        per_security_decision_hash, readiness_decision_hash,
                        construction_status, publication_decision, generated_at,
                        implementation_release_sha, reconciliation_summary_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    attestation.universe_build_id,
                    attestation.universe_id,
                    attestation.universe_version,
                    attestation.source_population_authority,
                    attestation.source_population_snapshot_id,
                    attestation.source_population_as_of,
                    attestation.source_population_count,
                    attestation.source_population_hash,
                    attestation.eligibility_rule_version,
                    attestation.universe_definition_hash,
                    attestation.normalization_version,
                    attestation.normalization_hash,
                    attestation.eligibility_input_snapshot_id,
                    attestation.eligibility_input_hash,
                    attestation.eligible_count,
                    attestation.ineligible_count,
                    attestation.eligibility_unresolved_count,
                    attestation.data_ready_count,
                    attestation.data_unavailable_count,
                    attestation.data_unresolved_count,
                    attestation.scannable_count,
                    attestation.eligible_membership_hash,
                    attestation.scannable_membership_hash,
                    attestation.per_security_decision_hash,
                    attestation.readiness_decision_hash,
                    attestation.construction_status.value,
                    attestation.publication_decision.value,
                    attestation.generated_at,
                    attestation.implementation_release_sha,
                    json.dumps(attestation.reconciliation_summary or {}),
                ))

                for er in eligibility_rows:
                    cursor.execute("""
                        INSERT INTO universe_eligibility_ledger (
                            universe_build_id, security_id, symbol, exchange,
                            source_record_hash, normalization_version,
                            normalized_security_hash, normalization_status,
                            universe_version, eligibility_rule_version,
                            universe_definition_hash, eligibility_decision,
                            eligibility_reason_code, eligibility_input_hash,
                            rule_evaluation_hash, decision_hash, observed_at,
                            implementation_release_sha, rule_results_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        er.universe_build_id,
                        er.security_id,
                        er.symbol,
                        er.exchange,
                        er.source_record_hash,
                        er.normalization_version,
                        er.normalized_security_hash,
                        er.normalization_status,
                        er.universe_version,
                        er.eligibility_rule_version,
                        er.universe_definition_hash,
                        er.eligibility_decision.value,
                        er.eligibility_reason_code,
                        er.eligibility_input_hash,
                        er.rule_evaluation_hash,
                        er.decision_hash,
                        er.observed_at,
                        er.implementation_release_sha,
                        json.dumps(er.rule_results),
                    ))

                for rr in readiness_rows:
                    cursor.execute("""
                        INSERT INTO universe_readiness_ledger (
                            universe_build_id, symbol, required_input_role,
                            data_authority, data_as_of, freshness_status,
                            completeness_status, content_hash, readiness_result,
                            readiness_reason_code, readiness_hash, candle_count,
                            has_live_price
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rr.universe_build_id,
                        rr.symbol,
                        rr.required_input_role,
                        rr.data_authority,
                        rr.data_as_of,
                        rr.freshness_status,
                        rr.completeness_status,
                        rr.content_hash,
                        rr.readiness_result.value,
                        rr.readiness_reason_code,
                        rr.readiness_hash,
                        rr.candle_count,
                        1 if rr.has_live_price else 0,
                    ))
        finally:
            conn.close()

    def get_latest_published_universe(self, universe_id: str = "ARX_US_EQUITIES") -> Optional[Dict[str, Any]]:
        """Retrieves latest published and complete universe build."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM universe_builds
                WHERE universe_id = ? AND publication_decision = 'PUBLISH' AND construction_status = 'COMPLETE'
                ORDER BY rowid DESC LIMIT 1
            """, (universe_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_scannable_membership(self, universe_build_id: str) -> List[str]:
        """Returns sorted list of symbols marked DATA_READY for the universe build."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM universe_readiness_ledger
                WHERE universe_build_id = ? AND readiness_result = 'DATA_READY'
                ORDER BY symbol ASC
            """, (universe_build_id,))
            return [r[0] for r in cursor.fetchall()]
        finally:
            conn.close()
