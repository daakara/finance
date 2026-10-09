"""
Immutable Snapshot Store for Scanner Publication Records.

Persists completed scanner snapshots with cryptographic fingerprints,
version tuples, and immutable SQLite triggers preventing mutation or deletion.
"""

import sqlite3
import os
import json
import logging
from typing import Dict, Any, List, Optional
from analyst_dashboard.data.market_db import retry_sqlite, DATA_DIR

logger = logging.getLogger(__name__)

SCANNER_DB_PATH = os.path.join(DATA_DIR, ".finance_scanner_store.db")


class ScannerSnapshotStore:
    """Production-grade SQLite persistent store for immutable scanner snapshots."""

    def __init__(self, db_path: str = SCANNER_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    @retry_sqlite()
    def _init_schema(self):
        """Initialize immutable scanner_snapshots table and append-only triggers."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scanner_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        scanner_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        api_contract_version TEXT NOT NULL,
                        ruleset_version TEXT NOT NULL,
                        evidence_schema_version TEXT NOT NULL,
                        score_model_version TEXT NOT NULL,
                        data_provenance_version TEXT NOT NULL,
                        universe_version TEXT NOT NULL,
                        freshness_policy_version TEXT NOT NULL,
                        implementation_release_sha TEXT NOT NULL,
                        semantic_fingerprint TEXT NOT NULL,
                        generated_at TEXT NOT NULL,
                        data_as_of TEXT NOT NULL,
                        status_at_publication TEXT NOT NULL,
                        universe_id TEXT NOT NULL,
                        universe_size INTEGER NOT NULL,
                        matched_count INTEGER NOT NULL,
                        results_json TEXT NOT NULL,
                        provenance_json TEXT NOT NULL,
                        freshness_json TEXT NOT NULL,
                        publication_decision TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_scanner_created ON scanner_snapshots (scanner_id, created_at DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_run_id ON scanner_snapshots (run_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_fingerprint ON scanner_snapshots (semantic_fingerprint)")

                # Strict immutability triggers (prevent update/delete)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_scanner_snapshots_no_update
                    BEFORE UPDATE ON scanner_snapshots
                    BEGIN
                        SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be updated in place');
                    END;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS trg_scanner_snapshots_no_delete
                    BEFORE DELETE ON scanner_snapshots
                    BEGIN
                        SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: scanner_snapshots rows cannot be deleted');
                    END;
                """)
        finally:
            conn.close()

    @retry_sqlite()
    def save_snapshot(self, snapshot_data: Dict[str, Any]) -> str:
        """Persist a completed snapshot immutably."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO scanner_snapshots (
                        snapshot_id,
                        scanner_id,
                        run_id,
                        api_contract_version,
                        ruleset_version,
                        evidence_schema_version,
                        score_model_version,
                        data_provenance_version,
                        universe_version,
                        freshness_policy_version,
                        implementation_release_sha,
                        semantic_fingerprint,
                        generated_at,
                        data_as_of,
                        status_at_publication,
                        universe_id,
                        universe_size,
                        matched_count,
                        results_json,
                        provenance_json,
                        freshness_json,
                        publication_decision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot_data["snapshot_id"],
                    snapshot_data["scanner_id"],
                    snapshot_data["run_id"],
                    snapshot_data["api_contract_version"],
                    snapshot_data["ruleset_version"],
                    snapshot_data["evidence_schema_version"],
                    snapshot_data["score_model_version"],
                    snapshot_data["data_provenance_version"],
                    snapshot_data["universe_version"],
                    snapshot_data["freshness_policy_version"],
                    snapshot_data["implementation_release_sha"],
                    snapshot_data["semantic_fingerprint"],
                    snapshot_data["generated_at"],
                    snapshot_data["data_as_of"],
                    snapshot_data["status_at_publication"],
                    snapshot_data["universe_id"],
                    snapshot_data["universe_size"],
                    snapshot_data["matched_count"],
                    json.dumps(snapshot_data["results"], separators=(",", ":")),
                    json.dumps(snapshot_data.get("provenance", {}), separators=(",", ":")),
                    json.dumps(snapshot_data.get("freshness", {}), separators=(",", ":")),
                    snapshot_data.get("publication_decision", "PUBLISH"),
                ))
            return snapshot_data["snapshot_id"]
        finally:
            conn.close()

    @retry_sqlite()
    def get_latest_active_snapshot(self, scanner_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest verified, published snapshot for a scanner."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scanner_snapshots
                WHERE scanner_id = ? AND publication_decision = 'PUBLISH'
                ORDER BY created_at DESC
                LIMIT 1
            """, (scanner_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)
        finally:
            conn.close()

    @retry_sqlite()
    def get_snapshot_by_id(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific immutable snapshot by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scanner_snapshots WHERE snapshot_id = ?", (snapshot_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)
        finally:
            conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row into a structured snapshot dictionary."""
        return {
            "snapshot_id": row["snapshot_id"],
            "scanner_id": row["scanner_id"],
            "run_id": row["run_id"],
            "api_contract_version": row["api_contract_version"],
            "ruleset_version": row["ruleset_version"],
            "evidence_schema_version": row["evidence_schema_version"],
            "score_model_version": row["score_model_version"],
            "data_provenance_version": row["data_provenance_version"],
            "universe_version": row["universe_version"],
            "freshness_policy_version": row["freshness_policy_version"],
            "implementation_release_sha": row["implementation_release_sha"],
            "semantic_fingerprint": row["semantic_fingerprint"],
            "generated_at": row["generated_at"],
            "data_as_of": row["data_as_of"],
            "status_at_publication": row["status_at_publication"],
            "universe_id": row["universe_id"],
            "universe_size": row["universe_size"],
            "matched_count": row["matched_count"],
            "results": json.loads(row["results_json"]),
            "provenance": json.loads(row["provenance_json"]),
            "freshness": json.loads(row["freshness_json"]),
            "publication_decision": row["publication_decision"],
            "created_at": row["created_at"],
        }
