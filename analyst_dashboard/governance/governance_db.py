"""ARX Production Governance SQLite Engine & Immutability Ledger.

Manages persistent SQLite storage for:
1. epoch_certification_results (Canonical machine-readable audit evidence)
2. epoch_release_authorizations (Certified runtime authorizations)
3. epoch_release_revocations (Append-only runtime participation revocations)
4. epoch_activation_records (Immutable epoch activation boundary)

GOVERNING INVARIANTS:
1. PERSISTENT STORAGE: Default DB path is /root/analyst_dashboard/data/governance.db
   (Railway persistent volume mount), overridable via ARX_GOVERNANCE_DB_PATH.
2. STRICT RELATIONAL INTEGRITY: PRAGMA foreign_keys = ON; STRICT tables.
3. DATABASE-ENFORCED IMMUTABILITY: All 4 tables strictly reject UPDATE and DELETE
   via SQLite triggers. No mutation relies solely on application convention.
4. APPEND-ONLY REVOCATION: Revoking runtime participation records an append-only event
   without rewriting historical authorization or resetting the active epoch boundary.
"""

import os
import time
import functools
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("arx.governance.db")

# Default production path on Railway persistent volume mount
DEFAULT_PRODUCTION_GOVERNANCE_DB_PATH = "/root/analyst_dashboard/data/governance.db"


def retry_sqlite(max_retries: int = 5, base_delay: float = 0.05):
    """Decorator to retry SQLite operations with exponential backoff on database lock contention."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_err = e
                    err_msg = str(e).lower()
                    if "locked" in err_msg or "busy" in err_msg:
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                    raise
                except Exception:
                    raise
            if last_err:
                raise last_err
        return wrapper
    return decorator


from analyst_dashboard.governance.storage import (
    resolve_governance_db_path as _storage_resolve_governance_db_path,
    ensure_data_root,
    attest_persistent_storage,
    is_production_runtime,
)


def resolve_governance_db_path(custom_path: Optional[str] = None) -> str:
    """Resolves the authoritative SQLite governance database path using shared storage authority."""
    return _storage_resolve_governance_db_path(custom_path)


def get_governance_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Establishes an isolated SQLite connection with strict pragmas."""
    target_path = resolve_governance_db_path(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path, timeout=15.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass  # In-memory or restricted filesystem fallback
    conn.execute("PRAGMA busy_timeout = 10000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_governance_db(db_path: Optional[str] = None) -> None:
    """Initializes tables, indexes, and immutability triggers for governance storage."""
    conn = get_governance_connection(db_path)
    try:
        with conn:
            conn.executescript("""
            PRAGMA foreign_keys = ON;

            -- 1. Full machine-readable certification audit payloads
            CREATE TABLE IF NOT EXISTS epoch_certification_results (
                certification_result_sha256 TEXT PRIMARY KEY,
                epoch_id TEXT NOT NULL,
                release_sha TEXT NOT NULL,
                deployment_id TEXT NOT NULL,
                overall_status TEXT NOT NULL CHECK(overall_status IN ('PASS', 'FAIL')),
                result_payload_json TEXT NOT NULL,
                certified_at_utc TEXT NOT NULL
            ) STRICT;

            -- 2. Certified release authorizations (Immutable)
            CREATE TABLE IF NOT EXISTS epoch_release_authorizations (
                epoch_id TEXT NOT NULL,
                authorized_release_sha TEXT NOT NULL,
                authorized_deployment_id TEXT NOT NULL,
                production_certification_status TEXT NOT NULL CHECK(production_certification_status = 'PASS'),
                certification_result_sha256 TEXT NOT NULL,
                certified_at_utc TEXT NOT NULL,
                PRIMARY KEY (epoch_id, authorized_release_sha, authorized_deployment_id),
                FOREIGN KEY (certification_result_sha256) REFERENCES epoch_certification_results(certification_result_sha256)
            ) STRICT;

            -- 3. Append-only runtime revocations
            CREATE TABLE IF NOT EXISTS epoch_release_revocations (
                revocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch_id TEXT NOT NULL,
                release_sha TEXT NOT NULL,
                deployment_id TEXT NOT NULL,
                revoked_at_utc TEXT NOT NULL,
                revocation_reason TEXT NOT NULL CHECK(length(revocation_reason) >= 16),
                revoked_by TEXT NOT NULL,
                FOREIGN KEY (epoch_id, release_sha, deployment_id) 
                    REFERENCES epoch_release_authorizations(epoch_id, authorized_release_sha, authorized_deployment_id)
            ) STRICT;

            -- 4. Authoritative Epoch Activation Records
            CREATE TABLE IF NOT EXISTS epoch_activation_records (
                epoch_id TEXT PRIMARY KEY,
                release_sha TEXT NOT NULL,
                deployment_id TEXT NOT NULL,
                activated_at_utc TEXT NOT NULL,
                activation_source TEXT NOT NULL,
                activation_auth_token_hash TEXT NOT NULL,
                FOREIGN KEY (epoch_id, release_sha, deployment_id) 
                    REFERENCES epoch_release_authorizations(epoch_id, authorized_release_sha, authorized_deployment_id)
            ) STRICT;

            -- 5. Append-only Epoch Supersession Records
            CREATE TABLE IF NOT EXISTS epoch_supersession_records (
                supersession_id INTEGER PRIMARY KEY AUTOINCREMENT,
                previous_epoch_id TEXT NOT NULL,
                target_epoch_id TEXT NOT NULL,
                superseded_at_utc TEXT NOT NULL,
                supersession_status TEXT NOT NULL CHECK(supersession_status = 'SUPERSEDED_PRE_OBSERVATION'),
                reason TEXT NOT NULL,
                clean_prospective_signals_captured INTEGER NOT NULL,
                empirical_evidence_lost INTEGER NOT NULL,
                supersession_payload_json TEXT NOT NULL
            ) STRICT;

            -- IMMUTABILITY TRIGGERS
            CREATE TRIGGER IF NOT EXISTS prevent_update_epoch_certification_results
            BEFORE UPDATE ON epoch_certification_results
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_certification_results is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_delete_epoch_certification_results
            BEFORE DELETE ON epoch_certification_results
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_certification_results cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_update_epoch_release_authorizations
            BEFORE UPDATE ON epoch_release_authorizations
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_release_authorizations is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_delete_epoch_release_authorizations
            BEFORE DELETE ON epoch_release_authorizations
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_release_authorizations cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_update_epoch_release_revocations
            BEFORE UPDATE ON epoch_release_revocations
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_release_revocations is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_delete_epoch_release_revocations
            BEFORE DELETE ON epoch_release_revocations
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_release_revocations cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_update_epoch_activation_records
            BEFORE UPDATE ON epoch_activation_records
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_activation_records is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_delete_epoch_activation_records
            BEFORE DELETE ON epoch_activation_records
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_activation_records cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_update_epoch_supersession_records
            BEFORE UPDATE ON epoch_supersession_records
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_supersession_records is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_delete_epoch_supersession_records
            BEFORE DELETE ON epoch_supersession_records
            BEGIN
                SELECT RAISE(FAIL, 'FAIL_CLOSED: epoch_supersession_records cannot be deleted');
            END;
            """)
    finally:
        conn.close()


class GovernanceDatabaseEngine:
    """Operations interface for the governance SQLite ledger."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = resolve_governance_db_path(db_path)
        init_governance_db(self.db_path)

    def get_connection(self) -> sqlite3.Connection:
        return get_governance_connection(self.db_path)

    @retry_sqlite()
    def record_certification_and_authorization(
        self,
        epoch_id: str,
        release_sha: str,
        deployment_id: str,
        overall_status: str,
        result_payload_json: str,
        result_sha256: str,
        certified_at_utc: str,
    ) -> None:
        """Atomically inserts certification results and authorization row on PASS."""
        conn = self.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO epoch_certification_results (
                        certification_result_sha256,
                        epoch_id,
                        release_sha,
                        deployment_id,
                        overall_status,
                        result_payload_json,
                        certified_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result_sha256,
                        epoch_id,
                        release_sha,
                        deployment_id,
                        overall_status,
                        result_payload_json,
                        certified_at_utc,
                    ),
                )
                if overall_status == "PASS":
                    conn.execute(
                        """
                        INSERT INTO epoch_release_authorizations (
                            epoch_id,
                            authorized_release_sha,
                            authorized_deployment_id,
                            production_certification_status,
                            certification_result_sha256,
                            certified_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            epoch_id,
                            release_sha,
                            deployment_id,
                            "PASS",
                            result_sha256,
                            certified_at_utc,
                        ),
                    )
        finally:
            conn.close()

    @retry_sqlite()
    def record_activation(
        self,
        epoch_id: str,
        release_sha: str,
        deployment_id: str,
        activated_at_utc: str,
        activation_source: str,
        activation_auth_token_hash: str,
    ) -> Tuple[bool, str]:
        """Inserts an immutable epoch activation record.
        Uses BEGIN IMMEDIATE to acquire writer lock before checking eligibility reads,
        eliminating TOCTOU races between authorization/revocation check and activation insert.
        Fails closed on conflicts or rollbacks.
        """
        conn = self.get_connection()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # Check for existing activation record for this epoch
            cur = conn.execute(
                "SELECT release_sha, deployment_id, activated_at_utc FROM epoch_activation_records WHERE epoch_id = ?",
                (epoch_id,),
            )
            existing = cur.fetchone()
            if existing:
                conn.execute("ROLLBACK;")
                if (
                    existing["release_sha"] == release_sha
                    and existing["deployment_id"] == deployment_id
                ):
                    return True, "IDEMPOTENT_ALREADY_ACTIVATED"
                return False, f"CONFLICT: Epoch {epoch_id} already activated by release {existing['release_sha']}"

            # Ensure release authorization exists with PASS status
            auth_cur = conn.execute(
                """
                SELECT production_certification_status FROM epoch_release_authorizations
                WHERE epoch_id = ? AND authorized_release_sha = ? AND authorized_deployment_id = ?
                """,
                (epoch_id, release_sha, deployment_id),
            )
            auth = auth_cur.fetchone()
            if not auth or auth["production_certification_status"] != "PASS":
                conn.execute("ROLLBACK;")
                return False, "UNAUTHORIZED: No valid PASS release authorization found"

            # Ensure exact runtime is not revoked
            rev_cur = conn.execute(
                """
                SELECT COUNT(*) FROM epoch_release_revocations
                WHERE epoch_id = ? AND release_sha = ? AND deployment_id = ?
                """,
                (epoch_id, release_sha, deployment_id),
            )
            if rev_cur.fetchone()[0] > 0:
                conn.execute("ROLLBACK;")
                return False, f"RUNTIME_REVOKED: Target release authorization for {release_sha} on deployment {deployment_id} has been revoked"

            # Invariant: Persistent storage assertion (Fail-closed defense-in-depth)
            if is_production_runtime():
                attest = attest_persistent_storage()
                if not attest.get("isValid", False):
                    conn.execute("ROLLBACK;")
                    return False, f"STORAGE_NOT_PERSISTENT: Cannot activate epoch on non-persistent storage: {attest.get('error', 'attestation failed')}"

            conn.execute(
                """
                INSERT INTO epoch_activation_records (
                    epoch_id,
                    release_sha,
                    deployment_id,
                    activated_at_utc,
                    activation_source,
                    activation_auth_token_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    epoch_id,
                    release_sha,
                    deployment_id,
                    activated_at_utc,
                    activation_source,
                    activation_auth_token_hash,
                ),
            )
            conn.execute("COMMIT;")
            return True, "ACTIVATION_SUCCESSFUL"
        except Exception:
            if conn.in_transaction:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def record_revocation(
        self,
        epoch_id: str,
        release_sha: str,
        deployment_id: str,
        revoked_at_utc: str,
        revocation_reason: str,
        revoked_by: str,
    ) -> Tuple[bool, str]:
        """Appends a revocation record for a specific authorized runtime.
        Uses BEGIN IMMEDIATE to acquire writer lock before checking target authorization.
        """
        conn = self.get_connection()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # Target must exist in authorizations
            auth_cur = conn.execute(
                """
                SELECT 1 FROM epoch_release_authorizations
                WHERE epoch_id = ? AND authorized_release_sha = ? AND authorized_deployment_id = ?
                """,
                (epoch_id, release_sha, deployment_id),
            )
            if not auth_cur.fetchone():
                conn.execute("ROLLBACK;")
                return False, "NOT_FOUND: Target release authorization does not exist"

            conn.execute(
                """
                INSERT INTO epoch_release_revocations (
                    epoch_id,
                    release_sha,
                    deployment_id,
                    revoked_at_utc,
                    revocation_reason,
                    revoked_by
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    epoch_id,
                    release_sha,
                    deployment_id,
                    revoked_at_utc,
                    revocation_reason,
                    revoked_by,
                ),
            )
            conn.execute("COMMIT;")
            return True, "REVOCATION_RECORDED"
        except Exception:
            if conn.in_transaction:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
            raise
        finally:
            conn.close()

    def get_activation_record(self, epoch_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM epoch_activation_records WHERE epoch_id = ?",
                (epoch_id,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_release_authorization(
        self, epoch_id: str, release_sha: str, deployment_id: str
    ) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT * FROM epoch_release_authorizations
                WHERE epoch_id = ? AND authorized_release_sha = ? AND authorized_deployment_id = ?
                """,
                (epoch_id, release_sha, deployment_id),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def is_runtime_revoked(
        self, epoch_id: str, release_sha: str, deployment_id: str
    ) -> bool:
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT COUNT(*) FROM epoch_release_revocations
                WHERE epoch_id = ? AND release_sha = ? AND deployment_id = ?
                """,
                (epoch_id, release_sha, deployment_id),
            )
            return cur.fetchone()[0] > 0
        finally:
            conn.close()

    @retry_sqlite()
    def evaluate_capture_authorization_predicate(
        self,
        epoch_id: str,
        release_sha: Optional[str] = None,
        deployment_id: Optional[str] = None,
        now_utc: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Evaluates whether prospective capture is authorized for the given runtime.
        Fails closed on any error, missing metadata, un-activated state, unauthorized release, or revocation.
        """
        current_release = release_sha or os.getenv("RAILWAY_GIT_COMMIT_SHA")
        current_deployment = deployment_id or os.getenv("RAILWAY_DEPLOYMENT_ID")
        eval_now = now_utc or datetime.now(timezone.utc).isoformat()

        if not current_release or not current_deployment:
            return False, "MISSING_RUNTIME_IDENTITY"

        conn = self.get_connection()
        try:
            # 1. Epoch active check
            act_cur = conn.execute(
                "SELECT activated_at_utc FROM epoch_activation_records WHERE epoch_id = ?",
                (epoch_id,),
            )
            act_row = act_cur.fetchone()
            if not act_row:
                return False, "EPOCH_NOT_ACTIVATED"
            if eval_now < act_row["activated_at_utc"]:
                return False, f"TIME_PRECEDES_ACTIVATION: now={eval_now} < act={act_row['activated_at_utc']}"

            # 2. Release authorization check
            auth_cur = conn.execute(
                """
                SELECT production_certification_status FROM epoch_release_authorizations
                WHERE epoch_id = ? AND authorized_release_sha = ? AND authorized_deployment_id = ?
                """,
                (epoch_id, current_release, current_deployment),
            )
            auth_row = auth_cur.fetchone()
            if not auth_row or auth_row["production_certification_status"] != "PASS":
                return False, "RUNTIME_NOT_AUTHORIZED"

            # 3. Revocation check
            rev_cur = conn.execute(
                """
                SELECT COUNT(*) FROM epoch_release_revocations
                WHERE epoch_id = ? AND release_sha = ? AND deployment_id = ?
                """,
                (epoch_id, current_release, current_deployment),
            )
            if rev_cur.fetchone()[0] > 0:
                return False, "RUNTIME_REVOKED"

            return True, "AUTHORIZED"
        except Exception as e:
            logger.error("Error evaluating capture authorization predicate: %s", e)
            return False, f"PREDICATE_ERROR: {str(e)}"
        finally:
            conn.close()

    @retry_sqlite()
    def record_epoch_supersession(
        self,
        previous_epoch_id: str,
        target_epoch_id: str,
        superseded_at_utc: str,
        reason: str,
        clean_prospective_signals_captured: int,
        empirical_evidence_lost: int,
        supersession_payload: Dict[str, Any],
    ) -> int:
        """Atomically inserts an append-only epoch supersession record."""
        import json
        conn = self.get_connection()
        try:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO epoch_supersession_records (
                        previous_epoch_id,
                        target_epoch_id,
                        superseded_at_utc,
                        supersession_status,
                        reason,
                        clean_prospective_signals_captured,
                        empirical_evidence_lost,
                        supersession_payload_json
                    ) VALUES (?, ?, ?, 'SUPERSEDED_PRE_OBSERVATION', ?, ?, ?, ?)
                    """,
                    (
                        previous_epoch_id,
                        target_epoch_id,
                        superseded_at_utc,
                        reason,
                        clean_prospective_signals_captured,
                        empirical_evidence_lost,
                        json.dumps(supersession_payload, sort_keys=True),
                    ),
                )
                return cur.lastrowid
        finally:
            conn.close()

    def get_epoch_supersession_record(
        self, previous_epoch_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the latest supersession record for a given epoch."""
        import json
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT supersession_id, previous_epoch_id, target_epoch_id,
                       superseded_at_utc, supersession_status, reason,
                       clean_prospective_signals_captured, empirical_evidence_lost,
                       supersession_payload_json
                FROM epoch_supersession_records
                WHERE previous_epoch_id = ?
                ORDER BY supersession_id DESC LIMIT 1
                """,
                (previous_epoch_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "supersession_id": row["supersession_id"],
                "previous_epoch_id": row["previous_epoch_id"],
                "target_epoch_id": row["target_epoch_id"],
                "superseded_at_utc": row["superseded_at_utc"],
                "supersession_status": row["supersession_status"],
                "reason": row["reason"],
                "clean_prospective_signals_captured": row["clean_prospective_signals_captured"],
                "empirical_evidence_lost": row["empirical_evidence_lost"],
                "supersession_payload": json.loads(row["supersession_payload_json"]),
            }
        finally:
            conn.close()
