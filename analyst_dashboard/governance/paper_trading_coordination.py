"""ARX Paper-Trading Persistent Coordination Store.

Governed by:
- ARX Paper-Trading Outcome Evaluation Contract V1.0.1
- docs/planning/ARX_PAPER_TRADING_ROADMAP.md (PR #322)

Provides:
1. Transactional SQLite coordination store with WAL mode and BEGIN IMMEDIATE locking.
2. Single coordination authority for:
   - Session claims by (cohort_id, session_date) across branches/runners.
   - Cumulative request reservations (reserved durably on disk before transmission).
   - Capture completion tracking and idempotent recovery.
3. Fail-closed storage attestation: store failures block network acquisition before transmission.
4. Separate tracking of invocation, source cutoff, completed capture times, and outside-corridor classifications.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arx.governance.paper_trading_coordination")

DEFAULT_COORDINATION_DB_FILENAME = ".paper_trading_coordination.db"


class PaperTradingCoordinationError(Exception):
    """Base exception for paper-trading coordination failures."""
    pass


class CoordinationStoreUnavailableError(PaperTradingCoordinationError):
    """Raised when the persistent coordination store cannot be accessed or verified (fail-closed)."""
    pass


class ConcurrentClaimConflictError(PaperTradingCoordinationError):
    """Raised when another runner actively holds the lease for a cohort session."""
    pass


class BudgetExhaustedError(PaperTradingCoordinationError):
    """Raised when cumulative request attempts reach or exceed authorized session budget."""
    pass


class CaptureAlreadyCompletedError(PaperTradingCoordinationError):
    """Raised when attempting to capture an already completed cohort session."""
    pass


class PaperTradingCoordinationStore:
    """Production-grade SQLite transactional store for forward paper-trading coordination."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = self._resolve_db_path(db_path)
        self._ensure_store_ready()

    @staticmethod
    def _resolve_db_path(custom_path: Optional[str | Path]) -> Path:
        if custom_path is not None:
            return Path(custom_path).resolve()

        env_path = os.getenv("ARX_FORWARD_COORDINATION_DB_PATH")
        if env_path:
            return Path(env_path).resolve()

        # Resolve forward pilot data root
        try:
            from analyst_dashboard.governance.storage import resolve_forward_pilot_data_root
            data_root = resolve_forward_pilot_data_root()
        except Exception:
            repo_root = Path(__file__).resolve().parents[2]
            data_root = repo_root / "data" / "forward_pilot"

        return Path(data_root).resolve() / DEFAULT_COORDINATION_DB_FILENAME

    def _get_connection(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            raise CoordinationStoreUnavailableError(
                f"FAIL_CLOSED: Cannot establish connection to coordination store at '{self.db_path}': {e}"
            ) from e

    def _ensure_store_ready(self) -> None:
        """Initializes database schema under immediate transaction; fails closed if storage is unavailable."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_claims (
                        cohort_id TEXT NOT NULL,
                        session_date TEXT NOT NULL,
                        claimed_by TEXT NOT NULL,
                        claimed_at_utc TEXT NOT NULL,
                        claim_status TEXT NOT NULL,
                        lease_expires_at_utc TEXT NOT NULL,
                        invocation_time_utc TEXT NOT NULL,
                        source_cutoff_time_utc TEXT NOT NULL,
                        completed_at_utc TEXT,
                        batch_sha256 TEXT,
                        records_count INTEGER DEFAULT 0,
                        valid_setup_count INTEGER DEFAULT 0,
                        insufficient_data_count INTEGER DEFAULT 0,
                        outside_corridor_count INTEGER DEFAULT 0,
                        waiting_pullback_count INTEGER DEFAULT 0,
                        note TEXT,
                        PRIMARY KEY (cohort_id, session_date)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session_claims_status ON session_claims(claim_status);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS request_reservations (
                        cohort_id TEXT NOT NULL,
                        session_date TEXT NOT NULL,
                        attempt_index INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        target_url TEXT NOT NULL,
                        reserved_at_utc TEXT NOT NULL,
                        status TEXT NOT NULL,
                        completed_at_utc TEXT,
                        http_code INTEGER,
                        duration_ms REAL,
                        sha256 TEXT,
                        byte_length INTEGER,
                        error_message TEXT,
                        PRIMARY KEY (cohort_id, session_date, attempt_index)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_request_reservations_symbol ON request_reservations(symbol, status);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_budgets (
                        cohort_id TEXT NOT NULL,
                        session_date TEXT NOT NULL,
                        max_budget INTEGER NOT NULL,
                        consumed_budget INTEGER NOT NULL DEFAULT 0,
                        updated_at_utc TEXT NOT NULL,
                        PRIMARY KEY (cohort_id, session_date)
                    );
                """)
                conn.execute("COMMIT;")
            except Exception as e:
                conn.execute("ROLLBACK;")
                raise CoordinationStoreUnavailableError(
                    f"FAIL_CLOSED: Schema initialization failed for coordination store at '{self.db_path}': {e}"
                ) from e
            finally:
                conn.close()
        except CoordinationStoreUnavailableError:
            raise
        except Exception as e:
            raise CoordinationStoreUnavailableError(
                f"FAIL_CLOSED: Failed to access coordination store storage at '{self.db_path}': {e}"
            ) from e

    def claim_session(
        self,
        *,
        cohort_id: str,
        session_date: str,
        claimed_by: str,
        invocation_time_utc: str,
        source_cutoff_time_utc: str,
        max_budget: int,
        lease_seconds: int = 300,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Atomically acquires session claim across runners or branches; handles recovery of expired claims."""
        now = now_utc or datetime.now(timezone.utc)
        now_iso = now.isoformat()
        lease_expires_iso = datetime.fromtimestamp(now.timestamp() + lease_seconds, tz=timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # Check existing claim
            row = conn.execute(
                "SELECT * FROM session_claims WHERE cohort_id = ? AND session_date = ?;",
                (cohort_id, session_date),
            ).fetchone()

            if row:
                status = row["claim_status"]
                if status == "COMPLETED":
                    conn.execute("COMMIT;")
                    return {
                        "status": "ALREADY_COMPLETED",
                        "cohortId": cohort_id,
                        "sessionDate": session_date,
                        "claimedBy": row["claimed_by"],
                        "completedAtUtc": row["completed_at_utc"],
                        "batchSha256": row["batch_sha256"],
                    }

                # If claim is active and unexpired, check lease ownership
                lease_exp = row["lease_expires_at_utc"]
                is_expired = now_iso >= lease_exp

                if not is_expired and row["claimed_by"] != claimed_by:
                    conn.execute("COMMIT;")
                    raise ConcurrentClaimConflictError(
                        f"CONCURRENT_CLAIM_REJECTED: Session '{session_date}' for '{cohort_id}' is actively claimed "
                        f"by runner '{row['claimed_by']}' until {lease_exp}."
                    )

                # Active lease by same runner or expired lease: extend/reclaim
                conn.execute("""
                    UPDATE session_claims
                    SET claimed_by = ?,
                        claim_status = 'IN_PROGRESS',
                        lease_expires_at_utc = ?,
                        invocation_time_utc = ?,
                        source_cutoff_time_utc = ?
                    WHERE cohort_id = ? AND session_date = ?;
                """, (claimed_by, lease_expires_iso, invocation_time_utc, source_cutoff_time_utc, cohort_id, session_date))

                # Ensure budget row is tracked
                conn.execute("""
                    INSERT INTO session_budgets (cohort_id, session_date, max_budget, consumed_budget, updated_at_utc)
                    VALUES (?, ?, ?, (SELECT COUNT(*) FROM request_reservations WHERE cohort_id = ? AND session_date = ?), ?)
                    ON CONFLICT(cohort_id, session_date) DO UPDATE SET max_budget = excluded.max_budget, updated_at_utc = excluded.updated_at_utc;
                """, (cohort_id, session_date, max_budget, cohort_id, session_date, now_iso))

                conn.execute("COMMIT;")
                return {
                    "status": "RECLAIMED" if is_expired else "RENEWED",
                    "cohortId": cohort_id,
                    "sessionDate": session_date,
                    "claimedBy": claimed_by,
                    "leaseExpiresAtUtc": lease_expires_iso,
                }

            # Insert new claim
            conn.execute("""
                INSERT INTO session_claims (
                    cohort_id, session_date, claimed_by, claimed_at_utc,
                    claim_status, lease_expires_at_utc, invocation_time_utc,
                    source_cutoff_time_utc
                ) VALUES (?, ?, ?, ?, 'CLAIMED', ?, ?, ?);
            """, (cohort_id, session_date, claimed_by, now_iso, lease_expires_iso, invocation_time_utc, source_cutoff_time_utc))

            conn.execute("""
                INSERT INTO session_budgets (cohort_id, session_date, max_budget, consumed_budget, updated_at_utc)
                VALUES (?, ?, ?, 0, ?);
            """, (cohort_id, session_date, max_budget, now_iso))

            conn.execute("COMMIT;")
            return {
                "status": "CLAIMED",
                "cohortId": cohort_id,
                "sessionDate": session_date,
                "claimedBy": claimed_by,
                "leaseExpiresAtUtc": lease_expires_iso,
            }
        except (ConcurrentClaimConflictError, CoordinationStoreUnavailableError):
            raise
        except Exception as e:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise CoordinationStoreUnavailableError(f"Claim transaction failed: {e}") from e
        finally:
            conn.close()

    def reserve_http_attempt(
        self,
        *,
        cohort_id: str,
        session_date: str,
        symbol: str,
        target_url: str,
        max_budget: int,
        now_utc: Optional[datetime] = None,
    ) -> int:
        """Atomically reserves an attempt in the store BEFORE transmission; raises BudgetExhaustedError if limit reached."""
        now = now_utc or datetime.now(timezone.utc)
        now_iso = now.isoformat()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # Count existing reserved/completed attempts
            row = conn.execute(
                "SELECT COUNT(*) as count FROM request_reservations WHERE cohort_id = ? AND session_date = ?;",
                (cohort_id, session_date),
            ).fetchone()
            current_attempts = row["count"] if row else 0

            if current_attempts >= max_budget:
                conn.execute("COMMIT;")
                raise BudgetExhaustedError(
                    f"BUDGET_EXHAUSTED: Consumed {current_attempts}/{max_budget} request attempts for "
                    f"cohort {cohort_id} session {session_date}. No further requests allowed."
                )

            next_index = current_attempts + 1

            conn.execute("""
                INSERT INTO request_reservations (
                    cohort_id, session_date, attempt_index, symbol, target_url,
                    reserved_at_utc, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'RESERVED');
            """, (cohort_id, session_date, next_index, symbol, target_url, now_iso))

            conn.execute("""
                INSERT INTO session_budgets (cohort_id, session_date, max_budget, consumed_budget, updated_at_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cohort_id, session_date) DO UPDATE SET
                    consumed_budget = excluded.consumed_budget,
                    updated_at_utc = excluded.updated_at_utc;
            """, (cohort_id, session_date, max_budget, next_index, now_iso))

            conn.execute("COMMIT;")
            return next_index
        except (BudgetExhaustedError, CoordinationStoreUnavailableError):
            raise
        except Exception as e:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise CoordinationStoreUnavailableError(f"Attempt reservation failed: {e}") from e
        finally:
            conn.close()

    def finalize_http_attempt(
        self,
        *,
        cohort_id: str,
        session_date: str,
        attempt_index: int,
        status: str,
        http_code: Optional[int] = None,
        duration_ms: float = 0.0,
        sha256: Optional[str] = None,
        byte_length: int = 0,
        error_message: Optional[str] = None,
        now_utc: Optional[datetime] = None,
    ) -> None:
        """Durable completion update for a reserved request attempt."""
        now = now_utc or datetime.now(timezone.utc)
        now_iso = now.isoformat()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("""
                UPDATE request_reservations
                SET status = ?,
                    completed_at_utc = ?,
                    http_code = ?,
                    duration_ms = ?,
                    sha256 = ?,
                    byte_length = ?,
                    error_message = ?
                WHERE cohort_id = ? AND session_date = ? AND attempt_index = ?;
            """, (status, now_iso, http_code, duration_ms, sha256, byte_length, error_message, cohort_id, session_date, attempt_index))
            conn.execute("COMMIT;")
        except Exception as e:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise CoordinationStoreUnavailableError(f"Attempt finalization failed: {e}") from e
        finally:
            conn.close()

    def get_consumed_budget(self, cohort_id: str, session_date: str) -> int:
        """Returns the total number of attempts reserved/consumed so far for the session."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM request_reservations WHERE cohort_id = ? AND session_date = ?;",
                (cohort_id, session_date),
            ).fetchone()
            return row["count"] if row else 0
        finally:
            conn.close()

    def get_session_recovery_state(self, cohort_id: str, session_date: str) -> Dict[str, Any]:
        """Returns comprehensive state for restarting or recovering a session without duplicate work."""
        conn = self._get_connection()
        try:
            claim_row = conn.execute(
                "SELECT * FROM session_claims WHERE cohort_id = ? AND session_date = ?;",
                (cohort_id, session_date),
            ).fetchone()

            attempts_rows = conn.execute(
                "SELECT * FROM request_reservations WHERE cohort_id = ? AND session_date = ? ORDER BY attempt_index ASC;",
                (cohort_id, session_date),
            ).fetchall()

            completed_symbols: Dict[str, Dict[str, Any]] = {}
            for r in attempts_rows:
                if r["status"] == "SUCCESS" and r["sha256"]:
                    completed_symbols[r["symbol"]] = {
                        "sha256": r["sha256"],
                        "byteLength": r["byte_length"],
                        "httpCode": r["http_code"],
                        "attemptIndex": r["attempt_index"],
                    }

            return {
                "cohortId": cohort_id,
                "sessionDate": session_date,
                "claim": dict(claim_row) if claim_row else None,
                "totalAttemptsConsumed": len(attempts_rows),
                "completedSymbols": completed_symbols,
                "attempts": [dict(r) for r in attempts_rows],
            }
        finally:
            conn.close()

    def mark_capture_completed(
        self,
        *,
        cohort_id: str,
        session_date: str,
        batch_sha256: str,
        records_count: int,
        valid_setup_count: int,
        insufficient_data_count: int,
        outside_corridor_count: int,
        waiting_pullback_count: int,
        note: Optional[str] = None,
        now_utc: Optional[datetime] = None,
    ) -> None:
        """Atomically records capture completion with fine-grained classification counts."""
        now = now_utc or datetime.now(timezone.utc)
        now_iso = now.isoformat()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("""
                UPDATE session_claims
                SET claim_status = 'COMPLETED',
                    completed_at_utc = ?,
                    batch_sha256 = ?,
                    records_count = ?,
                    valid_setup_count = ?,
                    insufficient_data_count = ?,
                    outside_corridor_count = ?,
                    waiting_pullback_count = ?,
                    note = ?
                WHERE cohort_id = ? AND session_date = ?;
            """, (
                now_iso, batch_sha256, records_count, valid_setup_count,
                insufficient_data_count, outside_corridor_count, waiting_pullback_count,
                note, cohort_id, session_date
            ))
            conn.execute("COMMIT;")
        except Exception as e:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise CoordinationStoreUnavailableError(f"Marking capture completed failed: {e}") from e
        finally:
            conn.close()

    def is_capture_completed(self, cohort_id: str, session_date: str) -> bool:
        """Checks if the cohort session is already marked completed."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT claim_status FROM session_claims WHERE cohort_id = ? AND session_date = ?;",
                (cohort_id, session_date),
            ).fetchone()
            return bool(row and row["claim_status"] == "COMPLETED")
        finally:
            conn.close()
