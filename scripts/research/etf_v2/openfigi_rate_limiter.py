"""
scripts/research/etf_v2/openfigi_rate_limiter.py

Cross-process atomic rolling-window rate limiter for OpenFIGI v3 integration.
Enforces the ratified global project ceiling:
    20 requests per rolling 60-second window across all processes.

Invariants Enforced:
- OFIGI-RLR07: Global reservation operation atomic across processes via SQLite BEGIN IMMEDIATE.
- OFIGI-RLR08: Rolling 60-second window semantics: count(reservations in (T - 60s, T]) <= 20.
- OFIGI-RLR09: Initial requests consume global capacity.
- OFIGI-RLR10: Every retry consumes global capacity.
- OFIGI-RLR11: Local pre-dispatch rejection consumes no capacity.
- OFIGI-RLR12: Crash reservation semantics fail safe (unexpired reservations remain consumed).
- OFIGI-RLR14: Deterministic clock seam for zero-sleep offline testing.
- OFIGI-RLR15: Safe handling of clock regression (future timestamps count toward capacity).
- OFIGI-RLR16: Canonical database contamination strictly prohibited.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, List, Optional, Tuple, Union
import uuid

from .openfigi_config import (
    CANONICAL_DB_NAME,
    DEFAULT_OPERATIONAL_DB_PATH,
    resolve_openfigi_operational_db_path,
)

GLOBAL_CAPACITY = 20
GLOBAL_WINDOW_SECONDS = 60.0


class CanonicalStoreContaminationError(RuntimeError):
    """Raised when an operational component attempts to connect to a canonical database."""
    pass


class OpenFIGIRateLimitTimeoutError(TimeoutError):
    """Raised when acquiring a global rate limit reservation times out."""
    pass


class GlobalSQLiteRateLimiter:
    """
    Cross-process atomic rolling-window rate limiter backed by SQLite.
    Guarantees that at every dispatch time T:
        count(successfully reserved attempts in (T - 60s, T]) <= 20
    across all processes and threads.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        capacity: int = GLOBAL_CAPACITY,
        window_seconds: float = GLOBAL_WINDOW_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        sleep_func: Optional[Callable[[float], None]] = None,
        timeout: float = 30.0
    ):
        self.db_path = resolve_openfigi_operational_db_path(db_path)

        self.capacity = capacity
        self.window_seconds = window_seconds
        self.clock = clock or time.time
        self.sleep_func = sleep_func or time.sleep
        self.busy_timeout = timeout
        self._thread_lock = threading.Lock()
        self._schema_initialized = False

        # Enforce canonical path firewall
        resolved = str(self.db_path.resolve()).replace("\\", "/")
        if CANONICAL_DB_NAME in resolved or "data/canonical" in resolved:
            raise CanonicalStoreContaminationError(
                f"CANONICAL_FIREWALL_VIOLATION: Rate limiter cannot target canonical store {self.db_path}"
            )

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout,
            isolation_level=None  # Explicit transaction control
        )
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout * 1000)};")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Initializes the reservation table atomically if not present."""
        if not self._schema_initialized:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS openfigi_rate_limit_reservations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reservation_id TEXT NOT NULL UNIQUE,
                    process_id INTEGER NOT NULL,
                    reserved_at REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_limit_reserved_at
                ON openfigi_rate_limit_reservations(reserved_at);
            """)
            self._schema_initialized = True

    def try_reserve(self, now: Optional[float] = None) -> Tuple[bool, float]:
        """
        Attempts to reserve a slot atomically across processes.
        Returns:
            (True, 0.0) if slot is reserved immediately.
            (False, wait_seconds) if capacity is exhausted in the current rolling window.
        """
        current_t = now if now is not None else self.clock()
        start_attempt = time.monotonic()

        with self._thread_lock:
            while True:
                try:
                    conn = self._get_connection()
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        self._ensure_schema(conn)

                        # Prune very old reservations (> 24 hours) to keep table bounded
                        conn.execute(
                            "DELETE FROM openfigi_rate_limit_reservations WHERE reserved_at < ?",
                            (current_t - 86400.0,)
                        )

                        # Query active reservations in the rolling window.
                        # Window: (current_t - window_seconds, infinity).
                        # Note: We include any reserved_at > current_t to fail safe against clock regression!
                        window_cutoff = current_t - self.window_seconds
                        row = conn.execute(
                            "SELECT COUNT(*) FROM openfigi_rate_limit_reservations WHERE reserved_at > ?",
                            (window_cutoff,)
                        ).fetchone()
                        active_count = row[0] if row else 0

                        if active_count < self.capacity:
                            # Slot available: reserve immediately
                            res_id = str(uuid.uuid4())
                            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                            conn.execute(
                                """
                                INSERT INTO openfigi_rate_limit_reservations
                                (reservation_id, process_id, reserved_at, created_at)
                                VALUES (?, ?, ?, ?)
                                """,
                                (res_id, os.getpid(), current_t, now_iso)
                            )
                            conn.execute("COMMIT")
                            return True, 0.0
                        else:
                            # Capacity full: calculate wait time based on earliest slot to expire
                            offset = active_count - self.capacity
                            row = conn.execute(
                                """
                                SELECT reserved_at FROM openfigi_rate_limit_reservations
                                WHERE reserved_at > ?
                                ORDER BY reserved_at ASC
                                LIMIT 1 OFFSET ?
                                """,
                                (window_cutoff, offset)
                            ).fetchone()

                            conn.execute("COMMIT")
                            if row:
                                oldest_active = row[0]
                                free_at = oldest_active + self.window_seconds
                                wait_needed = max(0.001, free_at - current_t)
                            else:
                                wait_needed = self.window_seconds

                            return False, wait_needed
                    except sqlite3.OperationalError as e:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        if ("locked" in str(e).lower() or "busy" in str(e).lower()) and (time.monotonic() - start_attempt < self.busy_timeout):
                            time.sleep(0.005)
                            continue
                        raise
                    finally:
                        conn.close()
                except sqlite3.OperationalError as e:
                    if ("locked" in str(e).lower() or "busy" in str(e).lower()) and (time.monotonic() - start_attempt < self.busy_timeout):
                        time.sleep(0.005)
                        continue
                    raise

    def acquire_delay(self) -> float:
        """
        Evaluates reservation for a single dispatch attempt.
        If capacity is available, reserves a slot immediately and returns 0.0.
        If full, returns the required delay in seconds WITHOUT reserving.
        """
        acquired, wait_sec = self.try_reserve()
        if acquired:
            return 0.0
        return wait_sec

    def acquire(self, timeout: float = 120.0) -> float:
        """
        Acquires a global rate-limit reservation, waiting if necessary.
        Returns the total wait time experienced in seconds.
        """
        start_t = self.clock()
        total_waited = 0.0

        while True:
            current_t = self.clock()
            if (current_t - start_t) > timeout:
                raise OpenFIGIRateLimitTimeoutError(
                    f"Rate limit reservation timed out after {total_waited:.2f}s (timeout={timeout}s)"
                )

            acquired, wait_sec = self.try_reserve(now=current_t)
            if acquired:
                return total_waited

            sleep_duration = min(wait_sec, max(0.001, timeout - (current_t - start_t)))
            self.sleep_func(sleep_duration)
            total_waited += sleep_duration

    def get_active_count(self, now: Optional[float] = None) -> int:
        """Returns the number of active reservations in the current rolling window."""
        current_t = now if now is not None else self.clock()
        conn = self._get_connection()
        try:
            self._ensure_schema(conn)
            window_cutoff = current_t - self.window_seconds
            row = conn.execute(
                "SELECT COUNT(*) FROM openfigi_rate_limit_reservations WHERE reserved_at > ?",
                (window_cutoff,)
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def get_all_reservations(self) -> List[Tuple[int, str, int, float, str]]:
        """Returns all reservation records for auditing."""
        conn = self._get_connection()
        try:
            self._ensure_schema(conn)
            return conn.execute(
                "SELECT id, reservation_id, process_id, reserved_at, created_at FROM openfigi_rate_limit_reservations ORDER BY reserved_at ASC"
            ).fetchall()
        finally:
            conn.close()
