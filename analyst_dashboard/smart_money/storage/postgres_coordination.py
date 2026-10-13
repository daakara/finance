"""
ARX Terminal — PostgreSQL Distributed Coordination & Advisory Lock Manager.

Provides deterministic advisory lock namespaces, mutual exclusion for SEC egress ownership,
and transaction-scoped run serialization.
"""

import hashlib
import logging
from contextlib import contextmanager
from typing import Generator
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

# Deterministic 64-bit BigInt lock keys derived via SHA-256
def _derive_lock_key(name: str) -> int:
    h = hashlib.sha256(f"ARX_LOCK_{name}".encode("utf-8")).hexdigest()
    # Mask to signed 63-bit integer to fit PostgreSQL bigint advisory lock parameter
    return int(h[:15], 16) & 0x7FFFFFFFFFFFFFFF

LOCK_SEC_EGRESS_OWNER = _derive_lock_key("SEC_EGRESS_OWNER")
LOCK_DISCOVERY_RUN = _derive_lock_key("DISCOVERY_RUN")
LOCK_RECONCILIATION_RUN = _derive_lock_key("RECONCILIATION_RUN")
LOCK_INGESTION_RUN = _derive_lock_key("INGESTION_RUN")
LOCK_SNAPSHOT_PUBLICATION = _derive_lock_key("SNAPSHOT_PUBLICATION")

LOCK_NAMES = {
    LOCK_SEC_EGRESS_OWNER: "SEC_EGRESS_OWNER",
    LOCK_DISCOVERY_RUN: "DISCOVERY_RUN",
    LOCK_RECONCILIATION_RUN: "RECONCILIATION_RUN",
    LOCK_INGESTION_RUN: "INGESTION_RUN",
    LOCK_SNAPSHOT_PUBLICATION: "SNAPSHOT_PUBLICATION",
}


def try_advisory_lock(conn: PgConnection, lock_key: int) -> bool:
    """
    Attempts to acquire a session-level PostgreSQL advisory lock non-blockingly.
    Returns True if acquired, False otherwise.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s);", (lock_key,))
        res = cur.fetchone()
        acquired = bool(res and res[0])
        name = LOCK_NAMES.get(lock_key, str(lock_key))
        if acquired:
            logger.info(f"Acquired session advisory lock: {name} ({lock_key})")
        else:
            logger.warning(f"Failed to acquire session advisory lock: {name} ({lock_key})")
        return acquired


def advisory_unlock(conn: PgConnection, lock_key: int) -> bool:
    """
    Releases a previously acquired session-level advisory lock.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s);", (lock_key,))
        res = cur.fetchone()
        released = bool(res and res[0])
        name = LOCK_NAMES.get(lock_key, str(lock_key))
        logger.info(f"Released session advisory lock: {name} ({lock_key}) -> {released}")
        return released


def acquire_sec_owner_lock(conn: PgConnection) -> bool:
    """
    Acquires the exclusive SEC Egress Owner lock.
    Mandatory invariant: Exactly one active SEC egress owner in production.
    """
    return try_advisory_lock(conn, LOCK_SEC_EGRESS_OWNER)


def release_sec_owner_lock(conn: PgConnection) -> bool:
    """
    Releases the SEC Egress Owner lock.
    """
    return advisory_unlock(conn, LOCK_SEC_EGRESS_OWNER)


@contextmanager
def advisory_xact_lock(conn: PgConnection, lock_key: int) -> Generator[None, None, None]:
    """
    Acquires a transaction-scoped advisory lock that is automatically released upon COMMIT or ROLLBACK.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s);", (lock_key,))
        name = LOCK_NAMES.get(lock_key, str(lock_key))
        logger.debug(f"Acquired transaction advisory lock: {name}")
    yield


@contextmanager
def session_run_lock(conn: PgConnection, lock_key: int) -> Generator[bool, None, None]:
    """
    Context manager for session-level run lock. Yields True if acquired, False otherwise.
    Releases the lock on exit if acquired.
    """
    acquired = try_advisory_lock(conn, lock_key)
    try:
        yield acquired
    finally:
        if acquired:
            advisory_unlock(conn, lock_key)
