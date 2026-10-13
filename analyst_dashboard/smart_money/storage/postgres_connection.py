"""
ARX Terminal — PostgreSQL Connection Pool Manager.

Provides thread-safe connection pooling, rational statement timeouts per role,
and transactional context managers for Smart Money services.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Generator, Optional
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

from analyst_dashboard.smart_money.storage.config import (
    get_database_url,
    get_database_role,
    ROLE_WEB,
    ROLE_INGESTION,
    ROLE_MIGRATION,
)

logger = logging.getLogger(__name__)

# Role-specific pool defaults
POOL_SETTINGS = {
    ROLE_WEB: {"minconn": 1, "maxconn": 10, "statement_timeout_ms": 30000},
    ROLE_INGESTION: {"minconn": 1, "maxconn": 5, "statement_timeout_ms": 120000},
    ROLE_MIGRATION: {"minconn": 1, "maxconn": 2, "statement_timeout_ms": 300000},
}

_pool_lock = threading.Lock()
_global_pool: Optional[ThreadedConnectionPool] = None
_pool_dsn: Optional[str] = None


def get_connection_pool(dsn: Optional[str] = None, role: Optional[str] = None) -> ThreadedConnectionPool:
    """
    Returns or initializes the global ThreadedConnectionPool.
    """
    global _global_pool, _pool_dsn
    target_dsn = dsn or get_database_url()
    if not target_dsn:
        raise ValueError("DATABASE_URL_NOT_CONFIGURED: PostgreSQL connection string required.")

    target_role = role or get_database_role()
    settings = POOL_SETTINGS.get(target_role, POOL_SETTINGS[ROLE_WEB])

    with _pool_lock:
        if _global_pool is None or _pool_dsn != target_dsn:
            if _global_pool is not None:
                try:
                    _global_pool.closeall()
                except Exception as e:
                    logger.warning(f"Error closing previous pool: {e}")

            # Ensure connect_timeout is set
            conn_args = {"connect_timeout": 10}
            _global_pool = ThreadedConnectionPool(
                minconn=settings["minconn"],
                maxconn=settings["maxconn"],
                dsn=target_dsn,
                **conn_args,
            )
            _pool_dsn = target_dsn
            logger.info(f"Initialized PostgreSQL pool for role '{target_role}' (max={settings['maxconn']})")

        return _global_pool


def close_connection_pool() -> None:
    """Closes all connections in the global pool."""
    global _global_pool, _pool_dsn
    with _pool_lock:
        if _global_pool is not None:
            try:
                _global_pool.closeall()
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
            _global_pool = None
            _pool_dsn = None


@contextmanager
def get_connection(dsn: Optional[str] = None, role: Optional[str] = None) -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Context manager yielding a connection from the pool and returning it upon exit.
    Applies role-appropriate statement_timeout.
    """
    pool = get_connection_pool(dsn=dsn, role=role)
    conn = pool.getconn()
    active_role = role or get_database_role()
    timeout_ms = POOL_SETTINGS.get(active_role, POOL_SETTINGS[ROLE_WEB])["statement_timeout_ms"]

    try:
        # Enforce statement timeout and UTC timezone per session
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)};")
            cur.execute("SET TIME ZONE 'GMT';")
        yield conn
    finally:
        pool.putconn(conn)


@contextmanager
def transaction(conn: Optional[psycopg2.extensions.connection] = None, dsn: Optional[str] = None, role: Optional[str] = None) -> Generator[psycopg2.extensions.cursor, None, None]:
    """
    Transactional context manager. Automatically commits on success or rolls back on exception.
    """
    if conn is not None:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        with get_connection(dsn=dsn, role=role) as c:
            try:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    yield cur
                c.commit()
            except Exception:
                c.rollback()
                raise


def ping_database(dsn: Optional[str] = None) -> bool:
    """Performs an active health check query."""
    try:
        with get_connection(dsn=dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                res = cur.fetchone()
                return res is not None and res[0] == 1
    except Exception as e:
        logger.warning(f"PostgreSQL ping failed: {e}")
        return False
