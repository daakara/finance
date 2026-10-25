"""
ARX Terminal — Smart Money Canonical Storage Factory.

Provides unified runtime persistence authority selection across:
- API routes
- Scanner runners
- Data access layers
- Dedicated ingestion workers
- Operator control plane

Enforces strict fail-closed backend resolution and eliminates scattered backend conditionals.
"""

import os
import logging
from typing import TYPE_CHECKING, Optional, Any

from analyst_dashboard.smart_money.storage.config import (
    get_storage_backend,
    get_database_url,
    validate_storage_configuration,
)

if TYPE_CHECKING:
    from analyst_dashboard.smart_money.storage.postgres_repository import (
        PostgresSmartMoneyRepository,
    )

logger = logging.getLogger(__name__)


def get_smart_money_storage_backend() -> str:
    """Returns the canonical storage backend ('postgres' or 'sqlite')."""
    return get_storage_backend()


def get_smart_money_repository(dsn: Optional[str] = None) -> Any:
    """
    Returns an instance of PostgresSmartMoneyRepository configured with the authoritative DSN.
    Raises RuntimeError if SMART_MONEY_DATABASE_URL is not configured,
    or if required PostgreSQL driver dependencies (psycopg2) are unavailable.
    """
    url = dsn or get_database_url()
    if not url:
        raise RuntimeError(
            "CANONICAL_POSTGRES_REPOSITORY_UNAVAILABLE: "
            "SMART_MONEY_DATABASE_URL is not configured."
        )

    try:
        from analyst_dashboard.smart_money.storage.postgres_repository import (
            PostgresSmartMoneyRepository,
            PSYCOPG2_AVAILABLE,
        )
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is not available")
    except ImportError as exc:
        raise RuntimeError(
            "CANONICAL_POSTGRES_DRIVER_UNAVAILABLE / POSTGRESQL_DRIVER_UNAVAILABLE: "
            "PostgreSQL storage backend was selected, "
            "but the required PostgreSQL client libraries (psycopg2) could not be loaded. "
            "Install psycopg2-binary to use PostgreSQL storage."
        ) from exc

    return PostgresSmartMoneyRepository(dsn=url)


def get_smart_money_snapshot_store(backend: Optional[str] = None) -> Any:
    """
    Returns the authoritative snapshot store for the current runtime profile.
    In 'postgres' profile: returns PostgresSmartMoneyRepository.
    In 'sqlite' profile: returns ScannerSnapshotStore.
    Fails closed with actionable error if postgres is selected but psycopg2 is missing.
    No silent fallback from required production PostgreSQL to ephemeral SQLite.
    """
    active_backend = backend or get_smart_money_storage_backend()
    if active_backend == "postgres":
        return get_smart_money_repository()
    elif active_backend == "sqlite":
        # Lazy import to prevent SQLite database creation during PostgreSQL profile import
        from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
        return ScannerSnapshotStore()
    else:
        raise RuntimeError(
            f"UNSUPPORTED_STORAGE_BACKEND: Unknown storage backend '{active_backend}'. "
            f"Must be 'postgres' or 'sqlite'."
        )


def get_smart_money_scanner_runner(backend: Optional[str] = None) -> Any:
    """
    Returns the canonical SmartMoneyScannerRunner configured for the active storage backend.
    """
    active_backend = backend or get_smart_money_storage_backend()
    if active_backend not in ("postgres", "sqlite"):
        raise RuntimeError(
            f"UNSUPPORTED_STORAGE_BACKEND: Unknown storage backend '{active_backend}'. "
            f"Must be 'postgres' or 'sqlite'."
        )
    from analyst_dashboard.analyzers.scanner_runner import SmartMoneyScannerRunner
    return SmartMoneyScannerRunner(backend=active_backend)
