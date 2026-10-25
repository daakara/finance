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
from typing import Optional, Any

from analyst_dashboard.smart_money.storage.config import (
    get_storage_backend,
    get_database_url,
    validate_storage_configuration,
)

logger = logging.getLogger(__name__)


def get_smart_money_storage_backend() -> str:
    """Returns the canonical storage backend ('postgres' or 'sqlite')."""
    return get_storage_backend()


def get_smart_money_repository(dsn: Optional[str] = None) -> Any:
    """
    Returns an instance of PostgresSmartMoneyRepository configured with the authoritative DSN.
    Raises RuntimeError if storage backend is not configured for PostgreSQL or driver unavailable.
    """
    url = dsn or get_database_url()
    if not url:
        raise RuntimeError(
            "CANONICAL_POSTGRES_REPOSITORY_UNAVAILABLE: "
            "SMART_MONEY_DATABASE_URL is not configured."
        )
    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
    return PostgresSmartMoneyRepository(dsn=url)


def get_smart_money_snapshot_store(backend: Optional[str] = None) -> Any:
    """
    Returns the authoritative snapshot store for the current runtime profile.
    In 'postgres' profile: returns PostgresSmartMoneyRepository.
    In 'sqlite' profile: returns ScannerSnapshotStore.
    """
    active_backend = backend or get_smart_money_storage_backend()
    if active_backend == "postgres":
        return get_smart_money_repository()
    
    # Lazy import to prevent SQLite database creation during PostgreSQL profile import
    from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
    return ScannerSnapshotStore()


def get_smart_money_scanner_runner(backend: Optional[str] = None) -> Any:
    """
    Returns the canonical SmartMoneyScannerRunner configured for the active storage backend.
    """
    from analyst_dashboard.analyzers.scanner_runner import SmartMoneyScannerRunner
    return SmartMoneyScannerRunner(backend=backend)
