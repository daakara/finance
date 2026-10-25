"""
ARX Terminal — Smart Money Storage Subsystem.
Provides PostgreSQL operational persistence, schema management, distributed coordination,
and migration tooling for the market-wide SEC Form 4 pipeline.
"""

from typing import TYPE_CHECKING

from analyst_dashboard.smart_money.storage.config import (
    SMART_MONEY_POSTGRES_SCHEMA_VERSION,
    POSTGRES_MAJOR_VERSION_SUPPORTED,
    SMART_MONEY_POSTGRES_MAJOR,
    PRODUCTION_TARGET_POSTGRES_MAJOR,
    RAILWAY_POSTGRES_IMAGE_TARGET,
    get_storage_backend,
    get_database_url,
    get_database_role,
    validate_storage_configuration,
)
from analyst_dashboard.smart_money.storage.factory import (
    get_smart_money_storage_backend,
    get_smart_money_repository,
    get_smart_money_snapshot_store,
    get_smart_money_scanner_runner,
)

if TYPE_CHECKING:
    from analyst_dashboard.smart_money.storage.migration import (
        SmartMoneySqliteMigrator,
        SourceIntegrityConflictError,
    )


def __getattr__(name: str):
    if name in ("SmartMoneySqliteMigrator", "SourceIntegrityConflictError"):
        from analyst_dashboard.smart_money.storage import migration
        return getattr(migration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SMART_MONEY_POSTGRES_SCHEMA_VERSION",
    "POSTGRES_MAJOR_VERSION_SUPPORTED",
    "SMART_MONEY_POSTGRES_MAJOR",
    "PRODUCTION_TARGET_POSTGRES_MAJOR",
    "RAILWAY_POSTGRES_IMAGE_TARGET",
    "get_storage_backend",
    "get_database_url",
    "get_database_role",
    "validate_storage_configuration",
    "get_smart_money_storage_backend",
    "get_smart_money_repository",
    "get_smart_money_snapshot_store",
    "get_smart_money_scanner_runner",
    "SmartMoneySqliteMigrator",
    "SourceIntegrityConflictError",
]
