"""
ARX Terminal — Smart Money Storage Subsystem.
Provides PostgreSQL operational persistence, schema management, distributed coordination,
and migration tooling for the market-wide SEC Form 4 pipeline.
"""

from analyst_dashboard.smart_money.storage.config import (
    SMART_MONEY_POSTGRES_SCHEMA_VERSION,
    POSTGRES_MAJOR_VERSION_SUPPORTED,
    get_storage_backend,
    validate_storage_configuration,
)

__all__ = [
    "SMART_MONEY_POSTGRES_SCHEMA_VERSION",
    "POSTGRES_MAJOR_VERSION_SUPPORTED",
    "get_storage_backend",
    "validate_storage_configuration",
]
