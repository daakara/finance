"""
ARX Terminal — Smart Money Production Database Configuration Contract.

Defines explicit configuration contracts, fail-closed production rules,
role identities, and supported PostgreSQL version standards.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SMART_MONEY_POSTGRES_SCHEMA_VERSION = "SMART_MONEY_POSTGRES_SCHEMA_V1"

# Explicitly pinned PostgreSQL major version (Option A — Pin 16)
SMART_MONEY_POSTGRES_MAJOR = 16
POSTGRES_MAJOR_VERSION_SUPPORTED = 16
PRODUCTION_TARGET_POSTGRES_MAJOR = 16
RAILWAY_POSTGRES_IMAGE_TARGET = "postgres:16-alpine"

# Canonical role identifiers
ROLE_WEB = "smart_money_web"
ROLE_INGESTION = "smart_money_ingestion"
ROLE_MIGRATION = "smart_money_migration"
VALID_ROLES = {ROLE_WEB, ROLE_INGESTION, ROLE_MIGRATION}


def get_database_url() -> Optional[str]:
    """
    Returns the authoritative PostgreSQL connection URL for Smart Money.
    Checks SMART_MONEY_DATABASE_URL first, then DATABASE_URL.
    """
    return os.environ.get("SMART_MONEY_DATABASE_URL") or os.environ.get("DATABASE_URL")


def get_database_role() -> str:
    """
    Returns the active database principal role. Defaults to smart_money_web for least-privilege.
    """
    role = os.environ.get("SMART_MONEY_DATABASE_ROLE", ROLE_WEB).strip().lower()
    if role in {"web", ROLE_WEB}:
        return ROLE_WEB
    if role in {"ingestion", "worker", ROLE_INGESTION}:
        return ROLE_INGESTION
    if role in {"migration", ROLE_MIGRATION}:
        return ROLE_MIGRATION
    return ROLE_WEB


def get_storage_backend() -> str:
    """
    Determines whether the subsystem is operating against PostgreSQL or local SQLite.
    Follows ARX_STORAGE_BACKEND with strict fail-closed validation.
    In production environments, SQLite is strictly disallowed.
    """
    configured_backend = os.environ.get("ARX_STORAGE_BACKEND", "").strip().lower()
    env = os.environ.get("ARX_ENVIRONMENT", "").strip().lower()
    db_url = get_database_url()
    has_valid_pg_url = bool(db_url and (db_url.startswith("postgres://") or db_url.startswith("postgresql://")))

    if configured_backend == "postgres":
        if not has_valid_pg_url:
            raise RuntimeError(
                "POSTGRES_STORAGE_BACKEND_CONFIGURATION_VIOLATION: "
                "ARX_STORAGE_BACKEND is set to 'postgres', but SMART_MONEY_DATABASE_URL "
                "is missing or invalid. Ephemeral SQLite fallback is strictly prohibited."
            )
        return "postgres"

    if configured_backend == "sqlite":
        if env == "production":
            raise RuntimeError(
                "PRODUCTION_STORAGE_CONFIGURATION_VIOLATION: "
                "ARX_STORAGE_BACKEND='sqlite' is strictly prohibited in production environment."
            )
        return "sqlite"

    # If ARX_STORAGE_BACKEND is not explicitly set:
    if env == "production":
        if not has_valid_pg_url:
            raise RuntimeError(
                "PRODUCTION_STORAGE_CONFIGURATION_VIOLATION: "
                "SMART_MONEY_DATABASE_URL is strictly required in production environment. "
                "Ephemeral SQLite fallback is strictly prohibited."
            )
        return "postgres"

    # Default for local non-production development
    if has_valid_pg_url:
        return "postgres"
    return "sqlite"


def validate_storage_configuration() -> None:
    """
    Enforces fail-closed configuration invariants:
    1. In production (ARX_ENVIRONMENT=production), PostgreSQL is mandatory.
    2. Missing configuration or attempted SQLite fallback in production raises RuntimeError.
    3. Invalid role raises ValueError.
    """
    env = os.environ.get("ARX_ENVIRONMENT", "").strip().lower()
    backend = get_storage_backend()
    db_url = get_database_url()

    if env == "production":
        if backend != "postgres":
            raise RuntimeError(
                "PRODUCTION_STORAGE_CONFIGURATION_VIOLATION: "
                "Production environment requires PostgreSQL backend."
            )
        if not db_url:
            raise RuntimeError(
                "PRODUCTION_STORAGE_CONFIGURATION_VIOLATION: "
                "SMART_MONEY_DATABASE_URL is strictly required in production environment. "
                "Ephemeral SQLite fallback is strictly prohibited."
            )
        if not (db_url.startswith("postgres://") or db_url.startswith("postgresql://")):
            raise RuntimeError(
                "PRODUCTION_STORAGE_CONFIGURATION_VIOLATION: "
                "SMART_MONEY_DATABASE_URL must be a valid PostgreSQL connection string in production. "
                f"Invalid URI scheme detected."
            )

    role = get_database_role()
    if role not in VALID_ROLES:
        raise ValueError(f"INVALID_DATABASE_ROLE: '{role}' is not a valid Smart Money principal role.")
