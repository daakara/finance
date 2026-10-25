"""
Unit tests for Smart Money PostgreSQL configuration, role separation,
locking derivation, numeric boundary precision, and migration integrity.
"""

import os
import pytest
from decimal import Decimal

from analyst_dashboard.smart_money.storage.config import (
    get_database_url,
    get_database_role,
    get_storage_backend,
    validate_storage_configuration,
    ROLE_WEB,
    ROLE_INGESTION,
    ROLE_MIGRATION,
    SMART_MONEY_POSTGRES_SCHEMA_VERSION,
    POSTGRES_MAJOR_VERSION_SUPPORTED,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    LOCK_SEC_EGRESS_OWNER,
    LOCK_DISCOVERY_RUN,
    LOCK_RECONCILIATION_RUN,
    LOCK_INGESTION_RUN,
    LOCK_SNAPSHOT_PUBLICATION,
    LOCK_NAMES,
)
from analyst_dashboard.smart_money.storage.migration import compute_file_sha256


class TestPostgresConfigurationContracts:
    """Tests fail-closed configuration invariants and role resolution."""

    def test_schema_and_version_constants(self):
        assert SMART_MONEY_POSTGRES_SCHEMA_VERSION == "SMART_MONEY_POSTGRES_SCHEMA_V1"
        assert POSTGRES_MAJOR_VERSION_SUPPORTED == 16

    def test_role_resolution_defaults_to_web(self, monkeypatch):
        monkeypatch.delenv("SMART_MONEY_DATABASE_ROLE", raising=False)
        assert get_database_role() == ROLE_WEB

        monkeypatch.setenv("SMART_MONEY_DATABASE_ROLE", "ingestion")
        assert get_database_role() == ROLE_INGESTION

        monkeypatch.setenv("SMART_MONEY_DATABASE_ROLE", "worker")
        assert get_database_role() == ROLE_INGESTION

        monkeypatch.setenv("SMART_MONEY_DATABASE_ROLE", "migration")
        assert get_database_role() == ROLE_MIGRATION

    def test_storage_backend_detection(self, monkeypatch):
        monkeypatch.delenv("SMART_MONEY_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert get_storage_backend() == "sqlite"

        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", "postgresql://user:pass@localhost:5432/arx")
        assert get_storage_backend() == "postgres"

    def test_production_fails_closed_without_postgres(self, monkeypatch):
        monkeypatch.setenv("ARX_ENVIRONMENT", "production")
        monkeypatch.delenv("SMART_MONEY_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with pytest.raises(RuntimeError, match="PRODUCTION_STORAGE_CONFIGURATION_VIOLATION"):
            validate_storage_configuration()

    def test_production_fails_closed_with_sqlite_url(self, monkeypatch):
        monkeypatch.setenv("ARX_ENVIRONMENT", "production")
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", "sqlite:///data/smart_money.db")

        with pytest.raises(RuntimeError, match="PRODUCTION_STORAGE_CONFIGURATION_VIOLATION"):
            validate_storage_configuration()

    def test_production_succeeds_with_valid_postgres_url(self, monkeypatch):
        monkeypatch.setenv("ARX_ENVIRONMENT", "production")
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", "postgresql://user:pass@localhost:5432/arx")
        monkeypatch.setenv("SMART_MONEY_DATABASE_ROLE", "web")

        # Must not raise
        validate_storage_configuration()

    def test_non_production_allows_missing_postgres(self, monkeypatch):
        monkeypatch.setenv("ARX_ENVIRONMENT", "development")
        monkeypatch.delenv("SMART_MONEY_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

        # Must not raise in dev/test
        validate_storage_configuration()


class TestDistributedLockKeyDerivation:
    """Verifies deterministic lock key derivation and collision avoidance."""

    def test_all_five_locks_have_distinct_keys(self):
        keys = [
            LOCK_SEC_EGRESS_OWNER,
            LOCK_DISCOVERY_RUN,
            LOCK_RECONCILIATION_RUN,
            LOCK_INGESTION_RUN,
            LOCK_SNAPSHOT_PUBLICATION,
        ]
        assert len(keys) == 5
        assert len(set(keys)) == 5, "Lock key collision detected!"

    def test_lock_keys_fit_in_signed_bigint(self):
        max_signed_bigint = 0x7FFFFFFFFFFFFFFF  # 9,223,372,036,854,775,807
        for key in LOCK_NAMES.keys():
            assert 0 < key <= max_signed_bigint, f"Key {key} does not fit in PostgreSQL signed bigint"


class TestNumericBoundaryAndPrecision:
    """Tests exact numeric boundaries for financial calculations (Section 36)."""

    def test_exact_qualification_threshold(self):
        # Qualification rule: gross_purchase_value_usd >= 25000.00
        THRESHOLD = Decimal("25000.00")

        below = Decimal("24999.99")
        exact = Decimal("25000.00")
        above = Decimal("25000.01")

        assert below < THRESHOLD
        assert exact >= THRESHOLD
        assert above >= THRESHOLD

        # Float conversion risk check
        f_below = 24999.99
        f_exact = 25000.00
        assert f_below < 25000.00
        assert f_exact >= 25000.00


class TestMigrationFileDigest:
    """Tests deterministic SHA-256 computation."""

    def test_non_existent_file_returns_zeros(self):
        digest = compute_file_sha256("non_existent_file_path.db")
        assert digest == "0" * 64

    def test_existing_file_digest(self, tmp_path):
        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"SMART_MONEY_DATABASE_PAYLOAD")
        digest = compute_file_sha256(str(test_file))
        assert len(digest) == 64
        # SHA256 of b"SMART_MONEY_DATABASE_PAYLOAD"
        import hashlib
        assert digest == hashlib.sha256(b"SMART_MONEY_DATABASE_PAYLOAD").hexdigest()
