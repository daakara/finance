"""
ARX Terminal — Smart Money Release Provenance and Conflict Detection Test Suite.

Verifies:
1. resolve_release_identity fail-closed resolution.
2. Conflict detection between ARX_RELEASE_SHA and RAILWAY_GIT_COMMIT_SHA.
3. Format validation (strict 40-char hex).
4. require_continuous_authorization rejection of conflicting/unverified release identities.
5. Repository receipt release_sha fail-closed enforcement (no silent hardcoded fallback).
"""

from __future__ import annotations

import os
import re
import pytest
from unittest.mock import patch, MagicMock

from analyst_dashboard.smart_money.continuous_authorization import (
    resolve_release_identity,
    require_continuous_authorization,
    SCOPE_TRIAL_2,
)
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.canary_runner import NvdaCanaryRunner


VALID_SHA_A = "16549be1acb9fdafef3c6b97f9115b674f9f4842"
VALID_SHA_B = "b485e44141fc2435c33c71eb7e4af107b02fa129"
INVALID_SHA = "not-a-valid-sha"


class TestReleaseProvenanceResolution:
    """Verifies resolve_release_identity behavior across environment permutations."""

    def test_matching_arx_and_railway_shas_accepted(self):
        env = {
            "ARX_RELEASE_SHA": VALID_SHA_A,
            "RAILWAY_GIT_COMMIT_SHA": VALID_SHA_A.upper(),  # case-insensitive match
        }
        resolved = resolve_release_identity(env)
        assert resolved == VALID_SHA_A.lower()

    def test_conflicting_arx_and_railway_shas_rejected(self):
        env = {
            "ARX_RELEASE_SHA": VALID_SHA_B,
            "RAILWAY_GIT_COMMIT_SHA": VALID_SHA_A,
        }
        with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_CONFLICT"):
            resolve_release_identity(env)

    def test_invalid_arx_release_sha_rejected(self):
        env = {"ARX_RELEASE_SHA": INVALID_SHA}
        with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_INVALID"):
            resolve_release_identity(env)

    def test_invalid_railway_git_commit_sha_rejected(self):
        env = {"RAILWAY_GIT_COMMIT_SHA": INVALID_SHA}
        with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_INVALID"):
            resolve_release_identity(env)

    def test_single_railway_git_commit_sha_accepted(self):
        env = {"RAILWAY_GIT_COMMIT_SHA": VALID_SHA_A}
        assert resolve_release_identity(env) == VALID_SHA_A

    def test_single_arx_release_sha_accepted(self):
        env = {"ARX_RELEASE_SHA": VALID_SHA_A}
        assert resolve_release_identity(env) == VALID_SHA_A

    def test_git_env_sha_fallback_accepted(self):
        env = {"GIT_COMMIT_SHA": VALID_SHA_A}
        assert resolve_release_identity(env) == VALID_SHA_A

    def test_git_repo_fallback_accepted(self):
        # Empty env falls back to local git repo (which exists in test environment)
        resolved = resolve_release_identity({})
        assert re.fullmatch(r"^[0-9a-f]{40}$", resolved)

    def test_missing_identity_fails_closed(self):
        with patch("subprocess.run", side_effect=Exception("Git unavailable")):
            with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_MISSING"):
                resolve_release_identity({})


class TestContinuousAuthorizationProvenanceIntegration:
    """Verifies that require_continuous_authorization catches conflicting release provenance."""

    def test_auth_fails_closed_on_conflicting_release_provenance(self):
        env = {
            "ARX_SEC_EGRESS_ROLE": "OWNER",
            "ARX_CONTINUOUS_INGESTION_AUTHORIZED": "true",
            "ARX_CONTINUOUS_AUTH_SCOPE": SCOPE_TRIAL_2,
            "ARX_CONTINUOUS_AUTH_UNIVERSE_ID": "UB-US-EQUITIES-20261025-001",
            "ARX_CONTINUOUS_AUTH_ID": "b" * 32,
            "ARX_CONTINUOUS_AUTH_RELEASE_SHA": VALID_SHA_A,
            "ARX_RELEASE_SHA": VALID_SHA_B,  # stale value
            "RAILWAY_GIT_COMMIT_SHA": VALID_SHA_A,  # authoritative value
            "ARX_CONTINUOUS_AUTH_ISSUED_AT": "2026-11-20T10:00:00Z",
            "ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-21T10:00:00Z",
        }
        with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_CONFLICT"):
            require_continuous_authorization(
                env=env,
                issuers=[
                    {"ticker": "NVDA", "cik": "0001045810"},
                    {"ticker": "AAPL", "cik": "0000320193"},
                ],
            )


class TestRepositoryProvenanceEnforcement:
    """Verifies that PostgresSmartMoneyRepository rejects invalid or conflicting release SHA."""

    def test_save_reconciliation_receipt_rejects_invalid_release_sha(self):
        repo = PostgresSmartMoneyRepository(dsn=None)
        receipt_payload = {
            "receipt_id": "test-id",
            "interval_start": "2026-11-19T00:00:00Z",
            "interval_end": "2026-11-19T23:59:59Z",
            "certified_watermark": "2026-11-19T23:59:59Z",
            "expected_count": 1,
            "accounted_count": 1,
            "missing_count": 0,
            "material_gap_count": 0,
            "coverage_status": "PARTIAL",
            "failure_reasons": [],
            "reconciliation_hash": "a" * 64,
            "release_sha": INVALID_SHA,
        }
        with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_INVALID"):
            repo.save_reconciliation_receipt(receipt_payload)

    def test_record_issuer_coverage_receipt_rejects_invalid_release_sha(self):
        repo = PostgresSmartMoneyRepository(dsn=None)
        with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_INVALID"):
            repo.record_issuer_coverage_receipt(
                ticker="NVDA",
                issuer_cik="0001045810",
                interval_start="2026-11-19T00:00:00Z",
                interval_end="2026-11-19T23:59:59Z",
                certified_watermark="2026-11-19T23:59:59Z",
                release_sha=INVALID_SHA,
            )

    def test_record_issuer_coverage_receipt_fails_on_conflicting_env(self, monkeypatch):
        monkeypatch.setenv("ARX_RELEASE_SHA", VALID_SHA_B)
        monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", VALID_SHA_A)
        repo = PostgresSmartMoneyRepository(dsn=None)
        with pytest.raises(RuntimeError, match="RELEASE_PROVENANCE_CONFLICT"):
            repo.record_issuer_coverage_receipt(
                ticker="NVDA",
                issuer_cik="0001045810",
                interval_start="2026-11-19T00:00:00Z",
                interval_end="2026-11-19T23:59:59Z",
                certified_watermark="2026-11-19T23:59:59Z",
                release_sha=None,  # triggers runtime resolution
            )


class TestCanaryRunnerProvenanceBinding:
    """Verifies that NvdaCanaryRunner stores and passes release_sha."""

    def test_runner_accepts_and_stores_release_sha(self):
        runner = NvdaCanaryRunner(
            release_sha=VALID_SHA_A,
            ticker="NVDA",
            cik="0001045810",
        )
        assert runner.release_sha == VALID_SHA_A
