"""
tests/test_security_master_contract.py

Comprehensive test suite for ARX Terminal Canonical Security Master.
Verifies domain models, provider adapters, normalization, conflict resolution,
eligibility policy, persistence, API contracts, PLSE resolution, negative subtypes,
provider failure, and secret safety.
"""

import json
import pytest
from fastapi.testclient import TestClient

from analyst_dashboard.security_master.models import (
    AssetClass,
    SecurityType,
    ListingStatus,
    ClassificationStatus,
    ExecutionEligibility,
    AnalyticsCapability,
    CanonicalInstrument,
)
from analyst_dashboard.security_master.alpaca_adapter import AlpacaAssetEvidence
from analyst_dashboard.security_master.openfigi_adapter import OpenFIGISubtypeEvidence
from analyst_dashboard.security_master.normalization import SecurityMasterNormalizationEngine
from analyst_dashboard.security_master.eligibility import evaluate_execution_eligibility
from analyst_dashboard.security_master.persistence import SecurityMasterRepository
from analyst_dashboard.security_master.config import (
    resolve_security_master_db_path,
    SecurityMasterFirewallError,
)
from analyst_dashboard.security_master.service import (
    SecurityMasterService,
    set_security_master_service,
)
from tests.fixtures.security_master_fixtures import (
    MockAlpacaAdapter,
    MockOpenFIGIAdapter,
    FIXTURE_EVIDENCE_REGISTRY,
)
from api.main import app


@pytest.fixture
def mock_service():
    """Provides an isolated in-memory SecurityMasterService."""
    repo = SecurityMasterRepository(db_path=":memory:")
    service = SecurityMasterService(
        alpaca_adapter=MockAlpacaAdapter(),
        openfigi_adapter=MockOpenFIGIAdapter(),
        normalization_engine=SecurityMasterNormalizationEngine(),
        repository=repo,
    )
    set_security_master_service(service)
    yield service
    set_security_master_service(None)


# 1. Canonical Domain Model Tests
def test_canonical_domain_model_structure():
    inst = CanonicalInstrument(
        symbol="PLSE",
        provider_symbol="PLSE",
        asset_class=AssetClass.EQUITY,
        security_type=SecurityType.COMMON_STOCK,
        primary_exchange="NASDAQ",
        listing_status=ListingStatus.ACTIVE,
        classification_status=ClassificationStatus.VERIFIED,
        execution_eligibility=ExecutionEligibility.STOCK_EXECUTION,
        analytics_capability=AnalyticsCapability.SUPPORTED,
        classification_authority="ARX_SERVER_SECURITY_MASTER",
        source_provenance={"test": True},
        classification_timestamp="2026-10-06T00:00:00Z",
        stable_identifiers={"figi": "BBG00BRBHVD0"},
    )
    d = inst.to_dict()
    assert d["symbol"] == "PLSE"
    assert d["asset_class"] == "EQUITY"
    assert d["security_type"] == "COMMON_STOCK"
    assert d["execution_eligibility"] == "STOCK_EXECUTION"
    # Ensure booleans like isStock/isETF are NOT fields of CanonicalInstrument
    assert "isStock" not in d
    assert "isETF" not in d
    assert "isCrypto" not in d


# 2. Alpaca Identity Adapter Invariant
def test_alpaca_broad_class_is_not_subtype_authority():
    adapter = MockAlpacaAdapter()
    ev = adapter.fetch_asset_evidence("PLSE")
    assert ev.success is True
    assert ev.broad_asset_class == AssetClass.EQUITY
    # Alpaca broad class us_equity cannot establish COMMON_STOCK subtype
    assert not hasattr(ev, "security_type")


def test_alpaca_failure_returns_unresolved():
    adapter = MockAlpacaAdapter()
    ev = adapter.fetch_asset_evidence("INVALID_XYZ_999")
    assert ev.success is False
    assert ev.listing_status == ListingStatus.UNKNOWN


# 3. OpenFIGI Subtype Adapter Normalization
def test_openfigi_subtype_normalization():
    adapter = MockOpenFIGIAdapter()
    # Known mapping
    t, status = adapter.normalize_security_type("Common Stock", None)
    assert t == SecurityType.COMMON_STOCK
    assert status == ClassificationStatus.VERIFIED

    t, status = adapter.normalize_security_type("ETP", "Mutual Fund")
    assert t == SecurityType.ETF
    assert status == ClassificationStatus.VERIFIED

    t, status = adapter.normalize_security_type("Equity WRT", "Warrant")
    assert t == SecurityType.WARRANT
    assert status == ClassificationStatus.VERIFIED

    # Unknown vocabulary must NOT default to Common Stock
    t, status = adapter.normalize_security_type("UNKNOWN_STRUCTURE_XYZ", None)
    assert t == SecurityType.UNKNOWN
    assert status == ClassificationStatus.UNVERIFIED


# 4. Normalization and Conflict Resolution
def test_material_provider_conflict_fails_closed():
    engine = SecurityMasterNormalizationEngine()
    # Alpaca says crypto, OpenFIGI says Common Stock Equity
    alpaca_ev = FIXTURE_EVIDENCE_REGISTRY["CONFLICTED_MOCK"]["alpaca"]
    figi_ev = FIXTURE_EVIDENCE_REGISTRY["CONFLICTED_MOCK"]["openfigi"]

    inst = engine.normalize("CONFLICTED_MOCK", alpaca_ev, figi_ev)
    assert inst.classification_status == ClassificationStatus.CONFLICTED
    assert inst.execution_eligibility == ExecutionEligibility.FAIL_CLOSED
    assert inst.source_provenance["conflict_detected"] is True


def test_missing_subtype_marks_unverified():
    engine = SecurityMasterNormalizationEngine()
    alpaca_ev = FIXTURE_EVIDENCE_REGISTRY["NO_SUBTYPE_MOCK"]["alpaca"]
    figi_ev = FIXTURE_EVIDENCE_REGISTRY["NO_SUBTYPE_MOCK"]["openfigi"]

    inst = engine.normalize("NO_SUBTYPE_MOCK", alpaca_ev, figi_ev)
    assert inst.classification_status == ClassificationStatus.UNVERIFIED
    assert inst.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


# 5. Execution Eligibility Policy Engine
def test_execution_eligibility_matrix():
    # 1. Verified Common Stock + Active -> STOCK_EXECUTION
    assert evaluate_execution_eligibility(
        AssetClass.EQUITY, SecurityType.COMMON_STOCK, ListingStatus.ACTIVE, ClassificationStatus.VERIFIED
    ) == ExecutionEligibility.STOCK_EXECUTION

    # 2. Inactive Common Stock -> FAIL_CLOSED
    assert evaluate_execution_eligibility(
        AssetClass.EQUITY, SecurityType.COMMON_STOCK, ListingStatus.INACTIVE, ClassificationStatus.VERIFIED
    ) == ExecutionEligibility.FAIL_CLOSED

    # 3. Verified ETF + Active -> ETF_EXECUTION
    assert evaluate_execution_eligibility(
        AssetClass.ETF, SecurityType.ETF, ListingStatus.ACTIVE, ClassificationStatus.VERIFIED
    ) == ExecutionEligibility.ETF_EXECUTION

    # 4. All unsupported subtypes fail closed even if Active and Verified
    unsupported = [
        SecurityType.ADR,
        SecurityType.REIT,
        SecurityType.PREFERRED,
        SecurityType.WARRANT,
        SecurityType.UNIT,
        SecurityType.RIGHT,
        SecurityType.OTHER,
        SecurityType.UNKNOWN,
    ]
    for sub in unsupported:
        res = evaluate_execution_eligibility(
            AssetClass.EQUITY, sub, ListingStatus.ACTIVE, ClassificationStatus.VERIFIED
        )
        assert res == ExecutionEligibility.FAIL_CLOSED, f"{sub} must fail closed"

    # 5. Non-verified or Conflicted -> FAIL_CLOSED
    assert evaluate_execution_eligibility(
        AssetClass.EQUITY, SecurityType.COMMON_STOCK, ListingStatus.ACTIVE, ClassificationStatus.CONFLICTED
    ) == ExecutionEligibility.FAIL_CLOSED

    assert evaluate_execution_eligibility(
        AssetClass.EQUITY, SecurityType.COMMON_STOCK, ListingStatus.ACTIVE, ClassificationStatus.UNVERIFIED
    ) == ExecutionEligibility.FAIL_CLOSED


# 6. Persistence & Cache Tests
def test_persistence_hybrid_lru_and_provenance():
    repo = SecurityMasterRepository(db_path=":memory:", ttl_seconds=60.0)
    inst = CanonicalInstrument(
        symbol="TEST",
        provider_symbol="TEST",
        asset_class=AssetClass.EQUITY,
        security_type=SecurityType.COMMON_STOCK,
        primary_exchange="NASDAQ",
        listing_status=ListingStatus.ACTIVE,
        classification_status=ClassificationStatus.VERIFIED,
        execution_eligibility=ExecutionEligibility.STOCK_EXECUTION,
        analytics_capability=AnalyticsCapability.SUPPORTED,
        classification_authority="ARX_SERVER_SECURITY_MASTER",
        source_provenance={"raw_evidence": "verified"},
        classification_timestamp="2026-10-06T00:00:00Z",
        stable_identifiers={"figi": "BBG_TEST_FIGI"},
    )
    repo.save(inst, now_epoch=1000.0)

    # Cache hit
    hit = repo.get("TEST", now_epoch=1010.0)
    assert hit is not None
    assert hit.symbol == "TEST"
    assert hit.stable_identifiers["figi"] == "BBG_TEST_FIGI"
    assert hit.source_provenance["raw_evidence"] == "verified"

    # Evict LRU to test SQLite retrieval
    repo.clear_cache()
    from_db = repo.get("TEST", now_epoch=1010.0)
    assert from_db is not None
    assert from_db.execution_eligibility == ExecutionEligibility.STOCK_EXECUTION
    assert from_db.stable_identifiers["figi"] == "BBG_TEST_FIGI"

    # Test TTL expiration
    stale = repo.get("TEST", now_epoch=1100.0)
    assert stale is None  # Exceeded 60s TTL


def test_firewall_prohibits_etf_v2_db_reuse():
    with pytest.raises(SecurityMasterFirewallError):
        resolve_security_master_db_path("data/operational/openfigi_operational.db")

    with pytest.raises(SecurityMasterFirewallError):
        resolve_security_master_db_path("data/canonical/etf_v2_canonical_population.db")


# 7. Acceptance Cases: PLSE & Negative Subtypes
def test_plse_acceptance_case(mock_service):
    inst = mock_service.get_or_resolve_instrument("PLSE")
    assert inst.symbol == "PLSE"
    assert inst.asset_class == AssetClass.EQUITY
    assert inst.security_type == SecurityType.COMMON_STOCK
    assert inst.listing_status == ListingStatus.ACTIVE
    assert inst.classification_status == ClassificationStatus.VERIFIED
    assert inst.execution_eligibility == ExecutionEligibility.STOCK_EXECUTION


def test_negative_subtypes_acceptance_cases(mock_service):
    # SPY -> ETF_EXECUTION
    spy = mock_service.get_or_resolve_instrument("SPY")
    assert spy.execution_eligibility == ExecutionEligibility.ETF_EXECUTION

    # TSM (ADR) -> FAIL_CLOSED
    tsm = mock_service.get_or_resolve_instrument("TSM")
    assert tsm.security_type == SecurityType.ADR
    assert tsm.execution_eligibility == ExecutionEligibility.FAIL_CLOSED

    # O (REIT) -> FAIL_CLOSED
    o = mock_service.get_or_resolve_instrument("O")
    assert o.security_type == SecurityType.REIT
    assert o.execution_eligibility == ExecutionEligibility.FAIL_CLOSED

    # CORZW (Warrant) -> FAIL_CLOSED
    corzw = mock_service.get_or_resolve_instrument("CORZW")
    assert corzw.security_type == SecurityType.WARRANT
    assert corzw.execution_eligibility == ExecutionEligibility.FAIL_CLOSED

    # BAC.PRK (Preferred) -> FAIL_CLOSED
    pref = mock_service.get_or_resolve_instrument("BAC.PRK")
    assert pref.security_type == SecurityType.PREFERRED
    assert pref.execution_eligibility == ExecutionEligibility.FAIL_CLOSED

    # Inactive common stock -> FAIL_CLOSED
    inactive = mock_service.get_or_resolve_instrument("INACTIVE_MOCK")
    assert inactive.listing_status == ListingStatus.INACTIVE
    assert inactive.execution_eligibility == ExecutionEligibility.FAIL_CLOSED

    # Invalid symbol -> FAIL_CLOSED
    inv = mock_service.get_or_resolve_instrument("INVALID_XYZ_999")
    assert inv.classification_status == ClassificationStatus.UNVERIFIED
    assert inv.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


# 8. Provider Outage Test
def test_provider_outage_fails_closed(mock_service):
    outage = mock_service.get_or_resolve_instrument("PROVIDER_OUTAGE_MOCK")
    assert outage.classification_status == ClassificationStatus.UNVERIFIED
    assert outage.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


# 9. Secret Redaction Test
def test_provenance_contains_no_secrets(mock_service):
    plse = mock_service.get_or_resolve_instrument("PLSE")
    prov_str = json.dumps(plse.source_provenance).lower()
    assert "api_key" not in prov_str
    assert "secret" not in prov_str
    assert "bearer" not in prov_str
    assert "authorization" not in prov_str


# 10. API Route Tests: /api/v1/market/instruments/{symbol}
def test_market_instruments_api_endpoint(mock_service):
    client = TestClient(app)
    resp = client.get("/api/v1/market/instruments/PLSE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "PLSE"
    assert data["asset_class"] == "EQUITY"
    assert data["security_type"] == "COMMON_STOCK"
    assert data["execution_eligibility"] == "STOCK_EXECUTION"
    assert data["classification_authority"] == "ARX_SERVER_SECURITY_MASTER"
    assert "isStock" not in data


def test_market_instruments_api_negative_subtype(mock_service):
    client = TestClient(app)
    resp = client.get("/api/v1/market/instruments/CORZW")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "CORZW"
    assert data["security_type"] == "WARRANT"
    assert data["execution_eligibility"] == "FAIL_CLOSED"
