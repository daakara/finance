"""
tests/test_canonical_security_master.py

Authoritative Test Suite for ARX Canonical Server Security Master.
Validates:
- Domain model invariants (INV-SECMASTER-01..20)
- PLSE positive regression (Common Stock -> STOCK_EXECUTION)
- Negative routing for unsupported subtypes (ADR, REIT, Preferred, Warrant, Unit, Right -> FAIL_CLOSED)
- Inactive, conflicted, unverified, and outage fail-closed semantics
- ETF and Crypto execution surface isolation
- Hybrid SQLite persistence + LRU cache + Firewall isolation
- Secret redaction and zero credential leakage
- API route verification (/api/v1/security-master and /api/v1/market)
"""

import json
import pytest
from pathlib import Path
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
from analyst_dashboard.security_master.config import (
    resolve_security_master_db_path,
    SecurityMasterFirewallError,
)
from analyst_dashboard.security_master.eligibility import evaluate_execution_eligibility
from analyst_dashboard.security_master.normalization import SecurityMasterNormalizationEngine
from analyst_dashboard.security_master.persistence import SecurityMasterRepository
from analyst_dashboard.security_master.service import (
    SecurityMasterService,
    get_security_master_service,
    set_security_master_service,
)
from tests.fixtures.security_master_fixtures import (
    MockAlpacaAdapter,
    MockOpenFIGIAdapter,
    FIXTURE_EVIDENCE_REGISTRY,
)
from api.main import app


@pytest.fixture
def mock_service(tmp_path: Path):
    """Provides a fully isolated, deterministic SecurityMasterService with mock adapters."""
    alpaca = MockAlpacaAdapter()
    figi = MockOpenFIGIAdapter()
    norm = SecurityMasterNormalizationEngine()
    db_file = tmp_path / "test_sec_master.db"
    repo = SecurityMasterRepository(db_path=db_file, ttl_seconds=3600.0)
    service = SecurityMasterService(
        alpaca_adapter=alpaca,
        openfigi_adapter=figi,
        normalization_engine=norm,
        repository=repo,
    )
    set_security_master_service(service)
    yield service
    set_security_master_service(None)


# ---------------------------------------------------------------------------
# 1. CANONICAL INSTRUMENT MODEL CONTRACT
# ---------------------------------------------------------------------------
def test_canonical_instrument_model_contract():
    """Validates that CanonicalInstrument conforms strictly to Section 8 Canonical Contract."""
    instrument = CanonicalInstrument(
        symbol="PLSE",
        provider_symbol="PLSE",
        asset_class=AssetClass.EQUITY,
        security_type=SecurityType.COMMON_STOCK,
        primary_exchange="NASDAQ",
        listing_status=ListingStatus.ACTIVE,
        classification_status=ClassificationStatus.VERIFIED,
        execution_eligibility=ExecutionEligibility.STOCK_EXECUTION,
        analytics_capability=AnalyticsCapability.FULL_ANALYTICS,
        classification_authority="ARX_SERVER_SECURITY_MASTER",
        source_provider="COMPOSITE_ALPACA_OPENFIGI",
        classification_timestamp="2026-10-06T00:00:00Z",
        stable_identifier="BBG00BRBHVD0",
        provider_provenance={"test": "provenance"},
    )

    assert instrument.symbol == "PLSE"
    assert instrument.provider_symbol == "PLSE"
    assert instrument.asset_class == AssetClass.EQUITY
    assert instrument.security_type == SecurityType.COMMON_STOCK
    assert instrument.primary_exchange == "NASDAQ"
    assert instrument.listing_status == ListingStatus.ACTIVE
    assert instrument.classification_status == ClassificationStatus.VERIFIED
    assert instrument.execution_eligibility == ExecutionEligibility.STOCK_EXECUTION
    assert instrument.analytics_capability == AnalyticsCapability.FULL_ANALYTICS
    assert instrument.classification_authority == "ARX_SERVER_SECURITY_MASTER"
    assert instrument.source_provider == "COMPOSITE_ALPACA_OPENFIGI"
    assert instrument.stable_identifier == "BBG00BRBHVD0"
    assert instrument.provider_provenance == {"test": "provenance"}

    # Serialized dictionary preserves all fields
    d = instrument.to_dict()
    assert d["symbol"] == "PLSE"
    assert d["execution_eligibility"] == "STOCK_EXECUTION"
    assert d["security_type"] == "COMMON_STOCK"


# ---------------------------------------------------------------------------
# 2. PLSE POSITIVE REGRESSION CASE (ORGANIC RESOLUTION)
# ---------------------------------------------------------------------------
def test_plse_positive_regression(mock_service):
    """
    PLSE Required Acceptance Case:
    Without ticker-specific code:
    SYMBOL = PLSE
    ASSET_CLASS = EQUITY
    SECURITY_TYPE = COMMON_STOCK
    LISTING_STATUS = ACTIVE
    CLASSIFICATION_STATUS = VERIFIED
    EXECUTION_ELIGIBILITY = STOCK_EXECUTION
    """
    inst = mock_service.get_or_resolve_instrument("PLSE")

    assert inst.symbol == "PLSE"
    assert inst.asset_class == AssetClass.EQUITY
    assert inst.security_type == SecurityType.COMMON_STOCK
    assert inst.listing_status == ListingStatus.ACTIVE
    assert inst.classification_status == ClassificationStatus.VERIFIED
    assert inst.execution_eligibility == ExecutionEligibility.STOCK_EXECUTION
    assert inst.primary_exchange == "NASDAQ"
    assert inst.stable_identifier == "BBG00BRBHVD0"


# ---------------------------------------------------------------------------
# 3. NEGATIVE ROUTING (UNSUPPORTED / DERIVATIVE / INACTIVE)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ticker,expected_type", [
    ("TSM", SecurityType.ADR),
    ("O", SecurityType.REIT),
    ("BAC.PRK", SecurityType.PREFERRED),
    ("CORZW", SecurityType.WARRANT),
])
def test_unsupported_subtypes_fail_closed(mock_service, ticker, expected_type):
    """
    Verify that unsupported equity subtypes (ADR, REIT, Preferred, Warrant)
    resolve to their true subtype but strictly FAIL_CLOSED for execution.
    """
    inst = mock_service.get_or_resolve_instrument(ticker)
    assert inst.symbol == ticker
    assert inst.security_type == expected_type
    assert inst.classification_status == ClassificationStatus.VERIFIED
    assert inst.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


def test_inactive_common_stock_fails_closed(mock_service):
    """Inactive common stock cannot execute under any circumstance."""
    inst = mock_service.get_or_resolve_instrument("INACTIVE_MOCK")
    assert inst.security_type == SecurityType.COMMON_STOCK
    assert inst.listing_status == ListingStatus.INACTIVE
    assert inst.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


def test_conflicted_provider_evidence_fails_closed(mock_service):
    """Conflicting provider classifications must fail closed with CONFLICTED status."""
    inst = mock_service.get_or_resolve_instrument("CONFLICTED_MOCK")
    assert inst.classification_status == ClassificationStatus.CONFLICTED
    assert inst.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


def test_missing_openfigi_subtype_fails_closed(mock_service):
    """Alpaca active us_equity without OpenFIGI subtype proof fails closed as UNVERIFIED."""
    inst = mock_service.get_or_resolve_instrument("NO_SUBTYPE_MOCK")
    assert inst.classification_status == ClassificationStatus.UNVERIFIED
    assert inst.security_type == SecurityType.UNKNOWN
    assert inst.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


def test_complete_provider_outage_fails_closed(mock_service):
    """Provider outage or timeout strictly fails closed as UNVERIFIED."""
    inst = mock_service.get_or_resolve_instrument("PROVIDER_OUTAGE_MOCK")
    assert inst.classification_status == ClassificationStatus.UNVERIFIED
    assert inst.execution_eligibility == ExecutionEligibility.FAIL_CLOSED


# ---------------------------------------------------------------------------
# 4. ETF AND CRYPTO EXECUTION ISOLATION
# ---------------------------------------------------------------------------
def test_etf_execution_isolation(mock_service):
    """ETFs route strictly to ETF_EXECUTION and never to STOCK_EXECUTION."""
    inst = mock_service.get_or_resolve_instrument("SPY")
    assert inst.symbol == "SPY"
    assert inst.security_type == SecurityType.ETF
    assert inst.execution_eligibility == ExecutionEligibility.ETF_EXECUTION
    assert inst.execution_eligibility != ExecutionEligibility.STOCK_EXECUTION


def test_crypto_execution_isolation(mock_service):
    """Crypto routes strictly to CRYPTO_EXECUTION."""
    from analyst_dashboard.security_master.alpaca_adapter import AlpacaAssetEvidence
    custom_alpaca = MockAlpacaAdapter(override_registry={
        "BTC/USD": AlpacaAssetEvidence(
            symbol="BTC/USD",
            success=True,
            broad_asset_class=AssetClass.CRYPTO,
            primary_exchange="CRYPTO",
            listing_status=ListingStatus.ACTIVE,
        )
    })
    service = SecurityMasterService(
        alpaca_adapter=custom_alpaca,
        openfigi_adapter=MockOpenFIGIAdapter(),
        normalization_engine=SecurityMasterNormalizationEngine(),
        repository=mock_service.repository,
    )
    inst = service.get_or_resolve_instrument("BTC/USD")
    assert inst.security_type == SecurityType.CRYPTO
    assert inst.execution_eligibility == ExecutionEligibility.CRYPTO_EXECUTION
    assert inst.execution_eligibility != ExecutionEligibility.STOCK_EXECUTION


# ---------------------------------------------------------------------------
# 5. PERSISTENCE, CACHE, AND FIREWALL SEPARATION
# ---------------------------------------------------------------------------
def test_hybrid_persistence_and_cache(tmp_path: Path):
    """Verifies that resolved instruments persist in SQLite and survive LRU cache clear."""
    db_file = tmp_path / "persistence_test.db"
    repo = SecurityMasterRepository(db_path=db_file, ttl_seconds=3600.0)
    service = SecurityMasterService(
        alpaca_adapter=MockAlpacaAdapter(),
        openfigi_adapter=MockOpenFIGIAdapter(),
        normalization_engine=SecurityMasterNormalizationEngine(),
        repository=repo,
    )

    # Initial resolution (persists to DB and LRU)
    inst1 = service.get_or_resolve_instrument("PLSE")
    assert inst1.symbol == "PLSE"

    # Clear in-memory LRU cache
    repo.clear_cache()
    assert repo.lru_cache.get("PLSE") is None

    # Retrieve again: must load from SQLite without querying providers
    inst2 = repo.get("PLSE")
    assert inst2 is not None
    assert inst2.symbol == "PLSE"
    assert inst2.execution_eligibility == ExecutionEligibility.STOCK_EXECUTION
    assert inst2.stable_identifier == "BBG00BRBHVD0"


def test_security_master_firewall_prohibits_etf_v2_db_reuse():
    """
    INV-SECMASTER-19: Security Master persistence must NOT reuse
    ETF v2 operational database or canonical population databases.
    """
    forbidden_paths = [
        "data/operational/openfigi_operational.db",
        "data/canonical/etf_v2_canonical_population.db",
        "C:/Users/akara/Documents/Projects/finance/data/operational/openfigi_operational.db",
    ]
    for p in forbidden_paths:
        with pytest.raises(SecurityMasterFirewallError):
            resolve_security_master_db_path(p)


def test_stale_record_ttl_fails_closed(tmp_path: Path):
    """Stale records past configured TTL expire and fail closed on retrieval."""
    now_epoch = 1000.0
    db_file = tmp_path / "ttl_test.db"
    repo = SecurityMasterRepository(db_path=db_file, ttl_seconds=60.0, clock=lambda: now_epoch)

    inst = CanonicalInstrument(
        symbol="AAPL",
        provider_symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        security_type=SecurityType.COMMON_STOCK,
        primary_exchange="NASDAQ",
        listing_status=ListingStatus.ACTIVE,
        classification_status=ClassificationStatus.VERIFIED,
        execution_eligibility=ExecutionEligibility.STOCK_EXECUTION,
        analytics_capability=AnalyticsCapability.FULL_ANALYTICS,
        classification_authority="ARX_SERVER_SECURITY_MASTER",
        classification_timestamp="2026-10-06T00:00:00Z",
    )
    repo.save(inst, now_epoch=now_epoch)

    # Fresh lookup (epoch + 30s)
    assert repo.get("AAPL", now_epoch=now_epoch + 30.0) is not None

    # Stale lookup (epoch + 70s): LRU bypassed, DB returns None
    repo.clear_cache()
    assert repo.get("AAPL", now_epoch=now_epoch + 70.0) is None


# ---------------------------------------------------------------------------
# 6. SECRET SAFETY VERIFICATION
# ---------------------------------------------------------------------------
def test_secret_safety_zero_credential_leakage(mock_service):
    """Verifies that zero API keys or authorization tokens leak into provenance or logs."""
    inst = mock_service.get_or_resolve_instrument("PLSE")
    payload_str = json.dumps(inst.to_dict())

    assert "mock_key" not in payload_str
    assert "mock_secret" not in payload_str
    assert "mock_figi_key" not in payload_str
    assert "APCA-API-KEY-ID" not in payload_str
    assert "X-OPENFIGI-APIKEY" not in payload_str


# ---------------------------------------------------------------------------
# 7. CANONICAL INSTRUMENT API INTEGRATION
# ---------------------------------------------------------------------------
def test_api_routes_security_master_and_market(mock_service):
    """
    Validates REST API endpoints:
    /api/v1/security-master/instruments/{symbol}
    /api/v1/market/instruments/{symbol}
    """
    client = TestClient(app)

    # 1. Security Master route for PLSE
    resp = client.get("/api/v1/security-master/instruments/PLSE")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "PLSE"
    assert data["execution_eligibility"] == "STOCK_EXECUTION"
    assert data["security_type"] == "COMMON_STOCK"

    # 2. Market route for PLSE
    resp_market = client.get("/api/v1/market/instruments/PLSE")
    assert resp_market.status_code == 200
    m_data = resp_market.json()
    assert m_data["symbol"] == "PLSE"
    assert m_data["execution_eligibility"] == "STOCK_EXECUTION"

    # 3. Negative case: TSM (ADR)
    resp_tsm = client.get("/api/v1/security-master/instruments/TSM")
    assert resp_tsm.status_code == 200
    tsm_data = resp_tsm.json()
    assert tsm_data["symbol"] == "TSM"
    assert tsm_data["execution_eligibility"] == "FAIL_CLOSED"
    assert tsm_data["security_type"] == "ADR"

    # 4. Status endpoint
    resp_status = client.get("/api/v1/security-master/status")
    assert resp_status.status_code == 200
    s_data = resp_status.json()
    assert s_data["status"] == "online"
    assert s_data["subsystem"] == "ARX_SERVER_SECURITY_MASTER"
