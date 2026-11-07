"""
tests/test_radar_single_authoritative_universe.py

Focused Verification Suite for Directive ARX-RADAR-UNIVERSE-001:
Radar Single Authoritative Universe & Investor-Facing Curated Deprecation.

Validates the 10 mandatory checks:
1. Only the published ARX_US_EQUITIES universe is used for normal Radar discovery.
2. Unavailable published universe fails closed without curated fallback (UNIVERSE_UNAVAILABLE).
3. Empty universe membership fails closed without curated fallback (UNIVERSE_MEMBERSHIP_EMPTY).
4. Curated catalog (43) is never automatically substituted in default discovery.
5. Historical fixtures remain available exclusively to explicit test harnesses (PRE_SCREENED, ALL_CATALOG).
6. Stage 2 classification is independent of VCP.
7. Non-Stage-2 assets are excluded from primary opportunities (accessible via view='diagnostic').
8. Actual published universe membership count, build ID, and data-as-of are displayed dynamically without hardcoded sizes.
9. Actionability hurdle (2.0:1) and discovery-only boundaries (isActionable=False, canSizeTrade=False) are preserved.
10. Search and custom ticker navigation do not silently substitute curated universes.
"""

import pytest
from unittest.mock import MagicMock
from fastapi import Response

from api.routes.screener import (
    run_screener_get,
    MASTER_CATALOG_CANDIDATES,
    HISTORICAL_VCP_35_REGRESSION_FIXTURE,
    vcp_scanner_runner,
)
from analyst_dashboard.analyzers.decision_hierarchy import (
    DecisionHierarchyEngine,
    DecisionState,
)
from analyst_dashboard.universe.store import UniverseStore


pytestmark = [pytest.mark.tier2b, pytest.mark.usefixtures("published_screener_universe")]


def test_1_normal_discovery_uses_only_published_universe():
    """
    Check 1: Only the published ARX_US_EQUITIES universe is used for normal Radar discovery.
    Default requests resolve strictly to ARX_US_EQUITIES with active build ID and matching count.
    """
    resp = Response()
    data = run_screener_get(resp)

    assert data.get("status") in ("SUCCESS", "PARTIAL")
    assert data.get("activeUniverseId") == "ARX_US_EQUITIES"
    assert data.get("universeScope") == "ARX_US_EQUITIES"
    assert data.get("activeUniverseBuildId") is not None
    assert data.get("activeView") == "STAGE_2_PRIMARY"

    pop = data.get("population", {})
    assert pop.get("universeScope") == "ARX_US_EQUITIES"
    assert pop.get("universeBuildId") == data.get("activeUniverseBuildId")
    assert pop.get("referenceUniverseCount") == data.get("totalCandidates")


def test_2_unavailable_published_universe_fails_closed():
    """
    Check 2: Unavailable published universe fails closed without curated fallback.
    Returns status='ERROR', datasetState='UNIVERSE_UNAVAILABLE', with zero candidates.
    Does NOT fall back to 35-fixture or 43-catalog.
    """
    mock_store = MagicMock(spec=UniverseStore)
    mock_store.get_latest_published_universe.return_value = None

    orig_store = vcp_scanner_runner.universe_store
    vcp_scanner_runner.universe_store = mock_store
    try:
        resp = Response()
        data = run_screener_get(resp)

        assert data.get("status") == "ERROR"
        assert data.get("scanState") == "FAILED"
        assert data.get("datasetState") == "UNIVERSE_UNAVAILABLE"
        assert "UNPUBLISHED_UNIVERSE_BUILD" in data.get("error", "")
        assert len(data.get("candidates", [])) == 0
        assert data.get("gemsFound") == 0
        assert data.get("totalCandidates") == 0
        assert data.get("activeUniverseId") == "ARX_US_EQUITIES"
    finally:
        vcp_scanner_runner.universe_store = orig_store


def test_3_empty_universe_membership_fails_closed():
    """
    Check 3: Empty universe membership fails closed without curated fallback.
    When build exists but contains 0 scannable symbols, returns UNIVERSE_MEMBERSHIP_EMPTY.
    """
    mock_store = MagicMock(spec=UniverseStore)
    mock_store.get_latest_published_universe.return_value = {
        "universe_build_id": "TEST-EMPTY-BUILD-001",
        "universe_id": "ARX_US_EQUITIES",
        "publication_decision": "PUBLISH",
        "construction_status": "COMPLETE",
    }
    mock_store.get_scannable_membership.return_value = []

    orig_store = vcp_scanner_runner.universe_store
    vcp_scanner_runner.universe_store = mock_store
    try:
        resp = Response()
        data = run_screener_get(resp)

        assert data.get("status") == "ERROR"
        assert data.get("scanState") == "FAILED"
        assert data.get("datasetState") == "UNIVERSE_MEMBERSHIP_EMPTY"
        assert "EMPTY_UNIVERSE_MEMBERSHIP" in data.get("error", "")
        assert len(data.get("candidates", [])) == 0
        assert data.get("gemsFound") == 0
        assert data.get("activeUniverseBuildId") == "TEST-EMPTY-BUILD-001"
    finally:
        vcp_scanner_runner.universe_store = orig_store


def test_4_no_automatic_fallback_to_43_asset_catalog():
    """
    Check 4: Curated 43-asset catalog is never automatically substituted in default discovery.
    """
    resp = Response()
    data = run_screener_get(resp)

    # In default discovery, universeScope must not be MASTER_CATALOG or ALL_CATALOG
    assert data.get("universeScope") == "ARX_US_EQUITIES"
    assert data.get("activeUniverseId") == "ARX_US_EQUITIES"
    assert data.get("universeScope") not in ("MASTER_CATALOG", "ALL_CATALOG")


def test_5_curated_selectors_removed_and_cannot_override_published_universe():
    """
    Check 5: Curated Pre-Screened VCP (35) and Master Catalog (43) selectors are removed.
    Explicit PRE_SCREENED and ALL_CATALOG requests no longer substitute curated lists,
    strictly resolving to the authoritative published ARX_US_EQUITIES universe.
    """
    resp = Response()

    # Attempted PRE_SCREENED request: must not switch to curated 35 fixture
    data_ps = run_screener_get(resp, universe="PRE_SCREENED")
    assert data_ps.get("activeUniverseId") == "ARX_US_EQUITIES"
    assert data_ps.get("universeScope") == "ARX_US_EQUITIES"

    # Attempted ALL_CATALOG request: must not switch to curated 43 catalog
    data_cat = run_screener_get(resp, universe="ALL_CATALOG")
    assert data_cat.get("activeUniverseId") == "ARX_US_EQUITIES"
    assert data_cat.get("universeScope") == "ARX_US_EQUITIES"


def test_6_stage2_classification_independent_of_vcp():
    """
    Check 6: Stage 2 classification is independent of VCP.
    Stage 2 candidates are discoverable even when 0 VCP patterns qualify.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all")
    candidates = data.get("candidates", [])

    assert len(candidates) > 0, "Primary Stage 2 discovery must return candidate equities"
    for c in candidates:
        assert c.get("stageClassification") == "STAGE_2"

    # Non-VCP Stage 2 equities must be present and distinct from VCP qualifications
    non_vcp = [c for c in candidates if not c.get("stageProvenance", {}).get("isVcpQualified", False)]
    assert len(non_vcp) > 0, "Stage 2 candidates without VCP must be discoverable in primary view"


def test_7_non_stage2_excluded_from_primary_opportunities():
    """
    Check 7: Non-Stage-2 assets are excluded from primary opportunities.
    Primary view contains strictly STAGE_2 assets. Stage 1, 3, 4, and unresolved are in view='diagnostic'.
    """
    resp = Response()
    primary = run_screener_get(resp, filter_type="all", view="stage_2_primary")
    for c in primary.get("candidates", []):
        assert c.get("stageClassification") == "STAGE_2"
        assert c.get("stageClassification") not in ("STAGE_1", "STAGE_3", "STAGE_4", "STAGE_UNRESOLVED")

    diag = run_screener_get(resp, filter_type="all", view="diagnostic")
    diag_stages = {c.get("stageClassification") for c in diag.get("candidates", [])}
    assert len(diag_stages) >= 1
    assert diag.get("activeView") == "DIAGNOSTIC_ALL_STAGES"


def test_8_published_universe_metadata_transported():
    """
    Check 8: Display the actual published universe membership count, build ID, and data-as-of date.
    Never hardcode the universe size.
    """
    resp = Response()
    data = run_screener_get(resp)

    assert data.get("activeUniverseId") == "ARX_US_EQUITIES"
    assert data.get("activeUniverseBuildId") is not None
    assert isinstance(data.get("totalCandidates"), int)
    assert data.get("totalCandidates") > 0

    pop = data.get("population", {})
    assert pop.get("referenceUniverseCount") == data.get("totalCandidates")
    assert pop.get("universeBuildId") == data.get("activeUniverseBuildId")
    assert pop.get("stage2OpportunityCount") == sum(
        1 for c in data.get("candidates", []) if c.get("stageClassification") == "STAGE_2"
    )


def test_9_actionability_hurdle_preserved_and_discovery_is_actionable_false():
    """
    Check 9: Preserve the 2.0:1 actionability hurdle and discovery-only boundaries.
    In Radar screener, isActionable is strictly False, canSizeTrade is strictly False.
    DecisionHierarchyEngine enforces 2.0:1 ratio.
    """
    resp = Response()
    data = run_screener_get(resp)
    for c in data.get("candidates", []):
        assert c.get("isActionable") is False, "Discovery screener must never grant trade actionability"
        assert c.get("canSizeTrade") is False, "Position sizing must be disabled in discovery screener"

    # Engine hurdle enforcement: 1.85:1 vs 2.2:1
    sub2 = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST",
        current_price=100.0,
        candle_count=200,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=1.85,
    )
    assert sub2["isActionable"] is False
    assert "2.0:1" in sub2["disqualificationReason"]

    above2 = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST",
        current_price=100.0,
        candle_count=200,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.2,
    )
    assert above2["isActionable"] is True


def test_10_search_and_custom_tickers_do_not_substitute_curated_universes():
    """
    Check 10: Search and custom ticker navigation do not silently substitute curated universes.
    Custom ticker scan scopes strictly to requested assets.
    """
    resp = Response()
    data = run_screener_get(resp, custom_tickers="AAPL,MSFT")

    assert data.get("activeUniverseId") == "CUSTOM_WATCHLIST"
    assert data.get("totalCandidates") == 2
    symbols = [c["symbol"] for c in data.get("candidates", [])]
    assert set(symbols) == {"AAPL", "MSFT"}
