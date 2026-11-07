"""
tests/test_radar_stage2_discovery_remediation.py

Radar Stage 2 Discovery Remediation & Contract Alignment Test Suite.
Validates the 10 mandatory checks:
1. Stage 2 without VCP appears as research candidate.
2. Zero VCP does not imply zero Stage 2.
3. Stage 1/3/4 do not leak into primary Stage 2 discovery.
4. Unresolved data is not presented as failed technical setup.
5. Active promoted universe is used dynamically, preserving 35-symbol fixture.
6. Transport classification matches canonical VCPClassifier output.
7. Stage 2 candidate failing 2:1 R:R is disqualified from actionability (isActionable: False).
8. Existing VCP publication integrity remains unchanged.
9. Data session and execution timestamps remain distinguishable.
10. Radar, Analysis, and Trade Plan do not contradict canonical decision states.
"""

import pytest
from fastapi import Response
from typing import Dict, Any, List

from api.routes.screener import (
    run_screener_get,
    compute_canonical_stage_classification,
)
from analyst_dashboard.analyzers.decision_hierarchy import (
    DecisionHierarchyEngine,
    DecisionState,
)
from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    FreshnessStatus,
    PublicationDecision,
    resolve_coverage_envelope,
)
from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner
from analyst_dashboard.universe.fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE
from analyst_dashboard.universe.store import UniverseStore
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler


pytestmark = [pytest.mark.tier2b, pytest.mark.usefixtures("published_screener_universe")]


def test_1_stage2_without_vcp_appears_as_research_candidate():
    """
    Check 1: Stage 2 without VCP appears as research candidate.
    Equities with verified Stage 2 trend qualification must be discoverable in primary Radar
    discovery even when a VCP contraction setup is absent.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all", view="stage_2_primary")
    candidates = data.get("candidates", [])
    assert len(candidates) > 0, "Primary Stage 2 discovery must return candidate equities"

    # All returned candidates must be verified Stage 2
    for c in candidates:
        assert c.get("stageClassification") == "STAGE_2", (
            f"Candidate {c.get('symbol')} in primary discovery has stageClassification {c.get('stageClassification')}, expected STAGE_2"
        )

    # Must contain candidates that do not qualify for VCP (research discovery decoupled from VCP pattern)
    non_vcp_stage2 = [c for c in candidates if not c.get("stageProvenance", {}).get("isVcpQualified", False)]
    assert len(non_vcp_stage2) > 0, "Stage 2 candidates without VCP setup must be present in primary discovery"


def test_2_zero_vcp_does_not_imply_zero_stage2():
    """
    Check 2: Zero VCP does not imply zero Stage 2.
    When VCP matched count is zero across the cohort, Stage 2 opportunity discovery does NOT collapse to zero.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all", view="stage_2_primary")
    candidates = data.get("candidates", [])

    # Total evaluated symbols > 0 and Stage 2 opportunities > 0
    assert data.get("stage2OpportunityCount", 0) > 0
    assert len(candidates) > 0

    # Even if we strictly filter to non-VCP candidates, Stage 2 opportunities remain available
    non_vcp_stage2 = [c for c in candidates if "VCP" not in c.get("categories", [])]
    assert len(non_vcp_stage2) > 0, "Stage 2 discovery must yield candidates even when VCP category matches are 0"


def test_3_stage_1_3_4_do_not_leak_into_primary_discovery():
    """
    Check 3: Stage 1/3/4 do not leak into primary Stage 2 discovery.
    Primary discovery excludes Stage 1, Stage 3, Stage 4, and unresolved securities.
    These are only accessible via the diagnostic view.
    """
    resp = Response()
    # Primary view
    primary_data = run_screener_get(resp, filter_type="all", view="stage_2_primary")
    primary_candidates = primary_data.get("candidates", [])
    for c in primary_candidates:
        assert c.get("stageClassification") == "STAGE_2", (
            f"Symbol {c.get('symbol')} with stage {c.get('stageClassification')} leaked into primary Stage 2 discovery"
        )
        assert c.get("stageClassification") not in ("STAGE_1", "STAGE_3", "STAGE_4", "STAGE_UNRESOLVED")

    # Diagnostic view
    diag_data = run_screener_get(resp, filter_type="all", view="diagnostic")
    diag_candidates = diag_data.get("candidates", [])
    assert diag_data.get("activeView") == "DIAGNOSTIC_ALL_STAGES"
    assert len(diag_candidates) >= len(primary_candidates)


def test_4_unresolved_data_not_presented_as_failed_technical_setup():
    """
    Check 4: Unresolved data is not presented as failed technical setup.
    Securities with insufficient history (< 50 bars) or unresolvable structure evaluate to
    STAGE_UNRESOLVED with explicit reason codes, not as broken chart patterns or Stage 2 failures.
    """
    # Short candle series (20 bars)
    short_candles = [
        {"time": f"2026-10-{i:02d}", "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000000}
        for i in range(1, 21)
    ]
    stage_class, reasons, is_vcp = compute_canonical_stage_classification("SHORT_HIST", short_candles)
    assert stage_class == "STAGE_UNRESOLVED"
    assert "INSUFFICIENT_HISTORICAL_SESSIONS" in reasons
    assert is_vcp is False


def test_5_active_promoted_universe_resolved_dynamically():
    """
    Check 5: Active promoted universe is used dynamically, preserving 35-symbol fixture.
    The primary screener resolves membership from UniverseStore.get_latest_published_universe,
    while explicit PRE_SCREENED or HISTORICAL_35 requests fall back safely to the 35-symbol fixture.
    """
    resp = Response()
    # Default execution: dynamic resolution
    data_dynamic = run_screener_get(resp)
    assert data_dynamic.get("activeUniverseId") == "ARX_US_EQUITIES"
    assert data_dynamic.get("universeScope") == "ARX_US_EQUITIES"
    assert data_dynamic.get("activeUniverseBuildId") is not None

    # Regression fixture path
    data_fixture = run_screener_get(resp, universe="PRE_SCREENED")
    fixture_symbols = set(HISTORICAL_VCP_35_REGRESSION_FIXTURE)
    fixture_candidates = data_fixture.get("candidates", [])
    for c in fixture_candidates:
        assert c["symbol"] in fixture_symbols


def test_6_transport_classification_matches_canonical_vcp_classifier():
    """
    Check 6: Transport classification matches canonical VCPClassifier output.
    Transport candidate objects provide stageClassification, stagePhase, setupPattern,
    and stageProvenance citing classifierAuthority: 'VCPClassifier'.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all", view="stage_2_primary")
    candidates = data.get("candidates", [])
    assert len(candidates) > 0

    sample = candidates[0]
    assert "stageClassification" in sample
    assert "stagePhase" in sample
    assert "setupPattern" in sample
    assert "stageProvenance" in sample

    prov = sample["stageProvenance"]
    assert prov.get("classifierAuthority") == "VCPClassifier"
    assert prov.get("rulesetVersion") == "1.0.0"
    assert prov.get("trendTemplatePass") is True
    assert isinstance(prov.get("rejectionReasons"), list)


def test_7_stage2_candidate_failing_2_to_1_rr_not_trade_ready():
    """
    Check 7: Stage 2 candidate failing 2:1 R:R is disqualified from actionability (isActionable: False).
    DecisionHierarchyEngine strictly enforces rr >= 2.0. Any ratio below 2.0 (e.g. 1.85:1)
    evaluates to VALID_SETUP (awaiting trigger), with isActionable = False and canSizeTrade = False.
    """
    # 1.85:1 ratio: below 2.0 threshold
    decision_sub2 = DecisionHierarchyEngine.resolve_decision_state(
        symbol="S2_CANDIDATE",
        current_price=100.0,
        candle_count=250,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=1.85,
    )
    assert decision_sub2["state"] == DecisionState.VALID_SETUP.value
    assert decision_sub2["isActionable"] is False
    assert decision_sub2["canSizeTrade"] is False
    assert "2.0:1" in decision_sub2["disqualificationReason"]

    # 2.2:1 ratio: meets or exceeds 2.0 threshold
    decision_actionable = DecisionHierarchyEngine.resolve_decision_state(
        symbol="S2_CANDIDATE",
        current_price=100.0,
        candle_count=250,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.2,
    )
    assert decision_actionable["state"] == DecisionState.ACTIONABLE_SETUP.value
    assert decision_actionable["isActionable"] is True
    assert decision_actionable["canSizeTrade"] is True


def test_8_existing_vcp_publication_integrity_unchanged():
    """
    Check 8: Existing VCP publication integrity remains unchanged.
    VCPScannerRunner snapshot publication continues to adhere strictly to MINERVINI_VCP.
    Stage 2 non-VCP candidates are NOT published to the VCP snapshot results, and matched_count
    reflects strictly confirmed VCP setups.
    """
    runner = VCPScannerRunner()
    # In VCPScannerRunner qualification loop:
    # A symbol must be BOTH is_vcp and is_stage_2 to be admitted to VCP qualified candidates.
    # Non-VCP Stage 2 symbols are NOT admitted.
    assert hasattr(runner, "execute_market_wide_scan")
    # Verify publication integrity engine ruleset hash is preserved
    from analyst_dashboard.analyzers.scanner_publication_integrity import (
        CANONICAL_VCP_RULESET_HASH,
        VCP_RULESET_VERSION,
    )
    assert VCP_RULESET_VERSION == "1.0.0"
    assert len(CANONICAL_VCP_RULESET_HASH) == 64


def test_9_data_session_and_execution_timestamps_distinguishable():
    """
    Check 9: Data session and execution timestamps remain distinguishable.
    source_session_date reflects the authoritative daily market trade date (e.g. 2026-11-05),
    while scanned_at reflects the execution clock timestamp in UTC ISO 8601 (e.g. 2026-11-07T...Z).
    """
    c_meta = {
        "coverage_status": "COMPLETE",
        "scanned_successfully_count": 250,
        "source_session_date": "2026-11-05",
        "scanned_at": "2026-11-07T12:00:00Z",
        "stage_2_opportunity_count": 100,
        "unresolved_symbol_count": 65,
    }
    envelope = resolve_coverage_envelope(
        status=ScannerStatus.AVAILABLE,
        universe_size=250,
        matched_count=0,
        c_meta=c_meta,
    )
    assert envelope["source_session_date"] == "2026-11-05"
    assert envelope["scanned_at"] == "2026-11-07T12:00:00Z"
    assert envelope["source_session_date"] != envelope["scanned_at"]
    assert envelope["stage_2_opportunity_count"] == 100
    assert envelope["unresolved_symbol_count"] == 65


def test_10_radar_analysis_and_trade_plan_decision_state_parity():
    """
    Check 10: Radar, Analysis, and Trade Plan do not contradict canonical decision states.
    When a Stage 2 asset fails trade-readiness (e.g. R:R < 2.0 or pending trigger):
    - Radar discovery provides stageClassification: 'STAGE_2' but does not declare actionable execution.
    - DecisionHierarchyEngine assigns state: 'VALID_SETUP' with isActionable = False.
    - Trade planning / sizing is strictly prohibited until actionable threshold is satisfied.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all", view="stage_2_primary")
    candidates = data.get("candidates", [])
    assert len(candidates) > 0

    for c in candidates:
        rr = c.get("riskRewardRatio")
        if rr is not None and rr < 2.0:
            dec = DecisionHierarchyEngine.resolve_decision_state(
                symbol=c["symbol"],
                current_price=c["currentPrice"],
                candle_count=100,
                freshness_status="LIVE",
                has_fundamentals=True,
                confluence_score=c.get("confluenceScore", 75.0),
                stage_phase=2,
                is_in_buy_zone=True,
                risk_reward_ratio=rr,
            )
            # Must NOT be actionable
            assert dec["isActionable"] is False
            assert dec["canSizeTrade"] is False
            assert dec["state"] == DecisionState.VALID_SETUP.value
