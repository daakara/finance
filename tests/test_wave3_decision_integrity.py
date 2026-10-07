"""tests/test_wave3_decision_integrity.py

Targeted test suite for Synthesis E Wave 3:
- Epistemic Integrity & Prohibited Terms
- Canonical Instrument Evidence Applicability Router
- ETF Non-Disqualification for Missing 10-K
- UNKNOWN Instrument Fail-Closed
- Decision Trace Instrument Explainability
- Smart Money Recency and Status Semantics
"""

import pytest
from analyst_dashboard.security_master.models import SecurityType, AssetClass
from analyst_dashboard.security_master.applicability import (
    get_required_evidence_for_instrument,
    INSTRUMENT_EVIDENCE_REGISTRY,
)
from analyst_dashboard.analyzers.decision_hierarchy import (
    DecisionHierarchyEngine,
    DecisionState,
)
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine


def test_instrument_evidence_contracts_defined():
    """Verify registry covers required instrument classes."""
    for st in [SecurityType.COMMON_STOCK, SecurityType.ETF, SecurityType.ADR, SecurityType.REIT, SecurityType.UNKNOWN]:
        contract = get_required_evidence_for_instrument(st)
        assert contract is not None
        assert contract.security_type == st


def test_etf_contract_exempts_corporate_10k():
    """Verify ETF contract exempts corporate 10-K/10-Q and insider filings."""
    contract = get_required_evidence_for_instrument(SecurityType.ETF)
    assert "CORPORATE_FINANCIALS_10K_10Q" in contract.not_applicable_evidence
    assert "FORM_4_CSUITE_INSIDERS" in contract.not_applicable_evidence
    assert "FUND_STRUCTURE_PROFILE" in contract.required_evidence
    assert "Fund / ETF Profile" in contract.profile_label


def test_common_stock_requires_corporate_fundamentals():
    """Verify common stock requires corporate financial filings and fails if missing."""
    contract = get_required_evidence_for_instrument(SecurityType.COMMON_STOCK)
    assert "CORPORATE_FINANCIALS_10K_10Q" in contract.required_evidence

    res = DecisionHierarchyEngine.resolve_decision_state(
        symbol="AAPL",
        current_price=180.0,
        candle_count=100,
        freshness_status="REALTIME",
        has_fundamentals=False,  # missing 10-K/10-Q
        confluence_score=80.0,
        security_type=SecurityType.COMMON_STOCK,
    )
    assert res["state"] == DecisionState.EVIDENCE_INCOMPLETE.value
    assert "10-K" in res["disqualificationReason"] or "unverified" in res["disqualificationReason"]
    assert res["isActionable"] is False


def test_etf_not_disqualified_for_missing_corporate_fundamentals():
    """INV-W3-01: An ETF must NEVER fail or be disqualified for missing SEC Form 10-K."""
    res = DecisionHierarchyEngine.resolve_decision_state(
        symbol="SPY",
        current_price=540.0,
        candle_count=100,
        freshness_status="REALTIME",
        has_fundamentals=False,  # Corporate 10-K missing, which is NOT_APPLICABLE for ETFs
        confluence_score=80.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_confirmed=True,
        security_type=SecurityType.ETF,
        asset_class=AssetClass.ETF,
    )
    # Must NOT fail with EVIDENCE_INCOMPLETE for 10-K
    assert res["state"] != DecisionState.EVIDENCE_INCOMPLETE.value
    assert res["state"] in (DecisionState.ACTIONABLE_SETUP.value, DecisionState.VALID_SETUP.value)
    if res.get("disqualificationReason"):
        assert "10-K" not in res["disqualificationReason"]


def test_unknown_instrument_fails_closed():
    """INV-W3-02: UNKNOWN instrument must fail closed to UNVERIFIED."""
    res = DecisionHierarchyEngine.resolve_decision_state(
        symbol="UNKNOWN_XYZ",
        current_price=100.0,
        candle_count=100,
        freshness_status="REALTIME",
        has_fundamentals=True,
        confluence_score=85.0,
        security_type=SecurityType.UNKNOWN,
    )
    assert res["state"] == DecisionState.UNVERIFIED.value
    assert res["isActionable"] is False
    assert "Unclassified instrument" in res["disqualificationReason"] or "unconfirmed" in res["disqualificationReason"]


def test_decision_trace_etf_narrative():
    """INV-W3-03: Decision trace produces instrument-aware Fund Profile explanation for ETFs."""
    trace = DecisionTraceEngine.build_decision_trace(
        symbol="QQQ",
        current_price=460.0,
        candles=[{"time": "2026-10-01", "close": 460.0}] * 60,
        freshness={"status": "REALTIME", "providerSource": "alpaca"},
        technicals={"rsi_14": 55.0},
        confluence={"confluenceScore": 78.0},
        factor_scores={},  # No corporate factors
        optimal_execution={"execution_status": "IN_BUY_ZONE", "risk_reward_ratio": 2.2, "stage_phase": 2},
        security_type=SecurityType.ETF,
        asset_class=AssetClass.ETF,
    )
    assert trace["instrumentProfile"]["securityType"] == "ETF"
    assert "Corporate 10-K financial filings are not applicable" in trace["explanation"]
    assert "Fund / ETF Profile" in trace["explanation"]


def test_smart_money_single_asset_status():
    """INV-W3-04: Single asset smart money options flow handles curated and live distinction."""
    live_flow = SmartMoneyEngine.get_options_flow(symbol="NVDA", include_curated=False)
    assert isinstance(live_flow, list)
    curated_flow = SmartMoneyEngine.get_options_flow(symbol="NVDA", include_curated=True)
    assert isinstance(curated_flow, list)
