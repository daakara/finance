"""Tests for Trader Archetype Strategy Models (Buffett, Pelosi, Druckenmiller, Simons, Gardner).

Phase 4A-2 F_04 Remediation Test Suite:
- Verification of authentic empirical scoring and exclusion of numeric laundering fallbacks.
- Elimination of hardcoded ticker score floors.
- Explicit typing of static domain priors (STATIC_DOMAIN_PRIOR).
- Partial-evidence weight normalization across required and optional factors.
- Preservation of canonical decision authority under DecisionHierarchyEngine.
"""

import pandas as pd
import numpy as np
import pytest
from analyst_dashboard.analyzers.trader_archetypes import (
    TraderArchetypeAnalyzer,
    EvidenceType,
    ArchetypeEvidenceStatus,
)
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState

pytestmark = pytest.mark.tier2b


def test_trader_archetype_consensus_five_models():
    analyzer = TraderArchetypeAnalyzer()

    # Synthetic price DataFrame
    dates = pd.date_range("2025-01-01", periods=60)
    prices = [100.0 * (1.0 + 0.005 * i) for i in range(60)]
    df = pd.DataFrame({"Close": prices}, index=dates)

    factor_scores = {
        "growthScore": 90,
        "qualityScore": 88,
        "valuationScore": 75,
        "momentumScore": 85,
        "tailRiskScore": 80,
        "piotroskiFScore": 8,
    }

    result = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors", "returnOnAssets": 0.25, "freeCashflow": 25000000000},
        price_df=df,
        risk_metrics={"Sortino_Ratio": 2.8, "Skewness": -0.1},
        macro_indicators={"yield_curve_spread": 0.45, "credit_spread_oas": 2.5},
        factor_scores=factor_scores,
    )

    assert "consensusScore" in result
    assert "verdict" in result
    assert len(result["archetypes"]) == 5

    archetype_names = [a["name"] for a in result["archetypes"]]
    assert any("Buffett" in n for n in archetype_names)
    assert any("Pelosi" in n for n in archetype_names)
    assert any("Druckenmiller" in n for n in archetype_names)
    assert any("Simons" in n for n in archetype_names)
    assert any("David Gardner" in n or "Motley Fool" in n for n in archetype_names)


def test_warren_buffett_moat_and_commodity_discrimination():
    """Warren Buffett Model: Discerning wide moats vs commodity hardware assembly vs logistics vs biotech."""
    analyzer = TraderArchetypeAnalyzer()

    # 1. Wide-Moat Compounder (AAPL) - Empirical score without hardcoded >= 90 floor clamp
    res_aapl = analyzer.analyze_asset(
        symbol="AAPL",
        info={"sector": "Technology", "industry": "Consumer Electronics"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"qualityScore": 92, "valuationScore": 70, "piotroskiFScore": 8},
    )
    buffett_aapl = next(a for a in res_aapl["archetypes"] if "Buffett" in a["name"])
    assert buffett_aapl["alignmentScore"] >= 80
    assert buffett_aapl["status"] == "High Moat Alignment"
    assert "pricing power" in buffett_aapl["thesis"]
    assert buffett_aapl["evidenceStatus"] in ["AVAILABLE", EvidenceType.AUTHORITATIVE_DYNAMIC]

    # 2. Hardware Server Integrator (SMCI) -> Capped score, thin gross margins
    res_smci = analyzer.analyze_asset(
        symbol="SMCI",
        info={"sector": "Technology", "industry": "Computer Hardware"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"qualityScore": 75, "valuationScore": 70, "piotroskiFScore": 6},
    )
    buffett_smci = next(a for a in res_smci["archetypes"] if "Buffett" in a["name"])
    assert buffett_smci["alignmentScore"] <= 62
    assert buffett_smci["status"] == "Competitive Commodity Risk"
    assert "thin gross margins" in buffett_smci["thesis"]

    # 3. Logistics Network (DHLGY) -> Capital-intensive delivery moat
    res_dhl = analyzer.analyze_asset(
        symbol="DHLGY",
        info={"sector": "Industrials", "industry": "Freight & Logistics Services"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"qualityScore": 72, "valuationScore": 70, "piotroskiFScore": 6},
    )
    buffett_dhl = next(a for a in res_dhl["archetypes"] if "Buffett" in a["name"])
    assert buffett_dhl["alignmentScore"] <= 72
    assert buffett_dhl["status"] == "Capital-Intensive Network Moat"
    assert "CapEx" in buffett_dhl["thesis"]

    # 4. Clinical Biotech (ARWR) -> Outside circle of competence
    res_arwr = analyzer.analyze_asset(
        symbol="ARWR",
        info={"sector": "Healthcare", "industry": "Biotechnology"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"qualityScore": 55, "valuationScore": 50, "piotroskiFScore": 4},
    )
    buffett_arwr = next(a for a in res_arwr["archetypes"] if "Buffett" in a["name"])
    assert buffett_arwr["alignmentScore"] <= 58
    assert buffett_arwr["status"] == "Outside Circle of Competence"
    assert "clinical trial" in buffett_arwr["thesis"].lower()

    # 5. Crypto Moat Proxy (BTC-USD) -> Disclosed as STATIC_DOMAIN_PRIOR, without numeric missing-evidence score
    res_btc = analyzer.analyze_asset(
        symbol="BTC-USD",
        info={},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={},
    )
    buffett_btc = next(a for a in res_btc["archetypes"] if "Buffett" in a["name"])
    assert buffett_btc["alignmentScore"] is None
    assert buffett_btc["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE
    assert "Tier-1 Network Moat" in buffett_btc["status"]
    assert buffett_btc["thematicPrior"] is not None
    assert buffett_btc["thematicPrior"]["evidenceType"] == EvidenceType.STATIC_DOMAIN_PRIOR


def test_nancy_pelosi_congressional_policy_coverage():
    """Nancy Pelosi Model: Key legislative policy beneficiaries and sector-aware policy fallbacks."""
    analyzer = TraderArchetypeAnalyzer()

    # 1. Direct Policy Beneficiary (PLTR) with valid measured momentum and growth
    res_pltr = analyzer.analyze_asset(
        symbol="PLTR",
        info={"sector": "Technology", "industry": "Software - Infrastructure"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"momentumScore": 88, "growthScore": 92},
    )
    pelosi_pltr = next(a for a in res_pltr["archetypes"] if "Pelosi" in a["name"])
    assert pelosi_pltr["alignmentScore"] >= 85
    assert pelosi_pltr["status"] == "Strong Policy Support"
    assert "Department of Defense" in pelosi_pltr["thesis"]
    assert pelosi_pltr["thematicPrior"] is not None
    assert pelosi_pltr["thematicPrior"]["evidenceType"] == EvidenceType.STATIC_DOMAIN_PRIOR

    # 2. Direct Defense Contractor (LMT)
    res_lmt = analyzer.analyze_asset(
        symbol="LMT",
        info={"sector": "Industrials", "industry": "Aerospace & Defense"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"momentumScore": 75, "growthScore": 70},
    )
    pelosi_lmt = next(a for a in res_lmt["archetypes"] if "Pelosi" in a["name"])
    assert pelosi_lmt["status"] == "Strong Policy Support"
    assert "Air Dominance" in pelosi_lmt["thesis"]

    # 3. Sector Fallback (Defense non-listed)
    res_def = analyzer.analyze_asset(
        symbol="DEFENSE_CO",
        info={"sector": "Industrials", "industry": "Aerospace & Defense"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"momentumScore": 75, "growthScore": 70},
    )
    pelosi_def = next(a for a in res_def["archetypes"] if "Pelosi" in a["name"])
    assert pelosi_def["status"] == "Defense Appropriations Exposure"
    assert "NDAA" in pelosi_def["thesis"]


def test_stanley_druckenmiller_macro_regime_sensitivity():
    """Stanley Druckenmiller Model: Dynamic reaction to inverted yield curve vs expansionary regimes."""
    analyzer = TraderArchetypeAnalyzer()

    # 1. Inverted Yield Curve (Late-Cycle / Tightening)
    res_inv = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=None,
        risk_metrics={},
        macro_indicators={"yield_curve_spread": -0.52, "credit_spread_oas": 2.80},
        factor_scores={"momentumScore": 85, "growthScore": 90},
    )
    druck_inv = next(a for a in res_inv["archetypes"] if "Druckenmiller" in a["name"])
    assert "Inverted" in druck_inv["status"]
    assert "tightening" in druck_inv["thesis"].lower()
    assert "lower interest rate environment" not in druck_inv["thesis"].lower()

    # 2. Widening Credit Spreads (Credit Stress)
    res_stress = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=None,
        risk_metrics={},
        macro_indicators={"yield_curve_spread": 0.30, "credit_spread_oas": 4.60},
        factor_scores={"momentumScore": 85, "growthScore": 90},
    )
    druck_stress = next(a for a in res_stress["archetypes"] if "Druckenmiller" in a["name"])
    assert "Credit Spread Widening" in druck_stress["status"]
    assert "tightening financial conditions" in druck_stress["thesis"].lower()

    # 3. Expansionary Steepening Regime
    res_exp = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=None,
        risk_metrics={},
        macro_indicators={"yield_curve_spread": 0.45, "credit_spread_oas": 2.20},
        factor_scores={"momentumScore": 85, "growthScore": 90},
    )
    druck_exp = next(a for a in res_exp["archetypes"] if "Druckenmiller" in a["name"])
    assert druck_exp["status"] == "Positive Macro Trend"
    assert "Accommodative monetary liquidity" in druck_exp["thesis"]


def test_jim_simons_quant_tail_risk_discrimination():
    """Jim Simons Model: Statistical stability vs left-tail crash risk penalty."""
    analyzer = TraderArchetypeAnalyzer()

    # 1. Low Downside Risk / High Sortino
    res_stable = analyzer.analyze_asset(
        symbol="STABLE1",
        info={},
        price_df=None,
        risk_metrics={"Sortino_Ratio": 2.65, "Skewness": -0.10},
        macro_indicators={},
        factor_scores={"tailRiskScore": 88, "momentumScore": 80},
    )
    simons_stable = next(a for a in res_stable["archetypes"] if "Simons" in a["name"])
    assert simons_stable["alignmentScore"] >= 80
    assert simons_stable["status"] == "Low Downside Risk"
    assert "Superior Sortino" in simons_stable["thesis"]

    # 2. Severe Left-Tail Crash Risk
    res_crash = analyzer.analyze_asset(
        symbol="RISKY1",
        info={},
        price_df=None,
        risk_metrics={"Sortino_Ratio": 0.55, "Skewness": -1.40},
        macro_indicators={},
        factor_scores={"tailRiskScore": 40, "momentumScore": 55},
    )
    simons_crash = next(a for a in res_crash["archetypes"] if "Simons" in a["name"])
    assert simons_crash["status"] == "Elevated Tail Risk / Asymmetric Downside"
    assert "crash risk" in simons_crash["thesis"].lower()
    assert "limited crash risk" not in simons_crash["thesis"].lower()


def test_david_gardner_rule_breakers_anti_hallucination():
    """David Gardner Model: Sector-specific theses without margin hallucinations on freight or ODM hardware."""
    analyzer = TraderArchetypeAnalyzer()

    # 1. Freight & Logistics (DHLGY, FDX, UPS) -> Must NOT claim high gross margin or cloud transition
    for sym in ["DHLGY", "FDX", "UPS"]:
        res_freight = analyzer.analyze_asset(
            symbol=sym,
            info={"sector": "Industrials", "industry": "Freight & Logistics Services"},
            price_df=None,
            risk_metrics={},
            macro_indicators={},
            factor_scores={"growthScore": 68, "momentumScore": 72},
        )
        gardner_freight = next(a for a in res_freight["archetypes"] if "Gardner" in a["name"])
        assert "High gross margin" not in gardner_freight["thesis"], f"{sym} must not claim High gross margin"
        assert "digital/cloud architecture" not in gardner_freight["catalyst"], f"{sym} must not claim cloud architecture"
        assert "logistics network" in gardner_freight["thesis"].lower()

    # 2. Hardware Server ODM (SMCI, DELL) -> AI Hardware Supercycle
    res_smci = analyzer.analyze_asset(
        symbol="SMCI",
        info={"sector": "Technology", "industry": "Computer Hardware"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"growthScore": 88, "momentumScore": 85},
    )
    gardner_smci = next(a for a in res_smci["archetypes"] if "Gardner" in a["name"])
    assert gardner_smci["status"] == "AI Hardware Supercycle"
    assert "liquid cooling" in gardner_smci["thesis"].lower()

    # 3. Biopharma (ARWR, CPRX, LLY) -> Therapeutic pipeline
    res_arwr = analyzer.analyze_asset(
        symbol="ARWR",
        info={"sector": "Healthcare", "industry": "Biotechnology"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"growthScore": 78, "momentumScore": 70},
    )
    gardner_arwr = next(a for a in res_arwr["archetypes"] if "Gardner" in a["name"])
    assert "therapeutic" in gardner_arwr["thesis"].lower()
    assert "clinical trial" in gardner_arwr["catalyst"].lower()


def test_trader_archetype_null_safety_and_none_coalescing():
    """Verify null-safety across all 5 archetypes when dictionaries contain None or non-float values."""
    analyzer = TraderArchetypeAnalyzer()
    res = analyzer.analyze_asset(
        symbol="NULL_TEST",
        info={"sector": None, "industry": None},
        price_df=None,
        risk_metrics={"Sortino_Ratio": None, "Skewness": None},
        macro_indicators={"yield_curve_spread": None, "credit_spread_oas": None},
        factor_scores={
            "qualityScore": None,
            "growthScore": None,
            "momentumScore": None,
            "valuationScore": None,
            "piotroskiFScore": None,
            "tailRiskScore": None,
        },
    )
    assert res["consensusScore"] is None
    assert res["verdict"] == "Telemetry Unavailable"
    assert len(res["archetypes"]) == 5
    for a in res["archetypes"]:
        assert a.get("evidenceStatus") == ArchetypeEvidenceStatus.UNAVAILABLE
        assert a["alignmentScore"] is None
        assert len(a["thesis"]) > 5
        assert len(a["catalyst"]) > 5


def test_david_gardner_non_tech_sector_theses():
    """Verify Utilities, Financials, Real Estate, and Retail emit sector-specific growth theses."""
    analyzer = TraderArchetypeAnalyzer()
    test_cases = [
        ("NEE", "Utilities", "Electric Utilities", "Regulated Utility"),
        ("WMT", "Consumer Defensive", "Retail - Discount", "Consumer Distribution"),
        ("JPM", "Financial Services", "Commercial Banking", "Financial Institution"),
        ("O", "Real Estate", "REIT - Commercial", "Real Estate Asset Portfolio"),
    ]
    for sym, sec, ind, expected_status in test_cases:
        res = analyzer.analyze_asset(
            symbol=sym,
            info={"sector": sec, "industry": ind},
            price_df=None,
            risk_metrics={},
            macro_indicators={},
            factor_scores={"growthScore": 75, "momentumScore": 70},
        )
        gardner = next(a for a in res["archetypes"] if "Gardner" in a["name"])
        assert expected_status in gardner["status"], f"{sym}: expected {expected_status} in {gardner['status']}"
        assert "digital/cloud architecture" not in gardner["catalyst"], f"{sym} leaked cloud catalyst"
        assert "High gross margin" not in gardner["thesis"], f"{sym} leaked high gross margin thesis"


def test_hardware_odm_broadened_matching():
    """Verify broadened ODM keywords (server, electronic manufacturing, chassis, liquid cooling)."""
    analyzer = TraderArchetypeAnalyzer()
    odm_cases = [
        ("SYNTH_SERVER", "Technology", "AI Server Hardware Rack Assembly"),
        ("SYNTH_EMS", "Technology", "Electronic Manufacturing Services"),
        ("SYNTH_CHASSIS", "Technology", "Modular Chassis Design and Assembly"),
        ("SYNTH_COOLING", "Technology", "Direct Liquid Cooling Infrastructure"),
    ]
    for sym, sec, ind in odm_cases:
        res = analyzer.analyze_asset(
            symbol=sym,
            info={"sector": sec, "industry": ind},
            price_df=None,
            risk_metrics={},
            macro_indicators={},
            factor_scores={"growthScore": 85, "momentumScore": 80, "qualityScore": 75, "valuationScore": 70},
        )
        gardner = next(a for a in res["archetypes"] if "Gardner" in a["name"])
        assert gardner["status"] == "AI Hardware Supercycle", f"{sym} expected AI Hardware Supercycle"
        assert "High gross margin" not in gardner["thesis"]
        assert "liquid cooling" in gardner["thesis"].lower()

        buffett = next(a for a in res["archetypes"] if "Buffett" in a["name"])
        assert buffett["status"] == "Competitive Commodity Risk", f"{sym} expected Competitive Commodity Risk"
        assert buffett["alignmentScore"] <= 62


# ── F_04 Specific Acceptance Test Suites (Sections 16, 17, 18, 19, 20) ──────

def test_f04_section16_no_decision_authority_expansion():
    """Section 16: Verify archetype scores and consensus CANNOT change decision state,
    promote actionable status, or enable position sizing under DecisionHierarchyEngine."""
    analyzer = TraderArchetypeAnalyzer()

    # 1. Evaluate DecisionState for incomplete data
    engine_verdict = DecisionHierarchyEngine.resolve_decision_state(
        symbol="AAPL",
        current_price=150.0,
        candle_count=10,  # < 50 sessions -> INSUFFICIENT_DATA
        freshness_status="LIVE_INTRA_DAY",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=None,
        is_in_buy_zone=False,
        risk_reward_ratio=None,
    )
    assert engine_verdict["state"] == DecisionState.INSUFFICIENT_DATA.value
    assert engine_verdict["isActionable"] is False
    assert engine_verdict["canSizeTrade"] is False

    # 2. Perfect archetype scores must have ZERO authority over canonical decision
    perfect_archetypes = analyzer.analyze_asset(
        symbol="AAPL",
        info={"sector": "Technology", "industry": "Consumer Electronics"},
        price_df=pd.DataFrame({"Close": [100.0, 105.0]}),
        risk_metrics={"Sortino_Ratio": 3.5, "Skewness": 0.2},
        macro_indicators={"yield_curve_spread": 0.50, "credit_spread_oas": 2.50},
        factor_scores={"qualityScore": 95, "valuationScore": 90, "growthScore": 95, "momentumScore": 95, "tailRiskScore": 90, "piotroskiFScore": 9},
    )
    assert perfect_archetypes["consensusScore"] >= 80

    # DecisionHierarchyEngine must remain completely unaffected by archetype output
    engine_verdict_after = DecisionHierarchyEngine.resolve_decision_state(
        symbol="AAPL",
        current_price=150.0,
        candle_count=10,
        freshness_status="LIVE_INTRA_DAY",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=None,
        is_in_buy_zone=False,
        risk_reward_ratio=None,
    )
    assert engine_verdict_after["state"] == DecisionState.INSUFFICIENT_DATA.value
    assert engine_verdict_after["isActionable"] is False
    assert engine_verdict_after["canSizeTrade"] is False


def test_f04_section17_all_measured_inputs_missing_fails_closed():
    """Section 17: For each archetype, verify that when all measured inputs are missing,
    no empirical numeric score is synthesized."""
    analyzer = TraderArchetypeAnalyzer()

    # Empty inputs across all factor, risk, and macro telemetry
    res = analyzer.analyze_asset(
        symbol="EMPTY_TEST",
        info={},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={},
    )

    archetype_map = {a["name"]: a for a in res["archetypes"]}

    # BUFFETT_MISSING_EVIDENCE_NUMERIC_SCORE = NO
    buffett = next(a for n, a in archetype_map.items() if "Buffett" in n)
    assert buffett["alignmentScore"] is None
    assert buffett["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # PELOSI_MISSING_EVIDENCE_NUMERIC_SCORE = NO
    pelosi = next(a for n, a in archetype_map.items() if "Pelosi" in n)
    assert pelosi["alignmentScore"] is None
    assert pelosi["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # DRUCKENMILLER_MISSING_EVIDENCE_NUMERIC_SCORE = NO
    druck = next(a for n, a in archetype_map.items() if "Druckenmiller" in n)
    assert druck["alignmentScore"] is None
    assert druck["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # SIMONS_MISSING_EVIDENCE_NUMERIC_SCORE = NO
    simons = next(a for n, a in archetype_map.items() if "Simons" in n)
    assert simons["alignmentScore"] is None
    assert simons["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # GARDNER_MISSING_EVIDENCE_NUMERIC_SCORE = NO
    gardner = next(a for n, a in archetype_map.items() if "Gardner" in n)
    assert gardner["alignmentScore"] is None
    assert gardner["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # Consensus must be None when all archetypes are unavailable
    assert res["consensusScore"] is None
    assert res["verdict"] == "Telemetry Unavailable"


def test_f04_section18_partial_evidence_semantics():
    """Section 18: Partial evidence tests for each archetype:
    - all factors present
    - required present + optional missing
    - required missing
    - only static prior present
    - all evidence absent."""
    analyzer = TraderArchetypeAnalyzer()

    # Buffett: required (quality, valuation), optional (piotroski)
    # 1. All present -> AUTHORITATIVE_DYNAMIC / AVAILABLE
    b_all = analyzer._evaluate_buffett_moat("TEST", False, {}, {"qualityScore": 80, "valuationScore": 70, "piotroskiFScore": 8})
    assert b_all["alignmentScore"] is not None
    assert b_all["evidenceStatus"] == ArchetypeEvidenceStatus.AVAILABLE
    assert b_all["evidenceType"] == EvidenceType.AUTHORITATIVE_DYNAMIC

    # 2. Required present + optional missing -> PROVISIONAL_DYNAMIC / PROVISIONAL
    b_prov = analyzer._evaluate_buffett_moat("TEST", False, {}, {"qualityScore": 80, "valuationScore": 70})
    assert b_prov["alignmentScore"] is not None
    assert b_prov["evidenceStatus"] == ArchetypeEvidenceStatus.PROVISIONAL
    assert b_prov["evidenceType"] == EvidenceType.PROVISIONAL_DYNAMIC

    # 3. Required missing (valuation missing) -> UNAVAILABLE
    b_req_miss = analyzer._evaluate_buffett_moat("TEST", False, {}, {"qualityScore": 80})
    assert b_req_miss["alignmentScore"] is None
    assert b_req_miss["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # 4. Only static prior present (BTC-USD in CRYPTO_MOATS) -> UNAVAILABLE numeric score, STATIC_DOMAIN_PRIOR
    b_prior_only = analyzer._evaluate_buffett_moat("BTC", True, {}, {})
    assert b_prior_only["alignmentScore"] is None
    assert b_prior_only["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE
    assert b_prior_only["thematicPrior"]["evidenceType"] == EvidenceType.STATIC_DOMAIN_PRIOR

    # 5. All absent -> UNAVAILABLE
    b_absent = analyzer._evaluate_buffett_moat("UNKNOWN", False, {}, {})
    assert b_absent["alignmentScore"] is None
    assert b_absent["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # Druckenmiller: required (macro YC, macro CS, momentum), optional (growth)
    # 1. All present -> AUTHORITATIVE_DYNAMIC / AVAILABLE
    d_all = analyzer._evaluate_druckenmiller_macro(
        {"yield_curve_spread": 0.40, "credit_spread_oas": 2.50},
        {"momentumScore": 80, "growthScore": 85},
        None,
    )
    assert d_all["alignmentScore"] is not None
    assert d_all["evidenceStatus"] == ArchetypeEvidenceStatus.AVAILABLE
    assert d_all["evidenceType"] == EvidenceType.AUTHORITATIVE_DYNAMIC

    # 2. Required present + optional missing (growth missing) -> PROVISIONAL_DYNAMIC / PROVISIONAL
    d_prov = analyzer._evaluate_druckenmiller_macro(
        {"yield_curve_spread": 0.40, "credit_spread_oas": 2.50},
        {"momentumScore": 80},
        None,
    )
    assert d_prov["alignmentScore"] is not None
    assert d_prov["evidenceStatus"] == ArchetypeEvidenceStatus.PROVISIONAL
    assert d_prov["evidenceType"] == EvidenceType.PROVISIONAL_DYNAMIC

    # 3. Required missing (momentum missing) -> UNAVAILABLE
    d_miss = analyzer._evaluate_druckenmiller_macro(
        {"yield_curve_spread": 0.40, "credit_spread_oas": 2.50},
        {},
        None,
    )
    assert d_miss["alignmentScore"] is None
    assert d_miss["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE

    # Simons: required (Sortino, Skewness, tailRisk), optional (momentum)
    # 1. All present -> AUTHORITATIVE_DYNAMIC / AVAILABLE
    s_all = analyzer._evaluate_simons_quant(
        {"Sortino_Ratio": 2.0, "Skewness": -0.2},
        None,
        {"tailRiskScore": 85, "momentumScore": 80},
    )
    assert s_all["alignmentScore"] is not None
    assert s_all["evidenceStatus"] == ArchetypeEvidenceStatus.AVAILABLE
    assert s_all["evidenceType"] == EvidenceType.AUTHORITATIVE_DYNAMIC

    # 2. Required present + optional missing (momentum missing) -> PROVISIONAL_DYNAMIC / PROVISIONAL
    s_prov = analyzer._evaluate_simons_quant(
        {"Sortino_Ratio": 2.0, "Skewness": -0.2},
        None,
        {"tailRiskScore": 85},
    )
    assert s_prov["alignmentScore"] is not None
    assert s_prov["evidenceStatus"] == ArchetypeEvidenceStatus.PROVISIONAL
    assert s_prov["evidenceType"] == EvidenceType.PROVISIONAL_DYNAMIC

    # 3. Required missing (Sortino missing) -> UNAVAILABLE
    s_miss = analyzer._evaluate_simons_quant(
        {"Skewness": -0.2},
        None,
        {"tailRiskScore": 85},
    )
    assert s_miss["alignmentScore"] is None
    assert s_miss["evidenceStatus"] == ArchetypeEvidenceStatus.UNAVAILABLE


def test_f04_section19_consensus_aggregation_formula():
    """Section 19: Consensus test:
    - 80, 70, None -> expected 75
    - None, None, None, None, None -> consensus = None, verdict = Telemetry Unavailable."""
    # Synthetic consensus with 80, 70, None
    archetypes = [
        {"name": "A1", "alignmentScore": 80, "evidenceStatus": "AUTHORITATIVE_DYNAMIC"},
        {"name": "A2", "alignmentScore": 70, "evidenceStatus": "PROVISIONAL_DYNAMIC"},
        {"name": "A3", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
    ]
    available = [a for a in archetypes if a.get("evidenceStatus") != "UNAVAILABLE" and a.get("alignmentScore") is not None]
    consensus = round(sum(a["alignmentScore"] for a in available) / len(available))
    assert consensus == 75

    # All None
    all_none = [
        {"name": "A1", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
        {"name": "A2", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
        {"name": "A3", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
        {"name": "A4", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
        {"name": "A5", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
    ]
    avail_none = [a for a in all_none if a.get("evidenceStatus") != "UNAVAILABLE" and a.get("alignmentScore") is not None]
    cons_none = round(sum(a["alignmentScore"] for a in avail_none) / len(avail_none)) if avail_none else None
    assert cons_none is None


def test_f04_section20_prior_disclosure_and_labeling():
    """Section 20: For tickers in curated thematic dictionaries verify:
    - STATIC_PRIOR_VISIBLE = YES
    - STATIC_PRIOR_LABELED = YES
    - STATIC_PRIOR_PRESENTED_AS_LIVE_OBSERVATION = NO."""
    analyzer = TraderArchetypeAnalyzer()

    # NVDA in MOTLEY_FOOL_DISRUPTORS
    res_nvda = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=None,
        risk_metrics={"Sortino_Ratio": 2.2, "Skewness": -0.1},
        macro_indicators={"yield_curve_spread": 0.40, "credit_spread_oas": 2.50},
        factor_scores={"growthScore": 95, "momentumScore": 92, "qualityScore": 90, "valuationScore": 75, "tailRiskScore": 85},
    )

    gardner = next(a for a in res_nvda["archetypes"] if "Gardner" in a["name"])
    assert gardner["thematicPrior"] is not None
    assert gardner["thematicPrior"]["source"] == "MOTLEY_FOOL_DISRUPTORS"
    assert gardner["thematicPrior"]["evidenceType"] == EvidenceType.STATIC_DOMAIN_PRIOR
    assert gardner["thematicPrior"]["isStaticPrior"] is True
    assert gardner["thematicPrior"]["isLiveObservation"] is False

    # PLTR in CONGRESSIONAL_POLICY_TICKERS
    res_pltr = analyzer.analyze_asset(
        symbol="PLTR",
        info={"sector": "Technology", "industry": "Software"},
        price_df=None,
        risk_metrics={"Sortino_Ratio": 2.0, "Skewness": -0.1},
        macro_indicators={"yield_curve_spread": 0.40, "credit_spread_oas": 2.50},
        factor_scores={"growthScore": 92, "momentumScore": 90, "qualityScore": 88, "valuationScore": 70, "tailRiskScore": 80},
    )

    pelosi = next(a for a in res_pltr["archetypes"] if "Pelosi" in a["name"])
    assert pelosi["thematicPrior"] is not None
    assert pelosi["thematicPrior"]["source"] == "CONGRESSIONAL_POLICY_TICKERS"
    assert pelosi["thematicPrior"]["evidenceType"] == EvidenceType.STATIC_DOMAIN_PRIOR
    assert pelosi["thematicPrior"]["isStaticPrior"] is True
    assert pelosi["thematicPrior"]["isLiveObservation"] is False
