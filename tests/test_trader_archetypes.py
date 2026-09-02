"""Tests for Trader Archetype Strategy Models (Buffett, Pelosi, Druckenmiller, Simons, Gardner)."""

import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.trader_archetypes import TraderArchetypeAnalyzer


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

    # 1. Wide-Moat Compounder (AAPL)
    res_aapl = analyzer.analyze_asset(
        symbol="AAPL",
        info={"sector": "Technology", "industry": "Consumer Electronics"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"qualityScore": 92, "valuationScore": 70, "piotroskiFScore": 8},
    )
    buffett_aapl = next(a for a in res_aapl["archetypes"] if "Buffett" in a["name"])
    assert buffett_aapl["alignmentScore"] >= 90
    assert buffett_aapl["status"] == "High Moat Alignment"
    assert "pricing power" in buffett_aapl["thesis"]

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

    # 5. Crypto Moat Proxy (BTC-USD)
    res_btc = analyzer.analyze_asset(
        symbol="BTC-USD",
        info={},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={},
    )
    buffett_btc = next(a for a in res_btc["archetypes"] if "Buffett" in a["name"])
    assert buffett_btc["alignmentScore"] >= 75
    assert buffett_btc["status"] == "Tier-1 Network Moat"


def test_nancy_pelosi_congressional_policy_coverage():
    """Nancy Pelosi Model: Key legislative policy beneficiaries and sector-aware policy fallbacks."""
    analyzer = TraderArchetypeAnalyzer()

    # 1. Direct Policy Beneficiary (PLTR)
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
    assert "consensusScore" in res
    assert len(res["archetypes"]) == 5
    for a in res["archetypes"]:
        assert isinstance(a["alignmentScore"], (int, float))
        assert 0 <= a["alignmentScore"] <= 100
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



