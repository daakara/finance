"""Tests for Elite Trader Archetype Models."""

import pytest
import pandas as pd
from analyst_dashboard.analyzers.trader_archetypes import TraderArchetypeAnalyzer


def test_trader_archetype_analyzer_equity():
    analyzer = TraderArchetypeAnalyzer()
    res = analyzer.analyze_asset(
        symbol="AAPL",
        info={"returnOnAssets": 0.22, "grossMargins": 0.45},
        price_df=pd.DataFrame({"Close": [300, 305, 310]}),
        risk_metrics={"Sortino_Ratio": 2.1, "Skewness": -0.1},
        macro_indicators={"yield_curve_spread": 0.47, "credit_spread_oas": 2.69},
        factor_scores={"qualityScore": 90, "valuationScore": 75, "momentumScore": 80, "growthScore": 85, "tailRiskScore": 82, "piotroskiFScore": 8},
    )

    assert "consensusScore" in res
    assert "verdict" in res
    assert len(res["archetypes"]) == 4
    assert 0 <= res["consensusScore"] <= 100

    names = [a["name"] for a in res["archetypes"]]
    assert any("Warren Buffett" in n for n in names)
    assert any("Nancy Pelosi" in n for n in names)
    assert any("Stanley Druckenmiller" in n for n in names)
    assert any("Jim Simons" in n for n in names)


def test_trader_archetype_analyzer_crypto():
    analyzer = TraderArchetypeAnalyzer()
    
    # Tier-1 network moat (BTC)
    res_btc = analyzer.analyze_asset(
        symbol="BTC-USD",
        info={},
        price_df=pd.DataFrame({"Close": [60000, 62000, 64000]}),
        risk_metrics={"Sortino_Ratio": 1.9, "Skewness": 0.2},
        macro_indicators={"yield_curve_spread": 0.47, "credit_spread_oas": 2.69},
        factor_scores={"qualityScore": 92, "valuationScore": 70, "momentumScore": 85, "growthScore": 90, "tailRiskScore": 75},
    )
    buffett_btc = next(a for a in res_btc["archetypes"] if "Warren Buffett" in a["name"])
    assert buffett_btc["alignmentScore"] >= 70
    assert "Tier-1 Network Moat" in buffett_btc["status"]

    # Speculative altcoin (DOGE)
    res_alt = analyzer.analyze_asset(
        symbol="DOGE-USD",
        info={},
        price_df=pd.DataFrame({"Close": [0.10, 0.09, 0.08]}),
        risk_metrics={"Sortino_Ratio": -0.5, "Skewness": -0.4},
        macro_indicators={"yield_curve_spread": 0.47, "credit_spread_oas": 2.69},
        factor_scores={"qualityScore": 50, "valuationScore": 50, "momentumScore": 40, "growthScore": 40, "tailRiskScore": 40},
    )
    buffett_alt = next(a for a in res_alt["archetypes"] if "Warren Buffett" in a["name"])
    assert buffett_alt["alignmentScore"] < 50
    assert "Speculative Altcoin" in buffett_alt["status"]

