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
        info={"returnOnAssets": 0.25, "freeCashflow": 25000000000},
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

