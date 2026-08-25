"""Tests for Self-Healing Forecast Auditor and Market Graph Contagion Engine."""

import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.self_healing_engine import SelfHealingForecastAuditor
from analyst_dashboard.analyzers.market_graph import MarketGraphEngine


def test_self_healing_forecast_auditor():
    auditor = SelfHealingForecastAuditor()
    
    # Create synthetic price series
    dates = pd.date_range("2025-01-01", periods=60)
    prices = [100.0 * (1.0 + 0.005 * i + 0.01 * np.sin(i)) for i in range(60)]
    df = pd.DataFrame({"Close": prices}, index=dates)

    audit = auditor.audit_and_calibrate(
        symbol="NVDA",
        price_df=df,
        current_risk_metrics={"Modified_VaR_95": -2.8},
        expected_return_data={"p50Expected": 18.5},
    )

    assert "accuracyScore" in audit
    assert "hitRatePct" in audit
    assert "rmsePct" in audit
    assert "varBreachRatePct" in audit
    assert "varBreachStatus" in audit
    assert audit["accuracyScore"] > 70.0


def test_market_graph_engine():
    engine = MarketGraphEngine()
    
    graph_nvda = engine.get_relationship_graph("NVDA")
    assert graph_nvda["rootNode"] == "NVDA"
    assert "upstream" in graph_nvda["topology"]
    assert "downstream" in graph_nvda["topology"]
    assert "macro" in graph_nvda["topology"]
    assert "peers" in graph_nvda["topology"]
    assert len(graph_nvda["topology"]["upstream"]) > 0

    graph_btc = engine.get_relationship_graph("BTC-USD")
    assert graph_btc["rootNode"] == "BTC-USD"

