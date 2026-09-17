"""Tests for Live-API-Only Epistemic Purity and Zero Fabricated Data Invariants."""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np

from analyst_dashboard.analyzers.self_healing_engine import SelfHealingForecastAuditor
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.catalysts import CatalystEngine
from analyst_dashboard.data.market_db import MarketDatabaseEngine


from analyst_dashboard.analyzers.volatility_forecaster import VolatilityForecaster


def test_self_healing_auditor_rejects_insufficient_history():
    """Verify that fewer than 35 bars returns None for accuracy/hit rate, never fabricated 92.4%."""
    auditor = SelfHealingForecastAuditor()
    short_df = pd.DataFrame({"Close": [100.0 + i for i in range(20)]})
    result = auditor.audit_and_calibrate(
        symbol="TEST",
        price_df=short_df,
        current_risk_metrics={"Modified_VaR_95": 2.5},
        expected_return_data={},
    )
    assert result["accuracyScore"] is None, "Accuracy score must be None for N < 35"
    assert result["hitRatePct"] is None, "Hit rate must be None for N < 35"
    assert result["rmsePct"] is None, "RMSE must be None for N < 35"
    assert "Awaiting Minimum Historical Sample" in result["auditStatus"]


def test_smart_money_options_flow_no_fabricated_live_feed():
    """Verify that get_options_flow returns empty list for live queries without streaming provider."""
    live_flow = SmartMoneyEngine.get_options_flow(symbol="NVDA", include_curated=False)
    assert live_flow == [], "Live options flow without OPRA feed must be empty"

    curated_flow = SmartMoneyEngine.get_options_flow(symbol="NVDA", include_curated=True)
    assert len(curated_flow) > 0, "Curated research archive should be retrievable when explicitly requested"


def test_smart_money_overview_live_and_curated():
    """Verify that live overview sets None for rolling totals, while curated mode calculates archive totals."""
    # Live mode (include_curated=False)
    live_overview = SmartMoneyEngine.get_smart_money_overview(include_curated=False)
    assert live_overview["total_congress_filings_30d"] is None, "Live 30d congress count must be None without live feed"
    assert live_overview["total_sec_insiders_30d"] is None, "Live 30d SEC count must be None without live feed"
    assert live_overview["net_political_sentiment"] is None, "Live political sentiment must be None without live feed"
    assert live_overview["unusual_flow_volume_today"] is None
    assert live_overview["call_to_put_dollar_ratio"] is None
    assert live_overview["congress_trades"] == []

    # Curated mode (include_curated=True)
    curated_overview = SmartMoneyEngine.get_smart_money_overview(include_curated=True)
    assert curated_overview["total_congress_filings_30d"] > 0
    assert curated_overview["total_sec_insiders_30d"] > 0
    assert "Bullish" in curated_overview["net_political_sentiment"] or "Bearish" in curated_overview["net_political_sentiment"]


def test_volatility_forecaster_rejects_insufficient_history():
    """Verify that insufficient history returns is_available: False and None volatility, never numeric 0.0."""
    forecaster = VolatilityForecaster()
    short_data = pd.DataFrame({
        "Close": [100.0 + i * 0.1 for i in range(30)],
    })
    forecast = forecaster.generate_volatility_forecast(short_data, forecast_horizon=30)
    assert forecast.get("is_available") is False, "Volatility forecast must be unavailable for N < 100"
    assert forecast.get("current_volatility") is None, "Current volatility must be None, never 0.0"
    assert forecast.get("forecasted_volatility") == [], "Forecasted volatility series must be empty list"

    # Test fallback forecast with empty returns
    empty_forecast = forecaster._create_fallback_forecast(pd.Series(dtype=float), horizon=10, fallback_type="empty_test")
    assert empty_forecast.current_volatility is None, "Fallback volatility must be None for empty returns"
    assert empty_forecast.forecasted_volatility == [], "Fallback forecasted volatility must be empty list"
    assert empty_forecast.volatility_trend == "insufficient_data"


def test_market_db_no_default_snapshot_pollution():
    """Verify SQLite factor snapshot stores NULL for missing scores, never defaulting to 80 or Strong Buy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = os.path.join(tmpdir, "test_market.db")
        db = MarketDatabaseEngine(db_path=test_db_path)

        # Save incomplete snapshot
        db.save_factor_snapshot("PARTIAL", {
            "currentPrice": 123.45,
            "growthScore": None,
            "qualityScore": None,
            "piotroskiFScore": None,
            "verdict": None,
        })

        retrieved = db.get_factor_snapshot("PARTIAL")
        assert retrieved is not None
        assert retrieved["current_price"] == 123.45
        assert retrieved["growth_score"] is None, "Missing growth score must remain NULL"
        assert retrieved["quality_score"] is None, "Missing quality score must remain NULL"
        assert retrieved["piotroski_f"] is None, "Missing Piotroski must remain NULL"
        assert retrieved["verdict"] is None, "Missing verdict must remain NULL"


def test_confluence_engine_purges_fundamental_imputation():
    """Verify that missing fundamentals produce unavailable status and 0 score, not defaulted 70/65."""
    engine = ConfluenceEngine()

    # Asset with NO fundamentals provided
    result_empty = engine.calculate_confluence(
        symbol="UNKNOWN",
        technical_data={"current_price": 50.0, "rsi": 50.0, "stage": "Stage 2"},
        fundamental_data={},
    )
    fund_pillar = next(p for p in result_empty["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
    assert fund_pillar["status"] == "unavailable"
    assert fund_pillar["score"] == 0.0
    assert "unavailable" in fund_pillar["detail"].lower()

    # Asset with partial fundamentals: only quality provided, no Piotroski or growth
    result_partial = engine.calculate_confluence(
        symbol="PARTIAL",
        technical_data={"current_price": 50.0, "rsi": 50.0, "stage": "Stage 2"},
        fundamental_data={"qualityScore": 85.0},
    )
    fund_pillar_partial = next(p for p in result_partial["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
    assert fund_pillar_partial["status"] == "positive"
    assert fund_pillar_partial["score"] == 85.0
    assert "Piotroski Unassessed" in fund_pillar_partial["detail"]


def test_catalyst_archive_provenance_disclosure():
    """Verify that catalyst reports purge static forecasts in live mode, disclosing provenance in curated mode."""
    engine = CatalystEngine()
    # Live mode (default): zero static multi-year projections or fake milestones
    live_report = engine.get_asset_catalyst_report("NVDA", current_price=125.0, include_curated=False)
    assert live_report["upcoming_milestones"] == [], "Live catalyst report must not include static milestones"
    assert live_report["multi_year_forecast"] == [], "Live catalyst report must not include static multi-year forecasts"

    # Curated mode
    curated_report = engine.get_asset_catalyst_report("NVDA", current_price=125.0, include_curated=True)
    assert curated_report.get("isCuratedArchive") is True
    assert curated_report.get("forecastProvenance") == "Curated Historical Consensus"
    assert curated_report.get("asOfDate") == "2026-09-01"
    assert len(curated_report["multi_year_forecast"]) > 0


def test_smart_money_congress_and_options_routes_provenance():
    """Verify that /congress and /options-flow routes enforce live feed honesty."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    # 1. /congress route defaults to live mode (empty, unavailable)
    res_live_congress = client.get("/api/v1/smart-money/congress")
    assert res_live_congress.status_code == 200
    data_live = res_live_congress.json()
    assert data_live["available"] is False
    assert data_live["status"] == "UNAVAILABLE"
    assert data_live["trades"] == []
    assert "disconnected" in data_live["disclosure"].lower()

    # 2. /congress route returns curated data only when explicitly requested
    res_curated_congress = client.get("/api/v1/smart-money/congress?include_curated=true")
    assert res_curated_congress.status_code == 200
    data_curated = res_curated_congress.json()
    assert data_curated["available"] is True
    assert data_curated["status"] == "CURATED"
    assert len(data_curated["trades"]) > 0
    assert data_curated["dataset_date"] == "2026-08-28"

    # 3. /options-flow route reports available: False when no live stream is connected
    res_options = client.get("/api/v1/smart-money/options-flow")
    assert res_options.status_code == 200
    data_opts = res_options.json()
    assert data_opts["available"] is False
    assert data_opts["is_live"] is False
    assert data_opts["flow"] == []
    assert "unavailable" in data_opts["message"].lower()

    # 4. /options-flow route with include_curated=true must NOT claim "Live OPRA options feed active"
    res_curated_opts = client.get("/api/v1/smart-money/options-flow?include_curated=true")
    assert res_curated_opts.status_code == 200
    data_curated_opts = res_curated_opts.json()
    assert data_curated_opts["status"] == "CURATED"
    assert data_curated_opts["is_live"] is False
    assert "live opra options feed active" not in data_curated_opts["message"].lower()
    assert "archive" in data_curated_opts["message"].lower()


def test_analytics_observed_at_and_fetched_at_separation():
    """Verify that /analytics endpoint reports separated observedAt and fetchedAt metadata."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    res = client.get("/api/v1/analytics/AAPL?period=1mo&interval=1d")
    if res.status_code == 200:
        data = res.json()
        assert "fetchedAt" in data, "fetchedAt must be present in response"
        assert isinstance(data["fetchedAt"], int), "fetchedAt must be millisecond epoch int"
        assert "freshness" in data
        assert "fetchedAt" in data["freshness"]
        if data.get("observedAt") is not None:
            assert isinstance(data["observedAt"], int)
            # observedAt must represent authentic market observation, distinct from fetch time
            assert data["observedAt"] <= data["fetchedAt"], "Observation time cannot be in the future of fetch time"

