"""Tests for Legendary Hidden Gems Discovery Screener Engine."""

import pytest
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from api.routes.screener import run_screener, ScreenerRequest


def test_hidden_gems_screener_scoring():
    screener = HiddenGemsScreener()
    results = screener.evaluate_candidates(["PLTR", "CRWD", "NVDA", "BTC-USD"])

    assert len(results) == 4
    for gem in results:
        assert "composite_score" in gem
        assert "lynch_score" in gem
        assert "greenblatt_score" in gem
        assert "growth_score" in gem
        assert "expert_model" in gem
        assert gem["composite_score"] > 50


def test_screener_api_endpoint():
    req = ScreenerRequest(tickers=["PLTR", "ENPH", "ETH-USD"])
    res = run_screener(req)
    assert res["total_candidates"] == 3
    assert res["gems_found"] == 3
    assert len(res["results"]) == 3

