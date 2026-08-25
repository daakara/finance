"""Tests for Model Calibration: ETF scoring, ER drift damping, and Crypto Value moats."""

import pytest
from api.routes.analytics import get_asset_analytics


def test_etf_quality_calibration():
    # SPY should have a robust factor score (>68) due to fund diversification
    res = get_asset_analytics("SPY", "1y")
    assert res["factorScores"]["compositeFactorScore"] >= 68
    assert res["factorScores"]["qualityScore"] >= 80


def test_er_drift_damping():
    # INTC has a massive 1-year surge (>250%), verify 90-day E[R] P50 is reasonably damped (<60%)
    res = get_asset_analytics("INTC", "1y")
    assert res["expectedReturn"]["p50Expected"] < 60.0


def test_crypto_buffett_moat_calibration():
    # ETH has staking yields and protocol fees, verify Buffett Moat recognition
    res = get_asset_analytics("ETH-USD", "1y")
    buffett = next(a for a in res["traderArchetypes"]["archetypes"] if "Warren Buffett" in a["name"])
    assert buffett["alignmentScore"] >= 70
    assert "Tier-1 Network Moat" in buffett["status"]

