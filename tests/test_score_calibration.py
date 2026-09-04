"""ARX Score Calibration & Monotonicity Test Suite (Phase 20C).

Validates:
  1. Score Monotonicity: Increasing fundamental or technical quality strictly improves conviction score.
  2. Missing-Data Mathematical Compression: Incomplete disclosures cap maximum achievable score below 50.
  3. Single Pillar Containment: One perfect pillar cannot overwhelm missing/deteriorating pillars.
  4. Binary Risk Calibration: Imminent earnings penalizes score to prevent reckless execution before gap events.
"""

import pytest
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine


def test_score_monotonicity_fundamental_quality():
    """Higher fundamental solvency strictly increases the fundamental pillar and overall confluence."""
    base_tech = {"setup_pattern": "Stage 2 Accumulation", "rsi_14": 52.0, "risk_reward_ratio": 2.5}
    base_macro = {"yield_curve_10y2y": 0.25, "credit_spread": 3.2}

    # High quality firm: Piotroski 8, Quality 85
    high_fund = {"piotroski_f": 8, "qualityScore": 85.0, "growthScore": 80.0, "valuationScore": 75.0}
    res_high = ConfluenceEngine.calculate_confluence(
        symbol="TST_HIGH_QUAL",
        technical_data=base_tech,
        fundamental_data=high_fund,
        macro_data=base_macro,
    )

    # Low quality firm: Piotroski 3, Quality 40
    low_fund = {"piotroski_f": 3, "qualityScore": 40.0, "growthScore": 45.0, "valuationScore": 50.0}
    res_low = ConfluenceEngine.calculate_confluence(
        symbol="TST_LOW_QUAL",
        technical_data=base_tech,
        fundamental_data=low_fund,
        macro_data=base_macro,
    )

    assert res_high["confluenceScore"] > res_low["confluenceScore"]
    high_pillar = next(p for p in res_high["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
    low_pillar = next(p for p in res_low["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
    assert high_pillar["score"] > low_pillar["score"]
    assert high_pillar["status"] == "positive"
    assert low_pillar["status"] == "warning"


def test_score_monotonicity_stage_discipline():
    """Stage 2 Accumulation strictly scores higher than Stage 4 Markdown distribution."""
    fund_data = {"piotroski_f": 7, "qualityScore": 75.0, "growthScore": 70.0, "valuationScore": 65.0}

    # Stage 2 Markup setup
    tech_stage_2 = {"setup_pattern": "Stage 2 Accumulation", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8}
    res_stage_2 = ConfluenceEngine.calculate_confluence(
        symbol="TST_STAGE2",
        technical_data=tech_stage_2,
        fundamental_data=fund_data,
    )

    # Stage 4 Distribution setup
    tech_stage_4 = {"setup_pattern": "Stage 4 Correction", "stage_phase": "Stage 4", "rsi_14": 38.0, "risk_reward_ratio": 1.1}
    res_stage_4 = ConfluenceEngine.calculate_confluence(
        symbol="TST_STAGE4",
        technical_data=tech_stage_4,
        fundamental_data=fund_data,
    )

    assert res_stage_2["confluenceScore"] > res_stage_4["confluenceScore"]
    stage2_tech = next(p for p in res_stage_2["pillars"] if p["pillar"] == "TECHNICAL_STRUCTURE")
    stage4_tech = next(p for p in res_stage_4["pillars"] if p["pillar"] == "TECHNICAL_STRUCTURE")
    assert stage2_tech["score"] >= 80.0
    assert stage4_tech["score"] <= 35.0


def test_missing_data_score_compression():
    """When fundamentals and smart money are missing, confluence cannot exceed 50.0 (compressed to defensive zone)."""
    # Perfect technicals (88/100) and healthy macro (60-85/100)
    tech = {"setup_pattern": "Breakout", "rsi_14": 50.0, "risk_reward_ratio": 3.0}
    macro = {"yield_curve_10y2y": 0.30, "credit_spread": 3.0}

    # Empty fundamentals and empty smart money
    res = ConfluenceEngine.calculate_confluence(
        symbol="TST_COMPRESSED",
        technical_data=tech,
        fundamental_data={},
        smart_money_data={},
        macro_data=macro,
    )

    # Missing pillars score 0.0, compressing the composite
    assert res["confluenceScore"] < 50.0
    assert res["badgeColor"] in ["rose", "amber"]
    assert "GREEN LIGHT" not in res["plainRating"]


def test_single_pillar_cannot_overwhelm_deteriorating_system():
    """A perfect technical score cannot overpower weak fundamentals, bad macro, and binary catalyst risk."""
    perfect_tech = {"setup_pattern": "Stage 2 VCP Breakout", "rsi_14": 50.0, "risk_reward_ratio": 4.0}
    distressed_fund = {"piotroski_f": 2, "qualityScore": 30.0, "growthScore": 25.0, "valuationScore": 30.0}
    recession_macro = {"yield_curve_10y2y": -0.80, "credit_spread": 7.5}
    imminent_earnings = {"days_to_earnings": 0.5}

    res = ConfluenceEngine.calculate_confluence(
        symbol="TST_CONTAINMENT",
        technical_data=perfect_tech,
        fundamental_data=distressed_fund,
        macro_data=recession_macro,
        catalyst_data=imminent_earnings,
    )

    assert res["confluenceScore"] <= 40.0
    assert res["confluenceRating"] == "DEFENSIVE / CAPITAL PRESERVATION MODE"


def test_binary_catalyst_penalty_calibration():
    """Earnings event within 24h triggers a mandatory -25.0 binary gap risk penalty."""
    tech = {"setup_pattern": "Stage 2 Accumulation", "rsi_14": 55.0, "risk_reward_ratio": 2.5}
    fund = {"piotroski_f": 7, "qualityScore": 75.0}

    res_no_earnings = ConfluenceEngine.calculate_confluence(
        symbol="TST_EARN_CALIB",
        technical_data=tech,
        fundamental_data=fund,
        catalyst_data=None,
    )

    res_imminent_earnings = ConfluenceEngine.calculate_confluence(
        symbol="TST_EARN_CALIB",
        technical_data=tech,
        fundamental_data=fund,
        catalyst_data={"days_to_earnings": 0.8},
    )

    delta = res_no_earnings["confluenceScore"] - res_imminent_earnings["confluenceScore"]
    assert delta >= 24.0
    assert any("HIGH BINARY GAP RISK" in w for w in res_imminent_earnings["warnings"])
