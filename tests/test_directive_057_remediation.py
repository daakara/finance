"""
ARX TERMINAL - DIRECTIVE 057 TARGETED REGRESSION & REMEDIATION VERIFICATION SUITE

Validates:
1. Confluence Evidence-Truthfulness (DEF-056-001 / Remediation R1)
   - Edge Cases 1-10: All pillar combinations, dynamic reweighting, narrative accuracy.
   - Invariant: Never assert Smart Money agreement when Smart Money is unavailable.
   - Invariant: Differentiate institutional confirmation from multi-factor modeled alignment.
2. Position Sizing Integrity & Risk-Budget Floor (DEF-056-002 / Remediation R2)
   - Boundary Fixtures R2-A to R2-I: Sub-1-share, exact 1-share, multiple, fractional floor,
     invalid geometry, non-actionable, execution hazard, margin illustration, invalid params.
3. Modeled Risk Disclosures & Economic Truthfulness (DEF-056-003 / Remediation R3)
   - Guidance explicitly states modeled risk to stop and gap/slippage disclosures.
"""

import math
import pytest
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine


# ==============================================================================
# REMEDIATION R1: CONFLUENCE EVIDENCE-TRUTHFULNESS TESTS (EDGE CASES 1-10)
# ==============================================================================

def test_r1_case_01_all_four_pillars_present_and_positive():
    """Case 1: All 4 pillars positive -> Full institutional confirmation."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="AAPL",
        technical_data={"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8, "current_price": 150.0, "stop_loss": 140.0},
        smart_money_data={"has_insider_buy": True, "has_congress_buy": True},
        fundamental_data={"piotroski_f": 9, "qualityScore": 90.0, "growthScore": 85.0, "valuationScore": 80.0},
        macro_data={"yield_curve_10y2y": 0.35, "credit_spread": 3.0},
    )
    assert res["confluenceScore"] >= 80.0
    assert res["confluenceRating"] == "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT"
    assert "smart money flow" in res["bottomLine"]
    assert "4/4" in res["plainBadge"] or "4-Pillar" in res["confluenceBadge"]


def test_r1_case_02_smart_money_unavailable_three_positive():
    """Case 2: Tech, Fund, Macro positive; Smart Money unavailable (DEF-056-001 core)."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="AAPL",
        technical_data={"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8, "current_price": 150.0, "stop_loss": 140.0},
        smart_money_data=None,  # Smart money unavailable
        fundamental_data={"piotroski_f": 8, "qualityScore": 88.0, "growthScore": 80.0, "valuationScore": 75.0},
        macro_data={"yield_curve_10y2y": 0.35, "credit_spread": 3.0},
    )
    assert res["confluenceScore"] >= 80.0
    # Must differentiate from full institutional confirmation
    assert res["confluenceRating"] == "HIGH-CONVICTION MULTI-FACTOR ALIGNMENT"
    # Narrative MUST NOT claim smart money flow is synchronised
    assert "smart money flow" not in res["bottomLine"]
    assert "technicals, balance sheet quality and macro tailwinds are synchronised" in res["bottomLine"].lower()
    assert res["coverageRatio"] == 0.75


def test_r1_case_03_smart_money_neutral():
    """Case 3: Smart Money present but neutral -> Does not claim smart money synchronised."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="AAPL",
        technical_data={"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8, "current_price": 150.0, "stop_loss": 140.0},
        smart_money_data={"has_insider_buy": False, "has_congress_buy": False},  # Neutral 50.0
        fundamental_data={"piotroski_f": 9, "qualityScore": 90.0, "growthScore": 85.0, "valuationScore": 80.0},
        macro_data={"yield_curve_10y2y": 0.35, "credit_spread": 3.0},
    )
    # Even if score reaches 80 via high other pillars, smart money is neutral, so NOT in positive list
    if res["confluenceScore"] >= 80.0:
        assert res["confluenceRating"] == "HIGH-CONVICTION MULTI-FACTOR ALIGNMENT"
        assert "smart money flow" not in res["bottomLine"]


def test_r1_case_04_smart_money_present_and_warning():
    """Case 4: Smart Money has net insider selling/warning flags."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="AAPL",
        technical_data={"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8, "current_price": 150.0, "stop_loss": 140.0},
        smart_money_data={"net_institutional_flow_mil": -35.0, "insider_sell_count_90d": 6, "insider_buy_count_90d": 0},
        fundamental_data={"piotroski_f": 7},
        macro_data={"yield_curve_10y2y": 0.20, "credit_spread": 3.5},
    )
    assert "smart money flow" not in res["bottomLine"]


def test_r1_case_05_two_unavailable_pillars():
    """Case 5: Smart Money and Macro unavailable -> Rescaled across Tech and Fund."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="AAPL",
        technical_data={"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8, "current_price": 150.0, "stop_loss": 140.0},
        smart_money_data=None,
        fundamental_data={"piotroski_f": 9, "qualityScore": 90.0, "growthScore": 85.0, "valuationScore": 80.0},
        macro_data=None,
    )
    assert res["coverageRatio"] == 0.50
    if res["confluenceScore"] >= 80.0:
        assert res["confluenceRating"] == "HIGH-CONVICTION MULTI-FACTOR ALIGNMENT"
        assert "smart money flow" not in res["bottomLine"]
        assert "technicals and balance sheet quality are synchronised" in res["bottomLine"].lower()


def test_r1_case_06_missing_technical_data():
    """Case 6: Technical data empty/neutral."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="AAPL",
        technical_data={},
        fundamental_data={"piotroski_f": 8, "qualityScore": 80.0},
        macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.2},
    )
    assert res["pillars"][0]["status"] == "neutral"


def test_r1_case_07_foreign_adr_smart_money_treatment():
    """Case 7: Foreign ADR without SEC Form 4 filings -> Discloses neutral governance status."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="ASML",
        technical_data={"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8, "current_price": 800.0, "stop_loss": 760.0},
        smart_money_data=None,
        fundamental_data={"piotroski_f": 8, "qualityScore": 85.0},
        macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.2},
    )
    smart_pillar = next(p for p in res["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["status"] == "neutral"
    assert "Foreign ADR" in smart_pillar["detail"]
    assert "smart money flow" not in res["bottomLine"]


def test_r1_case_08_high_score_partial_coverage():
    """Case 8: High score (>=80.0) with coverageRatio < 1.0 never claims 4 pillars."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="AAPL",
        technical_data={"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8, "current_price": 150.0, "stop_loss": 140.0},
        smart_money_data=None,
        fundamental_data={"piotroski_f": 9, "qualityScore": 90.0, "growthScore": 85.0, "valuationScore": 80.0},
        macro_data={"yield_curve_10y2y": 0.35, "credit_spread": 3.0},
    )
    assert res["confluenceScore"] >= 80.0
    assert "4-Pillar" not in res["confluenceBadge"]
    assert "4/4" not in res["plainBadge"]
    assert res["coverageRatio"] == 0.75


def test_r1_case_09_medium_score_divergence():
    """Case 9: Mixed signals produce selective rating without false claims."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="XYZ",
        technical_data={"setup_pattern": "Consolidation", "stage_phase": "Stage 1", "rsi_14": 50.0},
        fundamental_data={"piotroski_f": 4},
    )
    assert res["confluenceScore"] < 65.0
    assert "smart money flow" not in res["bottomLine"]


def test_r1_case_10_stale_evidence_handling():
    """Case 10: Stale evidence passes through without asserting live synchronization."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="CPRX",
        technical_data={"setup_pattern": "Correction", "executionStatus": "STOPPED_OUT", "riskRewardRatio": 1.2},
        fundamental_data={},
    )
    assert res["confluenceScore"] <= 48.0
    assert "DEFENSIVE" in res["confluenceRating"]


# ==============================================================================
# REMEDIATION R2: POSITION SIZING INTEGRITY & BOUNDARY FIXTURES (R2-A TO R2-I)
# ==============================================================================

def test_r2_fixture_a_below_one_share_budget():
    """R2-A: $10,000 equity, 1% risk ($100 budget), Entry $200, Stop $50 (Risk/share $150).
    Strict risk budget floor MUST return 0 shares and actualDollarRisk = 0.0.
    """
    res = ConfluenceEngine.calculate_position_size(
        account_equity=10000.0,
        risk_pct=1.0,
        entry_price=200.0,
        stop_loss=50.0,
        take_profit_1=250.0,
    )
    assert res["shares"] == 0
    assert res["executableShares"] == 0
    assert res["actualDollarRisk"] == 0.0
    assert res["positionValue"] == 0.0
    assert res["portfolioAllocationPct"] == 0.0
    assert res["isExecutable"] is False
    assert "Position sizing suppressed (0 shares)" in res["guidance"]
    assert "Buy 0 shares" not in res["guidance"]


def test_r2_fixture_b_exactly_one_share():
    """R2-B: $10,000 equity, 1% risk ($100 budget), Entry $200, Stop $100 (Risk/share $100).
    Exactly 1 share permitted.
    """
    res = ConfluenceEngine.calculate_position_size(
        account_equity=10000.0,
        risk_pct=1.0,
        entry_price=200.0,
        stop_loss=100.0,
        take_profit_1=300.0,
    )
    assert res["shares"] == 1
    assert res["executableShares"] == 1
    assert res["actualDollarRisk"] == 100.0
    assert res["positionValue"] == 200.0
    assert res["isExecutable"] is True
    assert "Buy 1 shares at $200.00" in res["guidance"]


def test_r2_fixture_c_multiple_shares():
    """R2-C: $25,000 equity, 1% risk ($250 budget), Entry $100, Stop $95 (Risk/share $5).
    250 / 5 = 50 shares.
    """
    res = ConfluenceEngine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
    )
    assert res["shares"] == 50
    assert res["executableShares"] == 50
    assert res["actualDollarRisk"] == 250.0
    assert res["positionValue"] == 5000.0
    assert res["isExecutable"] is True


def test_r2_fixture_d_fractional_remainder():
    """R2-D: $10,000 equity, 1% risk ($100 budget), Entry $100, Stop $93 (Risk/share $7).
    Floor(100 / 7) = 14 shares ($98 actual risk, 0.98% actual risk).
    """
    res = ConfluenceEngine.calculate_position_size(
        account_equity=10000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=93.0,
        take_profit_1=115.0,
    )
    assert res["shares"] == 14
    assert res["executableShares"] == 14
    assert res["actualDollarRisk"] == 98.0
    assert "0.98% of portfolio" in res["guidance"]


def test_r2_fixture_e_invalid_geometry():
    """R2-E: Entry <= Stop loss -> defensive error, zero shares."""
    res = ConfluenceEngine.calculate_position_size(
        account_equity=10000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit_1=120.0,
    )
    assert "error" in res
    assert res["shares"] == 0
    assert res["executableShares"] == 0
    assert res["isExecutable"] is False


def test_r2_fixture_f_non_actionable_setup():
    """R2-F: is_actionable=False -> zero executable shares."""
    res = ConfluenceEngine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        is_actionable=False,
        disqualification_reason="Setup below 50 SMA",
    )
    assert res["shares"] == 0
    assert res["executableShares"] == 0
    assert res["isExecutable"] is False
    assert "Execution restricted" in res["guidance"]


def test_r2_fixture_g_liquidity_hazard():
    """R2-G: execution_hazard=True -> zero executable shares."""
    res = ConfluenceEngine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        execution_hazard=True,
    )
    assert res["shares"] == 0
    assert res["executableShares"] == 0
    assert res["isExecutable"] is False
    assert "Execution restricted" in res["guidance"]


def test_r2_fixture_h_valid_margin_illustration():
    """R2-H: Tight stop $0.50 -> 200 shares ($20k exposure = 200% equity).
    Raw mathematical output is preserved without an artificial 100% equity cap in core engine.
    """
    res = ConfluenceEngine.calculate_position_size(
        account_equity=10000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=99.50,
        take_profit_1=102.0,
    )
    assert res["shares"] == 200
    assert res["positionValue"] == 20000.0
    assert res["portfolioAllocationPct"] == 200.0
    assert res["isExecutable"] is True


def test_r2_fixture_i_invalid_financial_parameters():
    """R2-I: Zero/negative equity, NaN equity, zero/negative risk_pct -> fails closed."""
    # Zero equity
    res_zero_eq = ConfluenceEngine.calculate_position_size(
        account_equity=0.0, risk_pct=1.0, entry_price=100.0, stop_loss=95.0, take_profit_1=110.0
    )
    assert res_zero_eq["shares"] == 0
    assert res_zero_eq["isExecutable"] is False

    # Negative equity
    res_neg_eq = ConfluenceEngine.calculate_position_size(
        account_equity=-5000.0, risk_pct=1.0, entry_price=100.0, stop_loss=95.0, take_profit_1=110.0
    )
    assert res_neg_eq["shares"] == 0
    assert res_neg_eq["isExecutable"] is False

    # NaN equity
    res_nan_eq = ConfluenceEngine.calculate_position_size(
        account_equity=float("nan"), risk_pct=1.0, entry_price=100.0, stop_loss=95.0, take_profit_1=110.0
    )
    assert res_nan_eq["shares"] == 0
    assert res_nan_eq["isExecutable"] is False

    # Zero risk pct
    res_zero_risk = ConfluenceEngine.calculate_position_size(
        account_equity=10000.0, risk_pct=0.0, entry_price=100.0, stop_loss=95.0, take_profit_1=110.0
    )
    assert res_zero_risk["shares"] == 0
    assert res_zero_risk["isExecutable"] is False


# ==============================================================================
# REMEDIATION R3: MODELED-RISK LANGUAGE & DISCLOSURE TESTS
# ==============================================================================

def test_r3_modeled_risk_guidance_language():
    """Verify guidance text explicitly qualifies risk as modeled to stop with gap/slippage disclosures."""
    res = ConfluenceEngine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_1=110.0,
    )
    assert "Modeled Risk to Stop" in res["guidance"]
    assert "Actual losses may exceed this amount due to price gaps or slippage" in res["guidance"]
    assert "hard stop" in res["guidance"]
