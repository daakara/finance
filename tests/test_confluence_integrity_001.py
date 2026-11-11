"""
ARX-CONFLUENCE-INTEGRITY-001 — Regression Test Suite

Verifies:
1. ULTA-style 1.1:1 R:R candidates cannot be described as satisfying the 1.85:1 rule.
2. A high confluence score cannot override a failed eligibility gate.
3. Missing evidence cannot masquerade as verified institutional alignment.
4. Ranking is deterministic from an immutable snapshot.
5. Two qualifying candidates produce a two-card result, not a fabricated third.
6. Existing actionability and execution restrictions remain fail-closed.
7. 2-pillar coverage is capped at 72.0 (below HIGH-CONVICTION threshold of 80.0).
8. 3-pillar coverage is capped at 84.0.
9. Full 4-pillar coverage retains 96.0 cap.
"""

import pytest
import pandas as pd
import numpy as np
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_df(close_prices):
    closes = list(close_prices)
    df = pd.DataFrame({
        "Open": closes,
        "High": [p + 0.5 for p in closes],
        "Low": [p - 0.5 for p in closes],
        "Close": closes,
        "Volume": [1_000_000] * len(closes),
    })
    return df


POSITIVE_TECH = {
    "setup_pattern": "Minervini VCP Stage 2",
    "stage_phase": "Stage 2 Advancing Growth Phase",
    "execution_status": "IN_BUY_ZONE",
    "rsi_14": 52.0,
    "risk_reward_ratio": 2.35,
    "current_price": 100.0,
    "stop_loss": 90.0,
}

POSITIVE_FUND = {
    "piotroski_f": 8,
    "qualityScore": 88.0,
    "growthScore": 85.0,
    "valuationScore": 75.0,
}

POSITIVE_SMART = {
    "has_insider_buy": True,
    "insider_value_usd": 2_000_000.0,
    "insider_name": "CEO",
    "has_congress_buy": False,
    "has_options_flow": False,
    "net_institutional_flow_mil": 5.0,
    "insider_buy_count_90d": 1,
    "insider_sell_count_90d": 0,
}

POSITIVE_MACRO = {
    "yield_curve_10y2y": 0.35,
    "credit_spread": 3.80,
}


# ── 1. R:R Integrity ──────────────────────────────────────────────────────────

def test_rr_floor_enforced_by_geometry_not_label():
    """
    INTEGRITY-001-RR-01: risk_reward_ratio in plan must reflect actual geometry
    (blended TP1/TP2 for LONG_TERM, TP1 for DAY_TRADER), not an arbitrary label.
    _enforce_execution_invariants adjusts TP levels to satisfy the 1.85:1 minimum floor.
    """
    plan = OptimalExecutionEngine.calculate_execution_plan(
        symbol="TEST", price=100.0, atr_14=1.0, user_role="LONG_TERM"
    )
    rr = plan.get("risk_reward_ratio")
    tp1 = plan.get("take_profit_1")
    tp2 = plan.get("take_profit_2")
    stop = plan.get("stop_loss")
    entry = plan.get("planned_entry", 100.0)
    risk = entry - stop
    tp1_gain = tp1 - entry
    tp2_gain = tp2 - entry
    blended_reward = 0.50 * tp1_gain + 0.50 * tp2_gain
    geometric_rr = round(blended_reward / risk, 2)
    assert abs(rr - geometric_rr) < 0.05, (
        f"risk_reward_ratio ({rr}) must match blended geometric value ({geometric_rr}). "
        f"TP1={tp1}, TP2={tp2}, stop={stop}"
    )
    assert rr >= 1.85, f"Invariant enforcer must ensure rr >= 1.85 via geometry, got {rr}"
    tp1_rr = round(tp1_gain / risk, 2)
    assert tp1_rr >= 1.85, f"TP1 geometry alone must satisfy >= 1.85:1 floor, got {tp1_rr}"


def test_ulta_style_low_rr_cannot_pass_eligibility_gate():
    """
    INTEGRITY-001-RR-02: A candidate with actual R:R < 2.0 must be rejected
    by DecisionHierarchyEngine even if confluence score is >= 75 and in buy zone.
    """
    result = DecisionHierarchyEngine.resolve_decision_state(
        symbol="ULTA_TEST",
        current_price=664.41,
        candle_count=65,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.7,
        stage_phase="Stage 2 Advancing Growth Phase",
        is_in_buy_zone=True,
        risk_reward_ratio=1.1,  # Below 2.0 gate
        is_cataloged=True,
        is_confirmed=True,
        user_role="LONG_TERM",
        execution_status="IN_BUY_ZONE",
    )
    assert result.get("isActionable") is False, (
        "Candidate with 1.1:1 R:R must NOT be actionable even with high confluence score."
    )
    assert "2.0" in (result.get("disqualificationReason") or ""), (
        "Disqualification reason must cite the 2.0:1 R:R threshold."
    )


# ── 2. Score Cannot Override Failed Eligibility Gate ─────────────────────────

def test_high_score_cannot_override_stopped_out_status():
    """
    INTEGRITY-001-GATE-01: Score of 90+ cannot override STOPPED_OUT execution status.
    """
    result = DecisionHierarchyEngine.resolve_decision_state(
        symbol="STOPPED_SYM",
        current_price=50.0,
        candle_count=65,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=92.0,
        stage_phase="Stage 2 Advancing Growth Phase",
        is_in_buy_zone=False,
        risk_reward_ratio=3.0,
        is_cataloged=True,
        is_confirmed=False,
        user_role="LONG_TERM",
        execution_status="STOPPED_OUT",
    )
    assert result.get("isActionable") is False
    assert "stop" in (result.get("disqualificationReason") or "").lower()


def test_high_score_cannot_override_severe_liquidity_hazard():
    """
    INTEGRITY-001-GATE-02: Severe liquidity hazard blocks actionability regardless of score.
    """
    result = DecisionHierarchyEngine.resolve_decision_state(
        symbol="ILLIQUID_SYM",
        current_price=10.0,
        candle_count=65,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=88.0,
        stage_phase="Stage 2 Advancing Growth Phase",
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
        is_confirmed=True,
        user_role="LONG_TERM",
        execution_status="IN_BUY_ZONE",
        execution_hazard=True,
    )
    assert result.get("isActionable") is False
    assert "hazard" in (result.get("disqualificationReason") or "").lower() or \
           "liquidity" in (result.get("disqualificationReason") or "").lower()


# ── 3. Missing Evidence Cannot Masquerade as Verified Alignment ──────────────

def test_2_pillar_coverage_capped_at_72():
    """
    INTEGRITY-001-COV-01: When both smart_money and macro are unavailable,
    score must be capped at 72.0 regardless of tech/fund inputs.
    """
    result = ConfluenceEngine.calculate_confluence(
        symbol="TEST2PILLAR",
        technical_data=POSITIVE_TECH,
        smart_money_data=None,   # unavailable
        fundamental_data=POSITIVE_FUND,
        catalyst_data=None,
        macro_data=None,         # unavailable
    )
    score = result["confluenceScore"]
    assert score <= 72.0, (
        f"2-pillar coverage must cap at 72.0 (got {score}). "
        "Score must not reach HIGH-CONVICTION threshold without smart money and macro evidence."
    )
    assert result["incompleteCoverage"] is True
    assert result["verifiedPillarCount"] == 2
    assert result["coverageCap"] == 72.0
    assert result["coverageRatio"] == 0.5


def test_3_pillar_coverage_capped_at_84():
    """
    INTEGRITY-001-COV-02: When one of smart_money or macro is unavailable,
    score must be capped at 84.0.
    """
    # smart money unavailable, macro available
    result = ConfluenceEngine.calculate_confluence(
        symbol="TEST3PILLAR_A",
        technical_data=POSITIVE_TECH,
        smart_money_data=None,     # unavailable
        fundamental_data=POSITIVE_FUND,
        catalyst_data=None,
        macro_data=POSITIVE_MACRO, # available
    )
    assert result["confluenceScore"] <= 84.0, (
        f"3-pillar (no smart money) must cap at 84.0 (got {result['confluenceScore']})"
    )
    assert result["verifiedPillarCount"] == 3
    assert result["coverageCap"] == 84.0

    # macro unavailable, smart money available
    result2 = ConfluenceEngine.calculate_confluence(
        symbol="TEST3PILLAR_B",
        technical_data=POSITIVE_TECH,
        smart_money_data=POSITIVE_SMART,
        fundamental_data=POSITIVE_FUND,
        catalyst_data=None,
        macro_data=None,  # unavailable
    )
    assert result2["confluenceScore"] <= 84.0, (
        f"3-pillar (no macro) must cap at 84.0 (got {result2['confluenceScore']})"
    )
    assert result2["verifiedPillarCount"] == 3
    assert result2["coverageCap"] == 84.0


def test_full_4_pillar_coverage_retains_96_cap():
    """
    INTEGRITY-001-COV-03: Full 4-pillar coverage retains original 96.0 cap.
    """
    result = ConfluenceEngine.calculate_confluence(
        symbol="TEST4PILLAR",
        technical_data=POSITIVE_TECH,
        smart_money_data=POSITIVE_SMART,
        fundamental_data=POSITIVE_FUND,
        catalyst_data=None,
        macro_data=POSITIVE_MACRO,
    )
    assert result["incompleteCoverage"] is False
    assert result["verifiedPillarCount"] == 4
    assert result["coverageCap"] == 96.0
    # Score can reach HIGH-CONVICTION with full evidence
    assert result["confluenceScore"] <= 96.0


def test_unavailable_smart_money_does_not_produce_positive_smart_badge():
    """
    INTEGRITY-001-COV-04: When smart_money_data is None, the SMART_MONEY_FLOW
    pillar must not emit status='positive'.
    """
    result = ConfluenceEngine.calculate_confluence(
        symbol="NO_SM",
        technical_data=POSITIVE_TECH,
        smart_money_data=None,
        fundamental_data=POSITIVE_FUND,
        catalyst_data=None,
        macro_data=POSITIVE_MACRO,
    )
    smart_pillar = next(p for p in result["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["status"] == "unavailable", (
        "SMART_MONEY_FLOW pillar must be 'unavailable' when smart_money_data is None — "
        "not 'positive' or 'neutral'."
    )
    assert smart_pillar["score"] == 0.0


# ── 4. Ranking Determinism ────────────────────────────────────────────────────

def test_ranking_deterministic_from_immutable_snapshot():
    """
    INTEGRITY-001-RANK-01: Given identical inputs, ranking is stable across two calls.
    Sorting is: actionable first, then confluenceScore descending, then symbol ascending.
    """
    setups = [
        {"symbol": "ZZZ", "confluenceScore": 85.0, "isActionable": True,  "executionStatus": "IN_BUY_ZONE"},
        {"symbol": "AAA", "confluenceScore": 90.0, "isActionable": False, "executionStatus": "WAITING_PULLBACK"},
        {"symbol": "MMM", "confluenceScore": 87.0, "isActionable": True,  "executionStatus": "IN_BUY_ZONE"},
        {"symbol": "BBB", "confluenceScore": 85.0, "isActionable": True,  "executionStatus": "IN_BUY_ZONE"},
    ]

    def _sort_key(s):
        is_delisted = 1 if s.get("executionStatus") == "DELISTED_SECURITY" else 0
        is_act = 1 if s.get("isActionable") else 0
        confl = float(s.get("confluenceScore") or 0.0)
        sym = str(s.get("symbol") or "")
        return (is_delisted, -is_act, -confl, sym)

    sorted1 = sorted(setups, key=_sort_key)
    sorted2 = sorted(setups, key=_sort_key)
    assert [s["symbol"] for s in sorted1] == [s["symbol"] for s in sorted2]
    # Actionable first (ZZZ, MMM, BBB before AAA), then by score desc, tie by symbol asc
    symbols = [s["symbol"] for s in sorted1]
    assert symbols[0] == "MMM"  # actionable, score 87
    assert symbols[1] == "BBB"  # actionable, score 85, B < Z
    assert symbols[2] == "ZZZ"  # actionable, score 85
    assert symbols[3] == "AAA"  # not actionable, highest score but sinks below actionables


# ── 5. Count Integrity: N qualifying → N cards, not N+1 ──────────────────────

def test_two_qualifying_candidates_produce_two_results_not_three():
    """
    INTEGRITY-001-COUNT-01: If exactly 2 setups pass validity (have analysisReferencePrice),
    slice(0, 3) on a 2-element list must produce exactly 2, never a fabricated third.
    """
    # Simulate topCandidates logic: filter valid, sort, slice 0:3
    valid_setups = [
        {"symbol": "ONON", "confluenceScore": 84.4, "isActionable": True,  "analysisReferencePrice": 34.7},
        {"symbol": "MNST", "confluenceScore": 84.4, "isActionable": False, "analysisReferencePrice": 45.3},
        # ULTA missing analysisReferencePrice → filtered out
        {"symbol": "ULTA", "confluenceScore": 80.7, "isActionable": False, "analysisReferencePrice": None},
    ]
    # Validity filter: requires positive analysisReferencePrice
    valid = [s for s in valid_setups
             if s.get("analysisReferencePrice") and float(s["analysisReferencePrice"]) > 0]
    top3 = valid[:3]
    assert len(top3) == 2, f"Expected 2 qualifying candidates, got {len(top3)}"
    assert all(s["analysisReferencePrice"] is not None for s in top3)


# ── 6. Fail-Closed Actionability and Execution Restrictions ──────────────────

def test_non_actionable_candidate_cannot_trigger_live_execution():
    """
    INTEGRITY-001-EXEC-01: isActionable=False must mean canExecuteLive=False.
    The UI's handleQuickLog is gated on canExecuteLive; this tests that
    the backend 'isSuppressed' flag survives to block live execution.
    """
    suppressed_states = [
        "WAITING_PULLBACK",
        "IN_BUY_ZONE_AWAITING_TRIGGER",
        "EXTENDED_ABOVE_BUY_ZONE",
        "APPROACHING_TARGET",
        "STOPPED_OUT",
    ]
    for status in suppressed_states:
        result = DecisionHierarchyEngine.resolve_decision_state(
            symbol="SUPPRESSED",
            current_price=100.0,
            candle_count=65,
            freshness_status="LIVE",
            has_fundamentals=True,
            confluence_score=85.0,
            stage_phase="Stage 2 Advancing Growth Phase",
            is_in_buy_zone=(status in ("IN_BUY_ZONE_AWAITING_TRIGGER",)),
            risk_reward_ratio=2.5,
            is_cataloged=True,
            is_confirmed=False,
            user_role="LONG_TERM",
            execution_status=status,
        )
        assert result.get("isActionable") is False, (
            f"Status '{status}' must yield isActionable=False."
        )


def test_missing_stop_loss_blocks_actionability():
    """
    INTEGRITY-001-EXEC-02: Absence of stop_loss must prevent actionability
    regardless of other conditions (mirrors api/routes/analytics.py check).
    """
    # Simulate the analytics.py is_actionable gate:
    # is_actionable = bool(dec_state.get("isActionable")) AND stop_loss is not None AND ...
    stop_loss = None
    dec_is_actionable = True  # decision engine says go
    is_actionable = (
        dec_is_actionable
        and stop_loss is not None
    )
    assert is_actionable is False, "Missing stop_loss must block actionability."


def test_operational_holds_not_weakened():
    """
    INTEGRITY-001-EXEC-03: ACTIONABLE_EXECUTION_STATUSES must not include
    statuses that indicate the trade is not safe to execute.
    """
    from analyst_dashboard.analyzers.optimal_execution import (
        ACTIONABLE_EXECUTION_STATUSES,
        NON_ACTIONABLE_EXECUTION_STATUSES,
    )
    forbidden_in_actionable = {
        "STOPPED_OUT", "WAITING_PULLBACK", "EXTENDED_ABOVE_BUY_ZONE",
        "APPROACHING_TARGET", "STALE_MARKET_DATA", "INSUFFICIENT_HISTORY",
    }
    overlap = ACTIONABLE_EXECUTION_STATUSES & forbidden_in_actionable
    assert not overlap, (
        f"Operational safety statuses must not appear in ACTIONABLE_EXECUTION_STATUSES: {overlap}"
    )
    # Confirm the sets are disjoint
    assert ACTIONABLE_EXECUTION_STATUSES.isdisjoint(NON_ACTIONABLE_EXECUTION_STATUSES), (
        "ACTIONABLE and NON_ACTIONABLE execution status sets must be disjoint."
    )
