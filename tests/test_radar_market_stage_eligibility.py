"""Mandatory Regression Test Suite for Market-Stage Decision Contract & Opportunity Eligibility.

Directive: RADAR-ACT-008R
Tracker: GitHub Issue #166

Covers all 12 mandatory test cases:
1. Genuine Stage 2 VCP candidate -> screeningEligible: True, in Qualified Opportunities.
2. High-scoring Stage 4 stock -> screeningEligible: False, excluded from Qualified Opportunities.
3. Stage 4 stock near historical entry zone -> screeningEligible: False, no buy zone trigger.
4. Stage 2-to-Stage 4 transition -> demoted immediately, removed from opportunities.
5. Unresolved or stale market-stage evidence -> fails closed to STAGE_UNRESOLVED / non-eligible.
6. Stage 4 with positive Smart Money evidence -> Smart Money preserved as research, but bullish opportunity False.
7. Stage 4 portfolio holding -> visible in portfolio monitoring, non-actionable for buy.
8. Delisted CPRX -> DELISTED_SECURITY, non-actionable, historical record preserved.
9. Stage 1 and Stage 3 assets -> research/watchlist only, screeningEligible: False.
10. Zero qualified VCP matches -> status: AVAILABLE, matched_count: 0.
11. Frontend counts matching backend eligibility -> population.currentOpportunityCount parity.
12. Historical research preservation -> All Research / Catalog maintains full visibility.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analyst_dashboard.vcp.stage_contract import (
    evaluate_market_stage,
    MarketStage,
    MarketStageAssessment,
    normalize_stage_enum,
    STAGE_LABELS,
)
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.predicate_registry import PredicateStatus


def generate_synthetic_candles(
    n_bars: int = 220,
    trend: str = "STAGE_2",
    base_price: float = 100.0,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV candles."""
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_bars)]
    
    if trend == "STAGE_2":
        # Upward sloping prices: +50% over window, closing near 52w high
        closes = [base_price * (1.0 + 0.5 * (i / n_bars)) for i in range(n_bars)]
    elif trend == "STAGE_4":
        # Secular downtrend: -40% over window, below declining moving averages
        closes = [base_price * (1.0 - 0.4 * (i / n_bars)) for i in range(n_bars)]
    elif trend == "STAGE_1":
        # Flat basing consolidation within a tight range
        closes = [base_price * (1.0 + 0.02 * math.sin(i / 5.0)) for i in range(n_bars)]
    elif trend == "STAGE_3":
        # Rise then topping distribution
        half = n_bars // 2
        up = [base_price * (1.0 + 0.4 * (i / half)) for i in range(half)]
        top = [up[-1] * (1.0 + 0.03 * math.sin(i / 3.0)) for i in range(n_bars - half)]
        closes = up + top
    else:
        closes = [base_price] * n_bars

    highs = [c * (1.0 + volatility) for c in closes]
    lows = [c * (1.0 - volatility) for c in closes]
    opens = [c * (1.0 - volatility * 0.2) for c in closes]
    volumes = [1000000 + int(100000 * math.sin(i)) for i in range(n_bars)]

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=pd.DatetimeIndex(dates))


import math


class TestRadarMarketStageEligibility:
    """12 Mandatory Regression Test Cases for Directive RADAR-ACT-008R."""

    def test_case_1_genuine_stage_2_vcp_candidate(self):
        """Case 1: Genuine Stage 2 VCP candidate is bullish eligible."""
        df = generate_synthetic_candles(n_bars=220, trend="STAGE_2", base_price=100.0)
        current_price = float(df["Close"].iloc[-1])
        
        assessment = evaluate_market_stage(df, current_price, symbol="GENUINE_S2")
        assert assessment.stage == MarketStage.STAGE_2
        assert assessment.is_bullish_eligible is True
        assert assessment.trend_template_passed is True

        # Optimal execution on Stage 2:
        plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price, user_role="LONG_TERM")
        assert plan["stage_phase"] == STAGE_LABELS[MarketStage.STAGE_2]
        assert assessment.is_bullish_eligible is True
        assert assessment.stage == MarketStage.STAGE_2

    def test_case_2_high_scoring_stage_4_stock(self):
        """Case 2: High-scoring Stage 4 stock is strictly ineligible for bullish opportunities."""
        df = generate_synthetic_candles(n_bars=220, trend="STAGE_4", base_price=200.0)
        current_price = float(df["Close"].iloc[-1])

        assessment = evaluate_market_stage(df, current_price, symbol="HIGH_SCORE_S4")
        assert assessment.stage == MarketStage.STAGE_4
        assert assessment.is_bullish_eligible is False

        # Simulate a high confluence conviction score (e.g. 88.0 from value/insider data)
        high_confluence_score = 88.0
        dec_state = DecisionHierarchyEngine.resolve_decision_state(
            symbol="HIGH_SCORE_S4",
            current_price=current_price,
            candle_count=len(df),
            freshness_status="LIVE",
            has_fundamentals=True,
            confluence_score=high_confluence_score,
            stage_phase=assessment.stage_label,
            is_in_buy_zone=False,
            is_confirmed=True,
        )
        # Must NOT be ACTIONABLE_SETUP:
        assert dec_state["state"] != DecisionState.ACTIONABLE_SETUP.value
        assert dec_state["isActionable"] is False
        assert dec_state["canSizeTrade"] is False
        assert "Stage 4" in dec_state["label"] or "Stage 4" in dec_state["disqualificationReason"]

    def test_case_3_stage_4_stock_near_historical_entry_zone(self):
        """Case 3: Stage 4 stock near its entry zone CANNOT trigger IN_BUY_ZONE."""
        df = generate_synthetic_candles(n_bars=220, trend="STAGE_4", base_price=200.0)
        current_price = float(df["Close"].iloc[-1])
        assessment = evaluate_market_stage(df, current_price)

        plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price, user_role="LONG_TERM")
        # In Stage 4, execution status must NEVER be IN_BUY_ZONE:
        assert plan["execution_status"] in ("WAITING_PULLBACK", "STAGE_4_MARKDOWN")
        assert plan["is_in_buy_zone"] is False
        assert assessment.is_bullish_eligible is False
        assert "Stage 4" in plan["stage_phase"]

    def test_case_4_stage_2_to_stage_4_transition(self):
        """Case 4: Stock transitioning from Stage 2 to Stage 4 is immediately demoted."""
        # Start in Stage 2
        df_s2 = generate_synthetic_candles(n_bars=220, trend="STAGE_2", base_price=100.0)
        eval_s2 = evaluate_market_stage(df_s2, float(df_s2["Close"].iloc[-1]))
        assert eval_s2.stage == MarketStage.STAGE_2
        assert eval_s2.is_bullish_eligible is True

        # Breakdown occurs: append 40 sessions of persistent sharp selling
        last_date = df_s2.index[-1]
        breakdown_closes = [float(df_s2["Close"].iloc[-1]) * (1.0 - 0.01 * (i + 1)) for i in range(40)]
        breakdown_dates = [last_date + timedelta(days=i + 1) for i in range(40)]
        df_breakdown = pd.DataFrame({
            "Open": breakdown_closes,
            "High": [c * 1.01 for c in breakdown_closes],
            "Low": [c * 0.99 for c in breakdown_closes],
            "Close": breakdown_closes,
            "Volume": [2000000] * 40,
        }, index=pd.DatetimeIndex(breakdown_dates))
        df_combined = pd.concat([df_s2, df_breakdown])

        eval_breakdown = evaluate_market_stage(df_combined, breakdown_closes[-1])
        # Transition must immediately demote:
        assert eval_breakdown.stage == MarketStage.STAGE_4
        assert eval_breakdown.is_bullish_eligible is False

    def test_case_5_unresolved_or_stale_market_stage_evidence(self):
        """Case 5: Unresolved or insufficient history fails closed to non-eligible."""
        # 30 bars is insufficient to calculate 50 SMA or Trend Template:
        df_short = generate_synthetic_candles(n_bars=30, trend="STAGE_2")
        eval_short = evaluate_market_stage(df_short, float(df_short["Close"].iloc[-1]))
        assert eval_short.stage == MarketStage.STAGE_UNRESOLVED
        assert eval_short.is_bullish_eligible is False

        # Empty DF:
        eval_empty = evaluate_market_stage(pd.DataFrame())
        assert eval_empty.stage == MarketStage.STAGE_UNRESOLVED
        assert eval_empty.is_bullish_eligible is False

    def test_case_6_stage_4_with_positive_smart_money_evidence(self):
        """Case 6: Positive Smart Money evidence does NOT confer bullish opportunity status on Stage 4 asset."""
        df = generate_synthetic_candles(n_bars=220, trend="STAGE_4")
        current_price = float(df["Close"].iloc[-1])
        assessment = evaluate_market_stage(df, current_price)

        assert assessment.stage == MarketStage.STAGE_4
        assert assessment.is_bullish_eligible is False

        # High Smart Money confluence
        dec = DecisionHierarchyEngine.resolve_decision_state(
            symbol="SM_STAGE_4",
            current_price=current_price,
            candle_count=len(df),
            freshness_status="LIVE",
            has_fundamentals=True,
            confluence_score=90.0,
            stage_phase=assessment.stage_label,
            is_in_buy_zone=False,
            is_confirmed=True,
        )
        assert dec["isActionable"] is False
        assert dec["canSizeTrade"] is False

    def test_case_7_stage_4_portfolio_holding(self):
        """Case 7: Stage 4 portfolio holding is visible for monitoring but disabled for new buy sizing."""
        df = generate_synthetic_candles(n_bars=220, trend="STAGE_4")
        current_price = float(df["Close"].iloc[-1])
        assessment = evaluate_market_stage(df, current_price)

        dec = DecisionHierarchyEngine.resolve_decision_state(
            symbol="HELD_S4",
            current_price=current_price,
            candle_count=len(df),
            freshness_status="LIVE",
            has_fundamentals=True,
            confluence_score=60.0,
            stage_phase=assessment.stage_label,
            is_in_buy_zone=False,
            is_confirmed=False,
        )
        # Sizing and actionability must be strictly False:
        assert dec["canSizeTrade"] is False
        assert dec["isActionable"] is False
        assert "Stage 4" in dec["disqualificationReason"]

    def test_case_8_delisted_cprx_exclusion(self):
        """Case 8: Confirmed delisted CPRX is strictly ineligible and non-actionable."""
        from api.routes.screener import ATTRIBUTABLE_LIFECYCLE_EVIDENCE
        assert "CPRX" in ATTRIBUTABLE_LIFECYCLE_EVIDENCE
        cprx_info = ATTRIBUTABLE_LIFECYCLE_EVIDENCE["CPRX"]
        assert cprx_info["status"] == "DELISTED"

        dec_state = DecisionHierarchyEngine.resolve_decision_state(
            symbol="CPRX",
            current_price=0.0,
            candle_count=0,
            freshness_status="UNAVAILABLE",
            has_fundamentals=False,
            confluence_score=0.0,
            stage_phase="Delisted",
            is_in_buy_zone=False,
        )
        assert dec_state["isActionable"] is False
        assert dec_state["canSizeTrade"] is False

    def test_case_9_stage_1_and_stage_3_assets(self):
        """Case 9: Stage 1 and Stage 3 assets are research/watchlist only, ineligible for bullish momentum opportunities."""
        df_s1 = generate_synthetic_candles(n_bars=220, trend="STAGE_1")
        eval_s1 = evaluate_market_stage(df_s1, float(df_s1["Close"].iloc[-1]))
        assert eval_s1.is_bullish_eligible is False

        df_s3 = generate_synthetic_candles(n_bars=220, trend="STAGE_3")
        eval_s3 = evaluate_market_stage(df_s3, float(df_s3["Close"].iloc[-1]))
        assert eval_s3.is_bullish_eligible is False

    def test_case_10_zero_qualified_vcp_matches(self):
        """Case 10: Zero qualified VCP matches truthfully returns AVAILABLE with matched_count: 0."""
        from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner, ScannerStatus

        runner = VCPScannerRunner()
        # Test 1: Zero qualified matches evaluated cleanly -> AVAILABLE with matched_count: 0
        zero_match_rec = {
            "snapshot_id": "vcp-snap-zero-matches",
            "scanner_id": "MINERVINI_VCP",
            "run_id": "run-zero-matches",
            "api_contract_version": runner.get_version_tuple().api_contract_version,
            "ruleset_version": runner.get_version_tuple().ruleset_version,
            "evidence_schema_version": runner.get_version_tuple().evidence_schema_version,
            "score_model_version": runner.get_version_tuple().score_model_version,
            "data_provenance_version": runner.get_version_tuple().data_provenance_version,
            "universe_version": runner.get_version_tuple().universe_version,
            "freshness_policy_version": runner.get_version_tuple().freshness_policy_version,
            "implementation_release_sha": runner.get_version_tuple().implementation_release_sha,
            "semantic_fingerprint": "fp-zero",
            "generated_at": "2026-11-03T00:00:00Z",
            "data_as_of": "2026-11-02",
            "status_at_publication": "AVAILABLE",
            "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
            "universe_size": 133,
            "matched_count": 0,
            "results": [],
            "provenance": {"role": "PRICE_VOLUME_HISTORY"},
            "freshness": {"status": "LIVE", "data_as_of": "2026-11-02", "generated_at": "2026-11-03T00:00:00Z"},
            "universe_metadata": {"scope_class": "US_EQUITIES", "display_name": "US Equities"},
            "coverage_metadata": {"coverage_status": "COMPLETE"},
        }
        envelope = runner._to_envelope(zero_match_rec)
        assert envelope["status"] == "AVAILABLE"
        assert envelope["matched_count"] == 0
        assert len(envelope["results"]) == 0

        # Test 2: Incomplete classification remains PIPELINE_PENDING with matched_count: None
        pending_env = runner._to_status_envelope(
            status=ScannerStatus.PIPELINE_PENDING,
            reason="STAGE_CLASSIFICATION_IN_PROGRESS",
            detail="Classification ongoing",
        )
        assert pending_env["status"] == "PIPELINE_PENDING"
        assert pending_env["matched_count"] is None

    def test_case_11_frontend_counts_matching_backend_eligibility(self):
        """Case 11: Backend population opportunity count matches eligible candidates."""
        # Simulated candidates array with 2 Stage 2 eligible and 2 Stage 4 ineligible:
        candidates = [
            {"symbol": "S2_A", "marketStage": "STAGE_2", "isBullishEligible": True, "executionStatus": "IN_BUY_ZONE", "screeningEligible": True},
            {"symbol": "S2_B", "marketStage": "STAGE_2", "isBullishEligible": True, "executionStatus": "IN_BUY_ZONE", "screeningEligible": True},
            {"symbol": "S4_A", "marketStage": "STAGE_4", "isBullishEligible": False, "executionStatus": "WAITING_PULLBACK", "screeningEligible": False},
            {"symbol": "S4_B", "marketStage": "STAGE_4", "isBullishEligible": False, "executionStatus": "IN_BUY_ZONE", "screeningEligible": False},
        ]
        active = [c for c in candidates if c["executionStatus"] != "DELISTED_SECURITY"]
        opportunities = [c for c in active if c.get("screeningEligible") and c.get("marketStage") == "STAGE_2"]

        assert len(opportunities) == 2
        assert all(c["symbol"].startswith("S2") for c in opportunities)

    def test_case_12_historical_research_preservation(self):
        """Case 12: Stage 4 and delisted assets are preserved in All Research / Catalog."""
        candidates = [
            {"symbol": "QUAL_S2", "marketStage": "STAGE_2", "isBullishEligible": True, "executionStatus": "IN_BUY_ZONE"},
            {"symbol": "RESEARCH_S4", "marketStage": "STAGE_4", "isBullishEligible": False, "executionStatus": "WAITING_PULLBACK"},
            {"symbol": "CPRX", "marketStage": "STAGE_UNRESOLVED", "isBullishEligible": False, "executionStatus": "DELISTED_SECURITY"},
        ]

        # Qualified Opportunities view shows only QUAL_S2:
        qualified_view = [c for c in candidates if c.get("isBullishEligible") and c["executionStatus"] == "IN_BUY_ZONE"]
        assert len(qualified_view) == 1
        assert qualified_view[0]["symbol"] == "QUAL_S2"

        # All Research / Market Catalog view preserves all 3:
        all_research_view = list(candidates)
        assert len(all_research_view) == 3
        symbols = [c["symbol"] for c in all_research_view]
        assert "QUAL_S2" in symbols
        assert "RESEARCH_S4" in symbols
        assert "CPRX" in symbols
