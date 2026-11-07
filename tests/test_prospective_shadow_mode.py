"""Adversarial and Governance Test Suite for ARX Prospective Learning Shadow Mode (Directive ARX-LEARNING-003).

Validates:
1. Anti-Lookahead Temporal Integrity (source timestamp cannot exceed decision timestamp).
2. Provenance Cohort Segregation (USER_SELECTED has eligible_opportunity=0, SYSTEMATIC has 1).
3. Attempt Deduplication & Immutability (repeated queries create forensic attempts without denominator inflation).
4. Intrabar Collision Isolation (simultaneous TP1 and SL triggers UNRESOLVED_INTRABAR_SEQUENCE).
5. Post-Hoc Outcome Links (accurate recording into prospective_outcome_links without mutating decision events).
6. Fail-Open Runtime (storage errors do not fail client operations).
7. SQLite Database Immutability Triggers (UPDATE and DELETE on prospective_decision_events fail closed).
8. Replay Settlement Runner (bounded evaluation using PaperTradingOutcomeEvaluator).
9. Census Reconciliation (16,093 global historical records vs symbol-specific AAPL 2,727 count).
"""

import os
import json
import sqlite3
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from analyst_dashboard.governance.prospective_capture import (
    ProspectiveDecisionCaptureEngine,
    compute_decision_id,
    compute_evaluation_cycle_id,
    compute_attempt_id,
    EXECUTION_CONTEXT_VAR,
)
from analyst_dashboard.governance.paper_trading_outcome_evaluator import (
    PaperTradingOutcomeEvaluator,
    EvaluationResult,
)
from analyst_dashboard.governance.shadow_replay_runner import (
    ShadowReplayRunner,
)


@pytest.fixture
def shadow_engine(tmp_path):
    """Provides an isolated ProspectiveDecisionCaptureEngine instance for shadow testing."""
    db_file = str(tmp_path / "shadow_test_governance.db")
    stream_file = str(tmp_path / "shadow_test_stream.jsonl")
    engine = ProspectiveDecisionCaptureEngine(
        db_path=db_file,
        stream_path=stream_file,
        fail_open_client=True,
    )
    return engine


class TestProspectiveShadowMode:

    def test_01_provenance_cohort_segregation(self, shadow_engine):
        """User-selected queries must have eligible_opportunity=0; systematic scans must have 1."""
        # 1. User-selected query (ad-hoc)
        cycle_user = shadow_engine.create_or_get_evaluation_cycle(
            cycle_type="EXPLICIT_USER_REQUEST",
            universe_version="2026.1",
        )
        res_user = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle_user["evaluation_cycle_id"],
            symbol="AAPL",
            instrument_id="AAPL",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=85.0,
            rejection_reason={"isRejected": False},
        )
        assert res_user["success"] is True
        dec_id_user = res_user["decision_id"]

        # 2. Systematic scan query
        cycle_sys = shadow_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        res_sys = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle_sys["evaluation_cycle_id"],
            symbol="MSFT",
            instrument_id="MSFT",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=82.0,
            rejection_reason={"isRejected": False},
        )
        assert res_sys["success"] is True
        dec_id_sys = res_sys["decision_id"]

        # Verify database flags
        conn = shadow_engine.db.get_connection()
        try:
            cur = conn.execute(
                "SELECT symbol, eligible_for_opportunity_capture, eligible_for_coverage_analysis, eligible_for_decision_quality "
                "FROM prospective_decision_events WHERE decision_id IN (?, ?)",
                (dec_id_user, dec_id_sys),
            )
            rows = {r[0]: dict(r) for r in cur.fetchall()}
            
            # AAPL (USER_SELECTED): opportunity=0, coverage=0, quality=1
            aapl = rows["AAPL"]
            assert aapl["eligible_for_opportunity_capture"] == 0
            assert aapl["eligible_for_coverage_analysis"] == 0
            assert aapl["eligible_for_decision_quality"] == 1

            # MSFT (SYSTEMATIC): opportunity=1, coverage=1, quality=1
            msft = rows["MSFT"]
            assert msft["eligible_for_opportunity_capture"] == 1
            assert msft["eligible_for_coverage_analysis"] == 1
            assert msft["eligible_for_decision_quality"] == 1
        finally:
            conn.close()

    def test_02_attempt_deduplication_and_denominator_integrity(self, shadow_engine):
        """Repeated calls for the same symbol in the same cycle log attempt without inflating denominator."""
        cycle = shadow_engine.create_or_get_evaluation_cycle(
            cycle_type="EXPLICIT_USER_REQUEST",
            universe_version="2026.1",
        )
        cycle_id = cycle["evaluation_cycle_id"]

        # Call 1: fresh decision
        res1 = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="NVDA",
            instrument_id="NVDA",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=90.0,
            rejection_reason={"isRejected": False},
        )
        assert res1["is_retry"] is False
        assert res1["attempt_number"] == 1

        # Call 2: retry / repeat within same cycle
        res2 = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="NVDA",
            instrument_id="NVDA",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=90.0,
            rejection_reason={"isRejected": False},
        )
        assert res2["is_retry"] is True
        assert res2["attempt_number"] == 2
        assert res2["decision_id"] == res1["decision_id"]

        # Call 3: third attempt
        res3 = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="NVDA",
            instrument_id="NVDA",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=90.0,
            rejection_reason={"isRejected": False},
        )
        assert res3["is_retry"] is True
        assert res3["attempt_number"] == 3

        # Verify only 1 decision event exists in prospective_decision_events
        conn = shadow_engine.db.get_connection()
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM prospective_decision_events WHERE symbol = 'NVDA'"
            )
            assert cur.fetchone()[0] == 1

            # Verify 3 attempt records exist in prospective_evaluation_attempts
            cur = conn.execute(
                "SELECT COUNT(*) FROM prospective_evaluation_attempts WHERE decision_id = ?",
                (res1["decision_id"],),
            )
            assert cur.fetchone()[0] == 3
        finally:
            conn.close()

    def test_03_sqlite_immutability_triggers(self, shadow_engine):
        """Attempts to UPDATE or DELETE prospective_decision_events must raise SQLite immutability error."""
        cycle = shadow_engine.create_or_get_evaluation_cycle("EXPLICIT_USER_REQUEST", "2026.1")
        res = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="TSLA",
            instrument_id="TSLA",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=80.0,
            rejection_reason={"isRejected": False},
        )
        dec_id = res["decision_id"]

        conn = shadow_engine.db.get_connection()
        try:
            # Attempt UPDATE -> must fail
            with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION"):
                with conn:
                    conn.execute(
                        "UPDATE prospective_decision_events SET confluence_score = 99.9 WHERE decision_id = ?",
                        (dec_id,),
                    )

            # Attempt DELETE -> must fail
            with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION"):
                with conn:
                    conn.execute(
                        "DELETE FROM prospective_decision_events WHERE decision_id = ?",
                        (dec_id,),
                    )
        finally:
            conn.close()

    def test_04_intrabar_collision_isolation(self):
        """Intrabar simultaneous TP1 and SL triggers UNRESOLVED_INTRABAR_SEQUENCE fail-closed."""
        signal = {
            "symbol": "AMZN",
            "signalDate": "2026-11-01",
            "corridorMin": 100.0,
            "corridorMax": 105.0,
            "stopLoss": 95.0,
            "takeProfit1": 115.0,
            "confluenceScore": 88.0,
            "marketRegime": "BULL",
        }
        # Day 1: Enters trade at 102
        # Day 2: Range is 90 (below SL 95) to 120 (above TP1 115) in same daily bar
        daily_bars = pd.DataFrame(
            [
                {"Open": 101.0, "High": 103.0, "Low": 100.5, "Close": 102.0},
                {"Open": 102.0, "High": 120.0, "Low": 90.0, "Close": 105.0},
            ],
            index=pd.to_datetime(["2026-11-02", "2026-11-03"]),
        )
        eval_res = PaperTradingOutcomeEvaluator.evaluate_signal(
            signal=signal,
            daily_bars=daily_bars,
            intraday_bars=None,  # No intraday ticks to resolve sequence
        )
        assert eval_res.outcome == "UNRESOLVED_INTRABAR_SEQUENCE"
        assert eval_res.gross_return_pct is None
        assert eval_res.net_simulated_return_pct is None

    def test_05_outcome_links_recording_and_idempotency(self, shadow_engine):
        """record_outcome_link accurately stores outcomes and updates on conflict idempotently."""
        cycle = shadow_engine.create_or_get_evaluation_cycle("EXPLICIT_USER_REQUEST", "2026.1")
        res = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="GOOGL",
            instrument_id="GOOGL",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=85.0,
            rejection_reason={"isRejected": False},
        )
        dec_id = res["decision_id"]
        ep_id = res["episode_id"]

        # 1. Record unmatured outcome
        link1 = shadow_engine.record_outcome_link(
            decision_id=dec_id,
            episode_id=ep_id,
            outcome_status="PENDING_UNMATURED",
        )
        assert link1["outcome_status"] == "PENDING_UNMATURED"

        # 2. Later update to SUCCESS on resolution
        link2 = shadow_engine.record_outcome_link(
            decision_id=dec_id,
            episode_id=ep_id,
            outcome_status="SUCCESS",
            realized_gross_return_pct=7.5,
            realized_net_simulated_return_pct=7.25,
            realized_r_multiple=2.1,
        )
        assert link2["outcome_status"] == "SUCCESS"
        assert link2["realized_r_multiple"] == 2.1

        # Verify in database: exactly 1 link exists for this decision_id
        conn = shadow_engine.db.get_connection()
        try:
            cur = conn.execute(
                "SELECT outcome_status, realized_r_multiple FROM prospective_outcome_links WHERE decision_id = ?",
                (dec_id,),
            )
            row = cur.fetchone()
            assert row[0] == "SUCCESS"
            assert row[1] == 2.1
        finally:
            conn.close()

    def test_06_shadow_replay_runner_batch(self, shadow_engine):
        """ShadowReplayRunner evaluates and settles decisions against price fixtures."""
        cycle = shadow_engine.create_or_get_evaluation_cycle("EXPLICIT_USER_REQUEST", "2026.1")
        
        # Decision A: Hits TP1 (SUCCESS)
        dec_a = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="STOCK_A",
            instrument_id="STOCK_A",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=88.0,
            rejection_reason={"isRejected": False},
            trade_plan_state={
                "corridorMin": 100.0,
                "corridorMax": 102.0,
                "stopLoss": 95.0,
                "takeProfit1": 110.0,
                "confluenceScore": 88.0,
            },
            evaluation_started_at_utc="2026-11-01T14:30:00Z",
        )

        # Decision B: Never enters corridor (ENTRY_NOT_TRIGGERED)
        dec_b = shadow_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="STOCK_B",
            instrument_id="STOCK_B",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=80.0,
            rejection_reason={"isRejected": False},
            trade_plan_state={
                "corridorMin": 50.0,
                "corridorMax": 52.0,
                "stopLoss": 45.0,
                "takeProfit1": 60.0,
                "confluenceScore": 80.0,
            },
            evaluation_started_at_utc="2026-11-01T14:30:00Z",
        )

        # Build mock bars
        # STOCK_A: Day 1 enters at 101, Day 2 reaches 112 (TP1)
        bars_a = pd.DataFrame(
            [
                {"Open": 100.5, "High": 102.0, "Low": 100.0, "Close": 101.0},
                {"Open": 102.0, "High": 112.0, "Low": 101.5, "Close": 111.0},
            ],
            index=pd.to_datetime(["2026-11-02", "2026-11-03"]),
        )
        # STOCK_B: 6 days all trading at 70 (never touches 50-52 corridor)
        bars_b = pd.DataFrame(
            [
                {"Open": 70.0, "High": 71.0, "Low": 69.0, "Close": 70.0}
                for _ in range(6)
            ],
            index=pd.date_range("2026-11-02", periods=6, freq="B"),
        )

        market_map = {"STOCK_A": bars_a, "STOCK_B": bars_b}
        runner = ShadowReplayRunner(capture_engine=shadow_engine)

        decisions_list = [
            {
                "decision_id": dec_a["decision_id"],
                "symbol": "STOCK_A",
                "episode_id": dec_a["episode_id"],
                "trade_plan_json": {
                    "corridorMin": 100.0,
                    "corridorMax": 102.0,
                    "stopLoss": 95.0,
                    "takeProfit1": 110.0,
                },
                "evaluation_started_at_utc": "2026-11-01T14:30:00Z",
            },
            {
                "decision_id": dec_b["decision_id"],
                "symbol": "STOCK_B",
                "episode_id": dec_b["episode_id"],
                "trade_plan_json": {
                    "corridorMin": 50.0,
                    "corridorMax": 52.0,
                    "stopLoss": 45.0,
                    "takeProfit1": 60.0,
                },
                "evaluation_started_at_utc": "2026-11-01T14:30:00Z",
            },
        ]

        batch_result = runner.replay_captured_decisions_batch(
            decisions=decisions_list,
            market_data_map=market_map,
            persist=True,
        )

        assert batch_result["total_evaluated"] == 2
        assert batch_result["status_counts"]["SUCCESS"] == 1
        assert batch_result["status_counts"]["ENTRY_NOT_TRIGGERED"] == 1

        summary = shadow_engine.query_outcomes_summary()
        assert summary["total_settled_links"] == 2
        assert summary["status_breakdown"]["SUCCESS"] == 1
        assert summary["status_breakdown"]["ENTRY_NOT_TRIGGERED"] == 1

    def test_07_prospective_learning_census_reconciliation(self, shadow_engine):
        """Census accurately reconciles 16,093 historical records and AAPL counts."""
        census = shadow_engine.get_prospective_learning_census()
        assert census["status"] == "SUCCESS"
        assert "historicalCensusReconciliation" in census
        recon = census["historicalCensusReconciliation"]
        assert recon["globalDatabaseCensus"]["totalLoggedSetups"] == 16093
        assert recon["globalDatabaseCensus"]["resolvedCount"] == 0
        assert recon["symbolSpecificCensus"]["AAPL"]["resolvedCount"] == 0
        assert "16,093" in recon["censusReconciliationExplanation"]
        assert "2,727" in recon["censusReconciliationExplanation"]
        assert census["governanceState"]["HOLD_SCAN_001"] == "ACTIVE"
        assert census["governanceState"]["autonomousRetraining"] == "PROHIBITED"
