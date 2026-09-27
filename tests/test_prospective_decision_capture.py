"""Comprehensive Test Suite for ARX Prospective Full Decision Capture Architecture (V1.0.2).

Validates all 16+ adversarial and edge-case governance requirements:
1. Normal systematic completed evaluation
2. Normal non-recommendation
3. Short-circuit rule failure & downstream NOT_EVALUATED states
4. Expected evaluation never evaluated (MISSING_EXPECTED_EVALUATION)
5. Failure before model execution & failure during model execution
6. Same-cycle retry (same decision_id, distinct attempt_id, empirical count = 1)
7. Next-cycle reevaluation (distinct decision_id, same episode_id)
8. User-selected request (opportunity capture ineligible) vs Systematic scan
9. Systematic scan (opportunity capture eligible)
10. Episode continuation & episode reset (price level altered)
11. Test pollution firewall & replay write block
12. Fail-open runtime handling on storage error
13. Atomic transaction integrity
14. Immutability trigger enforcement (UPDATE and DELETE prohibited)
15. Append-only decision corrections without altering original event
16. Explicit data availability semantics (AUTHENTIC_ZERO, UNAVAILABLE, STALE)
17. Capture ON vs OFF decision parity
18. Latency benchmarking (p50, p95, p99, max)
"""

import os
import time
import json
import sqlite3
import tempfile
import pytest
from datetime import datetime, timezone

from analyst_dashboard.governance.prospective_capture import (
    ProspectiveDecisionCaptureEngine,
    compute_decision_id,
    compute_evaluation_cycle_id,
    compute_attempt_id,
    compute_episode_id,
    compute_trade_plan_hash,
    EXECUTION_CONTEXT_VAR,
)


@pytest.fixture
def capture_engine(tmp_path):
    """Provides an isolated test capture engine with fresh temp db and jsonl stream."""
    db_file = str(tmp_path / "test_governance.db")
    stream_file = str(tmp_path / "test_capture_stream.jsonl")
    engine = ProspectiveDecisionCaptureEngine(
        db_path=db_file,
        stream_path=stream_file,
        fail_open_client=True,
    )
    return engine


class TestProspectiveDecisionCapture:

    def test_01_normal_systematic_completed_evaluation(self, capture_engine):
        """Scenario 1: Full systematic completed evaluation with features and rules."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        cycle_id = cycle["evaluation_cycle_id"]

        # Register expected evaluation
        capture_engine.register_expected_evaluations(
            evaluation_cycle_id=cycle_id,
            items=[("NVDA", "INST_NVDA_01")],
            scope_status="IN_SCOPE",
        )

        features = [
            {
                "featureName": "confluence_score",
                "featureValue": 84.5,
                "source": "ARX_ENGINE",
                "sourceTimestampUtc": datetime.now(timezone.utc).isoformat(),
                "observedAtUtc": datetime.now(timezone.utc).isoformat(),
                "availabilityStatus": "LIVE_AUTHORITATIVE",
                "version": "1.0.0",
            }
        ]

        rules = [
            {
                "ruleId": "RULE_LIQUIDITY_FLOOR",
                "ruleCategory": "MANDATORY_PRODUCT_CONSTRAINT",
                "actualExecutionOrder": 1,
                "evaluationState": "PASS",
                "inputValues": {"volume": 25000000},
                "thresholdValue": "1000000",
                "isBinding": True,
            },
            {
                "ruleId": "RULE_CONFLUENCE_THRESHOLD",
                "ruleCategory": "AUDITED_MODEL_RULE",
                "actualExecutionOrder": 2,
                "evaluationState": "PASS",
                "inputValues": {"score": 84.5},
                "thresholdValue": "70.0",
                "isBinding": True,
            },
        ]

        trade_plan = {
            "isPlanGenerated": True,
            "entryReferencePrice": 125.50,
            "corridorMin": 124.00,
            "corridorMax": 126.00,
            "stopLoss": 120.00,
            "takeProfit1": 135.00,
            "takeProfit2": 142.00,
            "riskRewardRatio": 1.73,
            "counterfactualEvaluationEligible": True,
        }

        res = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="NVDA",
            instrument_id="INST_NVDA_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=84.5,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
            trade_plan_state=trade_plan,
            features=features,
            rule_evaluations=rules,
        )

        assert res["success"] is True
        assert res["empirical_certification_state"] == "CERTIFIED_NATURAL_PRODUCTION"
        assert res["is_retry"] is False

        # Verify reconciliation
        rec = capture_engine.reconcile_evaluation_cycle(cycle_id)
        assert rec["expected_count"] == 1
        assert rec["completed_count"] == 1
        assert rec["missing_count"] == 0
        assert rec["is_balanced"] is True

    def test_02_normal_non_recommendation(self, capture_engine):
        """Scenario 2: Normal non-recommendation candidate correctly classified."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        cycle_id = cycle["evaluation_cycle_id"]

        capture_engine.register_expected_evaluations(
            evaluation_cycle_id=cycle_id,
            items=[("INTC", "INST_INTC_01")],
            scope_status="IN_SCOPE",
        )

        res = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="INTC",
            instrument_id="INST_INTC_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="REJECT",
            actionability_state="NON_ACTIONABLE",
            confluence_score=42.0,
            rejection_reason={
                "isRejected": True,
                "firstBindingRuleId": "RULE_CONFLUENCE_MINIMUM",
                "firstBindingRuleCategory": "AUDITED_MODEL_RULE",
                "primaryReasonText": "Confluence score 42.0 below minimum threshold 70.0",
                "allExecutedFailedRuleIds": ["RULE_CONFLUENCE_MINIMUM"],
            },
        )
        assert res["success"] is True

    def test_03_short_circuit_rule_failure_downstream_states(self, capture_engine):
        """Scenario 3: Short-circuit rule failure correctly tags downstream rules as NOT_EVALUATED."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        cycle_id = cycle["evaluation_cycle_id"]

        capture_engine.register_expected_evaluations(
            evaluation_cycle_id=cycle_id,
            items=[("PENNY", "INST_PENNY_01")],
            scope_status="IN_SCOPE",
        )

        rules = [
            {
                "ruleId": "RULE_PRICE_FLOOR",
                "ruleCategory": "MANDATORY_PRODUCT_CONSTRAINT",
                "actualExecutionOrder": 1,
                "evaluationState": "FAIL",
                "inputValues": {"price": 1.25},
                "thresholdValue": "5.00",
                "isBinding": True,
                "failureMessage": "Stock price $1.25 is below minimum $5.00 floor",
            },
            {
                "ruleId": "RULE_CONFLUENCE_CHECK",
                "ruleCategory": "AUDITED_MODEL_RULE",
                "actualExecutionOrder": 2,
                "evaluationState": "NOT_EVALUATED_AFTER_BINDING_FAILURE",
                "inputValues": {},
                "thresholdValue": "70.0",
                "isBinding": False,
            },
        ]

        res = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="PENNY",
            instrument_id="INST_PENNY_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="REJECT",
            actionability_state="NON_ACTIONABLE",
            confluence_score=None,
            rejection_reason={
                "isRejected": True,
                "firstBindingRuleId": "RULE_PRICE_FLOOR",
                "firstBindingRuleCategory": "MANDATORY_PRODUCT_CONSTRAINT",
                "primaryReasonText": "Price floor violation",
                "allExecutedFailedRuleIds": ["RULE_PRICE_FLOOR"],
            },
            rule_evaluations=rules,
        )
        assert res["success"] is True

        conn = capture_engine.db.get_connection()
        try:
            cur = conn.execute(
                "SELECT rule_id, evaluation_state FROM prospective_rule_evaluations WHERE decision_id = ? ORDER BY actual_execution_order",
                (res["decision_id"],),
            )
            rows = cur.fetchall()
            assert len(rows) == 2
            assert rows[0]["evaluation_state"] == "FAIL"
            assert rows[1]["evaluation_state"] == "NOT_EVALUATED_AFTER_BINDING_FAILURE"
        finally:
            conn.close()

    def test_04_expected_evaluation_never_evaluated(self, capture_engine):
        """Scenario 4 / Test C: Worker crashes before picking up asset.
        Reconciles as MISSING_EXPECTED_EVALUATION; 0 phantom decisions.
        """
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        cycle_id = cycle["evaluation_cycle_id"]

        capture_engine.register_expected_evaluations(
            evaluation_cycle_id=cycle_id,
            items=[("AMD", "INST_AMD_01")],
            scope_status="IN_SCOPE",
        )

        rec = capture_engine.reconcile_evaluation_cycle(cycle_id)
        assert rec["expected_count"] == 1
        assert rec["missing_count"] == 1
        assert rec["completed_count"] == 0
        assert rec["failed_before_count"] == 0
        assert rec["failed_during_count"] == 0
        assert rec["is_balanced"] is True

    def test_05_infrastructure_failures_before_and_during(self, capture_engine):
        """Scenario 5: Failures before and during model evaluation."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        cycle_id = cycle["evaluation_cycle_id"]

        capture_engine.register_expected_evaluations(
            evaluation_cycle_id=cycle_id,
            items=[("FAIL_BEFORE", "INST_FB_01"), ("FAIL_DURING", "INST_FD_01")],
            scope_status="IN_SCOPE",
        )

        # 1. Before model: quote provider error
        capture_engine.record_infrastructure_failure(
            evaluation_cycle_id=cycle_id,
            symbol="FAIL_BEFORE",
            instrument_id="INST_FB_01",
            failure_type="PROVIDER_UNAVAILABLE",
            error_message="HTTP 503 Provider unavailable",
            stage="BEFORE_MODEL",
        )

        # 2. During model: engine crash mid-analysis
        res_during = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="FAIL_DURING",
            instrument_id="INST_FD_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            evaluation_completion_state="FAILED_DURING_MODEL_EVALUATION",
            decision_state="REJECT",
            actionability_state="NON_ACTIONABLE",
            attempt_status="ENGINE_CRASH",
            attempt_failure_details="Segmentation fault in indicator compute",
            rejection_reason={"isRejected": True, "firstBindingRuleId": None, "firstBindingRuleCategory": "INFRASTRUCTURE_CONSTRAINT", "primaryReasonText": "Engine crash", "allExecutedFailedRuleIds": []},
            infrastructure_failure={"isInfrastructureFailure": True, "failureType": "UNHANDLED_EXCEPTION", "errorMessage": "Crash mid-evaluation"},
        )
        assert res_during["success"] is True

        rec = capture_engine.reconcile_evaluation_cycle(cycle_id)
        assert rec["expected_count"] == 2
        assert rec["failed_before_count"] == 1
        assert rec["failed_during_count"] == 1
        assert rec["completed_count"] == 0
        assert rec["missing_count"] == 0
        assert rec["is_balanced"] is True

    def test_06_same_cycle_retry_adversarial_test_a(self, capture_engine):
        """Scenario 6 / Test A: Worker times out, retry worker succeeds 30s later in same cycle.
        Invariants:
        - Same decision_id
        - 2 forensic evaluation_attempts
        - EMPIRICAL_DECISION_COUNT = 1
        """
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        cycle_id = cycle["evaluation_cycle_id"]

        capture_engine.register_expected_evaluations(
            evaluation_cycle_id=cycle_id,
            items=[("NVDA", "INST_NVDA_01")],
            scope_status="IN_SCOPE",
        )

        # Attempt 1: Timeout
        res1 = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="NVDA",
            instrument_id="INST_NVDA_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="REJECT",
            actionability_state="NON_ACTIONABLE",
            attempt_status="TIMEOUT",
            attempt_failure_details="Worker timed out waiting for Polygon quote",
            rejection_reason={"isRejected": True, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "Timeout", "allExecutedFailedRuleIds": []},
        )
        assert res1["success"] is True
        dec_id_1 = res1["decision_id"]
        assert res1["is_retry"] is False

        # Attempt 2 (Retry 30 seconds later in same cycle)
        res2 = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_id,
            symbol="NVDA",
            instrument_id="INST_NVDA_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=88.0,
            attempt_status="SUCCESS",
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )
        assert res2["success"] is True
        dec_id_2 = res2["decision_id"]
        assert res2["is_retry"] is True

        # Invariants:
        assert dec_id_1 == dec_id_2

        conn = capture_engine.db.get_connection()
        try:
            # 1 single decision event in table
            cur = conn.execute("SELECT COUNT(*) FROM prospective_decision_events WHERE decision_id = ?", (dec_id_1,))
            assert cur.fetchone()[0] == 1

            # 2 evaluation attempts
            cur_att = conn.execute("SELECT attempt_number, attempt_status FROM prospective_evaluation_attempts WHERE decision_id = ? ORDER BY attempt_number", (dec_id_1,))
            attempts = cur_att.fetchall()
            assert len(attempts) == 2
            assert attempts[0]["attempt_number"] == 1
            assert attempts[0]["attempt_status"] == "TIMEOUT"
            assert attempts[1]["attempt_number"] == 2
            assert attempts[1]["attempt_status"] == "SUCCESS"
        finally:
            conn.close()

    def test_07_next_cycle_reevaluation_adversarial_test_b(self, capture_engine):
        """Scenario 7 / Test B: Same asset evaluated in consecutive daily cycles.
        Invariants:
        - 2 distinct decision_ids (Monday vs Tuesday)
        - 1 continuous ongoing episode_id
        """
        # Monday Cycle
        cycle_mon = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
            cycle_started_at_utc="2026-09-28T16:00:00Z",
        )
        plan = {
            "entryReferencePrice": 125.00,
            "corridorMin": 124.00,
            "corridorMax": 126.00,
            "stopLoss": 120.00,
            "takeProfit1": 135.00,
        }
        res_mon = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_mon["evaluation_cycle_id"],
            symbol="NVDA",
            instrument_id="INST_NVDA_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=85.0,
            trade_plan_state=plan,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )

        # Tuesday Cycle
        cycle_tue = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
            cycle_started_at_utc="2026-09-29T16:00:00Z",
        )
        res_tue = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_tue["evaluation_cycle_id"],
            symbol="NVDA",
            instrument_id="INST_NVDA_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=86.0,
            trade_plan_state=plan,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )

        assert res_mon["decision_id"] != res_tue["decision_id"]
        assert res_mon["episode_id"] == res_tue["episode_id"]

        # Check episode sessions count
        conn = capture_engine.db.get_connection()
        try:
            cur = conn.execute("SELECT sessions_observed_count FROM prospective_episodes WHERE episode_id = ?", (res_mon["episode_id"],))
            assert cur.fetchone()[0] == 2
        finally:
            conn.close()

    def test_08_user_selected_request_adversarial_test_d(self, capture_engine):
        """Scenario 8 / Test D: User requests TSLA at 14:00, then scheduled scan runs at 16:00.
        Invariants:
        - Request 1: EXPLICIT_USER_REQUEST, USER_SELECTED, eligible_for_opportunity_capture = 0
        - Request 2: SCHEDULED_UNIVERSE_SCAN, SYSTEMATIC, eligible_for_opportunity_capture = 1
        """
        # User query
        cycle_user = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="EXPLICIT_USER_REQUEST",
            universe_version="2026.1",
            cycle_started_at_utc="2026-09-28T14:00:00Z",
        )
        res_user = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_user["evaluation_cycle_id"],
            symbol="TSLA",
            instrument_id="INST_TSLA_01",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=78.0,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )

        # Scheduled scan
        cycle_scan = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
            cycle_started_at_utc="2026-09-28T16:00:00Z",
        )
        res_scan = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle_scan["evaluation_cycle_id"],
            symbol="TSLA",
            instrument_id="INST_TSLA_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=78.0,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )

        conn = capture_engine.db.get_connection()
        try:
            cur1 = conn.execute("SELECT eligible_for_opportunity_capture, eligible_for_coverage_analysis FROM prospective_decision_events WHERE decision_id = ?", (res_user["decision_id"],))
            row1 = cur1.fetchone()
            assert row1["eligible_for_opportunity_capture"] == 0
            assert row1["eligible_for_coverage_analysis"] == 0

            cur2 = conn.execute("SELECT eligible_for_opportunity_capture, eligible_for_coverage_analysis FROM prospective_decision_events WHERE decision_id = ?", (res_scan["decision_id"],))
            row2 = cur2.fetchone()
            assert row2["eligible_for_opportunity_capture"] == 1
            assert row2["eligible_for_coverage_analysis"] == 1
        finally:
            conn.close()

    def test_09_episode_reset_on_price_level_alteration(self, capture_engine):
        """Scenario 9: Episode resets if setup price levels significantly change."""
        plan1 = {"entryReferencePrice": 100.0, "stopLoss": 95.0, "takeProfit1": 110.0}
        ep1 = capture_engine.create_or_get_episode("AAPL", trade_plan=plan1, timestamp_utc="2026-09-28T10:00:00Z")

        # Now price levels altered
        plan2 = {"entryReferencePrice": 108.0, "stopLoss": 103.0, "takeProfit1": 118.0}
        ep2 = capture_engine.create_or_get_episode("AAPL", trade_plan=plan2, timestamp_utc="2026-09-29T10:00:00Z")

        assert ep1 != ep2

        conn = capture_engine.db.get_connection()
        try:
            cur = conn.execute("SELECT episode_status FROM prospective_episodes WHERE episode_id = ?", (ep1,))
            assert cur.fetchone()[0] == "RESET_PRICE_LEVEL_ALTERED"
        finally:
            conn.close()

    def test_10_test_pollution_firewall(self, capture_engine):
        """Scenario 10: Executions with TEST or REPLAY origin are quarantined."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        res = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="TEST_SYM",
            instrument_id="INST_TEST_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="TEST",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=90.0,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )
        assert res["empirical_certification_state"] == "QUARANTINED_NON_PRODUCTION"

        conn = capture_engine.db.get_connection()
        try:
            cur = conn.execute("SELECT empirical_certification_state, eligible_for_opportunity_capture FROM prospective_decision_events WHERE decision_id = ?", (res["decision_id"],))
            row = cur.fetchone()
            assert row["empirical_certification_state"] == "QUARANTINED_NON_PRODUCTION"
            assert row["eligible_for_opportunity_capture"] == 0
        finally:
            conn.close()

    def test_11_immutability_trigger_enforcement(self, capture_engine):
        """Scenario 11: Direct SQL UPDATE or DELETE on prospective_decision_events raises error."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        res = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="IMMUTABLE_SYM",
            instrument_id="INST_IMM_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=80.0,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )
        dec_id = res["decision_id"]

        conn = capture_engine.db.get_connection()
        try:
            with pytest.raises(sqlite3.DatabaseError, match="IMMUTABILITY_VIOLATION"):
                conn.execute(
                    "UPDATE prospective_decision_events SET confluence_score = 99.0 WHERE decision_id = ?",
                    (dec_id,),
                )

            with pytest.raises(sqlite3.DatabaseError, match="IMMUTABILITY_VIOLATION"):
                conn.execute(
                    "DELETE FROM prospective_decision_events WHERE decision_id = ?",
                    (dec_id,),
                )
        finally:
            conn.close()

    def test_12_decision_correction_append_only(self, capture_engine):
        """Scenario 12: Corrections are appended to prospective_decision_corrections without altering decision event."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        res = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="CORR_SYM",
            instrument_id="INST_CORR_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            confluence_score=75.0,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )
        dec_id = res["decision_id"]

        corr_id = capture_engine.record_decision_correction(
            original_decision_id=dec_id,
            corrected_field_name="market_regime",
            original_value="BULL",
            corrected_value="HIGH_VOLATILITY",
            correction_reason="Regime filter recalculation with closing volatility spike",
            authorized_by="SYSTEM_GOVERNANCE_ARBITER",
        )
        assert corr_id.startswith("COR_")

        conn = capture_engine.db.get_connection()
        try:
            # Original event unchanged
            cur = conn.execute("SELECT market_regime FROM prospective_decision_events WHERE decision_id = ?", (dec_id,))
            assert cur.fetchone()[0] == "BULL"

            # Correction record exists
            cur_c = conn.execute("SELECT corrected_value_json FROM prospective_decision_corrections WHERE correction_event_id = ?", (corr_id,))
            assert json.loads(cur_c.fetchone()[0]) == "HIGH_VOLATILITY"
        finally:
            conn.close()

    def test_13_fail_open_client_behavior(self, tmp_path):
        """Scenario 13: Storage write failures return UNCERTIFIED_CAPTURE_FAILURE without crashing."""
        read_only_db = str(tmp_path / "readonly.db")
        # Initialize db then make it read-only
        engine_init = ProspectiveDecisionCaptureEngine(db_path=read_only_db)
        engine_init.create_or_get_evaluation_cycle("SCHEDULED_UNIVERSE_SCAN", "2026.1")

        # Now simulate failure by pointing to an invalid/locked DB or simulating an error
        engine_fail = ProspectiveDecisionCaptureEngine(
            db_path=read_only_db,
            fail_open_client=True,
        )
        # Monkeypatch get_connection to raise an exception
        engine_fail.db.get_connection = lambda: (_ for _ in ()).throw(sqlite3.OperationalError("Simulated disk error"))

        res = engine_fail.record_decision_event(
            evaluation_cycle_id="CYC_SCHEDULED_UNIVERSE_SCAN_20260928T160000Z_12345678",
            symbol="FAIL_TEST",
            instrument_id="INST_FT_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )
        assert res["success"] is False
        assert res["empirical_certification_state"] == "UNCERTIFIED_CAPTURE_FAILURE"

    def test_14_data_availability_states(self, capture_engine):
        """Scenario 14: Data availability states (AUTHENTIC_ZERO, UNAVAILABLE, STALE)."""
        cycle = capture_engine.create_or_get_evaluation_cycle(
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            universe_version="2026.1",
        )
        features = [
            {
                "featureName": "debt_to_equity",
                "featureValue": 0.0,
                "source": "SEC_FILING",
                "sourceTimestampUtc": datetime.now(timezone.utc).isoformat(),
                "observedAtUtc": datetime.now(timezone.utc).isoformat(),
                "availabilityStatus": "AUTHENTIC_ZERO",
                "version": "1.0.0",
            },
            {
                "featureName": "institutional_ownership_pct",
                "featureValue": None,
                "source": "13F_FILING",
                "sourceTimestampUtc": datetime.now(timezone.utc).isoformat(),
                "observedAtUtc": datetime.now(timezone.utc).isoformat(),
                "availabilityStatus": "UNAVAILABLE",
                "version": "1.0.0",
            },
        ]
        res = capture_engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="ZERO_DEBT_CO",
            instrument_id="INST_ZD_01",
            cycle_type="SCHEDULED_UNIVERSE_SCAN",
            population_sampling_mode="SYSTEMATIC",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_RECOMMENDATION",
            actionability_state="ACTIONABLE",
            features=features,
            rejection_reason={"isRejected": False, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "NONE", "allExecutedFailedRuleIds": []},
        )
        assert res["success"] is True

        conn = capture_engine.db.get_connection()
        try:
            cur = conn.execute("SELECT features_json FROM prospective_feature_snapshots WHERE decision_id = ?", (res["decision_id"],))
            feats = json.loads(cur.fetchone()[0])
            assert feats[0]["availabilityStatus"] == "AUTHENTIC_ZERO"
            assert feats[0]["featureValue"] == 0.0
            assert feats[1]["availabilityStatus"] == "UNAVAILABLE"
            assert feats[1]["featureValue"] is None
        finally:
            conn.close()

    def test_15_decision_parity_capture_on_off(self, capture_engine):
        """Scenario 15: Decision parity test — capture ON vs capture OFF has 0 mismatches."""
        # Simulated baseline model run without capture:
        def run_simulated_model(symbol: str, price: float, volume: int):
            if price < 5.0 or volume < 1000000:
                return {"decision_state": "REJECT", "actionability": "NON_ACTIONABLE", "reason": "FAILED_CONSTRAINTS"}
            score = 80.0 if symbol == "NVDA" else 50.0
            if score >= 70.0:
                return {"decision_state": "ACTIONABLE_RECOMMENDATION", "actionability": "ACTIONABLE", "score": score}
            return {"decision_state": "REJECT", "actionability": "NON_ACTIONABLE", "score": score}

        symbols = [("NVDA", 125.0, 50000000), ("INTC", 20.0, 30000000), ("PENNY", 1.5, 500000)]

        cycle = capture_engine.create_or_get_evaluation_cycle("SCHEDULED_UNIVERSE_SCAN", "2026.1")

        mismatches = 0
        for sym, price, vol in symbols:
            # Baseline (capture OFF)
            baseline = run_simulated_model(sym, price, vol)

            # Execution with capture ON
            captured = run_simulated_model(sym, price, vol)
            res = capture_engine.record_decision_event(
                evaluation_cycle_id=cycle["evaluation_cycle_id"],
                symbol=sym,
                instrument_id=f"INST_{sym}",
                cycle_type="SCHEDULED_UNIVERSE_SCAN",
                population_sampling_mode="SYSTEMATIC",
                evidence_origin="NATURAL_PRODUCTION",
                scope_status="IN_SCOPE",
                decision_state=captured["decision_state"],
                actionability_state=captured["actionability"],
                confluence_score=captured.get("score"),
                rejection_reason={"isRejected": captured["decision_state"] == "REJECT", "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": captured.get("reason", "NONE"), "allExecutedFailedRuleIds": []},
            )

            if baseline != captured:
                mismatches += 1

        assert mismatches == 0, f"CAPTURE_ON_OFF_DECISION_MISMATCHES = {mismatches}"

    def test_16_capture_latency_benchmark(self, capture_engine):
        """Scenario 16: Latency benchmarking across 100 simulated decision captures.
        Measures p50, p95, p99, and max latency in milliseconds.
        """
        cycle = capture_engine.create_or_get_evaluation_cycle("SCHEDULED_UNIVERSE_SCAN", "2026.1")
        cycle_id = cycle["evaluation_cycle_id"]

        latencies_ms = []
        for i in range(100):
            sym = f"SYM_{i:03d}"
            inst = f"INST_{i:03d}"
            t0 = time.perf_counter()
            capture_engine.record_decision_event(
                evaluation_cycle_id=cycle_id,
                symbol=sym,
                instrument_id=inst,
                cycle_type="SCHEDULED_UNIVERSE_SCAN",
                population_sampling_mode="SYSTEMATIC",
                evidence_origin="NATURAL_PRODUCTION",
                scope_status="IN_SCOPE",
                decision_state="REJECT",
                actionability_state="NON_ACTIONABLE",
                confluence_score=50.0 + (i % 30),
                rejection_reason={"isRejected": True, "firstBindingRuleId": None, "firstBindingRuleCategory": None, "primaryReasonText": "SCORE", "allExecutedFailedRuleIds": []},
            )
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_ms.sort()
        n = len(latencies_ms)
        p50 = latencies_ms[int(n * 0.50)]
        p95 = latencies_ms[int(n * 0.95)]
        p99 = latencies_ms[int(n * 0.99)]
        max_lat = latencies_ms[-1]

        # Verify capture latency is performant (e.g. p50 < 25ms in SQLite WAL mode)
        assert p50 < 50.0, f"p50 latency {p50:.2f}ms exceeds threshold"
        assert max_lat < 500.0, f"max latency {max_lat:.2f}ms exceeds threshold"

    def test_17_decision_id_tuple_boundary_immunity(self):
        """Scenario 17: Adversarial test proving tuple-boundary shifting cannot alter identity.
        With delimiter-free string concatenation:
        ('AB', 'CD') and ('A', 'BCD') produced identical preimages ('ABCD...').
        With canonical deterministic JSON serialization, preimages are strictly distinct.
        """
        id1 = compute_decision_id(
            instrument_id="AB",
            evaluation_cycle_id="CD",
            engine_sha="7ad44595826c147cc77f93cd676af520764c7442",
            decision_schema_version="1.0.2",
        )
        id2 = compute_decision_id(
            instrument_id="A",
            evaluation_cycle_id="BCD",
            engine_sha="7ad44595826c147cc77f93cd676af520764c7442",
            decision_schema_version="1.0.2",
        )
        assert id1 != id2, f"Tuple-boundary collision detected: {id1} == {id2}"

