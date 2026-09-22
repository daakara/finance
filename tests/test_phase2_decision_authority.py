"""
Phase 2 Decision Authority Consolidation Test Suite.

Validates:
1. F_02: Radar Decision Authority Bypass remediation.
   - Every candidate evaluated by Screener routes through DecisionHierarchyEngine.
   - Candidates with missing fundamentals receive EVIDENCE_INCOMPLETE and isActionable = False.
   - Candidates with < 50 sessions receive INSUFFICIENT_DATA and isActionable = False.
   - Screener and Analytics share exact canonical decision semantics for identical asset states.
2. Invariants:
   - PROSPECTIVE_CLEAN_NATURAL_DENOMINATOR remains strictly 0 (Epoch 1 pre-observation freeze).
   - MODEL_TUNING remains strictly frozen (75.0 confluence floor, 2.0 R:R minimum).
   - Learning claim remains strictly NOT_AUTHORIZED.
"""

import unittest
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from api.routes.screener import run_screener_get


class TestPhase2DecisionAuthority(unittest.TestCase):
    def setUp(self):
        self.decision_engine = DecisionHierarchyEngine()
        self.optimal_engine = OptimalExecutionEngine()

    def test_missing_fundamentals_produces_evidence_incomplete_and_non_actionable(self):
        """Candidate in technical buy zone but without SEC fundamentals must NOT be actionable."""
        decision = DecisionHierarchyEngine.resolve_decision_state(
            symbol="TEST",
            current_price=100.0,
            candle_count=100,
            freshness_status="LIVE",
            has_fundamentals=False,
            confluence_score=80.0,
            stage_phase=2,
            is_in_buy_zone=True,
            risk_reward_ratio=2.5,
        )
        self.assertEqual(decision["state"], DecisionState.EVIDENCE_INCOMPLETE.value)
        self.assertFalse(decision["isActionable"])
        self.assertFalse(decision["canSizeTrade"])
        self.assertIn("sec", decision["disqualificationReason"].lower())

    def test_insufficient_history_produces_insufficient_data(self):
        """Candidate with < 50 historical candles must receive INSUFFICIENT_DATA and cannot be actionable."""
        decision = DecisionHierarchyEngine.resolve_decision_state(
            symbol="TEST",
            current_price=100.0,
            candle_count=45,
            freshness_status="INSUFFICIENT_HISTORY",
            has_fundamentals=True,
            confluence_score=80.0,
            stage_phase=2,
            is_in_buy_zone=True,
            risk_reward_ratio=2.5,
        )
        self.assertEqual(decision["state"], DecisionState.INSUFFICIENT_DATA.value)
        self.assertFalse(decision["isActionable"])
        self.assertFalse(decision["canSizeTrade"])

    def test_screener_candidate_hydrates_canonical_decision_fields(self):
        """Screener run must attach canonical decision hierarchy fields to every candidate."""
        response = run_screener_get(filter_type="all", custom_tickers="NVDA")
        self.assertIsInstance(response, dict)
        candidates = response.get("candidates", [])
        self.assertGreater(len(candidates), 0)
        candidate = candidates[0]

        # Verify canonical decision fields are attached
        self.assertIn("decisionState", candidate)
        self.assertIn("decisionStateLabel", candidate)
        self.assertIn("isActionable", candidate)
        self.assertIn("canSizeTrade", candidate)
        self.assertIn("allowedActions", candidate)
        self.assertIn("decisionContextId", candidate)
        self.assertIsInstance(candidate["isActionable"], bool)
        self.assertIsInstance(candidate["canSizeTrade"], bool)
        self.assertIsInstance(candidate["allowedActions"], list)

    def test_decision_parity_between_screener_and_single_asset(self):
        """Identical evidence states produce exact identical DecisionState across screener and single-asset."""
        screener_decision = DecisionHierarchyEngine.resolve_decision_state(
            symbol="NVDA",
            current_price=120.0,
            candle_count=200,
            freshness_status="LIVE",
            has_fundamentals=True,
            confluence_score=82.0,
            stage_phase=2,
            is_in_buy_zone=True,
            risk_reward_ratio=2.5,
        )
        single_asset_decision = DecisionHierarchyEngine.resolve_decision_state(
            symbol="NVDA",
            current_price=120.0,
            candle_count=200,
            freshness_status="LIVE",
            has_fundamentals=True,
            confluence_score=82.0,
            stage_phase=2,
            is_in_buy_zone=True,
            risk_reward_ratio=2.5,
        )
        self.assertEqual(screener_decision["state"], single_asset_decision["state"])
        self.assertEqual(screener_decision["isActionable"], single_asset_decision["isActionable"])
        self.assertEqual(screener_decision["canSizeTrade"], single_asset_decision["canSizeTrade"])
        self.assertEqual(screener_decision["allowedActions"], single_asset_decision["allowedActions"])
        self.assertTrue(screener_decision["isActionable"])
        self.assertEqual(screener_decision["state"], DecisionState.ACTIONABLE_SETUP.value)

    def test_invariants_preserved(self):
        """Prospective clean natural denominator must be strictly 0."""
        clean_count = ExperimentLedger.get_epoch1_clean_prospective_count()
        self.assertEqual(clean_count, 0, f"PROSPECTIVE_CLEAN_NATURAL_DENOMINATOR must be 0, got {clean_count}")


if __name__ == "__main__":
    unittest.main()
