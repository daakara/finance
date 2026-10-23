#!/usr/bin/env python3
"""
ARX Terminal — Phase 2 Calibration Scoring Utility Regression Test Suite
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
Verifies 15 deterministic edge-cases and invariants for tests/scoring_utility.py
"""

import sys
import os
import json
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.scoring_utility import (
    calculate_cohens_kappa,
    compute_contingency,
    validate_ratings_payload,
    validate_gold_key_payload,
    evaluate_observer_vs_gold,
    evaluate_pairwise_observers,
    ALL_CODES
)

class TestScoringUtilityRegression(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def _write_json(self, filename: str, data: dict) -> str:
        path = os.path.join(self.temp_dir.name, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    # 1. Perfect agreement
    def test_01_perfect_agreement(self):
        po, pe, kappa, is_def = calculate_cohens_kappa(a=5, b=0, c=0, d=5)
        self.assertTrue(is_def)
        self.assertEqual(po, 1.0)
        self.assertEqual(kappa, 1.0)

    # 2. Systematic disagreement
    def test_02_systematic_disagreement(self):
        po, pe, kappa, is_def = calculate_cohens_kappa(a=0, b=5, c=5, d=0)
        self.assertTrue(is_def)
        self.assertEqual(po, 0.0)
        self.assertLess(kappa, 0.0)

    # 3. Rare positive categories
    def test_03_rare_positive_categories(self):
        # 1 TP out of 100 trials, 99 TN
        po, pe, kappa, is_def = calculate_cohens_kappa(a=1, b=0, c=0, d=99)
        self.assertTrue(is_def)
        self.assertEqual(po, 1.0)
        self.assertEqual(kappa, 1.0)

    # 4. All-negative category
    def test_04_all_negative_category(self):
        # 0 TP, 0 FP, 0 FN, 20 TN -> Pe = 1.0
        po, pe, kappa, is_def = calculate_cohens_kappa(a=0, b=0, c=0, d=20)
        self.assertFalse(is_def)
        self.assertIsNone(kappa)
        self.assertEqual(po, 1.0)
        self.assertEqual(pe, 1.0)

    # 5. Undefined kappa handling (never fabricated as 1.0)
    def test_05_undefined_kappa_handling(self):
        po, pe, kappa, is_def = calculate_cohens_kappa(a=0, b=0, c=0, d=10)
        self.assertIsNone(kappa, "Must return None when Pe >= 1.0, NOT 1.0")
        self.assertFalse(is_def)

    # 6. Multilabel overlap
    def test_06_multilabel_overlap(self):
        r1 = [{"trial_id": "T1", "assigned_cm_codes": ["CM-07"], "assigned_sm_codes": ["SM-06"]}]
        r2 = [{"trial_id": "T1", "assigned_cm_codes": ["CM-07"], "assigned_sm_codes": ["SM-06"]}]
        s1 = [set(r1[0]["assigned_cm_codes"] + r1[0]["assigned_sm_codes"])]
        s2 = [set(r2[0]["assigned_cm_codes"] + r2[0]["assigned_sm_codes"])]
        
        a7, b7, c7, d7 = compute_contingency(s1, s2, "CM-07")
        self.assertEqual((a7, b7, c7, d7), (1, 0, 0, 0))
        a6, b6, c6, d6 = compute_contingency(s1, s2, "SM-06")
        self.assertEqual((a6, b6, c6, d6), (1, 0, 0, 0))

    # 7. Missing responses
    def test_07_missing_responses(self):
        gold = {
            "gold_key_id": "G1",
            "items": {
                "T1": {"authoritative_cm_codes": []},
                "T2": {"authoritative_cm_codes": []}
            }
        }
        obs = {
            "observer_id": "O1",
            "ratings": [{"trial_id": "T1", "assigned_cm_codes": []}]
        }
        gp = self._write_json("gold7.json", gold)
        op = self._write_json("obs7.json", obs)
        with self.assertRaises(ValueError) as ctx:
            evaluate_observer_vs_gold(gp, op)
        self.assertIn("missing required trials", str(ctx.exception).lower())

    # 8. Duplicate trial IDs
    def test_08_duplicate_trial_ids(self):
        obs = {
            "observer_id": "O1",
            "ratings": [
                {"trial_id": "T1", "assigned_cm_codes": []},
                {"trial_id": "T1", "assigned_cm_codes": []}
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_ratings_payload(obs, "Test")
        self.assertIn("duplicate trial_id detected", str(ctx.exception).lower())

    # 9. Missing gold-key IDs
    def test_09_missing_gold_key_ids(self):
        gold = {"items": {"T1": {}}}
        with self.assertRaises(ValueError) as ctx:
            validate_gold_key_payload(gold)
        self.assertIn("missing or invalid 'gold_key_id'", str(ctx.exception).lower())

    # 10. Incorrect taxonomy codes
    def test_10_incorrect_taxonomy_codes(self):
        obs = {
            "observer_id": "O1",
            "ratings": [{"trial_id": "T1", "assigned_cm_codes": ["INVALID_CODE_99"]}]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_ratings_payload(obs, "Test")
        self.assertIn("invalid taxonomy code", str(ctx.exception).lower())

    # 11. False positives
    def test_11_false_positives(self):
        gold = {"gold_key_id": "G1", "items": {"T1": {"authoritative_cm_codes": []}}}
        obs = {"observer_id": "O1", "ratings": [{"trial_id": "T1", "assigned_cm_codes": ["CM-07"]}]}
        gp = self._write_json("gold11.json", gold)
        op = self._write_json("obs11.json", obs)
        res = evaluate_observer_vs_gold(gp, op)
        self.assertEqual(res["per_code_metrics"]["CM-07"]["fp"], 1)
        self.assertEqual(len(res["disputes"]), 1)
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_FAILED")

    # 12. False negatives
    def test_12_false_negatives(self):
        gold = {"gold_key_id": "G1", "items": {"T1": {"authoritative_cm_codes": ["CM-07"]}}}
        obs = {"observer_id": "O1", "ratings": [{"trial_id": "T1", "assigned_cm_codes": []}]}
        gp = self._write_json("gold12.json", gold)
        op = self._write_json("obs12.json", obs)
        res = evaluate_observer_vs_gold(gp, op)
        self.assertEqual(res["per_code_metrics"]["CM-07"]["fn"], 1)
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_FAILED")

    # 13. Category prevalence imbalance
    def test_13_prevalence_imbalance(self):
        # 10 trials: 1 positive, 9 negative. Observer guesses 0 on all 10 trials.
        # Po = 9/10 = 0.90, but chance agreement Pe = 0.90.
        # Kappa should be 0.0, despite 90% raw accuracy!
        po, pe, kappa, is_def = calculate_cohens_kappa(a=0, b=0, c=1, d=9)
        self.assertTrue(is_def)
        self.assertEqual(po, 0.9)
        self.assertEqual(pe, 0.9)
        self.assertEqual(kappa, 0.0)

    # 14. Human observer versus gold-key comparisons (Measurement A)
    def test_14_observer_vs_gold_measurement_a(self):
        gold_path = os.path.join(os.path.dirname(__file__), "fixtures", "calibration", "v2", "gold_key_alpha_v2.json")
        obs = {
            "observer_id": "SYN-TEST-PERFECT",
            "ratings": [
                {"trial_id": f"ALPHA-{i:02d}", "assigned_cm_codes": [], "assigned_sm_codes": [], "interpretation_state": "INTERPRETATION_CORRECT", "action_state": "ACTION_APPROPRIATE"}
                for i in range(1, 11)
            ]
        }
        # Inject exact matching codes for ALPHA-02, 03, 04, 07
        obs["ratings"][1]["assigned_sm_codes"] = ["SM-06"]
        obs["ratings"][1]["action_state"] = "ACTION_INAPPROPRIATE"
        obs["ratings"][2]["assigned_sm_codes"] = ["SM-01", "SM-02"]
        obs["ratings"][2]["interpretation_state"] = "INTERPRETATION_INCORRECT"
        obs["ratings"][2]["action_state"] = "ACTION_INAPPROPRIATE"
        obs["ratings"][3]["assigned_sm_codes"] = ["SM-03"]
        obs["ratings"][3]["interpretation_state"] = "INTERPRETATION_INCORRECT"
        obs["ratings"][3]["action_state"] = "ACTION_INAPPROPRIATE"
        obs["ratings"][6]["assigned_cm_codes"] = ["CM-07"]
        obs["ratings"][6]["assigned_sm_codes"] = ["SM-06"]
        obs["ratings"][6]["interpretation_state"] = "INTERPRETATION_INCORRECT"
        obs["ratings"][6]["action_state"] = "ACTION_INAPPROPRIATE"
        
        op = self._write_json("obs_perfect.json", obs)
        res = evaluate_observer_vs_gold(gold_path, op)
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_PASSED")
        self.assertEqual(res["evaluation_summary"]["exact_match_rate"], 1.0)
        self.assertEqual(res["evaluation_summary"]["micro_pooled_kappa"], 1.0)
        self.assertEqual(res["evaluation_summary"]["critical_decision_false_negatives"], 0)

    # 15. Human observer pairwise comparisons (Measurement B)
    def test_15_observer_pairwise_measurement_b(self):
        obs1 = {
            "observer_id": "OBS-ALPHA-1",
            "ratings": [
                {"trial_id": "T1", "assigned_cm_codes": ["CM-07"], "assigned_sm_codes": []},
                {"trial_id": "T2", "assigned_cm_codes": [], "assigned_sm_codes": []}
            ]
        }
        obs2 = {
            "observer_id": "OBS-ALPHA-2",
            "ratings": [
                {"trial_id": "T1", "assigned_cm_codes": ["CM-07"], "assigned_sm_codes": []},
                {"trial_id": "T2", "assigned_cm_codes": [], "assigned_sm_codes": []}
            ]
        }
        p1 = self._write_json("obs1.json", obs1)
        p2 = self._write_json("obs2.json", obs2)
        res = evaluate_pairwise_observers(p1, p2)
        self.assertEqual(res["pairwise_summary"]["pairwise_verdict"], "PAIRWISE_RELIABILITY_SUFFICIENT")
        self.assertEqual(res["pairwise_summary"]["exact_match_rate"], 1.0)
        self.assertEqual(res["pairwise_summary"]["micro_pooled_kappa"], 1.0)

    # 16. Supplemental Gamma corpus execution
    def test_16_supplemental_gamma_corpus_execution(self):
        res = evaluate_observer_vs_gold(
            "tests/fixtures/calibration/v2/gold_key_gamma_v1.json",
            "tests/fixtures/calibration/v2/observer_gamma_v1_calibrated.json"
        )
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_PASSED")
        self.assertEqual(res["evaluation_summary"]["critical_decision_false_negatives"], 0)
        # Verify previously missing categories are tested positive
        for code in ["CM-01", "CM-02", "CM-04", "CM-05", "CM-06", "CM-07"]:
            self.assertIn(code, res["evaluation_summary"]["categories_tested_positive"])

    # 17. Consolidated 30-trial full coverage (Zero untested categories)
    def test_17_consolidated_30_trial_complete_coverage(self):
        res = evaluate_observer_vs_gold(
            "tests/fixtures/calibration/v2/gold_key_consolidated_30.json",
            "tests/fixtures/calibration/v2/observer_consolidated_30_calibrated.json"
        )
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_PASSED")
        self.assertEqual(res["evaluation_summary"]["total_trials"], 30)
        self.assertEqual(len(res["evaluation_summary"]["categories_untested_in_corpus"]), 0)
        self.assertEqual(len(res["evaluation_summary"]["categories_tested_positive"]), 15)

if __name__ == "__main__":
    unittest.main()
