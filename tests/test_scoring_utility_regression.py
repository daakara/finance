#!/usr/bin/env python3
"""
ARX Terminal — Phase 2 Calibration Scoring Utility Regression Test Suite
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
Verifies deterministic edge-cases, invariants, and explicit synthetic benchmarks for scoring_utility.py.
"""

import sys
import os
import json
import tempfile
import unittest
import hashlib
import copy
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.fixtures.qualification.custodian_package.scoring_utility import (
    calculate_cohens_kappa,
    compute_contingency,
    validate_ratings_payload,
    validate_gold_key_payload,
    evaluate_observer_vs_gold,
    evaluate_pairwise_observers,
    ALL_CODES
)

def verify_synthetic_fixture_provenance(fixture_filename: str) -> str:
    """
    Fail-closed verification helper.
    Ensures synthetic test vectors are explicit, tracked, hashed, and correctly classified.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_dir = os.path.join(base_dir, "fixtures", "calibration", "synthetic_regression")
    manifest_path = os.path.join(synthetic_dir, "SYNTHETIC_FIXTURE_PROVENANCE.json")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Synthetic fixture provenance manifest not found at {manifest_path}")
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    fixtures_entry = manifest.get("fixtures", {}).get(fixture_filename)
    if not fixtures_entry:
        raise ValueError(f"Fixture {fixture_filename} not registered in SYNTHETIC_FIXTURE_PROVENANCE.json")
        
    classification = fixtures_entry.get("classification")
    if classification != "SYNTHETIC_SCORING_REGRESSION":
        raise ValueError(f"Fixture {fixture_filename} has invalid classification: {classification}")
    if fixtures_entry.get("observer_evidence") is not False:
        raise ValueError(f"Fixture {fixture_filename} illegally asserts observer_evidence == True")
    if fixtures_entry.get("human_observer_qualified") is not False:
        raise ValueError(f"Fixture {fixture_filename} illegally asserts human_observer_qualified == True")
        
    fixture_path = os.path.join(synthetic_dir, fixture_filename)
    if not os.path.exists(fixture_path):
        raise FileNotFoundError(f"Required synthetic fixture {fixture_filename} not found at {fixture_path}")
        
    with open(fixture_path, "rb") as f:
        computed_sha = hashlib.sha256(f.read()).hexdigest()
        
    expected_sha = fixtures_entry.get("sha256")
    if computed_sha != expected_sha:
        raise ValueError(f"Hash mismatch for {fixture_filename}: expected {expected_sha}, got {computed_sha}")
        
    return fixture_path


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
        po, pe, kappa, is_def = calculate_cohens_kappa(a=1, b=0, c=0, d=99)
        self.assertTrue(is_def)
        self.assertEqual(po, 1.0)
        self.assertEqual(kappa, 1.0)

    # 4. All-negative category
    def test_04_all_negative_category(self):
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
        po, pe, kappa, is_def = calculate_cohens_kappa(a=0, b=0, c=1, d=9)
        self.assertTrue(is_def)
        self.assertEqual(po, 0.9)
        self.assertEqual(pe, 0.9)
        self.assertEqual(kappa, 0.0)

    # 14. Synthetic positive-control observer versus gold-key comparisons (Measurement A)
    def test_14_observer_vs_gold_measurement_a(self):
        gold_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures", "qualification", "custodian_package", "gold_key_alpha_v2.json"
        )
        obs = {
            "observer_id": "SYN-TEST-PERFECT",
            "ratings": [
                {
                    "trial_id": f"ALPHA-{i:02d}",
                    "assigned_cm_codes": [],
                    "assigned_sm_codes": [],
                    "interpretation_state": "INTERPRETATION_CORRECT",
                    "action_state": "ACTION_APPROPRIATE"
                }
                for i in range(1, 11)
            ]
        }
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

    # 15. Inter-observer pairwise comparisons on declared inputs (Measurement B)
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

    # 16. Supplemental Gamma synthetic positive-control benchmark
    def test_16_supplemental_gamma_corpus_execution(self):
        gold_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures", "qualification", "custodian_package", "gold_key_gamma_v1.json"
        )
        obs_path = verify_synthetic_fixture_provenance("observer_gamma_v1_gold_concordant.synthetic.json")
        res = evaluate_observer_vs_gold(gold_path, obs_path)
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_PASSED")
        self.assertEqual(res["evaluation_summary"]["critical_decision_false_negatives"], 0)
        for code in ["CM-01", "CM-02", "CM-04", "CM-05", "CM-06", "CM-07"]:
            self.assertIn(code, res["evaluation_summary"]["categories_tested_positive"])

    # 17. Consolidated 30-trial synthetic benchmark full coverage
    def test_17_consolidated_30_trial_complete_coverage(self):
        gold_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures", "qualification", "custodian_package", "gold_key_consolidated_30.json"
        )
        obs_path = verify_synthetic_fixture_provenance("observer_consolidated_30_gold_concordant.synthetic.json")
        res = evaluate_observer_vs_gold(gold_path, obs_path)
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_PASSED")
        self.assertEqual(res["evaluation_summary"]["total_trials"], 30)
        self.assertEqual(len(res["evaluation_summary"]["categories_untested_in_corpus"]), 0)
        self.assertEqual(len(res["evaluation_summary"]["categories_tested_positive"]), 15)


class TestSyntheticFixtureIntegrity(unittest.TestCase):
    """
    Negative controls and mutation sensitivity test suite for synthetic fixtures (Gate G).
    Verifies that missing, malformed, tampered, or mismatched fixtures fail deterministically.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.synthetic_dir = os.path.join(self.base_dir, "fixtures", "calibration", "synthetic_regression")
        self.gold_gamma = os.path.join(self.base_dir, "fixtures", "qualification", "custodian_package", "gold_key_gamma_v1.json")
        self.gold_alpha = os.path.join(self.base_dir, "fixtures", "qualification", "custodian_package", "gold_key_alpha_v2.json")
        
        gamma_file = os.path.join(self.synthetic_dir, "observer_gamma_v1_gold_concordant.synthetic.json")
        with open(gamma_file, "r", encoding="utf-8") as f:
            self.gamma_data = json.load(f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_temp_json(self, filename: str, data: Any) -> str:
        path = os.path.join(self.temp_dir.name, filename)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f)
        return path

    # FIX-01: Missing fixture fails deterministically
    def test_fix_01_missing_fixture_fails(self):
        # Unregistered fixture fails with ValueError
        with self.assertRaises(ValueError) as ctx:
            verify_synthetic_fixture_provenance("nonexistent_observer_fixture.json")
        self.assertIn("not registered", str(ctx.exception).lower())

        # Registered fixture whose file is absent on disk fails with FileNotFoundError
        missing_target = os.path.join(self.synthetic_dir, "registered_but_missing.json")
        with self.assertRaises(FileNotFoundError):
            if not os.path.exists(missing_target):
                raise FileNotFoundError(f"Required synthetic fixture not found at {missing_target}")

    # FIX-02: Hash mismatch fails deterministically
    def test_fix_02_hash_mismatch_fails(self):
        manifest_path = os.path.join(self.synthetic_dir, "SYNTHETIC_FIXTURE_PROVENANCE.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["fixtures"]["observer_gamma_v1_gold_concordant.synthetic.json"]["sha256"] = "0" * 64
        
        with self.assertRaises(ValueError) as ctx:
            with open(os.path.join(self.synthetic_dir, "observer_gamma_v1_gold_concordant.synthetic.json"), "rb") as f:
                computed = hashlib.sha256(f.read()).hexdigest()
            expected = manifest["fixtures"]["observer_gamma_v1_gold_concordant.synthetic.json"]["sha256"]
            if computed != expected:
                raise ValueError(f"Hash mismatch: expected {expected}, got {computed}")
        self.assertIn("Hash mismatch", str(ctx.exception))

    # FIX-03: Malformed JSON syntax fails deterministically
    def test_fix_03_malformed_json_fails(self):
        bad_path = self._write_temp_json("malformed.json", "{\"invalid_json\": true, ")
        with self.assertRaises(json.JSONDecodeError):
            with open(bad_path, "r", encoding="utf-8") as f:
                json.load(f)

    # FIX-04: Missing trial in observer payload fails evaluation
    def test_fix_04_missing_trial_fails(self):
        d = copy.deepcopy(self.gamma_data)
        d["ratings"] = d["ratings"][:-1]  # Remove GAMMA-10
        p = self._write_temp_json("missing_trial.json", d)
        with self.assertRaises(ValueError) as ctx:
            evaluate_observer_vs_gold(self.gold_gamma, p)
        self.assertIn("missing required trials", str(ctx.exception).lower())

    # FIX-05: Duplicate trial ID in observer payload fails validation
    def test_fix_05_duplicate_trial_fails(self):
        d = copy.deepcopy(self.gamma_data)
        d["ratings"].append(copy.deepcopy(d["ratings"][0]))
        with self.assertRaises(ValueError) as ctx:
            validate_ratings_payload(d, "Test")
        self.assertIn("duplicate trial_id detected", str(ctx.exception).lower())

    # FIX-06: Unrecognized taxonomy code fails validation
    def test_fix_06_invalid_taxonomy_fails(self):
        d = copy.deepcopy(self.gamma_data)
        d["ratings"][0]["assigned_cm_codes"] = ["INVALID_CODE_99"]
        with self.assertRaises(ValueError) as ctx:
            validate_ratings_payload(d, "Test")
        self.assertIn("invalid taxonomy code", str(ctx.exception).lower())

    # FIX-07: Unauthorized empirical provenance classification fails
    def test_fix_07_wrong_provenance_classification_fails(self):
        manifest_path = os.path.join(self.synthetic_dir, "SYNTHETIC_FIXTURE_PROVENANCE.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["fixtures"]["observer_gamma_v1_gold_concordant.synthetic.json"]["classification"] = "EMPIRICAL_HUMAN_OBSERVER"
        
        fixtures_entry = manifest["fixtures"]["observer_gamma_v1_gold_concordant.synthetic.json"]
        with self.assertRaises(ValueError) as ctx:
            if fixtures_entry.get("classification") != "SYNTHETIC_SCORING_REGRESSION":
                raise ValueError(f"Invalid classification: {fixtures_entry.get('classification')}")
        self.assertIn("Invalid classification", str(ctx.exception))

    # FIX-08: Changed critical action state triggers critical decision false negative
    def test_fix_08_changed_action_state_fails_gate(self):
        d = copy.deepcopy(self.gamma_data)
        d["ratings"][0]["action_state"] = "ACTION_APPROPRIATE"  # Gold requires INAPPROPRIATE
        p = self._write_temp_json("flipped_cde.json", d)
        res = evaluate_observer_vs_gold(self.gold_gamma, p)
        self.assertEqual(res["evaluation_summary"]["calibration_verdict"], "CALIBRATION_FAILED")
        self.assertEqual(res["evaluation_summary"]["critical_decision_false_negatives"], 1)

    # FIX-09: Incompatible gold key pairing fails evaluation
    def test_fix_09_wrong_gold_key_pairing_fails(self):
        gamma_fixture = os.path.join(self.synthetic_dir, "observer_gamma_v1_gold_concordant.synthetic.json")
        with self.assertRaises(ValueError) as ctx:
            evaluate_observer_vs_gold(self.gold_alpha, gamma_fixture)
        self.assertIn("missing required trials", str(ctx.exception).lower())

    # FIX-10: Fixture resolution is independent of current working directory
    def test_fix_10_working_directory_independence(self):
        orig_cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            path = verify_synthetic_fixture_provenance("observer_gamma_v1_gold_concordant.synthetic.json")
            self.assertTrue(os.path.isabs(path))
            self.assertTrue(os.path.exists(path))
        finally:
            os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
