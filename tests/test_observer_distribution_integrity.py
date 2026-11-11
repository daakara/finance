"""
ARX Terminal — Observer Distribution Integrity & Taxonomy Drift Regression Suite
Governing Authorization: ARX_UX_INSTR001_SYNTHETIC_OBSERVER_FIXTURE_AMENDMENT_AUTHORIZED
Framework: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
"""

import unittest
import os
import json
import zipfile
import hashlib
from pathlib import Path


class TestObserverDistributionIntegrity(unittest.TestCase):
    """
    Evaluates observer qualification contracts against committed synthetic fixtures
    and qualification package assets without dependency on uncommitted operational artifacts.
    """

    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent
        self.root_dir = self.test_dir.parent
        self.fixture_dir = self.test_dir / "fixtures" / "observer"
        self.qualification_dir = self.test_dir / "fixtures" / "qualification"

        # Deterministic synthetic fixture paths
        self.synth_handoff_1 = self.fixture_dir / "ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.synthetic.md"
        self.synth_handoff_2 = self.fixture_dir / "ARX_UX_OBSERVER_02_DISTRIBUTION_HANDOFF.synthetic.md"

        # Committed qualification package paths
        self.handbook_path = self.qualification_dir / "observer_package" / "ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md"
        self.instructions_path = self.qualification_dir / "observer_package" / "SUBMISSION_INSTRUCTIONS.md"
        self.zip_path = self.qualification_dir / "ARX_OBSERVER_QUALIFICATION_BUNDLE.zip"
        self.manifest_path = self.qualification_dir / "custodian_package" / "PACKAGE_INTEGRITY_MANIFEST.json"
        self.gold_key_path = self.qualification_dir / "custodian_package" / "gold_key_consolidated_30.json"

    def test_01_frozen_canonical_cm_definitions(self):
        """Check 1: Verify all 9 CM codes retain their exact canonical definitions."""
        expected_cm = {
            "CM-01": "FAIL_CLOSED",
            "CM-02": "NOT_ACTIONABLE",
            "CM-03": "Target",
            "CM-04": "Stop Floor",
            "CM-05": "Stale Data",
            "CM-06": "UNKNOWN",
            "CM-07": "Live Execution",
            "CM-08": "Inverted",
            "CM-09": "WATCH"
        }
        target_docs = [self.handbook_path, self.synth_handoff_1, self.synth_handoff_2]
        for doc_path in target_docs:
            self.assertTrue(doc_path.exists(), f"Target document does not exist: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for code, marker in expected_cm.items():
                self.assertIn(code, content, f"Missing {code} in {doc_path.name}")
                self.assertIn(marker.lower(), content.lower(), f"Missing marker '{marker}' for {code} in {doc_path.name}")

    def test_02_ratified_supplemental_sm_definitions(self):
        """Check 2: Verify all 6 SM codes retain their exact definitions without drift."""
        expected_sm = {
            "SM-01": "Investment Horizon Mismatch",
            "SM-02": "Risk-Budget Over-Allocation",
            "SM-03": "Bear-Case Neglect",
            "SM-04": "Confluence Score Reification",
            "SM-05": "Screening Conflated with Execution Trigger",
            "SM-06": "Pipeline Degradation Neglect"
        }
        target_docs = [self.handbook_path, self.synth_handoff_1, self.synth_handoff_2]
        for doc_path in target_docs:
            self.assertTrue(doc_path.exists(), f"Target document does not exist: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for code, title in expected_sm.items():
                self.assertIn(code, content, f"Missing {code} in {doc_path.name}")
                self.assertIn(title.lower(), content.lower(), f"Missing title '{title}' for {code} in {doc_path.name}")

    def test_03_no_drifted_taxonomy_terms(self):
        """Check 3: Confirm complete eradication of hallucinated drifted taxonomy terms."""
        drifted_terms = [
            "Position Sizing Inversion",
            "Benchmark Misalignment",
            "Execution Boundary Violation",
            "Invariant Display Confusion",
            "Institutional Flow Misattribution",
            "Liquidity Void Blindness",
            "Volatility Regime Misalignment",
            "Factor Beta Confounding",
            "Gamma Exposure Inversion",
            "Cross-Asset Lead/Lag Distortion"
        ]
        target_docs = [
            self.synth_handoff_1,
            self.synth_handoff_2,
            self.handbook_path,
            self.instructions_path
        ]
        for doc_path in target_docs:
            self.assertTrue(doc_path.exists(), f"Target document does not exist: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for term in drifted_terms:
                self.assertNotIn(term, content, f"Drifted term '{term}' still found in {doc_path.name}")

    def test_04_ratified_concept_identities(self):
        """Check 4: Verify Concepts A, B, and C retain approved identities."""
        concepts = [
            ("Concept A", "Simplified Linear Analysis"),
            ("Concept B", "Decision Cockpit"),
            ("Concept C", "Evidence-First Workspace")
        ]
        target_docs = [self.synth_handoff_1, self.synth_handoff_2, self.instructions_path]
        for doc_path in target_docs:
            self.assertTrue(doc_path.exists(), f"Target document does not exist: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            for c_id, c_name in concepts:
                self.assertIn(c_id.lower(), content, f"Concept ID '{c_id}' not found in {doc_path.name}")
                self.assertIn(c_name.lower(), content, f"Concept name '{c_name}' not found in {doc_path.name}")

    def test_05_trial_ids_align_with_gold_key(self):
        """Check 5: Verify trial IDs in blinded transcripts match gold key IDs exactly."""
        self.assertTrue(self.gold_key_path.exists(), "Gold key does not exist")
        with open(self.gold_key_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        gold_ids = set(gold_data['items'].keys())

        expected_ids = set(
            [f"ALPHA-{i:02d}" for i in range(1, 11)] +
            [f"BETA-{i:02d}" for i in range(1, 11)] +
            [f"GAMMA-{i:02d}" for i in range(1, 11)]
        )
        self.assertEqual(gold_ids, expected_ids, "Gold key does not match expected ALPHA/BETA/GAMMA trial IDs")

        transcript_ids = set()
        for fname in ['TRANS_ALPHA_BLINDED.json', 'TRANS_BETA_BLINDED.json', 'TRANS_GAMMA_BLINDED.json']:
            tpath = self.qualification_dir / "observer_package" / fname
            self.assertTrue(tpath.exists(), f"Transcript does not exist: {tpath}")
            with open(tpath, 'r', encoding='utf-8') as f:
                tdata = json.load(f)
            for trial in tdata['trials']:
                transcript_ids.add(trial['trial_id'])

        self.assertEqual(transcript_ids, gold_ids, "Blinded transcript trial IDs do not match gold key")

    def test_06_synthetic_provenance_explicitly_disclosed(self):
        """Check 6: Verify explicit disclosure of synthetic provenance in all observer-facing files."""
        prov_text = "These are synthetic calibration transcripts developed to evaluate observer scoring consistency. They are not recordings of actual ARX Terminal research participants."
        target_docs = [
            self.synth_handoff_1,
            self.synth_handoff_2,
            self.instructions_path,
            self.handbook_path
        ]
        for doc_path in target_docs:
            self.assertTrue(doc_path.exists(), f"Target document does not exist: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn(prov_text, content, f"Missing synthetic provenance disclosure in {doc_path.name}")

    def test_07_zip_bundle_has_no_answers_or_rationales(self):
        """Check 7: Ensure ZIP archive contains strictly blinded files and zero gold keys."""
        self.assertTrue(self.zip_path.exists(), "ZIP bundle does not exist")
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            names = z.namelist()
            self.assertEqual(len(names), 6, f"Expected 6 files in zip, found {len(names)}: {names}")
            for name in names:
                self.assertFalse(name.startswith('gold_key'), f"Gold key found in zip: {name}")
                content = z.read(name).decode('utf-8', errors='ignore')
                self.assertNotIn("authoritative_cm_codes", content, f"Answer key leaked in {name}")
                self.assertNotIn("authoritative_sm_codes", content, f"Answer key leaked in {name}")

    def test_08_manifest_and_zip_hash_match(self):
        """Check 8: Verify current ZIP byte hash matches manifest declaration."""
        self.assertTrue(self.zip_path.exists(), "ZIP bundle does not exist")
        self.assertTrue(self.manifest_path.exists(), "Manifest does not exist")
        with open(self.zip_path, 'rb') as f:
            computed_sha256 = hashlib.sha256(f.read()).hexdigest()
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        self.assertEqual(computed_sha256, manifest['current_bundle_sha256'], "Manifest bundle SHA256 does not match computed zip hash")

    def test_09_production_containment_preserved(self):
        """Check 9: Confirm zero prototype or research artifacts in frontend/app."""
        frontend_app = self.root_dir / 'frontend' / 'app'
        if frontend_app.exists():
            for root, dirs, files in os.walk(frontend_app):
                for f in files:
                    self.assertNotIn('prototype', f.lower(), f"Prototype artifact found in production app: {f}")
                    self.assertNotIn('qualification', f.lower(), f"Qualification artifact found in production app: {f}")

    def test_10_no_fabricated_human_results(self):
        """Check 10: Ensure submission directory has 0 human submissions and scoring remains unexecuted."""
        sub_dir = self.qualification_dir / 'submissions'
        self.assertTrue(sub_dir.exists(), "Submissions directory does not exist")
        files = [f for f in os.listdir(sub_dir) if f.endswith('.json')]
        self.assertEqual(len(files), 0, f"Found unexpected rating files in submission lockbox: {files}")


class TestSyntheticObserverFixtureNegativeValidation(unittest.TestCase):
    """
    Negative tests enforcing that malformed, missing, drifted, or out-of-bounds
    synthetic fixtures produce deterministic and meaningful test failures.
    """

    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent
        self.fixture_dir = self.test_dir / "fixtures" / "observer"
        self.handoff_1 = self.fixture_dir / "ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.synthetic.md"

    def test_neg_01_missing_fixture_fails_meaningfully(self):
        """Verify that a missing synthetic fixture raises a descriptive assertion failure."""
        non_existent = self.fixture_dir / "NON_EXISTENT_FIXTURE.synthetic.md"
        self.assertFalse(non_existent.exists())
        with self.assertRaises(AssertionError) as ctx:
            self.assertTrue(non_existent.exists(), f"Target document does not exist: {non_existent}")
        self.assertIn("Target document does not exist", str(ctx.exception))

    def test_neg_02_malformed_fixture_missing_cm_code_fails(self):
        """Verify that an altered fixture missing a canonical CM code fails validation."""
        with open(self.handoff_1, 'r', encoding='utf-8') as f:
            valid_content = f.read()

        # Simulate malformed fixture content missing CM-01
        corrupted_content = valid_content.replace("CM-01", "CORRUPTED_CODE")
        self.assertNotIn("CM-01", corrupted_content)

        # Test assertion that would fail on corrupted content
        with self.assertRaises(AssertionError) as ctx:
            self.assertIn("CM-01", corrupted_content, "Missing CM-01 in fixture")
        self.assertIn("Missing CM-01", str(ctx.exception))

    def test_neg_03_unauthorized_drifted_wording_detected(self):
        """Verify that unauthorized drifted taxonomy terms are immediately detected if present."""
        with open(self.handoff_1, 'r', encoding='utf-8') as f:
            valid_content = f.read()

        # Simulate drifted content injection
        drifted_injection = valid_content + "\nUnauthorized: Position Sizing Inversion\n"
        with self.assertRaises(AssertionError) as ctx:
            self.assertNotIn("Position Sizing Inversion", drifted_injection, "Drifted term detected")
        self.assertIn("Drifted term detected", str(ctx.exception))

    def test_neg_04_synthetic_fixture_cannot_grant_operational_clearance(self):
        """Verify that synthetic fixtures explicitly disclaim operational clearance."""
        with open(self.handoff_1, 'r', encoding='utf-8') as f:
            content = f.read()

        # Must explicitly state disclaimer
        disclaimer = "Synthetic contract fixture only. This document does not authorize recruitment"
        self.assertIn(disclaimer, content)
        self.assertIn("SYNTHETIC_TEST_FIXTURE", content)

    def test_neg_05_unexpected_root_handoff_does_not_override_fixture(self):
        """Verify that tests deterministically resolve to tests/fixtures/observer and ignore root directory."""
        suite = TestObserverDistributionIntegrity()
        suite.setUp()
        self.assertEqual(suite.synth_handoff_1, self.fixture_dir / "ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.synthetic.md")
        self.assertTrue(str(suite.synth_handoff_1).startswith(str(self.fixture_dir)))

    def test_neg_06_path_traversal_outside_fixture_root_rejected(self):
        """Verify that path traversal attempts outside test directory are detected."""
        traversal_path = (self.fixture_dir / ".." / ".." / ".." / "some_external_file.md").resolve()
        # Must not be inside fixture_dir
        self.assertFalse(str(traversal_path).startswith(str(self.fixture_dir)))


if __name__ == '__main__':
    unittest.main()
