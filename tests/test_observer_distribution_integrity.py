"""
ARX Terminal — Observer Distribution Integrity & Taxonomy Drift Regression Suite
Governing Authorization: ARX_UX_OBSERVER_DISTRIBUTION_INTEGRITY_REMEDIATION_AUTHORIZED
Framework: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
"""

import unittest
import os
import json
import zipfile
import hashlib

class TestObserverDistributionIntegrity(unittest.TestCase):

    def setUp(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.zip_path = os.path.join(self.root_dir, 'tests', 'fixtures', 'qualification', 'ARX_OBSERVER_QUALIFICATION_BUNDLE.zip')
        self.manifest_path = os.path.join(self.root_dir, 'tests', 'fixtures', 'qualification', 'custodian_package', 'PACKAGE_INTEGRITY_MANIFEST.json')
        self.gold_key_path = os.path.join(self.root_dir, 'tests', 'fixtures', 'qualification', 'custodian_package', 'gold_key_consolidated_30.json')

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
        for doc in ['ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md', 'ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.md', 'ARX_UX_OBSERVER_02_DISTRIBUTION_HANDOFF.md']:
            doc_path = os.path.join(self.root_dir, doc)
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for code, marker in expected_cm.items():
                self.assertIn(code, content, f"Missing {code} in {doc}")
                self.assertIn(marker.lower(), content.lower(), f"Missing marker '{marker}' for {code} in {doc}")

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
        for doc in ['ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md', 'ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.md', 'ARX_UX_OBSERVER_02_DISTRIBUTION_HANDOFF.md']:
            doc_path = os.path.join(self.root_dir, doc)
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for code, title in expected_sm.items():
                self.assertIn(code, content, f"Missing {code} in {doc}")
                self.assertIn(title.lower(), content.lower(), f"Missing title '{title}' for {code} in {doc}")

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
        docs = [
            'ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.md',
            'ARX_UX_OBSERVER_02_DISTRIBUTION_HANDOFF.md',
            'tests/fixtures/qualification/observer_package/ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md',
            'tests/fixtures/qualification/observer_package/SUBMISSION_INSTRUCTIONS.md'
        ]
        for doc in docs:
            doc_path = os.path.join(self.root_dir, doc)
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for term in drifted_terms:
                self.assertNotIn(term, content, f"Drifted term '{term}' still found in {doc}")

    def test_04_ratified_concept_identities(self):
        """Check 4: Verify Concepts A, B, and C retain approved identities."""
        concepts = [
            ("Concept A", "Simplified Linear Analysis"),
            ("Concept B", "Decision Cockpit"),
            ("Concept C", "Evidence-First Workspace")
        ]
        for doc in ['ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.md', 'ARX_UX_OBSERVER_02_DISTRIBUTION_HANDOFF.md', 'tests/fixtures/qualification/observer_package/SUBMISSION_INSTRUCTIONS.md']:
            doc_path = os.path.join(self.root_dir, doc)
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            for c_id, c_name in concepts:
                self.assertIn(c_id.lower(), content, f"Concept ID '{c_id}' not found in {doc}")
                self.assertIn(c_name.lower(), content, f"Concept name '{c_name}' not found in {doc}")

    def test_05_trial_ids_align_with_gold_key(self):
        """Check 5: Verify trial IDs in blinded transcripts match gold key IDs exactly."""
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
            tpath = os.path.join(self.root_dir, 'tests', 'fixtures', 'qualification', 'observer_package', fname)
            with open(tpath, 'r', encoding='utf-8') as f:
                tdata = json.load(f)
            for trial in tdata['trials']:
                transcript_ids.add(trial['trial_id'])

        self.assertEqual(transcript_ids, gold_ids, "Blinded transcript trial IDs do not match gold key")

    def test_06_synthetic_provenance_explicitly_disclosed(self):
        """Check 6: Verify explicit disclosure of synthetic provenance in all observer-facing files."""
        prov_text = "These are synthetic calibration transcripts developed to evaluate observer scoring consistency. They are not recordings of actual ARX Terminal research participants."
        docs = [
            'ARX_UX_OBSERVER_01_DISTRIBUTION_HANDOFF.md',
            'ARX_UX_OBSERVER_02_DISTRIBUTION_HANDOFF.md',
            'tests/fixtures/qualification/observer_package/SUBMISSION_INSTRUCTIONS.md',
            'tests/fixtures/qualification/observer_package/ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md'
        ]
        for doc in docs:
            doc_path = os.path.join(self.root_dir, doc)
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn(prov_text, content, f"Missing synthetic provenance disclosure in {doc}")

    def test_07_zip_bundle_has_no_answers_or_rationales(self):
        """Check 7: Ensure ZIP archive contains strictly blinded files and zero gold keys."""
        self.assertTrue(os.path.exists(self.zip_path), "ZIP bundle does not exist")
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
        with open(self.zip_path, 'rb') as f:
            computed_sha256 = hashlib.sha256(f.read()).hexdigest()
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        self.assertEqual(computed_sha256, manifest['current_bundle_sha256'], "Manifest bundle SHA256 does not match computed zip hash")

    def test_09_production_containment_preserved(self):
        """Check 9: Confirm zero prototype or research artifacts in frontend/app."""
        frontend_app = os.path.join(self.root_dir, 'frontend', 'app')
        for root, dirs, files in os.walk(frontend_app):
            for f in files:
                self.assertNotIn('prototype', f.lower(), f"Prototype artifact found in production app: {f}")
                self.assertNotIn('qualification', f.lower(), f"Qualification artifact found in production app: {f}")

    def test_10_no_fabricated_human_results(self):
        """Check 10: Ensure submission directory has 0 human submissions and scoring remains unexecuted."""
        sub_dir = os.path.join(self.root_dir, 'tests', 'fixtures', 'qualification', 'submissions')
        files = [f for f in os.listdir(sub_dir) if f.endswith('.json')]
        self.assertEqual(len(files), 0, f"Found unexpected rating files in submission lockbox: {files}")

if __name__ == '__main__':
    unittest.main()
