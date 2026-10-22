import unittest
import json
import hashlib
from pathlib import Path
import sys

# Ensure scripts directory is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "scripts" / "research"))

from verify_discovery_pilot import (
    verify_pilot,
    FROZEN_BASELINE_LEDGER_SHA256,
    AUTHORIZED_PUBLISHED_SYMBOLS,
    OFFLINE_ARITHMETIC_FIXTURES
)


class TestEvidenceLedDiscoveryPilot(unittest.TestCase):
    """
    Verification Suite for Evidence-Led Discovery Pilot (Issue #43).
    
    Test Scope & Limitations:
    - Schema assertions verify internal JSON document structure, not external economic validity.
    - Checksum checks verify file immutability, not empirical investment performance.
    - Arithmetic checks reconcile derived metrics to reported components using offline fixtures.
    - Synthetic example checks verify template documentation cleanliness.
    """

    def test_discovery_pilot_verification(self):
        res = verify_pilot(repo_root)
        self.assertTrue(res["passed"], f"Verification failed: {res['errors']}")
        self.assertEqual(res["records_verified"], 3)

    def test_public_publication_ineligibility(self):
        ledger_path = repo_root / "docs" / "research" / "evidence_led_discovery" / "discovery_opportunity_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for symbol, rec in data["deep_investigations"].items():
            dim = rec["dimensional_classification"]
            self.assertEqual(
                dim["public_publication_eligibility"],
                "INTERNAL_PILOT_RECORD_NOT_ELIGIBLE",
                f"Record {symbol} must have public_publication_eligibility == INTERNAL_PILOT_RECORD_NOT_ELIGIBLE"
            )

    def test_published_watch_manifest_integrity(self):
        manifest_path = repo_root / "docs" / "research" / "opportunity_discovery" / "published_watch_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["published_symbols"], AUTHORIZED_PUBLISHED_SYMBOLS)
        self.assertEqual(manifest["published_count"], 6)

    def test_odp05_baseline_ledger_immutability(self):
        odp05_path = repo_root / "docs" / "research" / "opportunity_discovery" / "odp05_discovery_ledger.json"
        with open(odp05_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(actual_sha, FROZEN_BASELINE_LEDGER_SHA256)

    def test_standardized_decision_process_specification(self):
        proc_path = repo_root / "docs" / "research" / "DECISION_FOCUSED_RESEARCH_PROCESS.md"
        self.assertTrue(proc_path.exists(), "DECISION_FOCUSED_RESEARCH_PROCESS.md must exist")
        text = proc_path.read_text(encoding="utf-8")
        # Assert the 6 steps exist
        for step in [
            "Step A: Define the Decision",
            "Step B: Identify Decisive Uncertainties",
            "Step C: Connect Outcomes to Consequences",
            "Step D: Investigate Proportionately",
            "Step E: Make the Assessment",
            "Step F: Set Review Triggers and Stop"
        ]:
            self.assertIn(step, text)
        # Assert illustrative business examples exist
        for ticker in ["DECK", "FIX", "DUOL", "ULTA", "MEDP", "ISRG"]:
            self.assertIn(ticker, text)
        # Assert prospective adoption invariants
        self.assertIn("Prospective-Only Application", text)
        self.assertIn("WATCH is not BUY", text)

    def test_synthetic_template_example_cleanliness(self):
        proc_path = repo_root / "docs" / "research" / "DECISION_FOCUSED_RESEARCH_PROCESS.md"
        text = proc_path.read_text(encoding="utf-8")
        # Verify synthetic demonstration label
        self.assertIn("SYNTHETIC DEMONSTRATION EXAMPLE — NOT INVESTMENT RESEARCH", text)
        self.assertIn('"symbol": "SYNTH"', text)
        self.assertIn('"reference_price": null', text)
        # Verify absence of company-specific or expired artifacts
        self.assertNotIn("REC-ANET-20261022-V1", text)
        self.assertNotIn("Form 10-Q Q3 2024", text)

    def test_fcf_offline_arithmetic_reconciliation(self):
        for sym, fixture in OFFLINE_ARITHMETIC_FIXTURES.items():
            fcf = fixture["cfo"] - fixture["capex"]
            self.assertAlmostEqual(fcf, fixture["expected_fcf"], places=2)
            conv = fcf / fixture["cfo"]
            self.assertAlmostEqual(conv, fixture["expected_conversion"], places=3)

    def test_snps_merger_completion_reconciliation(self):
        triage_path = repo_root / "docs" / "research" / "evidence_led_discovery" / "SEARCH_UNIVERSE_AND_TRIAGE_MATRIX.md"
        text = triage_path.read_text(encoding="utf-8")
        self.assertIn("July 17, 2025", text)
        self.assertNotIn("pending $35B", text.lower())

    def test_pilot_evaluation_epistemic_qualifications(self):
        eval_path = repo_root / "docs" / "research" / "evidence_led_discovery" / "PILOT_EVALUATION_REPORT.md"
        text = eval_path.read_text(encoding="utf-8")
        # Check seven-dimensional structure
        self.assertIn("1. Research Structure and Isolation", text)
        self.assertIn("2. Source Accuracy", text)
        self.assertIn("3. Useful Questions Surfaced", text)
        self.assertIn("4. Defensible Priority Changes", text)
        self.assertIn("5. Candidate-Generation Effectiveness", text)
        self.assertIn("6. Effort Measurement", text)
        self.assertIn("7. Investment Performance", text)
        # Check removal of unwarranted claims
        text_lower = text.lower()
        self.assertNotIn("binding continue", text_lower)
        self.assertNotIn("fundamentally superior", text_lower)
        self.assertNotIn("proved signal value", text_lower)


if __name__ == "__main__":
    unittest.main()
