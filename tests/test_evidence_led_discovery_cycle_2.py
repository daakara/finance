import unittest
import json
import hashlib
from pathlib import Path
import sys

# Ensure scripts directory is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "scripts" / "research"))

from verify_discovery_cycle_2 import (
    verify_cycle_2,
    FROZEN_BASELINE_LEDGER_SHA256,
    AUTHORIZED_PUBLISHED_SYMBOLS,
    CYCLE_2_OFFLINE_ARITHMETIC_FIXTURES
)


class TestEvidenceLedDiscoveryCycle2(unittest.TestCase):
    """
    Verification Suite for Evidence-Led Discovery Cycle 2 (BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-046).

    Test Scope & Epistemic Boundaries:
    - Schema assertions verify internal JSON document structure, not external market validity.
    - Checksum checks verify file immutability, not predictive investment alpha.
    - Arithmetic checks reconcile derived metrics to reported components using offline fixtures.
    - Separation of internal exploratory research from authorized public Quality Watch publications.
    """

    def test_cycle_2_verification(self):
        res = verify_cycle_2(repo_root)
        self.assertTrue(res["passed"], f"Verification failed: {res['errors']}")
        self.assertEqual(res["records_verified"], 2)

    def test_public_publication_ineligibility(self):
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for symbol, rec in data["deep_investigations"].items():
            dim = rec["dimensional_classification"]
            self.assertEqual(
                dim["public_publication_eligibility"],
                "INTERNAL_PILOT_RECORD_NOT_ELIGIBLE",
                f"Record {symbol} must have public_publication_eligibility == INTERNAL_PILOT_RECORD_NOT_ELIGIBLE"
            )
            self.assertEqual(
                dim["expectation_gap_status"],
                "VALUATION_EXPECTATION_GAP_UNRESOLVED",
                f"Record {symbol} must have expectation_gap_status == VALUATION_EXPECTATION_GAP_UNRESOLVED"
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

    def test_cycle_2_fcf_offline_arithmetic_reconciliation(self):
        for sym, fixture in CYCLE_2_OFFLINE_ARITHMETIC_FIXTURES.items():
            fcf = fixture["cfo"] - fixture["capex"]
            self.assertAlmostEqual(fcf, fixture["expected_fcf"], places=2)
            conv = fcf / fixture["cfo"]
            self.assertAlmostEqual(conv, fixture["expected_conversion"], places=3)

    def test_cycle_2_evaluation_effort_and_decision(self):
        eval_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "CYCLE_2_EVALUATION_REPORT.md"
        self.assertTrue(eval_path.exists(), "CYCLE_2_EVALUATION_REPORT.md must exist")
        text = eval_path.read_text(encoding="utf-8")

        # Verify qualified CONTINUE recommendation
        self.assertIn("CONTINUE (Qualified Recommendation for Iterative Discovery Integration)", text)

        # Verify active effort accounting by stage
        self.assertIn("Stage 1: Starting State & Isolation", text)
        self.assertIn("Stage 2: Experiment Specification", text)
        self.assertIn("Stage 3: Candidate Generation", text)
        self.assertIn("Stage 4: Six-Step Triage Matrix", text)
        self.assertIn("Stage 5: Deep Investigations", text)
        self.assertIn("Stage 6: Automated Verification & Testing", text)
        self.assertIn("Total Active Research Time", text)

        # Verify prohibition of unsupported claims
        self.assertNotIn("proved signal value", text)
        self.assertNotIn("guaranteed alpha", text)
        self.assertNotIn("demonstrated market outperformance", text)

    def test_backlog_housekeeping_reconciliation(self):
        backlog_path = repo_root / "docs" / "planning" / "ENGINEERING_BACKLOG.md"
        self.assertTrue(backlog_path.exists(), "ENGINEERING_BACKLOG.md must exist")
        text = backlog_path.read_text(encoding="utf-8")

        # Verify Issue #43 / PR #45 completed adoption milestone
        self.assertIn("BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-043", text)
        self.assertIn("COMPLETED_AND_ADOPTED", text)
        self.assertIn("9d74ea903098e6c3628f8fab433de425a82b024c", text)

        # Verify Cycle 2 registered item
        self.assertIn("BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-046", text)
        self.assertIn("Evidence-Led Opportunity Discovery Cycle 2", text)


if __name__ == "__main__":
    unittest.main()
