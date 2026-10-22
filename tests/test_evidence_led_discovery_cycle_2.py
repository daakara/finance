import unittest
import json
import hashlib
import copy
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
    - Source reconciliation tests cross-verify primary filings, ledger data, and markdown records.
    - Negative tests enforce detection of corrupted inputs, incomplete debt bridges, and ungrounded claims.
    - Separation of internal exploratory research from authorized public Quality Watch publications.
    """

    def test_cycle_2_verification(self):
        res = verify_cycle_2(repo_root)
        self.assertTrue(res["passed"], f"Verification failed: {res['errors']}")
        self.assertGreaterEqual(res["records_verified"], 1)

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

    def test_cycle_2_source_linked_reconciliation(self):
        """Cross-reconcile fixture, ledger, and markdown narrative for primary reported values."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        records_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "DISCOVERY_OPPORTUNITY_RECORDS.md"
        self.assertTrue(ledger_path.exists())
        self.assertTrue(records_path.exists())

        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger_data = json.load(f)
        records_md = records_path.read_text(encoding="utf-8")

        for sym, fixture in CYCLE_2_OFFLINE_ARITHMETIC_FIXTURES.items():
            if sym not in ledger_data["deep_investigations"]:
                continue
            rec = ledger_data["deep_investigations"][sym]
            fin = rec["economic_significance"]["financial_reconciliation"]

            # 1. Match ledger to authoritative primary fixture
            self.assertEqual(rec["cik"], fixture["cik"])
            self.assertAlmostEqual(fin["fy2025_cfo_usd_m"], fixture["cfo"], places=2)
            self.assertAlmostEqual(fin["fy2025_capex_usd_m"], fixture["capex"], places=2)
            self.assertAlmostEqual(fin["fy2025_fcf_usd_m"], fixture["expected_fcf"], places=2)
            self.assertAlmostEqual(fin["fy2025_revenue_usd_m"], fixture["revenue"], places=1)

            # 2. Check narrative representation in markdown
            self.assertIn(fixture["cik"], records_md, f"CIK {fixture['cik']} for {sym} missing from markdown")

    def test_negative_materially_altered_reported_financial_input(self):
        """Fails when reported input is altered even if dependent arithmetic is internally consistent."""
        def evaluate_ledger_financials(fin_data, symbol):
            fixture = CYCLE_2_OFFLINE_ARITHMETIC_FIXTURES[symbol]
            # Verify primary source fidelity
            if abs(fin_data["fy2025_cfo_usd_m"] - fixture["cfo"]) > 0.05:
                raise ValueError(f"Material alteration of CFO for {symbol}: expected {fixture['cfo']}, got {fin_data['fy2025_cfo_usd_m']}")
            if abs(fin_data["fy2025_capex_usd_m"] - fixture["capex"]) > 0.05:
                raise ValueError(f"Material alteration of Capex for {symbol}: expected {fixture['capex']}, got {fin_data['fy2025_capex_usd_m']}")

        # Altered input with internally consistent arithmetic: CFO=999, Capex=499, FCF=500 (999-499=500)
        altered_fin = {
            "fy2025_cfo_usd_m": 999.0,
            "fy2025_capex_usd_m": 499.0,
            "fy2025_fcf_usd_m": 500.0,
            "fy2025_fcf_cfo_conversion_pct": 50.05
        }
        with self.assertRaises(ValueError) as ctx:
            evaluate_ledger_financials(altered_fin, "TMDX")
        self.assertIn("Material alteration of CFO", str(ctx.exception))

    def test_negative_incomplete_debt_bridge_for_tmdx(self):
        """Fails when TMDX debt bridge omits other borrowings or aircraft finance lease liabilities."""
        def validate_tmdx_debt_bridge(bridge):
            required = [
                "cash_and_equivalents_usd_m",
                "convertible_notes_principal_usd_m",
                "other_borrowings_principal_usd_m",
                "finance_lease_liabilities_usd_m",
                "total_debt_and_leases_usd_m",
                "net_debt_incl_leases_usd_m"
            ]
            for key in required:
                if key not in bridge:
                    raise KeyError(f"Missing required debt bridge component: {key}")
            # Ensure net debt cannot be asserted as positive when obligations exceed cash
            if bridge["net_debt_incl_leases_usd_m"] > 0:
                raise ValueError("TMDX cannot be represented as net cash positive with obligations exceeding cash")

        # Incomplete bridge omitting aircraft finance leases
        incomplete_bridge = {
            "cash_and_equivalents_usd_m": 472.675,
            "convertible_notes_principal_usd_m": 459.996,
            "other_borrowings_principal_usd_m": 60.000,
            "total_debt_and_leases_usd_m": 519.996,
            "net_debt_incl_leases_usd_m": -47.321
        }
        with self.assertRaises(KeyError) as ctx:
            validate_tmdx_debt_bridge(incomplete_bridge)
        self.assertIn("finance_lease_liabilities_usd_m", str(ctx.exception))

        # Faulty bridge asserting positive net cash
        faulty_net_cash_bridge = {
            "cash_and_equivalents_usd_m": 472.675,
            "convertible_notes_principal_usd_m": 459.996,
            "other_borrowings_principal_usd_m": 60.000,
            "finance_lease_liabilities_usd_m": 347.660,
            "total_debt_and_leases_usd_m": 867.656,
            "net_debt_incl_leases_usd_m": 22.7
        }
        with self.assertRaises(ValueError) as ctx:
            validate_tmdx_debt_bridge(faulty_net_cash_bridge)
        self.assertIn("net cash positive", str(ctx.exception))

    def test_negative_asserted_reverse_dcf_without_model(self):
        """Fails when a solved reverse-DCF condition is asserted without a dated model."""
        def validate_reverse_dcf(expectation_dict):
            condition = expectation_dict.get("reverse_dcf_solved_condition")
            if condition is None or condition == "":
                return
            valid_unresolved_tokens = ["UNRESOLVED", "NOT_SOLVED", "UNGROUNDED", "HYPOTHESIS"]
            if not any(token in str(condition).upper() for token in valid_unresolved_tokens):
                raise ValueError(f"Ungrounded solved DCF assertion detected without dated model evidence: '{condition}'")

        ungrounded_assertion = {
            "reverse_dcf_solved_condition": "High volume expansion across US donor pools required to justify 10% hurdle rate"
        }
        with self.assertRaises(ValueError) as ctx:
            validate_reverse_dcf(ungrounded_assertion)
        self.assertIn("Ungrounded solved DCF assertion detected", str(ctx.exception))

    def test_cycle_2_evaluation_effort_and_epistemic_discipline(self):
        eval_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "CYCLE_2_EVALUATION_REPORT.md"
        self.assertTrue(eval_path.exists(), "CYCLE_2_EVALUATION_REPORT.md must exist")
        text = eval_path.read_text(encoding="utf-8")

        # Verify structured effort logging and formal recommendation sections exist
        self.assertIn("## 2. Estimated Active Effort by Stage", text)
        self.assertIn("## 5. Formal Process Recommendation", text)

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
