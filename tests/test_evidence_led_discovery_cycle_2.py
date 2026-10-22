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
    CYCLE_2_PRIMARY_SOURCE_FIXTURES
)


class TestEvidenceLedDiscoveryCycle2(unittest.TestCase):
    """
    Verification Suite for Evidence-Led Discovery Cycle 2 (BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-046).

    Test Scope & Epistemic Boundaries:
    - Exercises the real verification path (verify_cycle_2) for both control and mutated cases.
    - Demonstrates that source fidelity checks reject altered inputs even with internally consistent arithmetic.
    - Demonstrates enforcement of the standard net-debt sign convention (net debt = debt - cash).
    - Verifies exclusion of unsupported solved-DCF conditions and metric identity validation.
    - Enforces invariant immutability and separation of internal research from public manifests.
    """

    def test_a_corrected_control_records_pass(self):
        """A. Corrected control records pass through the real verifier."""
        res = verify_cycle_2(repo_root)
        self.assertTrue(res["passed"], f"Verification failed with errors: {res.get('errors')}")
        self.assertEqual(res["records_verified"], 2)

    def test_b_consistently_altered_financial_inputs_fail(self):
        """B. Consistently altered financial inputs fail: arithmetic consistent but source fidelity broken."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        mutated = copy.deepcopy(ledger)
        fin = mutated["deep_investigations"]["TMDX"]["economic_significance"]["financial_reconciliation"]
        # Consistent arithmetic: CFO=999, Capex=499, FCF=500, Conv=50.05%
        fin["fy2025_cfo_usd_m"] = 999.0
        fin["fy2025_capex_usd_m"] = 499.0
        fin["fy2025_fcf_usd_m"] = 500.0
        fin["fy2025_fcf_cfo_conversion_pct"] = round((500.0 / 999.0) * 100, 2)

        res = verify_cycle_2(repo_root, ledger_data=mutated)
        self.assertFalse(res["passed"], "Real verifier must fail on altered reported CFO/Capex despite consistent arithmetic")
        self.assertTrue(
            any("does not match primary source fixture" in err for err in res["errors"]),
            f"Expected source fidelity mismatch error, got: {res['errors']}"
        )

    def test_c_missing_other_borrowings_fails(self):
        """C. Missing other borrowings fail: omit TMDX other borrowing component."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        mutated = copy.deepcopy(ledger)
        bridge = mutated["deep_investigations"]["TMDX"]["downside_and_survival"]["debt_bridge_q2_2026"]
        del bridge["other_borrowings_principal_usd_m"]

        res = verify_cycle_2(repo_root, ledger_data=mutated)
        self.assertFalse(res["passed"], "Real verifier must fail when other borrowings component is missing")
        self.assertTrue(
            any("missing required field: other_borrowings_principal_usd_m" in err for err in res["errors"]),
            f"Expected missing field error, got: {res['errors']}"
        )

    def test_d_incorrect_other_borrowings_fails(self):
        """D. Incorrect other borrowings fail: set to zero and consistently recalculate totals."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        mutated = copy.deepcopy(ledger)
        bridge = mutated["deep_investigations"]["TMDX"]["downside_and_survival"]["debt_bridge_q2_2026"]
        bridge["other_borrowings_principal_usd_m"] = 0.0
        # Consistently recalculate dependent arithmetic
        bridge["funded_debt_excl_leases_usd_m"] = 459.996
        bridge["total_debt_and_leases_usd_m"] = 807.656
        bridge["net_debt_excl_leases_usd_m"] = -12.679
        bridge["net_debt_incl_leases_usd_m"] = 334.981

        res = verify_cycle_2(repo_root, ledger_data=mutated)
        self.assertFalse(res["passed"], "Real verifier must fail when other borrowings value is altered")
        self.assertTrue(
            any("TMDX other borrowings mismatch" in err for err in res["errors"]),
            f"Expected other borrowings fixture mismatch error, got: {res['errors']}"
        )

    def test_e_reversed_net_debt_signs_fail(self):
        """E. Reversed net-debt signs fail: negative net debt asserted despite obligations exceeding cash."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        mutated = copy.deepcopy(ledger)
        bridge = mutated["deep_investigations"]["TMDX"]["downside_and_survival"]["debt_bridge_q2_2026"]
        # Invert sign to negative despite total debt ($867.7M) exceeding cash ($472.7M)
        bridge["net_debt_incl_leases_usd_m"] = -394.981

        res = verify_cycle_2(repo_root, ledger_data=mutated)
        self.assertFalse(res["passed"], "Real verifier must fail on reversed net debt sign")
        self.assertTrue(
            any("Sign reversal error" in err for err in res["errors"]),
            f"Expected sign reversal error, got: {res['errors']}"
        )

    def test_f_correct_positive_net_debt_passes(self):
        """F. Correct positive net debt passes without disqualifying indebted companies."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        bridge_tmdx = ledger["deep_investigations"]["TMDX"]["downside_and_survival"]["debt_bridge_q2_2026"]
        bridge_etn = ledger["deep_investigations"]["ETN"]["downside_and_survival"]["debt_bridge_fy2025"]

        # Standard convention: net debt = debt - cash. When debt > cash, net debt is positive.
        self.assertEqual(bridge_tmdx["net_debt_incl_leases_usd_m"], 394.981)
        self.assertEqual(bridge_tmdx["net_debt_excl_leases_usd_m"], 47.321)
        self.assertEqual(bridge_etn["net_debt_usd_m"], 9092.0)

        res = verify_cycle_2(repo_root, ledger_data=ledger)
        self.assertTrue(res["passed"], "Valid positive net debt must pass cleanly without arbitrary penalty")

    def test_g_unsupported_solved_dcf_assertions_fail(self):
        """G. Unsupported solved-DCF assertions fail through the real verification path."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        mutated = copy.deepcopy(ledger)
        exp = mutated["deep_investigations"]["TMDX"]["price_and_expectations"]["expectation_disaggregation"]
        exp["reverse_dcf_solved_condition"] = "High volume expansion across US donor pools required to justify 10% hurdle rate"

        res = verify_cycle_2(repo_root, ledger_data=mutated)
        self.assertFalse(res["passed"], "Real verifier must fail when solved DCF assertions lack explicit model evidence")
        self.assertTrue(
            any("asserts reverse_dcf_solved_condition without explicit qualification" in err for err in res["errors"]),
            f"Expected ungrounded solved DCF error, got: {res['errors']}"
        )

    def test_h_eaton_total_segment_profit_identity_validation(self):
        """H. Enforce correct metric identity: reject total segment operating profit labeled as operating income."""
        ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        # Mutate to old defect: storing segment profit as operating income
        mutated = copy.deepcopy(ledger)
        fin = mutated["deep_investigations"]["ETN"]["economic_significance"]["financial_reconciliation"]
        fin["fy2025_operating_income_usd_m"] = 6713.0

        res = verify_cycle_2(repo_root, ledger_data=mutated)
        self.assertFalse(res["passed"], "Real verifier must reject segment profit mislabeled as consolidated operating income")
        self.assertTrue(
            any("incorrectly stores total segment operating profit as fy2025_operating_income_usd_m" in err for err in res["errors"]),
            f"Expected segment profit mislabel error, got: {res['errors']}"
        )

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

        for sym, fixture in CYCLE_2_PRIMARY_SOURCE_FIXTURES.items():
            if sym not in ledger_data["deep_investigations"]:
                continue
            rec = ledger_data["deep_investigations"][sym]
            fin = rec["economic_significance"]["financial_reconciliation"]
            annual_comps = fixture["measurement_periods"]["annual_fy2025"]["reported_components"]

            self.assertEqual(rec["cik"], fixture["cik"])
            self.assertAlmostEqual(fin["fy2025_cfo_usd_m"], annual_comps["cfo"], places=2)
            self.assertAlmostEqual(fin["fy2025_capex_usd_m"], annual_comps["capex"], places=2)
            self.assertAlmostEqual(fin["fy2025_revenue_usd_m"], annual_comps.get("revenue", annual_comps.get("net_sales")), places=1)
            self.assertIn(fixture["cik"], records_md, f"CIK {fixture['cik']} for {sym} missing from markdown")

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
