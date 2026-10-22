#!/usr/bin/env python3
"""
ARX Terminal — Evidence-Led Discovery Cycle 2 Automated Verification Engine.

Governing Standard: ARX-DECISION-RESEARCH-PROCESS-V1 & ARX Closure Model v2.

Methodological Disclaimer on Verification Scope:
- Schema checks establish internal structural compliance of JSON ledgers; they do not establish economic validity.
- Date and accession format checks verify syntactic presence of required disclosures; they do not independently confirm external factual accuracy.
- Offline calculation checks verify arithmetic consistency between reported components and derived metrics.
- Source reconciliation verifies agreement between verified filing disclosures, internal JSON ledgers, and narrative reports.
- Frontend isolation checks verify architectural decoupling; they do not establish investment merit.
"""

import sys
import json
import hashlib
from pathlib import Path

FROZEN_BASELINE_LEDGER_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d"
AUTHORIZED_PUBLISHED_SYMBOLS = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"]

CYCLE_2_OFFLINE_ARITHMETIC_FIXTURES = {
    "TMDX": {
        "cik": "0001756262",
        "cfo": 192.840,
        "capex": 59.251,
        "expected_fcf": 133.589,
        "expected_conversion": 0.6927452810620203,
        "gross_profit": 362.806,
        "operating_income": 108.583,
        "net_income": 190.291,
        "revenue": 605.494,
        "debt_bridge": {
            "cash_and_equivalents": 472.675,
            "convertible_notes_principal": 459.996,
            "other_borrowings_principal": 60.000,
            "finance_lease_liabilities": 347.660,
            "total_debt_and_leases": 867.656,
            "net_debt_excl_leases": -47.321,
            "net_debt_incl_leases": -394.981
        }
    },
    "ETN": {
        "cik": "0001551182",
        "cfo": 4472.0,
        "capex": 919.0,
        "expected_fcf": 3553.0,
        "expected_conversion": 0.7944991055456172,
        "revenue": 27448.0,
        "gross_profit": 10317.0,
        "operating_income": 6713.0,
        "net_income": 4087.0,
        "debt_bridge": {
            "total_debt": 9895.0,
            "cash_and_short_term_investments": 803.0,
            "net_debt": 9092.0
        }
    }
}


def verify_cycle_2(repo_root: Path) -> dict:
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "records_verified": 0
    }

    # 1. Verify Immutability of Frozen Baseline Ledger (ODP-05)
    odp05_path = repo_root / "docs" / "research" / "opportunity_discovery" / "odp05_discovery_ledger.json"
    if not odp05_path.exists():
        results["passed"] = False
        results["errors"].append(f"Baseline ODP-05 ledger not found: {odp05_path}")
    else:
        with open(odp05_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if actual_sha != FROZEN_BASELINE_LEDGER_SHA256:
            results["passed"] = False
            results["errors"].append(
                f"Frozen baseline ledger SHA mismatch. Expected {FROZEN_BASELINE_LEDGER_SHA256}, got {actual_sha}"
            )

    # 2. Verify Published Watch Manifest Integrity
    manifest_path = repo_root / "docs" / "research" / "opportunity_discovery" / "published_watch_manifest.json"
    if not manifest_path.exists():
        results["passed"] = False
        results["errors"].append(f"Published watch manifest not found: {manifest_path}")
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        published_symbols = manifest_data.get("published_symbols", [])
        if published_symbols != AUTHORIZED_PUBLISHED_SYMBOLS:
            results["passed"] = False
            results["errors"].append(
                f"Published watch manifest violation: expected {AUTHORIZED_PUBLISHED_SYMBOLS}, got {published_symbols}"
            )
        if manifest_data.get("published_count") != 6:
            results["passed"] = False
            results["errors"].append(f"Published count must be exactly 6, got {manifest_data.get('published_count')}")

    # 3. Verify Cycle 2 Discovery Ledger Structure
    ledger_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "discovery_cycle_2_ledger.json"
    if not ledger_path.exists():
        results["passed"] = False
        results["errors"].append(f"Cycle 2 discovery ledger not found: {ledger_path}")
        return results

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    if ledger.get("schema_version") != "ARX-DISCOVERY-CYCLE-2-LEDGER-V1":
        results["passed"] = False
        results["errors"].append(f"Invalid schema version: {ledger.get('schema_version')}")

    if ledger.get("governing_backlog_item") != "BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-046":
        results["passed"] = False
        results["errors"].append(f"Invalid governing backlog item: {ledger.get('governing_backlog_item')}")

    examined_universe = ledger.get("examined_universe", [])
    if not isinstance(examined_universe, list) or len(examined_universe) == 0:
        results["passed"] = False
        results["errors"].append("Examined universe must be a non-empty list of candidate symbols")

    deep_recs = ledger.get("deep_investigations", {})
    if not isinstance(deep_recs, dict) or len(deep_recs) == 0:
        results["passed"] = False
        results["errors"].append("Deep investigations must contain at least one investigated candidate")

    # 4. Verify Individual Deep Records
    for symbol, rec in deep_recs.items():
        results["records_verified"] += 1
        dim = rec.get("dimensional_classification", {})

        # Enforce publication ineligibility
        if dim.get("public_publication_eligibility") != "INTERNAL_PILOT_RECORD_NOT_ELIGIBLE":
            results["passed"] = False
            results["errors"].append(
                f"Record {symbol} violates publication eligibility invariant: {dim.get('public_publication_eligibility')}"
            )

        # Enforce expectation gap status is qualified as unresolved
        if dim.get("expectation_gap_status") != "VALUATION_EXPECTATION_GAP_UNRESOLVED":
            results["passed"] = False
            results["errors"].append(
                f"Record {symbol} must qualify expectation_gap_status as VALUATION_EXPECTATION_GAP_UNRESOLVED"
            )

        # CIK Verification against authoritative fixture if available
        if symbol in CYCLE_2_OFFLINE_ARITHMETIC_FIXTURES:
            expected_cik = CYCLE_2_OFFLINE_ARITHMETIC_FIXTURES[symbol]["cik"]
            actual_cik = rec.get("cik")
            if actual_cik != expected_cik:
                results["passed"] = False
                results["errors"].append(f"Record {symbol} CIK mismatch: expected {expected_cik}, got {actual_cik}")

        # Check evidence citation and canonical SEC Archives URL
        dev = rec.get("observable_development", {})
        evidence_str = dev.get("primary_filing_evidence", "")
        if "10-K" not in evidence_str or "Form" not in evidence_str:
            results["passed"] = False
            results["errors"].append(f"Record {symbol} lacks explicit Form 10-K primary evidence reference")
        if "https://www.sec.gov/Archives/edgar/data/" not in evidence_str:
            results["passed"] = False
            results["errors"].append(f"Record {symbol} primary evidence must use canonical SEC Archives URL")

        # Check expectations disaggregation and ensure reverse_dcf_solved_condition is not an ungrounded claim
        expectations = rec.get("price_and_expectations", {}).get("expectation_disaggregation", {})
        consensus_status = expectations.get("sourced_analyst_consensus", "")
        if "UNRESOLVED_CONSENSUS" not in consensus_status:
            results["passed"] = False
            results["errors"].append(
                f"Record {symbol} must explicitly label consensus as UNRESOLVED_CONSENSUS when feeds are unavailable"
            )

        reverse_dcf = expectations.get("reverse_dcf_solved_condition")
        if reverse_dcf is not None and reverse_dcf != "":
            # If not null or empty, it must explicitly state it is unresolved / ungrounded
            if not any(token in str(reverse_dcf).upper() for token in ["UNRESOLVED", "NOT_SOLVED", "UNGROUNDED", "HYPOTHESIS"]):
                results["passed"] = False
                results["errors"].append(
                    f"Record {symbol} asserts reverse_dcf_solved_condition without explicit qualification or model: '{reverse_dcf}'"
                )

        # Arithmetic Reconciliation of Reported Financials
        fin = rec.get("economic_significance", {}).get("financial_reconciliation", {})
        cfo = fin.get("fy2025_cfo_usd_m")
        capex = fin.get("fy2025_capex_usd_m")
        fcf = fin.get("fy2025_fcf_usd_m")
        fcf_conv = fin.get("fy2025_fcf_cfo_conversion_pct")

        if cfo is None or capex is None or fcf is None or fcf_conv is None:
            results["passed"] = False
            results["errors"].append(f"Record {symbol} missing required FCF reconciliation fields")
        else:
            calc_fcf = round(cfo - capex, 3)
            if abs(calc_fcf - fcf) > 0.05:
                results["passed"] = False
                results["errors"].append(
                    f"Record {symbol} FCF arithmetic mismatch: CFO({cfo}) - Capex({capex}) = {calc_fcf}, reported {fcf}"
                )

            calc_conv = round((fcf / cfo) * 100, 2)
            if abs(calc_conv - round(fcf_conv, 2)) > 0.2:
                results["passed"] = False
                results["errors"].append(
                    f"Record {symbol} FCF conversion mismatch: FCF({fcf}) / CFO({cfo}) = {calc_conv}%, reported {fcf_conv}%"
                )

        # Full Debt Bridge Verification for TMDX
        if symbol == "TMDX":
            downside = rec.get("downside_and_survival", {})
            bridge = downside.get("debt_bridge_q2_2026")
            if not bridge or not isinstance(bridge, dict):
                results["passed"] = False
                results["errors"].append("TMDX requires explicit debt_bridge_q2_2026 object")
            else:
                required_debt_keys = [
                    "cash_and_equivalents_usd_m",
                    "convertible_notes_principal_usd_m",
                    "other_borrowings_principal_usd_m",
                    "finance_lease_liabilities_usd_m",
                    "total_debt_and_leases_usd_m",
                    "net_debt_incl_leases_usd_m"
                ]
                for rk in required_debt_keys:
                    if rk not in bridge:
                        results["passed"] = False
                        results["errors"].append(f"TMDX debt bridge missing required field: {rk}")

                if all(rk in bridge for rk in required_debt_keys):
                    conv = bridge["convertible_notes_principal_usd_m"]
                    other = bridge["other_borrowings_principal_usd_m"]
                    leases = bridge["finance_lease_liabilities_usd_m"]
                    gross_cash = bridge["cash_and_equivalents_usd_m"]
                    total_debt = bridge["total_debt_and_leases_usd_m"]
                    net_debt = bridge["net_debt_incl_leases_usd_m"]

                    calc_total = round(conv + other + leases, 3)
                    if abs(calc_total - total_debt) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"TMDX debt bridge sum mismatch: {conv} + {other} + {leases} = {calc_total}, reported {total_debt}"
                        )

                    calc_net = round(gross_cash - total_debt, 3)
                    if abs(calc_net - net_debt) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"TMDX net debt mismatch: {gross_cash} - {total_debt} = {calc_net}, reported {net_debt}"
                        )

                    if net_debt > 0:
                        results["passed"] = False
                        results["errors"].append(
                            f"TMDX cannot be represented as net cash positive with total debt & leases exceeding gross cash (net debt: {net_debt})"
                        )

    # 5. Source-Linked Reconciliation with Markdown Narrative
    records_md_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "DISCOVERY_OPPORTUNITY_RECORDS.md"
    if records_md_path.exists():
        records_text = records_md_path.read_text(encoding="utf-8")
        # TMDX reconciled narrative checks
        if "192.8" not in records_text or "133.5" not in records_text and "133.6" not in records_text:
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect reconciled TMDX CFO ($192.8M) or FCF ($133.6M)")
        # ETN reconciled narrative checks
        if "0001551182" not in records_text:
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect reconciled ETN CIK (0001551182)")
        if "4,472" not in records_text or "3,553" not in records_text:
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect reconciled ETN CFO ($4,472M) or FCF ($3,553M)")

    # 6. Verify Frontend Decoupling
    frontend_dir = repo_root / "frontend"
    if frontend_dir.exists():
        for file in frontend_dir.rglob("*.ts*"):
            if file.is_file():
                content = file.read_text(encoding="utf-8", errors="ignore")
                if "discovery_cycle_2" in content:
                    results["passed"] = False
                    results["errors"].append(f"Frontend file {file.name} illegally imports internal discovery cycle 2")

    # 7. Verify Required Markdown Documentation
    required_docs = [
        repo_root / "docs" / "research" / "discovery_cycle_2" / "EXPERIMENT_SPECIFICATION_AND_SEARCH_RECEIPTS.md",
        repo_root / "docs" / "research" / "discovery_cycle_2" / "SEARCH_UNIVERSE_AND_TRIAGE_MATRIX.md",
        repo_root / "docs" / "research" / "discovery_cycle_2" / "DISCOVERY_OPPORTUNITY_RECORDS.md",
        repo_root / "docs" / "research" / "discovery_cycle_2" / "CYCLE_2_EVALUATION_REPORT.md",
    ]
    for doc in required_docs:
        if not doc.exists():
            results["passed"] = False
            results["errors"].append(f"Required Cycle 2 documentation missing: {doc.name}")

    return results


def main():
    repo_root = Path(__file__).resolve().parents[2]
    print(f"Verifying Evidence-Led Discovery Cycle 2 in: {repo_root}")
    results = verify_cycle_2(repo_root)

    print(f"Records verified: {results['records_verified']}")
    if results["passed"]:
        print("[PASS] All Cycle 2 verification checks passed cleanly.")
        sys.exit(0)
    else:
        print("[FAIL] Verification failed with errors:")
        for err in results["errors"]:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
