#!/usr/bin/env python3
"""
ARX Terminal — Evidence-Led Discovery Cycle 2 Automated Verification Engine.

Governing Standard: ARX-DECISION-RESEARCH-PROCESS-V1 & ARX Closure Model v2.

Methodological Disclaimer on Verification Scope:
- Schema checks establish internal structural compliance of JSON ledgers; they do not establish economic validity.
- Date and accession format checks verify syntactic presence of required disclosures; they do not independently confirm external factual accuracy.
- Source reconciliation verifies exact agreement between verified primary filings, internal JSON ledgers, and narrative reports.
- Offline calculation checks verify arithmetic consistency and adherence to the standard net-debt sign convention.
- Frontend isolation checks verify architectural decoupling; they do not establish investment merit.
"""

import sys
import json
import hashlib
from pathlib import Path

FROZEN_BASELINE_LEDGER_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d"
AUTHORIZED_PUBLISHED_SYMBOLS = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"]

# Compact, primary-source linked fixtures with full filing locators and measurement periods
CYCLE_2_PRIMARY_SOURCE_FIXTURES = {
    "TMDX": {
        "cik": "0001756262",
        "company_name": "TransMedics Group, Inc.",
        "measurement_periods": {
            "annual_fy2025": {
                "filing_form": "Form 10-K",
                "accession": "0001193125-26-067032",
                "filing_url": "https://www.sec.gov/Archives/edgar/data/1756262/000119312526067032/tmdx-20251231.htm",
                "period_ended": "2025-12-31",
                "units": "USD_MILLIONS",
                "statement_locators": {
                    "revenue": "Item 8, Consolidated Statements of Operations, p. 74",
                    "gross_profit": "Item 8, Consolidated Statements of Operations, p. 74",
                    "operating_income": "Item 8, Consolidated Statements of Operations, p. 74",
                    "net_income": "Item 8, Consolidated Statements of Operations, p. 74",
                    "cfo": "Item 8, Consolidated Statements of Cash Flows, p. 77",
                    "capex": "Item 8, Consolidated Statements of Cash Flows, p. 77 ('Purchases of property and equipment')"
                },
                "reported_components": {
                    "revenue": 605.494,
                    "gross_profit": 362.806,
                    "operating_income": 108.583,
                    "net_income": 190.291,
                    "cfo": 192.840,
                    "capex": 59.251,
                    "tax_valuation_allowance_benefit": 103.3
                },
                "derived_metrics": {
                    "fcf": 133.589,
                    "fcf_conversion_pct": 69.2745
                }
            },
            "interim_q2_2026": {
                "filing_form": "Form 10-Q",
                "accession": "0001193125-26-332843",
                "filing_url": "https://www.sec.gov/Archives/edgar/data/1756262/000119312526332843/tmdx-20260630.htm",
                "period_ended": "2026-06-30",
                "units": "USD_MILLIONS",
                "calculation_convention": "net_debt = included_debt_obligations - cash_and_equivalents",
                "statement_locators": {
                    "cash_and_equivalents": "Item 1, Condensed Consolidated Balance Sheets, p. 3",
                    "convertible_notes_principal": "Note 7, Debt and Lease Obligations, p. 11",
                    "other_borrowings_principal": "Note 7, Debt and Lease Obligations, p. 11",
                    "finance_lease_liabilities": "Item 1, Condensed Consolidated Balance Sheets, p. 3 & Note 7, p. 13",
                    "finance_lease_additions_1h2026": "Item 1, Condensed Consolidated Statements of Cash Flows, Supplemental, p. 6"
                },
                "reported_components": {
                    "cash_and_equivalents": 472.675,
                    "convertible_notes_principal": 459.996,
                    "other_borrowings_principal": 60.000,
                    "finance_lease_liabilities": 347.660,
                    "finance_lease_additions_six_months_ended_june_30_2026": 340.040
                },
                "derived_debt_metrics": {
                    "funded_debt_excl_leases": 519.996,
                    "net_debt_excl_leases": 47.321,
                    "total_debt_and_leases": 867.656,
                    "net_debt_incl_leases": 394.981
                }
            }
        }
    },
    "ETN": {
        "cik": "0001551182",
        "company_name": "Eaton Corporation plc",
        "measurement_periods": {
            "annual_fy2025": {
                "filing_form": "Form 10-K",
                "accession": "0001551182-26-000007",
                "filing_url": "https://www.sec.gov/Archives/edgar/data/1551182/000155118226000007/etn-20251231.htm",
                "period_ended": "2025-12-31",
                "units": "USD_MILLIONS",
                "calculation_convention": "net_debt = total_debt - cash_and_short_term_investments",
                "statement_locators": {
                    "net_sales": "Item 8, Consolidated Statements of Income, p. 52",
                    "gross_profit": "Item 8, Consolidated Statements of Income, p. 52",
                    "total_segment_operating_profit": "Item 7, MD&A, Results of Operations by Segment, p. 33",
                    "electrical_americas_sales": "Item 7, MD&A, Results of Operations by Segment, p. 33",
                    "electrical_americas_segment_operating_profit": "Item 7, MD&A, Results of Operations by Segment, p. 33",
                    "net_income_ordinary_shareholders": "Item 8, Consolidated Statements of Income, p. 52",
                    "cfo": "Item 8, Consolidated Statements of Cash Flows, p. 55",
                    "capex": "Item 8, Consolidated Statements of Cash Flows, p. 55 ('Capital expenditures for property, plant and equipment')",
                    "total_debt": "Item 8, Consolidated Balance Sheets, p. 54",
                    "cash_and_short_term_investments": "Item 8, Consolidated Balance Sheets, p. 54"
                },
                "reported_components": {
                    "net_sales": 27448.0,
                    "gross_profit": 10317.0,
                    "total_segment_operating_profit": 6713.0,
                    "electrical_americas_sales": 13276.0,
                    "electrical_americas_segment_operating_profit": 3972.0,
                    "net_income": 4087.0,
                    "cfo": 4472.0,
                    "capex": 919.0,
                    "total_debt": 9895.0,
                    "cash_and_short_term_investments": 803.0
                },
                "derived_metrics": {
                    "fcf": 3553.0,
                    "fcf_conversion_pct": 79.4499,
                    "total_segment_operating_margin_pct": 24.4572,
                    "electrical_americas_margin_pct": 29.9186,
                    "net_debt": 9092.0
                }
            }
        }
    }
}


def verify_cycle_2(repo_root: Path, ledger_data: dict = None) -> dict:
    """
    Executes the full Cycle 2 verification engine across empirical authorities,
    JSON discovery ledgers, primary-source arithmetic fixtures, and narrative records.
    """
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

    # 3. Load and Verify Cycle 2 Discovery Ledger Structure
    if ledger_data is not None:
        ledger = ledger_data
    else:
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

    # 4. Verify Individual Deep Records against Source-Linked Fixtures & Rules
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

        # CIK Verification against authoritative fixture
        if symbol in CYCLE_2_PRIMARY_SOURCE_FIXTURES:
            expected_cik = CYCLE_2_PRIMARY_SOURCE_FIXTURES[symbol]["cik"]
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
            valid_unresolved_tokens = ["UNRESOLVED", "NOT_SOLVED", "UNGROUNDED", "HYPOTHESIS", "NULL"]
            if not any(token in str(reverse_dcf).upper() for token in valid_unresolved_tokens):
                results["passed"] = False
                results["errors"].append(
                    f"Record {symbol} asserts reverse_dcf_solved_condition without explicit qualification or model: '{reverse_dcf}'"
                )

        # Source Fidelity & Arithmetic Reconciliation of Reported Financials
        fin = rec.get("economic_significance", {}).get("financial_reconciliation", {})
        cfo = fin.get("fy2025_cfo_usd_m")
        capex = fin.get("fy2025_capex_usd_m")
        fcf = fin.get("fy2025_fcf_usd_m")
        fcf_conv = fin.get("fy2025_fcf_cfo_conversion_pct")

        if cfo is None or capex is None or fcf is None or fcf_conv is None:
            results["passed"] = False
            results["errors"].append(f"Record {symbol} missing required FCF reconciliation fields")
        else:
            # 1. Compare reported components to authoritative primary fixtures (catches altered inputs)
            if symbol in CYCLE_2_PRIMARY_SOURCE_FIXTURES:
                fixture_annual = CYCLE_2_PRIMARY_SOURCE_FIXTURES[symbol]["measurement_periods"]["annual_fy2025"]
                exp_comps = fixture_annual["reported_components"]

                if abs(cfo - exp_comps["cfo"]) > 0.05:
                    results["passed"] = False
                    results["errors"].append(
                        f"Record {symbol} reported CFO ({cfo}) does not match primary source fixture ({exp_comps['cfo']})"
                    )

                if abs(capex - exp_comps["capex"]) > 0.05:
                    results["passed"] = False
                    results["errors"].append(
                        f"Record {symbol} reported Capex ({capex}) does not match primary source fixture ({exp_comps['capex']})"
                    )

                # Metric identity check for Eaton: total segment operating profit vs consolidated op income
                if symbol == "ETN":
                    if "fy2025_operating_income_usd_m" in fin:
                        results["passed"] = False
                        results["errors"].append(
                            "Record ETN incorrectly stores total segment operating profit as fy2025_operating_income_usd_m"
                        )
                    seg_profit = fin.get("fy2025_total_segment_operating_profit_usd_m")
                    if seg_profit is None or abs(seg_profit - exp_comps["total_segment_operating_profit"]) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"Record ETN missing or incorrect fy2025_total_segment_operating_profit_usd_m: reported {seg_profit}, expected {exp_comps['total_segment_operating_profit']}"
                        )

            # 2. Recompute derived FCF from verified reported components
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

        # Full Debt Bridge & Net-Debt Sign Convention Verification
        downside = rec.get("downside_and_survival", {})
        if symbol == "TMDX":
            bridge = downside.get("debt_bridge_q2_2026")
            if not bridge or not isinstance(bridge, dict):
                results["passed"] = False
                results["errors"].append("TMDX requires explicit debt_bridge_q2_2026 object")
            else:
                required_debt_keys = [
                    "cash_and_equivalents_usd_m",
                    "convertible_notes_principal_usd_m",
                    "other_borrowings_principal_usd_m",
                    "funded_debt_excl_leases_usd_m",
                    "finance_lease_liabilities_usd_m",
                    "total_debt_and_leases_usd_m",
                    "net_debt_excl_leases_usd_m",
                    "net_debt_incl_leases_usd_m"
                ]
                for rk in required_debt_keys:
                    if rk not in bridge:
                        results["passed"] = False
                        results["errors"].append(f"TMDX debt bridge missing required field: {rk}")

                if all(rk in bridge for rk in required_debt_keys):
                    conv = bridge["convertible_notes_principal_usd_m"]
                    other = bridge["other_borrowings_principal_usd_m"]
                    funded_excl = bridge["funded_debt_excl_leases_usd_m"]
                    leases = bridge["finance_lease_liabilities_usd_m"]
                    gross_cash = bridge["cash_and_equivalents_usd_m"]
                    total_debt = bridge["total_debt_and_leases_usd_m"]
                    net_excl = bridge["net_debt_excl_leases_usd_m"]
                    net_incl = bridge["net_debt_incl_leases_usd_m"]

                    # Primary source fidelity check for components
                    fixture_q2 = CYCLE_2_PRIMARY_SOURCE_FIXTURES["TMDX"]["measurement_periods"]["interim_q2_2026"]["reported_components"]
                    if abs(conv - fixture_q2["convertible_notes_principal"]) > 0.05:
                        results["passed"] = False
                        results["errors"].append(f"TMDX convertible notes mismatch: expected {fixture_q2['convertible_notes_principal']}, got {conv}")
                    if abs(other - fixture_q2["other_borrowings_principal"]) > 0.05:
                        results["passed"] = False
                        results["errors"].append(f"TMDX other borrowings mismatch: expected {fixture_q2['other_borrowings_principal']}, got {other}")
                    if abs(leases - fixture_q2["finance_lease_liabilities"]) > 0.05:
                        results["passed"] = False
                        results["errors"].append(f"TMDX finance lease liabilities mismatch: expected {fixture_q2['finance_lease_liabilities']}, got {leases}")
                    if abs(gross_cash - fixture_q2["cash_and_equivalents"]) > 0.05:
                        results["passed"] = False
                        results["errors"].append(f"TMDX cash mismatch: expected {fixture_q2['cash_and_equivalents']}, got {gross_cash}")

                    # Sum checks
                    calc_funded = round(conv + other, 3)
                    if abs(calc_funded - funded_excl) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"TMDX funded debt excl leases mismatch: {conv} + {other} = {calc_funded}, reported {funded_excl}"
                        )

                    calc_total = round(conv + other + leases, 3)
                    if abs(calc_total - total_debt) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"TMDX total debt & leases sum mismatch: {conv} + {other} + {leases} = {calc_total}, reported {total_debt}"
                        )

                    # Standard Net-Debt Convention: net_debt = included_debt - cash
                    calc_net_excl = round(funded_excl - gross_cash, 3)
                    if abs(calc_net_excl - net_excl) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"TMDX net debt excl leases arithmetic mismatch: {funded_excl} - {gross_cash} = {calc_net_excl}, reported {net_excl}"
                        )

                    calc_net_incl = round(total_debt - gross_cash, 3)
                    if abs(calc_net_incl - net_incl) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"TMDX net debt incl leases arithmetic mismatch: {total_debt} - {gross_cash} = {calc_net_incl}, reported {net_incl}"
                        )

                    # Sign convention enforcement
                    if total_debt > gross_cash and net_incl <= 0:
                        results["passed"] = False
                        results["errors"].append(
                            f"Sign reversal error: TMDX included debt ({total_debt}) exceeds cash ({gross_cash}), so net debt must be positive, but got {net_incl}"
                        )
                    if gross_cash > total_debt and net_incl >= 0:
                        results["passed"] = False
                        results["errors"].append(
                            f"Sign reversal error: TMDX cash ({gross_cash}) exceeds included debt ({total_debt}), so net debt must be negative (net cash), but got {net_incl}"
                        )

        elif symbol == "ETN":
            bridge = downside.get("debt_bridge_fy2025")
            if not bridge or not isinstance(bridge, dict):
                results["passed"] = False
                results["errors"].append("ETN requires explicit debt_bridge_fy2025 object")
            else:
                total_debt = bridge.get("total_debt_usd_m")
                cash_st = bridge.get("cash_and_short_term_investments_usd_m")
                net_debt = bridge.get("net_debt_usd_m")

                if total_debt is None or cash_st is None or net_debt is None:
                    results["passed"] = False
                    results["errors"].append("ETN debt bridge missing required debt or cash fields")
                else:
                    fixture_etn = CYCLE_2_PRIMARY_SOURCE_FIXTURES["ETN"]["measurement_periods"]["annual_fy2025"]["reported_components"]
                    if abs(total_debt - fixture_etn["total_debt"]) > 0.05:
                        results["passed"] = False
                        results["errors"].append(f"ETN total debt mismatch: expected {fixture_etn['total_debt']}, got {total_debt}")
                    if abs(cash_st - fixture_etn["cash_and_short_term_investments"]) > 0.05:
                        results["passed"] = False
                        results["errors"].append(f"ETN cash/ST investments mismatch: expected {fixture_etn['cash_and_short_term_investments']}, got {cash_st}")

                    calc_net = round(total_debt - cash_st, 3)
                    if abs(calc_net - net_debt) > 0.05:
                        results["passed"] = False
                        results["errors"].append(
                            f"ETN net debt arithmetic mismatch: {total_debt} - {cash_st} = {calc_net}, reported {net_debt}"
                        )
                    if total_debt > cash_st and net_debt <= 0:
                        results["passed"] = False
                        results["errors"].append(
                            f"Sign reversal error: ETN total debt ({total_debt}) exceeds cash ({cash_st}), so net debt must be positive, but got {net_debt}"
                        )

    # 5. Source-Linked Reconciliation with Markdown Narrative (only run on repo root files)
    records_md_path = repo_root / "docs" / "research" / "discovery_cycle_2" / "DISCOVERY_OPPORTUNITY_RECORDS.md"
    if records_md_path.exists() and ledger_data is None:
        records_text = records_md_path.read_text(encoding="utf-8")
        if "192.8" not in records_text or ("133.5" not in records_text and "133.6" not in records_text):
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect reconciled TMDX CFO ($192.8M) or FCF ($133.6M)")
        if "0001551182" not in records_text:
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect reconciled ETN CIK (0001551182)")
        if "4,472" not in records_text or "3,553" not in records_text:
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect reconciled ETN CFO ($4,472M) or FCF ($3,553M)")
        if "6,713" not in records_text:
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect ETN total segment operating profit ($6,713M)")
        if "+$394.981M" not in records_text and "+$395.0M" not in records_text:
            results["passed"] = False
            results["errors"].append("DISCOVERY_OPPORTUNITY_RECORDS.md does not reflect positive net debt for TMDX under the standard convention")

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
