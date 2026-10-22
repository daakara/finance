#!/usr/bin/env python3
"""
ARX Terminal — Evidence-Led Discovery Cycle 2 Automated Verification Engine.

Governing Standard: ARX-DECISION-RESEARCH-PROCESS-V1 & ARX Closure Model v2.

Methodological Disclaimer on Verification Scope:
- Schema checks establish internal structural compliance of JSON ledgers; they do not establish economic validity.
- Date and accession format checks verify syntactic presence of required disclosures; they do not independently confirm external factual accuracy.
- Offline calculation checks verify arithmetic consistency between reported components and derived metrics.
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
        "cfo": 84.2,
        "capex": 58.0,
        "expected_fcf": 26.2,
        "expected_conversion": 0.31116389548693587
    },
    "ETN": {
        "cfo": 4500.0,
        "capex": 810.0,
        "expected_fcf": 3690.0,
        "expected_conversion": 0.82
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
    if len(examined_universe) != 14:
        results["passed"] = False
        results["errors"].append(f"Examined universe must contain exactly 14 symbols, got {len(examined_universe)}")

    deep_recs = ledger.get("deep_investigations", {})
    if set(deep_recs.keys()) != {"TMDX", "ETN"}:
        results["passed"] = False
        results["errors"].append(f"Deep investigations must contain exactly TMDX and ETN, got {list(deep_recs.keys())}")

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

        # Check evidence citation and canonical SEC Archives URL
        dev = rec.get("observable_development", {})
        evidence_str = dev.get("primary_filing_evidence", "")
        if "10-K" not in evidence_str or "Form" not in evidence_str:
            results["passed"] = False
            results["errors"].append(f"Record {symbol} lacks explicit Form 10-K primary evidence reference")
        if "https://www.sec.gov/Archives/edgar/data/" not in evidence_str:
            results["passed"] = False
            results["errors"].append(f"Record {symbol} primary evidence must use canonical SEC Archives URL")

        # Check that expectations acknowledge unresolved consensus
        expectations = rec.get("price_and_expectations", {}).get("expectation_disaggregation", {})
        consensus_status = expectations.get("sourced_analyst_consensus", "")
        if "UNRESOLVED_CONSENSUS" not in consensus_status:
            results["passed"] = False
            results["errors"].append(
                f"Record {symbol} must explicitly label consensus as UNRESOLVED_CONSENSUS when feeds are unavailable"
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
            calc_fcf = round(cfo - capex, 2)
            if abs(calc_fcf - fcf) > 0.05:
                results["passed"] = False
                results["errors"].append(
                    f"Record {symbol} FCF arithmetic mismatch: CFO({cfo}) - Capex({capex}) = {calc_fcf}, reported {fcf}"
                )

            calc_conv = round((fcf / cfo) * 100, 1)
            if abs(calc_conv - round(fcf_conv, 1)) > 0.2:
                results["passed"] = False
                results["errors"].append(
                    f"Record {symbol} FCF conversion mismatch: FCF({fcf}) / CFO({cfo}) = {calc_conv}%, reported {fcf_conv}%"
                )

    # 5. Verify Frontend Decoupling
    frontend_dir = repo_root / "frontend"
    if frontend_dir.exists():
        for file in frontend_dir.rglob("*.ts*"):
            if file.is_file():
                content = file.read_text(encoding="utf-8", errors="ignore")
                if "discovery_cycle_2" in content:
                    results["passed"] = False
                    results["errors"].append(f"Frontend file {file.name} illegally imports internal discovery cycle 2")

    # 6. Verify Required Markdown Documentation
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
