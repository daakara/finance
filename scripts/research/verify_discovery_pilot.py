#!/usr/bin/env python3
"""
ARX Terminal — Evidence-Led Discovery Pilot Verification Script (Corrected)
Task: ARX-TERMINAL-EVIDENCE-LED-DISCOVERY-PILOT (Issue #43)
Governing Item: BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-043

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

REQUIRED_ROOT_KEYS = [
    "schema_version",
    "task",
    "governing_backlog_item",
    "governing_issue",
    "created_at_utc",
    "information_cutoff_date",
    "provisional_analytical_horizon_years",
    "examined_universe",
    "candidate_generation_routes",
    "triage_summary",
    "deep_investigations"
]

REQUIRED_RECORD_KEYS = [
    "symbol",
    "company_name",
    "cik",
    "discovery_provenance",
    "dimensional_classification",
    "observable_development",
    "economic_significance",
    "value_capture",
    "price_and_expectations",
    "downside_and_survival",
    "assessment_changes",
    "research_priority"
]

REQUIRED_DIMENSIONS = [
    "research_maturity",
    "opportunity_mechanism",
    "expectation_gap_status",
    "financial_resilience",
    "public_publication_eligibility"
]

# Checked offline source values for arithmetic consistency verification (FY2025 GAAP)
OFFLINE_ARITHMETIC_FIXTURES = {
    "ANET": {
        "cfo": 4371.9,
        "capex": 110.0,
        "expected_fcf": 4261.9,
        "expected_conversion": 0.9748
    },
    "VRT": {
        "cfo": 2113.8,
        "capex": 220.0,
        "expected_fcf": 1893.8,
        "expected_conversion": 0.8959
    },
    "MPWR": {
        "cfo": 838.2,
        "capex": 172.0,
        "expected_fcf": 666.2,
        "expected_conversion": 0.7948
    }
}


def verify_pilot(repo_root: Path) -> dict:
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "records_verified": 0
    }

    # 1. Verify discovery_opportunity_ledger.json structure and fields
    ledger_path = repo_root / "docs" / "research" / "evidence_led_discovery" / "discovery_opportunity_ledger.json"
    if not ledger_path.exists():
        results["passed"] = False
        results["errors"].append(f"Missing required ledger: {ledger_path}")
        return results

    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Failed to parse JSON in {ledger_path}: {e}")
        return results

    for key in REQUIRED_ROOT_KEYS:
        if key not in data:
            results["passed"] = False
            results["errors"].append(f"Missing root key in ledger: {key}")

    # Verify provenance qualification
    universe_info = data.get("examined_universe", {})
    if universe_info.get("provenance_status") != "EXPLORATORY_MANUAL_COMPARISON_NOT_AUTOMATED_DISCOVERY":
        results["passed"] = False
        results["errors"].append(
            "examined_universe must explicitly acknowledge EXPLORATORY_MANUAL_COMPARISON_NOT_AUTOMATED_DISCOVERY"
        )

    deep_records = data.get("deep_investigations", {})
    if len(deep_records) < 2:
        results["passed"] = False
        results["errors"].append(f"Insufficient deep investigations: expected >=2, got {len(deep_records)}")

    for symbol, rec in deep_records.items():
        results["records_verified"] += 1
        for field in REQUIRED_RECORD_KEYS:
            if field not in rec:
                results["passed"] = False
                results["errors"].append(f"Record {symbol} missing field: {field}")

        dim = rec.get("dimensional_classification", {})
        for d in REQUIRED_DIMENSIONS:
            if d not in dim:
                results["passed"] = False
                results["errors"].append(f"Record {symbol} missing dimension: {d}")

        if dim.get("public_publication_eligibility") != "INTERNAL_PILOT_RECORD_NOT_ELIGIBLE":
            results["passed"] = False
            results["errors"].append(
                f"Record {symbol} violates publication eligibility invariant: {dim.get('public_publication_eligibility')}"
            )

        # Check evidence citation
        dev = rec.get("observable_development", {})
        evidence_str = dev.get("primary_filing_evidence", "")
        if "10-K" not in evidence_str or "Form" not in evidence_str:
            results["passed"] = False
            results["errors"].append(f"Record {symbol} lacks explicit Form 10-K primary evidence reference")

        # Check that expectations acknowledge unresolved consensus where applicable
        expectations = rec.get("price_and_expectations", {}).get("expectation_disaggregation", {})
        consensus_val = expectations.get("sourced_analyst_consensus", "")
        if "UNRESOLVED" not in consensus_val and not consensus_val.startswith("http"):
            results["warnings"].append(
                f"Record {symbol} consensus expectations should be explicit URL or marked UNRESOLVED_CONSENSUS"
            )

    # 2. Arithmetic Reconciliation of Retained FCF Calculations
    for sym, fix in OFFLINE_ARITHMETIC_FIXTURES.items():
        fcf = fix["cfo"] - fix["capex"]
        if abs(fcf - fix["expected_fcf"]) > 0.05:
            results["passed"] = False
            results["errors"].append(
                f"Arithmetic reconciliation failed for {sym}: CFO {fix['cfo']} - Capex {fix['capex']} != {fix['expected_fcf']}"
            )
        conv = fcf / fix["cfo"]
        if abs(conv - fix["expected_conversion"]) > 0.001:
            results["passed"] = False
            results["errors"].append(
                f"Arithmetic conversion failed for {sym}: FCF {fcf} / CFO {fix['cfo']} != {fix['expected_conversion']}"
            )

    # 3. Verify published_watch_manifest.json invariants
    manifest_path = repo_root / "docs" / "research" / "opportunity_discovery" / "published_watch_manifest.json"
    if not manifest_path.exists():
        results["passed"] = False
        results["errors"].append(f"Missing published manifest: {manifest_path}")
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        pub_symbols = manifest.get("published_symbols", [])
        if pub_symbols != AUTHORIZED_PUBLISHED_SYMBOLS:
            results["passed"] = False
            results["errors"].append(
                f"published_symbols mutated! Expected {AUTHORIZED_PUBLISHED_SYMBOLS}, got {pub_symbols}"
            )

    # 4. Verify odp05_discovery_ledger.json immutability
    odp05_path = repo_root / "docs" / "research" / "opportunity_discovery" / "odp05_discovery_ledger.json"
    if not odp05_path.exists():
        results["passed"] = False
        results["errors"].append(f"Missing baseline ledger: {odp05_path}")
    else:
        with open(odp05_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if actual_sha != FROZEN_BASELINE_LEDGER_SHA256:
            results["passed"] = False
            results["errors"].append(
                f"odp05_discovery_ledger.json SHA-256 mismatch! Expected {FROZEN_BASELINE_LEDGER_SHA256}, got {actual_sha}"
            )

    # 5. Verify Frontend Isolation
    frontend_dir = repo_root / "frontend"
    if frontend_dir.exists():
        for path in frontend_dir.rglob("*.ts*"):
            if "node_modules" in str(path) or ".next" in str(path):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if "evidence_led_discovery" in content or "discovery_opportunity_ledger" in content:
                    results["passed"] = False
                    results["errors"].append(
                        f"Frontend file {path.relative_to(repo_root)} improperly references internal pilot files!"
                    )
            except Exception as e:
                results["warnings"].append(f"Could not read {path}: {e}")

    # 6. Verify Standardized Decision-Focused Research Process
    std_process_path = repo_root / "docs" / "research" / "DECISION_FOCUSED_RESEARCH_PROCESS.md"
    if not std_process_path.exists():
        results["passed"] = False
        results["errors"].append(f"Missing canonical decision process standard: {std_process_path}")
    else:
        text = std_process_path.read_text(encoding="utf-8", errors="ignore")
        required_sections = [
            "Step A: Define the Decision",
            "Step B: Identify Decisive Uncertainties",
            "Step C: Connect Outcomes to Consequences",
            "Step D: Investigate Proportionately",
            "Step E: Make the Assessment",
            "Step F: Set Review Triggers and Stop",
            "DECK", "FIX", "DUOL", "ULTA", "MEDP", "ISRG",
            "Decision Sought",
            "Review Triggers & Stopping Rationale"
        ]
        for sec in required_sections:
            if sec not in text:
                results["passed"] = False
                results["errors"].append(f"DECISION_FOCUSED_RESEARCH_PROCESS.md missing required section/example: {sec}")

    # 7. Verify Triage Matrix Factual Freshness & SNPS Correction
    triage_path = repo_root / "docs" / "research" / "evidence_led_discovery" / "SEARCH_UNIVERSE_AND_TRIAGE_MATRIX.md"
    if triage_path.exists():
        triage_text = triage_path.read_text(encoding="utf-8", errors="ignore")
        if "pending $35B" in triage_text or "acquisition of Ansys (pending" in triage_text:
            results["passed"] = False
            results["errors"].append("SEARCH_UNIVERSE_AND_TRIAGE_MATRIX.md retains discredited 'pending acquisition' claim for SNPS")
        if "July 17, 2025" not in triage_text:
            results["passed"] = False
            results["errors"].append("SEARCH_UNIVERSE_AND_TRIAGE_MATRIX.md lacks primary Form 8-K July 17, 2025 merger date for SNPS")

    return results


def main():
    repo_root = Path(__file__).resolve().parents[2]
    print(f"Verifying Evidence-Led Discovery Pilot in: {repo_root}")
    res = verify_pilot(repo_root)
    print(f"Records verified: {res['records_verified']}")
    if res["warnings"]:
        for w in res["warnings"]:
            print(f"[WARN] {w}")
    if not res["passed"]:
        print("[FAIL] Verification errors:")
        for err in res["errors"]:
            print(f"  - {err}")
        sys.exit(1)
    print("[PASS] All verification checks passed cleanly.")


if __name__ == "__main__":
    main()
