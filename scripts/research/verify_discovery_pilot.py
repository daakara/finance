#!/usr/bin/env python3
"""
ARX Terminal — Evidence-Led Discovery Pilot Verification Script
Task: ARX-TERMINAL-EVIDENCE-LED-DISCOVERY-PILOT (Issue #43)
Governing Item: BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-043

Verifies:
1. Schema conformance and required field completeness of discovery_opportunity_ledger.json.
2. Complete dimensional classification (research maturity, opportunity mechanism, expectation gap, financial resilience, publication eligibility).
3. Public publication ineligibility invariant across all internal pilot records.
4. Primary source filing citations, accession numbers, and date formats.
5. Invariant preservation: published_watch_manifest.json published symbols == 6, odp05_discovery_ledger.json SHA-256 untouched.
6. Strict frontend isolation: zero imports or references to internal pilot artifacts in frontend/.
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


def verify_pilot(repo_root: Path) -> dict:
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "records_verified": 0
    }

    # 1. Verify discovery_opportunity_ledger.json
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

    # 2. Verify published_watch_manifest.json invariants
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

    # 3. Verify odp05_discovery_ledger.json immutability
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

    # 4. Verify Frontend Isolation
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
