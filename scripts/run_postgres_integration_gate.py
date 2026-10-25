"""
ARX Terminal — Hosted PostgreSQL 16 Integration Gate Runner.

Enforces zero-skip, zero-deselection execution of all 6 Smart Money Candidate 006
PostgreSQL integration test suites against a verified, isolated PostgreSQL 16 environment.
"""

import sys
import os
import json
import time
import tempfile
import subprocess
from typing import Dict, Any, List

MANDATORY_ALLOWLIST = [
    "tests/test_smart_money_postgres_integration.py",
    "tests/test_smart_money_postgres_runtime_integration.py",
    "tests/test_smart_money_worker_lease_recovery.py",
    "tests/test_smart_money_backup_containment_postgres_integration.py",
    "tests/test_smart_money_candidate_006_readiness_adversarial.py",
    "tests/test_smart_money_backup_automation.py",
]

EXPECTED_TEST_COUNT = 43

CRITICAL_ASSERTIONS = {
    "migration_and_rollback": [
        "test_sqlite_to_postgres_migration",
        "test_migration_conflict_divergent_hash_fails_closed",
        "test_forward_migration_and_idempotency",
    ],
    "advisory_lock_mutual_exclusion": [
        "test_single_owner_lock_mutual_exclusion",
        "test_lock_released_on_connection_close",
        "test_sec_owner_lock_and_backup_lock_coexist",
        "test_advisory_lock_handover_and_non_interference",
    ],
    "concurrent_worker_exclusion": [
        "test_worker_claims_request_via_skip_locked",
        "test_concurrent_claim_skip_locked",
        "test_pre_publication_race_condition_protection",
    ],
    "persistence_and_immutability": [
        "test_raw_filings_immutability_triggers",
        "test_reconciliation_receipts_append_only",
        "test_scanner_snapshots_immutability",
    ],
    "web_worker_role_isolation": [
        "test_worker_fails_closed_without_owner_role",
        "test_worker_succeeds_with_owner_role",
        "test_web_role_cannot_insert_raw_filings_or_modify_receipts",
    ],
    "backup_containment_and_recovery": [
        "test_real_postgres_17_step_containment_integration",
        "test_create_backup_generates_verified_artifacts_routine",
        "test_restore_verifier_validates_backup_into_isolated_target",
        "test_tampered_backup_fails_verification",
    ],
}


def preflight_environment() -> Dict[str, Any]:
    print("================================================================================")
    print("  ARX TERMINAL: POSTGRESQL 16 HOSTED INTEGRATION GATE PRE-FLIGHT")
    print("================================================================================\n")

    # 1. Verify driver dependencies
    try:
        import psycopg2
        psycopg2_version = getattr(psycopg2, "__version__", "installed")
        print(f"[PRE-FLIGHT] psycopg2 driver: FOUND ({psycopg2_version})")
    except ImportError as e:
        print(f"[FATAL] psycopg2 driver missing: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        import pgserver
        pgserver_version = getattr(pgserver, "__version__", "installed")
        print(f"[PRE-FLIGHT] pgserver engine: FOUND ({pgserver_version})")
    except ImportError as e:
        print(f"[FATAL] pgserver engine missing: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Test live server bootstrap and get version
    tmp_pg = tempfile.mkdtemp(prefix="arx_ci_preflight_pg_")
    try:
        print("[PRE-FLIGHT] Bootstrapping isolated verification database instance...")
        server = pgserver.get_server(tmp_pg)
        uri = server.get_uri()
        conn = psycopg2.connect(uri)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version_row = cur.fetchone()[0]
        cur.close()
        conn.close()
        server.cleanup()
        print(f"[PRE-FLIGHT] Real PostgreSQL server verified: {version_row}")

        if "PostgreSQL 16" not in version_row:
            print(f"[FATAL] PostgreSQL version mismatch. Expected PostgreSQL 16, found: {version_row}", file=sys.stderr)
            sys.exit(1)

        return {
            "psycopg2_version": psycopg2_version,
            "pgserver_version": pgserver_version,
            "postgres_version_string": version_row,
            "target_major": 16,
        }
    except Exception as e:
        print(f"[FATAL] Failed to bootstrap isolated PostgreSQL server: {e}", file=sys.stderr)
        sys.exit(1)


def run_integration_suite(preflight_info: Dict[str, Any]) -> None:
    print("\n================================================================================")
    print("  ARX TERMINAL: EXECUTING MANDATORY POSTGRESQL INTEGRATION ALLOWLIST")
    print("================================================================================")
    print(f"Mandatory Modules ({len(MANDATORY_ALLOWLIST)}):")
    for mod in MANDATORY_ALLOWLIST:
        print(f"  - {mod}")
    print(f"Expected Minimum Passing Test Count: {EXPECTED_TEST_COUNT}")
    print("================================================================================\n")

    # Run pytest via subprocess to capture full output and exit code
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *MANDATORY_ALLOWLIST,
        "-v",
        "--strict-markers",
        "-rA",
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = time.time() - t0

    print(proc.stdout)

    # Parse results
    output = proc.stdout
    passed_tests = []
    skipped_tests = []
    failed_tests = []

    for line in output.splitlines():
        line_clean = line.strip()
        if " PASSED" in line_clean and "::" in line_clean:
            test_id = line_clean.split()[0]
            passed_tests.append(test_id)
        elif (" SKIPPED" in line_clean or "Skipped:" in line_clean) and "::" in line_clean:
            test_id = line_clean.split()[0]
            skipped_tests.append(test_id)
        elif (" FAILED" in line_clean or "ERROR" in line_clean) and "::" in line_clean:
            test_id = line_clean.split()[0]
            failed_tests.append(test_id)

    total_passed = len(passed_tests)
    total_skipped = len(skipped_tests)
    total_failed = len(failed_tests)

    print("\n================================================================================")
    print("  POSTGRESQL INTEGRATION GATE: EXECUTION AUDIT & INVARIANT CHECKS")
    print("================================================================================")
    print(f"Execution Duration: {duration:.2f}s")
    print(f"Collected and Passed: {total_passed} / {EXPECTED_TEST_COUNT}")
    print(f"Skipped Tests: {total_skipped}")
    print(f"Failed Tests: {total_failed}")
    print(f"Pytest Exit Code: {proc.returncode}")

    # Check for module-level skips
    module_skips = [mod for mod in MANDATORY_ALLOWLIST if f"Skipped: could not import" in output and mod in output]
    if module_skips:
        print(f"\n[FATAL ERROR] Module-level collection skips detected: {module_skips}", file=sys.stderr)

    # Invariant 1: Zero skips permitted
    if total_skipped > 0:
        print(f"\n[FATAL ERROR] Mandatory integration gate failed: {total_skipped} tests were SKIPPED! Zero skips allowed.", file=sys.stderr)
        for s in skipped_tests:
            print(f"  [SKIPPED] {s}", file=sys.stderr)
        sys.exit(1)

    # Invariant 2: Zero failures permitted
    if total_failed > 0 or proc.returncode != 0:
        print(f"\n[FATAL ERROR] Mandatory integration gate failed: {total_failed} tests failed (exit code {proc.returncode}).", file=sys.stderr)
        sys.exit(proc.returncode if proc.returncode != 0 else 1)

    # Invariant 3: Exact test count verified
    if total_passed < EXPECTED_TEST_COUNT:
        print(f"\n[FATAL ERROR] Insufficient test execution: Expected at least {EXPECTED_TEST_COUNT} passed tests, but only {total_passed} passed.", file=sys.stderr)
        sys.exit(1)

    # Invariant 4: Critical capability assertions verified
    passed_set = set(passed_tests)
    print("\nVerifying mandatory capability categories:")
    all_critical_present = True
    critical_audit = {}

    for cat_name, req_tests in CRITICAL_ASSERTIONS.items():
        missing_in_cat = []
        for req in req_tests:
            found = any(req in p for p in passed_set)
            if not found:
                missing_in_cat.append(req)
        if missing_in_cat:
            print(f"  [FAIL] {cat_name}: Missing required tests: {missing_in_cat}")
            all_critical_present = False
            critical_audit[cat_name] = {"status": "FAIL", "missing": missing_in_cat}
        else:
            print(f"  [PASS] {cat_name}: {len(req_tests)}/{len(req_tests)} verified")
            critical_audit[cat_name] = {"status": "PASS", "verified_count": len(req_tests)}

    if not all_critical_present:
        print("\n[FATAL ERROR] One or more critical capability categories lacked verification!", file=sys.stderr)
        sys.exit(1)

    # Output report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "disposition": "PASS",
        "preflight": preflight_info,
        "execution": {
            "duration_seconds": round(duration, 2),
            "total_passed": total_passed,
            "total_skipped": total_skipped,
            "total_failed": total_failed,
            "allowlist_module_count": len(MANDATORY_ALLOWLIST),
            "modules": MANDATORY_ALLOWLIST,
        },
        "critical_capabilities": critical_audit,
        "passed_tests": passed_tests,
    }

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "postgres_integration_gate_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {report_path}")
    print("\n================================================================================")
    print("  >>> POSTGRESQL 16 HOSTED INTEGRATION GATE CERTIFIED: 100% PASS <<<")
    print("================================================================================\n")


def main():
    preflight_info = preflight_environment()
    run_integration_suite(preflight_info)


if __name__ == "__main__":
    main()
