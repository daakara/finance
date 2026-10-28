#!/usr/bin/env python3
"""
ARX Release Qualification - Master Local Test Suite Runner
Executes all 12 Release Qualification checks (RQ-01 through RQ-12) sequentially.
Guarantees stage receipts are emitted for all controls and aggregates final verdict via RQ-12.
Fails closed if any stage fails or is blocked.
"""

import sys, os, subprocess, time, json, shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECEIPTS_DIR = os.path.join(os.path.dirname(__file__), "receipts")

CHECKS = [
    ("RQ-01", "Candidate Identity", [sys.executable, "scripts/release/rq01_candidate_identity.py", "--allow-main-target"]),
    ("RQ-02", "Release Diff Integrity", [sys.executable, "scripts/release/rq02_diff_integrity.py"]),
    ("RQ-03", "Backend Regression", [sys.executable, "scripts/release/rq03_backend_regression.py"]),
    ("RQ-04", "Frontend Regression", [sys.executable, "scripts/release/rq04_frontend_regression.py"]),
    ("RQ-05", "Cross-Engine Invariants", [sys.executable, "scripts/release/rq05_cross_engine_invariants.py"]),
    ("RQ-06", "Governance Invariants", [sys.executable, "scripts/release/rq06_governance_invariants.py"]),
    ("RQ-07", "Deployment Safety", [sys.executable, "scripts/release/rq07_deployment_safety.py"]),
    ("RQ-08", "Artifact Provenance", [sys.executable, "scripts/release/rq08_artifact_provenance.py"]),
    ("RQ-09", "Compatibility Suite", [sys.executable, "scripts/release/rq09_compatibility_suite.py"]),
    ("RQ-10", "Configuration Integrity", [sys.executable, "scripts/release/rq10_config_integrity.py"]),
    ("RQ-11", "Rollback Readiness", [sys.executable, "scripts/release/rq11_rollback_readiness.py"]),
    ("RQ-12", "Qualification Summary", [sys.executable, "scripts/release/rq12_qualification_summary.py"])
]

def main():
    print("===============================================================")
    print("   ARX TERMINAL - FULL RELEASE QUALIFICATION CI SUITE (RQ-01..12)")
    print("===============================================================")

    # Clean previous run receipts to prevent cross-run contamination
    if os.path.exists(RECEIPTS_DIR):
        for f in os.listdir(RECEIPTS_DIR):
            fpath = os.path.join(RECEIPTS_DIR, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    
    start_time = time.time()
    results = {}
    passed = 0
    failed = 0

    for code, name, cmd in CHECKS:
        print(f"\n>>> Executing [{code}] {name}...")
        t0 = time.time()
        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed = round(time.time() - t0, 2)
        if res.returncode == 0:
            print(f">>> [{code}] PASSED in {elapsed}s")
            results[code] = {"name": name, "status": "PASS", "elapsed_sec": elapsed}
            passed += 1
        else:
            print(f">>> [{code}] FAILED (exit code {res.returncode}) in {elapsed}s")
            results[code] = {"name": name, "status": "FAIL", "exit_code": res.returncode, "elapsed_sec": elapsed}
            failed += 1

    total_time = round(time.time() - start_time, 2)
    print("\n===============================================================")
    print("   RELEASE QUALIFICATION MASTER RUNNER SUMMARY")
    print("===============================================================")
    print(f"Total Checks: {len(CHECKS)} | Passed: {passed} | Failed: {failed} | Duration: {total_time}s")
    for code, r in results.items():
        print(f"  [{code}] {r['name']:<35} : {r['status']}")

    summary_file = os.path.join(PROJECT_ROOT, "scripts", "release", "master_ci_run_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"total": len(CHECKS), "passed": passed, "failed": failed, "duration_sec": total_time, "results": results}, f, indent=2)

    if failed > 0:
        print("\nOVERALL VERDICT: FAIL CLOSED (Release Blocked)")
        sys.exit(1)
    else:
        print("\nOVERALL VERDICT: ALL RELEASE QUALIFICATION CHECKS PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()