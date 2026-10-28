#!/usr/bin/env python3
"""
ARX Release Qualification - Negative Test Suite (Stage C Verification)
Adversarially exercises failure modes across the RQ suite to prove fail-closed enforcement:
1. Unauthorized Target Branch
2. Substituted / Fabricated Commit SHA
3. Tampered File Hash in Manifest
4. Completely Missing Manifest File
5. Non-existent Base Commit in Ancestry
6. Direct Pytest Non-Existent Target Test
7. Corrupted Backlog Syntax
"""

import sys, os, subprocess, json, tempfile, shutil

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def run_test_case(name, cmd, should_fail=True):
    print(f"\n--- Testing Negative Case: {name} ---")
    res = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if should_fail:
        if res.returncode != 0:
            print(f"  [PASS] Failed closed as expected (exit code {res.returncode}).")
            return {"case": name, "expected": "FAIL_CLOSED", "actual": f"EXIT_{res.returncode}", "verdict": "PASS"}
        else:
            print(f"  [FAIL] Unexpectedly PASSED when it should have failed closed!")
            return {"case": name, "expected": "FAIL_CLOSED", "actual": "EXIT_0", "verdict": "FAIL"}
    else:
        if res.returncode == 0:
            print(f"  [PASS] Passed as expected (exit code 0).")
            return {"case": name, "expected": "PASS", "actual": "EXIT_0", "verdict": "PASS"}
        else:
            print(f"  [FAIL] Failed unexpectedly (exit code {res.returncode})!")
            return {"case": name, "expected": "PASS", "actual": f"EXIT_{res.returncode}", "verdict": "FAIL"}

def main():
    print("===============================================================")
    print("   ARX TERMINAL - ADVERSARIAL RELEASE CI NEGATIVE MATRIX")
    print("===============================================================")

    results = []

    # Case 1: Incorrect target branch
    results.append(run_test_case(
        "NEG-01: Unauthorized Target Branch (feature-branch)",
        "python scripts/release/rq01_candidate_identity.py --target-branch feature/unauthorized"
    ))

    # Case 2: Substituted unapproved commit SHA
    results.append(run_test_case(
        "NEG-02: Substituted / Fabricated Commit SHA",
        "python scripts/release/rq01_candidate_identity.py --expected-head 0000000000000000000000000000000000000000"
    ))

    # Case 3: Tampered Release Manifest (Hash Mismatch)
    temp_dir = tempfile.mkdtemp()
    try:
        tampered_manifest = os.path.join(temp_dir, "tampered_manifest.json")
        with open(os.path.join(PROJECT_ROOT, "scripts", "release", "release_manifest.json"), "r") as f:
            data = json.load(f)
        for k in data["files"]:
            data["files"][k]["sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            break
        with open(tampered_manifest, "w") as f:
            json.dump(data, f)

        results.append(run_test_case(
            "NEG-03: Tampered File Hash in Manifest",
            f"python scripts/release/rq02_diff_integrity.py --manifest {tampered_manifest}"
        ))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Case 4: Missing Manifest File
    results.append(run_test_case(
        "NEG-04: Completely Missing Manifest File",
        "python scripts/release/rq02_diff_integrity.py --manifest nonexistent_file.json"
    ))

    # Case 5: Non-existent Base Commit in Ancestry
    results.append(run_test_case(
        "NEG-05: Non-existent Base Commit in Ancestry",
        "python scripts/release/rq01_candidate_identity.py --base deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    ))

    # Case 6: Pytest Non-Existent Target Test
    results.append(run_test_case(
        "NEG-06: Direct Pytest Non-Existent Target Test",
        "python -m pytest tests/test_nonexistent_broken_test.py"
    ))

    # Case 7: Backlog Integrity Breach Simulation (Corrupted JSON)
    temp_backlog = os.path.join(PROJECT_ROOT, "docs", "planning", "temp_corrupted_backlog.json")
    try:
        with open(temp_backlog, "w") as f:
            f.write("{invalid_json: true")
        results.append(run_test_case(
            "NEG-07: Corrupted Backlog Syntax",
            f"python tools/governance/validate_backlog_integrity.py --file {temp_backlog}"
        ))
    finally:
        if os.path.exists(temp_backlog):
            os.remove(temp_backlog)

    # Print Summary Matrix
    print("\n===============================================================")
    print("   ADVERSARIAL NEGATIVE TEST MATRIX RESULTS")
    print("===============================================================")
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    total = len(results)
    print(f"Total Negative Cases Tested: {total} | Pass (Failed Closed): {passed} | Flaws: {total - passed}")
    for r in results:
        print(f"  {r['case']:<55} Expected: {r['expected']:<12} Actual: {r['actual']:<10} [{r['verdict']}]")

    csv_path = os.path.join(PROJECT_ROOT, "scripts", "release", "negative_test_results.json")
    with open(csv_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if passed == total:
        print("\nALL NEGATIVE CASES SUCCESSFULLY FAILED CLOSED!")
        sys.exit(0)
    else:
        print("\nFAIL: One or more negative cases did not fail closed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
