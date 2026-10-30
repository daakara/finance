#!/usr/bin/env python3
"""
ARX Release Qualification - RQ-11 Adversarial Falsification Suite
Directives: ARX — Final RQ-11 Production Rollback Authority Closure & 074-A4.1

Adversarially tests Check RQ-11 under negative controls:
1. PC-00: Baseline Positive Control (Active Railway & Cloudflare recovery targets, 0-DDL, valid anchors) -> RC 0
2. NC-01: Stale Railway Rollback Target Rejection (Historical d1f8cf37 marked REMOVED) -> RC 1 (STALE_ROLLBACK_TARGET_REJECTED)
3. NC-02: Invalid Railway Deployment UUID Format -> RC 1 (INVALID_ROLLBACK_TARGET)
4. NC-03: Invalid Cloudflare Deployment UUID Format -> RC 1 (INVALID_ROLLBACK_TARGET)
5. NC-04: Unrecognized Railway Target UUID -> RC 1 (UNRECOGNIZED_ROLLBACK_TARGET)
6. NC-05: Unrecognized Cloudflare Target UUID -> RC 1 (UNRECOGNIZED_ROLLBACK_TARGET)
7. NC-06: Simulated Destructive Database Schema DDL Mutation -> RC 1 (DDL_MIGRATIONS_DETECTED)
8. NC-07: Missing/Invalid Git Base Anchor Object -> RC 1 (MISSING_GIT_ANCHOR)

Preserves canonical qualification evidence via ARX_RECEIPTS_DIR_OVERRIDE.
"""

import sys, os, subprocess, json, tempfile, shutil, time, datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEMP_RECEIPTS_DIR = None

def run_rq11(args_list, env=None):
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "release", "rq11_rollback_readiness.py")] + args_list
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    if TEMP_RECEIPTS_DIR and "ARX_RECEIPTS_DIR_OVERRIDE" not in current_env:
        current_env["ARX_RECEIPTS_DIR_OVERRIDE"] = TEMP_RECEIPTS_DIR
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=current_env)
    return res.returncode, res.stdout, res.stderr

def main():
    print("================================================================================")
    print("ARX RELEASE QUALIFICATION - RQ-11 ADVERSARIAL FALSIFICATION SUITE (8 CONTROLS)")
    print("Directive: ARX — Final RQ-11 Production Rollback Authority Closure")
    print("================================================================================")

    results = []
    global TEMP_RECEIPTS_DIR
    temp_dir = tempfile.mkdtemp(prefix="arx_rq11_adversarial_")
    TEMP_RECEIPTS_DIR = os.path.join(temp_dir, "receipts")
    os.makedirs(TEMP_RECEIPTS_DIR, exist_ok=True)

    try:
        # PC-00: Baseline Positive Control
        rc, out, err = run_rq11([])
        passed = (rc == 0 and "=== [RQ-11] RESULT: PASS" in out)
        results.append({
            "control": "PC-00: Baseline Positive Control",
            "expected_rc": 0,
            "actual_rc": rc,
            "passed": passed,
            "details": "All 4 assertions pass on authoritative candidate and recovery anchors"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] PC-00: Baseline Positive Control (rc={rc})")

        # NC-01: Stale Railway target d1f8cf37 (marked REMOVED in deployment register)
        rc, out, err = run_rq11(["--assert-railway-target", "d1f8cf37-7592-4ad5-b986-4bbc9b4ed042"])
        passed = (rc == 1 and "STALE_ROLLBACK_TARGET_REJECTED" in out)
        results.append({
            "control": "NC-01: Stale Railway target rejection (d1f8cf37)",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "details": "Fails closed with STALE_ROLLBACK_TARGET_REJECTED"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-01: Stale Railway target rejection (rc={rc})")

        # NC-02: Invalid Railway deployment UUID format
        rc, out, err = run_rq11(["--assert-railway-target", "not-a-valid-uuid-string"])
        passed = (rc == 1 and "INVALID_ROLLBACK_TARGET" in out and "Railway" in out)
        results.append({
            "control": "NC-02: Invalid Railway deployment UUID format",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "details": "Fails closed with INVALID_ROLLBACK_TARGET on malformed UUID"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-02: Invalid Railway UUID format (rc={rc})")

        # NC-03: Invalid Cloudflare deployment UUID format
        rc, out, err = run_rq11(["--assert-cloudflare-target", "cloudflare-invalid-uuid-12345"])
        passed = (rc == 1 and "INVALID_ROLLBACK_TARGET" in out and "Cloudflare" in out)
        results.append({
            "control": "NC-03: Invalid Cloudflare deployment UUID format",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "details": "Fails closed with INVALID_ROLLBACK_TARGET on malformed Cloudflare UUID"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-03: Invalid Cloudflare UUID format (rc={rc})")

        # NC-04: Unrecognized Railway target UUID (valid UUID, but unapproved/drifted)
        rc, out, err = run_rq11(["--assert-railway-target", "00000000-0000-0000-0000-000000000000"])
        passed = (rc == 1 and "UNRECOGNIZED_ROLLBACK_TARGET" in out and "Railway" in out)
        results.append({
            "control": "NC-04: Unrecognized Railway target UUID",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "details": "Fails closed with UNRECOGNIZED_ROLLBACK_TARGET"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-04: Unrecognized Railway target UUID (rc={rc})")

        # NC-05: Unrecognized Cloudflare target UUID (valid UUID, but unapproved/drifted)
        rc, out, err = run_rq11(["--assert-cloudflare-target", "11111111-1111-1111-1111-111111111111"])
        passed = (rc == 1 and "UNRECOGNIZED_ROLLBACK_TARGET" in out and "Cloudflare" in out)
        results.append({
            "control": "NC-05: Unrecognized Cloudflare target UUID",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "details": "Fails closed with UNRECOGNIZED_ROLLBACK_TARGET"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-05: Unrecognized Cloudflare target UUID (rc={rc})")

        # NC-06: Non-zero database schema DDL mutation
        rc, out, err = run_rq11(["--simulate-ddl"])
        passed = (rc == 1 and "DDL_MIGRATIONS_DETECTED" in out)
        results.append({
            "control": "NC-06: Non-zero database schema DDL mutation",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "details": "Fails closed with DDL_MIGRATIONS_DETECTED"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-06: Non-zero schema DDL mutation (rc={rc})")

        # NC-07: Missing/invalid Git base anchor object
        rc, out, err = run_rq11(["--base", "0000000000000000000000000000000000000000"])
        passed = (rc == 1 and "MISSING_GIT_ANCHOR" in out)
        results.append({
            "control": "NC-07: Missing/invalid Git base anchor object",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "details": "Fails closed with MISSING_GIT_ANCHOR when base object not in object DB"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-07: Missing Git base anchor object (rc={rc})")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("--------------------------------------------------------------------------------")
    total_passed = sum(1 for r in results if r["passed"])
    total_tests = len(results)
    print(f"SUMMARY: {total_passed}/{total_tests} controls passed.")

    # Write adversarial test receipt
    receipt = {
        "suite": "RQ-11 Adversarial Falsification Suite",
        "directive": "ARX — Final RQ-11 Production Rollback Authority Closure",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_controls": total_tests,
        "passed_controls": total_passed,
        "all_passed": (total_passed == total_tests),
        "results": results
    }

    receipt_path = os.path.join(PROJECT_ROOT, "scripts", "release", "receipts", "rq11_adversarial_results.json")
    os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"Adversarial receipt written to: {receipt_path}")

    if total_passed == total_tests:
        print("=== RQ-11 ADVERSARIAL FALSIFICATION SUITE: ALL CONTROLS PASSED (FAIL-CLOSED VERIFIED) ===")
        sys.exit(0)
    else:
        print("=== RQ-11 ADVERSARIAL FALSIFICATION SUITE: FAILURES DETECTED ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
