#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-06: Governance & Backlog Invariants
Validates canonical backlog integrity, JSON schema, initiative dependencies,
and strictly enforces that Sprint 5 remains UNAUTHORIZED (Rule P5-010).
Emits a verified machine-readable stage receipt.
"""

import sys, os, subprocess, time, datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, get_git_commit

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    print("=== [RQ-06] GOVERNANCE & BACKLOG INVARIANTS ===")

    # 1. Backlog integrity check
    validator_script = os.path.join(PROJECT_ROOT, "tools", "governance", "validate_backlog_integrity.py")
    res_val = subprocess.run([sys.executable, validator_script], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
    print(res_val.stdout)
    val_passed = (res_val.returncode == 0)
    assertions.append({
        "assertion": "Backlog schema and dependency integrity validated",
        "observed": {"exit_code": res_val.returncode},
        "passed": val_passed
    })
    if not val_passed:
        errors.append(f"Backlog validator failed with exit code {res_val.returncode}")

    # 2. Assert Sprint 5 Lock (Rule P5-010)
    sprint_check = os.path.join(PROJECT_ROOT, "tools", "governance", "check_active_sprint.py")
    res_s5 = subprocess.run([sys.executable, sprint_check, "--assert-sprint", "5"], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Rule P5-010: Asserting sprint 5 MUST fail (exit code != 0) because sprint 5 is not authorized
    s5_locked = (res_s5.returncode != 0)
    assertions.append({
        "assertion": "Sprint 5 remains UNAUTHORIZED (Rule P5-010 enforced)",
        "observed": {"exit_code": res_s5.returncode, "sprint_5_blocked": s5_locked},
        "passed": s5_locked
    })
    if not s5_locked:
        errors.append("Rule P5-010 breach: Sprint 5 unexpectedly reported as authorized!")

    duration = time.time() - t0
    status = "PASS" if len(errors) == 0 else "FAIL"
    candidate_sha = get_git_commit("HEAD")

    evidence = [
        "tools/governance/validate_backlog_integrity.py",
        "tools/governance/check_active_sprint.py",
        "docs/planning/canonical_backlog.json"
    ]

    receipt = create_stage_receipt(
        control_id="RQ-06",
        name="Governance & Backlog Invariants",
        status=status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=candidate_sha,
        target_branch="release/production",
        assertions=assertions,
        evidence_paths=evidence,
        errors=errors,
        metadata={"sprint_5_rule": "P5-010", "backlog_validator_exit": res_val.returncode}
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        print("=== [RQ-06] RESULT: PASS (Backlog valid, Sprint 5 locked) ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-06] RESULT: FAIL (Errors: {errors}) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()