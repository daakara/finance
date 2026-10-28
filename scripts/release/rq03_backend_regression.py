#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-03: Backend Regression Suite
Executes pytest regression tests for confluence scoring, sizing guards, and Kelly capping.
Emits a verified machine-readable stage receipt.
"""

import sys, os, subprocess, time, datetime, re

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

    print("=== [RQ-03] BACKEND REGRESSION SUITE ===")
    test_files = [
        "tests/test_directive_057_remediation.py",
        "tests/test_confluence_engine.py"
    ]

    cmd = [sys.executable, "-m", "pytest"] + test_files + ["-v"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = time.time() - t0
    output = res.stdout

    print(output)

    # Parse pytest results
    match = re.search(r"==+ (?:(\d+) passed)?(?:, )?(?:(\d+) failed)?(?:, )?(?:(\d+) error)? in ([\d\.]+)s ==+", output)
    passed_count = 0
    failed_count = 0
    if match:
        passed_count = int(match.group(1)) if match.group(1) else 0
        failed_count = int(match.group(2)) if match.group(2) else 0
    else:
        # Fallback count
        passed_count = len(re.findall(r"PASSED", output))
        failed_count = len(re.findall(r"FAILED", output))

    assertions.append({
        "assertion": "All backend regression tests pass",
        "observed": {"passed": passed_count, "failed": failed_count, "exit_code": res.returncode},
        "passed": (res.returncode == 0 and failed_count == 0 and passed_count >= 25)
    })

    if res.returncode != 0 or failed_count > 0:
        errors.append(f"Pytest suite failed with returncode {res.returncode}, {failed_count} failures.")
    elif passed_count < 25:
        errors.append(f"Expected at least 25 tests, only observed {passed_count}.")

    status = "PASS" if len(errors) == 0 else "FAIL"
    candidate_sha = get_git_commit("HEAD")

    receipt = create_stage_receipt(
        control_id="RQ-03",
        name="Backend Regression Suite",
        status=status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=candidate_sha,
        target_branch="release/production",
        assertions=assertions,
        evidence_paths=test_files,
        errors=errors,
        metadata={"pytest_passed": passed_count, "pytest_failed": failed_count, "pytest_duration": duration}
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        print(f"=== [RQ-03] RESULT: PASS ({passed_count} passed) ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-03] RESULT: FAIL (Errors: {errors}) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
