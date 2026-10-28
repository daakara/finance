#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-04: Frontend Regression & Typecheck
Executes frontend typecheck and permanent governor sizing regression tests.
Emits a verified machine-readable stage receipt.
"""

import sys, os, subprocess, time, datetime, re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, get_git_commit

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    print("=== [RQ-04] FRONTEND REGRESSION & TYPECHECK ===")

    # 1. Typecheck
    print("Running npm run type-check...")
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    res_tc = subprocess.run(
        [npm_cmd, "run", "type-check"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    tc_passed = (res_tc.returncode == 0)
    print(res_tc.stdout)
    assertions.append({
        "assertion": "Frontend TypeScript compiler typecheck passes",
        "observed": {"exit_code": res_tc.returncode},
        "passed": tc_passed
    })
    if not tc_passed:
        errors.append(f"Frontend typecheck failed with exit code {res_tc.returncode}")

    # 2. Sizing regression tests
    print("Running npm run test:sizing...")
    res_sizing = subprocess.run(
        [npm_cmd, "run", "test:sizing"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    sizing_out = res_sizing.stdout
    print(sizing_out)

    # Robust matching regardless of UTF-8 checkmark rendering in Windows console
    passed_tests = len(re.findall(r"Test \d+ Passed", sizing_out))
    m_all = re.search(r"ALL (\d+) GOVERNOR SIZING ENGINE REGRESSION TESTS PASSED", sizing_out)
    if m_all:
        passed_tests = max(passed_tests, int(m_all.group(1)))

    sizing_passed = (res_sizing.returncode == 0 and passed_tests >= 15)
    assertions.append({
        "assertion": "Governor sizing permanent regression suite passes",
        "observed": {"passed_tests": passed_tests, "exit_code": res_sizing.returncode},
        "passed": sizing_passed
    })
    if not sizing_passed:
        errors.append(f"Frontend sizing tests failed (observed {passed_tests}/15 passed, exit code {res_sizing.returncode})")

    duration = time.time() - t0
    status = "PASS" if len(errors) == 0 else "FAIL"
    candidate_sha = get_git_commit("HEAD")

    evidence = [
        "frontend/tests/governorSizingEngine.test.ts",
        "frontend/lib/simulation/governorSizingEngine.ts",
        "frontend/components/PositionSizerModal.tsx"
    ]

    receipt = create_stage_receipt(
        control_id="RQ-04",
        name="Frontend Regression & Typecheck",
        status=status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=candidate_sha,
        target_branch="release/production",
        assertions=assertions,
        evidence_paths=evidence,
        errors=errors,
        metadata={"typecheck_exit": res_tc.returncode, "sizing_tests_passed": passed_tests}
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        print(f"=== [RQ-04] RESULT: PASS ({passed_tests} tests passed, typecheck clean) ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-04] RESULT: FAIL (Errors: {errors}) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
