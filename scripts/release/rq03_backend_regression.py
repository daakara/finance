#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-03: Backend Regression Suite
Executes pytest regression tests for confluence scoring, sizing guards, and Kelly capping.
Emits a verified machine-readable stage receipt.
"""

import sys, os, subprocess, time, datetime, re, tempfile

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, get_git_commit, resolve_identity_contract

OFFLINE_SEC_FILES = [
    'tests/test_sec_archived_replay_preparation.py',
    'tests/test_sec_bounded_resumable_ingestion.py',
    'tests/test_sec_resume_safety_regressions.py',
    'tests/test_sec_expanded_delivery.py',
    'tests/test_smart_money_two_issuer_trial.py',
    'tests/test_sec_ingestion_evidence_integrity.py',
    'tests/test_smart_money_continuous_ingestion.py',
]

def offline_sec_verdict(returncode, output):
    def count(label):
        matches = re.findall(r'(\d+) ' + label, output)
        return int(matches[-1]) if matches else 0
    counts = {label: count(label) for label in ('passed', 'skipped', 'failed', 'errors?')}
    return (returncode == 0 and counts['passed'] >= 242 and counts['skipped'] == 0
            and counts['failed'] == 0 and counts['errors?'] == 0), counts

def run_offline_sec_suite():
    env = os.environ.copy()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env['PYTHONPATH'] = os.pathsep.join([os.path.join(PROJECT_ROOT, 'tests'), os.path.join(PROJECT_ROOT, 'tools', 'qualification'), PROJECT_ROOT])
    env['ARX_REQUIRE_LOCAL_POSTGRES_TESTS'] = 'true'
    with tempfile.TemporaryDirectory(prefix='arx-sec-rq03-', dir=os.environ.get('RUNNER_TEMP')) as tmp:
        cmd = [sys.executable, '-m', 'pytest', '-p', 'sec_offline_test_support', '--basetemp=' + tmp] + OFFLINE_SEC_FILES + ['-q', '--tb=short', '-rs']
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)
    passed, counts = offline_sec_verdict(result.returncode, output)
    return passed, dict(counts, exit_code=result.returncode)


def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    print("=== [RQ-03] BACKEND REGRESSION SUITE ===")
    test_files = [
        "tests/test_directive_057_remediation.py",
        "tests/test_confluence_engine.py",
        "tests/test_confluence_integrity_002.py",
        "tests/test_confluence_provenance_wiring_003.py",
        "tests/test_confluence_cross_component_parity.py",
        "tests/test_financial_correctness_authority_035.py",
        "tests/test_smart_money_backup_retention_safety.py",
        "tests/test_smart_money_backup_retention_bounded_adversarial.py",
        "tests/test_rq10_manifest_authority_negative_cases.py",
        "tests/test_smart_money_provenance.py",
        "tests/test_smart_money_two_issuer_trial.py",
        "tests/test_sm_act_020_continuous_authorization.py",
        "tests/test_smart_money_worker_lifecycle_and_fencing.py",
        "tests/test_sec_disclosure_postgres_integration.py",
        "tests/test_sec_disclosure_view.py",
        "tests/test_fred_sec_presentation_integrity.py",
        "tests/test_release_sec_gate.py",
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
        "passed": (res.returncode == 0 and failed_count == 0 and passed_count >= 30)
    })

    if res.returncode != 0 or failed_count > 0:
        errors.append(f"Pytest suite failed with returncode {res.returncode}, {failed_count} failures.")
    elif passed_count < 30:
        errors.append(f"Expected at least 30 tests, only observed {passed_count}.")

    core_duration = duration
    offline_ok, offline_counts = run_offline_sec_suite()
    duration = time.time() - t0
    assertions.append({'assertion': 'Mandatory offline SEC delivery/resume suites pass without skips',
                       'observed': offline_counts, 'passed': offline_ok})
    if not offline_ok:
        errors.append('Mandatory offline SEC suite failed, skipped tests, or observed fewer than 242 passing checks.')

    status = "PASS" if len(errors) == 0 else "FAIL"
    ok, identity, err = resolve_identity_contract()
    candidate_sha = identity["source_sha"] if ok else None

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
        evidence_paths=test_files + OFFLINE_SEC_FILES,
        errors=errors,
        metadata={"pytest_passed": passed_count, "pytest_failed": failed_count, "pytest_duration": core_duration, "offline_sec": offline_counts}
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
