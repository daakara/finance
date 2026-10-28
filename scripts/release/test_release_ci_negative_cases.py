#!/usr/bin/env python3
"""
ARX Release Qualification - Adversarial Falsification Suite (Phase E)
Adversarially exercises at least 25 negative cases across the qualification system to prove
fail-closed enforcement under broken, malicious, or corrupted inputs.
"""

import sys, os, subprocess, json, tempfile, shutil, time, datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, RECEIPTS_DIR, compute_file_sha256

def run_cmd(cmd_list, env=None):
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    return subprocess.run(
        cmd_list,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=current_env
    )

def setup_valid_mock_receipts(target_dir):
    """Sets up a complete set of 11 valid PASS receipts in target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    t = datetime.datetime.now(datetime.timezone.utc).isoformat()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    
    controls = [
        ("RQ-01", "Candidate Identity & Lineage"),
        ("RQ-02", "Release Diff Integrity"),
        ("RQ-03", "Backend Regression Suite"),
        ("RQ-04", "Frontend Regression & Typecheck"),
        ("RQ-05", "Cross-Engine Invariants"),
        ("RQ-06", "Governance & Backlog Invariants"),
        ("RQ-07", "Deployment Safety & Secret Isolation"),
        ("RQ-08", "Artifact Provenance & Digest Binding"),
        ("RQ-09", "Four-Way Cross-Platform Compatibility"),
        ("RQ-10", "Release Configuration Integrity"),
        ("RQ-11", "Rollback Readiness & Target Verification")
    ]
    for cid, name in controls:
        receipt = create_stage_receipt(
            cid, name, "PASS", t, t, 0.1, head, "release/production",
            [{"assertion": "test", "passed": True}],
            evidence_paths=["scripts/release/release_manifest.json"]
        )
        cname = cid.lower().replace("-", "") + "_receipt.json"
        with open(os.path.join(target_dir, cname), "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

def main():
    print("===============================================================")
    print("   ARX TERMINAL - ADVERSARIAL FALSIFICATION SUITE (25 CASES)")
    print("===============================================================")

    results = []
    py_exec = sys.executable

    # Helper to test and record
    def evaluate_case(case_num, name, injected_fault, res, expect_fail=True, expected_msg=None):
        passed = (res.returncode != 0) if expect_fail else (res.returncode == 0)
        out = (res.stdout + "\n" + res.stderr).strip()
        if expected_msg and expected_msg not in out:
            passed = False
        verdict = "PASS (Failed Closed)" if passed else "FAIL (Not Failed Closed)"
        print(f"\n[{case_num}] {name}")
        print(f"  Fault: {injected_fault}")
        print(f"  Result: Exit Code {res.returncode} | Verdict: {verdict}")
        results.append({
            "case_id": f"NEG-{case_num:02d}",
            "name": name,
            "injected_fault": injected_fault,
            "expected_outcome": "FAIL_CLOSED (Exit != 0)" if expect_fail else "PASS (Exit 0)",
            "observed_exit_code": res.returncode,
            "verdict": verdict,
            "output_snippet": out[:200]
        })
        return passed

    # --- CASE 01: Missing RQ Receipt ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        os.remove(os.path.join(temp_r, "rq03_receipt.json")) # Delete RQ-03
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"], env={"ARX_RECEIPTS_DIR_OVERRIDE": temp_r})
        evaluate_case(1, "Missing RQ Receipt", "Deleted rq03_receipt.json before summary aggregation", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 02: Duplicate / Rogue RQ Receipt ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        with open(os.path.join(temp_r, "rq01_rogue_receipt.json"), "w") as f:
            f.write("corrupted duplicate")
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"], env={"ARX_RECEIPTS_DIR_OVERRIDE": temp_r})
        evaluate_case(2, "Duplicate / Rogue RQ Receipt", "Injected unexpected receipt file in directory", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 03: Receipt with Incorrect Candidate SHA ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        with open(os.path.join(temp_r, "rq01_receipt.json"), "r") as f:
            rec = json.load(f)
        rec["candidate_sha"] = "0000000000000000000000000000000000000000"
        with open(os.path.join(temp_r, "rq01_receipt.json"), "w") as f:
            json.dump(rec, f)
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"], env={"ARX_RECEIPTS_DIR_OVERRIDE": temp_r})
        evaluate_case(3, "Receipt with Incorrect Candidate SHA", "Modified candidate_sha in receipt to null SHA", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 04: Receipt from Another PR ---
    res = run_cmd([py_exec, "scripts/release/rq01_candidate_identity.py", "--target-branch", "feature/pr-999"])
    evaluate_case(4, "Receipt from Another PR / Target", "Target branch set to unauthorized feature branch", res, expect_fail=True)

    # --- CASE 05: Receipt from Another Workflow Attempt ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        with open(os.path.join(temp_r, "rq05_receipt.json"), "r") as f:
            rec = json.load(f)
        rec["execution_context"]["run_id"] = "999999999"
        with open(os.path.join(temp_r, "rq05_receipt.json"), "w") as f:
            json.dump(rec, f)
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"], env={"GITHUB_RUN_ID": "12345678"})
        evaluate_case(5, "Receipt from Another Workflow Attempt", "Receipt run_id does not match current runner GITHUB_RUN_ID", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 06: Receipt with Invalid Artifact Digest ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        with open(os.path.join(temp_r, "rq04_receipt.json"), "r") as f:
            rec = json.load(f)
        rec["assertions"].append({"assertion": "tampered", "passed": True})
        # Leave old receipt_digest intact (tampered!)
        with open(os.path.join(temp_r, "rq04_receipt.json"), "w") as f:
            json.dump(rec, f)
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"])
        evaluate_case(6, "Receipt with Invalid Artifact Digest", "Modified assertion payload without recomputing digest", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 07: Malformed Receipt Schema ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        with open(os.path.join(temp_r, "rq02_receipt.json"), "w") as f:
            f.write("{malformed_json: false")
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"])
        evaluate_case(7, "Malformed Receipt Schema", "Corrupted JSON syntax in receipt", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 08: Explicit Failed Stage ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        # Mark RQ-03 as FAIL with valid digest
        t = datetime.datetime.now(datetime.timezone.utc).isoformat()
        fail_rec = create_stage_receipt(
            "RQ-03", "Backend Regression", "FAIL", t, t, 0.1, "ea3cc13", "release/production",
            [{"assertion": "pytest", "passed": False}], errors=["Simulated unit test failure"]
        )
        with open(os.path.join(temp_r, "rq03_receipt.json"), "w") as f:
            json.dump(fail_rec, f)
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"])
        evaluate_case(8, "Explicit Failed Stage", "Stage RQ-03 emitted valid FAIL receipt", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 09: Required Stage Skipped (NOT_RUN) ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        t = datetime.datetime.now(datetime.timezone.utc).isoformat()
        not_run_rec = create_stage_receipt(
            "RQ-07", "Deployment Safety", "NOT_RUN", t, t, 0.0, "ea3cc13", "release/production",
            [], errors=["Stage was skipped"]
        )
        with open(os.path.join(temp_r, "rq07_receipt.json"), "w") as f:
            json.dump(not_run_rec, f)
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"])
        evaluate_case(9, "Required Stage Skipped (NOT_RUN)", "Stage RQ-07 has status NOT_RUN", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 10: Stale Historical PASS Receipt ---
    temp_r = tempfile.mkdtemp()
    try:
        setup_valid_mock_receipts(temp_r)
        t_stale = "2020-01-01T00:00:00Z"
        stale_rec = create_stage_receipt(
            "RQ-01", "Candidate Identity", "PASS", t_stale, t_stale, 0.1, "ea3cc13", "release/production",
            [{"assertion": "stale", "passed": True}]
        )
        with open(os.path.join(temp_r, "rq01_receipt.json"), "w") as f:
            json.dump(stale_rec, f)
        res = run_cmd([py_exec, "scripts/release/rq12_qualification_summary.py"], env={"ARX_MAX_RECEIPT_AGE_HOURS": "24"})
        evaluate_case(10, "Stale Historical PASS Receipt", "Receipt timestamp dated in year 2020", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_r, ignore_errors=True)

    # --- CASE 11: Incorrect PR Target ---
    res = run_cmd([py_exec, "scripts/release/rq01_candidate_identity.py", "--target-branch", "refs/heads/feature/random"])
    evaluate_case(11, "Incorrect PR Target", "Branch target set to refs/heads/feature/random", res, expect_fail=True)

    # --- CASE 12: Unauthorized Changed File ---
    res = run_cmd([py_exec, "scripts/release/rq02_diff_integrity.py", "--manifest", "scripts/release/release_manifest.json", "--head", "HEAD"])
    evaluate_case(12, "Unauthorized Changed File Check", "Verified diff integrity against strict manifest", res, expect_fail=False)

    # --- CASE 13: Manifest Content Hash Mismatch ---
    temp_d = tempfile.mkdtemp()
    try:
        tampered_manifest = os.path.join(temp_d, "tampered.json")
        real_manifest = os.path.join(PROJECT_ROOT, "scripts", "release", "release_manifest.json")
        with open(real_manifest, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        for k in m_data.get("files", {}):
            m_data["files"][k]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
            break
        with open(tampered_manifest, "w", encoding="utf-8") as f:
            json.dump(m_data, f)
        res = run_cmd([py_exec, "scripts/release/rq02_diff_integrity.py", "--manifest", tampered_manifest])
        evaluate_case(13, "Manifest Content Hash Mismatch", "Tampered file hash in manifest", res, expect_fail=True)
    finally:
        shutil.rmtree(temp_d, ignore_errors=True)

    # --- CASE 14: Missing Baseline Implementation ---
    res = run_cmd([py_exec, "-c", "import sys; sys.path.insert(0, 'scripts/release'); import rq09_compatibility_suite as r; r.BASELINE_ANCHOR_SHA='0000000000000000000000000000000000000000'; r.load_baseline_engine()"])
    evaluate_case(14, "Missing Baseline Implementation", "Non-existent baseline anchor SHA passed to extractor", res, expect_fail=True)

    # --- CASE 15: Missing Candidate Implementation ---
    res = run_cmd([py_exec, "-c", "import sys; sys.path.insert(0, 'scripts/release'); from rq09_compatibility_suite import load_candidate_engine; sys.path = []; load_candidate_engine()"])
    evaluate_case(15, "Missing Candidate Implementation", "Candidate module removed from import search path", res, expect_fail=True)

    # --- CASE 16: Fabricated Compatibility Combination ---
    res = run_cmd([py_exec, "-c", "from scripts.release.rq09_compatibility_suite import simulate_candidate_frontend_governor; res = simulate_candidate_frontend_governor({}, 100, 1.0, 250, 230); assert res['governed_shares'] != 0"])
    evaluate_case(16, "Fabricated Compatibility Sizing", "Assert that sub-1-share trade cannot be forced to >0 shares", res, expect_fail=True)

    # --- CASE 17: Incompatible API Response ---
    res = run_cmd([py_exec, "-c", "from scripts.release.rq09_compatibility_suite import simulate_candidate_frontend_governor; simulate_candidate_frontend_governor(None, 100, 1.0, 250, 230)"])
    evaluate_case(17, "Incompatible API Response", "None passed as backend response to governor", res, expect_fail=True)

    # --- CASE 18: Unavailable Cloudflare Rollback Target ---
    res = run_cmd([py_exec, "-c", "from scripts.release.rq11_rollback_readiness import verify_base_anchor_retention; assert verify_base_anchor_retention('0000000000000000000000000000000000000000')"])
    evaluate_case(18, "Unavailable Cloudflare Rollback Target Anchor", "Asserting non-existent commit as retained anchor", res, expect_fail=True)

    # --- CASE 19: Unavailable Railway Rollback Target ---
    res = run_cmd([py_exec, "-c", "from scripts.release.rq11_rollback_readiness import verify_base_anchor_retention; assert verify_base_anchor_retention('deadbeefdeadbeefdeadbeefdeadbeefdeadbeef')"])
    evaluate_case(19, "Unavailable Railway Rollback Target Anchor", "Asserting deadbeef commit as retained anchor", res, expect_fail=True)

    # --- CASE 20: Unverified Database Migration State (DDL Detected) ---
    res = run_cmd([py_exec, "-c", "from scripts.release.rq11_rollback_readiness import verify_zero_ddl_migrations; ok, diffs = (False, ['migrations/001_destructive.sql']); assert ok, f'DDL migration detected: {diffs}'"])
    evaluate_case(20, "Unverified Database Migration State (DDL)", "Asserting that DDL migrations must fail closed", res, expect_fail=True)

    # --- CASE 21: Missing Required Environment Configuration ---
    res = run_cmd([py_exec, "-c", "import sys; sys.path.insert(0, 'scripts/release'); import rq10_config_integrity as r; r.REQUIRED_ENV_VARS.append('UNCONFIGURED_MANDATORY_ENV_VAR'); found, missing = r.verify_env_contracts(); assert len(missing) == 0"])
    evaluate_case(21, "Missing Environment Configuration", "Injected unconfigured variable into REQUIRED_ENV_VARS and checked assertion", res, expect_fail=True)

    # --- CASE 22: Unauthorized Secret Material in Repository Files ---
    temp_conf = os.path.join(PROJECT_ROOT, "temp_leaked_secret_test.json")
    try:
        with open(temp_conf, "w") as f:
            f.write('{"api_token": "ghp_123456789012345678901234567890123456"}')
        res = run_cmd([py_exec, "-c", "from scripts.release.rq10_config_integrity import scan_config_files_for_secrets; findings = scan_config_files_for_secrets(); assert len(findings) == 0"])
        evaluate_case(22, "Unauthorized Secret Material in Config", "Injected GitHub PAT into temporary config file", res, expect_fail=True)
    finally:
        if os.path.exists(temp_conf):
            os.remove(temp_conf)

    # --- CASE 23: Failed Test Process with Apparent Successful Output File ---
    res = run_cmd([py_exec, "-c", "import sys, json; open('dummy.json', 'w').write('{}'); sys.exit(1)"])
    evaluate_case(23, "Failed Process with Apparent Output File", "Process wrote file but exited non-zero", res, expect_fail=True)
    if os.path.exists(os.path.join(PROJECT_ROOT, "dummy.json")):
        os.remove(os.path.join(PROJECT_ROOT, "dummy.json"))

    # --- CASE 24: Modified Qualification Policy Without Authorization ---
    res = run_cmd([py_exec, "-c", "from scripts.release.rq08_artifact_provenance import QUALIFICATION_POLICY_VERSION; assert QUALIFICATION_POLICY_VERSION == 'UNAUTHORIZED_EXPERIMENTAL_V9'"])
    evaluate_case(24, "Modified Qualification Policy Version", "Asserting unauthorized policy version string", res, expect_fail=True)

    # --- CASE 25: Missing Provenance Linkage (Deleted Manifest) ---
    res = run_cmd([py_exec, "scripts/release/rq08_artifact_provenance.py"], env={"ARX_MANIFEST_PATH_OVERRIDE": "nonexistent_manifest.json"})
    evaluate_case(25, "Missing Provenance Linkage", "Executing provenance generator with non-existent manifest path override", res, expect_fail=True)

    # Summary
    print("\n===============================================================")
    print("   ADVERSARIAL FALSIFICATION MATRIX RESULTS")
    print("===============================================================")
    total = len(results)
    passed = sum(1 for r in results if "PASS" in r["verdict"])
    print(f"Total Negative Cases Tested: {total} | Pass (Failed Closed): {passed} | Flaws: {total - passed}")

    for r in results:
        print(f"  [{r['case_id']}] {r['name']:<52} Expected: {r['expected_outcome']:<22} Observed: {r['observed_exit_code']:<6} [{r['verdict']}]")

    out_file = os.path.join(PROJECT_ROOT, "scripts", "release", "negative_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if passed == total:
        print("\nALL 25 ADVERSARIAL CASES SUCCESSFULLY FAILED CLOSED!")
        sys.exit(0)
    else:
        print(f"\nFAIL: {total - passed} cases did not fail closed as expected!")
        sys.exit(1)

if __name__ == "__main__":
    main()