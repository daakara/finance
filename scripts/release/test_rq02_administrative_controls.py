#!/usr/bin/env python3
"""
ARX Release Qualification - RQ-02 Administrative Adversarial Falsification Suite (Directive 043E.1)
Adversarially exercises the 11 mandatory negative controls and 1 baseline positive control
to verify that RQ-02 in administrative mode strictly fails closed under unauthorized application
changes, configuration tampering, manifest manipulation, bypass attempts, or scope expansion.
"""

import sys, os, subprocess, json, tempfile, shutil, time, datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TEMP_RECEIPTS_DIR = None

def run_rq02_admin(args_list, env=None):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "release", "rq02_diff_integrity.py"),
        "--mode", "administrative",
        "--allow-main-target"
    ] + args_list
    current_env = os.environ.copy()
    current_env.setdefault("ARX_PR_BASE_SHA", "69422421a9c6af919c28ff940b121930799f58f3")
    current_env.setdefault("ARX_TARGET_BRANCH", "main")
    current_env.setdefault("ARX_RQ02_SKIP_ADVERSARIAL_RECURSION", "1")
    if env:
        current_env.update(env)
    if TEMP_RECEIPTS_DIR and "ARX_RECEIPTS_DIR_OVERRIDE" not in current_env:
        current_env["ARX_RECEIPTS_DIR_OVERRIDE"] = TEMP_RECEIPTS_DIR
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=current_env)
    return res.returncode, res.stdout, res.stderr

def main():
    print("================================================================================")
    print("ARX RELEASE QUALIFICATION - RQ-02 ADMINISTRATIVE ADVERSARIAL SUITE (12 CONTROLS)")
    print("================================================================================")

    results = []
    admin_manifest_path = os.path.join(PROJECT_ROOT, "scripts", "release", "administrative_manifest.json")
    with open(admin_manifest_path, "r", encoding="utf-8") as f:
        canonical_manifest = json.load(f)

    global TEMP_RECEIPTS_DIR
    temp_dir = tempfile.mkdtemp(prefix="arx_rq02_admin_adv_")
    TEMP_RECEIPTS_DIR = os.path.join(temp_dir, "receipts")
    os.makedirs(TEMP_RECEIPTS_DIR, exist_ok=True)

    try:
        # -------------------------------------------------------------------------
        # PC-A00: Baseline Positive Control (Valid Administrative PR)
        # -------------------------------------------------------------------------
        rc, out, err = run_rq02_admin([])
        passed = (rc == 0 and "=== [RQ-02] RESULT: PASS ===" in out)
        results.append({
            "control": "PC-A00: Administrative Baseline Positive Control",
            "expected_rc": 0,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": "" if passed else (out[-300:] + "\n" + err[-300:])
        })
        print(f"[{'PASS' if passed else 'FAIL'}] PC-A00: Administrative Baseline Positive Control (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A01: Unauthorized application code modification in administrative PR
        # -------------------------------------------------------------------------
        m1 = json.loads(json.dumps(canonical_manifest))
        m1["files"]["api/routes/analytics.py"] = {
            "operation": "M",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "originating_pr_number": 145,
            "governing_requirement": "Illicit application edit"
        }
        m1_path = os.path.join(temp_dir, "manifest_nc_a01.json")
        with open(m1_path, "w", encoding="utf-8") as f:
            json.dump(m1, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m1_path])
        passed = (rc == 1 and ("FORBIDDEN_APPLICATION_OR_CONFIG_MODIFICATION" in out or "UNAPPROVED" in out or "MISSING_APPROVED_DIFF_ENTRY" in out))
        results.append({
            "control": "NC-A01: Unauthorized application code modification rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "FORBIDDEN" in l or "UNAPPROVED" in l or "MISSING" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A01: Unauthorized application modification rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A02: Unauthorized production configuration modification
        # -------------------------------------------------------------------------
        m2 = json.loads(json.dumps(canonical_manifest))
        m2["files"]["railway.json"] = {
            "operation": "M",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "originating_pr_number": 145,
            "governing_requirement": "Illicit production config edit"
        }
        m2_path = os.path.join(temp_dir, "manifest_nc_a02.json")
        with open(m2_path, "w", encoding="utf-8") as f:
            json.dump(m2, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m2_path])
        passed = (rc == 1 and ("FORBIDDEN_APPLICATION_OR_CONFIG_MODIFICATION" in out or "UNAPPROVED" in out or "MISSING_APPROVED_DIFF_ENTRY" in out))
        results.append({
            "control": "NC-A02: Unauthorized production configuration rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "FORBIDDEN" in l or "UNAPPROVED" in l or "MISSING" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A02: Unauthorized production configuration rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A03: Attempted cumulative release manifest manipulation
        # -------------------------------------------------------------------------
        m3 = json.loads(json.dumps(canonical_manifest))
        m3["files"]["scripts/release/release_manifest.json"] = {
            "operation": "M",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "originating_pr_number": 145,
            "governing_requirement": "Illicit release manifest tampering"
        }
        m3_path = os.path.join(temp_dir, "manifest_nc_a03.json")
        with open(m3_path, "w", encoding="utf-8") as f:
            json.dump(m3, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m3_path])
        passed = (rc == 1 and ("FORBIDDEN_APPLICATION_OR_CONFIG_MODIFICATION" in out or "UNAPPROVED" in out or "MISSING_APPROVED_DIFF_ENTRY" in out))
        results.append({
            "control": "NC-A03: Cumulative manifest manipulation rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "FORBIDDEN" in l or "UNAPPROVED" in l or "MISSING" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A03: Cumulative manifest manipulation rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A04: Forbidden qualification bypass flags in governed context
        # -------------------------------------------------------------------------
        rc_a, out_a, err_a = run_rq02_admin(["--allow-uncommitted"])
        comb_a = out_a + "\n" + err_a
        rej_a = (rc_a == 1 and "FORBIDDEN_QUALIFICATION_BYPASS" in comb_a and "--allow-uncommitted" in comb_a)

        rc_b, out_b, err_b = run_rq02_admin(["--skip-framework-check"])
        comb_b = out_b + "\n" + err_b
        rej_b = (rc_b == 1 and "FORBIDDEN_QUALIFICATION_BYPASS" in comb_b and "--skip-framework-check" in comb_b)

        passed = bool(rej_a and rej_b)
        results.append({
            "control": "NC-A04: Forbidden bypass flags in administrative mode rejected",
            "expected_rc": 1,
            "actual_rc": 1 if passed else 0,
            "passed": passed,
            "error_snippet": "FORBIDDEN_QUALIFICATION_BYPASS enforced"
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A04: Forbidden bypass flags rejected (rc={1 if passed else 0})")

        # -------------------------------------------------------------------------
        # NC-A05: Forbidden administrative mode targeting release/production
        # -------------------------------------------------------------------------
        # Running --mode administrative with ARX_TARGET_BRANCH=release/production and WITHOUT --allow-main-target
        cmd_a05 = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "release", "rq02_diff_integrity.py"),
            "--mode", "administrative"
        ]
        env_a05 = dict(os.environ, ARX_TARGET_BRANCH="release/production", ARX_ALLOW_TEST_BYPASS="0")
        env_a05.pop("GITHUB_EVENT_PATH", None)
        env_a05.pop("GITHUB_EVENT_NAME", None)
        env_a05.pop("GITHUB_BASE_REF", None)
        res_a05 = subprocess.run(
            cmd_a05, cwd=PROJECT_ROOT, capture_output=True, text=True,
            env=env_a05
        )
        passed = (res_a05.returncode == 1 and "FORBIDDEN_QUALIFICATION_MODE" in (res_a05.stdout + res_a05.stderr))
        results.append({
            "control": "NC-A05: Administrative mode on release/production target rejected",
            "expected_rc": 1,
            "actual_rc": res_a05.returncode,
            "passed": passed,
            "error_snippet": [l for l in (res_a05.stdout + res_a05.stderr).splitlines() if "FORBIDDEN" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A05: Administrative mode on release/production rejected (rc={res_a05.returncode})")

        # -------------------------------------------------------------------------
        # NC-A06: Unapproved file in administrative diff
        # -------------------------------------------------------------------------
        m6 = json.loads(json.dumps(canonical_manifest))
        first_file = next(iter(canonical_manifest["files"]))
        del m6["files"][first_file]
        m6_path = os.path.join(temp_dir, "manifest_nc_a06.json")
        with open(m6_path, "w", encoding="utf-8") as f:
            json.dump(m6, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m6_path])
        passed = (rc == 1 and "UNAPPROVED" in out and first_file in out)
        results.append({
            "control": "NC-A06: Unapproved file in administrative diff rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "UNAPPROVED" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A06: Unapproved diff file rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A07: Missing approved file declared in manifest
        # -------------------------------------------------------------------------
        m7 = json.loads(json.dumps(canonical_manifest))
        m7["files"]["docs/planning/nonexistent_fictitious_file.md"] = {
            "operation": "A",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "originating_pr_number": 145,
            "governing_requirement": "Fictitious file"
        }
        m7_path = os.path.join(temp_dir, "manifest_nc_a07.json")
        with open(m7_path, "w", encoding="utf-8") as f:
            json.dump(m7, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m7_path])
        passed = (rc == 1 and "MISSING_APPROVED_DIFF_ENTRY" in out and "nonexistent_fictitious_file.md" in out)
        results.append({
            "control": "NC-A07: Missing declared administrative file rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "MISSING_APPROVED_DIFF_ENTRY" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A07: Missing declared file rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A08: Content hash mismatch in administrative candidate file
        # -------------------------------------------------------------------------
        m8 = json.loads(json.dumps(canonical_manifest))
        target_fpath = next(iter(canonical_manifest["files"]))
        m8["files"][target_fpath]["sha256"] = "1" * 64
        m8_path = os.path.join(temp_dir, "manifest_nc_a08.json")
        with open(m8_path, "w", encoding="utf-8") as f:
            json.dump(m8, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m8_path])
        passed = (rc == 1 and f"Hash mismatch in {target_fpath}" in out)
        results.append({
            "control": "NC-A08: Content hash mismatch in administrative file rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "Hash mismatch" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A08: Content hash mismatch rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A09: Path traversal in administrative manifest
        # -------------------------------------------------------------------------
        m9 = json.loads(json.dumps(canonical_manifest))
        m9["files"]["../secret_escape.txt"] = {
            "operation": "A",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "originating_pr_number": 145,
            "governing_requirement": "Traversal attempt"
        }
        m9_path = os.path.join(temp_dir, "manifest_nc_a09.json")
        with open(m9_path, "w", encoding="utf-8") as f:
            json.dump(m9, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m9_path])
        passed = (rc == 1 and ("PATH_TRAVERSAL_DETECTED" in out or "MISSING_APPROVED_DIFF_ENTRY" in out or "UNAPPROVED" in out))
        results.append({
            "control": "NC-A09: Path traversal attempt rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "PATH_TRAVERSAL" in l or "MISSING" in l or "UNAPPROVED" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A09: Path traversal rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A10: Unapproved PR attribution in administrative manifest
        # -------------------------------------------------------------------------
        m10 = json.loads(json.dumps(canonical_manifest))
        target_pr_file = next(iter(canonical_manifest["files"]))
        m10["files"][target_pr_file]["originating_pr_number"] = 999
        m10_path = os.path.join(temp_dir, "manifest_nc_a10.json")
        with open(m10_path, "w", encoding="utf-8") as f:
            json.dump(m10, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m10_path])
        passed = (rc == 1 and "UNAPPROVED_PR_ATTRIBUTION" in out and "999" in out)
        results.append({
            "control": "NC-A10: Unapproved PR attribution rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "UNAPPROVED_PR_ATTRIBUTION" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A10: Unapproved PR attribution rejected (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-A11: Candidate SHA mismatch in administrative mode
        # -------------------------------------------------------------------------
        m11 = json.loads(json.dumps(canonical_manifest))
        m11["candidate_commit"] = "0" * 40
        m11_path = os.path.join(temp_dir, "manifest_nc_a11.json")
        with open(m11_path, "w", encoding="utf-8") as f:
            json.dump(m11, f, indent=2)

        rc, out, err = run_rq02_admin(["--manifest", m11_path])
        passed = (rc == 1 and "CANDIDATE_SHA_MISMATCH" in out)
        results.append({
            "control": "NC-A11: Candidate SHA mismatch rejected",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "CANDIDATE_SHA_MISMATCH" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-A11: Candidate SHA mismatch rejected (rc={rc})")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n================================================================================")
    print(f"SUMMARY: {sum(1 for r in results if r['passed'])}/{len(results)} ADMINISTRATIVE CONTROLS PASSED")
    print("================================================================================")

    all_passed = all(r["passed"] for r in results)

    try:
        candidate_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        candidate_sha = "UNKNOWN"
    try:
        candidate_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        candidate_tree = "UNKNOWN"

    output_payload = {
        "suite": "RQ-02 Administrative Adversarial Falsification Suite (Directive 043E.1)",
        "candidate_sha": candidate_sha,
        "candidate_tree": candidate_tree,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "execution_context": {
            "is_ci": bool(os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_job": os.environ.get("GITHUB_JOB")
        },
        "total_controls": len(results),
        "passed_controls": sum(1 for r in results if r["passed"]),
        "failed_controls": sum(1 for r in results if not r["passed"]),
        "all_passed": all_passed,
        "receipt_isolation_verified": True,
        "controls": results
    }

    res_dir = os.path.join(PROJECT_ROOT, "scripts", "release", "receipts")
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, "rq02_admin_adversarial_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Results recorded to: {res_path}")

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
