#!/usr/bin/env python3
"""
ARX Release Qualification - RQ-02 Adversarial Falsification Suite (Directive ARX-036-D12A)
Adversarially exercises the 10 mandatory negative controls to prove that RQ-02 fails closed
under broken, unauthorized, drifted, or malicious inputs.
"""

import sys, os, subprocess, json, tempfile, shutil, time

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TEMP_RECEIPTS_DIR = None

def run_rq02(args_list, env=None):
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "release", "rq02_diff_integrity.py")] + args_list
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    if TEMP_RECEIPTS_DIR and "ARX_RECEIPTS_DIR_OVERRIDE" not in current_env:
        current_env["ARX_RECEIPTS_DIR_OVERRIDE"] = TEMP_RECEIPTS_DIR
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=current_env)
    return res.returncode, res.stdout, res.stderr

def main():
    print("================================================================================")
    print("ARX RELEASE QUALIFICATION - RQ-02 ADVERSARIAL FALSIFICATION SUITE (10 CONTROLS)")
    print("================================================================================")

    results = []
    real_manifest_path = os.path.join(PROJECT_ROOT, "scripts", "release", "release_manifest.json")
    with open(real_manifest_path, "r", encoding="utf-8") as f:
        canonical_manifest = json.load(f)

    global TEMP_RECEIPTS_DIR
    temp_dir = tempfile.mkdtemp(prefix="arx_rq02_adversarial_")
    TEMP_RECEIPTS_DIR = os.path.join(temp_dir, "receipts")
    os.makedirs(TEMP_RECEIPTS_DIR, exist_ok=True)

    try:
        # Positive Control: Canonical manifest & HEAD against base anchor
        rc, out, err = run_rq02(["--allow-uncommitted"])
        passed = (rc == 0 and "=== [RQ-02] RESULT: PASS ===" in out)
        results.append({
            "control": "PC-00: Baseline Positive Control",
            "expected_rc": 0,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": "" if passed else out[-200:]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] PC-00: Baseline Positive Control (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-01: An unapproved added file
        # -------------------------------------------------------------------------
        m1 = json.loads(json.dumps(canonical_manifest))
        # Remove an added file from approved manifest (e.g. ARX_UX_OWNER_REVIEW_PACKAGE.md)
        del m1["files"]["ARX_UX_OWNER_REVIEW_PACKAGE.md"]
        m1_path = os.path.join(temp_dir, "manifest_nc01.json")
        with open(m1_path, "w", encoding="utf-8") as f:
            json.dump(m1, f, indent=2)

        rc, out, err = run_rq02(["--manifest", m1_path, "--allow-uncommitted"])
        passed = (rc == 1 and "UNAPPROVED" in out and "ARX_UX_OWNER_REVIEW_PACKAGE.md" in out)
        results.append({
            "control": "NC-01: Unapproved added file in diff",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "UNAPPROVED" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-01: Unapproved added file (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-02: An unapproved modification
        # -------------------------------------------------------------------------
        m2 = json.loads(json.dumps(canonical_manifest))
        # Remove a modified file from approved manifest (e.g. api/routes/screener.py)
        del m2["files"]["api/routes/screener.py"]
        m2_path = os.path.join(temp_dir, "manifest_nc02.json")
        with open(m2_path, "w", encoding="utf-8") as f:
            json.dump(m2, f, indent=2)

        rc, out, err = run_rq02(["--manifest", m2_path, "--allow-uncommitted"])
        passed = (rc == 1 and "UNAPPROVED" in out and "api/routes/screener.py" in out)
        results.append({
            "control": "NC-02: Unapproved modification in diff",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "UNAPPROVED" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-02: Unapproved modification (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-03: A missing approved file
        # -------------------------------------------------------------------------
        m3 = json.loads(json.dumps(canonical_manifest))
        # Add a fictitious approved file not present in diff
        m3["files"]["unreal/fictitious_file.py"] = {
            "operation": "A",
            "sha256": "0" * 64,
            "size_bytes": 100,
            "originating_pr_number": 131,
            "governing_requirement": "Test"
        }
        m3_path = os.path.join(temp_dir, "manifest_nc03.json")
        with open(m3_path, "w", encoding="utf-8") as f:
            json.dump(m3, f, indent=2)

        rc, out, err = run_rq02(["--manifest", m3_path, "--allow-uncommitted"])
        passed = (rc == 1 and "MISSING_APPROVED_DIFF_ENTRY" in out and "unreal/fictitious_file.py" in out)
        results.append({
            "control": "NC-03: Missing approved file declared in manifest",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "MISSING_APPROVED_DIFF_ENTRY" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-03: Missing approved file (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-04: A changed approved file hash
        # -------------------------------------------------------------------------
        m4 = json.loads(json.dumps(canonical_manifest))
        # Corrupt the hash of frontend/lib/api.ts
        m4["files"]["frontend/lib/api.ts"]["sha256"] = "1" * 64
        m4_path = os.path.join(temp_dir, "manifest_nc04.json")
        with open(m4_path, "w", encoding="utf-8") as f:
            json.dump(m4, f, indent=2)

        rc, out, err = run_rq02(["--manifest", m4_path, "--allow-uncommitted"])
        passed = (rc == 1 and "Hash mismatch in frontend/lib/api.ts" in out)
        results.append({
            "control": "NC-04: Changed approved file content hash",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "Hash mismatch" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-04: Changed approved file hash (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-05: An incorrect candidate SHA
        # -------------------------------------------------------------------------
        m5 = json.loads(json.dumps(canonical_manifest))
        m5["release_candidate_commit"] = "0" * 40
        m5_path = os.path.join(temp_dir, "manifest_nc05.json")
        with open(m5_path, "w", encoding="utf-8") as f:
            json.dump(m5, f, indent=2)

        rc, out, err = run_rq02(["--manifest", m5_path, "--allow-uncommitted"])
        passed = (rc == 1 and "CANDIDATE_SHA_MISMATCH" in out)
        results.append({
            "control": "NC-05: Incorrect candidate SHA",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "CANDIDATE_SHA_MISMATCH" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-05: Incorrect candidate SHA (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-06: An incorrect release base
        # -------------------------------------------------------------------------
        # Evaluating against an unapproved release base anchor (6d4020c merge base instead of cdc47a2eee release base)
        rc, out, err = run_rq02(["--base", "6d4020ca3fef3ae07ac01204b03e0629b6a7cc79", "--allow-uncommitted"])
        passed = (rc == 1 and "BASE_ANCHOR_MISMATCH" in out)
        results.append({
            "control": "NC-06: Incorrect release base anchor",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "BASE_ANCHOR_MISMATCH" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-06: Incorrect release base anchor (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-07: A staged or dirty working-tree mismatch
        # -------------------------------------------------------------------------
        # Running WITHOUT --allow-uncommitted when an approved candidate file is uncommitted
        target_probe = os.path.join(PROJECT_ROOT, "README.md")
        with open(target_probe, "rb") as f:
            orig_probe_bytes = f.read()
        try:
            with open(target_probe, "wb") as f:
                f.write(orig_probe_bytes + b"\n# dirty working tree probe for NC-07 validation\n")
            rc, out, err = run_rq02([])
            passed = (rc == 1 and "DIRTY_WORKING_TREE_MISMATCH" in out)
        finally:
            with open(target_probe, "wb") as f:
                f.write(orig_probe_bytes)
        results.append({
            "control": "NC-07: Staged or dirty working-tree mismatch",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "DIRTY_WORKING_TREE_MISMATCH" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-07: Dirty working-tree mismatch (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-08: Unauthorized qualification-script modification
        # -------------------------------------------------------------------------
        m8 = json.loads(json.dumps(canonical_manifest))
        m8["qualification_framework_files"]["scripts/release/rq01_candidate_identity.py"]["sha256"] = "9" * 64
        m8_path = os.path.join(temp_dir, "manifest_nc08.json")
        with open(m8_path, "w", encoding="utf-8") as f:
            json.dump(m8, f, indent=2)

        rc, out, err = run_rq02(["--manifest", m8_path, "--allow-uncommitted"])
        passed = (rc == 1 and "UNAUTHORIZED_QUALIFICATION_FRAMEWORK_MODIFICATION" in out)
        results.append({
            "control": "NC-08: Unauthorized qualification-script modification",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "UNAUTHORIZED_QUALIFICATION_FRAMEWORK_MODIFICATION" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-08: Unauthorized qualification-script modification (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-09: Duplicate or conflicting manifest entries
        # -------------------------------------------------------------------------
        # Raw text injection of duplicate key
        raw_json = json.dumps(canonical_manifest, indent=2)
        # Duplicate a key in the files dictionary
        dup_json = raw_json.replace(
            '"README.md": {',
            '"README.md": { "sha256": "dup" },\n    "README.md": {'
        )
        m9_path = os.path.join(temp_dir, "manifest_nc09.json")
        with open(m9_path, "w", encoding="utf-8") as f:
            f.write(dup_json)

        rc, out, err = run_rq02(["--manifest", m9_path, "--allow-uncommitted"])
        passed = (rc == 1 and "DUPLICATE_MANIFEST_ENTRY" in out)
        results.append({
            "control": "NC-09: Duplicate or conflicting manifest entries",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "DUPLICATE_MANIFEST_ENTRY" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-09: Duplicate manifest entry rejection (rc={rc})")

        # -------------------------------------------------------------------------
        # NC-10: Changed file attributed to an unrelated PR
        # -------------------------------------------------------------------------
        m10 = json.loads(json.dumps(canonical_manifest))
        # Attribute a file to an unauthorized/unrelated PR #999
        m10["files"]["frontend/app/page.tsx"]["originating_pr_number"] = 999
        m10_path = os.path.join(temp_dir, "manifest_nc10.json")
        with open(m10_path, "w", encoding="utf-8") as f:
            json.dump(m10, f, indent=2)

        rc, out, err = run_rq02(["--manifest", m10_path, "--allow-uncommitted"])
        passed = (rc == 1 and "UNAPPROVED_PR_ATTRIBUTION" in out and "999" in out)
        results.append({
            "control": "NC-10: Changed file attributed to unapproved PR without release authority",
            "expected_rc": 1,
            "actual_rc": rc,
            "passed": passed,
            "error_snippet": [l for l in out.splitlines() if "UNAPPROVED_PR_ATTRIBUTION" in l][:1]
        })
        print(f"[{'PASS' if passed else 'FAIL'}] NC-10: Unapproved PR attribution (rc={rc})")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n================================================================================")
    print(f"SUMMARY: {sum(1 for r in results if r['passed'])}/{len(results)} CONTROLS PASSED")
    print("================================================================================")

    # Dump results JSON for artifact inclusion
    res_dir = os.path.join(PROJECT_ROOT, "scripts", "release", "receipts")
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, "rq02_adversarial_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results recorded to: {res_path}")

    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
