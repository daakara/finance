#!/usr/bin/env python3
"""
ARX Release Qualification - Synthetic Merge Reproduction Fixture
Reproduces the exact failure observed in hosted GitHub Actions run 40696182896.
Source SHA:    4c9e8e2ecd27d765424bac35e384b8f7c6c9861f
Execution SHA: 65cc4686afec8a71dbcac7c768fe0ca519dda943
Demonstrates that legacy receipt generation causes RQ-02..11 to emit the execution SHA
as candidate_sha, which RQ-12 rejects as CANDIDATE_SUBSTITUTED.
"""

import sys, os, json, tempfile, subprocess, shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, compute_sha256, compute_file_sha256

SOURCE_SHA = "4c9e8e2ecd27d765424bac35e384b8f7c6c9861f"
EXECUTION_SHA = "65cc4686afec8a71dbcac7c768fe0ca519dda943"
BASE_SHA = "ea3cc13eeeb6d3e6c250ecfd3ed6aecb4ec44d6c"

MANDATORY_CONTROLS = [
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

def main():
    print("===============================================================")
    print("   ARX TERMINAL - REPRODUCING HOSTED FAILURE (RUN 40696182896)")
    print("===============================================================")

    temp_dir = tempfile.mkdtemp(prefix="arx_hosted_repro_")
    event_file = os.path.join(temp_dir, "event.json")
    with open(event_file, "w", encoding="utf-8") as f:
        json.dump({
            "pull_request": {
                "number": 121,
                "head": {"sha": SOURCE_SHA, "ref": "feat/release-qualification-ci-074a1"},
                "base": {"sha": BASE_SHA, "ref": "main"}
            }
        }, f)

    env = dict(os.environ)
    env["ARX_RECEIPTS_DIR_OVERRIDE"] = temp_dir
    env["GITHUB_ACTIONS"] = "true"
    env["GITHUB_RUN_ID"] = "40696182896"
    env["GITHUB_RUN_ATTEMPT"] = "1"
    env["GITHUB_REPOSITORY"] = "daakara/finance"
    env["GITHUB_EVENT_NAME"] = "pull_request"
    env["GITHUB_EVENT_PATH"] = event_file
    env["GITHUB_SHA"] = EXECUTION_SHA
    env["GITHUB_BASE_REF"] = "main"
    env["GITHUB_HEAD_REF"] = "feat/release-qualification-ci-074a1"
    env["ARX_MOCK_MERGE_PARENTS"] = f"{BASE_SHA},{SOURCE_SHA}"
    env["ARX_MOCK_EXECUTION_SHA"] = EXECUTION_SHA

    # Simulate legacy receipt generation from candidate 4c9e8e2
    for cid, name in MANDATORY_CONTROLS:
        # In run 40696182896:
        # RQ-01 used target_sha = source_sha (4c9e8e2...)
        # RQ-02..11 used get_git_commit('HEAD') = execution_sha (65cc468...)
        cand_sha = SOURCE_SHA if cid == "RQ-01" else EXECUTION_SHA

        ctx = {
            "is_github_actions": True,
            "repository": "daakara/finance",
            "workflow": "Release Qualification Suite",
            "run_id": "40696182896",
            "run_attempt": 1,
            "workflow_run_id": "40696182896",
            "workflow_run_attempt": "1",
            "event_name": "pull_request",
            "pr_number": 121,
            "pr_head_sha": SOURCE_SHA,
            "pr_base_sha": BASE_SHA,
            "base_ref": "main",
            "head_ref": "feat/release-qualification-ci-074a1",
            "github_sha": EXECUTION_SHA,
            "checked_out_sha": EXECUTION_SHA,
            "tree_sha": "test_tree",
            "is_merge_commit": True,
            "merge_parent1": BASE_SHA,
            "merge_parent2": SOURCE_SHA,
            "source_sha": cand_sha,
            "execution_sha": EXECUTION_SHA,
            "release_sha": cand_sha,
            "base_sha": BASE_SHA,
            "target_branch": "main",
            "qualification_policy_version": "2.2.0-D074A4"
        }

        manifest_path = os.path.join(PROJECT_ROOT, "scripts", "release", "release_manifest.json")
        manifest_digest = compute_file_sha256(manifest_path)

        receipt_core = {
            "control_id": cid,
            "name": name,
            "status": "PASS",
            "start_time": "2026-10-28T13:00:00Z",
            "end_time": "2026-10-28T13:00:01Z",
            "duration_sec": 1.0,
            "candidate_sha": cand_sha,
            "source_sha": cand_sha,
            "execution_sha": EXECUTION_SHA,
            "target_branch": "main",
            "execution_context": ctx,
            "assertions": [{"assertion": f"{name} check", "passed": True}],
            "evidence_paths": ["scripts/release/release_manifest.json"],
            "evidence_digests": {"scripts/release/release_manifest.json": manifest_digest},
            "errors": [],
            "metadata": {"policy_version": "2.2.0-D074A4"}
        }
        canonical_json = json.dumps(receipt_core, sort_keys=True, indent=2)
        receipt_core["receipt_digest"] = compute_sha256(canonical_json.encode("utf-8"))

        cid_clean = cid.lower().replace("-", "")
        with open(os.path.join(temp_dir, f"{cid_clean}_receipt.json"), "w", encoding="utf-8") as f:
            json.dump(receipt_core, f, indent=2)

    # Now execute RQ-12
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "rq12_qualification_summary.py"),
        "--candidate-sha", SOURCE_SHA,
        "--target-branch", "main"
    ]
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    print("Process Exit Code:", res.returncode)

    shutil.rmtree(temp_dir, ignore_errors=True)

    if res.returncode != 0 and "Total Controls: 11 | Passed: 1" in res.stdout and "RELEASE_QUALIFICATION_BLOCKED" in res.stdout:
        print("\n>>> HOSTED RUN 40696182896 FAILURE REPRODUCED PERFECTLY (1 PASS, 10 FAILED CLOSED)! <<<")
        repro_ok = True
    else:
        print("\n>>> FAILED TO REPRODUCE EXPECTED HOSTED FAILURE STATE <<<")
        repro_ok = False

    # -------------------------------------------------------------
    # Now verify the REPAIRED synthetic merge contract (Directive 074-A4.1)
    # -------------------------------------------------------------
    print("\n===============================================================")
    print("   ARX TERMINAL - VERIFYING REPAIRED SYNTHETIC MERGE CONTRACT")
    print("===============================================================")
    temp_dir2 = tempfile.mkdtemp(prefix="arx_repaired_test_")
    event_file2 = os.path.join(temp_dir2, "event.json")
    with open(event_file2, "w", encoding="utf-8") as f:
        json.dump({
            "pull_request": {
                "number": 121,
                "head": {"sha": SOURCE_SHA, "ref": "feat/release-qualification-ci-074a1"},
                "base": {"sha": BASE_SHA, "ref": "main"}
            }
        }, f)

    env2 = dict(os.environ)
    env2["ARX_RECEIPTS_DIR_OVERRIDE"] = temp_dir2
    env2["GITHUB_ACTIONS"] = "true"
    env2["GITHUB_RUN_ID"] = "40696182896"
    env2["GITHUB_RUN_ATTEMPT"] = "1"
    env2["GITHUB_REPOSITORY"] = "daakara/finance"
    env2["GITHUB_EVENT_NAME"] = "pull_request"
    env2["GITHUB_EVENT_PATH"] = event_file2
    env2["GITHUB_SHA"] = EXECUTION_SHA
    env2["GITHUB_BASE_REF"] = "main"
    env2["GITHUB_HEAD_REF"] = "feat/release-qualification-ci-074a1"
    env2["ARX_MOCK_MERGE_PARENTS"] = f"{BASE_SHA},{SOURCE_SHA}"
    env2["ARX_MOCK_EXECUTION_SHA"] = EXECUTION_SHA

    # Temporarily update os.environ so create_stage_receipt resolves repaired PR identity
    orig_env = dict(os.environ)
    try:
        os.environ.update(env2)
        # Create receipts using the updated create_stage_receipt
        # Simulate execution of all 11 stages under repaired contract
        for cid, name in MANDATORY_CONTROLS:
            receipt = create_stage_receipt(
                control_id=cid,
                name=name,
                status="PASS",
                start_time="2026-10-28T13:00:00Z",
                end_time="2026-10-28T13:00:01Z",
                duration_sec=1.0,
                candidate_sha=SOURCE_SHA, # Authoritative source SHA
                target_branch="main",
                assertions=[{"assertion": f"{name} check", "passed": True}],
                evidence_paths=["scripts/release/release_manifest.json"],
                metadata={"test_repaired": True}
            )
            cid_clean = cid.lower().replace("-", "")
            with open(os.path.join(temp_dir2, f"{cid_clean}_receipt.json"), "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
    finally:
        os.environ.clear()
        os.environ.update(orig_env)

    cmd2 = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "rq12_qualification_summary.py"),
        "--candidate-sha", SOURCE_SHA,
        "--execution-sha", EXECUTION_SHA,
        "--target-branch", "main"
    ]
    res2 = subprocess.run(cmd2, env=env2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res2.stdout)
    print("Process Exit Code:", res2.returncode)

    shutil.rmtree(temp_dir2, ignore_errors=True)

    repaired_ok = (
        res2.returncode == 0 and
        "Total Controls: 11 | Passed: 11 | Failed: 0 | Blocked: 0" in res2.stdout and
        "OVERALL VERDICT: RELEASE_QUALIFIED" in res2.stdout
    )

    if repaired_ok:
        print("\n>>> REPAIRED SYNTHETIC MERGE CONTRACT VERIFIED: 11/11 PASSED (RELEASE_QUALIFIED) <<<")
    else:
        print("\n>>> REPAIRED SYNTHETIC MERGE CONTRACT FAILED <<<")

    return 0 if (repro_ok and repaired_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
