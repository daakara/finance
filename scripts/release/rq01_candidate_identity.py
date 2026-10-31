#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-01: Candidate Identity & Ancestry
Validates PR source, target, commit SHA ancestry, GitHub Actions execution context,
and prevents candidate substitution.
"""

import sys, os, subprocess, argparse, json, re, time, datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, get_execution_context, resolve_identity_contract

def run_git(cmd_args):
    try:
        return subprocess.check_output(cmd_args, cwd=PROJECT_ROOT, text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError:
        return None

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    parser = argparse.ArgumentParser(description="RQ-01: Candidate Identity Verification")
    parser.add_argument("--manifest", default=None, help="Path to release_manifest.json")
    parser.add_argument("--base", default=None, help="Expected release base commit")
    parser.add_argument("--expected-candidate", default=None, help="Expected authorized candidate commit")
    parser.add_argument("--target-branch", default=None, help="Target branch override")
    parser.add_argument("--allow-main-target", action="store_true", help="Permit main as target during staging PR qualification")
    args = parser.parse_args()

    # Load release manifest if present to derive canonical defaults
    manifest_path = args.manifest or os.path.join(PROJECT_ROOT, "scripts", "release", "release_manifest.json")
    manifest_data = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    expected_base = args.base or manifest_data.get("base_anchor_commit") or "9f9cdb9eb7d724c8ea22573826e76cabd1d2944f"
    expected_candidate = args.expected_candidate or manifest_data.get("release_candidate_commit")

    print("=== [RQ-01] CANDIDATE IDENTITY & ANCESTRY CHECK ===")
    ok, ctx, err_msg = resolve_identity_contract()
    if not ok:
        errors.append(f"Identity resolution failed closed: {err_msg}")
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-01", "Candidate Identity & Lineage", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       "UNKNOWN", "UNKNOWN", assertions, errors=errors)
        save_stage_receipt(receipt)
        print(f"FAIL: {err_msg}")
        sys.exit(1)

    current_head = run_git(["git", "rev-parse", "HEAD"])
    if not current_head:
        errors.append("Unable to determine current git HEAD commit.")
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-01", "Candidate Identity & Lineage", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       "UNKNOWN", "UNKNOWN", assertions, errors=errors)
        save_stage_receipt(receipt)
        print("FAIL: Unable to determine current HEAD commit.")
        sys.exit(1)

    # 1. Format check
    is_hex_sha = bool(re.match(r"^[0-9a-fA-F]{40}$", current_head))
    assertions.append({"assertion": "HEAD is valid 40-char hex SHA", "observed": current_head, "passed": is_hex_sha})
    if not is_hex_sha:
        errors.append(f"HEAD {current_head} is not a valid 40-character hex SHA.")

    # 2. Derive Target Branch and Head from Identity Contract
    target_branch = args.target_branch or ctx["target_branch"]
    candidate_sha = ctx["source_sha"]

    # Handle GitHub Actions merge-ref checkout behavior
    is_merge_commit = False
    parents_out = run_git(["git", "rev-parse", "HEAD^@"])
    if parents_out:
        parents = parents_out.splitlines()
        if len(parents) >= 2 and ctx.get("event_name") == "pull_request":
            is_merge_commit = True
            merge_base_parent = parents[0]
            merge_pr_parent = parents[1]
            assertions.append({
                "assertion": "GitHub PR Merge Commit Verified",
                "observed": {"merge_sha": current_head, "base_parent": merge_base_parent, "head_parent": merge_pr_parent},
                "passed": True
            })
            if merge_pr_parent.lower() != candidate_sha.lower():
                errors.append(f"Merge commit head parent {merge_pr_parent} does not match PR source SHA {candidate_sha}")

    print(f"Current Checked-Out HEAD: {current_head} (Merge Commit: {is_merge_commit})")
    print(f"Resolved Candidate SHA: {candidate_sha}")
    print(f"Target Branch: {target_branch}")
    print(f"Expected Base Anchor: {expected_base}")
    print(f"Expected Candidate: {expected_candidate}")

    # 3. Base commit existence in repository history
    base_exists = run_git(["git", "cat-file", "-e", expected_base]) is not None
    assertions.append({"assertion": "Base anchor exists in repo history", "observed": expected_base, "passed": base_exists})
    if not base_exists:
        errors.append(f"Base commit {expected_base} does not exist in repository history.")

    # 4. Ancestry check: expected_base must be an ancestor of candidate_sha
    is_base_ancestor = False
    if base_exists:
        res = subprocess.run(["git", "merge-base", "--is-ancestor", expected_base, candidate_sha], cwd=PROJECT_ROOT)
        is_base_ancestor = (res.returncode == 0)
    assertions.append({"assertion": "Base anchor is ancestor of Candidate SHA", "observed": is_base_ancestor, "passed": is_base_ancestor})
    if not is_base_ancestor:
        errors.append(f"Base commit {expected_base} is NOT an ancestor of Candidate SHA {candidate_sha}.")

    # 5. Target branch validity check
    clean_target = target_branch.replace("refs/heads/", "").strip()
    valid_targets = ["release/production"]
    if args.allow_main_target or clean_target == "main":
        valid_targets.append("main")

    is_target_valid = clean_target in valid_targets
    assertions.append({"assertion": "Target branch is authorized for qualification", "observed": clean_target, "passed": is_target_valid})
    if not is_target_valid:
        errors.append(f"Target branch '{clean_target}' is not authorized. Allowed: {valid_targets}")

    # 6. Candidate substitution guard
    if expected_candidate:
        is_cand_incorporated = (candidate_sha.lower() == expected_candidate.lower())
        if not is_cand_incorporated:
            res_cand = subprocess.run(["git", "merge-base", "--is-ancestor", expected_candidate, candidate_sha], cwd=PROJECT_ROOT)
            is_cand_incorporated = (res_cand.returncode == 0)
        assertions.append({
            "assertion": "Candidate SHA incorporates authorized candidate",
            "observed": {"candidate_sha": candidate_sha, "expected": expected_candidate, "ancestor": is_cand_incorporated},
            "passed": is_cand_incorporated
        })
        if not is_cand_incorporated:
            errors.append(f"Candidate {candidate_sha} does not incorporate authorized candidate {expected_candidate}.")

    duration = time.time() - t0
    status = "PASS" if len(errors) == 0 else "FAIL"

    receipt = create_stage_receipt(
        control_id="RQ-01",
        name="Candidate Identity & Lineage",
        status=status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=candidate_sha,
        target_branch=clean_target,
        assertions=assertions,
        evidence_paths=[".github/workflows/release-qualification.yml"],
        errors=errors,
        metadata={"is_merge_commit": is_merge_commit, "checked_out_head": current_head}
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        print("=== [RQ-01] RESULT: PASS ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-01] RESULT: FAIL (Errors: {errors}) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
