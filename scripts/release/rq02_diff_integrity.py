#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-02: Release Diff Integrity (Hardened Directive ARX-036-D12A)
Verifies changed-file inventory against the approved release candidate manifest.
Enforces context-specific baseline selection, duplicate manifest entry rejection,
cryptographic qualification-framework self-integrity, working-tree dirty check,
exact PR origin attribution, and fail-closed security.
"""

import sys, os, subprocess, argparse, json, hashlib, time, datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import (
    create_stage_receipt, save_stage_receipt, compute_file_sha256,
    compute_sha256, get_execution_context, get_git_commit, resolve_identity_contract,
    is_git_ancestor, SHA40_REGEX
)

APPROVED_PRS = {123, 125, 126, 127, 129, 130, 131, 132, 133}

def check_duplicate_keys_hook(pairs):
    """Custom JSON hook that rejects duplicate keys in manifest dictionaries."""
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"DUPLICATE_MANIFEST_ENTRY: Duplicate key '{k}' detected in manifest JSON.")
        seen[k] = v
    return seen

def run_git_diff_status(base_sha: str, head_sha: str):
    """Safely runs git diff without shell=True, returning list of (status, filepath)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-status", base_sha, head_sha],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.PIPE
        ).strip()
        diffs = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                status, path = parts[0], parts[1].strip()
                diffs.append((status, path))
        return diffs
    except subprocess.CalledProcessError as e:
        print(f"FAIL: git diff command failed: {e}")
        return None

def compute_git_blob_sha256(ref: str, path: str):
    """Computes SHA256 of a file blob directly from the immutable Git object database."""
    try:
        blob = subprocess.check_output(
            ["git", "show", f"{ref}:{path}"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.PIPE
        )
        return hashlib.sha256(blob).hexdigest()
    except subprocess.CalledProcessError:
        return None

def git_commit_exists(sha: str) -> bool:
    """Verifies that a full 40-char SHA exists as a valid commit in the Git repository."""
    if not sha or not SHA40_REGEX.match(sha):
        return False
    try:
        res = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return res.returncode == 0
    except Exception:
        return False

def run_git_merge_base(commit1: str, commit2: str):
    """Finds common ancestor commit between two commits."""
    try:
        out = subprocess.check_output(
            ["git", "merge-base", commit1, commit2],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.PIPE
        ).strip()
        return out if out else None
    except Exception:
        return None

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    parser = argparse.ArgumentParser(description="RQ-02: Release Diff Integrity Verification")
    parser.add_argument("--mode", choices=["auto", "cumulative", "pr", "artifact"], default="auto",
                        help="Qualification mode: auto, cumulative release, PR differential, or artifact")
    parser.add_argument("--base", default=None, help="Release base anchor (defaults to manifest base_anchor_commit)")
    parser.add_argument("--head", default=None, help="Candidate commit to evaluate")
    parser.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "release_manifest.json"),
                        help="Approved release manifest")
    parser.add_argument("--allow-uncommitted", action="store_true", help="Permit uncommitted changes in working tree (testing only)")
    parser.add_argument("--skip-framework-check", action="store_true", help="Skip framework self-integrity check (testing only)")
    args = parser.parse_args()

    print("=== [RQ-02] RELEASE DIFF INTEGRITY CHECK (HARDENED ARX-036-D12A) ===")
    print(f"Qualification Mode: {args.mode}")
    print(f"Manifest Path: {args.manifest}")

    # 1. Manifest file existence & parsing with duplicate-key detection
    if not os.path.exists(args.manifest):
        errors.append(f"Release manifest not found at {args.manifest}")
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-02", "Release Diff Integrity", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       "UNKNOWN", "release/production", assertions, errors=errors)
        save_stage_receipt(receipt)
        print(f"FAIL: Release manifest not found at {args.manifest}")
        sys.exit(1)

    try:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f, object_pairs_hook=check_duplicate_keys_hook)
        assertions.append({
            "assertion": "Manifest JSON parsed successfully with zero duplicate keys",
            "passed": True
        })
    except ValueError as ve:
        err = f"DUPLICATE_MANIFEST_ENTRY: {ve}"
        errors.append(err)
        assertions.append({"assertion": "Zero duplicate manifest keys", "passed": False, "error": str(ve)})
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-02", "Release Diff Integrity", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       "UNKNOWN", "release/production", assertions, errors=errors)
        save_stage_receipt(receipt)
        print(f"FAIL: {err}")
        sys.exit(1)
    except Exception as e:
        err = f"Manifest JSON parsing error: {e}"
        errors.append(err)
        manifest_data = {}

    manifest_base = manifest_data.get("base_anchor_commit")
    manifest_candidate = manifest_data.get("release_candidate_commit")
    approved_files = manifest_data.get("files", {})
    qualification_files = manifest_data.get("qualification_framework_files", {})

    # 2. Baseline & Candidate Commit Resolution
    ok, identity, err = resolve_identity_contract()
    source_sha = identity["source_sha"] if ok else get_git_commit("HEAD")
    head_sha = get_git_commit(args.head) if args.head else source_sha

    # Context-specific baseline selection
    if args.base:
        base_sha = get_git_commit(args.base) or args.base
    elif args.mode == "pr" and ok and identity.get("base_sha"):
        base_sha = identity["base_sha"]
    elif manifest_base:
        base_sha = manifest_base
    else:
        base_sha = "cdc47a2eee8a427aaa6cf3ab5d01cad7f35d71e1"

    print(f"Diff range: {base_sha[:7]}..{head_sha[:7]}")
    print(f"Expected Base Anchor (Manifest): {manifest_base}")
    print(f"Expected Candidate (Manifest):   {manifest_candidate}")

    # 3. Release Base & Candidate Alignment Verification
    base_exists = git_commit_exists(base_sha)
    assertions.append({
        "assertion": "Base anchor exists in git repository history",
        "observed": {"base_sha": base_sha},
        "passed": base_exists
    })
    if not base_exists:
        errors.append(f"BASE_ANCHOR_INVALID: Base commit '{base_sha}' does not exist in git history.")

    head_exists = git_commit_exists(head_sha)
    assertions.append({
        "assertion": "Candidate commit exists in git repository history",
        "observed": {"head_sha": head_sha},
        "passed": head_exists
    })
    if not head_exists:
        errors.append(f"CANDIDATE_COMMIT_INVALID: Candidate commit '{head_sha}' does not exist in git history.")

    if base_exists and head_exists:
        anc_ok = is_git_ancestor(base_sha, head_sha)
        mb = base_sha if anc_ok else run_git_merge_base(base_sha, head_sha)
        lineage_ok = bool(anc_ok or (mb and git_commit_exists(mb)))
        assertions.append({
            "assertion": "Base anchor shares verified Git lineage or ancestry with candidate",
            "observed": {"base_sha": base_sha, "head_sha": head_sha, "is_direct_ancestor": anc_ok, "merge_base": mb},
            "passed": lineage_ok
        })
        if not lineage_ok:
            errors.append(f"BASE_LINEAGE_INVALID: Base commit '{base_sha}' shares no common git lineage with candidate '{head_sha}'.")

    if manifest_base:
        base_match = (base_sha.lower() == manifest_base.lower())
        assertions.append({
            "assertion": "Release base matches manifest base_anchor_commit",
            "observed": {"base_sha": base_sha, "expected": manifest_base},
            "passed": base_match
        })
        if not base_match:
            errors.append(f"BASE_ANCHOR_MISMATCH: Evaluated base {base_sha} does not match manifest base {manifest_base}")

    if manifest_candidate and args.mode in ("auto", "cumulative"):
        # Candidate SHA must match manifest release_candidate_commit
        cand_match = (head_sha.lower() == manifest_candidate.lower())
        # In CI merge-commits, head_sha parent2 can also be checked
        if not cand_match:
            try:
                res_anc = subprocess.run(
                    ["git", "merge-base", "--is-ancestor", manifest_candidate, head_sha],
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if res_anc.returncode == 0:
                    cand_match = True
            except Exception:
                pass

        assertions.append({
            "assertion": "Candidate SHA matches or incorporates manifest release_candidate_commit",
            "observed": {"head_sha": head_sha, "expected": manifest_candidate},
            "passed": cand_match
        })
        if not cand_match:
            errors.append(f"CANDIDATE_SHA_MISMATCH: Evaluated candidate {head_sha} does not match manifest candidate {manifest_candidate}")

    # 4. Qualification Framework Self-Integrity Check
    if not args.skip_framework_check and qualification_files:
        framework_mismatches = []
        for qpath, qmeta in qualification_files.items():
            expected_qhash = qmeta.get("sha256")
            disk_path = os.path.join(PROJECT_ROOT, qpath)
            if not os.path.exists(disk_path):
                framework_mismatches.append((qpath, expected_qhash, "FILE_MISSING"))
                continue
            act_qhash = compute_file_sha256(disk_path)
            if act_qhash.lower() != expected_qhash.lower():
                framework_mismatches.append((qpath, expected_qhash, act_qhash))

        assertions.append({
            "assertion": "Qualification framework scripts cryptographic self-integrity verified",
            "observed": {"mismatch_count": len(framework_mismatches)},
            "passed": len(framework_mismatches) == 0
        })
        if framework_mismatches:
            for qf, exp, act in framework_mismatches:
                errors.append(f"UNAUTHORIZED_QUALIFICATION_FRAMEWORK_MODIFICATION: {qf} expected {exp[:10]}... got {act[:10]}...")

    # 5. Staged or Dirty Working-Tree Check on Candidate Files
    if not args.allow_uncommitted:
        try:
            status_out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.PIPE
            ).strip()
            dirty_candidate_files = []
            for line in status_out.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                code, fpath = parts[0], parts[1].strip()
                if " -> " in fpath:
                    fpath = fpath.split(" -> ")[1].strip()
                fpath = fpath.replace("\\", "/")
                # If an approved candidate file or manifest is dirty in working tree, fail closed
                if fpath in approved_files or fpath in qualification_files or fpath == "scripts/release/release_manifest.json":
                    dirty_candidate_files.append((code, fpath))

            assertions.append({
                "assertion": "Zero uncommitted changes to approved release candidate files",
                "observed": {"dirty_count": len(dirty_candidate_files), "files": dirty_candidate_files},
                "passed": len(dirty_candidate_files) == 0
            })
            if dirty_candidate_files:
                for code, df in dirty_candidate_files:
                    errors.append(f"DIRTY_WORKING_TREE_MISMATCH: Candidate file '{df}' has uncommitted modifications ({code})")
        except Exception as e:
            errors.append(f"Unable to check git working tree status: {e}")

    # 6. Diff Evaluation & Unapproved File Detection
    if not base_exists or not head_exists:
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-02", "Release Diff Integrity", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       head_sha, "release/production", assertions, errors=errors)
        save_stage_receipt(receipt)
        print("=== [RQ-02] RESULT: FAIL (Invalid Commits) ===")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    diff_entries = run_git_diff_status(base_sha, head_sha)
    if diff_entries is None:
        errors.append(f"Unable to execute git diff {base_sha} {head_sha}")
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-02", "Release Diff Integrity", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       head_sha, "release/production", assertions, errors=errors)
        save_stage_receipt(receipt)
        sys.exit(1)

    all_approved = {}
    all_approved.update(approved_files)
    all_approved.update(qualification_files)

    unapproved = []
    renamed = []
    deleted = []
    modified = []
    added = []
    diff_paths_seen = set()

    for status_code, path in diff_entries:
        diff_paths_seen.add(path)
        if status_code.startswith("R"):
            renamed.append(path)
        elif status_code.startswith("D"):
            deleted.append(path)
        elif status_code.startswith("A"):
            added.append(path)
        elif status_code.startswith("M"):
            modified.append(path)

        is_manifest_itself = (path == "scripts/release/release_manifest.json")
        is_dynamic_receipt = is_manifest_itself or path.startswith("scripts/release/receipts/") or path in [
            "scripts/release/master_ci_run_results.json",
            "scripts/release/negative_test_results.json",
            "scripts/release/release_qualification_summary.json",
            "scripts/release/rq_compatibility_results.json",
            "scripts/release/rq_provenance_receipt.json",
            "scripts/release/rq_rollback_targets.json"
        ]

        if path not in all_approved and not is_dynamic_receipt:
            unapproved.append((status_code, path))

    assertions.append({
        "assertion": "Zero unapproved files in release diff",
        "observed": {"unapproved_count": len(unapproved), "unapproved_files": unapproved},
        "passed": len(unapproved) == 0
    })

    if unapproved:
        print(f"FAIL: Detected {len(unapproved)} unapproved files in diff:")
        for code, uf in unapproved:
            print(f"  [UNAPPROVED ({code})] {uf}")
            errors.append(f"Unapproved file in diff ({code}): {uf}")

    # 7. Completeness Verification (Zero Missing Approved Files)
    missing_approved = []
    for fpath in approved_files.keys():
        if fpath not in diff_paths_seen:
            missing_approved.append(fpath)

    assertions.append({
        "assertion": "All approved candidate files present in release diff",
        "observed": {"missing_count": len(missing_approved), "missing_files": missing_approved},
        "passed": len(missing_approved) == 0
    })
    if missing_approved:
        for mf in missing_approved:
            errors.append(f"MISSING_APPROVED_DIFF_ENTRY: Manifest specifies approved file '{mf}' which is missing from release diff")

    # 8. Cryptographic Content Hashes & PR Attribution Verification
    hash_mismatches = []
    unapproved_prs = []

    for fpath, meta in approved_files.items():
        expected_hash = meta.get("sha256")
        if not expected_hash:
            errors.append(f"Approved file {fpath} lacks sha256 in manifest.")
            continue

        # Verify PR attribution
        orig_pr = meta.get("originating_pr_number") or meta.get("pr")
        if orig_pr is not None:
            try:
                pr_num = int(orig_pr)
                if pr_num not in APPROVED_PRS:
                    unapproved_prs.append((fpath, pr_num))
            except Exception:
                unapproved_prs.append((fpath, str(orig_pr)))
        else:
            unapproved_prs.append((fpath, "MISSING_PR_ATTRIBUTION"))

        # Compute blob hash from immutable git tree at head_sha
        blob_hash = compute_git_blob_sha256(head_sha, fpath)
        disk_path = os.path.join(PROJECT_ROOT, fpath)
        disk_hash = compute_file_sha256(disk_path) if os.path.exists(disk_path) else "NOT_ON_DISK"

        target_hash = blob_hash if blob_hash else disk_hash
        if not target_hash or target_hash.lower() != expected_hash.lower():
            hash_mismatches.append((fpath, expected_hash, target_hash))

    assertions.append({
        "assertion": "Candidate file content hashes strictly match manifest",
        "observed": {"mismatch_count": len(hash_mismatches)},
        "passed": len(hash_mismatches) == 0
    })
    if hash_mismatches:
        for f, exp, act in hash_mismatches:
            errors.append(f"Hash mismatch in {f}: expected {exp[:10]} got {act[:10]}")

    assertions.append({
        "assertion": "All changed files strictly attributed to approved release PRs",
        "observed": {"unapproved_pr_count": len(unapproved_prs), "unapproved_prs": unapproved_prs},
        "passed": len(unapproved_prs) == 0
    })
    if unapproved_prs:
        for f, pr_val in unapproved_prs:
            errors.append(f"UNAPPROVED_PR_ATTRIBUTION: File {f} attributed to unapproved or missing PR ({pr_val})")

    duration = time.time() - t0
    status = "PASS" if len(errors) == 0 else "FAIL"

    receipt = create_stage_receipt(
        control_id="RQ-02",
        name="Release Diff Integrity",
        status=status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=head_sha,
        target_branch="release/production",
        assertions=assertions,
        evidence_paths=[args.manifest],
        errors=errors,
        metadata={
            "mode": args.mode,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "total_diff_entries": len(diff_entries),
            "approved_files_count": len(approved_files),
            "qualification_files_count": len(qualification_files),
            "unapproved_count": len(unapproved),
            "missing_approved_count": len(missing_approved),
            "hash_mismatch_count": len(hash_mismatches),
            "unapproved_pr_count": len(unapproved_prs)
        }
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        # 9. Hosted Adversarial Falsification Gate
        # In canonical CI execution (clean run against default manifest), execute both adversarial falsification suites
        is_canonical_run = (
            not args.allow_uncommitted
            and not args.skip_framework_check
            and os.path.abspath(args.manifest) == os.path.abspath(os.path.join(os.path.dirname(__file__), "release_manifest.json"))
            and os.environ.get("ARX_RQ02_SKIP_ADVERSARIAL_RECURSION") != "1"
        )
        if is_canonical_run:
            sub_env = os.environ.copy()
            sub_env["ARX_RQ02_SKIP_ADVERSARIAL_RECURSION"] = "1"
            print("\n>>> [RQ-02-ADV] Executing RQ-02 Adversarial Controls...")
            res_adv1 = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "test_rq02_adversarial_controls.py")], cwd=PROJECT_ROOT, env=sub_env)
            if res_adv1.returncode != 0:
                print("FAIL: RQ-02 Adversarial Controls failed!")
                sys.exit(1)

            print("\n>>> [Gate H] Executing Gate H 12-Scenario Workflow Attacks...")
            res_adv2 = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "test_d12c_workflow_attacks.py")], cwd=PROJECT_ROOT, env=sub_env)
            if res_adv2.returncode != 0:
                print("FAIL: Gate H Workflow Attacks failed!")
                sys.exit(1)

        print(f"PASS: All {len(diff_entries)} changed files strictly match approved release manifest (0 unapproved).")
        print("=== [RQ-02] RESULT: PASS ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-02] RESULT: FAIL (Errors: {len(errors)}) ===")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
