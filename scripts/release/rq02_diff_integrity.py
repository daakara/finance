#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-02: Release Diff Integrity (Hardened Directive ARX-036-D12A / D043E.1)
Verifies changed-file inventory against approved candidate manifests.
Supports:
1. Cumulative production-release qualification targeting release/production.
2. PR-scoped administrative/governance qualification targeting main.
Enforces strict fail-closed boundary isolation, immutable content hashes,
rejection of unauthorized application code in administrative PRs,
cryptographic qualification-framework self-integrity, working-tree dirty check,
and zero bypass without explicit authorization.
"""

import sys, os, subprocess, argparse, json, hashlib, time, datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import (
    create_stage_receipt, save_stage_receipt, load_stage_receipt, compute_file_sha256,
    compute_sha256, get_execution_context, get_git_commit, resolve_identity_contract,
    is_git_ancestor, SHA40_REGEX
)

APPROVED_PRS = {123, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136, 139, 140, 145, 147, 149, 150, 151, 153, 155, 156}

FORBIDDEN_ADMIN_PREFIXES = (
    "api/",
    "analyst_dashboard/",
    "frontend/",
    "migrations/",
    ".github/rulesets/",
    ".secrets/",
)

FORBIDDEN_ADMIN_FILES = (
    "railway.json",
    "wrangler.toml",
    "scripts/release/release_manifest.json",
)

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
    parser.add_argument("--mode", choices=["auto", "cumulative", "pr", "administrative", "artifact"], default="auto",
                        help="Qualification mode: auto, cumulative release, PR differential, administrative, or artifact")
    parser.add_argument("--base", default=None, help="Release base anchor (defaults to manifest base_anchor_commit)")
    parser.add_argument("--head", default=None, help="Candidate commit to evaluate")
    parser.add_argument("--manifest", default=None, help="Approved release or administrative manifest")
    parser.add_argument("--allow-main-target", action="store_true", help="Permit main as target during staging/administrative PR qualification")
    parser.add_argument("--target-branch", default=None, help="Target branch override")
    parser.add_argument("--allow-uncommitted", action="store_true", help="Permit uncommitted changes in working tree (testing only)")
    parser.add_argument("--skip-framework-check", action="store_true", help="Skip framework self-integrity check (testing only)")
    args = parser.parse_args()

    # Identity Resolution & Target Branch Derivation
    ok, identity, err_msg = resolve_identity_contract()
    rq01_receipt = load_stage_receipt("RQ-01") if os.path.exists(os.path.join(PROJECT_ROOT, "scripts", "release", "receipts", "rq01_receipt.json")) else None
    rec_target = None
    if rq01_receipt and rq01_receipt.get("status") == "PASS":
        rec_target = rq01_receipt.get("target_branch") or rq01_receipt.get("execution_context", {}).get("target_branch")

    if identity.get("event_name") == "local_execution" and rec_target:
        target_branch = args.target_branch or os.environ.get("ARX_TARGET_BRANCH") or rec_target
    else:
        target_branch = args.target_branch or os.environ.get("ARX_TARGET_BRANCH") or (identity.get("target_branch") if ok else None) or rec_target or "release/production"
    clean_target = target_branch.replace("refs/heads/", "").strip() if target_branch else "release/production"
    if args.allow_main_target and clean_target == "release/production":
        clean_target = "main"

    mode = args.mode
    if mode == "auto":
        if clean_target == "release/production":
            mode = "cumulative"
        else:
            # Targeting main or non-production staging
            if args.manifest:
                try:
                    with open(args.manifest, "r", encoding="utf-8") as f:
                        m_peek = json.load(f)
                    if m_peek.get("pr_type") == "ADMINISTRATIVE_GOVERNANCE":
                        mode = "administrative"
                    else:
                        mode = "pr"
                except Exception:
                    mode = "administrative"
            else:
                # Differentiate administrative PR vs application-remediation PR
                check_source = identity["source_sha"] if ok else get_git_commit("HEAD")
                check_head = get_git_commit(args.head) if args.head else check_source
                check_base = get_git_commit(args.base) if args.base else (
                    identity.get("base_sha") if (ok and identity.get("base_sha") and identity.get("event_name") == "pull_request") else (
                        run_git_merge_base("origin/main", check_head) or get_git_commit("origin/main") or "3f3b6c4afda9624eb11cf08d34ddfaebc642b434"
                    )
                )
                sample_diff = run_git_diff_status(check_base, check_head) if (check_base and check_head) else []
                is_app_remediation = any(
                    any(p.startswith(pfx) for pfx in FORBIDDEN_ADMIN_PREFIXES)
                    for _, p in (sample_diff or [])
                )
                if is_app_remediation:
                    mode = "pr"
                elif identity.get("pr_number") == 145 or args.allow_main_target:
                    mode = "administrative"
                else:
                    mode = "pr"

    print("=== [RQ-02] RELEASE DIFF INTEGRITY CHECK (HARDENED ARX-036-D12A / D043E.1) ===")
    print(f"Qualification Mode: {mode} (Requested: {args.mode})")
    print(f"Target Branch:      {clean_target}")

    # Security Guard: Administrative mode is strictly forbidden for production target unless explicitly overridden
    if mode in ("administrative", "pr") and clean_target == "release/production" and not args.allow_main_target:
        err = "FORBIDDEN_QUALIFICATION_MODE: Administrative qualification mode is not permitted when targeting release/production."
        print(f"FAIL: {err}", file=sys.stderr)
        errors.append(err)
        assertions.append({
            "assertion": "Administrative qualification mode strictly restricted to non-production targets",
            "passed": False,
            "error": err
        })
        duration = time.time() - t0
        receipt = create_stage_receipt(
            control_id="RQ-02",
            name="Release Diff Integrity",
            status="FAIL",
            start_time=start_time,
            end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            duration_sec=duration,
            candidate_sha="UNKNOWN",
            target_branch=clean_target,
            assertions=assertions,
            errors=errors
        )
        save_stage_receipt(receipt)
        sys.exit(1)

    # 0. Fail-Closed Enforcement: Reject Testing Bypass Flags in Governed Contexts
    is_ci = (os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true")
    is_governed_run = (os.environ.get("ARX_GOVERNED_QUALIFICATION") == "1")
    is_production_eval = (mode == "cumulative" and (not args.base or "release/production" in str(args.base) or "cdc47a2" in str(args.base)))
    is_governed_context = is_ci or is_governed_run or is_production_eval or (os.environ.get("ARX_ALLOW_TEST_BYPASS") != "1")

    if args.allow_uncommitted or args.skip_framework_check:
        forbidden = []
        if args.allow_uncommitted:
            forbidden.append("--allow-uncommitted")
        if args.skip_framework_check:
            forbidden.append("--skip-framework-check")

        if is_governed_context:
            bypass_err = (
                f"FORBIDDEN_QUALIFICATION_BYPASS: Bypass flag(s) {forbidden} are strictly forbidden "
                f"in governed CI and production release qualification environments."
            )
            print(f"FAIL: {bypass_err}", file=sys.stderr)
            errors.append(bypass_err)
            assertions.append({
                "assertion": "Zero forbidden qualification bypass flags used in governed environment",
                "passed": False,
                "error": bypass_err
            })
            duration = time.time() - t0
            receipt = create_stage_receipt(
                control_id="RQ-02",
                name="Release Diff Integrity",
                status="FAIL",
                start_time=start_time,
                end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                duration_sec=duration,
                candidate_sha="UNKNOWN",
                target_branch=clean_target,
                assertions=assertions,
                errors=errors
            )
            save_stage_receipt(receipt)
            sys.exit(1)

    # Resolve manifest file path
    manifest_data = {}
    if args.manifest:
        manifest_path = args.manifest
    elif mode == "administrative":
        manifest_path = os.path.join(os.path.dirname(__file__), "administrative_manifest.json")
    elif mode == "cumulative":
        manifest_path = os.path.join(os.path.dirname(__file__), "release_manifest.json")
    else:  # mode == "pr"
        manifest_path = None

    print(f"Manifest Path:      {manifest_path}")

    # 1. Manifest file existence & parsing with duplicate-key detection
    if manifest_path:
        if not os.path.exists(manifest_path):
            err = f"Release manifest not found at {manifest_path}"
            errors.append(err)
            status = "FAIL"
            duration = time.time() - t0
            receipt = create_stage_receipt("RQ-02", "Release Diff Integrity", status, start_time,
                                           datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                           "UNKNOWN", clean_target, assertions, errors=errors)
            save_stage_receipt(receipt)
            print(f"FAIL: {err}")
            sys.exit(1)

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
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
                                           "UNKNOWN", clean_target, assertions, errors=errors)
            save_stage_receipt(receipt)
            print(f"FAIL: {err}")
            sys.exit(1)
        except Exception as e:
            err = f"Manifest JSON parsing error: {e}"
            errors.append(err)
            manifest_data = {}

    # =========================================================================
    # TRACK 1: CUMULATIVE PRODUCTION RELEASE QUALIFICATION
    # =========================================================================
    if mode == "cumulative":
        manifest_base = manifest_data.get("base_anchor_commit")
        manifest_candidate = manifest_data.get("release_candidate_commit")
        approved_files = manifest_data.get("files", {})
        qualification_files = manifest_data.get("qualification_framework_files", {})

        source_sha = identity["source_sha"] if ok else get_git_commit("HEAD")
        head_sha = get_git_commit(args.head) if args.head else source_sha

        if args.base:
            base_sha = get_git_commit(args.base) or args.base
        elif manifest_base:
            base_sha = manifest_base
        else:
            base_sha = "cdc47a2eee8a427aaa6cf3ab5d01cad7f35d71e1"

        print(f"Diff range:         {base_sha[:7]}..{head_sha[:7]}")
        print(f"Expected Base:      {manifest_base}")
        print(f"Expected Candidate: {manifest_candidate}")

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

        if manifest_candidate:
            cand_match = (head_sha.lower() == manifest_candidate.lower())
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

        # Framework self-integrity
        if not args.skip_framework_check and qualification_files:
            framework_mismatches = []
            for qpath, qmeta in qualification_files.items():
                expected_qhash = qmeta.get("sha256")
                disk_path = os.path.join(PROJECT_ROOT, qpath)
                if not os.path.exists(disk_path):
                    framework_mismatches.append((qpath, expected_qhash, "FILE_MISSING"))
                    continue
                act_qhash = compute_file_sha256(disk_path)
                candidate_hash = compute_git_blob_sha256(head_sha, qpath)
                has_match = (
                    isinstance(expected_qhash, str)
                    and isinstance(candidate_hash, str)
                    and candidate_hash.lower() == expected_qhash.lower()
                    and act_qhash.lower() == expected_qhash.lower()
                )
                if not has_match:
                    framework_mismatches.append((qpath, expected_qhash, act_qhash))

            assertions.append({
                "assertion": "Qualification framework scripts cryptographic self-integrity verified",
                "observed": {"mismatch_count": len(framework_mismatches)},
                "passed": len(framework_mismatches) == 0
            })
            if framework_mismatches:
                for qf, exp, act in framework_mismatches:
                    errors.append(f"UNAUTHORIZED_QUALIFICATION_FRAMEWORK_MODIFICATION: {qf} expected {exp[:10]}... got {act[:10]}...")

        # Dirty working tree check
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

        diff_entries = run_git_diff_status(base_sha, head_sha)
        if diff_entries is None:
            errors.append(f"Unable to execute git diff {base_sha} {head_sha}")
            diff_entries = []

        all_approved = {}
        all_approved.update(approved_files)
        all_approved.update(qualification_files)

        unapproved = []
        diff_paths_seen = set()

        for status_code, path in diff_entries:
            diff_paths_seen.add(path)
            is_manifest_itself = (path in ("scripts/release/release_manifest.json", "scripts/release/administrative_manifest.json"))
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

        hash_mismatches = []
        unapproved_prs = []

        for fpath, meta in approved_files.items():
            expected_hash = meta.get("sha256")
            if not expected_hash:
                errors.append(f"Approved file {fpath} lacks sha256 in manifest.")
                continue

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
                err_msg = f"UNAPPROVED_PR_ATTRIBUTION: File {f} attributed to unapproved or missing PR ({pr_val})"
                print(f"FAIL: {err_msg}")
                errors.append(err_msg)

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
            evidence_paths=[manifest_path],
            errors=errors,
            metadata={
                "mode": mode,
                "base_sha": base_sha,
                "head_sha": head_sha,
                "total_diff_entries": len(diff_entries),
                "approved_files_count": len(approved_files),
                "unapproved_count": len(unapproved),
                "missing_approved_count": len(missing_approved),
                "hash_mismatch_count": len(hash_mismatches),
                "unapproved_pr_count": len(unapproved_prs)
            }
        )
        receipt_path = save_stage_receipt(receipt)
        print(f"Receipt written to: {receipt_path}")

        if status == "PASS":
            is_canonical_run = (
                not args.allow_uncommitted
                and not args.skip_framework_check
                and os.environ.get("ARX_RQ02_SKIP_ADVERSARIAL_RECURSION") != "1"
            )
            if is_canonical_run:
                sub_env = os.environ.copy()
                sub_env["ARX_RQ02_SKIP_ADVERSARIAL_RECURSION"] = "1"
                print("\n>>> [RQ-02-ADV] Executing RQ-02 Cumulative Adversarial Controls...")
                res_adv1 = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "test_rq02_adversarial_controls.py")], cwd=PROJECT_ROOT, env=sub_env)
                if res_adv1.returncode != 0:
                    print("FAIL: RQ-02 Adversarial Controls failed!")
                    sys.exit(1)

                print("\n>>> [Gate H] Executing Gate H 12-Scenario Workflow Attacks...")
                res_adv2 = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "test_d12c_workflow_attacks.py")], cwd=PROJECT_ROOT, env=sub_env)
                if res_adv2.returncode != 0:
                    print("FAIL: Gate H Workflow Attacks failed!")
                    sys.exit(1)

                save_stage_receipt(receipt)

            print(f"PASS: All {len(diff_entries)} changed files strictly match approved cumulative release manifest (0 unapproved).")
            print("=== [RQ-02] RESULT: PASS ===")
            sys.exit(0)
        else:
            print(f"=== [RQ-02] RESULT: FAIL (Errors: {len(errors)}) ===")
            for err in errors[:10]:
                print(f"  - {err}")
            sys.exit(1)

    # =========================================================================
    # TRACK 3: APPLICATION REMEDIATION PR QUALIFICATION
    # =========================================================================
    elif mode == "pr":
        source_sha = identity["source_sha"] if ok else get_git_commit("HEAD")
        head_sha = get_git_commit(args.head) if args.head else source_sha
        pr_number = identity.get("pr_number") if ok else None

        if args.base:
            base_sha = get_git_commit(args.base) or args.base
        elif ok and identity.get("base_sha") and identity.get("event_name") == "pull_request":
            base_sha = identity["base_sha"]
        else:
            target_ref = f"origin/{clean_target}" if get_git_commit(f"origin/{clean_target}") else clean_target
            target_sha = get_git_commit(target_ref) or get_git_commit(clean_target) or "3f3b6c4afda9624eb11cf08d34ddfaebc642b434"
            base_sha = (
                run_git_merge_base(target_sha, head_sha)
                or run_git_merge_base("origin/main", head_sha)
                or target_sha
                or "3f3b6c4afda9624eb11cf08d34ddfaebc642b434"
            )

        print(f"Diff range:         {base_sha[:7]}..{head_sha[:7]}")
        print(f"Candidate HEAD:     {head_sha}")
        print(f"Target PR Number:   {pr_number}")
        print(f"Target Branch:      {clean_target}")

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
                "assertion": "PR base shares verified Git ancestry with candidate HEAD",
                "observed": {"base_sha": base_sha, "head_sha": head_sha, "is_direct_ancestor": anc_ok, "merge_base": mb},
                "passed": lineage_ok
            })
            if not lineage_ok:
                errors.append(f"BASE_LINEAGE_INVALID: Base commit '{base_sha}' is not an ancestor of candidate '{head_sha}'.")

        # Candidate PR Authorization check
        try:
            pr_val = int(pr_number)
            is_approved_pr = pr_val in APPROVED_PRS
        except Exception:
            is_approved_pr = False
        assertions.append({
            "assertion": "Candidate PR is an approved release or remediation workstream",
            "observed": {"pr_number": pr_number, "approved_prs": sorted(list(APPROVED_PRS))},
            "passed": is_approved_pr
        })
        if not is_approved_pr:
            errors.append(f"UNAPPROVED_PR_NUMBER: PR #{pr_number} is not in approved release PR set.")

        # Diff evaluation & security checks
        diff_entries = run_git_diff_status(base_sha, head_sha)
        if diff_entries is None:
            errors.append(f"Unable to execute git diff {base_sha} {head_sha}")
            diff_entries = []

        candidate_diff_paths = {p for _, p in diff_entries}

        # Dirty working tree check
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
                    if fpath in candidate_diff_paths:
                        dirty_candidate_files.append((code, fpath))

                assertions.append({
                    "assertion": "Zero uncommitted changes to candidate application files",
                    "observed": {"dirty_count": len(dirty_candidate_files), "files": dirty_candidate_files},
                    "passed": len(dirty_candidate_files) == 0
                })
                if dirty_candidate_files:
                    for code, df in dirty_candidate_files:
                        errors.append(f"DIRTY_WORKING_TREE_MISMATCH: Candidate file '{df}' has uncommitted modifications ({code})")
            except Exception as e:
                errors.append(f"Unable to check git working tree status: {e}")

        path_traversal_violations = []
        forbidden_modifications = []
        unauthorized_operations = []
        unauthorized_paths = []
        candidate_blobs = {}

        ALLOWED_APP_PREFIXES = ("api/", "analyst_dashboard/", "frontend/", "tests/", "scripts/release/", "docs/")

        for status_code, path in diff_entries:
            # 1. Path traversal check
            if ".." in path or path.startswith("/") or path.startswith("\\") or "\x00" in path:
                path_traversal_violations.append(path)
                errors.append(f"PATH_TRAVERSAL_DETECTED: Malformed or traversing path '{path}' detected in PR diff.")
                continue

            # 2. Scope & Boundary Checks: Strictly Protect Secrets, Production Manifests, Rulesets, and Environment
            if path in FORBIDDEN_ADMIN_FILES or path.startswith(".secrets/") or path.startswith(".github/rulesets/") or path.startswith(".env"):
                forbidden_modifications.append((status_code, path, "PROTECTED_CONFIG_OR_SECRETS"))
                errors.append(f"FORBIDDEN_PROTECTED_PATH_MODIFICATION: Protected file '{path}' cannot be modified in application PR.")
                continue

            # 3. Permitted Operations Check: Reject Unexpected Renames or Deletions of application code
            if status_code.startswith("D") or status_code.startswith("R"):
                unauthorized_operations.append((status_code, path))
                errors.append(f"UNAUTHORIZED_FILE_OPERATION: Operation '{status_code}' on '{path}' is not permitted in remediation PR.")
                continue

            # 4. Allowed directory tree boundaries
            if not any(path.startswith(prefix) for prefix in ALLOWED_APP_PREFIXES):
                unauthorized_paths.append(path)
                errors.append(f"UNAUTHORIZED_DIFF_PATH: File '{path}' is outside authorized application trees.")
                continue

            # 5. Compute cryptographic blob hash directly from Git Merkle tree at candidate HEAD
            blob_hash = compute_git_blob_sha256(head_sha, path)
            if not blob_hash:
                errors.append(f"BLOB_HASH_COMPUTATION_FAILED: Unable to compute git blob hash for '{path}' at {head_sha}.")
            else:
                candidate_blobs[path] = {
                    "operation": status_code,
                    "sha256": blob_hash,
                    "originating_pr_number": pr_number
                }

        assertions.append({
            "assertion": "Zero path traversal or malformed paths detected",
            "observed": {"traversal_count": len(path_traversal_violations), "violations": path_traversal_violations},
            "passed": len(path_traversal_violations) == 0
        })

        assertions.append({
            "assertion": "Zero forbidden modifications to secrets, environment, rulesets, or release manifests",
            "observed": {"forbidden_count": len(forbidden_modifications), "violations": forbidden_modifications},
            "passed": len(forbidden_modifications) == 0
        })

        assertions.append({
            "assertion": "Zero unauthorized file renames or destructive deletions in application tree",
            "observed": {"unauthorized_ops_count": len(unauthorized_operations), "violations": unauthorized_operations},
            "passed": len(unauthorized_operations) == 0
        })

        assertions.append({
            "assertion": "All changed files strictly contained within authorized application boundaries",
            "observed": {"unauthorized_path_count": len(unauthorized_paths), "violations": unauthorized_paths},
            "passed": len(unauthorized_paths) == 0
        })

        assertions.append({
            "assertion": "Cryptographic Merkle blob hashes computed for all candidate files",
            "observed": {"diff_count": len(diff_entries), "blob_count": len(candidate_blobs)},
            "passed": len(candidate_blobs) == len(diff_entries) and len(diff_entries) > 0
        })

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
            target_branch=clean_target,
            assertions=assertions,
            errors=errors,
            metadata={
                "mode": mode,
                "pr_number": pr_number,
                "base_sha": base_sha,
                "head_sha": head_sha,
                "total_diff_entries": len(diff_entries),
                "candidate_blobs": candidate_blobs
            }
        )
        receipt_path = save_stage_receipt(receipt)
        print(f"Receipt written to: {receipt_path}")

        if status == "PASS":
            print(f"PASS: All {len(diff_entries)} changed files strictly match approved application remediation policy (0 unapproved).")
            print("=== [RQ-02] RESULT: PASS ===")
            sys.exit(0)
        else:
            print(f"=== [RQ-02] RESULT: FAIL (Errors: {len(errors)}) ===")
            for err in errors[:10]:
                print(f"  - {err}")
            sys.exit(1)

    # =========================================================================
    # TRACK 2: ADMINISTRATIVE / GOVERNANCE PR QUALIFICATION
    # =========================================================================
    elif mode == "administrative":
        manifest_pr = manifest_data.get("pr_number", 145)
        manifest_base = manifest_data.get("base_commit") or manifest_data.get("base_anchor_commit") or "69422421a9c6af919c28ff940b121930799f58f3"
        manifest_candidate = manifest_data.get("candidate_commit") or manifest_data.get("release_candidate_commit")
        approved_files = manifest_data.get("files", {})
        qualification_files = manifest_data.get("qualification_framework_files", {})

        source_sha = identity["source_sha"] if ok else get_git_commit("HEAD")
        head_sha = (get_git_commit(args.head) if args.head else None) or os.environ.get("ARX_ADMIN_CANDIDATE_SHA") or os.environ.get("ARX_PR_HEAD_SHA") or (source_sha if ok else manifest_candidate)

        if args.base:
            base_sha = get_git_commit(args.base) or args.base
        elif os.environ.get("ARX_PR_BASE_SHA"):
            base_sha = os.environ.get("ARX_PR_BASE_SHA")
        elif manifest_base:
            base_sha = manifest_base
        elif ok and identity.get("base_sha"):
            base_sha = identity["base_sha"]
        else:
            base_sha = "69422421a9c6af919c28ff940b121930799f58f3"

        print(f"Diff range:         {base_sha[:7]}..{head_sha[:7]}")
        print(f"Expected Base:      {manifest_base}")
        print(f"Expected Candidate: {manifest_candidate}")
        print(f"Target PR Number:   {manifest_pr}")

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
                "assertion": "PR base shares verified Git ancestry with candidate HEAD",
                "observed": {"base_sha": base_sha, "head_sha": head_sha, "is_direct_ancestor": anc_ok, "merge_base": mb},
                "passed": lineage_ok
            })
            if not lineage_ok:
                errors.append(f"BASE_LINEAGE_INVALID: Base commit '{base_sha}' is not an ancestor of candidate '{head_sha}'.")

        if manifest_base:
            base_match = (base_sha.lower() == manifest_base.lower()) or is_git_ancestor(manifest_base, base_sha) or is_git_ancestor(base_sha, manifest_base)
            assertions.append({
                "assertion": "PR base matches or shares lineage with administrative manifest base",
                "observed": {"base_sha": base_sha, "expected": manifest_base},
                "passed": base_match
            })
            if not base_match:
                errors.append(f"BASE_ANCHOR_MISMATCH: Evaluated base {base_sha} does not match manifest base {manifest_base}")

        if manifest_candidate:
            cand_match = (head_sha.lower() == manifest_candidate.lower()) or is_git_ancestor(manifest_candidate, head_sha)
            assertions.append({
                "assertion": "PR head incorporates approved administrative candidate commit",
                "observed": {"head_sha": head_sha, "expected": manifest_candidate},
                "passed": cand_match
            })
            if not cand_match:
                errors.append(f"CANDIDATE_SHA_MISMATCH: Evaluated candidate {head_sha} does not match or incorporate manifest candidate {manifest_candidate}")

        # Qualification Framework self-integrity check (if declared)
        if not args.skip_framework_check and qualification_files:
            framework_mismatches = []
            for qpath, qmeta in qualification_files.items():
                expected_qhash = qmeta.get("sha256")
                disk_path = os.path.join(PROJECT_ROOT, qpath)
                if not os.path.exists(disk_path):
                    framework_mismatches.append((qpath, expected_qhash, "FILE_MISSING"))
                    continue
                act_qhash = compute_file_sha256(disk_path)
                candidate_hash = compute_git_blob_sha256(head_sha, qpath)
                has_match = (
                    isinstance(expected_qhash, str)
                    and isinstance(candidate_hash, str)
                    and candidate_hash.lower() == expected_qhash.lower()
                    and act_qhash.lower() == expected_qhash.lower()
                )
                if not has_match:
                    framework_mismatches.append((qpath, expected_qhash, act_qhash))

            assertions.append({
                "assertion": "Administrative qualification framework self-integrity verified",
                "observed": {"mismatch_count": len(framework_mismatches)},
                "passed": len(framework_mismatches) == 0
            })
            if framework_mismatches:
                for qf, exp, act in framework_mismatches:
                    errors.append(f"UNAUTHORIZED_QUALIFICATION_FRAMEWORK_MODIFICATION: {qf} expected {exp[:10]}... got {act[:10]}...")

        # Dirty working tree check
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
                    if fpath in approved_files or fpath == manifest_path or fpath == "scripts/release/administrative_manifest.json":
                        dirty_candidate_files.append((code, fpath))

                assertions.append({
                    "assertion": "Zero uncommitted changes to approved administrative files",
                    "observed": {"dirty_count": len(dirty_candidate_files), "files": dirty_candidate_files},
                    "passed": len(dirty_candidate_files) == 0
                })
                if dirty_candidate_files:
                    for code, df in dirty_candidate_files:
                        errors.append(f"DIRTY_WORKING_TREE_MISMATCH: Administrative file '{df}' has uncommitted modifications ({code})")
            except Exception as e:
                errors.append(f"Unable to check git working tree status: {e}")

        diff_entries = run_git_diff_status(base_sha, head_sha)
        if diff_entries is None:
            errors.append(f"Unable to execute git diff {base_sha} {head_sha}")
            diff_entries = []

        all_approved = {}
        all_approved.update(approved_files)
        all_approved.update(qualification_files)

        diff_paths_seen = set()
        unapproved = []
        forbidden_modifications = []
        path_traversal_violations = []
        unauthorized_operations = []

        for status_code, path in diff_entries:
            diff_paths_seen.add(path)

            # 1. Path Traversal & Malformed Path Check
            if ".." in path or path.startswith("/") or path.startswith("\\") or "\x00" in path:
                path_traversal_violations.append(path)
                errors.append(f"PATH_TRAVERSAL_DETECTED: Malformed or traversing path '{path}' detected in PR diff.")
                continue

            # 2. Scope & Boundary Checks: Strictly Reject Application Code, Production Config, Cumulative Manifest
            if path in FORBIDDEN_ADMIN_FILES:
                forbidden_modifications.append((status_code, path, "PRODUCTION_CONFIG_OR_RELEASE_MANIFEST"))
                errors.append(f"FORBIDDEN_APPLICATION_OR_CONFIG_MODIFICATION: Forbidden production/manifest file '{path}' in administrative PR.")
                continue

            if any(path.startswith(prefix) for prefix in FORBIDDEN_ADMIN_PREFIXES):
                forbidden_modifications.append((status_code, path, "APPLICATION_SOURCE_TREE"))
                errors.append(f"FORBIDDEN_APPLICATION_OR_CONFIG_MODIFICATION: Application code path '{path}' cannot be modified in administrative PR.")
                continue

            if path.startswith(".env") or path.startswith("Dockerfile"):
                forbidden_modifications.append((status_code, path, "ENVIRONMENT_CONFIG"))
                errors.append(f"FORBIDDEN_APPLICATION_OR_CONFIG_MODIFICATION: Production configuration file '{path}' cannot be modified in administrative PR.")
                continue

            # 3. Permitted Operations Check: Reject Unexpected Renames or Deletions
            if status_code.startswith("D") or status_code.startswith("R"):
                meta = approved_files.get(path, {})
                op = meta.get("operation", "")
                if not op or not status_code.startswith(op):
                    unauthorized_operations.append((status_code, path))
                    errors.append(f"UNAUTHORIZED_FILE_OPERATION: Operation '{status_code}' on '{path}' is not permitted in administrative PR.")
                    continue

            # 4. Unapproved File Check
            is_manifest_itself = (path in ("scripts/release/administrative_manifest.json", "scripts/release/release_manifest.json"))
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
            "assertion": "Zero path traversal or malformed paths detected",
            "observed": {"traversal_count": len(path_traversal_violations), "violations": path_traversal_violations},
            "passed": len(path_traversal_violations) == 0
        })

        assertions.append({
            "assertion": "Zero forbidden application or configuration modifications in administrative scope",
            "observed": {"forbidden_count": len(forbidden_modifications), "violations": forbidden_modifications},
            "passed": len(forbidden_modifications) == 0
        })

        assertions.append({
            "assertion": "Zero unauthorized renames or deletions",
            "observed": {"unauthorized_ops_count": len(unauthorized_operations), "violations": unauthorized_operations},
            "passed": len(unauthorized_operations) == 0
        })

        assertions.append({
            "assertion": "Zero unapproved files in administrative PR diff",
            "observed": {"unapproved_count": len(unapproved), "unapproved_files": unapproved},
            "passed": len(unapproved) == 0
        })
        if unapproved:
            print(f"FAIL: Detected {len(unapproved)} unapproved files in administrative diff:")
            for code, uf in unapproved:
                print(f"  [UNAPPROVED ({code})] {uf}")
                errors.append(f"Unapproved file in diff ({code}): {uf}")

        # Completeness Check
        missing_approved = []
        for fpath in approved_files.keys():
            if fpath not in diff_paths_seen:
                missing_approved.append(fpath)

        assertions.append({
            "assertion": "All approved administrative files present in PR diff",
            "observed": {"missing_count": len(missing_approved), "missing_files": missing_approved},
            "passed": len(missing_approved) == 0
        })
        if missing_approved:
            for mf in missing_approved:
                errors.append(f"MISSING_APPROVED_DIFF_ENTRY: Manifest specifies approved file '{mf}' which is missing from PR diff")

        # Cryptographic Hash & PR Attribution Check
        hash_mismatches = []
        unapproved_prs = []

        for fpath, meta in approved_files.items():
            expected_hash = meta.get("sha256")
            if not expected_hash:
                errors.append(f"Approved administrative file {fpath} lacks sha256 in manifest.")
                continue

            orig_pr = meta.get("originating_pr_number") or meta.get("pr")
            if orig_pr is not None:
                try:
                    pr_num = int(orig_pr)
                    if pr_num != int(manifest_pr):
                        unapproved_prs.append((fpath, pr_num))
                except Exception:
                    unapproved_prs.append((fpath, str(orig_pr)))
            else:
                unapproved_prs.append((fpath, "MISSING_PR_ATTRIBUTION"))

            blob_hash = compute_git_blob_sha256(head_sha, fpath)
            disk_path = os.path.join(PROJECT_ROOT, fpath)
            disk_hash = compute_file_sha256(disk_path) if os.path.exists(disk_path) else "NOT_ON_DISK"

            target_hash = blob_hash if blob_hash else disk_hash
            if not target_hash or target_hash.lower() != expected_hash.lower():
                hash_mismatches.append((fpath, expected_hash, target_hash))

        assertions.append({
            "assertion": "Administrative file content hashes strictly match manifest",
            "observed": {"mismatch_count": len(hash_mismatches)},
            "passed": len(hash_mismatches) == 0
        })
        if hash_mismatches:
            for f, exp, act in hash_mismatches:
                errors.append(f"Hash mismatch in {f}: expected {exp[:10]} got {act[:10]}")

        assertions.append({
            "assertion": "All administrative changes strictly attributed to target PR",
            "observed": {"unapproved_pr_count": len(unapproved_prs), "unapproved_prs": unapproved_prs},
            "passed": len(unapproved_prs) == 0
        })
        if unapproved_prs:
            for f, pr_val in unapproved_prs:
                err_msg = f"UNAPPROVED_PR_ATTRIBUTION: File {f} attributed to unapproved or missing PR ({pr_val})"
                print(f"FAIL: {err_msg}")
                errors.append(err_msg)

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
            target_branch=clean_target,
            assertions=assertions,
            evidence_paths=[manifest_path],
            errors=errors,
            metadata={
                "mode": mode,
                "pr_number": manifest_pr,
                "base_sha": base_sha,
                "head_sha": head_sha,
                "total_diff_entries": len(diff_entries),
                "approved_files_count": len(approved_files),
                "unapproved_count": len(unapproved),
                "missing_approved_count": len(missing_approved),
                "hash_mismatch_count": len(hash_mismatches),
                "unapproved_pr_count": len(unapproved_prs)
            }
        )
        receipt_path = save_stage_receipt(receipt)
        print(f"Receipt written to: {receipt_path}")

        if status == "PASS":
            is_canonical_run = (
                not args.allow_uncommitted
                and not args.skip_framework_check
                and os.environ.get("ARX_RQ02_SKIP_ADVERSARIAL_RECURSION") != "1"
            )
            if is_canonical_run:
                sub_env = os.environ.copy()
                sub_env["ARX_RQ02_SKIP_ADVERSARIAL_RECURSION"] = "1"
                print("\n>>> [RQ-02-ADMIN-ADV] Executing RQ-02 Administrative Adversarial Controls...")
                res_adv = subprocess.run(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "test_rq02_administrative_controls.py")],
                    cwd=PROJECT_ROOT,
                    env=sub_env
                )
                if res_adv.returncode != 0:
                    print("FAIL: RQ-02 Administrative Adversarial Controls failed!")
                    sys.exit(1)

                save_stage_receipt(receipt)

            print(f"PASS: All {len(diff_entries)} administrative changes strictly match approved administrative manifest (0 unapproved).")
            print("=== [RQ-02] RESULT: PASS ===")
            sys.exit(0)
        else:
            print(f"=== [RQ-02] RESULT: FAIL (Errors: {len(errors)}) ===")
            for err in errors[:10]:
                print(f"  - {err}")
            sys.exit(1)

if __name__ == "__main__":
    main()
