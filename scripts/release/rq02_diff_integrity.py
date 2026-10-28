#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-02: Release Diff Integrity
Verifies the changed-file inventory against the approved release candidate manifest.
Rejects unapproved runtime, configuration, dependency, infrastructure, or qualification script changes.
Eliminates blanket exemptions and shell injection vectors.
"""

import sys, os, subprocess, argparse, json, hashlib, time, datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, compute_file_sha256, get_execution_context, get_git_commit

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

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    parser = argparse.ArgumentParser(description="RQ-02: Release Diff Integrity Verification")
    parser.add_argument("--base", default="6d4020ca3fef3ae07ac01204b03e0629b6a7cc79", help="Release base anchor")
    parser.add_argument("--head", default=None, help="Candidate commit to evaluate")
    parser.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "release_manifest.json"), help="Approved release manifest")
    args = parser.parse_args()

    head_sha = args.head or get_git_commit("HEAD")

    print("=== [RQ-02] RELEASE DIFF INTEGRITY CHECK ===")
    print(f"Diff range: {args.base[:7]}..{head_sha[:7]}")
    print(f"Manifest: {args.manifest}")

    if not os.path.exists(args.manifest):
        errors.append(f"Release manifest not found at {args.manifest}")
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-02", "Release Diff Integrity", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       head_sha, "release/production", assertions, errors=errors)
        save_stage_receipt(receipt)
        print(f"FAIL: Release manifest not found at {args.manifest}")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        try:
            manifest_data = json.load(f)
        except Exception as e:
            errors.append(f"Manifest JSON parsing error: {e}")
            manifest_data = {}

    approved_files = manifest_data.get("files", {})
    qualification_files = manifest_data.get("qualification_framework_files", {})
    all_approved = {}
    all_approved.update(approved_files)
    all_approved.update(qualification_files)

    diff_entries = run_git_diff_status(args.base, head_sha)
    if diff_entries is None:
        errors.append(f"Unable to execute git diff {args.base} {head_sha}")
        status = "FAIL"
        duration = time.time() - t0
        receipt = create_stage_receipt("RQ-02", "Release Diff Integrity", status, start_time,
                                       datetime.datetime.now(datetime.timezone.utc).isoformat(), duration,
                                       head_sha, "release/production", assertions, errors=errors)
        save_stage_receipt(receipt)
        sys.exit(1)

    print(f"Total diff entries detected: {len(diff_entries)}")
    print(f"Approved candidate files in manifest: {len(approved_files)}")
    print(f"Approved qualification files in manifest: {len(qualification_files)}")

    unapproved = []
    renamed = []
    deleted = []
    modified = []
    added = []

    for status_code, path in diff_entries:
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

    # Verify content hashes for approved candidate files against target candidate commit
    hash_mismatches = []
    for fpath, meta in approved_files.items():
        expected_hash = meta.get("sha256")
        if not expected_hash:
            errors.append(f"Approved file {fpath} lacks sha256 in manifest.")
            continue
        try:
            # Check file blob at HEAD or local working tree
            local_hash = compute_file_sha256(os.path.join(PROJECT_ROOT, fpath))
            if local_hash.lower() != expected_hash.lower():
                hash_mismatches.append((fpath, expected_hash, local_hash))
        except Exception as e:
            hash_mismatches.append((fpath, expected_hash, str(e)))

    assertions.append({
        "assertion": "Candidate file content hashes match manifest",
        "observed": {"mismatch_count": len(hash_mismatches)},
        "passed": len(hash_mismatches) == 0
    })

    if hash_mismatches:
        print(f"FAIL: Hash mismatch in {len(hash_mismatches)} approved files:")
        for f, exp, act in hash_mismatches:
            print(f"  [HASH MISMATCH] {f}: expected {exp[:10]}... got {act[:10]}...")
            errors.append(f"Hash mismatch in {f}: expected {exp[:10]} got {act[:10]}")

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
            "total_diff_entries": len(diff_entries),
            "added_count": len(added),
            "modified_count": len(modified),
            "deleted_count": len(deleted),
            "renamed_count": len(renamed)
        }
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        print(f"PASS: All changed files strictly match approved release manifest (0 unapproved).")
        print("=== [RQ-02] RESULT: PASS ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-02] RESULT: FAIL (Errors: {errors}) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
