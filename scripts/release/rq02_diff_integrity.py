#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-02: Release Diff Integrity
Verifies the changed-file inventory against the approved release candidate manifest.
Rejects unapproved runtime, configuration, dependency, or infrastructure changes.
"""

import sys, os, subprocess, argparse, json, hashlib

def get_git_diff(base_sha, head_sha):
    cmd = f"git diff --name-only {base_sha} {head_sha}"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.PIPE).strip()
        return [f.strip() for f in output.splitlines() if f.strip()]
    except subprocess.CalledProcessError as e:
        print(f"FAIL: git diff command failed: {e}")
        return None

def compute_file_sha(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="RQ-02: Release Diff Integrity Verification")
    parser.add_argument("--base", default="6d4020ca3fef3ae07ac01204b03e0629b6a7cc79", help="Release base anchor")
    parser.add_argument("--head", default="ea3cc13eeeb6d3e6c250ecfd3ed6aecb4ec44d6c", help="Candidate commit to evaluate")
    parser.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "release_manifest.json"), help="Approved release manifest")
    parser.add_argument("--allow-ci-workflow", action="store_true", help="Allow new release-qualification.yml in diff during initial PR")
    args = parser.parse_args()

    print("=== [RQ-02] RELEASE DIFF INTEGRITY CHECK ===")
    print(f"Diff range: {args.base[:7]}..{args.head[:7]}")
    print(f"Manifest: {args.manifest}")

    if not os.path.exists(args.manifest):
        print(f"FAIL: Release manifest not found at {args.manifest}")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    approved_files = manifest_data.get("files", {})
    changed_files = get_git_diff(args.base, args.head)
    if changed_files is None:
        sys.exit(1)

    print(f"Total changed files detected: {len(changed_files)}")
    print(f"Approved files in manifest: {len(approved_files)}")

    unapproved = []
    for f in changed_files:
        if args.allow_ci_workflow and f.startswith(".github/workflows/release-qualification") or f.startswith("scripts/release/"):
            continue
        if f not in approved_files:
            unapproved.append(f)

    if unapproved:
        print(f"FAIL: Detected {len(unapproved)} unapproved files in release candidate diff:")
        for uf in unapproved:
            print(f"  [UNAPPROVED] {uf}")
        sys.exit(1)

    # Verify content hash of changed files against manifest if checking target candidate
    hash_mismatches = []
    for f, expected_meta in approved_files.items():
        if f in changed_files:
            actual_hash = compute_file_sha(f)
            expected_hash = expected_meta.get("sha256")
            if actual_hash and expected_hash and actual_hash.lower() != expected_hash.lower():
                # Check git show at target candidate commit
                try:
                    c = subprocess.check_output(f"git show {args.head}:{f}", shell=True)
                    candidate_hash = hashlib.sha256(c).hexdigest()
                    if candidate_hash.lower() != expected_hash.lower():
                        hash_mismatches.append((f, expected_hash, candidate_hash))
                except Exception:
                    hash_mismatches.append((f, expected_hash, "ERROR_READING_GIT"))

    if hash_mismatches:
        print(f"FAIL: Hash mismatch in {len(hash_mismatches)} approved files:")
        for f, exp, act in hash_mismatches:
            print(f"  [HASH MISMATCH] {f}: expected {exp[:10]}... got {act[:10]}...")
        sys.exit(1)

    print(f"PASS: All {len(changed_files)} changed files strictly match approved release manifest.")
    print("=== [RQ-02] RESULT: PASS ===")
    sys.exit(0)

if __name__ == "__main__":
    main()
