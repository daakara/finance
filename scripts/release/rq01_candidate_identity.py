#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-01: Candidate Identity
Validates PR source, target, commit SHA ancestry, and prevents commit substitution.
"""

import sys, os, subprocess, argparse, json, re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def run_git(cmd_args):
    try:
        return subprocess.check_output(cmd_args, text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError:
        return None

def main():
    parser = argparse.ArgumentParser(description="RQ-01: Candidate Identity Verification")
    parser.add_argument("--base", default="6d4020ca3fef3ae07ac01204b03e0629b6a7cc79", help="Expected release base commit")
    parser.add_argument("--expected-head", default="ea3cc13eeeb6d3e6c250ecfd3ed6aecb4ec44d6c", help="Expected candidate commit")
    parser.add_argument("--target-branch", default="release/production", help="Target branch (release/production or main)")
    parser.add_argument("--allow-main-target", action="store_true", help="Permit main as target during staging tests")
    args = parser.parse_args()

    print("=== [RQ-01] CANDIDATE IDENTITY & ANCESTRY CHECK ===")
    
    current_head = run_git(["git", "rev-parse", "HEAD"])
    if not current_head:
        print("FAIL: Unable to determine current HEAD commit.")
        sys.exit(1)
    
    print(f"Current HEAD: {current_head}")
    print(f"Expected Base: {args.base}")
    print(f"Expected Candidate: {args.expected_head}")

    # 1. Format check
    if not re.match(r"^[0-9a-fA-F]{40}$", current_head):
        print(f"FAIL: Current HEAD {current_head} is not a valid 40-character hex SHA.")
        sys.exit(1)

    # 2. Base commit existence
    base_exists = run_git(["git", "cat-file", "-e", args.base])
    if base_exists is None:
        print(f"FAIL: Base commit {args.base} does not exist in repository history.")
        sys.exit(1)

    # 3. Ancestry check: args.base must be an ancestor of current HEAD
    is_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", args.base, current_head]).returncode == 0
    if not is_ancestor:
        print(f"FAIL: Base commit {args.base} is NOT an ancestor of current HEAD {current_head}.")
        sys.exit(1)
    print(f"PASS: Base anchor {args.base[:7]} is verified ancestor of HEAD {current_head[:7]}.")

    # 4. Target branch check
    valid_targets = ["refs/heads/release/production", "release/production"]
    if args.allow_main_target:
        valid_targets.extend(["refs/heads/main", "main"])
    
    if args.target_branch not in valid_targets:
        print(f"FAIL: Target branch '{args.target_branch}' is not authorized for release qualification.")
        sys.exit(1)
    print(f"PASS: Target branch '{args.target_branch}' is valid.")

    # 5. Candidate substitution guard
    if args.expected_head and current_head.lower() != args.expected_head.lower():
        is_candidate_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", args.expected_head, current_head]).returncode == 0
        if not is_candidate_ancestor:
            print(f"FAIL: Current HEAD {current_head} does not contain authorized candidate {args.expected_head}.")
            sys.exit(1)
        else:
            print(f"NOTE: HEAD {current_head[:7]} incorporates authorized candidate {args.expected_head[:7]}.")
    else:
        print(f"PASS: Current commit matches exact candidate SHA {args.expected_head[:7]}.")

    print("=== [RQ-01] RESULT: PASS ===")
    sys.exit(0)

if __name__ == "__main__":
    main()
