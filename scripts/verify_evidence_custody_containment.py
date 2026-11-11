#!/usr/bin/env python3
"""
ARX Terminal — Evidence Custody Repository Containment Verifier (Gate A)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
Mandate:
  - Verify .gitignore exclusions for sensitive research evidence and keys
  - Verify that no sensitive observer submissions, identities, or keys are tracked
  - Verify that canonical public fixtures and policies remain intact and tracked
  - Fail closed if any sensitive evidence is staged or committed in Git
"""

import sys
import os
import subprocess
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patterns that MUST NEVER be committed to Git
FORBIDDEN_TRACKED_PATTERNS = [
    re.compile(r"^tests/fixtures/qualification/submissions/.*\.json$", re.IGNORECASE),
    re.compile(r".*/qualification/.*identities/.*", re.IGNORECASE),
    re.compile(r".*\.vault/.*", re.IGNORECASE),
    re.compile(r".*\.custody/.*", re.IGNORECASE),
    re.compile(r".*custody_secrets.*", re.IGNORECASE),
    re.compile(r".*observer_identity.*\.json$", re.IGNORECASE),
    re.compile(r".*observer_keys.*", re.IGNORECASE),
    re.compile(r".*qualification.*receipts.*\.jsonl$", re.IGNORECASE),
    re.compile(r".*OBS-HUMAN.*", re.IGNORECASE),
]

# Patterns that MUST remain tracked/accessible
REQUIRED_CANONICAL_PATHS = [
    "tests/fixtures/qualification/submissions/SUBMISSION_LOCKBOX_POLICY.md",
    "tests/fixtures/qualification/custodian_package/PACKAGE_INTEGRITY_MANIFEST.json",
    "tests/fixtures/qualification/custodian_package/gold_key_consolidated_30.json",
    "tests/fixtures/qualification/custodian_package/scoring_utility.py",
    "tests/fixtures/qualification/observer_package/OBSERVER_RATING_SCHEMA.json",
    "tests/fixtures/qualification/observer_package/ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md",
]


def check_gitignore_rules() -> Tuple[bool, List[str]]:
    """Verify that .gitignore contains mandatory custody exclusion patterns."""
    gitignore_path = REPO_ROOT / ".gitignore"
    if not gitignore_path.exists():
        return False, ["Missing .gitignore file in repository root"]

    content = gitignore_path.read_text(encoding="utf-8")
    required_rules = [
        "tests/fixtures/qualification/submissions/*.json",
        "tests/fixtures/qualification/submissions/vault/",
        "tests/fixtures/qualification/submissions/identities/",
        "tests/fixtures/qualification/submissions/*.key",
        "tests/fixtures/qualification/submissions/*receipts*.jsonl",
        "*.vault/",
        ".vault/",
        "*custody_secrets*",
        "*identity_mapping*.json",
        "!tests/fixtures/qualification/submissions/SUBMISSION_LOCKBOX_POLICY.md",
    ]

    missing = []
    for rule in required_rules:
        if rule not in content:
            missing.append(rule)

    return (len(missing) == 0), missing


def check_tracked_files() -> Tuple[bool, List[str]]:
    """Verify that git ls-files does not track any forbidden custody paths."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        return False, [f"Failed to execute git ls-files: {e}"]

    violations = []
    for file_path in tracked_files:
        norm_path = file_path.replace("\\", "/")
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if pattern.search(norm_path):
                # Ensure exception for allowed files
                if norm_path == "tests/fixtures/qualification/submissions/SUBMISSION_LOCKBOX_POLICY.md":
                    continue
                violations.append(f"Tracked forbidden custody file: {norm_path}")

    return (len(violations) == 0), violations


def check_staged_files() -> Tuple[bool, List[str]]:
    """Verify that git diff --cached does not stage any forbidden custody files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        staged_files = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        return False, [f"Failed to execute git diff --cached: {e}"]

    violations = []
    for file_path in staged_files:
        norm_path = file_path.replace("\\", "/")
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if pattern.search(norm_path):
                if norm_path == "tests/fixtures/qualification/submissions/SUBMISSION_LOCKBOX_POLICY.md":
                    continue
                violations.append(f"Staged forbidden custody file: {norm_path}")

    return (len(violations) == 0), violations


def check_git_history_leaks() -> Tuple[bool, List[str]]:
    """Scan git commit history for accidental human evidence or key commits."""
    check_paths = [
        "tests/fixtures/qualification/submissions/*.json",
        "*observer_identity*.json",
        "*custody_secrets*",
    ]
    violations = []
    for path_glob in check_paths:
        try:
            result = subprocess.run(
                ["git", "log", "--all", "--full-history", "-n", "5", "--", path_glob],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                violations.append(f"Historical commits detected for pattern '{path_glob}'")
        except subprocess.CalledProcessError as e:
            violations.append(f"Git log check failed for '{path_glob}': {e}")

    return (len(violations) == 0), violations


def verify_canonical_fixtures_present() -> Tuple[bool, List[str]]:
    """Verify that canonical public research artifacts and policies remain present."""
    missing = []
    for req_path in REQUIRED_CANONICAL_PATHS:
        full_path = REPO_ROOT / req_path
        if not full_path.exists():
            missing.append(f"Missing canonical fixture or policy: {req_path}")
    return (len(missing) == 0), missing


def verify_test_forced_add_containment() -> Tuple[bool, List[str]]:
    """Test that git check-ignore rejects forbidden files."""
    test_paths = [
        "tests/fixtures/qualification/submissions/OBS-TEST-99.json",
        "tests/fixtures/qualification/submissions/vault/ciphertext.enc",
        "tests/fixtures/qualification/submissions/staging/temp.json",
        "tests/fixtures/qualification/submissions/identities/map.json",
        "tests/fixtures/qualification/submissions/custody.db",
        "tests/fixtures/qualification/submissions/custody_receipts.jsonl",
    ]
    failures = []
    for test_p in test_paths:
        try:
            res = subprocess.run(
                ["git", "check-ignore", test_p],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                failures.append(f"Path '{test_p}' is NOT ignored by git")
        except Exception as e:
            failures.append(f"Error checking ignore for '{test_p}': {e}")

    return (len(failures) == 0), failures


def main() -> int:
    print("=" * 78)
    print("ARX EVIDENCE CUSTODY REPOSITORY CONTAINMENT VERIFIER (GATE A)")
    print("=" * 78)

    all_passed = True

    # Check 1: .gitignore rules
    ok, details = check_gitignore_rules()
    if ok:
        print("[PASS] Gate A.1: .gitignore contains mandatory custody exclusion rules.")
    else:
        print(f"[FAIL] Gate A.1: Missing .gitignore rules: {details}")
        all_passed = False

    # Check 2: Tracked files
    ok, details = check_tracked_files()
    if ok:
        print("[PASS] Gate A.2: No forbidden custody files or identities tracked in Git.")
    else:
        print(f"[FAIL] Gate A.2: Forbidden tracked files detected: {details}")
        all_passed = False

    # Check 3: Staged files
    ok, details = check_staged_files()
    if ok:
        print("[PASS] Gate A.3: No forbidden custody files staged in Git index.")
    else:
        print(f"[FAIL] Gate A.3: Forbidden staged files detected: {details}")
        all_passed = False

    # Check 4: Git history leak scan
    ok, details = check_git_history_leaks()
    if ok:
        print("[PASS] Gate A.4: Zero historical commits matching custody-sensitive patterns.")
    else:
        print(f"[FAIL] Gate A.4: Historical leak detected: {details}")
        all_passed = False

    # Check 5: Canonical fixtures presence
    ok, details = verify_canonical_fixtures_present()
    if ok:
        print("[PASS] Gate A.5: All required canonical fixtures and policies present.")
    else:
        print(f"[FAIL] Gate A.5: Missing canonical files: {details}")
        all_passed = False

    # Check 6: Check-ignore behavior
    ok, details = verify_test_forced_add_containment()
    if ok:
        print("[PASS] Gate A.6: Git check-ignore correctly blocks all test custody paths.")
    else:
        print(f"[FAIL] Gate A.6: Git check-ignore failed: {details}")
        all_passed = False

    print("-" * 78)
    if all_passed:
        print("VERDICT: ARX_UX_GIT_EXPOSURE_CONTAINMENT_VERIFIED")
        print("=" * 78)
        return 0
    else:
        print("VERDICT: ARX_UX_GIT_EXPOSURE_CONTAINMENT_FAILED")
        print("=" * 78)
        return 1


if __name__ == "__main__":
    sys.exit(main())
