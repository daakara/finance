"""Trusted-base PR admission gate for pull_request_target and issue_comment.

This script is only run from the base/default branch checked out by a
pull_request_target/issue_comment workflow, NEVER from candidate PR code.
It reads GitHub API metadata (repository, PR, files and owner comment) without
checking out, importing, installing or executing any code from the candidate.
"""
import json
import os
import re
import sys

from rq02_owner_admission import _github_json, verify_owner_admission

ALLOWED_PATHS = ("api/", "analyst_dashboard/", "frontend/", "tests/", "docs/")
PROTECTED = (
    "scripts/release/", ".github/", ".secrets/", ".env", "Dockerfile",
    "migrations/",
)
PROTECTED_EXACT = ("railway.json", "wrangler.toml")
MAX_PAGES = 10


def validate_files(files):
    if not files:
        return False, "EMPTY_CANDIDATE_DIFF"
    seen = set()
    for item in files:
        path = item.get("filename")
        status = item.get("status")
        if not isinstance(path, str) or not path or path in seen:
            return False, "DUPLICATE_OR_INVALID_DIFF_PATH"
        seen.add(path)
        if ".." in path.split("/") or path.startswith(("/", "\\")) or "\x00" in path:
            return False, "MALFORMED_DIFF_PATH"
        if path in PROTECTED_EXACT or path.startswith(PROTECTED):
            return False, "PROTECTED_PATH_MODIFICATION:" + path
        if not path.startswith(ALLOWED_PATHS):
            return False, "OUTSIDE_APPLICATION_BOUNDARY:" + path
        if status not in ("added", "modified"):
            return False, "DESTRUCTIVE_OR_UNEXPECTED_OPERATION:" + path
    return True, "SCOPED_APPLICATION_DIFF"


def get_all_files(repo, number):
    files = []
    for page in range(1, MAX_PAGES + 1):
        part = _github_json(
            f"https://api.github.com/repos/{repo}/pulls/{number}/files?per_page=100&page={page}"
        )
        if not isinstance(part, list):
            return None
        files.extend(part)
        if len(part) < 100:
            return files
    return None  # fail closed on truncated inventory


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        print("FAIL: Only GitHub-hosted execution is accepted")
        return 1
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not path:
        print("FAIL: Invalid GitHub workflow identity")
        return 1
    try:
        with open(path, encoding="utf-8") as stream:
            event = json.load(stream)
        pr = event.get("pull_request")
        if pr is None:
            issue = event.get("issue", {})
            if not issue.get("pull_request"):
                print("SKIP: Not a pull request")
                return 0
            number = issue["number"]
            pr = _github_json(f"https://api.github.com/repos/{repo}/pulls/{number}")
        number = pr.get("number")
        sha = (pr.get("head") or {}).get("sha")
        base = (pr.get("base") or {}).get("ref")
        if base != "main":
            print("SKIP: No main-target application admission on this event")
            return 0
        if not isinstance(number, int) or not isinstance(sha, str):
            print("FAIL: PR identity missing")
            return 1
        # Always obtain live PR metadata, not potentially stale event state.
        live = _github_json(f"https://api.github.com/repos/{repo}/pulls/{number}")
        if (live.get("head") or {}).get("sha") != sha or (live.get("base") or {}).get("ref") != base:
            print("FAIL: PR head/base changed since event")
            return 1
        changed = get_all_files(repo, number)
        if changed is None:
            print("FAIL: Candidate diff pagination limit or API error")
            return 1
        okay, reason = validate_files(changed)
        if not okay:
            print("FAIL: " + reason)
            return 1
        admitted, evidence = verify_owner_admission(
            pr_number=number, head_sha=sha, target=base
        )
        if not admitted:
            print("FAIL: GitHub owner admission absent or invalid: " + evidence)
            return 1
        print(f"PASS: owner-approved PR #{number} exact SHA {sha}; {len(changed)} files; {evidence}")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("FAIL: Trusted metadata evaluation failed: " + type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
