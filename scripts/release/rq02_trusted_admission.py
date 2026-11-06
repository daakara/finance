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
import urllib.request

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


def _evaluate_main():
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



# This distinct commit-status context is the ruleset-required gate. GitHub's
# issue_comment workflow jobs do not themselves satisfy PR required checks.
OWNER_STATUS_CONTEXT = "ARX Owner Admission Attestation"


def _event_head():
    """Resolve a target PR head from trusted GitHub workflow payload."""
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not event_path:
        return None
    with open(event_path, encoding="utf-8") as handle:
        event = json.load(handle)
    pr = event.get("pull_request")
    if pr is None:
        issue = event.get("issue") or {}
        if not issue.get("pull_request"):
            return None
        pr = _github_json(f"https://api.github.com/repos/{repo}/pulls/{issue['number']}")
    if (pr.get("base") or {}).get("ref") != "main":
        return None
    sha = (pr.get("head") or {}).get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid PR head identity for commit status")
    return repo, sha


def post_commit_status(repo, sha, state, description):
    """Publish a SHA-specific required commit status using trusted base code.

    Uses only the GitHub token's statuses:write permission. The candidate PR
    is never checked out or executed in this workflow.
    """
    if state not in ("pending", "success", "failure"):
        raise ValueError("Unsupported owner admission status")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Untrusted status SHA")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("Trusted GitHub status token unavailable")
    data = {
        "state": state,
        "context": OWNER_STATUS_CONTEXT,
        "description": description[:130],
    }
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id and run_id.isdigit():
        data["target_url"] = f"https://github.com/{repo}/actions/runs/{run_id}"
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/statuses/{sha}",
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "User-Agent": "arx-rq02-trusted-owner-status",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        if response.status != 201:
            raise OSError("Commit status API did not accept the attestation")


def main():
    """Emit pending before evaluation, then exact-head success/failure."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return _evaluate_main()
    try:
        head = _event_head()
        if head is None:
            return _evaluate_main()
        repo, sha = head
        post_commit_status(repo, sha, "pending", "Owner admission verification in progress")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("FAIL: Could not initialize owner admission commit status: " + type(exc).__name__)
        return 1

    result = _evaluate_main()
    try:
        post_commit_status(
            repo, sha, "success" if result == 0 else "failure",
            "Verified owner admission and bounded diff" if result == 0 else "Owner admission missing or scope rejected",
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("FAIL: Could not publish owner admission result: " + type(exc).__name__)
        return 1
    return result


if __name__ == "__main__":
    sys.exit(main())
