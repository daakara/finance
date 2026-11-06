"""RQ-02 owner admission from GitHub-authenticated, SHA-bound PR comments.

No static PR-number list is required for future main-target application PRs.
Approval MUST be a GitHub comment authored by the repository owner's account
on the exact PR and bound to its current 40-character head SHA. The RQ-02
caller separately enforces changed-file scope and protected-path restrictions.

This module does not grant authorization to production/release branch PRs.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Tuple

APPROVAL = re.compile(
    r"\AARX-RQ02-APPROVE v1 pr=([1-9][0-9]*) sha=([0-9a-f]{40}) base=main\Z"
)
SHA = re.compile(r"\A[0-9a-f]{40}\Z")
MAX_AGE = datetime.timedelta(days=7)
TIMEOUT_SECONDS = 10


def _github_json(url: str) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "arx-rq02-owner-admission"}
    # Public repository GitHub metadata is readable without auth.
    # Optional read-only token reduces anonymous rate limits; never log it.
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def _time(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def evaluate_owner_comment(
    comment: Dict[str, Any], *, owner_id: int, pr_number: int,
    head_sha: str, now: datetime.datetime,
) -> Tuple[bool, str]:
    """Strictly parse GitHub's verified author identity and SHA-bound approval."""
    m = APPROVAL.fullmatch(str(comment.get("body") or "").strip())
    if not m or int(m.group(1)) != pr_number or m.group(2) != head_sha:
        return False, "APPROVAL_NOT_SHA_BOUND"
    user = comment.get("user")
    if not isinstance(user, dict) or user.get("id") != owner_id or user.get("type") != "User":
        return False, "NOT_REPOSITORY_OWNER"
    if not isinstance(comment.get("id"), int) or not isinstance(comment.get("html_url"), str):
        return False, "UNATTESTED_COMMENT_IDENTITY"
    try:
        created = _time(comment["created_at"])
        updated = _time(comment["updated_at"])
    except (ValueError, KeyError, TypeError):
        return False, "UNATTESTED_COMMENT_TIMESTAMP"
    if created.tzinfo is None or updated.tzinfo is None:
        return False, "UNATTESTED_COMMENT_TIMESTAMP"
    if created > now or updated > now or now - updated > MAX_AGE:
        return False, "EXPIRED_OR_FUTURE_APPROVAL"
    if updated < created:
        return False, "INVALID_APPROVAL_EDIT_HISTORY"
    return True, str(comment["html_url"])


def verify_owner_admission(*, pr_number: int, head_sha: str, target: str) -> Tuple[bool, str]:
    """Read authoritative GitHub repository, PR and owner-comment metadata."""
    if target != "main" or pr_number < 1 or not SHA.fullmatch(head_sha):
        return False, "INVALID_MAIN_CANDIDATE_IDENTITY"
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return False, "NON_HOSTED_OWNER_ADMISSION_DISALLOWED"
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        return False, "UNTRUSTED_REPOSITORY_CONTEXT"
    api = "https://api.github.com/repos/" + repo
    try:
        metadata = _github_json(api)
        pr = _github_json(api + "/pulls/" + str(pr_number))
        if not isinstance(metadata, dict) or not isinstance(pr, dict):
            return False, "GITHUB_METADATA_UNAVAILABLE"
        owner_id = (metadata.get("owner") or {}).get("id")
        if not isinstance(owner_id, int) or (metadata.get("full_name") or "").lower() != repo.lower():
            return False, "REPOSITORY_OWNER_UNVERIFIED"
        if pr.get("number") != pr_number or pr.get("state") != "open":
            return False, "PR_IDENTITY_MISMATCH"
        if (pr.get("base") or {}).get("ref") != "main":
            return False, "PR_TARGET_MISMATCH"
        if (pr.get("head") or {}).get("sha") != head_sha:
            return False, "PR_HEAD_CHANGED_SINCE_APPROVAL"
        if (pr.get("head") or {}).get("repo", {}).get("full_name", "").lower() != repo.lower():
            return False, "CROSS_REPOSITORY_HEAD_DISALLOWED"
        # Do not use the GitHub API's final page alone: read bounded pagination.
        # An unexpectedly enormous comment stream fails closed.
        now = datetime.datetime.now(datetime.timezone.utc)
        for page in range(1, 11):
            data = _github_json(api + f"/issues/{pr_number}/comments?per_page=100&page={page}")
            if not isinstance(data, list):
                return False, "GITHUB_COMMENT_RESPONSE_INVALID"
            for comment in reversed(data):
                if isinstance(comment, dict):
                    accepted, evidence = evaluate_owner_comment(
                        comment, owner_id=owner_id, pr_number=pr_number,
                        head_sha=head_sha, now=now,
                    )
                    if accepted:
                        return True, evidence
            if len(data) < 100:
                return False, "NO_OWNER_APPROVAL_FOR_EXACT_HEAD"
        return False, "COMMENT_PAGINATION_LIMIT_EXCEEDED"
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, OSError):
        return False, "GITHUB_OWNER_ATTESTATION_UNAVAILABLE"
