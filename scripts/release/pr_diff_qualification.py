#!/usr/bin/env python3
"""Strict PR-scoped diff qualification prototype.

Approval manifests are read ONLY from the trusted base commit, never the PR head.
This is independent of the historical cumulative-release RQ-02 validator.
No workflow is switched to this candidate until governance review authorizes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PROTECTED_PREFIXES = (
    ".github/",
    "scripts/release/",
)
APPROVAL_ROOT = "scripts/release/pr_approvals"


class QualificationFailure(Exception):
    pass


def git(*args: str, binary: bool = False):
    result = subprocess.run(["git", *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise QualificationFailure(
            "GIT_FAILURE: " + " ".join(args[:2]) + " (" +
            result.stderr.decode("utf-8", errors="replace").strip()[:200] + ")"
        )
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def require_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise QualificationFailure(f"INVALID_{label}: requires full lowercase 40-character SHA")
    return value


def require_exact_fields(obj: dict, required: set[str], label: str):
    if not isinstance(obj, dict) or set(obj) != required:
        raise QualificationFailure(f"INVALID_{label}_SCHEMA: unexpected or missing fields")


def unique_json(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise QualificationFailure(f"DUPLICATE_APPROVAL_KEY: {key}")
        obj[key] = value
    return obj


def qualify(base: str, head: str, pr: int) -> dict:
    base = require_sha(base, "BASE")
    head = require_sha(head, "HEAD")
    if not isinstance(pr, int) or pr <= 0:
        raise QualificationFailure("INVALID_PR_NUMBER")
    if base == head:
        raise QualificationFailure("EMPTY_CANDIDATE")
    git("cat-file", "-e", base + "^{commit}")
    git("cat-file", "-e", head + "^{commit}")
    git("merge-base", "--is-ancestor", base, head)

    approval_path = f"{APPROVAL_ROOT}/{pr}.json"
    try:
        raw = git("show", f"{base}:{approval_path}")
    except QualificationFailure:
        raise QualificationFailure("NO_TRUSTED_BASE_APPROVAL")
    try:
        approval = json.loads(raw, object_pairs_hook=unique_json)
    except (ValueError, TypeError) as exc:
        raise QualificationFailure("INVALID_APPROVAL_JSON: " + str(exc)) from exc
    require_exact_fields(
        approval, {"schema_version", "pr_number", "files"},
        "APPROVAL",
    )
    if approval["schema_version"] != 1 or type(approval["pr_number"]) is not int or approval["pr_number"] != pr:
        raise QualificationFailure("APPROVAL_PR_OR_VERSION_MISMATCH")
    # Base-committed approvals pin content hashes, never future descendant SHAs.
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        if not event_path or os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
            raise QualificationFailure("UNTRUSTED_CI_EVENT_CONTEXT")
        try:
            with open(event_path, encoding="utf-8") as stream:
                event = json.load(stream, object_pairs_hook=unique_json)
            event_pr = event["pull_request"]
            repo = os.environ.get("GITHUB_REPOSITORY")
            if (not repo or event["repository"]["full_name"] != repo
                    or event_pr["base"]["repo"]["full_name"] != repo
                    or event_pr["head"]["repo"]["full_name"] != repo):
                raise QualificationFailure("CI_REPOSITORY_IDENTITY_MISMATCH")
            if (type(event["number"]) is not int or event["number"] != pr
                    or event_pr["number"] != pr
                    or event_pr["head"]["sha"] != head
                    or event_pr["base"]["sha"] != base):
                raise QualificationFailure("CI_EVENT_IDENTITY_MISMATCH")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise QualificationFailure("INVALID_CI_EVENT_IDENTITY") from exc
    files = approval["files"]
    if not isinstance(files, dict) or not files:
        raise QualificationFailure("EMPTY_OR_INVALID_APPROVED_FILE_INVENTORY")

    changes = git("diff", "--name-status", "--no-renames", base, head).splitlines()
    observed = {}
    for line in changes:
        cols = line.split("\t")
        if len(cols) != 2:
            raise QualificationFailure("INVALID_GIT_DIFF_ENTRY")
        op, path = cols
        if op not in ("A", "M", "D") or not path or path in observed:
            raise QualificationFailure("INVALID_OR_DUPLICATE_DIFF_ENTRY")
        if path.startswith(PROTECTED_PREFIXES) or path == approval_path:
            raise QualificationFailure("PROTECTED_QUALIFICATION_FILE_CHANGED: " + path)
        if path.startswith("/") or ".." in path.split("/") or "\\" in path:
            raise QualificationFailure("UNSAFE_DIFF_PATH")
        observed[path] = op
    if set(files) != set(observed):
        raise QualificationFailure(
            "UNAPPROVED_OR_MISSING_FILES: " + json.dumps({
                "unapproved": sorted(set(observed) - set(files)),
                "missing": sorted(set(files) - set(observed)),
            }, sort_keys=True)
        )

    receipt_files = {}
    for path, meta in files.items():
        require_exact_fields(meta, {"operation", "sha256"}, "FILE")
        if meta["operation"] != observed[path]:
            raise QualificationFailure("OPERATION_MISMATCH: " + path)
        if meta["operation"] == "D":
            if meta["sha256"] is not None:
                raise QualificationFailure("DELETION_HASH_MUST_BE_NULL: " + path)
            actual = None
        else:
            digest = meta["sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise QualificationFailure("INVALID_EXPECTED_HASH: " + path)
            actual = hashlib.sha256(git("show", f"{head}:{path}", binary=True)).hexdigest()
            if actual != digest:
                raise QualificationFailure("CONTENT_HASH_MISMATCH: " + path)
        receipt_files[path] = {"operation": observed[path], "sha256": actual}
    return {
        "verdict": "PASS",
        "qualification": "PR_DIFFERENTIAL_ONLY_NOT_PRODUCTION_RELEASE",
        "pr_number": pr,
        "base_sha": base,
        "head_sha": head,
        "approval_source": f"{base}:{approval_path}",
        "files": receipt_files,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--pr", required=True, type=int)
    args = parser.parse_args()
    try:
        result = qualify(args.base, args.head, args.pr)
    except QualificationFailure as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
