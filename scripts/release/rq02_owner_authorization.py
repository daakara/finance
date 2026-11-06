"""ARX Terminal — Release Qualification Check RQ-02 Solo-Owner PR Authorization.

Provides an authoritative, sustainable pull-request admission mechanism for solo-owner
governance, eliminating hardcoded single-use exception scripts.

GOVERNING INVARIANTS:
1. SOLO-OWNER AUTHORITY: Only authorizations attested by repository owner 'daakara' are valid.
2. IMMUTABLE SHA BINDING: Candidate HEAD SHA must strictly match the authorization record.
   Any commit modification, rebase, or tamper invalidates the authorization.
3. STRICT SCOPE CONTAINMENT: All changed files must be explicitly declared in allowed_paths.
   Zero unexpected file additions or modifications are admitted.
4. ABSOLUTE BOUNDARY ISOLATION: Modification of secrets (.secrets/), rulesets (.github/rulesets/),
   environment (.env*), production release manifests (release_manifest.json), or deployment
   configs (railway.json, wrangler.toml) strictly fails closed.
5. BACKWARDS COMPATIBILITY: Historical approved PRs (123..205) are preserved via registry.
6. ZERO BYPASS: Evaluates fail-closed under missing, corrupted, or tampered registry.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arx.release.rq02_owner_authorization")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "pr_authorizations.json")
AUTHORIZED_OWNER = "daakara"
SHA40_REGEX = re.compile(r"^[0-9a-fA-F]{40}$")

FORBIDDEN_ADMIN_PREFIXES = (
    "api/",
    "analyst_dashboard/",
    "frontend/",
    "migrations/",
    ".github/rulesets/",
    ".secrets/",
)

FORBIDDEN_PROTECTED_FILES = (
    "railway.json",
    "wrangler.toml",
    "scripts/release/release_manifest.json",
)


def _check_duplicate_keys_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Custom JSON hook that rejects duplicate keys in registry dictionaries."""
    seen: Dict[str, Any] = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"DUPLICATE_REGISTRY_ENTRY: Duplicate key '{k}' detected in JSON.")
        seen[k] = v
    return seen


def load_authorization_registry(registry_path: Optional[str] = None) -> Tuple[bool, Dict[str, Any], str]:
    """Loads and validates the authoritative solo-owner PR authorization registry.

    Returns:
        (is_valid, registry_data, error_reason)
    """
    path = registry_path or DEFAULT_REGISTRY_PATH
    if not os.path.exists(path):
        return False, {}, f"AUTHORIZATION_REGISTRY_MISSING: File not found at '{path}'."

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=_check_duplicate_keys_hook)
    except Exception as e:
        return False, {}, f"AUTHORIZATION_REGISTRY_CORRUPT: Failed to parse registry JSON: {e}"

    if not isinstance(data, dict):
        return False, {}, "AUTHORIZATION_REGISTRY_INVALID: Root object must be a dictionary."

    owner = data.get("owner")
    if owner != AUTHORIZED_OWNER:
        return False, {}, f"AUTHORIZATION_REGISTRY_UNAUTHORIZED_OWNER: Expected owner '{AUTHORIZED_OWNER}', got '{owner}'."

    policy = data.get("policy_name")
    if policy != "ARX_SOLO_OWNER_PR_AUTHORIZATION_POLICY":
        return False, {}, f"AUTHORIZATION_REGISTRY_INVALID_POLICY: Expected 'ARX_SOLO_OWNER_PR_AUTHORIZATION_POLICY', got '{policy}'."

    return True, data, ""


def verify_candidate_authorization(
    *,
    pr_number: Optional[int],
    head_sha: str,
    base_sha: str,
    clean_target: str,
    diff_entries: List[Tuple[str, str]],
    identity: Optional[Dict[str, Any]] = None,
    registry_path: Optional[str] = None,
    allowed_app_prefixes: Tuple[str, ...] = ("api/", "analyst_dashboard/", "frontend/", "tests/", "scripts/release/", "docs/"),
) -> Tuple[bool, str, Dict[str, Any]]:
    """Authoritative solo-owner gatekeeper for candidate PR diff admission.

    Validates candidate identity, cryptographic SHA binding, source scope,
    forbidden paths, and historical admissions.

    Returns:
        (is_authorized, reason_code, audit_details)
    """
    audit_details: Dict[str, Any] = {
        "pr_number": pr_number,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "target_branch": clean_target,
        "mode": "UNKNOWN",
        "violations": [],
    }

    if pr_number is None:
        return False, "PR_NUMBER_REQUIRED", audit_details

    try:
        pr_val = int(pr_number)
    except (TypeError, ValueError):
        return False, "INVALID_PR_NUMBER_FORMAT", audit_details

    ok, registry, err = load_authorization_registry(registry_path)
    if not ok:
        audit_details["violations"].append(err)
        return False, err, audit_details

    historical_set: Set[int] = set(registry.get("historical_approved_prs", []))
    authorizations: Dict[str, Any] = registry.get("authorizations", {})

    str_pr = str(pr_val)

    # Path A: Explicit Solo-Owner Candidate Authorization
    if str_pr in authorizations:
        audit_details["mode"] = "EXPLICIT_OWNER_AUTHORIZATION"
        auth = authorizations[str_pr]

        if not isinstance(auth, dict):
            return False, "MALFORMED_AUTHORIZATION_RECORD", audit_details

        # 1. Authorizer Verification
        authorized_by = auth.get("authorized_by")
        if authorized_by != AUTHORIZED_OWNER:
            err_msg = f"UNAUTHORIZED_SIGNER: Expected authorizer '{AUTHORIZED_OWNER}', got '{authorized_by}'."
            audit_details["violations"].append(err_msg)
            return False, err_msg, audit_details

        # 2. Immutable Candidate Head SHA Binding
        authorized_head = auth.get("candidate_head_sha", "")
        if not authorized_head or not SHA40_REGEX.match(authorized_head):
            err_msg = f"MALFORMED_AUTHORIZED_HEAD_SHA: '{authorized_head}' is not a valid 40-char SHA."
            audit_details["violations"].append(err_msg)
            return False, err_msg, audit_details

        if head_sha.lower() != authorized_head.lower():
            err_msg = f"CANDIDATE_SHA_MISMATCH: Authorization bound to '{authorized_head}', candidate HEAD is '{head_sha}'."
            audit_details["violations"].append(err_msg)
            return False, err_msg, audit_details

        # 3. Target Branch Binding
        auth_target = auth.get("target_branch", "main")
        if clean_target != auth_target:
            err_msg = f"TARGET_BRANCH_MISMATCH: Authorization target '{auth_target}' does not match '{clean_target}'."
            audit_details["violations"].append(err_msg)
            return False, err_msg, audit_details

        # 4. Hosted Execution Context Verification (Prevent Impersonation in CI)
        if identity and identity.get("event_name") == "pull_request":
            pr_author = identity.get("pr_author") or identity.get("user")
            pr_association = identity.get("author_association")
            if pr_author and pr_author != AUTHORIZED_OWNER and pr_association != "OWNER":
                err_msg = f"FORBIDDEN_OWNER_IMPERSONATION: PR author '{pr_author}' cannot claim solo-owner authorization."
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

        # 5. Source Scope Containment & Boundary Verification
        allowed_paths = set(auth.get("allowed_paths", []))
        if not allowed_paths:
            err_msg = "EMPTY_ALLOWED_PATHS: Explicit authorization must declare at least one allowed path."
            audit_details["violations"].append(err_msg)
            return False, err_msg, audit_details

        for status_code, path in diff_entries:
            # Traversal check
            if ".." in path or path.startswith("/") or path.startswith("\\") or "\x00" in path:
                err_msg = f"PATH_TRAVERSAL_DETECTED: '{path}'"
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

            # Strict Boundary Isolation: Protected Files
            if path in FORBIDDEN_PROTECTED_FILES or path.startswith(".secrets/") or path.startswith(".github/rulesets/") or path.startswith(".env"):
                err_msg = f"FORBIDDEN_PROTECTED_PATH_MODIFICATION: '{path}'"
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

            # Operation check: Deletion or rename of application code forbidden
            if status_code.startswith("D") or status_code.startswith("R"):
                err_msg = f"UNAUTHORIZED_FILE_OPERATION: Operation '{status_code}' on '{path}' is forbidden."
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

            # Scope Containment
            if path not in allowed_paths:
                err_msg = f"UNAUTHORIZED_DIFF_PATH: File '{path}' is not within authorized scope for PR #{pr_val}."
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

        audit_details["directive"] = auth.get("directive", "UNSPECIFIED")
        audit_details["scope_description"] = auth.get("scope_description", "")
        return True, "OWNER_AUTHORIZATION_VERIFIED", audit_details

    # Path B: Historical Approved PR (Backwards Compatibility)
    elif pr_val in historical_set:
        audit_details["mode"] = "HISTORICAL_APPROVAL"

        for status_code, path in diff_entries:
            # Traversal check
            if ".." in path or path.startswith("/") or path.startswith("\\") or "\x00" in path:
                err_msg = f"PATH_TRAVERSAL_DETECTED: '{path}'"
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

            # Protected Files
            if path in FORBIDDEN_PROTECTED_FILES or path.startswith(".secrets/") or path.startswith(".github/rulesets/") or path.startswith(".env"):
                err_msg = f"FORBIDDEN_PROTECTED_PATH_MODIFICATION: '{path}'"
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

            # Forbidden Deletions/Renames
            if status_code.startswith("D") or status_code.startswith("R"):
                err_msg = f"UNAUTHORIZED_FILE_OPERATION: Operation '{status_code}' on '{path}' is forbidden."
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

            # Allowed prefix check
            if not any(path.startswith(prefix) for prefix in allowed_app_prefixes):
                err_msg = f"UNAUTHORIZED_DIFF_PATH: File '{path}' is outside authorized application trees."
                audit_details["violations"].append(err_msg)
                return False, err_msg, audit_details

        return True, "HISTORICAL_PR_AUTHORIZED", audit_details

    # Path C: Unapproved PR -> Fail Closed
    else:
        err_msg = f"UNAPPROVED_PR_NUMBER: PR #{pr_val} is not in approved release or owner-authorized PR registry."
        audit_details["violations"].append(err_msg)
        return False, err_msg, audit_details
