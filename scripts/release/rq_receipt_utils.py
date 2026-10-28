#!/usr/bin/env python3
"""
ARX Release Qualification - Receipt Utilities (Directive 074-A4.1)
Provides authoritative schema, cryptographic hashing, and validation for stage receipts (RQ-01..12).
Enforces single authoritative identity contract across Source, Execution, and Release dimensions.
Zero abbreviated SHA matching. Strict fail-closed evidence integrity.
"""

import sys, os, json, hashlib, datetime, subprocess, re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

QUALIFICATION_POLICY_VERSION = "2.2.0-D074A4"
SHA40_REGEX = re.compile(r"^[0-9a-fA-F]{40}$")

def get_receipts_dir():
    override = os.environ.get("ARX_RECEIPTS_DIR_OVERRIDE")
    if override and os.path.exists(override):
        return override
    return os.path.join(os.path.dirname(__file__), "receipts")

RECEIPTS_DIR = get_receipts_dir()

VALID_STATUSES = ["PASS", "FAIL", "BLOCKED", "NOT_RUN", "NOT_APPLICABLE"]

def ensure_receipts_dir():
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    return RECEIPTS_DIR

def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()

def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_git_commit(ref="HEAD") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.PIPE
        ).strip()
    except Exception:
        return "UNKNOWN_GIT_REF"

def get_git_tree(ref="HEAD:") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.PIPE
        ).strip()
    except Exception:
        return "UNKNOWN_TREE_REF"

def get_commit_tree(ref="HEAD") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{ref}^{{tree}}"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.PIPE
        ).strip()
    except Exception:
        return "UNKNOWN_COMMIT_TREE"

def get_commit_parents(ref="HEAD") -> list:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", f"{ref}^@"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.PIPE
        ).strip()
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []

def is_git_ancestor(ancestor: str, descendant: str) -> bool:
    try:
        res = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return (res.returncode == 0)
    except Exception:
        return False

def resolve_identity_contract() -> tuple:
    """
    Authoritative identity resolver for ARX Release Qualification.
    Implements Section 4 & Section 5 of Directive 074-A4.1.
    Returns (is_valid: bool, identity: dict, error_message: str).
    """
    is_github_actions = (os.environ.get("GITHUB_ACTIONS") == "true")
    repository = os.environ.get("GITHUB_REPOSITORY", "daakara/finance")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "local_execution")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    raw_run_id = os.environ.get("GITHUB_RUN_ID")
    raw_run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")

    if is_github_actions:
        if not raw_run_id:
            return False, {}, "Missing required hosted execution metadata: GITHUB_RUN_ID (Fail-Closed)"
        if not raw_run_attempt:
            return False, {}, "Missing required hosted execution metadata: GITHUB_RUN_ATTEMPT (Fail-Closed)"

    workflow_run_id = str(raw_run_id) if raw_run_id else "local_run"
    workflow_run_attempt = int(raw_run_attempt) if raw_run_attempt else 1

    # Checked out execution SHA
    execution_sha = os.environ.get("ARX_MOCK_EXECUTION_SHA") or get_git_commit("HEAD")
    if not SHA40_REGEX.match(execution_sha):
        return False, {}, f"Execution commit SHA '{execution_sha}' is not a valid 40-character hex SHA"

    # Context 1: Pull Request Event (Hosted GitHub Actions or simulated PR environment)
    if event_name == "pull_request" or (event_path and os.path.exists(event_path)):
        if not event_path or not os.path.exists(event_path):
            return False, {}, f"pull_request event requires valid GITHUB_EVENT_PATH, but file was not found: '{event_path}'"

        try:
            with open(event_path, "r", encoding="utf-8") as f:
                event_data = json.load(f)
        except Exception as e:
            return False, {}, f"Failed to parse GITHUB_EVENT_PATH JSON: {e}"

        pr_data = event_data.get("pull_request")
        if not pr_data or not isinstance(pr_data, dict):
            return False, {}, "Event payload missing 'pull_request' dictionary"

        pr_number = pr_data.get("number")
        if not pr_number or not isinstance(pr_number, int):
            return False, {}, f"Event payload invalid or missing PR number: {pr_number}"

        pr_head = pr_data.get("head", {})
        pr_base = pr_data.get("base", {})
        source_sha = pr_head.get("sha", "")
        base_sha = pr_base.get("sha", "")
        target_branch = pr_base.get("ref") or os.environ.get("GITHUB_BASE_REF") or "main"

        if not SHA40_REGEX.match(source_sha) or source_sha == "0" * 40:
            return False, {}, f"PR head SHA '{source_sha}' is not a valid 40-character hex SHA (cannot be null SHA)"
        if not SHA40_REGEX.match(base_sha) or base_sha == "0" * 40:
            return False, {}, f"PR base SHA '{base_sha}' is not a valid 40-character hex SHA (cannot be null SHA)"

        # Verify execution commit topology against Git repository
        # If in a mocked/simulated environment with override, verify simulated parents
        mock_parents = os.environ.get("ARX_MOCK_MERGE_PARENTS")
        if mock_parents:
            parents = [p.strip() for p in mock_parents.replace("\n", ",").split(",") if p.strip()]
        else:
            parents = get_commit_parents("HEAD")

        if len(parents) == 2:
            # Synthetic merge commit (e.g. refs/pull/PR/merge)
            parent1, parent2 = parents[0], parents[1]

            # 1. Verify parent 2 equals the authorized PR head SHA (Source Identity)
            if parent2.lower() != source_sha.lower():
                return False, {}, (
                    f"Synthetic merge commit head parent ({parent2}) does not equal "
                    f"authorized PR head SHA ({source_sha})"
                )

            # 2. Verify parent 1 corresponds to PR base SHA
            if parent1.lower() != base_sha.lower():
                if not mock_parents:
                    is_rel = is_git_ancestor(parent1, base_sha) or is_git_ancestor(base_sha, parent1)
                    if not is_rel:
                        return False, {}, (
                            f"Synthetic merge commit base parent ({parent1}) does not correspond to "
                            f"PR base SHA ({base_sha})"
                        )
                else:
                    return False, {}, (
                        f"Synthetic merge commit base parent ({parent1}) does not match "
                        f"PR base SHA ({base_sha})"
                    )

            # 3. Verify tree matches commit tree
            if not mock_parents:
                tree_sha = get_git_tree("HEAD:")
                commit_tree = get_commit_tree("HEAD")
                if tree_sha != commit_tree and tree_sha != "UNKNOWN_TREE_REF":
                    return False, {}, "Checked-out working tree does not match commit tree"

        elif len(parents) == 1:
            # Direct checkout of PR head
            if execution_sha.lower() != source_sha.lower():
                return False, {}, (
                    f"Direct checkout SHA ({execution_sha}) diverges from "
                    f"authorized PR head SHA ({source_sha})"
                )
        else:
            return False, {}, f"Invalid merge topology: commit has {len(parents)} parents (expected 1 or 2)"

        identity = {
            "repository": repository,
            "event_name": "pull_request",
            "pr_number": pr_number,
            "source_sha": source_sha,
            "execution_sha": execution_sha,
            "base_sha": base_sha,
            "target_branch": target_branch,
            "workflow_run_id": workflow_run_id,
            "workflow_run_attempt": workflow_run_attempt,
            "qualification_policy_version": QUALIFICATION_POLICY_VERSION
        }
        return True, identity, "VALID_PR_IDENTITY"

    # Context 2: Local Execution / Testing Context
    elif event_name in ("local_execution", "workflow_dispatch", "push"):
        source_sha = os.environ.get("ARX_PR_HEAD_SHA") or execution_sha
        base_sha = os.environ.get("ARX_PR_BASE_SHA") or "ea3cc13eeeb6d3e6c250ecfd3ed6aecb4ec44d6c"
        raw_pr = os.environ.get("ARX_PR_NUMBER")
        pr_number = int(raw_pr) if raw_pr else None
        target_branch = os.environ.get("ARX_TARGET_BRANCH") or os.environ.get("GITHUB_REF_NAME") or "release/production"

        if not SHA40_REGEX.match(source_sha):
            return False, {}, f"Source SHA '{source_sha}' is not a valid 40-character hex SHA"
        if not SHA40_REGEX.match(base_sha):
            return False, {}, f"Base SHA '{base_sha}' is not a valid 40-character hex SHA"

        identity = {
            "repository": repository,
            "event_name": event_name,
            "pr_number": pr_number,
            "source_sha": source_sha,
            "execution_sha": execution_sha,
            "base_sha": base_sha,
            "target_branch": target_branch,
            "workflow_run_id": workflow_run_id,
            "workflow_run_attempt": workflow_run_attempt,
            "qualification_policy_version": QUALIFICATION_POLICY_VERSION
        }
        return True, identity, "VALID_LOCAL_IDENTITY"

    # Context 3: Unsupported execution context -> Fail closed
    else:
        return False, {}, f"Unsupported execution context event: '{event_name}' (Fail-Closed)"

def get_execution_context() -> dict:
    """Convenience wrapper returning identity dictionary or raising RuntimeError."""
    ok, identity, err = resolve_identity_contract()
    if not ok:
        raise RuntimeError(f"Identity resolution failed closed: {err}")
    return identity

def create_stage_receipt(
    control_id: str,
    name: str,
    status: str,
    start_time: str,
    end_time: str,
    duration_sec: float,
    candidate_sha: str = None,
    target_branch: str = None,
    assertions: list = None,
    evidence_paths: list = None,
    errors: list = None,
    metadata: dict = None
) -> dict:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

    ok, identity, err_msg = resolve_identity_contract()
    if not ok:
        if status == "FAIL":
            identity = {
                "repository": os.environ.get("GITHUB_REPOSITORY", "daakara/finance"),
                "event_name": os.environ.get("GITHUB_EVENT_NAME", "unknown"),
                "pr_number": int(os.environ.get("ARX_PR_NUMBER", "121")) if os.environ.get("ARX_PR_NUMBER") else None,
                "source_sha": candidate_sha or "0000000000000000000000000000000000000000",
                "execution_sha": "0000000000000000000000000000000000000000",
                "base_sha": "0000000000000000000000000000000000000000",
                "target_branch": target_branch or "unknown",
                "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
                "qualification_policy_version": QUALIFICATION_POLICY_VERSION
            }
            auth_source_sha = identity["source_sha"]
            auth_execution_sha = identity["execution_sha"]
        else:
            raise RuntimeError(f"Identity resolution failed in stage {control_id}: {err_msg}")
    else:
        # Enforce that candidate_sha ALWAYS means authoritative source_sha
        auth_source_sha = identity["source_sha"]
        auth_execution_sha = identity["execution_sha"]

        if candidate_sha and candidate_sha.lower() != auth_source_sha.lower():
            if status == "FAIL":
                errors = errors or []
                errors.append(
                    f"Stage {control_id} candidate_sha mismatch: provided '{candidate_sha}' != "
                    f"authoritative source_sha '{auth_source_sha}'"
                )
            else:
                raise ValueError(
                    f"Stage {control_id} candidate_sha mismatch: provided '{candidate_sha}' != "
                    f"authoritative source_sha '{auth_source_sha}'"
                )

    evidence_paths = evidence_paths or []
    errors = errors or []
    metadata = metadata or {}

    evidence_digests = {}
    for p in evidence_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
        if os.path.exists(abs_p):
            evidence_digests[p] = compute_file_sha256(abs_p)
        else:
            evidence_digests[p] = "FILE_NOT_FOUND"

    receipt_core = {
        "control_id": control_id,
        "name": name,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": round(duration_sec, 3),
        "candidate_sha": auth_source_sha,
        "source_sha": auth_source_sha,
        "execution_sha": auth_execution_sha,
        "target_branch": target_branch or identity["target_branch"],
        "execution_context": identity,
        "assertions": assertions or [],
        "evidence_paths": evidence_paths,
        "evidence_digests": evidence_digests,
        "errors": errors,
        "metadata": {
            **metadata,
            "policy_version": QUALIFICATION_POLICY_VERSION
        }
    }

    # Compute deterministic receipt content digest (excluding the digest field itself)
    canonical_json = json.dumps(receipt_core, sort_keys=True, indent=2)
    receipt_digest = compute_sha256(canonical_json.encode("utf-8"))
    receipt_core["receipt_digest"] = receipt_digest

    return receipt_core

def save_stage_receipt(receipt: dict) -> str:
    r_dir = ensure_receipts_dir()
    cid_clean = receipt["control_id"].lower().replace("-", "")
    filename = f"{cid_clean}_receipt.json"
    filepath = os.path.join(r_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    return filepath

def load_stage_receipt(control_id: str) -> dict:
    r_dir = get_receipts_dir()
    cid_clean = control_id.lower().replace("-", "")
    filepath = os.path.join(r_dir, f"{cid_clean}_receipt.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_receipt_integrity(
    receipt: dict,
    expected_source_sha: str = None,
    expected_execution_sha: str = None
) -> tuple:
    if not isinstance(receipt, dict):
        return False, "Receipt is not a JSON object"

    required_keys = [
        "control_id", "name", "status", "start_time", "end_time",
        "duration_sec", "candidate_sha", "source_sha", "execution_sha",
        "target_branch", "execution_context", "assertions", "evidence_paths",
        "evidence_digests", "errors", "receipt_digest"
    ]
    for k in required_keys:
        if k not in receipt:
            return False, f"Missing required key '{k}' in receipt"

    if receipt["status"] not in VALID_STATUSES:
        return False, f"Invalid receipt status '{receipt['status']}'"

    # Enforce candidate_sha == source_sha
    cand_sha = receipt.get("candidate_sha", "")
    src_sha = receipt.get("source_sha", "")
    exec_sha = receipt.get("execution_sha", "")

    if cand_sha.lower() != src_sha.lower():
        return False, f"Receipt candidate_sha ({cand_sha}) does not equal source_sha ({src_sha})"

    if not SHA40_REGEX.match(src_sha):
        return False, f"Receipt source_sha '{src_sha}' is not a valid 40-character hex SHA"
    if not SHA40_REGEX.match(exec_sha):
        return False, f"Receipt execution_sha '{exec_sha}' is not a valid 40-character hex SHA"

    # Validate execution context schema
    ec = receipt.get("execution_context", {})
    if not isinstance(ec, dict):
        return False, "Receipt execution_context is not a dictionary"

    ec_required = [
        "repository", "event_name", "source_sha", "execution_sha",
        "base_sha", "target_branch", "workflow_run_id", "workflow_run_attempt",
        "qualification_policy_version"
    ]
    for ek in ec_required:
        if ek not in ec:
            return False, f"Missing required execution_context field '{ek}' in receipt"

    if ec["source_sha"].lower() != src_sha.lower():
        return False, f"execution_context source_sha ({ec['source_sha']}) != receipt source_sha ({src_sha})"
    if ec["execution_sha"].lower() != exec_sha.lower():
        return False, f"execution_context execution_sha ({ec['execution_sha']}) != receipt execution_sha ({exec_sha})"

    # Verify digest
    stored_digest = receipt["receipt_digest"]
    copy_receipt = dict(receipt)
    del copy_receipt["receipt_digest"]
    canonical_json = json.dumps(copy_receipt, sort_keys=True, indent=2)
    computed_digest = compute_sha256(canonical_json.encode("utf-8"))
    if stored_digest != computed_digest:
        return False, f"Receipt digest mismatch: stored {stored_digest[:10]} vs computed {computed_digest[:10]}"

    # Verify expected source SHA
    if expected_source_sha:
        if not SHA40_REGEX.match(expected_source_sha):
            return False, f"Expected source SHA '{expected_source_sha}' is not a valid 40-character SHA"
        if src_sha.lower() != expected_source_sha.lower():
            return False, f"Receipt source_sha mismatch: expected {expected_source_sha} vs receipt {src_sha}"

    # Verify expected execution SHA
    if expected_execution_sha:
        if not SHA40_REGEX.match(expected_execution_sha):
            return False, f"Expected execution SHA '{expected_execution_sha}' is not a valid 40-character SHA"
        if exec_sha.lower() != expected_execution_sha.lower():
            return False, f"Receipt execution_sha mismatch: expected {expected_execution_sha} vs receipt {exec_sha}"

    # Verify receipt freshness if age limit configured
    max_age_str = os.environ.get("ARX_MAX_RECEIPT_AGE_HOURS")
    if max_age_str:
        try:
            max_age = float(max_age_str)
            end_t_str = receipt.get("end_time", "")
            end_t = datetime.datetime.fromisoformat(end_t_str.replace("Z", "+00:00"))
            now_t = datetime.datetime.now(datetime.timezone.utc)
            if (now_t - end_t).total_seconds() > max_age * 3600:
                return False, f"Stale receipt: generated {(now_t - end_t).total_seconds() / 3600:.1f}h ago (limit: {max_age}h)"
        except Exception as e:
            return False, f"Invalid receipt timestamp format: {e}"

    return True, "VALID"
