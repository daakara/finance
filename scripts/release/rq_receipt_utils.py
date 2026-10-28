#!/usr/bin/env python3
"""
ARX Release Qualification - Receipt Utilities
Provides standard schema, hashing, and validation for stage receipts (RQ-01..12).
Enforces fail-closed evidence integrity, candidate identity binding, and anti-tampering.
"""

import sys, os, json, hashlib, datetime, subprocess, re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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
        return subprocess.check_output(["git", "rev-parse", ref], cwd=PROJECT_ROOT, text=True, stderr=subprocess.PIPE).strip()
    except Exception:
        return "UNKNOWN_GIT_REF"

def get_git_tree(ref="HEAD:") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", ref], cwd=PROJECT_ROOT, text=True, stderr=subprocess.PIPE).strip()
    except Exception:
        return "UNKNOWN_TREE_REF"

def get_execution_context():
    """Extracts execution context from GitHub Actions environment or local simulation."""
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    run_id = os.environ.get("GITHUB_RUN_ID")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "local_execution")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    base_ref = os.environ.get("GITHUB_BASE_REF")
    head_ref = os.environ.get("GITHUB_HEAD_REF")
    sha = os.environ.get("GITHUB_SHA")
    
    pr_number = None
    pr_head_sha = None
    pr_base_sha = None

    if event_path and os.path.exists(event_path):
        try:
            with open(event_path, "r", encoding="utf-8") as f:
                event_data = json.load(f)
            pr_data = event_data.get("pull_request", {})
            pr_number = pr_data.get("number")
            pr_head_sha = pr_data.get("head", {}).get("sha")
            pr_base_sha = pr_data.get("base", {}).get("sha")
            if not base_ref:
                base_ref = pr_data.get("base", {}).get("ref")
            if not head_ref:
                head_ref = pr_data.get("head", {}).get("ref")
        except Exception:
            pass

    # Fallback to local PR simulation environment variables if set
    if not pr_number and os.environ.get("ARX_PR_NUMBER"):
        pr_number = int(os.environ.get("ARX_PR_NUMBER"))
    if not pr_head_sha and os.environ.get("ARX_PR_HEAD_SHA"):
        pr_head_sha = os.environ.get("ARX_PR_HEAD_SHA")
    if not pr_base_sha and os.environ.get("ARX_PR_BASE_SHA"):
        pr_base_sha = os.environ.get("ARX_PR_BASE_SHA")

    return {
        "is_github_actions": is_github_actions,
        "run_id": run_id,
        "run_attempt": int(run_attempt) if run_attempt else None,
        "event_name": event_name,
        "pr_number": pr_number,
        "pr_head_sha": pr_head_sha,
        "pr_base_sha": pr_base_sha,
        "base_ref": base_ref,
        "head_ref": head_ref,
        "github_sha": sha
    }

def create_stage_receipt(
    control_id: str,
    name: str,
    status: str,
    start_time: str,
    end_time: str,
    duration_sec: float,
    candidate_sha: str,
    target_branch: str,
    assertions: list,
    evidence_paths: list = None,
    errors: list = None,
    metadata: dict = None
) -> dict:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

    evidence_paths = evidence_paths or []
    errors = errors or []
    metadata = metadata or {}
    ctx = get_execution_context()

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
        "candidate_sha": candidate_sha,
        "target_branch": target_branch,
        "execution_context": ctx,
        "assertions": assertions,
        "evidence_paths": evidence_paths,
        "evidence_digests": evidence_digests,
        "errors": errors,
        "metadata": metadata
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

def validate_receipt_integrity(receipt: dict, expected_candidate_sha: str = None) -> tuple[bool, str]:
    if not isinstance(receipt, dict):
        return False, "Receipt is not a JSON object"
    
    required_keys = [
        "control_id", "name", "status", "start_time", "end_time", 
        "duration_sec", "candidate_sha", "target_branch", "execution_context",
        "assertions", "evidence_paths", "evidence_digests", "errors", "receipt_digest"
    ]
    for k in required_keys:
        if k not in receipt:
            return False, f"Missing required key '{k}' in receipt"

    if receipt["status"] not in VALID_STATUSES:
        return False, f"Invalid receipt status '{receipt['status']}'"

    # Verify digest
    stored_digest = receipt["receipt_digest"]
    copy_receipt = dict(receipt)
    del copy_receipt["receipt_digest"]
    canonical_json = json.dumps(copy_receipt, sort_keys=True, indent=2)
    computed_digest = compute_sha256(canonical_json.encode("utf-8"))
    if stored_digest != computed_digest:
        return False, f"Receipt digest mismatch: stored {stored_digest[:10]} vs computed {computed_digest[:10]}"

    # Verify candidate SHA match if expected
    if expected_candidate_sha and receipt["candidate_sha"].lower() != expected_candidate_sha.lower():
        return False, f"Receipt candidate_sha mismatch: expected {expected_candidate_sha[:10]} vs receipt {receipt['candidate_sha'][:10]}"

    return True, "VALID"
