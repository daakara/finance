"""
ARX TERMINAL — SMART MONEY
WORKLOAD IDENTITY EXACT VERIFIER (EPOCH 001)

Enforces exact tuple verification of ambient GitHub Actions OIDC claims
against the frozen workload identity policy. Replaces broad-pattern matching.
"""

import datetime
from typing import Dict, Any, Tuple, Optional


def verify_workload_identity(
    claims: Dict[str, Any],
    policy: Dict[str, Any],
    current_timestamp: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Verifies that claims exactly match all frozen policy dimensions.
    Returns (True, 'PASS') or (False, '<exact rejection reason>').
    """
    if current_timestamp is None:
        current_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

    constraints = policy.get("workload_identity_constraints", {})
    if not constraints:
        return False, "REJECTED: Policy missing workload_identity_constraints"

    # 1. Issuer
    expected_iss = constraints.get("authorized_issuer", "")
    if claims.get("iss") != expected_iss:
        return False, f"REJECTED: Unauthorized issuer '{claims.get('iss')}' (expected '{expected_iss}')"

    # 2. Audience
    expected_aud = constraints.get("authorized_audience", "")
    if claims.get("aud") != expected_aud:
        return False, f"REJECTED: Unauthorized audience '{claims.get('aud')}' (expected '{expected_aud}')"

    # 3. Repository
    expected_repo = constraints.get("authorized_repository", "")
    if claims.get("repository") != expected_repo:
        return False, f"REJECTED: Unauthorized repository '{claims.get('repository')}' (expected '{expected_repo}')"

    # 4. Repository ID
    expected_repo_id = str(constraints.get("authorized_repository_id", ""))
    if str(claims.get("repository_id", "")) != expected_repo_id:
        return False, f"REJECTED: Unauthorized repository_id '{claims.get('repository_id')}' (expected '{expected_repo_id}')"

    # 5. Repository Owner
    expected_owner = constraints.get("authorized_repository_owner", "")
    if claims.get("repository_owner") != expected_owner:
        return False, f"REJECTED: Unauthorized repository_owner '{claims.get('repository_owner')}' (expected '{expected_owner}')"

    # 6. Repository Owner ID
    expected_owner_id = str(constraints.get("authorized_repository_owner_id", ""))
    if str(claims.get("repository_owner_id", "")) != expected_owner_id:
        return False, f"REJECTED: Unauthorized repository_owner_id '{claims.get('repository_owner_id')}' (expected '{expected_owner_id}')"

    # 7. Workflow Name
    expected_wf = constraints.get("authorized_workflow", "")
    if claims.get("workflow") != expected_wf:
        return False, f"REJECTED: Unauthorized workflow '{claims.get('workflow')}' (expected '{expected_wf}')"

    # 8. Job Workflow Ref
    expected_job_wf_ref = constraints.get("authorized_job_workflow_ref", "")
    if claims.get("job_workflow_ref") != expected_job_wf_ref:
        return False, f"REJECTED: Unauthorized job_workflow_ref '{claims.get('job_workflow_ref')}' (expected '{expected_job_wf_ref}')"

    # 9. Ref
    expected_ref = constraints.get("authorized_ref", "")
    if claims.get("ref") != expected_ref:
        return False, f"REJECTED: Unauthorized ref '{claims.get('ref')}' (expected '{expected_ref}')"

    # 10. Subject Rule
    expected_sub = constraints.get("subject_rule", "")
    if claims.get("sub") != expected_sub:
        return False, f"REJECTED: Unauthorized subject '{claims.get('sub')}' (expected '{expected_sub}')"

    # 11. Event Name
    authorized_events = constraints.get("authorized_event_names", [])
    prohibited_events = constraints.get("prohibited_event_names", [])
    event_name = claims.get("event_name", "")

    if event_name in prohibited_events:
        return False, f"REJECTED: Prohibited event_name '{event_name}'"
    if event_name not in authorized_events:
        return False, f"REJECTED: Unauthorized event_name '{event_name}' (allowed: {authorized_events})"

    # 12. Environment
    expected_env = constraints.get("authorized_environment", "")
    claim_env = claims.get("environment") or ""
    if expected_env in ("NONE", "", None):
        if claim_env not in ("", "NONE", None):
            return False, f"REJECTED: Unexpected environment '{claim_env}' (expected NONE)"
    elif claim_env != expected_env:
        return False, f"REJECTED: Unauthorized environment '{claim_env}' (expected '{expected_env}')"

    # 13. Expiration & Lifetime
    exp = claims.get("exp")
    if exp is not None and current_timestamp > int(exp):
        return False, f"REJECTED: Token expired ({exp} < {current_timestamp})"

    nbf = claims.get("nbf")
    max_lifetime = constraints.get("maximum_token_lifetime_seconds", 600)
    if exp is not None and nbf is not None:
        lifetime = int(exp) - int(nbf)
        if lifetime > max_lifetime:
            return False, f"REJECTED: Token lifetime {lifetime}s exceeds maximum {max_lifetime}s"

    return True, "PASS"
