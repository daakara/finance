"""
Workspace and Resource Authorization Boundary for ARX SaaS Foundation (Wave 1F-A).

This module defines:
1. WorkspaceAuthorizer: Protocol distinguishing workspace access authorization
   from commercial entitlement resolution.
2. DefaultWorkspaceAuthorizer: Deterministic, fail-closed authorizer for the current
   phase (prior to full database-backed workspace memberships in Phase 1G).

Architectural Rule:
Authorization (actor identity and workspace access) must NEVER be collapsed
with Entitlements (capabilities and limits) into a single Boolean check.
"""

import hashlib
from typing import Optional, Protocol

# Attestation: Full database-backed tenancy and membership await Phase 1G
WORKSPACE_MEMBERSHIP_RESOLUTION: str = "NOT_IMPLEMENTED"


class WorkspaceAuthorizer(Protocol):
    """
    Protocol governing workspace and resource access control.
    Separated from entitlement resolution.
    """

    def authorize_workspace_access(
        self,
        actor_id: Optional[str],
        workspace_id: str,
    ) -> bool:
        """Verify whether an actor is authorized to access a given workspace."""
        ...

    def authorize_resource_access(
        self,
        actor_id: Optional[str],
        workspace_id: str,
        resource_owner_actor_id: Optional[str],
    ) -> bool:
        """Verify whether an actor is authorized to access a specific resource within a workspace."""
        ...


class DefaultWorkspaceAuthorizer:
    """
    Default deterministic authorizer for the current pre-membership phase.

    Rules:
    1. 'ws_default': Open to anonymous actors and identified legacy actors.
    2. 'ws_usr_<hash>': Authorized only if actor_id matches the deterministic hash.
    3. Custom / foreign workspace IDs: Fail-closed (False) until Phase 1G membership tables exist.
    """

    def authorize_workspace_access(
        self,
        actor_id: Optional[str],
        workspace_id: str,
    ) -> bool:
        if not workspace_id or not isinstance(workspace_id, str):
            return False

        clean_ws = workspace_id.strip()

        # Legacy anonymous workspace is accessible
        if clean_ws == "ws_default":
            return True

        # Personal deterministic migration workspace requires matching actor
        if clean_ws.startswith("ws_usr_"):
            if not actor_id:
                return False
            expected_hash = hashlib.sha256(actor_id.encode("utf-8")).hexdigest()[:16]
            return clean_ws == f"ws_usr_{expected_hash}"

        # Unknown workspace format in current phase fails closed
        return False

    def authorize_resource_access(
        self,
        actor_id: Optional[str],
        workspace_id: str,
        resource_owner_actor_id: Optional[str],
    ) -> bool:
        # First verify workspace access
        if not self.authorize_workspace_access(actor_id, workspace_id):
            return False

        # If resource has a specific actor owner, verify matching actor (or unowned)
        if resource_owner_actor_id is not None:
            if actor_id is None:
                return False
            return actor_id == resource_owner_actor_id

        return True
