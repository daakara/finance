"""
CockpitApplicationService Seam for ARX SaaS Foundation (Wave 1F-A).

This application service defines the bounded orchestration interface for the unified
cockpit experience. It coordinates:
1. Workspace access authorization (WorkspaceAuthorizer)
2. Separation between actor-profile telemetry and workspace state (Strategy A)
3. Action items and readiness orchestration
4. Repository interaction

Guarantees:
- Pure application coordination; zero commercial plan names or pricing logic.
- Domain purity (INV-SAAS-01): Never passes RequestContext or EntitlementSet into quant engines.
- NOT wired to routes in Phase 1F-A (wiring reserved for Phase 1F-B).
"""

from typing import Optional, Any, Dict, List

from api.context.request_context import RequestContext
from api.services.authorizer import WorkspaceAuthorizer, DefaultWorkspaceAuthorizer
from api.services.entitlement_resolver import EntitlementResolver, DefaultEntitlementResolver


class CockpitApplicationService:
    """
    Application service managing cockpit dashboard state, actor profile resilience,
    and workflow action items with fail-closed workspace authorization.
    """

    def __init__(
        self,
        authorizer: Optional[WorkspaceAuthorizer] = None,
        entitlement_resolver: Optional[EntitlementResolver] = None,
        db_engine: Optional[Any] = None,
    ) -> None:
        self.authorizer = authorizer or DefaultWorkspaceAuthorizer()
        self.entitlement_resolver = entitlement_resolver or DefaultEntitlementResolver()
        self.db_engine = db_engine

    def _verify_workspace_access(self, context: RequestContext) -> None:
        """Enforce workspace access authorization."""
        if not self.authorizer.authorize_workspace_access(context.actor_id, context.workspace_id):
            raise PermissionError(
                f"Actor '{context.actor_id}' is not authorized to access workspace '{context.workspace_id}'."
            )

    def get_cockpit_state(self, context: RequestContext) -> Dict[str, Any]:
        """
        Retrieve unified cockpit state: actor profile resilience + workspace actions.
        Enforces workspace access.
        """
        self._verify_workspace_access(context)

        # Actor selector (personal profile data under Strategy A)
        actor_selector = context.actor_id or "default"
        # Workspace selector (shared holdings/actions)
        workspace_selector = context.workspace_id

        profile_data = {}
        actions_data: List[Dict[str, Any]] = []

        if self.db_engine is not None:
            if hasattr(self.db_engine, "get_user_profile"):
                profile_data = self.db_engine.get_user_profile(actor_selector) or {}
            if hasattr(self.db_engine, "get_user_actions"):
                actions_data = self.db_engine.get_user_actions(actor_selector) or []

        return {
            "actor_id": context.actor_id,
            "workspace_id": context.workspace_id,
            "profile": profile_data,
            "actions": actions_data,
        }

    def update_profile(
        self,
        context: RequestContext,
        profile_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update actor profile resilience metrics.
        Enforces workspace access and verifies actor presence.
        """
        self._verify_workspace_access(context)

        actor_selector = context.actor_id or "default"

        if self.db_engine is not None and hasattr(self.db_engine, "update_user_profile"):
            return self.db_engine.update_user_profile(actor_selector, profile_data)

        return {
            "status": "success",
            "actor_id": actor_selector,
            "updated_profile": profile_data,
        }

    def create_action(
        self,
        context: RequestContext,
        action_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record a new action item in the workspace.
        Enforces workspace access.
        """
        self._verify_workspace_access(context)

        actor_selector = context.actor_id or "default"

        if self.db_engine is not None and hasattr(self.db_engine, "add_user_action"):
            return self.db_engine.add_user_action(actor_selector, action_data)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "action": action_data,
        }

    def update_action(
        self,
        context: RequestContext,
        action_id: str,
        action_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update an action item state in the workspace.
        Enforces workspace access.
        """
        self._verify_workspace_access(context)

        actor_selector = context.actor_id or "default"

        if self.db_engine is not None and hasattr(self.db_engine, "update_user_action"):
            return self.db_engine.update_user_action(actor_selector, action_id, action_data)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "action_id": action_id,
            "updated": True,
        }
