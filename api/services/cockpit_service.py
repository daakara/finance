"""
CockpitApplicationService Seam for ARX SaaS Foundation (Wave 1F-B).

This application service defines the bounded orchestration interface for the unified
cockpit experience. It coordinates:
1. Workspace access authorization (WorkspaceAuthorizer)
2. Separation between actor-profile telemetry and workspace state (Strategy A)
3. Action items and readiness orchestration
4. Repository interaction

Guarantees:
- Pure application coordination; zero commercial plan names or pricing logic.
- Domain purity (INV-SAAS-01): Never passes RequestContext or EntitlementSet into quant engines.
- Private cache safety (INV-SAAS-02): All operations operate within private boundaries.
"""

from datetime import datetime, timezone
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

    def _verify_actor_bound_workspace(self, context: RequestContext) -> None:
        """Enforce that private persistence operations require an authenticated or actor-bound workspace (INV-SAAS-07)."""
        if not context.actor_id or context.workspace_id == "ws_default":
            raise PermissionError(
                "Private persistence operations require an authenticated or actor-bound workspace. "
                "Shared 'ws_default' cannot own persistent data (INV-SAAS-07)."
            )

    def _get_storage_selector(self, context: RequestContext) -> str:
        """Derive storage selector for actor profile records (Strategy A: ACTOR_PROFILE_DATA)."""
        return context.actor_id or "default"

    def get_cockpit_state(self, context: RequestContext) -> Dict[str, Any]:
        """
        Retrieve unified cockpit state: actor profile resilience + workspace actions and holdings.
        Enforces workspace access.
        """
        self._verify_workspace_access(context)

        # Under INV-SAAS-07, ws_default cannot access private persisted data
        if not context.actor_id or context.workspace_id == "ws_default":
            return {
                "actor_id": context.actor_id,
                "workspace_id": context.workspace_id,
                "profile": {},
                "holdings": [],
                "actions": [],
            }

        sel_id = self._get_storage_selector(context)

        profile_data: Dict[str, Any] = {}
        holdings_data: List[Dict[str, Any]] = []
        actions_data: List[Dict[str, Any]] = []

        if self.db_engine is not None:
            if hasattr(self.db_engine, "get_user_profile"):
                profile_data = self.db_engine.get_user_profile(sel_id) or {}
            if hasattr(self.db_engine, "get_workspace_portfolio"):
                holdings_data = self.db_engine.get_workspace_portfolio(context.workspace_id, user_id=context.actor_id) or []
            elif hasattr(self.db_engine, "get_user_portfolio"):
                holdings_data = self.db_engine.get_user_portfolio(sel_id) or []
            if hasattr(self.db_engine, "get_workspace_actions"):
                actions_data = self.db_engine.get_workspace_actions(context.workspace_id, user_id=context.actor_id) or []
            elif hasattr(self.db_engine, "get_user_actions"):
                actions_data = self.db_engine.get_user_actions(sel_id) or []

        return {
            "actor_id": context.actor_id,
            "workspace_id": context.workspace_id,
            "profile": profile_data,
            "holdings": holdings_data,
            "actions": actions_data,
        }

    def update_profile(
        self,
        context: RequestContext,
        profile_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update actor profile resilience metrics.
        Enforces workspace access, actor-bound workspace (INV-SAAS-07), and verifies actor presence.
        """
        self._verify_workspace_access(context)
        self._verify_actor_bound_workspace(context)

        sel_id = self._get_storage_selector(context)

        if self.db_engine is not None:
            if hasattr(self.db_engine, "save_user_profile"):
                success = self.db_engine.save_user_profile(sel_id, profile_data)
                if not success:
                    raise RuntimeError("Failed to persist profile to SQLite store.")
            elif hasattr(self.db_engine, "update_user_profile"):
                return self.db_engine.update_user_profile(sel_id, profile_data)

        return {
            "status": "SUCCESS",
            "message": f"Profile persisted for record selector '{sel_id}'.",
            "profileId": sel_id,
            "userId": sel_id,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }

    def create_action(
        self,
        context: RequestContext,
        action_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record a new action item in the workspace.
        Enforces workspace access and actor-bound workspace (INV-SAAS-07).
        """
        self._verify_workspace_access(context)
        self._verify_actor_bound_workspace(context)

        sel_id = self._get_storage_selector(context)

        if self.db_engine is not None:
            if hasattr(self.db_engine, "save_workspace_action"):
                success = self.db_engine.save_workspace_action(
                    workspace_id=context.workspace_id,
                    user_id=context.actor_id or "default",
                    action=action_data,
                )
                if not success:
                    raise RuntimeError("Failed to persist action item to SQLite store.")
            elif hasattr(self.db_engine, "save_user_action"):
                success = self.db_engine.save_user_action(sel_id, action_data)
                if not success:
                    raise RuntimeError("Failed to persist action item to SQLite store.")
            elif hasattr(self.db_engine, "add_user_action"):
                return self.db_engine.add_user_action(sel_id, action_data)

        action_id = str(action_data.get("id", ""))
        return {
            "status": "SUCCESS",
            "message": f"Action item '{action_id}' saved for selector '{sel_id}'.",
            "actionId": action_id,
        }

    def update_action(
        self,
        context: RequestContext,
        action_id: str,
        action_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update an action item state in the workspace.
        Enforces workspace access and actor-bound workspace (INV-SAAS-07).
        """
        self._verify_workspace_access(context)
        self._verify_actor_bound_workspace(context)

        sel_id = self._get_storage_selector(context)

        if self.db_engine is not None and hasattr(self.db_engine, "update_user_action"):
            return self.db_engine.update_user_action(sel_id, action_id, action_data)

        action_dict = dict(action_data)
        action_dict["id"] = action_id
        return self.create_action(context, action_dict)
