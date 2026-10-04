"""
JournalApplicationService Seam for ARX SaaS Foundation (Wave 1F-A).

This application service defines the bounded orchestration interface for journal
and trade lifecycle operations. It coordinates:
1. Workspace access authorization (WorkspaceAuthorizer)
2. Capability entitlement enforcement (EntitlementResolver: 'journal.read', 'journal.write')
3. Repository interaction

Guarantees:
- Pure application coordination; zero commercial plan names or pricing logic.
- Domain purity (INV-SAAS-01): Never passes RequestContext or EntitlementSet into quant engines.
- NOT wired to routes in Phase 1F-A (wiring reserved for Phase 1F-B).
"""

from typing import Optional, Any, Dict, List

from api.context.request_context import RequestContext
from api.services.authorizer import WorkspaceAuthorizer, DefaultWorkspaceAuthorizer
from api.services.entitlement_resolver import EntitlementResolver, DefaultEntitlementResolver


class JournalApplicationService:
    """
    Application service managing trade journal records with fail-closed
    workspace authorization and entitlement validation.
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

    def get_trades(
        self,
        context: RequestContext,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trade journal entries for the given workspace context.
        Enforces workspace access and 'journal.read' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("journal.read"):
            raise PermissionError("Workspace is not entitled to capability 'journal.read'.")

        if self.db_engine is not None and hasattr(self.db_engine, "get_user_trades"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.get_user_trades(selector, status=status)

        return []

    def record_trade(
        self,
        context: RequestContext,
        trade_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Log a new intended trade in the workspace journal.
        Enforces workspace access and 'journal.write' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("journal.write"):
            raise PermissionError("Workspace is not entitled to capability 'journal.write'.")

        if self.db_engine is not None and hasattr(self.db_engine, "log_trade"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.log_trade(selector, trade_data)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "trade": trade_data,
        }

    def record_fill(
        self,
        context: RequestContext,
        fill_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record execution fill for a journal trade.
        Enforces workspace access and 'journal.write' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("journal.write"):
            raise PermissionError("Workspace is not entitled to capability 'journal.write'.")

        if self.db_engine is not None and hasattr(self.db_engine, "record_fill"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.record_fill(selector, fill_data)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "fill": fill_data,
        }

    def close_trade(
        self,
        context: RequestContext,
        trade_id: Any,
        close_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Close an existing trade in the workspace journal.
        Enforces workspace access and 'journal.write' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("journal.write"):
            raise PermissionError("Workspace is not entitled to capability 'journal.write'.")

        if self.db_engine is not None and hasattr(self.db_engine, "close_trade"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.close_trade(selector, trade_id, close_data)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "trade_id": trade_id,
            "closed": True,
        }

    def get_telemetry(self, context: RequestContext) -> Dict[str, Any]:
        """
        Retrieve journal behavioral risk telemetry for the workspace.
        Enforces workspace access and 'journal.read' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("journal.read"):
            raise PermissionError("Workspace is not entitled to capability 'journal.read'.")

        if self.db_engine is not None and hasattr(self.db_engine, "get_journal_telemetry"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.get_journal_telemetry(selector)

        return {
            "workspace_id": context.workspace_id,
            "trade_count": 0,
            "win_rate": 0.0,
            "drawdown": 0.0,
        }
