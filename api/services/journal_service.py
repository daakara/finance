"""
JournalApplicationService Seam for ARX SaaS Foundation (Wave 1F-B).

This application service defines the bounded orchestration interface for journal
and trade lifecycle operations. It coordinates:
1. Workspace access authorization (WorkspaceAuthorizer)
2. Capability entitlement enforcement (EntitlementResolver: 'journal.read', 'journal.write')
3. Repository interaction

Guarantees:
- Pure application coordination; zero commercial plan names or pricing logic.
- Domain purity (INV-SAAS-01): Never passes RequestContext or EntitlementSet into quant engines.
- Private cache safety (INV-SAAS-02): All operations operate within private boundaries.
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

    def _get_storage_selector(self, context: RequestContext) -> str:
        """Derive storage selector for the pre-tenancy database compatibility seam."""
        return context.actor_id or "default_user"

    def get_trades(
        self,
        context: RequestContext,
        limit: int = 50,
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

        selector = self._get_storage_selector(context)
        if self.db_engine is not None:
            if hasattr(self.db_engine, "get_journal_trades"):
                return self.db_engine.get_journal_trades(selector, limit=limit)
            elif hasattr(self.db_engine, "get_user_trades"):
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

        selector = self._get_storage_selector(context)
        if self.db_engine is not None:
            if hasattr(self.db_engine, "save_journal_trade"):
                return self.db_engine.save_journal_trade(selector, trade_data)
            elif hasattr(self.db_engine, "log_trade"):
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

        selector = self._get_storage_selector(context)
        if self.db_engine is not None:
            if hasattr(self.db_engine, "record_trade_fill"):
                return self.db_engine.record_trade_fill(selector, fill_data)
            elif hasattr(self.db_engine, "record_fill"):
                return self.db_engine.record_fill(selector, fill_data)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "fill": fill_data,
        }

    def record_exit(
        self,
        context: RequestContext,
        exit_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record a trade exit or partial position reduction.
        Enforces workspace access and 'journal.write' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("journal.write"):
            raise PermissionError("Workspace is not entitled to capability 'journal.write'.")

        selector = self._get_storage_selector(context)
        if self.db_engine is not None and hasattr(self.db_engine, "record_trade_exit"):
            return self.db_engine.record_trade_exit(selector, exit_data)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "exit": exit_data,
        }

    def record_close(
        self,
        context: RequestContext,
        close_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Close 100% of an active holding.
        Enforces workspace access and 'journal.write' capability.
        """
        close_data_copy = dict(close_data)
        close_data_copy["shares"] = None
        return self.record_exit(context, close_data_copy)

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

        selector = self._get_storage_selector(context)
        if self.db_engine is not None:
            if hasattr(self.db_engine, "close_trade"):
                return self.db_engine.close_trade(selector, trade_id, close_data)
            elif hasattr(self.db_engine, "record_trade_exit"):
                exit_payload = dict(close_data)
                exit_payload["tradeId"] = trade_id
                return self.db_engine.record_trade_exit(selector, exit_payload)

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

        selector = self._get_storage_selector(context)
        if self.db_engine is not None:
            if hasattr(self.db_engine, "get_risk_telemetry"):
                return self.db_engine.get_risk_telemetry(selector)
            elif hasattr(self.db_engine, "get_journal_telemetry"):
                return self.db_engine.get_journal_telemetry(selector)

        return {
            "workspace_id": context.workspace_id,
            "trade_count": 0,
            "win_rate": 0.0,
            "drawdown": 0.0,
        }
