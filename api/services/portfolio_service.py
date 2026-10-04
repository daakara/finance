"""
PortfolioApplicationService Seam for ARX SaaS Foundation (Wave 1F-A).

This application service defines the bounded orchestration interface for portfolio
operations. It coordinates:
1. Workspace access authorization (WorkspaceAuthorizer)
2. Capability and limit entitlement enforcement (EntitlementResolver)
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


class PortfolioApplicationService:
    """
    Application service managing portfolio holding operations with fail-closed
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

    def get_portfolio(self, context: RequestContext) -> Dict[str, Any]:
        """
        Retrieve portfolio holdings for the given workspace context.
        Enforces workspace access and 'portfolio.read' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.read"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.read'.")

        if self.db_engine is not None and hasattr(self.db_engine, "get_user_portfolio"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.get_user_portfolio(selector)

        return {
            "workspace_id": context.workspace_id,
            "holdings": [],
            "total_value": 0.0,
            "cash": 0.0,
        }

    def add_holding(
        self,
        context: RequestContext,
        symbol: str,
        shares: float,
        cost_basis: float,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add or update a holding within the workspace.
        Enforces workspace access, 'portfolio.manage' capability, and 'portfolio.max_holdings' limit.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.manage"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.manage'.")

        max_holdings = entitlements.get_limit("portfolio.max_holdings")
        if max_holdings is not None and self.db_engine is not None:
            selector = context.actor_id or context.workspace_id
            if hasattr(self.db_engine, "get_user_portfolio"):
                current_portfolio = self.db_engine.get_user_portfolio(selector)
                holdings_list = current_portfolio.get("holdings", []) if isinstance(current_portfolio, dict) else []
                if len(holdings_list) >= max_holdings:
                    raise ValueError(
                        f"Workspace holding count ({len(holdings_list)}) exceeds limit of {max_holdings}."
                    )

        if self.db_engine is not None and hasattr(self.db_engine, "add_holding"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.add_holding(selector, symbol.upper().strip(), shares, cost_basis)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "symbol": symbol.upper().strip(),
            "shares": shares,
            "cost_basis": cost_basis,
            "notes": notes,
        }

    def remove_holding(self, context: RequestContext, symbol: str) -> Dict[str, Any]:
        """
        Remove a holding from the workspace.
        Enforces workspace access and 'portfolio.manage' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.manage"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.manage'.")

        if self.db_engine is not None and hasattr(self.db_engine, "delete_holding"):
            selector = context.actor_id or context.workspace_id
            return self.db_engine.delete_holding(selector, symbol.upper().strip())

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "symbol": symbol.upper().strip(),
        }
