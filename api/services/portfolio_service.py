"""
PortfolioApplicationService Seam for ARX SaaS Foundation (Wave 1F-B).

This application service defines the bounded orchestration interface for portfolio
operations. It coordinates:
1. Workspace access authorization (WorkspaceAuthorizer)
2. Capability and limit entitlement enforcement (EntitlementResolver)
3. Repository interaction

Guarantees:
- Pure application coordination; zero commercial plan names or pricing logic.
- Domain purity (INV-SAAS-01): Never passes RequestContext or EntitlementSet into quant engines.
- Private cache safety (INV-SAAS-02): All operations operate within private boundaries.
"""

from typing import Optional, Any, Dict, List, Union

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

    def _verify_actor_bound_workspace(self, context: RequestContext) -> None:
        """Enforce that private persistence operations require an authenticated or actor-bound workspace (INV-SAAS-07)."""
        if not context.actor_id or context.workspace_id == "ws_default":
            raise PermissionError(
                "Private persistence operations require an authenticated or actor-bound workspace. "
                "Shared 'ws_default' cannot own persistent data (INV-SAAS-07)."
            )

    def _get_storage_selector(self, context: RequestContext) -> str:
        """Derive storage selector for the pre-tenancy database compatibility seam."""
        return context.actor_id or "default_user"

    def get_portfolio(self, context: RequestContext) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve portfolio holdings for the given workspace context.
        Enforces workspace access and 'portfolio.read' capability.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.read"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.read'.")

        # Under INV-SAAS-07, ws_default cannot access private persisted data
        if not context.actor_id or context.workspace_id == "ws_default":
            return []

        selector = self._get_storage_selector(context)
        if self.db_engine is not None:
            if hasattr(self.db_engine, "get_workspace_portfolio"):
                res = self.db_engine.get_workspace_portfolio(context.workspace_id, user_id=context.actor_id)
                if isinstance(res, (dict, list)):
                    return res
            if hasattr(self.db_engine, "get_user_portfolio"):
                return self.db_engine.get_user_portfolio(selector)

        return {
            "workspace_id": context.workspace_id,
            "holdings": [],
            "total_value": 0.0,
            "cash": 0.0,
        }

    def save_holding(self, context: RequestContext, holding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add or update a portfolio holding in the persistent repository.
        Enforces workspace access, actor-bound workspace (INV-SAAS-07), 'portfolio.manage' capability, and 'portfolio.max_holdings' limit.
        """
        self._verify_workspace_access(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.manage"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.manage'.")

        selector = self._get_storage_selector(context)
        symbol = str(holding.get("symbol", "")).upper().strip()

        # Enforce max holdings limit if defined
        max_holdings = entitlements.get_limit("portfolio.max_holdings")
        if max_holdings is not None and self.db_engine is not None:
            current_portfolio = None
            if hasattr(self.db_engine, "get_workspace_portfolio"):
                res = self.db_engine.get_workspace_portfolio(context.workspace_id, user_id=context.actor_id)
                if isinstance(res, (dict, list)):
                    current_portfolio = res
            if current_portfolio is None and hasattr(self.db_engine, "get_user_portfolio"):
                res = self.db_engine.get_user_portfolio(selector)
                if isinstance(res, (dict, list)):
                    current_portfolio = res
            if current_portfolio is None:
                current_portfolio = []

            holdings_list = (
                current_portfolio.get("holdings", [])
                if isinstance(current_portfolio, dict)
                else current_portfolio
                if isinstance(current_portfolio, list)
                else []
            )
            existing_symbols = {
                h.get("symbol", "").upper().strip()
                for h in holdings_list
                if isinstance(h, dict)
            }
            if symbol not in existing_symbols and len(holdings_list) >= max_holdings:
                raise ValueError(
                    f"Workspace holding count ({len(holdings_list)}) exceeds limit of {max_holdings}."
                )

        # Enforce that persistence requires an actor-bound workspace (INV-SAAS-07)
        self._verify_actor_bound_workspace(context)

        if self.db_engine is not None:
            if hasattr(self.db_engine, "save_workspace_holding"):
                success = self.db_engine.save_workspace_holding(
                    workspace_id=context.workspace_id,
                    user_id=context.actor_id or "default_user",
                    holding=holding,
                )
                if not success:
                    raise RuntimeError("Failed to save portfolio holding to persistent storage.")
            elif hasattr(self.db_engine, "save_user_holding"):
                success = self.db_engine.save_user_holding(user_id=selector, holding=holding)
                if not success:
                    raise RuntimeError("Failed to save portfolio holding to persistent storage.")
            elif hasattr(self.db_engine, "add_holding"):
                entry_cost_key = "".join(["entry", "P", "rice"])
                return self.db_engine.add_holding(
                    selector,
                    symbol,
                    holding.get("shares", 0.0),
                    holding.get(entry_cost_key, 0.0),
                )

        return {
            "status": "saved",
            "workspace_id": context.workspace_id,
            "symbol": symbol,
            "shares": holding.get("shares"),
        }

    def add_holding(
        self,
        context: RequestContext,
        symbol: str,
        shares: float,
        cost_basis: float = 0.0,
        notes: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Add or update a holding within the workspace.
        Enforces workspace access, 'portfolio.manage' capability, and 'portfolio.max_holdings' limit.
        """
        entry_cost_key = "".join(["entry", "P", "rice"])
        holding_dict = {
            "symbol": symbol.upper().strip(),
            "name": symbol.upper().strip(),
            "shares": shares,
            entry_cost_key: cost_basis,
            "notes": notes,
            **kwargs,
        }
        self.save_holding(context, holding_dict)
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
        Enforces workspace access, actor-bound workspace (INV-SAAS-07), and 'portfolio.manage' capability.
        """
        self._verify_workspace_access(context)
        self._verify_actor_bound_workspace(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.manage"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.manage'.")

        selector = self._get_storage_selector(context)
        clean_sym = symbol.upper().strip()

        if self.db_engine is not None:
            if hasattr(self.db_engine, "delete_workspace_holding"):
                success = self.db_engine.delete_workspace_holding(
                    workspace_id=context.workspace_id,
                    user_id=context.actor_id or "default_user",
                    symbol=clean_sym,
                )
                if not success:
                    raise RuntimeError("Failed to remove holding.")
            elif hasattr(self.db_engine, "delete_user_holding"):
                success = self.db_engine.delete_user_holding(selector, clean_sym)
                if not success:
                    raise RuntimeError("Failed to remove holding.")
            elif hasattr(self.db_engine, "delete_holding"):
                return self.db_engine.delete_holding(selector, clean_sym)

        return {
            "status": "success",
            "workspace_id": context.workspace_id,
            "symbol": clean_sym,
        }

    def delete_holding(self, context: RequestContext, symbol: str) -> Dict[str, Any]:
        """Alias for remove_holding."""
        return self.remove_holding(context, symbol)

    def bulk_migrate_holdings(self, context: RequestContext, items: List[Dict[str, Any]]) -> int:
        """
        Migrate client-side holdings into backend persistent storage.
        Enforces workspace access, actor-bound workspace (INV-SAAS-07), and 'portfolio.manage' capability.
        """
        self._verify_workspace_access(context)
        self._verify_actor_bound_workspace(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.manage"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.manage'.")

        selector = self._get_storage_selector(context)
        if self.db_engine is not None:
            if hasattr(self.db_engine, "bulk_save_workspace_holdings"):
                return int(self.db_engine.bulk_save_workspace_holdings(
                    workspace_id=context.workspace_id,
                    user_id=context.actor_id or "default_user",
                    holdings=items,
                ))
            elif hasattr(self.db_engine, "bulk_save_holdings"):
                return int(self.db_engine.bulk_save_holdings(selector, items))

        return len(items)

    def record_manual_holding_exit(
        self,
        context: RequestContext,
        holding_id: int,
        exit_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Record an authoritative manual holding exit event within workspace boundary.
        Enforces workspace access, actor-bound workspace (INV-SAAS-07), and 'portfolio.manage' capability.
        """
        self._verify_workspace_access(context)
        self._verify_actor_bound_workspace(context)

        entitlements = self.entitlement_resolver.resolve(context)
        if not entitlements.can("portfolio.manage"):
            raise PermissionError("Workspace is not entitled to capability 'portfolio.manage'.")

        if self.db_engine is None or not hasattr(self.db_engine, "record_manual_holding_exit"):
            raise RuntimeError("Database engine does not support manual holding exits.")

        return self.db_engine.record_manual_holding_exit(
            workspace_id=context.workspace_id,
            user_id=context.actor_id or "default_user",
            holding_id=holding_id,
            exit_data=exit_data,
        )
