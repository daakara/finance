"""
Unit tests for Application Service interfaces and contracts (Wave 1F-A).

Verifies:
1. Workspace access authorization checks (WorkspaceAuthorizer)
2. Entitlement capability checks (EntitlementResolver)
3. Numeric limit enforcement (e.g. portfolio.max_holdings)
4. Domain purity: zero passing of RequestContext/EntitlementSet into quant engines
5. Separation of authorization from entitlements
6. Zero commercial plan names in service code
"""

import ast
import hashlib
import inspect
import pytest
from unittest.mock import MagicMock

from api.context.request_context import RequestContext
from api.services.authorizer import (
    WorkspaceAuthorizer,
    DefaultWorkspaceAuthorizer,
    WORKSPACE_MEMBERSHIP_RESOLUTION,
)
from api.services.entitlement_resolver import (
    EntitlementSet,
    DefaultEntitlementResolver,
)
from api.services.portfolio_service import PortfolioApplicationService
from api.services.journal_service import JournalApplicationService
from api.services.cockpit_service import CockpitApplicationService


# --- 1. Workspace Authorizer Tests ---

def test_authorizer_ws_default_access():
    """Verify ws_default is accessible anonymously and with actor."""
    authorizer = DefaultWorkspaceAuthorizer()
    assert authorizer.authorize_workspace_access(None, "ws_default") is True
    assert authorizer.authorize_workspace_access("trader_bob", "ws_default") is True


def test_authorizer_personal_workspace_access():
    """Verify personal deterministic workspace requires matching actor."""
    authorizer = DefaultWorkspaceAuthorizer()
    actor = "trader_alice"
    actor_hash = hashlib.sha256(actor.encode("utf-8")).hexdigest()[:16]
    valid_ws = f"ws_usr_{actor_hash}"
    foreign_ws = "ws_usr_0123456789abcdef"

    # Matching actor
    assert authorizer.authorize_workspace_access(actor, valid_ws) is True
    # Anonymous actor accessing personal workspace -> False
    assert authorizer.authorize_workspace_access(None, valid_ws) is False
    # Mismatched actor
    assert authorizer.authorize_workspace_access("trader_charlie", valid_ws) is False
    # Foreign workspace
    assert authorizer.authorize_workspace_access(actor, foreign_ws) is False


def test_authorizer_unknown_workspace_fail_closed():
    """Verify arbitrary unknown workspaces fail closed in pre-membership phase."""
    authorizer = DefaultWorkspaceAuthorizer()
    assert authorizer.authorize_workspace_access("trader_alice", "ws_custom_tenant") is False
    assert authorizer.authorize_workspace_access(None, "ws_custom_tenant") is False


# --- 2. PortfolioApplicationService Tests ---

def test_portfolio_service_authorization_enforced():
    """Verify portfolio operations fail closed if workspace unauthorized."""
    service = PortfolioApplicationService()
    unauthorized_ctx = RequestContext(
        actor_id="trader_alice",
        workspace_id="ws_unauthorized",
        request_id="req-1",
    )

    with pytest.raises(PermissionError, match="not authorized"):
        service.get_portfolio(unauthorized_ctx)

    with pytest.raises(PermissionError, match="not authorized"):
        service.add_holding(unauthorized_ctx, "AAPL", 10.0, 150.0)

    with pytest.raises(PermissionError, match="not authorized"):
        service.remove_holding(unauthorized_ctx, "AAPL")


def test_portfolio_service_capability_enforced():
    """Verify portfolio operations fail if capability is not entitled."""
    # Resolver denying capabilities
    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = EntitlementSet(capabilities={})

    service = PortfolioApplicationService(entitlement_resolver=mock_resolver)
    valid_ctx = RequestContext(actor_id=None, workspace_id="ws_default", request_id="req-1")

    with pytest.raises(PermissionError, match="not entitled to capability 'portfolio.read'"):
        service.get_portfolio(valid_ctx)

    with pytest.raises(PermissionError, match="not entitled to capability 'portfolio.manage'"):
        service.add_holding(valid_ctx, "AAPL", 10.0, 150.0)


def test_portfolio_service_limit_enforced():
    """Verify holdings limit is enforced when max_holdings is exceeded."""
    # Resolver with limit of 2 holdings
    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = EntitlementSet(
        capabilities={"portfolio.read": True, "portfolio.manage": True},
        limits={"portfolio.max_holdings": 2},
    )

    mock_db = MagicMock()
    # Mock current holding count of 2
    mock_db.get_user_portfolio.return_value = {"holdings": [{"symbol": "A"}, {"symbol": "B"}]}

    service = PortfolioApplicationService(
        entitlement_resolver=mock_resolver,
        db_engine=mock_db,
    )
    valid_ctx = RequestContext(actor_id=None, workspace_id="ws_default", request_id="req-1")

    with pytest.raises(ValueError, match="exceeds limit of 2"):
        service.add_holding(valid_ctx, "C", 5.0, 100.0)


# --- 3. JournalApplicationService Tests ---

def test_journal_service_authorization_enforced():
    """Verify journal operations fail closed if workspace unauthorized."""
    service = JournalApplicationService()
    unauthorized_ctx = RequestContext(
        actor_id="trader_alice",
        workspace_id="ws_unauthorized",
        request_id="req-1",
    )

    with pytest.raises(PermissionError, match="not authorized"):
        service.get_trades(unauthorized_ctx)

    with pytest.raises(PermissionError, match="not authorized"):
        service.record_trade(unauthorized_ctx, {"symbol": "MSFT"})

    with pytest.raises(PermissionError, match="not authorized"):
        service.get_telemetry(unauthorized_ctx)


def test_journal_service_capability_enforced():
    """Verify journal operations fail if capability is not entitled."""
    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = EntitlementSet(capabilities={})

    service = JournalApplicationService(entitlement_resolver=mock_resolver)
    valid_ctx = RequestContext(actor_id=None, workspace_id="ws_default", request_id="req-1")

    with pytest.raises(PermissionError, match="not entitled to capability 'journal.read'"):
        service.get_trades(valid_ctx)

    with pytest.raises(PermissionError, match="not entitled to capability 'journal.write'"):
        service.record_trade(valid_ctx, {"symbol": "MSFT"})


# --- 4. CockpitApplicationService Tests ---

def test_cockpit_service_authorization_enforced():
    """Verify cockpit operations fail closed if workspace unauthorized."""
    service = CockpitApplicationService()
    unauthorized_ctx = RequestContext(
        actor_id="trader_alice",
        workspace_id="ws_unauthorized",
        request_id="req-1",
    )

    with pytest.raises(PermissionError, match="not authorized"):
        service.get_cockpit_state(unauthorized_ctx)

    with pytest.raises(PermissionError, match="not authorized"):
        service.update_profile(unauthorized_ctx, {"lhi": 80.0})


def test_cockpit_service_strategy_a_separation():
    """Verify cockpit service distinguishes actor profile from workspace state."""
    mock_db = MagicMock()
    mock_db.get_user_profile.return_value = {"lhi": 85.0, "hhi": 90.0}
    mock_db.get_user_actions.return_value = [{"id": "act-1", "title": "Review stop"}]

    service = CockpitApplicationService(db_engine=mock_db)
    ctx = RequestContext(actor_id="trader_alice", workspace_id="ws_usr_test", request_id="req-1")

    # Authorizer grant for this test
    service.authorizer.authorize_workspace_access = MagicMock(return_value=True)

    result = service.get_cockpit_state(ctx)
    assert result["actor_id"] == "trader_alice"
    assert result["workspace_id"] == "ws_usr_test"
    assert result["profile"]["lhi"] == 85.0
    mock_db.get_user_profile.assert_called_with("trader_alice")


# --- 5. Architectural & Domain Purity Tests ---

@pytest.mark.parametrize("service_cls", [
    PortfolioApplicationService,
    JournalApplicationService,
    CockpitApplicationService,
])
def test_services_ast_no_commercial_plans(service_cls):
    """Verify service implementations contain zero commercial plan names or prices."""
    src = inspect.getsource(service_cls)
    forbidden = ["pro_plan", "free_plan", "fund_tier", "stripe", "price", "billing"]
    for word in forbidden:
        assert word not in src.lower(), f"Forbidden word '{word}' found in {service_cls.__name__}"


@pytest.mark.parametrize("service_cls", [
    PortfolioApplicationService,
    JournalApplicationService,
    CockpitApplicationService,
])
def test_services_ast_no_quant_engine_imports(service_cls):
    """Verify services do not import or couple directly to protected quant engines."""
    module_obj = inspect.getmodule(service_cls)
    src = inspect.getsource(module_obj)
    forbidden_imports = [
        "optimalexecutionengine",
        "confluenceengine",
        "decisionhierarchyengine",
        "decisiontraceengine",
        "technical_engine",
    ]
    for eng in forbidden_imports:
        assert eng not in src.lower(), f"Forbidden quant engine import '{eng}' found in {service_cls.__name__}"
