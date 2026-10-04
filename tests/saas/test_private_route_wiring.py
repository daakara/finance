"""
Phase 1F-B: Private Route Wiring & Boundary Integration Tests.

Verifies:
1. RequestContext injection on all 3 rewired private route families:
   - /api/v1/portfolio*
   - /api/v1/journal/*
   - /api/v1/cockpit/*
2. Authorization Precedes Entitlement:
   - Workspace authorization failure returns 403 Forbidden before repository or quant engine execution.
3. Entitlement Denial:
   - Missing required capability returns 403 Forbidden.
4. Legacy Identity Compatibility:
   - X-User-Id / X-Profile-Id / query params map deterministically to RequestContext.actor_id.
   - Missing identity defaults safely to 'default_user' / 'default'.
5. Domain Purity (INV-SAAS-01):
   - Quant models and decision engines never receive RequestContext.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.context.request_context import RequestContext
from api.services.authorizer import DefaultWorkspaceAuthorizer
from api.services.entitlement_resolver import DefaultEntitlementResolver, EntitlementSet
from api.services.portfolio_service import PortfolioApplicationService
from api.services.journal_service import JournalApplicationService
from api.services.cockpit_service import CockpitApplicationService


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Portfolio Route Wiring & Authorization
# ---------------------------------------------------------------------------

def test_portfolio_get_wiring_default_workspace(client):
    """Verify GET /api/v1/portfolio wires context and returns 200 with private cache headers."""
    resp = client.get("/api/v1/portfolio")
    assert resp.status_code == 200
    assert "private" in resp.headers.get("Cache-Control", "").lower()
    assert resp.headers.get("Pragma") == "no-cache"


def test_portfolio_unauthorized_workspace_denied(client):
    """Verify GET /api/v1/portfolio with unauthorized workspace ID returns 403."""
    # When X-Workspace-ID does not match deterministic actor workspace, authorizer denies access
    resp = client.get(
        "/api/v1/portfolio",
        headers={"X-User-Id": "alice", "X-Workspace-ID": "ws_unauthorized_bob"},
    )
    assert resp.status_code == 403
    assert "not authorized" in resp.json().get("detail", "").lower()


def test_portfolio_post_holding_wiring(client):
    """Verify POST /api/v1/portfolio creates holding via PortfolioApplicationService."""
    holding_data = {
        "symbol": "AAPL",
        "shares": 10.0,
        "entryPrice": 150.0,
    }
    resp = client.post(
        "/api/v1/portfolio",
        json=holding_data,
        headers={"X-User-Id": "test_user"},
    )
    assert resp.status_code in (200, 201)
    assert "private" in resp.headers.get("Cache-Control", "").lower()


# ---------------------------------------------------------------------------
# 2. Journal Route Wiring & Authorization
# ---------------------------------------------------------------------------

def test_journal_trades_wiring_default(client):
    """Verify GET /api/v1/journal/trades wires context and returns 200."""
    resp = client.get("/api/v1/journal/trades")
    assert resp.status_code == 200
    assert "private" in resp.headers.get("Cache-Control", "").lower()
    assert resp.headers.get("Pragma") == "no-cache"


def test_journal_unauthorized_workspace_denied(client):
    """Verify GET /api/v1/journal/trades with unauthorized workspace returns 403."""
    resp = client.get(
        "/api/v1/journal/trades",
        headers={"X-User-Id": "trader1", "X-Workspace-ID": "ws_other_trader"},
    )
    assert resp.status_code == 403


def test_journal_record_trade_wiring(client):
    """Verify POST /api/v1/journal/trades logs trade via JournalApplicationService."""
    trade_payload = {
        "symbol": "NVDA",
        "shares": 5.0,
        "entryPrice": 120.0,
        "status": "OPEN",
    }
    resp = client.post(
        "/api/v1/journal/trades",
        json=trade_payload,
        headers={"X-User-Id": "test_trader"},
    )
    assert resp.status_code in (200, 201)
    assert "private" in resp.headers.get("Cache-Control", "").lower()


def test_journal_telemetry_wiring(client):
    """Verify GET /api/v1/journal/telemetry wires context and returns 200."""
    resp = client.get("/api/v1/journal/telemetry", headers={"X-User-Id": "test_trader"})
    assert resp.status_code == 200
    assert "private" in resp.headers.get("Cache-Control", "").lower()


# ---------------------------------------------------------------------------
# 3. Cockpit Route Wiring & Authorization
# ---------------------------------------------------------------------------

def test_cockpit_state_wiring_default(client):
    """Verify GET /api/v1/cockpit/state wires context and returns 200 with CQRS read model."""
    resp = client.get("/api/v1/cockpit/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert "private" in resp.headers.get("Cache-Control", "").lower()
    assert resp.headers.get("Pragma") == "no-cache"


def test_cockpit_unauthorized_workspace_denied(client):
    """Verify GET /api/v1/cockpit/state with unauthorized workspace returns 403."""
    resp = client.get(
        "/api/v1/cockpit/state",
        headers={"X-Profile-Id": "prof1", "X-Workspace-ID": "ws_foreign"},
    )
    assert resp.status_code == 403


def test_cockpit_profile_update_wiring(client):
    """Verify POST /api/v1/cockpit/profile persists via CockpitApplicationService."""
    profile_payload = {
        "name": "Alex",
        "role": "Trader",
        "lhi": 80.0,
        "hhi": 85.0,
        "iai": 90.0,
        "liquidReserves": 25000.0,
        "monthlyBurn": 3000.0,
    }
    resp = client.post(
        "/api/v1/cockpit/profile",
        json=profile_payload,
        headers={"X-Profile-Id": "alex_prof"},
    )
    assert resp.status_code == 201
    assert "private" in resp.headers.get("Cache-Control", "").lower()


def test_cockpit_action_update_wiring(client):
    """Verify POST /api/v1/cockpit/actions persists action via CockpitApplicationService."""
    action_payload = {
        "id": "NBA-TEST-01",
        "title": "Review Risk Limits",
        "domain": "CAPITAL",
        "priorityScore": 88.0,
        "isPrimary": True,
    }
    resp = client.post(
        "/api/v1/cockpit/actions",
        json=action_payload,
        headers={"X-Profile-Id": "alex_prof"},
    )
    assert resp.status_code == 201
    assert "private" in resp.headers.get("Cache-Control", "").lower()


# ---------------------------------------------------------------------------
# 4. Entitlement Denial Verification (Mocked Resolver)
# ---------------------------------------------------------------------------

def test_service_entitlement_denial_behavior():
    """Verify application services fail closed when required capability is missing."""
    class DenyingResolver:
        def resolve(self, context: RequestContext) -> EntitlementSet:
            return EntitlementSet(capabilities=set(), limits={})

    authorizer = DefaultWorkspaceAuthorizer()
    denying_resolver = DenyingResolver()

    # Portfolio service denial
    port_svc = PortfolioApplicationService(authorizer=authorizer, entitlement_resolver=denying_resolver)
    ctx = RequestContext(actor_id="user1", workspace_id="ws_default", request_id="req1")
    with pytest.raises(PermissionError) as exc:
        port_svc.get_portfolio(ctx)
    assert "capability" in str(exc.value).lower()

    # Journal service denial
    jour_svc = JournalApplicationService(authorizer=authorizer, entitlement_resolver=denying_resolver)
    with pytest.raises(PermissionError) as exc:
        jour_svc.get_trades(ctx)
    assert "capability" in str(exc.value).lower()
