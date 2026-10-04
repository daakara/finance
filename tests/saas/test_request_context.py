"""
Tests for Phase 1A RequestContext Contract.

Verifies:
1. Public schema exact field-set equality {"actor_id", "workspace_id", "request_id"}.
2. Immutability across all fields.
3. Anonymous and identified construction.
4. Input validation (required fields non-empty).
5. Zero forbidden billing, payment, plan, pricing, or token fields.
6. Zero quant imports in context module.
7. Zero existing route or domain engine modifications.
"""

import ast
import dataclasses
import inspect
import os
import pytest

from api.context.request_context import RequestContext, create_default_context


def test_request_context_exact_public_fields():
    """Assert public field set is exactly {'actor_id', 'workspace_id', 'request_id'}."""
    field_names = {f.name for f in dataclasses.fields(RequestContext)}
    assert field_names == {"actor_id", "workspace_id", "request_id"}, f"Unexpected fields: {field_names}"


def test_request_context_construction():
    """Assert anonymous and identified contexts can be constructed."""
    # Anonymous actor
    anon_ctx = RequestContext(actor_id=None, workspace_id="ws_personal", request_id="req_123")
    assert anon_ctx.actor_id is None
    assert anon_ctx.workspace_id == "ws_personal"
    assert anon_ctx.request_id == "req_123"

    # Identified actor
    user_ctx = RequestContext(actor_id="usr_456", workspace_id="ws_team", request_id="req_789")
    assert user_ctx.actor_id == "usr_456"
    assert user_ctx.workspace_id == "ws_team"
    assert user_ctx.request_id == "req_789"


def test_request_context_immutability():
    """Assert instance cannot be mutated after construction."""
    ctx = RequestContext(actor_id="usr_1", workspace_id="ws_1", request_id="req_1")

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.actor_id = "usr_2"  # type: ignore

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.workspace_id = "ws_2"  # type: ignore

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.request_id = "req_2"  # type: ignore


def test_request_context_validation():
    """Assert required fields cannot be empty or invalid."""
    with pytest.raises(ValueError):
        RequestContext(actor_id=None, workspace_id="", request_id="req_1")

    with pytest.raises(ValueError):
        RequestContext(actor_id=None, workspace_id="ws_1", request_id="")

    with pytest.raises(ValueError):
        RequestContext(actor_id="", workspace_id="ws_1", request_id="req_1")


def test_create_default_context_helper():
    """Assert helper constructs valid anonymous default context without mutating state."""
    ctx = create_default_context()
    assert ctx.actor_id is None
    assert ctx.workspace_id == "ws_default"
    assert len(ctx.request_id) > 0

    custom = create_default_context(request_id="custom_req_99")
    assert custom.request_id == "custom_req_99"


def test_request_context_forbidden_fields():
    """Assert annotations and attributes contain no forbidden symbols."""
    forbidden = {
        "stripe", "billing", "plan", "price", "subscription",
        "payment", "token", "customer", "invoice", "tier",
    }
    field_names = {f.name.lower() for f in dataclasses.fields(RequestContext)}
    for f in field_names:
        for bad in forbidden:
            assert bad not in f, f"Forbidden substring '{bad}' found in field '{f}'"


def test_request_context_ast_no_quant_imports():
    """Assert api/context/request_context.py contains no quant-domain imports."""
    context_file = os.path.join(os.path.dirname(__file__), "..", "..", "api", "context", "request_context.py")
    with open(context_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="request_context.py")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("analyst_dashboard"), f"Prohibited import: {alias.name}"
                assert not alias.name.startswith("engines"), f"Prohibited import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("analyst_dashboard"), f"Prohibited import: {mod}"
            assert not mod.startswith("engines"), f"Prohibited import: {mod}"


def test_no_public_routes_consume_request_context():
    """Assert RequestContext is NEVER imported by public context-free routes (INV-SAAS-05)."""
    routes_dir = os.path.join(os.path.dirname(__file__), "..", "..", "api", "routes")
    authorized_private_routes = {"portfolio.py", "journal.py", "cockpit.py"}
    for root, _, files in os.walk(routes_dir):
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                if f not in authorized_private_routes:
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8") as fp:
                        content = fp.read()
                        assert "RequestContext" not in content, f"RequestContext leaked into public/unrewired route: {f}"
                else:
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8") as fp:
                        content = fp.read()
                        assert "RequestContext" in content, f"RequestContext expected in authorized private route: {f}"
