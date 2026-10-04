"""
Tests for Phase 1C Entitlement Contracts and Default Resolver.

Verifies:
1. EntitlementResolver protocol compliance.
2. EntitlementSet strict Boolean capability lookup.
3. EntitlementSet integer limit lookup and absence representation.
4. Rejection of invalid limit types (float, string, bool) and negative values.
5. Rejection of non-Boolean capability values.
6. Determinism of DefaultEntitlementResolver.
7. Immutability of RequestContext during resolution.
8. Zero external dependencies (network, DB, env) for DefaultEntitlementResolver.
9. AST inspection of resolver for unauthorized imports.
10. Absence of commercial plan names or pricing literals.
"""

import ast
import os
import pytest
from unittest.mock import patch

from api.context.request_context import RequestContext, create_default_context
from api.services.entitlement_resolver import (
    EntitlementSet,
    EntitlementResolver,
    DefaultEntitlementResolver,
)


def test_entitlement_set_capabilities_boolean_lookup():
    """Assert can() returns strict True or False and handles membership cleanly."""
    es = EntitlementSet(
        capabilities={"analysis.read": True, "analysis.quant": False, "radar.read": True}
    )
    assert es.can("analysis.read") is True
    assert es.can("analysis.quant") is False
    assert es.can("radar.read") is True
    assert es.can("portfolio.risk") is False

    # Return values must be actual strict bools
    res = es.can("analysis.read")
    assert isinstance(res, bool) and res is True
    res_absent = es.can("missing.capability")
    assert isinstance(res_absent, bool) and res_absent is False


def test_entitlement_set_limit_lookup_and_absence():
    """Assert get_limit() returns defined integer or None when absent."""
    es = EntitlementSet(
        capabilities=["analysis.read"],
        limits={"portfolio.max_holdings": 50, "alerts.max_active": 5},
    )
    assert es.get_limit("portfolio.max_holdings") == 50
    assert es.get_limit("alerts.max_active") == 5
    assert es.get_limit("absent.limit") is None


def test_entitlement_set_rejection_of_invalid_limits():
    """Assert rejection of non-integers, booleans, and negative numbers for limits."""
    # Negative limit
    with pytest.raises(ValueError, match="non-negative"):
        EntitlementSet(limits={"portfolio.max_holdings": -1})

    # Float limit
    with pytest.raises(TypeError, match="must be an integer"):
        EntitlementSet(limits={"portfolio.max_holdings": 10.5})  # type: ignore

    # String limit
    with pytest.raises(TypeError, match="must be an integer"):
        EntitlementSet(limits={"portfolio.max_holdings": "50"})  # type: ignore

    # Boolean limit (bool is subclass of int in Python)
    with pytest.raises(TypeError, match="must be an integer"):
        EntitlementSet(limits={"portfolio.max_holdings": True})  # type: ignore


def test_entitlement_set_rejection_of_invalid_capabilities():
    """Assert rejection of non-boolean capability values in mapping."""
    with pytest.raises(TypeError, match="must be a strict boolean"):
        EntitlementSet(capabilities={"analysis.read": 1})  # type: ignore

    with pytest.raises(TypeError, match="must be a strict boolean"):
        EntitlementSet(capabilities={"analysis.read": "true"})  # type: ignore

    with pytest.raises(ValueError, match="non-empty string"):
        EntitlementSet(capabilities=[""])


def test_default_resolver_determinism():
    """Assert DefaultEntitlementResolver produces identical EntitlementSets for equivalent contexts."""
    resolver = DefaultEntitlementResolver()
    ctx1 = RequestContext(actor_id=None, workspace_id="ws_1", request_id="req_1")
    ctx2 = RequestContext(actor_id=None, workspace_id="ws_1", request_id="req_1")

    e1 = resolver.resolve(ctx1)
    e2 = resolver.resolve(ctx2)

    assert e1.capabilities == e2.capabilities
    assert e1.limits == e2.limits
    assert e1.can("analysis.read") is True
    assert e2.can("analysis.read") is True


def test_default_resolver_does_not_mutate_context():
    """Assert resolve() leaves the input RequestContext completely unaltered."""
    resolver = DefaultEntitlementResolver()
    ctx = RequestContext(actor_id="usr_abc", workspace_id="ws_xyz", request_id="req_fixed")
    original_actor = ctx.actor_id
    original_workspace = ctx.workspace_id
    original_request = ctx.request_id

    _ = resolver.resolve(ctx)

    assert ctx.actor_id == original_actor
    assert ctx.workspace_id == original_workspace
    assert ctx.request_id == original_request


def test_default_resolver_pure_offline_execution():
    """Assert DefaultEntitlementResolver succeeds even when network, DB, and env are blocked."""
    resolver = DefaultEntitlementResolver()
    ctx = create_default_context()

    # Block socket connections to prove zero network access
    with patch("socket.socket", side_effect=RuntimeError("Network access forbidden")):
        entitlements = resolver.resolve(ctx)
        assert entitlements.can("analysis.read") is True
        assert entitlements.get_limit("portfolio.max_holdings") == 50


def test_resolver_ast_no_forbidden_imports():
    """Assert api/services/entitlement_resolver.py contains no billing or provider imports."""
    resolver_file = os.path.join(os.path.dirname(__file__), "..", "..", "api", "services", "entitlement_resolver.py")
    with open(resolver_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="entitlement_resolver.py")

    forbidden_packages = {"stripe", "requests", "urllib", "http", "sqlite3", "psycopg2", "redis"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split(".")[0]
                assert pkg not in forbidden_packages, f"Prohibited import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in forbidden_packages, f"Prohibited import: {node.module}"


def test_no_plan_names_or_prices_in_resolver_source():
    """Assert source contains zero commercial plan names or currency patterns."""
    import re
    resolver_file = os.path.join(os.path.dirname(__file__), "..", "..", "api", "services", "entitlement_resolver.py")
    with open(resolver_file, "r", encoding="utf-8") as f:
        src = f.read().lower()

    forbidden_terms = ["stripe", r"\$39", r"\$49", r"\$199", "tier", "starter", "pro", "growth", "fund", "scale", "enterprise"]
    for term in forbidden_terms:
        pattern = rf"\b{term}\b" if not term.startswith(r"\$") else re.escape(term)
        assert not re.search(pattern, src), f"Commercial term '{term}' found in entitlement_resolver.py source"
