"""
Tests for SaaS Boundary Isolation and Architectural Seams Integrity.

Verifies:
1. Public canonical analytics (/api/v1/analytics/{symbol}) has zero dependencies on SaaS context or capabilities.
2. Public radar routes (/api/v1/radar) have zero dependencies on SaaS context or capabilities.
3. Seams contracts (RequestContext, EntitlementSet, DefaultEntitlementResolver) adhere to purity standards.
"""

import ast
import os
import pytest

from api.context import RequestContext, create_default_context
from api.capabilities import CAPABILITIES, LIMITS, ALL_IDENTIFIERS
from api.services import EntitlementSet, EntitlementResolver, DefaultEntitlementResolver

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SAAS_MODULE_PREFIXES = ("api.context", "api.capabilities", "api.services")
SAAS_TYPE_NAMES = {
    "requestcontext",
    "entitlementset",
    "entitlementresolver",
    "defaultentitlementresolver",
    "capabilities",
    "limits",
}


def _check_route_ast_for_saas(route_rel_path: str):
    full_path = os.path.join(REPO_ROOT, route_rel_path)
    assert os.path.exists(full_path), f"Route file missing: {route_rel_path}"

    with open(full_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=route_rel_path)

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in SAAS_MODULE_PREFIXES:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        violations.append(f"{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for prefix in SAAS_MODULE_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{node.lineno}: from {mod} import ...")
            for alias in node.names:
                if alias.name.lower() in SAAS_TYPE_NAMES:
                    violations.append(f"{node.lineno}: imported symbol {alias.name}")

    assert len(violations) == 0, (
        f"Route {route_rel_path} illegally references SaaS boundary:\n" + "\n".join(violations)
    )


REWIRED_PRIVATE_ROUTES = {"portfolio.py", "journal.py", "cockpit.py"}


def test_public_and_unrewired_routes_boundary_isolation():
    """Verify all public and unmigrated routes in api/routes/ remain completely isolated from SaaS."""
    routes_dir = os.path.join(REPO_ROOT, "api", "routes")
    route_files = [f for f in os.listdir(routes_dir) if f.endswith(".py") and f != "__init__.py"]
    assert len(route_files) >= 5, f"Expected at least 5 route files, found {len(route_files)}"

    for rfile in route_files:
        if rfile not in REWIRED_PRIVATE_ROUTES:
            rel_path = os.path.join("api", "routes", rfile)
            _check_route_ast_for_saas(rel_path)


def test_rewired_private_routes_use_dedicated_application_services():
    """Verify rewired private routes correctly integrate with their application service seams."""
    routes_dir = os.path.join(REPO_ROOT, "api", "routes")
    expected_service_wiring = {
        "portfolio.py": ("PortfolioApplicationService", "portfolio_service"),
        "journal.py": ("JournalApplicationService", "journal_service"),
        "cockpit.py": ("CockpitApplicationService", "cockpit_service"),
    }
    for rfile, (service_cls, service_inst) in expected_service_wiring.items():
        rel_path = os.path.join(routes_dir, rfile)
        with open(rel_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert service_cls in content, f"Expected {service_cls} in {rfile}"
        assert service_inst in content, f"Expected {service_inst} in {rfile}"
        assert "RequestContext" in content, f"Expected RequestContext in {rfile}"
        assert "resolve_request_context" in content, f"Expected resolve_request_context in {rfile}"


def test_request_context_contract_integrity():
    """Verify RequestContext contract matches exact specification."""
    ctx = create_default_context("test_req")
    assert ctx.actor_id is None
    assert ctx.workspace_id == "ws_default"
    assert ctx.request_id == "test_req"

    with pytest.raises(Exception):
        ctx.workspace_id = "other"  # Immutability check


def test_entitlement_set_contract_integrity():
    """Verify EntitlementSet contract adheres to strict typing and immutability."""
    es = EntitlementSet(capabilities={"analysis.read"}, limits={"portfolio.max_holdings": 25})
    assert es.can("analysis.read") is True
    assert es.can("analysis.quant") is False
    assert es.get_limit("portfolio.max_holdings") == 25
    assert es.get_limit("alerts.max_active") is None

    # Cannot mutate internal dicts
    with pytest.raises(Exception):
        es.capabilities.add("analysis.quant")


def test_default_resolver_offline_compliance():
    """Verify DefaultEntitlementResolver resolves synchronously and deterministically."""
    resolver = DefaultEntitlementResolver()
    ctx = create_default_context()
    es = resolver.resolve(ctx)
    assert isinstance(es, EntitlementSet)
    assert es.can("analysis.read") is True
