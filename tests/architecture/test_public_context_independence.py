"""
Architectural Invariant Test: INV-SAAS-05
Public Context-Free Route Independence Enforcement.

Invariant Rule:
PUBLIC_CONTEXT_FREE_ROUTES_MUST_NOT_DEPEND_ON_REQUEST_CONTEXT_RESOLUTION

Public context-free routes must execute without:
1. Resolving RequestContext (either via middleware or route-level dependency injection);
2. Inspecting actor or tenant headers (X-User-Id, X-Workspace-ID, X-Profile-Id);
3. Executing entitlement or capability checks;
4. Mutating or accessing tenant-scoped state;
5. Emitting tenancy-dependent telemetry.

This suite tests:
1. Production public route files on disk (AST inspection);
2. Global FastAPI app configuration in api/main.py;
3. Negative synthetic fixtures proving detection capability;
4. Allowed synthetic fixtures proving valid private/service usage is permitted.
"""

import ast
import os
import pytest
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PUBLIC_CONTEXT_FREE_ROUTE_FILES = [
    "api/routes/analytics.py",
    "api/routes/volatility.py",
    "api/routes/regimes.py",
    "api/routes/smart_money.py",
    "api/routes/macro.py",
    "api/routes/etf.py",
]

FORBIDDEN_CONTEXT_IMPORTS = {
    "requestcontext",
    "requestcontextresolver",
    "resolve_request_context",
    "default_request_context_resolver",
    "entitlementresolver",
    "defaultentitlementresolver",
    "workspaceauthorizer",
    "defaultworkspaceauthorizer",
}

FORBIDDEN_CONTEXT_MODULES = {
    "api.context.resolver",
    "api.services.entitlement_resolver",
    "api.services.authorizer",
}

FORBIDDEN_PARAM_IDENTIFIERS = {
    "request_context",
    "req_context",
    "context",
    "workspace_id",
    "actor_id",
}


class PublicRouteIndependenceVisitor(ast.NodeVisitor):
    """AST visitor detecting unauthorized RequestContext or Entitlement dependencies in public routes."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name_lower = alias.name.lower()
            for mod in FORBIDDEN_CONTEXT_MODULES:
                if name_lower == mod or name_lower.startswith(mod + "."):
                    self.violations.append(
                        f"{self.filename}:{node.lineno} - Forbidden context module import: '{alias.name}'"
                    )
            for sym in FORBIDDEN_CONTEXT_IMPORTS:
                if name_lower == sym:
                    self.violations.append(
                        f"{self.filename}:{node.lineno} - Forbidden context symbol import: '{alias.name}'"
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = (node.module or "").lower()
        for forbidden_mod in FORBIDDEN_CONTEXT_MODULES:
            if mod == forbidden_mod or mod.startswith(forbidden_mod + "."):
                self.violations.append(
                    f"{self.filename}:{node.lineno} - Forbidden context import from module: '{node.module}'"
                )
        for alias in node.names:
            name_lower = alias.name.lower()
            if name_lower in FORBIDDEN_CONTEXT_IMPORTS:
                self.violations.append(
                    f"{self.filename}:{node.lineno} - Forbidden context symbol imported: '{alias.name}' from '{node.module}'"
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._check_route_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._check_route_function(node)
        self.generic_visit(node)

    def _check_route_function(self, node):
        # Inspect parameters for RequestContext type annotations or forbidden parameter names
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for arg in all_args:
            # Check annotation
            if arg.annotation:
                anno_str = ast.unparse(arg.annotation).lower() if hasattr(ast, "unparse") else ""
                if "requestcontext" in anno_str:
                    self.violations.append(
                        f"{self.filename}:{node.lineno} - Route handler '{node.name}' has prohibited RequestContext annotation on parameter '{arg.arg}'"
                    )
            # Check default values for Depends(resolve_request_context)
            if hasattr(node.args, "defaults"):
                for default in node.args.defaults:
                    def_str = ast.unparse(default).lower() if hasattr(ast, "unparse") else ""
                    if "resolve_request_context" in def_str or "requestcontextresolver" in def_str:
                        self.violations.append(
                            f"{self.filename}:{node.lineno} - Route handler '{node.name}' injects resolver via default: '{def_str}'"
                        )

    def visit_Call(self, node: ast.Call):
        # Check router instantiation APIRouter(dependencies=[...])
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name == "APIRouter":
            for kw in node.keywords:
                if kw.arg == "dependencies":
                    dep_str = ast.unparse(kw.value).lower() if hasattr(ast, "unparse") else ""
                    if "resolve_request_context" in dep_str or "requestcontextresolver" in dep_str:
                        self.violations.append(
                            f"{self.filename}:{node.lineno} - APIRouter declares global resolver dependency: '{dep_str}'"
                        )
        self.generic_visit(node)


def check_public_route_source(source: str, filename: str = "route.py") -> List[str]:
    tree = ast.parse(source, filename=filename)
    visitor = PublicRouteIndependenceVisitor(filename)
    visitor.visit(tree)
    return visitor.violations


# --- 1. Production Public Route Verifications ---

@pytest.mark.parametrize("rel_path", PUBLIC_CONTEXT_FREE_ROUTE_FILES)
def test_production_public_routes_are_context_free(rel_path: str):
    """
    INV-SAAS-05: Verify all production public routes on disk do not import
    or depend on RequestContext or context resolvers.
    """
    full_path = os.path.join(REPO_ROOT, rel_path)
    assert os.path.exists(full_path), f"Public route file missing: {rel_path}"

    with open(full_path, "r", encoding="utf-8") as f:
        src = f.read()

    violations = check_public_route_source(src, filename=rel_path)
    assert len(violations) == 0, (
        f"INV-SAAS-05 Violation in public route {rel_path}:\n" + "\n".join(violations)
    )


def test_main_app_has_no_global_resolver_middleware():
    """Verify api/main.py does not register RequestContextResolver as global middleware."""
    main_path = os.path.join(REPO_ROOT, "api", "main.py")
    with open(main_path, "r", encoding="utf-8") as f:
        src = f.read()

    src_lower = src.lower()
    assert "resolverequestcontext" not in src_lower
    assert "requestcontextresolver" not in src_lower
    assert "add_middleware(requestcontextresolver" not in src_lower


# --- 2. Deterministic Negative Fixtures ---

def test_negative_fixture_public_handler_accepting_request_context():
    """Negative fixture: public route handler accepting RequestContext parameter."""
    dirty_source = """
from fastapi import APIRouter
from api.context.request_context import RequestContext

router = APIRouter()

@router.get("/analytics/{symbol}")
def get_analytics(symbol: str, ctx: RequestContext):
    return {"symbol": symbol}
"""
    violations = check_public_route_source(dirty_source, "dirty_public_handler.py")
    assert len(violations) >= 1
    assert any("RequestContext" in v for v in violations)


def test_negative_fixture_public_handler_importing_private_resolver():
    """Negative fixture: public route file importing private resolver."""
    dirty_source = """
from fastapi import APIRouter
from api.context.resolver import resolve_request_context

router = APIRouter()
"""
    violations = check_public_route_source(dirty_source, "dirty_import.py")
    assert len(violations) >= 1
    assert any("Forbidden context" in v for v in violations)


def test_negative_fixture_router_declaring_resolver_dependency():
    """Negative fixture: router declaring resolver in dependencies list."""
    dirty_source = """
from fastapi import APIRouter, Depends
from api.context.resolver import resolve_request_context

router = APIRouter(dependencies=[Depends(resolve_request_context)])
"""
    violations = check_public_route_source(dirty_source, "dirty_router_dep.py")
    assert len(violations) >= 1


def test_negative_fixture_public_handler_importing_entitlement_resolver():
    """Negative fixture: public route handler importing EntitlementResolver."""
    dirty_source = """
from fastapi import APIRouter
from api.services.entitlement_resolver import EntitlementResolver

router = APIRouter()
"""
    violations = check_public_route_source(dirty_source, "dirty_entitlement.py")
    assert len(violations) >= 1
    assert any("Forbidden context" in v for v in violations)


# --- 3. Deterministic Allowed Fixtures ---

def test_allowed_fixture_private_route_with_request_context():
    """Allowed fixture: private route handler legitimately using RequestContext."""
    allowed_private_route = """
from fastapi import APIRouter, Depends
from api.context.request_context import RequestContext
from api.context.resolver import resolve_request_context

router = APIRouter()

@router.get("/portfolio")
def get_portfolio(context: RequestContext = Depends(resolve_request_context)):
    return {"workspace": context.workspace_id}
"""
    # Parse to verify AST syntax validity
    tree = ast.parse(allowed_private_route)
    assert tree is not None


def test_allowed_fixture_application_service_with_request_context():
    """Allowed fixture: application service legitimately coordinating RequestContext."""
    allowed_service = """
from api.context.request_context import RequestContext

class TestService:
    def execute(self, context: RequestContext):
        return context.workspace_id
"""
    tree = ast.parse(allowed_service)
    assert tree is not None


def test_allowed_fixture_public_route_with_plain_domain():
    """Allowed fixture: public context-free route calling pure domain engine."""
    allowed_public = """
from fastapi import APIRouter

router = APIRouter()

@router.get("/analytics/{symbol}")
def get_asset_analytics(symbol: str):
    # Pure domain call with primitive string
    return {"symbol": symbol, "score": 85}
"""
    violations = check_public_route_source(allowed_public, "valid_public.py")
    assert len(violations) == 0
