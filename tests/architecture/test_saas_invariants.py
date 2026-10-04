"""
Architectural Invariant Test: INV-SAAS-01

Quantitative and Analytical Purity Invariant:
Authoritative quantitative models, analytical engines, and decision-hierarchy
logic must remain pure domain functions. They must NEVER import, accept, or
reference SaaS concepts such as RequestContext, User, Workspace, Subscription,
Plan, BillingCustomer, PaymentStatus, SeatCount, or EntitlementSet.
"""

import ast
import os
import pytest
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PROTECTED_QUANT_MODULES = [
    "analyst_dashboard/analyzers/advanced_risk_analyzer.py",
    "analyst_dashboard/analyzers/catalysts.py",
    "analyst_dashboard/analyzers/confluence_engine.py",
    "analyst_dashboard/analyzers/decision_hierarchy.py",
    "analyst_dashboard/analyzers/decision_trace.py",
    "analyst_dashboard/analyzers/market_graph.py",
    "analyst_dashboard/analyzers/optimal_execution.py",
    "analyst_dashboard/analyzers/self_healing_engine.py",
    "analyst_dashboard/analyzers/smart_money.py",
    "analyst_dashboard/analyzers/trader_archetypes.py",
    "analyst_dashboard/data/market_price_state.py",
    "engines/technical_engine.py",
]

FORBIDDEN_IMPORT_SYMBOLS = {
    "requestcontext",
    "user",
    "workspace",
    "subscription",
    "plan",
    "billingcustomer",
    "paymentstatus",
    "seatcount",
    "entitlementset",
    "entitlementresolver",
    "defaultentitlementresolver",
}

FORBIDDEN_IMPORT_MODULES = {
    "api.context",
    "api.capabilities",
    "api.services.entitlement_resolver",
}

FORBIDDEN_PARAM_NAMES = {
    "request_context",
    "req_context",
    "entitlement_set",
    "entitlements",
    "workspace_id",
    "actor_id",
    "subscription_tier",
    "billing_plan",
}


class QuantPurityVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name_lower = alias.name.lower()
            for forbidden_mod in FORBIDDEN_IMPORT_MODULES:
                if name_lower == forbidden_mod or name_lower.startswith(forbidden_mod + "."):
                    self.violations.append(
                        f"{self.filename}:{node.lineno} - Prohibited module import: '{alias.name}'"
                    )
            for forbidden_sym in FORBIDDEN_IMPORT_SYMBOLS:
                if name_lower == forbidden_sym:
                    self.violations.append(
                        f"{self.filename}:{node.lineno} - Prohibited symbol import: '{alias.name}'"
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = (node.module or "").lower()
        for forbidden_mod in FORBIDDEN_IMPORT_MODULES:
            if mod == forbidden_mod or mod.startswith(forbidden_mod + "."):
                self.violations.append(
                    f"{self.filename}:{node.lineno} - Prohibited module import from: '{node.module}'"
                )
        for alias in node.names:
            name_lower = alias.name.lower()
            if name_lower in FORBIDDEN_IMPORT_SYMBOLS:
                self.violations.append(
                    f"{self.filename}:{node.lineno} - Prohibited symbol imported: '{alias.name}' from '{node.module}'"
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._check_args(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._check_args(node)
        self.generic_visit(node)

    def _check_args(self, node):
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for arg in all_args:
            arg_name = arg.arg.lower()
            if arg_name in FORBIDDEN_PARAM_NAMES:
                self.violations.append(
                    f"{self.filename}:{node.lineno} - Function '{node.name}' has prohibited SaaS parameter '{arg.arg}'"
                )


def check_source_purity(source: str, filename: str = "test.py") -> List[str]:
    tree = ast.parse(source, filename=filename)
    visitor = QuantPurityVisitor(filename)
    visitor.visit(tree)
    return visitor.violations


def test_protected_quant_modules_exist():
    """Verify all 12 protected quant modules exist on disk."""
    for rel_path in PROTECTED_QUANT_MODULES:
        full_path = os.path.join(REPO_ROOT, rel_path)
        assert os.path.exists(full_path), f"Protected module missing: {rel_path}"


@pytest.mark.parametrize("rel_path", PROTECTED_QUANT_MODULES)
def test_inv_saas_01_quant_purity(rel_path: str):
    """
    INV-SAAS-01: Verify protected quant modules contain zero SaaS imports or parameters.
    """
    full_path = os.path.join(REPO_ROOT, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
        src = f.read()

    violations = check_source_purity(src, filename=rel_path)
    assert len(violations) == 0, (
        f"INV-SAAS-01 Violation detected in {rel_path}:\n" + "\n".join(violations)
    )


def test_quant_purity_visitor_fixture_detects_violations():
    """Verify QuantPurityVisitor correctly flags violations in synthetic code."""
    dirty_source_1 = """
from api.context import RequestContext

def calculate_confluence(symbol: str, context: RequestContext):
    return 42
"""
    violations = check_source_purity(dirty_source_1, "dirty1.py")
    assert len(violations) >= 1
    assert any("Prohibited module import from: 'api.context'" in v for v in violations)

    dirty_source_2 = """
def compute_trade_risk(symbol: str, workspace_id: str, actor_id: str):
    return 100
"""
    violations_2 = check_source_purity(dirty_source_2, "dirty2.py")
    assert len(violations_2) == 2
    assert any("prohibited SaaS parameter 'workspace_id'" in v for v in violations_2)
    assert any("prohibited SaaS parameter 'actor_id'" in v for v in violations_2)
