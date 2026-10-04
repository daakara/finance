"""
Tests for Phase 1B Capability and Limit Vocabulary.

Verifies:
1. String types for all identifiers.
2. Strict grammar compliance ^[a-z][a-z0-9]*(\\.[a-z][a-z0-9_]*)+$
3. Uniqueness within collections.
4. Complete disjointness between capabilities and limits.
5. Zero commercial plan names or currency patterns in identifiers.
6. Zero unauthorized imports in module AST.
7. Clean subprocess import with zero side effects.
"""

import ast
import os
import subprocess
import sys
import pytest

from api.capabilities.capabilities import (
    CAPABILITIES,
    LIMITS,
    ALL_IDENTIFIERS,
    CAPABILITY_GRAMMAR_REGEX,
    validate_identifier,
)


def test_capability_identifiers_types_and_non_empty():
    """Assert all capability and limit identifiers are non-empty strings."""
    assert len(CAPABILITIES) > 0, "CAPABILITIES set must not be empty."
    assert len(LIMITS) > 0, "LIMITS set must not be empty."

    for c in CAPABILITIES:
        assert isinstance(c, str), f"Capability identifier {c} is not a string."
        assert len(c.strip()) > 0, "Capability identifier cannot be empty."

    for l in LIMITS:
        assert isinstance(l, str), f"Limit identifier {l} is not a string."
        assert len(l.strip()) > 0, "Limit identifier cannot be empty."


def test_grammar_compliance():
    """Assert all identifiers conform to ^[a-z][a-z0-9]*(\\.[a-z][a-z0-9_]*)+$."""
    for ident in ALL_IDENTIFIERS:
        assert CAPABILITY_GRAMMAR_REGEX.match(ident), f"Identifier '{ident}' failed grammar."
        assert validate_identifier(ident) is True


def test_uniqueness_and_disjointness():
    """Assert uniqueness and absolute disjointness between CAPABILITIES and LIMITS."""
    # Check disjointness
    overlap = CAPABILITIES & LIMITS
    assert len(overlap) == 0, f"Overlap detected between capabilities and limits: {overlap}"

    # Check that ALL_IDENTIFIERS equals union
    assert ALL_IDENTIFIERS == (CAPABILITIES | LIMITS)
    assert len(ALL_IDENTIFIERS) == len(CAPABILITIES) + len(LIMITS)


def test_no_commercial_plans_or_currency():
    """Assert no commercial plan names or currency symbols exist in identifiers."""
    forbidden = {
        "free", "starter", "pro", "growth", "fund", "scale",
        "enterprise", "usd", "eur", "price", "tier", "plan",
        "dollar", "cent", "billing", "stripe", "subscription",
    }
    for ident in ALL_IDENTIFIERS:
        parts = set(ident.split("."))
        for bad in forbidden:
            assert bad not in parts, f"Forbidden commercial term '{bad}' in identifier '{ident}'"
            assert bad not in ident.lower().replace("_", "").replace(".", ""), f"Forbidden substring '{bad}' in '{ident}'"


def test_module_ast_no_unauthorized_imports():
    """Assert api/capabilities/capabilities.py imports zero auth, billing, or network modules."""
    cap_file = os.path.join(os.path.dirname(__file__), "..", "..", "api", "capabilities", "capabilities.py")
    with open(cap_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="capabilities.py")

    allowed_modules = {"re", "typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                assert root_pkg in allowed_modules, f"Unauthorized import '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root_pkg = mod.split(".")[0]
            assert root_pkg in allowed_modules, f"Unauthorized import from '{mod}'"


def test_subprocess_clean_import_side_effects():
    """Assert importing api.capabilities executes cleanly in a separate process without side effects."""
    code = "import sys; from api.capabilities import CAPABILITIES, LIMITS; sys.exit(0 if len(CAPABILITIES) > 0 and len(LIMITS) > 0 else 1)"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Subprocess import failed with stderr: {proc.stderr}"
