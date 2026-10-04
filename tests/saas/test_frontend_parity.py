"""
Tests for Parity between Frontend and Backend SaaS Contracts.

Verifies:
1. Exact 1:1 identifier parity between Python and TypeScript capability lists.
2. Exact 1:1 identifier parity between Python and TypeScript limit lists.
3. Zero commercial plan names or currency symbols in frontend SaaS contracts.
4. Clean standalone TypeScript files without external dependencies.
"""

import os
import re
import pytest

from api.capabilities.capabilities import (
    CAPABILITIES as BACKEND_CAPABILITIES,
    LIMITS as BACKEND_LIMITS,
)


def _extract_ts_string_array(file_path: str, const_name: str) -> set:
    """Extract string values from a TypeScript array const."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = rf"export\s+const\s+{const_name}\s*=\s*\[(.*?)\]\s*as\s*const"
    match = re.search(pattern, content, re.DOTALL)
    assert match is not None, f"Could not find const {const_name} in {file_path}"

    array_body = match.group(1)
    items = re.findall(r'["\']([^"\']+)["\']', array_body)
    return set(items)


def test_capabilities_frontend_backend_parity():
    """Assert frontend CAPABILITIES array matches backend CAPABILITIES set exactly."""
    ts_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "lib", "saas", "capabilities.ts"
    )
    assert os.path.exists(ts_file), f"Frontend capabilities file does not exist: {ts_file}"

    frontend_caps = _extract_ts_string_array(ts_file, "CAPABILITIES")
    assert frontend_caps == BACKEND_CAPABILITIES, (
        f"Parity mismatch in CAPABILITIES.\n"
        f"Frontend only: {frontend_caps - BACKEND_CAPABILITIES}\n"
        f"Backend only: {BACKEND_CAPABILITIES - frontend_caps}"
    )


def test_limits_frontend_backend_parity():
    """Assert frontend LIMITS array matches backend LIMITS set exactly."""
    ts_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "lib", "saas", "capabilities.ts"
    )
    assert os.path.exists(ts_file), f"Frontend capabilities file does not exist: {ts_file}"

    frontend_limits = _extract_ts_string_array(ts_file, "LIMITS")
    assert frontend_limits == BACKEND_LIMITS, (
        f"Parity mismatch in LIMITS.\n"
        f"Frontend only: {frontend_limits - BACKEND_LIMITS}\n"
        f"Backend only: {BACKEND_LIMITS - frontend_limits}"
    )


def test_frontend_saas_no_commercial_plans_or_currency():
    """Assert frontend SaaS contract files contain zero commercial plan names or currency patterns."""
    saas_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "lib", "saas")
    ts_files = [
        os.path.join(saas_dir, f)
        for f in os.listdir(saas_dir)
        if f.endswith(".ts") or f.endswith(".tsx")
    ]
    assert len(ts_files) >= 3, f"Expected at least 3 frontend SaaS files, found {len(ts_files)}"

    forbidden_tokens = [
        "free", "starter", "pro", "growth", "fund", "scale", "enterprise",
        "tier", "stripe", "billing", "usd", "eur", r"\$39", r"\$49", r"\$199",
    ]

    for file_path in ts_files:
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read().lower()

        for token in forbidden_tokens:
            pattern = rf"\b{token}\b" if not token.startswith(r"\$") else token
            assert not re.search(pattern, src), (
                f"Forbidden commercial term '{token}' found in {os.path.basename(file_path)}"
            )


def test_frontend_contracts_pure_types_no_side_effects():
    """Assert frontend contracts import zero network, state, or DOM libraries."""
    saas_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "lib", "saas")
    for fname in os.listdir(saas_dir):
        if not fname.endswith(".ts"):
            continue
        file_path = os.path.join(saas_dir, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line_str = line.strip()
            if line_str.startswith("import ") and "from" in line_str:
                # Disallow network/react/dom imports
                assert "react" not in line_str.lower(), f"React imported in contract {fname}"
                assert "axios" not in line_str.lower(), f"Axios imported in contract {fname}"
                assert "fetch" not in line_str.lower(), f"Fetch imported in contract {fname}"
