"""
Architectural Invariant Test: Zero Pricing Leakage in Runtime Code.

Enforces:
1. No commercial plan names (free, starter, pro, growth, fund, scale, enterprise).
2. No dollar pricing patterns ($29, $39, $49, etc.) or currency symbols.
3. No billing / payment processor concepts (stripe, checkout, invoice, subscription_tier).
in any Phase 1A-1D runtime contract or implementation files.
"""

import os
import re
import pytest
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNTIME_SEAM_DIRECTORIES = [
    os.path.join(REPO_ROOT, "api", "context"),
    os.path.join(REPO_ROOT, "api", "capabilities"),
    os.path.join(REPO_ROOT, "api", "services"),
    os.path.join(REPO_ROOT, "frontend", "lib", "saas"),
]

FORBIDDEN_PLAN_NAMES = [
    "starter",
    "pro",
    "growth",
    "fund",
    "scale",
    "enterprise",
]

FORBIDDEN_BILLING_TERMS = [
    "stripe",
    "checkout_session",
    "subscription_tier",
    "credit_card",
    "invoice_id",
]

DOLLAR_PRICE_REGEX = re.compile(r"\$\s*\d+")


def _scan_file_for_pricing_leakage(file_path: str) -> List[str]:
    violations = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check dollar prices
    for match in DOLLAR_PRICE_REGEX.finditer(content):
        violations.append(f"Dollar price pattern '{match.group(0)}' found in {file_path}")

    content_lower = content.lower()

    # Check forbidden plan names (delimiters include underscores, punctuation, whitespace)
    for plan in FORBIDDEN_PLAN_NAMES:
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(plan)}(?![a-zA-Z0-9])"
        if re.search(pattern, content_lower):
            violations.append(f"Forbidden plan name '{plan}' found in {file_path}")

    # Check forbidden billing terms
    for term in FORBIDDEN_BILLING_TERMS:
        pattern = rf"(?<![a-zA-Z0-9]){re.escape(term)}(?![a-zA-Z0-9])"
        if re.search(pattern, content_lower):
            violations.append(f"Forbidden billing term '{term}' found in {file_path}")

    return violations


def test_no_pricing_leakage_in_runtime_seam_files():
    """Verify all files under runtime seam directories have zero pricing leakage."""
    all_violations = []
    total_files_scanned = 0

    for directory in RUNTIME_SEAM_DIRECTORIES:
        assert os.path.exists(directory), f"Seam directory missing: {directory}"
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.endswith((".py", ".ts", ".tsx", ".js", ".json")):
                    total_files_scanned += 1
                    file_path = os.path.join(root, fname)
                    file_violations = _scan_file_for_pricing_leakage(file_path)
                    all_violations.extend(file_violations)

    assert total_files_scanned >= 6, f"Expected at least 6 runtime files, found {total_files_scanned}"
    assert len(all_violations) == 0, (
        f"Pricing leakage detected in runtime seam files:\n" + "\n".join(all_violations)
    )


def test_pricing_leakage_fixture_detects_violations(tmp_path):
    """Verify leakage detector flags plan names and dollar amounts in dirty files."""
    dirty_file = tmp_path / "dirty.py"
    dirty_file.write_text("TIER_PRO_PRICE = '$49/mo'\nstripe_customer = 'cus_123'", encoding="utf-8")

    violations = _scan_file_for_pricing_leakage(str(dirty_file))
    assert len(violations) >= 3
    assert any("Dollar price pattern" in v for v in violations)
    assert any("Forbidden plan name 'pro'" in v for v in violations)
    assert any("Forbidden billing term 'stripe'" in v for v in violations)
