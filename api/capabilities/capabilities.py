"""
Authoritative backend vocabulary for Capabilities and Limits in ARX Terminal.

This module defines:
1. Boolean capability identifiers (atomic product privileges).
2. Numeric limit identifiers (quantitative resource boundaries).

Constraints:
- Strictly lowercase dot-separated identifiers adhering to grammar:
    ^[a-z][a-z0-9]*(\\.[a-z][a-z0-9_]*)+$
- Zero commercial plan names or commercial tier designations.
- Zero currency or pricing symbols.
- Zero imports of billing, authentication, or payment packages.
- Zero startup side effects.
"""

import re
from typing import FrozenSet

CAPABILITY_GRAMMAR_REGEX = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)+$")

# Authoritative set of Boolean capability identifiers
CAPABILITIES: FrozenSet[str] = frozenset({
    "analysis.read",
    "analysis.quant",
    "analysis.simulation",
    "radar.read",
    "radar.advanced_filters",
    "radar.custom_scan",
    "portfolio.read",
    "portfolio.manage",
    "portfolio.risk",
    "journal.read",
    "journal.write",
    "alerts.create",
    "alerts.realtime",
    "export.csv",
    "team.read",
    "team.manage",
    "api.access",
})

# Authoritative set of numeric limit identifiers
LIMITS: FrozenSet[str] = frozenset({
    "portfolio.max_holdings",
    "portfolio.max_workspaces",
    "alerts.max_active",
    "team.max_members",
    "api.requests_per_day",
})

ALL_IDENTIFIERS: FrozenSet[str] = CAPABILITIES | LIMITS


def validate_identifier(ident: str) -> bool:
    """Validate that an identifier strictly matches canonical grammar."""
    if not isinstance(ident, str):
        return False
    return bool(CAPABILITY_GRAMMAR_REGEX.match(ident))


# Validation invariant check on load
assert len(CAPABILITIES & LIMITS) == 0, "CAPABILITIES and LIMITS must be strictly disjoint."
assert all(validate_identifier(c) for c in CAPABILITIES), "All capability identifiers must conform to grammar."
assert all(validate_identifier(l) for l in LIMITS), "All limit identifiers must conform to grammar."
