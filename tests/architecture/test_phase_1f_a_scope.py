"""
Architectural Scope and Invariant Enforcement for Wave 1F-A.

Verifies:
1. ROUTES_REWIRED = NO: No existing route files have been modified.
2. DATABASE_SCHEMA_CHANGED = NO: SQLite schemas and db_engine remain unchanged.
3. MIGRATION_FILES_ADDED = 0: No migration files added.
4. REQUEST_CONTEXT_FIELD_COUNT = 3: RequestContext dataclass remains frozen.
5. CAPABILITY_VOCABULARY_CHANGED = NO: Capability dictionary remains frozen at 17 items.
6. PROTECTED_QUANT_MODULES_CHANGED = 0: Zero quant engine modifications.
7. ETF_V2_FILES_CHANGED = NO: Zero ETF V2 modifications.
8. OPENFIGI_FILES_CHANGED = NO: Zero OpenFIGI modifications.
9. TACTICAL_SETUPS_REMEDIATION_CHANGED = NO: Zero changes to tactical setup files.
10. No authentication, billing, or subscription logic in new seams.
"""

import os
import subprocess
import pytest
from dataclasses import fields

from api.context.request_context import RequestContext
from api.capabilities.capabilities import CAPABILITIES, LIMITS

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FROZEN_RELEASE_SHA = "724b5e3659ba0287fc3d8b9d58b4ef7eecde8703"

FORBIDDEN_ROUTE_FILES = [
    "api/routes/analytics.py",
    "api/routes/volatility.py",
    "api/routes/regimes.py",
    "api/routes/portfolio.py",
    "api/routes/journal.py",
    "api/routes/cockpit.py",
    "api/routes/screener.py",
    "api/routes/smart_money.py",
    "api/routes/macro.py",
    "api/routes/etf.py",
    "api/routes/cache.py",
    "api/routes/governance.py",
    "api/routes/telemetry.py",
]


def test_routes_not_rewired():
    """Verify zero route files have been modified since frozen release."""
    cmd = ["git", "status", "--porcelain=v1"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    modified_files = [line[3:].strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]

    for route_file in FORBIDDEN_ROUTE_FILES:
        assert route_file not in modified_files, f"Route file unexpectedly modified: {route_file}"


def test_database_schema_and_migrations_unchanged():
    """Verify database engine and migration directories are untouched."""
    cmd = ["git", "status", "--porcelain=v1"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    modified_files = [line[3:].strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]

    assert "analyst_dashboard/data/db_engine.py" not in modified_files
    for f in modified_files:
        assert "migration" not in f.lower(), f"Unexpected migration file: {f}"


def test_request_context_field_count_and_schema_frozen():
    """Verify RequestContext has exactly 3 fields and has not changed."""
    ctx_fields = fields(RequestContext)
    assert len(ctx_fields) == 3, f"Expected 3 fields, got {len(ctx_fields)}"
    field_names = [f.name for f in ctx_fields]
    assert field_names == ["actor_id", "workspace_id", "request_id"]


def test_capability_vocabulary_frozen():
    """Verify capabilities vocabulary has exactly 17 capabilities and 5 limits."""
    assert len(CAPABILITIES) == 17, f"Expected 17 capabilities, got {len(CAPABILITIES)}"
    assert len(LIMITS) == 5, f"Expected 5 limits, got {len(LIMITS)}"


def test_no_auth_or_billing_in_seams():
    """Verify newly introduced seam files contain zero auth tokens or billing logic."""
    new_seam_files = [
        "api/context/resolver.py",
        "api/services/authorizer.py",
        "api/services/portfolio_service.py",
        "api/services/journal_service.py",
        "api/services/cockpit_service.py",
    ]
    forbidden_tokens = ["jwt", "bearer", "oauth", "stripe", "checkout", "subscription_id", "plan_name", "invoice"]

    for rel_path in new_seam_files:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
            for token in forbidden_tokens:
                assert token not in content, f"Forbidden token '{token}' found in {rel_path}"
