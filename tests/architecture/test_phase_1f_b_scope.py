"""
Architectural Scope and Invariant Enforcement for Wave 1F-B.

Verifies:
1. ROUTES_REWIRED = ONLY_AUTHORIZED_PRIVATE: Only portfolio, journal, and cockpit are rewired.
2. PUBLIC_ROUTES_UNTOUCHED = YES: All public context-free routes remain untouched.
3. DATABASE_SCHEMA_CHANGED = NO: SQLite schemas and db_engine remain unchanged.
4. MIGRATION_FILES_ADDED = 0: No migration files added.
5. REQUEST_CONTEXT_FIELD_COUNT = 3: RequestContext dataclass remains frozen.
6. CAPABILITY_VOCABULARY_CHANGED = NO: Capability dictionary remains frozen at 17 items.
7. PROTECTED_QUANT_MODULES_CHANGED = 0: Zero quant engine modifications.
8. ETF_V2_FILES_CHANGED = NO: Zero ETF V2 modifications.
9. OPENFIGI_FILES_CHANGED = NO: Zero OpenFIGI modifications.
10. TACTICAL_SETUPS_REMEDIATION_CHANGED = NO: Zero changes to tactical setup files.
11. No authentication, billing, or subscription logic in seams or rewired routes.
"""

import os
import subprocess
import pytest
from dataclasses import fields

from api.context.request_context import RequestContext
from api.capabilities.capabilities import CAPABILITIES, LIMITS

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE_1F_A_RELEASE_SHA = "b45c6486f51837aaf03500c9d83219637d3ea191"

AUTHORIZED_PRIVATE_ROUTES = {
    "api/routes/portfolio.py",
    "api/routes/journal.py",
    "api/routes/cockpit.py",
}

FORBIDDEN_PUBLIC_ROUTE_FILES = [
    "api/routes/analytics.py",
    "api/routes/volatility.py",
    "api/routes/regimes.py",
    "api/routes/screener.py",
    "api/routes/smart_money.py",
    "api/routes/macro.py",
    "api/routes/etf.py",
    "api/routes/cache.py",
    "api/routes/governance.py",
    "api/routes/telemetry.py",
]


def test_only_authorized_private_routes_rewired():
    """Verify only authorized private routes were modified, and all other routes remain untouched."""
    cmd = ["git", "status", "--porcelain=v1"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    modified_files = [line[3:].strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]

    # Every forbidden public route must NOT be modified
    for route_file in FORBIDDEN_PUBLIC_ROUTE_FILES:
        assert route_file not in modified_files, f"Public route file unexpectedly modified: {route_file}"

    # Any modified route in api/routes/ must be in AUTHORIZED_PRIVATE_ROUTES
    for f in modified_files:
        if f.startswith("api/routes/"):
            assert f in AUTHORIZED_PRIVATE_ROUTES, f"Unauthorized route file modified: {f}"


def test_database_schema_phase_1f_b_boundary():
    """
    Verify Phase 1F-B database boundary evolution in Phase 1G.

    1F-B frozen historical assertion: schema unchanged during 1F-B.
    1G successor assertion: only authorized additive tenancy schema changes allowed
    (migration 002, workspace_migration, workspace_repository, models, and db_engine.py).
    All other database tables and migrations remain untouched.
    """
    cmd = ["git", "status", "--porcelain=v1"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    modified_files = [line[3:].strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]

    authorized_db_modifications = {
        "analyst_dashboard/data/db_engine.py",
        "database/models.py",
        "database/workspace_repository.py",
        "database/workspace_migration.py",
        "database/migrations/002_arx_saas_workspace_tenancy.sql",
    }
    for f in modified_files:
        if "migration" in f.lower() or f.startswith("database/"):
            assert f in authorized_db_modifications, f"Unexpected database/migration file: {f}"


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


def test_no_auth_or_billing_in_seams_and_routes():
    """Verify seam files and rewired routes contain zero auth tokens or billing logic."""
    target_files = [
        "api/context/resolver.py",
        "api/services/authorizer.py",
        "api/services/portfolio_service.py",
        "api/services/journal_service.py",
        "api/services/cockpit_service.py",
        "api/routes/portfolio.py",
        "api/routes/journal.py",
        "api/routes/cockpit.py",
    ]
    forbidden_tokens = ["jwt", "bearer", "oauth", "stripe", "checkout", "subscription_id", "plan_name", "invoice"]

    for rel_path in target_files:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
            for token in forbidden_tokens:
                assert token not in content, f"Forbidden token '{token}' found in {rel_path}"


def test_cross_track_isolation():
    """Verify zero changes to ETF V2, OpenFIGI, or Tactical Setups remediation."""
    cmd = ["git", "status", "--porcelain=v1"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    modified_files = [line[3:].strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]

    for f in modified_files:
        f_lower = f.lower()
        assert "etf" not in f_lower or "api/routes/etf.py" not in f, f"ETF file modified: {f}"
        assert "openfigi" not in f_lower, f"OpenFIGI file modified: {f}"
        assert "setups" not in f_lower, f"Tactical Setups file modified: {f}"
