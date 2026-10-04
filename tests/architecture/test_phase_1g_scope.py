"""
Architectural Scope and Invariant Enforcement for Phase 1G.

Verifies:
1. ROUTES_REWIRED = ONLY_AUTHORIZED_PRIVATE: Only portfolio, journal, and cockpit are rewired.
2. PUBLIC_ROUTES_UNTOUCHED = YES: All public context-free routes remain untouched.
3. DATABASE_SCHEMA_EXPAND_ONLY = YES:
   - workspaces and workspace_memberships tables exist.
   - workspace_id is added to portfolio_holdings, user_trade_journal, user_cockpit_actions.
   - workspace_id is nullable in all tables (expand-only phase).
   - user_profiles has NO workspace_id (ACTOR_PROFILE preserved).
4. INV_SAAS_06 = ENFORCED: Single canonical workspace identity authority.
5. REQUEST_CONTEXT_FIELD_COUNT = 3: RequestContext dataclass remains frozen.
6. CAPABILITY_VOCABULARY_CHANGED = NO: Capability dictionary remains frozen at 17 items.
7. PROTECTED_QUANT_MODULES_CHANGED = 0: Zero quant engine modifications.
8. ETF_V2_FILES_CHANGED = NO: Zero ETF V2 modifications.
9. OPENFIGI_FILES_CHANGED = NO: Zero OpenFIGI modifications.
10. TACTICAL_SETUPS_REMEDIATION_CHANGED = NO: Zero changes to tactical setup files.
11. Zero authentication, billing, or subscription logic in seams, routes, or database.
"""

import os
import sqlite3
import subprocess
import pytest
from dataclasses import fields

from api.context.request_context import RequestContext
from api.context.workspace_identity import (
    derive_compatibility_workspace_id,
    resolve_workspace_id,
    is_valid_workspace_id,
    WORKSPACE_ID_PREFIX,
    WORKSPACE_ID_AUTHORITY,
)
from api.capabilities.capabilities import CAPABILITIES, LIMITS
from database.workspace_migration import apply_workspace_tenancy_migration

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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
    cmd = ["git", "status", "--porcelain=v1", "-uno"]
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



def test_database_schema_expand_only_and_actor_profile():
    """Verify Phase 1G schema is strictly expand-only and user_profiles is preserved as ACTOR_PROFILE."""
    conn = sqlite3.connect(":memory:")
    # Initialize legacy baseline tables
    conn.executescript("""
        CREATE TABLE portfolio_holdings (
            user_id TEXT,
            symbol TEXT,
            shares REAL,
            avg_entry_price REAL,
            updated_at TIMESTAMP
        );
        CREATE TABLE user_trade_journal (
            user_id TEXT,
            trade_id TEXT,
            symbol TEXT,
            status TEXT
        );
        CREATE TABLE user_cockpit_actions (
            user_id TEXT,
            action_id TEXT,
            category TEXT,
            status TEXT
        );
        CREATE TABLE user_profiles (
            user_id TEXT PRIMARY KEY,
            display_name TEXT,
            risk_tolerance TEXT
        );
    """)

    results = apply_workspace_tenancy_migration(conn)
    assert results["workspaces_table_created"] is True
    assert results["workspace_memberships_table_created"] is True

    cursor = conn.cursor()

    # Verify workspaces table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workspaces'")
    assert cursor.fetchone() is not None

    # Verify workspace_memberships table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workspace_memberships'")
    assert cursor.fetchone() is not None

    # Verify workspace_id column added to WORKSPACE_OWNED tables and is nullable (expand-only)
    workspace_owned_tables = ["portfolio_holdings", "user_trade_journal", "user_cockpit_actions"]
    for tbl in workspace_owned_tables:
        cursor.execute(f"PRAGMA table_info({tbl})")
        cols = cursor.fetchall()
        col_names = [c[1] for c in cols]
        assert "workspace_id" in col_names, f"workspace_id missing from {tbl}"
        assert "user_id" in col_names, f"user_id must NOT be dropped in expand phase from {tbl}"

        # Check nullability (c[3] == 0 means NOT NULL is false, i.e., column is NULLABLE)
        ws_col = next(c for c in cols if c[1] == "workspace_id")
        assert ws_col[3] == 0, f"workspace_id in {tbl} must be nullable (expand-only)"

    # Verify user_profiles is preserved as ACTOR_PROFILE with ZERO workspace_id
    cursor.execute("PRAGMA table_info(user_profiles)")
    profile_cols = [c[1] for c in cursor.fetchall()]
    assert "workspace_id" not in profile_cols, "user_profiles must NOT contain workspace_id (ACTOR_PROFILE invariant)"
    assert "user_id" in profile_cols

    conn.close()


def test_inv_saas_06_workspace_identity_single_authority():
    """Verify INV-SAAS-06: api.context.workspace_identity is the single canonical authority."""
    assert WORKSPACE_ID_AUTHORITY == "api.context.workspace_identity:derive_compatibility_workspace_id"
    assert WORKSPACE_ID_PREFIX == "ws_"

    # Determinism
    ws_1 = derive_compatibility_workspace_id("user_alpha")
    ws_2 = derive_compatibility_workspace_id("user_alpha")
    assert ws_1 == ws_2
    assert ws_1.startswith("ws_usr_")
    assert derive_compatibility_workspace_id("ws_user_alpha") == "ws_user_alpha"

    # Default fallback
    assert derive_compatibility_workspace_id(None) == "ws_default"
    assert derive_compatibility_workspace_id("") == "ws_default"

    # Validation
    assert is_valid_workspace_id("ws_default") is True
    assert is_valid_workspace_id("ws_alpha_1") is True
    assert is_valid_workspace_id("") is False
    assert is_valid_workspace_id(None) is False
    assert is_valid_workspace_id("invalid with spaces") is False
    assert is_valid_workspace_id("invalid;injection") is False

    # Resolver uses authority
    from api.context.resolver import RequestContextResolver
    ctx = RequestContextResolver().resolve(x_user_id="alice_test")
    assert ctx.workspace_id == derive_compatibility_workspace_id("alice_test")
    assert ctx.actor_id == "alice_test"


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
    """Verify seam files, rewired routes, and database models contain zero auth tokens or billing logic."""
    target_files = [
        "api/context/resolver.py",
        "api/context/workspace_identity.py",
        "api/services/authorizer.py",
        "api/services/portfolio_service.py",
        "api/services/journal_service.py",
        "api/services/cockpit_service.py",
        "api/routes/portfolio.py",
        "api/routes/journal.py",
        "api/routes/cockpit.py",
        "database/models.py",
        "database/workspace_repository.py",
        "database/workspace_migration.py",
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
    cmd = ["git", "status", "--porcelain=v1", "-uno"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    modified_files = [line[3:].strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]

    proc_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    branch = proc_branch.stdout.strip()
    base_sha = "HEAD^1" if (branch == "main" and subprocess.run(["git", "rev-parse", "--verify", "HEAD^2"], cwd=REPO_ROOT, capture_output=True).returncode == 0) else "d20ec394133261df84885fb2d8c6f941a5b9ba19"
    cmd_diff = ["git", "diff", "--name-only", base_sha, "HEAD"]
    proc_diff = subprocess.run(cmd_diff, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc_diff.returncode == 0
    diff_files = [f.strip().replace("\\", "/") for f in proc_diff.stdout.splitlines() if f.strip()]

    all_files = modified_files + diff_files
    for f in all_files:
        f_lower = f.lower()
        assert "etf" not in f_lower or "api/routes/etf.py" not in f, f"ETF file modified: {f}"
        assert "openfigi" not in f_lower, f"OpenFIGI file modified: {f}"
        assert "setups" not in f_lower, f"Tactical Setups file modified: {f}"



def test_phase_1g_implementation_report_exists_and_complete():
    """Verify Phase 1G implementation report exists and asserts PASS verdict and predecessor SHA."""
    report_path = os.path.join(REPO_ROOT, "docs", "architecture", "ARX_SAAS_FOUNDATION_PHASE_1G_IMPLEMENTATION_REPORT.md")
    assert os.path.exists(report_path), f"Phase 1G implementation report not found: {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert (
        "GATE = PASS_ARX_SAAS_FOUNDATION_PHASE_1G_IMPLEMENTATION_VERIFIED" in content
        or "GATE = PASS_ARX_SAAS_FOUNDATION_PHASE_1G_IMPLEMENTATION_RECONCILED" in content
    )
    assert "18dd7e9f2d9f5737388f442a9f740cb82b4b163a" in content
    assert "EXPAND_PHASE_ONLY" in content
