"""
ARX SaaS Foundation Phase 1G: Workspace Persistence and Tenancy Test Suite.

Verifies:
1. Schema creation and expand-only properties (nullable workspace_id, indexes).
2. Idempotency of migrations and backfill.
3. Zero data loss and legacy data preservation.
4. Deterministic backfill with zero guessing (synthetic_or_guessed_workspace_assignments = 0).
5. Cross-workspace isolation (INV-SAAS-03: zero data leakage across workspaces).
6. Single authority enforcement (INV-SAAS-06: derive_compatibility_workspace_id).
7. Actor profile independence (user_profiles has no workspace_id).
8. Dual-write behavior on new persistence operations.
"""

import os
import sqlite3
import tempfile
import pytest

from api.context.workspace_identity import (
    derive_compatibility_workspace_id,
    is_valid_workspace_id,
    WORKSPACE_ID_PREFIX,
)
from api.context.request_context import RequestContext
from api.services.portfolio_service import PortfolioApplicationService
from api.services.journal_service import JournalApplicationService
from api.services.cockpit_service import CockpitApplicationService
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine
from database.workspace_migration import (
    apply_workspace_tenancy_migration,
    backfill_workspace_tenancy,
    WORKSPACE_OWNED_TABLES,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


@pytest.fixture
def engine(temp_db_path):
    """Instantiate HistoryDatabaseEngine against temporary database."""
    return HistoryDatabaseEngine(db_path=temp_db_path)


# ---------------------------------------------------------------------------
# 1. Authority Tests (INV-SAAS-06)
# ---------------------------------------------------------------------------

def test_workspace_identity_authority():
    """Verify single canonical workspace identity authority (INV-SAAS-06)."""
    # Deterministic generation
    ws_1 = derive_compatibility_workspace_id("user_alpha")
    ws_2 = derive_compatibility_workspace_id("user_alpha")
    assert ws_1 == ws_2
    assert ws_1.startswith(WORKSPACE_ID_PREFIX)
    assert is_valid_workspace_id(ws_1)

    # Different users yield different workspaces
    ws_beta = derive_compatibility_workspace_id("user_beta")
    assert ws_1 != ws_beta

    # Fallback to default for empty/whitespace/None
    assert derive_compatibility_workspace_id("") == "ws_default"
    assert derive_compatibility_workspace_id("   ") == "ws_default"
    assert derive_compatibility_workspace_id(None) == "ws_default"


# ---------------------------------------------------------------------------
# 2. Schema Creation & Expand-Only Properties
# ---------------------------------------------------------------------------

def test_schema_expand_only_properties(temp_db_path):
    """Verify schema migration adds tables, indexes, and nullable workspace_id without altering user_profiles."""
    engine = HistoryDatabaseEngine(db_path=temp_db_path)
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    # 1. Check workspaces table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workspaces';")
    assert cursor.fetchone() is not None

    # 2. Check workspace_memberships table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workspace_memberships';")
    assert cursor.fetchone() is not None

    # 3. Check WORKSPACE_OWNED tables have nullable workspace_id
    for table in WORKSPACE_OWNED_TABLES:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = {row[1]: {"notnull": row[3], "pk": row[5]} for row in cursor.fetchall()}
        assert "workspace_id" in cols, f"workspace_id missing from {table}"
        assert cols["workspace_id"]["notnull"] == 0, f"workspace_id must be nullable in expand phase on {table}"
        assert "user_id" in cols, f"user_id must be preserved on {table}"

    # 4. Check user_profiles has NO workspace_id (ACTOR_PROFILE)
    cursor.execute("PRAGMA table_info(user_profiles);")
    profile_cols = [row[1] for row in cursor.fetchall()]
    assert "workspace_id" not in profile_cols, "user_profiles must NOT have workspace_id"
    assert "user_id" in profile_cols

    # 5. Check indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    index_names = {row[0] for row in cursor.fetchall()}
    assert "idx_portfolio_holdings_ws" in index_names
    assert "idx_portfolio_holdings_ws_sym" in index_names
    assert "idx_user_trade_journal_ws" in index_names
    assert "idx_user_cockpit_actions_ws" in index_names

    conn.close()


def test_migration_and_backfill_idempotency(temp_db_path):
    """Verify migrations and backfills can be executed repeatedly with zero errors or duplicates."""
    conn = sqlite3.connect(temp_db_path)
    # Run migration and backfill multiple times
    res1 = apply_workspace_tenancy_migration(conn)
    bf1 = backfill_workspace_tenancy(conn)
    res2 = apply_workspace_tenancy_migration(conn)
    bf2 = backfill_workspace_tenancy(conn)

    assert isinstance(res1, dict)
    assert isinstance(res2, dict)
    # In second run, no new tables should be created
    assert res2["workspaces_table_created"] is False
    assert res2["workspace_memberships_table_created"] is False
    conn.close()


# ---------------------------------------------------------------------------
# 3. Deterministic Backfill & Zero Guessing
# ---------------------------------------------------------------------------

def test_deterministic_backfill_and_zero_guessing(temp_db_path):
    """Verify legacy rows are backfilled deterministically with zero guessing and zero data loss."""
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    # Create base legacy tables without workspace_id
    cursor.execute("""
        CREATE TABLE portfolio_holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            symbol TEXT,
            name TEXT,
            shares REAL,
            entry_price REAL,
            current_price REAL,
            target_price REAL,
            stop_loss_price REAL,
            added_at TEXT,
            asset_type TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Insert legacy rows: 2 for user_A, 1 for user_B, 1 with NULL user_id (unattributed)
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES ('user_a', 'AAPL', 'Apple', 10, 150.0, '2026-01-01', 'Stock');")
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES ('user_a', 'MSFT', 'Microsoft', 5, 300.0, '2026-01-01', 'Stock');")
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES ('user_b', 'NVDA', 'Nvidia', 20, 120.0, '2026-01-01', 'Stock');")
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES (NULL, 'ORPHAN', 'Orphaned', 1, 10.0, '2026-01-01', 'Stock');")
    conn.commit()

    # Apply migration and backfill
    apply_workspace_tenancy_migration(conn)
    ledger = backfill_workspace_tenancy(conn)

    # Verify ledger statistics
    ph_stats = ledger["tables"]["portfolio_holdings"]
    assert ph_stats["row_count_before"] == 4
    assert ph_stats["row_count_after"] == 4  # Zero data loss
    assert ph_stats["rows_eligible"] == 4
    assert ph_stats["rows_backfilled"] == 3
    assert ph_stats["rows_unresolved"] == 1  # NULL user_id remains NULL (zero guessing)
    assert ph_stats["rows_conflicting"] == 0

    # Verify backfilled values
    cursor.execute("SELECT symbol, user_id, workspace_id FROM portfolio_holdings;")
    rows = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

    expected_ws_a = derive_compatibility_workspace_id("user_a")
    expected_ws_b = derive_compatibility_workspace_id("user_b")

    assert rows["AAPL"] == ("user_a", expected_ws_a)
    assert rows["MSFT"] == ("user_a", expected_ws_a)
    assert rows["NVDA"] == ("user_b", expected_ws_b)
    assert rows["ORPHAN"] == (None, None)  # Zero guessed assignments

    # Verify workspaces and memberships were created
    cursor.execute("SELECT workspace_id, name FROM workspaces;")
    ws_rows = {r[0]: r[1] for r in cursor.fetchall()}
    assert expected_ws_a in ws_rows
    assert expected_ws_b in ws_rows

    cursor.execute("SELECT workspace_id, user_id, role FROM workspace_memberships;")
    memberships = {(r[0], r[1]): r[2] for r in cursor.fetchall()}
    assert (expected_ws_a, "user_a") in memberships
    assert memberships[(expected_ws_a, "user_a")] == "owner"
    assert (expected_ws_b, "user_b") in memberships

    conn.close()


# ---------------------------------------------------------------------------
# 4. Cross-Workspace Isolation (INV-SAAS-03)
# ---------------------------------------------------------------------------

def test_cross_workspace_isolation_portfolio(engine):
    """Verify holding mutations and queries are strictly isolated between workspaces (INV-SAAS-03)."""
    ws_1 = derive_compatibility_workspace_id("user_1")
    ws_2 = derive_compatibility_workspace_id("user_2")

    holding_1 = {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "shares": 10.0,
        "entryPrice": 150.0,
    }
    holding_2 = {
        "symbol": "MSFT",
        "name": "Microsoft Corp.",
        "shares": 20.0,
        "entryPrice": 300.0,
    }

    # Save to workspace 1 and 2
    assert engine.save_workspace_holding(workspace_id=ws_1, user_id="user_1", holding=holding_1)
    assert engine.save_workspace_holding(workspace_id=ws_2, user_id="user_2", holding=holding_2)

    # Read workspace 1
    port_1 = engine.get_workspace_portfolio(workspace_id=ws_1, user_id="user_1")
    symbols_1 = [h["symbol"] for h in port_1]
    assert symbols_1 == ["AAPL"]
    assert "MSFT" not in symbols_1  # No leak from workspace 2

    # Read workspace 2
    port_2 = engine.get_workspace_portfolio(workspace_id=ws_2, user_id="user_2")
    symbols_2 = [h["symbol"] for h in port_2]
    assert symbols_2 == ["MSFT"]
    assert "AAPL" not in symbols_2  # No leak from workspace 1

    # Delete in workspace 1 does not affect workspace 2
    engine.delete_workspace_holding(workspace_id=ws_1, user_id="user_1", symbol="AAPL")
    assert len(engine.get_workspace_portfolio(workspace_id=ws_1, user_id="user_1")) == 0
    assert len(engine.get_workspace_portfolio(workspace_id=ws_2, user_id="user_2")) == 1


def test_cross_workspace_isolation_journal_and_telemetry(engine):
    """Verify journal trades and risk telemetry are strictly isolated between workspaces (INV-SAAS-03)."""
    ws_1 = derive_compatibility_workspace_id("trader_alpha")
    ws_2 = derive_compatibility_workspace_id("trader_beta")

    # Trader Alpha fills a trade
    fill_alpha = {
        "symbol": "NVDA",
        "entryPrice": 100.0,
        "shares": 50.0,
        "setupName": "VCP Breakout",
        "confidence": 85.0,
        "notes": "Alpha trade",
    }
    trade_alpha = engine.record_workspace_trade_fill(workspace_id=ws_1, user_id="trader_alpha", fill_data=fill_alpha)
    assert trade_alpha["workspaceId"] == ws_1

    # Trader Beta fills a trade
    fill_beta = {
        "symbol": "TSLA",
        "entryPrice": 200.0,
        "shares": 25.0,
        "setupName": "Pocket Pivot",
        "confidence": 70.0,
        "notes": "Beta trade",
    }
    trade_beta = engine.record_workspace_trade_fill(workspace_id=ws_2, user_id="trader_beta", fill_data=fill_beta)
    assert trade_beta["workspaceId"] == ws_2

    # Verify journal log isolation
    trades_alpha = engine.get_workspace_journal_trades(workspace_id=ws_1, user_id="trader_alpha")
    trades_beta = engine.get_workspace_journal_trades(workspace_id=ws_2, user_id="trader_beta")

    assert len(trades_alpha) == 1
    assert trades_alpha[0]["symbol"] == "NVDA"
    assert len(trades_beta) == 1
    assert trades_beta[0]["symbol"] == "TSLA"

    # Close Alpha trade at a loss
    exit_alpha = {
        "tradeId": trade_alpha["id"],
        "exitPrice": 90.0,
        "followedRules": True,
    }
    engine.record_workspace_trade_exit(workspace_id=ws_1, user_id="trader_alpha", exit_data=exit_alpha)

    # Check telemetry
    telem_alpha = engine.get_workspace_risk_telemetry(workspace_id=ws_1, user_id="trader_alpha")
    telem_beta = engine.get_workspace_risk_telemetry(workspace_id=ws_2, user_id="trader_beta")

    # Alpha had a loss trade
    assert telem_alpha["consecutiveLossStreak"] == 1
    # Beta has open trade with 0 closed trades, so consecutive loss streak is 0
    assert telem_beta["consecutiveLossStreak"] == 0


def test_cross_workspace_isolation_cockpit_actions(engine):
    """Verify cockpit action items are isolated to workspaces while profiles remain actor-scoped."""
    ws_1 = derive_compatibility_workspace_id("actor_x")
    ws_2 = derive_compatibility_workspace_id("actor_y")

    action_1 = {
        "id": "act-1",
        "title": "Review Alpha Risk",
        "domain": "PORTFOLIO",
        "priorityScore": 90.0,
    }
    action_2 = {
        "id": "act-2",
        "title": "Review Beta Cash",
        "domain": "MACRO",
        "priorityScore": 75.0,
    }

    engine.save_workspace_action(workspace_id=ws_1, user_id="actor_x", action=action_1)
    engine.save_workspace_action(workspace_id=ws_2, user_id="actor_y", action=action_2)

    acts_1 = engine.get_workspace_actions(workspace_id=ws_1, user_id="actor_x")
    acts_2 = engine.get_workspace_actions(workspace_id=ws_2, user_id="actor_y")

    assert len(acts_1) == 1
    assert acts_1[0]["title"] == "Review Alpha Risk"
    assert len(acts_2) == 1
    assert acts_2[0]["title"] == "Review Beta Cash"


# ---------------------------------------------------------------------------
# 5. Application Services Integration & Fallback
# ---------------------------------------------------------------------------

def test_application_services_workspace_wiring(engine):
    """Verify PortfolioApplicationService, JournalApplicationService, and CockpitApplicationService coordinate via workspace_id."""
    ws_id = derive_compatibility_workspace_id("ctx_user")
    ctx = RequestContext(actor_id="ctx_user", workspace_id=ws_id, request_id="req-123")

    port_svc = PortfolioApplicationService(db_engine=engine)
    jour_svc = JournalApplicationService(db_engine=engine)
    cock_svc = CockpitApplicationService(db_engine=engine)

    # 1. Portfolio Service
    port_svc.save_holding(ctx, {"symbol": "GOOGL", "shares": 15.0, "entryPrice": 175.0})
    portfolio = port_svc.get_portfolio(ctx)
    assert len(portfolio) == 1
    assert portfolio[0]["symbol"] == "GOOGL"
    assert portfolio[0]["workspaceId"] == ws_id

    # 2. Journal Service
    trade = jour_svc.record_fill(ctx, {"symbol": "GOOGL", "shares": 10.0, "entryPrice": 175.0})
    assert trade["workspaceId"] == ws_id
    trades = jour_svc.get_trades(ctx)
    assert len(trades) == 1

    # 3. Cockpit Service: profile is actor-scoped, actions & holdings are workspace-scoped
    cock_svc.update_profile(ctx, {"name": "Test User", "role": "Macro Specialist"})
    cock_svc.create_action(ctx, {"id": "act-c1", "title": "Check Macro Exposure", "priorityScore": 80.0})

    state = cock_svc.get_cockpit_state(ctx)
    assert state["actor_id"] == "ctx_user"
    assert state["workspace_id"] == ws_id
    assert state["profile"]["name"] == "Test User"
    assert len(state["holdings"]) >= 1
    assert len(state["actions"]) == 1
    assert state["actions"][0]["workspaceId"] == ws_id


# ---------------------------------------------------------------------------
# 6. Default Workspace & Anonymous Persistence Isolation (INV-SAAS-07)
# ---------------------------------------------------------------------------

def test_inv_saas_07_ws_default_cannot_own_private_persisted_data(engine):
    """
    Verify INV-SAAS-07: Shared default workspace must NOT own private persisted user data.
    WS_DEFAULT_POLICY = NON_PERSISTENT_ONLY.
    PERSISTENT_WORKSPACE_WRITE_REQUIRES_ACTOR_BOUND_WORKSPACE = YES.
    """
    port_svc = PortfolioApplicationService(db_engine=engine)
    jour_svc = JournalApplicationService(db_engine=engine)
    cock_svc = CockpitApplicationService(db_engine=engine)

    ctx_anon = RequestContext(actor_id=None, workspace_id="ws_default", request_id="req-anon")

    # 1. Anonymous writes are rejected fail-closed
    with pytest.raises(PermissionError) as exc_port:
        port_svc.save_holding(ctx_anon, {"symbol": "AAPL", "shares": 10.0, "entryPrice": 150.0})
    assert "INV-SAAS-07" in str(exc_port.value)

    with pytest.raises(PermissionError) as exc_jour:
        jour_svc.record_trade(ctx_anon, {"symbol": "AAPL", "shares": 10.0, "entryPrice": 150.0})
    assert "INV-SAAS-07" in str(exc_jour.value)

    with pytest.raises(PermissionError) as exc_fill:
        jour_svc.record_fill(ctx_anon, {"symbol": "AAPL", "shares": 10.0, "entryPrice": 150.0})
    assert "INV-SAAS-07" in str(exc_fill.value)

    with pytest.raises(PermissionError) as exc_exit:
        jour_svc.record_exit(ctx_anon, {"symbol": "AAPL", "exitPrice": 160.0})
    assert "INV-SAAS-07" in str(exc_exit.value)

    with pytest.raises(PermissionError) as exc_cock_prof:
        cock_svc.update_profile(ctx_anon, {"name": "Anon User"})
    assert "INV-SAAS-07" in str(exc_cock_prof.value)

    with pytest.raises(PermissionError) as exc_cock_act:
        cock_svc.create_action(ctx_anon, {"id": "act-anon", "title": "Anon Action", "priorityScore": 50.0})
    assert "INV-SAAS-07" in str(exc_cock_act.value)

    # 2. Database engine direct methods also fail closed when ws_default is targeted
    with pytest.raises(PermissionError):
        engine.save_workspace_holding("ws_default", "anon_user", {"symbol": "AAPL", "shares": 1.0, "entryPrice": 100.0})
    with pytest.raises(PermissionError):
        engine.delete_workspace_holding("ws_default", "anon_user", "AAPL")
    with pytest.raises(PermissionError):
        engine.bulk_save_workspace_holdings("ws_default", "anon_user", [{"symbol": "AAPL", "shares": 1.0, "entryPrice": 100.0}])
    with pytest.raises(PermissionError):
        engine.save_workspace_journal_trade("ws_default", "anon_user", {"symbol": "AAPL", "shares": 1.0, "entryPrice": 100.0})
    with pytest.raises(PermissionError):
        engine.record_workspace_trade_fill("ws_default", "anon_user", {"symbol": "AAPL", "shares": 1.0, "entryPrice": 100.0})
    with pytest.raises(PermissionError):
        engine.record_workspace_trade_exit("ws_default", "anon_user", {"symbol": "AAPL", "exitPrice": 110.0})
    with pytest.raises(PermissionError):
        engine.save_workspace_action("ws_default", "anon_user", {"id": "act-1", "title": "Test", "priorityScore": 50.0})


def test_ws_default_membership_auto_provisioning_prohibited(temp_db_path):
    """Verify WS_DEFAULT_MEMBERSHIP_AUTO_PROVISIONING = PROHIBITED."""
    from database.workspace_repository import WorkspaceRepository
    repo = WorkspaceRepository(db_path=temp_db_path)

    # Cannot create workspace entity for ws_default
    with pytest.raises(ValueError) as exc_ws:
        repo.create_workspace("ws_default", "Default Workspace")
    assert "INV-SAAS-07" in str(exc_ws.value)

    # Cannot add membership for ws_default
    with pytest.raises(ValueError) as exc_mem:
        repo.add_membership("ws_default", "user_1")
    assert "INV-SAAS-07" in str(exc_mem.value)

    # Cannot add membership for empty or default user
    with pytest.raises(ValueError):
        repo.add_membership("ws_usr_test", "")
    with pytest.raises(ValueError):
        repo.add_membership("ws_usr_test", "default_user")


def test_adversarial_cross_anonymous_and_actor_isolation(engine):
    """
    Adversarial cross-anonymous isolation verification (Section 9):
    - ANONYMOUS_A_CAN_READ_ANONYMOUS_B_PRIVATE_DATA = NO
    - ANONYMOUS_A_CAN_MUTATE_ANONYMOUS_B_PRIVATE_DATA = NO
    - ANONYMOUS_CAN_ACCESS_ACTOR_A_PRIVATE_DATA = NO
    - ACTOR_A_CAN_ACCESS_ACTOR_B_PRIVATE_DATA = NO
    - ANONYMOUS_PRIVATE_PERSISTENCE_ACCESS = REJECTED
    """
    port_svc = PortfolioApplicationService(db_engine=engine)
    jour_svc = JournalApplicationService(db_engine=engine)
    cock_svc = CockpitApplicationService(db_engine=engine)

    ws_a = derive_compatibility_workspace_id("actor_a")
    ws_b = derive_compatibility_workspace_id("actor_b")

    ctx_actor_a = RequestContext(actor_id="actor_a", workspace_id=ws_a, request_id="req-a")
    ctx_actor_b = RequestContext(actor_id="actor_b", workspace_id=ws_b, request_id="req-b")
    ctx_anon_1 = RequestContext(actor_id=None, workspace_id="ws_default", request_id="req-anon-1")
    ctx_anon_2 = RequestContext(actor_id=None, workspace_id="ws_default", request_id="req-anon-2")

    # Actor A persists data
    port_svc.save_holding(ctx_actor_a, {"symbol": "AAPL", "shares": 100.0, "entryPrice": 150.0})
    jour_svc.record_fill(ctx_actor_a, {"symbol": "AAPL", "shares": 50.0, "entryPrice": 150.0})
    cock_svc.update_profile(ctx_actor_a, {"name": "Actor A Real Name"})
    cock_svc.create_action(ctx_actor_a, {"id": "act-a", "title": "Actor A Private Action", "priorityScore": 95.0})

    # Actor B persists data
    port_svc.save_holding(ctx_actor_b, {"symbol": "MSFT", "shares": 200.0, "entryPrice": 300.0})
    jour_svc.record_fill(ctx_actor_b, {"symbol": "MSFT", "shares": 100.0, "entryPrice": 300.0})

    # 1. ANONYMOUS_CAN_ACCESS_ACTOR_A_PRIVATE_DATA = NO
    anon_port = port_svc.get_portfolio(ctx_anon_1)
    assert anon_port == []
    anon_trades = jour_svc.get_trades(ctx_anon_1)
    assert anon_trades == []
    anon_telem = jour_svc.get_telemetry(ctx_anon_1)
    assert anon_telem["available"] is False
    assert anon_telem["userId"] is None
    anon_cockpit = cock_svc.get_cockpit_state(ctx_anon_1)
    assert anon_cockpit["profile"] == {}
    assert anon_cockpit["holdings"] == []
    assert anon_cockpit["actions"] == []

    # 2. ANONYMOUS_A_CAN_READ_ANONYMOUS_B_PRIVATE_DATA = NO
    # (Both receive clean empty views, zero leakage)
    assert port_svc.get_portfolio(ctx_anon_2) == []
    assert jour_svc.get_trades(ctx_anon_2) == []

    # 3. ANONYMOUS_A_CAN_MUTATE_ANONYMOUS_B_PRIVATE_DATA = NO
    # ANONYMOUS_PRIVATE_PERSISTENCE_ACCESS = REJECTED
    with pytest.raises(PermissionError):
        port_svc.save_holding(ctx_anon_1, {"symbol": "LEAK", "shares": 1.0, "entryPrice": 1.0})
    with pytest.raises(PermissionError):
        jour_svc.record_fill(ctx_anon_2, {"symbol": "LEAK", "shares": 1.0, "entryPrice": 1.0})

    # 4. ACTOR_A_CAN_ACCESS_ACTOR_B_PRIVATE_DATA = NO
    port_a = port_svc.get_portfolio(ctx_actor_a)
    port_b = port_svc.get_portfolio(ctx_actor_b)
    symbols_a = [h["symbol"] for h in port_a]
    symbols_b = [h["symbol"] for h in port_b]
    assert symbols_a == ["AAPL"]
    assert "MSFT" not in symbols_a
    assert symbols_b == ["MSFT"]
    assert "AAPL" not in symbols_b

    # 5. Actor A cannot spoof Actor B's workspace
    ctx_spoof = RequestContext(actor_id="actor_a", workspace_id=ws_b, request_id="req-spoof")
    with pytest.raises(PermissionError):
        port_svc.get_portfolio(ctx_spoof)
    with pytest.raises(PermissionError):
        jour_svc.get_trades(ctx_spoof)


def test_null_empty_legacy_owner_not_backfilled_to_ws_default(temp_db_path):
    """
    Verify backfill safety for missing user IDs (Section 10):
    NULL/empty legacy user_id != ws_default assignment.
    ROWS_WITH_NULL_OR_EMPTY_USER_ID = WORKSPACE_ID_REMAINS_NULL.
    SYNTHETIC_OR_GUESSED_WORKSPACE_ASSIGNMENTS = 0.
    """
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE portfolio_holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            symbol TEXT,
            name TEXT,
            shares REAL,
            entry_price REAL,
            current_price REAL,
            target_price REAL,
            stop_loss_price REAL,
            added_at TEXT,
            asset_type TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Insert rows with NULL, empty string, and whitespace user_id
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES (NULL, 'NULL_SYM', 'Null Owner', 10, 100.0, '2026-01-01', 'Stock');")
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES ('', 'EMPTY_SYM', 'Empty Owner', 5, 50.0, '2026-01-01', 'Stock');")
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES ('   ', 'SPACE_SYM', 'Space Owner', 2, 20.0, '2026-01-01', 'Stock');")
    cursor.execute("INSERT INTO portfolio_holdings (user_id, symbol, name, shares, entry_price, added_at, asset_type) VALUES ('valid_user', 'VALID_SYM', 'Valid Owner', 1, 10.0, '2026-01-01', 'Stock');")
    conn.commit()

    apply_workspace_tenancy_migration(conn)
    ledger = backfill_workspace_tenancy(conn)

    # Check that rows with NULL/empty/space user_id were NOT assigned to ws_default
    cursor.execute("SELECT symbol, user_id, workspace_id FROM portfolio_holdings WHERE symbol IN ('NULL_SYM', 'EMPTY_SYM', 'SPACE_SYM');")
    rows = cursor.fetchall()
    for sym, u_id, ws_id in rows:
        assert ws_id is None, f"Row {sym} had workspace_id={ws_id}; expected NULL, NEVER ws_default!"
        assert ws_id != "ws_default", f"Row {sym} was erroneously assigned to ws_default!"

    # Valid user row IS backfilled
    cursor.execute("SELECT workspace_id FROM portfolio_holdings WHERE symbol = 'VALID_SYM';")
    valid_ws = cursor.fetchone()[0]
    expected_ws = derive_compatibility_workspace_id("valid_user")
    assert valid_ws == expected_ws
    assert valid_ws != "ws_default"

    # ws_default must NEVER exist in workspaces or workspace_memberships tables
    cursor.execute("SELECT COUNT(*) FROM workspaces WHERE workspace_id = 'ws_default';")
    assert cursor.fetchone()[0] == 0
    cursor.execute("SELECT COUNT(*) FROM workspace_memberships WHERE workspace_id = 'ws_default';")
    assert cursor.fetchone()[0] == 0

    conn.close()
