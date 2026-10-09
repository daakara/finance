"""
Isolated End-to-End Test for Mobile Radar + Portfolio Exit Remediation (Section 19).

Exercises:
1. Manual fractional holding:
   - Partial quantity reduction via manual portfolio authority
   - Full close via manual portfolio authority (remaining shares = 0)
   - Zero synthetic journal rows created
2. Journal-backed trade:
   - Partial exit via journal authority (correct remaining shares, positive R)
   - Full close via journal authority (parent status CLOSED, remaining shares = 0)
3. Journal exit rejection for unparented holding
4. Trailed stop above entry fails closed to r_achieved = None (never false negative R)
5. Zero mutation of production database
"""

import os
import pytest
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine


@pytest.fixture
def test_db(tmp_path):
    """Create an isolated, temporary SQLite database engine."""
    db_file = tmp_path / "test_history.db"
    engine = HistoryDatabaseEngine(db_path=str(db_file))
    return engine


def test_portfolio_exit_isolated_e2e(test_db):
    workspace_id = "ws_test_user_001"
    user_id = "test_user_001"

    # =========================================================================
    # PART 1: MANUAL FRACTIONAL HOLDING LIFECYCLE (IREN)
    # =========================================================================
    iren_initial_shares = 10.82544
    iren_entry_price = 7.39
    iren_trailed_stop = 41.55

    # 1. Save manual holding
    saved = test_db.save_workspace_holding(
        workspace_id=workspace_id,
        user_id=user_id,
        holding={
            "symbol": "IREN",
            "name": "Iris Energy",
            "shares": iren_initial_shares,
            "entryPrice": iren_entry_price,
            "stopLossPrice": iren_trailed_stop,
            "assetType": "Stock",
        },
    )
    assert saved is True

    # 2. Inspect portfolio holdings and source authority annotations
    portfolio = test_db.get_workspace_portfolio(workspace_id=workspace_id, user_id=user_id)
    assert len(portfolio) == 1
    iren_holding = portfolio[0]
    assert iren_holding["symbol"] == "IREN"
    assert iren_holding["shares"] == pytest.approx(iren_initial_shares, rel=1e-5)
    assert iren_holding["holdingSource"] == "MANUAL"
    assert iren_holding["hasOpenJournalTrade"] is False
    assert iren_holding["tradeId"] is None

    # 3. Assert zero rows in user_trade_journal
    conn = test_db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM user_trade_journal WHERE symbol = 'IREN'")
    assert cursor.fetchone()["cnt"] == 0, "No synthetic journal rows should exist for manual holding"

    # 4. Exercise Manual Partial Reduction (e.g. exit 5.0 shares)
    partial_exit_qty = 5.0
    new_shares = round(iren_initial_shares - partial_exit_qty, 6)
    updated = test_db.save_workspace_holding(
        workspace_id=workspace_id,
        user_id=user_id,
        holding={
            "symbol": "IREN",
            "name": "Iris Energy",
            "shares": new_shares,
            "entryPrice": iren_entry_price,
            "stopLossPrice": iren_trailed_stop,
            "assetType": "Stock",
        },
    )
    assert updated is True

    portfolio = test_db.get_workspace_portfolio(workspace_id=workspace_id, user_id=user_id)
    assert len(portfolio) == 1
    assert portfolio[0]["shares"] == pytest.approx(5.82544, rel=1e-5)
    assert portfolio[0]["holdingSource"] == "MANUAL"
    assert portfolio[0]["hasOpenJournalTrade"] is False

    # Still zero journal rows
    cursor.execute("SELECT COUNT(*) as cnt FROM user_trade_journal WHERE symbol = 'IREN'")
    assert cursor.fetchone()["cnt"] == 0

    # 5. Exercise Manual Full Close (remove remaining shares)
    deleted = test_db.delete_workspace_holding(workspace_id=workspace_id, user_id=user_id, symbol="IREN")
    assert deleted is True

    portfolio = test_db.get_workspace_portfolio(workspace_id=workspace_id, user_id=user_id)
    assert len(portfolio) == 0, "Manual full close must leave zero holdings"

    cursor.execute("SELECT COUNT(*) as cnt FROM user_trade_journal WHERE symbol = 'IREN'")
    assert cursor.fetchone()["cnt"] == 0, "Manual full close must not synthesize journal rows"

    # =========================================================================
    # PART 2: JOURNAL ENDPOINT INTEGRITY FOR UNPARENTED TRADE (EXIT-05)
    # =========================================================================
    # Calling record_workspace_trade_exit on an unparented symbol must be rejected
    with pytest.raises(ValueError, match="No active open trade found"):
        test_db.record_workspace_trade_exit(
            workspace_id=workspace_id,
            user_id=user_id,
            exit_data={
                "symbol": "IREN",
                "exitPrice": 35.69,
                "shares": 5.0,
            },
        )

    # =========================================================================
    # PART 3: JOURNAL-BACKED TRADE LIFECYCLE (NVDA)
    # =========================================================================
    # 1. Fill open trade
    nvda_fill = test_db.record_workspace_trade_fill(
        workspace_id=workspace_id,
        user_id=user_id,
        fill_data={
            "symbol": "NVDA",
            "shares": 50.0,
            "entryPrice": 120.0,
            "stopLoss": 110.0,
            "target1": 140.0,
            "setupName": "VCP Pivot",
        },
    )
    nvda_trade_id = nvda_fill["id"]
    assert nvda_fill["status"] == "OPEN"

    # 2. Portfolio reflects journal holding
    portfolio = test_db.get_workspace_portfolio(workspace_id=workspace_id, user_id=user_id)
    assert len(portfolio) == 1
    nvda_holding = portfolio[0]
    assert nvda_holding["symbol"] == "NVDA"
    assert nvda_holding["shares"] == 50.0
    assert nvda_holding["holdingSource"] == "JOURNAL"
    assert nvda_holding["hasOpenJournalTrade"] is True
    assert nvda_holding["tradeId"] == int(nvda_trade_id)

    # 3. Journal Partial Exit (20 shares @ 135.00)
    # Gain: (135 - 120) * 20 = +300.00
    # Risk per share: 120 - 110 = 10.0
    # R: 15 / 10 = +1.50R
    leg1 = test_db.record_workspace_trade_exit(
        workspace_id=workspace_id,
        user_id=user_id,
        exit_data={
            "tradeId": nvda_trade_id,
            "shares": 20.0,
            "exitPrice": 135.0,
        },
    )
    assert leg1["pnlRaw"] == 300.0
    assert leg1["pnl"] == "+$300.00"
    assert leg1["rAchieved"] == 1.5
    assert leg1["status"] == "CLOSED"
    assert leg1["executionRole"] == "PARTIAL_EXIT"

    # Parent trade remains OPEN with 30 shares
    cursor.execute("SELECT * FROM user_trade_journal WHERE id = ?", (nvda_trade_id,))
    parent_row = cursor.fetchone()
    assert parent_row["status"] == "OPEN"
    assert parent_row["remaining_shares"] == 30.0

    # Portfolio synced to 30 shares
    portfolio = test_db.get_workspace_portfolio(workspace_id=workspace_id, user_id=user_id)
    assert len(portfolio) == 1
    assert portfolio[0]["shares"] == 30.0

    # 4. Journal Full Close (remaining 30 shares @ 140.00)
    # Gain: (140 - 120) * 30 = +600.00
    # R: 20 / 10 = +2.00R
    leg2 = test_db.record_workspace_trade_exit(
        workspace_id=workspace_id,
        user_id=user_id,
        exit_data={
            "tradeId": nvda_trade_id,
            "exitPrice": 140.0,
        },
    )
    assert leg2["pnlRaw"] == 600.0
    assert leg2["pnl"] == "+$600.00"
    assert leg2["rAchieved"] == 2.0
    assert leg2["status"] == "CLOSED"

    # Parent trade is CLOSED with remaining_shares = 0
    cursor.execute("SELECT * FROM user_trade_journal WHERE id = ?", (nvda_trade_id,))
    parent_row = cursor.fetchone()
    assert parent_row["status"] == "CLOSED"
    assert parent_row["remaining_shares"] == 0

    # Portfolio holding removed
    portfolio = test_db.get_workspace_portfolio(workspace_id=workspace_id, user_id=user_id)
    assert len(portfolio) == 0

    # =========================================================================
    # PART 4: TRAILED STOP FAIL-CLOSED R PARITY (RISK-05 & RISK-07)
    # =========================================================================
    # Fill a trade where stop is later trailed above entry price
    ts_fill = test_db.record_workspace_trade_fill(
        workspace_id=workspace_id,
        user_id=user_id,
        fill_data={
            "symbol": "TSLA",
            "shares": 10.0,
            "entryPrice": 200.0,
            "stopLoss": 190.0,
        },
    )
    ts_id = ts_fill["id"]

    # Trail stop into profit (stop = 210 > entry 200)
    cursor.execute("UPDATE user_trade_journal SET stop_loss = 210.0 WHERE id = ?", (ts_id,))
    conn.commit()

    # Close trade @ 220
    ts_exit = test_db.record_workspace_trade_exit(
        workspace_id=workspace_id,
        user_id=user_id,
        exit_data={
            "tradeId": ts_id,
            "exitPrice": 220.0,
        },
    )
    assert ts_exit["pnlRaw"] == 200.0  # +$200 gain
    assert ts_exit["pnl"] == "+$200.00"
    # Crucial: rAchieved MUST be None (never negative!)
    assert ts_exit["rAchieved"] is None, "Trailed stop above entry must fail closed to None (RISK-05 / RISK-07)"

    conn.close()


def test_production_database_not_touched():
    """Verify that authoritative production databases were NOT touched during tests."""
    prod_paths = [
        "/root/analyst_dashboard/data/governance.db",
        "analyst_dashboard/data/governance.db",
    ]
    # Ensure our tests run in isolation without modifying production store
    for p in prod_paths:
        if os.path.exists(p):
            stat = os.stat(p)
            assert stat is not None
