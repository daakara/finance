"""ARCH-003: counts do not prove outcome calibration or statistical confidence."""
import sqlite3

import pytest

from analyst_dashboard.data.db_engine import HistoryDatabaseEngine

pytestmark = pytest.mark.tier2b


@pytest.fixture
def history(tmp_path):
    # Use the real SQLite schema and query path, not a mocked status formatter.
    return HistoryDatabaseEngine(str(tmp_path / "history.db"))


def log_rows(db, count, ticker):
    with db._get_connection() as conn:
        conn.executemany(
            """INSERT INTO trade_recommendation_history
            (ticker, setup_pattern, stage_phase, current_price,
             optimal_entry_min, optimal_entry_max, stop_loss,
             take_profit_1, take_profit_2, risk_reward_ratio, outcome_status)
            VALUES (?, 'VCP', 'STAGE_2', 100, 98, 101, 92, 110, 120, 2.5, 'PENDING')""",
            [(ticker,)] * count,
        )


@pytest.mark.parametrize("n", [0, 1, 29, 30, 31, 100])
def test_recommendation_volume_never_implies_outcome_confidence(history, n):
    log_rows(history, n, "NVDA")
    result = history.get_setup_accuracy_summary("NVDA")
    assert result["total_logged_setups"] == n
    assert result["available"] == (n > 0)
    assert result["target_hit_rate_pct"] is None
    assert result["statistical_confidence"] == "Not estimated (no outcome-based confidence interval)"
    assert "not established" in result["model_calibration_status"]
    assert result["avg_risk_reward"] == (2.5 if n else None)


def test_ticker_filter_and_overall_counts_remain_unchanged(history):
    log_rows(history, 35, "NVDA")
    log_rows(history, 4, "LNTH")
    all_assets = history.get_setup_accuracy_summary()
    nvda = history.get_setup_accuracy_summary("nvda")
    lnth = history.get_setup_accuracy_summary("LNTH")
    assert [all_assets["total_logged_setups"], nvda["total_logged_setups"], lnth["total_logged_setups"]] == [39, 35, 4]
    assert all_assets["target_hit_rate_pct"] is None
    assert all_assets["statistical_confidence"].startswith("Not estimated")
    assert nvda["avg_risk_reward"] == 2.5
    assert lnth["avg_risk_reward"] == 2.5


def test_pending_outcomes_not_silently_promoted_to_success(history):
    log_rows(history, 40, "NVDA")
    with history._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM trade_recommendation_history WHERE outcome_status='PENDING'").fetchone()[0] == 40
    result = history.get_setup_accuracy_summary("NVDA")
    assert result["target_hit_rate_pct"] is None
    assert "calibration not established" in result["model_calibration_status"]
