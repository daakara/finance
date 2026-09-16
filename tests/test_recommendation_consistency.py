import pytest
from unittest.mock import patch, MagicMock
from api.routes.analytics import _build_tactical_setup
from analyst_dashboard.analyzers.optimal_execution import (
    OptimalExecutionEngine,
    ACTIONABLE_EXECUTION_STATUSES,
    ALL_EXECUTION_STATUSES,
)
import pandas as pd

DATES = pd.date_range("2026-01-01", periods=60, freq="B").strftime("%Y-%m-%d").tolist()
CANDLES = [
    {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000000, "date": d}
    for d in DATES
]

TAXONOMY_ACTIONABILITY_TABLE = [
    ("IN_BUY_ZONE", True),
    ("READY_TO_BUY", True),
    ("WAITING_PULLBACK", False),
    ("IN_BUY_ZONE_AWAITING_TRIGGER", False),
    ("APPROACHING_TARGET", False),
    ("STOPPED_OUT", False),
    ("INSUFFICIENT_HISTORY", False),
    ("UNVERIFIED_ASSET", False),
    ("STALE_MARKET_DATA", False),
    ("UNKNOWN", False),
    ("UNAUTHORIZED_FUTURE_STATUS", False),
]

@pytest.mark.parametrize("status,expected_actionable", TAXONOMY_ACTIONABILITY_TABLE)
def test_global_status_taxonomy_actionability(status, expected_actionable):
    """Verify that every status in the taxonomy evaluates to exact expected actionability when other criteria are met."""
    with patch('api.routes.analytics.optimal_execution_engine.calculate_trade_levels') as mock_calc, \
         patch('api.routes.analytics.confluence_engine.calculate_confluence') as mock_conf, \
         patch('api.routes.analytics.market_db.get_factor_snapshot', return_value={"quality_score": 85}), \
         patch('api.routes.analytics.smart_money_engine.get_sec_insider_trades', return_value=[]), \
         patch('api.routes.analytics.smart_money_engine.get_congressional_trades', return_value=[]):
        
        mock_conf.return_value = {"confluenceScore": 80.0}
        mock_calc.return_value = {
            "execution_status": status,
            "stop_loss": 95.0,
            "optimal_entry_max": 100.0,
            "take_profit_1": 115.0,
            "risk_reward_ratio": 2.5,
            "stage_phase": 2,
            "setup_pattern": "VCP Setup",
            "entry_thesis": "Test thesis",
        }
        
        setup = _build_tactical_setup('AAPL', 'SWING_TRADER', CANDLES, is_stale=(status == "STALE_MARKET_DATA"))
        assert setup['isActionable'] is expected_actionable, f"Status {status} expected isActionable={expected_actionable}"
        assert setup['isSuppressed'] is (not expected_actionable)
        if expected_actionable:
            assert setup['executionStatus'] in ACTIONABLE_EXECUTION_STATUSES
            assert setup['decisionState'] == "ACTIONABLE_SETUP"
        else:
            assert setup['executionStatus'] not in ACTIONABLE_EXECUTION_STATUSES or not expected_actionable
            assert setup['decisionState'] != "ACTIONABLE_SETUP"

def test_missing_levels_never_actionable_even_in_buy_zone():
    """Verify that IN_BUY_ZONE with missing levels is strictly NON-ACTIONABLE."""
    with patch('api.routes.analytics.optimal_execution_engine.calculate_trade_levels') as mock_calc, \
         patch('api.routes.analytics.confluence_engine.calculate_confluence') as mock_conf, \
         patch('api.routes.analytics.smart_money_engine.get_sec_insider_trades', return_value=[]), \
         patch('api.routes.analytics.smart_money_engine.get_congressional_trades', return_value=[]):
        
        mock_conf.return_value = {"confluenceScore": 80.0}
        
        # Case A: Missing stop_loss
        mock_calc.return_value = {
            "execution_status": "IN_BUY_ZONE",
            "stop_loss": None,
            "optimal_entry_max": 100.0,
            "take_profit_1": 115.0,
        }
        setup_no_stop = _build_tactical_setup('AAPL', 'SWING_TRADER', CANDLES, is_stale=False)
        assert setup_no_stop['isActionable'] is False

        # Case B: Missing entry_max
        mock_calc.return_value = {
            "execution_status": "IN_BUY_ZONE",
            "stop_loss": 95.0,
            "optimal_entry_max": None,
            "take_profit_1": 115.0,
        }
        setup_no_entry = _build_tactical_setup('AAPL', 'SWING_TRADER', CANDLES, is_stale=False)
        assert setup_no_entry['isActionable'] is False

def test_optimal_execution_enforce_invariants_actionability():
    """Verify that _enforce_execution_invariants embeds authoritative is_actionable field."""
    raw_plan = {
        "current_price": 100.0,
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 101.0,
        "stop_loss": 95.0,
        "take_profit_1": 115.0,
        "take_profit_2": 125.0,
        "risk_reward_ratio": 2.5,
        "execution_status": "IN_BUY_ZONE",
        "setup_pattern": "VCP",
    }
    plan = OptimalExecutionEngine._enforce_execution_invariants(raw_plan, "SWING_TRADER")
    assert plan["is_actionable"] is True

    raw_plan_waiting = {**raw_plan, "execution_status": "WAITING_PULLBACK"}
    plan_waiting = OptimalExecutionEngine._enforce_execution_invariants(raw_plan_waiting, "SWING_TRADER")
    assert plan_waiting["is_actionable"] is False


def test_missing_fundamentals_disqualifies_setup_even_in_buy_zone():
    """Verify that IN_BUY_ZONE without fundamentals is strictly EVIDENCE_INCOMPLETE and non-actionable."""
    with patch('api.routes.analytics.optimal_execution_engine.calculate_trade_levels') as mock_calc, \
         patch('api.routes.analytics.confluence_engine.calculate_confluence') as mock_conf, \
         patch('api.routes.analytics.market_db.get_factor_snapshot', return_value=None), \
         patch('api.routes.analytics.smart_money_engine.get_sec_insider_trades', return_value=[]), \
         patch('api.routes.analytics.smart_money_engine.get_congressional_trades', return_value=[]):
        
        mock_conf.return_value = {"confluenceScore": 85.0}
        mock_calc.return_value = {
            "execution_status": "IN_BUY_ZONE",
            "stop_loss": 95.0,
            "optimal_entry_max": 100.0,
            "take_profit_1": 115.0,
            "risk_reward_ratio": 3.0,
            "stage_phase": 2,
            "setup_pattern": "VCP Setup",
        }
        setup = _build_tactical_setup('NONETF_STOCK', 'SWING_TRADER', CANDLES, is_stale=False)
        assert setup['isActionable'] is False
        assert setup['isSuppressed'] is True
        assert setup['decisionState'] == "EVIDENCE_INCOMPLETE"
        assert "financial filings" in setup['reasonSuppressed'].lower() or "unverified" in setup['reasonSuppressed'].lower()


def test_low_confluence_disqualifies_setup_even_in_buy_zone():
    """Verify that IN_BUY_ZONE with confluence < 75 is VALID_SETUP (awaiting conviction) and non-actionable."""
    with patch('api.routes.analytics.optimal_execution_engine.calculate_trade_levels') as mock_calc, \
         patch('api.routes.analytics.confluence_engine.calculate_confluence') as mock_conf, \
         patch('api.routes.analytics.market_db.get_factor_snapshot', return_value={"quality_score": 85}), \
         patch('api.routes.analytics.smart_money_engine.get_sec_insider_trades', return_value=[]), \
         patch('api.routes.analytics.smart_money_engine.get_congressional_trades', return_value=[]):
        
        mock_conf.return_value = {"confluenceScore": 68.0}  # Below 75 threshold
        mock_calc.return_value = {
            "execution_status": "IN_BUY_ZONE",
            "stop_loss": 95.0,
            "optimal_entry_max": 100.0,
            "take_profit_1": 115.0,
            "risk_reward_ratio": 3.0,
            "stage_phase": 2,
            "setup_pattern": "VCP Setup",
        }
        setup = _build_tactical_setup('AAPL', 'SWING_TRADER', CANDLES, is_stale=False)
        assert setup['isActionable'] is False
        assert setup['isSuppressed'] is True
        assert setup['decisionState'] == "VALID_SETUP"
        assert "confluence" in setup['reasonSuppressed'].lower()


def test_low_risk_reward_disqualifies_setup_even_in_buy_zone():
    """Verify that IN_BUY_ZONE with R:R < 2.0 is non-actionable."""
    with patch('api.routes.analytics.optimal_execution_engine.calculate_trade_levels') as mock_calc, \
         patch('api.routes.analytics.confluence_engine.calculate_confluence') as mock_conf, \
         patch('api.routes.analytics.market_db.get_factor_snapshot', return_value={"quality_score": 85}), \
         patch('api.routes.analytics.smart_money_engine.get_sec_insider_trades', return_value=[]), \
         patch('api.routes.analytics.smart_money_engine.get_congressional_trades', return_value=[]):
        
        mock_conf.return_value = {"confluenceScore": 85.0}
        mock_calc.return_value = {
            "execution_status": "IN_BUY_ZONE",
            "stop_loss": 95.0,
            "optimal_entry_max": 100.0,
            "take_profit_1": 105.0,
            "risk_reward_ratio": 1.0,  # Below 2.0 institutional threshold
            "stage_phase": 2,
            "setup_pattern": "VCP Setup",
        }
        setup = _build_tactical_setup('AAPL', 'SWING_TRADER', CANDLES, is_stale=False)
        assert setup['isActionable'] is False
        assert setup['isSuppressed'] is True
        assert setup['decisionState'] == "VALID_SETUP"
        assert "risk/reward" in setup['reasonSuppressed'].lower()
