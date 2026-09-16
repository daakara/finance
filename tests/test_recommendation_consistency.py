import pytest
from unittest.mock import patch, MagicMock
from api.routes.analytics import _build_tactical_setup

def test_tactical_setup_actionable_only_in_buy_zone():
    candles = [
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000000, "date": "2026-03-01"},
        {"open": 104.0, "high": 106.0, "low": 103.0, "close": 105.0, "volume": 1200000, "date": "2026-03-02"},
    ]

    # Test Case 1: Stale data must NEVER be actionable
    stale_setup = _build_tactical_setup('AAPL', 'SWING_TRADER', candles, is_stale=True)
    assert stale_setup['isActionable'] is False
    assert stale_setup['isSuppressed'] is True
    assert stale_setup['executionStatus'] == 'STALE_MARKET_DATA'

    # Test Case 2: WAITING_PULLBACK must NOT be actionable
    with patch('api.routes.analytics.optimal_execution_engine.calculate_trade_levels') as mock_calc, \
         patch('api.routes.analytics.confluence_engine.calculate_confluence') as mock_conf, \
         patch('api.routes.analytics.smart_money_engine.get_sec_insider_trades', return_value=[]), \
         patch('api.routes.analytics.smart_money_engine.get_congressional_trades', return_value=[]):
        
        mock_conf.return_value = {"confluenceScore": 75.0}
        
        mock_calc.return_value = {
            "execution_status": "WAITING_PULLBACK",
            "stop_loss": 95.0,
            "optimal_entry_max": 101.0,
            "take_profit_1": 115.0,
            "setup_pattern": "Cup and Handle",
            "entry_thesis": "Waiting for consolidation"
        }
        setup_pullback = _build_tactical_setup('AAPL', 'SWING_TRADER', candles, is_stale=False)
        assert setup_pullback['isActionable'] is False
        assert setup_pullback['isSuppressed'] is True
        assert setup_pullback['executionStatus'] == 'WAITING_PULLBACK'
        assert setup_pullback['userRole'] == 'SWING_TRADER'

        # Test Case 3: IN_BUY_ZONE_AWAITING_TRIGGER must NOT be actionable
        mock_calc.return_value['execution_status'] = 'IN_BUY_ZONE_AWAITING_TRIGGER'
        setup_await = _build_tactical_setup('AAPL', 'SWING_TRADER', candles, is_stale=False)
        assert setup_await['isActionable'] is False
        assert setup_await['isSuppressed'] is True

        # Test Case 4: IN_BUY_ZONE with valid levels MUST be actionable
        mock_calc.return_value['execution_status'] = 'IN_BUY_ZONE'
        setup_active = _build_tactical_setup('AAPL', 'SWING_TRADER', candles, is_stale=False)
        assert setup_active['isActionable'] is True
        assert setup_active['isSuppressed'] is False
        assert setup_active['stopLoss'] == 95.0
        assert setup_active['entryPivot'] == 101.0
        assert setup_active['userRole'] == 'SWING_TRADER'

        # Test Case 5: READY_TO_BUY with valid levels MUST be actionable
        mock_calc.return_value['execution_status'] = 'READY_TO_BUY'
        setup_ready = _build_tactical_setup('AAPL', 'DAY_TRADER', candles, is_stale=False)
        assert setup_ready['isActionable'] is True
        assert setup_ready['isSuppressed'] is False
        assert setup_ready['userRole'] == 'DAY_TRADER'
