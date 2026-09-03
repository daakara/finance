'''Phase 15 Adversarial Production Launch & Reliability Verification Suite.'''

import pytest
import pandas as pd
import numpy as np
import math
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine


def test_confluence_engine_fails_closed_on_missing_fundamentals():
    engine = ConfluenceEngine()
    result = engine.calculate_confluence(
        symbol='UNCATALOGED_XYZ',
        technical_data={'executionStatus': 'IN_BUY_ZONE', 'riskRewardRatio': 2.5},
        smart_money_data={'has_insider_buy': False, 'has_congress_buy': False},
        fundamental_data=None,
        catalyst_data=None,
        macro_data=None,
    )
    fund_pillar = next(p for p in result['pillars'] if p['pillar'] == 'FUNDAMENTAL_SOLVENCY')
    assert fund_pillar['score'] == 0.0
    assert 'unavailable' in fund_pillar['detail'].lower()
    # Ensure composite score does not artificially award fundamental points
    assert result['confluenceScore'] < 70.0


def test_confluence_engine_awards_points_on_valid_fundamentals():
    engine = ConfluenceEngine()
    fund_data = {
        'piotroski_f': 9,
        'qualityScore': 95.0,
        'growthScore': 90.0,
        'valuationScore': 80.0,
        'roic': 25.4,
    }
    result = engine.calculate_confluence(
        symbol='PRISTINE_CO',
        technical_data={'executionStatus': 'IN_BUY_ZONE', 'riskRewardRatio': 2.8},
        smart_money_data={'has_insider_buy': True, 'has_congress_buy': True},
        fundamental_data=fund_data,
        catalyst_data={'days_to_earnings': 35},
        macro_data=None,
    )
    fund_pillar = next(p for p in result['pillars'] if p['pillar'] == 'FUNDAMENTAL_SOLVENCY')
    assert fund_pillar['score'] > 80.0
    assert fund_pillar['status'] == 'positive'
    assert 'Fortress Solvency' in fund_pillar['detail']


def test_optimal_execution_strictly_requires_50_sessions_for_50_sma():
    prices = [100.0 + i * 0.5 for i in range(35)]
    df = pd.DataFrame({
        'Open': prices,
        'High': [p + 1.0 for p in prices],
        'Low': [p - 1.0 for p in prices],
        'Close': prices,
        'Volume': [1000000] * 35,
    })
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=prices[-1], user_role='LONG_TERM')
    assert 'Trend Evidence Incomplete' in plan['setup_pattern']
    assert '50 Sessions' in plan['setup_pattern']
    assert 'Awaiting Historical Base' in plan['stage_phase']


def test_optimal_execution_with_50_plus_sessions_computes_sma50_and_vcp():
    prices = [100.0 + i * 0.5 for i in range(60)]
    df = pd.DataFrame({
        'Open': prices,
        'High': [p + 1.0 for p in prices],
        'Low': [p - 1.0 for p in prices],
        'Close': prices,
        'Volume': [1000000] * 60,
    })
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=prices[-1], user_role='LONG_TERM')
    assert 'Minervini VCP' in plan['setup_pattern']
    assert 'Stage 2' in plan['stage_phase']
    assert plan['stop_loss'] < plan['optimal_entry_min']
    assert plan['risk_reward_ratio'] >= 1.85


def test_optimal_execution_stage_4_markdown():
    prices = [150.0] * 50 + [100.0] * 5
    df = pd.DataFrame({
        'Open': prices,
        'High': [p + 1.0 for p in prices],
        'Low': [p - 1.0 for p in prices],
        'Close': prices,
        'Volume': [1000000] * len(prices),
    })
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price=100.0, user_role='LONG_TERM')
    assert 'Stage 4' in plan['setup_pattern']
    assert 'Stage 4 Markdown' in plan['stage_phase']


def test_position_sizing_fails_closed_on_invalid_inputs():
    res1 = ConfluenceEngine.calculate_position_size(
        account_equity=100000,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit_1=120.0,
    )
    assert res1['shares'] == 0
    assert 'error' in res1

    res2 = ConfluenceEngine.calculate_position_size(
        account_equity=0,
        risk_pct=1.0,
        entry_price=100.0,
        stop_loss=93.0,
        take_profit_1=120.0,
    )
    assert res2['shares'] == 0
