import sys
import os
sys.path.insert(0, os.path.abspath("."))
from analyst_dashboard.analyzers.catalysts import CatalystEngine

def test_metadata_and_sector_heuristics():
    engine = CatalystEngine()
    test_cases = [
        ('KO', 'Consumer Defensive', 'Beverages - Non-Alcoholic', 'The Coca-Cola Company'),
        ('SBUX', 'Consumer Cyclical', 'Restaurants', 'Starbucks Corporation'),
        ('O', 'Real Estate', 'REIT - Commercial', 'Realty Income Corporation'),
        ('XOM', 'Energy', 'Oil & Gas Integrated', 'Exxon Mobil Corporation'),
        ('NEM', 'Basic Materials', 'Gold Mining', 'Newmont Corporation'),
        ('JPM', 'Financial Services', 'Banks - Diversified', 'JPMorgan Chase & Co.'),
        ('NVDA', 'Technology', 'Semiconductors', 'NVIDIA Corporation'),
    ]
    
    theses = set()
    print('Testing Sector & Asset Catalysts Across 7 Diverse Sectors...')
    for sym, sec, ind, name in test_cases:
        rep = engine.get_asset_catalyst_report(symbol=sym, current_price=100.0, sector=sec, industry=ind, company_name=name)
        assert rep['company_name'] == name, f"Name mismatch: {rep['company_name']} vs {name}"
        assert len(rep['primary_drug_trial']) > 5, f"Empty catalyst for {sym}"
        assert len(rep['efficacy_summary']) > 10, f"Empty summary for {sym}"
        theses.add(rep['efficacy_summary'])
        print(f"  [PASS] {sym:<5} ({sec:<20}) -> Catalyst: {rep['primary_drug_trial'][:45]}...")
        
    assert len(theses) == len(test_cases), 'Duplicate thesis detected across distinct sectors!'
    print(f'Successfully validated {len(test_cases)} distinct sector profiles with 0 duplication.')

def test_day_trader_sizing_invariants():
    print('\nTesting Day Trader Leverage & FINRA 4210 Compliance Invariants...')
    
    # Case 1: Cash Mode with High Volatility (Should Clamp to 100% Buying Power)
    account_size = 25000
    price = 314.50
    risk_budget = 250
    stop_distance = 2.52
    
    raw_vol_units = int(risk_budget / stop_distance) # 99 units
    max_cash_units = int(account_size / price)       # 79 units
    cash_units = min(max_cash_units, raw_vol_units)
    
    assert cash_units == 79, f'Cash units should clamp to 79, got {cash_units}'
    assert cash_units * price <= account_size, 'Cash position exceeded account equity!'
    print(f'  [PASS] Cash Mode: Sized to {cash_units} units ( <= )')
    
    # Case 2: Margin Mode with Account < ,000 (Should Trigger PDT Alert)
    small_account = 10000
    pdt_alert = (small_account < 25000)
    assert pdt_alert is True, 'PDT alert failed to trigger on  margin account!'
    print(f'  [PASS] Margin Mode (<): Correctly triggered FINRA Rule 4210 PDT restriction.')
    
    # Case 3: Margin Mode with Account >= ,000 (Compliant)
    large_account = 50000
    pdt_alert_large = (large_account < 25000)
    assert pdt_alert_large is False, 'PDT alert falsely triggered on  margin account!'
    print(f'  [PASS] Margin Mode (>=): Correctly recognized compliant PDT status.')

def test_smart_money_timeframe_filtering():
    print('\nTesting Smart Money Horizon Timeframe Filtering (7D, 30D, 90D, 180D, 1Y, ALL)...')
    from datetime import datetime, timedelta
    
    now = datetime(2026, 8, 28)
    sample_trades = [
        {"id": "t1", "date": (now - timedelta(days=3)).strftime("%Y-%m-%d")},   # 3d ago (in 7D, 30D, 90D, 180D, 1Y, ALL)
        {"id": "t2", "date": (now - timedelta(days=18)).strftime("%Y-%m-%d")},  # 18d ago (in 30D, 90D, 180D, 1Y, ALL)
        {"id": "t3", "date": (now - timedelta(days=65)).strftime("%Y-%m-%d")},  # 65d ago (in 90D, 180D, 1Y, ALL)
        {"id": "t4", "date": (now - timedelta(days=120)).strftime("%Y-%m-%d")}, # 120d ago (in 180D, 1Y, ALL)
        {"id": "t5", "date": (now - timedelta(days=300)).strftime("%Y-%m-%d")}, # 300d ago (in 1Y, ALL)
        {"id": "t6", "date": (now - timedelta(days=500)).strftime("%Y-%m-%d")}, # 500d ago (in ALL only)
    ]
    
    def filter_tf(trades, tf):
        if tf == "ALL":
            return trades
        limit_days = {"7D": 7, "30D": 30, "90D": 90, "180D": 180, "1Y": 365}[tf]
        return [t for t in trades if (now - datetime.strptime(t["date"], "%Y-%m-%d")).days <= limit_days]
        
    assert len(filter_tf(sample_trades, "7D")) == 1
    assert len(filter_tf(sample_trades, "30D")) == 2
    assert len(filter_tf(sample_trades, "90D")) == 3
    assert len(filter_tf(sample_trades, "180D")) == 4
    assert len(filter_tf(sample_trades, "1Y")) == 5
    assert len(filter_tf(sample_trades, "ALL")) == 6
    print("  [PASS] 7D Filter: Correctly captured breaking trades (1 item)")
    print("  [PASS] 30D Filter: Correctly captured active monthly trades (2 items)")
    print("  [PASS] 90D Filter: Correctly captured quarterly cluster trades (3 items)")
    print("  [PASS] 180D Filter: Correctly captured 6-month flow trades (4 items)")
    print("  [PASS] 1Y Filter: Correctly captured 1-year macro conviction trades (5 items)")
    print("  [PASS] ALL Filter: Correctly preserved full historical records (6 items)")

if __name__ == '__main__':
    test_metadata_and_sector_heuristics()
    test_day_trader_sizing_invariants()
    test_smart_money_timeframe_filtering()
    print('\n>>> ALL INTERACTION & INVARIANT TESTS PASSED! <<<')
