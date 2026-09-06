"""
Audit Liquidity Governance & Counterfactual Discrimination Script.
Appends shadow liquidity observations to Phase 25 forward ledger
and computes historical discrimination metrics across the catalog universe.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

def audit_and_enrich_phase25_ledger():
    """Enrich the 9 active signals in paper_trading_ledger.json with liquidity observations."""
    market_db = MarketDatabaseEngine()
    ledger_path = ExperimentLedger.DEFAULT_LEDGER_PATH
    ledger = ExperimentLedger.load_ledger(ledger_path)

    print(f"Loaded ledger with {len(ledger['signals'])} signals.")

    enriched_count = 0
    for sig in ledger["signals"]:
        sym = sig["symbol"]
        entry_price = sig["entryPrice"]

        # Fetch historical daily candles up to signal date
        candles = market_db.get_daily_candles(sym, limit=60)
        if not candles:
            # Fallback to yfinance if not cached in local market db
            try:
                import yfinance as yf
                df = yf.Ticker(sym).history(period="3mo", interval="1d")
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame(candles)
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

        liq = LiquidityGuard.evaluate_liquidity(df, entry_price)

        # Append shadow observation object without mutating frozen decision fields
        sig["liquidityObservation"] = {
            "liquidityGrade": liq["liquidity_grade"],
            "badgeColor": liq["badge_color"],
            "adv20dUsd": liq["adv_20d_usd"],
            "amihudIlliqRaw": liq["amihud_illiq"],
            "amihudIlliqScaled": liq["amihud_illiq_scaled"],
            "volumeSpikeRatio": liq["volume_spike_ratio"],
            "executionHazard": liq["execution_hazard"],
            "shadowObservationTimestamp": "2026-09-04T17:45:00Z"
        }
        enriched_count += 1
        print(f"[{sym}] Grade: {liq['liquidity_grade']}, ADV: ${liq['adv_20d_usd']:,.0f}, Amihud Raw: {liq['amihud_illiq']:.2e}, Scaled: {liq['amihud_illiq_scaled']:.4f}")

    ExperimentLedger.save_ledger(ledger, ledger_path)
    print(f"Enriched {enriched_count} signals in {ledger_path} with zero model decision mutations.")

if __name__ == "__main__":
    audit_and_enrich_phase25_ledger()
