"""
Historical Liquidity Discrimination & Counterfactual Simulator.
Evaluates historical setups across the catalog universe using the frozen v2.4.0 engine,
classifies them into Liquidity Tiers (DEEP_LIQUIDITY, LIMIT_ORDER_REQUIRED, EXECUTION_RISK),
and computes realized forward outcomes (TP1 vs Stop) to produce:
1. Liquidity Cohort Table
2. Counterfactual Filter Analysis
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from api.routes.screener import DAY_TRADER_CANDIDATES, LONG_TERM_CANDIDATES

def run_simulation():
    market_db = MarketDatabaseEngine()
    universe = list(set(DAY_TRADER_CANDIDATES + LONG_TERM_CANDIDATES))

    # We also include smaller-cap / illiquid tickers to ensure all cohorts have observations
    extended_universe = universe + [
        "SIVE.ST", "AAOI", "RPI.L", "POWI", "CPRX", "MEDP", "LNTH", "ACLS",
        "TMDX", "CELH", "IONQ", "RKLB", "APP", "MARA", "HOOD"
    ]
    extended_universe = sorted(list(set(extended_universe)))

    events: List[Dict[str, Any]] = []

    print(f"Scanning {len(extended_universe)} assets for historical setups...")

    for sym in extended_universe:
        candles = market_db.get_daily_candles(sym, limit=252)
        if not candles or len(candles) < 60:
            continue

        df = pd.DataFrame(candles)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
        date_col = "time" if "time" in df.columns else "date"

        # Walk through historical sessions with a 40-bar minimum burn-in
        # Sample every 10 sessions to avoid excessive autocorrelation
        for t in range(40, len(df) - 20, 10):
            sub_df = df.iloc[:t].copy()
            current_price = float(sub_df["Close"].iloc[-1])
            if current_price <= 0:
                continue

            # Run frozen v2.4.0 execution ladder calculation
            plan = OptimalExecutionEngine.calculate_trade_levels(sub_df, current_price, user_role="LONG_TERM")

            entry_min = plan["optimal_entry_min"]
            entry_max = plan["optimal_entry_max"]
            stop = plan["stop_loss"]
            tp1 = plan["take_profit_1"]

            # Check if setup was technically ACTIONABLE (in accumulation zone with R:R >= 1.85)
            if entry_min is None or entry_max is None or plan.get("risk_reward_ratio") is None:
                continue
            is_actionable = (entry_min <= current_price <= entry_max * 1.008) and (plan["risk_reward_ratio"] >= 1.85)
            if not is_actionable:
                continue

            # Calculate shadow liquidity classification at timestamp t
            liq = LiquidityGuard.evaluate_liquidity(sub_df, current_price)
            cohort = liq["liquidity_grade"]

            # Measure forward outcome over the subsequent 20 sessions
            forward_df = df.iloc[t:t+20]
            highs = forward_df["High"].values
            lows = forward_df["Low"].values
            closes = forward_df["Close"].values

            tp1_hit = False
            tp1_session = None
            stop_hit = False
            stop_session = None

            for s_idx in range(len(forward_df)):
                if highs[s_idx] >= tp1 and not tp1_hit:
                    tp1_hit = True
                    tp1_session = s_idx
                if lows[s_idx] <= stop and not stop_hit:
                    stop_hit = True
                    stop_session = s_idx

            if tp1_hit and (not stop_hit or tp1_session <= stop_session):
                outcome = "TP1_WIN"
                realized_ret = round(((tp1 - current_price) / current_price) * 100.0, 2)
            elif stop_hit and (not tp1_hit or stop_session < tp1_session):
                outcome = "STOP_LOSS"
                realized_ret = round(((stop - current_price) / current_price) * 100.0, 2)
            else:
                outcome = "TIME_EXPIRED"
                final_close = float(closes[-1])
                realized_ret = round(((final_close - current_price) / current_price) * 100.0, 2)

            events.append({
                "symbol": sym,
                "date": sub_df[date_col].iloc[-1],
                "cohort": cohort,
                "advUsd": liq["adv_20d_usd"],
                "amihudRaw": liq["amihud_illiq"],
                "amihudScaled": liq["amihud_illiq_scaled"],
                "outcome": outcome,
                "realizedReturnPct": realized_ret,
                "isWin": outcome == "TP1_WIN" or realized_ret > 0,
                "isLoss": outcome == "STOP_LOSS" or realized_ret < 0,
            })

    print(f"Total simulated historical actionable setups evaluated: {len(events)}")
    if not events:
        print("No historical events matched criteria.")
        return

    ev_df = pd.DataFrame(events)

    # 1. Produce Cohort Table
    cohorts = ["DEEP_LIQUIDITY", "LIMIT_ORDER_REQUIRED", "EXECUTION_RISK"]
    cohort_stats = []

    for c in cohorts:
        c_sub = ev_df[ev_df["cohort"] == c]
        n = len(c_sub)
        if n == 0:
            continue

        wins = c_sub[c_sub["outcome"] == "TP1_WIN"]
        stops = c_sub[c_sub["outcome"] == "STOP_LOSS"]
        tp1_rate = len(wins) / n * 100.0
        stop_rate = len(stops) / n * 100.0
        mean_ret = float(c_sub["realizedReturnPct"].mean())

        avg_w = float(wins["realizedReturnPct"].mean()) if len(wins) > 0 else 0.0
        avg_l = float(abs(stops["realizedReturnPct"].mean())) if len(stops) > 0 else 0.0
        expectancy = (len(wins)/n * avg_w) - (len(stops)/n * avg_l)

        tot_gains = sum(wins["realizedReturnPct"]) if len(wins) > 0 else 0.0
        tot_losses = sum(abs(stops["realizedReturnPct"])) if len(stops) > 0 else 0.0
        pf = tot_gains / tot_losses if tot_losses > 0 else (999.0 if tot_gains > 0 else 0.0)

        cohort_stats.append({
            "Cohort": c,
            "N": n,
            "TP1 Rate (%)": round(tp1_rate, 1),
            "Stop Rate (%)": round(stop_rate, 1),
            "Mean Return (%)": round(mean_ret, 2),
            "Expectancy (%)": round(expectancy, 2),
            "Profit Factor": round(pf, 2),
        })

    cohort_table = pd.DataFrame(cohort_stats)
    print("\n--- LIQUIDITY COHORT TABLE ---")
    print(cohort_table.to_string(index=False))

    # 2. Produce Counterfactual Comparison Table
    all_n = len(ev_df)
    filtered_df = ev_df[ev_df["cohort"] != "EXECUTION_RISK"]
    filt_n = len(filtered_df)

    def calc_metrics(df_sub):
        n = len(df_sub)
        if n == 0:
            return {}
        w = df_sub[df_sub["outcome"] == "TP1_WIN"]
        s = df_sub[df_sub["outcome"] == "STOP_LOSS"]
        tp1_r = len(w) / n * 100.0
        stop_r = len(s) / n * 100.0
        m_ret = float(df_sub["realizedReturnPct"].mean())
        med_ret = float(df_sub["realizedReturnPct"].median())
        avg_w = float(w["realizedReturnPct"].mean()) if len(w) > 0 else 0.0
        avg_l = float(abs(s["realizedReturnPct"].mean())) if len(s) > 0 else 0.0
        exp = (len(w)/n * avg_w) - (len(s)/n * avg_l)
        tg = sum(w["realizedReturnPct"]) if len(w) > 0 else 0.0
        tl = sum(abs(s["realizedReturnPct"])) if len(s) > 0 else 0.0
        pf = tg / tl if tl > 0 else (999.0 if tg > 0 else 0.0)
        return {
            "N": n,
            "TP1 Rate": f"{tp1_r:.1f}%",
            "Stop Rate": f"{stop_r:.1f}%",
            "Mean Return": f"{m_ret:+.2f}%",
            "Median Return": f"{med_ret:+.2f}%",
            "Expectancy": f"{exp:+.2f}%",
            "Profit Factor": f"{pf:.2f}",
        }

    m_existing = calc_metrics(ev_df)
    m_filtered = calc_metrics(filtered_df)

    # Discard analysis
    discarded = ev_df[ev_df["cohort"] == "EXECUTION_RISK"]
    n_disc = len(discarded)
    disc_wins = len(discarded[discarded["isWin"]])
    disc_losses = len(discarded[discarded["isLoss"]])
    total_wins = len(ev_df[ev_df["isWin"]])
    total_losses = len(ev_df[ev_df["isLoss"]])

    pct_good_discarded = (disc_wins / total_wins * 100.0) if total_wins > 0 else 0.0
    pct_bad_discarded = (disc_losses / total_losses * 100.0) if total_losses > 0 else 0.0

    print("\n--- COUNTERFACTUAL ANALYSIS ---")
    cf_data = [
        {"Model": "Frozen v2.4.0 Baseline (All Actionable)", **m_existing},
        {"Model": "Hypothetical Liquidity-Filtered (Excl. Risk)", **m_filtered},
    ]
    print(pd.DataFrame(cf_data).to_string(index=False))

    print("\n--- DISCARD TRADEOFF RATIO ---")
    print(f"Total Actionable Setups Discarded: {n_disc} ({n_disc/all_n*100:.1f}%)")
    print(f"Profitable Signals Discarded: {disc_wins}/{total_wins} ({pct_good_discarded:.1f}%)")
    print(f"Losing Signals Discarded: {disc_losses}/{total_losses} ({pct_bad_discarded:.1f}%)")
    print(f"Signal Filtering Efficiency (Bad Discarded / Good Discarded): {pct_bad_discarded / max(0.01, pct_good_discarded):.2f}x")

if __name__ == "__main__":
    run_simulation()
