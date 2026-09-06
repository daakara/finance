"""Baseline Engines for ARX Model Governance.

Provides mathematically frozen, reproducible comparative baselines:
1. RandomEntryMonteCarlo: 1,000+ matched random-entry simulations yielding an empirical percentile rank.
2. MomentumBaselineEngine: Frozen 30-day momentum (Close(T0) / Close(T-30) - 1) benchmark.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class BaselineEngine:
    """Rigorous comparative baseline evaluation suite.

    Governance Standards & Epistemic Invariants:
    1. Pre-Registered Execution Convention: When High >= TP1 and Low <= SL within the same daily
       candle, both the strategy and the baseline resolve pessimistically to STOP_LOSS (fail-closed).
       This is an explicit pre-registered convention, not an empirical proof of intra-candle physical path.
    2. Linear Additive Friction Convention: Realized percentage returns are adjusted via additive
       deduction of 30 bps (0.30%) round-trip fee drag: R_net = R_gross - 0.30%.
    3. Survivorship Caveat: For prospective cohorts, the universe represents live assets currently
       available. However, 'actively quoted at T0' is NOT equivalent to a point-in-time historical
       investable universe (delistings, mergers, bankruptcies, suspensions). No claim of having solved
       historical survivorship bias is made.
    4. Dual Random Baselines: Both Unconditional Random and Sector-Conditioned Random must be
       evaluated and reported simultaneously; selective post-hoc choice of baseline is prohibited.
    """

    BASELINE_SPEC_VERSION = "2.0-FROZEN"
    PRIMARY_FRICTION_BPS = 30.0  # 30 bps round-trip transaction drag (additive deduction)

    @classmethod
    def compute_30d_momentum(cls, candles: List[Dict[str, Any]], sig_date: str) -> Optional[float]:
        """Calculates frozen 30-session momentum: Close(T0) / Close(T-30) - 1."""
        if not candles or len(candles) < 31:
            return None
        df = pd.DataFrame(candles)
        date_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
        if not date_col:
            return None
        prior = df[df[date_col] <= sig_date]
        if len(prior) < 31:
            return None
        close_col = "close" if "close" in prior.columns else "Close"
        c_t0 = float(prior.iloc[-1][close_col])
        c_t30 = float(prior.iloc[-31][close_col])
        if c_t30 <= 0:
            return None
        return round(((c_t0 - c_t30) / c_t30) * 100.0, 2)

    @classmethod
    def evaluate_matched_trade(
        cls,
        candles: List[Dict[str, Any]],
        sig_date: str,
        stop_loss_pct: float,
        take_profit_pct: float,
        max_sessions: int = 20,
        friction_bps: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Simulates a trade forward with matched risk geometry and pre-registered fail-closed convention."""
        if not candles:
            return None
        df = pd.DataFrame(candles)
        date_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
        if not date_col:
            return None

        subsequent = df[df[date_col] > sig_date].copy()
        if subsequent.empty:
            return None

        subsequent.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        sessions = min(len(subsequent), max_sessions)
        if sessions == 0:
            return None

        p0 = float(subsequent.iloc[0]["Open"])
        if p0 <= 0 or math.isnan(p0):
            return None

        stop_price = p0 * (1.0 - abs(stop_loss_pct) / 100.0)
        tp_price = p0 * (1.0 + abs(take_profit_pct) / 100.0)

        highs = subsequent["High"].values[:sessions]
        lows = subsequent["Low"].values[:sessions]
        closes = subsequent["Close"].values[:sessions]

        tp_hit = False
        stop_hit = False
        tp_session = None
        stop_session = None
        intrabar_collision = False

        for idx in range(sessions):
            h = highs[idx]
            l = lows[idx]
            curr_tp = (h >= tp_price)
            curr_stop = (l <= stop_price)

            if curr_tp and curr_stop:
                # Intrabar collision: fail-closed to stop loss
                intrabar_collision = True
                if not stop_hit:
                    stop_hit = True
                    stop_session = idx + 1
                if not tp_hit:
                    tp_hit = True
                    tp_session = idx + 1
                break

            if curr_tp and not tp_hit:
                tp_hit = True
                tp_session = idx + 1
            if curr_stop and not stop_hit:
                stop_hit = True
                stop_session = idx + 1

            if tp_hit or stop_hit:
                break

        # Resolve outcome
        friction_pct = friction_bps / 100.0
        if intrabar_collision or (stop_hit and (not tp_hit or stop_session <= tp_session)):
            gross_return = round(((stop_price - p0) / p0) * 100.0, 2)
            outcome = "STOP_LOSS"
        elif tp_hit and (not stop_hit or tp_session < stop_session):
            gross_return = round(((tp_price - p0) / p0) * 100.0, 2)
            outcome = "TP1_WIN"
        else:
            latest = float(closes[-1])
            gross_return = round(((latest - p0) / p0) * 100.0, 2)
            outcome = "TIME_EXPIRED"

        net_return = round(gross_return - friction_pct, 2)
        mfe = round(float(np.max(highs) - p0) / p0 * 100.0, 2)
        mae = round(float(np.min(lows) - p0) / p0 * 100.0, 2)

        return {
            "outcome": outcome,
            "grossReturnPct": gross_return,
            "netReturnPct": net_return,
            "mfePct": mfe,
            "maePct": mae,
            "sessionsObserved": sessions,
            "intrabarCollision": intrabar_collision
        }

    @classmethod
    def run_random_monte_carlo(
        cls,
        signals: List[Dict[str, Any]],
        universe_candles_map: Dict[str, List[Dict[str, Any]]],
        n_simulations: int = 1000,
        seed: int = 42,
        friction_bps: float = 30.0,
        match_sector: bool = False,
        asset_sectors_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Runs a Monte Carlo simulation of matched random trade sequences.
        Yields an empirical distribution and percentile rank for the actual trading engine.
        Supports sector-matched random sampling to prevent comparing tech-heavy signals to unconstrained beta.
        """
        if not signals or not universe_candles_map:
            return {"status": "INSUFFICIENT_DATA", "nSimulations": 0}

        rng = np.random.default_rng(seed)
        all_symbols = list(universe_candles_map.keys())
        if len(all_symbols) < 2:
            return {"status": "UNIVERSE_TOO_SMALL", "nSimulations": 0}

        # Actual engine results for comparison
        resolved_signals = [s for s in signals if s.get("status") == "RESOLVED"]
        if not resolved_signals:
            return {"status": "NO_RESOLVED_ACTUAL_SIGNALS", "nSimulations": 0}

        actual_returns = [
            s.get("forwardTracking", {}).get("realizedReturnPct", 0.0) - (friction_bps / 100.0)
            for s in resolved_signals
        ]
        actual_mean_ret = float(np.mean(actual_returns))
        actual_wins = sum(1 for r in actual_returns if r > 0)
        actual_win_rate = (actual_wins / len(actual_returns)) * 100.0

        sim_mean_returns: List[float] = []
        sim_win_rates: List[float] = []
        sim_expectancies: List[float] = []

        for _ in range(n_simulations):
            sim_portfolio_returns = []
            sim_wins = 0

            for actual_sig in resolved_signals:
                sig_date = actual_sig.get("signalDate", "2026-09-01")
                stop_pct = abs(float(actual_sig.get("stopLossPct", -6.5)))
                tp_pct = abs(float(actual_sig.get("takeProfit1Pct", 15.0)))
                actual_sym = actual_sig.get("symbol", "")
                actual_sec = actual_sig.get("inputs", {}).get("sector")

                # Sample an alternative symbol randomly (excluding the actual symbol to test pure random selection)
                candidates = [s for s in all_symbols if s != actual_sym and universe_candles_map.get(s)]
                if match_sector and asset_sectors_map and actual_sec:
                    sec_matches = [s for s in candidates if asset_sectors_map.get(s) == actual_sec]
                    if sec_matches:
                        candidates = sec_matches

                if not candidates:
                    candidates = [s for s in all_symbols if universe_candles_map.get(s)] or all_symbols
                random_sym = rng.choice(candidates)
                r_candles = universe_candles_map.get(random_sym, [])

                res = cls.evaluate_matched_trade(
                    candles=r_candles,
                    sig_date=sig_date,
                    stop_loss_pct=stop_pct,
                    take_profit_pct=tp_pct,
                    max_sessions=20,
                    friction_bps=friction_bps
                )
                if res:
                    ret = res["netReturnPct"]
                    sim_portfolio_returns.append(ret)
                    if res["outcome"] == "TP1_WIN":
                        sim_wins += 1

            if sim_portfolio_returns:
                p_mean = float(np.mean(sim_portfolio_returns))
                p_wr = (sim_wins / len(sim_portfolio_returns)) * 100.0
                sim_mean_returns.append(round(p_mean, 2))
                sim_win_rates.append(round(p_wr, 1))
                sim_expectancies.append(round(p_mean, 2))

        if not sim_mean_returns:
            return {"status": "SIMULATION_FAILED", "nSimulations": 0}

        # Compute percentiles
        p5 = float(np.percentile(sim_mean_returns, 5))
        p25 = float(np.percentile(sim_mean_returns, 25))
        p50 = float(np.percentile(sim_mean_returns, 50))
        p75 = float(np.percentile(sim_mean_returns, 75))
        p95 = float(np.percentile(sim_mean_returns, 95))

        # Empirical percentile rank of the strategy
        count_below = sum(1 for r in sim_mean_returns if r < actual_mean_ret)
        percentile_rank = round((count_below / len(sim_mean_returns)) * 100.0, 1)

        return {
            "status": "COMPLETED",
            "nSimulations": len(sim_mean_returns),
            "seed": seed,
            "assumedFrictionBps": friction_bps,
            "actualStrategy": {
                "sampleSize": len(resolved_signals),
                "netMeanReturnPct": round(actual_mean_ret, 2),
                "winRatePct": round(actual_win_rate, 1),
                "percentileRankVsRandom": percentile_rank,
            },
            "randomDistribution": {
                "meanNetReturnPct": round(float(np.mean(sim_mean_returns)), 2),
                "medianNetReturnPct": round(float(np.median(sim_mean_returns)), 2),
                "p5": round(p5, 2),
                "p25": round(p25, 2),
                "p50": round(p50, 2),
                "p75": round(p75, 2),
                "p95": round(p95, 2),
                "meanWinRatePct": round(float(np.mean(sim_win_rates)), 1),
            },
            "interpretation": (
                f"ArxTerminal net return ({round(actual_mean_ret, 2)}%) ranks at the "
                f"{percentile_rank}th percentile of 1,000 matched random-entry simulations."
            )
        }

    @classmethod
    def evaluate_dual_random_baselines(
        cls,
        signals: List[Dict[str, Any]],
        universe_candles_map: Dict[str, List[Dict[str, Any]]],
        n_simulations: int = 1000,
        seed: int = 42,
        friction_bps: float = 30.0,
        asset_sectors_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Executes BOTH mandatory random benchmarks to prevent selective post-hoc reporting:
        1. Unconditional Random: Is ArxTerminal better than selecting an investable stock at random?
        2. Sector-Conditioned Random: Is ArxTerminal better than simply selecting another stock from the same sector?
        """
        unconditional = cls.run_random_monte_carlo(
            signals=signals,
            universe_candles_map=universe_candles_map,
            n_simulations=n_simulations,
            seed=seed,
            friction_bps=friction_bps,
            match_sector=False
        )
        sector_conditioned = cls.run_random_monte_carlo(
            signals=signals,
            universe_candles_map=universe_candles_map,
            n_simulations=n_simulations,
            seed=seed + 1,  # Distinct deterministic seed
            friction_bps=friction_bps,
            match_sector=True,
            asset_sectors_map=asset_sectors_map
        ) if asset_sectors_map else {"status": "NO_SECTOR_MAP_PROVIDED"}

        return {
            "unconditionalRandom": unconditional,
            "sectorConditionedRandom": sector_conditioned,
            "reportingRequirement": "Both baselines must be reported simultaneously; selecting whichever looks better is prohibited."
        }

    @classmethod
    def evaluate_momentum_baseline(
        cls,
        universe_candles_map: Dict[str, List[Dict[str, Any]]],
        signal_dates: List[str],
        top_pct: float = 10.0,
        stop_loss_pct: float = 6.5,
        take_profit_pct: float = 15.0,
        friction_bps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Evaluates the top-decile 30-day momentum benchmark on matched signal dates.
        Momentum = Close(T0) / Close(T-30) - 1.
        """
        trade_outcomes = []
        for s_date in set(signal_dates):
            mom_scores = []
            for sym, candles in universe_candles_map.items():
                m = cls.compute_30d_momentum(candles, s_date)
                if m is not None:
                    mom_scores.append((sym, m))

            if not mom_scores:
                continue

            mom_scores.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(len(mom_scores) * (top_pct / 100.0)))
            top_momentum_symbols = [s[0] for s in mom_scores[:k]]

            for sym in top_momentum_symbols:
                res = cls.evaluate_matched_trade(
                    candles=universe_candles_map[sym],
                    sig_date=s_date,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    friction_bps=friction_bps
                )
                if res:
                    trade_outcomes.append(res)

        if not trade_outcomes:
            return {"status": "NO_TRADES_EVALUATED", "nTrades": 0}

        rets = [t["netReturnPct"] for t in trade_outcomes]
        wins = sum(1 for t in trade_outcomes if t["outcome"] == "TP1_WIN")

        return {
            "status": "COMPLETED",
            "nTrades": len(trade_outcomes),
            "definition": "Close(T0) / Close(T-30) - 1; Top 10% selected at T0",
            "netMeanReturnPct": round(float(np.mean(rets)), 2),
            "netMedianReturnPct": round(float(np.median(rets)), 2),
            "winRatePct": round((wins / len(trade_outcomes)) * 100.0, 1),
            "assumedFrictionBps": friction_bps,
        }
