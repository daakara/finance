"""ARX Model Governance & Experiment Ledger Engine.

Maintains an immutable audit trail:
engine_version -> signal -> timestamp -> inputs -> decision -> entry -> stop -> TP1 -> TP2 -> outcome

Tracks forward paper-trading positions without post-hoc modification.
"""

import os
import json
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ExperimentLedger:
    """Production-grade Model Governance and Forward Experiment Tracker."""

    DEFAULT_LEDGER_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "paper_trading_ledger.json"
    )

    @classmethod
    def load_ledger(cls, ledger_path: Optional[str] = None) -> Dict[str, Any]:
        path = ledger_path or cls.DEFAULT_LEDGER_PATH
        if not os.path.exists(path):
            return {
                "version": "1.0.0",
                "engineCommit": "4e36862",
                "tag": "v2.4.0-phase24-freeze",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "totalActiveSignals": 0,
                "signals": []
            }
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def save_ledger(cls, data: Dict[str, Any], ledger_path: Optional[str] = None) -> None:
        path = ledger_path or cls.DEFAULT_LEDGER_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def register_signal(
        cls,
        symbol: str,
        entry_price: float,
        opt_exec: Dict[str, Any],
        confluence_score: float,
        inputs_meta: Dict[str, Any],
        engine_commit: str = "4e36862",
        engine_tag: str = "v2.4.0-phase24-freeze",
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record an immutable new signal in the ledger."""
        ledger = cls.load_ledger(ledger_path)
        signal_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        signal_id = f"{symbol}_{signal_date}"

        # Prevent duplicate entries for the same symbol on the same date
        existing = next((s for s in ledger["signals"] if s["signalId"] == signal_id), None)
        if existing:
            return existing

        record = {
            "signalId": signal_id,
            "symbol": symbol.upper().strip(),
            "signalDate": signal_date,
            "engineVersion": engine_commit,
            "engineTag": engine_tag,
            "decisionState": "ACTIONABLE_SETUP",
            "entryPrice": float(entry_price),
            "corridorMin": opt_exec.get("optimal_entry_min"),
            "corridorMax": opt_exec.get("optimal_entry_max"),
            "stopLoss": opt_exec.get("stop_loss"),
            "stopLossPct": opt_exec.get("stop_loss_pct"),
            "takeProfit1": opt_exec.get("take_profit_1"),
            "takeProfit1Pct": opt_exec.get("take_profit_1_pct"),
            "takeProfit2": opt_exec.get("take_profit_2"),
            "takeProfit2Pct": opt_exec.get("take_profit_2_pct"),
            "riskRewardRatio": opt_exec.get("risk_reward_ratio"),
            "confluenceScore": float(confluence_score),
            "inputs": {
                "atr14": opt_exec.get("atr_14"),
                "atrPct": round((opt_exec.get("atr_14", 0) / max(0.01, entry_price)) * 100, 2),
                "setupPattern": opt_exec.get("setup_pattern"),
                "stagePhase": opt_exec.get("stage_phase"),
                "marketRegime": inputs_meta.get("market_regime", "BULL"),
                "sector": inputs_meta.get("sector", "EQUITY"),
                "assetClass": inputs_meta.get("asset_class", "US_EQUITY"),
            },
            "status": "OPEN",
            "forwardTracking": {
                "sessionsObserved": 0,
                "currentPrice": float(entry_price),
                "maxFavorableExcursionPct": 0.0,
                "maxAdverseExcursionPct": 0.0,
                "tp1Hit": False,
                "tp1Session": None,
                "stopHit": False,
                "stopSession": None,
                "return5d": None,
                "return10d": None,
                "return20d": None,
                "signalPersistence": {
                    "day1Valid": True,
                    "day3Valid": None,
                    "day5Valid": None,
                    "day10Valid": None
                },
                "resolutionDate": None,
                "resolvedOutcome": None,  # "TP1_WIN", "STOP_LOSS", "TIME_EXPIRED"
                "realizedReturnPct": None
            }
        }
        ledger["signals"].append(record)
        ledger["totalActiveSignals"] = len([s for s in ledger["signals"] if s["status"] == "OPEN"])
        cls.save_ledger(ledger, ledger_path)
        return record

    @classmethod
    def update_forward_observations(
        cls,
        db_engine,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Harvest subsequent price candles and update forward outcomes objectively."""
        ledger = cls.load_ledger(ledger_path)
        updated_count = 0

        for sig in ledger["signals"]:
            if sig["status"] != "OPEN":
                continue

            sym = sig["symbol"]
            entry_price = sig["entryPrice"]
            stop = sig["stopLoss"]
            tp1 = sig["takeProfit1"]
            sig_date = sig["signalDate"]

            candles = db_engine.get_daily_candles(sym, limit=100)
            if not candles:
                continue

            df = pd.DataFrame(candles)
            date_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
            if not date_col:
                continue

            # Filter candles occurring AFTER the signal date
            subsequent = df[df[date_col] > sig_date].copy()
            if subsequent.empty:
                continue

            subsequent.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            sessions = len(subsequent)
            highs = subsequent["High"].values
            lows = subsequent["Low"].values
            closes = subsequent["Close"].values

            latest_close = float(closes[-1])
            mfe = float(np.max(highs) - entry_price) / entry_price * 100.0
            mae = float(np.min(lows) - entry_price) / entry_price * 100.0

            ft = sig["forwardTracking"]
            ft["sessionsObserved"] = sessions
            ft["currentPrice"] = latest_close
            ft["maxFavorableExcursionPct"] = round(mfe, 2)
            ft["maxAdverseExcursionPct"] = round(mae, 2)

            # Check Returns at milestones
            if sessions >= 5 and ft["return5d"] is None:
                ft["return5d"] = round(((closes[4] - entry_price) / entry_price) * 100.0, 2)
            if sessions >= 10 and ft["return10d"] is None:
                ft["return10d"] = round(((closes[9] - entry_price) / entry_price) * 100.0, 2)
            if sessions >= 20 and ft["return20d"] is None:
                ft["return20d"] = round(((closes[19] - entry_price) / entry_price) * 100.0, 2)

            # Check TP1 vs Stop sequence
            for idx in range(sessions):
                if highs[idx] >= tp1 and not ft["tp1Hit"]:
                    ft["tp1Hit"] = True
                    ft["tp1Session"] = idx + 1
                if lows[idx] <= stop and not ft["stopHit"]:
                    ft["stopHit"] = True
                    ft["stopSession"] = idx + 1

            # Resolve outcome
            if ft["tp1Hit"] and (not ft["stopHit"] or (ft["tp1Session"] and ft["stopSession"] and ft["tp1Session"] <= ft["stopSession"])):
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "TP1_WIN"
                ft["realizedReturnPct"] = round(((tp1 - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[ft["tp1Session"] - 1] if ft["tp1Session"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif ft["stopHit"] and (not ft["tp1Hit"] or (ft["tp1Session"] and ft["stopSession"] and ft["stopSession"] < ft["tp1Session"])):
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "STOP_LOSS"
                ft["realizedReturnPct"] = round(((stop - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[ft["stopSession"] - 1] if ft["stopSession"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif sessions >= 20:
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "TIME_EXPIRED"
                ft["realizedReturnPct"] = round(((latest_close - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[19]

            updated_count += 1

        ledger["totalActiveSignals"] = len([s for s in ledger["signals"] if s["status"] == "OPEN"])
        cls.save_ledger(ledger, ledger_path)
        return {"updatedSignals": updated_count, "openSignals": ledger["totalActiveSignals"]}

    @classmethod
    def compute_governance_scorecard(cls, ledger_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate performance against Phase 25 Model Governance criteria."""
        ledger = cls.load_ledger(ledger_path)
        signals = ledger.get("signals", [])
        if not signals:
            return {"status": "NO_SIGNALS", "n": 0}

        resolved = [s for s in signals if s["status"] == "RESOLVED"]
        open_signals = [s for s in signals if s["status"] == "OPEN"]

        n_resolved = len(resolved)
        wins = [s for s in resolved if s["forwardTracking"]["resolvedOutcome"] == "TP1_WIN"]
        stops = [s for s in resolved if s["forwardTracking"]["resolvedOutcome"] == "STOP_LOSS"]

        p_win = (len(wins) / n_resolved) if n_resolved > 0 else 0.0
        p_loss = (len(stops) / n_resolved) if n_resolved > 0 else 0.0

        avg_win = float(np.mean([w["forwardTracking"]["realizedReturnPct"] for w in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(s["forwardTracking"]["realizedReturnPct"]) for s in stops])) if stops else 0.0

        expectancy = (p_win * avg_win) - (p_loss * avg_loss)
        total_gains = sum([w["forwardTracking"]["realizedReturnPct"] for w in wins]) if wins else 0.0
        total_losses = sum([abs(s["forwardTracking"]["realizedReturnPct"]) for s in stops]) if stops else 0.0
        profit_factor = (total_gains / total_losses) if total_losses > 0 else (999.0 if total_gains > 0 else 0.0)

        # Statistical Uncertainty & Confidence Intervals (Phase 25 Governance)
        # N=100 is an evidence-building target, NOT a formal significance threshold.
        # Compute standard error of returns and 95% confidence intervals around Expectancy.
        all_realized = [s["forwardTracking"]["realizedReturnPct"] for s in resolved if s["forwardTracking"].get("realizedReturnPct") is not None]
        if len(all_realized) >= 2:
            std_err = float(np.std(all_realized, ddof=1) / np.sqrt(len(all_realized)))
            ci_95_lower = round(expectancy - 1.96 * std_err, 2)
            ci_95_upper = round(expectancy + 1.96 * std_err, 2)
            
            # Wilson score interval for binomial Win Rate
            z = 1.96
            p = p_win
            denom = 1 + (z**2 / n_resolved)
            center = (p + (z**2 / (2 * n_resolved))) / denom
            spread = (z * np.sqrt((p * (1 - p) / n_resolved) + (z**2 / (4 * n_resolved**2)))) / denom
            win_ci_lower = round(max(0.0, (center - spread) * 100.0), 1)
            win_ci_upper = round(min(100.0, (center + spread) * 100.0), 1)
        else:
            std_err = None
            ci_95_lower = None
            ci_95_upper = None
            win_ci_lower = None
            win_ci_upper = None

        return {
            "totalSignals": len(signals),
            "openSignals": len(open_signals),
            "resolvedSignals": n_resolved,
            "winRate": round(p_win * 100.0, 1),
            "winRateCI95": [win_ci_lower, win_ci_upper] if win_ci_lower is not None else None,
            "stopRate": round(p_loss * 100.0, 1),
            "avgWinPct": round(avg_win, 2),
            "avgLossPct": round(avg_loss, 2),
            "expectancyPct": round(expectancy, 2),
            "expectancyStandardError": round(std_err, 2) if std_err is not None else None,
            "expectancyCI95": [ci_95_lower, ci_95_upper] if ci_95_lower is not None else None,
            "profitFactor": round(profit_factor, 2),
            "governanceGates": {
                "expectancyPositive": bool(expectancy > 0),
                "profitFactorAbove1_5": bool(profit_factor >= 1.5),
                "stopRateBelow50": bool(p_loss < 0.50 if n_resolved > 0 else True),
                "statisticallySignificant": bool(ci_95_lower is not None and ci_95_lower > 0)
            },
            "governancePrinciple": (
                "A model change cannot be justified by a single metric moving outside its target. "
                "It requires a reproducible failure pattern across a predefined cohort and sufficient forward observations."
            )
        }
