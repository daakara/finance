"""ARX Shadow Replay Settlement Runner (Directive: ARX-LEARNING-003).

Provides bounded, idempotent non-production replay evaluation of prospective
decisions and historical recommendation setups under Contract V1.0.1.

Governing Invariants:
1. CONTRACT REUSE: Evaluates outcomes strictly via PaperTradingOutcomeEvaluator
   (ARX Paper-Trading Outcome Contract V1.0.1). Zero duplicate evaluation logic.
2. OUTCOME SEPARATION: Formally records into prospective_outcome_links:
   - SUCCESS
   - FAILURE
   - ENTRY_NOT_TRIGGERED
   - HORIZON_EXIT
   - PENDING_UNMATURED
   - UNRESOLVED_INTRABAR_SEQUENCE
   - UNRESOLVED_DATA_CONFLICT
   - OUTCOME_DATA_MISSING
3. UNRESOLVED TRUTH INTEGRITY: Unresolved observations are NEVER counted as wins or losses.
4. PERIODIC WORKER CONTAINMENT: Settlement is strictly manual/dry-run replay only.
   Automatic periodic settlement workers remain disabled per HOLD_SCAN_001.
5. IMMUTABILITY RESPECT: Decision events are never mutated; outcome links are
   relational post-hoc bindings with cryptographic identity.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd

from analyst_dashboard.governance.paper_trading_outcome_evaluator import (
    PaperTradingOutcomeEvaluator,
    EvaluationResult,
)
from analyst_dashboard.governance.prospective_capture import (
    ProspectiveDecisionCaptureEngine,
)

logger = logging.getLogger("arx.governance.shadow_replay")

# Map PaperTradingOutcomeEvaluator outcome strings to schema enum
OUTCOME_STATUS_MAP = {
    "SUCCESS": "SUCCESS",
    "FAILURE": "FAILURE",
    "ENTRY_NOT_TRIGGERED": "ENTRY_NOT_TRIGGERED",
    "HORIZON_EXIT": "HORIZON_EXIT",
    "ENTRY_WINDOW_STILL_OPEN": "PENDING_UNMATURED",
    "UNMATURED": "PENDING_UNMATURED",
    "UNRESOLVED_INTRABAR_SEQUENCE": "UNRESOLVED_INTRABAR_SEQUENCE",
    "UNRESOLVED_DATA_CONFLICT": "UNRESOLVED_DATA_CONFLICT",
    "OUTCOME_DATA_MISSING": "OUTCOME_DATA_MISSING",
}


class ShadowReplayRunner:
    """Bounded, idempotent replay settlement engine."""

    def __init__(
        self,
        capture_engine: Optional[ProspectiveDecisionCaptureEngine] = None,
        db_path: Optional[str] = None,
    ):
        self.capture_engine = capture_engine or ProspectiveDecisionCaptureEngine(db_path=db_path)
        self.db = self.capture_engine.db

    def replay_single_decision(
        self,
        decision_id: str,
        symbol: str,
        episode_id: str,
        trade_plan: Dict[str, Any],
        signal_date: str,
        daily_bars: pd.DataFrame,
        intraday_bars: Optional[pd.DataFrame] = None,
        provider: str = "POLYGON_AUTHORITATIVE",
        provider_conflict: bool = False,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Evaluates a single prospective decision against subsequent market tape."""
        c_min = (
            trade_plan.get("corridorMin")
            or trade_plan.get("corridor_min")
            or trade_plan.get("optimal_entry_min")
            or 0.0
        )
        c_max = (
            trade_plan.get("corridorMax")
            or trade_plan.get("corridor_max")
            or trade_plan.get("optimal_entry_max")
            or 0.0
        )
        stop = trade_plan.get("stopLoss") or trade_plan.get("stop_loss") or 0.0
        tp1 = (
            trade_plan.get("takeProfit1")
            or trade_plan.get("take_profit_1")
            or trade_plan.get("target_1")
            or 0.0
        )

        signal_dict = {
            "symbol": symbol,
            "signalDate": signal_date,
            "corridorMin": float(c_min),
            "corridorMax": float(c_max),
            "stopLoss": float(stop),
            "takeProfit1": float(tp1),
            "confluenceScore": float(trade_plan.get("confluenceScore", 0.0)),
            "marketRegime": trade_plan.get("marketRegime", "BULL"),
        }

        eval_result = PaperTradingOutcomeEvaluator.evaluate_signal(
            signal=signal_dict,
            daily_bars=daily_bars,
            intraday_bars=intraday_bars,
            provider=provider,
            provider_conflict=provider_conflict,
        )

        outcome_status = OUTCOME_STATUS_MAP.get(eval_result.outcome, "PENDING_UNMATURED")

        persisted_link = None
        if persist:
            trade_plan_id = trade_plan.get("tradePlanId") or trade_plan.get("plan_id")
            persisted_link = self.capture_engine.record_outcome_link(
                decision_id=decision_id,
                episode_id=episode_id,
                governing_contract_id="CONTRACT_ARX_PAPER_TRADING_V1",
                governing_contract_version="1.0.1",
                outcome_status=outcome_status,
                realized_gross_return_pct=eval_result.gross_return_pct,
                realized_net_simulated_return_pct=eval_result.net_simulated_return_pct,
                realized_r_multiple=eval_result.r_multiple,
                trade_plan_id=trade_plan_id,
            )

        return {
            "decision_id": decision_id,
            "symbol": symbol,
            "episode_id": episode_id,
            "eval_result": eval_result.to_dict(),
            "outcome_status": outcome_status,
            "persisted_link": persisted_link,
        }

    def replay_captured_decisions_batch(
        self,
        decisions: List[Dict[str, Any]],
        market_data_map: Dict[str, pd.DataFrame],
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Replays a batch of captured decisions against provided market data."""
        results = []
        counts = {
            "SUCCESS": 0,
            "FAILURE": 0,
            "ENTRY_NOT_TRIGGERED": 0,
            "HORIZON_EXIT": 0,
            "PENDING_UNMATURED": 0,
            "UNRESOLVED_INTRABAR_SEQUENCE": 0,
            "UNRESOLVED_DATA_CONFLICT": 0,
            "OUTCOME_DATA_MISSING": 0,
        }

        for dec in decisions:
            dec_id = dec["decision_id"]
            sym = dec["symbol"]
            ep_id = dec.get("episode_id", f"EP_{sym}")
            tp_json = dec.get("trade_plan_json")
            if isinstance(tp_json, str):
                try:
                    tp = json.loads(tp_json)
                except Exception:
                    tp = {}
            elif isinstance(tp_json, dict):
                tp = tp_json
            else:
                tp = {}

            sig_date = dec.get("evaluation_started_at_utc", "")[:10]
            daily_bars = market_data_map.get(sym, pd.DataFrame())

            res = self.replay_single_decision(
                decision_id=dec_id,
                symbol=sym,
                episode_id=ep_id,
                trade_plan=tp,
                signal_date=sig_date,
                daily_bars=daily_bars,
                persist=persist,
            )
            results.append(res)
            st = res["outcome_status"]
            counts[st] = counts.get(st, 0) + 1

        return {
            "total_evaluated": len(results),
            "status_counts": counts,
            "resolved_count": counts["SUCCESS"] + counts["FAILURE"] + counts["HORIZON_EXIT"] + counts["ENTRY_NOT_TRIGGERED"],
            "unresolved_count": counts["PENDING_UNMATURED"] + counts["UNRESOLVED_INTRABAR_SEQUENCE"] + counts["UNRESOLVED_DATA_CONFLICT"] + counts["OUTCOME_DATA_MISSING"],
            "results": results,
            "governance_notice": (
                "Non-production shadow replay settlement. Automatic periodic settlement worker remains disabled per HOLD_SCAN_001."
            ),
        }
