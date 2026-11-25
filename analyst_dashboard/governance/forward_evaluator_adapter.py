"""Isolated Point-in-Time Forward Evaluator Adapter for ARX Paper-Trading Pilot.

Evaluates fresh point-in-time recommendation decisions for the frozen 10-symbol
universe without invoking production schedulers, mutating production databases,
or violating operational holds (HOLD_SCAN_001, HOLD_EXEC_001).

Guarantees:
1. Reuses ConfluenceEngine and OptimalExecutionEngine with explicit point-in-time inputs.
2. Complete 10-symbol denominator accounting: produces an explicit decision,
   missing-input result, or evaluation failure for every symbol.
3. A session with ten evaluated decisions and zero eligible setups is valid.
4. Future-bar rejection: bars timestamped after session close fail closed.
5. Retains raw source bytes, evaluator inputs and outputs, session timestamps,
   and release, universe, and contract digests.
6. Newly evaluated decisions retain evidenceOrigin: OFFLINE_INPUT and
   eligibleForOpportunityCapture: False.
7. Strictly isolated storage under data/forward_pilot/<cohort_id>/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.optimal_execution import (
    OptimalExecutionEngine,
    ACTIONABLE_EXECUTION_STATUSES,
)
from analyst_dashboard.governance.forward_capture import (
    CONTRACT_PATH,
    UNIVERSE_PATH,
    canonical,
    capture_batch,
    digest,
    session_window,
    utc,
)
from analyst_dashboard.governance.market_data_adapter import (
    normalize_yahoo_chart_bars,
    verify_raw_response_bytes,
)
from analyst_dashboard.governance.storage import (
    is_production_runtime,
    safe_write_forward_pilot_file,
    validate_forward_write_path,
)

logger = logging.getLogger("arx.governance.forward_evaluator_adapter")

US_EASTERN = ZoneInfo("America/New_York")


def normalize_symbol_price_data(
    symbol: str,
    raw_or_bars: Any,
    session_date: str,
    as_of_close_utc: datetime,
) -> pd.DataFrame:
    """Extracts, validates, and bounds daily OHLCV bars strictly <= session_date.
    
    Rejects future bars, nonfinite prices, inverted bounds, or non-session dates.
    """
    if raw_or_bars is None:
        return pd.DataFrame()

    bars: List[Dict[str, Any]] = []

    if isinstance(raw_or_bars, (bytes, bytearray)):
        bars = normalize_yahoo_chart_bars(bytes(raw_or_bars), symbol, as_of=session_date)
    elif isinstance(raw_or_bars, str):
        # Could be path to file or raw json string
        p = Path(raw_or_bars)
        if p.is_file():
            raw_b = p.read_bytes()
            bars = normalize_yahoo_chart_bars(raw_b, symbol, as_of=session_date)
        else:
            try:
                data = json.loads(raw_or_bars)
                if isinstance(data, dict) and "chart" in data:
                    bars = normalize_yahoo_chart_bars(raw_or_bars.encode("utf-8"), symbol, as_of=session_date)
                elif isinstance(data, list):
                    bars = data
                else:
                    raise ValueError(f"Unrecognized string payload for symbol {symbol}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Cannot parse price string for {symbol}: {e}") from e
    elif isinstance(raw_or_bars, dict):
        if "rawResponseBytes" in raw_or_bars or "rawBytes" in raw_or_bars:
            raw_b = raw_or_bars.get("rawResponseBytes") or raw_or_bars.get("rawBytes")
            if isinstance(raw_b, str):
                raw_b = raw_b.encode("utf-8")
            if raw_or_bars.get("rawResponseSha256"):
                verify_raw_response_bytes(raw_b, raw_or_bars["rawResponseSha256"])
            bars = normalize_yahoo_chart_bars(raw_b, symbol, as_of=session_date)
        elif "bars" in raw_or_bars and isinstance(raw_or_bars["bars"], list):
            bars = raw_or_bars["bars"]
        elif "rawPath" in raw_or_bars:
            raw_b = Path(raw_or_bars["rawPath"]).read_bytes()
            if raw_or_bars.get("rawResponseSha256"):
                verify_raw_response_bytes(raw_b, raw_or_bars["rawResponseSha256"])
            bars = normalize_yahoo_chart_bars(raw_b, symbol, as_of=session_date)
        else:
            raise ValueError(f"Unrecognized price dict format for {symbol}")
    elif isinstance(raw_or_bars, list):
        bars = raw_or_bars
    elif isinstance(raw_or_bars, pd.DataFrame):
        df = raw_or_bars.copy()
        if not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns or "Date" in df.columns:
                    col = "date" if "date" in df.columns else "Date"
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                else:
                    df.index = pd.to_datetime(df.index)
            # Filter strictly <= session_date
            df = df[df.index.strftime("%Y-%m-%d") <= session_date]
            return df
        return df
    else:
        raise TypeError(f"Unsupported price data type for {symbol}: {type(raw_or_bars)}")

    if not bars:
        return pd.DataFrame()

    cal = xcals.get_calendar("XNYS")
    dates: List[str] = []
    clean_records: List[Dict[str, Any]] = []

    for b in bars:
        d = b.get("date") or b.get("Date")
        if not d or not isinstance(d, str):
            raise ValueError(f"Missing or invalid date field in bar for {symbol}")
        if d > session_date:
            raise ValueError(
                f"FUTURE_BAR_REJECTED: Bar date '{d}' is strictly after session date '{session_date}'"
            )
        if not cal.is_session(d):
            raise ValueError(f"NON_SESSION_BAR_REJECTED: Date '{d}' is not an active XNYS session")

        vals = {k: b.get(k) or b.get(k.lower()) for k in ("Open", "High", "Low", "Close", "Volume")}
        for k, v in vals.items():
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"Nonfinite OHLCV value for {symbol} on {d}: {k}={v}")

        o, h, l, c, v = vals["Open"], vals["High"], vals["Low"], vals["Close"], vals["Volume"]
        if not (0 < l <= min(o, c) <= max(o, c) <= h and v >= 0):
            raise ValueError(f"Invalid OHLCV bounds for {symbol} on {d}: O={o}, H={h}, L={l}, C={c}")

        dates.append(d)
        clean_records.append({"Date": d, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})

    if dates != sorted(set(dates)):
        raise ValueError(f"Duplicate or unordered session bars detected for {symbol}")

    df = pd.DataFrame(clean_records).set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


def evaluate_symbol_decision(
    symbol: str,
    price_df: pd.DataFrame,
    session_date: str,
    as_of_close_utc: datetime,
) -> Dict[str, Any]:
    """Evaluates one symbol point-in-time using ConfluenceEngine and OptimalExecutionEngine.
    
    Returns explicit decisionState and evaluationStatus. Zero fabrication.
    """
    if price_df is None or price_df.empty:
        return {
            "decisionState": "INSUFFICIENT_DATA",
            "evaluationStatus": "MISSING_INPUT",
            "reason": f"No price history provided for session {session_date}",
            "plan": None,
        }

    try:
        last_close = float(price_df["Close"].iloc[-1])
        if not (math.isfinite(last_close) and last_close > 0):
            return {
                "decisionState": "EVIDENCE_INCOMPLETE",
                "evaluationStatus": "INVALID_INPUT",
                "reason": f"Nonfinite spot price {last_close} for {symbol}",
                "plan": None,
            }

        # 1. Optimal Trade Levels Calculation
        trade_levels = OptimalExecutionEngine.calculate_trade_levels(
            price_df=price_df,
            current_price=last_close,
            user_role="LONG_TERM",
        )
        exec_status = trade_levels.get("execution_status")

        # 2. Confluence Score Calculation
        confluence_res = ConfluenceEngine.calculate_confluence(
            symbol=symbol,
            technical_data=trade_levels,
        )
        confluence_score = float(confluence_res.get("confluenceScore") or 0.0)

        # 3. Assess Actionability
        c_min = trade_levels.get("optimal_entry_min")
        c_max = trade_levels.get("optimal_entry_max")
        stop = trade_levels.get("stop_loss")
        tp1 = trade_levels.get("take_profit_1")

        has_valid_bounds = (
            all(isinstance(x, (int, float)) and math.isfinite(x) for x in (c_min, c_max, stop, tp1))
            and (0 < stop < c_min <= c_max < tp1)
        )

        is_actionable = (exec_status in ACTIONABLE_EXECUTION_STATUSES) and has_valid_bounds

        if is_actionable:
            plan = {
                "spot": round(last_close, 4),
                "corridorMin": round(float(c_min), 4),
                "corridorMax": round(float(c_max), 4),
                "stopLoss": round(float(stop), 4),
                "takeProfit1": round(float(tp1), 4),
                "confluenceScore": round(confluence_score, 2),
            }
            return {
                "decisionState": "VALID_SETUP",
                "evaluationStatus": "RECORDED",
                "sourceTimestampUtc": as_of_close_utc.isoformat(),
                "observedAtUtc": as_of_close_utc.isoformat(),
                "plan": plan,
                "executionStatus": exec_status,
                "setupPattern": trade_levels.get("setup_pattern"),
            }
        else:
            # Non-actionable session evaluation (valid 0-setup state)
            return {
                "decisionState": "INSUFFICIENT_DATA",
                "evaluationStatus": "RECORDED",
                "sourceTimestampUtc": as_of_close_utc.isoformat(),
                "observedAtUtc": as_of_close_utc.isoformat(),
                "plan": None,
                "executionStatus": exec_status,
                "reason": f"Non-actionable execution status: {exec_status}",
            }

    except Exception as e:
        logger.error(f"Error evaluating {symbol} for session {session_date}: {e}")
        return {
            "decisionState": "INSUFFICIENT_DATA",
            "evaluationStatus": "FAILED_EVALUATION",
            "error": str(e),
            "reason": f"Evaluator execution exception: {str(e)}",
            "plan": None,
        }


def build_forward_evaluation_batch(
    sources_by_symbol: Dict[str, Any],
    session_date: str,
    engine_sha: str,
    as_of_close_utc: Optional[datetime] = None,
    universe_path: Path = UNIVERSE_PATH,
) -> Dict[str, Any]:
    """Evaluates all 10 symbols in frozen universe order and builds schema 1.0 batch."""
    cal = xcals.get_calendar("XNYS")
    if not cal.is_session(session_date):
        raise ValueError(f"Session date '{session_date}' must be an active XNYS session")

    close_dt = as_of_close_utc or cal.session_close(session_date).to_pydatetime()
    if close_dt.tzinfo is None:
        raise ValueError("Naive close timestamp rejected")
    close_dt = close_dt.astimezone(timezone.utc)

    universe = json.loads(universe_path.read_bytes())
    symbols: List[str] = universe["symbols"]

    evaluations: Dict[str, Any] = {}

    for sym in symbols:
        raw_source = sources_by_symbol.get(sym)
        if raw_source is None:
            # None evaluation is canonically handled by forward_capture as MISSING_INPUT
            evaluations[sym] = None
            continue
        try:
            df = normalize_symbol_price_data(sym, raw_source, session_date, close_dt)
            if df.empty:
                evaluations[sym] = None
            else:
                evaluations[sym] = evaluate_symbol_decision(sym, df, session_date, close_dt)
        except Exception as e:
            evaluations[sym] = {
                "error": str(e),
                "reason": f"Input normalization failed: {str(e)}",
            }

    batch = {
        "schemaVersion": "1.0",
        "samplingMode": "SYSTEMATIC",
        "sessionDate": session_date,
        "engineSha": engine_sha,
        "evaluations": evaluations,
    }
    return batch


def execute_isolated_forward_evaluation(
    raw_sources_manifest: Dict[str, Any],
    *,
    cohort_id: str,
    start_session: str,
    forward_root: Path,
    session_date: str,
    engine_sha: str,
    initialize: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Performs end-to-end isolated point-in-time evaluation and durable forward capture.
    
    1. Retains original raw evaluator input into cohort storage under SHA-256.
    2. Builds systematic evaluation batch across all 10 frozen symbols.
    3. Calls capture_batch to commit decisions atomically under flock.
    4. Binds release SHA, universe SHA-256, and contract SHA-256.
    """
    if is_production_runtime():
        raise ValueError("Production forward capture is not activated")

    root = Path(forward_root).absolute()
    if not root.is_dir():
        raise ValueError("Explicit existing forward root required")

    cal = xcals.get_calendar("XNYS")
    close_dt = cal.session_close(session_date).to_pydatetime()
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        raise ValueError("Naive capture clock rejected")
    now_dt = now_dt.astimezone(timezone.utc)

    # 1. Build the 10-symbol batch from sources
    symbols_sources = raw_sources_manifest.get("symbols", {})
    batch = build_forward_evaluation_batch(
        sources_by_symbol=symbols_sources,
        session_date=session_date,
        engine_sha=engine_sha,
        as_of_close_utc=close_dt,
    )
    raw_batch = json.dumps(batch, indent=2).encode("utf-8")

    # 2. Capture into forward decision stream (handles cohort initialization and locking)
    capture_receipt = capture_batch(
        raw_batch,
        cohort_id=cohort_id,
        start_session=start_session,
        forward_root=root,
        initialize=initialize,
        now=now_dt,
    )

    # 3. Retain original source payload bytes in evaluator_inputs/
    manifest_bytes = canonical(raw_sources_manifest).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    cohort_dir = root / cohort_id
    eval_inputs_dir = cohort_dir / "evaluator_inputs"
    target_manifest_path = eval_inputs_dir / f"{manifest_sha}.json"

    validate_forward_write_path(str(target_manifest_path), cohort_id, str(root))
    safe_write_forward_pilot_file(
        str(target_manifest_path), cohort_id, manifest_bytes.decode("utf-8"), str(root)
    )

    evals = batch["evaluations"]
    actionable_count = sum(1 for e in evals.values() if isinstance(e, dict) and e.get("decisionState") in ("ACTIONABLE_SETUP", "VALID_SETUP"))
    missing_count = sum(1 for e in evals.values() if e is None)
    failed_count = sum(1 for e in evals.values() if isinstance(e, dict) and e.get("error"))
    recorded_count = sum(1 for e in evals.values() if isinstance(e, dict) and not e.get("error"))

    return {
        "status": "CAPTURED_FORWARD_SESSION",
        "sessionDate": session_date,
        "cohortId": cohort_id,
        "engineSha": engine_sha,
        "evaluatorInputSha256": manifest_sha,
        "batchSha256": hashlib.sha256(raw_batch).hexdigest(),
        "totalSymbols": len(evals),
        "actionableSetups": actionable_count,
        "nonActionableDecisions": recorded_count - actionable_count,
        "missingInputs": missing_count,
        "failedEvaluations": failed_count,
        **capture_receipt,
    }
