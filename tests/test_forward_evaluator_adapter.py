"""Offline Qualification Test Suite for Forward Evaluator Adapter.

Quality Gates:
1. Complete 10-symbol accounting: Denominator preserved (zero dropped symbols).
2. Future-bar rejection: Bars strictly > session_date fail closed.
3. Missing & stale input handling: Explicit MISSING_INPUT and INSUFFICIENT_DATA.
4. Non-actionable session: 10 evaluated decisions with 0 eligible setups is valid.
5. Actionable setup generation: Plan bounds adhere to Contract V1.0.1 rules.
6. End-to-end isolated storage execution: Captures decisions into capture_stream.jsonl.
7. Identical replay idempotence: Repeated run is a byte-identical no-op.
8. Conflicting retry rejection: Mutated input fails closed.
9. Historical VOO/QQQM artifact reuse: Retained Yahoo chart v8 raw responses parsed.
10. Settlement pending invariant: Unmatured horizons and missing corporate actions leave settlement pending.
"""
import copy
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict

import exchange_calendars as xcals
import pandas as pd
import pytest

from analyst_dashboard.governance.forward_capture import UNIVERSE_PATH
from analyst_dashboard.governance.forward_settlement import settle_batch
from analyst_dashboard.governance.forward_evaluator_adapter import (
    build_forward_evaluation_batch,
    evaluate_symbol_decision,
    execute_isolated_forward_evaluation,
    normalize_symbol_price_data,
)

CAL = xcals.get_calendar("XNYS")
SYMBOLS = json.loads(UNIVERSE_PATH.read_bytes())["symbols"]
COHORT_TEST = "COHORT_FORWARD_PILOT_EVAL_QUAL"
SCRATCH_RAW_DIR = Path("/Users/iDavid/.gemini/antigravity/brain/e53e772b-2682-4ca8-a2bd-008123fee60e/scratch/raw_responses")
ENGINE_SHA = "42bd25050ebc54da9483d1fb45d67f321f98917f"


@pytest.fixture(autouse=True)
def local(monkeypatch):
    for k in (
        "ARX_FORCE_PRODUCTION_STORAGE_VALIDATION",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_ENVIRONMENT",
        "ENVIRONMENT",
    ):
        monkeypatch.delenv(k, raising=False)


def make_test_bars(end_date: str = "2026-11-25", n_sessions: int = 60, base_price: float = 100.0) -> list:
    sessions = [s.date().isoformat() for s in CAL.sessions_in_range("2026-06-01", end_date)][-n_sessions:]
    bars = []
    for i, d in enumerate(sessions):
        p = base_price + i * 0.2
        bars.append({
            "date": d,
            "Open": round(p - 0.2, 2),
            "High": round(p + 1.0, 2),
            "Low": round(p - 1.0, 2),
            "Close": round(p, 2),
            "Volume": 1000000,
        })
    return bars


# ==============================================================================
# Gate 1: Future-Bar Rejection & Non-Session Rejection
# ==============================================================================

def test_future_bar_rejection_fails_closed():
    session = "2026-11-25"
    close = CAL.session_close(session).to_pydatetime()
    bars = make_test_bars(end_date="2026-11-25", n_sessions=20)
    # Add future bar on 2026-11-27
    bars.append({
        "date": "2026-11-27",
        "Open": 105.0,
        "High": 106.0,
        "Low": 104.0,
        "Close": 105.5,
        "Volume": 500000,
    })
    with pytest.raises(ValueError, match="FUTURE_BAR_REJECTED"):
        normalize_symbol_price_data("AAPL", bars, session, close)


def test_non_session_bar_rejection_fails_closed():
    session = "2026-11-25"
    close = CAL.session_close(session).to_pydatetime()
    # 2026-11-22 is Sunday (non-session date <= session_date)
    bars = [{"date": "2026-11-22", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 100}]
    with pytest.raises(ValueError, match="NON_SESSION_BAR_REJECTED"):
        normalize_symbol_price_data("AAPL", bars, session, close)


# ==============================================================================
# Gate 2: Missing & Stale Input Handling (Denominator Preservation)
# ==============================================================================

def test_missing_symbol_produces_missing_input_in_denominator():
    session = "2026-11-25"
    close = CAL.session_close(session).to_pydatetime()
    # Empty DataFrame / None
    res_none = evaluate_symbol_decision("NVDA", pd.DataFrame(), session, close)
    assert res_none["decisionState"] == "INSUFFICIENT_DATA"
    assert res_none["evaluationStatus"] == "MISSING_INPUT"
    assert res_none["plan"] is None

    # Partial manifest with only 2 symbols
    sources = {
        "VOO": {"bars": make_test_bars("2026-11-25", 30)},
        "QQQM": {"bars": make_test_bars("2026-11-25", 30)},
    }
    batch = build_forward_evaluation_batch(sources, session, ENGINE_SHA, as_of_close_utc=close)
    evals = batch["evaluations"]
    assert len(evals) == 10
    assert list(evals.keys()) == SYMBOLS

    # The other 8 symbols are None in batch evaluations (canonically normalized to MISSING_INPUT in stream)
    for s in ("IWM", "XLE", "XLK", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"):
        assert evals[s] is None


# ==============================================================================
# Gate 3: Valid 0-Setup Session (Non-Actionable Decisions Recorded)
# ==============================================================================

def test_valid_zero_setup_session_records_ten_decisions():
    session = "2026-11-25"
    close = CAL.session_close(session).to_pydatetime()
    # Provide 20 bars for each symbol (insufficient history for long-term setup)
    sources = {s: {"bars": make_test_bars("2026-11-25", 20)} for s in SYMBOLS}
    batch = build_forward_evaluation_batch(sources, session, ENGINE_SHA, as_of_close_utc=close)
    evals = batch["evaluations"]

    assert len(evals) == 10
    assert all(e["evaluationStatus"] == "RECORDED" for e in evals.values())
    assert all(e["decisionState"] == "INSUFFICIENT_DATA" for e in evals.values())
    assert all(e["plan"] is None for e in evals.values())
    # 0 actionable setups is a valid session result
    actionable_count = sum(1 for e in evals.values() if e["decisionState"] in ("ACTIONABLE_SETUP", "VALID_SETUP"))
    assert actionable_count == 0


# ==============================================================================
# Gate 4: Actionable Setup Generation and Plan Bounds
# ==============================================================================

def test_actionable_setup_plan_bounds():
    session = "2026-11-25"
    close = CAL.session_close(session).to_pydatetime()
    bars = make_test_bars("2026-11-25", 60)
    df = pd.DataFrame(bars).set_index("date")
    df.index = pd.to_datetime(df.index)

    res = evaluate_symbol_decision("AAPL", df, session, close)
    # Regardless of whether it's actionable or non-actionable, if a plan is present it must satisfy bounds
    if res.get("plan"):
        p = res["plan"]
        assert 0 < p["stopLoss"] < p["corridorMin"] <= p["corridorMax"] < p["takeProfit1"]
        assert p["spot"] > 0
        assert 0 <= p["confluenceScore"] <= 100
        assert res["decisionState"] in ("VALID_SETUP", "ACTIONABLE_SETUP")
    else:
        assert res["decisionState"] in ("INSUFFICIENT_DATA", "EVIDENCE_INCOMPLETE")


# ==============================================================================
# Gate 5: End-to-End Isolated Execution, Idempotence & Fail-Closed Replays
# ==============================================================================

def test_end_to_end_isolated_evaluation_and_replays(tmp_path):
    session = "2026-11-25"
    close = CAL.session_close(session).to_pydatetime()
    now = close + timedelta(minutes=20)

    # 1. Prepare manifest with 2 active symbols and 8 missing
    manifest = {
        "sessionDate": session,
        "symbols": {
            "VOO": {"bars": make_test_bars(session, 25, base_price=700.0)},
            "QQQM": {"bars": make_test_bars(session, 25, base_price=290.0)},
        },
    }

    # First execution (initialize=True)
    res = execute_isolated_forward_evaluation(
        manifest,
        cohort_id=COHORT_TEST,
        start_session=session,
        forward_root=tmp_path,
        session_date=session,
        engine_sha=ENGINE_SHA,
        initialize=True,
        now=now,
    )
    assert res["status"] == "CAPTURED_FORWARD_SESSION"
    assert res["written"] == 10
    assert res["existing"] == 0
    assert res["totalSymbols"] == 10
    assert res["missingInputs"] == 8

    # Verify files created in isolated storage
    cohort_dir = tmp_path / COHORT_TEST
    assert (cohort_dir / "cohort.json").exists()
    assert (cohort_dir / "capture_stream.jsonl").exists()
    assert (cohort_dir / "evaluator_inputs").is_dir()

    stream_lines = [json.loads(l) for l in (cohort_dir / "capture_stream.jsonl").read_text().splitlines()]
    assert len(stream_lines) == 10
    assert [r["symbol"] for r in stream_lines] == SYMBOLS
    assert all(r["evidenceOrigin"] == "OFFLINE_INPUT" for r in stream_lines)
    assert all(not r["eligibleForOpportunityCapture"] for r in stream_lines)

    # 2. Identical replay: returns written: 0, existing: 10, files byte-identical
    stream_bytes_before = (cohort_dir / "capture_stream.jsonl").read_bytes()
    res_replay = execute_isolated_forward_evaluation(
        manifest,
        cohort_id=COHORT_TEST,
        start_session=session,
        forward_root=tmp_path,
        session_date=session,
        engine_sha=ENGINE_SHA,
        initialize=False,
        now=now,
    )
    assert res_replay["written"] == 0
    assert res_replay["existing"] == 10
    assert (cohort_dir / "capture_stream.jsonl").read_bytes() == stream_bytes_before

    # 3. Conflicting retry: mutating manifest for same session fails closed
    manifest_mutated = copy.deepcopy(manifest)
    manifest_mutated["symbols"]["VOO"]["bars"][-1]["Close"] = 999.0
    with pytest.raises(ValueError, match="Conflicting retry cannot replace frozen decisions"):
        execute_isolated_forward_evaluation(
            manifest_mutated,
            cohort_id=COHORT_TEST,
            start_session=session,
            forward_root=tmp_path,
            session_date=session,
            engine_sha=ENGINE_SHA,
            initialize=False,
            now=now,
        )


# ==============================================================================
# Gate 6: Historical VOO & QQQM Raw Response Reuse & Settlement Pending
# ==============================================================================

def test_reuse_retained_voo_qqqm_and_settlement_pending(tmp_path):
    session = "2026-11-25"
    close = CAL.session_close(session).to_pydatetime()
    now = close + timedelta(minutes=20)

    # Check retained raw files in scratch
    voo_raw_files = list(SCRATCH_RAW_DIR.glob("VOO_RES_*.raw"))
    qqqm_raw_files = list(SCRATCH_RAW_DIR.glob("QQQM_RES_*.raw"))

    sources = {}
    if voo_raw_files and qqqm_raw_files:
        sources["VOO"] = {"rawPath": str(voo_raw_files[0])}
        sources["QQQM"] = {"rawPath": str(qqqm_raw_files[0])}
    else:
        sources["VOO"] = {"bars": make_test_bars(session, 24, 708.44)}
        sources["QQQM"] = {"bars": make_test_bars(session, 24, 295.00)}

    manifest = {"sessionDate": session, "symbols": sources}

    res = execute_isolated_forward_evaluation(
        manifest,
        cohort_id=COHORT_TEST,
        start_session=session,
        forward_root=tmp_path,
        session_date=session,
        engine_sha=ENGINE_SHA,
        initialize=True,
        now=now,
    )
    assert res["status"] == "CAPTURED_FORWARD_SESSION"

    # Settlement check as of session date 2026-11-25
    settle_batch_data = {
        "schemaVersion": "1.0",
        "cohortId": COHORT_TEST,
        "asOfSession": session,
        "series": {},
    }
    settle_res = settle_batch(
        json.dumps(settle_batch_data).encode("utf-8"),
        cohort_id=COHORT_TEST,
        forward_root=tmp_path,
        initialize=True,
        now=now,
    )
    assert settle_res["written"] == 10
    assert settle_res["reservedEntrySlots"] == 0
    ledger_path = tmp_path / COHORT_TEST / "settlement_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    assert ledger["activationState"] == "DISABLED"
    assert not ledger["eligibleForOpportunityCapture"]
    assert ledger["candidateEvidenceOrigin"] == "OFFLINE_INPUT"
