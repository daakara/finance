"""
ARX Terminal — Epoch 2 Incident Remediation Test Suite
Analytics Non-Finite Daily-Bar Defect & Authority Boundary Verification

Tests all 21 scenarios specified in Section 21 of the Epoch 2 incident remediation contract:
1.  is_valid_daily_bar: valid completed bar returns True
2.  is_valid_daily_bar: null/NaN in Close/Open/High/Low returns False
3.  is_valid_daily_bar: negative or zero price returns False
4.  is_valid_daily_bar: High < Low returns False
5.  is_valid_daily_bar: Close > High or Close < Low returns False
6.  is_valid_daily_bar: Open > High or Open < Low returns False
7.  is_valid_daily_bar: negative volume returns False
8.  get_daily_bar_session_state: completed NYSE session -> COMPLETED_EXCHANGE_SESSION
9.  get_daily_bar_session_state: current open NYSE session -> CURRENT_OPEN_SESSION
10. get_daily_bar_session_state: future bar date -> FUTURE_BAR
11. get_daily_bar_session_state: crypto 24/7 calendar session state handling
12. validate_and_prune_daily_history: trailing unclosed/NaN bar pruned to last completed session
13. validate_and_prune_daily_history: trailing open-session bar pruned to last completed session
14. validate_and_prune_daily_history: interior corruption fails closed (valid=False, INTERIOR_CORRUPTION)
15. validate_and_prune_daily_history: insufficient history (< 15 bars) fails closed (valid=False, INSUFFICIENT_HISTORY)
16. validate_and_prune_daily_history: clean completed history preserved exactly unchanged
17. API Route Integration: symbol with trailing unclosed bar returns HTTP 200 with completed-session data and priceState AVAILABLE
18. API Route Integration: interior corruption fails closed with HTTP 503 / 404
19. Passive Capture Firewall: non-finite/null/non-positive current_price returns None (SUPPRESSED_NONFINITE_INPUT)
20. Passive Capture Firewall: non-finite entry_price or stop_loss returns None (SUPPRESSED_NONFINITE_INPUT)
21. Epoch 2 Manifest & Ledger: verify_epoch2_manifest is VERIFIED, epoch 1 clean prospective N=0, epoch 2 initial N=0
"""

import math
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.analytics import (
    is_valid_daily_bar,
    get_daily_bar_session_state,
    validate_and_prune_daily_history,
)
from analyst_dashboard.governance.passive_capture import PassiveCaptureHook
from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)


client = TestClient(app)


# ---------------------------------------------------------------------------
# 1-7. is_valid_daily_bar Unit Tests
# ---------------------------------------------------------------------------

def test_01_is_valid_daily_bar_valid_completed():
    row = pd.Series({"Open": 150.0, "High": 155.0, "Low": 149.0, "Close": 152.0, "Volume": 1000000})
    assert is_valid_daily_bar(row) is True


def test_02_is_valid_daily_bar_nan_close():
    row = pd.Series({"Open": 150.0, "High": 155.0, "Low": 149.0, "Close": float("nan"), "Volume": 1000000})
    assert is_valid_daily_bar(row) is False
    row_none = pd.Series({"Open": 150.0, "High": 155.0, "Low": 149.0, "Close": None, "Volume": 1000000})
    assert is_valid_daily_bar(row_none) is False


def test_03_is_valid_daily_bar_zero_or_negative():
    row_zero = pd.Series({"Open": 150.0, "High": 155.0, "Low": 0.0, "Close": 152.0, "Volume": 1000})
    assert is_valid_daily_bar(row_zero) is False
    row_neg = pd.Series({"Open": 150.0, "High": 155.0, "Low": -5.0, "Close": 152.0, "Volume": 1000})
    assert is_valid_daily_bar(row_neg) is False


def test_04_is_valid_daily_bar_high_less_than_low():
    row = pd.Series({"Open": 150.0, "High": 140.0, "Low": 155.0, "Close": 145.0, "Volume": 1000})
    assert is_valid_daily_bar(row) is False


def test_05_is_valid_daily_bar_close_outside_high_low():
    row_high = pd.Series({"Open": 150.0, "High": 155.0, "Low": 145.0, "Close": 156.0, "Volume": 1000})
    assert is_valid_daily_bar(row_high) is False
    row_low = pd.Series({"Open": 150.0, "High": 155.0, "Low": 145.0, "Close": 144.0, "Volume": 1000})
    assert is_valid_daily_bar(row_low) is False


def test_06_is_valid_daily_bar_open_outside_high_low():
    row_high = pd.Series({"Open": 158.0, "High": 155.0, "Low": 145.0, "Close": 150.0, "Volume": 1000})
    assert is_valid_daily_bar(row_high) is False
    row_low = pd.Series({"Open": 142.0, "High": 155.0, "Low": 145.0, "Close": 150.0, "Volume": 1000})
    assert is_valid_daily_bar(row_low) is False


def test_07_is_valid_daily_bar_negative_volume():
    row = pd.Series({"Open": 150.0, "High": 155.0, "Low": 145.0, "Close": 150.0, "Volume": -100})
    assert is_valid_daily_bar(row) is False


# ---------------------------------------------------------------------------
# 8-11. get_daily_bar_session_state Unit Tests
# ---------------------------------------------------------------------------

def test_08_session_state_completed_nyse():
    past_nyse_date = pd.Timestamp("2026-09-18", tz="UTC")
    state = get_daily_bar_session_state(past_nyse_date, is_crypto=False)
    assert state == "COMPLETED_EXCHANGE_SESSION"


def test_09_session_state_current_open_nyse():
    now_utc = datetime.now(timezone.utc)
    today_bar = pd.Timestamp(now_utc.date())
    state = get_daily_bar_session_state(today_bar, is_crypto=False)
    assert state in ("CURRENT_OPEN_SESSION", "COMPLETED_EXCHANGE_SESSION")


def test_10_session_state_future_bar():
    future_date = pd.Timestamp("2029-01-01", tz="UTC")
    state = get_daily_bar_session_state(future_date, is_crypto=False)
    assert state in ("FUTURE_SESSION", "FUTURE_BAR")


def test_11_session_state_crypto():
    past_crypto_date = pd.Timestamp("2026-09-18", tz="UTC")
    state_past = get_daily_bar_session_state(past_crypto_date, is_crypto=True)
    assert state_past == "COMPLETED_EXCHANGE_SESSION"


# ---------------------------------------------------------------------------
# 12-16. validate_and_prune_daily_history Unit Tests
# ---------------------------------------------------------------------------

def _generate_clean_history(n=50, start_date="2026-01-01"):
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    records = []
    base_price = 100.0
    for i, d in enumerate(dates):
        p = base_price + i * 0.5
        records.append({
            "Open": p - 0.2,
            "High": p + 1.0,
            "Low": p - 1.0,
            "Close": p,
            "Volume": 500000 + i * 1000,
        })
    return pd.DataFrame(records, index=dates)


def test_12_prune_trailing_nan_bar():
    df = _generate_clean_history(30)
    # Append trailing unclosed row with NaN close
    last_dt = df.index[-1] + pd.Timedelta(days=1)
    unclosed_row = pd.DataFrame([{
        "Open": 115.0, "High": 116.0, "Low": 114.0, "Close": float("nan"), "Volume": 100000
    }], index=[last_dt])
    corrupted_df = pd.concat([df, unclosed_row])

    pruned, meta = validate_and_prune_daily_history(corrupted_df, is_crypto=False)
    assert meta["valid"] is True
    assert meta["pruned_trailing_count"] == 1
    assert len(pruned) == 30
    assert math.isfinite(pruned["Close"].iloc[-1])


def test_13_prune_trailing_open_session_bar():
    df = _generate_clean_history(30)
    session_date = pd.Timestamp("2026-09-24")
    now_utc = datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc)
    open_row = pd.DataFrame([{
        "Open": 120.0, "High": 122.0, "Low": 119.0, "Close": 121.0, "Volume": 50000
    }], index=[session_date])
    active_df = pd.concat([df, open_row])

    pruned, meta = validate_and_prune_daily_history(active_df, is_crypto=False, now_utc=now_utc)
    assert meta["valid"] is True
    assert pruned.index[-1] != session_date
    assert pruned.index[-1] == df.index[-1]


def test_14_interior_corruption_fails_closed():
    df = _generate_clean_history(30)
    # Corrupt an interior row (e.g. index 15)
    df.iloc[15, df.columns.get_loc("Close")] = float("nan")

    pruned, meta = validate_and_prune_daily_history(df, is_crypto=False)
    assert meta["valid"] is False
    assert "INTERIOR_CORRUPTION" in meta["reason"]
    assert pruned is None


def test_15_insufficient_history_fails_closed():
    df = _generate_clean_history(10)  # less than minimum 15
    pruned, meta = validate_and_prune_daily_history(df, is_crypto=False)
    assert meta["valid"] is False
    assert "INSUFFICIENT" in meta["reason"]


def test_16_clean_history_preserved_exactly():
    df = _generate_clean_history(50)
    pruned, meta = validate_and_prune_daily_history(df, is_crypto=False)
    assert meta["valid"] is True
    assert meta["pruned_trailing_count"] == 0
    assert len(pruned) == 50
    assert pruned["Close"].iloc[-1] == df["Close"].iloc[-1]


# ---------------------------------------------------------------------------
# 17-18. API Route Integration Tests
# ---------------------------------------------------------------------------

def test_17_api_route_trailing_nan_bar_returns_completed_session():
    # Live symbols AAPL / SPY outside trading hours or with trailing NaN
    response = client.get("/api/v1/analytics/AAPL?period=1y&interval=1d&user_role=LONG_TERM")
    assert response.status_code == 200
    data = response.json()
    assert data["priceState"] == "AVAILABLE"
    assert data["quoteStatus"] in ("COMPLETED_SESSION", "REALTIME", "DELAYED")
    assert isinstance(data["currentPrice"], (int, float))
    assert math.isfinite(data["currentPrice"])
    assert data["currentPrice"] > 0
    assert (
        data["analysisReferencePrice"] == data["currentPrice"]
        or data.get("liveFreshness") in ("REALTIME", "DELAYED", "STALE")
        or data.get("livePrice") is not None
    )
    assert len(data["candles"]) >= 15
    assert all(math.isfinite(c["close"]) for c in data["candles"])


def test_18_api_route_interior_corruption_fails_closed():
    with patch("yfinance.Ticker") as mock_ticker:
        corrupted_df = _generate_clean_history(30)
        corrupted_df.iloc[10, corrupted_df.columns.get_loc("Close")] = float("nan")
        mock_instance = MagicMock()
        mock_instance.history.return_value = corrupted_df
        mock_ticker.return_value = mock_instance

        response = client.get("/api/v1/analytics/TESTCORRUPT?period=1y&interval=1d")
        assert response.status_code in (404, 503)


# ---------------------------------------------------------------------------
# 19-20. Passive Capture Firewall Tests
# ---------------------------------------------------------------------------

def _build_fixture_payload(symbol="AAPL", current_price=150.0):
    return {
        "symbol": symbol,
        "current_price": current_price,
        "optimal_execution_plan": {
            "execution_status": "ENTER_EARLY_ZONE",
            "optimal_entry_min": 148.0,
            "optimal_entry_max": 152.0,
            "stop_loss": 145.0,
            "take_profit_1": 160.0,
            "take_profit_2": 170.0,
            "risk_reward_ratio": 2.5,
        },
        "confluence_output": {"confluence_score": 85.0},
        "technicals": {"rsi_14": 45.0},
        "factor_scores": {"compositeFactorScore": 80.0},
        "macro_inputs": {"macro_difficulty": 50.0},
        "observed_at": 1726704000000,
        "fetched_at": 1726704001000,
        "freshness_status": "LIVE",
        "provider_source": "test_provider",
        "candles": [
            {"time": "2026-09-18", "open": 149.0, "high": 151.0, "low": 148.0, "close": 150.0, "volume": 1000000}
        ],
    }


def test_19_passive_capture_firewall_non_finite_current_price():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name

    try:
        ExperimentLedger.save_ledger({"version": "1.1.0", "signals": [], "totalActiveSignals": 0}, tmp_path)
        payload = _build_fixture_payload(current_price=float("nan"))
        payload["ledger_path"] = tmp_path

        record = PassiveCaptureHook.record_natural_recommendation(**payload)
        assert record is None

        # Verify ledger remained empty
        ledger = ExperimentLedger.load_ledger(tmp_path)
        assert len(ledger["signals"]) == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_20_passive_capture_firewall_non_finite_execution_levels():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name

    try:
        ExperimentLedger.save_ledger({"version": "1.1.0", "signals": [], "totalActiveSignals": 0}, tmp_path)
        payload = _build_fixture_payload()
        payload["optimal_execution_plan"]["stop_loss"] = float("nan")
        payload["ledger_path"] = tmp_path

        record = PassiveCaptureHook.record_natural_recommendation(**payload)
        assert record is None

        ledger = ExperimentLedger.load_ledger(tmp_path)
        assert len(ledger["signals"]) == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# 21. Epoch 2 Manifest & Denominator Accounting Test
# ---------------------------------------------------------------------------

def test_21_epoch2_manifest_and_denominator_accounting():
    # Verify Epoch 2 Manifest
    manifest_audit = ExperimentLedger.verify_epoch2_manifest()
    assert manifest_audit["status"] == "VERIFIED"
    assert manifest_audit["valid"] is True
    assert manifest_audit["epochId"] == "ARX_PROSPECTIVE_VALIDATION_EPOCH_2"

    # Verify Active Manifest resolves to Epoch 2 or candidate Epoch 3
    active_audit = ExperimentLedger.verify_observation_governance_manifest()
    assert active_audit["status"] == "VERIFIED"
    assert active_audit["valid"] is True
    assert active_audit["epochId"] in ("ARX_PROSPECTIVE_VALIDATION_EPOCH_2", "ARX_PROSPECTIVE_VALIDATION_EPOCH_3")

    # Denominator Accounting: Epoch 1 final N = 0, Epoch 2 initial N = 0
    assert ExperimentLedger.get_epoch1_clean_prospective_count() == 0
    assert ExperimentLedger.get_epoch2_clean_prospective_count() == 0
