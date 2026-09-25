"""Epoch 3 Prospective Write Firewall & Admission Governance Regression Suite.

Verifies the fail-closed prospective observation admission firewall:
1. DELAYED quote after activation -> blocked (QUOTE_NOT_REALTIME, delta=0)
2. STALE quote after activation -> blocked (QUOTE_NOT_REALTIME, delta=0)
3. AFTER_HOURS after activation -> blocked (MARKET_SESSION_NOT_REGULAR, delta=0)
4. PREMARKET after activation -> blocked (MARKET_SESSION_NOT_REGULAR, delta=0)
5. CLOSED / WEEKEND / HOLIDAY -> blocked (MARKET_SESSION_NOT_REGULAR, delta=0)
6. Non-actionable realtime regular session (IN_BUY_ZONE without canonical actionability) -> blocked (DECISION_NOT_ACTIONABLE, delta=0)
7. Fully eligible (realtime + regular + actionable) -> first write delta=+1, duplicate write delta=0
8. Existing firewall conditions (certification, test, replay, synthetic, unauthorized, revoked, unactivated) -> delta=0
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timezone

from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine, init_governance_db
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger, ProvenanceCohort
from analyst_dashboard.governance.passive_capture import (
    PassiveCaptureHook,
    ExecutionContext,
    governance_execution_context,
)

pytestmark = pytest.mark.tier1


@pytest.fixture
def test_firewall_env():
    """Sets up an isolated SQLite governance DB and paper trading ledger."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as ledger_f:
        db_path = db_f.name
        ledger_path = ledger_f.name

    ExperimentLedger.save_ledger({"version": "2.0.0", "signals": [], "totalActiveSignals": 0}, ledger_path)
    init_governance_db(db_path)
    gov_engine = GovernanceDatabaseEngine(db_path=db_path)

    epoch_id = ExperimentLedger.EPOCH_ID
    auth_release = "e9eec914b5467fca99e82ea791afbaf3f8fef2bb"
    auth_deployment = "dep_prod_authorized_epoch3"
    now_iso = datetime.now(timezone.utc).isoformat()

    gov_engine.record_certification_and_authorization(
        epoch_id=epoch_id,
        release_sha=auth_release,
        deployment_id=auth_deployment,
        overall_status="PASS",
        result_payload_json=json.dumps({"status": "PASS", "certifiedBy": "SecurityGuardian"}),
        result_sha256="cert_hash_000000000000000000000000000000000000000000000000000000000001",
        certified_at_utc=now_iso,
    )

    gov_engine.record_activation(
        epoch_id=epoch_id,
        release_sha=auth_release,
        deployment_id=auth_deployment,
        activated_at_utc=now_iso,
        activation_source="PRODUCTION_ACTIVATION_SUITE",
        activation_auth_token_hash="token_hash_0000000000000000000000000000000000000000000000000000000001",
    )

    yield {
        "db_path": db_path,
        "ledger_path": ledger_path,
        "gov_engine": gov_engine,
        "epoch_id": epoch_id,
        "auth_release": auth_release,
        "auth_deployment": auth_deployment,
    }

    for p in (db_path, ledger_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _build_payload(
    symbol="AAPL",
    current_price=220.0,
    live_freshness="REALTIME",
    market_session="REGULAR_SESSION",
    is_actionable=True,
    decision_state="ACTIONABLE_SETUP",
    execution_status="IN_BUY_ZONE",
):
    return {
        "symbol": symbol,
        "current_price": current_price,
        "is_actionable": is_actionable,
        "decision_state": decision_state,
        "optimal_execution_plan": {
            "execution_status": execution_status,
            "optimal_entry_min": 215.0,
            "optimal_entry_max": 225.0,
            "stop_loss": 208.0,
            "stop_loss_pct": -5.4,
            "take_profit_1": 245.0,
            "take_profit_1_pct": 11.3,
            "risk_reward_ratio": 2.1,
            "atr_14": 4.2,
        },
        "confluence_output": {
            "confluenceScore": 88.0,
            "overall_eligibility": "FULL",
            "market_regime": "BULL",
            "sector": "TECHNOLOGY",
        },
        "technicals": {"sma_50": 210.0, "ema_20": 216.0, "rsi_14": 56.0, "atr_14": 4.2},
        "factor_scores": {"quality_score": 92.0, "growth_score": 85.0, "valuation_score": 75.0, "asOfDate": "2026-09-20"},
        "macro_inputs": {"macro_difficulty": 40.0, "yield_curve_10y2y": 0.20, "credit_spread": 3.0},
        "observed_at": "2026-09-24T15:30:00Z",
        "fetched_at": "2026-09-24T15:30:01Z",
        "freshness_status": live_freshness,
        "provider_source": "ALPACA_IEX",
        "candles": [
            {"date": "2026-09-23", "close": 218.0, "volume": 60000000},
            {"date": "2026-09-24", "close": 220.0, "volume": 55000000},
        ],
        "live_spot_price": current_price,
        "market_price_state": {
            "liveObservedAt": "2026-09-24T15:30:00Z",
            "liveSource": "ALPACA_IEX",
            "liveFreshness": live_freshness,
            "marketSession": market_session,
        },
    }


def test_section_10_delayed_quote_after_activation(test_firewall_env):
    """Section 10: DELAYED quote after activation is blocked."""
    env = test_firewall_env
    payload = _build_payload(live_freshness="DELAYED", market_session="REGULAR_SESSION")

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol=payload["symbol"],
        is_actionable=payload["is_actionable"],
        decision_state=payload["decision_state"],
        market_price_state=payload["market_price_state"],
        live_spot_price=payload["live_spot_price"],
        current_price=payload["current_price"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
        db_path=env["db_path"],
        ledger_path=env["ledger_path"],
    )
    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "QUOTE_NOT_REALTIME"

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    ledger = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger["signals"]) == 0


def test_section_11_stale_quote_after_activation(test_firewall_env):
    """Section 11: STALE quote after activation is blocked."""
    env = test_firewall_env
    payload = _build_payload(live_freshness="STALE", market_session="REGULAR_SESSION")

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol=payload["symbol"],
        is_actionable=payload["is_actionable"],
        decision_state=payload["decision_state"],
        market_price_state=payload["market_price_state"],
        live_spot_price=payload["live_spot_price"],
        current_price=payload["current_price"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
        db_path=env["db_path"],
        ledger_path=env["ledger_path"],
    )
    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "QUOTE_NOT_REALTIME"

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    ledger = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger["signals"]) == 0


def test_section_12_after_hours_after_activation(test_firewall_env):
    """Section 12: REALTIME quote during AFTER_HOURS is blocked."""
    env = test_firewall_env
    payload = _build_payload(live_freshness="REALTIME", market_session="AFTER_HOURS")

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol=payload["symbol"],
        is_actionable=payload["is_actionable"],
        decision_state=payload["decision_state"],
        market_price_state=payload["market_price_state"],
        live_spot_price=payload["live_spot_price"],
        current_price=payload["current_price"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
        db_path=env["db_path"],
        ledger_path=env["ledger_path"],
    )
    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "MARKET_SESSION_NOT_REGULAR"

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    ledger = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger["signals"]) == 0


def test_section_13_premarket_after_activation(test_firewall_env):
    """Section 13: REALTIME quote during PREMARKET is blocked."""
    env = test_firewall_env
    payload = _build_payload(live_freshness="REALTIME", market_session="PREMARKET")

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol=payload["symbol"],
        is_actionable=payload["is_actionable"],
        decision_state=payload["decision_state"],
        market_price_state=payload["market_price_state"],
        live_spot_price=payload["live_spot_price"],
        current_price=payload["current_price"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
        db_path=env["db_path"],
        ledger_path=env["ledger_path"],
    )
    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "MARKET_SESSION_NOT_REGULAR"

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    ledger = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger["signals"]) == 0


@pytest.mark.parametrize("session", ["CLOSED", "WEEKEND", "HOLIDAY"])
def test_section_14_closed_weekend_holiday(test_firewall_env, session):
    """Section 14: CLOSED / WEEKEND / HOLIDAY produces zero ledger delta."""
    env = test_firewall_env
    payload = _build_payload(live_freshness="REALTIME", market_session=session)

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol=payload["symbol"],
        is_actionable=payload["is_actionable"],
        decision_state=payload["decision_state"],
        market_price_state=payload["market_price_state"],
        live_spot_price=payload["live_spot_price"],
        current_price=payload["current_price"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
        db_path=env["db_path"],
        ledger_path=env["ledger_path"],
    )
    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "MARKET_SESSION_NOT_REGULAR"

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    ledger = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger["signals"]) == 0


def test_section_15_non_actionable_realtime_regular_session(test_firewall_env):
    """Section 15: IN_BUY_ZONE alone with is_actionable=False produces zero ledger delta."""
    env = test_firewall_env
    payload = _build_payload(
        live_freshness="REALTIME",
        market_session="REGULAR_SESSION",
        is_actionable=False,
        decision_state="VALID_SETUP",
        execution_status="IN_BUY_ZONE",
    )

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol=payload["symbol"],
        is_actionable=payload["is_actionable"],
        decision_state=payload["decision_state"],
        market_price_state=payload["market_price_state"],
        live_spot_price=payload["live_spot_price"],
        current_price=payload["current_price"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
        db_path=env["db_path"],
        ledger_path=env["ledger_path"],
    )
    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "DECISION_NOT_ACTIONABLE"

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    ledger = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger["signals"]) == 0


def test_section_16_fully_eligible_and_duplicate(test_firewall_env):
    """Section 16: Fully eligible natural recommendation produces delta=+1; duplicate produces delta=0."""
    env = test_firewall_env
    payload = _build_payload(
        symbol="MSFT",
        current_price=450.0,
        live_freshness="REALTIME",
        market_session="REGULAR_SESSION",
        is_actionable=True,
        decision_state="ACTIONABLE_SETUP",
    )

    # First write
    res1 = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res1 is not None
    assert res1["symbol"] == "MSFT"
    assert res1["provenanceCohort"] == ProvenanceCohort.PROSPECTIVE_CLEAN
    assert res1["liveFreshness"] == "REALTIME"
    assert res1["marketSession"] == "REGULAR_SESSION"

    ledger1 = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger1["signals"]) == 1

    # Replay duplicate recommendation
    res2 = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res2 is not None
    assert res2["signalId"] == res1["signalId"]

    ledger2 = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger2["signals"]) == 1, "Duplicate replay must produce zero ledger delta"


def test_section_17_all_existing_write_firewalls(test_firewall_env, monkeypatch):
    """Section 17: Re-tests all existing firewall conditions for zero physical mutation."""
    env = test_firewall_env
    base_payload = _build_payload()

    # 1. CERTIFICATION context
    with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
        res = PassiveCaptureHook.record_natural_recommendation(
            **base_payload,
            ledger_path=env["ledger_path"],
            db_path=env["db_path"],
            runtime_release_sha=env["auth_release"],
            runtime_deployment_id=env["auth_deployment"],
        )
        assert res is None

    # 2. TEST mode
    monkeypatch.setenv("ARX_TEST_MODE", "1")
    res = PassiveCaptureHook.record_natural_recommendation(
        **base_payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    monkeypatch.delenv("ARX_TEST_MODE")

    # 3. REPLAY mode
    monkeypatch.setenv("ARX_REPLAY_MODE", "1")
    res = PassiveCaptureHook.record_natural_recommendation(
        **base_payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    monkeypatch.delenv("ARX_REPLAY_MODE")

    # 4. SYNTHETIC payload
    syn_payload = _build_payload()
    syn_payload["optimal_execution_plan"]["isSynthetic"] = True
    res = PassiveCaptureHook.record_natural_recommendation(
        **syn_payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None

    # 5. UNAUTHORIZED deployment
    res = PassiveCaptureHook.record_natural_recommendation(
        **base_payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha="unauthorized_release_sha",
        runtime_deployment_id="unauthorized_deployment_id",
    )
    assert res is None

    # 6. REVOKED deployment
    env["gov_engine"].record_revocation(
        epoch_id=env["epoch_id"],
        release_sha=env["auth_release"],
        deployment_id=env["auth_deployment"],
        revoked_at_utc=datetime.now(timezone.utc).isoformat(),
        revocation_reason="Security test revocation",
        revoked_by="TEST_REVOCATION",
    )
    res = PassiveCaptureHook.record_natural_recommendation(
        **base_payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None

    # 7. EPOCH NOT ACTIVATED
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as unact_db:
        unact_db_path = unact_db.name
    init_governance_db(unact_db_path)
    res = PassiveCaptureHook.record_natural_recommendation(
        **base_payload,
        ledger_path=env["ledger_path"],
        db_path=unact_db_path,
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )
    assert res is None
    if os.path.exists(unact_db_path):
        os.remove(unact_db_path)

    # 8. LIVE SPOT INVALID (None, NaN, <= 0)
    for invalid_spot in (None, float("nan"), -10.0, 0.0):
        bad_payload = _build_payload()
        bad_payload["live_spot_price"] = invalid_spot
        res = PassiveCaptureHook.record_natural_recommendation(
            **bad_payload,
            ledger_path=env["ledger_path"],
            db_path=env["db_path"],
            runtime_release_sha=env["auth_release"],
            runtime_deployment_id=env["auth_deployment"],
        )
        assert res is None

    ledger = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger["signals"]) == 0, "All firewall conditions must produce zero ledger delta"
