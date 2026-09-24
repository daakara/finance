"""Write-Boundary Governance & Physical Ledger Anti-Contamination Regression Suite.

Verifies the canonical write-time sequence:
request
→ determine execution context
→ verify production runtime identity
→ verify deployment-scoped authorization predicate
→ verify active epoch boundary
→ verify natural recommendation eligibility
→ deduplicate
→ ONLY THEN write prospective record

Asserts physical ledger file deltas on disk (zero_synthetic_data invariant B):
1. Certification request → zero ledger delta
2. Test mode request → zero ledger delta
3. Replay mode request → zero ledger delta
4. Synthetic recommendation payload → zero ledger delta
5. Unauthorized deployment identity → zero ledger delta
6. Revoked deployment identity → zero ledger delta
7. Natural authorized recommendation → exactly one physical ledger increment
8. Duplicate natural recommendation → zero additional ledger increment
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timezone

from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine, init_governance_db
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from analyst_dashboard.governance.passive_capture import (
    PassiveCaptureHook,
    ExecutionContext,
    governance_execution_context,
)

pytestmark = pytest.mark.tier1


@pytest.fixture
def test_governance_env():
    """Sets up an isolated SQLite governance DB and paper trading ledger."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as ledger_f:
        db_path = db_f.name
        ledger_path = ledger_f.name

    # Initialize empty ledger
    ExperimentLedger.save_ledger({"version": "1.2.0", "signals": [], "totalActiveSignals": 0}, ledger_path)

    # Initialize governance DB schema
    init_governance_db(db_path)
    gov_engine = GovernanceDatabaseEngine(db_path=db_path)

    epoch_id = "ARX_PROSPECTIVE_VALIDATION_EPOCH_3"
    auth_release = "authorized_commit_sha_1234567890abcdef"
    auth_deployment = "dep_prod_authorized_789"
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Authorize deployment
    gov_engine.record_certification_and_authorization(
        epoch_id=epoch_id,
        release_sha=auth_release,
        deployment_id=auth_deployment,
        overall_status="PASS",
        result_payload_json=json.dumps({"status": "PASS", "certifiedBy": "SecurityGuardian"}),
        result_sha256="cert_hash_000000000000000000000000000000000000000000000000000000000001",
        certified_at_utc=now_iso,
    )

    # 2. Activate epoch boundary
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

    # Teardown
    for p in (db_path, ledger_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _build_natural_payload(symbol="NVDA", current_price=120.0):
    """Builds a complete natural client recommendation payload."""
    return {
        "symbol": symbol,
        "current_price": current_price,
        "is_actionable": True,
        "decision_state": "ACTIONABLE_SETUP",
        "optimal_execution_plan": {
            "execution_status": "IN_BUY_ZONE",
            "optimal_entry_min": 118.0,
            "optimal_entry_max": 122.0,
            "stop_loss": 114.0,
            "stop_loss_pct": -5.0,
            "take_profit_1": 132.0,
            "take_profit_1_pct": 10.0,
            "risk_reward_ratio": 2.0,
            "atr_14": 3.5,
        },
        "confluence_output": {
            "confluenceScore": 86.5,
            "overall_eligibility": "FULL",
            "market_regime": "BULL",
            "sector": "TECHNOLOGY",
        },
        "technicals": {"sma_50": 110.0, "ema_20": 115.0, "rsi_14": 55.0, "atr_14": 3.5},
        "factor_scores": {"quality_score": 85.0, "growth_score": 90.0, "valuation_score": 70.0, "asOfDate": "2026-09-20"},
        "macro_inputs": {"macro_difficulty": 45.0, "yield_curve_10y2y": 0.15, "credit_spread": 3.2},
        "observed_at": "2026-09-24T15:30:00Z",
        "fetched_at": "2026-09-24T15:30:01Z",
        "freshness_status": "REALTIME",
        "provider_source": "ALPACA_IEX",
        "candles": [
            {"date": "2026-09-23", "close": 118.5, "volume": 50000000},
            {"date": "2026-09-24", "close": 120.0, "volume": 45000000},
        ],
        "live_spot_price": current_price,
        "market_price_state": {
            "liveObservedAt": "2026-09-24T15:30:00Z",
            "liveSource": "ALPACA_IEX",
            "liveFreshness": "REALTIME",
            "marketSession": "REGULAR_SESSION",
        },
    }


def test_write_boundary_certification_request_zero_ledger_delta(test_governance_env):
    """1. Certification execution context strictly produces zero physical ledger delta."""
    env = test_governance_env
    payload = _build_natural_payload("AAPL")

    with governance_execution_context(ExecutionContext.GOVERNANCE_CERTIFICATION):
        res = PassiveCaptureHook.record_natural_recommendation(
            **payload,
            ledger_path=env["ledger_path"],
            db_path=env["db_path"],
            runtime_release_sha=env["auth_release"],
            runtime_deployment_id=env["auth_deployment"],
        )

    assert res is None
    # Verify disk content: 0 physical signals
    ledger_disk = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk["signals"]) == 0


def test_write_boundary_test_mode_zero_ledger_delta(test_governance_env, monkeypatch):
    """2. ARX_TEST_MODE=1 environment strictly produces zero physical ledger delta."""
    env = test_governance_env
    payload = _build_natural_payload("MSFT")
    monkeypatch.setenv("ARX_TEST_MODE", "1")

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )

    assert res is None
    ledger_disk = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk["signals"]) == 0


def test_write_boundary_replay_mode_zero_ledger_delta(test_governance_env, monkeypatch):
    """3. ARX_REPLAY_MODE=1 environment strictly produces zero physical ledger delta."""
    env = test_governance_env
    payload = _build_natural_payload("AMZN")
    monkeypatch.setenv("ARX_REPLAY_MODE", "1")

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )

    assert res is None
    ledger_disk = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk["signals"]) == 0


def test_write_boundary_synthetic_recommendation_zero_ledger_delta(test_governance_env):
    """4. Synthetic payload flags strictly produce zero physical ledger delta."""
    env = test_governance_env
    payload = _build_natural_payload("GOOGL")
    payload["optimal_execution_plan"]["isSynthetic"] = True

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )

    assert res is None
    ledger_disk = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk["signals"]) == 0


def test_write_boundary_unauthorized_deployment_zero_ledger_delta(test_governance_env):
    """5. Unauthorized runtime deployment identity strictly produces zero physical ledger delta."""
    env = test_governance_env
    payload = _build_natural_payload("TSLA")

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha="unauthorized_release_sha_bad",
        runtime_deployment_id="unauthorized_deployment_bad",
    )

    assert res is None
    ledger_disk = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk["signals"]) == 0


def test_write_boundary_revoked_deployment_zero_ledger_delta(test_governance_env):
    """6. Revoked deployment identity strictly produces zero physical ledger delta."""
    env = test_governance_env
    payload = _build_natural_payload("META")

    # Record revocation
    env["gov_engine"].record_revocation(
        epoch_id=env["epoch_id"],
        release_sha=env["auth_release"],
        deployment_id=env["auth_deployment"],
        revoked_at_utc=datetime.now(timezone.utc).isoformat(),
        revocation_reason="SECURITY_AUDIT_REVOCATION_TEST_LONG_REASON",
        revoked_by="SecurityOfficer",
    )

    res = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )

    assert res is None
    ledger_disk = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk["signals"]) == 0


def test_write_boundary_natural_authorized_recommendation_and_deduplication(test_governance_env):
    """7 & 8. Natural authorized recommendation increments ledger by exactly 1; duplicate produces zero delta."""
    env = test_governance_env
    payload = _build_natural_payload("NVDA")

    # First write: Valid natural recommendation
    res1 = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )

    assert res1 is not None
    assert res1["symbol"] == "NVDA"
    assert res1["releaseSha"] == env["auth_release"]
    assert res1["deploymentId"] == env["auth_deployment"]
    assert res1["captureSource"] == "NATURAL_PRODUCTION_API"

    # Verify physical file count on disk increments to exactly 1
    ledger_disk_1 = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk_1["signals"]) == 1
    assert ledger_disk_1["signals"][0]["symbol"] == "NVDA"
    assert ledger_disk_1["signals"][0]["releaseSha"] == env["auth_release"]

    # Second write: Duplicate request on the same calendar date
    res2 = PassiveCaptureHook.record_natural_recommendation(
        **payload,
        ledger_path=env["ledger_path"],
        db_path=env["db_path"],
        runtime_release_sha=env["auth_release"],
        runtime_deployment_id=env["auth_deployment"],
    )

    assert res2 is not None
    assert res2["signalId"] == res1["signalId"]

    # Verify physical file count on disk remains EXACTLY 1 (zero additional increment)
    ledger_disk_2 = ExperimentLedger.load_ledger(env["ledger_path"])
    assert len(ledger_disk_2["signals"]) == 1
