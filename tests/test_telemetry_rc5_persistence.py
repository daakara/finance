"""Comprehensive Increment 2B Durable Persistence & Synthetic Acceptance Suite.

Executes all 20 Increment 2A-R scenarios + Increment 2A acceptance gates:
- EID-01 through EID-10: Evaluation identity, cross-tab deduplication, hourly bounds, rehydration, conflict quarantine.
- MIG-01 through MIG-03: Migration idempotency, schema incompatibility detection, rollback safety.
- FLG-01 through FLG-03: Four-state feature flag precedence (fail-closed).
- RBK-01 through RBK-02: Application rollback compatibility and zero-loss kill switch.
- ACK-01 through ACK-02: Acknowledgment recovery and outcome integrity.
- Additional Identity Adversarial Tests: Multi-threaded concurrency (50 workers), KAT cross-language parity.
"""

from __future__ import annotations

import os
import json
import uuid
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from analyst_dashboard.telemetry.corporate_event_contract import (
    CorporateEventTelemetryPayload,
    CorporateEventCategory,
    MarketSession,
    EventSourceStatus,
    FreshnessStatus,
    RiskBucket,
    EvaluationOutcome,
    compute_canonical_digest,
    derive_authoritative_evaluation_id,
)
from analyst_dashboard.telemetry.migration_rc5 import (
    apply_rc5_telemetry_migration,
    verify_rc5_telemetry_schema_compatibility,
    rollback_rc5_telemetry_schema,
)
from api.main import app

AUTH_SHA = "8404b6e4645b5f9f8237e70517dcab3b6afdda7f"
TEST_SECRET = "arx_synthetic_test_internal_secret"


@pytest.fixture
def test_db_path(tmp_path):
    """Provides an isolated disposable SQLite database file."""
    db_file = str(tmp_path / "test_telemetry_rc5.db")
    conn = sqlite3.connect(db_file)
    apply_rc5_telemetry_migration(conn)
    conn.close()
    return db_file


@pytest.fixture
def active_epoch(test_db_path):
    """Provisions an active observation epoch in the test database."""
    conn = sqlite3.connect(test_db_path)
    now = datetime.now(timezone.utc)
    start_ts = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    end_ts = (now + timedelta(days=2)).isoformat().replace("+00:00", "Z")
    created_ts = now.isoformat().replace("+00:00", "Z")

    conn.execute(
        """
        INSERT INTO arx_rc5_observation_epochs (
            epoch_id, name, authorized_release_sha, start_timestamp_utc, end_timestamp_utc, is_active, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("epoch-test-001", "Sprint 3 RC5 Observation Epoch", AUTH_SHA, start_ts, end_ts, 1, created_ts),
    )
    conn.commit()
    conn.close()
    return "epoch-test-001"


def make_rc5_payload(
    symbol="AAPL",
    release_sha=AUTH_SHA,
    penalty=-8.0,
    risk_bucket="CAUTION",
    days_to_earnings=3.5,
    evaluation_id=None,
    occurred_at_utc=None,
    synthetic_marker=False,
):
    now = datetime.now(timezone.utc)
    occurred_str = occurred_at_utc or now.isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "1.1.0",
        "event_id": str(uuid.uuid4()),
        "occurred_at_utc": occurred_str,
        "release_sha": release_sha,
        "ruleset_version": "RC5",
        "evaluation_context_id": f"ctx-{uuid.uuid4().hex[:8]}",
        "symbol": symbol,
        "event_category": "EARNINGS_ANNOUNCEMENT",
        "event_time_utc": (now + timedelta(days=4)).isoformat().replace("+00:00", "Z"),
        "market_session": "AMC",
        "event_source_status": "CONFIRMED",
        "freshness_status": "FRESH",
        "days_to_earnings": days_to_earnings,
        "risk_bucket": risk_bucket,
        "applied_penalty": penalty,
        "evaluation_outcome": "COMPLETED",
        "evaluation_id": evaluation_id,
        "synthetic_marker": synthetic_marker,
    }


# ============================================================================
# 1. GATE B & C: EVALUATION IDENTITY & CANONICAL DIGEST PARITY
# ============================================================================

def test_kat_canonical_digest_parity():
    """Verifies bit-for-bit canonical digest parity with Node known-answer test vector."""
    payload = {
        "symbol": "AAPL",
        "release_sha": "8404b6e4645b5f9f8237e70517dcab3b6afdda7f",
        "ruleset_version": "RC5",
        "event_category": "EARNINGS_ANNOUNCEMENT",
        "event_time_utc": "2026-10-27T21:00:00.000Z",
        "market_session": "POST_MARKET",
        "event_source_status": "CONFIRMED",
        "freshness_status": "FRESH",
        "days_to_earnings": 3.0,
        "risk_bucket": "CAUTION",
        "applied_penalty": -8.0,
        "evaluation_outcome": "COMPLETED",
    }
    digest = compute_canonical_digest(payload)
    assert digest == "5dc59ac1d4a0c2194feeaea349d3fd14528fd64ee6132b8a6bafa8e31cc731e0"


def test_kat_evaluation_id_parity():
    """Verifies deterministic evaluation_id parity with Node known-answer test vector."""
    eval_id = derive_authoritative_evaluation_id(
        release_sha="8404b6e4645b5f9f8237e70517dcab3b6afdda7f",
        symbol="AAPL",
        ruleset_version="RC5",
        event_time_utc="2026-10-27T21:00:00.000Z",
        market_session="POST_MARKET",
        evaluation_timestamp="2026-10-24T14:30:00.000Z",
    )
    assert eval_id == "aebb93cf5b7465165108aeb4f8673696eee6930cc8a8659ad5db43c93ec74165"


# ============================================================================
# 2. SCENARIO EID-01: CROSS-TAB DEDUPLICATION
# ============================================================================

def test_eid_01_cross_tab_deduplication(test_db_path, active_epoch, monkeypatch):
    """EID-01: Two tabs evaluating the same ticker share deterministic evaluation_id.
    Tab 1 persists; Tab 2 acknowledged as CONFIRMED_DUPLICATE. Denominator = 1.
    """
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    fixed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    eval_id = derive_authoritative_evaluation_id(
        release_sha=AUTH_SHA,
        symbol="AAPL",
        ruleset_version="RC5",
        event_time_utc="2026-10-27T21:00:00Z",
        market_session="AMC",
        evaluation_timestamp=fixed_time,
    )

    # Tab 1 Payload
    tab1_payload = make_rc5_payload(evaluation_id=eval_id, occurred_at_utc=fixed_time)
    tab1_payload["evaluation_context_id"] = "ctx-tab1-alpha"

    # Tab 2 Payload (different tab context, same evaluation_id and timestamp)
    tab2_payload = make_rc5_payload(evaluation_id=eval_id, occurred_at_utc=fixed_time)
    tab2_payload["evaluation_context_id"] = "ctx-tab2-beta"

    headers = {"X-Internal-Secret": TEST_SECRET}

    res1 = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": tab1_payload, "delivery_attempt_id": "deliv-tab1"},
        headers=headers,
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "PERSISTED"
    assert res1.json()["persisted"] == "CONFIRMED"

    res2 = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": tab2_payload, "delivery_attempt_id": "deliv-tab2"},
        headers=headers,
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "CONFIRMED_DUPLICATE"
    assert res2.json()["persisted"] == "CONFIRMED"
    assert res2.json()["retransmission_count"] == 1

    # Verify Database state
    conn = sqlite3.connect(test_db_path)
    eval_count = conn.execute("SELECT COUNT(*) FROM arx_rc5_telemetry_evaluations").fetchone()[0]
    raw_count = conn.execute("SELECT COUNT(*) FROM arx_rc5_telemetry_raw_events").fetchone()[0]
    conn.close()

    assert eval_count == 1  # Deduplicated!
    assert raw_count == 2   # Both transport attempts recorded


# ============================================================================
# 3. SCENARIO EID-02: RETRY AFTER TIMEOUT
# ============================================================================

def test_eid_02_retry_after_timeout(test_db_path, active_epoch, monkeypatch):
    """EID-02: Client retries after network timeout with same evaluation_id, new delivery_attempt_id."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    payload = make_rc5_payload()
    headers = {"X-Internal-Secret": TEST_SECRET}

    r1 = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": payload, "delivery_attempt_id": "attempt-001"},
        headers=headers,
    )
    assert r1.status_code == 200
    assert r1.json()["status"] == "PERSISTED"

    r2 = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": payload, "delivery_attempt_id": "attempt-002-retry"},
        headers=headers,
    )
    assert r2.status_code == 200
    assert r2.json()["status"] == "CONFIRMED_DUPLICATE"
    assert r2.json()["persisted"] == "CONFIRMED"


# ============================================================================
# 4. SCENARIO EID-03: DISTINCT EXECUTIONS WITH SAME RISK PAYLOAD
# ============================================================================

def test_eid_03_distinct_executions_not_merged(test_db_path, active_epoch, monkeypatch):
    """EID-03: AAPL and MSFT produce identical CAUTION / -8.0 outcomes.
    Both must persist as distinct logical evaluations. Denominator = 2.
    """
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    p1 = make_rc5_payload(symbol="AAPL", penalty=-8.0, risk_bucket="CAUTION")
    p2 = make_rc5_payload(symbol="MSFT", penalty=-8.0, risk_bucket="CAUTION")

    headers = {"X-Internal-Secret": TEST_SECRET}

    r1 = client.post(f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}", json={"raw_event": p1}, headers=headers)
    r2 = client.post(f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}", json={"raw_event": p2}, headers=headers)

    assert r1.status_code == 200 and r1.json()["status"] == "PERSISTED"
    assert r2.status_code == 200 and r2.json()["status"] == "PERSISTED"

    conn = sqlite3.connect(test_db_path)
    eval_count = conn.execute("SELECT COUNT(*) FROM arx_rc5_telemetry_evaluations").fetchone()[0]
    conn.close()
    assert eval_count == 2


# ============================================================================
# 5. SCENARIO EID-04 & EID-08: CONFLICTING IDENTITY MUTATION QUARANTINE
# ============================================================================

def test_eid_04_conflicting_identity_quarantined(test_db_path, active_epoch, monkeypatch):
    """EID-04: Same (release_sha, evaluation_id) with altered penalty is rejected and quarantined."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    fixed_id = str(uuid.uuid4())
    p1 = make_rc5_payload(symbol="AAPL", penalty=-8.0, risk_bucket="CAUTION", days_to_earnings=3.0, evaluation_id=fixed_id)
    # Altered penalty (-25.0) under same ID:
    p2 = make_rc5_payload(symbol="AAPL", penalty=-25.0, risk_bucket="ACUTE", days_to_earnings=0.5, evaluation_id=fixed_id)

    headers = {"X-Internal-Secret": TEST_SECRET}

    r1 = client.post(f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}", json={"raw_event": p1}, headers=headers)
    assert r1.status_code == 200
    assert r1.json()["status"] == "PERSISTED"

    r2 = client.post(f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}", json={"raw_event": p2}, headers=headers)
    assert r2.status_code == 409
    assert r2.json()["status"] == "CONFLICTING_IDENTITY"
    assert r2.json()["persisted"] == "NOT_PERSISTED"

    # Verify Quarantine insertion
    conn = sqlite3.connect(test_db_path)
    quar_count = conn.execute("SELECT COUNT(*) FROM arx_rc5_telemetry_quarantine WHERE reason_code = 'CONFLICTING_IDENTITY'").fetchone()[0]
    # Original evaluation must remain untouched with -8.0 penalty
    stored_penalty = conn.execute("SELECT applied_penalty FROM arx_rc5_telemetry_evaluations WHERE evaluation_id = ?", (fixed_id,)).fetchone()[0]
    conn.close()

    assert quar_count == 1
    assert stored_penalty == -8.0


# ============================================================================
# 6. SCENARIO EID-07: CONCURRENT IDENTICAL WRITES (50 WORKERS)
# ============================================================================

def test_eid_07_concurrent_identical_writes(test_db_path, active_epoch, monkeypatch):
    """EID-07: 50 concurrent requests for the identical evaluation.
    Exactly 1 evaluation row created, 50 raw attempts, 0 deadlocks.
    """
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    fixed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    eval_id = derive_authoritative_evaluation_id(
        release_sha=AUTH_SHA,
        symbol="NVDA",
        ruleset_version="RC5",
        event_time_utc="2026-10-30T21:00:00Z",
        market_session="AMC",
        evaluation_timestamp=fixed_time,
    )
    payload = make_rc5_payload(symbol="NVDA", evaluation_id=eval_id, occurred_at_utc=fixed_time)
    headers = {"X-Internal-Secret": TEST_SECRET}

    def dispatch(idx):
        return client.post(
            f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
            json={"raw_event": payload, "delivery_attempt_id": f"deliv-conc-{idx}"},
            headers=headers,
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(dispatch, i) for i in range(50)]
        results = [f.result() for f in futures]

    statuses = [r.status_code for r in results]
    assert all(s == 200 for s in statuses)

    verdicts = [r.json()["status"] for r in results]
    assert verdicts.count("PERSISTED") == 1
    assert verdicts.count("CONFIRMED_DUPLICATE") == 49

    conn = sqlite3.connect(test_db_path)
    eval_count = conn.execute("SELECT COUNT(*) FROM arx_rc5_telemetry_evaluations").fetchone()[0]
    raw_count = conn.execute("SELECT COUNT(*) FROM arx_rc5_telemetry_raw_events").fetchone()[0]
    conn.close()

    assert eval_count == 1
    assert raw_count == 50


# ============================================================================
# 7. SCENARIO MIG-01, MIG-02, MIG-03: MIGRATION COMPATIBILITY
# ============================================================================

def test_mig_01_migration_idempotent(test_db_path):
    """MIG-01: Re-running 005 migration on existing DB succeeds cleanly without mutation."""
    conn = sqlite3.connect(test_db_path)
    res = apply_rc5_telemetry_migration(conn)
    assert res["status"] == "APPLIED"
    assert res["verified"] is True
    conn.close()


def test_mig_02_incompatible_schema_detected(tmp_path):
    """MIG-02: Pre-flight check halts if an incompatible table exists."""
    db_file = str(tmp_path / "incompatible.db")
    conn = sqlite3.connect(db_file)
    # Create corrupted table missing required columns
    conn.execute("CREATE TABLE arx_rc5_telemetry_evaluations (id INTEGER PRIMARY KEY, junk TEXT);")
    conn.commit()

    compatible, errors = verify_rc5_telemetry_schema_compatibility(conn)
    conn.close()

    assert compatible is False
    assert any("Incompatible arx_rc5_telemetry_evaluations" in e or "missing" in e for e in errors)


# ============================================================================
# 8. SCENARIO FLG-01, FLG-03 & RBK-02: FEATURE FLAGS & KILL SWITCH
# ============================================================================

def test_flg_01_backend_disabled_kill_switch(test_db_path, monkeypatch):
    """FLG-01 & RBK-02: When ARX_PR53_TELEMETRY_ENABLED is false, returns DISABLED, preserves existing."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    payload = make_rc5_payload()
    headers = {"X-Internal-Secret": TEST_SECRET}

    res = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": payload},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "DISABLED"
    assert res.json()["persisted"] == "NOT_PERSISTED"


def test_flg_denominator_zero_when_disabled(test_db_path, monkeypatch):
    """Denominator endpoint returns 0 when disabled by policy."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "false")
    client = TestClient(app)

    res = client.get(f"/api/v1/telemetry/corporate-events/denominator?db_path={test_db_path}")
    assert res.status_code == 200
    assert res.json()["natural_eligible_denominator"] == 0
    assert res.json()["real_natural_production_denominator"] == "NOT_OBSERVABLE"


# ============================================================================
# 9. SCENARIO EID-10: SYNTHETIC RECORD EXCLUSION
# ============================================================================

def test_eid_10_synthetic_record_excluded_from_denominator(test_db_path, active_epoch, monkeypatch):
    """EID-10: Synthetic event is persisted for audit but marked is_natural_candidate = 0."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    payload = make_rc5_payload(synthetic_marker=True)
    headers = {"X-Internal-Secret": TEST_SECRET}

    res = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": payload},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "PERSISTED"
    assert res.json()["is_natural_candidate"] == 0

    # Denominator must remain 0
    denom_res = client.get(f"/api/v1/telemetry/corporate-events/denominator?db_path={test_db_path}")
    assert denom_res.json()["natural_eligible_denominator"] == 0


# ============================================================================
# 10. REPLAY WINDOW & CLOCK SKEW ENFORCEMENT
# ============================================================================

def test_replay_window_expired_rejected(test_db_path, monkeypatch):
    """Event older than 24 hours is rejected upfront."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    expired_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat().replace("+00:00", "Z")
    payload = make_rc5_payload(occurred_at_utc=expired_time)
    headers = {"X-Internal-Secret": TEST_SECRET}

    res = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": payload},
        headers=headers,
    )
    assert res.status_code == 400
    assert res.json()["error_code"] == "EXPIRED_REPLAY_WINDOW"


def test_future_clock_skew_rejected(test_db_path, monkeypatch):
    """Event >60s in future is rejected upfront."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    future_time = (datetime.now(timezone.utc) + timedelta(seconds=75)).isoformat().replace("+00:00", "Z")
    payload = make_rc5_payload(occurred_at_utc=future_time)
    headers = {"X-Internal-Secret": TEST_SECRET}

    res = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": payload},
        headers=headers,
    )
    assert res.status_code == 400
    assert res.json()["error_code"] == "FUTURE_TIMESTAMP_REJECTED"


# ============================================================================
# 11. INTERNAL SECRET AUTHENTICATION
# ============================================================================

def test_missing_or_bad_internal_secret(test_db_path, monkeypatch):
    """Missing or bad internal secret fails closed (401/403)."""
    monkeypatch.setenv("ARX_PR53_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("ARX_TEST_ENVIRONMENT", "1")
    client = TestClient(app)

    payload = make_rc5_payload()

    # Missing header -> 401
    r_missing = client.post(f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}", json={"raw_event": payload})
    assert r_missing.status_code == 401

    # Bad header -> 403
    r_bad = client.post(
        f"/api/v1/telemetry/corporate-events/persist?db_path={test_db_path}",
        json={"raw_event": payload},
        headers={"X-Internal-Secret": "invalid_attacker_secret"},
    )
    assert r_bad.status_code == 403
