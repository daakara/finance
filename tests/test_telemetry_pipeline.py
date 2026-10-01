"""Unit and Integration Tests for ARX ETF Cockpit P1 Telemetry Pipeline.

Covers:
1. Migration and schema integrity for P1 telemetry tables.
2. Invariant: When no active observation epoch exists, prospective denominator is strictly 0.
3. Strict fail-closed internal authentication via X-Internal-Secret.
4. Defense-in-depth: VALID classification requires active epoch and exact authorized release match.
5. Inactive / wrong epoch rejection.
6. Non-VALID states (EXCLUDED, QUARANTINED, INVALID) contract alignment.
7. Atomic persistence of raw candidate events and canonical audit records.
8. Zero side effects on rejected requests.
9. Race-safe deduplication: Unique deduplication_key prevents duplicate counting.
10. Distinct observation_unit_id counting in dynamic denominator derivation.
11. Multi-attempt semantics for distinct user attempts.
12. Read-only audit queries (/api/v1/telemetry/audit, /api/v1/telemetry/epochs).
"""

import os
import gc
import json
import tempfile
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.routes.telemetry import (
    init_telemetry_tables,
    get_db_connection,
)

pytestmark = pytest.mark.tier2c

client = TestClient(app)

TEST_SECRET = "test-internal-secret-p1-fixture"
PROD_RELEASE = "9d1cce52070ba242136c3346ce7fd6c83b6ad3ed"


@pytest.fixture(autouse=True)
def inject_test_secret(monkeypatch):
    """Inject test-only secret into environment. No hardcoded production fallback."""
    monkeypatch.setenv("TELEMETRY_INTERNAL_SECRET", TEST_SECRET)


@pytest.fixture
def temp_db():
    """Create an isolated temporary SQLite database for telemetry testing."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_telemetry_tables(db_path)
    yield db_path
    gc.collect()
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except Exception:
        pass


def auth_headers(secret: str = TEST_SECRET) -> dict:
    return {"X-Internal-Secret": secret}


def activate_test_epoch(db_path: str, epoch_id: str = "EPOCH_P1_TEST_ACTIVE", releases: list = None) -> None:
    if releases is None:
        releases = [PROD_RELEASE]
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO arx_p1_observation_epochs (
                epoch_id, name, start_timestamp, authorized_releases, activated_at, activated_by, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, 1);
            """,
            (
                epoch_id,
                "P1 Test Active Epoch",
                "2026-10-01T00:00:00Z",
                json.dumps(releases),
                "2026-10-01T00:00:00Z",
                "TEST_HARNESS",
            ),
        )
        conn.commit()


def test_telemetry_tables_creation(temp_db):
    """Verify all four canonical P1 telemetry tables and unique indexes are created."""
    with get_db_connection(temp_db) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        table_names = [r["name"] for r in tables]
        assert "arx_p1_observation_epochs" in table_names
        assert "arx_p1_telemetry_audit_records" in table_names
        assert "arx_p1_telemetry_invalid_events" in table_names
        assert "arx_p1_telemetry_raw_events" in table_names

        indexes = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='arx_p1_telemetry_audit_records';"
        ).fetchall()
        index_names = [r["name"] for r in indexes]
        assert "idx_audit_dedup" in index_names


def test_denominator_zero_when_no_active_epoch(temp_db):
    """Invariant: When no active epoch exists, prospective denominator is strictly 0."""
    res = client.get(f"/api/v1/telemetry/denominator?db_path={temp_db}")
    assert res.status_code == 200
    data = res.json()
    assert data["prospective_denominator"] == 0
    assert data["active_epoch"] is None
    assert data["epoch_status"] == "NOT_ESTABLISHED"


def test_direct_persistence_bypass_rejected(temp_db):
    """R24 Regression Test: Unauthenticated direct POST /persist is rejected with HTTP 401."""
    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "00000000-0000-0000-0000-000000000001",
            "session_id": "00000000-0000-0000-0000-000000000002",
            "observation_unit_id": "00000000-0000-0000-0000-000000000003",
            "deduplication_key": "1" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "intent_type": "ETF_SYMBOL_SELECT",
            "route": "/",
            "source_component": "handleSelectSymbol",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
        },
        "audit_record": {
            "observation_unit_id": "00000000-0000-0000-0000-000000000003",
            "event_id": "00000000-0000-0000-0000-000000000001",
            "session_id": "00000000-0000-0000-0000-000000000002",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "1" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
        },
    }

    # No header provided
    res = client.post(f"/api/v1/telemetry/persist?db_path={temp_db}", json=req_payload)
    assert res.status_code == 401
    assert "missing internal telemetry secret header" in res.json()["detail"]

    # Verify zero persistence
    with get_db_connection(temp_db) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_raw_events;").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_audit_records;").fetchone()["c"] == 0


def test_wrong_internal_secret_rejected(temp_db):
    """Wrong X-Internal-Secret is rejected with HTTP 403 Forbidden."""
    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "00000000-0000-0000-0000-000000000011",
            "session_id": "00000000-0000-0000-0000-000000000012",
            "observation_unit_id": "00000000-0000-0000-0000-000000000013",
            "deduplication_key": "2" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
        },
        "audit_record": {
            "observation_unit_id": "00000000-0000-0000-0000-000000000013",
            "event_id": "00000000-0000-0000-0000-000000000011",
            "session_id": "00000000-0000-0000-0000-000000000012",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "2" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
        },
    }

    res = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers("completely-wrong-secret"),
    )
    assert res.status_code == 403
    assert "invalid internal telemetry secret" in res.json()["detail"]

    with get_db_connection(temp_db) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_raw_events;").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_audit_records;").fetchone()["c"] == 0


def test_empty_internal_secret_rejected(temp_db):
    """Empty X-Internal-Secret is rejected with HTTP 401 Unauthorized."""
    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "00000000-0000-0000-0000-000000000021",
            "session_id": "00000000-0000-0000-0000-000000000022",
            "observation_unit_id": "00000000-0000-0000-0000-000000000023",
            "deduplication_key": "3" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
        },
        "audit_record": {
            "observation_unit_id": "00000000-0000-0000-0000-000000000023",
            "event_id": "00000000-0000-0000-0000-000000000021",
            "session_id": "00000000-0000-0000-0000-000000000022",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "3" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
        },
    }

    res = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers={"X-Internal-Secret": "   "},
    )
    assert res.status_code == 401


def test_missing_server_secret_fail_closed(temp_db, monkeypatch):
    """When TELEMETRY_INTERNAL_SECRET is unconfigured on the server, requests fail closed (HTTP 500)."""
    monkeypatch.delenv("TELEMETRY_INTERNAL_SECRET", raising=False)

    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "00000000-0000-0000-0000-000000000031",
            "session_id": "00000000-0000-0000-0000-000000000032",
            "observation_unit_id": "00000000-0000-0000-0000-000000000033",
            "deduplication_key": "4" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
        },
        "audit_record": {
            "observation_unit_id": "00000000-0000-0000-0000-000000000033",
            "event_id": "00000000-0000-0000-0000-000000000031",
            "session_id": "00000000-0000-0000-0000-000000000032",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "4" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
        },
    }

    res = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers(TEST_SECRET),
    )
    assert res.status_code == 500
    assert "internal secret not configured" in res.json()["detail"]


def test_valid_without_active_epoch_rejected(temp_db):
    """Defense-in-depth: Authenticated caller cannot persist VALID record when no active epoch exists."""
    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "00000000-0000-0000-0000-000000000041",
            "session_id": "00000000-0000-0000-0000-000000000042",
            "observation_unit_id": "00000000-0000-0000-0000-000000000043",
            "deduplication_key": "5" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "intent_type": "ETF_SYMBOL_SELECT",
            "route": "/",
            "source_component": "handleSelectSymbol",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
        },
        "audit_record": {
            "observation_unit_id": "00000000-0000-0000-0000-000000000043",
            "event_id": "00000000-0000-0000-0000-000000000041",
            "session_id": "00000000-0000-0000-0000-000000000042",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "5" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_NON_EXISTENT",
        },
    }

    res = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers(),
    )
    assert res.status_code == 400
    assert "no active observation epoch exists" in res.json()["detail"]

    with get_db_connection(temp_db) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_raw_events;").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_audit_records;").fetchone()["c"] == 0


def test_valid_unauthorized_release_rejected(temp_db):
    """Defense-in-depth: Authenticated caller cannot persist VALID record with unauthorized release_sha."""
    activate_test_epoch(temp_db, epoch_id="EPOCH_P1_TEST_ACTIVE", releases=[PROD_RELEASE])

    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "00000000-0000-0000-0000-000000000051",
            "session_id": "00000000-0000-0000-0000-000000000052",
            "observation_unit_id": "00000000-0000-0000-0000-000000000053",
            "deduplication_key": "6" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "intent_type": "ETF_SYMBOL_SELECT",
            "route": "/",
            "source_component": "handleSelectSymbol",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": "unauthorized_forged_release_sha",
        },
        "audit_record": {
            "observation_unit_id": "00000000-0000-0000-0000-000000000053",
            "event_id": "00000000-0000-0000-0000-000000000051",
            "session_id": "00000000-0000-0000-0000-000000000052",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": "unauthorized_forged_release_sha",
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Forged valid record",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "6" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_P1_TEST_ACTIVE",
        },
    }

    res = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers(),
    )
    assert res.status_code == 400
    assert "not authorized in active epoch" in res.json()["detail"]

    with get_db_connection(temp_db) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_raw_events;").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_audit_records;").fetchone()["c"] == 0


def test_inactive_epoch_authorization_rejected(temp_db):
    """Historical or inactive epochs cannot authorize VALID record persistence."""
    with get_db_connection(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO arx_p1_observation_epochs (
                epoch_id, name, start_timestamp, authorized_releases, activated_at, activated_by, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, 0);
            """,
            (
                "EPOCH_HISTORICAL_INACTIVE",
                "Historical Inactive Epoch",
                "2026-09-01T00:00:00Z",
                json.dumps([PROD_RELEASE]),
                "2026-09-01T00:00:00Z",
                "TEST_HARNESS",
            ),
        )
        conn.commit()

    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "00000000-0000-0000-0000-000000000061",
            "session_id": "00000000-0000-0000-0000-000000000062",
            "observation_unit_id": "00000000-0000-0000-0000-000000000063",
            "deduplication_key": "7" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
        },
        "audit_record": {
            "observation_unit_id": "00000000-0000-0000-0000-000000000063",
            "event_id": "00000000-0000-0000-0000-000000000061",
            "session_id": "00000000-0000-0000-0000-000000000062",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "7" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_HISTORICAL_INACTIVE",
        },
    }

    res = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers(),
    )
    assert res.status_code == 400
    assert "no active observation epoch exists" in res.json()["detail"]


def test_authorized_release_exact_match(temp_db):
    """Authorized release membership requires exact string match, not substring or prefix."""
    activate_test_epoch(temp_db, epoch_id="EPOCH_P1_TEST_ACTIVE", releases=[PROD_RELEASE])

    def make_payload(sha: str, key: str):
        return {
            "raw_event": {
                "schema_version": "1.0.0",
                "event_id": f"evt-{key}",
                "session_id": f"ses-{key}",
                "observation_unit_id": f"obs-{key}",
                "deduplication_key": key,
                "timestamp": "2026-10-01T12:00:00Z",
                "normalized_symbol": "SPY",
                "release_sha": sha,
            },
            "audit_record": {
                "observation_unit_id": f"obs-{key}",
                "event_id": f"evt-{key}",
                "session_id": f"ses-{key}",
                "timestamp": "2026-10-01T12:00:00Z",
                "normalized_symbol": "SPY",
                "environment": "production",
                "deployment_identity": "https://www.arxterminal.com",
                "release_sha": sha,
                "traffic_class": "NATURAL_PRODUCTION",
                "classification_state": "VALID",
                "classification_reason": "Clean natural attempt",
                "exclusion_code": "NONE",
                "quarantine_reason": "NONE",
                "deduplication_key": key,
                "replay_identity": "NONE",
                "source_component": "handleSelectSymbol",
                "epoch_id": "EPOCH_P1_TEST_ACTIVE",
            },
        }

    # 1. Prefix match should be rejected
    res_prefix = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=make_payload(PROD_RELEASE[:7], "8" * 64),
        headers=auth_headers(),
    )
    assert res_prefix.status_code == 400

    # 2. Suffix-extended match should be rejected
    res_extended = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=make_payload(PROD_RELEASE + "0", "9" * 64),
        headers=auth_headers(),
    )
    assert res_extended.status_code == 400

    # 3. Exact match succeeds
    res_exact = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=make_payload(PROD_RELEASE, "a" * 64),
        headers=auth_headers(),
    )
    assert res_exact.status_code == 200
    assert res_exact.json()["status"] == "PERSISTED"


def test_atomic_telemetry_persistence_with_active_epoch(temp_db):
    """Verify single atomic write persists raw event and audit record when authenticated & active epoch."""
    activate_test_epoch(temp_db, epoch_id="EPOCH_P1_TEST_ACTIVE", releases=[PROD_RELEASE])

    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "11111111-1111-4111-8111-111111111111",
            "session_id": "22222222-2222-4222-8222-222222222222",
            "observation_unit_id": "33333333-3333-4333-8333-333333333333",
            "deduplication_key": "b" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "intent_type": "ETF_SYMBOL_SELECT",
            "route": "/",
            "source_component": "handleSelectSymbol",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "synthetic_marker": False,
            "ci_run_marker": False,
            "manual_qa_marker": False,
            "user_agent_raw": "Mozilla/5.0",
        },
        "audit_record": {
            "observation_unit_id": "33333333-3333-4333-8333-333333333333",
            "event_id": "11111111-1111-4111-8111-111111111111",
            "session_id": "22222222-2222-4222-8222-222222222222",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural production attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "b" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_P1_TEST_ACTIVE",
        },
    }

    res = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers(),
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PERSISTED"
    assert data["deduplication_key"] == "b" * 64

    # Verify rows in database
    with get_db_connection(temp_db) as conn:
        raw_count = conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_raw_events;").fetchone()["c"]
        audit_count = conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_audit_records;").fetchone()["c"]
        assert raw_count == 1
        assert audit_count == 1


def test_non_valid_records_persisted_without_active_epoch(temp_db):
    """Section 15 Contract: Non-VALID records (EXCLUDED, QUARANTINED, INVALID) persist without active epoch.

    Crucially, they DO NOT increment the prospective denominator.
    """
    req_excluded = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "ex-0001",
            "session_id": "ses-0001",
            "observation_unit_id": "obs-ex-0001",
            "deduplication_key": "c" * 64,
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
        },
        "audit_record": {
            "observation_unit_id": "obs-ex-0001",
            "event_id": "ex-0001",
            "session_id": "ses-0001",
            "timestamp": "2026-10-01T12:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "CONTROLLED_TEST",
            "classification_state": "EXCLUDED",
            "classification_reason": "Synthetic marker detected",
            "exclusion_code": "EX_SYNTHETIC",
            "quarantine_reason": "NONE",
            "deduplication_key": "c" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
        },
    }

    req_quarantined = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "qu-0001",
            "session_id": "ses-0002",
            "observation_unit_id": "obs-qu-0001",
            "deduplication_key": "d" * 64,
            "timestamp": "2026-10-01T12:05:00Z",
            "normalized_symbol": "QQQ",
        },
        "audit_record": {
            "observation_unit_id": "obs-qu-0001",
            "event_id": "qu-0001",
            "session_id": "ses-0002",
            "timestamp": "2026-10-01T12:05:00Z",
            "normalized_symbol": "QQQ",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "UNKNOWN",
            "classification_state": "QUARANTINED",
            "classification_reason": "Missing origin verification",
            "exclusion_code": "NONE",
            "quarantine_reason": "UNRESOLVED_PROVENANCE",
            "deduplication_key": "d" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
        },
    }

    # Authenticated submission succeeds without active epoch
    res1 = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_excluded,
        headers=auth_headers(),
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "PERSISTED"

    res2 = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_quarantined,
        headers=auth_headers(),
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "PERSISTED"

    # Prospective denominator must remain strictly 0
    res_denom = client.get(f"/api/v1/telemetry/denominator?db_path={temp_db}")
    assert res_denom.status_code == 200
    assert res_denom.json()["prospective_denominator"] == 0


def test_race_safe_deduplication_idempotency(temp_db):
    """Submitting the same deduplication key multiple times does NOT create duplicate records."""
    activate_test_epoch(temp_db, epoch_id="EPOCH_P1_TEST_ACTIVE", releases=[PROD_RELEASE])

    req_payload = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "44444444-4444-4444-8444-444444444444",
            "session_id": "55555555-5555-4555-8555-555555555555",
            "observation_unit_id": "66666666-6666-4666-8666-666666666666",
            "deduplication_key": "e" * 64,
            "timestamp": "2026-10-01T12:05:00Z",
            "normalized_symbol": "QQQ",
            "intent_type": "ETF_SYMBOL_SELECT",
            "route": "/",
            "source_component": "handleSelectSymbol",
        },
        "audit_record": {
            "observation_unit_id": "66666666-6666-4666-8666-666666666666",
            "event_id": "44444444-4444-4444-8444-444444444444",
            "session_id": "55555555-5555-4555-8555-555555555555",
            "timestamp": "2026-10-01T12:05:00Z",
            "normalized_symbol": "QQQ",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural candidate",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "e" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_P1_TEST_ACTIVE",
        },
    }

    # First insertion
    res1 = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers(),
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "PERSISTED"

    # Second insertion with same deduplication_key
    res2 = client.post(
        f"/api/v1/telemetry/persist?db_path={temp_db}",
        json=req_payload,
        headers=auth_headers(),
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "DUPLICATE"

    # Audit records table must still have exactly 1 record
    with get_db_connection(temp_db) as conn:
        audit_count = conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_audit_records;").fetchone()["c"]
        assert audit_count == 1


def test_epoch_scoped_denominator_derivation(temp_db):
    """Prospective denominator dynamically counts distinct valid observation units in active epoch."""
    activate_test_epoch(temp_db, epoch_id="EPOCH_P1_TEST_ACTIVE", releases=[PROD_RELEASE])

    # 1. Insert valid observation 1
    req1 = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "e1-1111-4111-8111-111111111111",
            "session_id": "s1-2222-4222-8222-222222222222",
            "observation_unit_id": "obs-1111-4333-8333-333333333333",
            "deduplication_key": "f" * 64,
            "timestamp": "2026-10-01T14:00:00Z",
            "normalized_symbol": "SPY",
        },
        "audit_record": {
            "observation_unit_id": "obs-1111-4333-8333-333333333333",
            "event_id": "e1-1111-4111-8111-111111111111",
            "session_id": "s1-2222-4222-8222-222222222222",
            "timestamp": "2026-10-01T14:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "f" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_P1_TEST_ACTIVE",
        },
    }
    client.post(f"/api/v1/telemetry/persist?db_path={temp_db}", json=req1, headers=auth_headers())

    # 2. Insert valid observation 2
    req2 = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "e2-1111-4111-8111-111111111111",
            "session_id": "s2-2222-4222-8222-222222222222",
            "observation_unit_id": "obs-2222-4333-8333-333333333333",
            "deduplication_key": "0" * 64,
            "timestamp": "2026-10-01T14:05:00Z",
            "normalized_symbol": "QQQ",
        },
        "audit_record": {
            "observation_unit_id": "obs-2222-4333-8333-333333333333",
            "event_id": "e2-1111-4111-8111-111111111111",
            "session_id": "s2-2222-4222-8222-222222222222",
            "timestamp": "2026-10-01T14:05:00Z",
            "normalized_symbol": "QQQ",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean natural attempt",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "0" * 64,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_P1_TEST_ACTIVE",
        },
    }
    client.post(f"/api/v1/telemetry/persist?db_path={temp_db}", json=req2, headers=auth_headers())

    # 3. Query denominator
    res_denom = client.get(f"/api/v1/telemetry/denominator?db_path={temp_db}")
    assert res_denom.status_code == 200
    data = res_denom.json()
    assert data["prospective_denominator"] == 2
    assert data["active_epoch"] == "EPOCH_P1_TEST_ACTIVE"
    assert data["epoch_status"] == "ACTIVE"
    assert data["authority"] == "DERIVED_FROM_CANONICAL_AUDIT_LEDGER"


def test_multi_attempt_semantics_distinct(temp_db):
    """Section 21: Distinct user attempts with distinct observation unit IDs both count distinctly."""
    activate_test_epoch(temp_db, epoch_id="EPOCH_P1_TEST_ACTIVE", releases=[PROD_RELEASE])

    payload1 = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "evt-attempt-1",
            "session_id": "ses-common-01",
            "observation_unit_id": "obs-attempt-1",
            "deduplication_key": "11" * 32,
            "timestamp": "2026-10-01T15:00:00Z",
            "normalized_symbol": "SPY",
        },
        "audit_record": {
            "observation_unit_id": "obs-attempt-1",
            "event_id": "evt-attempt-1",
            "session_id": "ses-common-01",
            "timestamp": "2026-10-01T15:00:00Z",
            "normalized_symbol": "SPY",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean attempt 1",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "11" * 32,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_P1_TEST_ACTIVE",
        },
    }

    payload2 = {
        "raw_event": {
            "schema_version": "1.0.0",
            "event_id": "evt-attempt-2",
            "session_id": "ses-common-01",
            "observation_unit_id": "obs-attempt-2",
            "deduplication_key": "22" * 32,
            "timestamp": "2026-10-01T15:05:00Z",
            "normalized_symbol": "IVV",
        },
        "audit_record": {
            "observation_unit_id": "obs-attempt-2",
            "event_id": "evt-attempt-2",
            "session_id": "ses-common-01",
            "timestamp": "2026-10-01T15:05:00Z",
            "normalized_symbol": "IVV",
            "environment": "production",
            "deployment_identity": "https://www.arxterminal.com",
            "release_sha": PROD_RELEASE,
            "traffic_class": "NATURAL_PRODUCTION",
            "classification_state": "VALID",
            "classification_reason": "Clean attempt 2",
            "exclusion_code": "NONE",
            "quarantine_reason": "NONE",
            "deduplication_key": "22" * 32,
            "replay_identity": "NONE",
            "source_component": "handleSelectSymbol",
            "epoch_id": "EPOCH_P1_TEST_ACTIVE",
        },
    }

    res1 = client.post(f"/api/v1/telemetry/persist?db_path={temp_db}", json=payload1, headers=auth_headers())
    res2 = client.post(f"/api/v1/telemetry/persist?db_path={temp_db}", json=payload2, headers=auth_headers())
    assert res1.status_code == 200
    assert res2.status_code == 200

    res_denom = client.get(f"/api/v1/telemetry/denominator?db_path={temp_db}")
    assert res_denom.json()["prospective_denominator"] == 2


def test_rejected_requests_produce_zero_side_effects(temp_db):
    """Section 33: Rejected requests produce zero database side effects."""
    bad_requests = [
        ({}, auth_headers("wrong")),
        ({}, {}),
        (
            {
                "raw_event": {
                    "schema_version": "1.0.0",
                    "event_id": "bad-1",
                    "session_id": "bad-1",
                    "observation_unit_id": "obs-bad-1",
                    "deduplication_key": "33" * 32,
                    "timestamp": "2026-10-01T12:00:00Z",
                    "normalized_symbol": "SPY",
                },
                "audit_record": {
                    "observation_unit_id": "obs-bad-1",
                    "event_id": "bad-1",
                    "session_id": "bad-1",
                    "timestamp": "2026-10-01T12:00:00Z",
                    "normalized_symbol": "SPY",
                    "environment": "production",
                    "deployment_identity": "https://www.arxterminal.com",
                    "release_sha": "unauthorized",
                    "traffic_class": "NATURAL_PRODUCTION",
                    "classification_state": "VALID",
                    "classification_reason": "Bad",
                    "exclusion_code": "NONE",
                    "quarantine_reason": "NONE",
                    "deduplication_key": "33" * 32,
                    "replay_identity": "NONE",
                    "source_component": "handleSelectSymbol",
                },
            },
            auth_headers(),
        ),
    ]

    for body, headers in bad_requests:
        client.post(f"/api/v1/telemetry/persist?db_path={temp_db}", json=body, headers=headers)

    with get_db_connection(temp_db) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_raw_events;").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM arx_p1_telemetry_audit_records;").fetchone()["c"] == 0

    denom = client.get(f"/api/v1/telemetry/denominator?db_path={temp_db}").json()
    assert denom["prospective_denominator"] == 0


def test_read_only_audit_endpoint(temp_db):
    """Verify /api/v1/telemetry/audit returns filtered audit records without mutation."""
    res = client.get(f"/api/v1/telemetry/audit?db_path={temp_db}&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "count" in data
    assert "records" in data
    assert isinstance(data["records"], list)


def test_read_only_epochs_endpoint(temp_db):
    """Verify /api/v1/telemetry/epochs lists epochs without mutation."""
    res = client.get(f"/api/v1/telemetry/epochs?db_path={temp_db}")
    assert res.status_code == 200
    data = res.json()
    assert "epochs" in data
    assert isinstance(data["epochs"], list)


def test_canonical_telemetry_ingress_must_be_included_in_cloudflare_pages_deployment():
    """Regression Guard: Ensure canonical etf-intent edge function is located under frontend/functions/
    so Cloudflare Pages (project root: frontend) discovers and deploys the /api/telemetry/etf-intent route.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cf_func_path = os.path.join(repo_root, "frontend", "functions", "api", "telemetry", "etf-intent.ts")
    assert os.path.isfile(cf_func_path), f"Missing Cloudflare Pages ingress function: {cf_func_path}"

    # Verify stale handler at repo root is removed
    root_func_path = os.path.join(repo_root, "functions", "api", "telemetry", "etf-intent.ts")
    assert not os.path.exists(root_func_path), f"Stale ingress handler still exists at root: {root_func_path}"

    with open(cf_func_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'from "../../../lib/telemetry/etfDenominatorEngine"' in content
    assert "export const onRequest" in content
    assert "/api/telemetry/etf-intent" in content
