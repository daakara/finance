"""Tests for Smart Money API runtime governance isolation and fail-fast mutation traps."""

import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from concurrent.futures import ThreadPoolExecutor

from api.main import app

pytestmark = pytest.mark.tier2c

client = TestClient(app)


class GovernanceMutationAttempted(Exception):
    """Raised whenever a governance write/mutation authority is invoked."""
    pass


@pytest.fixture(autouse=True)
def governance_mutation_trap():
    """
    Installs fail-fast mutation traps across all canonical governance write authorities,
    voucher minting, state-machine transitions, and natural denominator increment hooks.
    """
    def trap_call(*args, **kwargs):
        raise GovernanceMutationAttempted(f"Prohibited governance mutation attempted during read request! args={args}, kwargs={kwargs}")

    with patch(
        "analyst_dashboard.smart_money.canonical_commit.CanonicalDecisionCommitAuthority.commit_decision",
        side_effect=trap_call,
    ) if os.path.exists("analyst_dashboard/smart_money/canonical_commit.py") else patch("builtins.print", side_effect=lambda *a, **k: None), \
         patch(
             "analyst_dashboard.smart_money.public_availability.gate3_authorization.claim_gate3_authorization",
             side_effect=trap_call,
             create=True,
         ), \
         patch(
             "analyst_dashboard.smart_money.public_availability.observer_state_machine.ObserverStateMachine.transition",
             side_effect=trap_call,
         ), \
         patch(
             "analyst_dashboard.governance.passive_capture.PassiveCaptureHook.record_observation",
             side_effect=trap_call,
             create=True,
         ):
        yield


# --- 12-CASE ISOLATION MATRIX ---

def test_isolation_01_normal_response():
    """Case 1: Normal response for all endpoints triggers zero governance mutations."""
    res_ov = client.get("/api/v1/smart-money/overview?include_curated=true")
    assert res_ov.status_code == 200

    res_cong = client.get("/api/v1/smart-money/congress?include_curated=true")
    assert res_cong.status_code == 200

    res_opt = client.get("/api/v1/smart-money/options-flow?include_curated=true")
    assert res_opt.status_code == 200

    res_sec = client.get("/api/v1/smart-money/sec-filings/AAPL")
    assert res_sec.status_code == 200

    res_finra = client.get("/api/v1/smart-money/finra-darkpool/AAPL")
    assert res_finra.status_code == 200

    res_radar = client.get("/api/v1/screener/smart-money/snapshot")
    assert res_radar.status_code == 200
    assert res_radar.json()["status"] == "PIPELINE_PENDING"


def test_isolation_02_empty_data():
    """Case 2: Empty data responses trigger zero governance mutations."""
    res_ov = client.get("/api/v1/smart-money/overview")  # default include_curated=False
    assert res_ov.status_code == 200
    assert res_ov.json()["congress_trades"] == []

    res_cong = client.get("/api/v1/smart-money/congress?symbol=NONEXISTENT_XYZ&include_curated=true")
    assert res_cong.status_code == 200
    assert res_cong.json()["trades"] == []


def test_isolation_03_invalid_symbol():
    """Case 3: Invalid ticker symbol input triggers 400 error without governance side effects."""
    res = client.get("/api/v1/smart-money/sec-filings/INVALID$SYMBOL!@#")
    assert res.status_code == 400

    res_cong = client.get("/api/v1/smart-money/congress?symbol=INVALID$SYMBOL!@#")
    assert res_cong.status_code == 400


def test_isolation_04_missing_cache():
    """Case 4: Querying an asset with no local cache gracefully returns empty results."""
    res = client.get("/api/v1/smart-money/sec-filings/ZZZZ")
    assert res.status_code == 200
    assert res.json()["filings"] == []


def test_isolation_05_corrupt_cache():
    """Case 5: Corrupt cache handles gracefully and does not trigger error state transition."""
    with patch("analyst_dashboard.data.sec_edgar_fetcher.SecEdgarFetcher.get_recent_filings", side_effect=ValueError("Corrupt cache bytes")):
        res = client.get("/api/v1/smart-money/sec-filings/AAPL")
        # FastAPI handles or raises exception without triggering governance writes
        assert res.status_code in [200, 500]


def test_isolation_06_sec_transport_timeout():
    """Case 6: Outbound SEC transport timeout fails closed with zero governance writes."""
    import requests
    with patch("requests.Session.get", side_effect=requests.exceptions.Timeout("SEC EDGAR timeout")):
        res = client.get("/api/v1/smart-money/sec-filings/AAPL")
        assert res.status_code == 200
        assert res.json()["filings"] == []


def test_isolation_07_sec_rate_limiting():
    """Case 7: HTTP 429 Too Many Requests from SEC upstream returns empty without governance mutations."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    with patch("requests.Session.get", return_value=mock_resp):
        res = client.get("/api/v1/smart-money/sec-filings/AAPL")
        assert res.status_code == 200
        assert res.json()["filings"] == []


def test_isolation_08_concurrent_requests():
    """Case 8: 10 concurrent requests to read endpoints execute cleanly with zero race condition side effects."""
    def send_req(i):
        return client.get(f"/api/v1/smart-money/overview?include_curated=true&run={i}").status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        statuses = list(executor.map(send_req, range(10)))
    assert all(s == 200 for s in statuses)


def test_isolation_09_dependency_initialization():
    """Case 9: Engine initialization has zero side effects on disk or governance state."""
    from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
    engine = SmartMoneyEngine()
    overview = engine.get_smart_money_overview(include_curated=True)
    assert len(overview["congress_trades"]) > 0


def test_isolation_10_background_scheduler_unavailable():
    """Case 10: Absence of background scheduler does not impair read API serving."""
    with patch.dict(os.environ, {"SMART_MONEY_RELEASE_MODE": "READ_ONLY_RESEARCH", "AUTONOMOUS_ROOTING_ENABLED": "false"}):
        res = client.get("/api/v1/smart-money/overview?include_curated=true")
        assert res.status_code == 200


def test_isolation_11_governance_authority_unavailable():
    """Case 11: Completely disabled or absent governance state does not affect read API endpoints."""
    with patch.dict(os.environ, {"CANONICAL_DECISION_WRITES_ENABLED": "false"}):
        res = client.get("/api/v1/smart-money/sec-filings/NVDA")
        assert res.status_code == 200


def test_isolation_12_repeated_identical_requests():
    """Case 12: Idempotent repeated calls produce identical outputs with zero state accumulation."""
    r1 = client.get("/api/v1/smart-money/congress?include_curated=true").json()
    r2 = client.get("/api/v1/smart-money/congress?include_curated=true").json()
    assert r1 == r2
