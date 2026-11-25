"""
ARX Issue #165 — Offline Test Suite for NVDA Submissions Diagnostic Acquisition.

Validates the bounded diagnostic acquisition contract:
1. Mocked Success: Exact raw bytes preserved, SHA-256 computed, schema/issuer/pagination validated,
   single request enforced, durable auth ID consumed atomically.
2. Mocked Timeout: Timeout handled cleanly, error recorded in persistent receipt, zero retries,
   auth ID remains durably consumed.
3. Mocked Redirect: Redirect refused (allow_redirects=False), status recorded, zero retries.
4. Mocked HTTP Error: 429 / 500 error recorded in persistent receipt, zero retries.
5. Mocked Malformed Response: Exact bytes preserved, SHA-256 recorded, JSON parse error captured.
6. Strict Budget Enforcement: Exactly one request allowed per diagnostic handler instance.
7. Catalog Validation Invariants: Issuer identity, schema, pagination, and interval coverage validated;
   latest accession date alone is never treated as proof of completeness.
8. Preserved Governance State & Intended Boundaries: Production snapshot and holds are explicitly
   labeled as INTENDED_BOUNDARY_UNVERIFIED_IN_DIAGNOSTIC; fixture tests do not establish live state.
9. Process Restart Safety: Durable authorization ID claimed atomically on disk survives process
   restarts; re-instantiated handler fails closed before network egress.
10. Concurrent Invocation Protection: Multi-threaded / multi-process execution with identical auth ID
    permits exactly one transmission; all others fail closed atomically.
11. Failed Transmission Token Consumption: Network timeout / failure consumes token; subsequent attempt
    with same auth ID fails before network egress.
12. Missing Auth ID: Rejection before network egress if auth ID is missing or empty.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.acquire_nvda_submissions_diagnostic import (
    DiagnosticAcquisitionHandler,
    DiagnosticBudgetExceededError,
    AuthorizationAlreadyConsumedError,
    AuthorizationMissingError,
    TARGET_URL,
    TARGET_CIK,
    TARGET_TICKER,
    KNOWN_IN_WINDOW_ACCESSIONS,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sec_edgar"
SUBMISSIONS_FIXTURE_PATH = FIXTURES_DIR / "CIK0001045810_submissions.json"
TEST_USER_AGENT = "ARXTerminal Research Akara.david26@gmail.com"
DEFAULT_TEST_AUTH_ID = "AUTH-NVDA-TEST-20261125-001"


@pytest.fixture
def authentic_submissions_bytes() -> bytes:
    """Authentic submissions catalog bytes from repository test fixtures."""
    assert SUBMISSIONS_FIXTURE_PATH.exists(), f"Missing fixture at {SUBMISSIONS_FIXTURE_PATH}"
    return SUBMISSIONS_FIXTURE_PATH.read_bytes()


@pytest.fixture
def evidence_tmp_dir(tmp_path: Path) -> Path:
    """Isolated temporary evidence directory for tests."""
    out = tmp_path / "sec_coverage_evidence" / "CIK0001045810"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ==============================================================================
# 1. Mocked Success Case
# ==============================================================================

def test_mocked_success(authentic_submissions_bytes: bytes, evidence_tmp_dir: Path):
    """
    Simulates a successful HTTP 200 response. Confirms exact raw bytes are saved,
    SHA-256 is computed, receipt is stored, catalog validation passes, durable
    authorization claim is recorded on disk, and exactly 1 attempt is made.
    """
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.content = authentic_submissions_bytes
    mock_resp.headers = {"Content-Type": "application/json"}

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-SUCCESS-001",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )

    receipt = handler.execute_single_request(session=mock_session)

    # 1. Single attempt verification
    assert handler.attempt_count == 1
    mock_session.get.assert_called_once_with(
        TARGET_URL,
        headers={"User-Agent": TEST_USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"},
        timeout=15.0,
        allow_redirects=False,
    )

    # 2. Receipt metadata
    assert receipt["success"] is True
    assert receipt["http_status"] == 200
    assert receipt["auth_id"] == "AUTH-TEST-SUCCESS-001"
    assert receipt["byte_count"] == len(authentic_submissions_bytes)
    expected_hash = hashlib.sha256(authentic_submissions_bytes).hexdigest()
    assert receipt["sha256"] == expected_hash

    # 3. Retained raw file verification
    raw_file = Path(receipt["raw_artifact_path"])
    full_raw_path = evidence_tmp_dir / raw_file.name
    assert full_raw_path.exists()
    assert full_raw_path.read_bytes() == authentic_submissions_bytes

    # 4. Retained receipt verification
    receipt_file = Path(receipt["receipt_path"])
    full_receipt_path = evidence_tmp_dir / receipt_file.name
    assert full_receipt_path.exists()

    # 5. On-disk auth claim verification
    claim_path = evidence_tmp_dir / "auth_claims" / "AUTH-TEST-SUCCESS-001.claim"
    assert claim_path.exists()
    claim_data = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim_data["auth_id"] == "AUTH-TEST-SUCCESS-001"
    assert claim_data["status"] == "CONSUMED_INTENT"

    # 6. Catalog validation verification
    val = receipt["validation"]
    assert val["is_valid"] is True
    assert val["issuer_identity_verified"] is True
    assert val["schema_verified"] is True
    assert val["pagination_verified"] is True
    assert val["interval_coverage_verified"] is True


# ==============================================================================
# 2. Mocked Timeout Case
# ==============================================================================

def test_mocked_timeout(evidence_tmp_dir: Path):
    """
    Simulates request timeout. Confirms error is recorded in persistent receipt,
    zero retries occur, and auth ID is durably marked consumed.
    """
    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out after 15.0s")

    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-TIMEOUT-001",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
        timeout_sec=15.0,
    )

    receipt = handler.execute_single_request(session=mock_session)

    assert handler.attempt_count == 1
    mock_session.get.assert_called_once()
    assert receipt["success"] is False
    assert receipt["http_status"] is None
    assert "TIMEOUT" in receipt["error"]

    # Verify receipt is persisted even on failure
    receipt_file = Path(receipt["receipt_path"])
    full_receipt_path = evidence_tmp_dir / receipt_file.name
    assert full_receipt_path.exists()

    # Auth claim was recorded atomically before timeout
    claim_path = evidence_tmp_dir / "auth_claims" / "AUTH-TEST-TIMEOUT-001.claim"
    assert claim_path.exists()


# ==============================================================================
# 3. Mocked Redirect Case
# ==============================================================================

def test_mocked_redirect(evidence_tmp_dir: Path):
    """
    Simulates HTTP 301/302 redirect. Confirms allow_redirects=False prevents following,
    redirect is rejected per contract, receipt is recorded, and zero retries occur.
    """
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 301
    mock_resp.headers = {"Location": "https://redirect.sec.gov/alternate"}
    mock_resp.content = b"<html>301 Moved Permanently</html>"

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-REDIRECT-001",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )

    receipt = handler.execute_single_request(session=mock_session)

    assert handler.attempt_count == 1
    mock_session.get.assert_called_once()
    assert receipt["success"] is False
    assert receipt["http_status"] == 301
    assert "REDIRECT_REFUSED" in receipt["error"]

    # Raw bytes of redirect response are persisted
    assert receipt["byte_count"] > 0
    assert receipt["sha256"] is not None


# ==============================================================================
# 4. Mocked HTTP Error Cases (429 and 500)
# ==============================================================================

@pytest.mark.parametrize("status_code", [429, 500, 403])
def test_mocked_http_error(status_code: int, evidence_tmp_dir: Path):
    """
    Simulates HTTP 429, 403, and 500 errors. Confirms errors are captured,
    receipt is written, and no automatic retry is attempted.
    """
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.content = f"Error {status_code}".encode("utf-8")

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    handler = DiagnosticAcquisitionHandler(
        auth_id=f"AUTH-TEST-HTTP-{status_code}-001",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )

    receipt = handler.execute_single_request(session=mock_session)

    assert handler.attempt_count == 1
    mock_session.get.assert_called_once()
    assert receipt["success"] is False
    assert receipt["http_status"] == status_code
    assert "HTTP_ERROR" in receipt["error"]
    assert receipt["byte_count"] > 0


# ==============================================================================
# 5. Mocked Malformed Response Case
# ==============================================================================

def test_mocked_malformed_response(evidence_tmp_dir: Path):
    """
    Simulates corrupted or truncated JSON response. Confirms exact corrupted bytes
    are preserved, SHA-256 is recorded, and JSON parse failure is captured in receipt.
    """
    corrupted_bytes = b"{\"cik\": \"0001045810\", \"filings\": { TRUNCATED..."

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.content = corrupted_bytes

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-MALFORMED-001",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )

    receipt = handler.execute_single_request(session=mock_session)

    assert handler.attempt_count == 1
    assert receipt["success"] is False
    assert receipt["http_status"] == 200
    assert receipt["byte_count"] == len(corrupted_bytes)
    assert receipt["sha256"] == hashlib.sha256(corrupted_bytes).hexdigest()
    assert "MALFORMED_JSON" in receipt["error"]

    # Corrupted bytes are preserved on disk for forensic audit
    raw_file = Path(receipt["raw_artifact_path"])
    full_raw_path = evidence_tmp_dir / raw_file.name
    assert full_raw_path.read_bytes() == corrupted_bytes


# ==============================================================================
# 6. Single Request Budget Enforcement (In-Memory Handler Instance)
# ==============================================================================

def test_single_request_budget_enforcement(authentic_submissions_bytes: bytes, evidence_tmp_dir: Path):
    """
    Confirms each handler instance can attempt at most one request.
    Any second attempt raises DiagnosticBudgetExceededError without making network calls.
    """
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.content = authentic_submissions_bytes

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-BUDGET-001",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )

    # First request: PASS
    receipt1 = handler.execute_single_request(session=mock_session)
    assert handler.attempt_count == 1

    # Second request on same handler instance: FAILS CLOSED with DiagnosticBudgetExceededError
    with pytest.raises(DiagnosticBudgetExceededError) as exc_info:
        handler.execute_single_request(session=mock_session)

    assert "DIAGNOSTIC_BUDGET_EXCEEDED" in str(exc_info.value)
    # Confirm session.get was only invoked exactly once
    assert mock_session.get.call_count == 1


# ==============================================================================
# 7. Process Restart Safety (Durable Auth ID Survives Restarts)
# ==============================================================================

def test_auth_id_survives_process_restart(authentic_submissions_bytes: bytes, evidence_tmp_dir: Path):
    """
    Simulates a process restart:
    1. Handler 1 claims AUTH-RESTART-001 and successfully executes.
    2. Handler 2 is newly instantiated with attempt_count=0 (fresh process memory),
       attempting the SAME AUTH-RESTART-001.
    3. Handler 2 fails closed with AuthorizationAlreadyConsumedError BEFORE any network call.
    4. session.get is called exactly ONCE across both handler lifecycles.
    """
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.content = authentic_submissions_bytes

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    test_auth_id = "AUTH-TEST-RESTART-001"

    # Process 1
    handler1 = DiagnosticAcquisitionHandler(
        auth_id=test_auth_id,
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )
    receipt1 = handler1.execute_single_request(session=mock_session)
    assert receipt1["success"] is True
    assert mock_session.get.call_count == 1

    # Process 2 (simulated restart: fresh handler instance with attempt_count=0)
    handler2 = DiagnosticAcquisitionHandler(
        auth_id=test_auth_id,
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )
    assert handler2.attempt_count == 0

    with pytest.raises(AuthorizationAlreadyConsumedError) as exc_info:
        handler2.execute_single_request(session=mock_session)

    assert "AUTHORIZATION_ALREADY_CONSUMED" in str(exc_info.value)
    # Verify no second HTTP request was transmitted
    assert mock_session.get.call_count == 1


# ==============================================================================
# 8. Concurrent Invocation Protection
# ==============================================================================

def test_auth_id_concurrent_invocation_protection(authentic_submissions_bytes: bytes, evidence_tmp_dir: Path):
    """
    Simulates concurrent execution across multiple threads/processes attempting to use the same auth ID.
    Confirms atomic os.O_CREAT | os.O_EXCL ensures exactly ONE invocation transmits while all other
    concurrent invocations fail closed with AuthorizationAlreadyConsumedError.
    """
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.content = authentic_submissions_bytes

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    test_auth_id = "AUTH-TEST-CONCURRENT-001"
    num_concurrent = 8

    success_receipts = []
    consumed_errors = []

    def run_worker():
        handler = DiagnosticAcquisitionHandler(
            auth_id=test_auth_id,
            user_agent=TEST_USER_AGENT,
            evidence_dir=evidence_tmp_dir,
        )
        return handler.execute_single_request(session=mock_session)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(run_worker) for _ in range(num_concurrent)]
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                success_receipts.append(res)
            except AuthorizationAlreadyConsumedError as exc:
                consumed_errors.append(exc)

    # Exactly 1 must have succeeded
    assert len(success_receipts) == 1
    # Exactly (num_concurrent - 1) must have failed closed with AuthorizationAlreadyConsumedError
    assert len(consumed_errors) == num_concurrent - 1
    # session.get must have been invoked exactly ONCE in total
    assert mock_session.get.call_count == 1


# ==============================================================================
# 9. Failed Transmission Token Consumption
# ==============================================================================

def test_auth_id_consumed_on_failed_transmission(evidence_tmp_dir: Path):
    """
    Confirms that if a transmission fails (e.g. timeout or server 500), the authorization
    ID remains durably consumed. Re-attempting with that same ID fails before network egress.
    """
    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    test_auth_id = "AUTH-TEST-FAILED-TRANS-001"

    handler1 = DiagnosticAcquisitionHandler(
        auth_id=test_auth_id,
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
        timeout_sec=5.0,
    )
    receipt1 = handler1.execute_single_request(session=mock_session)
    assert receipt1["success"] is False
    assert mock_session.get.call_count == 1

    # Second handler attempting same auth ID after timeout
    handler2 = DiagnosticAcquisitionHandler(
        auth_id=test_auth_id,
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )
    with pytest.raises(AuthorizationAlreadyConsumedError) as exc_info:
        handler2.execute_single_request(session=mock_session)

    assert "AUTHORIZATION_ALREADY_CONSUMED" in str(exc_info.value)
    # Network call count must remain 1
    assert mock_session.get.call_count == 1


# ==============================================================================
# 10. Missing Auth ID Fails Preconditions
# ==============================================================================

def test_missing_auth_id_fails_preconditions(evidence_tmp_dir: Path):
    """Handler fails closed if no auth ID is provided."""
    handler = DiagnosticAcquisitionHandler(
        auth_id="",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )
    valid, reason = handler.validate_preconditions()
    assert valid is False
    assert "MISSING_AUTH_ID" in reason

    mock_session = MagicMock(spec=requests.Session)
    with pytest.raises(ValueError) as exc_info:
        handler.execute_single_request(session=mock_session)
    assert "PRECONDITIONS_FAILED" in str(exc_info.value)
    assert mock_session.get.call_count == 0


# ==============================================================================
# 11. Catalog Validation Invariants & Anti-Inference Principle
# ==============================================================================

def test_catalog_validation_wrong_issuer(evidence_tmp_dir: Path):
    """Validation fails if catalog CIK or ticker does not match NVDA."""
    data = {
        "cik": "0000320193",  # AAPL
        "tickers": ["AAPL"],
        "filings": {"recent": {"accessionNumber": [], "filingDate": [], "acceptanceDateTime": [], "form": []}},
    }
    raw = json.dumps(data).encode("utf-8")
    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-WRONG-CIK",
        evidence_dir=evidence_tmp_dir,
    )
    res = handler.validate_catalog_bytes(raw, hashlib.sha256(raw).hexdigest())

    assert res["is_valid"] is False
    assert "WRONG_CIK" in res["error"]


def test_catalog_validation_missing_schema_columns(evidence_tmp_dir: Path):
    """Validation fails if required schema columns are missing."""
    data = {
        "cik": TARGET_CIK,
        "tickers": [TARGET_TICKER],
        "filings": {"recent": {"accessionNumber": ["0001-26-000001"]}},  # Missing filingDate, etc.
    }
    raw = json.dumps(data).encode("utf-8")
    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-MISSING-COL",
        evidence_dir=evidence_tmp_dir,
    )
    res = handler.validate_catalog_bytes(raw, hashlib.sha256(raw).hexdigest())

    assert res["is_valid"] is False
    assert "MALFORMED_COLUMNS" in res["error"]


def test_catalog_validation_overlapping_pagination_files(evidence_tmp_dir: Path):
    """
    If historical pagination files overlap the assessment interval, reports required
    pagination evidence without attempting additional downloads.
    """
    data = {
        "cik": TARGET_CIK,
        "tickers": [TARGET_TICKER],
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-000001"],
                "filingDate": ["2026-11-20"],
                "acceptanceDateTime": ["2026-11-20T21:00:00Z"],
                "form": ["4"],
            },
            "files": [
                {
                    "name": "CIK0001045810-submissions-002.json",
                    "filingFrom": "2026-09-01",
                    "filingTo": "2026-10-15",
                }
            ],
        },
    }
    raw = json.dumps(data).encode("utf-8")
    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-OVERLAP-FILES",
        evidence_dir=evidence_tmp_dir,
    )
    res = handler.validate_catalog_bytes(raw, hashlib.sha256(raw).hexdigest())

    assert res["is_valid"] is False
    assert res["pagination_verified"] is False
    assert "PAGINATION_EVIDENCE_REQUIRED" in res["error"]
    assert len(res["overlapping_pagination_files"]) == 1


def test_catalog_validation_anti_inference_rule(evidence_tmp_dir: Path):
    """
    Confirms anti-inference rule: a latest accession date within the window does NOT prove
    completeness if the catalog does not span back to interval start.
    """
    data = {
        "cik": TARGET_CIK,
        "tickers": [TARGET_TICKER],
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-000001"],
                "filingDate": ["2026-11-24"],  # Late date in interval
                "acceptanceDateTime": ["2026-11-24T21:00:00Z"],
                "form": ["4"],
            },
            "files": [],
        },
    }
    raw = json.dumps(data).encode("utf-8")
    # Interval start is 2026-08-26, but earliest filing in recent is 2026-11-24
    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-ANTI-INFERENCE",
        evidence_dir=evidence_tmp_dir,
    )
    res = handler.validate_catalog_bytes(raw, hashlib.sha256(raw).hexdigest())

    assert res["is_valid"] is False
    assert res["interval_coverage_verified"] is False
    assert "INTERVAL_INCOMPLETE" in res["error"]


# ==============================================================================
# 12. Preserved Governance State & Explicit Intended Boundaries
# ==============================================================================

def test_preserved_governance_state(authentic_submissions_bytes: bytes, evidence_tmp_dir: Path):
    """
    Confirms the receipt records preserved governance invariants and explicitly labels
    production snapshot and holds as intended boundaries unverified by offline diagnostic scripts.
    """
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.content = authentic_submissions_bytes

    mock_session = MagicMock(spec=requests.Session)
    mock_session.get.return_value = mock_resp

    handler = DiagnosticAcquisitionHandler(
        auth_id="AUTH-TEST-GOVERNANCE-001",
        user_agent=TEST_USER_AGENT,
        evidence_dir=evidence_tmp_dir,
    )

    receipt = handler.execute_single_request(session=mock_session)

    # 1. Verification of explicit intended boundaries
    bounds = receipt["governance_boundaries"]
    assert bounds["verification_status"] == "INTENDED_BOUNDARY_UNVERIFIED_IN_DIAGNOSTIC"
    assert "offline" in bounds["boundary_notes"].lower() or "unverified" in bounds["boundary_notes"].lower()
    assert bounds["intended_production_coverage_status"] == "PARTIAL_UNCERTIFIED"
    assert bounds["intended_active_snapshot_id"] == "SNAP_SM_1792939989_48f30db6"
    assert bounds["database_writes_performed"] == 0
    assert bounds["worker_activation"] is False
    assert "HOLD_SEC_001" in bounds["intended_operational_holds"]
    assert "HOLD_EXEC_001" in bounds["intended_operational_holds"]

    # 2. Compatibility check
    gov = receipt["preserved_governance_state"]
    assert gov["verification_status"] == "INTENDED_BOUNDARY_UNVERIFIED_IN_DIAGNOSTIC"
    assert gov["intended_production_coverage_status"] == "PARTIAL_UNCERTIFIED"
    assert gov["intended_active_snapshot_id"] == "SNAP_SM_1792939989_48f30db6"
    assert gov["database_writes_performed"] == 0
    assert gov["worker_activation"] is False
    assert "HOLD_SEC_001" in gov["intended_operational_holds"]
