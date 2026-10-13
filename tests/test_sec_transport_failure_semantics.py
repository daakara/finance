"""
Unit tests for SEC Transport Failure Semantics (Section 7.1).

Proves that:
1. Valid empty SEC response != timeout.
2. Valid empty SEC response != 429 rate limit.
3. HTTP 403 != verified absence.
4. HTTP 404 endpoint absence != generalized filing absence.
5. HTTP 5xx remains indeterminate.
6. Malformed payload is not represented as zero filings.
7. API consumer receives distinguishable status envelope.
"""

from unittest.mock import patch, MagicMock
import requests
import pytest
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher

client = TestClient(app)


def test_1_valid_empty_sec_response_distinct_from_timeout():
    """1. Valid empty SEC response != timeout."""
    fetcher = SecEdgarFetcher(user_agent="ValidApp valid@domain.com")

    # Mock 200 empty submissions
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "filings": {
            "recent": {
                "form": [],
                "filingDate": [],
                "accessionNumber": [],
                "primaryDocDescription": [],
                "acceptanceDateTime": [],
                "reportDate": [],
            }
        }
    }

    with patch.object(fetcher.session, "get", return_value=mock_resp_200):
        env_empty = fetcher.get_recent_filings_envelope("AAPL")

    assert env_empty["status"] == "AVAILABLE_NO_RESULTS"
    assert env_empty["items"] == []
    assert env_empty["status_code"] == 200

    # Mock timeout
    with patch.object(fetcher.session, "get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        env_timeout = fetcher.get_recent_filings_envelope("AAPL")

    assert env_timeout["status"] == "UPSTREAM_TIMEOUT"
    assert env_timeout["items"] is None
    assert env_timeout["reason"] == "CONNECTION_TIMEOUT"

    # Distinctness proof
    assert env_empty["status"] != env_timeout["status"]
    assert env_empty["items"] != env_timeout["items"]


def test_2_valid_empty_sec_response_distinct_from_429():
    """2. Valid empty SEC response != 429 rate limit."""
    fetcher = SecEdgarFetcher(user_agent="ValidApp valid@domain.com")

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429

    with patch.object(fetcher.session, "get", return_value=mock_resp_429):
        env_429 = fetcher.get_recent_filings_envelope("AAPL")

    assert env_429["status"] == "UPSTREAM_RATE_LIMITED"
    assert env_429["status_code"] == 429
    assert env_429["items"] is None
    assert env_429["reason"] == "RATE_LIMITED"


def test_3_http_403_not_verified_absence():
    """3. HTTP 403 != verified absence."""
    fetcher = SecEdgarFetcher(user_agent="ValidApp valid@domain.com")

    mock_resp_403 = MagicMock()
    mock_resp_403.status_code = 403

    with patch.object(fetcher.session, "get", return_value=mock_resp_403):
        env_403 = fetcher.get_recent_filings_envelope("AAPL")

    assert env_403["status"] == "UNKNOWN_FAIL_CLOSED"
    assert env_403["reason"] == "HTTP_403_FORBIDDEN"
    assert env_403["items"] is None, "403 must never collapse to items: [] (empty filings)"


def test_4_http_404_not_generalized_to_global_absence():
    """4. HTTP 404 endpoint/resource absence is not generalized to global filing absence."""
    fetcher = SecEdgarFetcher(user_agent="ValidApp valid@domain.com")

    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404

    with patch.object(fetcher.session, "get", return_value=mock_resp_404):
        env_404 = fetcher.get_recent_filings_envelope("AAPL")

    assert env_404["status"] == "UPSTREAM_NOT_FOUND"
    assert env_404["reason"] == "SUBMISSIONS_RESOURCE_NOT_FOUND"
    assert env_404["items"] is None, "404 submissions resource must not be represented as valid empty list"


def test_5_http_5xx_remains_indeterminate():
    """5. HTTP 5xx remains indeterminate."""
    fetcher = SecEdgarFetcher(user_agent="ValidApp valid@domain.com")

    for code in (500, 502, 503):
        mock_resp_5xx = MagicMock()
        mock_resp_5xx.status_code = code

        with patch.object(fetcher.session, "get", return_value=mock_resp_5xx):
            env_5xx = fetcher.get_recent_filings_envelope("AAPL")

        assert env_5xx["status"] == "UPSTREAM_SERVER_ERROR"
        assert env_5xx["status_code"] == code
        assert env_5xx["items"] is None


def test_6_malformed_payload_is_structurally_invalid():
    """6. Malformed JSON/payload is not represented as zero filings."""
    fetcher = SecEdgarFetcher(user_agent="ValidApp valid@domain.com")

    mock_resp_malformed = MagicMock()
    mock_resp_malformed.status_code = 200
    mock_resp_malformed.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")

    with patch.object(fetcher.session, "get", return_value=mock_resp_malformed):
        env_malformed = fetcher.get_recent_filings_envelope("AAPL")

    assert env_malformed["status"] == "STRUCTURALLY_INVALID"
    assert env_malformed["reason"] == "MALFORMED_JSON"
    assert env_malformed["items"] is None


def test_7_frontend_receives_distinguishable_envelope():
    """7. Frontend/API consumer receives distinguishable status and items: null on upstream failure."""
    with patch("api.routes.smart_money.sec_fetcher.get_recent_filings_envelope") as mock_env:
        # Simulate rate limit failure
        mock_env.return_value = {
            "status": "UPSTREAM_RATE_LIMITED",
            "status_code": 429,
            "cik": "0000320193",
            "reason": "RATE_LIMITED",
            "error_detail": "SEC EDGAR API rate limit exceeded",
            "items": None,
        }

        res = client.get("/api/v1/smart-money/sec-filings/AAPL")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "UPSTREAM_RATE_LIMITED"
        assert data["items"] is None
        assert "rate limit exceeded" in data["message"].lower()

        # Simulate valid empty response
        mock_env.return_value = {
            "status": "AVAILABLE_NO_RESULTS",
            "status_code": 200,
            "cik": "0000320193",
            "reason": None,
            "error_detail": None,
            "items": [],
        }

        res_empty = client.get("/api/v1/smart-money/sec-filings/AAPL")
        assert res_empty.status_code == 200
        data_empty = res_empty.json()
        assert data_empty["status"] == "AVAILABLE_NO_RESULTS"
        assert data_empty["available"] is True
        assert data_empty["items"] == []
        assert data_empty["filings"] == []
        assert "0 filings match criteria" in data_empty["message"]
