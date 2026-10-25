"""Tests for SEC User-Agent configuration, validation, and fair-access enforcement."""

import os
from unittest.mock import patch, MagicMock
import pytest

from analyst_dashboard.data.sec_edgar_fetcher import (
    validate_sec_user_agent,
    get_sec_user_agent,
    SecEdgarFetcher,
)
from analyst_dashboard.smart_money.form4_ingestion import (
    Form4IngestionManager,
    RateLimiter,
)


def test_valid_sec_user_agent():
    valid_ua = "ArxTerminalResearchApp compliance@arxterminal.com"
    is_valid, msg = validate_sec_user_agent(valid_ua)
    assert is_valid is True
    assert msg == "Valid"


def test_missing_and_empty_sec_user_agent():
    assert validate_sec_user_agent(None)[0] is False
    assert validate_sec_user_agent("")[0] is False
    assert validate_sec_user_agent("   ")[0] is False


def test_placeholder_sec_user_agent():
    # Disallowed default placeholder
    assert validate_sec_user_agent("FinanceTerminalResearchApp admin@financeterminal.internal")[0] is False
    assert validate_sec_user_agent("SampleApp test@example.com")[0] is False
    assert validate_sec_user_agent("MyCompany contact@sample.com")[0] is False
    assert validate_sec_user_agent("App user@test.local")[0] is False


def test_missing_email_in_user_agent():
    assert validate_sec_user_agent("JustAnAppName")[0] is False
    assert validate_sec_user_agent("AppNameWithoutEmail123")[0] is False


def test_missing_app_name_in_user_agent():
    assert validate_sec_user_agent("only_email@validcompany.org")[0] is False


def test_sec_edgar_fetcher_unconfigured_blocks_network():
    with patch.dict(os.environ, {}, clear=True):
        fetcher = SecEdgarFetcher()
        assert fetcher.is_configured is False
        assert fetcher.user_agent is None
        assert "User-Agent" not in fetcher.session.headers

        # resolve_cik for unknown symbol blocks without network request
        with patch.object(fetcher.session, "get") as mock_get:
            result = fetcher.resolve_cik("UNKNOWN_XYZ")
            assert result is None
            mock_get.assert_not_called()

        # get_company_submissions blocks without network request
        with patch.object(fetcher.session, "get") as mock_get:
            result = fetcher.get_company_submissions("AAPL")
            assert result is None
            mock_get.assert_not_called()


def test_sec_edgar_fetcher_header_propagation():
    valid_ua = "ArxTerminalProductionResearch ops@arxterminal.com"
    with patch.dict(os.environ, {"SEC_USER_AGENT": valid_ua}):
        fetcher = SecEdgarFetcher()
        assert fetcher.is_configured is True
        assert fetcher.user_agent == valid_ua
        assert fetcher.session.headers.get("User-Agent") == valid_ua

        # When network request is made with mock, header is preserved
        with patch.object(fetcher.session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"filings": {"recent": {}}}
            mock_get.return_value = mock_resp

            fetcher.get_company_submissions("AAPL")
            mock_get.assert_called_once()
            assert fetcher.session.headers["User-Agent"] == valid_ua


def test_form4_ingestion_manager_unconfigured_blocks_network():
    with patch.dict(os.environ, {}, clear=True):
        mgr = Form4IngestionManager(db_path=":memory:")
        assert mgr.is_configured is False
        with patch.object(mgr.session, "get") as mock_get:
            res = mgr._fetch_url_with_retry("https://data.sec.gov/submissions/CIK0000320193.json")
            assert res is None
            mock_get.assert_not_called()


def test_form4_ingestion_manager_rate_limiter_preserved():
    rl = RateLimiter(max_per_second=100.0)
    assert rl.interval == 0.01
    rl.acquire()
    assert rl.last_request_time > 0.0
