"""No-network integration tests for the production default continuous SEC fetch path."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from analyst_dashboard.smart_money.continuous_pipeline import (
    ContinuousOutboundSession,
    SmartMoneyContinuousPipeline,
)

CIK = "0001045810"
SYMBOL = "NVDA"
ACCESSION = "0001045810-26-000001"
XML = '<?xml version="1.0"?><ownershipDocument><schemaVersion>X0202</schemaVersion></ownershipDocument>'


def response(request, status, body="", headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = body.encode("utf-8")
    r.url = request.url
    r.request = request
    r.headers.update(headers or {})
    return r


def catalog(forms=None):
    forms = forms if forms is not None else ["4"]
    return json.dumps({"filings": {"recent": {
        "form": forms,
        "accessionNumber": [ACCESSION] * len(forms),
        "filingDate": ["2026-11-11"] * len(forms),
        "acceptanceDateTime": ["2026-11-11T19:00:00.000Z"] * len(forms),
        "primaryDocument": ["ownership.xml"] * len(forms),
    }}})


def make_pipeline():
    p = SmartMoneyContinuousPipeline(repository=MagicMock(), allow_test_mode=True)
    p.rate_limiter.acquire = lambda: None
    p.outbound_session.rate_limiter.acquire = lambda: None
    p.outbound_session.set_current_issuer(SYMBOL)
    return p


def test_default_worker_pipeline_fetches_form4_over_guarded_session_only():
    p = make_pipeline()
    assert p.sec_fetcher is None  # matches worker's default pipeline constructor
    attempts = []

    def fake_send(adapter, request, **kwargs):
        attempts.append(request.url)
        if request.url.startswith("https://data.sec.gov/"):
            return response(request, 200, catalog())
        return response(request, 200, XML)

    with patch(
        "analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
        return_value="ARX Terminal ops@example.org",
    ), patch.object(HTTPAdapter, "send", fake_send):
        filings = p._fetch_filings_from_sec(SYMBOL, CIK, "NVIDIA CORP")

    assert len(filings) == 1
    assert filings[0]["accession_number"] == ACCESSION
    assert filings[0]["xml_content"] == XML
    assert filings[0]["content_hash"]
    assert len(attempts) == 2
    assert p.outbound_session.total_transmissions == 2
    assert p.outbound_session.issuer_transmissions[SYMBOL] == 2
    assert attempts[0] == "https://data.sec.gov/submissions/CIK0001045810.json"
    assert attempts[1].startswith("https://www.sec.gov/Archives/edgar/data/")


def test_429_fails_closed_without_hidden_retry():
    p = make_pipeline()
    attempts = []

    def fake_send(adapter, request, **kwargs):
        attempts.append(request.url)
        return response(request, 429, "Rate limited")

    with patch(
        "analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
        return_value="ARX Terminal ops@example.org",
    ), patch.object(HTTPAdapter, "send", fake_send):
        with pytest.raises(RuntimeError, match="SEC_RATE_LIMITED_429"):
            p._fetch_filings_from_sec(SYMBOL, CIK, "NVIDIA CORP")
    assert len(attempts) == 1
    assert p.outbound_session.total_transmissions == 1


def test_sec_redirect_is_counted_and_untrusted_redirect_is_blocked():
    p = make_pipeline()
    attempts = []

    def fake_send(adapter, request, **kwargs):
        attempts.append(request.url)
        if len(attempts) == 1:
            return response(request, 302, headers={
                "Location": "https://attacker.invalid/collect"})
        raise AssertionError("untrusted destination reached transport")

    with patch(
        "analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
        return_value="ARX Terminal ops@example.org",
    ), patch.object(HTTPAdapter, "send", fake_send):
        with pytest.raises(RuntimeError, match="SEC_EGRESS_DESTINATION_REJECTED"):
            p._fetch_filings_from_sec(SYMBOL, CIK, "NVIDIA CORP")
    assert len(attempts) == 1
    assert p.outbound_session.total_transmissions == 1


def test_trusted_redirect_counts_each_http_transmission():
    p = make_pipeline()
    attempts = []

    def fake_send(adapter, request, **kwargs):
        attempts.append(request.url)
        if len(attempts) == 1:
            return response(request, 302, headers={
                "Location": "https://www.sec.gov/submissions/redirect.json"})
        if len(attempts) == 2:
            return response(request, 200, catalog(forms=[]))
        raise AssertionError("unexpected extra HTTP send")

    with patch(
        "analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
        return_value="ARX Terminal ops@example.org",
    ), patch.object(HTTPAdapter, "send", fake_send):
        assert p._fetch_filings_from_sec(SYMBOL, CIK, "NVIDIA CORP") == []
    assert len(attempts) == 2
    assert p.outbound_session.total_transmissions == 2


def test_adapter_managed_retries_fail_before_any_wire_send():
    session = ContinuousOutboundSession(allow_test_mode=True)
    session.set_current_issuer(SYMBOL)
    session.mount("https://", HTTPAdapter(max_retries=2))
    request = requests.Request(
        "GET", "https://data.sec.gov/submissions/CIK0001045810.json"
    ).prepare()
    with patch.object(HTTPAdapter, "send") as send:
        with pytest.raises(RuntimeError, match="SEC_EGRESS_UNACCOUNTED_RETRY_POLICY"):
            session.send(request)
        send.assert_not_called()


def test_no_valid_user_agent_fails_before_sec_transmission():
    p = make_pipeline()
    with patch(
        "analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
        return_value=None,
    ), patch.object(HTTPAdapter, "send") as send:
        with pytest.raises(RuntimeError, match="SEC_USER_AGENT_INVALID"):
            p._fetch_filings_from_sec(SYMBOL, CIK, "NVIDIA CORP")
        send.assert_not_called()


def test_invalid_accession_and_unsafe_primary_document_fail_closed():
    for bad_field, value in [
        ("accessionNumber", "../bad-accession"),
        ("primaryDocument", "../../secret.xml"),
    ]:
        p = make_pipeline()
        payload = json.loads(catalog())
        payload["filings"]["recent"][bad_field] = [value]
        with patch(
            "analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
            return_value="ARX Terminal ops@example.org",
        ), patch.object(
            HTTPAdapter, "send",
            side_effect=lambda request, **kwargs: response(
                request, 200, json.dumps(payload))
        ) as send:
            with pytest.raises(RuntimeError, match="SEC_(ACCESSION|PRIMARY_DOCUMENT)_INVALID"):
                p._fetch_filings_from_sec(SYMBOL, CIK, "NVIDIA CORP")
            assert send.call_count == 1


def test_session_does_not_publish_blank_live_snapshot():
    p = make_pipeline()
    p.allow_test_mode = False
    with patch.object(p, "_check_in_flight_authorization"), patch.object(
        p, "_fetch_filings_from_sec", return_value=[]
    ), patch(
        "analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
        return_value="ARX Terminal ops@example.org",
    ):
        with pytest.raises(RuntimeError, match="NO_NEW_FORM4_FILINGS"):
            p._step6_ingest_form4_filings()
