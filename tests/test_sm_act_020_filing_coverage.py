"""Offline coverage and watermark acceptance for continuous SEC Form 4.

The actual HTTP adapter is intercepted. No SEC traffic, DB writes or holds.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository

CIK = "0001045810"
SYMBOL = "NVDA"
ACCESSIONS = [f"0001045810-26-{i:06d}" for i in range(1, 6)]
XML = '<?xml version="1.0"?><ownershipDocument><schemaVersion>X0202</schemaVersion></ownershipDocument>'


def catalog(accessions):
    return json.dumps({"filings": {"recent": {
        "form": ["4"] * len(accessions),
        "accessionNumber": accessions,
        "filingDate": ["2026-11-11"] * len(accessions),
        "acceptanceDateTime": ["2026-11-11T19:00:00.000Z"] * len(accessions),
        "primaryDocument": ["ownership.xml"] * len(accessions),
    }}})


def response(request, body, status=200):
    value = requests.Response()
    value.status_code = status
    value._content = body.encode("utf-8")
    value.url = request.url
    value.request = request
    return value


def build_pipeline(known=()):
    repo = MagicMock()
    repo.get_existing_accessions_for_issuer.return_value = set(known)
    pipeline = SmartMoneyContinuousPipeline(repository=repo, allow_test_mode=True)
    pipeline.rate_limiter.acquire = lambda: None
    pipeline.outbound_session.rate_limiter.acquire = lambda: None
    pipeline.outbound_session.set_current_issuer(SYMBOL)
    return pipeline, repo


def exercise(pipeline, indexes):
    attempted = []

    def fake_send(adapter, request, **kwargs):
        attempted.append(request.url)
        if request.url.startswith("https://data.sec.gov/"):
            return response(request, catalog(indexes))
        return response(request, XML)

    with patch("analyst_dashboard.smart_money.continuous_pipeline.get_sec_user_agent",
               return_value="ARX Terminal ops@example.org"), patch.object(
            HTTPAdapter, "send", fake_send
        ):
        result = pipeline._fetch_filings_from_sec(SYMBOL, CIK, "NVIDIA CORP")
    return result, attempted


def test_three_unseen_filings_are_fetched_and_accounted_within_four_sends():
    pipeline, repo = build_pipeline()
    filings, sends = exercise(pipeline, ACCESSIONS[:3])
    assert [f["accession_number"] for f in filings] == ACCESSIONS[:3]
    assert len(sends) == 4
    assert pipeline.outbound_session.total_transmissions == 4
    assert pipeline.outbound_session.issuer_transmissions[SYMBOL] == 4
    repo.get_existing_accessions_for_issuer.assert_called_once_with(SYMBOL)


def test_known_filings_skipped_without_unnecessary_archive_requests():
    pipeline, _ = build_pipeline(known=[ACCESSIONS[1]])
    filings, sends = exercise(pipeline, ACCESSIONS[:3])
    assert [f["accession_number"] for f in filings] == [ACCESSIONS[0], ACCESSIONS[2]]
    assert len(sends) == 3


def test_duplicate_index_accessions_cannot_consume_extra_budget():
    pipeline, _ = build_pipeline()
    filings, sends = exercise(pipeline, [ACCESSIONS[0], ACCESSIONS[0]])
    assert len(filings) == 1
    assert len(sends) == 2


def test_four_unseen_filings_fail_before_any_archive_fetch():
    pipeline, _ = build_pipeline()
    with pytest.raises(RuntimeError, match="SEC_NEW_FILINGS_EXCEED_ISSUER_BUDGET"):
        exercise(pipeline, ACCESSIONS[:4])
    assert pipeline.outbound_session.total_transmissions == 1


def test_unverifiable_repository_watermark_fails_closed():
    pipeline, repo = build_pipeline()
    repo.get_existing_accessions_for_issuer.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="SEC_FILING_WATERMARK_UNVERIFIED"):
        exercise(pipeline, ACCESSIONS[:1])
    assert pipeline.outbound_session.total_transmissions == 1


def test_all_known_filings_returns_empty_without_archive_fetch():
    pipeline, _ = build_pipeline(known=[ACCESSIONS[0]])
    filings, sends = exercise(pipeline, ACCESSIONS[:1])
    assert filings == []
    assert len(sends) == 1


def test_repository_watermark_query_is_parameterized_and_read_only():
    repo = PostgresSmartMoneyRepository(dsn="postgresql://mock")
    cursor = MagicMock()
    cursor.fetchall.return_value = [(ACCESSIONS[0],), (ACCESSIONS[1],)]
    connection_context = MagicMock()
    connection_context.__enter__.return_value = cursor
    with patch("analyst_dashboard.smart_money.storage.postgres_repository.transaction",
               return_value=connection_context):
        known = repo.get_existing_accessions_for_issuer("NVDA")
    assert known == set(ACCESSIONS[:2])
    cursor.execute.assert_called_once_with(
        "SELECT accession_number FROM form4_raw_filings WHERE ticker = %s;",
        ("NVDA",),
    )
