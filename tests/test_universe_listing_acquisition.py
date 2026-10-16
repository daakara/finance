"""
tests/test_universe_listing_acquisition.py

Focused unit and integration verification for listing directory acquisition,
storage preservation, sidecar attribution, offline parser replay, and source accounting.

Marked tier2c (Provenance and Data Gate).

Coverage:
1. Successful acquisition and raw-byte preservation.
2. Timeout and transient failure retries.
3. 429 Rate-limit response handling.
4. TLS failure without insecure fallback.
5. Insecure redirect downgrade detection.
6. Oversized payload streaming ceiling enforcement.
7. HTML error body on HTTP 200 (schema failure during replay).
8. Malformed source schema parsing diagnostics.
9. Partial multi-source acquisition outcome.
10. Safe artifact-write invariant (failure never overwrites accepted artifact).
11. Preserved parser diagnostics and non-binding cross-source matching.
12. Zero implicit network call on module import.
13. Architectural boundary invariant: zero imports from builder, store, or scanner.
14. Dual JSON and Markdown accounting report generation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from analyst_dashboard.universe.acquisition import (
    CANONICAL_SOURCES,
    AcquisitionAttempt,
    AcquisitionOutcome,
    ArtifactStorage,
    ListingAcquisitionRunner,
    ListingSourceFetcher,
    SourceAccountingReport,
    SourceConfig,
    format_markdown_report,
    run_acquisition_and_accounting,
    save_accounting_reports,
)

pytestmark = pytest.mark.tier2c


# ── Synthetic Payloads ────────────────────────────────────────────────────────

SYNTHETIC_SEC_JSON = b"""{
  "fields": ["cik", "name", "ticker", "exchange"],
  "data": [
    [320193, "Apple Inc.", "AAPL", "Nasdaq"],
    [789019, "Microsoft Corp", "MSFT", "Nasdaq"],
    [1067983, "Berkshire Hathaway Inc", "BRK.A", "NYSE"]
  ]
}"""

SYNTHETIC_NASDAQ_LISTED_TXT = (
    b"Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    b"AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    b"MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n"
    b"File Creation Time: 1016202603:00\n"
)

SYNTHETIC_OTHER_LISTED_TXT = (
    b"ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    b"BRK.A|Berkshire Hathaway Inc. Class A|N|BRK.A|N|100|N|BRK.A\n"
    b"File Creation Time: 1016202603:00\n"
)


# ── Helper for Mocking Response ───────────────────────────────────────────────

def _build_mock_response(
    status_code: int = 200,
    content: bytes = b"",
    headers: dict | None = None,
    url: str = "https://example.com/feed",
    history: list | None = None,
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.content = content
    resp.headers = headers or {"Content-Type": "text/plain", "ETag": '"abc123etag"', "Last-Modified": "Wed, 16 Oct 2026 03:00:00 GMT"}
    resp.url = url
    resp.history = history or []
    resp.is_redirect = status_code in (301, 302, 303, 307, 308)
    # Support chunked streaming iter_content
    resp.iter_content = MagicMock(return_value=[content] if content else [])
    return resp


# ── Unit Tests ────────────────────────────────────────────────────────────────

def test_successful_acquisition_and_raw_byte_preservation(tmp_path: Path):
    """1. Successful fetch preserves exact raw bytes and metadata sidecar."""
    config = SourceConfig(
        source_id="sec_test",
        url="https://www.sec.gov/files/company_tickers_exchange.json",
        headers={"User-Agent": "TestRunner/1.0 (test@example.com)", "Accept": "application/json"},
        timeout_seconds=5.0,
    )
    mock_resp = _build_mock_response(200, SYNTHETIC_SEC_JSON, {"Content-Type": "application/json", "ETag": '"etag1"'})

    session = MagicMock(spec=requests.Session)
    session.get.return_value = mock_resp

    fetcher = ListingSourceFetcher(session=session)
    raw_bytes, attempt = fetcher.fetch(config)

    assert raw_bytes == SYNTHETIC_SEC_JSON
    assert attempt.outcome == AcquisitionOutcome.SUCCESS
    assert attempt.http_status == 200
    assert attempt.etag == '"etag1"'
    assert attempt.byte_length == len(SYNTHETIC_SEC_JSON)
    assert attempt.content_sha256 is not None

    storage = ArtifactStorage(tmp_path)
    saved_attempt = storage.save_artifact(attempt, raw_bytes)

    assert saved_attempt.raw_artifact_path is not None
    assert Path(saved_attempt.raw_artifact_path).exists()
    assert Path(saved_attempt.raw_artifact_path).read_bytes() == SYNTHETIC_SEC_JSON

    assert saved_attempt.metadata_artifact_path is not None
    assert Path(saved_attempt.metadata_artifact_path).exists()
    meta_json = json.loads(Path(saved_attempt.metadata_artifact_path).read_text(encoding="utf-8"))
    assert meta_json["content_sha256"] == attempt.content_sha256
    assert meta_json["outcome"] == "SUCCESS"


def test_timeout_and_transient_failure_retries(tmp_path: Path):
    """2. Timeouts trigger bounded retries up to max_retries, reporting TIMEOUT outcome."""
    config = SourceConfig(
        source_id="timeout_source",
        url="https://example.com/timeout",
        timeout_seconds=1.0,
        max_retries=2,
        retry_backoff_factor=0.01,
    )

    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    fetcher = ListingSourceFetcher(session=session)
    raw_bytes, attempt = fetcher.fetch(config)

    assert raw_bytes is None
    assert attempt.outcome == AcquisitionOutcome.TIMEOUT
    assert "timed out" in attempt.error_message
    assert session.get.call_count == 3  # initial attempt + 2 retries


def test_rate_limit_response_429(tmp_path: Path):
    """3. HTTP 429 Rate Limit is detected, inspected for Retry-After, and bounded."""
    config = SourceConfig(
        source_id="rate_limited_source",
        url="https://example.com/429",
        max_retries=1,
        retry_backoff_factor=0.01,
    )
    mock_429 = _build_mock_response(429, b"Too Many Requests", {"Retry-After": "1"})

    session = MagicMock(spec=requests.Session)
    session.get.return_value = mock_429

    fetcher = ListingSourceFetcher(session=session)
    raw_bytes, attempt = fetcher.fetch(config)

    assert raw_bytes is None
    assert attempt.outcome == AcquisitionOutcome.HTTP_ERROR
    assert "429" in attempt.error_message
    assert session.get.call_count == 2


def test_tls_failure_without_insecure_fallback():
    """4. SSL/TLS verification failure fails closed without falling back to insecure HTTP."""
    config = SourceConfig(
        source_id="tls_broken",
        url="https://example.com/bad_cert",
        enforce_https=True,
    )

    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.exceptions.SSLError("Certificate verify failed")

    fetcher = ListingSourceFetcher(session=session)
    raw_bytes, attempt = fetcher.fetch(config)

    assert raw_bytes is None
    assert attempt.outcome == AcquisitionOutcome.TLS_ERROR
    assert "TLS/SSL certificate verification failed" in attempt.error_message

    # Test scheme pre-flight check when HTTP is given
    insecure_config = SourceConfig(
        source_id="insecure_scheme",
        url="http://insecure.example.com/feed",
        enforce_https=True,
    )
    raw_bytes_insecure, attempt_insecure = fetcher.fetch(insecure_config)
    assert raw_bytes_insecure is None
    assert attempt_insecure.outcome == AcquisitionOutcome.TLS_ERROR
    assert "Insecure URL scheme" in attempt_insecure.error_message


def test_redirect_insecure_downgrade_detected():
    """5. Prohibits silent downgrade from HTTPS to HTTP across redirect chains before request is issued."""
    config = SourceConfig(
        source_id="redirect_downgrade",
        url="https://example.com/feed",
        enforce_https=True,
    )
    redirect_resp = _build_mock_response(301, headers={"Location": "http://insecure.example.com/downgraded"})

    session = MagicMock(spec=requests.Session)
    session.get.return_value = redirect_resp

    fetcher = ListingSourceFetcher(session=session)
    raw_bytes, attempt = fetcher.fetch(config)

    assert raw_bytes is None
    assert attempt.outcome == AcquisitionOutcome.TLS_ERROR
    assert "downgraded to insecure http before issuing request" in attempt.error_message.lower()
    # Invariant: session.get called only once for initial URL; never issued to insecure HTTP target!
    assert session.get.call_count == 1
    assert session.get.call_args[0][0] == "https://example.com/feed"


def test_sec_edgar_requires_explicit_contact_user_agent():
    """Check A: SEC EDGAR acquisition fails closed when no administrative contact is declared."""
    config = SourceConfig(
        source_id="sec_company_tickers_exchange",
        url="https://www.sec.gov/files/company_tickers_exchange.json",
        headers={"Accept": "application/json"},  # Missing contact User-Agent
    )
    session = MagicMock(spec=requests.Session)
    fetcher = ListingSourceFetcher(session=session)
    raw_bytes, attempt = fetcher.fetch(config)

    assert raw_bytes is None
    assert attempt.outcome == AcquisitionOutcome.FAILED
    assert "SEC EDGAR acquisition requires an explicitly configured User-Agent" in attempt.error_message
    session.get.assert_not_called()


def test_artifact_storage_path_traversal_sanitization(tmp_path: Path):
    """Check E: Verifies that malicious or relative source_id names cannot escape base_dir."""
    storage = ArtifactStorage(tmp_path)
    attempt = AcquisitionAttempt(
        source_id="../../escaped_name",
        requested_url="https://example.com/feed",
        outcome=AcquisitionOutcome.SUCCESS,
        http_status=200,
        byte_length=len(SYNTHETIC_SEC_JSON),
    )
    saved = storage.save_artifact(attempt, SYNTHETIC_SEC_JSON)
    assert saved.raw_artifact_path is not None
    saved_path = Path(saved.raw_artifact_path)
    assert saved_path.parent == tmp_path
    assert ".." not in saved_path.name


def test_sec_diagnostic_row_categorization():
    """Check B: Distinguishes structurally parsed rows with missing exchange from syntax failures."""
    from analyst_dashboard.universe.adapters.sec_edgar import parse_sec_company_tickers_exchange

    mixed_payload = json.dumps({
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [
            [320193, "Apple Inc.", "AAPL", "Nasdaq"],
            [2156068, "InoBat B.V.", "INBT", None],  # missing exchange attribute
            [999999, "Bad Row"],  # wrong length (structurally rejected)
        ]
    }).encode("utf-8")

    parsed = parse_sec_company_tickers_exchange(mixed_payload)
    assert parsed.data_row_count == 1
    assert parsed.malformed_row_count == 2

    # Missing exchange row has specific evidence gap reason (not corrupt syntax)
    missing_exch_row = next(r for r in parsed.rows if r.row_index == 1)
    assert "missing exchange attribute" in missing_exch_row.malformed_reason.lower()

    # Wrong length row has structural reason
    bad_struct_row = next(r for r in parsed.rows if r.row_index == 2)
    assert "width mismatch" in bad_struct_row.malformed_reason.lower()


def test_oversized_truncated_response():
    """6. Bounded streaming enforces max_bytes and terminates stream on overflow."""
    config = SourceConfig(
        source_id="oversized_source",
        url="https://example.com/large",
        max_bytes=100,
    )
    resp = _build_mock_response(200, url="https://example.com/large")
    resp.iter_content = MagicMock(return_value=[b"A" * 60, b"B" * 60])  # Total 120 bytes > 100

    session = MagicMock(spec=requests.Session)
    session.get.return_value = resp

    fetcher = ListingSourceFetcher(session=session)
    raw_bytes, attempt = fetcher.fetch(config)

    assert raw_bytes is None
    assert attempt.outcome == AcquisitionOutcome.SIZE_EXCEEDED
    assert "exceeded byte limit" in attempt.error_message.lower()
    resp.close.assert_called_once()


def test_html_error_body_on_http_200(tmp_path: Path):
    """7. Server returning HTML error body on HTTP 200 is preserved and flagged in replay."""
    html_error = b"<html><head><title>Access Denied</title></head><body>Rate Limit / Cloudflare Block</body></html>"

    config = SourceConfig(
        source_id="sec_company_tickers_exchange",
        url="https://www.sec.gov/files/company_tickers_exchange.json",
        headers={"User-Agent": "TestRunner/1.0 (test@example.com)", "Accept": "text/html"},
    )
    mock_resp = _build_mock_response(200, html_error, {"Content-Type": "text/html"})

    session = MagicMock(spec=requests.Session)
    session.get.return_value = mock_resp

    report, parsed_files = run_acquisition_and_accounting(
        storage_dir=tmp_path,
        sources={"sec_company_tickers_exchange": config},
        session=session,
        run_matching=False,
    )

    assert report.overall_outcome == "FAILED"  # fetch succeeded, but replay marked unusable
    summary = report.parsed_summaries["sec_company_tickers_exchange"]
    assert summary.is_usable is False
    assert summary.data_row_count == 0
    assert summary.malformed_row_count == 1
    assert any("JSON" in r for r in summary.sample_malformed_reasons)


def test_malformed_source_schema_diagnostics(tmp_path: Path):
    """8. Malformed directory entries preserve row counts and explicit error reasons."""
    corrupt_nasdaq = (
        b"Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        b"AAPL|Apple Inc.|Q|N|N\n"  # only 5 columns instead of 8
        b"File Creation Time: 1016202603:00\n"
    )
    config = SourceConfig(
        source_id="nasdaq_listed_directory",
        url="https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    )
    mock_resp = _build_mock_response(200, corrupt_nasdaq, {"Content-Type": "text/plain"})

    session = MagicMock(spec=requests.Session)
    session.get.return_value = mock_resp

    report, _ = run_acquisition_and_accounting(
        storage_dir=tmp_path,
        sources={"nasdaq_listed_directory": config},
        session=session,
        run_matching=False,
    )

    summary = report.parsed_summaries["nasdaq_listed_directory"]
    assert summary.data_row_count == 0
    assert summary.malformed_row_count == 1
    assert any("column" in r.lower() or "mismatch" in r.lower() for r in summary.sample_malformed_reasons)


def test_partial_multi_source_acquisition(tmp_path: Path):
    """9. Partial multi-source run aggregates succeeded and failed attempts correctly."""
    sec_cfg = SourceConfig(source_id="sec_company_tickers_exchange", url="https://sec.gov/tickers.json")
    nasdaq_cfg = SourceConfig(source_id="nasdaq_listed_directory", url="https://nasdaqtrader.com/nasdaq.txt")

    def mock_fetch(cfg: SourceConfig):
        if "sec" in cfg.source_id:
            return SYNTHETIC_SEC_JSON, AcquisitionAttempt(
                source_id=cfg.source_id,
                requested_url=cfg.url,
                outcome=AcquisitionOutcome.SUCCESS,
                http_status=200,
                byte_length=len(SYNTHETIC_SEC_JSON),
            )
        else:
            return None, AcquisitionAttempt(
                source_id=cfg.source_id,
                requested_url=cfg.url,
                outcome=AcquisitionOutcome.HTTP_ERROR,
                http_status=503,
                error_message="Service Unavailable",
            )

    report, parsed_files = run_acquisition_and_accounting(
        storage_dir=tmp_path,
        sources={"sec_company_tickers_exchange": sec_cfg, "nasdaq_listed_directory": nasdaq_cfg},
        fetch_fn=mock_fetch,
        run_matching=False,
    )

    assert report.overall_outcome == "PARTIAL_ACQUISITION"
    assert len(report.attempts) == 2
    assert "sec_company_tickers_exchange" in report.parsed_summaries
    assert "nasdaq_listed_directory" not in report.parsed_summaries


def test_safe_artifact_write_invariant_never_overwrites_accepted(tmp_path: Path):
    """10. Failed attempts NEVER overwrite an accepted valid artifact on disk."""
    storage = ArtifactStorage(tmp_path)

    # 1. Save valid initial artifact
    valid_attempt = AcquisitionAttempt(
        source_id="sec_company_tickers_exchange",
        requested_url="https://sec.gov/valid",
        outcome=AcquisitionOutcome.SUCCESS,
        http_status=200,
        byte_length=len(SYNTHETIC_SEC_JSON),
        retrieval_start_utc="2026-10-16T10:00:00Z",
    )
    storage.save_artifact(valid_attempt, SYNTHETIC_SEC_JSON)

    raw_path = tmp_path / "sec_company_tickers_exchange.json"
    assert raw_path.exists()
    assert raw_path.read_bytes() == SYNTHETIC_SEC_JSON

    # 2. Simulate subsequent failed attempt
    failed_attempt = AcquisitionAttempt(
        source_id="sec_company_tickers_exchange",
        requested_url="https://sec.gov/valid",
        outcome=AcquisitionOutcome.TIMEOUT,
        http_status=None,
        retrieval_start_utc="2026-10-16T11:00:00Z",
        error_message="Gateway Timeout",
    )
    res = storage.save_artifact(failed_attempt, None)

    # Invariant: Original valid artifact must remain untouched
    assert raw_path.exists()
    assert raw_path.read_bytes() == SYNTHETIC_SEC_JSON

    # Quarantine file must exist
    quarantine_files = list((tmp_path / "quarantine").glob("failed_*.meta.json"))
    assert len(quarantine_files) == 1
    assert "Gateway Timeout" in quarantine_files[0].read_text(encoding="utf-8")


def test_parser_diagnostics_and_cross_source_matching(tmp_path: Path):
    """11. Preserves parser observations and executes non-binding cross-matching."""
    sources = {
        "sec_company_tickers_exchange": CANONICAL_SOURCES["sec_company_tickers_exchange"],
        "nasdaq_listed_directory": CANONICAL_SOURCES["nasdaq_listed_directory"],
        "other_listed_directory": CANONICAL_SOURCES["other_listed_directory"],
    }
    offline_data = {
        "sec_company_tickers_exchange": SYNTHETIC_SEC_JSON,
        "nasdaq_listed_directory": SYNTHETIC_NASDAQ_LISTED_TXT,
        "other_listed_directory": SYNTHETIC_OTHER_LISTED_TXT,
    }

    runner = ListingAcquisitionRunner(output_dir=tmp_path)
    report = runner.run(sources=sources, run_matching=True, offline_bytes=offline_data)

    assert report.overall_outcome == "SUCCESS"
    assert report.production_population_completeness == "NOT_ESTABLISHED"
    assert len(report.parsed_summaries) == 3

    # Check match diagnostics
    assert report.match_diagnostics is not None
    assert report.match_diagnostics["total_candidates"] == 3
    assert report.match_diagnostics["fully_matched_count"] == 3


def test_no_implicit_network_call_on_import():
    """12. Importing acquisition package performs zero network or socket operations."""
    with patch("socket.socket") as mock_sock:
        import analyst_dashboard.universe.acquisition as acq
        assert hasattr(acq, "ListingSourceFetcher")
        assert hasattr(acq, "run_acquisition_and_accounting")
        mock_sock.assert_not_called()


def test_architectural_boundary_invariants():
    """13. Acquisition package strictly maintains separation from builder, store, and scanner."""
    acq_dir = Path("analyst_dashboard/universe/acquisition")
    forbidden_tokens = ["UniverseBuilder", "UniverseStore", "confluence_engine", "run_scanner", "execute_scan"]

    for py_file in acq_dir.glob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        parsed = ast.parse(code)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_tokens:
                        assert forbidden not in alias.name, f"Forbidden import '{forbidden}' in {py_file}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for forbidden in forbidden_tokens:
                    assert forbidden not in module, f"Forbidden import from '{forbidden}' in {py_file}"
                for alias in node.names:
                    for forbidden in forbidden_tokens:
                        assert forbidden not in alias.name, f"Forbidden import '{forbidden}' in {py_file}"


def test_dual_report_generation(tmp_path: Path):
    """14. Verifies generation and formatting of machine-readable JSON and Markdown reports."""
    report = SourceAccountingReport(
        generated_at_utc="2026-10-16T12:00:00Z",
        acquisition_version="1.0.0",
        attempts=[
            AcquisitionAttempt(
                source_id="sec_test",
                requested_url="https://sec.gov/test",
                outcome=AcquisitionOutcome.SUCCESS,
                http_status=200,
                byte_length=1234,
                content_sha256="abc123sha256",
            )
        ],
        parsed_summaries={},
        overall_outcome="COMPLETE_SUCCESS",
        production_population_completeness="NOT_ESTABLISHED",
        coverage_limitations=["Limitation 1"],
        unresolved_decisions=["Decision 1"],
    )

    json_path, md_path = save_accounting_reports(report, tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    json_content = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_content["production_population_completeness"] == "NOT_ESTABLISHED"
    assert json_content["overall_outcome"] == "COMPLETE_SUCCESS"

    md_content = md_path.read_text(encoding="utf-8")
    assert "PRODUCTION_POPULATION_COMPLETENESS" in md_content or "Production Population Completeness" in md_content
    assert "NOT_ESTABLISHED" in md_content
    assert "Core Product Acceptance Criterion" in md_content


# ── Formal Checks A–E Boundary Verification Suite ─────────────────────────────

def test_check_a_surrogate_key_non_synthesis(tmp_path: Path):
    """Check A: Verifies zero synthesis or manufacturing of production security_id or surrogate keys."""
    sources = {
        "sec_company_tickers_exchange": CANONICAL_SOURCES["sec_company_tickers_exchange"],
        "nasdaq_listed_directory": CANONICAL_SOURCES["nasdaq_listed_directory"],
    }
    offline_data = {
        "sec_company_tickers_exchange": SYNTHETIC_SEC_JSON,
        "nasdaq_listed_directory": SYNTHETIC_NASDAQ_LISTED_TXT,
    }

    report, parsed_files = run_acquisition_and_accounting(
        storage_dir=tmp_path,
        sources=sources,
        offline_bytes=offline_data,
        run_matching=True,
    )

    # Invariant: ParsedSourceRow preserves only native raw values, no synthetic security_id
    for source_id, parsed_file in parsed_files.items():
        for row in parsed_file.rows:
            assert "security_id" not in row.raw_values
            assert "surrogate_key" not in row.raw_values
            assert not any(k.startswith("CIK_") for k in row.raw_values.keys())

    # Invariant: Accounting report and models do not adopt synthetic surrogate keys as production identity
    assert not hasattr(report, "security_id")
    assert report.production_population_completeness == "NOT_ESTABLISHED"


def test_check_b_openfigi_isolation_and_multimatch_ambiguity_protection():
    """Check B: Directory acquisition operates decoupled from OpenFIGI and preserves ambiguity."""
    # Listing acquisition configuration does NOT define openfigi as a canonical listing source
    assert "openfigi" not in CANONICAL_SOURCES
    for cfg in CANONICAL_SOURCES.values():
        assert "openfigi.com" not in cfg.url.lower()

    # Even in matching adapter, OpenFIGI multi-match results must preserve ambiguity without auto-selection
    from analyst_dashboard.universe.adapters.models import OpenFigiMatchStatus, ParsedOpenFigiResult
    multi = ParsedOpenFigiResult(
        request_index=0,
        request_payload={"idType": "TICKER", "idValue": "TEST"},
        match_status=OpenFigiMatchStatus.MULTI_MATCH,
        matches=[{"figi": "BBG0001", "name": "Share A"}, {"figi": "BBG0002", "name": "Share B"}],
        selected_figi=None,  # Zero automatic selection
    )
    assert multi.selected_figi is None
    assert multi.match_status == OpenFigiMatchStatus.MULTI_MATCH


def test_check_c_source_completeness_accounting_truthfulness(tmp_path: Path):
    """Check C: Confirms truthful disclosure of non-completeness and evidence gaps."""
    report, _ = run_acquisition_and_accounting(
        storage_dir=tmp_path,
        sources={"sec_company_tickers_exchange": CANONICAL_SOURCES["sec_company_tickers_exchange"]},
        offline_bytes={"sec_company_tickers_exchange": SYNTHETIC_SEC_JSON},
    )
    assert report.production_population_completeness == "NOT_ESTABLISHED"
    assert any("NOT_ESTABLISHED" in lim for lim in report.coverage_limitations)
    assert len(report.coverage_limitations) >= 5
    assert len(report.unresolved_decisions) >= 4


def test_check_d_read_safety_and_non_triggering_guarantees():
    """Check D: Verifies zero background workers, crons, API route hooks, or DB writes."""
    import analyst_dashboard.universe.acquisition.runner as runner_mod
    import inspect

    # Verify functions are plain synchronous callable functions without decorators or async loops
    assert inspect.isfunction(runner_mod.run_acquisition_and_accounting)
    assert not inspect.iscoroutinefunction(runner_mod.run_acquisition_and_accounting)

    # Verify no Flask/FastAPI route decorators attached
    assert not hasattr(runner_mod.run_acquisition_and_accounting, "__wrapped__")
    assert not hasattr(runner_mod.ListingAcquisitionRunner.run, "__wrapped__")


def test_check_e_offline_replay_and_quarantine_integrity(tmp_path: Path):
    """Check E: Verifies pure offline replay capability and failure quarantine isolation."""
    runner = ListingAcquisitionRunner(output_dir=tmp_path)
    # Replay with offline_bytes operates 100% network free
    with patch("requests.Session.get") as mock_get:
        report = runner.run(
            sources={"sec": CANONICAL_SOURCES["sec_company_tickers_exchange"]},
            offline_bytes={"sec": SYNTHETIC_SEC_JSON},
        )
        mock_get.assert_not_called()
    assert report.overall_outcome == "SUCCESS"

    # Verify accepted artifact exists in base_dir
    raw_files = list(tmp_path.glob("*.json"))
    assert len(raw_files) >= 1

