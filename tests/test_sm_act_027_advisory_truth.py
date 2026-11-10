"""SM-ACT-027: source-contract regressions for held SEC evidence advisory projections.

These tests intentionally make no SEC, Railway, DB, or R2 calls. Full integration
verification must separately exercise the PostgreSQL/SQLite stores and API.
"""
from pathlib import Path
import sqlite3

from analyst_dashboard.analyzers.evidence_timestamp_projection import evidence_row_value

SOURCE = (Path(__file__).resolve().parents[1] / "analyst_dashboard" / "analyzers" / "smart_money.py").read_text(encoding="utf-8")

def _form4_projection():
    start = SOURCE.index('            if sec_rows:\n                for r in sec_rows:')
    end = SOURCE.index('        # ── FEED 2: OPTIONS FLOW SWEEPS')
    return SOURCE[start:end]


def test_empty_recent_window_not_claimed_as_verified_zero():
    src = _form4_projection()
    older = src.split("            elif has_older_in_db:", 1)[1].split("            else:", 1)[0]
    assert '"status": "PIPELINE_PENDING"' in older
    assert '"direction": "UNVERIFIED"' in older
    assert '"confidence": "NONE"' in older
    assert "complete SEC coverage" in older
    assert '"status": "NO_OBSERVATIONS"' not in older
    assert "0 insider transactions filed" not in older


def test_sec_publication_not_substituted_for_arx_ingestion():
    src = _form4_projection()
    assert '"acceptance_datetime": acc_dt' in src
    assert '"ingestion_timestamp": evidence_row_value(r, "arx_first_observed_at")' in src
    assert '"latest_ingestion_date": sec_observations[0].get("arx_first_observed_at")' in src
    assert '"latest_successful_ingestion": (sec_observations[0].get("arx_first_observed_at")' in src
    assert '"ingestion_timestamp": acc_dt' not in src
    assert '"latest_ingestion_date": sec_observations[0].get("acceptance_datetime")' not in src


def test_publication_is_not_false_source_health_check():
    src = _form4_projection()
    assert '"latest_successful_sec_source_check": None' in src
    assert '"latest_successful_sec_source_check": (sec_observations[0].get("acceptance_datetime")' not in src


def test_existing_insider_transaction_classification_retained():
    src = _form4_projection()
    assert 'is_buy = not is_deriv' in src
    assert 't_code == "F" or sig_type == "TAX_WITHHOLDING"' in src
    assert 'tx_type = "Tax Withholding"' in src
    assert 'is_sell = not is_deriv' in src
    assert '"replay_mode": replay_mode' in src
    assert '"as_of_watermark": as_of_watermark' in src


def test_database_row_timestamp_access_without_fabricated_fallback():
    values = {
        "acceptance_datetime": "2026-09-18T21:13:29Z",
        "arx_first_observed_at": "2026-09-19T10:20:00Z",
        "arx_processed_at": "2026-09-19T10:21:00Z",
        "sec_accepted_at": "2026-09-18T21:13:29Z",
    }
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "CREATE TABLE filings (acceptance_datetime TEXT, arx_first_observed_at TEXT, "
            "arx_processed_at TEXT, sec_accepted_at TEXT)"
        )
        conn.execute(
            "INSERT INTO filings VALUES (?, ?, ?, ?)",
            tuple(values.values()),
        )
        sqlite_row = conn.execute("SELECT * FROM filings").fetchone()
        for row in (values, sqlite_row):
            observed = evidence_row_value(row, "arx_first_observed_at")
            processed = evidence_row_value(row, "arx_processed_at")
            assert observed == "2026-09-19T10:20:00Z"
            assert processed == "2026-09-19T10:21:00Z"
            assert (observed or processed) != evidence_row_value(row, "acceptance_datetime")
            assert evidence_row_value(row, "not_a_real_column") is None
    finally:
        conn.close()


def test_missing_observation_never_becomes_sec_acceptance_time():
    row = {
        "acceptance_datetime": "2026-09-18T21:13:29Z",
        "arx_first_observed_at": None,
        "arx_processed_at": None,
    }
    assert (
        evidence_row_value(row, "arx_first_observed_at")
        or evidence_row_value(row, "arx_processed_at")
    ) is None


def test_public_route_forwards_watermark_and_mode_to_evidence_assessment():
    route_source = (
        Path(__file__).resolve().parents[1] / "api" / "routes" / "smart_money.py"
    ).read_text(encoding="utf-8")
    route = route_source.split('def get_smart_money_evidence(', 1)[1].split(
        '\n@router.get("/filings/{symbol}")', 1
    )[0]
    assert 'as_of=as_of_val' in route
    assert 'replay_mode=mode_val' in route

    wrapper = SOURCE.split('def assess_symbol(', 1)[1].split(
        'def get_smart_money_overview(', 1
    )[0]
    assert 'pit_evidence = cls.evaluate_smart_money_feeds(' in wrapper
    assert 'as_of=as_of' in wrapper
    assert 'replay_mode=replay_mode' in wrapper
    assert '"pit_evidence": pit_evidence' in wrapper
    assert '"latest_successful_ingestion": None' in wrapper
