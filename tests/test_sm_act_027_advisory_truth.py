"""SM-ACT-027: source-contract regressions for held SEC evidence advisory projections.

These tests intentionally make no SEC, Railway, DB, or R2 calls. Full integration
verification must separately exercise the PostgreSQL/SQLite stores and API.
"""
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "analyst_dashboard" / "analyzers" / "smart_money.py").read_text(encoding="utf-8")

def _form4_projection():
    start = SOURCE.index('            if sec_rows:')
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
    assert '"ingestion_timestamp": (r["arx_first_observed_at"]' in src
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
