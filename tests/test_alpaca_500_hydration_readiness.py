"""
tests/test_alpaca_500_hydration_readiness.py

Targeted regression tests for RADAR-P2-004:
- Validates load_manifest support for ARX Universe Manifest format (UB-ARX-EXPANSION-500).
- Validates configurable max_completed_session parameter while preserving 2026-10-19 backward compatibility.
- Validates bar validation and window boundary enforcement under configured session horizons.
- Validates non-mutating dry-run guarantees.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from analyst_dashboard.data.market_evidence import StructuralQuality
from scripts.ingest_alpaca_historical import (
    AlpacaHistoricalSipIngestionAdapter,
    BarValidationResult,
    MAX_COMPLETED_SESSION,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_manifest_universe_format():
    """Verify load_manifest parses ARX Universe Manifests with top-level 'symbols'."""
    p500 = REPO_ROOT / "docs" / "research" / "UB-ARX-EXPANSION-500-REHEARSAL.json"
    assert p500.exists(), "Rehearsal manifest must exist"

    info = AlpacaHistoricalSipIngestionAdapter.load_manifest(p500)
    assert len(info.symbols) == 500
    assert info.scope_type == "US_EQUITIES"
    assert "AAPL" in info.symbols
    assert "UNH" in info.symbols
    assert "WYNN" in info.symbols


def test_load_manifest_rejects_noncanonical():
    """Verify load_manifest rejects non-canonical symbol formats."""
    import tempfile
    import json

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"symbols": ["AAPL", "SEC_123456"]}, f)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="NONCANONICAL_IDENTITY_ALIAS_REJECTED"):
            AlpacaHistoricalSipIngestionAdapter.load_manifest(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def test_max_completed_session_default_backward_compatible():
    """Verify default max_completed_session preserves 2026-10-19."""
    adapter = AlpacaHistoricalSipIngestionAdapter()
    assert adapter.max_completed_session is None

    # Bars on 2026-11-06 must fail under default 2026-10-19 boundary
    bars_nov = [{"t": "2026-11-06T05:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.0, "v": 1000}]
    res_def = AlpacaHistoricalSipIngestionAdapter.validate_bar_series(
        "AAPL", bars_nov, "2026-11-06", "2026-11-06"
    )
    assert not res_def.is_valid
    assert "FUTURE_OR_INCOMPLETE_SESSION_BOUND" in (res_def.rejection_reason or "")


def test_max_completed_session_override_permits_november():
    """Verify max_completed_session override permits valid November 2026 sessions."""
    nov_max = date(2026, 11, 6)
    bars_nov = [{"t": "2026-11-06T05:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.0, "v": 1000}]

    res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series(
        "AAPL", bars_nov, "2026-11-06", "2026-11-06", max_completed_session=nov_max
    )
    assert res.is_valid
    assert res.bar_count == 1
    assert res.last_session == "2026-11-06"
    assert res.structural_quality == StructuralQuality.COMPLETE


def test_bar_validation_rejects_past_effective_max():
    """Verify sessions past effective max are strictly rejected."""
    nov_max = date(2026, 11, 6)
    # 2026-11-09 exceeds 2026-11-06
    bars_future = [{"t": "2026-11-09T05:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.0, "v": 1000}]

    res = AlpacaHistoricalSipIngestionAdapter.validate_bar_series(
        "AAPL", bars_future, "2026-11-09", "2026-11-09", max_completed_session=nov_max
    )
    assert not res.is_valid
    assert "FUTURE_OR_INCOMPLETE_SESSION_BOUND" in (res.rejection_reason or "")


def test_dry_run_bounded_execution():
    """Verify bounded execution in dry-run mode writes 0 bars and returns PREPARED_DRY_RUN."""
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_historical_bars.return_value = {
        "success": True,
        "symbol": "AAPL",
        "bars": [{"t": "2026-11-06T05:00:00Z", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.0, "v": 1000}],
        "next_page_token": None,
    }

    adapter = AlpacaHistoricalSipIngestionAdapter(
        fetcher=mock_fetcher,
        max_completed_session=date(2026, 11, 6),
    )

    p500 = REPO_ROOT / "docs" / "research" / "UB-ARX-EXPANSION-500-REHEARSAL.json"
    summary = adapter.execute_bounded_ingestion(
        manifest_path=p500,
        symbols_override=["AAPL"],
        start_date="2026-11-06",
        end_date="2026-11-06",
        apply_writes=False,
    )

    assert summary.mode == "DRY_RUN"
    assert summary.batch_outcome == "SUCCESS"
    assert summary.total_bars_ingested == 0
    assert summary.candidate_dispositions["AAPL"]["status"] == "PREPARED_DRY_RUN"


def test_rehearsal_500_excludes_kvue_qrvo_includes_lyft_chwy():
    """Verify RADAR-P2-005R: KVUE/QRVO excluded and replaced by LYFT/CHWY in 500 rehearsal."""
    import json
    p500 = REPO_ROOT / "docs" / "research" / "UB-ARX-EXPANSION-500-REHEARSAL.json"
    with open(p500, "r", encoding="utf-8") as f:
        data = json.load(f)

    symbols = set(data["symbols"])
    assert len(data["symbols"]) == 500
    assert len(symbols) == 500, "All 500 symbols must be strictly unique"

    # Excluded corporate action symbols
    assert "KVUE" not in symbols, "KVUE must be excluded due to corporate action"
    assert "QRVO" not in symbols, "QRVO must be excluded due to corporate action"

    # Qualified replacement symbols
    assert "LYFT" in symbols, "LYFT must be included in 500 rehearsal universe"
    assert "CHWY" in symbols, "CHWY must be included in 500 rehearsal universe"

    # Pre-existing hydrated symbols
    assert "CAG" in symbols, "CAG must remain in universe"
    assert "CPB" in symbols, "CPB must remain in universe"


def test_exclusions_ledger_corporate_action_audit():
    """Verify exclusions ledger documents KVUE, QRVO, and FVRR."""
    import json
    ex_file = REPO_ROOT / "docs" / "research" / "candidate_exclusions_ledger_500.json"
    with open(ex_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    breakdown = data["exclusion_breakdown"]
    assert breakdown.get("EXCLUDED_CORPORATE_ACTION_HALTED_OR_CEASED_TRADING") == 2
    assert breakdown.get("EXCLUDED_FAILS_LIQUIDITY_CONTRACT") == 1

    corp_and_liq = data.get("corporate_and_liquidity_exclusions", [])
    sym_to_reason = {item["symbol"]: item["exclusion_reason"] for item in corp_and_liq}
    assert sym_to_reason.get("KVUE") == "EXCLUDED_CORPORATE_ACTION_HALTED_OR_CEASED_TRADING"
    assert sym_to_reason.get("QRVO") == "EXCLUDED_CORPORATE_ACTION_HALTED_OR_CEASED_TRADING"
    assert sym_to_reason.get("FVRR") == "EXCLUDED_FAILS_LIQUIDITY_CONTRACT"


def test_replacement_uniqueness_invariance():
    """Verify replacing KVUE/QRVO with existing members CAG/CPB would violate uniqueness."""
    import json
    p500 = REPO_ROOT / "docs" / "research" / "UB-ARX-EXPANSION-500-REHEARSAL.json"
    with open(p500, "r", encoding="utf-8") as f:
        data = json.load(f)

    symbols = list(data["symbols"])
    # Both CAG and CPB are already distinct members
    assert "CAG" in symbols
    assert "CPB" in symbols
    # Adding them again as replacements would shrink set cardinality
    hypothetical_with_cag_cpb = symbols + ["CAG", "CPB"]
    assert len(set(hypothetical_with_cag_cpb)) == len(symbols), "CAG and CPB cannot serve as unique replacements"

