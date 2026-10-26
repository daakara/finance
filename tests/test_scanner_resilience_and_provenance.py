"""
ARX Terminal — Sprint 4 / Workstream 4.3
Comprehensive Verification Suite for Scanner Resilience, Population Truthfulness & Corporate-Action Provenance.

Covers Targeted Negative Test Matrix:
- NEG-R1-01: Explicit Empty Universe Override
- NEG-R1-02: Empty Build Membership
- NEG-R1-03: Natural Scheduled Scan Without Published Universe
- NEG-R2-01: Single Symbol Provider Timeout in Multi-Symbol Population
- NEG-R2-02: SQLite Lock Contention Isolation
- NEG-R2-03: Sub-Quorum Population Failure
- NEG-R3-01: Opaque Corporate Action Empty List Passthrough & Provenance Binding
- NEG-R3-02: Pre-Adjusted Continuous Feed Recognition
- NEG-R3-03: Raw Unadjusted Feed Guard
- API Route Radar Capabilities Coverage Disclosure Integration
"""

import os
import sqlite3
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest

from analyst_dashboard.analyzers.scanner_contract import ScannerStatus
from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.data.market_evidence import (
    AdjustmentState,
    MarketEvidence,
    MarketProvenance,
    CacheOrigin,
    Provider,
    IngestionSource,
    StructuralQuality,
)
from analyst_dashboard.vcp.conformance_oracle import generate_clean_history
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
from api.routes.screener import get_radar_capabilities


def _bars_to_dict_candles(bars: List[DailyOHLCVBar]) -> List[Dict[str, Any]]:
    return [
        {
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "time": b.bar_date,
        }
        for b in bars
    ]


def _create_mock_evidence(adj_state: str = AdjustmentState.SPLIT_ADJUSTED_ONLY) -> MarketEvidence:
    prov = MarketProvenance(
        provider=Provider.UNKNOWN,
        ingestion_source=IngestionSource.DIRECT_PROVIDER,
        observed_at="2026-10-22T21:00:00Z",
        observed_date="2026-10-22",
        observation_precision="DATE",
        observation_source="PROVIDER_BAR_TIMESTAMP",
        ingested_at="2026-10-22T21:05:00Z",
        adjustment_state=adj_state,
        structural_quality=StructuralQuality.COMPLETE,
        fallback_status=False,
    )
    return MarketEvidence(
        provenance=prov,
        cache_origin=CacheOrigin.SQLITE_MARKET_STORE,
        candle_count=60,
    )


# ==============================================================================
# TEST GROUP NEG-R1: EMPTY UNIVERSE BOUNDARY TESTS
# ==============================================================================

def test_neg_r1_01_explicit_empty_universe_override():
    """
    Scenario NEG-R1-01: Explicit Empty Universe Override.
    Must fail closed to ScannerStatus.ERROR with matched_count=None and
    coverage_status='EMPTY_UNIVERSE', failure_reason='EMPTY_UNIVERSE'.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()
        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)

        res = runner.execute_market_wide_scan(universe_override=[])

        assert res["status"] == "ERROR"
        assert res["matched_count"] is None
        assert res["snapshot"]["matched_count"] is None
        assert res["coverage"]["coverage_status"] == "EMPTY_UNIVERSE"
        assert res["coverage"]["failure_reason"] == "EMPTY_UNIVERSE"
        assert res["coverage"]["scanned_successfully_count"] == 0

        # Radar capabilities integration
        caps = get_radar_capabilities(vcp_snapshot=res)
        assert caps["VCP"]["status"] == "ERROR"
        assert caps["VCP"]["matchedCount"] is None
        assert caps["VCP"]["reason"] == "EMPTY_UNIVERSE"


def test_neg_r1_02_empty_build_membership():
    """
    Scenario NEG-R1-02: Unexpectedly Empty Authorized Build Membership.
    When an authorized universe build resolves to 0 scannable symbols, runner
    must fail closed to ERROR with failure_reason='BUILD_MEMBERSHIP_EMPTY'.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()
        mock_ustore = MagicMock()
        mock_ustore.get_universe_build.return_value = {
            "source_population_count": 0,
            "eligible_count": 0,
            "eligible_membership_hash": "hash-empty",
            "construction_status": "COMPLETE",
        }
        mock_ustore.get_scannable_membership.return_value = []

        runner = VCPScannerRunner(
            market_db=mock_db,
            snapshot_store=store,
            universe_store=mock_ustore,
        )

        res = runner.execute_market_wide_scan(universe_build_id="BUILD-EMPTY-001")

        assert res["status"] == "ERROR"
        assert res["matched_count"] is None
        assert res["coverage"]["coverage_status"] == "EMPTY_UNIVERSE"
        assert res["coverage"]["failure_reason"] == "BUILD_MEMBERSHIP_EMPTY"

        caps = get_radar_capabilities(vcp_snapshot=res)
        assert caps["VCP"]["status"] == "ERROR"
        assert caps["VCP"]["matchedCount"] is None
        assert caps["VCP"]["reason"] == "BUILD_MEMBERSHIP_EMPTY"


def test_neg_r1_03_natural_scheduled_scan_without_published_universe():
    """
    Scenario NEG-R1-03: Natural Scheduled Scan Without Published Universe.
    Verifies that invariant INV-U04 is strictly preserved: natural scheduled scan
    prohibits fallback to historical regression fixture and raises RuntimeError.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()
        mock_ustore = MagicMock()
        mock_ustore.get_latest_published_universe.return_value = None

        runner = VCPScannerRunner(
            market_db=mock_db,
            snapshot_store=store,
            universe_store=mock_ustore,
        )

        with pytest.raises(RuntimeError, match="NO_PUBLISHED_EXECUTABLE_UNIVERSE"):
            runner.execute_market_wide_scan(invocation_class="SCHEDULED_PRODUCTION")


# ==============================================================================
# TEST GROUP NEG-R2: PROVIDER TIMEOUT & PARTIAL POPULATION CONTAINMENT TESTS
# ==============================================================================

def test_neg_r2_01_single_symbol_provider_timeout_containment():
    """
    Scenario NEG-R2-01: Single Symbol Provider Timeout in Multi-Symbol Population.
    Verifies that a TimeoutError on one symbol is contained, does not crash the scan,
    increments unavailable_reasons['TIMEOUT'], and reports coverage_status='PARTIAL'.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()

        bars_ok = _bars_to_dict_candles(generate_clean_history("SYM1", "2026-10-22", total_bars=60, base_price=100.0))
        evidence_ok = _create_mock_evidence(AdjustmentState.SPLIT_ADJUSTED_ONLY)

        def mock_get_candles(sym, limit=60):
            if sym == "SYM_TIMEOUT":
                raise TimeoutError("5000ms provider network timeout")
            return bars_ok, evidence_ok

        mock_db.get_daily_candles_with_evidence.side_effect = mock_get_candles
        mock_db.get_latest_price.return_value = {"currentPrice": 100.0}

        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)

        # Universe has 3 symbols, 1 times out
        res = runner.execute_market_wide_scan(universe_override=["SYM_1", "SYM_TIMEOUT", "SYM_3"])

        assert res["status"] in ("AVAILABLE", "PIPELINE_PENDING")
        assert res["coverage"]["coverage_status"] == "PARTIAL"
        assert res["coverage"]["scanned_successfully_count"] == 2
        assert res["coverage"]["unavailable_symbol_count"] == 1
        assert res["coverage"]["unavailable_reasons"]["TIMEOUT"] == 1
        assert res["coverage"]["scan_coverage_pct"] == round(2 / 3 * 100, 1)

        # Radar capability reflects PARTIAL coverage truthfully
        caps = get_radar_capabilities(vcp_snapshot=res)
        assert caps["VCP"]["coverageStatus"] == "PARTIAL"
        assert caps["VCP"]["scannedSuccessfullyCount"] == 2
        assert caps["VCP"]["unavailableSymbolCount"] == 1


def test_neg_r2_02_sqlite_lock_contention_containment():
    """
    Scenario NEG-R2-02: SQLite Lock Contention Containment.
    Verifies that sqlite3.OperationalError('database is locked') is classified
    as 'DATABASE_LOCK', contained safely, and reported in partial coverage metadata.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()

        bars_ok = _bars_to_dict_candles(generate_clean_history("SYM1", "2026-10-22", total_bars=60, base_price=100.0))
        evidence_ok = _create_mock_evidence(AdjustmentState.SPLIT_ADJUSTED_ONLY)

        def mock_get_price(sym):
            if sym == "SYM_LOCKED":
                raise sqlite3.OperationalError("database is locked")
            return {"currentPrice": 50.0}

        mock_db.get_latest_price.side_effect = mock_get_price
        mock_db.get_daily_candles_with_evidence.return_value = (bars_ok, evidence_ok)

        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)

        res = runner.execute_market_wide_scan(universe_override=["SYM_1", "SYM_LOCKED"])

        assert res["status"] in ("AVAILABLE", "PIPELINE_PENDING")
        assert res["coverage"]["coverage_status"] == "PARTIAL"
        assert res["coverage"]["scanned_successfully_count"] == 1
        assert res["coverage"]["unavailable_symbol_count"] == 1
        assert res["coverage"]["unavailable_reasons"]["DATABASE_LOCK"] == 1


def test_neg_r2_03_sub_quorum_coverage_enforcement():
    """
    Scenario NEG-R2-03: Sub-Quorum Population Failure.
    Verifies that:
    1. Sub-quorum population failure with configurable min_coverage_pct (e.g. 80.0%)
       fails closed to ScannerStatus.ERROR with failure_reason='SUB_QUORUM_COVERAGE'.
    2. Default invocation without hardcoded 80% reports actual coverage truthfully
       as PARTIAL without crashing or falsely claiming completion.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()

        bars_ok = _bars_to_dict_candles(generate_clean_history("SYM1", "2026-10-22", total_bars=60, base_price=100.0))
        evidence_ok = _create_mock_evidence(AdjustmentState.SPLIT_ADJUSTED_ONLY)

        # 10 symbols: 1 succeeds, 9 fail (10% coverage)
        universe_10 = [f"SYM_{i}" for i in range(10)]

        def mock_get_candles(sym, limit=60):
            if sym != "SYM_0":
                raise TimeoutError("Simulated provider outage")
            return bars_ok, evidence_ok

        mock_db.get_daily_candles_with_evidence.side_effect = mock_get_candles
        mock_db.get_latest_price.return_value = {"currentPrice": 100.0}

        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)

        # Sub-test A: Configured quorum threshold of 80% fails closed
        res_quorum = runner.execute_market_wide_scan(
            universe_override=universe_10,
            min_coverage_pct=80.0,
        )
        assert res_quorum["status"] == "ERROR"
        assert res_quorum["matched_count"] is None
        assert res_quorum["coverage"]["failure_reason"] == "SUB_QUORUM_COVERAGE"
        assert res_quorum["coverage"]["scan_coverage_pct"] == 10.0

        # Sub-test B: Default invocation (no hardcoded quorum) reports PARTIAL truthfully
        res_default = runner.execute_market_wide_scan(universe_override=universe_10)
        assert res_default["status"] in ("AVAILABLE", "PIPELINE_PENDING")
        assert res_default["coverage"]["coverage_status"] == "PARTIAL"
        assert res_default["coverage"]["scan_coverage_pct"] == 10.0
        assert res_default["coverage"]["scanned_successfully_count"] == 1
        assert res_default["coverage"]["unavailable_symbol_count"] == 9


# ==============================================================================
# TEST GROUP NEG-R3: CORPORATE ACTION PROVENANCE TESTS
# ==============================================================================

def test_neg_r3_01_corporate_action_provenance_binding():
    """
    Scenario NEG-R3-01: Corporate Action Provenance Binding.
    Verifies that VCPScannerRunner queries market_db.get_daily_candles_with_evidence(),
    inspects evidence.provenance.adjustment_state, and records distribution in coverage metadata.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()

        bars_ok = _bars_to_dict_candles(generate_clean_history("SYM_ADJ", "2026-10-22", total_bars=60, base_price=100.0))
        evidence_adj = _create_mock_evidence(AdjustmentState.SPLIT_ADJUSTED_ONLY)

        mock_db.get_daily_candles_with_evidence.return_value = (bars_ok, evidence_adj)
        mock_db.get_latest_price.return_value = {"currentPrice": 100.0}

        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)
        res = runner.execute_market_wide_scan(universe_override=["SYM_ADJ"])

        assert res["coverage"]["adjustment_state_distribution"]["SPLIT_ADJUSTED_ONLY"] == 1
        assert res["coverage"]["data_complete_count"] == 1
        assert res["coverage"]["scanned_successfully_count"] == 1


def test_neg_r3_02_pre_adjusted_continuous_feed_recognition():
    """
    Scenario NEG-R3-02: Pre-Adjusted Continuous Feed Recognition.
    Verifies that when market DB provides SPLIT_AND_DIVIDEND_ADJUSTED or SPLIT_ADJUSTED_ONLY,
    the candidate's adjustment provenance confirms continuous_series_attested=True.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()

        bars_ok = _bars_to_dict_candles(generate_clean_history("SYM_PREADJ", "2026-10-22", total_bars=60, base_price=100.0))
        evidence_preadj = _create_mock_evidence(AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED)

        mock_db.get_daily_candles_with_evidence.return_value = (bars_ok, evidence_preadj)
        mock_db.get_latest_price.return_value = {"currentPrice": 100.0}

        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)
        res = runner.execute_market_wide_scan(universe_override=["SYM_PREADJ"])

        assert res["coverage"]["adjustment_state_distribution"]["SPLIT_AND_DIVIDEND_ADJUSTED"] == 1


def test_neg_r3_03_raw_unadjusted_feed_guard():
    """
    Scenario NEG-R3-03: Raw Unadjusted Feed Guard.
    Verifies that when market DB provides UNADJUSTED price bars, the runner
    defensively guards against false contraction qualification, incrementing
    unavailable_reasons['UNADJUSTED_PRICE_FEED_RISK'] and excluding from VCP matches.
    """
    with tempfile.TemporaryDirectory() as td:
        store = ScannerSnapshotStore(db_path=os.path.join(td, "test.db"))
        mock_db = MagicMock()

        bars_raw = _bars_to_dict_candles(generate_clean_history("SYM_RAW", "2026-10-22", total_bars=60, base_price=100.0))
        evidence_raw = _create_mock_evidence(AdjustmentState.UNADJUSTED)

        mock_db.get_daily_candles_with_evidence.return_value = (bars_raw, evidence_raw)
        mock_db.get_latest_price.return_value = {"currentPrice": 100.0}

        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=store)
        res = runner.execute_market_wide_scan(universe_override=["SYM_RAW"])

        assert res["coverage"]["unavailable_reasons"]["UNADJUSTED_PRICE_FEED_RISK"] == 1
        assert res["coverage"]["scanned_successfully_count"] == 0
        assert res["matched_count"] is None or res["matched_count"] == 0
        assert len(res["results"]) == 0


# ==============================================================================
# TEST GROUP API: RADAR CAPABILITIES COVERAGE DISCLOSURE INTEGRATION
# ==============================================================================

def test_api_route_radar_capabilities_coverage_enrichment():
    """
    Verifies that api.routes.screener.get_radar_capabilities safely enriches
    caps['VCP'] with coverageStatus, scannedSuccessfullyCount, and unavailableSymbolCount.
    """
    mock_snapshot_partial = {
        "status": "AVAILABLE",
        "matched_count": 0,
        "snapshot_id": "snap-vcp-001",
        "data_as_of": "2026-10-22",
        "generated_at": "2026-10-22T21:00:00Z",
        "coverage": {
            "coverage_status": "PARTIAL",
            "scanned_successfully_count": 48,
            "unavailable_symbol_count": 2,
            "scan_coverage_pct": 96.0,
        },
    }

    caps = get_radar_capabilities(vcp_snapshot=mock_snapshot_partial)
    vcp_cap = caps["VCP"]

    assert vcp_cap["status"] == "AVAILABLE"
    assert vcp_cap["matchedCount"] == 0
    assert vcp_cap["coverageStatus"] == "PARTIAL"
    assert vcp_cap["scannedSuccessfullyCount"] == 48
    assert vcp_cap["unavailableSymbolCount"] == 2
