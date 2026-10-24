"""
Comprehensive Acceptance and Verification Suite for Minervini VCP Universe Screening Activation.

Authorization: ARX_RADAR_MINERVINI_VCP_LOCAL_IMPLEMENTATION_AUTHORIZED
Remediation: ARX_VCP_HISTORY_BOUNDARY_VERIFICATION_AND_LOCAL_REMEDIATION_AUTHORIZED
Milestone 1: Minervini VCP Implementation & History Boundary Remediation

Tests:
1. History boundary requirement (199, 200, 221, 222, 251, 252 bars) across observations, predicates, and runner
2. Canonical VCP matching setup (Stage 2, progressive tightening, volume dry-up) -> AVAILABLE with match
3. Canonical non-matching setup (Stage 4 / uncontracted / flat) -> AVAILABLE with 0 matches
4. Sub-200 bar rejection -> INSUFFICIENT_DATA and uncomputed status -> PIPELINE_PENDING with matchedCount=None
5. Mixed universe partial coverage reporting (evaluated vs excluded counts and reasons)
6. All assets insufficient history -> PIPELINE_PENDING with matchedCount=None
7. Corrupt / invalid / non-finite candle data defense (fail-closed ERROR)
8. Radar API capability and count mapping parity for operational VCP
"""

import math
import os
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest
import pandas as pd

from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    CANONICAL_VCP_UNIVERSE,
)
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    CURRENT_IMPLEMENTATION_RELEASE_SHA,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.conformance_oracle import generate_clean_history
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
from api.routes.screener import get_radar_capabilities


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test_scanner.db")
        store = ScannerSnapshotStore(db_path=db_path)
        yield store


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


# =====================================================================
# 1. EXACT HISTORY BOUNDARY TESTS (199, 200, 221, 222, 251, 252)
# =====================================================================

def test_vcp_exact_history_boundary_observations_and_predicates():
    """
    Exhaustive Domain Evaluation Across Exact History Boundaries:
    Tests lengths 199, 200, 221, 222, 251, and 252 with an eligible positive VCP fixture.

    Asserts:
    - 199 bars: 200-SMA and slope are None. PRED_SUFFICIENT_HISTORY fails closed to INSUFFICIENT_DATA.
      PRED_TREND_TEMPLATE and PRED_STAGE_2 return INSUFFICIENT_DATA. vcp_qualified is False.
    - 200 bars: 200-SMA computable. 200-SMA slope computes fallback baseline (0.0). PRED_SUFFICIENT_HISTORY passes.
      52w window is min(200, 252) = 200 bars. Trend template and Stage 2 pass. vcp_qualified is True.
    - 221 bars: 200-SMA slope computes 21-session lag via earliest available SMA fallback. vcp_qualified is True.
    - 222 bars: 200-SMA slope computes exact 22-session lag without fallback (bar 221 vs bar 199). vcp_qualified is True.
    - 251 bars: 52w window is min(251, 252) = 251 bars. vcp_qualified is True.
    - 252 bars: 52w window is min(252, 252) = 252 bars (full 1-year benchmark). vcp_qualified is True.
    """
    compiler = VCPTemporalCaseCompiler()
    classifier = VCPClassifier()

    lengths = [199, 200, 221, 222, 251, 252]
    eval_results = {}

    for L in lengths:
        bars = generate_clean_history(
            symbol=f"SYM_{L}",
            as_of_date="2026-10-22",
            total_bars=L,
            base_price=100.0,
            trend_type="STAGE_2_UPTREND",
            contractions=[(0.25, 30), (0.12, 20), (0.04, 10)],
            final_vol_ratio=0.35,
        )
        pkg = compiler.compile_case(
            case_id=f"CASE-{L}",
            security_id=f"SYM_{L}",
            symbol=f"SYM_{L}",
            evaluation_as_of="2026-10-22T21:00:00Z",
            raw_bars=bars,
            reference_data={"rs_rank": 85.0},
            corporate_actions=[],
        )
        obs = classifier.extract_observations(pkg)
        preds = classifier.evaluate_predicates(obs)
        assessment = classifier.assess_domain_classification(pkg, preds, obs)
        eval_results[L] = (obs, preds, assessment)

    # 199 Boundary Assertions
    obs_199, preds_199, ass_199 = eval_results[199]
    assert obs_199.session_count == 199
    assert obs_199.sma_50 is not None
    assert obs_199.sma_150 is not None
    assert obs_199.sma_200 is None
    assert obs_199.sma_200_slope_22 is None
    assert preds_199["PRED_SUFFICIENT_HISTORY"].status.value == "INSUFFICIENT_DATA"
    assert "INSUFFICIENT_HISTORY_SESSIONS" in preds_199["PRED_SUFFICIENT_HISTORY"].reason_codes
    assert preds_199["PRED_TREND_TEMPLATE"].status.value == "INSUFFICIENT_DATA"
    assert "TREND_TEMPLATE_DATA_INCOMPLETE" in preds_199["PRED_TREND_TEMPLATE"].reason_codes
    assert preds_199["PRED_STAGE_2"].status.value == "INSUFFICIENT_DATA"
    assert "STAGE_2_DATA_MISSING" in preds_199["PRED_STAGE_2"].reason_codes
    assert ass_199.stage_classification == "STAGE_UNRESOLVED"
    assert ass_199.vcp_classification == "VCP_INSUFFICIENT_DATA"
    assert ass_199.vcp_qualified is False

    # 200 Boundary Assertions
    obs_200, preds_200, ass_200 = eval_results[200]
    assert obs_200.session_count == 200
    assert obs_200.sma_200 is not None
    assert obs_200.sma_200_slope_22 == 0.0  # Fallback baseline per numeric contract line 143
    assert preds_200["PRED_SUFFICIENT_HISTORY"].status.value == "PASS"
    assert preds_200["PRED_TREND_TEMPLATE"].status.value == "PASS"
    assert preds_200["PRED_STAGE_2"].status.value == "PASS"
    assert ass_200.stage_classification == "STAGE_2"
    assert ass_200.vcp_classification == "VCP_QUALIFIED"
    assert ass_200.vcp_qualified is True

    # 221 Boundary Assertions
    obs_221, preds_221, ass_221 = eval_results[221]
    assert obs_221.session_count == 221
    assert obs_221.sma_200_slope_22 > 0.0  # Measured across 21 available SMA bars
    assert preds_221["PRED_SUFFICIENT_HISTORY"].status.value == "PASS"
    assert ass_221.vcp_qualified is True

    # 222 Boundary Assertions (Exact 22-session lag without fallback)
    obs_222, preds_222, ass_222 = eval_results[222]
    assert obs_222.session_count == 222
    assert obs_222.sma_200_slope_22 > 0.0  # Exact 22-session lag: bar 221 vs bar 199
    assert preds_222["PRED_SUFFICIENT_HISTORY"].status.value == "PASS"
    assert ass_222.vcp_qualified is True

    # 251 Boundary Assertions
    obs_251, preds_251, ass_251 = eval_results[251]
    assert obs_251.session_count == 251
    assert preds_251["PRED_SUFFICIENT_HISTORY"].status.value == "PASS"
    assert ass_251.vcp_qualified is True

    # 252 Boundary Assertions (Full 52-week trading year benchmark)
    obs_252, preds_252, ass_252 = eval_results[252]
    assert obs_252.session_count == 252
    assert preds_252["PRED_SUFFICIENT_HISTORY"].status.value == "PASS"
    assert ass_252.vcp_qualified is True


def test_vcp_runner_all_six_history_boundaries(temp_store):
    """
    Runner Integration Across All 6 Boundaries:
    Executes VCPScannerRunner across each boundary individually with the conforming positive fixture.

    - 199 bars: Excluded under INSUFFICIENT_HISTORY. Scanned=0, Unavailable=1. Status=PIPELINE_PENDING, matchedCount=None.
    - 200, 221, 222, 251, 252 bars: Scanned=1, Unavailable=0, Matched=1. Status=AVAILABLE, matchedCount=1, lifecycle=SCANNER_OPERATIONAL.
    """
    lengths = [199, 200, 221, 222, 251, 252]

    for L in lengths:
        bars = generate_clean_history(
            symbol=f"SYM_{L}",
            as_of_date="2026-10-22",
            total_bars=L,
            base_price=100.0,
            trend_type="STAGE_2_UPTREND",
            contractions=[(0.25, 30), (0.12, 20), (0.04, 10)],
            final_vol_ratio=0.35,
        )
        mock_db = MagicMock()
        mock_db.get_daily_candles.return_value = _bars_to_dict_candles(bars)
        mock_db.get_latest_price.return_value = {"currentPrice": bars[-1].close}

        runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
        res = runner.execute_market_wide_scan(universe_override=[f"SYM_{L}"])
        caps = get_radar_capabilities(vcp_snapshot=res)
        vcp_cap = caps["VCP"]

        if L == 199:
            assert res["status"] == ScannerStatus.PIPELINE_PENDING.value
            assert res["snapshot"]["matched_count"] is None
            assert res["coverage"]["scanned_successfully_count"] == 0
            assert res["coverage"]["unavailable_symbol_count"] == 1
            assert res["coverage"]["unavailable_reasons"]["INSUFFICIENT_HISTORY"] == 1
            assert len(res["results"]) == 0
            assert vcp_cap["status"] == "PIPELINE_PENDING"
            assert vcp_cap["matchedCount"] is None
        else:
            assert res["status"] == ScannerStatus.AVAILABLE.value
            assert res["snapshot"]["matched_count"] == 1
            assert res["coverage"]["scanned_successfully_count"] == 1
            assert res["coverage"]["unavailable_symbol_count"] == 0
            assert len(res["results"]) == 1
            assert vcp_cap["status"] == "AVAILABLE"
            assert vcp_cap["matchedCount"] == 1
            assert vcp_cap["lifecycleState"] == "SCANNER_OPERATIONAL"


# =====================================================================
# 2. CANONICAL VCP MATCHING AND NON-MATCHING TESTS
# =====================================================================

def test_vcp_canonical_matching_candidate_qualification(temp_store):
    """
    Canonical Positive Match:
    An asset exhibiting Stage 2 advance, progressive tightening contractions (25%, 12%, 4%),
    and volume dry-up qualifies authoritatively as VCP_QUALIFIED.
    It is published in the snapshot with score, breakout pivot, stop loss, and reason codes.
    """
    matching_bars = generate_clean_history(
        symbol="QUAL",
        as_of_date="2026-10-22",
        total_bars=252,
        base_price=100.0,
        trend_type="STAGE_2_UPTREND",
        contractions=[(0.25, 30), (0.12, 20), (0.04, 10)],
        final_vol_ratio=0.35,
    )
    last_price = matching_bars[-1].close
    mock_db = MagicMock()
    mock_db.get_daily_candles.return_value = _bars_to_dict_candles(matching_bars)
    mock_db.get_latest_price.return_value = {"currentPrice": last_price}

    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
    res = runner.execute_market_wide_scan(universe_override=["QUAL"])

    assert res["status"] == ScannerStatus.AVAILABLE.value
    assert res["snapshot"]["matched_count"] == 1
    assert len(res["results"]) == 1

    cand = res["results"][0]
    assert cand["symbol"] == "QUAL"
    assert cand["current_price"] == last_price
    assert 0.0 < cand["score"] <= 100.0
    assert cand["rank"] == 1

    ev = cand["scanner_evidence"]
    assert ev["vcp_stage"] == "VCP 3-Stage Compression Confirmed"
    assert ev["stage_phase"] == "Stage 2 Advancing Growth Phase"
    assert ev["vcp_classification"] == "VCP_QUALIFIED"
    assert ev["stage_classification"] == "STAGE_2"
    assert ev["breakout_pivot"] is not None
    assert ev["stop_loss"] is not None
    assert "PROGRESSIVE_TIGHTENING_CONFIRMED" in ev["classification_reason_codes"]
    assert "TREND_TEMPLATE_SATISFIED" in ev["classification_reason_codes"]


def test_vcp_canonical_stage_4_non_matching(temp_store):
    """
    Canonical Stage 4 Markdown Non-Match:
    An asset in a secular Stage 4 decline fails trend template and Stage 2 predicates.
    It is authoritatively classified as VCP_NON_QUALIFIED and excluded from matches.
    """
    bars_stage4 = generate_clean_history(
        symbol="DOWN",
        as_of_date="2026-10-22",
        total_bars=252,
        base_price=150.0,
        trend_type="STAGE_4_DECLINE",
    )
    mock_db = MagicMock()
    mock_db.get_daily_candles.return_value = _bars_to_dict_candles(bars_stage4)
    mock_db.get_latest_price.return_value = {"currentPrice": bars_stage4[-1].close}

    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
    res = runner.execute_market_wide_scan(universe_override=["DOWN"])

    assert res["status"] == ScannerStatus.AVAILABLE.value
    assert res["snapshot"]["matched_count"] == 0
    assert len(res["results"]) == 0
    assert res["coverage"]["scanned_successfully_count"] == 1


# =====================================================================
# 3. MIXED UNIVERSE AND PARTIAL COVERAGE TRUTHFULNESS
# =====================================================================

def test_vcp_mixed_universe_partial_coverage_truthfulness(temp_store):
    """
    Mixed Universe Partial Coverage Truthfulness:
    In a universe with:
    - QUAL252: 252 bars, positive VCP match
    - FLAT200: 200 bars, computed non-match (flat Stage 1 base)
    - SHORT199: 199 bars, sub-200 bar insufficient history

    Asserts:
    - Status is AVAILABLE (valid evaluations exist)
    - matched_count == 1 (QUAL252 matched)
    - results has 1 candidate (QUAL252)
    - scanned_successfully_count == 2 (QUAL252 and FLAT200 evaluated)
    - unavailable_symbol_count == 1 (SHORT199 excluded)
    - unavailable_reasons["INSUFFICIENT_HISTORY"] == 1
    - scan_coverage_pct == 66.7%
    - coverage_status == "PARTIAL"
    - Zero coverage inflation; SHORT199 is never counted as an authoritative non-match
    """
    bars_qual_252 = generate_clean_history("QUAL252", "2026-10-22", total_bars=252, base_price=100.0,
                                           trend_type="STAGE_2_UPTREND",
                                           contractions=[(0.25, 30), (0.12, 20), (0.04, 10)],
                                           final_vol_ratio=0.35)
    bars_flat_200 = generate_clean_history("FLAT200", "2026-10-22", total_bars=200, base_price=100.0,
                                           trend_type="STAGE_1_BASING")
    bars_short_199 = generate_clean_history("SHORT199", "2026-10-22", total_bars=199, base_price=50.0)

    data_map = {
        "QUAL252": _bars_to_dict_candles(bars_qual_252),
        "FLAT200": _bars_to_dict_candles(bars_flat_200),
        "SHORT199": _bars_to_dict_candles(bars_short_199),
    }

    mock_db = MagicMock()
    mock_db.get_daily_candles.side_effect = lambda sym, limit=252: data_map.get(sym, [])
    mock_db.get_latest_price.side_effect = lambda sym: {"currentPrice": 100.0}

    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
    res = runner.execute_market_wide_scan(universe_override=["QUAL252", "FLAT200", "SHORT199"])
    caps = get_radar_capabilities(vcp_snapshot=res)

    assert res["status"] == ScannerStatus.AVAILABLE.value
    assert res["snapshot"]["matched_count"] == 1
    assert len(res["results"]) == 1
    assert res["results"][0]["symbol"] == "QUAL252"
    assert res["coverage"]["scanned_successfully_count"] == 2
    assert res["coverage"]["unavailable_symbol_count"] == 1
    assert res["coverage"]["unavailable_reasons"]["INSUFFICIENT_HISTORY"] == 1
    assert res["coverage"]["coverage_status"] == "PARTIAL"
    assert res["coverage"]["scan_coverage_pct"] == 66.7
    assert caps["VCP"]["status"] == "AVAILABLE"
    assert caps["VCP"]["matchedCount"] == 1


def test_vcp_all_assets_insufficient_history_emits_pending(temp_store):
    """
    All Assets Insufficient History:
    In a universe where every asset has < 200 bars (e.g. 60, 150, 199 bars),
    the scanner cannot evaluate the 200-session trend template for any asset.
    Status fails closed to PIPELINE_PENDING with matched_count=None.
    """
    bars_60 = generate_clean_history("SHORT60", "2026-10-22", total_bars=60, base_price=50.0)
    bars_150 = generate_clean_history("SHORT150", "2026-10-22", total_bars=150, base_price=70.0)
    bars_199 = generate_clean_history("SHORT199", "2026-10-22", total_bars=199, base_price=90.0)

    data_map = {
        "SHORT60": _bars_to_dict_candles(bars_60),
        "SHORT150": _bars_to_dict_candles(bars_150),
        "SHORT199": _bars_to_dict_candles(bars_199),
    }

    mock_db = MagicMock()
    mock_db.get_daily_candles.side_effect = lambda sym, limit=252: data_map.get(sym, [])
    mock_db.get_latest_price.side_effect = lambda sym: {"currentPrice": 100.0}

    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
    res = runner.execute_market_wide_scan(universe_override=["SHORT60", "SHORT150", "SHORT199"])
    caps = get_radar_capabilities(vcp_snapshot=res)

    assert res["status"] == ScannerStatus.PIPELINE_PENDING.value
    assert res["snapshot"]["matched_count"] is None
    assert res["coverage"]["matched_count"] is None
    assert res["coverage"]["scanned_successfully_count"] == 0
    assert res["coverage"]["unavailable_symbol_count"] == 3
    assert res["coverage"]["unavailable_reasons"]["INSUFFICIENT_HISTORY"] == 3
    assert len(res["results"]) == 0
    assert caps["VCP"]["status"] == "PIPELINE_PENDING"
    assert caps["VCP"]["matchedCount"] is None


def test_vcp_valid_evaluated_universe_zero_matches(temp_store):
    """
    Valid Evaluated Universe with Zero Matches:
    In a universe where all assets have >= 200 bars but none qualify (e.g. flat Stage 1 basing),
    the scan genuinely completes and evaluates all assets.
    Status is AVAILABLE with matched_count=0, coverage_status=COMPLETE, lifecycleState=SCANNER_OPERATIONAL.
    """
    bars_200 = generate_clean_history("FLAT200", "2026-10-22", total_bars=200, base_price=100.0,
                                      trend_type="STAGE_1_BASING")
    bars_252 = generate_clean_history("FLAT252", "2026-10-22", total_bars=252, base_price=120.0,
                                      trend_type="STAGE_1_BASING")

    data_map = {
        "FLAT200": _bars_to_dict_candles(bars_200),
        "FLAT252": _bars_to_dict_candles(bars_252),
    }

    mock_db = MagicMock()
    mock_db.get_daily_candles.side_effect = lambda sym, limit=252: data_map.get(sym, [])
    mock_db.get_latest_price.side_effect = lambda sym: {"currentPrice": 100.0}

    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
    res = runner.execute_market_wide_scan(universe_override=["FLAT200", "FLAT252"])
    caps = get_radar_capabilities(vcp_snapshot=res)

    assert res["status"] == ScannerStatus.AVAILABLE.value
    assert res["snapshot"]["matched_count"] == 0
    assert res["coverage"]["matched_count"] == 0
    assert res["coverage"]["scanned_successfully_count"] == 2
    assert res["coverage"]["unavailable_symbol_count"] == 0
    assert res["coverage"]["coverage_status"] == "COMPLETE"
    assert len(res["results"]) == 0
    assert caps["VCP"]["status"] == "AVAILABLE"
    assert caps["VCP"]["matchedCount"] == 0
    assert caps["VCP"]["lifecycleState"] == "SCANNER_OPERATIONAL"


# =====================================================================
# 4. CORRUPT AND INVALID DATA DEFENSE
# =====================================================================

def test_vcp_corrupt_candle_excluded_fail_closed(temp_store):
    """
    Data Corruption Defense:
    If an asset contains non-positive or non-finite OHLC prices,
    it is defensively excluded under INVALID_PRICE_DATA and fails closed to ERROR.
    """
    bars_corrupt = _bars_to_dict_candles(generate_clean_history("CORRUPT", "2026-10-22", total_bars=252, base_price=100.0))
    # Inject non-finite price
    bars_corrupt[50]["close"] = float("nan")

    mock_db = MagicMock()
    mock_db.get_daily_candles.return_value = bars_corrupt
    mock_db.get_latest_price.return_value = {"currentPrice": 100.0}

    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
    res = runner.execute_market_wide_scan(universe_override=["CORRUPT"])

    assert res["status"] == ScannerStatus.ERROR.value
    assert res["coverage"]["scanned_successfully_count"] == 0
    assert res["coverage"]["unavailable_reasons"]["INVALID_PRICE_DATA"] == 1


# =====================================================================
# 5. RADAR API CAPABILITY MAPPING PARITY
# =====================================================================

def test_vcp_radar_capabilities_operational_mapping(temp_store):
    """
    API Mapping Parity:
    An AVAILABLE VCP snapshot correctly maps to capabilities['VCP'] with:
    - status = AVAILABLE
    - universeScreening = AVAILABLE
    - matchedCount = integer >= 0 (e.g. 0)
    - populationDenominator = universe_size (e.g. 35)
    - lifecycleState = SCANNER_OPERATIONAL
    """
    bars_252 = generate_clean_history("TEST", "2026-10-22", total_bars=252, base_price=100.0)
    mock_db = MagicMock()
    mock_db.get_daily_candles.return_value = _bars_to_dict_candles(bars_252)
    mock_db.get_latest_price.return_value = {"currentPrice": 100.0}

    runner = VCPScannerRunner(market_db=mock_db, snapshot_store=temp_store)
    scan_res = runner.execute_market_wide_scan(universe_override=["TEST"])

    caps = get_radar_capabilities(vcp_snapshot=scan_res)
    vcp_cap = caps["VCP"]

    assert vcp_cap["status"] == "AVAILABLE"
    assert vcp_cap["universeScreening"] == "AVAILABLE"
    assert vcp_cap["matchedCount"] == 0
    assert vcp_cap["lifecycleState"] == "SCANNER_OPERATIONAL"
    assert vcp_cap["dataAsOf"] == scan_res["data_as_of"]
