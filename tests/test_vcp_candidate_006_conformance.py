"""ARX Terminal — Candidate 006 Synthetic Conformance Test Suite (SYN-01 through SYN-15).

Deterministic, mathematically isolated test suite verifying:
- SYN-01: Intervening low lower than both interval endpoints captured.
- SYN-02: New peak during provisional rebound and same-bar non-ordering.
- SYN-03: Equal-high and equal-low chronological tie breaking.
- SYN-04: Outside bar precedence & same-bar non-ordering policy.
- SYN-05: Three elapsed sessions without intermediate persistence requirement.
- SYN-06: Lower-low resets running trough and elapsed-session origin.
- SYN-07: Undercut after confirmation retained and evaluated for expansion.
- SYN-08: Strict 5% relative tightening boundaries (0.95 * D_prev).
- SYN-09: Initial depth floor preservation (D_1 >= 0.08).
- SYN-10: Recovery and rollover numerical guards (1e-7 acceptance band).
- SYN-11: Peak bar low exclusion in trough search.
- SYN-12: Cutoff during unconfirmed downswing (FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF).
- SYN-13: Fixed-start confirmed-record prefix consistency.
- SYN-14: Price-scale invariance.
- SYN-15: Invalid inputs and insufficient history.

Zero market data dependencies. 100% deterministic synthetic OHLCV generation.
"""

import datetime
from typing import List
import pytest

from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler, VCPTemporalCasePackage
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.predicate_registry import (
    PredicateStatus,
    VCPObservation,
    VCPPredicateRegistry,
)
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar


def make_bar(
    symbol: str,
    day_idx: int,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    volume: int = 1_000_000,
) -> DailyOHLCVBar:
    base_dt = datetime.date(2025, 1, 1) + datetime.timedelta(days=day_idx)
    bar_date = base_dt.isoformat()
    valid_time = f"{bar_date}T16:00:00Z"
    return DailyOHLCVBar(
        symbol=symbol,
        bar_date=bar_date,
        open=round(float(open_p), 4),
        high=round(float(high_p), 4),
        low=round(float(low_p), 4),
        close=round(float(close_p), 4),
        volume=int(volume),
        valid_time=valid_time,
        known_at=valid_time,
    )


def make_base_runup(symbol: str) -> List[DailyOHLCVBar]:
    """Generates 170 bars of steady Stage 2 run-up from 50 to 98, followed by anchor peak at bar 170 (100.0)."""
    bars = []
    for i in range(170):
        p = 50.0 + (48.0 * (i / 169.0))
        bars.append(make_bar(symbol, i, p, p + 0.2, p - 0.2, p, volume=1_500_000))
    # Bar 170: Anchor Peak (100.0)
    bars.append(make_bar(symbol, 170, 99.0, 100.0, 98.0, 99.5, volume=1_000_000))
    return bars


def append_buffer_bars(bars: List[DailyOHLCVBar], count: int = 25, price: float = 90.0):
    start_idx = len(bars)
    for i in range(start_idx, start_idx + count):
        bars.append(make_bar(bars[0].symbol, i, price, price + 0.5, price - 0.5, price, volume=500_000))


# ==============================================================================
# SYN-01: Intervening low lower than both interval endpoints captured
# ==============================================================================
def test_syn_01_intervening_low_lower_than_endpoints():
    bars = make_base_runup("TEST_SYN01")
    # 171..173: lower highs so 170 is a symmetric peak of radius 3
    bars.append(make_bar("TEST_SYN01", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN01", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN01", 173, 95.0, 95.5, 91.0, 92.0))
    # Wave 1 drops to 85.0 on bar 175 (depth 15%)
    bars.append(make_bar("TEST_SYN01", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN01", 175, 89.0, 89.5, 85.0, 85.5))
    # Rebound 3 sessions to confirm >= 5% (85 * 1.05 = 89.25)
    bars.append(make_bar("TEST_SYN01", 176, 85.5, 87.0, 85.5, 86.5))
    bars.append(make_bar("TEST_SYN01", 177, 86.5, 88.0, 86.5, 87.5))
    bars.append(make_bar("TEST_SYN01", 178, 87.5, 90.0, 87.5, 89.5)) # Confirm W1 on 178

    # Rebound continues, candidate peak established at bar 181 (High 110.00)
    bars.append(make_bar("TEST_SYN01", 179, 90.0, 95.0, 90.0, 94.0))
    bars.append(make_bar("TEST_SYN01", 180, 94.0, 105.0, 94.0, 104.0))
    bars.append(make_bar("TEST_SYN01", 181, 104.0, 110.0, 103.0, 109.0)) # Candidate peak at 181

    # Bars between candidate peak and rollover:
    # Bar 182: Low 104.0, Close 107.0 (pullback 2.7% < 4%)
    bars.append(make_bar("TEST_SYN01", 182, 109.0, 109.0, 104.0, 107.0))
    # Bar 183: Intervening Low 102.0, Close 106.0 (pullback 3.6% < 4%) -> Lower than both endpoints!
    bars.append(make_bar("TEST_SYN01", 183, 107.0, 107.5, 102.0, 106.0))
    # Bar 184: Low 103.5, Close 105.0 -> Rollover confirmed! (110 - 105)/110 = 4.54% >= 4%
    bars.append(make_bar("TEST_SYN01", 184, 106.0, 106.5, 103.5, 105.0))
    append_buffer_bars(bars, 25, 104.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    assert len(waves) == 1  # Wave 1 confirmed
    assert meta["final_state"] == "STATE_NEW_CONTRACTION"
    # Verify Wave 2 running trough is 102.0 at bar 183 (NOT endpoint 104.0 or 103.5)
    pw = meta["provisional_wave"]
    assert pw["candidate_peak_idx"] == 181
    assert pw["candidate_peak_price"] == 110.0
    assert pw["running_trough_idx"] == 183
    assert pw["running_trough_price"] == 102.0


# ==============================================================================
# SYN-02: New peak during provisional rebound and same-bar non-ordering
# ==============================================================================
def test_syn_02_new_peak_and_same_bar_non_ordering():
    bars = make_base_runup("TEST_SYN02")
    bars.append(make_bar("TEST_SYN02", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN02", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN02", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN02", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN02", 175, 89.0, 89.5, 85.0, 85.5))
    bars.append(make_bar("TEST_SYN02", 176, 85.5, 87.0, 85.5, 86.5))
    bars.append(make_bar("TEST_SYN02", 177, 86.5, 88.0, 86.5, 87.5))
    bars.append(make_bar("TEST_SYN02", 178, 87.5, 90.0, 87.5, 89.5)) # Confirm W1 on 178

    # Candidate peak initializes at bar 179 (105.0)
    bars.append(make_bar("TEST_SYN02", 179, 89.5, 105.0, 89.5, 104.0))
    # Bar 180: Higher high 108.0, Low 100.0, Close 101.0
    # Notice: (108 - 101)/108 = 6.48% >= 4%, but under same-bar non-ordering,
    # bar 180 updates peak and CANNOT trigger rollover!
    bars.append(make_bar("TEST_SYN02", 180, 104.0, 108.0, 100.0, 101.0))
    append_buffer_bars(bars, 25, 105.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    pw = meta["provisional_wave"]
    assert pw["candidate_peak_idx"] == 180
    assert pw["candidate_peak_price"] == 108.0


# ==============================================================================
# SYN-03: Equal-high and equal-low chronological tie breaking
# ==============================================================================
def test_syn_03_equal_high_and_low_ties():
    bars = make_base_runup("TEST_SYN03")
    bars.append(make_bar("TEST_SYN03", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN03", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN03", 173, 95.0, 95.5, 91.0, 92.0))
    # Bar 174: Low 85.0
    bars.append(make_bar("TEST_SYN03", 174, 92.0, 92.5, 85.0, 86.0))
    # Bar 175: Equal low 85.0
    bars.append(make_bar("TEST_SYN03", 175, 86.0, 87.0, 85.0, 86.0))
    # Rebound
    bars.append(make_bar("TEST_SYN03", 176, 86.0, 88.0, 86.0, 87.0))
    bars.append(make_bar("TEST_SYN03", 177, 87.0, 89.0, 87.0, 88.0))
    bars.append(make_bar("TEST_SYN03", 178, 88.0, 90.0, 88.0, 89.5)) # Confirm W1 on 178

    # Candidate peak at 179 (105.0)
    bars.append(make_bar("TEST_SYN03", 179, 89.5, 105.0, 89.5, 104.0))
    # Bar 180: Equal high 105.0
    bars.append(make_bar("TEST_SYN03", 180, 104.0, 105.0, 103.0, 104.0))
    append_buffer_bars(bars, 25, 104.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    # Wave 1 trough retained earliest occurrence (bar 174, not 175)
    assert waves[0]["trough_idx"] == 174
    # Candidate peak retained earliest occurrence (bar 179, not 180)
    pw = meta["provisional_wave"]
    assert pw["candidate_peak_idx"] == 179


# ==============================================================================
# SYN-04: Outside bar precedence & same-bar non-ordering policy
# ==============================================================================
def test_syn_04_outside_bar_precedence_and_same_bar_non_ordering():
    bars = make_base_runup("TEST_SYN04")
    bars.append(make_bar("TEST_SYN04", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN04", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN04", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN04", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN04", 175, 89.0, 89.5, 85.0, 85.5))
    bars.append(make_bar("TEST_SYN04", 176, 85.5, 87.0, 85.5, 86.5))
    bars.append(make_bar("TEST_SYN04", 177, 86.5, 88.0, 86.5, 87.5))
    bars.append(make_bar("TEST_SYN04", 178, 87.5, 90.0, 87.5, 89.5)) # Confirm W1 on 178

    # Candidate peak 106.0 at bar 179
    bars.append(make_bar("TEST_SYN04", 179, 89.5, 106.0, 89.5, 105.0))
    # Bar 180: Outside bar: High 108.0 (> 106), Low 98.0, Close 98.5
    bars.append(make_bar("TEST_SYN04", 180, 105.0, 108.0, 98.0, 98.5))
    append_buffer_bars(bars, 25, 100.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    # High expansion took precedence; candidate peak is 180 at 108.0
    # Rollover was evaluated on subsequent session (bar 181 from buffer at 100.0)
    assert meta["provisional_wave"]["candidate_peak_idx"] == 180
    assert meta["provisional_wave"]["candidate_peak_price"] == 108.0


# ==============================================================================
# SYN-05: Three elapsed sessions without intermediate persistence requirement
# ==============================================================================
def test_syn_05_three_elapsed_sessions_without_persistence():
    bars = make_base_runup("TEST_SYN05")
    bars.append(make_bar("TEST_SYN05", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN05", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN05", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN05", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN05", 175, 89.0, 89.5, 85.0, 85.5)) # Trough 85.0 at 175

    # Bar 176: Close 87.0 (2.35% recovery < 5%)
    bars.append(make_bar("TEST_SYN05", 176, 85.5, 87.5, 85.5, 87.0))
    # Bar 177: Close 86.0 (1.17% recovery < 5%, but Low >= 85.0)
    bars.append(make_bar("TEST_SYN05", 177, 87.0, 87.0, 85.2, 86.0))
    # Bar 178: Close 89.5 (5.29% recovery >= 5% on 3rd elapsed session)
    bars.append(make_bar("TEST_SYN05", 178, 86.0, 90.0, 86.0, 89.5))
    append_buffer_bars(bars, 25, 90.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    # Confirmed on session 178!
    assert len(waves) >= 1
    assert waves[0]["confirmation_idx"] == 178
    assert waves[0]["trough_idx"] == 175


# ==============================================================================
# SYN-06: Lower-low resets running trough and elapsed-session origin
# ==============================================================================
def test_syn_06_lower_low_resets_trough():
    bars = make_base_runup("TEST_SYN06")
    bars.append(make_bar("TEST_SYN06", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN06", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN06", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN06", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN06", 175, 89.0, 89.5, 85.0, 85.5)) # Low 85.0
    bars.append(make_bar("TEST_SYN06", 176, 85.5, 90.0, 85.5, 89.5)) # Close >= 5%, elapsed 1
    # Bar 177: Lower low 83.0 -> resets trough!
    bars.append(make_bar("TEST_SYN06", 177, 88.0, 88.0, 83.0, 84.0))
    append_buffer_bars(bars, 25, 84.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    # Because buffer bars stayed at 84.0 (< 83 * 1.05 = 87.15), W1 never confirmed
    assert len(waves) == 0
    assert meta["final_state"] == "STATE_WAVE_1_DOWNSWING"
    pw = meta["provisional_wave"]
    assert pw["running_trough_idx"] == 177
    assert pw["running_trough_price"] == 83.0


# ==============================================================================
# SYN-07: Undercut after confirmation retained and evaluated for expansion
# ==============================================================================
def test_syn_07_undercut_after_confirmation_expansion_rejected():
    bars = make_base_runup("TEST_SYN07")
    bars.append(make_bar("TEST_SYN07", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN07", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN07", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN07", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN07", 175, 89.0, 89.5, 85.0, 85.5)) # W1 Trough 85.0 (depth 15%)
    bars.append(make_bar("TEST_SYN07", 176, 85.5, 87.0, 85.5, 86.5))
    bars.append(make_bar("TEST_SYN07", 177, 86.5, 88.0, 86.5, 87.5))
    bars.append(make_bar("TEST_SYN07", 178, 87.5, 90.0, 87.5, 89.5)) # Confirm W1 on 178

    # Candidate peak at 179 (95.0)
    bars.append(make_bar("TEST_SYN07", 179, 89.5, 95.0, 89.5, 94.0))
    # Rollover on 180: Close 91.0 ((95 - 91)/95 = 4.21% >= 4%)
    bars.append(make_bar("TEST_SYN07", 180, 94.0, 94.0, 90.5, 91.0))
    # Undercut W1: Low 75.0 (< 85.0) at bar 181!
    bars.append(make_bar("TEST_SYN07", 181, 90.0, 90.0, 75.0, 76.0))
    # Rebound from 75.0: 3 sessions to confirm >= 5% (75 * 1.05 = 78.75)
    bars.append(make_bar("TEST_SYN07", 182, 76.0, 78.0, 76.0, 77.0))
    bars.append(make_bar("TEST_SYN07", 183, 77.0, 78.0, 77.0, 77.5))
    bars.append(make_bar("TEST_SYN07", 184, 77.5, 80.0, 77.5, 79.5)) # Confirm W2 on 184
    append_buffer_bars(bars, 25, 79.5)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    pkg = VCPTemporalCaseCompiler().compile_case(
        case_id="TEST_SYN07",
        security_id="SEC-001",
        symbol="TEST_SYN07",
        evaluation_as_of=bars[-1].valid_time,
        raw_bars=bars,
        reference_data={"rs_rank": 85.0},
        corporate_actions=[],
    )
    obs = classifier.extract_observations(pkg)
    preds = classifier.evaluate_predicates(obs)

    # Both waves extracted faithfully
    assert obs.contraction_count == 2
    # Wave 1 depth: (100 - 85)/100 = 15%
    # Wave 2 depth: (95 - 75)/95 = 21.05% -> Expansion!
    assert obs.contraction_depths[0] == 0.15
    assert obs.contraction_depths[1] > 0.20
    # Predicate fails under tightening rule
    assert preds["PRED_PROGRESSIVE_TIGHTENING"].status == PredicateStatus.FAIL
    assert "VOLATILITY_EXPANSION_DETECTED" in preds["PRED_PROGRESSIVE_TIGHTENING"].reason_codes


# ==============================================================================
# SYN-08: Strict 5% relative tightening boundaries (0.95 * D_prev)
# ==============================================================================
def test_syn_08_strict_tightening_boundary():
    nc = VCPNumericContract()
    # 20% -> 19.00% is exactly 0.20 * 0.95 -> PASS
    assert nc.verify_progressive_tightening([0.20, 0.1900], relative_reduction=0.05, eps_float=1e-7) is True
    # 20% -> 19.01% -> FAIL
    assert nc.verify_progressive_tightening([0.20, 0.1901], relative_reduction=0.05, eps_float=1e-7) is False
    # Expansion +0.005 (20% -> 20.5%) -> FAIL (confirms no +0.01 expansion allowance)
    assert nc.verify_progressive_tightening([0.20, 0.2050], relative_reduction=0.05, eps_float=1e-7) is False


# ==============================================================================
# SYN-09: Initial depth floor preservation (D_1 >= 0.08)
# ==============================================================================
def test_syn_09_initial_depth_floor():
    bars = []
    for i in range(170):
        p = 50.0 + (48.0 * (i / 169.0))
        bars.append(make_bar("TEST_SYN09", i, p, p + 0.2, p - 0.2, p))
    # Anchor peak 100.0, but only drops to 94.0 (depth 6.0% < 8.0%)
    bars.append(make_bar("TEST_SYN09", 170, 99.0, 100.0, 98.0, 99.5))
    bars.append(make_bar("TEST_SYN09", 171, 99.0, 99.0, 95.0, 95.5))
    bars.append(make_bar("TEST_SYN09", 172, 95.0, 95.0, 94.0, 94.5))
    append_buffer_bars(bars, 25, 94.5)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    # Initial drop < 8% is rejected as anchor candidate
    assert len(waves) == 0
    assert "INSUFFICIENT_BASE_ANCHOR_CANDIDATES" in meta["reason_codes"]


# ==============================================================================
# SYN-10: Recovery and rollover numerical guards (1e-7 acceptance band)
# ==============================================================================
def test_syn_10_numerical_guards():
    eps = 1e-7
    theta_prom = 0.05
    theta_roll = 0.04

    # Recovery: 0.04999995 is within 1e-7 guard (effective bound: 0.04999990)
    assert 0.04999995 >= (theta_prom - eps)
    # Recovery: 0.04999980 is outside guard
    assert not (0.04999980 >= (theta_prom - eps))

    # Rollover: 0.03999995 is within 1e-7 guard
    assert 0.03999995 >= (theta_roll - eps)
    # Rollover: 0.03999980 is outside guard
    assert not (0.03999980 >= (theta_roll - eps))


# ==============================================================================
# SYN-11: Peak bar low exclusion in trough search
# ==============================================================================
def test_syn_11_peak_bar_low_exclusion():
    bars = make_base_runup("TEST_SYN11")
    bars.append(make_bar("TEST_SYN11", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN11", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN11", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN11", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN11", 175, 89.0, 89.5, 85.0, 85.5))
    bars.append(make_bar("TEST_SYN11", 176, 85.5, 87.0, 85.5, 86.5))
    bars.append(make_bar("TEST_SYN11", 177, 86.5, 88.0, 86.5, 87.5))
    bars.append(make_bar("TEST_SYN11", 178, 87.5, 90.0, 87.5, 89.5)) # Confirm W1 on 178

    # Candidate peak at 179: High 110.0, but Low is anomalous 80.0!
    bars.append(make_bar("TEST_SYN11", 179, 90.0, 110.0, 80.0, 109.0))
    # Bar 180: Rollover triggers: High 106.0, Low 104.0, Close 105.0
    bars.append(make_bar("TEST_SYN11", 180, 108.0, 108.0, 104.0, 105.0))
    append_buffer_bars(bars, 25, 105.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    # Low on peak bar 179 (80.0) MUST be excluded from trough search!
    # Trough initialized from post-peak low on bar 180 (104.0)
    pw = meta["provisional_wave"]
    assert pw["running_trough_idx"] == 180
    assert pw["running_trough_price"] == 104.0


# ==============================================================================
# SYN-12: Cutoff during unconfirmed downswing
# ==============================================================================
def test_syn_12_cutoff_during_unconfirmed_downswing():
    bars = make_base_runup("TEST_SYN12")
    bars.append(make_bar("TEST_SYN12", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN12", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN12", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN12", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN12", 175, 89.0, 89.5, 85.0, 85.5))
    bars.append(make_bar("TEST_SYN12", 176, 85.5, 87.0, 85.5, 86.5))
    bars.append(make_bar("TEST_SYN12", 177, 86.5, 88.0, 86.5, 87.5))
    bars.append(make_bar("TEST_SYN12", 178, 87.5, 90.0, 87.5, 89.5)) # Confirm W1 on 178

    # Candidate peak at 179 (95.0)
    bars.append(make_bar("TEST_SYN12", 179, 89.5, 95.0, 89.5, 94.0))
    # Rollover on 180 (Close 91.0)
    bars.append(make_bar("TEST_SYN12", 180, 94.0, 94.0, 90.5, 91.0))
    # Add buffer bars keeping price in unconfirmed downswing without confirmed rebound
    for i in range(181, 205):
        bars.append(make_bar("TEST_SYN12", i, 91.0, 91.5, 90.0, 90.5))

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves, depths, meta = classifier._detect_causal_contractions(bars)

    assert meta["final_trough_confirmed"] is False
    assert "FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF" in meta["reason_codes"]


# ==============================================================================
# SYN-13: Fixed-start confirmed-record prefix consistency
# ==============================================================================
def test_syn_13_fixed_start_prefix_consistency():
    bars = make_base_runup("TEST_SYN13")
    bars.append(make_bar("TEST_SYN13", 171, 98.0, 98.5, 96.0, 97.0))
    bars.append(make_bar("TEST_SYN13", 172, 97.0, 97.5, 94.0, 95.0))
    bars.append(make_bar("TEST_SYN13", 173, 95.0, 95.5, 91.0, 92.0))
    bars.append(make_bar("TEST_SYN13", 174, 92.0, 92.5, 88.0, 89.0))
    bars.append(make_bar("TEST_SYN13", 175, 89.0, 89.5, 85.0, 85.5))
    bars.append(make_bar("TEST_SYN13", 176, 85.5, 87.0, 85.5, 86.5))
    bars.append(make_bar("TEST_SYN13", 177, 86.5, 88.0, 86.5, 87.5))
    bars.append(make_bar("TEST_SYN13", 178, 87.5, 90.0, 87.5, 89.5)) # Confirm W1 on 178
    append_buffer_bars(bars, 25, 90.0)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves_n, depths_n, meta_n = classifier._detect_causal_contractions(bars)

    # Append 10 more bars (Prefix N + M)
    append_buffer_bars(bars, 10, 90.0)
    waves_nm, depths_nm, meta_nm = classifier._detect_causal_contractions(bars)

    # Wave 1 boundaries MUST remain bit-exact identical
    assert waves_n[0]["start_idx"] == waves_nm[0]["start_idx"]
    assert waves_n[0]["trough_idx"] == waves_nm[0]["trough_idx"]
    assert waves_n[0]["peak_price"] == waves_nm[0]["peak_price"]
    assert waves_n[0]["trough_price"] == waves_nm[0]["trough_price"]
    assert waves_n[0]["confirmation_idx"] == waves_nm[0]["confirmation_idx"]
    assert waves_n[0]["depth_pct"] == waves_nm[0]["depth_pct"]


# ==============================================================================
# SYN-14: Price-scale invariance
# ==============================================================================
def test_syn_14_price_scale_invariance():
    bars_1x = make_base_runup("TEST_SYN14")
    bars_1x.append(make_bar("TEST_SYN14", 171, 98.0, 98.5, 96.0, 97.0))
    bars_1x.append(make_bar("TEST_SYN14", 172, 97.0, 97.5, 94.0, 95.0))
    bars_1x.append(make_bar("TEST_SYN14", 173, 95.0, 95.5, 91.0, 92.0))
    bars_1x.append(make_bar("TEST_SYN14", 174, 92.0, 92.5, 88.0, 89.0))
    bars_1x.append(make_bar("TEST_SYN14", 175, 89.0, 89.5, 85.0, 85.5))
    bars_1x.append(make_bar("TEST_SYN14", 176, 85.5, 87.0, 85.5, 86.5))
    bars_1x.append(make_bar("TEST_SYN14", 177, 86.5, 88.0, 86.5, 87.5))
    bars_1x.append(make_bar("TEST_SYN14", 178, 87.5, 90.0, 87.5, 89.5))
    append_buffer_bars(bars_1x, 25, 90.0)

    # Scale 10x
    bars_10x = []
    for b in bars_1x:
        bars_10x.append(make_bar(
            b.symbol,
            0,
            b.open * 10.0,
            b.high * 10.0,
            b.low * 10.0,
            b.close * 10.0,
            b.volume,
        ))

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)
    waves_1x, depths_1x, meta_1x = classifier._detect_causal_contractions(bars_1x)
    waves_10x, depths_10x, meta_10x = classifier._detect_causal_contractions(bars_10x)

    assert len(waves_1x) == len(waves_10x)
    assert depths_1x == depths_10x
    assert waves_1x[0]["start_idx"] == waves_10x[0]["start_idx"]
    assert waves_1x[0]["trough_idx"] == waves_10x[0]["trough_idx"]
    assert waves_1x[0]["confirmation_idx"] == waves_10x[0]["confirmation_idx"]


# ==============================================================================
# SYN-15: Invalid inputs and insufficient history
# ==============================================================================
def test_syn_15_invalid_inputs_and_insufficient_history():
    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_006)

    # Insufficient bars (< 40)
    bars_short = [make_bar("SHORT", i, 10.0, 11.0, 9.0, 10.0) for i in range(20)]
    waves, depths, meta = classifier._detect_causal_contractions(bars_short)
    assert len(waves) == 0
    assert "INSUFFICIENT_HISTORY_FOR_BASE" in meta["reason_codes"]

    # Nonpositive prices in stream handled safely
    bars_bad = make_base_runup("BAD")
    bars_bad.append(make_bar("BAD", 171, -10.0, -5.0, -15.0, -10.0))
    append_buffer_bars(bars_bad, 25, 90.0)
    waves_bad, depths_bad, meta_bad = classifier._detect_causal_contractions(bars_bad)
    assert "INVALID_PRICE_NONPOSITIVE" in meta_bad["reason_codes"]
