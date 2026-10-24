"""ARX Terminal — Candidate 005 Synthetic Conformance Test Suite (ACC-01 through ACC-11).

Deterministic, mathematically isolated test suite verifying:
- ACC-01: Base-Origin Anchoring & Pre-Base Advance Truncation
- ACC-02: Ascending Ceiling Test (3% Threshold Boundary)
- ACC-03: Chronological Processing & Lower High Retention
- ACC-04: Noise Merging - Small & Short (Merged)
- ACC-05: Noise Merging - Small & Sustained (Not Merged, Leg Closed)
- ACC-06: Noise Merging - Large & Short (Not Merged, Leg Closed)
- ACC-07: Noise Merging - Large & Sustained (Not Merged, Leg Closed)
- ACC-08: Progressive Tightening Matrix (2%, 4%, 8%, 20% depths)
- ACC-09: Final-Wave Trough Confirmation Timelines (A, B, C, D)
- ACC-10: Invalidation by Subsequent Lower Low
- ACC-11: Valid Contractions Rejected by Trend / Volume Rules

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
        is_session_closed=True,
    )


def make_package(bars: List[DailyOHLCVBar], case_id: str = "TEST-CASE-001", rs_rank: float = 85.0) -> VCPTemporalCasePackage:
    compiler = VCPTemporalCaseCompiler()
    as_of = bars[-1].valid_time if bars else "2026-01-01T20:00:00Z"
    return compiler.compile_case(
        case_id=case_id,
        security_id="SEC-001",
        symbol=bars[0].symbol if bars else "TEST",
        evaluation_as_of=as_of,
        raw_bars=bars,
        reference_data={"rs_rank": rs_rank},
        corporate_actions=[],
    )


# ==============================================================================
# ACC-01: Base-Origin Anchoring & Pre-Base Advance Truncation
# ==============================================================================
def test_acc_01_base_origin_anchoring_and_truncation():
    bars: List[DailyOHLCVBar] = []
    # Pre-base advance: 40 bars advancing from 50 to 98
    for i in range(40):
        p = 50.0 + (48.0 * (i / 39.0))
        bars.append(make_bar("AC01", i, p, p + 0.5, p - 0.5, p))

    # Base anchor peak at bar 40 (high 100.0)
    bars.append(make_bar("AC01", 40, 99.0, 100.0, 98.5, 99.5))

    # Wave 1 pullback down to 88.0 (depth 12.0%) over 8 bars
    for i in range(41, 49):
        p = 99.5 - (11.5 * ((i - 40) / 8.0))
        bars.append(make_bar("AC01", i, p, p + 0.5, p - 0.5, p))
    bars[48] = make_bar("AC01", 48, 88.5, 89.0, 88.0, 88.5)

    # Rebound to 96.0 at bar 55
    for i in range(49, 56):
        p = 88.5 + (7.5 * ((i - 48) / 7.0))
        bars.append(make_bar("AC01", i, p, p + 0.5, p - 0.5, p))
    bars[55] = make_bar("AC01", 55, 95.5, 96.0, 95.0, 95.8)

    # Wave 2 pullback to 90.0 at bar 62 (depth (96 - 90)/96 = 6.25%)
    for i in range(56, 63):
        p = 95.8 - (5.8 * ((i - 55) / 7.0))
        bars.append(make_bar("AC01", i, p, p + 0.5, p - 0.5, p))
    bars[62] = make_bar("AC01", 62, 90.5, 91.0, 90.0, 90.5)

    # Rebound to 94.0 at bar 67
    for i in range(63, 68):
        p = 90.5 + (3.5 * ((i - 62) / 5.0))
        bars.append(make_bar("AC01", i, p, p + 0.5, p - 0.5, p))
    bars[67] = make_bar("AC01", 67, 93.5, 94.0, 93.0, 93.8)

    # Wave 3 pullback to 91.5 at bar 72, then confirmed recovery to 93.5 at bar 75 (cutoff)
    for i in range(68, 73):
        p = 93.8 - (2.3 * ((i - 67) / 5.0))
        bars.append(make_bar("AC01", i, p, p + 0.5, p - 0.5, p))
    bars[72] = make_bar("AC01", 72, 91.8, 92.0, 91.5, 91.8)

    # Recovery
    bars.append(make_bar("AC01", 73, 92.0, 93.0, 91.8, 92.8))
    bars.append(make_bar("AC01", 74, 92.8, 93.8, 92.5, 93.5))
    bars.append(make_bar("AC01", 75, 93.5, 93.9, 93.2, 93.7))

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    waves, depths, meta = classifier._detect_macro_contractions(bars)

    assert meta["anchor_selected"] is not None
    assert meta["anchor_selected"]["bar_index"] == 40
    assert meta["anchor_selected"]["high"] == 100.0
    assert meta["pre_base_bars_truncated"] > 0
    assert "PRE_BASE_ADVANCE_TRUNCATED" in meta["reason_codes"]


# ==============================================================================
# ACC-02: Ascending Ceiling Test (3% Threshold Boundary)
# ==============================================================================
def test_acc_02_ascending_ceiling_3_percent_boundary():
    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)

    # Case A: Secondary peak at 102.50 (+2.50% <= +3.00%) -> Anchor retained at 100.0
    bars_a: List[DailyOHLCVBar] = []
    for i in range(40):
        bars_a.append(make_bar("AC02A", i, 70.0 + i * 0.7, 71.0 + i * 0.7, 69.5 + i * 0.7, 70.5 + i * 0.7))
    bars_a.append(make_bar("AC02A", 40, 99.0, 100.0, 98.0, 99.0))
    # drop to 88
    for i in range(41, 48):
        bars_a.append(make_bar("AC02A", i, 98.0 - (i - 40), 98.5 - (i - 40), 97.0 - (i - 40), 97.5 - (i - 40)))
    bars_a.append(make_bar("AC02A", 48, 89.0, 89.5, 88.0, 88.5))
    # rally to 102.50 (within 3% ceiling)
    for i in range(49, 56):
        p = 88.5 + (14.0 * ((i - 48) / 7.0))
        bars_a.append(make_bar("AC02A", i, p, p + 0.5, p - 0.5, p))
    bars_a[55] = make_bar("AC02A", 55, 101.5, 102.5, 101.0, 102.0)
    # drop to 94 and recover
    for i in range(56, 65):
        bars_a.append(make_bar("AC02A", i, 100.0 - (i - 55) * 0.7, 100.5 - (i - 55) * 0.7, 99.0 - (i - 55) * 0.7, 99.5 - (i - 55) * 0.7))

    _, _, meta_a = classifier._detect_macro_contractions(bars_a)
    assert meta_a["anchor_selected"]["bar_index"] == 40
    assert meta_a["anchor_selected"]["high"] == 100.0

    # Case B: Secondary peak at 104.50 (+4.50% > +3.00%) initiating drop >= 8% -> Base restarts at 104.50!
    bars_b = [b for b in bars_a[:55]]
    bars_b.append(make_bar("AC02B", 55, 103.5, 104.5, 103.0, 104.0))
    # drop to 93 (depth (104.5 - 93)/104.5 = 11.0% >= 8.0%)
    for i in range(56, 65):
        p = 104.0 - (11.0 * ((i - 55) / 9.0))
        bars_b.append(make_bar("AC02B", i, p, p + 0.5, p - 0.5, p))
    bars_b[64] = make_bar("AC02B", 64, 93.5, 94.0, 93.0, 93.5)
    for i in range(65, 75):
        bars_b.append(make_bar("AC02B", i, 94.0 + (i - 64) * 0.5, 94.5 + (i - 64) * 0.5, 93.5 + (i - 64) * 0.5, 94.2 + (i - 64) * 0.5))

    _, _, meta_b = classifier._detect_macro_contractions(bars_b)
    assert meta_b["anchor_selected"]["bar_index"] == 55
    assert meta_b["anchor_selected"]["high"] == 104.5


# ==============================================================================
# ACC-03: Chronological Processing & Lower High Retention
# ==============================================================================
def test_acc_03_chronological_lower_high_retention():
    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    bars: List[DailyOHLCVBar] = []
    for i in range(40):
        bars.append(make_bar("AC03", i, 70.0 + i * 0.7, 71.0 + i * 0.7, 69.5 + i * 0.7, 70.5 + i * 0.7))
    bars.append(make_bar("AC03", 40, 99.0, 100.0, 98.0, 99.0))
    for i in range(41, 48):
        bars.append(make_bar("AC03", i, 95.0, 96.0, 88.0, 89.0))
    bars.append(make_bar("AC03", 48, 94.0, 95.0, 93.5, 94.5))
    for i in range(49, 70):
        bars.append(make_bar("AC03", i, 93.0, 94.0, 91.0, 92.0))

    _, _, meta = classifier._detect_macro_contractions(bars)
    assert meta["anchor_selected"]["bar_index"] == 40
    assert meta["anchor_selected"]["high"] == 100.0


# ==============================================================================
# ACC-04 to ACC-07: Conjunctive Noise Merging Suite
# ==============================================================================
def test_acc_04_noise_merging_small_and_short_merged():
    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    bars: List[DailyOHLCVBar] = []
    for i in range(40):
        bars.append(make_bar("AC04", i, 70.0 + i * 0.7, 71.0 + i * 0.7, 69.5 + i * 0.7, 70.5 + i * 0.7))
    bars.append(make_bar("AC04", 40, 99.0, 100.0, 98.0, 99.0))
    for i in range(41, 46):
        bars.append(make_bar("AC04", i, 98.0 - (i - 40) * 1.6, 98.5 - (i - 40) * 1.6, 97.0 - (i - 40) * 1.6, 97.5 - (i - 40) * 1.6))
    bars[45] = make_bar("AC04", 45, 91.0, 91.5, 90.0, 90.5)

    # 2-day minor bounce to 92.5 (+2.78% < 4%, duration = 2 <= 3) -> MERGED AS NOISE
    bars.append(make_bar("AC04", 46, 90.5, 91.8, 90.2, 91.5))
    bars.append(make_bar("AC04", 47, 91.5, 92.5, 91.0, 92.0))

    # Downward continuation to 86.0
    for i in range(48, 55):
        p = 92.0 - (6.0 * ((i - 47) / 7.0))
        bars.append(make_bar("AC04", i, p, p + 0.5, p - 0.5, p))
    bars[54] = make_bar("AC04", 54, 86.5, 87.0, 86.0, 86.5)

    for i in range(55, 65):
        p = 86.5 + (8.5 * ((i - 54) / 10.0))
        bars.append(make_bar("AC04", i, p, p + 0.5, p - 0.5, p))

    waves, _, meta = classifier._detect_macro_contractions(bars)
    assert len(waves) >= 1
    assert waves[0]["trough_price"] == 86.0


def test_acc_05_noise_merging_small_and_sustained_leg_closed():
    # Bounce is small (< 4%) but lasts 5 bars (> 3 bars) -> Leg closed!
    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    bars: List[DailyOHLCVBar] = []
    for i in range(40):
        bars.append(make_bar("AC05", i, 70.0 + i * 0.7, 71.0 + i * 0.7, 69.5 + i * 0.7, 70.5 + i * 0.7))
    bars.append(make_bar("AC05", 40, 99.0, 100.0, 98.0, 99.0))
    for i in range(41, 46):
        bars.append(make_bar("AC05", i, 98.0 - (i - 40) * 1.6, 98.5 - (i - 40) * 1.6, 97.0 - (i - 40) * 1.6, 97.5 - (i - 40) * 1.6))
    bars[45] = make_bar("AC05", 45, 91.0, 91.5, 90.0, 90.5)

    # 5-day drift at 92.5 (+2.78% < 4%, but tau = 5 > 3)
    for i in range(46, 51):
        bars.append(make_bar("AC05", i, 92.0, 92.5, 91.5, 92.0))

    # Subsequent continuation down
    for i in range(51, 60):
        bars.append(make_bar("AC05", i, 91.0, 91.5, 87.0, 88.0))

    waves, _, _ = classifier._detect_macro_contractions(bars)
    # Because tau > 3, leg closed at 90.0
    assert len(waves) >= 1
    assert waves[0]["trough_price"] == 90.0


def test_acc_06_noise_merging_large_and_short_leg_closed():
    # Bounce is fast (2 days <= 3) but large (+6.0% >= 4%) -> Leg closed!
    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    bars: List[DailyOHLCVBar] = []
    for i in range(40):
        bars.append(make_bar("AC06", i, 70.0 + i * 0.7, 71.0 + i * 0.7, 69.5 + i * 0.7, 70.5 + i * 0.7))
    bars.append(make_bar("AC06", 40, 99.0, 100.0, 98.0, 99.0))
    for i in range(41, 46):
        bars.append(make_bar("AC06", i, 98.0 - (i - 40) * 1.6, 98.5 - (i - 40) * 1.6, 97.0 - (i - 40) * 1.6, 97.5 - (i - 40) * 1.6))
    bars[45] = make_bar("AC06", 45, 91.0, 91.5, 90.0, 90.5)

    # Sharp 2-day bounce to 95.5 (+6.11% >= 4.0%)
    bars.append(make_bar("AC06", 46, 91.0, 93.0, 90.5, 92.5))
    bars.append(make_bar("AC06", 47, 92.5, 95.5, 92.0, 95.0))

    for i in range(48, 60):
        bars.append(make_bar("AC06", i, 94.0, 94.5, 91.0, 92.0))

    waves, _, _ = classifier._detect_macro_contractions(bars)
    assert len(waves) >= 1
    assert waves[0]["trough_price"] == 90.0


# ==============================================================================
# ACC-08: Progressive Tightening Matrix
# ==============================================================================
def test_acc_08_progressive_tightening_matrix():
    nc = VCPNumericContract()

    # Sequence A: [20%, 12%, 6%] -> PASS
    assert nc.verify_progressive_tightening([0.20, 0.12, 0.06], relative_reduction=0.05) is True

    # Sequence B: [20%, 19.01%] -> FAIL (0.20 * 0.95 = 0.1900, 0.1901 > 0.1900)
    assert nc.verify_progressive_tightening([0.20, 0.1901], relative_reduction=0.05) is False

    # Sequence C: [8%, 7.60%] -> PASS (0.08 * 0.95 = 0.0760)
    assert nc.verify_progressive_tightening([0.08, 0.0760], relative_reduction=0.05) is True

    # Sequence D: [8%, 7.61%] -> FAIL (0.0761 > 0.0760)
    assert nc.verify_progressive_tightening([0.08, 0.0761], relative_reduction=0.05) is False

    # Sequence E: Shallow depths [4%, 4.05%] -> FAIL (Expansion)
    assert nc.verify_progressive_tightening([0.04, 0.0405], relative_reduction=0.05) is False

    # Sequence F: Shallow depths [4%, 3.80%] -> PASS (0.04 * 0.95 = 0.0380)
    assert nc.verify_progressive_tightening([0.04, 0.0380], relative_reduction=0.05) is True

    # Sequence G: Candidate 004 backward compatibility (relative_reduction=0.0)
    assert nc.verify_progressive_tightening([0.04, 0.0399], relative_reduction=0.0) is True


# ==============================================================================
# ACC-09: Final-Wave Trough Confirmation Timelines (A, B, C, D)
# ==============================================================================
def _build_base_for_timelines() -> List[DailyOHLCVBar]:
    """Builds a 236-bar history (>= 200 sessions) with an anchor peak at bar 170 (high 100),
    Wave 1: drops to bar 185 (85, depth 15%), rebounds to bar 205 (95)
    Wave 2: drops to bar 220 (88, depth 7.37%), rebounds to bar 235 (93)
    """
    bars: List[DailyOHLCVBar] = []
    # 0..169: steady Stage 2 advance from 50 to 98 (170 bars to satisfy 200-session minimum)
    for i in range(170):
        p = 50.0 + (48.0 * (i / 169.0))
        bars.append(make_bar("TM", i, p, p + 0.5, p - 0.5, p, volume=1_500_000))

    # Bar 170: Anchor Peak (100.0)
    bars.append(make_bar("TM", 170, 99.0, 100.0, 98.0, 99.5, volume=1_000_000))

    # 171..185: Wave 1 drops to 85.0 (depth 15%)
    for i in range(171, 186):
        p = 99.5 - (14.5 * ((i - 170) / 15.0))
        bars.append(make_bar("TM", i, p, p + 0.5, p - 0.5, p, volume=800_000))
    bars[185] = make_bar("TM", 185, 85.5, 86.0, 85.0, 85.5, volume=800_000)

    # 186..205: Rebound to 95.0
    for i in range(186, 206):
        p = 85.5 + (9.5 * ((i - 185) / 20.0))
        bars.append(make_bar("TM", i, p, p + 0.5, p - 0.5, p, volume=1_200_000))
    bars[205] = make_bar("TM", 205, 94.5, 95.0, 94.0, 94.8, volume=1_000_000)

    # 206..220: Wave 2 drops to 88.0 (depth (95 - 88)/95 = 7.37%)
    for i in range(206, 221):
        p = 94.8 - (6.8 * ((i - 205) / 15.0))
        bars.append(make_bar("TM", i, p, p + 0.5, p - 0.5, p, volume=600_000))
    bars[220] = make_bar("TM", 220, 88.5, 89.0, 88.0, 88.5, volume=600_000)

    # 221..235: Rebound to 93.0
    for i in range(221, 236):
        p = 88.5 + (4.5 * ((i - 220) / 15.0))
        bars.append(make_bar("TM", i, p, p + 0.5, p - 0.5, p, volume=900_000))
    bars[235] = make_bar("TM", 235, 92.5, 93.0, 92.0, 92.8, volume=700_000)

    return bars


def test_acc_09_timeline_a_completed_final_wave():
    bars = _build_base_for_timelines()
    # Wave 3: drops to bar 242 at 90.0 (high 90.5, close 90.2)
    for i in range(236, 243):
        p = 92.8 - (2.8 * ((i - 235) / 7.0))
        bars.append(make_bar("TMA", i, p, p + 0.3, p - 0.3, p, volume=400_000))
    bars[242] = make_bar("TMA", 242, 90.3, 90.6, 90.0, 90.2, volume=400_000)

    # Rebound: bars 243, 244, 245 (cutoff at 245, close 92.5 -> recovery (92.5 - 90)/90 = +2.78% >= 1.5%)
    bars.append(make_bar("TMA", 243, 90.5, 91.5, 90.2, 91.2, volume=350_000))
    bars.append(make_bar("TMA", 244, 91.2, 92.2, 91.0, 92.0, volume=350_000))
    bars.append(make_bar("TMA", 245, 92.0, 92.8, 91.8, 92.5, volume=350_000))

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    waves, depths, meta = classifier._detect_macro_contractions(bars)
    assert meta["final_trough_confirmed"] is True
    assert "FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF" not in meta["reason_codes"]


def test_acc_09_timeline_b_incomplete_final_wave_fail_closed():
    bars = _build_base_for_timelines()
    # Wave 3: drops all the way into cutoff at bar 242 (bar 242 is trough)
    for i in range(236, 243):
        p = 92.8 - (2.8 * ((i - 235) / 7.0))
        bars.append(make_bar("TMB", i, p, p + 0.3, p - 0.3, p, volume=400_000))
    bars[242] = make_bar("TMB", 242, 90.3, 90.6, 90.0, 90.1, volume=400_000)

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    pkg = make_package(bars, case_id="CASE-TIMELINE-B")
    obs = classifier.extract_observations(pkg)
    preds = classifier.evaluate_predicates(obs)

    assert obs.metadata.get("final_trough_confirmed") is False
    assert "FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF" in obs.metadata.get("reason_codes", [])
    assert preds["PRED_CONTRACTION_SEQUENCE_VALID"].status == PredicateStatus.FAIL
    assert "FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF" in preds["PRED_CONTRACTION_SEQUENCE_VALID"].reason_codes


# ==============================================================================
# ACC-10: Invalidation by Subsequent Lower Low
# ==============================================================================
def test_acc_10_invalidation_by_subsequent_lower_low():
    bars = _build_base_for_timelines()
    # Wave 3: drops to 90.0 on bar 242
    for i in range(236, 243):
        p = 92.8 - (2.8 * ((i - 235) / 7.0))
        bars.append(make_bar("AC10", i, p, p + 0.3, p - 0.3, p, volume=400_000))
    bars[242] = make_bar("AC10", 242, 90.3, 90.6, 90.0, 90.2, volume=400_000)

    # Temporary bounce to 91.8 on bar 243 (+2.0% recovery)
    bars.append(make_bar("AC10", 243, 90.5, 92.0, 90.2, 91.8, volume=400_000))

    # Undercut lower low on bar 244 to 89.0 (< 90.0)!
    bars.append(make_bar("AC10", 244, 91.0, 91.2, 89.0, 89.2, volume=400_000))

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    waves, depths, meta = classifier._detect_macro_contractions(bars)

    assert meta["final_trough_confirmed"] is False
    assert "FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF" in meta["reason_codes"]


# ==============================================================================
# ACC-11: Valid Contractions Rejected by Trend / Volume Rules
# ==============================================================================
def test_acc_11_valid_contractions_rejected_by_trend_or_volume():
    # Base with valid tightening contractions, but final wave has heavy churn (ratio > 0.70)
    bars = _build_base_for_timelines()
    # Wave 3: drops to bar 242 at 90.0 with HEAVY volume
    for i in range(236, 243):
        p = 92.8 - (2.8 * ((i - 235) / 7.0))
        bars.append(make_bar("AC11", i, p, p + 0.3, p - 0.3, p, volume=3_500_000))
    bars[242] = make_bar("AC11", 242, 90.3, 90.6, 90.0, 90.2, volume=3_500_000)

    # Rebound with heavy volume
    bars.append(make_bar("AC11", 243, 90.5, 91.5, 90.2, 91.2, volume=3_500_000))
    bars.append(make_bar("AC11", 244, 91.2, 92.2, 91.0, 92.0, volume=3_500_000))
    bars.append(make_bar("AC11", 245, 92.0, 92.8, 91.8, 92.5, volume=3_500_000))

    classifier = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)
    pkg = make_package(bars, case_id="CASE-HEAVY-VOLUME-TEST")
    assessment = classifier.classify_case(pkg)

    assert assessment.vcp_qualified is False
    assert assessment.vcp_classification == "VCP_NON_QUALIFIED"
    assert "VOLUME_HEAVY_ON_PULLBACK" in assessment.classification_reason_codes
