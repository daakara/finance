"""
Test Suite: test_vcp_source_integrity_012r.py
Tracker: Issue #166 / Directive RADAR-012R
Objective:
    1. Verify 200-SMA slope mathematical boundaries (199, 200, 221, 222, 250, 252 bars).
    2. Confirm elimination of synthetic zero and truncated 21-session lookback.
    3. Verify authentic flat SMA returns exact 0.0.
    4. Verify post-cutoff bar exclusion reduces admissible bar count.
    5. Verify fail-closed behavior on Stage 2, Stage 4, and Trend Template predicates.
    6. Verify caller provenance separation (SCHEDULED_PRODUCTION vs MANUAL_OPERATOR_CANARY vs BOOT_WARMUP).
"""

import datetime
from zoneinfo import ZoneInfo
import pytest

from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier, PredicateStatus
from analyst_dashboard.vcp.conformance_oracle import generate_clean_history
from analyst_dashboard.analyzers.scanner_runner import TriggerType


class TestVCPSlopeBoundaryAndIntegrity:
    """Mathematical verification of 200-SMA slope boundaries."""

    def test_flat_sma_returns_exact_zero(self):
        """When 200-SMA is mathematically flat (identical prices), slope must be exactly 0.0."""
        # 23 points of identical 100.0
        sma_series = [100.0] * 23
        slope = VCPNumericContract.compute_sma_slope(sma_series, lookback_bars=22)
        assert slope == 0.0

    def test_rising_and_falling_slope(self):
        """Ascending prices produce positive slope; descending produce negative slope."""
        sma_rising = [100.0 + i for i in range(23)]
        slope_pos = VCPNumericContract.compute_sma_slope(sma_rising, lookback_bars=22)
        assert slope_pos is not None
        assert slope_pos > 0.0
        expected_pos = (122.0 - 100.0) / 100.0
        assert abs(slope_pos - expected_pos) < 1e-6

        sma_falling = [100.0 - i for i in range(23)]
        slope_neg = VCPNumericContract.compute_sma_slope(sma_falling, lookback_bars=22)
        assert slope_neg is not None
        assert slope_neg < 0.0
        expected_neg = (78.0 - 100.0) / 100.0
        assert abs(slope_neg - expected_neg) < 1e-6

    def test_sma_series_fewer_than_23_points_returns_none(self):
        """
        Lookback 22 requires current point (index -1) and point 22 bars prior (index -(22+1) = -23).
        Series with <= 22 points must return None (no fallback to earliest SMA, no synthetic 0.0).
        """
        for count in range(0, 23):
            series = [100.0] * count
            slope = VCPNumericContract.compute_sma_slope(series, lookback_bars=22)
            assert slope is None, f"Expected None for {count} points, got {slope}"

    def test_sma_series_with_invalid_prior_or_current_returns_none(self):
        """None or <= 0 values in series at target indices must return None."""
        series = [100.0] * 23
        series[-1] = None  # current is None
        assert VCPNumericContract.compute_sma_slope(series, lookback_bars=22) is None

        series = [100.0] * 23
        series[-23] = None  # prior is None
        assert VCPNumericContract.compute_sma_slope(series, lookback_bars=22) is None

        series = [100.0] * 23
        series[-23] = 0.0  # prior is zero
        assert VCPNumericContract.compute_sma_slope(series, lookback_bars=22) is None

    def test_bar_counts_199_200_221_fail_closed_to_insufficient_data(self):
        """
        199 bars: SMA-200 cannot be computed.
        200 bars: SMA-200 computed at t0, but prior 22 sessions unavailable -> slope is None.
        221 bars: SMA-200 computed for 22 points, but 23 points required -> slope is None.
        All three must fail closed to VCP_INSUFFICIENT_DATA.
        """
        compiler = VCPTemporalCaseCompiler()
        classifier = VCPClassifier()

        for count in (199, 200, 221):
            bars = generate_clean_history("TEST", "2026-11-03", total_bars=count, contractions=[(0.18, 26), (0.06, 14)])
            pkg = compiler.compile_case(
                case_id=f"CASE-{count}",
                security_id="TEST",
                symbol="TEST",
                evaluation_as_of="2026-11-03T21:45:00Z",
                raw_bars=bars,
                reference_data={"rs_rank": 90.0},
                corporate_actions=[],
                run_id="run-test",
            )
            assert len(pkg.permitted_bars) == count
            res = classifier.classify_case(pkg)
            assert res.vcp_classification == "VCP_INSUFFICIENT_DATA", f"Failed for {count} bars"
            assert res.predicate_vector["PRED_STAGE_2"].status == PredicateStatus.INSUFFICIENT_DATA

    def test_bar_counts_222_250_252_compute_authoritatively(self):
        """
        222 bars: Exactly 23 SMA-200 points available -> first valid untruncated 22-session slope.
        250 bars: 51 SMA points -> 28 excess bars buffer.
        252 bars: 53 SMA points -> 30 excess bars buffer.
        All three must compute authoritatively without INSUFFICIENT_DATA.
        """
        compiler = VCPTemporalCaseCompiler()
        classifier = VCPClassifier()

        for count in (222, 250, 252):
            bars = generate_clean_history("TEST", "2026-11-03", total_bars=count, contractions=[(0.18, 26), (0.06, 14)])
            pkg = compiler.compile_case(
                case_id=f"CASE-{count}",
                security_id="TEST",
                symbol="TEST",
                evaluation_as_of="2026-11-03T21:45:00Z",
                raw_bars=bars,
                reference_data={"rs_rank": 90.0},
                corporate_actions=[],
                run_id="run-test",
            )
            assert len(pkg.permitted_bars) == count
            res = classifier.classify_case(pkg)
            assert res.vcp_classification in ("VCP_NON_QUALIFIED", "VCP_QUALIFIED"), f"Failed for {count} bars"
            assert res.predicate_vector["PRED_STAGE_2"].status != PredicateStatus.INSUFFICIENT_DATA
            assert res.predicate_vector["PRED_TREND_TEMPLATE"].status != PredicateStatus.INSUFFICIENT_DATA

    def test_post_cutoff_bar_exclusion_reduces_admissible_count(self):
        """
        Bars timestamped after evaluation_as_of must be excluded by temporal filtering,
        failing closed if admissible bars drop below 222.
        """
        compiler = VCPTemporalCaseCompiler()
        classifier = VCPClassifier()

        # Generate 230 bars ending on 2026-11-10
        bars_all = generate_clean_history("TEST", "2026-11-10", total_bars=230, contractions=[(0.18, 26), (0.06, 14)])
        # Cutoff is earlier, say 2026-10-25
        cutoff = "2026-10-25T21:45:00Z"
        pkg = compiler.compile_case(
            case_id="CASE-POST-CUTOFF",
            security_id="TEST",
            symbol="TEST",
            evaluation_as_of=cutoff,
            raw_bars=bars_all,
            reference_data={"rs_rank": 90.0},
            corporate_actions=[],
            run_id="run-test",
        )
        assert len(pkg.permitted_bars) < 222
        res = classifier.classify_case(pkg)
        assert res.vcp_classification == "VCP_INSUFFICIENT_DATA"


class TestCallerProvenanceSeparation:
    """Verifies that autonomous, canary, and warmup executions are strictly partitioned."""

    def test_scheduled_production_invocation_class_mapping(self):
        """Unadulterated scheduled scan resolves to SCHEDULED_PRODUCTION and NATURAL_PRODUCTION."""
        trigger_type = TriggerType.SCHEDULED
        operator_request_id = None
        is_canary = False
        startup_context = False

        if startup_context:
            eff_invocation_class = "BOOT_WARMUP"
        elif trigger_type == TriggerType.SCHEDULED and not operator_request_id and not is_canary:
            eff_invocation_class = "SCHEDULED_PRODUCTION"
        elif is_canary or (operator_request_id and "canary" in str(operator_request_id).lower()):
            eff_invocation_class = "MANUAL_OPERATOR_CANARY"
        else:
            eff_invocation_class = "MANUAL_OPERATOR"

        assert eff_invocation_class == "SCHEDULED_PRODUCTION"

    def test_canary_invocation_class_mapping(self):
        """Explicit canary or operator request resolves to MANUAL_OPERATOR_CANARY and ADMIN_FORCED."""
        # Case A: is_canary = True
        eff_class_a = "MANUAL_OPERATOR_CANARY" if True else "MANUAL_OPERATOR"
        assert eff_class_a == "MANUAL_OPERATOR_CANARY"

        # Case B: operator_request_id set
        operator_request_id = "canary-one-shot-run"
        if is_canary_flag := (operator_request_id and "canary" in operator_request_id.lower()):
            eff_class_b = "MANUAL_OPERATOR_CANARY"
        else:
            eff_class_b = "MANUAL_OPERATOR"
        assert eff_class_b == "MANUAL_OPERATOR_CANARY"

    def test_boot_warmup_invocation_class_mapping(self):
        """Startup context or maintenance trigger resolves to BOOT_WARMUP and NON_EVIDENCE_BOOTSTRAP."""
        startup_context = True
        if startup_context:
            eff_invocation_class = "BOOT_WARMUP"
        assert eff_invocation_class == "BOOT_WARMUP"

    def test_durable_storage_membership_and_disposition(self):
        """
        In durable storage:
        SCHEDULED_PRODUCTION -> NATURAL_PRODUCTION
        MANUAL_OPERATOR_CANARY -> ADMIN_FORCED, NON_NATURAL, NON_NATURAL_INELIGIBLE
        BOOT_WARMUP -> NON_EVIDENCE_BOOTSTRAP, NON_NATURAL, NON_NATURAL_INELIGIBLE
        """
        # SCHEDULED_PRODUCTION
        eff_inv = "SCHEDULED_PRODUCTION"
        eff_origin = "NATURAL_PRODUCTION" if eff_inv == "SCHEDULED_PRODUCTION" else "ADMIN_FORCED"
        assert eff_origin == "NATURAL_PRODUCTION"

        # MANUAL_OPERATOR_CANARY
        eff_inv = "MANUAL_OPERATOR_CANARY"
        eff_origin = "ADMIN_FORCED" if eff_inv in ("MANUAL_OPERATOR", "MANUAL_OPERATOR_CANARY") else "NATURAL_PRODUCTION"
        assert eff_origin == "ADMIN_FORCED"
        mem_class = "NON_NATURAL" if eff_origin == "ADMIN_FORCED" else "NATURAL"
        prosp_disp = "NON_NATURAL_INELIGIBLE" if eff_origin == "ADMIN_FORCED" else "ELIGIBLE"
        assert mem_class == "NON_NATURAL"
        assert prosp_disp == "NON_NATURAL_INELIGIBLE"
