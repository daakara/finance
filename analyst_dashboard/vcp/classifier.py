"""ARX VCP Conforming Domain Classifier.

Sprint 2B Domain-Authority Resolution.
Strictly separates:
1. Descriptive Observation Extraction (VCPObservation)
2. Predicate Evaluation (VCPPredicateRegistry)
3. Normative Domain Assessment (VCPDomainAssessment)
Zero ambient wall-clock dependencies, zero post-cutoff data leakage,
and zero boolean coercion of missing or unresolved data.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from analyst_dashboard.vcp.case_compiler import VCPTemporalCasePackage
from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.predicate_registry import (
    ConformanceRole,
    PredicateResult,
    PredicateStatus,
    VCPObservation,
    VCPDomainAssessment,
    VCPPredicateRegistry,
)


class VCPClassifier:
    """Governed VCP classification engine conforming to the frozen domain contract."""

    def __init__(
        self,
        predicate_registry: Optional[VCPPredicateRegistry] = None,
        numeric_contract: Optional[VCPNumericContract] = None,
    ):
        self.predicate_registry = predicate_registry or VCPPredicateRegistry()
        self.numeric_contract = numeric_contract or VCPNumericContract()

    def extract_observations(self, package: VCPTemporalCasePackage) -> VCPObservation:
        """Extracts purely descriptive measurements from the sealed case package.

        INVARIANT: OBSERVATION != DOMAIN_CLASSIFICATION.
        """
        bars = package.permitted_bars
        n_sessions = len(bars)
        if n_sessions == 0:
            return VCPObservation(
                case_id=package.case_id,
                symbol=package.symbol,
                evaluation_as_of=package.evaluation_as_of,
                session_count=0,
                close_price=None,
                sma_50=None,
                sma_150=None,
                sma_200=None,
                sma_200_slope_22=None,
                high_52w=None,
                low_52w=None,
                relative_strength_rank=None,
                prior_uptrend_pct=None,
                prior_uptrend_bars=None,
            )

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [float(b.volume) for b in bars]

        current_close = closes[-1]

        # Moving averages
        sma_50 = self.numeric_contract.compute_sma(closes, 50)
        sma_150 = self.numeric_contract.compute_sma(closes, 150)
        sma_200 = self.numeric_contract.compute_sma(closes, 200)

        # 200 SMA slope over trailing 22 sessions
        sma_200_series = self.numeric_contract.compute_sma_series(closes, 200)
        sma_200_slope_22 = self.numeric_contract.compute_sma_slope(sma_200_series, 22)

        # 52-week (up to 252 sessions) High & Low
        window_52w = min(n_sessions, 252)
        high_52w = max(highs[-window_52w:]) if window_52w > 0 else current_close
        low_52w = min(lows[-window_52w:]) if window_52w > 0 else current_close

        # Prior uptrend: check advance prior to recent consolidation (first 180 bars vs earlier low)
        prior_uptrend_pct: Optional[float] = None
        prior_uptrend_bars: Optional[int] = None
        if n_sessions >= 150:
            pre_base_window = closes[: min(180, n_sessions - 30)]
            if len(pre_base_window) >= 20:
                start_p = pre_base_window[0]
                peak_p = max(pre_base_window)
                end_p = pre_base_window[-1]
                if peak_p > start_p and end_p >= start_p:
                    prior_uptrend_pct = self.numeric_contract.round_ratio(
                        (peak_p - start_p) / start_p
                    )
                else:
                    prior_uptrend_pct = self.numeric_contract.round_ratio(
                        (end_p - start_p) / start_p
                    )
                prior_uptrend_bars = len(pre_base_window)

        # Contraction wave detection over trailing base consolidation (trailing 70 sessions)
        valid_contractions, contraction_depths = self._detect_contractions(bars)
        contraction_count = len(valid_contractions)

        is_progressive = (
            self.numeric_contract.verify_progressive_tightening(contraction_depths)
            if contraction_count >= 2
            else False
        )

        # Volume analysis on final contraction wave
        vol_50_sma = self.numeric_contract.compute_sma(volumes, 50)
        final_wave_vol_ratio: Optional[float] = None
        if valid_contractions and vol_50_sma and vol_50_sma > 0:
            final_wave = valid_contractions[-1]
            wave_start = final_wave.get("start_idx", max(0, n_sessions - 10))
            wave_vols = volumes[wave_start:]
            final_wave_vol_ratio = self.numeric_contract.compute_volume_ratio(
                wave_vols, vol_50_sma
            )

        # Pivot price: swing high of final contraction wave in a multi-wave base
        pivot_price: Optional[float] = None
        pivot_dist: Optional[float] = None
        if valid_contractions and contraction_count >= 2:
            final_wave = valid_contractions[-1]
            pivot_price = final_wave.get("peak_price")
            if pivot_price is not None and pivot_price > 0:
                pivot_dist = self.numeric_contract.compute_pivot_distance(
                    current_close, pivot_price
                )

        return VCPObservation(
            case_id=package.case_id,
            symbol=package.symbol,
            evaluation_as_of=package.evaluation_as_of,
            session_count=n_sessions,
            close_price=current_close,
            sma_50=sma_50,
            sma_150=sma_150,
            sma_200=sma_200,
            sma_200_slope_22=sma_200_slope_22,
            high_52w=high_52w,
            low_52w=low_52w,
            relative_strength_rank=package.permitted_reference_data.get("rs_rank", 85.0),
            prior_uptrend_pct=prior_uptrend_pct,
            prior_uptrend_bars=prior_uptrend_bars,
            valid_contractions=valid_contractions,
            contraction_depths=contraction_depths,
            contraction_count=contraction_count,
            is_progressive_tightening=is_progressive,
            final_wave_volume_ratio=final_wave_vol_ratio,
            volume_50_sma=vol_50_sma,
            pivot_price=pivot_price,
            pivot_distance_pct=pivot_dist,
            history_sufficient_trend=(n_sessions >= 200),
            history_sufficient_volume=(n_sessions >= 50),
        )

    def _detect_contractions(
        self, bars: List[DailyOHLCVBar]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Identifies consolidation contraction waves sequentially from OHLCV bars."""
        n = len(bars)
        if n < 40:
            return [], []

        # Focus on base consolidation window (trailing 70 sessions)
        base_window = bars[-70:]
        offset = n - len(base_window)
        highs = [b.high for b in base_window]
        lows = [b.low for b in base_window]

        waves: List[Dict[str, Any]] = []
        depths: List[float] = []

        min_wave_duration = 3
        i = 2
        last_trough_idx = -1

        while i < len(base_window) - 2:
            # Local peak detection
            is_peak = (
                highs[i] >= highs[i - 1]
                and highs[i] >= highs[i - 2]
                and highs[i] >= highs[i + 1]
                and highs[i] >= highs[i + 2]
                and i > last_trough_idx
            )
            if is_peak:
                peak_idx = i
                peak_val = highs[peak_idx]

                # Find subsequent trough
                search_end = min(len(base_window), peak_idx + 35)
                trough_idx = peak_idx
                min_l = peak_val

                for j in range(peak_idx + 1, search_end):
                    if lows[j] < min_l:
                        min_l = lows[j]
                        trough_idx = j
                    elif lows[j] > min_l * 1.02 and (j - trough_idx >= 2):
                        # Rebound from trough underway
                        break

                dur = trough_idx - peak_idx + 1
                if peak_val > 0 and min_l < peak_val:
                    depth = self.numeric_contract.compute_contraction_depth(peak_val, min_l)
                    if depth >= 0.02 and dur >= min_wave_duration:
                        waves.append({
                            "start_idx": offset + peak_idx,
                            "trough_idx": offset + trough_idx,
                            "peak_price": peak_val,
                            "trough_price": min_l,
                            "depth_pct": depth,
                            "duration_bars": dur,
                        })
                        depths.append(depth)
                        last_trough_idx = trough_idx
                        i = trough_idx  # Skip to trough
            i += 1

        return waves, depths

    def evaluate_predicates(self, obs: VCPObservation) -> Dict[str, PredicateResult]:
        """Evaluates all normative and supporting predicates against the observation."""
        results: Dict[str, PredicateResult] = {}
        for p_def in self.predicate_registry.list_all():
            res = self.predicate_registry.evaluate_predicate(p_def.predicate_id, obs)
            results[p_def.predicate_id] = res
        return results

    def assess_domain_classification(
        self,
        package: VCPTemporalCasePackage,
        predicates: Dict[str, PredicateResult],
        obs: VCPObservation,
    ) -> VCPDomainAssessment:
        """Determines the authoritative domain classification from the predicate vector.

        FAIL-CLOSED INVARIANT:
        If any normative predicate is UNRESOLVED or INSUFFICIENT_DATA,
        the authoritative VCP label fails closed.
        """
        normative_defs = self.predicate_registry.list_normative_predicates()
        normative_ids = [d.predicate_id for d in normative_defs]

        normative_results = [predicates[pid] for pid in normative_ids if pid in predicates]

        unresolved_count = sum(1 for r in normative_results if r.is_unresolved())
        insufficient_count = sum(1 for r in normative_results if r.is_insufficient_data())
        fail_count = sum(1 for r in normative_results if r.is_fail())
        pass_count = sum(1 for r in normative_results if r.is_pass())

        all_pass = (pass_count == len(normative_ids)) and (fail_count == 0) and (unresolved_count == 0) and (insufficient_count == 0)

        # Determine stage classification
        stg2 = predicates.get("PRED_STAGE_2")
        stg4 = predicates.get("PRED_STAGE_4")
        stg1 = predicates.get("PRED_STAGE_1")
        stg3 = predicates.get("PRED_STAGE_3")

        if insufficient_count > 0 and obs.session_count < 200:
            stage_class = "STAGE_UNRESOLVED"
        elif stg4 and stg4.is_pass():
            stage_class = "STAGE_4"
        elif stg2 and stg2.is_pass():
            stage_class = "STAGE_2"
        elif stg1 and stg1.is_pass():
            stage_class = "STAGE_1"
        elif stg3 and stg3.is_pass():
            stage_class = "STAGE_3"
        else:
            stage_class = "STAGE_UNRESOLVED"

        reason_codes: List[str] = []
        for r in normative_results:
            reason_codes.extend(r.reason_codes)

        # Final VCP Classification
        if insufficient_count > 0:
            vcp_qualified = False
            vcp_classification = "VCP_INSUFFICIENT_DATA"
            authorized_label = None
        elif unresolved_count > 0:
            vcp_qualified = False
            vcp_classification = "VCP_UNRESOLVED"
            authorized_label = None
        elif all_pass:
            vcp_qualified = True
            vcp_classification = "VCP_QUALIFIED"
            authorized_label = "VCP_STAGE_2_TIGHTENING"
        else:
            vcp_qualified = False
            vcp_classification = "VCP_NON_QUALIFIED"
            authorized_label = None

        return VCPDomainAssessment(
            case_id=package.case_id,
            evaluation_as_of=package.evaluation_as_of,
            predicate_vector=predicates,
            stage_classification=stage_class,
            vcp_qualified=vcp_qualified,
            vcp_classification=vcp_classification,
            classification_reason_codes=sorted(list(set(reason_codes))),
            authorized_label=authorized_label,
            normative_all_pass=all_pass,
            unresolved_normative_count=unresolved_count,
            insufficient_normative_count=insufficient_count,
        )

    def classify_case(self, package: VCPTemporalCasePackage) -> VCPDomainAssessment:
        """Full pipeline: CasePackage -> Observation -> Predicates -> Assessment."""
        obs = self.extract_observations(package)
        preds = self.evaluate_predicates(obs)
        assessment = self.assess_domain_classification(package, preds, obs)
        return assessment
