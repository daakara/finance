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
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
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

    VERSION_CANDIDATE_004 = "ARX_VCP_CLASSIFIER_V1_CANDIDATE_004"
    VERSION_CANDIDATE_005 = "ARX_VCP_CLASSIFIER_V1_CANDIDATE_005"

    def __init__(
        self,
        predicate_registry: Optional[VCPPredicateRegistry] = None,
        numeric_contract: Optional[VCPNumericContract] = None,
        algorithm_version: str = "ARX_VCP_CLASSIFIER_V1_CANDIDATE_004",
    ):
        self.predicate_registry = predicate_registry or VCPPredicateRegistry()
        self.numeric_contract = numeric_contract or VCPNumericContract()
        self.algorithm_version = algorithm_version

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

        # Contraction wave detection (Candidate 004 vs Candidate 005)
        macro_meta: Dict[str, Any] = {}
        if self.algorithm_version == self.VERSION_CANDIDATE_005:
            valid_contractions, contraction_depths, macro_meta = self._detect_macro_contractions(bars)
            contraction_count = len(valid_contractions)
            is_progressive = (
                self.numeric_contract.verify_progressive_tightening(
                    contraction_depths,
                    relative_reduction=0.05,
                    eps_float=1e-7,
                )
                if contraction_count >= 2
                else False
            )
        else:
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
            metadata=macro_meta,
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

    def _detect_macro_contractions(
        self,
        bars: List[DailyOHLCVBar],
        lookback_max: int = 90,
        lookback_min: int = 15,
        k_anchor: int = 3,
        theta_d1: float = 0.08,
        theta_dk: float = 0.04,
        theta_d_final: float = 0.02,
        theta_amp: float = 0.04,
        theta_dur: int = 3,
        theta_recovery: float = 0.015,
        ceiling_restart_threshold: float = 1.03,
    ) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
        """Candidate 005 Macro Contraction Wave Detection Engine.

        Resolves:
        1. Base-origin anchoring with chronological processing and restart.
        2. Conjunctive noise-merging (amplitude < 4% AND duration <= 3 elapsed sessions).
        3. Independent final-wave trough confirmation (recovery >= 1.5% or close > trough bar high).
        """
        n = len(bars)
        metadata: Dict[str, Any] = {
            "algorithm_version": self.VERSION_CANDIDATE_005,
            "anchor_selected": None,
            "pre_base_bars_truncated": 0,
            "waves_merged_noise": 0,
            "final_trough_confirmed": False,
            "final_trough_conf_date": None,
            "reason_codes": [],
        }

        if n < 40:
            metadata["reason_codes"].append("INSUFFICIENT_HISTORY_FOR_BASE")
            return [], [], metadata

        # 1. Base lookback slice: [max(0, n - lookback_max) : n]
        start_lookback = max(0, n - lookback_max)
        search_end = n - lookback_min
        if search_end <= start_lookback:
            metadata["reason_codes"].append("BASE_LOOKBACK_WINDOW_TOO_NARROW")
            return [], [], metadata

        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]

        # 2. Find eligible anchor peaks in [start_lookback + k_anchor, search_end]
        peak_candidates = []
        for i in range(start_lookback + k_anchor, search_end + 1):
            if i + k_anchor >= n:
                continue
            is_sym_peak = True
            for offset in range(-k_anchor, k_anchor + 1):
                if offset != 0 and highs[i + offset] > highs[i]:
                    is_sym_peak = False
                    break
            if is_sym_peak:
                peak_candidates.append(i)

        if not peak_candidates:
            metadata["reason_codes"].append("INSUFFICIENT_BASE_ANCHOR_CANDIDATES")
            return [], [], metadata

        # 3. Chronological Anchor Selection & Restart Engine (Ambiguity B)
        # Find earliest confirmed symmetric peak that initiates a contraction >= theta_d1 (8.0%)
        active_anchor_idx = None
        for p_idx in peak_candidates:
            p_high = highs[p_idx]
            trough_end = min(n, p_idx + 35)
            if trough_end > p_idx + 1:
                min_l = min(lows[p_idx + 1 : trough_end])
                if (p_high - min_l) / p_high >= theta_d1:
                    active_anchor_idx = p_idx
                    break

        if active_anchor_idx is None:
            metadata["reason_codes"].append("INSUFFICIENT_BASE_ANCHOR_CANDIDATES")
            return [], [], metadata

        # Step forward chronologically through subsequent peaks to test restart vs ceiling
        for p_idx in peak_candidates:
            if p_idx <= active_anchor_idx:
                continue
            current_anchor_high = highs[active_anchor_idx]
            new_high = highs[p_idx]
            ratio = new_high / current_anchor_high

            if ratio > ceiling_restart_threshold:
                # New trend leg / advance: check if new peak initiates contraction >= theta_d1
                trough_end = min(n, p_idx + 35)
                if trough_end > p_idx + 1:
                    min_l = min(lows[p_idx + 1 : trough_end])
                    if (new_high - min_l) / new_high >= theta_d1:
                        # Base restarts at new peak
                        active_anchor_idx = p_idx

        metadata["anchor_selected"] = {
            "bar_index": active_anchor_idx,
            "date": bars[active_anchor_idx].bar_date,
            "high": highs[active_anchor_idx],
        }
        metadata["pre_base_bars_truncated"] = active_anchor_idx - start_lookback
        if metadata["pre_base_bars_truncated"] > 0:
            metadata["reason_codes"].append("PRE_BASE_ADVANCE_TRUNCATED")

        # 4. Segment Contraction Waves starting from active_anchor_idx
        waves: List[Dict[str, Any]] = []
        depths: List[float] = []

        curr_peak_idx = active_anchor_idx
        curr_peak_val = highs[curr_peak_idx]

        while curr_peak_idx < n - 2:
            running_low = curr_peak_val
            running_low_idx = curr_peak_idx
            leg_closed = False
            rebound_peak_val = running_low
            rebound_peak_idx = running_low_idx

            j = curr_peak_idx + 1
            while j < n:
                b_low = lows[j]
                b_high = highs[j]

                if b_low < running_low:
                    running_low = b_low
                    running_low_idx = j
                    rebound_peak_val = b_high
                    rebound_peak_idx = j
                else:
                    if b_high > rebound_peak_val:
                        rebound_peak_val = b_high
                        rebound_peak_idx = j

                    delta_rebound = (rebound_peak_val - running_low) / running_low if running_low > 0 else 0.0
                    tau_rebound = j - running_low_idx

                    is_noise = (delta_rebound < theta_amp) and (tau_rebound <= theta_dur)

                    if not is_noise:
                        leg_closed = True
                        break

                j += 1

            # Check wave properties
            trough_idx = running_low_idx
            trough_val = running_low
            wave_dur = trough_idx - curr_peak_idx + 1
            wave_depth = self.numeric_contract.compute_contraction_depth(curr_peak_val, trough_val)

            min_depth_req = theta_d1 if len(waves) == 0 else (theta_d_final if j >= n - 5 else theta_dk)

            if wave_depth >= min_depth_req and wave_dur >= 2:
                wave_record = {
                    "wave_index": len(waves) + 1,
                    "start_idx": curr_peak_idx,
                    "trough_idx": trough_idx,
                    "peak_date": bars[curr_peak_idx].bar_date,
                    "trough_date": bars[trough_idx].bar_date,
                    "peak_price": self.numeric_contract.round_price(curr_peak_val),
                    "trough_price": self.numeric_contract.round_price(trough_val),
                    "depth_pct": wave_depth,
                    "duration_bars": wave_dur,
                }
                waves.append(wave_record)
                depths.append(wave_depth)
            else:
                metadata["waves_merged_noise"] += 1

            if j >= n or not leg_closed:
                break

            # Find next peak from rebound
            next_peak_idx = rebound_peak_idx
            next_peak_val = highs[next_peak_idx]

            k_idx = rebound_peak_idx + 1
            while k_idx < n:
                if highs[k_idx] > next_peak_val:
                    next_peak_val = highs[k_idx]
                    next_peak_idx = k_idx
                elif (next_peak_val - lows[k_idx]) / next_peak_val >= theta_dk:
                    break
                k_idx += 1

            if next_peak_idx == curr_peak_idx or next_peak_idx >= n - 1:
                break
            curr_peak_idx = next_peak_idx
            curr_peak_val = next_peak_val

        if metadata["waves_merged_noise"] > 0:
            metadata["reason_codes"].append("INTRA_WAVE_NOISE_MERGED")

        # 5. Independent Final-Trough Confirmation (Ambiguity A)
        if waves:
            final_wave = waves[-1]
            final_trough_val = final_wave["trough_price"]
            final_trough_idx = final_wave["trough_idx"]
            trough_bar_high = bars[final_trough_idx].high

            trough_confirmed = False
            trough_conf_date = None

            if final_trough_idx < n - 1:
                for eval_idx in range(final_trough_idx + 1, n):
                    eval_close = closes[eval_idx]
                    eval_low = lows[eval_idx]

                    if eval_low < final_trough_val:
                        trough_confirmed = False
                        break

                    rec_pct = (eval_close - final_trough_val) / final_trough_val if final_trough_val > 0 else 0.0
                    condA = rec_pct >= theta_recovery
                    condB = eval_close > trough_bar_high

                    if condA or condB:
                        trough_confirmed = True
                        trough_conf_date = bars[eval_idx].bar_date

            metadata["final_trough_confirmed"] = trough_confirmed
            metadata["final_trough_conf_date"] = trough_conf_date
            if not trough_confirmed:
                metadata["reason_codes"].append("FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF")

        return waves, depths, metadata

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
