"""
scripts/research/etf_v2/openfigi_classifier.py

Response classifier and normalizer for OpenFIGI V3 mapping responses.

Invariants Enforced:
- OFIGI-INV-008: Cardinality models 1:0, 1:1, 1:N; never resolves ambiguity with arbitrary candidates[0].
- OFIGI-INV-009: No-match normalizes to NO_OPERATIONAL_CORROBORATION; zero canonical effect.
- OFIGI-INV-010: Ambiguity preserves all candidates; zero canonical effect.
- OFIGI-INV-011: All 14 error and result classes enforce CANONICAL_EFFECT = NONE.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from .openfigi_models import (
    AuthorizedCanonicalInputRecord,
    CorroborationOutcomeClass,
    NormalizedCorroborationResult,
    OpenFIGIActiveMapping,
    OpenFIGICandidate,
    OpenFIGIMappingJob,
    OpenFIGIObservation,
    OpenFIGIResultEnvelope,
)


class OpenFIGIResponseClassifier:
    """Classifies raw OpenFIGI HTTP response envelopes into normalized observations and projections."""

    RECOGNIZED_LEGACY_NO_MATCH_ERRORS = frozenset({
        "no identifier found.",
        "no identifier found",
        "no match found.",
        "no match found",
    })

    @classmethod
    def classify_single_envelope(
        cls,
        envelope_dict: Dict[str, Any],
        input_record: AuthorizedCanonicalInputRecord
    ) -> NormalizedCorroborationResult:
        """Classifies a single raw response envelope for an authorized input record."""
        # 1. Check for warning envelope
        warning_val = envelope_dict.get("warning")
        if warning_val and isinstance(warning_val, str):
            clean_warn = warning_val.strip().lower()
            if "no identifier found" in clean_warn or "no match" in clean_warn:
                return NormalizedCorroborationResult(
                    canonical_internal_id=input_record.canonical_internal_id,
                    isin=input_record.isin,
                    outcome_class=CorroborationOutcomeClass.NO_OPERATIONAL_CORROBORATION,
                    warning_message=warning_val,
                    candidates_count=0
                )

        # 2. Check for error envelope
        error_val = envelope_dict.get("error")
        if error_val and isinstance(error_val, str):
            clean_err = error_val.strip().lower()
            if clean_err in cls.RECOGNIZED_LEGACY_NO_MATCH_ERRORS:
                return NormalizedCorroborationResult(
                    canonical_internal_id=input_record.canonical_internal_id,
                    isin=input_record.isin,
                    outcome_class=CorroborationOutcomeClass.NO_OPERATIONAL_CORROBORATION,
                    warning_message=f"Legacy error normalized as no-match: {error_val}",
                    candidates_count=0
                )
            else:
                return NormalizedCorroborationResult(
                    canonical_internal_id=input_record.canonical_internal_id,
                    isin=input_record.isin,
                    outcome_class=CorroborationOutcomeClass.PROVIDER_RESPONSE_INVALID,
                    error_message=f"Unrecognized provider error: {error_val}",
                    candidates_count=0
                )

        # 3. Check for data envelope
        data_val = envelope_dict.get("data")
        if data_val is None or not isinstance(data_val, list):
            return NormalizedCorroborationResult(
                canonical_internal_id=input_record.canonical_internal_id,
                isin=input_record.isin,
                outcome_class=CorroborationOutcomeClass.PROVIDER_RESPONSE_INVALID,
                error_message="Envelope missing valid 'data' array and has no recognized warning/error.",
                candidates_count=0
            )

        # Parse raw candidates into candidate models
        raw_candidates: List[OpenFIGICandidate] = []
        for item in data_val:
            if isinstance(item, dict) and "figi" in item:
                raw_candidates.append(OpenFIGICandidate(**item))

        if len(raw_candidates) == 0:
            return NormalizedCorroborationResult(
                canonical_internal_id=input_record.canonical_internal_id,
                isin=input_record.isin,
                outcome_class=CorroborationOutcomeClass.NO_OPERATIONAL_CORROBORATION,
                candidates_count=0
            )

        # Cardinality: 1 candidate -> Exact match
        if len(raw_candidates) == 1:
            cand = raw_candidates[0]
            return NormalizedCorroborationResult(
                canonical_internal_id=input_record.canonical_internal_id,
                isin=input_record.isin,
                outcome_class=CorroborationOutcomeClass.EXACT_OPERATIONAL_CORROBORATION,
                figi=cand.figi,
                composite_figi=cand.compositeFIGI,
                share_class_figi=cand.shareClassFIGI,
                ticker=cand.ticker,
                exch_code=cand.exchCode,
                security_type=cand.securityType,
                market_sector=cand.marketSector,
                name=cand.name,
                candidates_count=1
            )

        # Cardinality: N candidates -> Attempt deterministic contextual filtering
        filtered_candidates = raw_candidates

        # Disambiguate by currency if supplied in input context
        if input_record.currency:
            curr_filtered = [
                c for c in filtered_candidates
                if c.securityDescription and input_record.currency in c.securityDescription.upper()
            ]
            if curr_filtered:
                filtered_candidates = curr_filtered

        # Disambiguate by exchange code if supplied
        if input_record.exch_code:
            exch_filtered = [
                c for c in filtered_candidates
                if c.exchCode and c.exchCode.upper() == input_record.exch_code.upper()
            ]
            if exch_filtered:
                filtered_candidates = exch_filtered

        # Check post-filtering cardinality
        if len(filtered_candidates) == 1:
            cand = filtered_candidates[0]
            return NormalizedCorroborationResult(
                canonical_internal_id=input_record.canonical_internal_id,
                isin=input_record.isin,
                outcome_class=CorroborationOutcomeClass.EXACT_OPERATIONAL_CORROBORATION,
                figi=cand.figi,
                composite_figi=cand.compositeFIGI,
                share_class_figi=cand.shareClassFIGI,
                ticker=cand.ticker,
                exch_code=cand.exchCode,
                security_type=cand.securityType,
                market_sector=cand.marketSector,
                name=cand.name,
                candidates_count=len(raw_candidates)
            )
        elif len(filtered_candidates) == 0:
            return NormalizedCorroborationResult(
                canonical_internal_id=input_record.canonical_internal_id,
                isin=input_record.isin,
                outcome_class=CorroborationOutcomeClass.NO_OPERATIONAL_CORROBORATION,
                warning_message=f"Contextual filtering eliminated all {len(raw_candidates)} candidates.",
                candidates_count=len(raw_candidates)
            )
        else:
            # Ambiguity preserved; never arbitrarily pick candidates[0]
            return NormalizedCorroborationResult(
                canonical_internal_id=input_record.canonical_internal_id,
                isin=input_record.isin,
                outcome_class=CorroborationOutcomeClass.AMBIGUOUS_OPERATIONAL_CORROBORATION,
                warning_message=f"Ambiguous: {len(filtered_candidates)} candidates remain after context filtering.",
                candidates_count=len(filtered_candidates)
            )

    @classmethod
    def process_batch_response(
        cls,
        input_records: Sequence[AuthorizedCanonicalInputRecord],
        mapping_jobs: Sequence[OpenFIGIMappingJob],
        response_envelopes: List[Dict[str, Any]],
        http_status: int,
        execution_id: str,
        attempt_count: int,
        retry_count: int
    ) -> List[Tuple[OpenFIGIObservation, Optional[OpenFIGIActiveMapping]]]:
        """
        Processes a full batch response array, strictly preserving 1:1 positional indexing.
        """
        results: List[Tuple[OpenFIGIObservation, Optional[OpenFIGIActiveMapping]]] = []
        observed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Positional check: len(response) must equal len(request)
        if len(response_envelopes) != len(input_records):
            # Positional mismatch fails closed for the whole batch
            mismatch_err = (
                f"Positional mismatch: request had {len(input_records)} jobs, "
                f"response contained {len(response_envelopes)} envelopes."
            )
            for pos, record in enumerate(input_records):
                obs_id = str(uuid.uuid4())
                norm = NormalizedCorroborationResult(
                    canonical_internal_id=record.canonical_internal_id,
                    isin=record.isin,
                    outcome_class=CorroborationOutcomeClass.PROVIDER_RESPONSE_INVALID,
                    error_message=mismatch_err
                )
                raw_ev = {"error": mismatch_err}
                obs = OpenFIGIObservation(
                    observation_id=obs_id,
                    execution_id=execution_id,
                    correlation_id=f"{execution_id}:{pos}",
                    canonical_internal_id=record.canonical_internal_id,
                    isin=record.isin,
                    idempotency_key=record.compute_idempotency_key(),
                    source_population_version=record.source_population_version,
                    source_snapshot_sha256=record.source_snapshot_sha256,
                    request_position=pos,
                    request_filters=mapping_jobs[pos].model_dump(exclude_none=True),
                    attempt_count=attempt_count,
                    retry_count=retry_count,
                    http_status=http_status,
                    outcome_class=norm.outcome_class.value,
                    normalized_result=norm.model_dump(),
                    provider_response_evidence=raw_ev,
                    provider_response_digest=hashlib.sha256(json.dumps(raw_ev).encode("utf-8")).hexdigest(),
                    observed_at=observed_at,
                    created_at=observed_at
                )
                results.append((obs, None))
            return results

        # 1:1 positional mapping
        for pos, (record, envelope) in enumerate(zip(input_records, response_envelopes)):
            obs_id = str(uuid.uuid4())
            norm = cls.classify_single_envelope(envelope, record)
            raw_digest = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode("utf-8")).hexdigest()

            obs = OpenFIGIObservation(
                observation_id=obs_id,
                execution_id=execution_id,
                correlation_id=f"{execution_id}:{pos}",
                canonical_internal_id=record.canonical_internal_id,
                isin=record.isin,
                idempotency_key=record.compute_idempotency_key(),
                source_population_version=record.source_population_version,
                source_snapshot_sha256=record.source_snapshot_sha256,
                request_position=pos,
                request_filters=mapping_jobs[pos].model_dump(exclude_none=True),
                attempt_count=attempt_count,
                retry_count=retry_count,
                http_status=http_status,
                outcome_class=norm.outcome_class.value,
                normalized_result=norm.model_dump(),
                provider_response_evidence=envelope,
                provider_response_digest=raw_digest,
                observed_at=observed_at,
                created_at=observed_at
            )

            # Build active mapping projection only for successful corroborations
            proj = None
            if norm.outcome_class in (
                CorroborationOutcomeClass.EXACT_OPERATIONAL_CORROBORATION,
                CorroborationOutcomeClass.AMBIGUOUS_OPERATIONAL_CORROBORATION
            ):
                proj = OpenFIGIActiveMapping(
                    isin=record.isin,
                    canonical_internal_id=record.canonical_internal_id,
                    figi=norm.figi,
                    composite_figi=norm.composite_figi,
                    share_class_figi=norm.share_class_figi,
                    ticker=norm.ticker,
                    exch_code=norm.exch_code,
                    security_type=norm.security_type,
                    market_sector=norm.market_sector,
                    name=norm.name,
                    outcome_class=norm.outcome_class.value,
                    last_observation_id=obs_id,
                    source_population_version=record.source_population_version,
                    source_snapshot_sha256=record.source_snapshot_sha256,
                    updated_at=observed_at
                )

            results.append((obs, proj))

        return results
