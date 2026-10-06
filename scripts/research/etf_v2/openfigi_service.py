"""
scripts/research/etf_v2/openfigi_service.py

High-level orchestrator for OpenFIGI Symbology Corroboration.
Coordinates normalizer, client, classifier, and persistence repository.

Invariants Enforced:
- OFIGI-INV-001: Operational evidence only; zero canonical authority.
- OFIGI-INV-003: Unidirectional data flow from canonical input to operational store.
- OFIGI-INV-016: Canonical database, backup, and snapshot remain byte-identical.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from typing import List, Optional, Sequence, Tuple
import uuid

from .openfigi_classifier import OpenFIGIResponseClassifier
from .openfigi_client import MAX_MAPPING_JOBS_PER_REQUEST, OpenFIGIClient
from .openfigi_models import (
    AuthorizedCanonicalInputRecord,
    CorroborationExecutionSummary,
    CorroborationOutcomeClass,
    NormalizedCorroborationResult,
    OpenFIGIActiveMapping,
    OpenFIGIMappingJob,
    OpenFIGIObservation,
)
from .openfigi_config import validate_store_path_parity
from .openfigi_normalizer import OpenFIGINormalizer
from .openfigi_persistence import OpenFIGIPersistenceRepository

logger = logging.getLogger(__name__)


class OpenFIGICorroborationService:
    """
    Governed service orchestrator for OpenFIGI Symbology Corroboration.
    Accepts ONLY caller-authorized canonical cohort records.
    """

    def __init__(
        self,
        client: OpenFIGIClient,
        repository: OpenFIGIPersistenceRepository
    ):
        self.client = client
        self.repository = repository
        if hasattr(client, "rate_limiter") and hasattr(client.rate_limiter, "db_path"):
            validate_store_path_parity(client.rate_limiter.db_path, repository.db_path)

    def run_corroboration(
        self,
        authorized_cohort_records: Sequence[AuthorizedCanonicalInputRecord],
        execution_id: Optional[str] = None
    ) -> CorroborationExecutionSummary:
        """
        Executes bounded OpenFIGI corroboration on caller-authorized canonical records.
        OpenFIGI service does NOT self-select the canonical population.
        """
        exec_id = execution_id or str(uuid.uuid4())
        total_records = len(authorized_cohort_records)
        observations_to_persist: List[Tuple[OpenFIGIObservation, Optional[OpenFIGIActiveMapping]]] = []

        exact_count = 0
        ambiguous_count = 0
        no_match_count = 0
        failure_count = 0

        # Separate valid jobs from pre-request validation failures (e.g. OFIGI-INV-017)
        valid_inputs: List[AuthorizedCanonicalInputRecord] = []
        valid_jobs: List[OpenFIGIMappingJob] = []

        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for idx, rec in enumerate(authorized_cohort_records):
            is_valid, job, err_msg = OpenFIGINormalizer.validate_and_normalize(rec)
            if not is_valid:
                # Pre-request failure: create non-retryable INVALID_REQUEST observation locally
                failure_count += 1
                obs_id = str(uuid.uuid4())
                norm = NormalizedCorroborationResult(
                    canonical_internal_id=rec.canonical_internal_id,
                    isin=rec.isin,
                    outcome_class=CorroborationOutcomeClass.INVALID_REQUEST,
                    error_message=err_msg
                )
                raw_ev = {"error": err_msg, "validation_failure": True}
                obs = OpenFIGIObservation(
                    observation_id=obs_id,
                    execution_id=exec_id,
                    correlation_id=f"{exec_id}:{idx}",
                    canonical_internal_id=rec.canonical_internal_id,
                    isin=rec.isin,
                    idempotency_key=rec.compute_idempotency_key(),
                    source_population_version=rec.source_population_version,
                    source_snapshot_sha256=rec.source_snapshot_sha256,
                    request_position=idx,
                    request_filters={"mic_code": rec.mic_code, "exch_code": rec.exch_code},
                    attempt_count=0,
                    retry_count=0,
                    http_status=None,
                    outcome_class=norm.outcome_class.value,
                    normalized_result=norm.model_dump(),
                    provider_response_evidence=raw_ev,
                    provider_response_digest=hashlib.sha256(json.dumps(raw_ev).encode("utf-8")).hexdigest(),
                    observed_at=now_str,
                    created_at=now_str
                )
                observations_to_persist.append((obs, None))
            else:
                valid_inputs.append(rec)
                valid_jobs.append(job)

        # Batch valid mapping jobs into chunks of up to 100
        batch_size = MAX_MAPPING_JOBS_PER_REQUEST
        for chunk_start in range(0, len(valid_jobs), batch_size):
            chunk_inputs = valid_inputs[chunk_start : chunk_start + batch_size]
            chunk_jobs = valid_jobs[chunk_start : chunk_start + batch_size]

            status_code, response_envelopes, attempt_cnt, retry_cnt = self.client.post_mapping_jobs(chunk_jobs)

            classified_results = OpenFIGIResponseClassifier.process_batch_response(
                input_records=chunk_inputs,
                mapping_jobs=chunk_jobs,
                response_envelopes=response_envelopes,
                http_status=status_code,
                execution_id=exec_id,
                attempt_count=attempt_cnt,
                retry_count=retry_cnt
            )

            for obs, proj in classified_results:
                observations_to_persist.append((obs, proj))
                if obs.outcome_class == CorroborationOutcomeClass.EXACT_OPERATIONAL_CORROBORATION.value:
                    exact_count += 1
                elif obs.outcome_class == CorroborationOutcomeClass.AMBIGUOUS_OPERATIONAL_CORROBORATION.value:
                    ambiguous_count += 1
                elif obs.outcome_class == CorroborationOutcomeClass.NO_OPERATIONAL_CORROBORATION.value:
                    no_match_count += 1
                else:
                    failure_count += 1

        # Persist all observations and active mapping projections atomically
        if observations_to_persist:
            self.repository.persist_batch(observations_to_persist)

        return CorroborationExecutionSummary(
            execution_id=exec_id,
            total_input_records=total_records,
            total_observations_recorded=len(observations_to_persist),
            exact_matches=exact_count,
            ambiguous_matches=ambiguous_count,
            no_matches=no_match_count,
            failures=failure_count,
            live_mapping_requests_executed=0
        )
