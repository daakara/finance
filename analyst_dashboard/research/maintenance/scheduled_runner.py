"""
analyst_dashboard/research/maintenance/scheduled_runner.py

Scheduled Detection Runner for ARX Research-Maintenance Pilot.
Governed by ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.

Operational Flow:
1. Enforce pilot bounds (FIX only, CIK 0001035983).
2. Acquire process concurrency lock (.maintenance.lock).
3. Evaluate activation flag and pilot lifespan limits (dormant/expired).
4. Hydrate durable state with cryptographic anti-rollback check.
5. Poll primary-source SEC metadata within fair-access request budget.
6. Skip already processed accessions (idempotent deduplication).
7. If no new qualifying event: checkpoint last_checked_utc without repo mutation.
8. If newly available periodic filing:
   a. If supported period (Q3 2026): run reconciliation -> assessment -> draft packet.
   b. If unsupported period or amendment: generate bounded event review artifact without valuation.
9. Commit run checkpoint to durable manifest atomically.
10. Output execution telemetry and GitHub step summary.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from .contracts import (
    EventProcessingState,
    MaintenanceEvent,
    RunOutcome,
)
from .pilot_config import (
    TARGET_SYMBOL,
    TARGET_CIK,
    TARGET_COMPANY_NAME,
    ACCEPTED_BASELINE_ID,
    ACCEPTED_CUTOFF_DATE,
    RECORDED_KNOWN_ACCESSIONS,
    QUALIFYING_PERIODIC_FORMS,
    PERIODIC_AMENDMENT_FORMS,
    MAX_PILOT_RUNS,
    MAX_REQUESTS_PER_RUN,
    MAX_RUNTIME_SECONDS,
    MIN_REQUEST_INTERVAL_SECONDS,
    OPERATIONAL_STATE_DIR,
    OPERATIONAL_OUTPUT_DIR,
    OPERATIONAL_CACHE_DIR,
    OPERATIONAL_ARTIFACT_NAME,
    OPERATIONAL_BOOTSTRAP_ENV_VAR,
    REHEARSAL_STATE_DIR,
    REHEARSAL_OUTPUT_DIR,
    REHEARSAL_CACHE_DIR,
    REHEARSAL_ARTIFACT_NAME,
    validate_rehearsal_paths,
    evaluate_pilot_status,
    is_reconciliation_supported,
    RunBudgetTracker,
    BudgetExhaustedError,
)
from .durable_state import (
    DurableStateManager,
    ConcurrentRunError,
    StateIntegrityError,
    StatePersistenceError,
    StateRecoveryError,
)
from .event_poller import (
    SecEventPoller,
    format_sec_primary_url,
    format_sec_index_url,
)
from .fact_reconciler import FactReconciler
from .decision_assessor import DecisionAssessor
from .packet_generator import PacketGenerator
from .service import ResearchMaintenanceService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scheduled_runner")


class ScheduledRunner:
    """
    Executes a single scheduled detection run adhering to all pilot constraints.
    """

    def __init__(
        self,
        symbol: str = TARGET_SYMBOL,
        state_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        offline_mode: bool = False,
        force_run: bool = False,
        is_rehearsal: Optional[bool] = None,
        allow_bootstrap: bool = False,
        operational_bootstrap: bool = False,
    ):
        clean_sym = symbol.strip().upper()
        if clean_sym != TARGET_SYMBOL:
            raise ValueError(
                f"BOUNDARY_VIOLATION: Pilot is strictly limited to Comfort Systems USA, Inc. ({TARGET_SYMBOL}). "
                f"Requested symbol '{clean_sym}' is prohibited."
            )
        self.symbol = clean_sym
        self.cik = TARGET_CIK
        self.company_name = TARGET_COMPANY_NAME
        self.offline_mode = offline_mode
        self.force_run = force_run
        self.is_rehearsal = is_rehearsal if is_rehearsal is not None else force_run
        self.allow_bootstrap = allow_bootstrap
        self.operational_bootstrap = (
            operational_bootstrap
            or os.environ.get(OPERATIONAL_BOOTSTRAP_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "enabled")
        )

        # Reject contradictory settings: operational bootstrap cannot be combined with rehearsal mode
        if self.operational_bootstrap and (self.is_rehearsal or self.force_run):
            raise ValueError(
                "BOUNDARY_VIOLATION: Contradictory settings: operational bootstrap cannot be combined with rehearsal mode."
            )

        # Reject contradictory settings that allow force_run rehearsal execution to use the operational namespace
        if self.force_run and not self.is_rehearsal:
            raise ValueError(
                "BOUNDARY_VIOLATION: Contradictory settings: 'force_run=True' is reserved for rehearsals "
                "and cannot execute against the operational namespace (is_rehearsal=False)."
            )

        # Resolve paths according to operational vs rehearsal namespace
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        if self.is_rehearsal:
            default_state = repo_root / REHEARSAL_STATE_DIR
            default_output = repo_root / REHEARSAL_OUTPUT_DIR
            default_cache = repo_root / REHEARSAL_CACHE_DIR
            self.artifact_name = REHEARSAL_ARTIFACT_NAME
        else:
            default_state = repo_root / OPERATIONAL_STATE_DIR
            default_output = repo_root / OPERATIONAL_OUTPUT_DIR
            default_cache = repo_root / OPERATIONAL_CACHE_DIR
            self.artifact_name = OPERATIONAL_ARTIFACT_NAME

        self.state_dir = Path(state_dir) if state_dir else default_state
        self.output_dir = Path(output_dir) if output_dir else default_output
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache

        # Rehearsal isolation validation for explicit paths:
        if self.is_rehearsal:
            validate_rehearsal_paths(
                state_dir=self.state_dir,
                output_dir=self.output_dir,
                cache_dir=self.cache_dir,
                repo_root=repo_root,
            )

        self.state_mgr = DurableStateManager(
            state_dir=self.state_dir,
            artifact_name=self.artifact_name,
            is_rehearsal=self.is_rehearsal,
        )
        self.start_time = time.time()
        self.request_count = 0

    def run(self, target_accession: Optional[str] = None) -> Tuple[RunOutcome, Dict[str, Any]]:
        """
        Executes the guarded scheduled run under the concurrency lock,
        unless dormant or expired (exits immediately with zero filesystem, lock, or network mutation).
        """
        # Step 0: Initial safety guard - evaluate expiry and activation before acquiring locks,
        # mutating directories, hydrating state, or making network calls.
        pre_status = evaluate_pilot_status(
            current_run_count=0,
            force_run=self.force_run,
            operational_bootstrap=self.operational_bootstrap,
        )
        if pre_status.is_expired:
            logger.info(f"Pilot execution halted (expired): {pre_status.message}")
            summary = {
                "outcome": RunOutcome.PILOT_EXPIRED.value,
                "message": pre_status.message,
                "run_count": 0,
                "is_active": False,
                "is_expired": True,
            }
            self._write_step_summary(RunOutcome.PILOT_EXPIRED, summary)
            return RunOutcome.PILOT_EXPIRED, summary

        if not pre_status.is_active:
            logger.info(f"Pilot is dormant: {pre_status.message}")
            summary = {
                "outcome": RunOutcome.PILOT_DORMANT.value,
                "message": pre_status.message,
                "run_count": 0,
                "is_active": False,
                "instructions": "Set ENABLE_RESEARCH_MAINTENANCE_PILOT=true (or ARX_MAINTENANCE_PILOT_ACTIVE=1) to activate.",
            }
            self._write_step_summary(RunOutcome.PILOT_DORMANT, summary)
            return RunOutcome.PILOT_DORMANT, summary

        # Step 1: Concurrency lock
        try:
            with self.state_mgr.acquire_lock():
                return self._execute_guarded_run(target_accession=target_accession)
        except ConcurrentRunError as ce:
            logger.warning(str(ce))
            summary = {
                "outcome": RunOutcome.STATE_PERSISTENCE_FAILURE.value,
                "error": str(ce),
                "symbol": self.symbol,
                "run_time_sec": time.time() - self.start_time,
            }
            self._write_step_summary(RunOutcome.STATE_PERSISTENCE_FAILURE, summary)
            return RunOutcome.STATE_PERSISTENCE_FAILURE, summary

    def _execute_guarded_run(self, target_accession: Optional[str] = None) -> Tuple[RunOutcome, Dict[str, Any]]:
        # Initialize monotonic runtime and request budget tracker BEFORE active state recovery begins
        budget_tracker = RunBudgetTracker(
            max_requests=MAX_REQUESTS_PER_RUN,
            max_runtime_seconds=MAX_RUNTIME_SECONDS,
            min_request_interval_seconds=MIN_REQUEST_INTERVAL_SECONDS,
        )

        # Step 2: Hydrate durable state (multi-tier recovery: local -> backup -> GitHub artifact -> genesis)
        try:
            ok, hydrate_msg = self.state_mgr.hydrate_or_recover(
                allow_bootstrap=self.allow_bootstrap,
                budget_tracker=budget_tracker,
            )
            logger.info(f"Durable state hydration: {hydrate_msg}")
        except StateRecoveryError as sre:
            logger.error(f"State recovery failure: {sre}")
            summary = {
                "outcome": RunOutcome.STATE_PERSISTENCE_FAILURE.value,
                "error": str(sre),
                "recovery_requests": budget_tracker.recovery_request_count,
                "run_time_sec": budget_tracker.elapsed_time(),
            }
            self._write_step_summary(RunOutcome.STATE_PERSISTENCE_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.STATE_PERSISTENCE_FAILURE, summary
        except StateIntegrityError as se:
            logger.error(f"State integrity failure: {se}")
            summary = {
                "outcome": RunOutcome.STATE_PERSISTENCE_FAILURE.value,
                "error": str(se),
                "recovery_requests": budget_tracker.recovery_request_count,
                "run_time_sec": budget_tracker.elapsed_time(),
            }
            self._write_step_summary(RunOutcome.STATE_PERSISTENCE_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.STATE_PERSISTENCE_FAILURE, summary

        # Step 3: Evaluate pilot activation and lifespan limits with hydrated run count
        status = evaluate_pilot_status(
            current_run_count=self.state_mgr.run_count,
            force_run=self.force_run,
            operational_bootstrap=self.operational_bootstrap,
        )

        if status.is_expired:
            logger.info(f"Pilot execution halted: {status.message}")
            self.state_mgr.commit_run_checkpoint(RunOutcome.PILOT_EXPIRED)
            summary = {
                "outcome": RunOutcome.PILOT_EXPIRED.value,
                "message": status.message,
                "run_count": self.state_mgr.run_count,
                "max_runs": status.max_runs,
                "checkpoint_sequence": self.state_mgr.checkpoint_sequence,
            }
            self._write_step_summary(RunOutcome.PILOT_EXPIRED, summary, budget_tracker=budget_tracker)
            return RunOutcome.PILOT_EXPIRED, summary

        if not status.is_active:
            logger.info(f"Pilot is dormant: {status.message}")
            summary = {
                "outcome": RunOutcome.PILOT_DORMANT.value,
                "message": status.message,
                "run_count": self.state_mgr.run_count,
                "is_active": False,
                "instructions": "Set ENABLE_RESEARCH_MAINTENANCE_PILOT=true (or ARX_MAINTENANCE_PILOT_ACTIVE=1) to activate.",
            }
            self._write_step_summary(RunOutcome.PILOT_DORMANT, summary, budget_tracker=budget_tracker)
            return RunOutcome.PILOT_DORMANT, summary

        # Step 4: Check monotonic deadline at boundary before SEC polling
        try:
            budget_tracker.check_deadline("pre_sec_polling")
        except BudgetExhaustedError as be:
            logger.error(f"Budget exhausted before polling: {be}")
            summary = {
                "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                "error": f"BUDGET_EXHAUSTED: {be}",
                "symbol": self.symbol,
                "run_count": self.state_mgr.run_count,
            }
            self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

        # Step 5: Setup services with unified budget tracker and completed accessions
        service = ResearchMaintenanceService(
            packets_dir=self.output_dir,
            cache_dir=self.cache_dir,
            offline_mode=self.offline_mode,
            budget_tracker=budget_tracker,
            completed_accessions=self.state_mgr.completed_accessions,
        )

        # Step 6: Poll primary-source metadata within active request budget
        logger.info(f"Checking primary-source filings for {self.symbol} (CIK: {self.cik})...")
        try:
            event, warnings = service.poller.discover_filing_event(
                symbol=self.symbol,
                cik=self.cik,
                company_name=self.company_name,
                cutoff_date=ACCEPTED_CUTOFF_DATE,
                target_accession=target_accession,
                is_replay=False,
                force_refresh=False,
            )
            self.request_count = budget_tracker.sec_request_count
        except BudgetExhaustedError as be:
            logger.error(f"Budget exhausted during SEC polling: {be}")
            summary = {
                "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                "error": f"BUDGET_EXHAUSTED: {be}",
                "symbol": self.symbol,
                "run_count": self.state_mgr.run_count,
                "request_count": budget_tracker.sec_request_count,
                "elapsed_time": budget_tracker.elapsed_time(),
            }
            self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.SEC_RETRIEVAL_FAILURE, summary
        except Exception as e:
            logger.error(f"Primary source retrieval failed: {e}")
            summary = {
                "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                "error": f"SEC retrieval failure: {e}",
                "symbol": self.symbol,
                "run_count": self.state_mgr.run_count,
                "request_count": budget_tracker.sec_request_count,
                "elapsed_time": budget_tracker.elapsed_time(),
            }
            self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

        # Check for retrieval failure warning from poller
        if event is None and warnings and any("RETRIEVAL_FAILED" in w for w in warnings):
            logger.error(f"Poller returned retrieval failure: {warnings}")
            summary = {
                "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                "error": "; ".join(warnings),
                "symbol": self.symbol,
                "run_count": self.state_mgr.run_count,
                "request_count": budget_tracker.sec_request_count,
                "elapsed_time": budget_tracker.elapsed_time(),
            }
            self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

        # Step 7: Case - No Qualifying Event
        if event is None:
            logger.info(f"No new qualifying periodic filing found for {self.symbol}.")
            self.state_mgr.commit_run_checkpoint(RunOutcome.NO_QUALIFYING_EVENT)
            summary = {
                "outcome": RunOutcome.NO_QUALIFYING_EVENT.value,
                "symbol": self.symbol,
                "accepted_baseline_id": ACCEPTED_BASELINE_ID,
                "cutoff_date": ACCEPTED_CUTOFF_DATE,
                "run_count": self.state_mgr.run_count,
                "checkpoint_sequence": self.state_mgr.checkpoint_sequence,
                "last_checked_utc": self.state_mgr.last_checked_utc,
                "message": "Routine poll completed: No new periodic filings postdating cutoff. Checkpoint committed with 0 code or public mutations.",
            }
            self._write_step_summary(RunOutcome.NO_QUALIFYING_EVENT, summary, budget_tracker=budget_tracker)
            return RunOutcome.NO_QUALIFYING_EVENT, summary

        # Step 8: Case - Already Processed / Known Accession
        if self.state_mgr.is_accession_completed(event.accession_number):
            logger.info(f"Filing {event.accession_number} is already completed/known in durable state.")
            self.state_mgr.commit_run_checkpoint(RunOutcome.ALREADY_PROCESSED_EVENT)
            summary = {
                "outcome": RunOutcome.ALREADY_PROCESSED_EVENT.value,
                "symbol": self.symbol,
                "accession_number": event.accession_number,
                "run_count": self.state_mgr.run_count,
                "checkpoint_sequence": self.state_mgr.checkpoint_sequence,
                "message": f"Filing {event.accession_number} was previously completed. Deduplicated without re-generating review packets.",
            }
            self._write_step_summary(RunOutcome.ALREADY_PROCESSED_EVENT, summary, budget_tracker=budget_tracker)
            return RunOutcome.ALREADY_PROCESSED_EVENT, summary

        # Step 9: Case - Non-periodic disclosure misidentified (e.g. 8-K)
        if event.processing_state == EventProcessingState.REJECTED_MISIDENTIFIED:
            logger.warning(f"Filing {event.accession_number} is a non-periodic disclosure (Form {event.form}).")
            self.state_mgr.record_completed_event(event)
            self.state_mgr.commit_run_checkpoint(RunOutcome.SOURCE_VALIDATION_CONFLICT)
            summary = {
                "outcome": RunOutcome.SOURCE_VALIDATION_CONFLICT.value,
                "symbol": self.symbol,
                "accession_number": event.accession_number,
                "form": event.form,
                "reason": "Non-periodic disclosure (e.g. 8-K) rejected under fail-closed periodic qualification rule.",
            }
            self._write_step_summary(RunOutcome.SOURCE_VALIDATION_CONFLICT, summary, budget_tracker=budget_tracker)
            return RunOutcome.SOURCE_VALIDATION_CONFLICT, summary

        # Consequential processing boundary check before baseline load
        try:
            budget_tracker.check_deadline("pre_baseline_load")
        except BudgetExhaustedError as be:
            logger.error(f"Budget exhausted before baseline load: {be}")
            self.state_mgr.record_failed_event(event, stage="BUDGET_EXHAUSTED", error_msg=str(be))
            summary = {
                "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                "error": f"BUDGET_EXHAUSTED: {be}",
                "symbol": self.symbol,
                "run_count": self.state_mgr.run_count,
            }
            self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

        # Step 10: Load accepted baseline dossier
        try:
            baseline_dossier = service.load_accepted_baseline_dossier(self.symbol)
        except Exception as be:
            logger.error(f"Baseline load error: {be}")
            summary = {"outcome": RunOutcome.SOURCE_VALIDATION_CONFLICT.value, "error": str(be)}
            self._write_step_summary(RunOutcome.SOURCE_VALIDATION_CONFLICT, summary, budget_tracker=budget_tracker)
            return RunOutcome.SOURCE_VALIDATION_CONFLICT, summary

        # Consequential processing boundary check before pipeline execution
        try:
            budget_tracker.check_deadline("pre_reconciliation")
        except BudgetExhaustedError as be:
            logger.error(f"Budget exhausted before reconciliation: {be}")
            self.state_mgr.record_failed_event(event, stage="BUDGET_EXHAUSTED", error_msg=str(be))
            summary = {
                "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                "error": f"BUDGET_EXHAUSTED: {be}",
                "symbol": self.symbol,
                "run_count": self.state_mgr.run_count,
            }
            self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
            return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

        # Step 11: Check if period/form reconciliation is supported
        supported = is_reconciliation_supported(self.symbol, event.form, event.reporting_period) and not event.is_amendment

        if supported:
            # PATH A: Fully Supported Period (e.g. Q3 2026 Form 10-Q)
            logger.info(f"Filing {event.accession_number} is supported for automated reconciliation.")
            try:
                packet, processed_event, pipe_warnings = service.run_maintenance_pipeline(
                    symbol=self.symbol,
                    target_accession=event.accession_number,
                    cutoff_date=ACCEPTED_CUTOFF_DATE,
                    is_replay=False,
                    force_refresh=False,
                )
                self.request_count = budget_tracker.sec_request_count
            except BudgetExhaustedError as be:
                logger.error(f"Budget exhausted during pipeline reconciliation: {be}")
                self.state_mgr.record_failed_event(event, stage="BUDGET_EXHAUSTED", error_msg=str(be))
                summary = {
                    "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                    "error": f"BUDGET_EXHAUSTED: {be}",
                    "symbol": self.symbol,
                    "run_count": self.state_mgr.run_count,
                    "request_count": budget_tracker.sec_request_count,
                    "elapsed_time": budget_tracker.elapsed_time(),
                }
                self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
                return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

            if packet is None:
                # Failed during processing
                err = "; ".join(pipe_warnings) if pipe_warnings else "Pipeline execution failed"
                self.state_mgr.record_failed_event(event, stage="PIPELINE", error_msg=err)
                self.state_mgr.commit_run_checkpoint(RunOutcome.SOURCE_VALIDATION_CONFLICT)
                summary = {"outcome": RunOutcome.SOURCE_VALIDATION_CONFLICT.value, "error": err}
                self._write_step_summary(RunOutcome.SOURCE_VALIDATION_CONFLICT, summary, budget_tracker=budget_tracker)
                return RunOutcome.SOURCE_VALIDATION_CONFLICT, summary

            # Consequential processing boundary check before committing checkpoint
            try:
                budget_tracker.check_deadline("pre_checkpoint_commit")
            except BudgetExhaustedError as be:
                logger.error(f"Budget exhausted before checkpoint commit: {be}")
                self.state_mgr.record_failed_event(event, stage="BUDGET_EXHAUSTED", error_msg=str(be))
                summary = {
                    "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                    "error": f"BUDGET_EXHAUSTED: {be}",
                    "symbol": self.symbol,
                    "run_count": self.state_mgr.run_count,
                }
                self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
                return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

            # Success: Record completed event and commit checkpoint
            self.state_mgr.record_completed_event(processed_event, review_packet_id=packet.packet_id)
            self.state_mgr.commit_run_checkpoint(
                RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED,
                last_processed_accession=event.accession_number,
            )

            summary = {
                "outcome": RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED.value,
                "symbol": self.symbol,
                "accession_number": event.accession_number,
                "form": event.form,
                "reporting_period": event.reporting_period,
                "packet_id": packet.packet_id,
                "proposed_disposition": packet.proposed_disposition.value,
                "changes_count": len(packet.material_financial_changes),
                "checkpoint_sequence": self.state_mgr.checkpoint_sequence,
                "run_count": self.state_mgr.run_count,
                "packet_json_path": str(self.output_dir / f"{packet.packet_id}.json"),
                "packet_md_path": str(self.output_dir / f"{packet.packet_id}.md"),
            }
            self._write_step_summary(RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED, summary, budget_tracker=budget_tracker)
            return RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED, summary

        else:
            # PATH B: Unsupported Period or Amendment (e.g. Q4 2026 10-K, Q1 2027 10-Q, 10-Q/A)
            unsupported_reason = event.unsupported_reason or (
                f"Automated reconciliation is validated ONLY for Q3 2026 Form 10-Q (period 2026-09-30). "
                f"Filing period '{event.reporting_period}' and Form '{event.form}' require manual accounting reconciliation."
            )
            logger.warning(
                f"Filing {event.accession_number} period '{event.reporting_period}' is unsupported: {unsupported_reason}"
            )
            event.unsupported_reason = unsupported_reason

            artifact, json_path, md_path = service.packet_generator.generate_unsupported_period_artifact(
                event=event,
                baseline_dossier=baseline_dossier,
                unsupported_reason=unsupported_reason,
            )

            # Consequential processing boundary check before committing checkpoint
            try:
                budget_tracker.check_deadline("pre_checkpoint_commit")
            except BudgetExhaustedError as be:
                logger.error(f"Budget exhausted before checkpoint commit: {be}")
                self.state_mgr.record_failed_event(event, stage="BUDGET_EXHAUSTED", error_msg=str(be))
                summary = {
                    "outcome": RunOutcome.SEC_RETRIEVAL_FAILURE.value,
                    "error": f"BUDGET_EXHAUSTED: {be}",
                    "symbol": self.symbol,
                    "run_count": self.state_mgr.run_count,
                }
                self._write_step_summary(RunOutcome.SEC_RETRIEVAL_FAILURE, summary, budget_tracker=budget_tracker)
                return RunOutcome.SEC_RETRIEVAL_FAILURE, summary

            event.processing_state = EventProcessingState.DETECTED_UNSUPPORTED_PERIOD
            self.state_mgr.record_completed_event(event, review_packet_id=artifact["artifact_id"])
            self.state_mgr.commit_run_checkpoint(
                RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD,
                last_processed_accession=event.accession_number,
            )

            summary = {
                "outcome": RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD.value,
                "symbol": self.symbol,
                "accession_number": event.accession_number,
                "form": event.form,
                "reporting_period": event.reporting_period,
                "is_amendment": event.is_amendment,
                "artifact_id": artifact["artifact_id"],
                "status": artifact["status"],
                "unsupported_reason": unsupported_reason,
                "checkpoint_sequence": self.state_mgr.checkpoint_sequence,
                "run_count": self.state_mgr.run_count,
                "artifact_json_path": str(json_path),
                "artifact_md_path": str(md_path),
            }
            self._write_step_summary(RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD, summary, budget_tracker=budget_tracker)
            return RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD, summary

    def _write_step_summary(
        self,
        outcome: RunOutcome,
        summary: Dict[str, Any],
        budget_tracker: Optional[RunBudgetTracker] = None,
    ):
        """Writes execution telemetry to GITHUB_STEP_SUMMARY when available."""
        is_reh = getattr(self, "is_rehearsal", False)
        is_boot = getattr(self, "operational_bootstrap", False)

        if is_boot:
            prefix = "[OPERATIONAL BOOTSTRAP] "
            invocation_mode_str = "Manual Operational Bootstrap"
            summary["invocation_mode"] = "MANUAL_OPERATIONAL_BOOTSTRAP"
        elif is_reh:
            prefix = "[REHEARSAL] "
            invocation_mode_str = "Rehearsal"
            summary["invocation_mode"] = "REHEARSAL"
        else:
            prefix = ""
            invocation_mode_str = "Scheduled / Operational"
            summary["invocation_mode"] = "SCHEDULED_OR_OPERATIONAL"

        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return

        req_count = budget_tracker.sec_request_count if budget_tracker else summary.get("request_count", self.request_count)
        rec_count = budget_tracker.recovery_request_count if budget_tracker else summary.get("recovery_requests", 0)
        run_sec = budget_tracker.elapsed_time() if budget_tracker else summary.get("run_time_sec", time.time() - self.start_time)

        md = f"""## {prefix}ARX Scheduled Research Maintenance Run Summary

| Metric | Value |
| :--- | :--- |
| **Issuer** | {self.company_name} (`{self.symbol}`, CIK: `{self.cik}`) |
| **Run Outcome** | `{outcome.value}` |
| **Invocation Mode** | `{invocation_mode_str}` |
| **Run Count** | {summary.get('run_count', self.state_mgr.run_count)} / {MAX_PILOT_RUNS} |
| **Checkpoint Sequence** | {summary.get('checkpoint_sequence', self.state_mgr.checkpoint_sequence)} |
| **Execution Time** | {run_sec:.2f}s (Budget: {MAX_RUNTIME_SECONDS}s) |
| **SEC HTTP Requests** | {req_count} / {MAX_REQUESTS_PER_RUN} |
| **Recovery HTTP Requests** | {rec_count} |
| **Namespace Mode** | `{'REHEARSAL' if is_reh else 'OPERATIONAL'}` |

"""
        if is_boot:
            md += """> [!NOTE]
> **Authorized Operational Bootstrap**: One-time manual invocation executing against the operational namespace while recurring schedule remains dormant.

"""
        if outcome == RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED:
            md += f"""### Verified Supported Filing Detected
- **Accession**: `{summary.get('accession_number')}`
- **Form**: `{summary.get('form')}` | **Reporting Period**: `{summary.get('reporting_period')}`
- **Review Packet**: `{summary.get('packet_id')}`
- **Proposed Disposition**: `{summary.get('proposed_disposition')}`
- **Financial Changes Reconciled**: {summary.get('changes_count')}
- **Awaiting**: Human research analyst review.
"""
        elif outcome == RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD:
            md += f"""### Periodic Filing Detected (Unsupported Automation Period)
- **Status**: `{summary.get('status')}`
- **Accession**: `{summary.get('accession_number')}`
- **Form**: `{summary.get('form')}` {'(Amendment)' if summary.get('is_amendment') else ''} | **Reporting Period**: `{summary.get('reporting_period')}`
- **Reason**: {summary.get('unsupported_reason')}
- **Valuation**: **Suppressed** (zero speculative or stale figures).
- **Review Artifact**: `{summary.get('artifact_id')}`
- **Awaiting**: Manual human accounting reconciliation.
"""
        elif outcome == RunOutcome.NO_QUALIFYING_EVENT:
            md += f"""### Routine Poll - No New Periodic Filings
- **Cutoff Date**: `{summary.get('cutoff_date')}`
- **Baseline Dossier**: `{summary.get('accepted_baseline_id')}`
- **Status**: Checked SEC EDGAR primary source metadata. No newer 10-Q/10-K found. Zero public or code mutations.
"""
        elif outcome == RunOutcome.ALREADY_PROCESSED_EVENT:
            md += f"""### Deduplicated Event
- **Accession**: `{summary.get('accession_number')}`
- **Status**: Already completed in durable state ledger. Skipped without duplicate review packets.
"""
        elif outcome == RunOutcome.PILOT_DORMANT:
            md += f"""### Pilot Inactive (Dormant Guard)
- **Status**: recurring pilot execution is disabled by default.
- **Activation Action**: Set repository variable `ENABLE_RESEARCH_MAINTENANCE_PILOT=true` to enable.
"""
        elif outcome == RunOutcome.PILOT_EXPIRED:
            md += f"""### Pilot Expired (Lifecycle Guard)
- **Reason**: {summary.get('message')}
- **Status**: Recurring execution halted by code guard.
"""

        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(md + "\n")
        except Exception as e:
            logger.warning(f"Failed to write GITHUB_STEP_SUMMARY: {e}")


def main():
    parser = argparse.ArgumentParser(description="ARX Scheduled Research Maintenance Runner")
    parser.add_argument("--symbol", default=TARGET_SYMBOL, help="Target issuer symbol (default: FIX)")
    parser.add_argument("--state-dir", default=None, help="Directory for durable state")
    parser.add_argument("--output-dir", default=None, help="Directory for review packets/artifacts")
    parser.add_argument("--cache-dir", default=None, help="Directory for raw SEC cache")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode using local fixtures")
    parser.add_argument("--force-run", action="store_true", help="Bypass dormant activation check")
    parser.add_argument("--rehearsal", action="store_true", help="Execute in isolated rehearsal namespace")
    parser.add_argument("--bootstrap", action="store_true", help="Authorize first-run genesis initialization")
    parser.add_argument("--operational-bootstrap", action="store_true", help="Authorize manual operational execution while pilot is dormant")
    parser.add_argument("--target-accession", default=None, help="Target specific accession for rehearsal")

    args = parser.parse_args()

    # Reject contradictory settings: operational bootstrap cannot be combined with rehearsal mode
    if args.operational_bootstrap and (args.rehearsal or args.force_run):
        print(
            "[ARX BOUNDARY ERROR] Contradictory settings: operational bootstrap cannot be combined with rehearsal mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    is_rehearsal = args.rehearsal or args.force_run

    try:
        runner = ScheduledRunner(
            symbol=args.symbol,
            state_dir=args.state_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            offline_mode=args.offline,
            force_run=args.force_run,
            is_rehearsal=is_rehearsal,
            allow_bootstrap=args.bootstrap,
            operational_bootstrap=args.operational_bootstrap,
        )
    except ValueError as e:
        print(f"[ARX BOUNDARY ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    outcome, summary = runner.run(target_accession=args.target_accession)
    print(f"\n[ARX SCHEDULED RUNNER OUTCOME] => {outcome.value}")
    print(json.dumps(summary, indent=2))

    # Exit codes: 0 for all expected normal outcomes (including dormant, expired, no event, or verified event)
    # Non-zero for network or state persistence errors
    if outcome in (
        RunOutcome.NO_QUALIFYING_EVENT,
        RunOutcome.ALREADY_PROCESSED_EVENT,
        RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED,
        RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD,
        RunOutcome.PILOT_DORMANT,
        RunOutcome.PILOT_EXPIRED,
    ):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
