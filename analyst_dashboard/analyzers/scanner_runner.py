"""
Market-Wide Scanner Runners for Radar Scanners.

Implements deterministic market-wide universe execution for:
- MINERVINI_VCP
- SMART_MONEY
"""

import os
import json
import hashlib
import time
import uuid
import logging
import threading
import socket
from typing import Dict, Any, List, Optional
import pandas as pd

from analyst_dashboard.data.market_evidence import AdjustmentState

from analyst_dashboard.vcp.sprint_3_shadow_governance import (
    Sprint3ShadowGovernanceSuite,
    get_default_shadow_suite,
)

from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    FreshnessStatus,
    PublicationDecision,
    ScannerVersionTuple,
    ScannerCandidateResult,
    ImmutableScannerSnapshot,
    resolve_coverage_envelope,
    RADAR_SCOPE_LABEL,
    VCP_API_CONTRACT_VERSION,
    VCP_RULESET_VERSION,
    VCP_EVIDENCE_SCHEMA_VERSION,
    VCP_SCORE_MODEL_VERSION,
    VCP_DATA_PROVENANCE_VERSION,
    VCP_UNIVERSE_VERSION,
    VCP_FRESHNESS_POLICY_VERSION,
    SMART_MONEY_API_CONTRACT_VERSION,
    SMART_MONEY_RULESET_VERSION,
    SMART_MONEY_EVIDENCE_SCHEMA_VERSION,
    SMART_MONEY_SCORE_MODEL_VERSION,
    SMART_MONEY_DATA_PROVENANCE_VERSION,
    SMART_MONEY_UNIVERSE_VERSION,
    SMART_MONEY_FRESHNESS_POLICY_VERSION,
    compute_hash,
)
from analyst_dashboard.analyzers.scanner_publication_integrity import (
    ScannerPublicationIntegrityEngine,
    CANONICAL_VCP_RULESET_HASH,
    CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
    CANONICAL_VCP_SCORE_MODEL_HASH,
    CANONICAL_VCP_DATA_PROVENANCE_HASH,
    CANONICAL_VCP_UNIVERSE_HASH,
    CANONICAL_VCP_FRESHNESS_HASH,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.universe.store import UniverseStore
from analyst_dashboard.universe.contracts import UNIVERSE_ID, UNIVERSE_VERSION

from analyst_dashboard.coordination import (
    DurableRunCoordinator,
    FencedPublisher,
    CoordinationStore,
    TriggerType,
    AcquisitionStatus,
    CurrentLease,
    LeasePolicy,
    PRODUCTION_LEASE_POLICY,
    RESOURCE_KEY_VCP_PIPELINE,
    StaleLeasePublicationError,
)

logger = logging.getLogger(__name__)

# Current implementation release identity
CURRENT_IMPLEMENTATION_RELEASE_SHA = "ed04de51160e0f2c3e82bd43d989897a4fd77f89"

# Canonical Universe Definition for VCP Market-Wide Scanning
from analyst_dashboard.analyzers.scanner_contract import CANONICAL_VCP_UNIVERSE


# Supported authoritative classification authorities for VCP
SUPPORTED_VCP_CLASSIFICATION_AUTHORITIES = {
    "CANONICAL_VCP_CLASSIFIER",
    "VCP_CLASSIFIER_V1",
}


class VCPScannerRunner:
    """Market-wide execution runner for Minervini Volatility Contraction Pattern."""

    def __init__(
        self,
        market_db: Optional[MarketDatabaseEngine] = None,
        snapshot_store: Optional[ScannerSnapshotStore] = None,
        confluence_engine: Optional[ConfluenceEngine] = None,
        universe_store: Optional[UniverseStore] = None,
        coordinator: Optional[DurableRunCoordinator] = None,
        fenced_publisher: Optional[FencedPublisher] = None,
        coordination_store: Optional[CoordinationStore] = None,
        lease_policy: Optional[LeasePolicy] = None,
        shadow_suite: Optional[Sprint3ShadowGovernanceSuite] = None,
    ):
        self.market_db = market_db or MarketDatabaseEngine()
        self.snapshot_store = snapshot_store or ScannerSnapshotStore()
        self.confluence_engine = confluence_engine or ConfluenceEngine()
        self.universe_store = universe_store or UniverseStore()
        self.lease_policy = lease_policy or PRODUCTION_LEASE_POLICY
        self.shadow_suite = shadow_suite or get_default_shadow_suite()

        coord_db_path = getattr(self.snapshot_store, "db_path", None)
        if coordination_store:
            self.coordination_store = coordination_store
        elif coord_db_path:
            self.coordination_store = CoordinationStore(db_path=coord_db_path)
        else:
            self.coordination_store = CoordinationStore()

        self.coordinator = coordinator or DurableRunCoordinator(
            resource_key=RESOURCE_KEY_VCP_PIPELINE,
            db_path=self.coordination_store.db_path,
            policy=self.lease_policy,
        )
        self.fenced_publisher = fenced_publisher or FencedPublisher(self.coordination_store)
        self._cached_active_snapshot: Optional[Dict[str, Any]] = None
        self._scan_lock = threading.Lock()

    @staticmethod
    def _classify_fetch_error(exc: Exception) -> str:
        """Classify per-symbol fetch and evaluation exceptions into authoritative reason categories."""
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__.lower()
        if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in exc_str or "timed out" in exc_str:
            return "TIMEOUT"
        if "lock" in exc_str or "operationalerror" in exc_type:
            return "DATABASE_LOCK"
        if "rate limit" in exc_str or "429" in exc_str:
            return "RATE_LIMIT"
        return "DATA_SOURCE_FETCH_ERROR"

    def get_version_tuple(self) -> ScannerVersionTuple:
        return ScannerVersionTuple(
            scanner_id="MINERVINI_VCP",
            api_contract_version=VCP_API_CONTRACT_VERSION,
            ruleset_version=VCP_RULESET_VERSION,
            evidence_schema_version=VCP_EVIDENCE_SCHEMA_VERSION,
            score_model_version=VCP_SCORE_MODEL_VERSION,
            data_provenance_version=VCP_DATA_PROVENANCE_VERSION,
            universe_version=VCP_UNIVERSE_VERSION,
            freshness_policy_version=VCP_FRESHNESS_POLICY_VERSION,
            implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        )

    def execute_market_wide_scan(
        self,
        universe_override: Optional[List[str]] = None,
        universe_build_id: Optional[str] = None,
        logical_job_key: Optional[str] = None,
        trigger_type: TriggerType = TriggerType.OPERATOR,
        scheduled_for: Optional[str] = None,
        operator_request_id: Optional[str] = None,
        owner_instance_id: Optional[str] = None,
        bypass_thread_lock: bool = False,
        shadow_trigger_override: Optional[str] = None,
        logical_scan_run_id: Optional[str] = None,
        logical_trigger_id: Optional[str] = None,
        invocation_class: Optional[str] = None,
        originating_principal_type: Optional[str] = None,
        originating_principal_id: Optional[str] = None,
        scheduler_job_id: Optional[str] = None,
        scheduler_event_id: Optional[str] = None,
        startup_context: bool = False,
        replay_of_logical_scan_run_id: Optional[str] = None,
        delivery_attempt_id: Optional[str] = None,
        execution_attempt_id: Optional[str] = None,
        candidate_generation_id: str = "CANDIDATE_GENERATION_004",
        min_coverage_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute deterministic market-wide scan across eligible universe:
        1. Resolve versioned universe (from universe_build_id or universe_store)
        2. Prevent overlapping executions via non-blocking lock & durable distributed lease
        3. Load canonical input data and compute coverage statistics
        4. Execute OptimalExecutionEngine
        5. Collect and score candidates
        6. Verify publication integrity
        7. Persist immutable snapshot via atomic fenced publication
        """
        # 1. Local Thread Lock (Non-authoritative fast optimization)
        acquired_thread_lock = False
        if not bypass_thread_lock:
            acquired_thread_lock = self._scan_lock.acquire(blocking=False)
            if not acquired_thread_lock:
                raise RuntimeError("OVERLAPPING_VCP_SCANS_PROHIBITED: A market-wide VCP scan is already in progress.")

        # 2. Durable Distributed Coordination (Authoritative mutual exclusion & idempotency)
        effective_instance_id = owner_instance_id or f"inst-{uuid.uuid4().hex[:8]}"
        effective_job_key = logical_job_key or (
            f"vcp:operator:{operator_request_id}" if operator_request_id else
            (f"vcp:scheduled:{scheduled_for}" if scheduled_for else f"vcp:run:{int(time.time())}:{uuid.uuid4().hex[:8]}")
        )

        acq_result = self.coordinator.acquire(
            logical_job_key=effective_job_key,
            trigger_type=trigger_type,
            owner_instance_id=effective_instance_id,
            implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
            scheduled_for=scheduled_for,
            operator_request_id=operator_request_id,
        )

        if acq_result.status == AcquisitionStatus.JOB_ALREADY_SUCCEEDED:
            if acquired_thread_lock:
                self._scan_lock.release()
            pub_id = acq_result.existing_job.successful_publication_id if acq_result.existing_job else None
            logger.info(f"Logical job {effective_job_key} already succeeded with publication {pub_id}. Returning active snapshot.")
            cached = self.get_active_or_latest_snapshot()
            if isinstance(cached, dict):
                cached = dict(cached)
                cached["deduplicated"] = True
                cached["acquisition_status"] = "JOB_ALREADY_SUCCEEDED"
                cached["successful_publication_id"] = pub_id
            return cached

        if acq_result.status == AcquisitionStatus.ALREADY_RUNNING:
            if acquired_thread_lock:
                self._scan_lock.release()
            raise RuntimeError(f"OVERLAPPING_VCP_SCANS_PROHIBITED: {acq_result.message}")

        lease = acq_result.lease
        lease_released_by_publisher = False

        try:
            run_id = lease.run_id if lease else f"vcp-run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
            generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Resolve scannable membership and universe metadata
            target_build_id = universe_build_id
            build_attestation = None
            if target_build_id:
                if hasattr(self.universe_store, "get_universe_build"):
                    build_attestation = self.universe_store.get_universe_build(target_build_id)
                scannable_symbols = self.universe_store.get_scannable_membership(target_build_id)
            else:
                latest_build = self.universe_store.get_latest_published_universe(UNIVERSE_ID)
                if latest_build:
                    target_build_id = latest_build["universe_build_id"]
                    scannable_symbols = self.universe_store.get_scannable_membership(target_build_id)
                    build_attestation = latest_build
                else:
                    scannable_symbols = []

            is_natural_scheduled = (
                invocation_class == "SCHEDULED_PRODUCTION"
                or trigger_type == TriggerType.SCHEDULED
            )

            if universe_override is not None:
                universe = universe_override
                eligible_count = len(universe)
                scannable_count = len(universe)
                unavailable_count = 0
            elif scannable_symbols:
                # For natural scheduled scan, enforce published build validity (INV-U02, INV-U08, INV-U09)
                if is_natural_scheduled and build_attestation:
                    if build_attestation.get("publication_decision") != "PUBLISH" or build_attestation.get("construction_status") != "COMPLETE":
                        raise RuntimeError(
                            f"UNPUBLISHED_UNIVERSE_BUILD: Build '{target_build_id}' has publication_decision='{build_attestation.get('publication_decision')}', construction_status='{build_attestation.get('construction_status')}'. Natural execution prohibited."
                        )
                universe = scannable_symbols
                eligible_count = build_attestation.get("eligible_count", len(universe)) if build_attestation else len(universe)
                scannable_count = len(scannable_symbols)
                unavailable_count = build_attestation.get("data_unavailable_count", 0) if build_attestation else 0
            else:
                # Historical regression fixture fallback ONLY for non-natural / bootstrap / test invocations (INV-U04)
                if is_natural_scheduled:
                    raise RuntimeError(
                        f"NO_PUBLISHED_EXECUTABLE_UNIVERSE: Natural scheduled scan requires an authorized published universe build. Historical fixture fallback is prohibited for invocation_class='{invocation_class}'."
                    )
                universe = list(dict.fromkeys(CANONICAL_VCP_UNIVERSE)) if target_build_id is None else []
                eligible_count = len(universe)
                scannable_count = len(universe)
                unavailable_count = 0

            universe_size = len(universe)
            scanned_successfully_count = 0
            computed_evaluations_count = 0
            uncomputed_evaluations_count = 0
            unavailable_reasons: Dict[str, int] = {}
            adjustment_state_counts: Dict[str, int] = {}
            qualified_candidates: List[Dict[str, Any]] = []

            for sym in universe:
                clean_sym = sym.strip().upper()
                try:
                    latest = self.market_db.get_latest_price(clean_sym)
                except Exception as exc:
                    err_reason = self._classify_fetch_error(exc)
                    unavailable_reasons[err_reason] = unavailable_reasons.get(err_reason, 0) + 1
                    continue

                if not latest or not isinstance(latest, dict) or not isinstance(latest.get("currentPrice"), (int, float)) or latest["currentPrice"] <= 0:
                    unavailable_reasons["MISSING_PRICE_DATA"] = unavailable_reasons.get("MISSING_PRICE_DATA", 0) + 1
                    continue

                current_price = latest["currentPrice"]

                # R3 Provenance Integration: Retrieve stored daily candles along with factual evidence provenance
                try:
                    db_candles = None
                    evidence = None
                    if hasattr(self.market_db, "get_daily_candles_with_evidence"):
                        ret = self.market_db.get_daily_candles_with_evidence(clean_sym, limit=60)
                        if isinstance(ret, tuple) and len(ret) == 2:
                            cand_candidate, evid_candidate = ret
                            if isinstance(cand_candidate, list):
                                db_candles = cand_candidate
                                evidence = evid_candidate

                    if db_candles is None and hasattr(self.market_db, "get_daily_candles"):
                        db_candles = self.market_db.get_daily_candles(clean_sym, limit=60)
                        adj_state = AdjustmentState.UNKNOWN
                        adj_prov = {
                            "adjustment_state": adj_state,
                            "provider": "UNKNOWN",
                            "structural_quality": "UNKNOWN",
                            "splits_locally_applied": False,
                            "continuous_series_attested": False,
                        }
                    elif db_candles is not None:
                        adj_state = (
                            getattr(evidence.provenance, "adjustment_state", AdjustmentState.UNKNOWN)
                            if (evidence and hasattr(evidence, "provenance"))
                            else AdjustmentState.UNKNOWN
                        )
                        adj_prov = {
                            "adjustment_state": adj_state,
                            "provider": getattr(evidence.provenance, "provider", "UNKNOWN") if (evidence and hasattr(evidence, "provenance")) else "UNKNOWN",
                            "structural_quality": getattr(evidence.provenance, "structural_quality", "UNKNOWN") if (evidence and hasattr(evidence, "provenance")) else "UNKNOWN",
                            "splits_locally_applied": False,
                            "continuous_series_attested": adj_state in (
                                AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,
                                AdjustmentState.SPLIT_ADJUSTED_ONLY,
                            ),
                        }
                    else:
                        raise ValueError(f"Unable to retrieve daily candles for {clean_sym}")
                except Exception as exc:
                    err_reason = self._classify_fetch_error(exc)
                    unavailable_reasons[err_reason] = unavailable_reasons.get(err_reason, 0) + 1
                    continue

                adjustment_state_counts[adj_state] = adjustment_state_counts.get(adj_state, 0) + 1

                # NEG-R3-03: Raw unadjusted price series guard against false contraction qualification
                if adj_state == AdjustmentState.UNADJUSTED:
                    unavailable_reasons["UNADJUSTED_PRICE_FEED_RISK"] = (
                        unavailable_reasons.get("UNADJUSTED_PRICE_FEED_RISK", 0) + 1
                    )
                    continue

                if not db_candles or len(db_candles) < 50:
                    unavailable_reasons["INSUFFICIENT_HISTORY"] = unavailable_reasons.get("INSUFFICIENT_HISTORY", 0) + 1
                    continue

                try:
                    scanned_successfully_count += 1
                    df = pd.DataFrame([{
                        "Open": c["open"], "High": c["high"], "Low": c["low"], "Close": c["close"], "Volume": c["volume"]
                    } for c in db_candles], index=pd.to_datetime([c["time"] for c in db_candles]))

                    exec_res = OptimalExecutionEngine.calculate_trade_levels(df, current_price, user_role="LONG_TERM")

                    # Compute actual moving averages directly from daily candles
                    close_s = df["Close"]
                    actual_sma_50 = round(float(close_s.rolling(50).mean().iloc[-1]), 4) if len(close_s) >= 50 else None
                    actual_ema_20 = round(float(close_s.ewm(span=20, adjust=False).mean().iloc[-1]), 4) if len(close_s) >= 20 else None

                    # Truthfulness Invariant: OptimalExecutionEngine evaluates tactical execution corridors (entry/stop/target/ATR).
                    # Its legacy strategy 2.0.0 heuristic strings ("Minervini VCP", "Stage 2") on 60-bar windows
                    # do not constitute authoritative VCP domain classification or 200-session trend template qualification.
                    # A 60-bar query window is epistemically incapable of evaluating a 200-session trend template.
                    # Operational runner rejects the legacy false-certification fallback and enforces uncomputed status:
                    vcp_authoritative_computed = bool(exec_res.get("vcp_authoritative_evaluated", False))

                    if vcp_authoritative_computed:
                        vcp_status = exec_res.get("vcp_contraction_status")
                        stage_status = exec_res.get("stage_phase")
                        setup_pattern = exec_res.get("setup_pattern")
                        computed_evaluations_count += 1
                    else:
                        vcp_status = "Consolidation Unverified"
                        stage_status = "Stage Unverified (Trend Template Uncomputed)"
                        setup_pattern = "Consolidation Setup (Pattern Unverified)"
                        uncomputed_evaluations_count += 1

                    # Qualification ruleset: Must be authoritatively confirmed VCP in Stage 2 Advancing Growth Phase
                    is_vcp = vcp_authoritative_computed and (vcp_status == "VCP 3-Stage Compression Confirmed")
                    is_stage_2 = vcp_authoritative_computed and (stage_status == "Stage 2 Advancing Growth Phase")

                    if is_vcp and is_stage_2:
                        # Calculate multi-factor confluence conviction score
                        conf_res = self.confluence_engine.calculate_confluence(
                            symbol=clean_sym,
                            technical_data={
                                "executionStatus": exec_res["execution_status"],
                                "riskRewardRatio": exec_res["risk_reward_ratio"],
                                "setup_pattern": setup_pattern,
                                "stage_phase": stage_status,
                                "rsi_14": exec_res.get("rsi_14"),
                                "stop_loss": exec_res["stop_loss"],
                                "current_price": current_price,
                            },
                        )
                        score = round(float(conf_res.get("confluenceScore", 75.0)), 1)

                        scanner_evidence = {
                            "vcp_stage": vcp_status,
                            "setup_pattern": setup_pattern,
                            "stage_phase": stage_status,
                            "sma_50": actual_sma_50,
                            "ema_20": actual_ema_20,
                            "atr_14": exec_res.get("atr_14"),
                            "breakout_pivot": exec_res.get("breakout_pivot"),
                            "optimal_entry_min": exec_res.get("optimal_entry_min"),
                            "optimal_entry_max": exec_res.get("optimal_entry_max"),
                            "stop_loss": exec_res.get("stop_loss"),
                            "take_profit_1": exec_res.get("take_profit_1"),
                            "take_profit_2": exec_res.get("take_profit_2"),
                            "risk_reward_ratio": exec_res.get("risk_reward_ratio"),
                            "execution_status": exec_res.get("execution_status"),
                            "confluence_score": score,
                            "adjustment_provenance": adj_prov,
                        }

                        qualified_candidates.append({
                            "symbol": clean_sym,
                            "score": score,
                            "current_price": current_price,
                            "scanner_evidence": scanner_evidence,
                        })
                except Exception as eval_exc:
                    err_reason = self._classify_fetch_error(eval_exc)
                    unavailable_reasons[err_reason] = unavailable_reasons.get(err_reason, 0) + 1
                    uncomputed_evaluations_count += 1
                    continue

            # Deterministic ranking: score descending, symbol ascending
            qualified_candidates.sort(key=lambda c: (-c["score"], c["symbol"]))
            for rank_idx, cand in enumerate(qualified_candidates, start=1):
                cand["rank"] = rank_idx

            matched_count = len(qualified_candidates)
            data_as_of = time.strftime("%Y-%m-%d", time.gmtime())

            version_tuple = self.get_version_tuple()
            semantic_fingerprint = ScannerPublicationIntegrityEngine.get_canonical_vcp_fingerprint()

            # Record shadow observations under Sprint 3 contamination-controlled governance (Candidate 003)
            if invocation_class is not None:
                eff_invocation_class = invocation_class
            elif shadow_trigger_override in ("TEST", "REPLAY", "SYNTHETIC"):
                eff_invocation_class = shadow_trigger_override
            elif startup_context or str(trigger_type).upper() in ("MAINTENANCE", "BOOT_WARMUP", "WARMUP"):
                eff_invocation_class = "BOOT_WARMUP"
            elif trigger_type == TriggerType.SCHEDULED and not operator_request_id:
                eff_invocation_class = "SCHEDULED_PRODUCTION"
            else:
                eff_invocation_class = "MANUAL_OPERATOR"

            eff_logical_run_id = logical_scan_run_id or run_id
            for cand in qualified_candidates:
                evidence_dict = cand.get("scanner_evidence", {})
                self.shadow_suite.record_shadow_observation(
                    security_id=cand["symbol"],
                    evaluation_as_of=data_as_of,
                    universe_build_id=target_build_id or "ARX_CANONICAL_UNIVERSE_BUILD",
                    snapshot_run_id=run_id,
                    logical_scan_run_id=eff_logical_run_id,
                    logical_trigger_id=logical_trigger_id,
                    candidate_generation_id=candidate_generation_id,
                    ruleset_id="MINERVINI_VCP",
                    ruleset_version=version_tuple.ruleset_version,
                    predicate_vector_hash=hashlib.sha256(json.dumps(evidence_dict, sort_keys=True).encode("utf-8")).hexdigest(),
                    classification="CONFIRMED_VCP_STAGE_2",
                    decision_posture="QUALIFIED_WATCHLIST",
                    input_fingerprint=hashlib.sha256(f"{cand['symbol']}:{cand['current_price']}".encode("utf-8")).hexdigest(),
                    group_or_episode_id=f"EPISODE:{cand['symbol']}:{data_as_of}",
                    invocation_class=eff_invocation_class,
                    originating_principal_type=originating_principal_type,
                    originating_principal_id=originating_principal_id,
                    scheduler_job_id=scheduler_job_id,
                    scheduler_event_id=scheduler_event_id,
                    startup_context=startup_context,
                    replay_of_logical_scan_run_id=replay_of_logical_scan_run_id,
                    delivery_attempt_id=delivery_attempt_id,
                    execution_attempt_id=execution_attempt_id,
                )

            # Coverage statistics
            data_completeness_pct = round((scannable_count / max(1, eligible_count)) * 100, 1) if eligible_count > 0 else 0.0
            scan_coverage_pct = round((scanned_successfully_count / max(1, eligible_count)) * 100, 1) if eligible_count > 0 else 0.0
            if universe_size == 0:
                coverage_status = "EMPTY_UNIVERSE"
            elif scanned_successfully_count == 0 and eligible_count > 0:
                coverage_status = "UNAVAILABLE"
            elif scan_coverage_pct >= 99.9:
                coverage_status = "COMPLETE"
            else:
                coverage_status = "PARTIAL"

            universe_metadata = {
                "scope_class": "US_EQUITIES",
                "display_name": RADAR_SCOPE_LABEL,
                "source_population_count": build_attestation.get("source_population_count", eligible_count) if build_attestation else eligible_count,
                "eligible_universe_count": eligible_count,
                "universe_version": version_tuple.universe_version,
                "universe_build_id": target_build_id,
                "membership_hash": build_attestation.get("eligible_membership_hash") if build_attestation else None,
                "construction_status": build_attestation.get("construction_status", "COMPLETE") if build_attestation else "COMPLETE",
            }
            coverage_metadata = {
                "coverage_status": coverage_status,
                "data_complete_count": scannable_count,
                "scanned_successfully_count": scanned_successfully_count,
                "matched_count": matched_count,
                "unavailable_symbol_count": unavailable_count + (scannable_count - scanned_successfully_count),
                "unresolved_symbol_count": 0,
                "data_completeness_pct": data_completeness_pct,
                "scan_coverage_pct": scan_coverage_pct,
                "unavailable_reasons": unavailable_reasons,
                "adjustment_state_distribution": adjustment_state_counts,
            }

            # Optional quorum enforcement check (owner configurable via min_coverage_pct)
            is_sub_quorum = (min_coverage_pct is not None and scan_coverage_pct < min_coverage_pct)

            # Publication Integrity Gate
            eval_report = ScannerPublicationIntegrityEngine.evaluate_candidate_publication(
                scanner_id="MINERVINI_VCP",
                version_tuple=version_tuple,
                ruleset_hash=CANONICAL_VCP_RULESET_HASH,
                evidence_schema_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
                score_model_hash=CANONICAL_VCP_SCORE_MODEL_HASH,
                data_provenance_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
                universe_hash=CANONICAL_VCP_UNIVERSE_HASH,
                freshness_policy_hash=CANONICAL_VCP_FRESHNESS_HASH,
            )

            # Resolve 4-way semantic state:
            # Scenario A: Computed evaluation with zero matches (e.g. empty universe or computed 0-match)
            # Scenario B: Uncomputed / pipeline pending (pattern unverified across evaluated assets)
            # Scenario C: Evaluation failed / insufficient data
            # Scenario D: Computed with supported matches
            if eval_report.decision != PublicationDecision.PUBLISH:
                status_at_pub = ScannerStatus.ERROR
                effective_matched_count = None
                coverage_metadata["coverage_status"] = "UNAVAILABLE"
                coverage_metadata["failure_reason"] = "INTEGRITY_PUBLICATION_REJECTED"
            elif universe_size == 0:
                status_at_pub = ScannerStatus.ERROR
                effective_matched_count = None
                failure_reason = "BUILD_MEMBERSHIP_EMPTY" if target_build_id and universe_override is None else "EMPTY_UNIVERSE"
                coverage_metadata["coverage_status"] = "EMPTY_UNIVERSE"
                coverage_metadata["failure_reason"] = failure_reason
            elif is_sub_quorum:
                status_at_pub = ScannerStatus.ERROR
                effective_matched_count = None
                coverage_metadata["coverage_status"] = "UNAVAILABLE"
                coverage_metadata["failure_reason"] = "SUB_QUORUM_COVERAGE"
            elif len(qualified_candidates) > 0:
                status_at_pub = ScannerStatus.AVAILABLE
                effective_matched_count = len(qualified_candidates)
            elif scanned_successfully_count == 0:
                status_at_pub = ScannerStatus.ERROR
                effective_matched_count = None
                coverage_metadata["coverage_status"] = "UNAVAILABLE"
                coverage_metadata["failure_reason"] = "ZERO_SYMBOLS_SCANNED"
            elif computed_evaluations_count == 0 and uncomputed_evaluations_count > 0:
                status_at_pub = ScannerStatus.PIPELINE_PENDING
                effective_matched_count = None
            else:
                status_at_pub = ScannerStatus.AVAILABLE
                effective_matched_count = 0

            coverage_metadata["matched_count"] = effective_matched_count

            snapshot_id = f"vcp-snap-{int(time.time())}-{uuid.uuid4().hex[:8]}"

            provenance_dict: Dict[str, Any] = {
                "role": "PRICE_VOLUME_HISTORY",
                "source": "LOCAL_MARKET_DB_SQLITE",
                "source_contract_version": "1.0.0",
                "data_as_of": data_as_of,
                "universe_metadata": universe_metadata,
                "coverage_metadata": coverage_metadata,
            }
            if lease:
                provenance_dict["coordination"] = lease.to_provenance()
            else:
                provenance_dict["coordination"] = "NOT_APPLICABLE_PRE_COORDINATION"
            provenance_dict["shadow_governance"] = {
                "sprint_3_shadow_engineering": "AUTHORIZED",
                "shadow_evidence_authority": "PRODUCTION_ENGINEERING_OBSERVATION",
                "shadow_domain_authority": "INTERNAL_REFERENCE_ONLY",
                "external_validation_status": "DEFERRED",
                "candidate_generation_id": "CANDIDATE_GENERATION_001",
            }

            snapshot_record = {
                "snapshot_id": snapshot_id,
                "scanner_id": "MINERVINI_VCP",
                "run_id": run_id,
                "api_contract_version": version_tuple.api_contract_version,
                "ruleset_version": version_tuple.ruleset_version,
                "evidence_schema_version": version_tuple.evidence_schema_version,
                "score_model_version": version_tuple.score_model_version,
                "data_provenance_version": version_tuple.data_provenance_version,
                "universe_version": version_tuple.universe_version,
                "freshness_policy_version": version_tuple.freshness_policy_version,
                "implementation_release_sha": version_tuple.implementation_release_sha,
                "semantic_fingerprint": semantic_fingerprint,
                "generated_at": generated_at,
                "data_as_of": data_as_of,
                "status_at_publication": status_at_pub.value,
                "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
                "universe_size": universe_size,
                "matched_count": effective_matched_count,
                "results": qualified_candidates,
                "provenance": provenance_dict,
                "freshness": {
                    "policy_version": version_tuple.freshness_policy_version,
                    "generated_at": generated_at,
                    "data_as_of": data_as_of,
                    "evaluated_at": generated_at,
                    "status": FreshnessStatus.LIVE.value,
                },
                "publication_decision": eval_report.decision.value,
                "universe_metadata": universe_metadata,
                "coverage_metadata": coverage_metadata,
            }

            if eval_report.decision == PublicationDecision.PUBLISH:
                if lease is not None:
                    # Transactional Fenced Publication (Atomic authority check + publication)
                    self.fenced_publisher.publish_snapshot_fenced(
                        lease=lease,
                        snapshot_data=snapshot_record,
                        universe_build_id=target_build_id,
                        policy=self.lease_policy,
                    )
                    lease_released_by_publisher = True
                else:
                    self.snapshot_store.save_snapshot(snapshot_record)
                self._cached_active_snapshot = snapshot_record
                logger.info(f"VCP Scanner snapshot {snapshot_id} published successfully with status={status_at_pub.value}, matches={effective_matched_count}.")
            else:
                if lease is not None:
                    self.coordinator.release(lease)
                    lease_released_by_publisher = True
                logger.warning(f"VCP Scanner snapshot {snapshot_id} quarantined due to violations: {eval_report.violations}")

            return ImmutableScannerSnapshot(
                scanner_id="MINERVINI_VCP",
                run_id=run_id,
                snapshot_id=snapshot_id,
                version_tuple=version_tuple,
                semantic_fingerprint=semantic_fingerprint,
                generated_at=generated_at,
                data_as_of=data_as_of,
                status_at_publication=status_at_pub,
                universe_id="ARX_CANONICAL_LONG_TERM_V1",
                universe_size=universe_size,
                matched_count=effective_matched_count,
                results=qualified_candidates,
                provenance=snapshot_record["provenance"],
                freshness=snapshot_record["freshness"],
                publication_decision=eval_report.decision,
                universe_metadata=universe_metadata,
                coverage_metadata=coverage_metadata,
            ).to_envelope()
        finally:
            if lease is not None and not lease_released_by_publisher:
                self.coordinator.release(lease)
            if acquired_thread_lock:
                self._scan_lock.release()

    @staticmethod
    def _is_candidate_vcp_classification_authoritative(
        cand: Dict[str, Any],
        snapshot_provenance: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate whether a single candidate possesses affirmative, verifiable,
        authoritative VCP classification.
        """
        if not isinstance(cand, dict):
            return False

        evidence = cand.get("scanner_evidence")
        if not isinstance(evidence, dict):
            return False

        setup_pattern = str(cand.get("setup_pattern") or evidence.get("setup_pattern") or "")
        vcp_stage = str(cand.get("vcp_stage") or evidence.get("vcp_stage") or "")
        stage_phase = str(cand.get("stage_phase") or evidence.get("stage_phase") or "")

        # 1. Reject unverified or uncomputed indicators
        if any("Unverified" in s or "Uncomputed" in s for s in (setup_pattern, vcp_stage, stage_phase)):
            return False

        # 2. Check provenance authority boundary:
        # If provenance indicates internal reference only / deferred validation,
        # it cannot claim public classification authority.
        prov = snapshot_provenance if isinstance(snapshot_provenance, dict) else {}
        shadow_gov = prov.get("shadow_governance", {})
        if isinstance(shadow_gov, dict):
            if (
                shadow_gov.get("shadow_domain_authority") == "INTERNAL_REFERENCE_ONLY"
                and shadow_gov.get("external_validation_status") == "DEFERRED"
            ):
                return False

        # 3. Must specify an authorized classification authority
        authority = (
            evidence.get("classification_authority")
            or cand.get("classification_authority")
            or prov.get("classification_authority")
        )
        if authority not in SUPPORTED_VCP_CLASSIFICATION_AUTHORITIES:
            return False

        # 4. Must declare affirmative evaluation flag
        if evidence.get("vcp_authoritative_evaluated") is not True and cand.get("vcp_authoritative_evaluated") is not True:
            return False

        # 5. Must have affirmative boolean pattern detection and stage 2 confirmation
        if evidence.get("vcp_pattern_detected") is not True:
            return False
        if evidence.get("stage2_status") != "STAGE_2_CONFIRMED":
            return False

        # 6. Must have calculated moving average (SMA50)
        sma_50 = evidence.get("sma_50")
        if sma_50 is None or not isinstance(sma_50, (int, float)) or sma_50 <= 0:
            return False

        return True

    def _has_unsupported_vcp_matches(self, rec: Dict[str, Any]) -> bool:
        """
        Detect if a snapshot claims AVAILABLE or STALE matches but contains unsupported,
        unverified, or legacy proxy classifications.

        Enforces affirmative classification trust:
        - If snapshot claims AVAILABLE or STALE status:
          - If matched_count is 0 and results is empty: returns False (valid zero-match evaluation).
          - If matched_count > 0 or results has candidates:
            EVERY candidate claiming VCP qualification or containing scanner evidence
            must be authoritatively verified via _is_candidate_vcp_classification_authoritative.
          - If any candidate contains legacy label strings ('Minervini', 'VCP', 'Stage 2')
            without authoritative classification, fails closed.
          - If any candidate contains unverified/uncomputed indicators, fails closed.
          - If matched_count is inconsistent with results length, fails closed.
        """
        if not isinstance(rec, dict):
            return True

        status_pub = rec.get("status_at_publication")
        if status_pub not in (ScannerStatus.AVAILABLE.value, "AVAILABLE", ScannerStatus.STALE.value, "STALE"):
            return False

        matched_count = rec.get("matched_count")
        results = rec.get("results")

        # Invariant: Reject boolean, non-int, or negative matched_count values
        if type(matched_count) is bool or (matched_count is not None and type(matched_count) is not int) or (type(matched_count) is int and matched_count < 0):
            return True

        # Inconsistent null matched_count on AVAILABLE/STALE snapshot
        if matched_count is None and (results is None or len(results) == 0):
            return True

        # Valid zero-match evaluation
        if matched_count == 0 and (results is None or (isinstance(results, list) and len(results) == 0)):
            return False

        if not isinstance(results, list):
            return True

        if (matched_count or 0) > 0 and len(results) == 0:
            return True
        if (matched_count or 0) == 0 and len(results) > 0:
            return True

        provenance = rec.get("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}

        for cand in results:
            if not isinstance(cand, dict):
                return True
            evidence = cand.get("scanner_evidence", {})
            setup_pattern = str(cand.get("setup_pattern") or evidence.get("setup_pattern") or "")
            vcp_stage = str(cand.get("vcp_stage") or evidence.get("vcp_stage") or "")
            stage_phase = str(cand.get("stage_phase") or evidence.get("stage_phase") or "")

            # 1. Any unverified or uncomputed indicators in candidates claiming VCP qualification
            if any("Unverified" in s or "Uncomputed" in s for s in (setup_pattern, vcp_stage, stage_phase)):
                return True

            # 2. Legacy proxy fallback or claims of VCP / Minervini / Stage 2
            is_claiming_vcp = any(
                term in s for s in (setup_pattern, vcp_stage, stage_phase)
                for term in ("Minervini", "VCP", "Stage 2")
            )
            # If candidate claims VCP or contains scanner_evidence
            if is_claiming_vcp or "scanner_evidence" in cand:
                if not self._is_candidate_vcp_classification_authoritative(cand, snapshot_provenance=provenance):
                    return True

        return False

    def get_active_or_latest_snapshot(self) -> Dict[str, Any]:
        """Retrieve active in-memory snapshot or load latest from store without initiating background scan execution."""
        try:
            if self._cached_active_snapshot is not None:
                if (
                    self._cached_active_snapshot.get("implementation_release_sha") == CURRENT_IMPLEMENTATION_RELEASE_SHA
                    and not self._has_unsupported_vcp_matches(self._cached_active_snapshot)
                ):
                    latest_id = self.snapshot_store.get_latest_active_snapshot_id("MINERVINI_VCP")
                    if latest_id == self._cached_active_snapshot.get("snapshot_id"):
                        return self._to_envelope(self._cached_active_snapshot)

            persisted = self.snapshot_store.get_latest_active_snapshot("MINERVINI_VCP")
            if persisted is not None:
                if persisted.get("implementation_release_sha") != CURRENT_IMPLEMENTATION_RELEASE_SHA:
                    return self._to_status_envelope(
                        status=ScannerStatus.PIPELINE_PENDING,
                        reason="RELEASE_MISMATCH",
                        detail=(
                            f"Latest snapshot implementation release ({persisted.get('implementation_release_sha')}) "
                            f"does not match current release ({CURRENT_IMPLEMENTATION_RELEASE_SHA}). Pipeline execution pending."
                        ),
                    )
                if self._has_unsupported_vcp_matches(persisted):
                    logger.warning("VCPScannerRunner: retained snapshot contains unsupported VCP matches; failing closed to PIPELINE_PENDING.")
                    return self._to_status_envelope(
                        status=ScannerStatus.PIPELINE_PENDING,
                        reason="UNSUPPORTED_LEGACY_MATCHES",
                        detail="Retained scanner snapshot contains unsupported or unverified VCP classifications. Pipeline execution pending.",
                    )
                self._cached_active_snapshot = persisted
                return self._to_envelope(persisted)

            # Store is empty / cache miss: fail closed to PIPELINE_PENDING without initiating execution
            return self._to_status_envelope(
                status=ScannerStatus.PIPELINE_PENDING,
                reason="NO_ACTIVE_SNAPSHOT",
                detail="No published VCP scanner snapshot is active for current implementation release. Pipeline execution pending.",
            )
        except Exception as e:
            logger.error(f"VCPScannerRunner: snapshot store lookup failed: {e}")
            return self._to_status_envelope(
                status=ScannerStatus.ERROR,
                reason="STORAGE_ERROR",
                detail=f"Failed to access scanner snapshot store: {e}",
            )

    def _evaluate_freshness_status(self, rec: Dict[str, Any]) -> str:
        freshness_dict = rec.get("freshness", {})
        if isinstance(freshness_dict, dict) and freshness_dict.get("status") in (FreshnessStatus.STALE.value, "STALE"):
            return FreshnessStatus.STALE.value
        if rec.get("status_at_publication") in (ScannerStatus.STALE.value, "STALE"):
            return FreshnessStatus.STALE.value
        return freshness_dict.get("status", FreshnessStatus.LIVE.value) if isinstance(freshness_dict, dict) else FreshnessStatus.LIVE.value

    def _to_status_envelope(
        self,
        status: ScannerStatus,
        reason: str,
        detail: str,
        freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN,
    ) -> Dict[str, Any]:
        version_tuple = self.get_version_tuple()
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {
            "scanner_id": "MINERVINI_VCP",
            "api_contract_version": version_tuple.api_contract_version,
            "status": status.value,
            "reason": reason,
            "detail": detail,
            "snapshot_id": None,
            "matched_count": None,
            "data_as_of": None,
            "generated_at": now_iso,
            "methodology": {
                "ruleset_version": version_tuple.ruleset_version,
                "evidence_schema_version": version_tuple.evidence_schema_version,
                "score_model_version": version_tuple.score_model_version,
            },
            "provenance": {
                "data_provenance_version": version_tuple.data_provenance_version,
                "universe_version": version_tuple.universe_version,
                "implementation_release_sha": version_tuple.implementation_release_sha,
                "source": "FAIL_CLOSED_READ_CONTRACT",
            },
            "freshness": {
                "policy_version": version_tuple.freshness_policy_version,
                "generated_at": now_iso,
                "data_as_of": None,
                "evaluated_at": now_iso,
                "status": freshness_status.value,
                "rationale": detail,
            },
            "snapshot": {
                "run_id": None,
                "snapshot_id": None,
                "universe_size": len(CANONICAL_VCP_UNIVERSE),
                "matched_count": None,
                "semantic_fingerprint": None,
            },
            "universe": {
                "scope_class": "US_EQUITIES",
                "display_name": RADAR_SCOPE_LABEL,
                "source_population_count": 0,
                "eligible_universe_count": 0,
                "universe_version": version_tuple.universe_version,
                "universe_build_id": None,
                "membership_hash": None,
                "construction_status": "NONE",
            },
            "coverage": {
                "coverage_status": "PENDING" if status == ScannerStatus.PIPELINE_PENDING else "UNAVAILABLE",
                "data_complete_count": 0,
                "scanned_successfully_count": 0,
                "matched_count": None,
                "unavailable_symbol_count": 0,
                "unresolved_symbol_count": 0,
                "data_completeness_pct": 0.0,
                "scan_coverage_pct": 0.0,
                "unavailable_reasons": {},
            },
            "results": [],
        }

    def _to_envelope(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        prov = rec.get("provenance", {}) if isinstance(rec.get("provenance"), dict) else {}
        u_meta = rec.get("universe_metadata") or prov.get("universe_metadata", {})
        c_meta = rec.get("coverage_metadata") or prov.get("coverage_metadata", {})
        u_size = rec.get("universe_size", 0)
        freshness_status = self._evaluate_freshness_status(rec)
        status = rec["status_at_publication"]
        if freshness_status == FreshnessStatus.STALE.value and status == ScannerStatus.AVAILABLE.value:
            status = ScannerStatus.STALE.value
        return {
            "scanner_id": rec["scanner_id"],
            "api_contract_version": rec["api_contract_version"],
            "status": status,
            "snapshot_id": rec.get("snapshot_id"),
            "matched_count": rec.get("matched_count"),
            "data_as_of": rec.get("data_as_of"),
            "generated_at": rec.get("generated_at"),
            "methodology": {
                "ruleset_version": rec["ruleset_version"],
                "evidence_schema_version": rec["evidence_schema_version"],
                "score_model_version": rec["score_model_version"],
            },
            "provenance": {
                "data_provenance_version": rec["data_provenance_version"],
                "universe_version": rec["universe_version"],
                "implementation_release_sha": rec["implementation_release_sha"],
                **prov,
            },
            "freshness": {
                "policy_version": rec["freshness_policy_version"],
                "generated_at": rec["generated_at"],
                "data_as_of": rec["data_as_of"],
                "evaluated_at": rec.get("freshness", {}).get("evaluated_at", rec["generated_at"]),
                "status": freshness_status,
            },
            "snapshot": {
                "run_id": rec["run_id"],
                "snapshot_id": rec["snapshot_id"],
                "universe_size": u_size,
                "matched_count": rec["matched_count"],
                "semantic_fingerprint": rec["semantic_fingerprint"],
            },
            "universe": {
                "scope_class": u_meta.get("scope_class", "US_EQUITIES"),
                "display_name": u_meta.get("display_name", RADAR_SCOPE_LABEL),
                "source_population_count": u_meta.get("source_population_count", u_size),
                "eligible_universe_count": u_meta.get("eligible_universe_count", u_size),
                "universe_version": rec["universe_version"],
                "universe_build_id": u_meta.get("universe_build_id"),
                "membership_hash": u_meta.get("membership_hash"),
                "construction_status": u_meta.get("construction_status", "COMPLETE"),
            },
            "coverage": resolve_coverage_envelope(
                status=status,
                universe_size=u_size,
                matched_count=rec.get("matched_count"),
                c_meta=c_meta,
                failure_reason_fallback=c_meta.get("failure_reason"),
            ),
            "results": rec.get("results", []),
        }


RESOURCE_KEY_SMART_MONEY_PIPELINE = "smart_money_pipeline"


class SmartMoneyScannerRunner:
    """
    Market-wide execution runner for Smart Money Form 4 Insider Flow.
    Manages immutable snapshot publication in ScannerSnapshotStore.
    If an active published snapshot exists, returns AVAILABLE snapshot envelope.
    Fails closed to PIPELINE_PENDING with zero fabricated results if no certified snapshot exists.
    """

    def __init__(
        self,
        snapshot_store: Optional[Any] = None,
        form4_scanner: Optional[Any] = None,
        coordinator: Optional[DurableRunCoordinator] = None,
        coordination_store: Optional[CoordinationStore] = None,
        lease_policy: Optional[LeasePolicy] = None,
        backend: Optional[str] = None,
    ):
        from analyst_dashboard.smart_money.storage.config import get_storage_backend
        self.backend = backend or get_storage_backend()

        if self.backend == "postgres":
            if snapshot_store is not None:
                self.snapshot_store = snapshot_store
            else:
                from analyst_dashboard.smart_money.storage.factory import get_smart_money_repository
                self.snapshot_store = get_smart_money_repository()
            self.form4_scanner = form4_scanner
            self.coordination_store = coordination_store
            self.coordinator = coordinator
            self.lease_policy = lease_policy
        else:
            self.snapshot_store = snapshot_store or ScannerSnapshotStore()
            if form4_scanner is not None:
                self.form4_scanner = form4_scanner
            else:
                from analyst_dashboard.smart_money.form4_scanner import SmartMoneyForm4Scanner
                self.form4_scanner = SmartMoneyForm4Scanner()

            coord_db_path = getattr(self.snapshot_store, "db_path", None)
            if coordination_store:
                self.coordination_store = coordination_store
            elif coord_db_path:
                self.coordination_store = CoordinationStore(db_path=coord_db_path)
            else:
                self.coordination_store = CoordinationStore()

            self.lease_policy = lease_policy or PRODUCTION_LEASE_POLICY
            self.coordinator = coordinator or DurableRunCoordinator(
                resource_key=RESOURCE_KEY_SMART_MONEY_PIPELINE,
                db_path=self.coordination_store.db_path,
                policy=self.lease_policy,
            )

        self._scan_lock = threading.Lock()
        self._cached_active_snapshot: Optional[Dict[str, Any]] = None

    def get_version_tuple(self) -> ScannerVersionTuple:
        return ScannerVersionTuple(
            scanner_id="SMART_MONEY",
            api_contract_version=SMART_MONEY_API_CONTRACT_VERSION,
            ruleset_version=SMART_MONEY_RULESET_VERSION,
            evidence_schema_version=SMART_MONEY_EVIDENCE_SCHEMA_VERSION,
            score_model_version=SMART_MONEY_SCORE_MODEL_VERSION,
            data_provenance_version=SMART_MONEY_DATA_PROVENANCE_VERSION,
            universe_version=SMART_MONEY_UNIVERSE_VERSION,
            freshness_policy_version=SMART_MONEY_FRESHNESS_POLICY_VERSION,
            implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        )

    @classmethod
    def get_status_envelope(cls) -> Dict[str, Any]:
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {
            "scanner_id": "SMART_MONEY",
            "api_contract_version": SMART_MONEY_API_CONTRACT_VERSION,
            "status": ScannerStatus.PIPELINE_PENDING.value,
            "methodology": {
                "ruleset_version": SMART_MONEY_RULESET_VERSION,
                "evidence_schema_version": SMART_MONEY_EVIDENCE_SCHEMA_VERSION,
                "score_model_version": SMART_MONEY_SCORE_MODEL_VERSION,
            },
            "provenance": {
                "data_provenance_version": SMART_MONEY_DATA_PROVENANCE_VERSION,
                "universe_version": SMART_MONEY_UNIVERSE_VERSION,
                "implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA,
                "subsignal": "INSIDER_FORM4",
                "role": "INSIDER_FORM4_POINT_IN_TIME_FLOW",
                "source": "UNPROVISIONED_MARKET_WIDE_FEED",
            },
            "freshness": {
                "policy_version": SMART_MONEY_FRESHNESS_POLICY_VERSION,
                "generated_at": now_str,
                "data_as_of": None,
                "evaluated_at": now_str,
                "status": FreshnessStatus.UNKNOWN.value,
                "rationale": "Live market-wide institutional positioning, 13F filing aggregation, and dark pool feeds are disconnected for general universe screening. Point-in-time Form 4 screening active via /screener/smart-money/scan.",
            },
            "snapshot": {
                "run_id": None,
                "snapshot_id": None,
                "universe_size": len(CANONICAL_VCP_UNIVERSE),
                "matched_count": None,  # Invariant: matched_count is NOT_APPLICABLE when PIPELINE_PENDING
            },
            "results": [],
        }

    def _to_envelope(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        prov = rec.get("provenance", {}) if isinstance(rec.get("provenance"), dict) else {}
        u_meta = rec.get("universe_metadata") or prov.get("universe_metadata", {})
        c_meta = rec.get("coverage_metadata") or prov.get("coverage_metadata", {})
        u_size = rec.get("universe_size", 0)
        status = rec.get("status_at_publication", ScannerStatus.AVAILABLE.value)
        return {
            "scanner_id": "SMART_MONEY",
            "api_contract_version": rec.get("api_contract_version", SMART_MONEY_API_CONTRACT_VERSION),
            "status": status,
            "methodology": {
                "ruleset_version": rec.get("ruleset_version", SMART_MONEY_RULESET_VERSION),
                "evidence_schema_version": rec.get("evidence_schema_version", SMART_MONEY_EVIDENCE_SCHEMA_VERSION),
                "score_model_version": rec.get("score_model_version", SMART_MONEY_SCORE_MODEL_VERSION),
            },
            "provenance": {
                "data_provenance_version": rec.get("data_provenance_version", SMART_MONEY_DATA_PROVENANCE_VERSION),
                "universe_version": rec.get("universe_version", SMART_MONEY_UNIVERSE_VERSION),
                "implementation_release_sha": rec.get("implementation_release_sha", CURRENT_IMPLEMENTATION_RELEASE_SHA),
                "subsignal": "INSIDER_FORM4",
                **prov,
            },
            "freshness": rec.get("freshness", {}),
            "snapshot": {
                "run_id": rec.get("run_id"),
                "snapshot_id": rec.get("snapshot_id"),
                "universe_size": u_size,
                "matched_count": rec.get("matched_count"),
                "semantic_fingerprint": rec.get("semantic_fingerprint"),
                "publication_decision": rec.get("publication_decision", "PUBLISH"),
            },
            "universe": {
                "scope_class": u_meta.get("scope_class", "US_EQUITIES"),
                "display_name": u_meta.get("display_name", RADAR_SCOPE_LABEL),
                "source_population_count": u_meta.get("source_population_count", u_size),
                "eligible_universe_count": u_meta.get("eligible_universe_count", u_size),
                "universe_version": rec.get("universe_version", SMART_MONEY_UNIVERSE_VERSION),
                "universe_build_id": u_meta.get("universe_build_id"),
                "membership_hash": u_meta.get("membership_hash"),
                "construction_status": "COMPLETE",
            },
            "coverage": resolve_coverage_envelope(
                status=status,
                universe_size=u_size,
                matched_count=rec.get("matched_count"),
                c_meta=c_meta,
                failure_reason_fallback=c_meta.get("failure_reason"),
            ),
            "results": rec.get("results", []),
        }

    def get_active_or_latest_snapshot(self) -> Dict[str, Any]:
        """
        Retrieves the latest verified, published snapshot from ScannerSnapshotStore.
        If no published snapshot is present, returns fail-closed PIPELINE_PENDING envelope.
        Enforces Control A: Emergency web read disablement via ARX_SMART_MONEY_POSTGRES_READ_ENABLED=false.
        """
        read_enabled = os.environ.get("ARX_SMART_MONEY_POSTGRES_READ_ENABLED", "true").strip().lower()
        if read_enabled in ("false", "0", "no"):
            logger.warning("Smart Money postgres read disabled via ARX_SMART_MONEY_POSTGRES_READ_ENABLED=false (Control A)")
            return self.get_status_envelope()

        if self.backend == "postgres":
            persisted = self.snapshot_store.get_latest_active_snapshot("SMART_MONEY")
            if persisted is not None:
                return self._to_envelope(persisted)
            return self.get_status_envelope()

        if self._cached_active_snapshot is not None:
            return self._to_envelope(self._cached_active_snapshot)

        persisted = self.snapshot_store.get_latest_active_snapshot("SMART_MONEY")
        if persisted is not None:
            self._cached_active_snapshot = persisted
            return self._to_envelope(persisted)

        return self.get_status_envelope()

    def execute_market_wide_scan(
        self,
        as_of_watermark: Optional[str] = None,
        universe_symbols: Optional[List[str]] = None,
        universe_build_id: Optional[str] = None,
        lookback_days: int = 30,
        trigger_type: TriggerType = TriggerType.OPERATOR,
        bypass_thread_lock: bool = False,
        operator_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes point-in-time market-wide Form 4 scan under distributed fencing (Gates B2, B5, B6).
        Persists completed snapshot into ScannerSnapshotStore atomically.
        Enforces receipt-driven certification before advancing active published pointer.
        """
        if self.backend == "postgres" and self.coordinator is None:
            raise RuntimeError(
                "POSTGRES_DIRECT_SCAN_UNSUPPORTED: In PostgreSQL profile, market-wide scans "
                "must be submitted via sm_run_requests asynchronous queue to ensure dedicated worker execution."
            )

        acquired_thread_lock = False
        if not bypass_thread_lock:
            if not self._scan_lock.acquire(blocking=False):
                raise RuntimeError("OVERLAPPING_SMART_MONEY_SCANS_PROHIBITED: Local execution already active.")
            acquired_thread_lock = True

        lease = None
        try:
            effective_instance_id = f"inst-sm-{uuid.uuid4().hex[:8]}"
            effective_job_key = (
                f"smart_money:operator:{operator_request_id}" if operator_request_id else
                f"smart_money:run:{int(time.time())}:{uuid.uuid4().hex[:8]}"
            )
            lease_acq = self.coordinator.acquire(
                logical_job_key=effective_job_key,
                trigger_type=trigger_type,
                owner_instance_id=effective_instance_id,
                implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
                operator_request_id=operator_request_id,
            )
            if lease_acq.status != AcquisitionStatus.ACQUIRED:
                raise RuntimeError(
                    f"OVERLAPPING_SMART_MONEY_SCANS_PROHIBITED: Distributed lease rejected: {lease_acq.status}. "
                    f"Message: {lease_acq.message}"
                )
            lease = lease_acq.lease

            now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            watermark = as_of_watermark or now_str

            # Perform PIT scan strictly bounded by watermark
            scan_snapshot = self.form4_scanner.scan_issuers_as_of(
                as_of_watermark=watermark,
                symbols=universe_symbols,
                universe_build_id=universe_build_id,
                lookback_days=lookback_days,
            )

            # Map results to ScannerCandidateResult objects
            results = []
            for rank, item in enumerate(scan_snapshot.items, start=1):
                results.append(
                    ScannerCandidateResult(
                        symbol=item["symbol"],
                        rank=rank,
                        score=float(item.get("convictionScore", 0.0)),
                        current_price=0.0,
                        scanner_evidence=item,
                    ).to_dict()
                )

            version_tuple = self.get_version_tuple()
            fingerprint_payload = {
                "scanner_id": "SMART_MONEY",
                "watermark": watermark,
                "version_tuple": version_tuple.to_dict(),
                "universe_build_id": scan_snapshot.universe_build_id,
                "matched_count": len(results),
                "results": results,
            }
            semantic_fingerprint = compute_hash(fingerprint_payload)

            snapshot_id = f"SNAP_SM_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            run_id = f"RUN_SM_{int(time.time())}_{uuid.uuid4().hex[:6]}"

            # Completeness & Publication State Machine
            is_complete = (scan_snapshot.coverage_status == "COMPLETE")
            status_at_pub = ScannerStatus.AVAILABLE.value if is_complete else ScannerStatus.PIPELINE_PENDING.value
            pub_decision = PublicationDecision.PUBLISH.value if is_complete else PublicationDecision.QUARANTINE.value

            universe_metadata = {
                "scope_class": "US_EQUITIES",
                "display_name": RADAR_SCOPE_LABEL,
                "source_population_count": scan_snapshot.scannable_count,
                "eligible_universe_count": scan_snapshot.scannable_count,
                "universe_build_id": scan_snapshot.universe_build_id,
                "membership_hash": scan_snapshot.universe_content_hash,
                "construction_status": "COMPLETE",
            }
            coverage_metadata = {
                "coverage_status": scan_snapshot.coverage_status or "PARTIAL",
                "data_complete_count": scan_snapshot.scannable_count,
                "scanned_successfully_count": scan_snapshot.scannable_count,
                "matched_count": len(results) if is_complete else 0,
                "unavailable_symbol_count": 0,
                "unresolved_symbol_count": 0,
                "data_completeness_pct": 100.0 if is_complete else 0.0,
                "scan_coverage_pct": 100.0,
                "unavailable_reasons": {},
            }

            snapshot_record = {
                "snapshot_id": snapshot_id,
                "scanner_id": "SMART_MONEY",
                "run_id": run_id,
                "api_contract_version": version_tuple.api_contract_version,
                "ruleset_version": version_tuple.ruleset_version,
                "evidence_schema_version": version_tuple.evidence_schema_version,
                "score_model_version": version_tuple.score_model_version,
                "data_provenance_version": version_tuple.data_provenance_version,
                "universe_version": version_tuple.universe_version,
                "freshness_policy_version": version_tuple.freshness_policy_version,
                "implementation_release_sha": version_tuple.implementation_release_sha,
                "semantic_fingerprint": semantic_fingerprint,
                "generated_at": now_str,
                "data_as_of": watermark,
                "status_at_publication": status_at_pub,
                "universe_id": scan_snapshot.universe_id,
                "universe_size": scan_snapshot.scannable_count,
                "matched_count": len(results) if is_complete else 0,
                "results": results,
                "provenance": {
                    **scan_snapshot.provenance,
                    "subsignal": "INSIDER_FORM4",
                    "role": "INSIDER_FORM4_POINT_IN_TIME_FLOW",
                    "universe_metadata": universe_metadata,
                    "coverage_metadata": coverage_metadata,
                },
                "freshness": {
                    **scan_snapshot.freshness,
                    "status": FreshnessStatus.LIVE.value,
                },
                "publication_decision": pub_decision,
                "universe_metadata": universe_metadata,
                "coverage_metadata": coverage_metadata,
            }

            self.snapshot_store.save_snapshot(snapshot_record)
            if pub_decision == PublicationDecision.PUBLISH.value:
                self._cached_active_snapshot = snapshot_record
                logger.info(f"Smart Money Scanner snapshot {snapshot_id} published successfully with {len(results)} matches.")
            else:
                logger.info(f"Smart Money Scanner snapshot {snapshot_id} marked {status_at_pub} ({scan_snapshot.coverage_status}) and quarantined from active pointer.")

            return self._to_envelope(snapshot_record)
        finally:
            if lease is not None:
                self.coordinator.release(lease)
            if acquired_thread_lock:
                self._scan_lock.release()

