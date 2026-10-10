"""
Market-Wide Scanner Runners for Radar Scanners.

Implements deterministic market-wide universe execution for:
- MINERVINI_VCP
- SMART_MONEY
"""

import json
import hashlib
import time
import uuid
import logging
import threading
from typing import Dict, Any, List, Optional
import pandas as pd

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
            logger.info(f"Logical job {effective_job_key} already succeeded. Returning active snapshot.")
            return self.get_active_or_latest_snapshot()

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
                scannable_symbols = self.universe_store.get_scannable_membership(target_build_id)
            else:
                latest_build = self.universe_store.get_latest_published_universe(UNIVERSE_ID)
                if latest_build:
                    target_build_id = latest_build["universe_build_id"]
                    scannable_symbols = self.universe_store.get_scannable_membership(target_build_id)
                    build_attestation = latest_build
                else:
                    scannable_symbols = []

            if universe_override is not None:
                universe = universe_override
                eligible_count = len(universe)
                scannable_count = len(universe)
                unavailable_count = 0
            elif scannable_symbols:
                universe = scannable_symbols
                eligible_count = build_attestation["eligible_count"] if build_attestation else len(universe)
                scannable_count = len(scannable_symbols)
                unavailable_count = build_attestation["data_unavailable_count"] if build_attestation else 0
            else:
                # Historical regression fixture fallback
                universe = list(dict.fromkeys(CANONICAL_VCP_UNIVERSE))
                eligible_count = len(universe)
                scannable_count = len(universe)
                unavailable_count = 0

            universe_size = len(universe)
            scanned_successfully_count = 0
            unavailable_reasons: Dict[str, int] = {}
            qualified_candidates: List[Dict[str, Any]] = []

            for sym in universe:
                clean_sym = sym.strip().upper()
                latest = self.market_db.get_latest_price(clean_sym)
                if not latest or not latest.get("currentPrice") or latest["currentPrice"] <= 0:
                    unavailable_reasons["MISSING_PRICE_DATA"] = unavailable_reasons.get("MISSING_PRICE_DATA", 0) + 1
                    continue

                current_price = latest["currentPrice"]
                db_candles = self.market_db.get_daily_candles(clean_sym, limit=60)
                if not db_candles or len(db_candles) < 50:
                    unavailable_reasons["INSUFFICIENT_HISTORY"] = unavailable_reasons.get("INSUFFICIENT_HISTORY", 0) + 1
                    continue

                scanned_successfully_count += 1
                df = pd.DataFrame([{
                    "Open": c["open"], "High": c["high"], "Low": c["low"], "Close": c["close"], "Volume": c["volume"]
                } for c in db_candles], index=pd.to_datetime([c["time"] for c in db_candles]))

                exec_res = OptimalExecutionEngine.calculate_trade_levels(df, current_price, user_role="LONG_TERM")

                # Qualification ruleset: Must be confirmed VCP in Stage 2 Advancing Growth Phase
                is_vcp = exec_res.get("vcp_contraction_status") == "VCP 3-Stage Compression Confirmed"
                is_stage_2 = exec_res.get("stage_phase") == "Stage 2 Advancing Growth Phase"

                if is_vcp and is_stage_2:
                    # Calculate multi-factor confluence conviction score
                    conf_res = self.confluence_engine.calculate_confluence(
                        symbol=clean_sym,
                        technical_data={
                            "executionStatus": exec_res["execution_status"],
                            "riskRewardRatio": exec_res["risk_reward_ratio"],
                            "setup_pattern": exec_res["setup_pattern"],
                            "stage_phase": exec_res["stage_phase"],
                            "rsi_14": exec_res.get("rsi_14"),
                            "stop_loss": exec_res["stop_loss"],
                            "current_price": current_price,
                        },
                    )
                    score = round(float(conf_res.get("confluenceScore", 75.0)), 1)

                    scanner_evidence = {
                        "vcp_stage": exec_res.get("vcp_contraction_status"),
                        "setup_pattern": exec_res.get("setup_pattern"),
                        "stage_phase": exec_res.get("stage_phase"),
                        "sma_50": exec_res.get("breakout_pivot"),
                        "ema_20": exec_res.get("optimal_entry_min"),
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
                    }

                    qualified_candidates.append({
                        "symbol": clean_sym,
                        "score": score,
                        "current_price": current_price,
                        "scanner_evidence": scanner_evidence,
                    })

            # Deterministic ranking: score descending, symbol ascending
            qualified_candidates.sort(key=lambda c: (-c["score"], c["symbol"]))
            for rank_idx, cand in enumerate(qualified_candidates, start=1):
                cand["rank"] = rank_idx

            matched_count = len(qualified_candidates)
            data_as_of = time.strftime("%Y-%m-%d", time.gmtime())

            version_tuple = self.get_version_tuple()
            semantic_fingerprint = ScannerPublicationIntegrityEngine.get_canonical_vcp_fingerprint()

            # Record shadow observations under Sprint 3 contamination-controlled governance
            shadow_trigger_class = (
                shadow_trigger_override if shadow_trigger_override in ("TEST", "REPLAY", "SYNTHETIC")
                else ("NATURAL_PRODUCTION" if trigger_type == TriggerType.SCHEDULED and not operator_request_id
                      else "ADMIN_FORCED")
            )
            for cand in qualified_candidates:
                evidence_dict = cand.get("scanner_evidence", {})
                self.shadow_suite.record_shadow_observation(
                    security_id=cand["symbol"],
                    evaluation_as_of=data_as_of,
                    universe_build_id=target_build_id or "ARX_CANONICAL_UNIVERSE_BUILD",
                    snapshot_run_id=run_id,
                    candidate_generation_id="CANDIDATE_GENERATION_002",
                    ruleset_id="MINERVINI_VCP",
                    ruleset_version=version_tuple.ruleset_version,
                    predicate_vector_hash=hashlib.sha256(json.dumps(evidence_dict, sort_keys=True).encode("utf-8")).hexdigest(),
                    classification="CONFIRMED_VCP_STAGE_2",
                    decision_posture="QUALIFIED_WATCHLIST",
                    input_fingerprint=hashlib.sha256(f"{cand['symbol']}:{cand['current_price']}".encode("utf-8")).hexdigest(),
                    group_or_episode_id=f"EPISODE:{cand['symbol']}:{data_as_of}",
                    trigger_class=shadow_trigger_class,
                )

            # Coverage statistics
            data_completeness_pct = round((scannable_count / max(1, eligible_count)) * 100, 1)
            scan_coverage_pct = round((scanned_successfully_count / max(1, eligible_count)) * 100, 1)
            coverage_status = "COMPLETE" if scan_coverage_pct >= 99.9 else "PARTIAL"

            universe_metadata = {
                "scope_class": "US_EQUITIES",
                "display_name": RADAR_SCOPE_LABEL,
                "source_population_count": build_attestation["source_population_count"] if build_attestation else eligible_count,
                "eligible_universe_count": eligible_count,
                "universe_version": version_tuple.universe_version,
                "universe_build_id": target_build_id,
                "membership_hash": build_attestation["eligible_membership_hash"] if build_attestation else None,
                "construction_status": build_attestation["construction_status"] if build_attestation else "COMPLETE",
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
            }

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

            snapshot_id = f"vcp-snap-{int(time.time())}-{uuid.uuid4().hex[:8]}"

            provenance_dict: Dict[str, Any] = {
                "role": "PRICE_VOLUME_HISTORY",
                "source": "LOCAL_MARKET_DB_SQLITE",
                "source_contract_version": "1.0.0",
                "data_as_of": data_as_of,
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
                "status_at_publication": ScannerStatus.AVAILABLE.value if eval_report.decision == PublicationDecision.PUBLISH else ScannerStatus.ERROR.value,
                "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
                "universe_size": universe_size,
                "matched_count": matched_count,
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
                logger.info(f"VCP Scanner snapshot {snapshot_id} published successfully with {matched_count} matches.")
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
                status_at_publication=ScannerStatus.AVAILABLE if eval_report.decision == PublicationDecision.PUBLISH else ScannerStatus.ERROR,
                universe_id="ARX_CANONICAL_LONG_TERM_V1",
                universe_size=universe_size,
                matched_count=matched_count,
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

    def get_active_or_latest_snapshot(self) -> Dict[str, Any]:
        """Retrieve active in-memory snapshot or load latest from store, executing initial scan if store is empty."""
        if self._cached_active_snapshot is not None:
            return self._to_envelope(self._cached_active_snapshot)

        persisted = self.snapshot_store.get_latest_active_snapshot("MINERVINI_VCP")
        if persisted is not None:
            if persisted.get("implementation_release_sha") == CURRENT_IMPLEMENTATION_RELEASE_SHA:
                self._cached_active_snapshot = persisted
                return self._to_envelope(persisted)

        # First run or new release candidate: execute canonical market-wide scan
        return self.execute_market_wide_scan(trigger_type=TriggerType.SCHEDULED)

    def _to_envelope(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        u_meta = rec.get("universe_metadata", {})
        c_meta = rec.get("coverage_metadata", {})
        u_size = rec.get("universe_size", 0)
        return {
            "scanner_id": rec["scanner_id"],
            "api_contract_version": rec["api_contract_version"],
            "status": rec["status_at_publication"],
            "methodology": {
                "ruleset_version": rec["ruleset_version"],
                "evidence_schema_version": rec["evidence_schema_version"],
                "score_model_version": rec["score_model_version"],
            },
            "provenance": {
                "data_provenance_version": rec["data_provenance_version"],
                "universe_version": rec["universe_version"],
                "implementation_release_sha": rec["implementation_release_sha"],
                **rec.get("provenance", {}),
            },
            "freshness": {
                "policy_version": rec["freshness_policy_version"],
                "generated_at": rec["generated_at"],
                "data_as_of": rec["data_as_of"],
                "evaluated_at": rec.get("freshness", {}).get("evaluated_at", rec["generated_at"]),
                "status": rec.get("freshness", {}).get("status", FreshnessStatus.LIVE.value),
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
            "coverage": {
                "coverage_status": c_meta.get("coverage_status", "COMPLETE"),
                "data_complete_count": c_meta.get("data_complete_count", u_size),
                "scanned_successfully_count": c_meta.get("scanned_successfully_count", u_size),
                "matched_count": rec["matched_count"],
                "unavailable_symbol_count": c_meta.get("unavailable_symbol_count", 0),
                "unresolved_symbol_count": c_meta.get("unresolved_symbol_count", 0),
                "data_completeness_pct": c_meta.get("data_completeness_pct", 100.0),
                "scan_coverage_pct": c_meta.get("scan_coverage_pct", 100.0),
                "unavailable_reasons": c_meta.get("unavailable_reasons", {}),
            },
            "results": rec.get("results", []),
        }


class SmartMoneyScannerRunner:
    """
    Market-wide execution runner for Smart Money / Institutional Flow.
    Data Authority Status: INCOMPLETE (Live market-wide institutional feed unprovisioned).
    Fails closed to PIPELINE_PENDING with zero fabricated results.
    """

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
                "role": "INSTITUTIONAL_13F_DARK_POOL_FLOW",
                "source": "UNPROVISIONED_MARKET_WIDE_FEED",
            },
            "freshness": {
                "policy_version": SMART_MONEY_FRESHNESS_POLICY_VERSION,
                "generated_at": now_str,
                "data_as_of": None,
                "evaluated_at": now_str,
                "status": FreshnessStatus.UNKNOWN.value,
                "rationale": "Live market-wide institutional positioning, 13F filing aggregation, and dark pool feeds are disconnected for general universe screening. Single-asset Form 4 and Capitol Hill disclosures remain available on /smart-money.",
            },
            "snapshot": {
                "run_id": None,
                "snapshot_id": None,
                "universe_size": len(CANONICAL_VCP_UNIVERSE),
                "matched_count": None,  # Invariant: matched_count is NOT_APPLICABLE when PIPELINE_PENDING
            },
            "results": [],
        }

    def get_active_or_latest_snapshot(self) -> Dict[str, Any]:
        """Smart Money is blocked on market-wide institutional data authority; returns PIPELINE_PENDING envelope."""
        return self.get_status_envelope()

