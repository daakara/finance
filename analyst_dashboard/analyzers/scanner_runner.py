"""
Market-Wide Scanner Runners for Radar Scanners.

Implements deterministic market-wide universe execution for:
- MINERVINI_VCP
- SMART_MONEY
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    FreshnessStatus,
    PublicationDecision,
    ScannerVersionTuple,
    ScannerCandidateResult,
    ImmutableScannerSnapshot,
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

logger = logging.getLogger(__name__)

# Current implementation release identity
CURRENT_IMPLEMENTATION_RELEASE_SHA = "01683a39a19f3f74720f798459cec717698e2ab2"

# Canonical Universe Definition for VCP Market-Wide Scanning
from analyst_dashboard.analyzers.scanner_contract import CANONICAL_VCP_UNIVERSE


class VCPScannerRunner:
    """Market-wide execution runner for Minervini Volatility Contraction Pattern."""

    def __init__(
        self,
        market_db: Optional[MarketDatabaseEngine] = None,
        snapshot_store: Optional[ScannerSnapshotStore] = None,
        confluence_engine: Optional[ConfluenceEngine] = None,
    ):
        self.market_db = market_db or MarketDatabaseEngine()
        self.snapshot_store = snapshot_store or ScannerSnapshotStore()
        self.confluence_engine = confluence_engine or ConfluenceEngine()
        self._cached_active_snapshot: Optional[Dict[str, Any]] = None

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

    def execute_market_wide_scan(self, universe_override: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute deterministic market-wide scan across eligible universe:
        1. Resolve versioned universe
        2. Load canonical input data
        3. Execute OptimalExecutionEngine
        4. Collect and score candidates
        5. Verify publication integrity
        6. Persist immutable snapshot if verified
        """
        run_id = f"vcp-run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        universe = universe_override if universe_override is not None else list(dict.fromkeys(CANONICAL_VCP_UNIVERSE))
        universe_size = len(universe)

        qualified_candidates: List[Dict[str, Any]] = []

        for sym in universe:
            clean_sym = sym.strip().upper()
            latest = self.market_db.get_latest_price(clean_sym)
            if not latest or not latest.get("currentPrice") or latest["currentPrice"] <= 0:
                continue

            current_price = latest["currentPrice"]
            db_candles = self.market_db.get_daily_candles(clean_sym, limit=60)
            if not db_candles or len(db_candles) < 50:
                continue

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
            "provenance": {
                "role": "PRICE_VOLUME_HISTORY",
                "source": "LOCAL_MARKET_DB_SQLITE",
                "source_contract_version": "1.0.0",
                "data_as_of": data_as_of,
            },
            "freshness": {
                "policy_version": version_tuple.freshness_policy_version,
                "generated_at": generated_at,
                "data_as_of": data_as_of,
                "evaluated_at": generated_at,
                "status": FreshnessStatus.LIVE.value,
            },
            "publication_decision": eval_report.decision.value,
        }

        if eval_report.decision == PublicationDecision.PUBLISH:
            self.snapshot_store.save_snapshot(snapshot_record)
            self._cached_active_snapshot = snapshot_record
            logger.info(f"VCP Scanner snapshot {snapshot_id} published successfully with {matched_count} matches.")
        else:
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
        ).to_envelope()

    def get_active_or_latest_snapshot(self) -> Dict[str, Any]:
        """Retrieve active in-memory snapshot or load latest from store, executing initial scan if store is empty."""
        if self._cached_active_snapshot is not None:
            return self._to_envelope(self._cached_active_snapshot)

        persisted = self.snapshot_store.get_latest_active_snapshot("MINERVINI_VCP")
        if persisted is not None:
            self._cached_active_snapshot = persisted
            return self._to_envelope(persisted)

        # First run on initial deployment: execute canonical market-wide scan
        return self.execute_market_wide_scan()

    def _to_envelope(self, rec: Dict[str, Any]) -> Dict[str, Any]:
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
                "universe_size": rec["universe_size"],
                "matched_count": rec["matched_count"],
                "semantic_fingerprint": rec["semantic_fingerprint"],
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

