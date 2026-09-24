"""ARX Prospective Validation Epoch 1 — Passive Capture Hook.

Provides immutable, fail-closed, zero-side-effect passive observation
of natural production recommendations for prospective prediction validation.

Invariants Enforced:
1. ZERO EXECUTION MUTATION: No broker interaction, no order placement, no capital allocation.
2. STRICT FAIL-CLOSED: Capture exceptions are logged; analytics responses are NEVER blocked or altered.
3. DUAL-SHA IDENTITY:
   - DECISION_ENGINE_SHA: 7ad44595826c147cc77f93cd676af520764c7442
   - OBSERVATION_GOVERNANCE_SHA: 9bc1854c729974ba03548549091c4735d1bf0414
4. COMPLETE CONTENT-ADDRESSED SNAPSHOTS:
   - Market data payload & timestamp
   - Fundamental data payload & filing timestamp
   - Canonical normalized FRED macro payload & timestamp
   - Model configuration hash
5. TEMPORAL INTEGRITY (ANTI-LOOKAHEAD):
   - Every source observation timestamp <= recommendation timestamp
   - NO_SOURCE_INFORMATION_AVAILABLE_AFTER_RECOMMENDATION
6. OUTCOME ISOLATION:
   - Initial outcome state is PENDING / OPEN
   - realizedOutcome = None, MFE/MAE = null (sessionsObserved = 0)
"""

import os
import json
import math
import logging
import hashlib
import contextvars
from enum import Enum
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)

logger = logging.getLogger("arx.governance.passive_capture")


class ExecutionContext(str, Enum):
    """Execution context boundary for prospective evidence isolation."""
    NATURAL_CLIENT = "NATURAL_CLIENT"
    GOVERNANCE_CERTIFICATION = "GOVERNANCE_CERTIFICATION"


# In-Process ContextVar: Default is NATURAL_CLIENT for normal production traffic.
# External HTTP headers have ZERO authority to set or alter this ContextVar.
CURRENT_EXECUTION_CONTEXT: contextvars.ContextVar[ExecutionContext] = contextvars.ContextVar(
    "current_execution_context", default=ExecutionContext.NATURAL_CLIENT
)


@contextmanager
def governance_execution_context(context: ExecutionContext):
    """Guarantees strict token-based lifecycle management across all execution paths."""
    token = CURRENT_EXECUTION_CONTEXT.set(context)
    try:
        yield
    finally:
        CURRENT_EXECUTION_CONTEXT.reset(token)


class PassiveCaptureHook:
    """Passively captures natural production recommendations into the governance ledger."""

    EPOCH_ID = ExperimentLedger.EPOCH_ID
    EPOCH_START_UTC = ExperimentLedger.EPOCH_START_UTC
    DECISION_ENGINE_SHA = ExperimentLedger.DECISION_ENGINE_SHA
    CONFIG_HASH = ExperimentLedger.CONFIG_HASH

    @classmethod
    def get_observation_governance_sha(cls) -> str:
        """Retrieves observation governance SHA from ledger."""
        return ExperimentLedger.get_observation_governance_sha()

    @classmethod
    def is_temporal_gate_satisfied(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> bool:
        """Evaluates whether prospective observation is authorized for the active Epoch.
        For Epoch 3 or Epoch 2 (PRE_ACTIVATION), requires an authoritative activation record in production.
        """
        if cls.EPOCH_ID == "ARX_PROSPECTIVE_VALIDATION_EPOCH_3":
            return ExperimentLedger.is_epoch3_observation_authorized(
                activation_record_path=activation_record_path,
                db_path=db_path,
            )
        if cls.EPOCH_ID == "ARX_PROSPECTIVE_VALIDATION_EPOCH_2":
            return ExperimentLedger.is_epoch2_observation_authorized(
                activation_record_path=activation_record_path,
                db_path=db_path,
            )
        if cls.EPOCH_START_UTC is not None:
            now_utc = datetime.now(timezone.utc).isoformat()
            return now_utc >= cls.EPOCH_START_UTC
        return False

    @classmethod
    def compute_sha256(cls, payload: Any) -> str:
        """Deterministic SHA-256 for snapshot payloads."""
        if payload is None:
            return ""
        if isinstance(payload, str) and len(payload) == 64 and all(c in "0123456789abcdefABCDEF" for c in payload):
            return payload.lower()
        try:
            encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()
        except Exception:
            return ""

    @classmethod
    def record_natural_recommendation(
        cls,
        symbol: str,
        current_price: float,
        optimal_execution_plan: Dict[str, Any],
        confluence_output: Dict[str, Any],
        technicals: Dict[str, Any],
        factor_scores: Dict[str, Any],
        macro_inputs: Optional[Dict[str, Any]],
        observed_at: Optional[str] = None,
        fetched_at: Optional[str] = None,
        freshness_status: str = "END_OF_DAY",
        provider_source: str = "YAHOO_AUTHENTIC",
        candles: Optional[list] = None,
        ledger_path: Optional[str] = None,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
        live_spot_price: Optional[float] = None,
        market_price_state: Optional[Dict[str, Any]] = None,
        runtime_release_sha: Optional[str] = None,
        runtime_deployment_id: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
    ) -> Optional[Dict[str, Any]]:
        """Passively captures a single natural production recommendation.

        CANONICAL WRITE-TIME SEQUENCE:
        1. Determine execution context (prohibit CERTIFICATION, TEST, REPLAY, SIMULATION)
        2. Verify production runtime identity & deployment-scoped authorization predicate
        3. Verify active epoch boundary
        4. Verify natural recommendation eligibility & finite prices
        5. Verify anti-lookahead temporal integrity
        6. Deduplicate before write
        7. ONLY THEN write prospective record to ledger

        Fail-closed: Returns the captured record on success, or None on failure/quarantine.
        Never raises exceptions to callers.
        Zero side-effects on capital, orders, or broker connections.
        """
        try:
            # 1. EXECUTION CONTEXT & SYNTHETIC ENVIRONMENT GATE
            effective_context = execution_context or CURRENT_EXECUTION_CONTEXT.get()
            if effective_context != ExecutionContext.NATURAL_CLIENT:
                logger.debug(
                    f"[PASSIVE_CAPTURE] Suppressed write: Execution context is {effective_context} for {symbol}"
                )
                return None

            # Environmental prohibitions
            if (
                os.getenv("ARX_TEST_MODE") == "1"
                or os.getenv("ARX_REPLAY_MODE") == "1"
                or os.getenv("ARX_SIMULATION_MODE") == "1"
                or os.getenv("ARX_CERTIFICATION_MODE") == "1"
            ):
                logger.debug(f"[PASSIVE_CAPTURE] Suppressed write: Synthetic/test/replay mode active for {symbol}")
                return None

            # Test runner safeguard: Never contaminate default production ledger
            if ledger_path is None and "PYTEST_CURRENT_TEST" in os.environ:
                logger.debug("[PASSIVE_CAPTURE] Suppressed write: Pytest runner targeting default production ledger.")
                return None

            # Payload-level synthetic/simulation/certification prohibition
            if (
                optimal_execution_plan.get("isSynthetic")
                or optimal_execution_plan.get("isSimulated")
                or optimal_execution_plan.get("isCertification")
                or optimal_execution_plan.get("isReplay")
                or optimal_execution_plan.get("isTest")
                or (macro_inputs or {}).get("isSynthetic")
                or (macro_inputs or {}).get("isCertification")
                or factor_scores.get("isSynthetic")
                or factor_scores.get("isCertification")
                or provider_source in ("SYNTHETIC_MOCK", "SYNTHETIC_FALLBACK", "SIMULATION", "REPLAY", "CERTIFICATION")
            ):
                logger.warning(f"[PASSIVE_CAPTURE] Suppressed write: Non-natural synthetic/certification payload for {symbol}")
                return None

            # 2. RUNTIME IDENTITY & DEPLOYMENT-SCOPED AUTHORIZATION PREDICATE GATE
            current_release = runtime_release_sha or os.getenv("RAILWAY_GIT_COMMIT_SHA")
            current_deployment = runtime_deployment_id or os.getenv("RAILWAY_DEPLOYMENT_ID")
            from analyst_dashboard.governance.storage import is_production_runtime
            from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine

            if is_production_runtime() or (ledger_path is None and db_path is None) or db_path is not None or runtime_release_sha is not None or runtime_deployment_id is not None:
                gov_db = GovernanceDatabaseEngine(db_path=db_path)
                auth_ok, auth_reason = gov_db.evaluate_capture_authorization_predicate(
                    epoch_id=cls.EPOCH_ID,
                    release_sha=current_release,
                    deployment_id=current_deployment,
                    now_utc=datetime.now(timezone.utc).isoformat(),
                )
                if not auth_ok:
                    logger.warning(
                        f"[PASSIVE_CAPTURE] Suppressed write: Runtime authorization check failed ({auth_reason}) for {symbol}"
                    )
                    return None

            # 3. ACTIVE EPOCH BOUNDARY GATE
            if (ledger_path is None or activation_record_path is not None or db_path is not None) and not cls.is_temporal_gate_satisfied(
                activation_record_path=activation_record_path, db_path=db_path
            ):
                logger.warning(f"[PASSIVE_CAPTURE] Suppressed write: Epoch not active for symbol {symbol}")
                return None

            # 4. NATURAL RECOMMENDATION ELIGIBILITY & FINITE PRICE FIREWALL
            if current_price is None:
                logger.warning(f"[PASSIVE_CAPTURE] SUPPRESSED_NONFINITE_INPUT: current_price=None for symbol {symbol}")
                return None
            try:
                cp_float = float(current_price)
                if not math.isfinite(cp_float) or cp_float <= 0.0:
                    logger.warning(f"[PASSIVE_CAPTURE] SUPPRESSED_NONFINITE_INPUT: current_price={current_price} for symbol {symbol}")
                    return None
            except (ValueError, TypeError):
                logger.warning(f"[PASSIVE_CAPTURE] SUPPRESSED_NONFINITE_INPUT: current_price={current_price} invalid float for symbol {symbol}")
                return None

            try:
                entry_price = float(
                    optimal_execution_plan.get("optimal_entry_min")
                    or current_price
                    or 0.0
                )
                if not math.isfinite(entry_price) or entry_price <= 0.0:
                    logger.warning(f"[PASSIVE_CAPTURE] SUPPRESSED_NONFINITE_INPUT: entry_price={entry_price} for symbol {symbol}")
                    return None
            except (ValueError, TypeError):
                return None

            for level_key in ("stop_loss", "take_profit_1", "take_profit_2"):
                val = optimal_execution_plan.get(level_key)
                if val is not None:
                    try:
                        f_val = float(val)
                        if not math.isfinite(f_val) or f_val <= 0.0:
                            logger.warning(f"[PASSIVE_CAPTURE] SUPPRESSED_NONFINITE_INPUT: {level_key}={val} for symbol {symbol}")
                            return None
                    except (ValueError, TypeError):
                        return None

            now_dt = datetime.now(timezone.utc)

            def _to_iso(ts_val: Any) -> Optional[str]:
                if ts_val is None:
                    return None
                if isinstance(ts_val, (int, float)):
                    sec = ts_val / 1000.0 if ts_val > 1e11 else float(ts_val)
                    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()
                s = str(ts_val).strip()
                return s if s else None

            rec_iso = _to_iso(fetched_at) or now_dt.isoformat()
            market_iso = _to_iso(observed_at) or rec_iso
            sig_date = rec_iso[:10] if len(rec_iso) >= 10 else now_dt.strftime("%Y-%m-%d")

            upper_sym = symbol.upper().strip()

            # 1. Content-addressed Market Snapshot
            candle_list = candles or []
            candle_summary = [
                {"d": c.get("date") or c.get("Date"), "c": c.get("close") or c.get("Close"), "v": c.get("volume") or c.get("Volume")}
                for c in candle_list[-50:]
            ] if candle_list else []
            market_snapshot_hash = cls.compute_sha256(candle_summary) if candle_summary else cls.compute_sha256({"price": current_price, "obs": market_iso})

            # 2. Content-addressed Fundamental Snapshot
            fundamental_as_of = factor_scores.get("as_of_date") or factor_scores.get("asOfDate") or ""
            fundamental_filing_ts = _to_iso(factor_scores.get("filing_timestamp") or factor_scores.get("filingTimestamp")) or ""
            fundamental_snapshot_hash = cls.compute_sha256(factor_scores) if factor_scores else ""

            # 3. Content-addressed Macro Snapshot
            macro_data = macro_inputs or {}
            macro_obs_at = _to_iso(
                macro_data.get("macro_observation_available_at")
                or macro_data.get("yield_observation_timestamp")
                or macro_data.get("macroObservationAvailableAt")
            ) or ""
            macro_snapshot_hash = macro_data.get("raw_payload_hash") or (cls.compute_sha256(macro_data) if macro_data else "")
            yc_10y2y = macro_data.get("yield_curve_10y2y")
            cr_spread = macro_data.get("high_yield_credit_spread") if macro_data.get("high_yield_credit_spread") is not None else macro_data.get("credit_spread")

            # 4. Anti-Lookahead Temporal Integrity Verification
            # Invariant: source_available_at <= recommended_at for every domain
            rec_dt = ExperimentLedger._parse_utc_timestamp(rec_iso)
            if rec_dt:
                if market_iso:
                    obs_dt = ExperimentLedger._parse_utc_timestamp(market_iso)
                    if obs_dt and obs_dt > rec_dt:
                        logger.warning(f"Anti-lookahead violation: market observedAt {market_iso} > rec {rec_iso}")
                        return None
                if macro_obs_at:
                    macro_dt = ExperimentLedger._parse_utc_timestamp(macro_obs_at)
                    if macro_dt and macro_dt > rec_dt:
                        logger.warning(f"Anti-lookahead violation: macro observedAt {macro_obs_at} > rec {rec_iso}")
                        return None
                if fundamental_filing_ts:
                    filing_dt = ExperimentLedger._parse_utc_timestamp(fundamental_filing_ts)
                    if filing_dt and filing_dt > rec_dt:
                        logger.warning(f"Anti-lookahead violation: fundamental filing {fundamental_filing_ts} > rec {rec_iso}")
                        return None

            # 5. Assemble Inputs Metadata
            inputs_meta = {
                "market_regime": confluence_output.get("market_regime", "BULL"),
                "sector": confluence_output.get("sector", "EQUITY"),
                "asset_class": "US_EQUITY",
                "marketDataSnapshotTimestamp": market_iso,
                "marketSnapshotObservedAt": market_iso,
                "marketSnapshotHash": market_snapshot_hash,
                "candleCount": len(candle_list),
                "candle_count": len(candle_list),
                "sma50": technicals.get("sma_50"),
                "ema20": technicals.get("ema_20"),
                "rsi14": technicals.get("rsi_14"),
                "atr14": technicals.get("atr_14"),
                "fundamentalAsOfDate": str(fundamental_as_of),
                "fundamentalFilingTimestamp": str(fundamental_filing_ts),
                "fundamentalSnapshotHash": fundamental_snapshot_hash,
                "macroObservationDate": str(macro_obs_at),
                "macroObservationAvailableAt": str(macro_obs_at),
                "macroSnapshotHash": macro_snapshot_hash,
                "yieldCurve10y2y": yc_10y2y,
                "yield_curve_10y2y": yc_10y2y,
                "creditSpread": cr_spread,
                "credit_spread": cr_spread,
                "dataProvider": provider_source,
                "quoteFreshness": freshness_status,
                "evidenceCompleteness": "COMPLETE" if confluence_output.get("overall_eligibility") == "FULL" else "PARTIAL",
                "modelConfigHash": cls.CONFIG_HASH,
                "pointInTimePrecision": "TIMESTAMP",
                "analysisReferencePrice": current_price,
                "analysisReferenceSource": "COMPLETED_SESSION",
                "liveSpotPrice": live_spot_price,
                "liveObservedAt": (market_price_state or {}).get("liveObservedAt"),
                "liveSource": (market_price_state or {}).get("liveSource", "UNAVAILABLE"),
                "liveFreshness": (market_price_state or {}).get("liveFreshness", "UNAVAILABLE"),
                "marketSession": (market_price_state or {}).get("marketSession", "UNKNOWN"),
                "rawMarketPayload": candle_summary if candle_summary else None,
                "rawFundamentalPayload": factor_scores if factor_scores else None,
                "rawMacroPayload": macro_data if macro_data else None,
                "recommended_at": rec_iso,
                "signalTimestamp": rec_iso,
            }

            component_scores = {
                "qualityScore": factor_scores.get("quality_score"),
                "growthScore": factor_scores.get("growth_score"),
                "valuationScore": factor_scores.get("valuation_score"),
                "technicalScore": technicals.get("technical_score") or (technicals.get("score") if isinstance(technicals.get("score"), (int, float)) else None),
                "smartMoneyScore": None,
                "macroScore": confluence_output.get("macro_score"),
                "catalystScore": None,
            }

            # 6. Deduplication Check BEFORE Physical Ledger Mutation
            ledger = ExperimentLedger.load_ledger(ledger_path)
            sig_id = f"{upper_sym}_{sig_date}"
            existing = next((s for s in ledger.get("signals", []) if s.get("signalId") == sig_id), None)
            if existing:
                logger.info(
                    f"[PASSIVE_CAPTURE] Deduplicated natural recommendation: {sig_id} already exists. Zero ledger mutation."
                )
                return existing

            # 7. Immutable Registration into Governance Ledger (ONLY AFTER all gates pass)
            conf_val = confluence_output.get("confluenceScore") if confluence_output.get("confluenceScore") is not None else confluence_output.get("overall_score", 0.0)
            record = ExperimentLedger.register_signal(
                symbol=upper_sym,
                entry_price=entry_price,
                opt_exec=optimal_execution_plan,
                confluence_score=float(conf_val or 0.0),
                inputs_meta=inputs_meta,
                engine_commit=cls.DECISION_ENGINE_SHA,
                engine_tag=getattr(ExperimentLedger, "FROZEN_ENGINE_TAG", "v2.5.0-live-dual-price-freeze"),
                ledger_path=ledger_path,
                signal_date=sig_date,
                component_scores=component_scores,
                epoch_id=cls.EPOCH_ID,
                provenance_cohort=ProvenanceCohort.PROSPECTIVE_CLEAN,
                release_sha=current_release,
                deployment_id=current_deployment,
                capture_source="NATURAL_PRODUCTION_API",
                analysis_reference_price=current_price,
                analysis_reference_source="COMPLETED_SESSION",
                live_spot_price=live_spot_price,
                live_observed_at=(market_price_state or {}).get("liveObservedAt"),
                live_source=(market_price_state or {}).get("liveSource", "UNAVAILABLE"),
                live_freshness=(market_price_state or {}).get("liveFreshness", "UNAVAILABLE"),
                market_session=(market_price_state or {}).get("marketSession", "UNKNOWN"),
            )

            # Ensure dual-SHA identity and dual-price attributes are explicitly annotated on the record
            record["decisionEngineSha"] = cls.DECISION_ENGINE_SHA
            record["observationGovernanceSha"] = cls.get_observation_governance_sha()
            record["recommended_at"] = rec_iso
            record["signalTimestamp"] = rec_iso
            record["analysisReferencePrice"] = current_price
            record["analysisReferenceSource"] = "COMPLETED_SESSION"
            record["liveSpotPrice"] = live_spot_price
            record["liveObservedAt"] = (market_price_state or {}).get("liveObservedAt")
            record["liveSource"] = (market_price_state or {}).get("liveSource", "UNAVAILABLE")
            record["liveFreshness"] = (market_price_state or {}).get("liveFreshness", "UNAVAILABLE")
            record["marketSession"] = (market_price_state or {}).get("marketSession", "UNKNOWN")

            # 7. Cohort Classification & Integrity Validation
            cohort = ExperimentLedger.classify_provenance_cohort(record)
            record["provenanceCohort"] = cohort

            logger.info(
                f"[PASSIVE_CAPTURE] Captured natural recommendation {record.get('signalId')} "
                f"symbol={upper_sym} cohort={cohort} epoch={cls.EPOCH_ID}"
            )
            return record

        except Exception as e:
            logger.error(f"[PASSIVE_CAPTURE] Fail-closed: capture error for symbol {symbol}: {e}", exc_info=True)
            return None
