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
from typing import Dict, Any, Optional, Tuple

from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)

logger = logging.getLogger("arx.governance.passive_capture")


# ==============================================================================
# EXECUTION LADDER REMEDIATION — PROSPECTIVE VALIDATION EPOCH 001 AUTHORITIES
# ==============================================================================
EXECUTION_LADDER_EPOCH_ID = "EXECUTION_LADDER_PROSPECTIVE_EPOCH_001"
EXECUTION_LADDER_OBSERVATION_STREAM = "EXECUTION_LADDER_PLANS"
EXECUTION_LADDER_AUTHORITY_SHA = "7bcb7780221f58cf596dabce484d83276e0a3c50"

# Centralized Ratified Status Set (Derived from OptimalExecutionEngine Authority)
# Strictly entry-readiness states for un-entered plans (NO_ACTIVE_POSITION).
# Active-position target-progress states (APPROACHING_TARGET, TARGET_REACHED) and aliases (READY_TO_BUY) are excluded.
from analyst_dashboard.analyzers.optimal_execution import (
    ACTIONABLE_EXECUTION_STATUSES,
    NON_ACTIONABLE_EXECUTION_STATUSES,
)

RATIFIED_EXECUTION_LADDER_STATUSES = frozenset({
    "IN_BUY_ZONE",
    "IN_BUY_ZONE_AWAITING_TRIGGER",
    "EXTENDED_ABOVE_BUY_ZONE",
    "WAITING_PULLBACK",
    "STOPPED_OUT",
})
assert RATIFIED_EXECUTION_LADDER_STATUSES.issubset(
    ACTIONABLE_EXECUTION_STATUSES | NON_ACTIONABLE_EXECUTION_STATUSES
)


def resolve_release_sha(override_sha: Optional[str] = None) -> Optional[str]:
    """Resolves authoritative runtime/build release SHA with fail-closed provenance semantics.

    Checks in priority order:
    1. Explicit override_sha (if provided and non-empty)
    2. ARX_RELEASE_SHA
    3. ARX_RELEASE
    4. NEXT_PUBLIC_ARX_RELEASE
    5. RAILWAY_GIT_COMMIT_SHA

    Returns None if missing (fail-closed, never fabricates placeholders).
    """
    if override_sha and str(override_sha).strip():
        return str(override_sha).strip()
    for env_k in ("ARX_RELEASE_SHA", "ARX_RELEASE", "NEXT_PUBLIC_ARX_RELEASE", "RAILWAY_GIT_COMMIT_SHA"):
        val = os.getenv(env_k)
        if val and val.strip():
            return val.strip()
    return None


def compute_execution_ladder_plan_id(snapshot: Dict[str, Any]) -> str:
    """Computes deterministic 24-character hex plan_id for an immutable execution ladder snapshot.

    Guarantees cross-request stability across page refreshes, multiple consumers, and surfaces
    by hashing canonical plan levels, symbol, role, epoch, release, authority, and source trading date.
    Volatile subsecond request timestamps are excluded from the hash preimage.
    """
    source_ts = str(snapshot.get("source_data_timestamp") or snapshot.get("generation_timestamp") or "")
    date_bucket = source_ts[:10]  # Canonical trading date (YYYY-MM-DD)
    elements = [
        str(snapshot.get("epoch_id") or ""),
        str(snapshot.get("symbol") or "").upper().strip(),
        str(snapshot.get("user_role") or "").upper().strip(),
        date_bucket,
        str(snapshot.get("release_sha") or ""),
        str(snapshot.get("execution_ladder_authority_sha") or ""),
        str(snapshot.get("planned_entry") or ""),
        str(snapshot.get("structural_invalidation") or ""),
        str(snapshot.get("take_profit_1") or ""),
        str(snapshot.get("take_profit_2") or ""),
        str(snapshot.get("execution_status") or ""),
    ]
    raw = "|".join(elements)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"PLAN_{h[:24]}"


def build_execution_ladder_snapshot(
    symbol: Optional[str] = None,
    optimal_execution_plan: Optional[Dict[str, Any]] = None,
    current_price: Optional[float] = None,
    user_role: Optional[str] = None,
    instrument_class: Optional[str] = None,
    generation_timestamp: Optional[str] = None,
    source_data_timestamp: Optional[str] = None,
    release_sha: Optional[str] = None,
    authority_sha: Optional[str] = None,
    live_spot_price: Optional[float] = None,
    is_actionable: Optional[bool] = None,
    snapshot_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Constructs and strictly validates an immutable execution ladder snapshot.

    Returns (snapshot_dict, None) on success, or (None, rejection_reason) on failure.
    Enforces all 20 mandatory snapshot fields from Section 6.
    """
    s_dict = snapshot_dict or {}
    opt_plan = optimal_execution_plan or {}

    # 1. symbol
    sym = s_dict.get("symbol") if "symbol" in s_dict else symbol
    if not sym or not str(sym).strip():
        return None, "MISSING_SYMBOL"

    # 2. instrument_class
    iclass = s_dict.get("instrument_class") if "instrument_class" in s_dict else instrument_class
    if not iclass or not str(iclass).strip():
        return None, "MISSING_INSTRUMENT_CLASS"

    # 3. user_role
    role = s_dict.get("user_role") if "user_role" in s_dict else (user_role or opt_plan.get("user_role"))
    if not role or role not in ("DAY_TRADER", "LONG_TERM"):
        return None, "MISSING_USER_ROLE"

    # 4. generation_timestamp
    gts = s_dict.get("generation_timestamp") if "generation_timestamp" in s_dict else generation_timestamp
    if not gts or not str(gts).strip():
        return None, "MISSING_GENERATION_TIMESTAMP"

    # 5. source_data_timestamp
    sdts = s_dict.get("source_data_timestamp") if "source_data_timestamp" in s_dict else source_data_timestamp
    if not sdts or not str(sdts).strip():
        return None, "MISSING_SOURCE_DATA_TIMESTAMP"

    # 6. release_sha
    rsha = s_dict.get("release_sha") if "release_sha" in s_dict else resolve_release_sha(release_sha)
    if not rsha or not str(rsha).strip():
        return None, "MISSING_RELEASE_SHA"

    # 7. execution_ladder_authority_sha
    asha = s_dict.get("execution_ladder_authority_sha") if "execution_ladder_authority_sha" in s_dict else authority_sha
    if not asha or not str(asha).strip():
        return None, "MISSING_AUTHORITY_SHA"
    if str(asha).strip() != EXECUTION_LADDER_AUTHORITY_SHA:
        return None, "INVALID_AUTHORITY_SHA"

    # 8. current_spot
    cspot = s_dict.get("current_spot") if "current_spot" in s_dict else (live_spot_price if live_spot_price is not None else current_price)
    if cspot is None:
        return None, "MISSING_CURRENT_SPOT"
    try:
        cspot_f = float(cspot)
        if cspot_f <= 0.0 or not math.isfinite(cspot_f):
            return None, "MISSING_CURRENT_SPOT"
    except (ValueError, TypeError):
        return None, "MISSING_CURRENT_SPOT"

    # 9. entry_min
    emin = s_dict.get("entry_min") if "entry_min" in s_dict else (opt_plan.get("optimal_entry_min") if opt_plan.get("optimal_entry_min") is not None else opt_plan.get("entry_min"))
    if emin is None:
        return None, "MISSING_ENTRY_MIN"
    try:
        emin_f = float(emin)
        if emin_f <= 0.0 or not math.isfinite(emin_f):
            return None, "MISSING_ENTRY_MIN"
    except (ValueError, TypeError):
        return None, "MISSING_ENTRY_MIN"

    # 10. entry_max
    emax = s_dict.get("entry_max") if "entry_max" in s_dict else (opt_plan.get("optimal_entry_max") if opt_plan.get("optimal_entry_max") is not None else opt_plan.get("entry_max"))
    if emax is None:
        return None, "MISSING_ENTRY_MAX"
    try:
        emax_f = float(emax)
        if emax_f <= 0.0 or not math.isfinite(emax_f):
            return None, "MISSING_ENTRY_MAX"
    except (ValueError, TypeError):
        return None, "MISSING_ENTRY_MAX"

    # 11. planned_entry
    pe = s_dict.get("planned_entry") if "planned_entry" in s_dict else opt_plan.get("planned_entry")
    if pe is None:
        return None, "MISSING_PLANNED_ENTRY"
    try:
        pe_f = float(pe)
        if pe_f <= 0.0 or not math.isfinite(pe_f):
            return None, "MISSING_PLANNED_ENTRY"
    except (ValueError, TypeError):
        return None, "MISSING_PLANNED_ENTRY"

    # 12. structural_invalidation
    si = s_dict.get("structural_invalidation") if "structural_invalidation" in s_dict else opt_plan.get("structural_invalidation")
    if si is None:
        return None, "MISSING_STRUCTURAL_INVALIDATION"
    try:
        si_f = float(si)
        if si_f <= 0.0 or not math.isfinite(si_f):
            return None, "MISSING_STRUCTURAL_INVALIDATION"
    except (ValueError, TypeError):
        return None, "MISSING_STRUCTURAL_INVALIDATION"

    # 13. execution_risk
    er = s_dict.get("execution_risk") if "execution_risk" in s_dict else opt_plan.get("execution_risk")
    if er is None:
        return None, "MISSING_EXECUTION_RISK"
    try:
        er_f = float(er)
        if er_f <= 0.0 or not math.isfinite(er_f):
            return None, "MISSING_EXECUTION_RISK"
    except (ValueError, TypeError):
        return None, "MISSING_EXECUTION_RISK"

    # 14. atr_14
    atr = s_dict.get("atr_14") if "atr_14" in s_dict else opt_plan.get("atr_14")
    if atr is None:
        return None, "MISSING_ATR_14"
    try:
        atr_f = float(atr)
        if atr_f <= 0.0 or not math.isfinite(atr_f):
            return None, "MISSING_ATR_14"
    except (ValueError, TypeError):
        return None, "MISSING_ATR_14"

    # 15. take_profit_1
    tp1 = s_dict.get("take_profit_1") if "take_profit_1" in s_dict else opt_plan.get("take_profit_1")
    if tp1 is None:
        return None, "MISSING_TAKE_PROFIT_1"
    try:
        tp1_f = float(tp1)
        if tp1_f <= 0.0 or not math.isfinite(tp1_f):
            return None, "MISSING_TAKE_PROFIT_1"
    except (ValueError, TypeError):
        return None, "MISSING_TAKE_PROFIT_1"

    # 16. take_profit_2
    tp2 = s_dict.get("take_profit_2") if "take_profit_2" in s_dict else opt_plan.get("take_profit_2")
    if tp2 is None:
        return None, "MISSING_TAKE_PROFIT_2"
    try:
        tp2_f = float(tp2)
        if tp2_f <= 0.0 or not math.isfinite(tp2_f):
            return None, "MISSING_TAKE_PROFIT_2"
    except (ValueError, TypeError):
        return None, "MISSING_TAKE_PROFIT_2"

    # 17. market_location
    mloc = s_dict.get("market_location") if "market_location" in s_dict else opt_plan.get("market_location")
    if not mloc or not str(mloc).strip():
        return None, "MISSING_MARKET_LOCATION"

    # 18. execution_status
    estatus = s_dict.get("execution_status") if "execution_status" in s_dict else opt_plan.get("execution_status")
    if not estatus or not str(estatus).strip():
        return None, "MISSING_EXECUTION_STATUS"
    if str(estatus).strip() not in RATIFIED_EXECUTION_LADDER_STATUSES:
        return None, "UNRATIFIED_EXECUTION_STATUS"

    # 19. is_actionable
    act = s_dict.get("is_actionable") if "is_actionable" in s_dict else is_actionable
    if act is None or not isinstance(act, bool):
        return None, "MISSING_IS_ACTIONABLE"

    # 20. execution_stop_visible
    esv = s_dict.get("execution_stop_visible") if "execution_stop_visible" in s_dict else opt_plan.get("execution_stop_visible")
    if esv is None or not isinstance(esv, bool):
        return None, "MISSING_EXECUTION_STOP_VISIBLE"

    # All fields validated successfully: assemble canonical immutable snapshot
    constructed = {
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": str(sym).upper().strip(),
        "instrument_class": str(iclass),
        "user_role": str(role),
        "generation_timestamp": str(gts),
        "source_data_timestamp": str(sdts),
        "release_sha": str(rsha),
        "execution_ladder_authority_sha": str(asha),
        "current_spot": round(float(cspot_f), 6),
        "entry_min": round(float(emin_f), 6),
        "entry_max": round(float(emax_f), 6),
        "planned_entry": round(float(pe_f), 6),
        "structural_invalidation": round(float(si_f), 6),
        "execution_risk": round(float(er_f), 6),
        "atr_14": round(float(atr_f), 6),
        "take_profit_1": round(float(tp1_f), 6),
        "take_profit_2": round(float(tp2_f), 6),
        "market_location": str(mloc),
        "execution_status": str(estatus),
        "is_actionable": bool(act),
        "execution_stop_visible": bool(esv),
    }
    plan_id = compute_execution_ladder_plan_id(constructed)
    constructed["plan_id"] = plan_id
    return constructed, None


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
        For Epoch 4, Epoch 3, or Epoch 2 (PRE_ACTIVATION), requires an authoritative activation record in production.
        """
        if cls.EPOCH_ID == "ARX_PROSPECTIVE_VALIDATION_EPOCH_4":
            return ExperimentLedger.is_epoch4_observation_authorized(
                activation_record_path=activation_record_path,
                db_path=db_path,
            )
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

    last_admission_result: Optional[Dict[str, Any]] = None

    @classmethod
    def evaluate_prospective_admission(
        cls,
        symbol: str,
        is_actionable: bool = False,
        decision_state: Optional[str] = None,
        market_price_state: Optional[Dict[str, Any]] = None,
        live_spot_price: Optional[float] = None,
        current_price: Optional[float] = None,
        execution_context: Optional[ExecutionContext] = None,
        runtime_release_sha: Optional[str] = None,
        runtime_deployment_id: Optional[str] = None,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
        ledger_path: Optional[str] = None,
        sig_date: Optional[str] = None,
        optimal_execution_plan: Optional[Dict[str, Any]] = None,
        factor_scores: Optional[Dict[str, Any]] = None,
        macro_inputs: Optional[Dict[str, Any]] = None,
        provider_source: Optional[str] = None,
        epoch_id: Optional[str] = None,
        user_role: Optional[str] = None,
        instrument_class: Optional[str] = None,
        generation_timestamp: Optional[str] = None,
        source_data_timestamp: Optional[str] = None,
        execution_ladder_authority_sha: Optional[str] = None,
        snapshot_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluates whether an observation meets all strict prospective admission criteria.

        Returns a dict with 'prospectiveCaptureEligible' and 'prospectiveCaptureRejectionReason'.
        Rejection reasons (in canonical priority order):
        - NON_NATURAL_CONTEXT
        - RELEASE_NOT_CERTIFIED / DEPLOYMENT_NOT_AUTHORIZED
        - EPOCH_NOT_ACTIVATED
        - DECISION_NOT_ACTIONABLE
        - QUOTE_NOT_REALTIME
        - MARKET_SESSION_NOT_REGULAR
        - LIVE_SPOT_INVALID
        - DUPLICATE
        """
        # ==============================================================================
        # EPOCH 001 OBSERVATIONAL EXECUTION LADDER ADMISSION PATH
        # ==============================================================================
        target_epoch = epoch_id or (snapshot_dict or {}).get("epoch_id") or cls.EPOCH_ID
        if target_epoch == EXECUTION_LADDER_EPOCH_ID:
            # 1. EXECUTION CONTEXT & SYNTHETIC ENVIRONMENT GATE
            effective_context = execution_context or CURRENT_EXECUTION_CONTEXT.get()
            if effective_context != ExecutionContext.NATURAL_CLIENT:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
                }

            if (
                os.getenv("ARX_TEST_MODE") == "1"
                or os.getenv("ARX_REPLAY_MODE") == "1"
                or os.getenv("ARX_SIMULATION_MODE") == "1"
                or os.getenv("ARX_CERTIFICATION_MODE") == "1"
            ):
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
                }

            if ledger_path is None and db_path is None and "PYTEST_CURRENT_TEST" in os.environ:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
                }

            opt_plan = optimal_execution_plan or (snapshot_dict or {})
            macro = macro_inputs or {}
            factors = factor_scores or {}
            if (
                opt_plan.get("isSynthetic")
                or opt_plan.get("isSimulated")
                or opt_plan.get("isCertification")
                or opt_plan.get("isReplay")
                or opt_plan.get("isTest")
                or macro.get("isSynthetic")
                or macro.get("isCertification")
                or factors.get("isSynthetic")
                or factors.get("isCertification")
                or provider_source in ("SYNTHETIC_MOCK", "SYNTHETIC_FALLBACK", "SIMULATION", "REPLAY", "CERTIFICATION")
            ):
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
                }

            # 2. RESOLVE RELEASE SHA (FAIL-CLOSED)
            rel_sha = resolve_release_sha(runtime_release_sha or (snapshot_dict or {}).get("release_sha"))
            if not rel_sha:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "MISSING_RELEASE_SHA",
                }

            # 3. AUTHORITY SHA (FAIL-CLOSED)
            auth_sha = execution_ladder_authority_sha or (snapshot_dict or {}).get("execution_ladder_authority_sha")
            if not auth_sha:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "MISSING_AUTHORITY_SHA",
                }
            if str(auth_sha).strip() != EXECUTION_LADDER_AUTHORITY_SHA:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "INVALID_AUTHORITY_SHA",
                }

            # 4. RATIFIED STATUS & SNAPSHOT COMPLETENESS
            now_iso = datetime.now(timezone.utc).isoformat()
            plan_snapshot, missing_reason = build_execution_ladder_snapshot(
                symbol=symbol,
                optimal_execution_plan=optimal_execution_plan,
                current_price=current_price,
                user_role=user_role or opt_plan.get("user_role") or (snapshot_dict or {}).get("user_role") or "LONG_TERM",
                instrument_class=instrument_class or (snapshot_dict or {}).get("instrument_class") or "EQUITY",
                generation_timestamp=generation_timestamp or (snapshot_dict or {}).get("generation_timestamp") or now_iso,
                source_data_timestamp=source_data_timestamp or (snapshot_dict or {}).get("source_data_timestamp") or (market_price_state or {}).get("liveObservedAt") or now_iso,
                release_sha=rel_sha,
                authority_sha=auth_sha,
                live_spot_price=live_spot_price if live_spot_price is not None else (snapshot_dict or {}).get("current_spot"),
                is_actionable=is_actionable if is_actionable is not None else (snapshot_dict or {}).get("is_actionable"),
                snapshot_dict=snapshot_dict,
            )
            if not plan_snapshot:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": missing_reason,
                }

            # 5. DEDUPLICATION GATE (IDEMPOTENT BY DETERMINISTIC PLAN_ID)
            plan_id = plan_snapshot["plan_id"]
            from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
            gov_db = GovernanceDatabaseEngine(db_path=db_path)
            existing = gov_db.get_execution_ladder_plan(plan_id)
            if existing:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "DUPLICATE",
                    "plan_id": plan_id,
                    "existingRecord": existing,
                }

            return {
                "prospectiveCaptureEligible": True,
                "prospectiveCaptureRejectionReason": None,
                "plan_id": plan_id,
                "observationStream": EXECUTION_LADDER_OBSERVATION_STREAM,
                "snapshot": plan_snapshot,
            }

        # ==============================================================================
        # GENERIC / LEGACY PROSPECTIVE ADMISSION GATES (PRESERVED UNCHANGED)
        # ==============================================================================
        # 1. EXECUTION CONTEXT & SYNTHETIC ENVIRONMENT GATE
        effective_context = execution_context or CURRENT_EXECUTION_CONTEXT.get()
        if effective_context != ExecutionContext.NATURAL_CLIENT:
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
            }

        if (
            os.getenv("ARX_TEST_MODE") == "1"
            or os.getenv("ARX_REPLAY_MODE") == "1"
            or os.getenv("ARX_SIMULATION_MODE") == "1"
            or os.getenv("ARX_CERTIFICATION_MODE") == "1"
        ):
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
            }

        if ledger_path is None and "PYTEST_CURRENT_TEST" in os.environ:
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
            }

        opt_exec = optimal_execution_plan or {}
        macro = macro_inputs or {}
        factors = factor_scores or {}
        if (
            opt_exec.get("isSynthetic")
            or opt_exec.get("isSimulated")
            or opt_exec.get("isCertification")
            or opt_exec.get("isReplay")
            or opt_exec.get("isTest")
            or macro.get("isSynthetic")
            or macro.get("isCertification")
            or factors.get("isSynthetic")
            or factors.get("isCertification")
            or provider_source in ("SYNTHETIC_MOCK", "SYNTHETIC_FALLBACK", "SIMULATION", "REPLAY", "CERTIFICATION")
        ):
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "NON_NATURAL_CONTEXT",
            }

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
                rejection = "DEPLOYMENT_NOT_AUTHORIZED"
                if auth_reason and "CERTIFICATION" in auth_reason:
                    rejection = "RELEASE_NOT_CERTIFIED"
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": rejection,
                    "detail": auth_reason,
                }

        # 3. ACTIVE EPOCH BOUNDARY GATE
        if (ledger_path is None or activation_record_path is not None or db_path is not None) and not cls.is_temporal_gate_satisfied(
            activation_record_path=activation_record_path, db_path=db_path
        ):
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "EPOCH_NOT_ACTIVATED",
            }

        # 4. CANONICAL DECISION ACTIONABILITY GATE
        # Must require canonical DecisionHierarchy result (isActionable == True, state == ACTIONABLE_SETUP)
        if not is_actionable or (decision_state is not None and decision_state != "ACTIONABLE_SETUP"):
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "DECISION_NOT_ACTIONABLE",
            }

        # 5. MARKET DATA FRESHNESS GATE (Dual-Price Contract)
        mps = market_price_state or {}
        freshness = mps.get("liveFreshness")
        if freshness != "REALTIME":
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "QUOTE_NOT_REALTIME",
            }

        # 6. MARKET SESSION GATE (Dual-Price Contract)
        session = mps.get("marketSession")
        if session != "REGULAR_SESSION":
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "MARKET_SESSION_NOT_REGULAR",
            }

        # 7. PRICE VALIDITY GATE (Live Spot Price & Execution Corridor)
        if live_spot_price is None:
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
            }
        try:
            lsp_float = float(live_spot_price)
            if not math.isfinite(lsp_float) or lsp_float <= 0.0:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
                }
        except (ValueError, TypeError):
            return {
                "prospectiveCaptureEligible": False,
                "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
            }

        if current_price is not None:
            try:
                cp_float = float(current_price)
                if not math.isfinite(cp_float) or cp_float <= 0.0:
                    return {
                        "prospectiveCaptureEligible": False,
                        "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
                    }
            except (ValueError, TypeError):
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
                }

        if opt_exec:
            try:
                entry_p = float(opt_exec.get("optimal_entry_min") or current_price or 0.0)
                if not math.isfinite(entry_p) or entry_p <= 0.0:
                    return {
                        "prospectiveCaptureEligible": False,
                        "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
                    }
            except (ValueError, TypeError):
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
                }
            for level_k in ("stop_loss", "take_profit_1", "take_profit_2"):
                v = opt_exec.get(level_k)
                if v is not None:
                    try:
                        fv = float(v)
                        if not math.isfinite(fv) or fv <= 0.0:
                            return {
                                "prospectiveCaptureEligible": False,
                                "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
                            }
                    except (ValueError, TypeError):
                        return {
                            "prospectiveCaptureEligible": False,
                            "prospectiveCaptureRejectionReason": "LIVE_SPOT_INVALID",
                        }

        # 8. DEDUPLICATION GATE
        if sig_date and symbol:
            ledger = ExperimentLedger.load_ledger(ledger_path)
            sig_id = f"{symbol.upper().strip()}_{sig_date}"
            existing = next((s for s in ledger.get("signals", []) if s.get("signalId") == sig_id), None)
            if existing:
                return {
                    "prospectiveCaptureEligible": False,
                    "prospectiveCaptureRejectionReason": "DUPLICATE",
                    "existingRecord": existing,
                }

        return {
            "prospectiveCaptureEligible": True,
            "prospectiveCaptureRejectionReason": None,
        }

    @classmethod
    def record_execution_ladder_plan(
        cls,
        symbol: str,
        optimal_execution_plan: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None,
        user_role: str = "LONG_TERM",
        instrument_class: str = "EQUITY",
        generation_timestamp: Optional[str] = None,
        source_data_timestamp: Optional[str] = None,
        release_sha: Optional[str] = None,
        authority_sha: Optional[str] = None,
        live_spot_price: Optional[float] = None,
        is_actionable: Optional[bool] = None,
        db_path: Optional[str] = None,
        ledger_path: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
        snapshot_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Passively captures an immutable execution ladder prospective plan snapshot into governance.db.

        Fail-closed: Returns captured snapshot dict on success, None on rejection/error.
        Idempotent: Duplicate plan_id returns existing record without denominator increment.
        Never blocks live API responses or mutates trading behavior.
        """
        try:
            admission = cls.evaluate_prospective_admission(
                symbol=symbol,
                is_actionable=is_actionable if is_actionable is not None else False,
                live_spot_price=live_spot_price,
                current_price=current_price,
                execution_context=execution_context,
                runtime_release_sha=release_sha,
                db_path=db_path,
                ledger_path=ledger_path,
                optimal_execution_plan=optimal_execution_plan,
                epoch_id=EXECUTION_LADDER_EPOCH_ID,
                user_role=user_role,
                instrument_class=instrument_class,
                generation_timestamp=generation_timestamp,
                source_data_timestamp=source_data_timestamp,
                execution_ladder_authority_sha=authority_sha or EXECUTION_LADDER_AUTHORITY_SHA,
                snapshot_dict=snapshot_dict,
            )
            cls.last_admission_result = admission

            if not admission.get("prospectiveCaptureEligible"):
                rejection = admission.get("prospectiveCaptureRejectionReason")
                if rejection == "DUPLICATE":
                    logger.info(
                        f"[PASSIVE_CAPTURE] Deduplicated execution ladder plan {admission.get('plan_id')}. Zero denominator increment."
                    )
                    return admission.get("existingRecord")
                logger.warning(
                    f"[PASSIVE_CAPTURE] Suppressed execution ladder plan: {rejection} for symbol {symbol}"
                )
                return None

            snapshot = admission["snapshot"]
            from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
            gov_db = GovernanceDatabaseEngine(db_path=db_path)
            inserted = gov_db.insert_execution_ladder_plan(snapshot)
            if not inserted:
                # Concurrent or existing insertion
                return gov_db.get_execution_ladder_plan(snapshot["plan_id"])

            try:
                ExperimentLedger.record_execution_ladder_plan_snapshot(snapshot, ledger_path=ledger_path)
            except Exception as e:
                logger.warning(f"[PASSIVE_CAPTURE] Non-blocking ledger sync failure: {e}")

            logger.info(
                f"[PASSIVE_CAPTURE] Admitted execution ladder plan {snapshot['plan_id']} "
                f"symbol={symbol} status={snapshot['execution_status']} role={snapshot['user_role']}"
            )
            return snapshot
        except Exception as e:
            logger.error(
                f"[PASSIVE_CAPTURE] Fail-closed: execution ladder capture error for symbol {symbol}: {e}",
                exc_info=True,
            )
            return None

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
        is_actionable: bool = False,
        decision_state: Optional[str] = None,
        epoch_id: Optional[str] = None,
        user_role: Optional[str] = None,
        instrument_class: Optional[str] = None,
        execution_ladder_authority_sha: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Passively captures a single natural production recommendation.

        CANONICAL WRITE-TIME SEQUENCE:
        1. Evaluate prospective admission criteria (context, authorization, epoch, actionability, freshness, session, prices, deduplication)
        2. Verify anti-lookahead temporal integrity
        3. ONLY THEN write prospective record to ledger

        Fail-closed: Returns the captured record on success, or None on failure/quarantine.
        Never raises exceptions to callers.
        Zero side-effects on capital, orders, or broker connections.
        """
        if epoch_id == EXECUTION_LADDER_EPOCH_ID:
            return cls.record_execution_ladder_plan(
                symbol=symbol,
                optimal_execution_plan=optimal_execution_plan,
                current_price=current_price,
                user_role=user_role or (optimal_execution_plan or {}).get("user_role") or "LONG_TERM",
                instrument_class=instrument_class or "EQUITY",
                generation_timestamp=fetched_at,
                source_data_timestamp=observed_at or (market_price_state or {}).get("liveObservedAt"),
                release_sha=runtime_release_sha,
                authority_sha=execution_ladder_authority_sha,
                live_spot_price=live_spot_price,
                is_actionable=is_actionable,
                db_path=db_path,
                ledger_path=ledger_path,
                execution_context=execution_context,
            )

        try:
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
            current_release = runtime_release_sha or os.getenv("RAILWAY_GIT_COMMIT_SHA")
            current_deployment = runtime_deployment_id or os.getenv("RAILWAY_DEPLOYMENT_ID")

            admission = cls.evaluate_prospective_admission(
                symbol=upper_sym,
                is_actionable=is_actionable,
                decision_state=decision_state,
                market_price_state=market_price_state,
                live_spot_price=live_spot_price,
                current_price=current_price,
                execution_context=execution_context,
                runtime_release_sha=runtime_release_sha,
                runtime_deployment_id=runtime_deployment_id,
                activation_record_path=activation_record_path,
                db_path=db_path,
                ledger_path=ledger_path,
                sig_date=sig_date,
                optimal_execution_plan=optimal_execution_plan,
                factor_scores=factor_scores,
                macro_inputs=macro_inputs,
                provider_source=provider_source,
            )
            cls.last_admission_result = admission

            if not admission["prospectiveCaptureEligible"]:
                rejection = admission.get("prospectiveCaptureRejectionReason")
                if rejection == "DUPLICATE":
                    logger.info(
                        f"[PASSIVE_CAPTURE] Deduplicated natural recommendation: {upper_sym}_{sig_date} already exists. Zero ledger mutation."
                    )
                    return admission.get("existingRecord")
                logger.warning(
                    f"[PASSIVE_CAPTURE] Suppressed write: {rejection} for symbol {upper_sym}"
                )
                return None

            entry_price = float(
                optimal_execution_plan.get("optimal_entry_min")
                or current_price
                or 0.0
            )

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
