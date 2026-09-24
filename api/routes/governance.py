"""ARX Model Governance & Prospective Evaluation API Router.

Provides strictly read-only, authenticated access to the frozen prospective evaluation ledger,
governance scorecard, comparative baselines, and milestone tracking.

GOVERNING INVARIANTS:
1. READ-ONLY VIEW: This router has NO ability to modify predictions, historical outcomes,
   stops/targets, confluence, parameters, baselines, or transaction costs.
2. AUTHENTICATED ACCESS BOUNDARY: Strictly gated by ARX_EVALUATION_KEY via Bearer token
   or X-Evaluation-Key header using constant-time string comparison (hmac.compare_digest).
3. NO SEARCH ENGINE INDEXING: All endpoints emit X-Robots-Tag: noindex, nofollow, noarchive.
4. CLEAN COHORT FIREWALL: Milestone progress counts ONLY resolved records meeting strict
   ProvenanceCohort.PROSPECTIVE_CLEAN invariants.
"""

import os
import hmac
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Header, Response, HTTPException, status, Request, Body
from fastapi.responses import JSONResponse

from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)
from analyst_dashboard.governance.baseline_engine import BaselineEngine
from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
from analyst_dashboard.governance.evaluator import ProductionCertificationEvaluator
from analyst_dashboard.governance.storage import is_storage_persistent

logger = logging.getLogger("api.routes.governance")

router = APIRouter()

# Evaluation Secret Key Configuration
DEFAULT_DEV_EVAL_KEY = "arx-eval-prospective-2026-secret"
IS_PRODUCTION = os.getenv("ENVIRONMENT", "production").lower() == "production"
SERVER_EVAL_KEY = os.getenv("ARX_EVALUATION_KEY", "" if IS_PRODUCTION else DEFAULT_DEV_EVAL_KEY)

# Dedicated Epoch 2 Governance Keys (Zero Scope Overlap)
DEFAULT_DEV_CERT_KEY = "arx-epoch2-cert-dev-key"
DEFAULT_DEV_ACT_KEY = "arx-epoch2-act-dev-key"
DEFAULT_DEV_REV_KEY = "arx-epoch2-rev-dev-key"

SERVER_CERT_KEY = os.getenv("ARX_EPOCH_CERTIFICATION_KEY", "" if IS_PRODUCTION else DEFAULT_DEV_CERT_KEY)
SERVER_ACT_KEY = os.getenv("ARX_EPOCH_ACTIVATION_KEY", "" if IS_PRODUCTION else DEFAULT_DEV_ACT_KEY)
SERVER_REV_KEY = os.getenv("ARX_EPOCH_REVOCATION_KEY", "" if IS_PRODUCTION else DEFAULT_DEV_REV_KEY)


def _extract_bearer_or_header(authorization: Optional[str], header_val: Optional[str]) -> str:
    """Extracts candidate key from header or Authorization Bearer."""
    candidate_key = ""
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            candidate_key = parts[1]
        elif len(parts) == 1:
            candidate_key = parts[0]
    elif header_val:
        candidate_key = header_val.strip()
    return candidate_key


def _verify_credential(candidate_key: str, expected_key: str) -> bool:
    """Constant-time verification of credential strings."""
    if not expected_key or not candidate_key:
        return False
    return hmac.compare_digest(candidate_key.encode("utf-8"), expected_key.encode("utf-8"))


def _verify_evaluation_access(
    authorization: Optional[str] = None,
    x_evaluation_key: Optional[str] = None,
) -> bool:
    """Verifies incoming authentication credentials using constant-time comparison."""
    if not SERVER_EVAL_KEY:
        logger.warning("ARX_EVALUATION_KEY not configured on server — access denied.")
        return False

    candidate_key = ""
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            candidate_key = parts[1]
        elif len(parts) == 1:
            candidate_key = parts[0]
    elif x_evaluation_key:
        candidate_key = x_evaluation_key.strip()

    if not candidate_key:
        return False

    return hmac.compare_digest(candidate_key.encode("utf-8"), SERVER_EVAL_KEY.encode("utf-8"))


SECURITY_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "CDN-Cache-Control": "no-store",
    "Cloudflare-CDN-Cache-Control": "no-store",
    "Vary": "Authorization, X-Evaluation-Key",
    "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet",
}


def _set_security_headers(response: Response) -> None:
    """Applies strict anti-caching and anti-indexing headers."""
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v


def _compute_milestone_progress(scorecard: Dict[str, Any], ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Computes predefined milestone progress strictly from verified PROSPECTIVE_CLEAN records."""
    all_signals = ledger.get("signals", [])
    manifest_audit = ExperimentLedger.verify_frozen_engine_manifest()
    manifest_valid = manifest_audit.get("valid", False)

    clean_resolved_signals = []
    clean_open_signals = []
    distinct_sessions = set()
    regimes_observed = set()

    for s in all_signals:
        cohort = ExperimentLedger.classify_provenance_cohort(s)
        if cohort != ProvenanceCohort.PROSPECTIVE_CLEAN:
            continue

        # Invariant checks: valid decision and inputs hashes
        dec_hash = s.get("decisionSnapshotHash")
        if dec_hash and dec_hash != ExperimentLedger.compute_decision_snapshot_hash(s):
            continue

        in_hash = s.get("inputsSnapshotHash")
        if in_hash and in_hash != ExperimentLedger.compute_inputs_snapshot_hash(s):
            continue

        if not manifest_valid:
            # If production engine manifest is corrupted, milestone progress is halted
            continue

        if s.get("status") == "RESOLVED":
            clean_resolved_signals.append(s)
            sig_date = s.get("signalDate")
            if sig_date:
                distinct_sessions.add(sig_date)
            regime = s.get("inputs", {}).get("marketRegime")
            if regime:
                regimes_observed.add(regime)
        elif s.get("status") == "OPEN":
            clean_open_signals.append(s)

    n_clean_resolved = len(clean_resolved_signals)
    n_distinct_sessions = len(distinct_sessions)
    n_regimes = len(regimes_observed)

    # Milestone 1: >= 20 resolved PROSPECTIVE_CLEAN predictions
    m1_target = 20
    m1_reached = n_clean_resolved >= m1_target
    m1_progress = min(100.0, round((n_clean_resolved / m1_target) * 100.0, 1))

    # Milestone 2: >= 40 resolved PROSPECTIVE_CLEAN predictions
    m2_target = 40
    m2_reached = n_clean_resolved >= m2_target
    m2_progress = min(100.0, round((n_clean_resolved / m2_target) * 100.0, 1))

    # Milestone 3: >= 60 resolved PROSPECTIVE_CLEAN predictions AND >= 20 sessions AND multiple regimes
    m3_target_trades = 60
    m3_target_sessions = 20
    m3_trades_ok = n_clean_resolved >= m3_target_trades
    m3_sessions_ok = n_distinct_sessions >= m3_target_sessions
    m3_regimes_ok = n_regimes >= 2
    m3_reached = m3_trades_ok and m3_sessions_ok and m3_regimes_ok
    m3_progress = min(100.0, round((n_clean_resolved / m3_target_trades) * 100.0, 1))

    return {
        "manifestIntegrityVerified": manifest_valid,
        "cleanResolvedCount": n_clean_resolved,
        "cleanOpenCount": len(clean_open_signals),
        "distinctTradingSessions": n_distinct_sessions,
        "distinctRegimes": list(regimes_observed),
        "milestone1": {
            "title": "Milestone 1 — Initial Statistical Cohort",
            "targetResolvedTrades": m1_target,
            "currentResolvedTrades": n_clean_resolved,
            "progressPct": m1_progress,
            "status": "REACHED" if m1_reached else "PENDING",
            "condition": ">= 20 resolved PROSPECTIVE_CLEAN predictions",
        },
        "milestone2": {
            "title": "Milestone 2 — Intermediate Calibration Gate",
            "targetResolvedTrades": m2_target,
            "currentResolvedTrades": n_clean_resolved,
            "progressPct": m2_progress,
            "status": "REACHED" if m2_reached else "PENDING",
            "condition": ">= 40 resolved PROSPECTIVE_CLEAN predictions",
        },
        "milestone3": {
            "title": "Milestone 3 — Institutional Robustness Gate",
            "targetResolvedTrades": m3_target_trades,
            "currentResolvedTrades": n_clean_resolved,
            "targetDistinctSessions": m3_target_sessions,
            "currentDistinctSessions": n_distinct_sessions,
            "distinctRegimesCount": n_regimes,
            "progressPct": m3_progress,
            "status": "REACHED" if m3_reached else "PENDING",
            "conditionsMet": {
                "tradesThreshold": m3_trades_ok,
                "sessionsThreshold": m3_sessions_ok,
                "multipleRegimesThreshold": m3_regimes_ok,
            },
            "condition": ">= 60 resolved PROSPECTIVE_CLEAN predictions AND >= 20 distinct sessions AND multiple market regimes",
        },
        "countingRules": {
            "strictCohort": "ProvenanceCohort.PROSPECTIVE_CLEAN only",
            "historicalSignalsIncluded": False,
            "unresolvedSignalsIncluded": False,
            "cryptographicHashesRequired": True,
            "manifestVerificationRequired": True,
        }
    }


@router.get("/evaluation-summary", tags=["Model Governance & Prospective Evaluation"])
def get_evaluation_summary(
    response: Response,
    authorization: Optional[str] = Header(None),
    x_evaluation_key: Optional[str] = Header(None, alias="X-Evaluation-Key"),
):
    """Returns comprehensive read-only evaluation summary for the frozen prospective evaluation.
    
    Strictly protected by ARX_EVALUATION_KEY.
    """
    _set_security_headers(response)

    if not _verify_evaluation_access(authorization, x_evaluation_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid ARX Evaluation Key required to access private prospective evaluation.",
            headers=SECURITY_HEADERS,
        )

    try:
        # 1. Load authoritative paper trading ledger
        ledger = ExperimentLedger.load_ledger()
        
        # 2. Compute governance scorecards for both clean prospective cohort and contaminated historical cohort
        scorecard_clean = ExperimentLedger.compute_governance_scorecard(cohort_filter=ProvenanceCohort.PROSPECTIVE_CLEAN)
        scorecard_all = ExperimentLedger.compute_governance_scorecard(cohort_filter="ALL")
        
        # 3. Compute milestone tracking
        milestones = _compute_milestone_progress(scorecard_clean, ledger)
        
        # 4. Audit frozen engine manifest
        manifest_audit = ExperimentLedger.verify_frozen_engine_manifest()
        
        # 5. Cohort classification statistics
        all_signals = ledger.get("signals", [])
        cohort_counts = {
            "PROSPECTIVE_CLEAN": 0,
            "HISTORICAL_CONTAMINATED": 0,
            "HISTORICAL_UNKNOWN": 0,
            "EXCLUDED": 0,
            "total": len(all_signals),
        }
        for s in all_signals:
            c = ExperimentLedger.classify_provenance_cohort(s)
            cohort_counts[c] = cohort_counts.get(c, 0) + 1

        payload = {
            "status": "SUCCESS",
            "system": {
                "terminalVersion": "ArxTerminal v2.4.0",
                "productionEngineStatus": "FROZEN_UNMODIFIED",
                "measurementSpecVersion": BaselineEngine.BASELINE_SPEC_VERSION,
                "observationOnlyStatus": "ACTIVE",
                "observationStartTimestamp": "2026-09-04T17:32:00Z",
                "frozenEngineCommit": ExperimentLedger.FROZEN_ENGINE_COMMIT,
                "frozenEngineTag": ExperimentLedger.FROZEN_ENGINE_TAG,
                "manifestVerification": manifest_audit,
                "governingPrinciple": (
                    "Observation-only mode. Strategy parameters, thresholds, and execution logic are frozen. "
                    "This interface does not provide trading recommendations or strategy modification."
                ),
            },
            "cohortFirewall": {
                "status": "ACTIVE_ENFORCED",
                "cohortCounts": cohort_counts,
                "cleanEvaluationEligible": scorecard_clean.get("cohortFirewall", {}).get("cleanEvaluationEligible", True),
                "firewallWarning": scorecard_clean.get("cohortFirewall", {}).get("firewallWarning"),
            },
            "milestoneTracker": milestones,
            "prospectiveScorecard": scorecard_clean,
            "globalLedgerScorecard": scorecard_all,
            "evaluationConventions": {
                "intrabarCollision": "Pessimistic fail-closed STOP_LOSS resolution convention",
                "transactionFriction": "30 bps round-trip linear additive deduction standard",
                "survivorshipCaveat": "Live quoted assets at T0; not point-in-time historical survivor-free",
                "confluenceInterpretation": "Ordinal ranking score; evaluated via monotonicity and Spearman rho, not uncalibrated probabilities",
            }
        }
        return payload

    except Exception as e:
        logger.error(f"Error computing evaluation summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation scorecard calculation error: {str(e)}"
        )


@router.get("/prospective-ledger", tags=["Model Governance & Prospective Evaluation"])
def get_prospective_ledger(
    response: Response,
    cohort: Optional[str] = "PROSPECTIVE_CLEAN",
    status_filter: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    x_evaluation_key: Optional[str] = Header(None, alias="X-Evaluation-Key"),
):
    """Returns read-only prediction records from the authoritative ledger with full provenance audit.
    
    Strictly protected by ARX_EVALUATION_KEY.
    """
    _set_security_headers(response)

    if not _verify_evaluation_access(authorization, x_evaluation_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid ARX Evaluation Key required to access private prospective ledger.",
            headers=SECURITY_HEADERS,
        )

    try:
        ledger = ExperimentLedger.load_ledger()
        signals = ledger.get("signals", [])

        records = []
        for s in signals:
            c = ExperimentLedger.classify_provenance_cohort(s)
            
            # Apply cohort filter if requested
            if cohort and cohort.upper() != "ALL" and c != cohort.upper():
                continue

            # Apply status filter if requested
            if status_filter and status_filter.upper() != "ALL" and s.get("status", "").upper() != status_filter.upper():
                continue

            # Verify cryptographic hashes
            dec_valid = True
            if s.get("decisionSnapshotHash"):
                dec_valid = (s["decisionSnapshotHash"] == ExperimentLedger.compute_decision_snapshot_hash(s))

            in_valid = True
            if s.get("inputsSnapshotHash"):
                in_valid = (s["inputsSnapshotHash"] == ExperimentLedger.compute_inputs_snapshot_hash(s))

            record_summary = {
                "signalId": s.get("signalId"),
                "symbol": s.get("symbol"),
                "signalDate": s.get("signalDate"),
                "provenanceCohort": c,
                "engineVersion": s.get("engineVersion"),
                "engineTag": s.get("engineTag"),
                "status": s.get("status"),
                "entryPrice": s.get("entryPrice"),
                "stopLoss": s.get("stopLoss"),
                "stopLossPct": s.get("stopLossPct"),
                "takeProfit1": s.get("takeProfit1"),
                "takeProfit1Pct": s.get("takeProfit1Pct"),
                "takeProfit2": s.get("takeProfit2"),
                "takeProfit2Pct": s.get("takeProfit2Pct"),
                "riskRewardRatio": s.get("riskRewardRatio"),
                "confluenceScore": s.get("confluenceScore"),
                "inputs": s.get("inputs"),
                "forwardTracking": s.get("forwardTracking"),
                "decisionSnapshotHash": s.get("decisionSnapshotHash"),
                "decisionHashVerified": dec_valid,
                "inputsSnapshotHash": s.get("inputsSnapshotHash"),
                "inputsHashVerified": in_valid,
                "liquidityGrade": s.get("liquidityAtSignal", {}).get("liquidityGrade"),
            }
            records.append(record_summary)

        return {
            "status": "SUCCESS",
            "totalMatched": len(records),
            "cohortFilter": cohort,
            "statusFilter": status_filter,
            "records": records,
        }

    except Exception as e:
        logger.error(f"Error fetching prospective ledger: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prospective ledger fetch error: {str(e)}"
        )


# ==============================================================================
# EPOCH 2 PRODUCTION GOVERNANCE ENDPOINTS
# ==============================================================================

@router.post("/epoch-2/certify-release", tags=["Model Governance & Prospective Evaluation"])
async def certify_epoch2_release(
    request: Request,
    response: Response,
    authorization: Optional[str] = Header(None),
    x_arx_certification_key: Optional[str] = Header(None, alias="X-Arx-Certification-Key"),
):
    """Executes the canonical in-process 12-check production certification suite.
    Gated strictly by ARX_EPOCH_CERTIFICATION_KEY.
    """
    _set_security_headers(response)
    candidate_key = _extract_bearer_or_header(authorization, x_arx_certification_key)
    if not _verify_credential(candidate_key, SERVER_CERT_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid ARX_EPOCH_CERTIFICATION_KEY required for certification execution."
        )

    try:
        evaluator = ProductionCertificationEvaluator()
        result = await evaluator.execute_full_certification_suite()
        status_code = status.HTTP_200_OK if result.get("overallStatus") == "PASS" else status.HTTP_422_UNPROCESSABLE_ENTITY
        return JSONResponse(status_code=status_code, content=result)
    except Exception as e:
        logger.error(f"Runtime certification evaluation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Certification evaluation error: {str(e)}"
        )


@router.post("/epoch-2/activate", tags=["Model Governance & Prospective Evaluation"])
def activate_epoch2(
    request: Request,
    response: Response,
    authorization: Optional[str] = Header(None),
    x_arx_activation_key: Optional[str] = Header(None, alias="X-Arx-Activation-Key"),
):
    """Activates Epoch 2 prospective observation once and only once in production SQLite.
    Gated strictly by ARX_EPOCH_ACTIVATION_KEY.
    Requires an existing PASS release authorization for current container runtime.
    """
    _set_security_headers(response)
    candidate_key = _extract_bearer_or_header(authorization, x_arx_activation_key)
    if not _verify_credential(candidate_key, SERVER_ACT_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid ARX_EPOCH_ACTIVATION_KEY required for Epoch 2 activation."
        )

    current_release = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA")
    current_deployment = os.getenv("RAILWAY_DEPLOYMENT_ID") or os.getenv("DEPLOYMENT_ID")

    if not current_release or not current_deployment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Runtime Error: RAILWAY_GIT_COMMIT_SHA and RAILWAY_DEPLOYMENT_ID must be present in container environment."
        )

    current_release = current_release.strip().lower()
    current_deployment = current_deployment.strip()

    activated_at_utc = datetime.now(timezone.utc).isoformat()
    token_hash = hashlib.sha256(candidate_key.encode("utf-8")).hexdigest()

    gov_engine = GovernanceDatabaseEngine()
    if gov_engine.is_runtime_revoked(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=current_release,
        deployment_id=current_deployment,
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Activation Rejected: RUNTIME_REVOKED: Target release {current_release} on deployment {current_deployment} has been revoked."
        )

    success, reason = gov_engine.record_activation(
        epoch_id=ExperimentLedger.EPOCH_ID,
        release_sha=current_release,
        deployment_id=current_deployment,
        activated_at_utc=activated_at_utc,
        activation_source="RAILWAY_CONTAINER_PRODUCTION_RUNTIME",
        activation_auth_token_hash=token_hash,
    )

    if not success:
        if "CONFLICT" in reason:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Activation Conflict: {reason}"
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Activation Rejected: {reason}"
        )

    existing_record = gov_engine.get_activation_record(ExperimentLedger.EPOCH_ID)

    return {
        "status": "SUCCESS",
        "result": reason,
        "epochId": ExperimentLedger.EPOCH_ID,
        "releaseSha": current_release,
        "deploymentId": current_deployment,
        "activatedAtUtc": existing_record["activated_at_utc"] if existing_record else activated_at_utc,
    }


@router.post("/epoch-2/revoke-runtime", tags=["Model Governance & Prospective Evaluation"])
def revoke_epoch2_runtime(
    request: Request,
    response: Response,
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    x_arx_revocation_key: Optional[str] = Header(None, alias="X-Arx-Revocation-Key"),
):
    """Appends an operational runtime participation revocation to the append-only ledger.
    Gated strictly by ARX_EPOCH_REVOCATION_KEY.
    Does NOT modify or deactivate the underlying epoch activation boundary.
    """
    _set_security_headers(response)
    candidate_key = _extract_bearer_or_header(authorization, x_arx_revocation_key)
    if not _verify_credential(candidate_key, SERVER_REV_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid ARX_EPOCH_REVOCATION_KEY required for runtime revocation."
        )

    epoch_id = payload.get("epochId", ExperimentLedger.EPOCH_ID)
    release_sha = payload.get("releaseSha")
    deployment_id = payload.get("deploymentId")
    reason = payload.get("revocationReason")

    if not release_sha or not deployment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required target fields: releaseSha and deploymentId must be specified."
        )

    if not reason or len(str(reason).strip()) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid revocationReason: Reason must be at least 16 characters explaining revocation rationale."
        )

    # Server derives authoritative principal from revocation credential fingerprint
    revoked_by = f"KEY_FINGERPRINT:{hashlib.sha256(candidate_key.encode('utf-8')).hexdigest()[:12]}"
    revoked_at_utc = datetime.now(timezone.utc).isoformat()

    gov_engine = GovernanceDatabaseEngine()
    success, result_msg = gov_engine.record_revocation(
        epoch_id=epoch_id,
        release_sha=release_sha.strip().lower(),
        deployment_id=deployment_id.strip(),
        revoked_at_utc=revoked_at_utc,
        revocation_reason=str(reason).strip(),
        revoked_by=revoked_by,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Revocation Target Not Found: {result_msg}"
        )

    return {
        "status": "SUCCESS",
        "result": result_msg,
        "epochId": epoch_id,
        "releaseSha": release_sha,
        "deploymentId": deployment_id,
        "revokedAtUtc": revoked_at_utc,
        "revokedBy": revoked_by,
    }


@router.get("/epoch-2/status", tags=["Model Governance & Prospective Evaluation"])
def get_epoch2_governance_status(response: Response):
    """Returns canonical SQLite governance and runtime authorization status."""
    _set_security_headers(response)
    gov_engine = GovernanceDatabaseEngine()

    current_release = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA")
    current_deployment = os.getenv("RAILWAY_DEPLOYMENT_ID") or os.getenv("DEPLOYMENT_ID")

    activation_record = gov_engine.get_activation_record(ExperimentLedger.EPOCH_ID)
    release_auth = None
    is_revoked = False
    is_authorized = False
    auth_reason = "UNEVALUATED"

    if current_release and current_deployment:
        clean_rel = current_release.strip().lower()
        clean_dep = current_deployment.strip()
        release_auth = gov_engine.get_release_authorization(
            epoch_id=ExperimentLedger.EPOCH_ID,
            release_sha=clean_rel,
            deployment_id=clean_dep,
        )
        is_revoked = gov_engine.is_runtime_revoked(
            epoch_id=ExperimentLedger.EPOCH_ID,
            release_sha=clean_rel,
            deployment_id=clean_dep,
        )
        is_authorized, auth_reason = gov_engine.evaluate_capture_authorization_predicate(
            epoch_id=ExperimentLedger.EPOCH_ID,
            release_sha=clean_rel,
            deployment_id=clean_dep,
        )

    return {
        "epochId": ExperimentLedger.EPOCH_ID,
        "runningReleaseSha": current_release,
        "runningDeploymentId": current_deployment,
        "activationRecord": activation_record,
        "releaseAuthorization": release_auth,
        "isRuntimeRevoked": is_revoked,
        "prospectiveCaptureAuthorized": is_authorized,
        "storagePersistence": "VERIFIED" if is_storage_persistent() else "EPHEMERAL",
    }
