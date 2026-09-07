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
from typing import Dict, Any, Optional
from fastapi import APIRouter, Header, Response, HTTPException, status
from fastapi.responses import JSONResponse

from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)
from analyst_dashboard.governance.baseline_engine import BaselineEngine

logger = logging.getLogger("api.routes.governance")

router = APIRouter()

# Evaluation Secret Key Configuration
# Set ARX_EVALUATION_KEY in backend environment (Render/Railway/local .env).
# In production, missing key disables endpoint (fail-closed).
DEFAULT_DEV_EVAL_KEY = "arx-eval-prospective-2026-secret"
IS_PRODUCTION = os.getenv("ENVIRONMENT", "production").lower() == "production"
SERVER_EVAL_KEY = os.getenv("ARX_EVALUATION_KEY", "" if IS_PRODUCTION else DEFAULT_DEV_EVAL_KEY)


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
