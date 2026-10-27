import os
from fastapi import APIRouter, Response, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import pandas as pd
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.analyzers.tactical_regime import get_shared_macro_snapshot
from analyst_dashboard.data.market_evidence import (
    MarketEvidence,
    MarketProvenance,
    Provider,
    IngestionSource,
    ServingSource,
    CacheOrigin,
    ObservationPrecision,
    ObservationSource,
    AdjustmentState,
    StructuralQuality,
)
from analyst_dashboard.analyzers.scanner_contract import CANONICAL_VCP_UNIVERSE, RADAR_SCOPE_LABEL
from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner, SmartMoneyScannerRunner
from analyst_dashboard.smart_money.storage import (
    get_smart_money_scanner_runner,
    get_smart_money_storage_backend,
    get_smart_money_repository,
)

from datetime import datetime
import hashlib
import hmac

def get_operator_token() -> str:
    """Returns the configured operator authorization token. Fails closed (empty string) if unset."""
    return os.getenv("ARX_OPERATOR_TOKEN", "").strip()

OPERATOR_TOKEN: str = get_operator_token()


def verify_operator_token(candidate_token: Optional[str]) -> bool:
    """
    Constant-time comparison against configured ARX_OPERATOR_TOKEN.
    Fails closed if ARX_OPERATOR_TOKEN is unset, empty, or candidate does not match.
    """
    expected = get_operator_token() or OPERATOR_TOKEN
    if not expected or not candidate_token:
        return False
    return hmac.compare_digest(candidate_token.strip().encode("utf-8"), expected.encode("utf-8"))

import hmac


def is_vcp_natural_scheduler_enabled() -> bool:
    """Check if natural VCP scheduler trigger is enabled via environment variable (default: disabled)."""
    return os.getenv("VCP_NATURAL_SCHEDULER_ENABLED", "0").strip().lower() in ("1", "true", "yes", "enabled")


def verify_scheduler_token(candidate_token: Optional[str]) -> bool:
    """
    Constant-time comparison against dedicated VCP_SCHEDULER_SECRET.
    Strictly prohibits cross-domain credential fallback to telemetry or epoch-activation keys.
    Fails closed if VCP_SCHEDULER_SECRET is unset, empty, or candidate does not match.
    """
    if not candidate_token:
        return False
    expected = os.getenv("VCP_SCHEDULER_SECRET", "").strip()
    if not expected:
        return False
    return hmac.compare_digest(candidate_token.strip().encode("utf-8"), expected.encode("utf-8"))


class VCPNaturalScanRequest(BaseModel):
    scheduled_date: Optional[str] = None
    universe_build_id: Optional[str] = None
    scheduler_job_id: Optional[str] = None
    scheduler_event_id: Optional[str] = None

router = APIRouter()
screener = HiddenGemsScreener()
optimal_engine = OptimalExecutionEngine()
confluence_engine = ConfluenceEngine()
smart_money_engine = SmartMoneyEngine()
market_db = MarketDatabaseEngine()
vcp_scanner_runner = VCPScannerRunner(market_db=market_db, confluence_engine=confluence_engine)
smart_money_scanner_runner = get_smart_money_scanner_runner()


# Authentic Multi-Sector Dual-Horizon Universes (60 Total Quality Assets)
DAY_TRADER_CANDIDATES = [
    # AI & Megacap Momentum
    "NVDA", "TSLA", "PLTR", "ARM", "SMCI", "AMD", "META", "AAPL", "MSFT", "AMZN",
    # Cloud & Cybersecurity
    "CRWD", "PANW", "NET", "DDOG", "MDB",
    # Crypto & FinTech Beta
    "COIN", "MARA", "MSTR", "HOOD",
    # High-Beta Volatility & Squeeze Runners
    "DUOL", "CELH", "IONQ", "RKLB", "APP",
]

LONG_TERM_CANDIDATES = CANONICAL_VCP_UNIVERSE

DEFAULT_CANDIDATES = LONG_TERM_CANDIDATES

# ── Canonical Corporate Actions & Listing Lifecycle Registry ───────────────
# Authoritative registry of confirmed corporate actions, mergers, and delistings
# with primary SEC filing citations. Inactive securities are retained for audit
# and historical valuation integrity, but strictly excluded from active discovery.
CONFIRMED_CORPORATE_ACTIONS: Dict[str, Dict[str, Any]] = {
    "CPRX": {
        "status": "DELISTED",
        "action_type": "CASH_MERGER_ACQUISITION",
        "effective_date": "2026-07-21",
        "acquirer": "Angelini Pharma S.p.A.",
        "consideration": "$31.50 per share in cash",
        "filings": [
            {"form": "25-NSE", "filing_date": "2026-07-10", "description": "Notification of Removal from Listing on Nasdaq"},
            {"form": "15-12B", "filing_date": "2026-07-20", "description": "Certification and Notice of Termination of Registration"},
        ],
        "execution_status": "DELISTED_SECURITY",
        "status_label": "🛑 Delisted / Inactive",
        "status_color": "rose",
        "decision_state": "INELIGIBLE_INACTIVE",
        "screening_eligible": False,
        "is_investable": False,
        "notes": "Acquired by Angelini Pharma for $31.50/share in July 2026. Common stock deregistered and delisted from Nasdaq.",
    }
}

# ── Attributable Lifecycle Evidence Registry ──────────────────────────────
# Invariant: Static reference catalog membership does NOT constitute lifecycle evidence.
# Listing status is reported as ACTIVE only where attributable, primary-source lifecycle
# evidence exists (e.g. verified exchange certificate, primary listing confirmation).
# Without attributable lifecycle evidence, candidates report UNVERIFIED.
ATTRIBUTABLE_LIFECYCLE_EVIDENCE: Dict[str, Dict[str, Any]] = {
    # Populated when attributable primary listing / lifecycle verification is supplied.
    # Candidates without verified lifecycle records report 'UNVERIFIED'.
}

# ── Canonical Radar Discovery Taxonomy ──────────────────────────────────────
# Exactly three canonical categories governed across backend transport, API contracts,
# and UI presentation (no transport aliases such as "VALUE").
CANONICAL_RADAR_CATEGORIES = ["VALUE_GARP", "VCP", "SMART_MONEY"]

# Root-level capability contract distinguishing universe screening from single-asset analysis
RADAR_CAPABILITY_CONTRACT = {
    "VALUE_GARP": {
        "status": "AVAILABLE",
        "universeScreening": "AVAILABLE",
        "singleAssetAnalysis": "AVAILABLE",
        "catalogType": "CURATED_REFERENCE_CATALOG",
        "countAuthority": "DYNAMIC_CATALOG",
        "sortingSemantics": "CONFLUENCE_TECHNICAL_FLOW",
        "rationale": "Curated multi-factor fundamental reference catalog active; candidates sorted by technical and macro confluence score rather than intrinsic valuation discount.",
    },
    "VCP": {
        "status": "AVAILABLE",
        "universeScreening": "AVAILABLE",
        "singleAssetAnalysis": "AVAILABLE",
        "rationale": "Authentic Minervini Volatility Contraction Pattern universe screening active with deterministic publication integrity.",
    },
    "SMART_MONEY": {
        "status": "PIPELINE_PENDING",
        "universeScreening": "PIPELINE_PENDING",
        "singleAssetAnalysis": "AVAILABLE",
        "rationale": "Single-asset SEC Form 4 and Congressional STOCK Act disclosures active on /smart-money; universe-level institutional accumulation screener pipeline is pending deployment.",
    },
}

# Whitelist of authentic fundamental models that establish descriptive classification
VALUE_GARP_MODELS = {
    "Peter Lynch GARP Compounder",
    "Greenblatt Magic Formula",
    "Deep Value & Capital Return (Decelerating Comp Watch)",
    "Deep Value & Brand Mean-Reversion",
}


def get_radar_capabilities(
    *args,
    vcp_snapshot: Optional[Dict[str, Any]] = None,
    publication_pointer: Optional[Any] = None,
    sm_snapshot: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    # Support positional calls from Candidate 006 (sm_snapshot) and main (vcp_snapshot, publication_pointer)
    if args:
        first_arg = args[0]
        if len(args) >= 2:
            vcp_snapshot = first_arg
            publication_pointer = args[1]
        elif isinstance(first_arg, dict) and ("snapshot_id" in first_arg or "universe" in first_arg or "published_symbols" in first_arg):
            vcp_snapshot = first_arg
        else:
            # Candidate 006 backwards compatibility: single positional dict is sm_snapshot
            sm_snapshot = first_arg

    caps = {k: dict(v) for k, v in RADAR_CAPABILITY_CONTRACT.items()}

    # Resolve publication identity & population denominator if pointer or snapshot provided
    pop_denom = None
    build_id = None
    univ_id = None
    pub_digest = None
    is_rollback = False
    rollback_reason = None
    membership_status = "UNAVAILABLE"

    if publication_pointer is not None and hasattr(publication_pointer, "published_symbols"):
        pop_denom = len(publication_pointer.published_symbols)
        build_id = getattr(publication_pointer, "build_id", None)
        univ_id = getattr(publication_pointer, "universe_id", None)
        pub_digest = getattr(publication_pointer, "publication_digest", None)
        is_rollback = getattr(publication_pointer, "is_rollback", False)
        rollback_reason = getattr(publication_pointer, "rollback_reason", None)
        membership_status = "AVAILABLE" if pop_denom else "UNAVAILABLE"

    # Invariant 6: Stale or rolled-back pointers cannot imply operational scanner availability
    if is_rollback:
        caps["VCP"] = {
            "status": "PIPELINE_PENDING",
            "universeScreening": "PIPELINE_PENDING",
            "singleAssetAnalysis": "AVAILABLE",
            "matchedCount": None,
            "coverageStatus": "ROLLED_BACK",
            "scannedSuccessfullyCount": 0,
            "unavailableSymbolCount": pop_denom or 0,
            "populationDenominator": pop_denom,
            "universeBuildId": build_id,
            "universeId": univ_id,
            "publicationDigest": pub_digest,
            "isRollback": True,
            "rollbackReason": rollback_reason,
            "membershipStatus": membership_status,
            "lifecycleState": "STALE_OR_ROLLED_BACK_PUBLICATION",
            "reason": "ROLLED_BACK_PUBLICATION_POINTER",
            "rationale": f"Publication build '{build_id}' was rolled back ({rollback_reason or 'No reason specified'}). Scanner execution is suspended.",
        }
    elif vcp_snapshot:
        snap_status = vcp_snapshot.get("status")
        freshness_info = vcp_snapshot.get("freshness", {})
        u_meta = vcp_snapshot.get("universe", {})
        cov_info = vcp_snapshot.get("coverage", {})
        if u_meta:
            pop_denom = pop_denom or u_meta.get("eligible_universe_count") or vcp_snapshot.get("snapshot", {}).get("universe_size")
            build_id = build_id or u_meta.get("universe_build_id")
            univ_id = univ_id or u_meta.get("scope_class") or "ARX_US_EQUITIES"
            pub_digest = pub_digest or u_meta.get("membership_hash")
            membership_status = "AVAILABLE" if pop_denom else membership_status

        if snap_status == "AVAILABLE":
            raw_matched = vcp_snapshot.get("matched_count")
            is_valid_op = type(raw_matched) is int and raw_matched >= 0
            matched_count = raw_matched if is_valid_op else None
            caps["VCP"] = {
                "status": "AVAILABLE" if is_valid_op else "PIPELINE_PENDING",
                "universeScreening": "AVAILABLE" if is_valid_op else "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": matched_count,
                "coverageStatus": cov_info.get("coverage_status", "COMPLETE" if pop_denom else "UNKNOWN"),
                "scannedSuccessfullyCount": cov_info.get("scanned_successfully_count", pop_denom),
                "unavailableSymbolCount": cov_info.get("unavailable_symbol_count", 0),
                "dataAsOf": freshness_info.get("data_as_of") or vcp_snapshot.get("data_as_of"),
                "generatedAt": freshness_info.get("generated_at") or vcp_snapshot.get("generated_at"),
                "snapshotId": vcp_snapshot.get("snapshot_id"),
                "freshnessStatus": freshness_info.get("status", "LIVE"),
                "populationDenominator": pop_denom,
                "universeBuildId": build_id,
                "universeId": univ_id,
                "publicationDigest": pub_digest,
                "isRollback": False,
                "rollbackReason": None,
                "membershipStatus": membership_status,
                "lifecycleState": "SCANNER_OPERATIONAL" if is_valid_op else "SCANNER_OUTPUT_INCOMPLETE",
                "rationale": "Authentic Minervini Volatility Contraction Pattern universe screening active with deterministic publication integrity.",
            }
        elif snap_status == "PIPELINE_PENDING":
            reason = vcp_snapshot.get("reason", "UNCOMPUTED_CLASSIFICATION" if vcp_snapshot.get("snapshot_id") else "NO_ACTIVE_SNAPSHOT")
            caps["VCP"] = {
                "status": "PIPELINE_PENDING",
                "universeScreening": "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "coverageStatus": cov_info.get("coverage_status", "PENDING"),
                "scannedSuccessfullyCount": cov_info.get("scanned_successfully_count", 0),
                "unavailableSymbolCount": cov_info.get("unavailable_symbol_count", 0),
                "dataAsOf": freshness_info.get("data_as_of") or vcp_snapshot.get("data_as_of"),
                "generatedAt": freshness_info.get("generated_at") or vcp_snapshot.get("generated_at"),
                "snapshotId": vcp_snapshot.get("snapshot_id"),
                "freshnessStatus": freshness_info.get("status", "UNKNOWN"),
                "reason": reason,
                "populationDenominator": pop_denom,
                "universeBuildId": build_id,
                "universeId": univ_id,
                "publicationDigest": pub_digest,
                "isRollback": is_rollback,
                "rollbackReason": rollback_reason,
                "membershipStatus": membership_status,
                "lifecycleState": "COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED" if pop_denom else "NO_COMMITTED_PUBLICATION",
                "rationale": vcp_snapshot.get("detail") or "Automated market-wide volatility contraction screening is pending authoritative classifier integration.",
            }
        elif snap_status == "ERROR":
            fail_reason = cov_info.get("failure_reason") or vcp_snapshot.get("reason", "STORAGE_ERROR")
            raw_cov = cov_info.get("coverage_status")
            if not raw_cov or raw_cov in ("COMPLETE", "ERROR"):
                cov_status = "EMPTY_UNIVERSE" if (pop_denom == 0 or fail_reason in ("BUILD_MEMBERSHIP_EMPTY", "EMPTY_UNIVERSE")) else "UNAVAILABLE"
            else:
                cov_status = raw_cov
            caps["VCP"] = {
                "status": "ERROR",
                "universeScreening": "ERROR",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "coverageStatus": cov_status,
                "scannedSuccessfullyCount": cov_info.get("scanned_successfully_count", 0),
                "unavailableSymbolCount": cov_info.get("unavailable_symbol_count", pop_denom or 0),
                "dataAsOf": None,
                "generatedAt": None,
                "snapshotId": None,
                "freshnessStatus": "UNKNOWN",
                "reason": fail_reason,
                "populationDenominator": pop_denom,
                "universeBuildId": build_id,
                "universeId": univ_id,
                "publicationDigest": pub_digest,
                "isRollback": is_rollback,
                "rollbackReason": rollback_reason,
                "membershipStatus": membership_status,
                "lifecycleState": "ERROR",
                "rationale": vcp_snapshot.get("detail") or (
                    f"VCP scanner execution failed: {fail_reason}."
                    if fail_reason != "STORAGE_ERROR"
                    else "VCP scanner snapshot store error."
                ),
            }
        elif snap_status == "STALE":
            raw_matched = vcp_snapshot.get("matched_count")
            is_valid_count = type(raw_matched) is int and raw_matched >= 0
            matched_count = raw_matched if is_valid_count else None
            caps["VCP"] = {
                "status": "AVAILABLE" if is_valid_count else "PIPELINE_PENDING",
                "universeScreening": "AVAILABLE" if is_valid_count else "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": matched_count,
                "coverageStatus": cov_info.get("coverage_status", "STALE"),
                "scannedSuccessfullyCount": cov_info.get("scanned_successfully_count", pop_denom),
                "unavailableSymbolCount": cov_info.get("unavailable_symbol_count", 0),
                "dataAsOf": freshness_info.get("data_as_of") or vcp_snapshot.get("data_as_of"),
                "generatedAt": freshness_info.get("generated_at") or vcp_snapshot.get("generated_at"),
                "snapshotId": vcp_snapshot.get("snapshot_id"),
                "freshnessStatus": "STALE",
                "populationDenominator": pop_denom,
                "universeBuildId": build_id,
                "universeId": univ_id,
                "publicationDigest": pub_digest,
                "isRollback": is_rollback,
                "rollbackReason": rollback_reason,
                "membershipStatus": membership_status,
                "lifecycleState": "SCANNER_OPERATIONAL",
                "rationale": "VCP scanner snapshot is stale past freshness policy cutoff.",
            }
    elif pop_denom:
        # No active scanner snapshot, but publication pointer exists at membership layer
        caps["VCP"] = {
            "status": "PIPELINE_PENDING",
            "universeScreening": "PIPELINE_PENDING",
            "singleAssetAnalysis": "AVAILABLE",
            "matchedCount": None,
            "coverageStatus": "PENDING",
            "scannedSuccessfullyCount": 0,
            "unavailableSymbolCount": 0,
            "populationDenominator": pop_denom,
            "universeBuildId": build_id,
            "universeId": univ_id,
            "publicationDigest": pub_digest,
            "isRollback": is_rollback,
            "rollbackReason": rollback_reason,
            "membershipStatus": "AVAILABLE",
            "lifecycleState": "COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED",
            "reason": "SCANNER_NOT_ENROLLED",
            "rationale": f"Universe publication '{build_id}' active with {pop_denom} scannable symbols; batch scanner pipeline is pending operational activation.",
        }

    # Evaluate SMART_MONEY capability from authoritative snapshot
    if sm_snapshot and sm_snapshot.get("status") == "AVAILABLE":
        raw_matched = sm_snapshot.get("matched_count")
        is_valid_count = type(raw_matched) is int and raw_matched >= 0
        matched_cnt = raw_matched if is_valid_count else 0
        caps["SMART_MONEY"] = {
            "status": "AVAILABLE",
            "universeScreening": "AVAILABLE",
            "singleAssetAnalysis": "AVAILABLE",
            "subsignal": "INSIDER_FORM4",
            "matchedCount": matched_cnt,
            "dataAsOf": sm_snapshot.get("freshness", {}).get("data_as_of") or sm_snapshot.get("data_as_of"),
            "populationDenominator": sm_snapshot.get("population_denominator", 0),
            "rationale": "Authentic market-wide SEC Form 4 insider accumulation screening active with point-in-time certification.",
        }
    elif sm_snapshot and sm_snapshot.get("status") == "ERROR":
        caps["SMART_MONEY"] = {
            "status": "ERROR",
            "universeScreening": "ERROR",
            "singleAssetAnalysis": "AVAILABLE",
            "subsignal": "INSIDER_FORM4",
            "matchedCount": None,
            "dataAsOf": None,
            "populationDenominator": 0,
            "rationale": sm_snapshot.get("detail", "Smart Money screening pipeline error."),
        }
    else:
        # Canonical baseline: PIPELINE_PENDING with matchedCount: None
        caps["SMART_MONEY"] = {
            "status": "PIPELINE_PENDING",
            "universeScreening": "PIPELINE_PENDING",
            "singleAssetAnalysis": "AVAILABLE",
            "subsignal": "INSIDER_FORM4",
            "matchedCount": None,
            "dataAsOf": None,
            "populationDenominator": 0,
            "rationale": RADAR_CAPABILITY_CONTRACT["SMART_MONEY"]["rationale"],
        }
    return caps


def derive_canonical_radar_categories(
    expert_model: Optional[str],
    peg_ratio: Optional[float],
    roic_pct: Optional[float],
    gross_margin_pct: Optional[float],
    symbol: str,
    vcp_matched_symbols: Optional[set] = None,
    smart_money_matched_symbols: Optional[set] = None,
) -> List[str]:
    """
    Authoritative deterministic classifier: analytical evidence -> explicit category.
    Strictly immune to arbitrary substring heuristics (e.g. 'Smart Growth' or 'Flow Traders').
    """
    cats: List[str] = []

    # Inactive / delisted securities are strictly excluded from active discovery categories
    if symbol in CONFIRMED_CORPORATE_ACTIONS:
        return cats

    # VALUE_GARP: Requires quantitative valuation/quality qualification.
    # Descriptive archetype strings alone do NOT certify qualification.
    # Pre-existing established quantitative criteria: PEG <= 1.05 and ROIC >= 15.0%
    # No invented replacement thresholds are permitted.
    if peg_ratio is not None and 0.0 < peg_ratio <= 1.05 and roic_pct is not None and roic_pct >= 15.0:
        cats.append("VALUE_GARP")

    # VCP: Backed by deterministic market-wide scanner runner and verified snapshot
    if vcp_matched_symbols is not None and symbol in vcp_matched_symbols:
        cats.append("VCP")

    # SMART_MONEY: Backed by active verified snapshot from SmartMoneyScannerRunner
    if smart_money_matched_symbols is not None and symbol in smart_money_matched_symbols:
        cats.append("SMART_MONEY")

    return cats


def classify_candidate_category_evidence(categories: List[str]) -> Dict[str, str]:
    """
    Produces candidate-level evidence state for performed evaluations only.
    Does NOT manufacture evidence for unperformed universe evaluations.
    """
    evidence: Dict[str, str] = {}
    for cat in categories:
        if cat in CANONICAL_RADAR_CATEGORIES:
            evidence[cat] = "CRITERIA_MATCHED"
    return evidence


class ScreenerRequest(BaseModel):
    tickers: Optional[List[str]] = None
    user_role: Optional[str] = "LONG_TERM"


@router.post("/run")
def run_screener(request: ScreenerRequest = None, response: Response = None):
    """Run the Hidden Gems Discovery Screener against Peter Lynch GARP and Greenblatt Magic Formula criteria."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
    role = request.user_role if request and request.user_role else "LONG_TERM"
    default_pool = DAY_TRADER_CANDIDATES if role == "DAY_TRADER" else LONG_TERM_CANDIDATES
    tickers = (request.tickers if request and request.tickers else default_pool)
    results = screener.evaluate_candidates(tickers)

    return {
        "total_candidates": len(tickers),
        "gems_found": len(results),
        "results": results,
    }


run_screener_post = run_screener


@router.get("/run")
def run_screener_get(
    response: Response = None,
    filter_type: str = "all",
    user_role: str = "LONG_TERM",
    custom_tickers: Optional[str] = None,
):
    """GET endpoint supporting live screener execution, archetype filtering, and on-demand custom ticker scanning."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"

    is_day_trader = (user_role == "DAY_TRADER")

    # On-demand custom watchlist input support
    if custom_tickers and custom_tickers.strip():
        parsed = [t.strip().upper() for t in custom_tickers.replace(",", " ").split() if t.strip()]
        active_universe = parsed if parsed else (DAY_TRADER_CANDIDATES if is_day_trader else LONG_TERM_CANDIDATES)
    else:
        active_universe = DAY_TRADER_CANDIDATES if is_day_trader else LONG_TERM_CANDIDATES

    results = screener.evaluate_candidates(active_universe)
    shared_macro = get_shared_macro_snapshot()
    shared_macro_context_id = shared_macro.get("macroContextId")

    # Retrieve active VCP scanner snapshot for authentic category qualification
    vcp_snapshot = vcp_scanner_runner.get_active_or_latest_snapshot()
    vcp_matched_symbols = {r["symbol"] for r in vcp_snapshot.get("results", []) if "symbol" in r}

    # Retrieve active Smart Money scanner snapshot for authentic category qualification
    sm_runner = get_smart_money_scanner_runner()
    sm_snapshot = sm_runner.get_active_or_latest_snapshot()
    sm_matched_symbols = (
        {r["symbol"] for r in sm_snapshot.get("results", []) if "symbol" in r}
        if sm_snapshot.get("status") == "AVAILABLE"
        else set()
    )

    # Map candidate fields with live optimal execution levels
    mapped_candidates = []
    for r in results:
        raw_sym = r.get("ticker") or r.get("symbol")
        if not raw_sym or not str(raw_sym).strip():
            continue
        sym = str(raw_sym).strip().upper()
        roic_val = r.get("roic_pct")
        margin_val = r.get("gross_margin_pct")

        # Check for confirmed corporate actions (delistings / acquisitions)
        corp_action = CONFIRMED_CORPORATE_ACTIONS.get(sym)
        is_delisted = corp_action is not None

        # Retrieve current price and candles from market database
        latest_info = market_db.get_latest_price(sym)
        current_price = latest_info.get("currentPrice") if (latest_info and latest_info.get("currentPrice") and latest_info["currentPrice"] > 0) else None
        raw_trade_date = latest_info.get("date") if latest_info else None
        price_date = str(raw_trade_date)[:10] if (raw_trade_date and current_price is not None and not is_delisted) else None
        price_basis = "DAILY_CLOSE" if (current_price is not None and price_date is not None) else None
        hist_df = pd.DataFrame()

        if is_delisted:
            execution = {
                "current_price": current_price,
                "optimal_entry_min": None,
                "optimal_entry_max": None,
                "stop_loss": None,
                "stop_loss_pct": None,
                "take_profit_1": None,
                "take_profit_1_pct": None,
                "take_profit_2": None,
                "take_profit_2_pct": None,
                "risk_reward_ratio": None,
                "execution_status": "DELISTED_SECURITY",
                "setup_pattern": "Delisted Security / Ceased Public Trading",
                "entry_thesis": f"Security ceased trading pursuant to {corp_action['action_type']} ({corp_action['notes']}). Trade execution suppressed.",
                "invalidation_condition": "Security delisted from exchange.",
                "stage_phase": "Delisted / Inactive",
                "vcp_contraction_status": "Inactive",
                "breakout_pivot": None,
                "atr_14": None,
                "liquidity_defense": None,
            }
        elif current_price is None or current_price <= 0:
            current_price = None
            execution = {
                "current_price": None,
                "optimal_entry_min": None,
                "optimal_entry_max": None,
                "stop_loss": None,
                "stop_loss_pct": None,
                "take_profit_1": None,
                "take_profit_1_pct": None,
                "take_profit_2": None,
                "take_profit_2_pct": None,
                "risk_reward_ratio": None,
                "execution_status": "UNVERIFIED_ASSET",
                "setup_pattern": "Unverified Asset Setup",
                "entry_thesis": "Pricing and exchange tape unavailable. Trade levels suppressed.",
                "invalidation_condition": "Awaiting market data.",
                "stage_phase": "Unverified Asset",
                "vcp_contraction_status": "Unverified",
                "breakout_pivot": None,
                "atr_14": None,
                "liquidity_defense": None,
            }
        else:
            db_candles = market_db.get_daily_candles(sym, limit=60)
            if db_candles:
                hist_df = pd.DataFrame([{
                    "Open": c["open"], "High": c["high"], "Low": c["low"], "Close": c["close"], "Volume": c["volume"]
                } for c in db_candles], index=pd.to_datetime([c["time"] for c in db_candles]))
            else:
                hist_df = pd.DataFrame()

            execution = optimal_engine.calculate_trade_levels(hist_df, current_price, user_role=user_role)

        entry_min = execution["optimal_entry_min"]
        entry_max = execution["optimal_entry_max"]
        stop_loss = execution["stop_loss"]
        tp1 = execution["take_profit_1"]
        tp2 = execution["take_profit_2"]
        rr_ratio = execution["risk_reward_ratio"]
        setup_pat = execution["setup_pattern"]
        entry_th = execution["entry_thesis"]
        atr_14 = execution.get("atr_14")

        # Pure Mathematical Execution State Determination
        if is_delisted:
            execution_status = "DELISTED_SECURITY"
            status_label = corp_action["status_label"]
            status_color = corp_action["status_color"]
        elif execution.get("execution_status") == "UNVERIFIED_ASSET" or current_price is None or current_price <= 0:
            execution_status = "UNVERIFIED_ASSET"
            status_label = "⚠️ Unverified Asset"
            status_color = "slate"
        elif execution.get("execution_status") == "INSUFFICIENT_HISTORY" or stop_loss is None:
            execution_status = "INSUFFICIENT_HISTORY"
            status_label = "⏳ Insufficient History"
            status_color = "cyan"
        elif execution.get("stage_phase") == "Stage 4 Markdown (Awaiting New Base)":
            execution_status = "WAITING_PULLBACK"
            status_label = "⏳ Awaiting Base Formation"
            status_color = "cyan"
        elif stop_loss is not None and current_price is not None and current_price < stop_loss:
            execution_status = "STOPPED_OUT"
            status_label = "🛑 Below Stop Loss"
            status_color = "rose"
        elif execution.get("execution_status") == "APPROACHING_TARGET" or (tp1 is not None and current_price is not None and current_price >= tp1 * 0.96):
            execution_status = "APPROACHING_TARGET"
            status_label = "🚀 Session ORB Breakout" if is_day_trader else "🚀 Near TP Target"
            status_color = "amber"
        elif entry_min is not None and entry_max is not None and current_price is not None and (entry_min <= current_price <= entry_max * 1.008 or abs(current_price - entry_max) / max(0.01, current_price) <= 0.015):
            execution_status = "IN_BUY_ZONE"
            status_label = "🎯 Active VWAP Bounce" if is_day_trader else "🎯 Near Screening Buy Zone"
            status_color = "emerald"
        else:
            execution_status = "WAITING_PULLBACK"
            status_label = "⏳ Pullback Pending"
            status_color = "cyan"

        # Liquidity Guard (Shadow Observation Mode): Informational execution metadata only; does not mutate frozen decision state
        liq_def = execution.get("liquidity_defense")

        # Deterministic Smart Money & Catalyst Attributes
        if is_delisted:
            confluence_res = {
                "confluenceScore": 0.0,
                "confluenceRating": "Delisted Security / Inactive Listing",
                "badgeColor": "rose",
                "reasons": [corp_action["notes"]],
                "warnings": ["Ceased public trading. Do not trade. Historical record retained for audit integrity."],
            }
            rvol_val = "N/A"
            short_float_val = "N/A"
        elif execution_status == "UNVERIFIED_ASSET" or current_price is None or current_price <= 0:
            confluence_res = {
                "confluenceScore": 0.0,
                "confluenceRating": "Unverified Asset / No Market Data",
                "badgeColor": "slate",
                "reasons": ["Pricing, tape, and regulatory filings unavailable for unverified security."],
                "warnings": ["Do not trade unverified assets. Trade levels suppressed."],
            }
            rvol_val = "N/A"
            short_float_val = "N/A"
        elif execution_status == "INSUFFICIENT_HISTORY":
            confluence_res = {
                "confluenceScore": 0.0,
                "confluenceRating": "Insufficient History (< 50 Bars)",
                "badgeColor": "cyan",
                "reasons": ["Asset has fewer than 50 historical trading sessions on exchange record."],
                "warnings": ["Trade levels and risk geometry suppressed until seasoning threshold met."],
            }
            rvol_val = "N/A"
            short_float_val = "N/A"
        else:
            sec_trades = smart_money_engine.get_sec_insider_trades(sym)
            cong_trades = smart_money_engine.get_congressional_trades(sym)
            has_insider = len(sec_trades) > 0
            has_congress = len(cong_trades) > 0
            insider_val = 0.0
            if has_insider:
                for t in sec_trades:
                    val_raw = t.get("total_value") or t.get("transaction_value") or 0.0
                    if isinstance(val_raw, str):
                        cleaned = val_raw.replace("$", "").replace(",", "").strip()
                        try:
                            insider_val += float(cleaned)
                        except ValueError:
                            pass
                    elif isinstance(val_raw, (int, float)):
                        insider_val += float(val_raw)
            days_to_earn = r.get("days_to_earnings")

            smart_data = {
                "has_insider_buy": has_insider,
                "insider_value_usd": insider_val if has_insider else 0.0,
                "insider_name": sec_trades[0].get("reporting_owner", "") if has_insider else "",
                "has_congress_buy": has_congress,
                "has_options_flow": False,
            } if (has_insider or has_congress) else None

            # Compute multi-factor confluence conviction score
            confluence_res = confluence_engine.calculate_confluence(
                symbol=sym,
                technical_data={
                    "executionStatus": execution_status,
                    "riskRewardRatio": rr_ratio,
                    "setup_pattern": setup_pat,
                    "stage_phase": "Stage 2 Breakout" if execution_status == "IN_BUY_ZONE" else "Institutional Accumulation",
                    "rsi_14": execution.get("rsi_14"),
                    "stop_loss": stop_loss,
                    "current_price": current_price,
                },
                smart_money_data=smart_data,
                fundamental_data={
                    "qualityScore": float(r.get("quality_score")) if r.get("quality_score") is not None else None,
                    "growthScore": float(r.get("growth_score")) if r.get("growth_score") is not None else None,
                    "valuationScore": float(r.get("valuation_score")) if r.get("valuation_score") is not None else None,
                    "piotroski_f": int(r.get("piotroski_f")) if r.get("piotroski_f") is not None else None,
                    "roic": roic_val,
                    "peg": float(r.get("peg_ratio")) if r.get("peg_ratio") is not None else None,
                } if any(r.get(k) is not None for k in ["quality_score", "growth_score", "valuation_score", "piotroski_f", "roic_pct", "peg_ratio"]) else None,
                catalyst_data={
                    "days_to_earnings": days_to_earn,
                } if days_to_earn is not None else None,
                macro_data=shared_macro,
            )

            if not hist_df.empty and len(hist_df) >= 5 and "Volume" in hist_df.columns:
                recent_vol = float(hist_df["Volume"].iloc[-1])
                avg_vol = float(hist_df["Volume"].tail(20).mean())
                if avg_vol > 0:
                    rvol_val = f"{round(recent_vol / avg_vol, 1)}x"
                else:
                    rvol_val = "1.0x"
            else:
                rvol_val = "1.0x"

            short_float_num = r.get("short_float_pct")
            if short_float_num is not None:
                short_float_val = f"{short_float_num}%"
            else:
                short_float_val = "N/A"

        # Derive authentic, machine-readable category taxonomy
        model_name = r.get("expert_model")
        raw_peg = float(r.get("peg_ratio")) if r.get("peg_ratio") is not None else None
        cand_categories = derive_canonical_radar_categories(
            expert_model=model_name,
            peg_ratio=raw_peg,
            roic_pct=roic_val,
            gross_margin_pct=margin_val,
            symbol=sym,
            vcp_matched_symbols=vcp_matched_symbols,
            smart_money_matched_symbols=sm_matched_symbols,
        )
        cand_evidence = classify_candidate_category_evidence(cand_categories)

        # Resolve Canonical Decision State under DecisionHierarchyEngine (Phase 2 Consolidation)
        if is_delisted:
            dec_state = {
                "state": corp_action["decision_state"],
                "label": "Delisted Security — Inactive",
                "isActionable": False,
                "canSizeTrade": False,
                "disqualificationReason": f"Security ceased trading: {corp_action['notes']}",
            }
        else:
            candle_count = len(hist_df) if (hist_df is not None and not hist_df.empty) else 0
            has_fundamentals = bool(
                execution_status != "UNVERIFIED_ASSET"
                and any(r.get(k) is not None for k in ["quality_score", "growth_score", "valuation_score", "piotroski_f", "roic_pct", "peg_ratio"])
            )
            if execution_status == "UNVERIFIED_ASSET" or current_price is None or current_price <= 0:
                freshness_status = "UNAVAILABLE"
            elif execution_status == "INSUFFICIENT_HISTORY" or candle_count < 50:
                freshness_status = "INSUFFICIENT_HISTORY"
            else:
                freshness_status = "LIVE"

            dec_state = DecisionHierarchyEngine.resolve_decision_state(
                symbol=sym,
                current_price=current_price or 0.0,
                candle_count=candle_count,
                freshness_status=freshness_status,
                has_fundamentals=has_fundamentals,
                confluence_score=float(confluence_res["confluenceScore"]),
                stage_phase=execution.get("stage_phase") or setup_pat,
                is_in_buy_zone=(execution_status == "IN_BUY_ZONE"),
                risk_reward_ratio=rr_ratio,
                is_cataloged=True,
                is_confirmed=False,  # Invariant: Screener represents structural discovery only; cannot confirm execution triggers
                user_role=user_role,
                execution_status=execution_status,
                execution_hazard=execution.get("execution_hazard"),
                liquidity_defense=liq_def,
            )

        mapped_candidates.append({
            "ticker": sym,
            "symbol": sym,
            "companyName": r.get("company_name", sym),
            "currentPrice": current_price,
            "priceBasis": price_basis,
            "priceDate": price_date,
            "gemScore": int(r.get("composite_score")) if (r.get("composite_score") is not None and execution_status != "UNVERIFIED_ASSET") else None,
            "expertArchetype": model_name or ("Unverified Asset" if execution_status == "UNVERIFIED_ASSET" else None),
            "categories": cand_categories,
            "categoryEvidence": cand_evidence,
            "roic": f"{roic_val}%" if roic_val is not None else "N/A",
            "pegRatio": str(r.get("peg_ratio")) if r.get("peg_ratio") is not None else "N/A",
            "grossMargin": f"{margin_val}%" if margin_val is not None else "N/A",
            "atr14": f"${atr_14:.2f}" if (atr_14 is not None and execution_status not in ["UNVERIFIED_ASSET", "INSUFFICIENT_HISTORY", "DELISTED_SECURITY"]) else "N/A",
            "rvol": rvol_val,
            "shortFloat": short_float_val,
            "dayTraderSetup": "Pricing and intraday VWAP tape unavailable." if execution_status == "UNVERIFIED_ASSET" else ("Trading suspended for delisted security." if is_delisted else "Intraday momentum trend continuation above 5m VWAP anchor with defined ATR risk."),
            "thesis": r.get("investment_thesis", "Disclosures and market data unavailable for unverified security." if execution_status == "UNVERIFIED_ASSET" else ("High relative volume momentum with clear intraday VWAP risk definition." if is_day_trader else "High return on capital with strong free cash flows.")),
            "catalyst": r.get("primary_catalyst", "Awaiting verified disclosures." if execution_status == "UNVERIFIED_ASSET" else ("Intraday institutional flow breakout." if is_day_trader else "Product cycle expansion and margin gains.")),
            "riskLevel": "Unverified Risk" if execution_status == "UNVERIFIED_ASSET" else ("Delisted / Closed Position" if is_delisted else ("High Volatility (Intraday)" if is_day_trader else r.get("risk_rating", "Low-to-Medium Risk"))),
            # Execution Scanner Levels & Structural Geometry
            "executionStatus": execution_status,
            "statusLabel": status_label,
            "statusColor": status_color,
            "screeningStatus": "DELISTED" if is_delisted else ("SCREENING_ZONE" if execution_status == "IN_BUY_ZONE" else execution_status),
            "screeningGeometry": "INACTIVE" if is_delisted else ("WITHIN_TOLERANCE" if execution_status == "IN_BUY_ZONE" else "OUTSIDE_TOLERANCE"),
            "screeningEligible": False if is_delisted else bool(execution_status in ["IN_BUY_ZONE", "APPROACHING_TARGET"]),
            "corporateAction": corp_action,
            "fundamentalEvidence": {
                "provenance": "STATIC_REFERENCE_CATALOG" if sym in screener.KNOWN_GEMS_DATA else "UNVERIFIED",
                "isCatalogReference": bool(sym in screener.KNOWN_GEMS_DATA),
                "metricBasis": "HISTORICAL_PRE_ACQUISITION_CATALOG" if is_delisted else ("HISTORICAL_CATALOG_VALUE" if sym in screener.KNOWN_GEMS_DATA else "MISSING"),
                "asOfDate": corp_action["effective_date"] if is_delisted else ("CURATED_CATALOG_BASELINE" if sym in screener.KNOWN_GEMS_DATA else None),
                "listingStatus": corp_action["status"] if is_delisted else (ATTRIBUTABLE_LIFECYCLE_EVIDENCE[sym]["status"] if sym in ATTRIBUTABLE_LIFECYCLE_EVIDENCE else "UNVERIFIED"),
                "lifecycleEvidenceSource": corp_action.get("action_type") if is_delisted else (ATTRIBUTABLE_LIFECYCLE_EVIDENCE[sym].get("source") if sym in ATTRIBUTABLE_LIFECYCLE_EVIDENCE else None),
                "isAudited": False,
                "qualificationState": "INELIGIBLE_INACTIVE" if is_delisted else ("REFERENCE_CRITERIA_MET" if "VALUE_GARP" in cand_categories else ("REFERENCE_CRITERIA_UNMET" if sym in screener.KNOWN_GEMS_DATA else "UNASSESSED_INSUFFICIENT_EVIDENCE")),
                "sortingBasis": "TECHNICAL_CONFLUENCE_MOMENTUM" if not is_delisted and execution_status != "UNVERIFIED_ASSET" else "UNAVAILABLE",
            },
            "optimalEntryMin": entry_min,
            "optimalEntryMax": entry_max,
            "stopLoss": stop_loss,
            "stopLossPct": round(((stop_loss - current_price) / current_price) * 100, 1) if (stop_loss is not None and current_price is not None and current_price > 0) else None,
            "takeProfit1": tp1,
            "takeProfit1Pct": round(((tp1 - current_price) / current_price) * 100, 1) if (tp1 is not None and current_price is not None and current_price > 0) else None,
            "takeProfit2": tp2,
            "takeProfit2Pct": round(((tp2 - current_price) / current_price) * 100, 1) if (tp2 is not None and current_price is not None and current_price > 0) else None,
            "riskRewardRatio": rr_ratio,
            "setupPattern": setup_pat,
            "entryThesis": entry_th,
            # Canonical Decision Authority Fields (Discovery-Only: Execution reserved for DecisionTrace)
            "decisionState": dec_state["state"],
            "decisionStateLabel": "Discovery Candidate — Analyze for Trigger" if dec_state["state"] == "ACTIONABLE_SETUP" else dec_state["label"],
            "isActionable": False,  # Screener cannot declare trade actionability; execution requires DecisionTrace
            "canSizeTrade": False,  # Sizing disabled in discovery screener
            "allowedActions": [] if is_delisted else (["RESEARCH_PROFILE"] if execution_status == "UNVERIFIED_ASSET" else ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"]),
            "disqualificationReason": dec_state.get("disqualificationReason") or "Discovery candidate: Open in Analysis hub to evaluate canonical DecisionTrace.",
            "decisionContextId": f"dec-radar-{sym}-{int(time.time() * 1000)}",
            "macroContextId": shared_macro_context_id,
            # Confluence Conviction Score & Position Sizing
            "confluenceScore": confluence_res["confluenceScore"],
            "confluenceRating": confluence_res["confluenceRating"],
            "confluenceBadgeColor": confluence_res["badgeColor"],
            "confluenceReasons": confluence_res["reasons"],
            "confluenceWarnings": confluence_res["warnings"],
            "liquidityDefense": liq_def,
        })

    # Apply Selected Filter dynamically based on numerical thresholds
    non_tradable = ["UNVERIFIED_ASSET", "DELISTED_SECURITY"]
    if filter_type in ["in_buy_zone", "vwap_pullback"]:
        filtered = [c for c in mapped_candidates if c["executionStatus"] == "IN_BUY_ZONE"]
    elif filter_type in ["approaching_target", "orb_breakout"]:
        filtered = [c for c in mapped_candidates if c["executionStatus"] == "APPROACHING_TARGET"]
    elif filter_type == "high_rr":
        filtered = [
            c for c in mapped_candidates
            if c["executionStatus"] not in non_tradable
            and c.get("riskRewardRatio") is not None and c["riskRewardRatio"] >= 2.0 and (c["executionStatus"] == "IN_BUY_ZONE" or (c.get("optimalEntryMax") is not None and c["currentPrice"] is not None and c["currentPrice"] <= c["optimalEntryMax"] * 1.02))
        ]
    elif filter_type == "high_confluence":
        filtered = [c for c in mapped_candidates if c["confluenceScore"] >= 80.0 and c["executionStatus"] not in non_tradable]
    elif filter_type == "high_rvol":
        filtered = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable and c["rvol"] != "N/A" and float(c["rvol"].replace("x", "")) >= 2.5]
    elif filter_type == "squeeze":
        filtered = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable and c["shortFloat"] != "N/A" and float(c["shortFloat"].replace("%", "")) >= 6.0]
    elif filter_type == "lynch":
        filtered = [
            c for c in mapped_candidates
            if c["executionStatus"] not in non_tradable
            and ((c.get("pegRatio") not in [None, "N/A"] and 0 < float(c["pegRatio"]) <= 1.05) or (c.get("expertArchetype") and ("Lynch" in c["expertArchetype"] or "GARP" in c["expertArchetype"])))
            and "VALUE_GARP" in c["categories"]
        ]
    elif filter_type == "greenblatt":
        filtered = [
            c for c in mapped_candidates
            if c["executionStatus"] not in non_tradable
            and (
                (c.get("roic") not in [None, "N/A"] and float(c["roic"].replace("%", "")) >= 28.0)
                or (c.get("expertArchetype") and ("Greenblatt" in c["expertArchetype"] or "Magic" in c["expertArchetype"]))
            )
        ]
    elif filter_type == "rule_breakers":
        filtered = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable and (c.get("grossMargin") not in [None, "N/A"] and float(c["grossMargin"].replace("%", "")) >= 65.0)]
    elif filter_type == "value_garp":
        filtered = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable and "VALUE_GARP" in c["categories"]]
    elif filter_type == "vcp":
        filtered = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable and "VCP" in c["categories"]]
    elif filter_type == "smart_money":
        filtered = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable and "SMART_MONEY" in c["categories"]]
    else:
        filtered = mapped_candidates

    radar_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    radar_evidence = MarketEvidence(
        provenance=MarketProvenance(
            provider=Provider.YFINANCE,
            ingestion_source=IngestionSource.DIRECT_PROVIDER,
            observed_at=None,
            observed_date=time.strftime("%Y-%m-%d", time.gmtime()),
            observation_precision=ObservationPrecision.DATE,
            observation_source=ObservationSource.DERIVED_FROM_TRADE_DATE,
            ingested_at=radar_now,
            adjustment_state=AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,
            structural_quality=StructuralQuality.COMPLETE if filtered else StructuralQuality.UNKNOWN,
            fallback_status=False,
        ),
        serving_source=ServingSource.DIRECT_PROVIDER,
        cache_origin=CacheOrigin.NONE,
        served_at=radar_now,
        candle_count=len(filtered),
    )

    # Derive canonical candidate population model per ARX Closure Model v2 Gate B
    reference_count = len(active_universe)
    active_candidates = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable]
    opportunity_candidates = [c for c in active_candidates if c.get("screeningEligible")]
    delisted_candidates = [c for c in mapped_candidates if c["executionStatus"] == "DELISTED_SECURITY"]
    unverified_candidates = [c for c in mapped_candidates if c["executionStatus"] == "UNVERIFIED_ASSET"]

    population_model = {
        "referenceUniverseCount": reference_count,
        "activeScreeningCount": len(active_candidates),
        "currentOpportunityCount": len(opportunity_candidates),
        "delistedRecordCount": len(delisted_candidates),
        "unverifiedRecordCount": len(unverified_candidates),
    }

    return {
        "totalCandidates": reference_count,
        "gemsFound": len(filtered),
        "activeFilter": filter_type,
        "userRole": user_role,
        "capabilities": get_radar_capabilities(vcp_snapshot=vcp_snapshot, sm_snapshot=sm_snapshot),
        "macroContextId": shared_macro_context_id,
        "macroDifficulty": shared_macro,
        "population": population_model,
        "candidates": filtered,
        "results": results,
        "evidence": radar_evidence.to_dict(),
    }


@router.get("/vcp/snapshot")
def get_vcp_snapshot():
    """Retrieve active immutable snapshot envelope for Minervini VCP scanner (Section 9 & 18)."""
    return vcp_scanner_runner.get_active_or_latest_snapshot()


@router.get("/smart-money/snapshot")
def get_smart_money_snapshot():
    """Retrieve status envelope for Smart Money scanner (Section 9 & 18; returns active snapshot or PIPELINE_PENDING)."""
    runner = get_smart_money_scanner_runner()
    return runner.get_active_or_latest_snapshot()


@router.post("/smart-money/scan")
def trigger_smart_money_scan(
    response: Response = None,
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    x_operator_request_id: Optional[str] = Header(None, alias="X-Operator-Request-Id"),
    as_of_watermark: Optional[str] = None,
    universe_build_id: Optional[str] = None,
):
    """
    Operator-only trigger for Smart Money Form 4 market-wide scan (Gate B5 & B6).
    Rejects public unauthenticated requests. Enforces distributed lease & idempotency.
    In PostgreSQL profile: Enqueues an asynchronous run request in sm_run_requests (HTTP 202).
    In SQLite profile: Executes local synchronous scan.
    """
    if not verify_operator_token(x_operator_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="PUBLIC_MARKET_WIDE_SCAN_FORBIDDEN: Operator authorization required to trigger market-wide scanner.",
        )

    if get_smart_money_storage_backend() == "postgres":
        repo = get_smart_money_repository()
        subject_hash = hashlib.sha256(
            f"{as_of_watermark}:{universe_build_id}:{x_operator_request_id}".encode("utf-8")
        ).hexdigest()
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash=subject_hash,
        )
        if response is not None and hasattr(response, "status_code"):
            response.status_code = status.HTTP_202_ACCEPTED
        return {
            "status": "ACCEPTED",
            "request_id": req_id,
            "request_type": "MARKET_WIDE_FORM4_SCAN",
            "submitted_at": datetime.utcnow().isoformat() + "Z",
            "mode": "ASYNCHRONOUS_QUEUE",
        }

    runner = get_smart_money_scanner_runner()
    try:
        return runner.execute_market_wide_scan(
            as_of_watermark=as_of_watermark,
            universe_build_id=universe_build_id,
            operator_request_id=x_operator_request_id,
        )
    except RuntimeError as e:
        if "OVERLAPPING_SMART_MONEY_SCANS_PROHIBITED" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        raise


@router.post("/vcp/scan")
def trigger_vcp_scan(
    x_operator_token: Optional[str] = Header(None, alias="X-Operator-Token"),
    x_operator_request_id: Optional[str] = Header(None, alias="X-Operator-Request-Id"),
    universe_build_id: Optional[str] = None,
    logical_job_key: Optional[str] = None,
):
    """
    Operator-only trigger for Minervini VCP market-wide scan (Section 16).
    Rejects public unauthenticated requests. Enforces distributed lease & idempotency.
    """
    if not verify_operator_token(x_operator_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="PUBLIC_MARKET_WIDE_SCAN_FORBIDDEN: Operator authorization required to trigger market-wide scanner.",
        )
    try:
        return vcp_scanner_runner.execute_market_wide_scan(
            universe_build_id=universe_build_id,
            operator_request_id=x_operator_request_id,
            logical_job_key=logical_job_key,
        )
    except RuntimeError as e:
        if "OVERLAPPING_VCP_SCANS_PROHIBITED" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        raise
    except Exception as e:
        if "FENCED_REJECTED" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        raise


@router.post("/vcp/natural-scan")
def trigger_natural_vcp_scan(
    payload: Optional[VCPNaturalScanRequest] = None,
    x_scheduler_token: Optional[str] = Header(None, alias="X-Scheduler-Token"),
    x_scheduler_job_id: Optional[str] = Header(None, alias="X-Scheduler-Job-Id"),
    x_scheduler_event_id: Optional[str] = Header(None, alias="X-Scheduler-Event-Id"),
):
    """
    Authoritative natural production trigger for Minervini VCP market-wide EOD scans.
    Disabled by default: requires VCP_NATURAL_SCHEDULER_ENABLED=1.
    Protected by constant-time X-Scheduler-Token authentication against dedicated VCP_SCHEDULER_SECRET.
    Invokes NaturalVCPTriggerService with invocation_class=SCHEDULED_PRODUCTION and candidate_generation_id=CANDIDATE_GENERATION_004.
    """
    if not is_vcp_natural_scheduler_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NATURAL_SCHEDULER_DISABLED: VCP natural scheduling entry point is disabled in current deployment.",
        )
    if not verify_scheduler_token(x_scheduler_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="SCHEDULER_AUTHENTICATION_FAILED: Valid X-Scheduler-Token required for natural trigger execution.",
        )

    from analyst_dashboard.vcp.natural_trigger import get_natural_vcp_trigger_service
    from analyst_dashboard.vcp.session_guard import evaluate_natural_session_eligibility

    req_date = payload.scheduled_date if payload else None
    req_build = payload.universe_build_id if payload else None
    eff_job = (payload.scheduler_job_id if payload and payload.scheduler_job_id else None) or x_scheduler_job_id
    eff_event = (payload.scheduler_event_id if payload and payload.scheduler_event_id else None) or x_scheduler_event_id

    # Step 3: Server-side exchange-session eligibility & settlement window enforcement
    # Evaluates against authorized published executable universe (INV-U01, INV-U02, INV-U04)
    universe_store = getattr(vcp_scanner_runner, "universe_store", None)
    eligibility = evaluate_natural_session_eligibility(
        session_date=req_date,
        check_data_freshness=True,
        universe_symbols=None,
        universe_build_id=req_build,
        universe_store=universe_store,
        market_db=vcp_scanner_runner.market_db,
    )
    if not eligibility.is_eligible:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": "REJECTED",
                "reason": eligibility.reason,
                "session_date": eligibility.session_date,
                "earliest_execution_utc": eligibility.earliest_execution_utc,
                "latest_execution_utc": eligibility.latest_execution_utc,
                "coverage_ratio": eligibility.coverage_ratio,
                "detail": eligibility.detail,
            },
        )

    trigger_svc = get_natural_vcp_trigger_service()

    try:
        return trigger_svc.trigger_natural_scan(
            scheduled_date=req_date,
            universe_build_id=eligibility.universe_build_id or req_build,
            scheduler_job_id=eff_job,
            scheduler_event_id=eff_event,
        )
    except RuntimeError as e:
        if "OVERLAPPING_VCP_SCANS_PROHIBITED" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        raise
    except Exception as e:
        if "FENCED_REJECTED" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        raise


@router.get("/vcp/coordination")
def get_vcp_coordination_status():
    """Retrieve non-sensitive run coordination status, active lease, and recent lifecycle events (Section 25)."""
    res_key = vcp_scanner_runner.coordinator.resource_key
    active_lease = vcp_scanner_runner.coordinator.active_lease
    return {
        "resource_key": res_key,
        "coordination_epoch": vcp_scanner_runner.coordinator.store.get_current_epoch(res_key),
        "active_lease": active_lease.to_provenance() if active_lease else None,
        "recent_events": vcp_scanner_runner.coordinator.store.get_lease_events(res_key, limit=10),
    }



@router.get("/position-size")
def calculate_trade_position_size(
    account_equity: float = 25000.0,
    risk_pct: float = 1.0,
    entry_price: float = 100.0,
    stop_loss: float = 95.0,
    take_profit_1: float = 105.0,
    is_actionable: bool = True,
    can_size_trade: bool = True,
    execution_hazard: Optional[str] = None,
    adv_20d: Optional[float] = None,
):
    """Interactive Position Sizer calculating exact share quantities, risk limit, and Kelly allocation."""
    hazard_bool = None
    if execution_hazard is not None:
        hazard_bool = execution_hazard.lower() in ("true", "1", "yes")
    return ConfluenceEngine.calculate_position_size(
        account_equity=account_equity,
        risk_pct=risk_pct,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        is_actionable=is_actionable,
        can_size_trade=can_size_trade,
        execution_hazard=hazard_bool,
        adv_20d=adv_20d,
    )


@router.get("/phase26/validation-report")
def get_phase26_validation_report():
    """Returns Phase 26 prospective liquidity validation report covering Experiment 26-A and 26-B."""
    from analyst_dashboard.governance.liquidity_validation import Phase26ValidationEngine
    return Phase26ValidationEngine.generate_phase26_validation_report()
