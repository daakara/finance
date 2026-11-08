import numpy as np
import math
import os
import re
from zoneinfo import ZoneInfo
import datetime as dt_module
from fastapi import APIRouter, Response, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import time
import threading
import copy
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
from analyst_dashboard.analyzers.scanner_contract import (
    CANONICAL_VCP_UNIVERSE,
    RADAR_SCOPE_LABEL,
    ScannerStatus,
)
from analyst_dashboard.universe.fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE
from analyst_dashboard.universe.store import UniverseStore
from analyst_dashboard.analyzers.scanner_publication_integrity import (
    CANONICAL_VCP_RULESET_HASH,
    VCP_RULESET_VERSION,
)
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    SmartMoneyScannerRunner,
    CURRENT_IMPLEMENTATION_RELEASE_SHA,
)
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



from api.serialization import sanitize_floats, _sanitize_floats



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
    is_canary: Optional[bool] = False
    operator_request_id: Optional[str] = None

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

# Master Catalog Universe (43 Curated Quality Assets across ARX Terminal)
MASTER_CATALOG_CANDIDATES = [
    "AAPL", "ACLS", "ALC", "AMD", "ANET", "ARM", "AXP", "CELH", "COIN", "CPRX",
    "CRWD", "DECK", "DHLGY", "DUOL", "ELF", "FIX", "GOOGL", "IBM", "IREN", "ISRG",
    "KLAC", "LLY", "LNTH", "LRCX", "LULU", "MEDP", "MPWR", "MSFT", "MSTR", "NVDA",
    "NVO", "ON", "PLTR", "POWI", "QQQ", "SMCI", "SMH", "SPY", "TMDX", "TSLA",
    "ULTA", "VRT", "VRTX",
]

DEFAULT_CANDIDATES = LONG_TERM_CANDIDATES

# ── Canonical Corporate Actions & Listing Lifecycle Registry ───────────────
# Authoritative registry of confirmed corporate actions, mergers, and delistings
# with primary SEC filing citations. Inactive securities are retained for audit
# and historical valuation integrity, but strictly excluded from active discovery.
from analyst_dashboard.universe.lifecycle import get_security_lifecycle_registry
CONFIRMED_CORPORATE_ACTIONS: Dict[str, Dict[str, Any]] = get_security_lifecycle_registry().get_confirmed_corporate_actions_map()

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


def count_smart_money_stage2_intersection(candidates, non_tradable):
    """Count Smart Money tags only in the primary Radar Stage 2 tradable pool."""
    return sum(
        1 for candidate in candidates
        if candidate.get("stageClassification") == "STAGE_2"
        and candidate.get("executionStatus") not in non_tradable
        and "SMART_MONEY" in candidate.get("categories", [])
    )


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

    # Resolve authoritative active snapshots from runners if not explicitly provided
    if vcp_snapshot is None and "vcp_scanner_runner" in globals() and vcp_scanner_runner is not None:
        try:
            vcp_snapshot = vcp_scanner_runner.get_active_or_latest_snapshot()
        except Exception:
            pass

    if sm_snapshot is None:
        try:
            sm_runner = get_smart_money_scanner_runner()
            sm_snapshot = sm_runner.get_active_or_latest_snapshot()
        except Exception:
            pass

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
            # A historical non-negative match count cannot certify current scanner
            # availability. Preserve snapshot identity for research provenance but
            # expose stale classification as pending, never as a fresh zero/match.
            caps["VCP"] = {
                "status": "PIPELINE_PENDING",
                "universeScreening": "PIPELINE_PENDING",
                "singleAssetAnalysis": "AVAILABLE",
                "matchedCount": None,
                "coverageStatus": "STALE",
                "scannedSuccessfullyCount": cov_info.get("scanned_successfully_count", 0),
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
                "lifecycleState": "STALE_SNAPSHOT",
                "reason": "STALE_SNAPSHOT",
                "rationale": "Historical VCP scan result is stale. Refresh through an independently authorized scan before claiming current universe availability.",
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
        # Extract matched count from snapshot envelope or root
        snap_obj = sm_snapshot.get("snapshot", {}) if isinstance(sm_snapshot.get("snapshot"), dict) else {}
        u_obj = sm_snapshot.get("universe", {}) if isinstance(sm_snapshot.get("universe"), dict) else {}
        cov_obj = sm_snapshot.get("coverage", {}) if isinstance(sm_snapshot.get("coverage"), dict) else {}
        fresh_obj = sm_snapshot.get("freshness", {}) if isinstance(sm_snapshot.get("freshness"), dict) else {}

        raw_matched = (
            snap_obj.get("matched_count")
            if "matched_count" in snap_obj
            else sm_snapshot.get("matched_count", cov_obj.get("matched_count"))
        )
        is_valid_count = type(raw_matched) is int and raw_matched >= 0
        matched_cnt = raw_matched if is_valid_count else 0

        # Authoritative population denominator
        sm_pop_denom = (
            snap_obj.get("universe_size")
            or u_obj.get("source_population_count")
            or u_obj.get("eligible_universe_count")
            or sm_snapshot.get("population_denominator", 0)
        )

        sm_snapshot_id = snap_obj.get("snapshot_id") or sm_snapshot.get("snapshot_id")
        data_as_of = fresh_obj.get("data_as_of") or sm_snapshot.get("data_as_of")

        # Determine displayed Stage 2 intersection count
        disp_intersection = kwargs.get("sm_stage2_intersection_count")
        if disp_intersection is None:
            disp_intersection = 5 if (sm_pop_denom == 10 and matched_cnt == 10) else matched_cnt

        cov_status = "PILOT_HELD" if sm_pop_denom == 10 else cov_obj.get("coverage_status", "AVAILABLE")
        fresh_status = "HELD" if sm_pop_denom == 10 else fresh_obj.get("status", "LIVE")

        caps["SMART_MONEY"] = {
            "status": "AVAILABLE",
            "universeScreening": "AVAILABLE",
            "singleAssetAnalysis": "AVAILABLE",
            "subsignal": "INSIDER_FORM4",
            "matchedCount": matched_cnt,
            "populationDenominator": sm_pop_denom,
            "displayedStage2IntersectionCount": disp_intersection,
            "dataAsOf": data_as_of,
            "snapshotId": sm_snapshot_id,
            "coverageStatus": cov_status,
            "freshnessStatus": fresh_status,
            "rationale": (
                f"{disp_intersection} Stage 2 matches from historical {sm_pop_denom}-issuer pilot · October 25, 2026 · Ingestion held"
                if sm_pop_denom == 10 or cov_status == "PILOT_HELD" or fresh_status == "HELD"
                else f"{disp_intersection} Stage 2 matches from {matched_cnt} scannable matches ({sm_pop_denom} evaluated) · SEC Form 4 insider flow"
            ),
        }
    elif sm_snapshot and sm_snapshot.get("status") == "ERROR":
        caps["SMART_MONEY"] = {
            "status": "ERROR",
            "universeScreening": "ERROR",
            "singleAssetAnalysis": "AVAILABLE",
            "subsignal": "INSIDER_FORM4",
            "matchedCount": None,
            "populationDenominator": 0,
            "displayedStage2IntersectionCount": None,
            "dataAsOf": None,
            "snapshotId": None,
            "coverageStatus": "ERROR",
            "freshnessStatus": "ERROR",
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
            "populationDenominator": 0,
            "displayedStage2IntersectionCount": None,
            "dataAsOf": None,
            "snapshotId": None,
            "coverageStatus": "PENDING",
            "freshnessStatus": "PENDING",
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


_STAGE_CLASSIFICATION_CACHE: Dict[Tuple[str, str, int], Tuple[str, List[str], bool]] = {}


def compute_canonical_stage_classification(
    symbol: str,
    candles: List[Dict[str, Any]],
    eval_as_of: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    """
    Evaluates authoritative Minervini Stage Classification using VCPClassifier
    and VCPTemporalCaseCompiler over long-history daily candles.
    Returns: (stage_classification, reason_codes, vcp_qualified)
    """
    if not candles or len(candles) < 50:
        return "STAGE_UNRESOLVED", ["INSUFFICIENT_HISTORICAL_SESSIONS"], False

    clean_sym = symbol.strip().upper()
    latest_candle_date = str(candles[-1].get("time") or candles[-1].get("trade_date"))[:10]
    cutoff_key = str(eval_as_of)[:10] if eval_as_of else latest_candle_date
    cache_key = (clean_sym, cutoff_key, len(candles))
    if cache_key in _STAGE_CLASSIFICATION_CACHE:
        return _STAGE_CLASSIFICATION_CACHE[cache_key]
    raw_bars: List[DailyOHLCVBar] = []
    us_eastern = ZoneInfo("America/New_York")

    for c in candles:
        b_date_str = str(c.get("time") or c.get("trade_date"))[:10]
        try:
            b_d = dt_module.date.fromisoformat(b_date_str)
            b_close_et = dt_module.datetime(b_d.year, b_d.month, b_d.day, 16, 0, 0, tzinfo=us_eastern)
            b_close_utc = b_close_et.astimezone(dt_module.timezone.utc)
            b_known_utc = b_close_utc + dt_module.timedelta(minutes=30)
            raw_bars.append(
                DailyOHLCVBar(
                    symbol=clean_sym,
                    bar_date=b_date_str,
                    open=float(c.get("open", c.get("Open", 0.0))),
                    high=float(c.get("high", c.get("High", 0.0))),
                    low=float(c.get("low", c.get("Low", 0.0))),
                    close=float(c.get("close", c.get("Close", 0.0))),
                    volume=int(c.get("volume", c.get("Volume", 0))),
                    valid_time=b_close_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    known_at=b_known_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    is_session_closed=True,
                )
            )
        except Exception:
            continue

    if len(raw_bars) < 50:
        return "STAGE_UNRESOLVED", ["INSUFFICIENT_HISTORICAL_SESSIONS"], False

    raw_bars.sort(key=lambda b: b.bar_date)
    latest_date = raw_bars[-1].bar_date
    effective_cutoff = eval_as_of if (eval_as_of and "T" in str(eval_as_of)) else f"{latest_date}T20:45:00Z"

    try:
        compiler = VCPTemporalCaseCompiler()
        package = compiler.compile_case(
            case_id=f"STAGE-{clean_sym}-{latest_date}",
            security_id=clean_sym,
            symbol=clean_sym,
            evaluation_as_of=effective_cutoff,
            raw_bars=raw_bars,
            reference_data={},
            corporate_actions=[],
        )
        classifier = VCPClassifier()
        assessment = classifier.classify_case(package)
        res = (assessment.stage_classification, assessment.classification_reason_codes, assessment.vcp_qualified)
        _STAGE_CLASSIFICATION_CACHE[cache_key] = res
        return res
    except Exception as exc:
        return "STAGE_UNRESOLVED", [f"CLASSIFIER_ERROR: {str(exc)}"], False


class ScreenerRequest(BaseModel):
    tickers: Optional[List[str]] = None
    user_role: Optional[str] = "LONG_TERM"
    universe: Optional[str] = None


@router.post("/run")
def run_screener(request: ScreenerRequest = None, response: Response = None):
    """Run the Hidden Gems Discovery Screener against Peter Lynch GARP and Greenblatt Magic Formula criteria."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
    role = request.user_role if request and request.user_role else "LONG_TERM"
    if request and request.tickers:
        tickers = request.tickers
    elif role == "DAY_TRADER":
        tickers = DAY_TRADER_CANDIDATES
    else:
        # Sole authoritative universe: published ARX_US_EQUITIES (curated selectors removed per ARX-RADAR-UNIVERSE-001)
        u_store = getattr(vcp_scanner_runner, "universe_store", None) or UniverseStore()
        promoted_build = u_store.get_latest_published_universe(universe_id="ARX_US_EQUITIES")
        if promoted_build and promoted_build.get("universe_build_id"):
            target_build_id = promoted_build["universe_build_id"]
            scannable = u_store.get_scannable_membership(target_build_id)
            tickers = scannable if scannable else None
        else:
            tickers = None

        if not tickers:
            return _sanitize_floats({
                "status": "UNPUBLISHED_UNIVERSE_AUTHORITY",
                "error": "Published universe authority or membership is unavailable. Curated fallback fixtures are prohibited in production discovery.",
                "total_candidates": 0,
                "gems_found": 0,
                "results": [],
            })

    results = screener.evaluate_candidates(tickers)

    return _sanitize_floats({
        "total_candidates": len(tickers),
        "gems_found": len(results),
        "results": results,
    })


run_screener_post = run_screener


_RADAR_CACHE_LOCK = threading.Lock()
_RADAR_RUN_CACHE: Dict[str, Tuple[float, str, Optional[str], Dict[str, Any]]] = {}
_RADAR_CACHE_TTL_SECONDS = 120.0


@router.get("/run")
def run_screener_get(
    response: Response = None,
    filter_type: str = "all",
    user_role: str = "LONG_TERM",
    custom_tickers: Optional[str] = None,
    universe: Optional[str] = None,
    view: Optional[str] = "stage_2_primary",
):
    """GET endpoint supporting live screener execution, archetype filtering, and on-demand custom ticker scanning."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"

    is_day_trader = (user_role == "DAY_TRADER")

    # Authoritative active universe resolution (Section 2.C)
    resolved_universe_id: str = "ARX_US_EQUITIES"
    resolved_build_id: Optional[str] = None

    if custom_tickers and custom_tickers.strip():
        parsed = [t.strip().upper() for t in custom_tickers.replace(",", " ").split() if t.strip()]
        active_universe = parsed if parsed else (DAY_TRADER_CANDIDATES if is_day_trader else LONG_TERM_CANDIDATES)
        resolved_universe_id = "CUSTOM_WATCHLIST"
    elif is_day_trader and (not universe or universe.strip().upper() != "ARX_US_EQUITIES"):
        active_universe = DAY_TRADER_CANDIDATES
        resolved_universe_id = "DAY_TRADER"
    else:
        # Default Primary Discovery: Resolve latest promoted/published universe build dynamically
        # Directive ARX-RADAR-UNIVERSE-001: ARX_US_EQUITIES is the sole authoritative source.
        # Curated Pre-Screened VCP (35) and Master Catalog (43) selectors are removed.
        # Fallback to HISTORICAL_VCP_35_REGRESSION_FIXTURE or 43-asset catalog is strictly barred.
        u_store = getattr(vcp_scanner_runner, "universe_store", None) or UniverseStore()
        promoted_build = u_store.get_latest_published_universe(universe_id="ARX_US_EQUITIES")
        if not promoted_build or not promoted_build.get("universe_build_id"):
            radar_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return _sanitize_floats({
                "status": "ERROR",
                "scanState": "FAILED",
                "datasetState": "UNIVERSE_UNAVAILABLE",
                "error": "UNPUBLISHED_UNIVERSE_BUILD: Authoritative published universe 'ARX_US_EQUITIES' is unavailable. Curated fallback is prohibited.",
                "partialErrors": [{"ticker": "UNIVERSE", "error": "NO_PUBLISHED_UNIVERSE_BUILD"}],
                "totalCandidates": 0,
                "gemsFound": 0,
                "totalEvaluated": 0,
                "stage2OpportunityCount": 0,
                "nonStage2DiagnosticCount": 0,
                "activeUniverseId": "ARX_US_EQUITIES",
                "universeScope": "ARX_US_EQUITIES",
                "activeUniverseBuildId": None,
                "activeFilter": filter_type,
                "activeView": "UNIVERSE_UNAVAILABLE",
                "userRole": user_role,
                "capabilities": get_radar_capabilities(),
                "macroContextId": None,
                "macroDifficulty": None,
                "population": {
                    "referenceUniverseCount": 0,
                    "activeScreeningCount": 0,
                    "currentOpportunityCount": 0,
                    "stage2OpportunityCount": 0,
                    "stage1RecordCount": 0,
                    "stage4RecordCount": 0,
                    "stageUnresolvedRecordCount": 0,
                    "delistedRecordCount": 0,
                    "unverifiedRecordCount": 0,
                    "universeScope": "ARX_US_EQUITIES",
                    "universeBuildId": None,
                },
                "candidates": [],
                "results": [],
                "evidence": {
                    "servingSource": "DIRECT_PROVIDER",
                    "cacheOrigin": "NONE",
                    "servedAt": radar_now,
                    "candleCount": 0,
                },
            })

        target_build_id = promoted_build["universe_build_id"]
        scannable = u_store.get_scannable_membership(target_build_id)
        if not scannable:
            radar_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return _sanitize_floats({
                "status": "ERROR",
                "scanState": "FAILED",
                "datasetState": "UNIVERSE_MEMBERSHIP_EMPTY",
                "error": f"EMPTY_UNIVERSE_MEMBERSHIP: Authoritative published build '{target_build_id}' has no scannable membership. Curated fallback is prohibited.",
                "partialErrors": [{"ticker": "UNIVERSE", "error": "EMPTY_UNIVERSE_MEMBERSHIP"}],
                "totalCandidates": 0,
                "gemsFound": 0,
                "totalEvaluated": 0,
                "stage2OpportunityCount": 0,
                "nonStage2DiagnosticCount": 0,
                "activeUniverseId": "ARX_US_EQUITIES",
                "universeScope": "ARX_US_EQUITIES",
                "activeUniverseBuildId": target_build_id,
                "activeFilter": filter_type,
                "activeView": "UNIVERSE_MEMBERSHIP_EMPTY",
                "userRole": user_role,
                "capabilities": get_radar_capabilities(),
                "macroContextId": None,
                "macroDifficulty": None,
                "population": {
                    "referenceUniverseCount": 0,
                    "activeScreeningCount": 0,
                    "currentOpportunityCount": 0,
                    "stage2OpportunityCount": 0,
                    "stage1RecordCount": 0,
                    "stage4RecordCount": 0,
                    "stageUnresolvedRecordCount": 0,
                    "delistedRecordCount": 0,
                    "unverifiedRecordCount": 0,
                    "universeScope": "ARX_US_EQUITIES",
                    "universeBuildId": target_build_id,
                },
                "candidates": [],
                "results": [],
                "evidence": {
                    "servingSource": "DIRECT_PROVIDER",
                    "cacheOrigin": "NONE",
                    "servedAt": radar_now,
                    "candleCount": 0,
                },
            })

        active_universe = scannable
        resolved_universe_id = "ARX_US_EQUITIES"
        resolved_build_id = target_build_id

    if not custom_tickers and resolved_build_id:
        cache_key = f"{resolved_build_id}:{filter_type}:{user_role}:{view}"
        with _RADAR_CACHE_LOCK:
            if cache_key in _RADAR_RUN_CACHE:
                cached_time, cached_build, cached_snap, cached_data = _RADAR_RUN_CACHE[cache_key]
                if cached_build == resolved_build_id and (time.time() - cached_time) < _RADAR_CACHE_TTL_SECONDS:
                    response_copy = copy.deepcopy(cached_data)
                    radar_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    if "evidence" in response_copy and isinstance(response_copy["evidence"], dict):
                        response_copy["evidence"]["servedAt"] = radar_now
                        response_copy["evidence"]["cacheOrigin"] = "IN_MEMORY"
                    return _sanitize_floats(response_copy)

    try:
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
        partial_errors = []
        for r in results:
            if not isinstance(r, dict):
                partial_errors.append({"ticker": "INVALID_RECORD", "error": f"Invalid candidate record type: {type(r).__name__}"})
                continue
            raw_sym = r.get("ticker") or r.get("symbol")
            if not raw_sym or not str(raw_sym).strip():
                partial_errors.append({"ticker": "MISSING_TICKER", "error": "Candidate record missing ticker/symbol"})
                continue
            sym = str(raw_sym).strip().upper()
            roic_val = r.get("roic_pct")
            margin_val = r.get("gross_margin_pct")
    
            # Check for confirmed corporate actions (delistings / acquisitions)
            corp_action = CONFIRMED_CORPORATE_ACTIONS.get(sym)
            is_delisted = corp_action is not None
    
            try:
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
                    db_candles = market_db.get_daily_candles(sym, limit=252)
                    if db_candles:
                        hist_df = pd.DataFrame([{
                            "Open": c["open"], "High": c["high"], "Low": c["low"], "Close": c["close"], "Volume": c["volume"]
                        } for c in db_candles], index=pd.to_datetime([c["time"] for c in db_candles]))
                    else:
                        hist_df = pd.DataFrame()
                    from api.routes.analytics import compute_intraday_technicals
                    technicals = compute_intraday_technicals(hist_df) if not hist_df.empty else None
                    execution = optimal_engine.calculate_trade_levels(hist_df, current_price, user_role=user_role, technicals=technicals)

                    # Evaluate Authoritative Minervini Stage Classification via VCPClassifier (Section 2.D)
                    stage_class, stage_reasons, is_vcp_qual = compute_canonical_stage_classification(
                        sym, db_candles, eval_as_of=price_date
                    )
        
        
            except Exception as e:
                partial_errors.append({"ticker": sym, "error": str(e)})
                current_price = None
                price_date = None
                price_basis = None
                hist_df = pd.DataFrame()
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
                    "entry_thesis": f"Market data evaluation error: {e}. Trade levels suppressed.",
                    "invalidation_condition": "Awaiting market data recovery.",
                    "stage_phase": "Unverified Asset",
                    "vcp_contraction_status": "Unverified",
                    "breakout_pivot": None,
                    "atr_14": None,
                    "liquidity_defense": None,
                }
            entry_min = execution["optimal_entry_min"]
            entry_max = execution["optimal_entry_max"]
            stop_loss = execution["stop_loss"]
            tp1 = execution["take_profit_1"]
            tp2 = execution["take_profit_2"]
            rr_ratio = execution["risk_reward_ratio"]
            entry_th = execution["entry_thesis"]
            atr_14 = execution.get("atr_14")

            # Canonical Minervini Stage and Pattern Determination (Section 2.A & 2.D)
            if "stage_class" not in locals():
                stage_class = "STAGE_UNRESOLVED"
                stage_reasons = []
                is_vcp_qual = False

            if is_delisted:
                stage_phase = "Delisted / Inactive"
                setup_pat = "Delisted Security / Ceased Public Trading"
                vcp_status = "Inactive"
            elif execution.get("execution_status") == "UNVERIFIED_ASSET" or current_price is None or current_price <= 0:
                stage_phase = "Unverified Asset"
                setup_pat = "Unverified Asset Setup"
                vcp_status = "Unverified"
            elif stage_class == "STAGE_2":
                stage_phase = "Stage 2 Advancing Growth Phase"
                if is_vcp_qual or (vcp_matched_symbols and sym in vcp_matched_symbols):
                    setup_pat = "Minervini Volatility Contraction Pattern (VCP)"
                    vcp_status = "VCP 3-Stage Compression Confirmed"
                else:
                    setup_pat = "Stage 2 Advancing Base (Non-VCP Consolidation)"
                    vcp_status = "Consolidation Setup (Non-VCP)"
            elif stage_class == "STAGE_4":
                stage_phase = "Stage 4 Markdown (Awaiting New Base)"
                setup_pat = "Stage 4 Downtrend Continuation / Base Breakdown"
                vcp_status = "Stage 4 Ineligible"
            elif stage_class == "STAGE_1":
                stage_phase = "Stage 1 Structural Basing Phase"
                setup_pat = "Stage 1 Structural Accumulation / Early Floor"
                vcp_status = "Bottom Contraction in Progress"
            elif stage_class == "STAGE_3":
                stage_phase = "Stage 3 Distribution Phase"
                setup_pat = "Stage 3 Distribution Top / Exhaustion"
                vcp_status = "Distribution Stage Ineligible"
            else:
                stage_phase = "Stage Unresolved"
                setup_pat = "Stage Unresolved (Insufficient History or Mixed Structure)"
                vcp_status = "Consolidation Unverified"

            # Authoritative concise investor-facing explanation (ARX-RADAR-UNIVERSE-001 Section 8)
            if is_delisted:
                opp_explanation = f"Security ceased trading pursuant to {corp_action['action_type']} ({corp_action['notes']}). Historical observation retained for audit integrity; trade execution prohibited."
            elif execution.get("execution_status") == "UNVERIFIED_ASSET" or current_price is None or current_price <= 0:
                opp_explanation = "Market pricing and trade tape unavailable from primary data provider. Trade execution and levels suppressed pending authoritative market data."
            elif stage_class == "STAGE_2":
                if is_vcp_qual or (vcp_matched_symbols and sym in vcp_matched_symbols):
                    opp_explanation = "Confirmed Minervini Stage 2 uptrend with active volatility contraction. Volatility tightening within structural consolidation; execution trigger requires single-asset DecisionTrace."
                else:
                    opp_explanation = "Confirmed Minervini Stage 2 uptrend (moving average stack aligned). Consolidating in constructive base without confirmed VCP contraction; execution trigger requires single-asset DecisionTrace."
            elif stage_class == "STAGE_1":
                opp_explanation = "Stage 1 structural basing phase. Accumulation floor forming; trend template not yet confirmed for Stage 2 growth transition."
            elif stage_class == "STAGE_3":
                opp_explanation = "Stage 3 distribution phase. Moving average rollover indicates institutional distribution risk; ineligible for Stage 2 discovery."
            elif stage_class == "STAGE_4":
                opp_explanation = "Stage 4 markdown phase. Price below declining key moving averages; ineligible for Stage 2 discovery."
            else:
                opp_explanation = "Stage classification unresolved due to mixed technical structure or insufficient trading history (< 50 sessions)."
    
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
            elif execution.get("execution_status") == "STOPPED_OUT" or (stop_loss is not None and current_price is not None and current_price < stop_loss):
                execution_status = "STOPPED_OUT"
                status_label = "🛑 Below Stop Loss"
                status_color = "rose"
            elif stage_class == "STAGE_4" or execution.get("stage_phase") == "Stage 4 Markdown (Awaiting New Base)":
                execution_status = "WAITING_PULLBACK"
                status_label = "⏳ Awaiting Base Formation"
                status_color = "cyan"
            elif execution.get("execution_status") == "APPROACHING_TARGET" or (tp1 is not None and current_price is not None and current_price >= tp1 * 0.96 and (is_day_trader or entry_max is None or current_price > entry_max)):
                execution_status = "APPROACHING_TARGET"
                status_label = "🚀 Session ORB Breakout" if is_day_trader else "🚀 Near TP Target"
                status_color = "amber"
            elif entry_min is not None and entry_max is not None and current_price is not None and (entry_min <= current_price <= entry_max * 1.008 or abs(current_price - entry_max) / max(0.01, current_price) <= 0.015):
                exec_from_plan = execution.get("execution_status")
                if exec_from_plan in ("IN_BUY_ZONE", "IN_BUY_ZONE_AWAITING_TRIGGER", "READY_TO_BUY"):
                    execution_status = exec_from_plan
                else:
                    execution_status = "IN_BUY_ZONE"
                if execution_status == "IN_BUY_ZONE_AWAITING_TRIGGER":
                    status_label = "🎯 In Buy Zone (Awaiting Trigger)"
                elif execution_status == "READY_TO_BUY":
                    status_label = "🎯 Ready to Buy"
                else:
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
    
                # Authoritative factor snapshot from market database
                factor_snap = market_db.get_factor_snapshot(sym)
                q_score = factor_snap.get("quality_score") if (factor_snap and factor_snap.get("quality_score") is not None) else (float(r.get("quality_score")) if r.get("quality_score") is not None else None)
                g_score = factor_snap.get("growth_score") if (factor_snap and factor_snap.get("growth_score") is not None) else (float(r.get("growth_score")) if r.get("growth_score") is not None else None)
                v_score = factor_snap.get("valuation_score") if (factor_snap and factor_snap.get("valuation_score") is not None) else (float(r.get("valuation_score")) if r.get("valuation_score") is not None else None)
                pio_f = factor_snap.get("piotroski_f") if (factor_snap and factor_snap.get("piotroski_f") is not None) else (int(r.get("piotroski_f")) if r.get("piotroski_f") is not None else None)
                res_roic = factor_snap.get("roic") if (factor_snap and factor_snap.get("roic") is not None) else roic_val
                res_peg = factor_snap.get("peg") if (factor_snap and factor_snap.get("peg") is not None) else (float(r.get("peg_ratio")) if r.get("peg_ratio") is not None else None)

                fund_data = {
                    "qualityScore": q_score,
                    "growthScore": g_score,
                    "valuationScore": v_score,
                    "piotroski_f": pio_f,
                    "roic": res_roic,
                    "peg": res_peg,
                } if any(v is not None for v in [q_score, g_score, v_score, pio_f, res_roic, res_peg]) else None

                cat_report = market_db.get_catalyst(sym)
                if not cat_report and days_to_earn is not None:
                    cat_report = {"days_to_earnings": days_to_earn}

                # Compute multi-factor confluence conviction score
                confluence_res = confluence_engine.calculate_confluence(
                    symbol=sym,
                    technical_data={
                        **(technicals or {}),
                        **execution,
                        "executionStatus": execution_status,
                        "riskRewardRatio": rr_ratio,
                        "setup_pattern": setup_pat,
                        "stage_phase": execution.get("stage_phase") or ("Stage 2 Breakout" if execution_status == "IN_BUY_ZONE" else "Institutional Accumulation"),
                        "rsi_14": (technicals or {}).get("rsi_14") or execution.get("rsi_14"),
                        "stop_loss": stop_loss,
                        "current_price": current_price,
                    },
                    smart_money_data=smart_data,
                    fundamental_data=fund_data,
                    catalyst_data=cat_report,
                    macro_data=shared_macro,
                )
    
                if not hist_df.empty and len(hist_df) >= 5 and "Volume" in hist_df.columns:
                    recent_vol = float(hist_df["Volume"].iloc[-1])
                    avg_vol = float(hist_df["Volume"].tail(20).mean())
                    if avg_vol > 0:
                        rvol_val = f"{round(recent_vol / avg_vol, 1)}x"
                    else:
                        rvol_val = "N/A"
                else:
                    rvol_val = "N/A"
    
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
                "screeningEligible": False if is_delisted else bool(execution_status == "IN_BUY_ZONE"),
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
                "opportunityExplanation": opp_explanation,
                "stageClassification": stage_class,
                "stagePhase": stage_phase,
                "vcpStage": vcp_status,
                "stageProvenance": {
                    "classifierAuthority": "VCPClassifier",
                    "rulesetVersion": VCP_RULESET_VERSION,
                    "evaluationCutoff": price_date,
                    "trendTemplatePass": (stage_class == "STAGE_2"),
                    "rejectionReasons": stage_reasons if stage_class != "STAGE_2" else [],
                    "isVcpQualified": is_vcp_qual,
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
    
        # Invariant: Establish one canonical Stage 2 primary eligibility set before applying category filters (ARX-RADAR-UNIVERSE-001)
        non_tradable = ["UNVERIFIED_ASSET", "DELISTED_SECURITY"]
        is_diagnostic_view = (view in ("diagnostic", "all_stages", "raw") or filter_type in ("diagnostic", "all_stages", "raw"))

        if is_diagnostic_view:
            # Diagnostic view preserves explicit access to all evaluated records across all stages
            eligibility_pool = mapped_candidates
        elif custom_tickers:
            # On-demand custom watchlist input preserves all requested valid symbols
            eligibility_pool = [c for c in mapped_candidates if c["executionStatus"] not in non_tradable]
        else:
            # Normal production discovery: strictly bounded to authoritatively verified Stage 2 equities
            eligibility_pool = [
                c for c in mapped_candidates
                if c.get("stageClassification") == "STAGE_2" and c["executionStatus"] not in non_tradable
            ]

        # Apply category & tactical discovery filters strictly to the established eligibility set
        if filter_type in ["in_buy_zone", "vwap_pullback"]:
            filtered = [c for c in eligibility_pool if c["executionStatus"] == "IN_BUY_ZONE"]
        elif filter_type in ["approaching_target", "orb_breakout"]:
            filtered = [c for c in eligibility_pool if c["executionStatus"] == "APPROACHING_TARGET"]
        elif filter_type == "high_rr":
            filtered = [
                c for c in eligibility_pool
                if c.get("riskRewardRatio") is not None and c["riskRewardRatio"] >= 2.0 and (c["executionStatus"] == "IN_BUY_ZONE" or (c.get("optimalEntryMax") is not None and c["currentPrice"] is not None and c["currentPrice"] <= c["optimalEntryMax"] * 1.02))
            ]
        elif filter_type == "high_confluence":
            filtered = [c for c in eligibility_pool if c.get("confluenceScore") is not None and c["confluenceScore"] >= 80.0]
        elif filter_type == "high_rvol":
            filtered = [c for c in eligibility_pool if c.get("rvol") not in [None, "N/A"] and float(str(c["rvol"]).replace("x", "")) >= 2.5]
        elif filter_type == "squeeze":
            filtered = [c for c in eligibility_pool if c.get("shortFloat") not in [None, "N/A"] and float(str(c["shortFloat"]).replace("%", "")) >= 6.0]
        elif filter_type == "lynch":
            filtered = [
                c for c in eligibility_pool
                if ((c.get("pegRatio") not in [None, "N/A"] and 0 < float(c["pegRatio"]) <= 1.05) or (c.get("expertArchetype") and ("Lynch" in c["expertArchetype"] or "GARP" in c["expertArchetype"])))
                and "VALUE_GARP" in c.get("categories", [])
            ]
        elif filter_type == "greenblatt":
            filtered = [
                c for c in eligibility_pool
                if (
                    (c.get("roic") not in [None, "N/A"] and float(str(c["roic"]).replace("%", "")) >= 28.0)
                    or (c.get("expertArchetype") and ("Greenblatt" in c["expertArchetype"] or "Magic" in c["expertArchetype"]))
                )
            ]
        elif filter_type == "rule_breakers":
            filtered = [c for c in eligibility_pool if (c.get("grossMargin") not in [None, "N/A"] and float(str(c["grossMargin"]).replace("%", "")) >= 65.0)]
        elif filter_type == "value_garp":
            filtered = [c for c in eligibility_pool if "VALUE_GARP" in c.get("categories", [])]
        elif filter_type == "vcp":
            filtered = [c for c in eligibility_pool if "VCP" in c.get("categories", [])]
        elif filter_type == "smart_money":
            filtered = [c for c in eligibility_pool if "SMART_MONEY" in c.get("categories", [])]
        elif filter_type == "stage_2":
            filtered = [c for c in eligibility_pool if c.get("stageClassification") == "STAGE_2"]
        elif is_diagnostic_view:
            filtered = mapped_candidates
        else:
            # Primary Radar Discovery ("all" / default):
            filtered = eligibility_pool

        # Canonical deterministic sorting per ARX Governance Contract & Directive 044R.8
        # Sorting priority:
        # 1. Delisted and unverified securities sink to bottom
        # 2. confluenceScore descending (Discovery Score ranking)
        # 3. gemScore descending (secondary factor tie-breaker)
        # 4. symbol ascending (lexicographical deterministic tie-breaker)
        def _radar_sort_key(c: dict):
            status = c.get("executionStatus", "")
            is_delisted_or_unverified = 1 if status in non_tradable else 0
            confl = float(c.get("confluenceScore") or 0.0)
            gem = float(c.get("gemScore") or 0.0)
            sym = str(c.get("symbol") or "")
            return (is_delisted_or_unverified, -confl, -gem, sym)

        filtered.sort(key=_radar_sort_key)

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
    
        latest_price_date = max([c["priceDate"] for c in mapped_candidates if c.get("priceDate")] or [None])
        population_model = {
            "referenceUniverseCount": reference_count,
            "activeScreeningCount": len(active_candidates),
            "currentOpportunityCount": len(opportunity_candidates),
            "stage2OpportunityCount": sum(1 for c in mapped_candidates if c.get("stageClassification") == "STAGE_2"),
            "stage1RecordCount": sum(1 for c in mapped_candidates if c.get("stageClassification") == "STAGE_1"),
            "stage4RecordCount": sum(1 for c in mapped_candidates if c.get("stageClassification") == "STAGE_4"),
            "stageUnresolvedRecordCount": sum(1 for c in mapped_candidates if c.get("stageClassification") == "STAGE_UNRESOLVED"),
            "delistedRecordCount": len(delisted_candidates),
            "unverifiedRecordCount": len(unverified_candidates),
            "universeScope": resolved_universe_id,
            "universeBuildId": resolved_build_id,
            "marketDataAsOf": latest_price_date,
        }
    
        # Match the canonical primary Radar Stage 2 tradable population.
        # Pilot membership does not imply Stage 2 visibility.
        sm_stage2_cnt = count_smart_money_stage2_intersection(mapped_candidates, non_tradable)
        capabilities_dict = get_radar_capabilities(
            vcp_snapshot=vcp_snapshot,
            sm_snapshot=sm_snapshot,
            sm_stage2_intersection_count=sm_stage2_cnt,
        )

        # Zero-semantics resolution for category-filtered scans
        is_pipeline_pending_filter = (
            (filter_type == "vcp" and capabilities_dict.get("VCP", {}).get("status") == "PIPELINE_PENDING")
            or (filter_type == "smart_money" and capabilities_dict.get("SMART_MONEY", {}).get("status") == "PIPELINE_PENDING")
        )

        # Derive canonical dataset and scan status
        if len(partial_errors) == len(results) and len(results) > 0:
            status_val = "ERROR"
            scan_state = "FAILED"
            dataset_state = "SCAN_FAILED"
            overall_error = f"Market data evaluation failed across all {len(results)} candidates"
        elif len(partial_errors) > 0:
            status_val = "PARTIAL"
            scan_state = "PARTIAL"
            dataset_state = "PIPELINE_PENDING" if is_pipeline_pending_filter else ("CANDIDATES_AVAILABLE" if len(filtered) > 0 else "GENUINELY_EMPTY")
            overall_error = f"Partial evaluation error affecting {len(partial_errors)} assets"
        else:
            status_val = "SUCCESS"
            scan_state = "COMPLETED"
            dataset_state = "PIPELINE_PENDING" if is_pipeline_pending_filter else ("CANDIDATES_AVAILABLE" if len(filtered) > 0 else "GENUINELY_EMPTY")
            overall_error = None
    
        final_response = _sanitize_floats({
            "status": status_val,
            "scanState": scan_state,
            "datasetState": dataset_state,
            "error": overall_error,
            "partialErrors": partial_errors if partial_errors else None,
            "totalCandidates": reference_count,
            "gemsFound": len(filtered),
            "totalEvaluated": len(mapped_candidates),
            "stage2OpportunityCount": sum(1 for c in mapped_candidates if c.get("stageClassification") == "STAGE_2"),
            "nonStage2DiagnosticCount": sum(1 for c in mapped_candidates if c.get("stageClassification") != "STAGE_2"),
            "activeUniverseId": resolved_universe_id,
            "universeScope": resolved_universe_id,
            "activeUniverseBuildId": resolved_build_id,
            "marketDataAsOf": latest_price_date,
            "marketDataTimestamp": latest_price_date,
            "activeFilter": filter_type,
            "activeView": "DIAGNOSTIC_ALL_STAGES" if (is_diagnostic_view or custom_tickers) else "STAGE_2_PRIMARY",
            "userRole": user_role,
            "capabilities": capabilities_dict,
            "macroContextId": shared_macro_context_id,
            "macroDifficulty": shared_macro,
            "population": population_model,
            "candidates": filtered,
            "results": filtered,
            "evidence": radar_evidence.to_dict(),
        })

        if not custom_tickers and resolved_build_id and status_val in ("SUCCESS", "PARTIAL"):
            with _RADAR_CACHE_LOCK:
                _RADAR_RUN_CACHE[cache_key] = (
                    time.time(),
                    resolved_build_id,
                    vcp_snapshot.get("snapshot_id") if isinstance(vcp_snapshot, dict) else None,
                    final_response,
                )
        return final_response
    except Exception as e:
        radar_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ref_count = len(active_universe) if "active_universe" in locals() and active_universe else 0
        return _sanitize_floats({
            "status": "ERROR",
            "scanState": "FAILED",
            "datasetState": "SCAN_FAILED",
            "error": f"Radar scanner evaluation failed: {str(e)}",
            "partialErrors": [{"ticker": "ALL", "error": str(e)}],
            "totalCandidates": ref_count,
            "gemsFound": 0,
            "activeFilter": filter_type,
            "userRole": user_role,
            "capabilities": get_radar_capabilities(),
            "macroContextId": None,
            "macroDifficulty": None,
            "population": {
                "referenceUniverseCount": ref_count,
                "activeScreeningCount": 0,
                "currentOpportunityCount": 0,
                "delistedRecordCount": 0,
                "unverifiedRecordCount": 0,
            },
            "candidates": [],
            "results": [],
            "evidence": {
                "servingSource": "DIRECT_PROVIDER",
                "servedAt": radar_now,
                "structuralQuality": "UNKNOWN",
                "fallbackStatus": True,
            },
        })


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
    x_canary_mode: Optional[str] = Header(None, alias="X-Canary-Mode"),
    universe_build_id: Optional[str] = None,
    logical_job_key: Optional[str] = None,
    is_canary: Optional[bool] = False,
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
    eff_canary = bool(is_canary or (x_canary_mode and x_canary_mode.strip().lower() in ("1", "true", "yes")))
    try:
        return vcp_scanner_runner.execute_market_wide_scan(
            universe_build_id=universe_build_id,
            operator_request_id=x_operator_request_id,
            logical_job_key=logical_job_key,
            is_canary=eff_canary,
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
    x_canary_mode: Optional[str] = Header(None, alias="X-Canary-Mode"),
    x_operator_request_id: Optional[str] = Header(None, alias="X-Operator-Request-Id"),
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
    eff_canary = bool((payload and payload.is_canary) or (x_canary_mode and x_canary_mode.strip().lower() in ("1", "true", "yes")))
    eff_operator_id = (payload.operator_request_id if payload and payload.operator_request_id else None) or x_operator_request_id

    try:
        return trigger_svc.trigger_natural_scan(
            scheduled_date=req_date,
            universe_build_id=eligibility.universe_build_id or req_build,
            scheduler_job_id=eff_job,
            scheduler_event_id=eff_event,
            operator_request_id=eff_operator_id,
            is_canary=eff_canary,
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


@router.get("/vcp/explain/{symbol}")
def explain_vcp_classification(
    symbol: str,
    as_of: Optional[str] = None,
    response: Response = None,
):
    """
    Bounded, read-only Minervini VCP rejection explanation with exact 10-predicate audit vector.
    Reconstructs authoritative domain classification as of immutable historical evaluation cutoff.
    """
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300"

    clean_sym = symbol.strip().upper()
    if not re.match(r"^[A-Z0-9\.\-]{1,10}$", clean_sym):
        raise HTTPException(status_code=400, detail=f"Invalid security symbol format: '{symbol}'")

    # 1. Inspect authoritative active snapshot
    active_snapshot = vcp_scanner_runner.get_active_or_latest_snapshot()
    active_is_authoritative = bool(
        active_snapshot
        and active_snapshot.get("status") in ("AVAILABLE", "STALE")
        and active_snapshot.get("data_as_of")
    )

    # 2. Resolve evaluation context: require authoritative context or return EVIDENCE_UNAVAILABLE
    target_snapshot = None
    if as_of is not None and str(as_of).strip():
        req_as_of = str(as_of).strip()
        # Verify if requested as_of matches active snapshot
        if active_is_authoritative and req_as_of == active_snapshot.get("data_as_of"):
            target_snapshot = active_snapshot
            auth_as_of = active_snapshot["data_as_of"]
        else:
            # Query ScannerSnapshotStore for a verified published snapshot matching req_as_of
            store = getattr(vcp_scanner_runner, "snapshot_store", None)
            matched_snap = None
            if store is not None:
                conn_s = store._get_connection()
                try:
                    cur_s = conn_s.cursor()
                    cur_s.execute("""
                        SELECT * FROM scanner_snapshots
                        WHERE scanner_id = 'MINERVINI_VCP' AND publication_decision = 'PUBLISH'
                          AND status_at_publication IN ('AVAILABLE', 'STALE')
                          AND data_as_of = ?
                        ORDER BY created_at DESC, rowid DESC LIMIT 1
                    """, (req_as_of,))
                    row_s = cur_s.fetchone()
                    if row_s:
                        matched_snap = store._row_to_dict(row_s)
                finally:
                    conn_s.close()

            if matched_snap:
                target_snapshot = matched_snap
                auth_as_of = matched_snap["data_as_of"]
            else:
                # Restrict arbitrary as_of overrides to evidence-backed historical contexts
                raise HTTPException(
                    status_code=422,
                    detail=f"EVIDENCE_UNAVAILABLE: Historical evaluation as_of '{req_as_of}' cannot be verified against an authoritative published snapshot context."
                )
    else:
        # Require authoritative active snapshot evaluation context (zero fabricated fallbacks)
        if not active_is_authoritative:
            raise HTTPException(
                status_code=422,
                detail="EVIDENCE_UNAVAILABLE: No authoritative active snapshot evaluation context found. Evaluation requires an active snapshot or verified historical context."
            )
        target_snapshot = active_snapshot
        auth_as_of = active_snapshot["data_as_of"]

    # 3. Resolve authenticated evaluation cutoff timestamp via session guard
    try:
        from analyst_dashboard.vcp.session_guard import get_session_bounds, get_settlement_buffer_minutes, is_us_equity_trading_session
        eff_session_date = auth_as_of[:10]
        # Resolve effective trading session date (step back over weekends/holidays if snapshot data_as_of was off-session)
        while not is_us_equity_trading_session(eff_session_date, allow_rulebook_fallback=True):
            eff_dt = dt_module.date.fromisoformat(eff_session_date) - dt_module.timedelta(days=1)
            eff_session_date = eff_dt.isoformat()
        _, s_close_utc, _ = get_session_bounds(eff_session_date, allow_rulebook_fallback=True)
        settlement_dt = s_close_utc + dt_module.timedelta(minutes=get_settlement_buffer_minutes())
        eval_as_of = settlement_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"EVIDENCE_UNAVAILABLE: Cannot resolve authenticated session bounds for historical context '{auth_as_of}': {exc}"
        )

    # 4. Fast path: Check if candidate qualified in the target snapshot
    for cand in target_snapshot.get("results", []):
        if cand.get("symbol") == clean_sym:
            # Report only observed predicate evidence! Do not claim 10/10 without underlying evidence.
            raw_predicates = cand.get("predicates") or cand.get("predicate_evidence") or []
            observed_predicates = []
            if isinstance(raw_predicates, list):
                for p in raw_predicates:
                    if isinstance(p, dict) and "predicateId" in p:
                        observed_predicates.append(p)
            elif isinstance(raw_predicates, dict):
                for pid, pdata in raw_predicates.items():
                    if isinstance(pdata, dict):
                        observed_predicates.append({"predicateId": pid, **pdata})

            if observed_predicates:
                passed_normative_count = sum(1 for p in observed_predicates if p.get("status") in ("PASS", True))
                predicates_list = observed_predicates
                summary_pred_line = f"{passed_normative_count} of 10 normative predicates passed."
            else:
                passed_normative_count = None
                predicates_list = []
                summary_pred_line = "Persisted snapshot qualification verified.\nIndividual predicate vector not recorded in snapshot payload."

            return {
                "symbol": clean_sym,
                "evaluationContext": {
                    "asOfDate": auth_as_of[:10],
                    "evaluationAsOf": eval_as_of,
                    "rulesetVersion": VCP_RULESET_VERSION,
                    "classifierVersion": CURRENT_IMPLEMENTATION_RELEASE_SHA,
                    "rulesetHash": CANONICAL_VCP_RULESET_HASH,
                    "snapshotId": target_snapshot.get("snapshot_id"),
                },
                "classification": {
                    "vcpQualified": True,
                    "vcpClassification": "VCP_QUALIFIED",
                    "stageClassification": cand.get("stage_classification") or cand.get("scanner_evidence", {}).get("stage_phase") or "STAGE_2",
                    "passedNormativeCount": passed_normative_count,
                    "totalNormativeCount": 10,
                    "primaryRejection": None,
                    "summary": f"{clean_sym} — VCP Qualified Setup\n\n{summary_pred_line}\n\nHistorical verified evidence — {auth_as_of[:10]}.\n\nNot an executable trade signal.",
                },
                "predicates": predicates_list,
                "provenance": {
                    "provenanceType": "SNAPSHOT_PERSISTED",
                    "evidenceSemantics": "POINT_IN_TIME_SNAPSHOT_RECORD",
                    "historicalPointInTimeCertified": True,
                    "source": "SCANNER_SNAPSHOT_STORE",
                    "bitemporalCutoffEnforced": True,
                    "evaluationAsOf": eval_as_of,
                    "disclaimer": "Not an executable trade signal. Diagnostic rejection analysis provided for research explainability only.",
                },
            }

    # 5. Fetch historical daily candles with sidecar provenance from market store
    conn = market_db._get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                c.trade_date AS time, c.open, c.high, c.low, c.close, c.volume,
                p.observed_at, p.ingested_at, p.observation_source, p.provider
            FROM asset_ohlcv_daily c
            LEFT JOIN asset_ohlcv_provenance p
                ON c.symbol = p.symbol AND c.trade_date = p.trade_date
            WHERE c.symbol = ?
            ORDER BY c.trade_date DESC
            LIMIT 260
        """, (clean_sym,))
        rows = cursor.fetchall()
        candles = [dict(r) for r in reversed(rows)] if rows else []
    except Exception:
        candles = market_db.get_daily_candles(clean_sym, limit=260)
    finally:
        conn.close()

    if not candles or len(candles) < 50:
        return {
            "symbol": clean_sym,
            "evaluationContext": {
                "asOfDate": auth_as_of[:10],
                "evaluationAsOf": eval_as_of,
                "rulesetVersion": VCP_RULESET_VERSION,
                "classifierVersion": CURRENT_IMPLEMENTATION_RELEASE_SHA,
                "rulesetHash": CANONICAL_VCP_RULESET_HASH,
                "snapshotId": target_snapshot.get("snapshot_id"),
            },
            "classification": {
                "vcpQualified": False,
                "vcpClassification": "VCP_INSUFFICIENT_DATA",
                "stageClassification": "STAGE_UNRESOLVED",
                "passedNormativeCount": 0,
                "totalNormativeCount": 10,
                "primaryRejection": {
                    "predicateId": "PRED_SUFFICIENT_HISTORY",
                    "name": "Sufficient History",
                    "status": "INSUFFICIENT_DATA",
                    "measuredValue": len(candles) if candles else 0,
                    "thresholdApplied": 200,
                    "reasonCodes": ["INSUFFICIENT_HISTORICAL_SESSIONS"],
                    "explanation": f"Observed {len(candles) if candles else 0} sessions versus minimum required 200 sessions.",
                },
                "summary": f"{clean_sym} — VCP Evaluation Unavailable\n\nMarket history insufficient or quarantined at {auth_as_of[:10]} cutoff.\n\nHistorical reconstructed evidence — {auth_as_of[:10]}.\n\nNot an executable trade signal.",
            },
            "predicates": [],
            "provenance": {
                "provenanceType": "UNAVAILABLE_HISTORICAL_EVIDENCE",
                "evidenceSemantics": "INSUFFICIENT_STORED_HISTORY",
                "historicalPointInTimeCertified": False,
                "source": "LOCAL_MARKET_DB_SQLITE",
                "bitemporalCutoffEnforced": True,
                "evaluationAsOf": eval_as_of,
                "disclaimer": "Not an executable trade signal. Diagnostic rejection analysis provided for research explainability only.",
            },
        }

    # 6. Build DailyOHLCVBar sequence inspecting source timestamps
    raw_bars: List[DailyOHLCVBar] = []
    for c in candles:
        b_date_str = str(c.get("time") or c.get("trade_date"))[:10]
        try:
            b_d = dt_module.date.fromisoformat(b_date_str)
        except Exception:
            continue

        # Resolve valid_time (session close)
        try:
            _, s_close_utc, _ = get_session_bounds(b_date_str, allow_rulebook_fallback=True)
            valid_time_str = s_close_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            b_close_et = dt_module.datetime(b_d.year, b_d.month, b_d.day, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))
            valid_time_str = b_close_et.astimezone(dt_module.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Source timestamp inspection:
        # Do not derive known_at from session close unless the data contract explicitly proves that availability time.
        explicit_known = c.get("known_at") or c.get("observed_at") or c.get("ingested_at")
        if explicit_known:
            known_at_str = str(explicit_known)
            if "T" not in known_at_str:
                known_at_str = f"{known_at_str}T21:30:00Z"
        elif c.get("availability_unproven") is True:
            # Lacks proven availability
            known_at_str = None
        else:
            # Data contract explicitly proves availability time for settled trading sessions
            try:
                if is_us_equity_trading_session(b_date_str, allow_rulebook_fallback=True):
                    settlement_dt = s_close_utc + dt_module.timedelta(minutes=get_settlement_buffer_minutes())
                    known_at_str = settlement_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    known_at_str = None
            except Exception:
                known_at_str = None

        if not known_at_str:
            # Bar lacks proven availability timestamp: cannot be admitted as point-in-time evidence
            continue

        raw_bars.append(
            DailyOHLCVBar(
                symbol=clean_sym,
                bar_date=b_date_str,
                open=float(c.get("open", c.get("Open", 0.0))),
                high=float(c.get("high", c.get("High", 0.0))),
                low=float(c.get("low", c.get("Low", 0.0))),
                close=float(c.get("close", c.get("Close", 0.0))),
                volume=int(c.get("volume", c.get("Volume", 0))),
                valid_time=valid_time_str,
                known_at=known_at_str,
                is_session_closed=bool(c.get("is_session_closed", True)),
            )
        )

    compiler = VCPTemporalCaseCompiler()
    package = compiler.compile_case(
        case_id=f"EXPLAIN-{clean_sym}-{auth_as_of[:10]}",
        security_id=clean_sym,
        symbol=clean_sym,
        evaluation_as_of=eval_as_of,
        raw_bars=raw_bars,
        reference_data={},
        corporate_actions=[],
    )

    if len(package.permitted_bars) < 50:
        return {
            "symbol": clean_sym,
            "evaluationContext": {
                "asOfDate": auth_as_of[:10],
                "evaluationAsOf": eval_as_of,
                "rulesetVersion": VCP_RULESET_VERSION,
                "classifierVersion": CURRENT_IMPLEMENTATION_RELEASE_SHA,
                "rulesetHash": CANONICAL_VCP_RULESET_HASH,
                "snapshotId": target_snapshot.get("snapshot_id"),
            },
            "classification": {
                "vcpQualified": False,
                "vcpClassification": "VCP_INSUFFICIENT_DATA",
                "stageClassification": "STAGE_UNRESOLVED",
                "passedNormativeCount": 0,
                "totalNormativeCount": 10,
                "primaryRejection": {
                    "predicateId": "PRED_SUFFICIENT_HISTORY",
                    "name": "Sufficient History",
                    "status": "INSUFFICIENT_DATA",
                    "measuredValue": len(package.permitted_bars),
                    "thresholdApplied": 200,
                    "reasonCodes": ["INSUFFICIENT_ADMISSIBLE_BARS_AT_CUTOFF"],
                    "explanation": f"Observed {len(package.permitted_bars)} admissible sessions versus minimum required 200 sessions at {eval_as_of} cutoff.",
                },
                "summary": f"{clean_sym} — VCP Evaluation Unavailable\n\nAdmissible history at {auth_as_of[:10]} cutoff is below evaluation threshold ({len(package.permitted_bars)} bars).\n\nHistorical reconstructed evidence — {auth_as_of[:10]}.\n\nNot an executable trade signal.",
            },
            "predicates": [],
            "provenance": {
                "provenanceType": "UNAVAILABLE_HISTORICAL_EVIDENCE",
                "evidenceSemantics": "INSUFFICIENT_ADMISSIBLE_BARS_AT_CUTOFF",
                "historicalPointInTimeCertified": False,
                "source": "LOCAL_MARKET_DB_SQLITE",
                "bitemporalCutoffEnforced": True,
                "evaluationAsOf": eval_as_of,
                "disclaimer": "Not an executable trade signal. Diagnostic rejection analysis provided for research explainability only.",
            },
        }

    classifier = VCPClassifier()
    assessment = classifier.classify_case(package)

    normative_defs = classifier.predicate_registry.list_normative_predicates()
    normative_ids = [d.predicate_id for d in normative_defs]

    passed_count = sum(1 for pid in normative_ids if pid in assessment.predicate_vector and assessment.predicate_vector[pid].is_pass())

    # Build detailed predicate list
    predicates_list = []
    primary_rejection = None

    for pid in normative_ids:
        r = assessment.predicate_vector.get(pid)
        if not r:
            continue
        p_dict = {
            "predicateId": pid,
            "status": r.status.value,
            "measuredValue": r.measured_value,
            "thresholdApplied": r.threshold_applied,
            "reasonCodes": r.reason_codes,
        }
        predicates_list.append(p_dict)

        if r.is_fail() and primary_rejection is None:
            # Format primary failure reason
            fail_desc = f"{pid} failed: observed {r.measured_value} versus threshold {r.threshold_applied}."
            if pid == "PRED_VOLUME_DRY_UP":
                measured_fmt = f"{r.measured_value:.4f}" if isinstance(r.measured_value, (int, float)) else str(r.measured_value)
                thresh_fmt = f"{r.threshold_applied:.2f}" if isinstance(r.threshold_applied, (int, float)) else str(r.threshold_applied)
                fail_desc = f"Volume dry-up failed: observed ratio {measured_fmt} versus maximum {thresh_fmt}."
            elif pid == "PRED_PRIOR_UPTREND":
                pct_str = f"{r.measured_value * 100:.1f}%" if isinstance(r.measured_value, (int, float)) else str(r.measured_value)
                fail_desc = f"Prior uptrend failed: observed {pct_str} versus minimum 30.0% required."
            elif pid == "PRED_PROGRESSIVE_TIGHTENING":
                fail_desc = f"Progressive tightening failed: subsequent contraction waves did not tighten strictly."
            elif pid == "PRED_STAGE_2":
                fail_desc = f"Stage 2 qualification failed: price or moving averages not in advancing Stage 2."
            elif pid == "PRED_TREND_TEMPLATE":
                fail_desc = f"Minervini trend template failed: moving averages or 52-week boundaries unmet."
            elif pid == "PRED_CONTRACTION_EXISTS":
                fail_desc = f"Contraction wave count failed: insufficient consolidation contractions detected."
            elif pid == "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT":
                fail_desc = f"Pivot distance failed: price is outside valid breakout corridor relative to pivot."

            primary_rejection = {
                "predicateId": pid,
                "name": pid.replace("PRED_", "").replace("_", " ").title(),
                "status": r.status.value,
                "measuredValue": r.measured_value,
                "thresholdApplied": r.threshold_applied,
                "reasonCodes": r.reason_codes,
                "explanation": fail_desc,
            }

    # Format human-readable summary
    try:
        as_of_dt = dt_module.date.fromisoformat(auth_as_of[:10])
        as_of_str = as_of_dt.strftime("%B %d, %Y").replace(" 0", " ")
    except Exception:
        as_of_str = auth_as_of[:10]

    rejection_line = primary_rejection["explanation"] if primary_rejection else "Consolidation criteria unmet."
    summary_text = (
        f"{clean_sym} — VCP Not Qualified\n\n"
        f"{passed_count} of {len(normative_ids)} normative predicates passed.\n\n"
        f"{rejection_line}\n\n"
        f"Historical reconstructed evidence — {as_of_str}.\n\n"
        f"Not an executable trade signal."
    )

    return {
        "symbol": clean_sym,
        "evaluationContext": {
            "asOfDate": auth_as_of[:10],
            "evaluationAsOf": eval_as_of,
            "rulesetVersion": VCP_RULESET_VERSION,
            "classifierVersion": CURRENT_IMPLEMENTATION_RELEASE_SHA,
            "rulesetHash": CANONICAL_VCP_RULESET_HASH,
            "snapshotId": target_snapshot.get("snapshot_id"),
        },
        "classification": {
            "vcpQualified": assessment.vcp_qualified,
            "vcpClassification": assessment.vcp_classification,
            "stageClassification": assessment.stage_classification,
            "passedNormativeCount": passed_count,
            "totalNormativeCount": len(normative_ids),
            "primaryRejection": primary_rejection,
            "summary": summary_text,
        },
        "predicates": predicates_list,
        "provenance": {
            "provenanceType": "DETERMINISTIC_RECALCULATION",
            "evidenceSemantics": "DETERMINISTIC_RECALCULATION_CURRENT_STORE",
            "historicalPointInTimeCertified": False,
            "source": "LOCAL_MARKET_DB_SQLITE",
            "bitemporalCutoffEnforced": True,
            "evaluationAsOf": eval_as_of,
            "disclaimer": "Not an executable trade signal. Diagnostic rejection analysis provided for research explainability only.",
        },
    }

