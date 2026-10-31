"""
ARX Terminal — Directive ARX-MW-001
Market-Wide Discovery and Top-35 Radar Proof of Concept Data Contracts.

Governs immutable data contracts, status semantics, accounting metrics,
and candidate presentation structures for the Market-Wide Discovery POC.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional


class EngineEvaluationState(str, Enum):
    """Authoritative state for deeper analysis engine evaluation."""
    EVALUATED = "EVALUATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    NOT_AUTHORIZED = "NOT_AUTHORIZED"


class RankingCohort(str, Enum):
    """Selection cohorts for deterministic Top-35 shortlist."""
    COHORT_A_QUALIFIED = "COHORT_A_QUALIFIED"
    COHORT_B_DEVELOPING = "COHORT_B_DEVELOPING"


class Stage1Outcome(str, Enum):
    """Explicit outcome for Stage 1 preliminary screening."""
    SURVIVED = "SURVIVED"
    REJECTED_PRICE_FLOOR = "REJECTED_PRICE_FLOOR"
    REJECTED_LIQUIDITY_FLOOR = "REJECTED_LIQUIDITY_FLOOR"
    REJECTED_TREND_GEOMETRY = "REJECTED_TREND_GEOMETRY"
    REJECTED_MOMENTUM = "REJECTED_MOMENTUM"
    DATA_BLOCKED = "DATA_BLOCKED"
    EVALUATION_FAILED = "EVALUATION_FAILED"


# Canonical POC versioning constants
POC_UNIVERSE_ID: str = "ARX_US_EQUITIES_MARKET_WIDE_POC"
POC_UNIVERSE_VERSION: str = "arx-mw-universe-v1.0"
POC_RANKING_RULESET_VERSION: str = "top35-ranking-ruleset-v1.0"
POC_CANDIDATE_GENERATION_ID: str = "ARX_MW001_CANDIDATE_GEN_001"
POC_SCOPE_LABEL: str = "ARX-eligible US common stocks (Market-Wide Discovery POC)"


def compute_deterministic_hash(data: Any) -> str:
    """Computes SHA-256 digest of arbitrary JSON-serializable structure."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class Stage1FilterResult:
    """Stage 1 preliminary screening evaluation for a single security."""
    symbol: str
    outcome: Stage1Outcome
    price: Optional[float] = None
    volume_20d: Optional[float] = None
    dollar_volume_20d: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    rejection_reason: Optional[str] = None
    candle_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketWideCandidate:
    """
    Candidate presentation contract for Top-35 Radar POC (Section 8).
    Preserves strict discovery invariants: non-executable, no sizing multipliers.
    """
    security_id: str
    symbol: str
    company_name: str
    rank: int
    ranking_cohort: RankingCohort
    strategy_classification: str
    discovery_reason: str
    evaluation_status: EngineEvaluationState
    confluence_score: float
    pillar_scores: Dict[str, float]
    evidence_completeness: Dict[str, str]
    technical_conditions: Dict[str, Any]
    fundamental_findings: Dict[str, Any]
    missing_evidence: List[str]
    current_price: float
    price_timestamp: str
    setup_status: str
    invalidation_condition: str
    optimal_entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    relative_strength: Optional[float] = None
    volume_dry_up: Optional[float] = None
    universe_build_id: str = ""
    candidate_generation_id: str = POC_CANDIDATE_GENERATION_ID
    ranking_ruleset_version: str = POC_RANKING_RULESET_VERSION
    is_actionable: bool = False       # Invariant: discovery candidate is non-executable
    can_size_trade: bool = False      # Invariant: no automated position sizing

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ranking_cohort"] = self.ranking_cohort.value
        d["evaluation_status"] = self.evaluation_status.value
        return d


@dataclass
class MarketWideCoverageAccounting:
    """
    Coverage and denominator accounting closure (Section 10).
    Ensures mathematical consistency across all discovery funnel stages.
    """
    source_market_population: int
    identity_resolved_population: int
    eligible_universe: int
    ineligible_population: int
    eligible_data_blocked: int
    scannable_securities: int
    stage1_evaluated: int
    stage1_rejected: int
    stage1_data_blocked: int
    stage1_failed_or_pending: int
    stage2_admitted: int
    stage2_fully_evaluated: int
    stage2_partially_evaluated: int
    qualified_candidates_cohort_a: int
    developing_discoveries_cohort_b: int
    displayed_candidates: int
    universe_completeness_pct: float
    processing_completeness_pct: float
    data_coverage_status: str
    arithmetic_closure: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketWidePublicationSnapshot:
    """
    Immutable POC publication envelope (Section 9).
    Shadow-only publication, isolated from production prospective-learning denominators.
    """
    snapshot_id: str
    universe_build_id: str
    logical_scan_run_id: str
    candidate_generation_id: str
    ranking_ruleset_version: str
    release_sha: str
    as_of: str
    created_at: str
    coverage_accounting: MarketWideCoverageAccounting
    candidates: List[MarketWideCandidate]
    candidates_digest: str
    is_production_authoritative: bool = False
    governance_label: str = "EXPERIMENTAL MARKET-WIDE RADAR — SHADOW POC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "universe_build_id": self.universe_build_id,
            "logical_scan_run_id": self.logical_scan_run_id,
            "candidate_generation_id": self.candidate_generation_id,
            "ranking_ruleset_version": self.ranking_ruleset_version,
            "release_sha": self.release_sha,
            "as_of": self.as_of,
            "created_at": self.created_at,
            "coverage_accounting": self.coverage_accounting.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "candidates_count": len(self.candidates),
            "candidates_digest": self.candidates_digest,
            "is_production_authoritative": self.is_production_authoritative,
            "governance_label": self.governance_label,
        }
