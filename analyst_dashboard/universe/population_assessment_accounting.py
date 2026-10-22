"""
analyst_dashboard/universe/population_assessment_accounting.py

ARX Terminal — Sprint 2 Task S2-08: Full-Population Assessment & Outcome Accounting.
Governing authority: ARX Closure Model v2.

Core Capabilities:
1. Eleven-tier denominator closure accounting:
   Source Observations -> Distinct Lookup Keys -> Validated Identities ->
   Policy-Eligible Candidates -> Committed Published Members -> Scanner-Enrolled Members ->
   Dispatch-Attempted Members -> Assessment-Completed Members -> Assessment-Failed Members ->
   Authoritative Scanner Matches -> Frontend-Displayed Results.
2. Canonical terminal outcome taxonomy with strict fail-closed precedence.
3. Mutual exclusivity and reproducible arithmetic closure verification:
   N_population = N_no_setup + N_qualifying_or_developing + N_excluded + N_unsupported + N_insufficient_data + N_failed
4. Standardized assessment telemetry envelope with rigorous measurement units and
   honest 'NOT_MEASURED' reporting for unmeasured or unavailable hardware/network metrics.
5. Guaranteed 100% offline-executable, pure functional evaluation with zero side effects,
   zero production DB writes, and zero scanner activations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arx.universe.population_assessment_accounting")


# ─── 1. Canonical Outcome Taxonomy ─────────────────────────────────────────────

class AssessmentTerminalOutcome(str, Enum):
    """
    Mutually exclusive terminal outcome for every security entering the assessment pipeline.
    Preserves existing canonical taxonomy from CoverageDiagnostics & SessionGuard without
    introducing competing enums.
    """
    # ── Category: Qualifying / Developing Setups ($N_{qualifying\_or\_developing}$) ──
    COMPLETED_MATCH = "COMPLETED_MATCH"

    # ── Category: No Setup ($N_{no\_setup}$) ───────────────────────────────────────
    COMPLETED_NONMATCH = "COMPLETED_NONMATCH"

    # ── Category: Insufficient Data ($N_{insufficient\_data}$) ────────────────────
    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    INVALID_OHLCV = "INVALID_OHLCV"
    STALE_QUOTE = "STALE_QUOTE"

    # ── Category: Excluded ($N_{excluded}$) ────────────────────────────────────────
    POLICY_INELIGIBLE = "POLICY_INELIGIBLE"
    UNENROLLED = "UNENROLLED"
    DISPATCH_DEFERRED = "DISPATCH_DEFERRED"

    # ── Category: Unsupported ($N_{unsupported}$) ─────────────────────────────────
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"

    # ── Category: Failed ($N_{failed}$) ───────────────────────────────────────────
    RUNTIME_ERROR = "RUNTIME_ERROR"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"


# Strict fail-closed evaluation precedence (lower number = higher priority).
# Pre-assessment failures and execution blocks strictly take precedence over downstream evaluations.
OUTCOME_PRECEDENCE: Dict[AssessmentTerminalOutcome, int] = {
    AssessmentTerminalOutcome.EXECUTION_TIMEOUT: 1,
    AssessmentTerminalOutcome.RUNTIME_ERROR: 2,
    AssessmentTerminalOutcome.AMBIGUOUS_IDENTITY: 3,
    AssessmentTerminalOutcome.POLICY_INELIGIBLE: 4,
    AssessmentTerminalOutcome.UNENROLLED: 5,
    AssessmentTerminalOutcome.DISPATCH_DEFERRED: 6,
    AssessmentTerminalOutcome.MISSING_MARKET_DATA: 7,
    AssessmentTerminalOutcome.INSUFFICIENT_HISTORY: 8,
    AssessmentTerminalOutcome.INVALID_OHLCV: 9,
    AssessmentTerminalOutcome.STALE_QUOTE: 10,
    AssessmentTerminalOutcome.COMPLETED_MATCH: 11,
    AssessmentTerminalOutcome.COMPLETED_NONMATCH: 12,
}


# ─── 2. Data Models ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SecurityAssessmentOutcome:
    """
    Immutable terminal assessment record for an individual security.
    """
    security_id: str
    symbol: str
    outcome: AssessmentTerminalOutcome
    detail: str = ""
    precedence_rank: int = field(init=False)
    execution_attempt_id: Optional[str] = None
    duration_ms: Optional[float] = None
    assessed_at: Optional[str] = None

    def __post_init__(self):
        rank = OUTCOME_PRECEDENCE.get(self.outcome, 99)
        object.__setattr__(self, "precedence_rank", rank)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


@dataclass
class AssessmentTelemetryEnvelope:
    """
    Standardized operational telemetry envelope for full-population assessment.
    Guarantees honest units, timestamps, denominators, and source attribution.
    Unmeasured operational values default strictly to 'NOT_MEASURED' (or None)
    with zero synthetic fabrication.
    """
    duration_ms: Optional[float] = None
    throughput_securities_per_sec: Optional[float] = None
    queue_wait_ms: Optional[float] = None
    db_read_latency_ms: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_rss_bytes: Optional[int] = None
    worker_concurrency: Optional[int] = None
    timeout_count: int = 0
    retry_count: int = 0
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    completion_fraction: float = 0.0
    measurement_units: Dict[str, str] = field(default_factory=lambda: {
        "duration_ms": "milliseconds",
        "throughput_securities_per_sec": "securities/second",
        "queue_wait_ms": "milliseconds",
        "db_read_latency_ms": "milliseconds",
        "cpu_percent": "percentage",
        "memory_rss_bytes": "bytes",
        "worker_concurrency": "worker_count",
        "timeout_count": "integer_count",
        "retry_count": "integer_count",
        "cache_hits": "integer_count",
        "cache_misses": "integer_count",
        "completion_fraction": "ratio_0_to_1",
    })
    sample_denominator: int = 0
    source_attribution: str = "ARX_S2_08_INSTRUMENTATION"
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": self.duration_ms if self.duration_ms is not None else "NOT_MEASURED",
            "throughput_securities_per_sec": self.throughput_securities_per_sec if self.throughput_securities_per_sec is not None else "NOT_MEASURED",
            "queue_wait_ms": self.queue_wait_ms if self.queue_wait_ms is not None else "NOT_MEASURED",
            "db_read_latency_ms": self.db_read_latency_ms if self.db_read_latency_ms is not None else "NOT_MEASURED",
            "cpu_percent": self.cpu_percent if self.cpu_percent is not None else "NOT_MEASURED",
            "memory_rss_bytes": self.memory_rss_bytes if self.memory_rss_bytes is not None else "NOT_MEASURED",
            "worker_concurrency": self.worker_concurrency if self.worker_concurrency is not None else "NOT_MEASURED",
            "timeout_count": self.timeout_count,
            "retry_count": self.retry_count,
            "cache_hits": self.cache_hits if self.cache_hits is not None else "NOT_MEASURED",
            "cache_misses": self.cache_misses if self.cache_misses is not None else "NOT_MEASURED",
            "completion_fraction": round(self.completion_fraction, 4),
            "measurement_units": self.measurement_units,
            "sample_denominator": self.sample_denominator,
            "source_attribution": self.source_attribution,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class FullPopulationAssessmentSummary:
    """
    Authoritative summary capturing eleven-tier denominator closure,
    reconciliation arithmetic, and operational telemetry.
    """
    # ── Eleven Distinct Accounting Quantities ─────────────────────────────────
    total_source_observations: int = 0         # 1. Source observations (raw rows)
    distinct_lookup_keys: int = 0              # 2. Distinct lookup keys (tickers)
    validated_security_identities: int = 0     # 3. Validated security identities (ARX_SEC_*)
    policy_eligible_candidates: int = 0        # 4. Policy-eligible candidates (R01-R07)
    committed_published_members: int = 0       # 5. Committed published members
    scanner_enrolled_members: int = 0          # 6. Scanner-enrolled members
    dispatch_attempted_members: int = 0        # 7. Dispatch-attempted members
    assessment_completed_members: int = 0      # 8. Assessment-completed members
    assessment_failed_or_excluded_members: int = 0  # 9. Assessment-failed or excluded
    authoritative_matches: int = 0             # 10. Authoritative scanner matches
    frontend_displayed_results: Optional[int] = None  # 11. Frontend-displayed results

    # ── Category Quantities (ENGINEERING_BACKLOG.md Formula) ───────────────────
    n_no_setup: int = 0                        # COMPLETED_NONMATCH
    n_qualifying_or_developing: int = 0        # COMPLETED_MATCH
    n_excluded: int = 0                        # POLICY_INELIGIBLE + UNENROLLED + DISPATCH_DEFERRED
    n_unsupported: int = 0                     # AMBIGUOUS_IDENTITY
    n_insufficient_data: int = 0               # MISSING_MARKET_DATA + INSUFFICIENT_HISTORY + INVALID_OHLCV + STALE_QUOTE
    n_failed: int = 0                          # RUNTIME_ERROR + EXECUTION_TIMEOUT

    # ── Closure Verification ──────────────────────────────────────────────────
    closure_verified: bool = False
    closure_delta: int = 0
    reconciliation_message: str = ""
    per_outcome_counts: Dict[str, int] = field(default_factory=dict)
    telemetry: Optional[AssessmentTelemetryEnvelope] = None
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_source_observations": self.total_source_observations,
            "distinct_lookup_keys": self.distinct_lookup_keys,
            "validated_security_identities": self.validated_security_identities,
            "policy_eligible_candidates": self.policy_eligible_candidates,
            "committed_published_members": self.committed_published_members,
            "scanner_enrolled_members": self.scanner_enrolled_members,
            "dispatch_attempted_members": self.dispatch_attempted_members,
            "assessment_completed_members": self.assessment_completed_members,
            "assessment_failed_or_excluded_members": self.assessment_failed_or_excluded_members,
            "authoritative_matches": self.authoritative_matches,
            "frontend_displayed_results": self.frontend_displayed_results,
            "category_reconciliation": {
                "n_no_setup": self.n_no_setup,
                "n_qualifying_or_developing": self.n_qualifying_or_developing,
                "n_excluded": self.n_excluded,
                "n_unsupported": self.n_unsupported,
                "n_insufficient_data": self.n_insufficient_data,
                "n_failed": self.n_failed,
                "sum_categories": (
                    self.n_no_setup
                    + self.n_qualifying_or_developing
                    + self.n_excluded
                    + self.n_unsupported
                    + self.n_insufficient_data
                    + self.n_failed
                ),
            },
            "closure_verified": self.closure_verified,
            "closure_delta": self.closure_delta,
            "reconciliation_message": self.reconciliation_message,
            "per_outcome_counts": self.per_outcome_counts,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "evaluated_at": self.evaluated_at,
        }


# ─── 3. Accounting Engine ─────────────────────────────────────────────────────

class FullPopulationAssessmentEngine:
    """
    Pure offline accounting and validation engine for full-population assessment.
    Guarantees:
    - Zero network calls.
    - Zero database mutations.
    - Zero scanner executions.
    - Strict mathematical closure verification.
    """

    @staticmethod
    def evaluate_outcomes(
        defined_population: List[str],
        individual_outcomes: List[SecurityAssessmentOutcome],
        total_source_observations: Optional[int] = None,
        distinct_lookup_keys: Optional[int] = None,
        validated_security_identities: Optional[int] = None,
        policy_eligible_candidates: Optional[int] = None,
        committed_published_members: Optional[int] = None,
        scanner_enrolled_members: Optional[int] = None,
        frontend_displayed_results: Optional[int] = None,
        telemetry: Optional[AssessmentTelemetryEnvelope] = None,
    ) -> FullPopulationAssessmentSummary:
        """
        Reconciles the defined population against individual security outcomes.
        Verifies:
        1. Population cardinality matches outcome set cardinality.
        2. Every defined security has exactly one reconciled terminal outcome.
        3. Mutual exclusivity of outcome partitions.
        4. Arithmetic equality of formula:
           N_population = N_no_setup + N_qualifying + N_excluded + N_unsupported + N_insufficient_data + N_failed
        """
        n_population = len(defined_population)
        pop_set = set(s.strip().upper() for s in defined_population if s.strip())

        # Check for duplicate outcomes per security
        seen_symbols: Set[str] = set()
        duplicate_symbols: Set[str] = set()
        for o in individual_outcomes:
            sym = o.symbol.strip().upper()
            if sym in seen_symbols:
                duplicate_symbols.add(sym)
            seen_symbols.add(sym)

        # Count per outcome
        counts: Dict[AssessmentTerminalOutcome, int] = {o: 0 for o in AssessmentTerminalOutcome}
        for o in individual_outcomes:
            counts[o.outcome] = counts.get(o.outcome, 0) + 1

        # Category allocations per ENGINEERING_BACKLOG.md
        n_qualifying = counts[AssessmentTerminalOutcome.COMPLETED_MATCH]
        n_no_setup = counts[AssessmentTerminalOutcome.COMPLETED_NONMATCH]
        n_excluded = (
            counts[AssessmentTerminalOutcome.POLICY_INELIGIBLE]
            + counts[AssessmentTerminalOutcome.UNENROLLED]
            + counts[AssessmentTerminalOutcome.DISPATCH_DEFERRED]
        )
        n_unsupported = counts[AssessmentTerminalOutcome.AMBIGUOUS_IDENTITY]
        n_insufficient_data = (
            counts[AssessmentTerminalOutcome.MISSING_MARKET_DATA]
            + counts[AssessmentTerminalOutcome.INSUFFICIENT_HISTORY]
            + counts[AssessmentTerminalOutcome.INVALID_OHLCV]
            + counts[AssessmentTerminalOutcome.STALE_QUOTE]
        )
        n_failed = (
            counts[AssessmentTerminalOutcome.RUNTIME_ERROR]
            + counts[AssessmentTerminalOutcome.EXECUTION_TIMEOUT]
        )

        n_completed = n_qualifying + n_no_setup
        n_failed_or_excluded = n_excluded + n_unsupported + n_insufficient_data + n_failed

        sum_categories = (
            n_no_setup
            + n_qualifying
            + n_excluded
            + n_unsupported
            + n_insufficient_data
            + n_failed
        )

        # Check missing symbols
        missing_from_outcomes = pop_set - seen_symbols
        unexpected_in_outcomes = seen_symbols - pop_set

        dispatch_attempted = len(individual_outcomes)
        closure_delta = sum_categories - n_population

        # Determine verification status
        closure_verified = (
            closure_delta == 0
            and len(missing_from_outcomes) == 0
            and len(unexpected_in_outcomes) == 0
            and len(duplicate_symbols) == 0
            and sum_categories == dispatch_attempted
        )

        if closure_verified:
            msg = (
                f"ARITHMETIC_CLOSURE_VERIFIED: Defined population ({n_population}) exactly matches "
                f"reconciled outcomes ({sum_categories}). Matches={n_qualifying}, NonMatches={n_no_setup}, "
                f"Excluded={n_excluded}, Unsupported={n_unsupported}, InsufficientData={n_insufficient_data}, "
                f"Failed={n_failed}."
            )
        else:
            reasons = []
            if closure_delta != 0:
                reasons.append(f"closure_delta={closure_delta}")
            if missing_from_outcomes:
                reasons.append(f"missing_symbols_count={len(missing_from_outcomes)}")
            if unexpected_in_outcomes:
                reasons.append(f"unexpected_symbols_count={len(unexpected_in_outcomes)}")
            if duplicate_symbols:
                reasons.append(f"duplicate_symbols_count={len(duplicate_symbols)}")
            msg = f"CLOSURE_FAILED: {'; '.join(reasons)}"

        # Final Summary
        summary = FullPopulationAssessmentSummary(
            total_source_observations=(
                total_source_observations
                if total_source_observations is not None
                else n_population
            ),
            distinct_lookup_keys=(
                distinct_lookup_keys
                if distinct_lookup_keys is not None
                else n_population
            ),
            validated_security_identities=(
                validated_security_identities
                if validated_security_identities is not None
                else n_population
            ),
            policy_eligible_candidates=(
                policy_eligible_candidates
                if policy_eligible_candidates is not None
                else n_population
            ),
            committed_published_members=(
                committed_published_members
                if committed_published_members is not None
                else n_population
            ),
            scanner_enrolled_members=(
                scanner_enrolled_members
                if scanner_enrolled_members is not None
                else n_population
            ),
            dispatch_attempted_members=dispatch_attempted,
            assessment_completed_members=n_completed,
            assessment_failed_or_excluded_members=n_failed_or_excluded,
            authoritative_matches=n_qualifying,
            frontend_displayed_results=frontend_displayed_results,
            n_no_setup=n_no_setup,
            n_qualifying_or_developing=n_qualifying,
            n_excluded=n_excluded,
            n_unsupported=n_unsupported,
            n_insufficient_data=n_insufficient_data,
            n_failed=n_failed,
            closure_verified=closure_verified,
            closure_delta=closure_delta,
            reconciliation_message=msg,
            per_outcome_counts={k.value: v for k, v in counts.items()},
            telemetry=telemetry,
        )

        return summary
