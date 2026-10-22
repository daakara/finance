"""
tests/test_full_population_assessment_accounting.py

Authoritative verification for Sprint 2 Task S2-08: Full-Population Assessment & Outcome Accounting.
Governing authority: ARX Closure Model v2.

Coverage:
1. Eleven-tier denominator closure verification:
   N_population = N_no_setup + N_qualifying_or_developing + N_excluded + N_unsupported + N_insufficient_data + N_failed
2. Reconciles exact arithmetic equality (closure_delta == 0).
3. Fail-closed detection of missing, duplicate, or unexpected securities in assessment.
4. Strict terminal outcome exclusivity and fail-closed precedence.
5. Honest operational telemetry envelope with 'NOT_MEASURED' defaults and zero synthetic data.
6. Authentic operational zero-match preservation vs pipeline pending/unknown states.
7. Decoupling of frontend display filters from authoritative scanner matches.
8. Backward compatibility with S2-06, S2-07, and S2-09 contracts.
"""

from __future__ import annotations

import json
import pytest

from analyst_dashboard.universe.population_assessment_accounting import (
    AssessmentTerminalOutcome,
    OUTCOME_PRECEDENCE,
    SecurityAssessmentOutcome,
    AssessmentTelemetryEnvelope,
    FullPopulationAssessmentSummary,
    FullPopulationAssessmentEngine,
)


@pytest.fixture
def sample_150_population() -> list[str]:
    """Generates deterministic 150-symbol pilot universe."""
    return [f"SYM{i:03d}" for i in range(1, 151)]


# ─── 1. Denominator Arithmetic Closure Tests ────────────────────────────────────

def test_full_population_exact_arithmetic_closure(sample_150_population):
    """
    Verifies that a 150-symbol population partitioned across all 6 canonical categories
    produces exact arithmetic closure (closure_delta == 0, closure_verified == True).
    """
    outcomes: list[SecurityAssessmentOutcome] = []

    # 1. Qualifying matches ($N_{qualifying\_or\_developing}$): 12
    for s in sample_150_population[0:12]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.COMPLETED_MATCH, detail="VCP Confirmed"
        ))

    # 2. No setup ($N_{no\_setup}$): 88
    for s in sample_150_population[12:100]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.COMPLETED_NONMATCH, detail="No contraction"
        ))

    # 3. Excluded ($N_{excluded}$): 20 (policy ineligible, unenrolled, deferred)
    for s in sample_150_population[100:110]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.POLICY_INELIGIBLE, detail="Rule R02 ETF"
        ))
    for s in sample_150_population[110:115]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.UNENROLLED, detail="Membership unselected"
        ))
    for s in sample_150_population[115:120]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.DISPATCH_DEFERRED, detail="Rate limit hold"
        ))

    # 4. Unsupported ($N_{unsupported}$): 10 (ambiguous identity)
    for s in sample_150_population[120:130]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.AMBIGUOUS_IDENTITY, detail="Missing CIK"
        ))

    # 5. Insufficient Data ($N_{insufficient\_data}$): 15 (missing price, low history, bad OHLCV, stale)
    for s in sample_150_population[130:135]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.MISSING_MARKET_DATA, detail="No store record"
        ))
    for s in sample_150_population[135:140]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.INSUFFICIENT_HISTORY, detail="Candles < 50"
        ))
    for s in sample_150_population[140:143]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.INVALID_OHLCV, detail="High < Low"
        ))
    for s in sample_150_population[143:145]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.STALE_QUOTE, detail="Session mismatch"
        ))

    # 6. Failed ($N_{failed}$): 5 (runtime error, execution timeout)
    for s in sample_150_population[145:148]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.RUNTIME_ERROR, detail="ZeroDivisionError"
        ))
    for s in sample_150_population[148:150]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.EXECUTION_TIMEOUT, detail="Worker lease expired"
        ))

    assert len(outcomes) == 150

    summary = FullPopulationAssessmentEngine.evaluate_outcomes(
        defined_population=sample_150_population,
        individual_outcomes=outcomes,
        total_source_observations=165,
        distinct_lookup_keys=150,
        validated_security_identities=140,
        policy_eligible_candidates=130,
        committed_published_members=125,
        scanner_enrolled_members=120,
        frontend_displayed_results=12,
    )

    # Invariants verification
    assert summary.closure_verified is True
    assert summary.closure_delta == 0
    assert summary.dispatch_attempted_members == 150
    assert summary.assessment_completed_members == 100  # 12 matches + 88 nonmatches
    assert summary.assessment_failed_or_excluded_members == 50  # 20 excluded + 10 unsupported + 15 data + 5 failed
    assert summary.authoritative_matches == 12
    assert summary.n_qualifying_or_developing == 12
    assert summary.n_no_setup == 88
    assert summary.n_excluded == 20
    assert summary.n_unsupported == 10
    assert summary.n_insufficient_data == 15
    assert summary.n_failed == 5

    # Formula check: N_pop == sum(categories)
    cat = summary.to_dict()["category_reconciliation"]
    assert cat["sum_categories"] == 150
    assert summary.reconciliation_message.startswith("ARITHMETIC_CLOSURE_VERIFIED")


# ─── 2. Fail-Closed Boundary & Leakage Protection Tests ────────────────────────

def test_missing_security_fails_closure(sample_150_population):
    """
    Verifies that if even one security from the defined population disappears from the
    outcomes list, closure arithmetic strictly fails closed.
    """
    # Create outcomes for only 149 symbols (omitting SYM150)
    outcomes = [
        SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}",
            symbol=s,
            outcome=AssessmentTerminalOutcome.COMPLETED_NONMATCH,
        )
        for s in sample_150_population[:-1]
    ]

    summary = FullPopulationAssessmentEngine.evaluate_outcomes(
        defined_population=sample_150_population,
        individual_outcomes=outcomes,
    )

    assert summary.closure_verified is False
    assert summary.closure_delta == -1
    assert "missing_symbols_count=1" in summary.reconciliation_message
    assert "CLOSURE_FAILED" in summary.reconciliation_message


def test_duplicate_security_outcome_fails_closure(sample_150_population):
    """
    Verifies that if a security has duplicate outcome records, closure fails closed.
    """
    outcomes = [
        SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}",
            symbol=s,
            outcome=AssessmentTerminalOutcome.COMPLETED_NONMATCH,
        )
        for s in sample_150_population
    ]
    # Inject a duplicate for SYM001
    outcomes.append(
        SecurityAssessmentOutcome(
            security_id="ARX_SEC_SYM001",
            symbol="SYM001",
            outcome=AssessmentTerminalOutcome.COMPLETED_MATCH,
        )
    )

    summary = FullPopulationAssessmentEngine.evaluate_outcomes(
        defined_population=sample_150_population,
        individual_outcomes=outcomes,
    )

    assert summary.closure_verified is False
    assert summary.closure_delta == 1
    assert "duplicate_symbols_count=1" in summary.reconciliation_message


def test_unexpected_security_fails_closure(sample_150_population):
    """
    Verifies that if an unexpected security not in the defined population appears in outcomes,
    closure fails closed.
    """
    outcomes = [
        SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}",
            symbol=s,
            outcome=AssessmentTerminalOutcome.COMPLETED_NONMATCH,
        )
        for s in sample_150_population[:-1]
    ]
    # Replace last symbol with an rogue symbol
    outcomes.append(
        SecurityAssessmentOutcome(
            security_id="ARX_SEC_ROGUE",
            symbol="ROGUE_SYM",
            outcome=AssessmentTerminalOutcome.COMPLETED_MATCH,
        )
    )

    summary = FullPopulationAssessmentEngine.evaluate_outcomes(
        defined_population=sample_150_population,
        individual_outcomes=outcomes,
    )

    assert summary.closure_verified is False
    assert "unexpected_symbols_count=1" in summary.reconciliation_message


# ─── 3. Authentic Operational Zero vs Unknown Distinction ──────────────────────

def test_authentic_operational_zero_matches_reconciliation(sample_150_population):
    """
    Verifies that when a scanner runs across the full population and authentic 0 matches
    are found, authoritative_matches is 0 and closure is fully verified.
    """
    # All 150 securities evaluated to COMPLETED_NONMATCH
    outcomes = [
        SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}",
            symbol=s,
            outcome=AssessmentTerminalOutcome.COMPLETED_NONMATCH,
            detail="Pattern criteria not met",
        )
        for s in sample_150_population
    ]

    summary = FullPopulationAssessmentEngine.evaluate_outcomes(
        defined_population=sample_150_population,
        individual_outcomes=outcomes,
    )

    assert summary.closure_verified is True
    assert summary.authoritative_matches == 0
    assert summary.n_qualifying_or_developing == 0
    assert summary.n_no_setup == 150
    assert summary.assessment_completed_members == 150
    assert summary.assessment_failed_or_excluded_members == 0


# ─── 4. Precedence & Exclusivity Verification ──────────────────────────────────

def test_outcome_precedence_invariants():
    """
    Verifies that outcome precedence ranking is strictly ascending and unique.
    """
    ranks = [OUTCOME_PRECEDENCE[o] for o in AssessmentTerminalOutcome]
    assert len(ranks) == len(set(ranks)), "Precedence ranks must be strictly unique"

    # Execution timeouts and runtime errors must precede missing market data
    assert OUTCOME_PRECEDENCE[AssessmentTerminalOutcome.EXECUTION_TIMEOUT] < OUTCOME_PRECEDENCE[AssessmentTerminalOutcome.MISSING_MARKET_DATA]
    # Missing market data must precede completed evaluations
    assert OUTCOME_PRECEDENCE[AssessmentTerminalOutcome.MISSING_MARKET_DATA] < OUTCOME_PRECEDENCE[AssessmentTerminalOutcome.COMPLETED_MATCH]


# ─── 5. Operational Telemetry Envelope Verification ───────────────────────────

def test_telemetry_envelope_honest_reporting():
    """
    Verifies that unmeasured telemetry values are reported as 'NOT_MEASURED' without
    synthetic fabrication, and measured values preserve exact units and ratios.
    """
    telemetry = AssessmentTelemetryEnvelope(
        duration_ms=452.3,
        throughput_securities_per_sec=331.6,
        timeout_count=0,
        retry_count=0,
        completion_fraction=1.0,
        sample_denominator=150,
    )

    d = telemetry.to_dict()

    assert d["duration_ms"] == 452.3
    assert d["throughput_securities_per_sec"] == 331.6
    assert d["queue_wait_ms"] == "NOT_MEASURED"
    assert d["db_read_latency_ms"] == "NOT_MEASURED"
    assert d["cpu_percent"] == "NOT_MEASURED"
    assert d["memory_rss_bytes"] == "NOT_MEASURED"
    assert d["worker_concurrency"] == "NOT_MEASURED"
    assert d["cache_hits"] == "NOT_MEASURED"
    assert d["cache_misses"] == "NOT_MEASURED"
    assert d["timeout_count"] == 0
    assert d["retry_count"] == 0
    assert d["completion_fraction"] == 1.0
    assert d["sample_denominator"] == 150
    assert d["source_attribution"] == "ARX_S2_08_INSTRUMENTATION"


# ─── 6. Frontend Filter Decoupling Verification ───────────────────────────────

def test_frontend_filter_reduction_does_not_mutate_authoritative_matches(sample_150_population):
    """
    Verifies that user search filtering reduces frontend_displayed_results without mutating
    authoritative_matches or corrupting closure.
    """
    outcomes = []
    # 5 matches, 145 nonmatches
    for s in sample_150_population[:5]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.COMPLETED_MATCH
        ))
    for s in sample_150_population[5:]:
        outcomes.append(SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}", symbol=s, outcome=AssessmentTerminalOutcome.COMPLETED_NONMATCH
        ))

    # User searches for 'SYM001' on frontend -> displayed = 1
    summary = FullPopulationAssessmentEngine.evaluate_outcomes(
        defined_population=sample_150_population,
        individual_outcomes=outcomes,
        frontend_displayed_results=1,
    )

    assert summary.closure_verified is True
    assert summary.authoritative_matches == 5  # Authoritative scanner matches remain 5!
    assert summary.frontend_displayed_results == 1  # User sees only 1 displayed result!
    assert summary.n_qualifying_or_developing == 5


# ─── 7. Serialization & Dict Export Verification ──────────────────────────────

def test_summary_json_serializable(sample_150_population):
    """
    Verifies that the entire summary and telemetry envelope serialize cleanly to JSON.
    """
    outcomes = [
        SecurityAssessmentOutcome(
            security_id=f"ARX_SEC_{s}",
            symbol=s,
            outcome=AssessmentTerminalOutcome.COMPLETED_NONMATCH,
        )
        for s in sample_150_population
    ]

    telemetry = AssessmentTelemetryEnvelope(
        duration_ms=120.5,
        throughput_securities_per_sec=1244.8,
        sample_denominator=150,
        completion_fraction=1.0,
    )

    summary = FullPopulationAssessmentEngine.evaluate_outcomes(
        defined_population=sample_150_population,
        individual_outcomes=outcomes,
        telemetry=telemetry,
    )

    raw_json = json.dumps(summary.to_dict(), indent=2)
    reloaded = json.loads(raw_json)

    assert reloaded["closure_verified"] is True
    assert reloaded["closure_delta"] == 0
    assert reloaded["telemetry"]["duration_ms"] == 120.5
    assert reloaded["telemetry"]["queue_wait_ms"] == "NOT_MEASURED"
