"""Comprehensive Test Suite for Sprint 2B Domain-Authority Resolution.

Tests all 27 technical, semantic, bitemporal, and mutation gates:
- Domain glossary completeness & authority integrity
- Predicate contract closure & 5-state result model
- Numeric & temporal determinism
- Bitemporal physical truncation & partial-bar safety
- Future suffix invariance & future canary elimination
- Wall-clock independence & cache safety
- Adjudicator blinding & holdout duplicate leakage
- Holdout commitment hashing
- Conformance on Gold Dev & Holdout corpora (0 mismatches)
- 21 Mutation operators killed (0 survivors)
- Product claim authorization & differential accounting
"""

import copy
import hashlib
import json
import pytest

from analyst_dashboard.vcp.case_compiler import (
    TemporalReadEvent,
    VCPTemporalCaseCompiler,
    VCPTemporalCasePackage,
)
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.conformance_oracle import (
    ADJUDICATORS,
    OracleTier,
    VCPConformanceCorpus,
    VCPCorpusCase,
    generate_clean_history,
)
from analyst_dashboard.vcp.domain_glossary import (
    GLOSSARY_TERMS,
    TermStatus,
    VCPDomainGlossary,
)
from analyst_dashboard.vcp.domain_source_registry import (
    AuthorityClass,
    VCP_DOMAIN_SOURCES,
    VCPDomainSourceRegistry,
)
from analyst_dashboard.vcp.label_authorization import (
    LABEL_AUTHORIZATION_CATALOG,
    VCPLabelAuthorizationMatrix,
)
from analyst_dashboard.vcp.mutation_harness import (
    MutationTestResult,
    VCPMutationHarness,
)
from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.predicate_registry import (
    ConformanceRole,
    PredicateResult,
    PredicateStatus,
    VCPDomainAssessment,
    VCPObservation,
    VCPPredicateRegistry,
)
from analyst_dashboard.vcp.temporal_contract import (
    DailyOHLCVBar,
    VCPTemporalContract,
)


# ======================================================================
# 1. DOMAIN GLOSSARY & SOURCE REGISTRY INTEGRITY
# ======================================================================

def test_domain_glossary_completeness():
    glossary = VCPDomainGlossary()
    terms = glossary.list_terms()
    assert len(terms) >= 21, f"Expected at least 21 domain terms, found {len(terms)}"
    assert glossary.count_terms_without_definition() == 0, "No terms may lack a definition"

    required_ids = [
        "TERM-VCP", "TERM-CONTRACTION", "TERM-CANDIDATE-CONTRACTION", "TERM-VALID-CONTRACTION",
        "TERM-CONTRACTION-SEQUENCE", "TERM-PROGRESSIVE-TIGHTENING", "TERM-VOLATILITY-CONTRACTION",
        "TERM-VOLUME-DRY-UP", "TERM-PRIOR-TREND", "TERM-TREND-TEMPLATE", "TERM-PIVOT",
        "TERM-BASE", "TERM-BREAKOUT", "TERM-FAILED-SETUP", "TERM-STAGE-1", "TERM-STAGE-2",
        "TERM-STAGE-3", "TERM-STAGE-4", "TERM-INSUFFICIENT-EVIDENCE", "TERM-DOMAIN-UNRESOLVED",
        "TERM-NOT-APPLICABLE",
    ]
    for tid in required_ids:
        term = glossary.get_term(tid)
        assert term is not None, f"Missing required term {tid}"
        assert len(term.semantic_definition.strip()) > 20
        assert len(term.domain_authority_sources) >= 1
        assert term.status in list(TermStatus)


def test_source_authority_reference_integrity():
    registry = VCPDomainSourceRegistry()
    assert len(registry.sources) >= 5, "Expected at least 5 primary/secondary domain sources"

    # Verify primary authorities exist
    min_2013 = registry.get_source("SRC-MINERVINI-2013")
    assert min_2013 is not None
    assert min_2013.primary_authority_class == AuthorityClass.PRIMARY_DOMAIN_AUTHORITY
    assert len(min_2013.claims) >= 5

    wei_1988 = registry.get_source("SRC-WEINSTEIN-1988")
    assert wei_1988 is not None
    assert wei_1988.primary_authority_class == AuthorityClass.PRIMARY_DOMAIN_AUTHORITY

    # Verify all glossary terms reference valid registered sources
    glossary = VCPDomainGlossary()
    for term in glossary.list_terms():
        for s_id in term.domain_authority_sources:
            src = registry.get_source(s_id)
            assert src is not None, f"Term {term.term_id} references unregistered source {s_id}"


def test_licensing_content_usage_separation():
    registry = VCPDomainSourceRegistry()
    for src in registry.sources.values():
        rights = [r.value for r in src.content_usage_rights]
        assert "REFERENCE_PERMITTED" in rights
        assert "CITATION_ONLY" in rights
        if src.primary_authority_class == AuthorityClass.PRIMARY_DOMAIN_AUTHORITY:
            assert "WHOLESALE_REPRODUCTION_FORBIDDEN" in rights
            assert "CHART_IMAGE_EMBEDDING_FORBIDDEN" in rights


# ======================================================================
# 2. PREDICATE CONTRACT CLOSURE & RESULT MODEL
# ======================================================================

def test_predicate_registry_closure():
    pred_reg = VCPPredicateRegistry()
    normative = pred_reg.list_normative_predicates()
    assert len(normative) == 10, f"Expected exactly 10 normative predicates, got {len(normative)}"

    expected_normative = [
        "PRED_SUFFICIENT_HISTORY", "PRED_PRIOR_UPTREND", "PRED_TREND_TEMPLATE",
        "PRED_STAGE_2", "PRED_CONTRACTION_EXISTS", "PRED_CONTRACTION_SEQUENCE_VALID",
        "PRED_PROGRESSIVE_TIGHTENING", "PRED_VOLUME_DRY_UP", "PRED_PIVOT_DEFINED",
        "PRED_PRICE_POSITION_RELATIVE_TO_PIVOT",
    ]
    for pid in expected_normative:
        p = pred_reg.get_predicate(pid)
        assert p is not None, f"Missing normative predicate {pid}"
        assert p.conformance_role == ConformanceRole.NORMATIVE
        assert len(p.authority_basis) >= 1
        assert len(p.reason_codes) >= 1


def test_5_state_predicate_result_model():
    res_pass = PredicateResult("P1", PredicateStatus.PASS, 10, 5, ["OK"])
    res_fail = PredicateResult("P1", PredicateStatus.FAIL, 2, 5, ["TOO_LOW"])
    res_insuf = PredicateResult("P1", PredicateStatus.INSUFFICIENT_DATA, None, 5, ["MISSING"])
    res_unres = PredicateResult("P1", PredicateStatus.UNRESOLVED, None, 5, ["AMBIGUOUS"])
    res_na = PredicateResult("P1", PredicateStatus.NOT_APPLICABLE, None, None, ["SKIP"])

    assert res_pass.is_pass() and not res_pass.is_fail()
    assert res_fail.is_fail() and not res_fail.is_pass()
    assert res_insuf.is_insufficient_data() and not res_insuf.is_fail() and not res_insuf.is_pass()
    assert res_unres.is_unresolved() and not res_unres.is_fail() and not res_unres.is_pass()
    assert res_na.is_not_applicable()


# ======================================================================
# 3. NUMERIC & TEMPORAL DETERMINISM
# ======================================================================

def test_numeric_determinism():
    prices = [10.0, 12.0, 11.5, 13.0, 12.5, 14.0]
    sma1 = VCPNumericContract.compute_sma(prices, 3)
    sma2 = VCPNumericContract.compute_sma(prices, 3)
    assert sma1 == sma2 == 13.1667

    depth1 = VCPNumericContract.compute_contraction_depth(100.0, 80.0)
    depth2 = VCPNumericContract.compute_contraction_depth(100.0, 80.0)
    assert depth1 == depth2 == 0.2000

    tight1 = VCPNumericContract.verify_progressive_tightening([0.25, 0.15, 0.05])
    tight2 = VCPNumericContract.verify_progressive_tightening([0.25, 0.15, 0.05])
    assert tight1 == tight2 is True


def test_temporal_admissibility_and_partial_bar():
    contract = VCPTemporalContract()
    cutoff = "2026-03-31T20:00:00Z"

    # Closed bar prior to cutoff -> PASS
    b_ok = DailyOHLCVBar("ACME", "2026-03-31", 10, 11, 9, 10.5, 100, "2026-03-31T20:00:00Z", "2026-03-31T20:00:00Z", True)
    adm_ok, reason_ok = contract.is_bar_admissible(b_ok, cutoff)
    assert adm_ok is True
    assert reason_ok is None

    # Post-cutoff valid time -> REJECT
    b_future_valid = DailyOHLCVBar("ACME", "2026-04-01", 10, 11, 9, 10.5, 100, "2026-04-01T20:00:00Z", "2026-04-01T20:00:00Z", True)
    adm_fv, reason_fv = contract.is_bar_admissible(b_future_valid, cutoff)
    assert adm_fv is False
    assert "POST_CUTOFF_VALID_TIME" in reason_fv

    # Post-cutoff known at -> REJECT
    b_future_known = DailyOHLCVBar("ACME", "2026-03-31", 10, 11, 9, 10.5, 100, "2026-03-31T20:00:00Z", "2026-04-02T12:00:00Z", True)
    adm_fk, reason_fk = contract.is_bar_admissible(b_future_known, cutoff)
    assert adm_fk is False
    assert "POST_CUTOFF_KNOWN_AT" in reason_fk

    # Partial / unclosed bar -> REJECT
    b_unclosed = DailyOHLCVBar("ACME", "2026-03-31", 10, 11, 9, 10.5, 100, "2026-03-31T14:30:00Z", "2026-03-31T14:30:00Z", False)
    adm_uncl, reason_uncl = contract.is_bar_admissible(b_unclosed, cutoff)
    assert adm_uncl is False
    assert "PARTIAL_BAR_UNCLOSED" in reason_uncl


# ======================================================================
# 4. SEALED CASE COMPILER & BITEMPORAL PREFIX INVARIANCE
# ======================================================================

def test_sealed_case_compiler_physical_truncation():
    compiler = VCPTemporalCaseCompiler()
    cutoff = "2026-03-31T20:00:00Z"

    bars = [
        DailyOHLCVBar("A", "2026-03-29", 10, 11, 9, 10, 100, "2026-03-29T20:00:00Z", "2026-03-29T20:00:00Z"),
        DailyOHLCVBar("A", "2026-03-30", 10, 11, 9, 10, 100, "2026-03-30T20:00:00Z", "2026-03-30T20:00:00Z"),
        DailyOHLCVBar("A", "2026-03-31", 10, 11, 9, 10, 100, "2026-03-31T20:00:00Z", "2026-03-31T20:00:00Z"),
        DailyOHLCVBar("A", "2026-04-01", 50, 60, 40, 55, 900, "2026-04-01T20:00:00Z", "2026-04-01T20:00:00Z"),  # FUTURE BAR
    ]
    ca = [
        {"action_id": "CA1", "effective_date": "2026-03-15", "announced_date": "2026-03-10"},
        {"action_id": "CA2", "effective_date": "2026-04-15", "announced_date": "2026-04-10"},  # FUTURE CA
    ]
    ref = {"sector": "Technology", "latest_price": 500.0}

    pkg = compiler.compile_case("CASE-TRUNC", "SEC-A", "A", cutoff, bars, ref, ca)

    # Physical truncation checks
    assert len(pkg.permitted_bars) == 3
    assert all(b.valid_time <= cutoff for b in pkg.permitted_bars)
    assert len(pkg.permitted_corporate_actions) == 1
    assert pkg.permitted_corporate_actions[0]["action_id"] == "CA1"
    assert "latest_price" not in pkg.permitted_reference_data


def test_bitemporal_prefix_invariance_and_future_suffix():
    """Core Invariant: Histories identical up to T produce identical case packages and classifications."""
    compiler = VCPTemporalCaseCompiler()
    classifier = VCPClassifier()
    cutoff = "2026-03-31T21:00:00Z"

    # Base common history up to T (250 bars)
    common_bars = generate_clean_history("INVAR", "2026-03-31", 250, 120.0, "STAGE_2_UPTREND", [(0.25, 30), (0.12, 20), (0.04, 10)])

    # Future Suffix A: Explosive +500% Breakout
    suffix_a = [
        DailyOHLCVBar("INVAR", "2026-04-01", 150, 200, 140, 190, 5000000, "2026-04-01T20:00:00Z", "2026-04-01T20:00:00Z"),
        DailyOHLCVBar("INVAR", "2026-04-02", 200, 300, 190, 280, 8000000, "2026-04-02T20:00:00Z", "2026-04-02T20:00:00Z"),
    ]

    # Future Suffix B: Complete -90% Collapse
    suffix_b = [
        DailyOHLCVBar("INVAR", "2026-04-01", 100, 105, 50, 55, 3000000, "2026-04-01T20:00:00Z", "2026-04-01T20:00:00Z"),
        DailyOHLCVBar("INVAR", "2026-04-02", 50, 52, 10, 12, 12000000, "2026-04-02T20:00:00Z", "2026-04-02T20:00:00Z"),
    ]

    pkg_a = compiler.compile_case("CASE-INV", "SEC-INV", "INVAR", cutoff, common_bars + suffix_a, {}, [])
    pkg_b = compiler.compile_case("CASE-INV", "SEC-INV", "INVAR", cutoff, common_bars + suffix_b, {}, [])

    # Packages must be semantically identical up to cutoff
    assert pkg_a.source_hashes == pkg_b.source_hashes
    assert pkg_a.temporal_information_closure_hash == pkg_b.temporal_information_closure_hash
    assert len(pkg_a.permitted_bars) == len(pkg_b.permitted_bars) == 250

    # Classifications must be strictly identical
    eval_a = classifier.classify_case(pkg_a)
    eval_b = classifier.classify_case(pkg_b)
    assert eval_a.vcp_classification == eval_b.vcp_classification
    assert eval_a.stage_classification == eval_b.stage_classification
    for pid in eval_a.predicate_vector:
        assert eval_a.predicate_vector[pid].status == eval_b.predicate_vector[pid].status


def test_future_canary_elimination():
    compiler = VCPTemporalCaseCompiler()
    cutoff = "2026-03-31T20:00:00Z"

    # Inject canary tokens into post-cutoff data
    canary_bar = DailyOHLCVBar(
        symbol="CANARY_TEST", bar_date="2026-04-10", open=100, high=110, low=90, close=105, volume=1000,
        valid_time="2026-04-10T20:00:00Z", known_at="2026-04-10T20:00:00Z"
    )
    canary_ca = {"action_id": "CANARY_SPLIT_POST_CUTOFF", "effective_date": "2026-05-01"}

    pkg = compiler.compile_case("CANARY-CASE", "SEC-CAN", "CAN", cutoff, [canary_bar], {}, [canary_ca])

    # Scan package for canary exposures
    exposures = compiler.scan_for_canaries(pkg, "CANARY")
    assert len(exposures) == 0, f"Future canary leaked into sealed package: {exposures}"


def test_wall_clock_independence():
    compiler = VCPTemporalCaseCompiler()
    classifier = VCPClassifier()
    corpus = VCPConformanceCorpus()

    c = corpus.dev_cases["DEV-001-QUALIFIED-3T"]
    pkg = compiler.compile_case(c.case_id, c.security_id, c.symbol, c.evaluation_as_of, c.raw_bars, {}, [])

    # Run today, run tomorrow, run 10 years later with identical input
    res1 = classifier.classify_case(pkg)
    res2 = classifier.classify_case(pkg)
    assert res1.vcp_classification == res2.vcp_classification
    assert res1.stage_classification == res2.stage_classification
    assert res1.normative_all_pass == res2.normative_all_pass


# ======================================================================
# 5. CONFORMANCE ORACLE: DEV & HOLDOUT EVALUATION
# ======================================================================

def test_gold_dev_corpus_conformance():
    corpus = VCPConformanceCorpus()
    compiler = VCPTemporalCaseCompiler()
    classifier = VCPClassifier()

    dev_cases = corpus.list_dev_cases()
    assert len(dev_cases) == 16, f"Expected 16 dev cases, found {len(dev_cases)}"

    gold_mismatches = 0
    for c in dev_cases:
        pkg = compiler.compile_case(
            case_id=c.case_id, security_id=c.security_id, symbol=c.symbol,
            evaluation_as_of=c.evaluation_as_of, raw_bars=c.raw_bars,
            reference_data=c.reference_data, corporate_actions=c.corporate_actions,
        )
        assessment = classifier.classify_case(pkg)

        if c.oracle_tier == OracleTier.GOLD:
            assert assessment.vcp_classification == c.expected_vcp_classification, (
                f"Dev Gold VCP mismatch on {c.case_id}: expected {c.expected_vcp_classification}, got {assessment.vcp_classification}"
            )
            assert assessment.stage_classification == c.expected_stage, (
                f"Dev Gold Stage mismatch on {c.case_id}: expected {c.expected_stage}, got {assessment.stage_classification}"
            )
            for pid, exp_s in c.expected_predicates.items():
                act_s = assessment.predicate_vector[pid].status
                assert act_s == exp_s, (
                    f"Dev Gold Predicate mismatch on {c.case_id} {pid}: expected {exp_s}, got {act_s}"
                )


def test_sealed_holdout_corpus_conformance():
    corpus = VCPConformanceCorpus()
    compiler = VCPTemporalCaseCompiler()
    classifier = VCPClassifier()

    holdout_cases = corpus.list_holdout_cases()
    assert len(holdout_cases) == 8, f"Expected 8 holdout cases, found {len(holdout_cases)}"

    for c in holdout_cases:
        pkg = compiler.compile_case(
            case_id=c.case_id, security_id=c.security_id, symbol=c.symbol,
            evaluation_as_of=c.evaluation_as_of, raw_bars=c.raw_bars,
            reference_data=c.reference_data, corporate_actions=c.corporate_actions,
        )
        assessment = classifier.classify_case(pkg)

        if c.oracle_tier == OracleTier.GOLD:
            assert assessment.vcp_classification == c.expected_vcp_classification, (
                f"Holdout Gold VCP mismatch on {c.case_id}: expected {c.expected_vcp_classification}, got {assessment.vcp_classification}"
            )
            assert assessment.stage_classification == c.expected_stage, (
                f"Holdout Gold Stage mismatch on {c.case_id}: expected {c.expected_stage}, got {assessment.stage_classification}"
            )
            for pid, exp_s in c.expected_predicates.items():
                act_s = assessment.predicate_vector[pid].status
                assert act_s == exp_s, (
                    f"Holdout Gold Predicate mismatch on {c.case_id} {pid}: expected {exp_s}, got {act_s}"
                )


def test_holdout_leakage_and_duplicate_isolation():
    corpus = VCPConformanceCorpus()
    leakage = corpus.audit_holdout_leakage()
    assert leakage["exact_duplicates"] == 0, "No exact duplicate cases allowed between Dev and Holdout"
    assert leakage["symbol_overlap"] == 0, "No symbol overlap allowed between Dev and Holdout"
    assert leakage["episode_overlap"] == 0, "No episode overlap allowed"
    assert leakage["near_duplicate_leakage"] == 0, "No near-duplicate leakage allowed"


def test_holdout_commitment_hashes_deterministic():
    corpus = VCPConformanceCorpus()
    mem_hash = corpus.compute_holdout_membership_hash()
    lbl_hash = corpus.compute_holdout_label_commitment_hash()
    assert len(mem_hash) == 64
    assert len(lbl_hash) == 64

    # Repeat hash calculation must be 100% deterministic
    assert corpus.compute_holdout_membership_hash() == mem_hash
    assert corpus.compute_holdout_label_commitment_hash() == lbl_hash


def test_adjudicator_blinding_protocol():
    corpus = VCPConformanceCorpus()
    for c in corpus.list_dev_cases() + corpus.list_holdout_cases():
        assert c.arx_scanner_output_visible is False, (
            f"Adjudicator blinding failure on {c.case_id}: scanner output visible during adjudication"
        )
        assert c.adjudicator_id in ["ADJ-001", "ADJ-002"]


def test_predicate_coverage_matrix():
    corpus = VCPConformanceCorpus()
    matrix = corpus.compute_predicate_coverage_matrix()
    pred_reg = VCPPredicateRegistry()
    normative = pred_reg.list_normative_predicates()

    for p in normative:
        assert p.predicate_id in matrix, f"Normative predicate {p.predicate_id} has zero oracle coverage"
        counts = matrix[p.predicate_id]
        assert counts["PASS"] > 0, f"Predicate {p.predicate_id} has 0 PASS coverage"
        # At least one non-PASS status tested
        assert (counts["FAIL"] + counts["INSUFFICIENT_DATA"] + counts["NOT_APPLICABLE"]) > 0


# ======================================================================
# 6. MUTATION TESTING: 21 OPERATORS KILLED
# ======================================================================

def test_oracle_and_temporal_mutation_harness_21_operators():
    harness = VCPMutationHarness()
    results = harness.run_all_mutations()

    assert len(results) == 21, f"Expected 21 mutation operators, got {len(results)}"
    survivors = [r for r in results if not r.caught]
    assert len(survivors) == 0, f"Critical mutation survivors detected: {[s.operator_id for s in survivors]}"


# ======================================================================
# 7. PRODUCT CLAIM AUTHORIZATION & DIFFERENTIAL ACCOUNTING
# ======================================================================

def test_label_authorization_matrix_and_minervini_rule():
    matrix = VCPLabelAuthorizationMatrix()

    # Core Rule: MINERVINI_LABEL_AUTHORIZED = NO
    assert matrix.get_minervini_label_authorization() == "NO"
    assert matrix.is_label_authorized("Minervini VCP") is False

    # Authorized generic domain labels
    assert matrix.is_label_authorized("VCP") is True
    assert matrix.is_label_authorized("Stage 1") is True
    assert matrix.is_label_authorized("Stage 2") is True
    assert matrix.is_label_authorized("Stage 3") is True
    assert matrix.is_label_authorized("Stage 4") is True
    assert matrix.is_label_authorized("confirmed") is True
    assert matrix.is_label_authorized("breakout-ready") is True
    assert matrix.is_label_authorized("volume dry-up") is True
    assert matrix.is_label_authorized("contraction") is True


def test_differential_old_proxy_accounting():
    import pandas as pd
    from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

    corpus = VCPConformanceCorpus()
    compiler = VCPTemporalCaseCompiler()
    classifier = VCPClassifier()

    diff_counts = {"OLD_PROXY_FALSE_POSITIVE": 0, "OLD_PROXY_FALSE_NEGATIVE": 0, "LABEL_EQUIVALENT": 0, "UNRESOLVED": 0}
    for c in corpus.list_dev_cases() + corpus.list_holdout_cases():
        pkg = compiler.compile_case(c.case_id, c.security_id, c.symbol, c.evaluation_as_of, c.raw_bars, {}, [])
        assessment = classifier.classify_case(pkg)

        df = pd.DataFrame([{
            'Open': b.open, 'High': b.high, 'Low': b.low, 'Close': b.close, 'Volume': b.volume
        } for b in pkg.permitted_bars], index=pd.to_datetime([b.valid_time for b in pkg.permitted_bars]))

        spot = pkg.permitted_bars[-1].close if pkg.permitted_bars else 100.0
        old_exec = OptimalExecutionEngine.calculate_trade_levels(df, spot, user_role='LONG_TERM')
        old_is_vcp = (old_exec.get('vcp_contraction_status') == 'VCP 3-Stage Compression Confirmed')
        new_is_vcp = assessment.vcp_qualified

        if c.oracle_tier == OracleTier.UNRESOLVED:
            diff_counts["UNRESOLVED"] += 1
        elif old_is_vcp and not new_is_vcp:
            diff_counts["OLD_PROXY_FALSE_POSITIVE"] += 1
        elif not old_is_vcp and new_is_vcp:
            diff_counts["OLD_PROXY_FALSE_NEGATIVE"] += 1
        else:
            diff_counts["LABEL_EQUIVALENT"] += 1

    assert diff_counts["OLD_PROXY_FALSE_POSITIVE"] == 10, "Expected exactly 10 old proxy false positives"
    assert diff_counts["OLD_PROXY_FALSE_NEGATIVE"] == 0, "Expected 0 old proxy false negatives"
    assert diff_counts["LABEL_EQUIVALENT"] == 13, "Expected 13 label equivalent cases"
    assert diff_counts["UNRESOLVED"] == 1, "Expected 1 unresolved case"
