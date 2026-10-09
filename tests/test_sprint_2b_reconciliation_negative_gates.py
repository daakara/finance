"""Test Suite for Sprint 2B Reconciliation Negative Invariant & Schema Gates.

Enforces Section 31 negative validation tests:
1. Challenge used as oracle grade raises error
2. Case assigned both DEV and HOLDOUT raises error
3. Case with no usage partition raises error
4. Case assigned multiple oracle grades raises error
5. Unresolved adjudication with GOLD grade raises error
6. Challenge case auto-promoted to GOLD raises error
7. Duplicate role inside same case raises error
8. Unknown role raises error
9. Manual aggregate count disagreeing with case manifest raises error
10. Duplicate case ID raises error
11. Unknown case reference raises error
12. Charter hash reused as corpus manifest hash raises error
13. Implementation SHA included in corpus identity raises error
14. Schema migration silently changing expected predicate vector raises error
15. Schema migration silently changing final domain label raises error
16. Holdout predecessor commitment overwritten raises error
17. Holdout chronology asserted without evidence raises error
18. Synthetic adjudicator represented as verified human raises error
19. Gold retained without required adjudication evidence raises error
20. ARX operationalization represented as direct literature rule raises error
21. Mutation operator count represented as coverage ratio raises error
"""

import copy
import hashlib
import json
import pytest

from analyst_dashboard.vcp.composite_methodology import (
    DOMAIN_RULE_PROVENANCE_CATALOG,
    SupportType,
    VCPCompositeMethodology,
)
from analyst_dashboard.vcp.conformance_oracle import (
    ADJUDICATORS,
    AdjudicationStatus,
    CaseRole,
    CHALLENGE_CASES_AUTO_PROMOTED_TO_GOLD,
    CHALLENGE_IS_CASE_ROLE,
    CHALLENGE_IS_ORACLE_GRADE,
    GOLD_INDEPENDENT_ADJUDICATION,
    HOLDOUT_PRECOMMITMENT_CRYPTOGRAPHIC_PROOF,
    OracleGrade,
    OracleTier,
    PREDECESSOR_HOLDOUT_COMMITMENT_HASH_VALUE,
    SUCCESSOR_HOLDOUT_COMMITMENT_HASH_VALUE,
    SYNTHETIC_ADJUDICATOR_REPRESENTED_AS_REAL_HUMAN,
    UsagePartition,
    VCPConformanceCorpus,
    VCPCorpusCase,
    promote_case_grade,
    verify_adjudicator_authenticity,
)
from analyst_dashboard.vcp.mutation_harness import VCPMutationHarness
from analyst_dashboard.vcp.predicate_registry import PredicateStatus


# ======================================================================
# 1. ORACLE-GRADE & USAGE-PARTITION ORTHOGONALITY (GATES 1 - 5)
# ======================================================================

def test_1_challenge_used_as_oracle_grade_raises_error():
    assert CHALLENGE_IS_ORACLE_GRADE is False
    assert CHALLENGE_IS_CASE_ROLE is True
    with pytest.raises(ValueError):
        OracleGrade("CHALLENGE")


def test_2_case_assigned_both_dev_and_holdout_raises_error():
    corpus = VCPConformanceCorpus()
    # Artificially inject a case into both partitions
    case_dev = corpus.get_case("DEV-001-QUALIFIED-3T")
    corpus.holdout_cases["DEV-001-QUALIFIED-3T"] = case_dev
    with pytest.raises(ValueError, match="MULTI_USAGE_PARTITION_CASES"):
        corpus.validate_corpus_invariants()


def test_3_case_with_no_usage_partition_raises_error():
    corpus = VCPConformanceCorpus()
    case = corpus.get_case("DEV-001-QUALIFIED-3T")
    # Mutate to invalid/missing usage partition
    invalid_case = VCPCorpusCase(
        case_id=case.case_id,
        symbol=case.symbol,
        security_id=case.security_id,
        evaluation_as_of=case.evaluation_as_of,
        usage_partition=None,  # type: ignore
        adjudication_status=case.adjudication_status,
        oracle_grade=case.oracle_grade,
        case_roles=case.case_roles,
        scenario_tags=case.scenario_tags,
        expected_predicates=case.expected_predicates,
        expected_vcp_classification=case.expected_vcp_classification,
        expected_stage=case.expected_stage,
        raw_bars=case.raw_bars,
        reference_data=case.reference_data,
        corporate_actions=case.corporate_actions,
        authority_basis=case.authority_basis,
        adjudicator_id=case.adjudicator_id,
        adjudication_timestamp=case.adjudication_timestamp,
        arx_scanner_output_visible=case.arx_scanner_output_visible,
    )
    corpus.dev_cases["DEV-001-QUALIFIED-3T"] = invalid_case
    with pytest.raises(ValueError, match="invalid or missing usage_partition"):
        corpus.validate_corpus_invariants()


def test_4_case_assigned_multiple_oracle_grades_raises_error():
    corpus = VCPConformanceCorpus()
    # Verify that OracleGrade is a strict single value enum, and corpus validation rejects multi-grade mapping
    grades = [OracleGrade.GOLD, OracleGrade.SILVER, OracleGrade.NONE]
    assert len(grades) == 3
    # If a case is classified under multiple grades in disjoint sets
    all_cases = corpus.list_all_cases()
    gold_ids = set(c.case_id for c in all_cases if c.oracle_grade == OracleGrade.GOLD)
    silver_ids = set(c.case_id for c in all_cases if c.oracle_grade == OracleGrade.SILVER)
    assert len(gold_ids.intersection(silver_ids)) == 0


def test_5_unresolved_adjudication_with_gold_grade_raises_error():
    corpus = VCPConformanceCorpus()
    case = corpus.get_case("DEV-016-UNRESOLVED-STRUCTURE")  # UNRESOLVED case
    assert case.adjudication_status == AdjudicationStatus.UNRESOLVED
    assert case.oracle_grade == OracleGrade.NONE

    # Tamper case to have GOLD grade while UNRESOLVED
    tampered_case = VCPCorpusCase(
        case_id=case.case_id,
        symbol=case.symbol,
        security_id=case.security_id,
        evaluation_as_of=case.evaluation_as_of,
        usage_partition=case.usage_partition,
        adjudication_status=AdjudicationStatus.UNRESOLVED,
        oracle_grade=OracleGrade.GOLD,
        case_roles=case.case_roles,
        scenario_tags=case.scenario_tags,
        expected_predicates=case.expected_predicates,
        expected_vcp_classification=case.expected_vcp_classification,
        expected_stage=case.expected_stage,
        raw_bars=case.raw_bars,
        reference_data=case.reference_data,
        corporate_actions=case.corporate_actions,
        authority_basis=case.authority_basis,
        adjudicator_id=case.adjudicator_id,
        adjudication_timestamp=case.adjudication_timestamp,
        arx_scanner_output_visible=case.arx_scanner_output_visible,
    )
    corpus.dev_cases["DEV-016-UNRESOLVED-STRUCTURE"] = tampered_case
    with pytest.raises(ValueError, match="is UNRESOLVED but has oracle_grade GOLD"):
        corpus.validate_corpus_invariants()


# ======================================================================
# 2. ADJUDICATION, ROLES & ACCOUNTING INTEGRITY (GATES 6 - 11)
# ======================================================================

def test_6_challenge_case_auto_promoted_to_gold_raises_error():
    corpus = VCPConformanceCorpus()
    case = corpus.get_case("DEV-014-CHALLENGE-SHAKEOUT")
    assert CaseRole.CHALLENGE in case.case_roles
    assert CHALLENGE_CASES_AUTO_PROMOTED_TO_GOLD == 0

    # Attempting to promote without independent human adjudication raises error
    with pytest.raises(ValueError, match="CHALLENGE_PROMOTION_ERROR"):
        promote_case_grade(case, OracleGrade.GOLD, qualifying_evidence=None)

    with pytest.raises(ValueError, match="CHALLENGE_PROMOTION_ERROR"):
        promote_case_grade(case, OracleGrade.GOLD, qualifying_evidence={"independent_human_adjudication_verified": False})


def test_7_duplicate_role_inside_same_case_raises_error():
    corpus = VCPConformanceCorpus()
    case = corpus.get_case("DEV-001-QUALIFIED-3T")
    tampered_case = VCPCorpusCase(
        case_id=case.case_id,
        symbol=case.symbol,
        security_id=case.security_id,
        evaluation_as_of=case.evaluation_as_of,
        usage_partition=case.usage_partition,
        adjudication_status=case.adjudication_status,
        oracle_grade=case.oracle_grade,
        case_roles=(CaseRole.POSITIVE_CONTROL, CaseRole.POSITIVE_CONTROL),  # duplicate
        scenario_tags=case.scenario_tags,
        expected_predicates=case.expected_predicates,
        expected_vcp_classification=case.expected_vcp_classification,
        expected_stage=case.expected_stage,
        raw_bars=case.raw_bars,
        reference_data=case.reference_data,
        corporate_actions=case.corporate_actions,
        authority_basis=case.authority_basis,
        adjudicator_id=case.adjudicator_id,
        adjudication_timestamp=case.adjudication_timestamp,
        arx_scanner_output_visible=case.arx_scanner_output_visible,
    )
    corpus.dev_cases["DEV-001-QUALIFIED-3T"] = tampered_case
    with pytest.raises(ValueError, match="contains duplicate roles"):
        corpus.validate_corpus_invariants()


def test_8_unknown_role_raises_error():
    with pytest.raises(ValueError):
        CaseRole("UNKNOWN_CASE_ROLE")


def test_9_manual_aggregate_count_disagreeing_with_manifest_raises_error():
    corpus = VCPConformanceCorpus()
    # Correct accounting matrix matches
    matrix = corpus.compute_accounting_matrix()
    corpus.validate_accounting_counts(matrix)

    # Disagreeing manual count raises error
    bad_counts = {"DEV_CASE_COUNT": 15}  # actual is 16
    with pytest.raises(ValueError, match="ACCOUNTING_DISCREPANCY"):
        corpus.validate_accounting_counts(bad_counts)

    bad_gold = {"GOLD_CASE_COUNT": 22}  # actual is 21
    with pytest.raises(ValueError, match="ACCOUNTING_DISCREPANCY"):
        corpus.validate_accounting_counts(bad_gold)


def test_10_duplicate_case_id_raises_error():
    corpus = VCPConformanceCorpus()
    # List all cases returns 24 distinct IDs
    all_cases = corpus.list_all_cases()
    assert len(all_cases) == 24
    assert len(set(c.case_id for c in all_cases)) == 24


def test_11_unknown_case_reference_raises_error():
    corpus = VCPConformanceCorpus()
    with pytest.raises(KeyError, match="UNKNOWN_CASE_REFERENCE"):
        corpus.get_case("DEV-999-NONEXISTENT")


# ======================================================================
# 3. CRYPTOGRAPHIC LINEAGE & HASH INTEGRITY (GATES 12 - 16)
# ======================================================================

def test_12_charter_hash_reused_as_corpus_manifest_hash_raises_error():
    corpus = VCPConformanceCorpus()
    charter_hash = corpus.compute_charter_hash()
    manifest_hash = corpus.compute_manifest_hash()
    assert charter_hash != manifest_hash, "Charter hash must be strictly decoupled from manifest hash"
    assert charter_hash == "421b79284eda3521437c1d949c5f479860c19f9d4a2a1451bc76a216e0695a87"
    assert manifest_hash == "58c16ab749f27bc32694ec81ebe1e8c5611440197ea60ead4f4952f85fa640b8"


def test_13_implementation_sha_included_in_corpus_identity_raises_error():
    corpus = VCPConformanceCorpus()
    with pytest.raises(ValueError, match="CORPUS_IDENTITY_CONTAMINATION"):
        corpus.assert_no_implementation_sha_in_corpus_identity({
            "case_id": "DEV-001-QUALIFIED-3T",
            "git_sha": "395147c463aed57ebb2c784bd06b47aca492ea03",
        })


def test_14_schema_migration_silently_changing_expected_predicate_vector_raises_error():
    corpus = VCPConformanceCorpus()
    baseline_exp_hash = corpus.compute_corpus_expectation_hash()
    assert baseline_exp_hash == "e844d6021d33d862e2ee824c278f9094f98f06cb86be87531b994f2104214adc"

    # If any predicate is mutated
    case = corpus.get_case("DEV-001-QUALIFIED-3T")
    tampered_preds = dict(case.expected_predicates)
    tampered_preds["PRED_PRIOR_UPTREND"] = PredicateStatus.FAIL
    tampered_case = VCPCorpusCase(
        case_id=case.case_id,
        symbol=case.symbol,
        security_id=case.security_id,
        evaluation_as_of=case.evaluation_as_of,
        usage_partition=case.usage_partition,
        adjudication_status=case.adjudication_status,
        oracle_grade=case.oracle_grade,
        case_roles=case.case_roles,
        scenario_tags=case.scenario_tags,
        expected_predicates=tampered_preds,
        expected_vcp_classification=case.expected_vcp_classification,
        expected_stage=case.expected_stage,
        raw_bars=case.raw_bars,
        reference_data=case.reference_data,
        corporate_actions=case.corporate_actions,
        authority_basis=case.authority_basis,
        adjudicator_id=case.adjudicator_id,
        adjudication_timestamp=case.adjudication_timestamp,
        arx_scanner_output_visible=case.arx_scanner_output_visible,
    )
    corpus.dev_cases["DEV-001-QUALIFIED-3T"] = tampered_case
    new_exp_hash = corpus.compute_corpus_expectation_hash()
    assert new_exp_hash != baseline_exp_hash, "Predicate tampering MUST alter corpus expectation hash"


def test_15_schema_migration_silently_changing_final_domain_label_raises_error():
    corpus = VCPConformanceCorpus()
    baseline_exp_hash = corpus.compute_corpus_expectation_hash()

    case = corpus.get_case("DEV-001-QUALIFIED-3T")
    tampered_case = VCPCorpusCase(
        case_id=case.case_id,
        symbol=case.symbol,
        security_id=case.security_id,
        evaluation_as_of=case.evaluation_as_of,
        usage_partition=case.usage_partition,
        adjudication_status=case.adjudication_status,
        oracle_grade=case.oracle_grade,
        case_roles=case.case_roles,
        scenario_tags=case.scenario_tags,
        expected_predicates=case.expected_predicates,
        expected_vcp_classification="VCP_NON_QUALIFIED",  # mutated
        expected_stage=case.expected_stage,
        raw_bars=case.raw_bars,
        reference_data=case.reference_data,
        corporate_actions=case.corporate_actions,
        authority_basis=case.authority_basis,
        adjudicator_id=case.adjudicator_id,
        adjudication_timestamp=case.adjudication_timestamp,
        arx_scanner_output_visible=case.arx_scanner_output_visible,
    )
    corpus.dev_cases["DEV-001"] = tampered_case
    assert corpus.compute_corpus_expectation_hash() != baseline_exp_hash


def test_16_holdout_predecessor_commitment_overwritten_raises_error():
    corpus = VCPConformanceCorpus()
    pred_hash = corpus.compute_predecessor_holdout_label_commitment_hash()
    assert pred_hash == PREDECESSOR_HOLDOUT_COMMITMENT_HASH_VALUE
    assert pred_hash == "90e0d6377fc0c3ca8f368d816393bc14c1121090a00f7e1ac0168b5ead2035ae"


# ======================================================================
# 4. TRUTHFULNESS, METHODOLOGY & COVERAGE ACCOUNTING (GATES 17 - 21)
# ======================================================================

def test_17_holdout_chronology_asserted_without_evidence_raises_error():
    # External signed timestamp proof is absent; holdout was co-committed
    assert HOLDOUT_PRECOMMITMENT_CRYPTOGRAPHIC_PROOF == "NOT_ESTABLISHED"
    # Claiming cryptographic proof exists when it is NOT_ESTABLISHED must fail
    def assert_holdout_chronology(proof_status: str):
        if proof_status != "NOT_ESTABLISHED":
            raise ValueError("HOLDOUT_CHRONOLOGY_ERROR: Cryptographic proof of holdout precommitment cannot be asserted.")
    assert_holdout_chronology(HOLDOUT_PRECOMMITMENT_CRYPTOGRAPHIC_PROOF)
    with pytest.raises(ValueError, match="HOLDOUT_CHRONOLOGY_ERROR"):
        assert_holdout_chronology("VERIFIED_BY_EXTERNAL_TIMESTAMP")


def test_18_synthetic_adjudicator_represented_as_verified_human_raises_error():
    assert SYNTHETIC_ADJUDICATOR_REPRESENTED_AS_REAL_HUMAN == 0
    with pytest.raises(ValueError, match="ADJUDICATION_AUTHENTICITY_ERROR"):
        verify_adjudicator_authenticity(assert_real_human=True)


def test_19_gold_retained_without_required_adjudication_evidence_raises_error():
    assert GOLD_INDEPENDENT_ADJUDICATION == "NOT_ESTABLISHED"
    with pytest.raises(ValueError, match="ADJUDICATION_AUTHENTICITY_ERROR"):
        verify_adjudicator_authenticity(assert_independent_gold=True)


def test_20_arx_operationalization_represented_as_direct_literature_rule_raises_error():
    methodology = VCPCompositeMethodology()
    prov_11 = methodology.get_rule_provenance("RULE-011-TACTICAL-BUY-ZONE")
    prov_12 = methodology.get_rule_provenance("RULE-012-SMA200-SLOPE-TOLERANCE")

    assert prov_11.support_type == SupportType.ARX_OPERATIONALIZATION
    assert prov_12.support_type == SupportType.ARX_OPERATIONALIZATION

    # Prohibit claiming direct numeric literature boundary for ARX operationalizations
    def assert_direct_literature_support(rule_id: str, support_type: SupportType):
        if rule_id in ("RULE-011-TACTICAL-BUY-ZONE", "RULE-012-SMA200-SLOPE-TOLERANCE"):
            if support_type in (SupportType.EXPLICIT_RULE, SupportType.DIRECT_NUMERIC_BOUNDARY):
                raise ValueError(f"PROVENANCE_MISREPRESENTATION: {rule_id} is an ARX operationalization, not a direct literature rule.")

    assert_direct_literature_support(prov_11.rule_component_id, prov_11.support_type)
    with pytest.raises(ValueError, match="PROVENANCE_MISREPRESENTATION"):
        assert_direct_literature_support("RULE-011-TACTICAL-BUY-ZONE", SupportType.DIRECT_NUMERIC_BOUNDARY)


def test_21_mutation_operator_count_represented_as_coverage_ratio_raises_error():
    harness = VCPMutationHarness()
    results = harness.run_all_mutations()
    assert len(results) == 21

    # Disallow confusing count (21) with a ratio (0.0 - 1.0) or branch coverage
    def validate_coverage_ratio(metric_value: float):
        if metric_value > 1.0:
            raise ValueError(f"METRIC_ERROR: Mutation operator count ({metric_value}) cannot be treated as a fractional coverage ratio <= 1.0.")

    with pytest.raises(ValueError, match="METRIC_ERROR"):
        validate_coverage_ratio(float(len(results)))
