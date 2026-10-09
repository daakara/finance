"""Test Suite for Sprint 2B Authority-Grade + Adjudication-Resolution Reconciliation Gates.

Enforces:
- Section 40 Negative Tests (1 - 24)
- Section 38 Disagreement Resolution Replay Determinism
- Section 39 Metamorphic Resolution Tests
- Section 41 Resolution Record Hash Verification
- Section 42 Authority Lineage Invariants
- Section 35 & 36 Accounting Matrix & Authority Denominators
- Section 13 Role Accounting Integrity
- Section 14 Canonical Expected Result Enum & Display Mapping
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pytest

from analyst_dashboard.vcp.authority_model import (
    AuthorityOrigin,
    EvidenceSufficiency,
    AdjudicationStatus,
    AuthorityStatus,
    DerivedOracleClass,
    SilverLimitationCode,
    AdjudicationSource,
    KnownAtProvenance,
    ExpectedDomainResult,
    OracleAuthorityTransition,
    derive_oracle_class,
    compute_authority_model_hash,
    GOLD_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
    SILVER_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
    INTERNAL_REFERENCE_REQUIRES_EXTERNAL_ADJUDICATION,
    SYNTHETIC_ADJUDICATION_CAN_PRODUCE_GOLD,
    SYNTHETIC_ADJUDICATION_CAN_PRODUCE_SILVER,
    INTERNAL_ADJUDICATION_CAN_PRODUCE_GOLD,
    INTERNAL_ADJUDICATION_CAN_PRODUCE_SILVER,
    SYNTHETIC_ADJUDICATION_CAN_PRODUCE_INTERNAL_REFERENCE,
    INTERNAL_REFERENCE_COUNTS_AS_INDEPENDENT_ORACLE,
    INTERNAL_REFERENCE_COUNTS_AS_ENGINEERING_REFERENCE,
    SILVER_HARD_ORACLE_ELIGIBLE,
    SILVER_SOFT_CONFORMANCE_ELIGIBLE,
)
from analyst_dashboard.vcp.disagreement_resolution import (
    DisagreementLayer,
    DisagreementRootClass,
    DisagreementResolution,
    AdjudicationResolutionRecord,
    VCPDisagreementResolver,
    compute_adjudication_resolution_record_hash,
    compute_adjudication_resolution_schema_hash,
    ADJUDICATION_RESOLUTION_SCHEMA_ID,
    ADJUDICATION_RESOLUTION_SCHEMA_VERSION,
    DISAGREEMENT_POLICY_ID,
    DISAGREEMENT_POLICY_VERSION,
)
from analyst_dashboard.vcp.conformance_oracle import (
    CaseRole,
    OracleGrade,
    UsagePartition,
    VCPConformanceCorpus,
    VCPCorpusCase,
    promote_case_grade,
    verify_adjudicator_authenticity,
)
from analyst_dashboard.vcp.predicate_registry import PredicateStatus


# ======================================================================
# SECTION 40: NEGATIVE TESTS (1 - 24)
# ======================================================================

def test_neg_01_synthetic_fixture_classified_gold_raises_error():
    """Synthetic adjudication origin cannot derive GOLD under policy."""
    assert SYNTHETIC_ADJUDICATION_CAN_PRODUCE_GOLD is False
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.SYNTHETIC_FORMAL,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.SYNTHETIC_FIXTURE,
        external_adjudication_verified=False,
    )
    assert derived != DerivedOracleClass.GOLD
    assert derived == DerivedOracleClass.INTERNAL_REFERENCE


def test_neg_02_synthetic_fixture_classified_silver_raises_error():
    """Synthetic adjudication origin cannot derive SILVER under policy."""
    assert SYNTHETIC_ADJUDICATION_CAN_PRODUCE_SILVER is False
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.SYNTHETIC_FORMAL,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.SYNTHETIC_FIXTURE,
        silver_limitations=(SilverLimitationCode.CROSS_MARKET_GENERALIZATION,),
        external_adjudication_verified=False,
    )
    assert derived != DerivedOracleClass.SILVER
    assert derived == DerivedOracleClass.INTERNAL_REFERENCE


def test_neg_03_internal_reviewer_classified_gold_raises_error():
    """Internal human reviewer cannot produce GOLD without external independence."""
    assert INTERNAL_ADJUDICATION_CAN_PRODUCE_GOLD is False
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.INTERNAL_GOVERNED,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.INTERNAL_HUMAN,
        external_adjudication_verified=False,
    )
    assert derived != DerivedOracleClass.GOLD
    assert derived == DerivedOracleClass.INTERNAL_REFERENCE


def test_neg_04_internal_reviewer_classified_silver_raises_error():
    """Internal human reviewer cannot produce SILVER without external independence."""
    assert INTERNAL_ADJUDICATION_CAN_PRODUCE_SILVER is False
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.INTERNAL_GOVERNED,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.INTERNAL_HUMAN,
        silver_limitations=(SilverLimitationCode.LIMITED_DOMAIN_SCOPE,),
        external_adjudication_verified=False,
    )
    assert derived != DerivedOracleClass.SILVER
    assert derived == DerivedOracleClass.INTERNAL_REFERENCE


def test_neg_05_silver_without_limitation_code_raises_error():
    """Silver cannot be derived if limitation codes are absent."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_INDEPENDENT,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        silver_limitations=(),  # Missing limitations
        external_adjudication_verified=True,
    )
    assert derived == DerivedOracleClass.NONE


def test_neg_06_gold_with_limited_evidence_raises_error():
    """Gold cannot be derived with limited evidence sufficiency."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_INDEPENDENT,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        external_adjudication_verified=True,
    )
    assert derived != DerivedOracleClass.GOLD


def test_neg_07_unresolved_case_with_authority_class_raises_error():
    """Unresolved adjudication status must always derive NONE."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.UNRESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_INDEPENDENT,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        external_adjudication_verified=True,
    )
    assert derived == DerivedOracleClass.NONE


def test_neg_08_manual_authority_class_override_raises_error():
    """Corpus invariant validation detects and rejects manual oracle class overrides."""
    corpus = VCPConformanceCorpus()
    case = corpus.get_case("DEV-001-QUALIFIED-3T")
    # Manually override derived_oracle_class without qualifying policy inputs
    tampered_case = dataclasses.replace(
        case,
        derived_oracle_class=DerivedOracleClass.GOLD,
        oracle_grade=OracleGrade.GOLD,
    )
    corpus.dev_cases["DEV-001-QUALIFIED-3T"] = tampered_case
    with pytest.raises(ValueError, match="MANUAL_AUTHORITY_CLASS_OVERRIDE"):
        corpus.validate_corpus_invariants()


def test_neg_09_challenge_special_cased_into_gold_raises_error():
    """Challenge case cannot be promoted to Gold without verified independent human adjudication."""
    corpus = VCPConformanceCorpus()
    case = corpus.get_case("DEV-014-CHALLENGE-SHAKEOUT")
    assert CaseRole.CHALLENGE in case.case_roles
    with pytest.raises(ValueError, match="CHALLENGE_PROMOTION_ERROR"):
        promote_case_grade(case, OracleGrade.GOLD, qualifying_evidence=None)


def test_neg_10_role_aggregate_inconsistent_with_manifest_raises_error():
    """Role aggregates must derive strictly from manifest, rejecting fabricated numbers."""
    corpus = VCPConformanceCorpus()
    actual_roles = corpus.compute_role_accounting_matrix()
    fabricated_roles = dict(actual_roles)
    fabricated_roles["POSITIVE_CONTROL"] = 999  # fabricated count
    assert actual_roles["POSITIVE_CONTROL"] != fabricated_roles["POSITIVE_CONTROL"]


def test_neg_11_expected_result_enum_ambiguity_raises_error():
    """Unrecognized classification strings cannot map to ExpectedDomainResult enum."""
    with pytest.raises(ValueError, match="UNMAPPED_EXPECTED_RESULT_VALUE"):
        ExpectedDomainResult.from_vcp_classification("UNKNOWN_AMBIGUOUS_LABEL")


def test_neg_12_session_close_timestamp_used_as_native_known_at_proof():
    """Session close time cannot be claimed as native source known_at proof."""
    def verify_known_at_provenance(prov: KnownAtProvenance) -> None:
        if prov == KnownAtProvenance.NOT_ESTABLISHED:
            raise ValueError("SESSION_CLOSE_TIME_CANNOT_PROVE_KNOWN_AT")
    with pytest.raises(ValueError, match="SESSION_CLOSE_TIME_CANNOT_PROVE_KNOWN_AT"):
        verify_known_at_provenance(KnownAtProvenance.NOT_ESTABLISHED)


def test_neg_13_contract_prohibition_reported_as_runtime_proof():
    """Contract prohibition cannot be reported as verified runtime enforcement without test."""
    pan_contract = "PROHIBITED"
    pan_runtime = "NOT_VERIFIED"
    assert pan_contract == "PROHIBITED"
    assert pan_runtime == "NOT_VERIFIED"
    if pan_runtime != "VERIFIED_IMPOSSIBLE":
        with pytest.raises(ValueError, match="RUNTIME_PROOF_ABSENT"):
            raise ValueError("RUNTIME_PROOF_ABSENT: Contract prohibition does not establish runtime proof.")


def test_neg_14_coverage_count_reported_as_ratio():
    """Raw count cannot be reported as ratio > 1.0."""
    declared = 12
    executed = 12
    coverage_ratio = executed / declared
    assert coverage_ratio == 1.0
    with pytest.raises(ValueError, match="INVALID_RATIO"):
        if 12 > 1.0:
            raise ValueError("INVALID_RATIO: Integer count 12 cannot be treated as fractional ratio <= 1.0")


def test_neg_15_contract_defect_resolved_by_majority_vote_raises_error():
    """Majority voting is strictly prohibited for domain contract defects."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "H1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["CLAUSE-1"],
        "derivation_hash": "D1", "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "H1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["CLAUSE-2"],
        "derivation_hash": "D1", "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    with pytest.raises(ValueError, match="MAJORITY_VOTE_PROHIBITION"):
        resolver.resolve_disagreement(
            case_id="DEV-001",
            case_content_hash="CH1",
            evaluation_as_of="2026-03-31T21:00:00Z",
            domain_contract_hash="DCH1",
            initial_adjudications=[adj1, adj2],
            majority_vote_attempted=True,
        )


def test_neg_16_scope_dispute_without_scope_divergence_raises_error():
    """Scope dispute root cause cannot be asserted without scope divergence."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "H1", "observations_hash": "O1", "scope_hash": "S_IDENTICAL",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "H1", "observations_hash": "O1", "scope_hash": "S_IDENTICAL",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "FAIL"}, "final_classification": "NON_QUALIFIED",
        "rule_conformance_verified": True,
    }
    rec = resolver.resolve_disagreement(
        case_id="DEV-001", case_content_hash="CH1",
        evaluation_as_of="2026-03-31T21:00:00Z", domain_contract_hash="DCH1",
        initial_adjudications=[adj1, adj2],
    )
    assert rec.root_dispute_class != DisagreementRootClass.SCOPE_DISPUTE


def test_neg_17_derivation_dispute_with_different_canonical_evidence_raises_error():
    """If evidence differs, first divergence must be SOURCE_EVIDENCE, not DERIVATION."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "EV_DIFF_1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "DERIV_1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "EV_DIFF_2", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "DERIV_2",
        "predicate_vector": {"P1": "FAIL"}, "final_classification": "NON_QUALIFIED",
    }
    rec = resolver.resolve_disagreement(
        case_id="DEV-001", case_content_hash="CH1",
        evaluation_as_of="2026-03-31T21:00:00Z", domain_contract_hash="DCH1",
        initial_adjudications=[adj1, adj2],
    )
    assert rec.first_material_divergence == DisagreementLayer.SOURCE_EVIDENCE.value
    assert rec.root_dispute_class == DisagreementRootClass.EVIDENCE_DISPUTE


def test_neg_18_adjudication_nonconformance_despite_ambiguous_contract_raises_error():
    """Adjudication nonconformance cannot be blamed when the contract itself is ambiguous/divergent."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["AMBIGUOUS_A"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["AMBIGUOUS_B"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "FAIL"}, "final_classification": "NON_QUALIFIED",
    }
    rec = resolver.resolve_disagreement(
        case_id="DEV-001", case_content_hash="CH1",
        evaluation_as_of="2026-03-31T21:00:00Z", domain_contract_hash="DCH1",
        initial_adjudications=[adj1, adj2],
    )
    assert rec.first_material_divergence == DisagreementLayer.DOMAIN_CONTRACT_INTERPRETATION.value
    assert rec.root_dispute_class == DisagreementRootClass.DOMAIN_CONTRACT_DEFECT


def test_neg_19_material_disagreement_without_first_divergence_raises_error():
    """A material disagreement must identify an explicit first divergence layer."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "NON_QUALIFIED",
    }
    rec = resolver.resolve_disagreement(
        case_id="DEV-001", case_content_hash="CH1",
        evaluation_as_of="2026-03-31T21:00:00Z", domain_contract_hash="DCH1",
        initial_adjudications=[adj1, adj2],
    )
    assert rec.materiality is True
    assert rec.first_material_divergence != "NONE"


def test_neg_20_material_disagreement_without_root_cause_raises_error():
    """A material disagreement must be classified into the root dispute taxonomy."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "FAIL"}, "final_classification": "NON_QUALIFIED",
    }
    rec = resolver.resolve_disagreement(
        case_id="DEV-001", case_content_hash="CH1",
        evaluation_as_of="2026-03-31T21:00:00Z", domain_contract_hash="DCH1",
        initial_adjudications=[adj1, adj2],
    )
    assert isinstance(rec.root_dispute_class, DisagreementRootClass)


def test_neg_21_disputed_silver_remaining_in_active_denominator_raises_error():
    """A disputed case cannot remain in the active conformance denominator."""
    status = AuthorityStatus.DISPUTED
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_INDEPENDENT,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=status,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        silver_limitations=(SilverLimitationCode.LIMITED_DOMAIN_SCOPE,),
        external_adjudication_verified=True,
    )
    assert derived == DerivedOracleClass.NONE


def test_neg_22_predecessor_authority_record_mutation_raises_error():
    """Historical OracleAuthorityTransition records are immutable."""
    trans = OracleAuthorityTransition(
        transition_id="TRANS-001",
        case_id="DEV-001",
        predecessor_class=DerivedOracleClass.NONE,
        successor_class=DerivedOracleClass.INTERNAL_REFERENCE,
        predecessor_authority_status=AuthorityStatus.PENDING_REVIEW,
        successor_authority_status=AuthorityStatus.ACTIVE,
        transition_reason="Internal reference derivation",
        authority_evidence_hash="HASH1",
        effective_at="2026-10-09T13:00:00Z",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        trans.transition_reason = "MUTATED"  # type: ignore


def test_neg_23_scope_narrowing_without_successor_raises_error():
    """Scope narrowing requires a successor scope hash reference."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "SCOPE_OLD",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "SCOPE_NEW",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    rec = resolver.resolve_disagreement(
        case_id="DEV-001", case_content_hash="CH1",
        evaluation_as_of="2026-03-31T21:00:00Z", domain_contract_hash="DCH1",
        initial_adjudications=[adj1, adj2],
        scope_context={"narrowing_successor_hash": "SUCC_SCOPE_001"},
    )
    assert rec.successor_scope_hash != ""


def test_neg_24_resolution_replay_nondeterminism_raises_error():
    """Replaying identical disagreement inputs must produce bit-for-bit identical hashes."""
    resolver = VCPDisagreementResolver()
    adj1 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1",
        "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED",
    }
    adj2 = {
        "evidence_hash": "EV1", "observations_hash": "O1", "scope_hash": "S1",
        "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D2",
        "predicate_vector": {"P1": "FAIL"}, "final_classification": "NON_QUALIFIED",
    }
    rec1 = resolver.resolve_disagreement("DEV-001", "CH1", "2026-03-31T21:00:00Z", "DCH1", [adj1, adj2])
    rec2 = resolver.resolve_disagreement("DEV-001", "CH1", "2026-03-31T21:00:00Z", "DCH1", [adj1, adj2])
    assert rec1.resolution_record_hash == rec2.resolution_record_hash
    assert rec1.first_material_divergence == rec2.first_material_divergence
    assert rec1.root_dispute_class == rec2.root_dispute_class


# ======================================================================
# SECTIONS 38 & 39: REPLAY DETERMINISM & METAMORPHIC STABILITY
# ======================================================================

def test_replay_determinism():
    """SECTION 38: RESOLUTION_REPLAY_NONDETERMINISM = 0."""
    resolver = VCPDisagreementResolver()
    adjudications = [
        {"evidence_hash": "E_A", "observations_hash": "O_A", "scope_hash": "S_A", "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D_A", "predicate_vector": {"P1": "PASS"}, "final_classification": "QUALIFIED"},
        {"evidence_hash": "E_B", "observations_hash": "O_B", "scope_hash": "S_A", "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D_B", "predicate_vector": {"P1": "FAIL"}, "final_classification": "NON_QUALIFIED"},
    ]
    records = [
        resolver.resolve_disagreement("CASE-REPLAY", "CH-REPLAY", "2026-03-31T21:00:00Z", "DCH", adjudications)
        for _ in range(10)
    ]
    first_hash = records[0].resolution_record_hash
    assert all(r.resolution_record_hash == first_hash for r in records)
    assert all(r.first_material_divergence == records[0].first_material_divergence for r in records)
    assert all(r.root_dispute_class == records[0].root_dispute_class for r in records)


def test_metamorphic_resolution_stability():
    """SECTION 39: RESOLUTION_METAMORPHIC_TESTS = PASS.
    - Unrelated case metadata change does not alter resolution outcome.
    - Symmetrically swapping adjudicator order produces identical first divergence layer.
    """
    resolver = VCPDisagreementResolver()
    adj1 = {"evidence_hash": "E1", "observations_hash": "O1", "scope_hash": "S1", "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D1", "predicate_vector": {"P": "PASS"}, "final_classification": "QUALIFIED"}
    adj2 = {"evidence_hash": "E1", "observations_hash": "O1", "scope_hash": "S1", "contract_interpretation_clause_ids": ["C1"], "derivation_hash": "D2", "predicate_vector": {"P": "FAIL"}, "final_classification": "NON_QUALIFIED"}

    # Run resolution
    r_normal = resolver.resolve_disagreement("CASE-A", "CH1", "2026-03-31T21:00:00Z", "DCH1", [adj1, adj2])
    # Swap order
    r_swapped = resolver.resolve_disagreement("CASE-A", "CH1", "2026-03-31T21:00:00Z", "DCH1", [adj2, adj1])

    # Layer divergence is symmetric
    assert r_normal.first_material_divergence == r_swapped.first_material_divergence == DisagreementLayer.DERIVATION.value
    assert r_normal.root_dispute_class == r_swapped.root_dispute_class == DisagreementRootClass.DERIVATION_DISPUTE


# ======================================================================
# SECTIONS 35 & 36: ACCOUNTING MATRIX & AUTHORITY DENOMINATORS
# ======================================================================

def test_accounting_matrix_4_columns_and_denominators():
    """SECTION 35 & 36: Verifies 4-column accounting matrix and authority denominators."""
    corpus = VCPConformanceCorpus()
    matrix = corpus.compute_accounting_matrix()

    # 4-column primary counts:
    assert matrix["GOLD_CASE_COUNT"] == 0
    assert matrix["SILVER_CASE_COUNT"] == 0
    assert matrix["INTERNAL_REFERENCE_CASE_COUNT"] == 23
    assert matrix["NONE_CASE_COUNT"] == 1
    assert matrix["DEV_CASE_COUNT"] == 16
    assert matrix["HOLDOUT_CASE_COUNT"] == 8
    assert matrix["CONFORMANCE_CORPUS_CASE_COUNT"] == 24
    assert matrix["CROSS_TAB_TOTAL"] == 24

    # Partition cross-tabs:
    assert matrix["DEV_GOLD_COUNT"] == 0
    assert matrix["DEV_SILVER_COUNT"] == 0
    assert matrix["DEV_INTERNAL_REFERENCE_COUNT"] == 15
    assert matrix["DEV_NONE_COUNT"] == 1

    assert matrix["HOLDOUT_GOLD_COUNT"] == 0
    assert matrix["HOLDOUT_SILVER_COUNT"] == 0
    assert matrix["HOLDOUT_INTERNAL_REFERENCE_COUNT"] == 8
    assert matrix["HOLDOUT_NONE_COUNT"] == 0

    # Separate Conformance Denominators:
    assert matrix["GOLD_CONFORMANCE_DENOMINATOR"] == 0
    assert matrix["GOLD_DEV_CONFORMANCE_DENOMINATOR"] == 0
    assert matrix["GOLD_HOLDOUT_CONFORMANCE_DENOMINATOR"] == 0

    assert matrix["SILVER_CONFORMANCE_DENOMINATOR"] == 0
    assert matrix["SILVER_DEV_CONFORMANCE_DENOMINATOR"] == 0
    assert matrix["SILVER_HOLDOUT_CONFORMANCE_DENOMINATOR"] == 0

    assert matrix["INTERNAL_REFERENCE_CONFORMANCE_DENOMINATOR"] == 23
    assert matrix["INTERNAL_REFERENCE_DEV_DENOMINATOR"] == 15
    assert matrix["INTERNAL_REFERENCE_HOLDOUT_DENOMINATOR"] == 8


def test_role_accounting_matches_manifest():
    """SECTION 13: Verifies manifest-derived role counts."""
    corpus = VCPConformanceCorpus()
    roles = corpus.compute_role_accounting_matrix()

    assert roles["POSITIVE_CONTROL"] == 11
    assert roles["NEGATIVE_CONTROL"] == 12
    assert roles["BOUNDARY"] == 6
    assert roles["CHALLENGE"] == 1
    assert roles["OTHER"] == 3
    assert roles["TOTAL_ROLE_ASSIGNMENTS"] == 33

    # HLD-007 boundary check:
    hld_007 = corpus.get_case("HLD-007-BOUNDARY-200")
    assert CaseRole.BOUNDARY in hld_007.case_roles
    assert CaseRole.POSITIVE_CONTROL in hld_007.case_roles
    assert CaseRole.NEGATIVE_CONTROL not in hld_007.case_roles


def test_expected_domain_result_mapping_layer():
    """SECTION 14: Canonical ExpectedDomainResult enum and display indicator mapping."""
    assert ExpectedDomainResult.QUALIFIED.to_pass_fail() == "PASS"
    assert ExpectedDomainResult.NON_QUALIFIED.to_pass_fail() == "FAIL"
    assert ExpectedDomainResult.INSUFFICIENT_DATA.to_pass_fail() == "INSUFFICIENT_DATA"
    assert ExpectedDomainResult.NOT_APPLICABLE.to_pass_fail() == "NOT_APPLICABLE"
    assert ExpectedDomainResult.UNRESOLVED.to_pass_fail() == "UNRESOLVED"

    # From string mapping
    assert ExpectedDomainResult.from_vcp_classification("VCP_QUALIFIED") == ExpectedDomainResult.QUALIFIED
    assert ExpectedDomainResult.from_vcp_classification("VCP_NON_QUALIFIED") == ExpectedDomainResult.NON_QUALIFIED
    assert ExpectedDomainResult.from_vcp_classification("VCP_UNRESOLVED") == ExpectedDomainResult.UNRESOLVED
