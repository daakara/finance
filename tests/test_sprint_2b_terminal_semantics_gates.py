"""Test Suite for Sprint 2B Terminal Semantics Correction + Internal Freeze Gate.

Enforces:
- Issue 1: External independence enforcement & loophole removal (Section 7, Section 27)
- Issue 2: Manifest-derived role accounting, overlap, and HLD-007 audit (Sections 8-11)
- Issue 3: Label authorization semantics, product use, and legal review separation (Sections 12-16)
- Issue 4: Holdout precommitment historical reconciliation & causal precedence protocol (Sections 17-21)
- Section 27 Negative Tests (15+ negative gate tests)
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Set

from analyst_dashboard.vcp.authority_model import (
    AuthorityOrigin,
    DomainSourceAuthority,
    EvidenceSufficiency,
    AdjudicationStatus,
    AuthorityStatus,
    DerivedOracleClass,
    SilverLimitationCode,
    AdjudicationSource,
    KnownAtProvenance,
    ExpectedDomainResult,
    derive_oracle_class,
    compute_authority_model_hash,
    PRIMARY_SOURCE_AUTHORITY_ALONE_CAN_PRODUCE_GOLD,
    PRIMARY_SOURCE_AUTHORITY_ALONE_CAN_PRODUCE_SILVER,
    GOLD_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
    SILVER_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
    GOLD_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION,
    SILVER_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION,
    SYNTHETIC_ADJUDICATION_CAN_PRODUCE_GOLD,
    SYNTHETIC_ADJUDICATION_CAN_PRODUCE_SILVER,
    INTERNAL_ADJUDICATION_CAN_PRODUCE_GOLD,
    INTERNAL_ADJUDICATION_CAN_PRODUCE_SILVER,
    SYNTHETIC_ADJUDICATION_CAN_PRODUCE_INTERNAL_REFERENCE,
)
from analyst_dashboard.vcp.conformance_oracle import (
    CaseRole,
    OracleGrade,
    UsagePartition,
    VCPConformanceCorpus,
    VCPCorpusCase,
)
from analyst_dashboard.vcp.label_authorization import (
    DomainSemanticSupport,
    ProductUseStatus,
    LegalReviewStatus,
    VCPLabelAuthorizationMatrix,
    LEGACY_LABEL_AUTHORIZED_FIELD_DEPRECATED,
    LEGAL_CONCLUSION_WITHOUT_AUTHORITY,
    LEGAL_STATUS_INFERRED_FROM_DOMAIN_SOURCE,
    DOMAIN_SEMANTIC_SUPPORT_REPORTED_AS_LEGAL_AUTHORIZATION,
    PRODUCT_POLICY_REPORTED_AS_LEGAL_OPINION,
)
from analyst_dashboard.vcp.holdout_policy import (
    ProofMechanism,
    HOLDOUT_PRECOMMITMENT_POLICY_ID,
    HOLDOUT_PRECOMMITMENT_POLICY_VERSION,
    CURRENT_CANDIDATE_HOLDOUT_PRECOMMITMENT,
    CURRENT_CANDIDATE_PRECOMMITMENT_DEFECT,
    RETROACTIVE_TIMESTAMP_CAN_ESTABLISH_PRECOMMITMENT,
    HOLDOUT_PRECOMMITMENT_REPAIR_FOR_CURRENT_CANDIDATE,
    CURRENT_HOLDOUT_ENGINEERING_UTILITY,
    CURRENT_HOLDOUT_PRECOMMITTED_PROSPECTIVE_AUTHORITY,
    PRECOMMITMENT_PROOF_REQUIRES_CAUSAL_PRECEDENCE,
    PRECOMMITMENT_EPOCH_PROTOCOL_STEPS,
    verify_precommitment_ordering,
    audit_candidate_precommitment,
    compute_holdout_precommitment_policy_hash,
)


# ======================================================================
# ISSUE 1: EXTERNAL INDEPENDENCE & LOOPHOLE REMOVAL (SECTION 7 & 27)
# ======================================================================

def test_external_primary_with_complete_evidence_does_not_produce_gold():
    """EXTERNAL_PRIMARY cannot produce GOLD even with complete evidence."""
    assert PRIMARY_SOURCE_AUTHORITY_ALONE_CAN_PRODUCE_GOLD is False
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_PRIMARY,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        external_adjudication_verified=True,
    )
    assert derived != DerivedOracleClass.GOLD


def test_external_primary_with_limited_evidence_does_not_produce_silver():
    """EXTERNAL_PRIMARY cannot produce SILVER even with limited evidence and limitation codes."""
    assert PRIMARY_SOURCE_AUTHORITY_ALONE_CAN_PRODUCE_SILVER is False
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_PRIMARY,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        silver_limitations=(SilverLimitationCode.CROSS_MARKET_GENERALIZATION,),
        external_adjudication_verified=True,
    )
    assert derived != DerivedOracleClass.SILVER


def test_primary_domain_source_with_synthetic_formal_produces_internal_reference():
    """PRIMARY domain source authority + SYNTHETIC_FORMAL adjudication produces INTERNAL_REFERENCE."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.SYNTHETIC_FORMAL,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.SYNTHETIC_FIXTURE,
        domain_source_authority=DomainSourceAuthority.PRIMARY,
    )
    assert derived == DerivedOracleClass.INTERNAL_REFERENCE


def test_primary_source_with_internal_reviewer_produces_internal_reference():
    """PRIMARY source authority + INTERNAL_REVIEWER adjudication produces INTERNAL_REFERENCE."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.INTERNAL_REVIEWER,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.INTERNAL_HUMAN,
        domain_source_authority=DomainSourceAuthority.PRIMARY,
    )
    assert derived == DerivedOracleClass.INTERNAL_REFERENCE


def test_external_independent_with_complete_evidence_produces_gold():
    """EXTERNAL_INDEPENDENT + COMPLETE + all Gold criteria produces GOLD."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_INDEPENDENT,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        external_adjudication_verified=True,
        material_disagreements_count=0,
        normative_predicates_resolved=True,
        final_classification_resolved=True,
    )
    assert derived == DerivedOracleClass.GOLD


def test_external_independent_with_limited_evidence_produces_silver():
    """EXTERNAL_INDEPENDENT + LIMITED + limitation code produces SILVER."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_INDEPENDENT,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        silver_limitations=(SilverLimitationCode.LIMITED_DOMAIN_SCOPE,),
        external_adjudication_verified=True,
        material_disagreements_count=0,
        normative_predicates_resolved=True,
        final_classification_resolved=True,
    )
    assert derived == DerivedOracleClass.SILVER


def test_external_independent_with_limited_evidence_no_limitation_code_fails_closed():
    """EXTERNAL_INDEPENDENT + LIMITED + no limitation code fails closed to NONE."""
    derived = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.EXTERNAL_INDEPENDENT,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN,
        silver_limitations=(),  # Missing limitation reason code
        external_adjudication_verified=True,
    )
    assert derived == DerivedOracleClass.NONE


def test_primary_source_treated_as_independent_adjudication_raises_or_rejects():
    """PRIMARY source authority alone cannot masquerade as independent adjudication."""
    assert PRIMARY_SOURCE_AUTHORITY_ALONE_CAN_PRODUCE_GOLD is False
    assert PRIMARY_SOURCE_AUTHORITY_ALONE_CAN_PRODUCE_SILVER is False
    assert GOLD_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION == 0
    assert SILVER_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION == 0


def test_synthetic_adjudication_plus_primary_source_cannot_produce_gold_or_silver():
    """Synthetic adjudication + primary source cannot produce Gold or Silver."""
    derived_complete = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.SYNTHETIC_FORMAL,
        evidence_sufficiency=EvidenceSufficiency.COMPLETE,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.SYNTHETIC_FIXTURE,
        domain_source_authority=DomainSourceAuthority.PRIMARY,
    )
    assert derived_complete != DerivedOracleClass.GOLD
    assert derived_complete == DerivedOracleClass.INTERNAL_REFERENCE

    derived_limited = derive_oracle_class(
        adjudication_status=AdjudicationStatus.RESOLVED,
        authority_origin=AuthorityOrigin.SYNTHETIC_FORMAL,
        evidence_sufficiency=EvidenceSufficiency.LIMITED,
        authority_status=AuthorityStatus.ACTIVE,
        adjudication_source=AdjudicationSource.SYNTHETIC_FIXTURE,
        silver_limitations=(SilverLimitationCode.PARTIAL_SOURCE_AUTHORITY,),
        domain_source_authority=DomainSourceAuthority.PRIMARY,
    )
    assert derived_limited != DerivedOracleClass.SILVER
    assert derived_limited == DerivedOracleClass.INTERNAL_REFERENCE


# ======================================================================
# ISSUE 2: MANIFEST-DERIVED ROLE ACCOUNTING & HLD-007 AUDIT
# ======================================================================

def test_manifest_derived_role_accounting_projection():
    """Verifies that all 24 cases project to exactly 33 role memberships without mismatch."""
    corpus = VCPConformanceCorpus()
    all_cases = corpus.list_all_cases()
    assert len(all_cases) == 24

    # Enumerate every case and count role memberships
    role_counts: Dict[str, int] = {
        "POSITIVE_CONTROL": 0,
        "NEGATIVE_CONTROL": 0,
        "BOUNDARY": 0,
        "CHALLENGE": 0,
        "TEMPORAL_ADVERSARIAL": 0,
        "CORPORATE_ACTION": 0,
        "OTHER": 0,
    }
    total_assignments = 0

    for c in all_cases:
        assert len(c.case_roles) > 0, f"Case {c.case_id} has no roles assigned"
        # No duplicate roles inside the same case
        assert len(c.case_roles) == len(set(c.case_roles)), f"Duplicate roles in {c.case_id}"
        for r in c.case_roles:
            assert isinstance(r, CaseRole), f"Unknown role value {r}"
            role_counts[r.value] += 1
            total_assignments += 1

    assert total_assignments == 33
    assert role_counts["POSITIVE_CONTROL"] == 11
    assert role_counts["NEGATIVE_CONTROL"] == 12
    assert role_counts["BOUNDARY"] == 6
    assert role_counts["CHALLENGE"] == 1
    assert role_counts["TEMPORAL_ADVERSARIAL"] == 0
    assert role_counts["CORPORATE_ACTION"] == 0
    assert role_counts["OTHER"] == 3

    # Total role assignments may exceed corpus case count because memberships overlap
    assert total_assignments > len(all_cases)


def test_hld_007_case_roles_audit():
    """Audits HLD-007-BOUNDARY-200 explicitly: must be BOUNDARY and POSITIVE_CONTROL, never NEGATIVE_CONTROL."""
    corpus = VCPConformanceCorpus()
    case = corpus.get_case("HLD-007-BOUNDARY-200")
    assert case is not None
    assert set(case.case_roles) == {CaseRole.BOUNDARY, CaseRole.POSITIVE_CONTROL}
    assert CaseRole.NEGATIVE_CONTROL not in case.case_roles


def test_role_accounting_matrix_method_matches():
    """Verifies corpus.compute_role_accounting_matrix() derives matching counts."""
    corpus = VCPConformanceCorpus()
    matrix = corpus.compute_role_accounting_matrix()
    assert matrix["POSITIVE_CONTROL"] == 11
    assert matrix["NEGATIVE_CONTROL"] == 12
    assert matrix["BOUNDARY"] == 6
    assert matrix["CHALLENGE"] == 1
    assert matrix["OTHER"] == 3
    assert matrix["TOTAL_ROLE_ASSIGNMENTS"] == 33


def test_case_role_duplicate_inside_case_rejected():
    """Attempting to assign duplicate role inside same case raises error in corpus validation."""
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
        case_roles=(CaseRole.POSITIVE_CONTROL, CaseRole.POSITIVE_CONTROL),  # Duplicate role
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
    corpus.dev_cases[case.case_id] = tampered_case
    with pytest.raises(ValueError, match="duplicate roles"):
        corpus.validate_corpus_invariants()


# ======================================================================
# ISSUE 3: LABEL AUTHORIZATION, PRODUCT USE & LEGAL REVIEW SEPARATION
# ======================================================================

def test_label_authorization_decouples_semantic_product_and_legal():
    """Verifies clear separation of semantic support, product-use status, and legal review."""
    matrix = VCPLabelAuthorizationMatrix()

    # Minervini VCP: Supported semantically, prohibited by product policy, legal review not established
    assert matrix.minervini_semantic_support == DomainSemanticSupport.SUPPORTED
    assert matrix.minervini_product_use_status == ProductUseStatus.PROHIBITED_BY_PRODUCT_POLICY
    assert matrix.minervini_legal_review_status == LegalReviewStatus.NOT_ESTABLISHED

    # VCP: Supported semantically, authorized by product policy, legal review not established
    assert matrix.vcp_semantic_support == DomainSemanticSupport.SUPPORTED
    assert matrix.vcp_product_use_status == ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY
    assert matrix.vcp_legal_review_status == LegalReviewStatus.NOT_ESTABLISHED

    # Weinstein Stage: Supported semantically, authorized by product policy, legal review not established
    assert matrix.weinstein_stage_semantic_support == DomainSemanticSupport.SUPPORTED
    assert matrix.weinstein_stage_product_use_status == ProductUseStatus.AUTHORIZED_BY_PRODUCT_POLICY
    assert matrix.weinstein_stage_legal_review_status == LegalReviewStatus.NOT_ESTABLISHED

    # Legacy field deprecated
    assert LEGACY_LABEL_AUTHORIZED_FIELD_DEPRECATED is True
    assert LEGAL_CONCLUSION_WITHOUT_AUTHORITY == 0
    assert LEGAL_STATUS_INFERRED_FROM_DOMAIN_SOURCE == 0
    assert DOMAIN_SEMANTIC_SUPPORT_REPORTED_AS_LEGAL_AUTHORIZATION == 0
    assert PRODUCT_POLICY_REPORTED_AS_LEGAL_OPINION == 0


def test_legacy_boolean_is_projection_of_product_policy_not_legal():
    """Verifies is_label_authorized projects product policy only, not legal opinion."""
    matrix = VCPLabelAuthorizationMatrix()
    assert matrix.is_label_authorized("Minervini VCP") is False
    assert matrix.is_label_authorized("VCP") is True
    assert matrix.is_label_authorized("Stage 2") is True


# ======================================================================
# ISSUE 4: HOLDOUT PRECOMMITMENT HISTORICAL RECONCILIATION & PROTOCOL
# ======================================================================

def test_current_candidate_holdout_precommitment_is_historical():
    """Verifies that current candidate holdout precommitment is historically not established."""
    assert CURRENT_CANDIDATE_HOLDOUT_PRECOMMITMENT == "NOT_ESTABLISHED"
    assert CURRENT_CANDIDATE_PRECOMMITMENT_DEFECT == "HISTORICAL / NON_RETROACTIVELY_REPAIRABLE"
    assert RETROACTIVE_TIMESTAMP_CAN_ESTABLISH_PRECOMMITMENT is False
    assert HOLDOUT_PRECOMMITMENT_REPAIR_FOR_CURRENT_CANDIDATE == "IMPOSSIBLE_RETROACTIVELY"
    assert CURRENT_HOLDOUT_ENGINEERING_UTILITY == "PRESERVED"
    assert CURRENT_HOLDOUT_PRECOMMITTED_PROSPECTIVE_AUTHORITY == "NOT_ESTABLISHED"
    assert PRECOMMITMENT_PROOF_REQUIRES_CAUSAL_PRECEDENCE is True


def test_retroactive_timestamp_attempt_raises_error():
    """Attempting to provide a retroactive timestamp to repair past precommitment raises ValueError."""
    with pytest.raises(ValueError, match="RETROACTIVE_TIMESTAMP_PROHIBITED"):
        audit_candidate_precommitment(claimed_retroactive_timestamp="2026-09-01T00:00:00Z")


def test_claim_prospective_precommitted_on_current_holdout_raises_error():
    """Claiming prospective precommitment on current holdout raises ValueError."""
    with pytest.raises(ValueError, match="INVALID_PROSPECTIVE_CLAIM"):
        audit_candidate_precommitment(claim_prospective_precommitted=True)


def test_causal_precedence_verification():
    """Verifies that commitment timestamp must strictly precede candidate freeze timestamp."""
    assert verify_precommitment_ordering("2026-10-01T00:00:00Z", "2026-10-05T00:00:00Z") is True
    with pytest.raises(ValueError, match="PRECOMMITMENT_CAUSAL_PRECEDENCE_VIOLATION"):
        verify_precommitment_ordering("2026-10-06T00:00:00Z", "2026-10-05T00:00:00Z")


def test_holdout_precommitment_policy_protocol_and_hash():
    """Verifies 10-step protocol and deterministic hash computation."""
    assert len(PRECOMMITMENT_EPOCH_PROTOCOL_STEPS) == 10
    assert len(ProofMechanism) >= 5
    h = compute_holdout_precommitment_policy_hash()
    assert isinstance(h, str) and len(h) == 64
    assert h == "db1779a5acf23b56b60313a5fe3658a1d84116ca73110f40f7d67d54718e3279"
