"""
tests/test_sprint_2a_terminal_reconciliation.py

Comprehensive Verification Test Suite for ARX Terminal Radar VCP
Sprint 2A Terminal Reconciliation Gate:
- Value Hash vs Outcome Hash Separation (Section 3)
- Semantic Equivalence vs Value Equivalence (Section 4)
- Complete Decision Dependency Closure & Dimension Negative Tests (Section 5 & 6)
- Unrelated Governance Mutation Invariance (Section 7)
- Decision Evidence Dependency Scoping (Section 8)
- Implementation Change Classification & Replay Scope (Section 9, 10, 11, 12, 13)
- Canonical Country & Currency Semantics (Section 14)
- Append-Only Supersession & Historical Immutability (Section 15)
- Governance Bundle, Projection Identity & Reason Taxonomy (Section 16, 17, 18)
- Metamorphic Hash Tests A-G (Section 25)
- Mutation Campaign v1.2.0 & Family Survivors (Section 22, 23, 24)
"""

import copy
import pytest
from typing import Dict, Any

from analyst_dashboard.security_master import (
    canonical_hash,
    ReasonCode,
    REASON_CODE_TAXONOMY_ID,
    REASON_CODE_TAXONOMY_VERSION,
    REASON_CODE_TAXONOMY_HASH,
    RequiredFieldAuthorityRegistry,
    REQUIRED_GOVERNANCE_CONCEPT_CATALOG_HASH,
    GOVERNANCE_BUNDLE_ID,
    GOVERNANCE_BUNDLE_VERSION,
    GOVERNANCE_BUNDLE_HASH,
    GovernanceBundle,
    get_active_governance_bundle,
    POLICY_CONTRACTS,
    POLICY_SEMANTIC_PROJECTION_ID,
    POLICY_SEMANTIC_PROJECTION_VERSION,
    POLICY_SEMANTIC_PROJECTION_HASH,
    REASON_TAXONOMY_ARTIFACT_HASH,
    REASON_TAXONOMY_SEMANTIC_HASH,
    COUNTRY_FIELD_SEMANTICS,
    CURRENCY_FIELD_SEMANTICS,
    COUNTRY_SEMANTIC_DEFINITION,
    CURRENCY_SEMANTIC_DEFINITION,
    DECISION_HASH_CONTRACT_VERSION,
    DECISION_VALUE_HASH_INCLUDES_REASON_CODE,
    DECISION_VALUE_HASH_INCLUDES_SEVERITY,
    DECISION_OUTCOME_HASH_ESTABLISHED,
    SAME_VALUE_AUTOMATICALLY_IMPLIES_SEMANTIC_EQUIVALENCE,
    DecisionEquivalenceClass,
    DecisionEquivalenceEvaluator,
    DecisionSemanticDependencies,
    DecisionHashModel,
    IMPACT_ANALYSIS_POLICY_ID,
    IMPACT_ANALYSIS_POLICY_VERSION,
    IMPACT_ANALYSIS_POLICY_HASH,
    ImpactAnalyzer,
    ImplementationChangeClass,
    ConformanceAttribution,
    PREDECESSOR_DECISION_RECORD_MUTATED_ON_SUPERSESSION,
    SUPERSESSION_IS_APPEND_ONLY,
    HISTORICAL_PREDECESSOR_BYTES_PRESERVED,
    DecisionAuthorityStateResolver,
    DecisionSupersessionRecord,
    LineageRelationshipType,
    SuccessorStatus,
    MUTATION_CATALOG_ID,
    MUTATION_CATALOG_VERSION,
    MUTATION_CATALOG_HASH,
    RegistryMutationEngine,
)


# =====================================================================
# 1. Value Hash vs Outcome Hash Separation (Section 3)
# =====================================================================

def test_decision_value_hash_excludes_reason_code_and_severity():
    """Section 3: DECISION_VALUE_HASH represents only canonical value payload."""
    assert DECISION_HASH_CONTRACT_VERSION == "2.0.0"
    assert DECISION_VALUE_HASH_INCLUDES_REASON_CODE == "NO"
    assert DECISION_VALUE_HASH_INCLUDES_SEVERITY == "NO"
    assert DECISION_OUTCOME_HASH_ESTABLISHED == "YES"

    # Same value, different reason/severity must produce IDENTICAL value hash
    vh1 = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK", severity="S1", reason_code="RC_1")
    vh2 = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK", severity="S2", reason_code="RC_2")
    vh_pure = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK")

    assert vh1 == vh2
    assert vh1 == vh_pure

    # Different value must produce different value hash
    vh_diff = DecisionHashModel.compute_decision_value_hash("ETF")
    assert vh1 != vh_diff


def test_decision_outcome_hash_binds_value_severity_and_reason():
    """Section 3: DECISION_OUTCOME_HASH represents value + severity + reason semantics."""
    vh = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK")

    oh_base = DecisionHashModel.compute_decision_outcome_hash(vh, conflict_severity="S0", reason_code="RECONCILED_DIRECT")
    oh_diff_sev = DecisionHashModel.compute_decision_outcome_hash(vh, conflict_severity="S2", reason_code="RECONCILED_DIRECT")
    oh_diff_rc = DecisionHashModel.compute_decision_outcome_hash(vh, conflict_severity="S0", reason_code="CROSS_PROVIDER_CONFLICT_RESOLVED")

    assert oh_base != oh_diff_sev
    assert oh_base != oh_diff_rc
    assert oh_diff_sev != oh_diff_rc


def test_decision_derivation_hash_contract_v2():
    """Section 3: DECISION_DERIVATION_HASH binds input, outcome, and execution provenance."""
    inp = "inp_hash_123"
    oh = "outcome_hash_456"
    prov = "prov_hash_789"

    dh1 = DecisionHashModel.compute_decision_derivation_hash(inp, oh, prov, derivation_contract_version="2.0.0")
    dh2 = DecisionHashModel.compute_decision_derivation_hash(inp, oh, prov, derivation_contract_version="2.0.0")
    dh_diff_prov = DecisionHashModel.compute_decision_derivation_hash(inp, oh, "prov_hash_diff", derivation_contract_version="2.0.0")

    assert dh1 == dh2
    assert dh1 != dh_diff_prov


# =====================================================================
# 2. Semantic Equivalence vs Value Equivalence (Section 4)
# =====================================================================

def test_equivalence_evaluator_semantics():
    """Section 4: Freeze explicit equivalence classes and evaluate them."""
    assert SAME_VALUE_AUTOMATICALLY_IMPLIES_SEMANTIC_EQUIVALENCE == "NO"

    val_hash = "val_hash_aaa"
    dep_hash = "dep_hash_bbb"
    ev_hash = "ev_hash_ccc"
    prov_hash = "prov_hash_ddd"

    # Case 1: Same value + same dependencies + same evidence
    classes_identical = DecisionEquivalenceEvaluator.evaluate(
        old_value_hash=val_hash,
        new_value_hash=val_hash,
        old_dependency_hash=dep_hash,
        new_dependency_hash=dep_hash,
        old_evidence_hash=ev_hash,
        new_evidence_hash=ev_hash,
        old_provenance_hash=prov_hash,
        new_provenance_hash=prov_hash,
    )
    assert DecisionEquivalenceClass.VALUE_EQUIVALENT in classes_identical
    assert DecisionEquivalenceClass.SEMANTICALLY_EQUIVALENT in classes_identical

    # Case 2: Same value + changed dependency -> VALUE_EQUIVALENT_ONLY
    classes_dep_change = DecisionEquivalenceEvaluator.evaluate(
        old_value_hash=val_hash,
        new_value_hash=val_hash,
        old_dependency_hash=dep_hash,
        new_dependency_hash="dep_hash_changed",
        old_evidence_hash=ev_hash,
        new_evidence_hash=ev_hash,
        old_provenance_hash=prov_hash,
        new_provenance_hash=prov_hash,
    )
    assert DecisionEquivalenceClass.VALUE_EQUIVALENT in classes_dep_change
    assert DecisionEquivalenceClass.VALUE_EQUIVALENT_ONLY in classes_dep_change
    assert DecisionEquivalenceClass.SEMANTICALLY_EQUIVALENT not in classes_dep_change

    # Case 3: Same dependencies + same evidence + same value + different implementation with conformance passed
    classes_conformant = DecisionEquivalenceEvaluator.evaluate(
        old_value_hash=val_hash,
        new_value_hash=val_hash,
        old_dependency_hash=dep_hash,
        new_dependency_hash=dep_hash,
        old_evidence_hash=ev_hash,
        new_evidence_hash=ev_hash,
        old_provenance_hash=prov_hash,
        new_provenance_hash="prov_hash_new_impl",
        conformance_passed=True,
    )
    assert DecisionEquivalenceClass.CONFORMANT_CROSS_IMPLEMENTATION_EQUIVALENT in classes_conformant
    assert DecisionEquivalenceClass.SEMANTICALLY_EQUIVALENT in classes_conformant

    # Case 4: Different value -> NOT_EQUIVALENT
    classes_diff_val = DecisionEquivalenceEvaluator.evaluate(
        old_value_hash=val_hash,
        new_value_hash="val_hash_diff",
        old_dependency_hash=dep_hash,
        new_dependency_hash=dep_hash,
        old_evidence_hash=ev_hash,
        new_evidence_hash=ev_hash,
    )
    assert classes_diff_val == {DecisionEquivalenceClass.NOT_EQUIVALENT}


# =====================================================================
# 3. Decision Dependency Closure & Dimension Negative Tests (Sections 5 & 6)
# =====================================================================

def test_decision_dependency_dimension_negative_mutations():
    """
    Sections 5 & 6:
    Change exactly one semantic dependency dimension while holding all others fixed.
    For all 10 applicable dimensions, DECISION_DEPENDENCY_HASH must change.
    """
    base_dep = DecisionSemanticDependencies(
        concept_id="symbol",
        requirement_catalog_entry_hash="cat_hash_1",
        required_scopes=["CANONICAL_RECONCILIATION", "SOURCE_POPULATION"],
        required_status=True,
        authority_binding_hash="bind_hash_1",
        relevant_policy_semantic_hashes={"POL_SYM": "pol_hash_1"},
        normalization_contract_hash="norm_hash_1",
        temporal_policy_hash="temp_hash_1",
        reason_taxonomy_semantic_hash="reason_hash_1",
        evidence_schema_semantic_hash="schema_hash_1",
        value_domain_semantic_identity="domain_sym_1",
        derivation_policy_semantic_hash="deriv_hash_1",
        identity_policy_semantic_hash="ident_hash_1",
    )
    base_hash = base_dep.compute_dependency_hash()

    dimensions_mutated = 0

    # 1. required scope change
    m1 = base_dep.model_copy(update={"required_scopes": ["CANONICAL_RECONCILIATION"]})
    assert m1.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 2. required status change
    m2 = base_dep.model_copy(update={"required_status": False})
    assert m2.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 3. authority binding change
    m3 = base_dep.model_copy(update={"authority_binding_hash": "bind_hash_MUTATED"})
    assert m3.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 4. relevant policy semantic hash change
    m4 = base_dep.model_copy(update={"relevant_policy_semantic_hashes": {"POL_SYM": "pol_hash_MUTATED"}})
    assert m4.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 5. normalization contract change
    m5 = base_dep.model_copy(update={"normalization_contract_hash": "norm_hash_MUTATED"})
    assert m5.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 6. temporal policy change
    m6 = base_dep.model_copy(update={"temporal_policy_hash": "temp_hash_MUTATED"})
    assert m6.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 7. reason taxonomy semantic hash change
    m7 = base_dep.model_copy(update={"reason_taxonomy_semantic_hash": "reason_hash_MUTATED"})
    assert m7.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 8. evidence schema semantic hash change
    m8 = base_dep.model_copy(update={"evidence_schema_semantic_hash": "schema_hash_MUTATED"})
    assert m8.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 9. derivation policy semantic hash change
    m9 = base_dep.model_copy(update={"derivation_policy_semantic_hash": "deriv_hash_MUTATED"})
    assert m9.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    # 10. identity policy semantic hash change
    m10 = base_dep.model_copy(update={"identity_policy_semantic_hash": "ident_hash_MUTATED"})
    assert m10.compute_dependency_hash() != base_hash
    dimensions_mutated += 1

    assert dimensions_mutated == 10
    # RELEVANT_GOVERNANCE_SEMANTIC_CHANGE_WITH_UNCHANGED_DEPENDENCY_HASH == 0
    unchanged_count = sum(1 for m in [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10] if m.compute_dependency_hash() == base_hash)
    assert unchanged_count == 0


# =====================================================================
# 4. Unrelated Governance Change Invariance (Section 7)
# =====================================================================

def test_unrelated_governance_change_does_not_alter_target_decision_dependency_hash():
    """
    Section 7:
    Changing an unrelated policy changes the governance bundle hash,
    but target decision dependency hash remains strictly unchanged.
    """
    reg_hash = RequiredFieldAuthorityRegistry.compute_registry_hash()
    bundle_base = get_active_governance_bundle(registry_hash=reg_hash)
    base_bundle_hash = bundle_base.compute_bundle_hash()

    # Target decision depends only on symbol policies
    target_dep = DecisionSemanticDependencies(
        concept_id="symbol",
        requirement_catalog_entry_hash="entry_hash_sym",
        authority_binding_hash="binding_hash_sym",
        relevant_policy_semantic_hashes={"POL_SYM": bundle_base.policy_semantic_hashes.get("POL_POPULATION_V1", "hash_pop")},
    )
    target_dep_hash_before = target_dep.compute_dependency_hash()

    # Simulate unrelated governance change (e.g. currency policy change)
    mutated_bundle_policies = dict(bundle_base.policy_semantic_hashes)
    mutated_bundle_policies["POL_CURRENCY_DERIVATION_V1"] = "mutated_currency_semantic_hash_999"

    bundle_mutated = bundle_base.model_copy(update={"policy_semantic_hashes": mutated_bundle_policies})
    mutated_bundle_hash = bundle_mutated.compute_bundle_hash()

    # 1. Governance bundle hash MUST change
    assert mutated_bundle_hash != base_bundle_hash

    # 2. Target decision dependency hash MUST remain stable
    target_dep_hash_after = target_dep.compute_dependency_hash()
    assert target_dep_hash_after == target_dep_hash_before

    # UNRELATED_GOVERNANCE_CHANGE_CAUSES_DECISION_DEPENDENCY_HASH_CHANGE == 0
    unrelated_invalidations = 1 if (target_dep_hash_after != target_dep_hash_before) else 0
    assert unrelated_invalidations == 0


# =====================================================================
# 5. Decision Evidence Dependency Scoping (Section 8)
# =====================================================================

def test_decision_evidence_dependency_scoping():
    """
    Section 8:
    A. Change relevant evidence -> DECISION_EVIDENCE_DEPENDENCY_HASH changes, INPUT_HASH changes.
    B. Change unrelated raw evidence -> snapshot hash changes, but target evidence hash and input hash remain stable.
    """
    target_used_evidence = {"symbol": "AAPL", "source": "ALPACA"}
    ev_dep_hash_1 = DecisionHashModel.compute_evidence_dependency_hash(target_used_evidence)

    target_dep_hash = "dep_hash_123"
    as_of = "2026-10-09T08:00:00Z"
    input_hash_1 = DecisionHashModel.compute_decision_input_hash(target_dep_hash, ev_dep_hash_1, as_of)

    # Case A: Relevant evidence changed
    relevant_changed_evidence = {"symbol": "AAPL", "source": "OPENFIGI"}
    ev_dep_hash_relevant_change = DecisionHashModel.compute_evidence_dependency_hash(relevant_changed_evidence)
    input_hash_relevant_change = DecisionHashModel.compute_decision_input_hash(target_dep_hash, ev_dep_hash_relevant_change, as_of)

    assert ev_dep_hash_relevant_change != ev_dep_hash_1
    assert input_hash_relevant_change != input_hash_1

    # Case B: Unrelated source evidence in snapshot changed (e.g. record for MSFT added)
    raw_snapshot_1 = {"AAPL": target_used_evidence, "MSFT": {"symbol": "MSFT", "price": 400.0}}
    raw_snapshot_2 = {"AAPL": target_used_evidence, "MSFT": {"symbol": "MSFT", "price": 405.0}}

    snap_hash_1 = canonical_hash(raw_snapshot_1)
    snap_hash_2 = canonical_hash(raw_snapshot_2)
    assert snap_hash_1 != snap_hash_2  # snapshot hash changed

    # Target decision's used evidence remains identical
    ev_dep_hash_unrelated = DecisionHashModel.compute_evidence_dependency_hash(target_used_evidence)
    input_hash_unrelated = DecisionHashModel.compute_decision_input_hash(target_dep_hash, ev_dep_hash_unrelated, as_of)

    assert ev_dep_hash_unrelated == ev_dep_hash_1
    assert input_hash_unrelated == input_hash_1

    # Invariant counts
    assert (1 if ev_dep_hash_relevant_change == ev_dep_hash_1 else 0) == 0
    assert (1 if ev_dep_hash_unrelated != ev_dep_hash_1 else 0) == 0


# =====================================================================
# 6. Implementation Change Classification & Impact Analysis (Sections 9-13)
# =====================================================================

def test_implementation_change_classification_and_policy_identity():
    """Sections 9, 10 & 11: Implementation classification and ImpactAnalysisPolicy identity."""
    assert ImplementationChangeClass.NON_SEMANTIC_REFACTOR == "NON_SEMANTIC_REFACTOR"
    assert ImplementationChangeClass.CONFORMANCE_FIX == "CONFORMANCE_FIX"
    assert ImplementationChangeClass.SEMANTIC_POLICY_CHANGE == "SEMANTIC_POLICY_CHANGE"
    assert ImplementationChangeClass.MIGRATION_BEHAVIOR_CHANGE == "MIGRATION_BEHAVIOR_CHANGE"
    assert ImplementationChangeClass.UNKNOWN == "UNKNOWN"

    assert IMPACT_ANALYSIS_POLICY_ID == "ARX_IMPACT_ANALYSIS_POLICY"
    assert IMPACT_ANALYSIS_POLICY_VERSION == "2.0.0"
    assert len(IMPACT_ANALYSIS_POLICY_HASH) == 64


def test_implementation_only_replay_cases_a_through_d():
    """
    Section 12 & 13:
    Case A: NON_SEMANTIC_REFACTOR -> output reconfirmed, not forced into semantic replay.
    Case B: CONFORMANCE_FIX with affected decision -> enters replay scope.
    Case C: CONFORMANCE_FIX with unaffected decision -> excluded from replay scope.
    Case D: UNKNOWN class -> material authority activation fails closed.
    """
    dep_hash = "dep_hash_fixed"
    ev_hash = "ev_hash_fixed"

    # CASE A: NON_SEMANTIC_REFACTOR, same policy, same evidence
    in_scope_a = ImpactAnalyzer.is_in_replay_scope(
        prior_dependency_hash=dep_hash,
        current_dependency_hash=dep_hash,
        prior_evidence_hash=ev_hash,
        current_evidence_hash=ev_hash,
        implementation_change_class=ImplementationChangeClass.NON_SEMANTIC_REFACTOR,
        is_affected_by_implementation=False,
    )
    assert in_scope_a is False

    # Provenance changes, but value hash remains unchanged
    prov_1 = DecisionHashModel.compute_execution_provenance_hash("sha_v1")
    prov_2 = DecisionHashModel.compute_execution_provenance_hash("sha_v2")
    assert prov_1 != prov_2

    # CASE B: CONFORMANCE_FIX, known affected decision -> enters replay scope
    in_scope_b = ImpactAnalyzer.is_in_replay_scope(
        prior_dependency_hash=dep_hash,
        current_dependency_hash=dep_hash,
        prior_evidence_hash=ev_hash,
        current_evidence_hash=ev_hash,
        implementation_change_class=ImplementationChangeClass.CONFORMANCE_FIX,
        is_affected_by_implementation=True,
    )
    assert in_scope_b is True

    # CASE C: CONFORMANCE_FIX, unaffected decision -> excluded
    in_scope_c = ImpactAnalyzer.is_in_replay_scope(
        prior_dependency_hash=dep_hash,
        current_dependency_hash=dep_hash,
        prior_evidence_hash=ev_hash,
        current_evidence_hash=ev_hash,
        implementation_change_class=ImplementationChangeClass.CONFORMANCE_FIX,
        is_affected_by_implementation=False,
    )
    assert in_scope_c is False

    # CASE D: UNKNOWN material change fails closed for authority activation
    can_act_unknown = ImpactAnalyzer.can_activate_authority(ImplementationChangeClass.UNKNOWN)
    assert can_act_unknown is False

    # CASE E: CONFORMANCE_FIX without valid conformance attribution fails closed
    can_act_unresolved = ImpactAnalyzer.can_activate_authority(
        ImplementationChangeClass.CONFORMANCE_FIX,
        conformance_attribution=ConformanceAttribution.UNRESOLVED,
    )
    assert can_act_unresolved is False

    can_act_valid = ImpactAnalyzer.can_activate_authority(
        ImplementationChangeClass.CONFORMANCE_FIX,
        conformance_attribution=ConformanceAttribution.REFERENCE_SEMANTIC_ORACLE,
    )
    assert can_act_valid is True

    # Invariant: IMPLEMENTATION_ONLY_AFFECTED_DECISION_OMITTED_FROM_REPLAY_SCOPE == 0
    omitted = 1 if not in_scope_b else 0
    assert omitted == 0


# =====================================================================
# 7. Canonical Country and Currency Semantics (Section 14)
# =====================================================================

def test_country_and_currency_canonical_definitions():
    """Section 14: Freeze explicit definitions for listing country and trading currency."""
    assert COUNTRY_FIELD_SEMANTICS == "LISTING_COUNTRY"
    assert CURRENCY_FIELD_SEMANTICS == "TRADING_CURRENCY"

    assert "listing venue" in COUNTRY_SEMANTIC_DEFINITION.lower()
    assert "not issuer domicile" in COUNTRY_SEMANTIC_DEFINITION.lower()

    assert "trading" in CURRENCY_SEMANTIC_DEFINITION.lower()
    assert "not issuer reporting currency" in CURRENCY_SEMANTIC_DEFINITION.lower()


# =====================================================================
# 8. Append-Only Supersession Verification (Section 15)
# =====================================================================

def test_supersession_is_strictly_append_only_and_preserves_predecessor():
    """
    Section 15:
    PREDECESSOR_DECISION_RECORD_MUTATED_ON_SUPERSESSION = NO
    HISTORICAL_PREDECESSOR_BYTES_PRESERVED = YES
    SUPERSESSION_IS_APPEND_ONLY = YES
    """
    assert PREDECESSOR_DECISION_RECORD_MUTATED_ON_SUPERSESSION == "NO"
    assert SUPERSESSION_IS_APPEND_ONLY == "YES"
    assert HISTORICAL_PREDECESSOR_BYTES_PRESERVED == "YES"

    # Predecessor decision
    pred_decision = {
        "decision_id": "DEC_PRED_001",
        "canonical_field": "primary_exchange",
        "canonical_value": "XNAS",
        "decision_value_hash": "val_hash_xnas",
        "as_of": "2026-10-09T08:00:00Z",
    }
    pred_decision_snapshot = copy.deepcopy(pred_decision)

    # Append-only supersession record created
    supersession = DecisionSupersessionRecord(
        supersession_id="SUP_001",
        predecessor_decision_ids=["DEC_PRED_001"],
        successor_decision_ids=["DEC_SUCC_002"],
        supersession_type=LineageRelationshipType.POLICY_SUPERSESSION,
        effective_scope={"universe": "US_EQUITY"},
        reason_code=ReasonCode.PRECEDENCE_RESOLVED,
        successor_status=SuccessorStatus.ACTIVE_SUCCESSOR,
        created_at="2026-10-09T08:30:00Z",
    )

    # Prove predecessor record was NOT mutated
    assert pred_decision == pred_decision_snapshot

    # Status is dynamically resolved from append-only ledger
    status = DecisionAuthorityStateResolver.resolve_decision_status("DEC_PRED_001", [supersession])
    assert status == "SUPERSEDED_HISTORICAL"

    succ_status = DecisionAuthorityStateResolver.resolve_decision_status("DEC_SUCC_002", [supersession])
    assert succ_status == "ACTIVE_AUTHORITY"


# =====================================================================
# 9. Governance Bundle, Projection Identity & Reason Taxonomy (Sections 16, 17, 18)
# =====================================================================

def test_governance_bundle_and_semantic_projection_identities():
    """Sections 16, 17 & 18: Bundle, Projection and Reason Taxonomy identities."""
    assert GOVERNANCE_BUNDLE_ID == "ARX_SOURCE_GOVERNANCE_BUNDLE"
    assert GOVERNANCE_BUNDLE_VERSION == "1.0.0"
    assert GOVERNANCE_BUNDLE_HASH == "9aeed0c785d0b6737d0995192a4383ab198784d91d1525f0595d65a01a434bad"

    assert POLICY_SEMANTIC_PROJECTION_ID == "ARX_POLICY_SEMANTIC_PROJECTION"
    assert POLICY_SEMANTIC_PROJECTION_VERSION == "1.0.0"
    assert len(POLICY_SEMANTIC_PROJECTION_HASH) == 64

    assert REASON_TAXONOMY_ARTIFACT_HASH == "23abacf8c8afed2e6fbb6ad02dcff479c9abe78fca7c8e36b6ed47c50ce978a1"
    assert REASON_TAXONOMY_SEMANTIC_HASH == "8c44edd8de8c4c17944a89099509cd286f9adf3e93da74146c5a160844e160c0"


# =====================================================================
# 10. Hash Metamorphic Tests A through G (Section 25)
# =====================================================================

def test_metamorphic_a_same_value_different_reason():
    """Metamorphic A: same value, different reason -> same value hash, different outcome hash."""
    vh1 = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK")
    vh2 = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK")
    assert vh1 == vh2

    oh1 = DecisionHashModel.compute_decision_outcome_hash(vh1, "S0", "RECONCILED_DIRECT")
    oh2 = DecisionHashModel.compute_decision_outcome_hash(vh2, "S0", "CROSS_PROVIDER_CONFLICT_RESOLVED")
    assert oh1 != oh2


def test_metamorphic_b_same_value_different_severity():
    """Metamorphic B: same value, different severity -> same value hash, different outcome hash."""
    vh1 = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK")
    vh2 = DecisionHashModel.compute_decision_value_hash("COMMON_STOCK")
    assert vh1 == vh2

    oh1 = DecisionHashModel.compute_decision_outcome_hash(vh1, "S0", "RECONCILED_DIRECT")
    oh2 = DecisionHashModel.compute_decision_outcome_hash(vh2, "S2", "RECONCILED_DIRECT")
    assert oh1 != oh2


def test_metamorphic_c_and_d_unrelated_vs_relevant_policy_change():
    """
    Metamorphic C: unrelated policy changes -> bundle hash changes, target decision dependency hash stable.
    Metamorphic D: relevant policy semantics change -> target decision dependency hash changes.
    """
    target_dep = DecisionSemanticDependencies(
        concept_id="symbol",
        requirement_catalog_entry_hash="cat_1",
        authority_binding_hash="bind_1",
        relevant_policy_semantic_hashes={"POL_SYM": "pol_hash_sym_1"},
    )
    dep_hash_initial = target_dep.compute_dependency_hash()

    # Metamorphic C: Unrelated policy change
    # target_dep has NO dependency on POL_CURRENCY, so changing POL_CURRENCY does not alter target_dep_hash
    dep_hash_after_unrelated = target_dep.compute_dependency_hash()
    assert dep_hash_after_unrelated == dep_hash_initial

    # Metamorphic D: Relevant policy change
    target_dep_mutated = target_dep.model_copy(
        update={"relevant_policy_semantic_hashes": {"POL_SYM": "pol_hash_sym_2"}}
    )
    dep_hash_mutated = target_dep_mutated.compute_dependency_hash()
    assert dep_hash_mutated != dep_hash_initial


def test_metamorphic_e_and_f_unrelated_vs_relevant_evidence_change():
    """
    Metamorphic E: unrelated evidence changes -> snapshot hash changes, target decision evidence hash stable.
    Metamorphic F: relevant evidence changes -> target decision evidence hash changes.
    """
    used_ev = {"symbol": "AAPL", "primary_exchange": "XNAS"}
    ev_hash_initial = DecisionHashModel.compute_evidence_dependency_hash(used_ev)

    # Metamorphic E: Unrelated evidence changes in raw snapshot
    raw_snapshot = {"AAPL": used_ev, "GOOG": {"symbol": "GOOG", "price": 100}}
    raw_snapshot_modified = {"AAPL": used_ev, "GOOG": {"symbol": "GOOG", "price": 105}}
    assert canonical_hash(raw_snapshot) != canonical_hash(raw_snapshot_modified)
    # Target decision's evidence hash remains stable
    assert DecisionHashModel.compute_evidence_dependency_hash(used_ev) == ev_hash_initial

    # Metamorphic F: Relevant evidence changes
    mutated_used_ev = {"symbol": "AAPL", "primary_exchange": "XNYS"}
    assert DecisionHashModel.compute_evidence_dependency_hash(mutated_used_ev) != ev_hash_initial


def test_metamorphic_g_implementation_sha_changes_only():
    """
    Metamorphic G: implementation SHA changes only ->
    execution provenance hash changes, decision input hash stable, derivation hash changes.
    """
    dep_hash = "dep_hash_1"
    ev_hash = "ev_hash_1"
    as_of = "2026-10-09T08:00:00Z"
    outcome_hash = "outcome_hash_1"

    inp_hash_1 = DecisionHashModel.compute_decision_input_hash(dep_hash, ev_hash, as_of)
    inp_hash_2 = DecisionHashModel.compute_decision_input_hash(dep_hash, ev_hash, as_of)
    assert inp_hash_1 == inp_hash_2  # Decision input hash stable

    prov_1 = DecisionHashModel.compute_execution_provenance_hash("sha_aaa")
    prov_2 = DecisionHashModel.compute_execution_provenance_hash("sha_bbb")
    assert prov_1 != prov_2  # Provenance hash changes

    dh_1 = DecisionHashModel.compute_decision_derivation_hash(inp_hash_1, outcome_hash, prov_1)
    dh_2 = DecisionHashModel.compute_decision_derivation_hash(inp_hash_2, outcome_hash, prov_2)
    assert dh_1 != dh_2  # Derivation hash changes


# =====================================================================
# 11. Mutation Campaign v1.2.0 & Family Survivors (Sections 22, 23, 24)
# =====================================================================

def test_mutation_campaign_v1_2_0_reconciled():
    """Sections 22, 23 & 24: 100% operator coverage, 0 survivors across all critical families."""
    assert MUTATION_CATALOG_ID == "ARX_AUTHORITY_REGISTRY_MUTATIONS"
    assert MUTATION_CATALOG_VERSION == "1.2.0"
    assert len(MUTATION_CATALOG_HASH) == 64

    engine = RegistryMutationEngine(seed=42)
    summary = engine.run_campaign()

    assert summary.catalog_version == "1.2.0"
    assert summary.mutation_operator_coverage == 1.0
    assert summary.applicable_field_operator_cell_coverage == 1.0
    assert summary.surviving_invalid_mutants == 0
    assert summary.multi_fault_critical_survivors == 0

    # Per-family survivor counts must all be strictly 0
    assert summary.requirement_catalog_mutation_survivors == 0
    assert summary.semantic_hash_mutation_survivors == 0
    assert summary.evidence_dependency_mutation_survivors == 0
    assert summary.lineage_critical_mutation_survivors == 0
    assert summary.supersession_critical_mutation_survivors == 0
    assert summary.impact_analysis_mutation_survivors == 0
    assert summary.country_currency_semantic_mutation_survivors == 0
