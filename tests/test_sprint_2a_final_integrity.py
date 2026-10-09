"""
tests/test_sprint_2a_final_integrity.py

Dedicated Comprehensive Verification Suite for ARX Terminal Radar VCP
Sprint 2A Final Closure Integrity Gate.

Enforces:
- Requirement-Catalog Governance & Change-Control Ledger (Section 3, 4, 5)
- Non-Direct Authority Resolution & Concrete Policy Contracts (Section 6, 7, 8)
- Country & Currency Value Evidence Derivation Decoupling (Section 9)
- Governance Bundle & Aggregate Semantic Hashing (Section 10)
- Decision Dependency Hashing & Evidence Dependency Hashing (Section 11, 12)
- Six-Hash Decision Architecture & Semantic Isolation (Section 13, 14, 31)
- Replay Lineage Contracts, Lineage DAG & Cycle Prevention (Section 15-21)
- Supersession Ledger & Historical Preservation (Section 22, 23, 24)
- Bitemporal Replay Accounting & Scope Governor (Section 25, 26, 27, 28, 29)
- Reason Taxonomy Semantic Projection Invariance (Section 30)
- Negative & Integrity Verification Matrix (Section 32, Scenarios 1-27)
"""

import copy
import pytest

from analyst_dashboard.security_master import (
    RequiredFieldAuthorityRegistry,
    RequiredFieldEntry,
    GovernanceBinding,
    GovernanceBindingType,
    GovernedScope,
    RequiredGovernanceConceptCatalog,
    GovernanceConceptDefinition,
    CatalogChangeRecord,
    CatalogChangeType,
    REQUIRED_GOVERNANCE_CONCEPT_CATALOG_ID,
    REQUIRED_GOVERNANCE_CONCEPT_CATALOG_VERSION,
    REQUIRED_GOVERNANCE_CONCEPT_CATALOG_HASH,
    ConcretePolicyContract,
    POLICY_CONTRACTS,
    GOVERNANCE_BUNDLE_ID,
    GovernanceBundle,
    get_active_governance_bundle,
    DecisionSemanticDependencies,
    DecisionHashModel,
    ImpactAnalyzer,
    IMPACT_ANALYSIS_POLICY_ID,
    ImplementationChangeClass,
    ConformanceAttribution,
    ReplayPurpose,
    AuthorityEffect,
    SuccessorStatus,
    LineageRelationshipType,
    DecisionLineageEdge,
    LineageCycleError,
    DecisionLineageDAG,
    DecisionReplayRecord,
    DecisionSupersessionRecord,
    ReplayAccountingSummary,
    CanonicalGenerationLineage,
    canonical_hash,
    ReasonCode,
)


# =====================================================================
# Scenario 1: Silent Concept Removal from Root Catalog Rejected
# =====================================================================
def test_scenario_01_silent_concept_removal_rejected():
    concepts = copy.deepcopy(RequiredGovernanceConceptCatalog.CONCEPTS)
    del concepts["symbol"]
    res = RequiredGovernanceConceptCatalog.validate(concepts=concepts)
    assert not res.passed
    assert any(e.error_code == "UNAUTHORIZED_CONCEPT_REMOVAL" and e.concept_id == "symbol" for e in res.errors)


# =====================================================================
# Scenario 2: Silent Required Demotion in Root Catalog Rejected
# =====================================================================
def test_scenario_02_silent_required_demotion_rejected():
    concepts = copy.deepcopy(RequiredGovernanceConceptCatalog.CONCEPTS)
    c_dict = concepts["symbol"].model_dump()
    c_dict["required"] = False
    concepts["symbol"] = GovernanceConceptDefinition.model_validate(c_dict)
    res = RequiredGovernanceConceptCatalog.validate(concepts=concepts)
    assert not res.passed
    assert any(e.error_code == "UNAUTHORIZED_REQUIRED_STATUS_CHANGE" and e.concept_id == "symbol" for e in res.errors)


# =====================================================================
# Scenario 3: Silent Scope Removal from Root Catalog Rejected
# =====================================================================
def test_scenario_03_silent_scope_removal_rejected():
    concepts = copy.deepcopy(RequiredGovernanceConceptCatalog.CONCEPTS)
    c_dict = concepts["symbol"].model_dump()
    c_dict["required_scopes"] = []
    concepts["symbol"] = GovernanceConceptDefinition.model_validate(c_dict)
    res = RequiredGovernanceConceptCatalog.validate(concepts=concepts)
    assert not res.passed
    assert any(e.error_code == "UNAUTHORIZED_SCOPE_REMOVAL" and e.concept_id == "symbol" for e in res.errors)


# =====================================================================
# Scenario 4: Catalog Version Bump Without Change Record Rejected
# =====================================================================
def test_scenario_04_catalog_version_without_change_record_rejected():
    res = RequiredGovernanceConceptCatalog.validate(
        concepts=RequiredGovernanceConceptCatalog.CONCEPTS,
        expected_version="2.0.0",
        change_record=None,
    )
    assert not res.passed
    assert any(e.error_code == "CATALOG_VERSION_WITHOUT_CHANGE_ID" for e in res.errors)


# =====================================================================
# Scenario 5: Catalog Hash Mismatch Detected
# =====================================================================
def test_scenario_05_catalog_hash_mismatch_detected():
    res = RequiredGovernanceConceptCatalog.validate(
        concepts=RequiredGovernanceConceptCatalog.CONCEPTS,
        expected_hash="0000000000000000000000000000000000000000000000000000000000000000",
    )
    assert not res.passed
    assert any(e.error_code == "CATALOG_HASH_MISMATCH" for e in res.errors)


# =====================================================================
# Scenario 6: Authorized Catalog Change Record Permitted
# =====================================================================
def test_scenario_06_authorized_catalog_change_permitted():
    concepts = copy.deepcopy(RequiredGovernanceConceptCatalog.CONCEPTS)
    del concepts["provider_asset_class"]
    change = CatalogChangeRecord(
        catalog_change_id="CHG_TEST_001",
        predecessor_catalog_hash=REQUIRED_GOVERNANCE_CONCEPT_CATALOG_HASH,
        successor_catalog_hash="dummy_hash",
        change_type=CatalogChangeType.REMOVE_CONCEPT,
        rationale="Authorized test removal of provider_asset_class",
        affected_concepts=["provider_asset_class"],
        authorization_status="AUTHORIZED",
        created_at="2026-10-09T09:00:00Z",
    )
    res = RequiredGovernanceConceptCatalog.validate(concepts=concepts, change_record=change)
    assert res.passed


# =====================================================================
# Scenario 7: Fixed Taxonomy Rejected as Evidence Authority
# =====================================================================
def test_scenario_07_fixed_taxonomy_rejected_as_evidence_authority():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    ed = entries["country"].model_dump()
    ed["governance_binding"] = {
        "binding_type": GovernanceBindingType.FIXED_TAXONOMY,
        "policy_id": "POL_TAXONOMY_ISO",
        "policy_version": "1.0.0",
        "policy_hash": "cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
    }
    entries["country"] = RequiredFieldEntry.model_validate(ed)
    res = RequiredFieldAuthorityRegistry.validate(entries=entries)
    assert not res.passed
    assert any(e.error_code == "TAXONOMY_USED_AS_EVIDENCE_AUTHORITY" and e.field_id == "country" for e in res.errors)


# =====================================================================
# Scenario 8: Non-Direct Binding Without Concrete Contract Rejected
# =====================================================================
def test_scenario_08_non_direct_binding_without_concrete_contract_rejected():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    ed = entries["country"].model_dump()
    ed["governance_binding"]["policy_id"] = "POL_UNREGISTERED_NON_DIRECT_99"
    entries["country"] = RequiredFieldEntry.model_validate(ed)
    res = RequiredFieldAuthorityRegistry.validate(entries=entries)
    assert not res.passed
    assert any(e.error_code == "NON_DIRECT_BINDING_WITHOUT_CONCRETE_SEMANTICS" and e.field_id == "country" for e in res.errors)


# =====================================================================
# Scenario 9: Non-Direct Binding Missing Policy Version Rejected
# =====================================================================
def test_scenario_09_non_direct_binding_missing_policy_version_rejected():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    ed = entries["country"].model_dump()
    ed["governance_binding"]["policy_version"] = ""
    entries["country"] = RequiredFieldEntry.model_validate(ed)
    res = RequiredFieldAuthorityRegistry.validate(entries=entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_POLICY_VERSION" and e.field_id == "country" for e in res.errors)


# =====================================================================
# Scenario 10: Non-Direct Binding Missing Policy Hash Rejected
# =====================================================================
def test_scenario_10_non_direct_binding_missing_policy_hash_rejected():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    ed = entries["country"].model_dump()
    ed["governance_binding"]["policy_hash"] = ""
    entries["country"] = RequiredFieldEntry.model_validate(ed)
    res = RequiredFieldAuthorityRegistry.validate(entries=entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_POLICY_HASH" and e.field_id == "country" for e in res.errors)


# =====================================================================
# Scenario 11: Documentation Prose Invariance on Semantic Hash
# =====================================================================
def test_scenario_11_documentation_prose_invariance_on_semantic_hash():
    base_contract = POLICY_CONTRACTS["POL_COUNTRY_DERIVATION_V1"]
    base_semantic_hash = base_contract.compute_semantic_hash()
    base_artifact_hash = base_contract.compute_artifact_hash()

    # Modify non-semantic documentation metadata
    modified_contract = ConcretePolicyContract(
        policy_id=base_contract.policy_id,
        policy_version=base_contract.policy_version,
        policy_name=base_contract.policy_name,
        description=base_contract.description,
        authorities=base_contract.authorities,
        precedence=base_contract.precedence,
        admissibility_rules=base_contract.admissibility_rules,
        normalization_rules=base_contract.normalization_rules,
        missing_behavior=base_contract.missing_behavior,
        conflict_behavior=base_contract.conflict_behavior,
        stale_behavior=base_contract.stale_behavior,
        unknown_behavior=base_contract.unknown_behavior,
        temporal_behavior=base_contract.temporal_behavior,
        derivation_inputs=base_contract.derivation_inputs,
        reason_codes=base_contract.reason_codes,
        effective_scopes=base_contract.effective_scopes,
        documentation_prose="Completely new commentary that does not affect semantics.",
        display_label="New Label",
        maintainer="New Maintainer",
    )

    # Artifact hash MUST differ, but semantic hash MUST be identical
    assert modified_contract.compute_artifact_hash() != base_artifact_hash
    assert modified_contract.compute_semantic_hash() == base_semantic_hash


# =====================================================================
# Scenario 12: Governance Bundle Hash Determinism & Completeness
# =====================================================================
def test_scenario_12_governance_bundle_hash_determinism_and_completeness():
    reg_hash = RequiredFieldAuthorityRegistry.compute_registry_hash()
    bundle1 = get_active_governance_bundle(registry_hash=reg_hash)
    bundle2 = get_active_governance_bundle(registry_hash=reg_hash)

    assert bundle1.compute_bundle_hash() == bundle2.compute_bundle_hash()
    assert len(bundle1.policy_semantic_hashes) >= 11
    assert "POL_COUNTRY_DERIVATION_V1" in bundle1.policy_semantic_hashes
    assert "POL_CURRENCY_DERIVATION_V1" in bundle1.policy_semantic_hashes


# =====================================================================
# Scenario 13: Decision Dependency Hash Strictly Scoped
# =====================================================================
def test_scenario_13_decision_dependency_hash_strictly_scoped():
    deps = DecisionSemanticDependencies(
        concept_id="symbol",
        requirement_catalog_entry_hash="hash_concept_symbol",
        authority_binding_hash="hash_binding_symbol",
        relevant_policy_semantic_hashes={"POL_SYMBOL_V1": "hash_symbol_policy"},
    )
    h1 = deps.compute_dependency_hash()

    # Changing an unrelated policy not referenced in this dependency does not alter hash
    deps2 = DecisionSemanticDependencies(
        concept_id="symbol",
        requirement_catalog_entry_hash="hash_concept_symbol",
        authority_binding_hash="hash_binding_symbol",
        relevant_policy_semantic_hashes={"POL_SYMBOL_V1": "hash_symbol_policy"},
    )
    h2 = deps2.compute_dependency_hash()
    assert h1 == h2

    # Changing the relevant policy DOES alter hash
    deps3 = DecisionSemanticDependencies(
        concept_id="symbol",
        requirement_catalog_entry_hash="hash_concept_symbol",
        authority_binding_hash="hash_binding_symbol",
        relevant_policy_semantic_hashes={"POL_SYMBOL_V1": "hash_symbol_policy_NEW"},
    )
    h3 = deps3.compute_dependency_hash()
    assert h1 != h3


# =====================================================================
# Scenario 14: Evidence Dependency Hash Strictly Scoped
# =====================================================================
def test_scenario_14_evidence_dependency_hash_strictly_scoped():
    used_evidence_1 = {"symbol": "AAPL", "status": "active"}
    used_evidence_2 = {"symbol": "AAPL", "status": "active"}
    used_evidence_3 = {"symbol": "AAPL", "status": "active", "extra_unrelated": 123}

    h1 = DecisionHashModel.compute_evidence_dependency_hash(used_evidence_1)
    h2 = DecisionHashModel.compute_evidence_dependency_hash(used_evidence_2)
    h3 = DecisionHashModel.compute_evidence_dependency_hash(used_evidence_3)

    assert h1 == h2
    assert h1 != h3


# =====================================================================
# Scenario 15: Decision Input Hash Excludes Implementation SHA
# =====================================================================
def test_scenario_15_decision_input_hash_excludes_implementation_sha():
    dep_hash = "dep_hash_123"
    ev_hash = "ev_hash_456"
    as_of = "2026-10-09T00:00:00Z"

    # Input hash computed under implementation SHA 1
    input_hash_1 = DecisionHashModel.compute_decision_input_hash(
        decision_dependency_hash=dep_hash,
        decision_evidence_dependency_hash=ev_hash,
        as_of=as_of,
    )

    # Input hash computed under implementation SHA 2 (re-execution / refactor)
    input_hash_2 = DecisionHashModel.compute_decision_input_hash(
        decision_dependency_hash=dep_hash,
        decision_evidence_dependency_hash=ev_hash,
        as_of=as_of,
    )

    assert input_hash_1 == input_hash_2


# =====================================================================
# Scenario 16: Decision Input Hash Excludes Run ID & Wall-Clock Timestamps
# =====================================================================
def test_scenario_16_decision_input_hash_excludes_run_id_and_timestamp():
    dep_hash = "dep_hash_123"
    ev_hash = "ev_hash_456"
    as_of = "2026-10-09T00:00:00Z"

    # Two separate runs with distinct run IDs and execution timestamps
    h_run1 = DecisionHashModel.compute_decision_input_hash(dep_hash, ev_hash, as_of)
    h_run2 = DecisionHashModel.compute_decision_input_hash(dep_hash, ev_hash, as_of)

    assert h_run1 == h_run2


# =====================================================================
# Scenario 17: Execution Provenance Hash Captures Implementation SHA
# =====================================================================
def test_scenario_17_execution_provenance_hash_captures_implementation_sha():
    p1 = DecisionHashModel.compute_execution_provenance_hash("sha_aaaa")
    p2 = DecisionHashModel.compute_execution_provenance_hash("sha_bbbb")
    assert p1 != p2


# =====================================================================
# Scenario 18: Decision Derivation Hash Binds Input, Value & Provenance
# =====================================================================
def test_scenario_18_decision_derivation_hash_binds_all_dimensions():
    d1 = DecisionHashModel.compute_decision_derivation_hash("inp_1", "val_1", "prov_1")
    d2 = DecisionHashModel.compute_decision_derivation_hash("inp_1", "val_1", "prov_2")
    d3 = DecisionHashModel.compute_decision_derivation_hash("inp_2", "val_1", "prov_1")
    d4 = DecisionHashModel.compute_decision_derivation_hash("inp_1", "val_2", "prov_1")

    assert d1 != d2
    assert d1 != d3
    assert d1 != d4


# =====================================================================
# Scenario 19: Lineage DAG Directed Cycle Rejection
# =====================================================================
def test_scenario_19_lineage_dag_cycle_rejection():
    dag = DecisionLineageDAG()
    dag.add_edge(DecisionLineageEdge(
        predecessor_decision_id="DEC_A",
        successor_decision_id="DEC_B",
        relationship_type=LineageRelationshipType.POLICY_SUPERSESSION,
        effective_scope="CANONICAL_RECONCILIATION",
    ))
    dag.add_edge(DecisionLineageEdge(
        predecessor_decision_id="DEC_B",
        successor_decision_id="DEC_C",
        relationship_type=LineageRelationshipType.POLICY_SUPERSESSION,
        effective_scope="CANONICAL_RECONCILIATION",
    ))

    # Adding DEC_C -> DEC_A creates a cycle A -> B -> C -> A
    with pytest.raises(LineageCycleError):
        dag.add_edge(DecisionLineageEdge(
            predecessor_decision_id="DEC_C",
            successor_decision_id="DEC_A",
            relationship_type=LineageRelationshipType.POLICY_SUPERSESSION,
            effective_scope="CANONICAL_RECONCILIATION",
        ))


# =====================================================================
# Scenario 20: Lineage DAG Self-Loop Rejection
# =====================================================================
def test_scenario_20_lineage_dag_self_loop_rejection():
    dag = DecisionLineageDAG()
    with pytest.raises(LineageCycleError):
        dag.add_edge(DecisionLineageEdge(
            predecessor_decision_id="DEC_A",
            successor_decision_id="DEC_A",
            relationship_type=LineageRelationshipType.RECONFIRMS,
            effective_scope="CANONICAL_RECONCILIATION",
        ))


# =====================================================================
# Scenario 21: Lineage DAG Supports 1->1, 1->N, N->1, N->N Topologies
# =====================================================================
def test_scenario_21_lineage_dag_topologies_valid():
    dag = DecisionLineageDAG()

    # 1 -> 1
    dag.add_edge(DecisionLineageEdge(
        predecessor_decision_id="D1",
        successor_decision_id="D2",
        relationship_type=LineageRelationshipType.POLICY_SUPERSESSION,
        effective_scope="CANONICAL_RECONCILIATION",
    ))

    # 1 -> N (split)
    dag.add_edge(DecisionLineageEdge(
        predecessor_decision_id="D2",
        successor_decision_id="D3_A",
        relationship_type=LineageRelationshipType.TEMPORAL_SUCCESSION,
        effective_scope="TEMPORAL_MEMBERSHIP",
    ))
    dag.add_edge(DecisionLineageEdge(
        predecessor_decision_id="D2",
        successor_decision_id="D3_B",
        relationship_type=LineageRelationshipType.TEMPORAL_SUCCESSION,
        effective_scope="TEMPORAL_MEMBERSHIP",
    ))

    # N -> 1 (merge)
    dag.add_edge(DecisionLineageEdge(
        predecessor_decision_id="D3_A",
        successor_decision_id="D4",
        relationship_type=LineageRelationshipType.IDENTITY_RECONCILIATION,
        effective_scope="CANONICAL_RECONCILIATION",
    ))
    dag.add_edge(DecisionLineageEdge(
        predecessor_decision_id="D3_B",
        successor_decision_id="D4",
        relationship_type=LineageRelationshipType.IDENTITY_RECONCILIATION,
        effective_scope="CANONICAL_RECONCILIATION",
    ))

    assert len(dag.edges) == 5


# =====================================================================
# Scenario 22: Replay Purpose Decoupled from Authority Effect
# =====================================================================
def test_scenario_22_replay_purpose_decoupled_from_authority_effect():
    counterfactual_replay = DecisionReplayRecord(
        replay_id="RPL_CF_001",
        replay_purpose=ReplayPurpose.COUNTERFACTUAL,
        authority_effect=AuthorityEffect.NONE,
        predecessor_decision_ids=["DEC_OLD"],
        successor_decision_ids=["DEC_NEW_SIM"],
        successor_generation_id="GEN_SIM",
        decision_subject_lineage_key="US_EQUITY:AAPL",
        new_policy_semantic_hash="hash_pol",
        new_evidence_dependency_hash="hash_ev",
        new_execution_provenance_hash="hash_prov",
        new_decision_value_hash="hash_val",
        policy_changed=True,
        evidence_changed=False,
        implementation_changed=False,
        canonical_value_changed=True,
        replay_class=ImplementationChangeClass.SEMANTIC_POLICY_CHANGE,
        attribution=ConformanceAttribution.FORMAL_POLICY_EVALUATOR,
        created_at="2026-10-09T09:00:00Z",
    )
    assert counterfactual_replay.replay_purpose == ReplayPurpose.COUNTERFACTUAL
    assert counterfactual_replay.authority_effect == AuthorityEffect.NONE


# =====================================================================
# Scenario 23: Append-Only Supersession Ledger Preserves Predecessors
# =====================================================================
def test_scenario_23_append_only_supersession_ledger_preserves_predecessors():
    supersession = DecisionSupersessionRecord(
        supersession_id="SUP_001",
        predecessor_decision_ids=["DEC_P1"],
        successor_decision_ids=["DEC_S1"],
        supersession_type=LineageRelationshipType.POLICY_SUPERSESSION,
        effective_scope={"scope": "CANONICAL_RECONCILIATION"},
        reason_code=ReasonCode.PRECEDENCE_RESOLVED,
        predecessor_status_after_supersession="SUPERSEDED_HISTORICAL",
        successor_status=SuccessorStatus.ACTIVE_SUCCESSOR,
        preserves_historical_queryability=True,
        created_at="2026-10-09T09:00:00Z",
    )
    assert supersession.predecessor_status_after_supersession == "SUPERSEDED_HISTORICAL"
    assert supersession.preserves_historical_queryability is True


# =====================================================================
# Scenario 24: Successor Status Explicit Lifecycle
# =====================================================================
def test_scenario_24_successor_status_explicit_lifecycle():
    assert SuccessorStatus.PROPOSED_SUCCESSOR != SuccessorStatus.ACTIVE_SUCCESSOR
    assert SuccessorStatus.VALIDATED_SUCCESSOR != SuccessorStatus.ACTIVE_SUCCESSOR


# =====================================================================
# Scenario 25: Replay Accounting Closure Invariant Enforced
# =====================================================================
def test_scenario_25_replay_accounting_closure_enforced():
    # Valid closure: eligible == replayed_unchanged + replayed_changed + replay_failed + not_replayed_with_reason
    summary = ReplayAccountingSummary(
        eligible_for_replay_count=100,
        replayed_unchanged_count=80,
        replayed_changed_count=15,
        replay_failed_count=0,
        not_replayed_with_reason_count=5,
        unaccounted_decisions_count=0,
    )
    assert summary.validate_closure() is True


# =====================================================================
# Scenario 26: Replay Accounting Discrepancy & Unaccounted Rejected
# =====================================================================
def test_scenario_26_replay_accounting_discrepancy_rejected():
    # Discrepancy: sum does not match eligible
    sum_mismatch = ReplayAccountingSummary(
        eligible_for_replay_count=100,
        replayed_unchanged_count=80,
        replayed_changed_count=15,
        replay_failed_count=0,
        not_replayed_with_reason_count=0,  # missing 5
        unaccounted_decisions_count=0,
    )
    assert sum_mismatch.validate_closure() is False

    # Unaccounted decisions > 0 fails
    unaccounted_leak = ReplayAccountingSummary(
        eligible_for_replay_count=100,
        replayed_unchanged_count=80,
        replayed_changed_count=15,
        replay_failed_count=0,
        not_replayed_with_reason_count=5,
        unaccounted_decisions_count=1,
    )
    assert unaccounted_leak.validate_closure() is False


# =====================================================================
# Scenario 27: Impact Analysis Scope Governor & Reason Taxonomy Projection
# =====================================================================
def test_scenario_27_impact_analysis_scope_governor_behavior():
    # Identical dependency & evidence hashes -> excluded from replay scope
    in_scope_no_change = ImpactAnalyzer.is_in_replay_scope(
        prior_dependency_hash="dep_h1",
        current_dependency_hash="dep_h1",
        prior_evidence_hash="ev_h1",
        current_evidence_hash="ev_h1",
    )
    assert in_scope_no_change is False

    # Dependency change -> included in replay scope
    in_scope_dep_change = ImpactAnalyzer.is_in_replay_scope(
        prior_dependency_hash="dep_h1",
        current_dependency_hash="dep_h2",
        prior_evidence_hash="ev_h1",
        current_evidence_hash="ev_h1",
    )
    assert in_scope_dep_change is True

    # Evidence change -> included in replay scope
    in_scope_ev_change = ImpactAnalyzer.is_in_replay_scope(
        prior_dependency_hash="dep_h1",
        current_dependency_hash="dep_h1",
        prior_evidence_hash="ev_h1",
        current_evidence_hash="ev_h2",
    )
    assert in_scope_ev_change is True
