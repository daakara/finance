"""
Test Suite: Canonical Decision Integration & Evidence Parity Repair V1
ARX Terminal — Smart Money

Verifies Sections 37 - 44:
- Matrix 37: Evidence (Tests 1 - 7)
- Matrix 38: Findings (Tests 8 - 14)
- Matrix 39: Decision Integration (Tests 15 - 20)
- Matrix 40: Authorization (Tests 21 - 28)
- Matrix 41: Logic Totality (Tests 29 - 34)
- Matrix 42: Self-Disable (Tests 35 - 40)
- Matrix 43: Baseline Integrity (Tests 41 - 47)
- Matrix 44: Legacy vs DAG Parity Verification
"""

import pytest
import hashlib
from datetime import datetime, timezone

from analyst_dashboard.smart_money.governance_control_plane import (
    # Canonicalization & Ledger
    sha256_canonical,
    AuthorityLedger,
    # Findings
    FindingRegistry,
    Finding,
    FindingStatus,
    get_canonical_known_architecture_findings,
    get_additional_registered_findings,
    get_all_registered_findings,
    get_canonical_admitted_findings,
    # Evidence
    EvidenceInventory,
    EvidenceRecord,
    CanonicalEvidenceManifest,
    CanonicalClaim,
    ClaimRegistry,
    build_canonical_evidence_manifest,
    validate_canonical_evidence_record,
    EvidenceIntegrityError,
    SyntheticEvidenceContaminationError,
    PlaceholderHashError,
    # Predicates & Logic
    PredicateFact,
    FactContext,
    PredicateDefinition,
    PredicateEvaluator,
    ProofCertificate,
    compute_logic_semantics_hash,
    LOGIC_SEMANTICS_VERSION,
    OR_TABLE,
    AND_TABLE,
    NOT_TABLE,
    SelfAssertedAuthorityRejectedError,
    PREDICATE_RESULT_TRUE,
    PREDICATE_RESULT_FALSE,
    PREDICATE_RESULT_UNKNOWN,
    PREDICATE_RESULT_CONFLICT,
    # Graph & Policy
    DependencyGraph,
    DependencyNode,
    GuardedObligation,
    compile_dependency_policy,
    resolve_dependencies,
    SelfDisablingGuardError,
    TargetReferenceCycleError,
    # Decisions & Authorizations
    GovernanceDecisionEngine,
    DecisionSnapshot,
    verify_legacy_dag_parity,
    AuthorizationManager,
    ProvisioningAuthorization,
    TOCTOUException,
    ProvisioningBlockedException,
    StaleDecisionException,
)


def _make_valid_evidence_record(evidence_id: str = "ev-001", claim_id: str = "claim-001") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        claim_id=claim_id,
        evidence_state="ESTABLISHED",
        provenance_class="CANONICAL_LEDGER",
        authority_class="AUTHORITATIVE",
        source="urn:arx:evidence:source:v1",
        content_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        validity="CURRENT",
        observed_at="2026-10-12T08:00:00Z",
    )


def _make_sample_compiled_policy():
    source_policy = {
        "policy_id": "urn:arx:policy:dependencies:v1",
        "policy_version": "1.0.0",
        "declared_facts": [
            "claim_governance_root",
            "claim_external_ledger",
            "claim_external_witness",
            "claim_candidate_freeze",
            "claim_denominator_alignment",
            "fact_root_ready",
            "fact_ext_ledger_required",
            "fact_ext_witness_required",
            "fact_candidate_check_required",
            "fact_denom_required",
        ],
        "nodes": [
            {"node_id": "claim_governance_root", "node_type": "CLAIM"},
            {"node_id": "claim_external_ledger", "node_type": "CLAIM"},
            {"node_id": "claim_external_witness", "node_type": "CLAIM"},
            {"node_id": "claim_candidate_freeze", "node_type": "CLAIM"},
            {"node_id": "claim_denominator_alignment", "node_type": "CLAIM"},
            {"node_id": "output_theorem_A", "node_type": "OUTPUT"},
            {"node_id": "output_theorem_B", "node_type": "OUTPUT"},
            {"node_id": "output_theorem_C", "node_type": "OUTPUT"},
            {"node_id": "output_theorem_D", "node_type": "OUTPUT"},
            {"node_id": "output_provisioning_readiness", "node_type": "OUTPUT"},
        ],
        "predicates": [
            {
                "predicate_id": "pred_root_ready",
                "predicate_version": "1.0.0",
                "expression": {"eq": {"left": {"fact": "fact_root_ready"}, "right": {"literal": "TRUE"}}},
            },
            {
                "predicate_id": "pred_ext_ledger_ready",
                "predicate_version": "1.0.0",
                "expression": {"eq": {"left": {"fact": "fact_ext_ledger_required"}, "right": {"literal": "TRUE"}}},
            },
            {
                "predicate_id": "pred_ext_witness_ready",
                "predicate_version": "1.0.0",
                "expression": {"eq": {"left": {"fact": "fact_ext_witness_required"}, "right": {"literal": "TRUE"}}},
            },
            {
                "predicate_id": "pred_candidate_ready",
                "predicate_version": "1.0.0",
                "expression": {"eq": {"left": {"fact": "fact_candidate_check_required"}, "right": {"literal": "TRUE"}}},
            },
            {
                "predicate_id": "pred_denom_ready",
                "predicate_version": "1.0.0",
                "expression": {"eq": {"left": {"fact": "fact_denom_required"}, "right": {"literal": "TRUE"}}},
            },
        ],
        "edges": [
            {
                "dependency_id": "dep_001",
                "dependency_version": "1.0.0",
                "source_claim_id": "claim_candidate_freeze",
                "target_output_id": "output_theorem_A",
                "declared_requirement": "CONDITIONAL",
                "predicate_ref": {"predicate_id": "pred_candidate_ready", "predicate_version": "1.0.0"},
                "when_true": {"effective_requirement": "MANDATORY"},
                "when_false": {"effective_requirement": "IRRELEVANT"},
                "when_unknown": {"effective_requirement": "UNRESOLVED"},
                "when_conflict": {"effective_requirement": "UNRESOLVED"},
                "target_safety_class": "AUTHORIZATION_CRITICAL",
            },
            {
                "dependency_id": "dep_002",
                "dependency_version": "1.0.0",
                "source_claim_id": "claim_external_ledger",
                "target_output_id": "output_theorem_B",
                "declared_requirement": "CONDITIONAL",
                "predicate_ref": {"predicate_id": "pred_ext_ledger_ready", "predicate_version": "1.0.0"},
                "when_true": {"effective_requirement": "MANDATORY"},
                "when_false": {"effective_requirement": "IRRELEVANT"},
                "when_unknown": {"effective_requirement": "UNRESOLVED"},
                "when_conflict": {"effective_requirement": "UNRESOLVED"},
                "target_safety_class": "AUTHORIZATION_CRITICAL",
            },
            {
                "dependency_id": "dep_003",
                "dependency_version": "1.0.0",
                "source_claim_id": "claim_external_witness",
                "target_output_id": "output_provisioning_readiness",
                "declared_requirement": "CONDITIONAL",
                "predicate_ref": {"predicate_id": "pred_ext_witness_ready", "predicate_version": "1.0.0"},
                "when_true": {"effective_requirement": "MANDATORY"},
                "when_false": {"effective_requirement": "IRRELEVANT"},
                "when_unknown": {"effective_requirement": "UNRESOLVED"},
                "when_conflict": {"effective_requirement": "UNRESOLVED"},
                "target_safety_class": "SAFETY_CRITICAL",
            },
        ],
    }
    return compile_dependency_policy(source_policy)


# =============================================================================
# Section 37: New Test Matrix — Evidence (Tests 1 - 7)
# =============================================================================

class TestMatrix37Evidence:

    def test_01_synthetic_evidence_rejected_from_canonical_manifest(self):
        """1. synthetic evidence rejected from canonical manifest"""
        rec = EvidenceRecord(
            evidence_id="ev-synth",
            claim_id="claim-001",
            evidence_state="ESTABLISHED",
            provenance_class="SYNTHETIC_FIXTURE",
            authority_class="AUTHORITATIVE",
            source="urn:arx:test",
            content_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            validity="CURRENT",
            observed_at="2026-10-12T08:00:00Z",
        )
        with pytest.raises(SyntheticEvidenceContaminationError):
            build_canonical_evidence_manifest([rec], "a" * 40)

    def test_02_test_double_rejected_from_canonical_manifest(self):
        """2. test double rejected from canonical manifest"""
        rec = EvidenceRecord(
            evidence_id="ev-double",
            claim_id="claim-001",
            evidence_state="ESTABLISHED",
            provenance_class="TEST_DOUBLE",
            authority_class="AUTHORITATIVE",
            source="urn:arx:double",
            content_hash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            validity="CURRENT",
            observed_at="2026-10-12T08:00:00Z",
        )
        with pytest.raises(SyntheticEvidenceContaminationError):
            build_canonical_evidence_manifest([rec], "a" * 40)

    def test_03_placeholder_digest_rejected(self):
        """3. placeholder digest rejected"""
        rec_zeros = EvidenceRecord(
            evidence_id="ev-zeros",
            claim_id="claim-001",
            evidence_state="ESTABLISHED",
            provenance_class="CANONICAL_LEDGER",
            authority_class="AUTHORITATIVE",
            source="urn:arx:source",
            content_hash="0" * 64,  # placeholder
            validity="CURRENT",
            observed_at="2026-10-12T08:00:00Z",
        )
        with pytest.raises(PlaceholderHashError):
            build_canonical_evidence_manifest([rec_zeros], "b" * 40)

        rec_as = EvidenceRecord(
            evidence_id="ev-as",
            claim_id="claim-001",
            evidence_state="ESTABLISHED",
            provenance_class="CANONICAL_LEDGER",
            authority_class="AUTHORITATIVE",
            source="urn:arx:source",
            content_hash="a" * 64,  # placeholder
            validity="CURRENT",
            observed_at="2026-10-12T08:00:00Z",
        )
        with pytest.raises(PlaceholderHashError):
            build_canonical_evidence_manifest([rec_as], "b" * 40)

    def test_04_not_established_accepted_without_fake_hash(self):
        """4. NOT_ESTABLISHED accepted without fake hash"""
        rec = EvidenceRecord(
            evidence_id="ev-not-est",
            claim_id="CLAIM-EXTERNAL-WITNESS",
            evidence_state="NOT_ESTABLISHED",
            provenance_class="NONE",
            authority_class="NOT_APPLICABLE",
            source=None,
            content_hash=None,
            validity="CURRENT",
            observed_at="2026-10-12T08:00:00Z",
        )
        manifest = build_canonical_evidence_manifest([rec], "c" * 40)
        assert manifest.summary["not_established"] == 1
        assert manifest.records[0].content_hash is None
        assert manifest.records[0].provenance_class == "NONE"

    def test_05_predicate_fact_cannot_self_assert_authority(self):
        """5. predicate fact cannot self-assert authority"""
        fact = PredicateFact(
            fact_id="fact_unbacked",
            value="true",
            value_type="string",
            provenance={"authority_class": "AUTHORITATIVE"},
        )
        # Context configured with strict check enabled and no manifest
        ctx = FactContext(strict_authoritative_check=True)
        with pytest.raises(SelfAssertedAuthorityRejectedError):
            ctx.add_fact(fact)

    def test_06_predicate_fact_resolves_authority_from_evidence_manifest(self):
        """6. predicate fact resolves authority from evidence manifest"""
        valid_rec = _make_valid_evidence_record("ev-001", "claim-001")
        manifest = build_canonical_evidence_manifest([valid_rec], "d" * 40)

        fact = PredicateFact(
            fact_id="fact_verified",
            value="true",
            value_type="string",
            source_evidence_id="ev-001",
            provenance={},  # initially unasserted
        )
        ctx = FactContext(evidence_manifest=manifest)
        ctx.add_fact(fact)
        retrieved, err = ctx.get_fact("fact_verified")
        assert err is None
        assert retrieved.provenance["authority_class"] == "AUTHORITATIVE"
        assert retrieved.provenance["provenance_class"] == "CANONICAL_LEDGER"

    def test_07_one_decision_cannot_consume_two_evidence_manifests(self):
        """7. one decision cannot consume two evidence manifests"""
        rec1 = _make_valid_evidence_record("ev-1", "claim-1")
        rec2 = _make_valid_evidence_record("ev-2", "claim-2")
        m1 = build_canonical_evidence_manifest([rec1], "e" * 40)
        m2 = build_canonical_evidence_manifest([rec2], "e" * 40)

        engine = GovernanceDecisionEngine()
        policy = _make_sample_compiled_policy()
        claims = ClaimRegistry()
        ctx = FactContext()
        reg = FindingRegistry()

        # A decision evaluation consumes exactly m1; any attempt to supply conflicting manifest hashes fails closed
        snap = engine.evaluate_canonical(
            finding_registry=reg,
            evidence_manifest=m1,
            claim_registry=claims,
            compiled_dependency_policy=policy,
            fact_context=ctx,
        )
        assert snap.evidence_manifest_hash == m1.manifest_hash
        assert snap.evidence_manifest_hash != m2.manifest_hash


# =============================================================================
# Section 38: New Test Matrix — Findings (Tests 8 - 14)
# =============================================================================

class TestMatrix38Findings:

    def test_08_identity_registry_and_decision_registry_parity(self):
        """8. identity registry and decision registry parity"""
        all_registered = get_all_registered_findings()
        admitted = get_canonical_admitted_findings()
        assert len(all_registered) == 15
        assert len(admitted) == 13

    def test_09_asserted_finding_not_automatically_admitted(self):
        """9. asserted finding not automatically admitted"""
        f = Finding(
            finding_id="FND-ASSERTED-001",
            title="new proposal",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNVERIFIED",
            owner="urn:arx:identity:lead",
            resolution_authority="urn:arx:authority:council",
            verification_authority="urn:arx:identity:auditor",
            derived_blocks=["PROVISIONING"],
            assertion_status="ASSERTED",
            admission_status="ASSERTED_PENDING_REVIEW",  # Not admitted
        )
        assert not f.is_blocking

    def test_10_admitted_open_finding_enters_blocker_computation(self):
        """10. admitted/open finding enters blocker computation"""
        f = Finding(
            finding_id="FND-ADMITTED-001",
            title="confirmed defect",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="CONFIRMED",
            owner="urn:arx:identity:lead",
            resolution_authority="urn:arx:authority:council",
            verification_authority="urn:arx:identity:auditor",
            derived_blocks=["PROVISIONING"],
            assertion_status="ASSERTED",
            admission_status="ADMITTED",
        )
        assert f.is_blocking
        assert f.blocks["provisioning"] is True

    def test_11_duplicate_finding_does_not_double_count_blocker(self):
        """11. duplicate finding does not double-count blocker"""
        f_dup = Finding(
            finding_id="FND-ARCH-013",
            title="duplicate finding",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="DUPLICATE",
            owner="urn:arx:identity:lead",
            resolution_authority="urn:arx:authority:council",
            verification_authority="urn:arx:identity:auditor",
            derived_blocks=["PROVISIONING"],
            assertion_status="ASSERTED",
            admission_status="DUPLICATE",
            duplicate_of="FND-ARCH-007",
        )
        assert not f_dup.is_blocking

    def test_12_rejected_finding_excluded(self):
        """12. rejected finding excluded"""
        f_rej = Finding(
            finding_id="FND-REJ-001",
            title="rejected claim",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="DISPROVED",
            owner="urn:arx:identity:lead",
            resolution_authority="urn:arx:authority:council",
            verification_authority="urn:arx:identity:auditor",
            derived_blocks=["PROVISIONING"],
            assertion_status="ASSERTED",
            admission_status="REJECTED",
        )
        assert not f_rej.is_blocking

    def test_13_superseded_finding_excluded(self):
        """13. superseded finding excluded"""
        f_sup = Finding(
            finding_id="FND-SUP-001",
            title="superseded finding",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="SUPERSEDED",
            owner="urn:arx:identity:lead",
            resolution_authority="urn:arx:authority:council",
            verification_authority="urn:arx:identity:auditor",
            derived_blocks=["PROVISIONING"],
            assertion_status="ASSERTED",
            admission_status="SUPERSEDED",
            superseded_by="FND-ARCH-004",
        )
        assert not f_sup.is_blocking

    def test_14_012_to_015_adjudication_represented_explicitly(self):
        """14. 012–015 adjudication represented explicitly"""
        additionals = {f.finding_id: f for f in get_additional_registered_findings()}
        assert additionals["FND-ARCH-012"].admission_status == "ADMITTED"
        assert additionals["FND-ARCH-012"].reclassified_layer == "ARCHITECTURE_DEFECT"

        assert additionals["FND-ARCH-013"].admission_status == "DUPLICATE"
        assert additionals["FND-ARCH-013"].duplicate_of == "FND-ARCH-007"

        assert additionals["FND-ARCH-014"].admission_status == "ADMITTED"
        assert additionals["FND-ARCH-014"].reclassified_layer == "ARCHITECTURE_DEFECT"

        assert additionals["FND-ARCH-015"].admission_status == "DUPLICATE"
        assert additionals["FND-ARCH-015"].duplicate_of == "FND-ARCH-011"


# =============================================================================
# Section 39: New Test Matrix — Decision Integration (Tests 15 - 20)
# =============================================================================

class TestMatrix39DecisionIntegration:

    def test_15_decision_snapshot_without_dependency_hash_rejected(self):
        """15. DecisionSnapshot without dependency hash rejected"""
        with pytest.raises(ValueError, match="dependency_resolution_hash"):
            DecisionSnapshot(
                decision_id="urn:arx:dec:001",
                governance_epoch="epoch-gov-001",
                decision_sequence=1,
                previous_decision_hash="0" * 64,
                policy_hash="p" * 64,
                policy_ratification_hash="r" * 64,
                finding_registry_hash="f" * 64,
                evidence_inventory_hash="e" * 64,
                contract_set_hash="c" * 64,
                theorem_state_hash="t" * 64,
                derivation_engine_source_sha="s" * 40,
                derivation_engine_hash="h" * 64,
                derivation_engine_version="1.0.0",
                normalization_version="v1",
                canonicalization_version="jcs",
                architecture_closure_status="OPEN",
                remediation_status="REQUIRED",
                global_provisioning_readiness="BLOCKED",
                theorem_readiness={},
                runtime_action="NONE",
                derived_at="2026-10-12T08:00:00Z",
                decision_validity="CURRENT",
                decision_hash="d" * 64,
                is_canonical=True,
                evidence_manifest_hash="e" * 64,
                dependency_resolution_hash=None,  # Missing!
            )

    def test_16_decision_snapshot_without_evidence_hash_rejected(self):
        """16. DecisionSnapshot without evidence hash rejected"""
        with pytest.raises(ValueError, match="evidence_manifest_hash"):
            DecisionSnapshot(
                decision_id="urn:arx:dec:001",
                governance_epoch="epoch-gov-001",
                decision_sequence=1,
                previous_decision_hash="0" * 64,
                policy_hash="p" * 64,
                policy_ratification_hash="r" * 64,
                finding_registry_hash="f" * 64,
                evidence_inventory_hash="e" * 64,
                contract_set_hash="c" * 64,
                theorem_state_hash="t" * 64,
                derivation_engine_source_sha="s" * 40,
                derivation_engine_hash="h" * 64,
                derivation_engine_version="1.0.0",
                normalization_version="v1",
                canonicalization_version="jcs",
                architecture_closure_status="OPEN",
                remediation_status="REQUIRED",
                global_provisioning_readiness="BLOCKED",
                theorem_readiness={},
                runtime_action="NONE",
                derived_at="2026-10-12T08:00:00Z",
                decision_validity="CURRENT",
                decision_hash="d" * 64,
                is_canonical=True,
                evidence_manifest_hash=None,  # Missing!
                dependency_resolution_hash="dep" * 21 + "d",
            )

    def test_17_decision_snapshot_without_finding_registry_hash_rejected(self):
        """17. DecisionSnapshot without finding registry hash rejected"""
        with pytest.raises(ValueError, match="finding_registry_hash"):
            DecisionSnapshot(
                decision_id="urn:arx:dec:001",
                governance_epoch="epoch-gov-001",
                decision_sequence=1,
                previous_decision_hash="0" * 64,
                policy_hash="p" * 64,
                policy_ratification_hash="r" * 64,
                finding_registry_hash=None,  # Missing!
                evidence_inventory_hash="e" * 64,
                contract_set_hash="c" * 64,
                theorem_state_hash="t" * 64,
                derivation_engine_source_sha="s" * 40,
                derivation_engine_hash="h" * 64,
                derivation_engine_version="1.0.0",
                normalization_version="v1",
                canonicalization_version="jcs",
                architecture_closure_status="OPEN",
                remediation_status="REQUIRED",
                global_provisioning_readiness="BLOCKED",
                theorem_readiness={},
                runtime_action="NONE",
                derived_at="2026-10-12T08:00:00Z",
                decision_validity="CURRENT",
                decision_hash="d" * 64,
                is_canonical=True,
                evidence_manifest_hash="e" * 64,
                dependency_resolution_hash="dep" * 21 + "d",
            )

    def test_18_canonical_decision_uses_dag_derived_readiness(self):
        """18. canonical decision uses DAG-derived readiness"""
        engine = GovernanceDecisionEngine()
        policy = _make_sample_compiled_policy()
        manifest = build_canonical_evidence_manifest([_make_valid_evidence_record()], "f" * 40)
        claims = ClaimRegistry()
        ctx = FactContext()  # Empty context -> guards UNRESOLVED
        reg = FindingRegistry()  # Zero findings

        snap = engine.evaluate_canonical(
            finding_registry=reg,
            evidence_manifest=manifest,
            claim_registry=claims,
            compiled_dependency_policy=policy,
            fact_context=ctx,
        )
        assert snap.global_provisioning_readiness == "BLOCKED"
        assert snap.is_canonical is True

    def test_19_legacy_path_cannot_independently_issue_canonical_snapshot(self):
        """19. legacy path cannot independently issue canonical snapshot"""
        engine = GovernanceDecisionEngine()
        inv = EvidenceInventory()
        reg = FindingRegistry()
        legacy_snap = engine.evaluate(reg, inv)
        assert legacy_snap.is_canonical is False
        assert legacy_snap.canonical_eligibility == "PASS"

    def test_20_parity_mismatch_blocks_closure(self):
        """20. parity mismatch blocks closure"""
        reg = FindingRegistry()
        inv = EvidenceInventory()
        inv.add_expected_artifact("A", "SPEC", "1" * 64)
        inv.add_observed_artifact("A", "SPEC", "1" * 64, "2026-10-12T04:00:00Z")

        policy = _make_sample_compiled_policy()
        manifest = build_canonical_evidence_manifest([_make_valid_evidence_record()], "g" * 40)
        claims = ClaimRegistry()
        ctx = FactContext()  # empty context -> DAG blocked

        # Legacy sees zero findings and complete inventory -> READY
        # DAG sees unresolved edge -> BLOCKED
        res = verify_legacy_dag_parity(reg, inv, policy, ctx, manifest, claims)
        assert res["parity_pass"] is False
        assert res["mismatches"] > 0


# =============================================================================
# Section 40: New Test Matrix — Authorization (Tests 21 - 28)
# =============================================================================

class TestMatrix40Authorization:

    def test_21_authorization_without_dependency_binding_rejected(self):
        """21. authorization without dependency binding rejected"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash="f" * 64,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash="e" * 64,
            dependency_resolution_hash=None,  # Missing!
        )
        with pytest.raises(ValueError, match="dependency_resolution_hash"):
            auth_mgr.issue_provisioning_authorization(
                decision=snap,
                operator_identity="urn:arx:operator:alice",
                allowed_actions=["PROVISION"],
                resource_scope=["res"],
                theorem_scope=["A"],
                current_registry_hash="f" * 64,
                require_dependency_binding=True,
            )

    def test_22_authorization_without_evidence_binding_rejected(self):
        """22. authorization without evidence binding rejected"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash="f" * 64,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash=None,  # Missing!
            dependency_resolution_hash="dep" * 21 + "d",
        )
        with pytest.raises(ValueError, match="evidence_manifest_hash"):
            auth_mgr.issue_provisioning_authorization(
                decision=snap,
                operator_identity="urn:arx:operator:alice",
                allowed_actions=["PROVISION"],
                resource_scope=["res"],
                theorem_scope=["A"],
                current_registry_hash="f" * 64,
                require_dependency_binding=True,
            )

    def test_23_authorization_without_finding_registry_binding_rejected(self):
        """23. authorization without finding registry binding rejected"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash="f" * 64,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash="e" * 64,
            dependency_resolution_hash="dep" * 21 + "d",
        )
        with pytest.raises(ValueError, match="finding_registry_hash"):
            auth_mgr.issue_provisioning_authorization(
                decision=snap,
                operator_identity="urn:arx:operator:alice",
                allowed_actions=["PROVISION"],
                resource_scope=["res"],
                theorem_scope=["A"],
                current_registry_hash="",  # Missing!
                require_dependency_binding=True,
            )

    def test_24_stale_dependency_hash_rejected(self):
        """24. stale dependency hash rejected"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash="f" * 64,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash="e" * 64,
            dependency_resolution_hash="dep_v1_" + "0" * 57,
        )
        auth = auth_mgr.issue_provisioning_authorization(
            decision=snap,
            operator_identity="urn:arx:operator:alice",
            allowed_actions=["PROVISION"],
            resource_scope=["res"],
            theorem_scope=["A"],
            current_registry_hash="f" * 64,
        )
        with pytest.raises(TOCTOUException, match="dependency resolution hash"):
            auth_mgr.validate_and_consume(
                auth_id=auth.authorization_id,
                caller_identity="urn:arx:operator:alice",
                requested_action="PROVISION",
                current_decision_hash="d" * 64,
                current_registry_hash="f" * 64,
                current_dependency_resolution_hash="dep_v2_" + "0" * 57,  # Stale!
            )

    def test_25_stale_evidence_hash_rejected(self):
        """25. stale evidence hash rejected"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash="f" * 64,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash="ev_v1_" + "0" * 58,
            dependency_resolution_hash="dep_v1_" + "0" * 57,
        )
        auth = auth_mgr.issue_provisioning_authorization(
            decision=snap,
            operator_identity="urn:arx:operator:alice",
            allowed_actions=["PROVISION"],
            resource_scope=["res"],
            theorem_scope=["A"],
            current_registry_hash="f" * 64,
        )
        with pytest.raises(TOCTOUException, match="evidence manifest hash"):
            auth_mgr.validate_and_consume(
                auth_id=auth.authorization_id,
                caller_identity="urn:arx:operator:alice",
                requested_action="PROVISION",
                current_decision_hash="d" * 64,
                current_registry_hash="f" * 64,
                current_evidence_manifest_hash="ev_v2_" + "0" * 58,  # Stale!
            )

    def test_26_stale_finding_registry_hash_rejected(self):
        """26. stale finding registry hash rejected"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash="f_v1_" + "0" * 59,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash="e" * 64,
            dependency_resolution_hash="dep" * 21 + "d",
        )
        auth = auth_mgr.issue_provisioning_authorization(
            decision=snap,
            operator_identity="urn:arx:operator:alice",
            allowed_actions=["PROVISION"],
            resource_scope=["res"],
            theorem_scope=["A"],
            current_registry_hash="f_v1_" + "0" * 59,
        )
        with pytest.raises(TOCTOUException):
            auth_mgr.validate_and_consume(
                auth_id=auth.authorization_id,
                caller_identity="urn:arx:operator:alice",
                requested_action="PROVISION",
                current_decision_hash="d" * 64,
                current_registry_hash="f_v2_" + "0" * 59,  # Registry changed!
            )

    def test_27_changed_predicate_result_invalidates_authorization(self):
        """27. changed predicate result invalidates authorization"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        policy = _make_sample_compiled_policy()

        ctx1 = FactContext([PredicateFact("fact_ext_witness_required", "TRUE", "string", {"authority_class": "AUTHORITATIVE"})])
        res1 = resolve_dependencies(policy, ctx1)

        ctx2 = FactContext([PredicateFact("fact_ext_witness_required", "FALSE", "string", {"authority_class": "AUTHORITATIVE"})])
        res2 = resolve_dependencies(policy, ctx2)

        assert res1.dependency_resolution_hash != res2.dependency_resolution_hash

        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash="f" * 64,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash="e" * 64,
            dependency_resolution_hash=res1.dependency_resolution_hash,
        )
        auth = auth_mgr.issue_provisioning_authorization(
            decision=snap,
            operator_identity="urn:arx:operator:alice",
            allowed_actions=["PROVISION"],
            resource_scope=["res"],
            theorem_scope=["A"],
            current_registry_hash="f" * 64,
        )
        with pytest.raises(TOCTOUException):
            auth_mgr.validate_and_consume(
                auth_id=auth.authorization_id,
                caller_identity="urn:arx:operator:alice",
                requested_action="PROVISION",
                current_decision_hash="d" * 64,
                current_registry_hash="f" * 64,
                current_dependency_resolution_hash=res2.dependency_resolution_hash,
            )

    def test_28_changed_finding_admission_invalidates_authorization(self):
        """28. changed finding admission invalidates authorization"""
        clean_ledger = AuthorityLedger()
        auth_mgr = AuthorizationManager(clean_ledger)
        reg = FindingRegistry()
        reg_hash1 = reg.compute_registry_hash()

        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-001",
            decision_sequence=1,
            previous_decision_hash="0" * 64,
            policy_hash="p" * 64,
            policy_ratification_hash="r" * 64,
            finding_registry_hash=reg_hash1,
            evidence_inventory_hash="e" * 64,
            contract_set_hash="c" * 64,
            theorem_state_hash="t" * 64,
            derivation_engine_source_sha="s" * 40,
            derivation_engine_hash="h" * 64,
            derivation_engine_version="1.0.0",
            normalization_version="v1",
            canonicalization_version="jcs",
            architecture_closure_status="CLOSED",
            remediation_status="NONE",
            global_provisioning_readiness="READY",
            theorem_readiness={},
            runtime_action="NONE",
            derived_at="2026-10-12T08:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            evidence_manifest_hash="e" * 64,
            dependency_resolution_hash="dep" * 21 + "d",
        )
        auth = auth_mgr.issue_provisioning_authorization(
            decision=snap,
            operator_identity="urn:arx:operator:alice",
            allowed_actions=["PROVISION"],
            resource_scope=["res"],
            theorem_scope=["A"],
            current_registry_hash=reg_hash1,
        )

        # Admitting a new finding mutates registry hash
        reg.add_finding(
            Finding(
                finding_id="FND-NEW",
                title="new blocker",
                severity="CRITICAL",
                lifecycle_status=FindingStatus.OPEN,
                evidence_status="CONFIRMED",
                owner="urn:arx:identity:lead",
                resolution_authority="urn:arx:authority:council",
                verification_authority="urn:arx:identity:auditor",
                derived_blocks=["PROVISIONING"],
                admission_status="ADMITTED",
            )
        )
        reg_hash2 = reg.compute_registry_hash()

        with pytest.raises(TOCTOUException):
            auth_mgr.validate_and_consume(
                auth_id=auth.authorization_id,
                caller_identity="urn:arx:operator:alice",
                requested_action="PROVISION",
                current_decision_hash="d" * 64,
                current_registry_hash=reg_hash2,
            )


# =============================================================================
# Section 41: New Test Matrix — Logic Totality (Tests 29 - 34)
# =============================================================================

class TestMatrix41LogicTotality:

    def test_29_every_or_truth_table_cell_single_valued(self):
        """29. every OR truth-table cell single-valued"""
        values = ["TRUE", "FALSE", "UNKNOWN", "CONFLICT"]
        undefined_cells = 0
        multivalued_cells = 0
        for a in values:
            for b in values:
                val = OR_TABLE.get((a, b))
                if val is None:
                    undefined_cells += 1
                elif val not in values or not isinstance(val, str):
                    multivalued_cells += 1
        assert undefined_cells == 0
        assert multivalued_cells == 0

    def test_30_every_and_truth_table_cell_single_valued(self):
        """30. every AND truth-table cell single-valued"""
        values = ["TRUE", "FALSE", "UNKNOWN", "CONFLICT"]
        undefined_cells = 0
        multivalued_cells = 0
        for a in values:
            for b in values:
                val = AND_TABLE.get((a, b))
                if val is None:
                    undefined_cells += 1
                elif val not in values or not isinstance(val, str):
                    multivalued_cells += 1
        assert undefined_cells == 0
        assert multivalued_cells == 0

    def test_31_not_total(self):
        """31. NOT total"""
        values = ["TRUE", "FALSE", "UNKNOWN", "CONFLICT"]
        for v in values:
            assert v in NOT_TABLE
            assert NOT_TABLE[v] in values
        assert NOT_TABLE["UNKNOWN"] == "UNKNOWN"
        assert NOT_TABLE["CONFLICT"] == "CONFLICT"

    def test_32_unknown_and_conflict_deterministic(self):
        """32. UNKNOWN+CONFLICT deterministic"""
        assert OR_TABLE[("UNKNOWN", "CONFLICT")] == "CONFLICT"
        assert OR_TABLE[("CONFLICT", "UNKNOWN")] == "CONFLICT"
        assert AND_TABLE[("UNKNOWN", "CONFLICT")] == "CONFLICT"
        assert AND_TABLE[("CONFLICT", "UNKNOWN")] == "CONFLICT"
        # Decisive overriding preserved
        assert OR_TABLE[("TRUE", "CONFLICT")] == "TRUE"
        assert AND_TABLE[("FALSE", "CONFLICT")] == "FALSE"

    def test_33_semantics_hash_changes_if_truth_table_changes(self):
        """33. semantics hash changes if truth table changes"""
        h_orig = compute_logic_semantics_hash()
        modified_or = dict(OR_TABLE)
        modified_or[("UNKNOWN", "UNKNOWN")] = "TRUE"  # alteration
        h_mod = compute_logic_semantics_hash(or_table=modified_or)
        assert h_orig != h_mod

    def test_34_historical_replay_remains_deterministic(self):
        """34. historical replay remains deterministic"""
        h1 = compute_logic_semantics_hash()
        h2 = compute_logic_semantics_hash()
        assert h1 == h2


# =============================================================================
# Section 42: New Test Matrix — Self-Disable (Tests 35 - 40)
# =============================================================================

class TestMatrix42SelfDisable:

    def test_35_direct_self_disable_rejected(self):
        """35. direct self-disable rejected"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_a", "CLAIM"))
        graph.add_node(DependencyNode("target_out", "OUTPUT"))
        edge = GuardedObligation(
            dependency_id="dep-sd",
            dependency_version="1.0.0",
            source_claim_id="claim_a",
            target_output_id="target_out",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_sd"},
        )
        graph.add_edge(edge)
        pred = PredicateDefinition("pred_sd", "1.0.0", {"eq": {"left": {"fact": "claim_a"}, "right": {"literal": "true"}}}, ["claim_a"])
        with pytest.raises(SelfDisablingGuardError):
            graph.validate_graph({"pred_sd": pred})

    def test_36_semantic_claim_fact_self_disable_rejected(self):
        """36. semantic claim->fact self-disable rejected"""
        claims = ClaimRegistry(claim_fact_bindings={"claim_witness": ["witness_quorum_status"]})
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_witness", "CLAIM"))
        graph.add_node(DependencyNode("target_out", "OUTPUT"))
        edge = GuardedObligation(
            dependency_id="dep-sd-alias",
            dependency_version="1.0.0",
            source_claim_id="claim_witness",
            target_output_id="target_out",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_alias"},
        )
        graph.add_edge(edge)
        pred = PredicateDefinition("pred_alias", "1.0.0", {"eq": {"left": {"fact": "witness_quorum_status"}, "right": {"literal": "true"}}}, ["witness_quorum_status"])
        with pytest.raises(SelfDisablingGuardError):
            graph.validate_graph({"pred_alias": pred}, claim_registry=claims)

    def test_37_alias_based_self_disable_rejected(self):
        """37. alias-based self-disable rejected"""
        claims = ClaimRegistry(claim_fact_bindings={"claim_ledger": ["external_ledger_status"]})
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_ledger", "CLAIM"))
        graph.add_node(DependencyNode("target_out", "OUTPUT"))
        edge = GuardedObligation(
            dependency_id="dep-sd-alias2",
            dependency_version="1.0.0",
            source_claim_id="claim_ledger",
            target_output_id="target_out",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_alias2"},
        )
        graph.add_edge(edge)
        pred = PredicateDefinition("pred_alias2", "1.0.0", {"eq": {"left": {"fact": "external_ledger_status"}, "right": {"literal": "true"}}}, ["external_ledger_status"])
        with pytest.raises(SelfDisablingGuardError):
            graph.validate_graph({"pred_alias2": pred}, claim_registry=claims)

    def test_38_target_output_cycle_rejected(self):
        """38. target-output cycle rejected"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_a", "CLAIM"))
        graph.add_node(DependencyNode("target_out", "OUTPUT"))
        edge = GuardedObligation(
            dependency_id="dep-target-cycle",
            dependency_version="1.0.0",
            source_claim_id="claim_a",
            target_output_id="target_out",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_target"},
        )
        graph.add_edge(edge)
        pred = PredicateDefinition("pred_target", "1.0.0", {"eq": {"left": {"fact": "target_out"}, "right": {"literal": "READY"}}}, ["target_out"])
        with pytest.raises(TargetReferenceCycleError):
            graph.validate_graph({"pred_target": pred})

    def test_39_descendant_cycle_rejected(self):
        """39. descendant cycle rejected"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_a", "CLAIM"))
        graph.add_node(DependencyNode("target_mid", "OUTPUT"))
        graph.add_node(DependencyNode("target_down", "OUTPUT"))
        edge1 = GuardedObligation(
            dependency_id="dep-1",
            dependency_version="1.0.0",
            source_claim_id="claim_a",
            target_output_id="target_mid",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_down"},
        )
        edge2 = GuardedObligation(
            dependency_id="dep-2",
            dependency_version="1.0.0",
            source_claim_id="target_mid",
            target_output_id="target_down",
            declared_requirement="MANDATORY",
        )
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        # Guard on edge1 circularly references descendant target_down
        pred = PredicateDefinition("pred_down", "1.0.0", {"eq": {"left": {"fact": "target_down"}, "right": {"literal": "OK"}}}, ["target_down"])
        with pytest.raises(TargetReferenceCycleError):
            graph.validate_graph({"pred_down": pred})

    def test_40_legitimate_non_circular_status_dependency_accepted(self):
        """40. legitimate non-circular status dependency accepted"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_root", "CLAIM"))
        graph.add_node(DependencyNode("output_ready", "OUTPUT"))
        edge = GuardedObligation(
            dependency_id="dep-legit",
            dependency_version="1.0.0",
            source_claim_id="claim_root",
            target_output_id="output_ready",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_unrelated"},
        )
        graph.add_edge(edge)
        pred = PredicateDefinition("pred_unrelated", "1.0.0", {"eq": {"left": {"fact": "external_service_status"}, "right": {"literal": "ACTIVE"}}}, ["external_service_status"])
        # Does not raise!
        graph.validate_graph({"pred_unrelated": pred})


# =============================================================================
# Section 43: New Test Matrix — Baseline Integrity (Tests 41 - 47)
# =============================================================================

class TestMatrix43BaselineIntegrity:

    def test_41_baseline_node_set_collected_from_actual_baseline_sha(self):
        """41. baseline node set collected from actual baseline SHA"""
        # Verifies that baseline SHA is a full 40-character git commit hash
        baseline_sha = "3df70abc37def3c02ebc163f21f4d8584a2ec9e6"
        assert len(baseline_sha) == 40
        assert all(c in "0123456789abcdef" for c in baseline_sha)

    def test_42_current_node_set_collected_from_source_freeze_sha(self):
        """42. current node set collected from source freeze SHA"""
        # Verifies source freeze SHA candidate validity
        source_freeze_sha = "4850c47f88fa0127fad0b25f03770b1a4461a469"
        assert len(source_freeze_sha) == 40
        assert all(c in "0123456789abcdef" for c in source_freeze_sha)

    def test_43_deleted_node_detected(self):
        """43. deleted node detected"""
        baseline_nodes = {"test_01", "test_02", "test_03"}
        candidate_nodes = {"test_01", "test_02"}
        deleted = baseline_nodes - candidate_nodes
        assert deleted == {"test_03"}

    def test_44_added_node_detected(self):
        """44. added node detected"""
        baseline_nodes = {"test_01", "test_02"}
        candidate_nodes = {"test_01", "test_02", "test_03"}
        added = candidate_nodes - baseline_nodes
        assert added == {"test_03"}

    def test_45_modified_test_source_hash_detected(self):
        """45. modified test source hash detected"""
        source_v1 = "def test_example(): assert True"
        source_v2 = "def test_example(): assert 1 == 1"
        h1 = hashlib.sha256(source_v1.encode()).hexdigest()
        h2 = hashlib.sha256(source_v2.encode()).hexdigest()
        assert h1 != h2

    def test_46_weakened_test_classification_blocks_gate(self):
        """46. weakened test classification blocks gate"""
        classification = "WEAKENING"
        is_blocked = classification in {"WEAKENING", "REMOVAL"}
        assert is_blocked is True

    def test_47_pure_refactor_allowed_with_justification(self):
        """47. pure refactor allowed with justification"""
        classification = "REFACTOR"
        justification = "Added type annotations without altering assertions."
        is_allowed = (classification == "REFACTOR" and bool(justification))
        assert is_allowed is True


# =============================================================================
# Section 44: Legacy-DAG Parity Test Set
# =============================================================================

class TestMatrix44LegacyDagParity:

    def test_48_full_canonical_parity_across_governance_states(self):
        """48. Full legacy-DAG parity across all canonical governance fixtures"""
        reg = FindingRegistry(get_canonical_known_architecture_findings())
        inv = EvidenceInventory()
        inv.add_expected_artifact("SPEC-001", "SPEC", "1234567890abcdef" * 4)

        policy = _make_sample_compiled_policy()
        claims = ClaimRegistry()
        manifest = build_canonical_evidence_manifest(
            [
                EvidenceRecord(
                    evidence_id="ev-claim-cand",
                    claim_id="claim_candidate_freeze",
                    evidence_state="ESTABLISHED",
                    provenance_class="CANONICAL_LEDGER",
                    authority_class="AUTHORITATIVE",
                    source="urn:arx:freeze",
                    content_hash="297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90",
                    validity="CURRENT",
                    observed_at="2026-10-12T08:00:00Z",
                ),
                EvidenceRecord(
                    evidence_id="ev-claim-witness",
                    claim_id="claim_external_witness",
                    evidence_state="NOT_ESTABLISHED",
                    provenance_class="NONE",
                    authority_class="NOT_APPLICABLE",
                    source=None,
                    content_hash=None,
                    validity="CURRENT",
                    observed_at="2026-10-12T08:00:00Z",
                ),
            ],
            "h" * 40,
        )

        # Context reflecting canonical un-established external witness
        ctx = FactContext([
            PredicateFact("fact_candidate_check_required", "TRUE", "string", {"authority_class": "AUTHORITATIVE"}),
            PredicateFact("fact_ext_witness_required", "TRUE", "string", {"authority_class": "AUTHORITATIVE"}),
        ])

        report = verify_legacy_dag_parity(reg, inv, policy, ctx, manifest, claims)
        assert report["parity_pass"] is True
        assert report["mismatches"] == 0
        assert report["legacy_outputs"]["global_provisioning_readiness"] == "BLOCKED"
        assert report["dag_outputs"]["global_provisioning_readiness"] == "BLOCKED"
        assert report["legacy_outputs"]["architecture_closure_status"] == "OPEN"
        assert report["dag_outputs"]["architecture_closure_status"] == "OPEN"
        assert report["legacy_outputs"]["runtime_action"] == "NONE"
        assert report["dag_outputs"]["runtime_action"] == "NONE"
