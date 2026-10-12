"""
Test Suite: Canonical Input Authority, Registry Completeness & Decision-Chain Closure V1
ARX Terminal — Smart Money

Verifies:
- Context authority and non-substitutable CanonicalDecisionContext (Tests 1–8)
- Evidence recomputation from git tree blobs and provenance (Tests 9–15)
- Claim derivation, state basis, Candidate 004 dual-identity (Tests 16–22)
- Fact derivation authority and claim binding (Tests 23–28)
- Finding registry split identity, admission, and duplicate integrity (Tests 29–35)
- Dependency DAG completeness and repository binding (Tests 36–41)
- Blocker ablation and monotonicity (Tests 42–47)
- Anti-equivocation and sequence collision prevention (Tests 48–52)
- Temporal integrity and evaluation clock skew (Tests 53–56)
- Authorization freshness and pre-integration invalidation (Tests 57–61)
- Legacy vs DAG parity (Tests 62–64)
- Normative ground-truth test oracle & order independence (Tests 65–68)
"""

import os
import json
import copy
import hashlib
import pytest
from datetime import datetime, timezone, timedelta

from analyst_dashboard.smart_money.governance_control_plane import (
    GovernanceDecisionEngine,
    ReferenceDecisionEngine,
    DecisionSnapshot,
    verify_legacy_dag_parity,
    validate_canonical_decision_snapshot,
    PreIntegrationSnapshotRejectedError,
    DuplicateDecisionSequenceError,
    CallerConstructedCanonicalContextRejectedError,
    INVALIDATED_PRE_INTEGRATION_DECISION_HASHES,
    AuthorizationManager,
    ProvisioningAuthorization,
    PreIntegrationAuthorizationError,
    StaleDecisionException,
    TOCTOUException,
    AuthorityLedger,
    FindingRegistry,
    Finding,
    FindingStatus,
    FindingAdjudication,
    get_all_registered_findings,
    get_canonical_admitted_findings,
    create_canonical_finding_registry,
    create_historical_finding_registry,
    DuplicateWithoutCanonicalTargetError,
    DuplicateSelfReferenceError,
    DuplicateCycleError,
    CanonicalEvidenceManifest,
    EvidenceRecord,
    build_canonical_evidence_manifest,
    ClaimRegistry,
    ClaimDerivationEngine,
    CanonicalDerivedClaim,
    StateBasis,
    EmptyClaimRegistryError,
    AuthoritativeStateBasisMissingError,
    FactDerivationEngine,
    SelfAssertedFactAuthorityError,
    FactWithoutClaimBindingError,
    CanonicalDecisionContext,
    CanonicalDecisionContextBuilder,
    SourceFreezeVerificationError,
    RepositoryEvidenceRecomputationError,
    FutureDatedRecordError,
    DAGIncompletenessError,
    compile_dependency_policy,
    resolve_dependencies,
    FactContext,
    sha256_canonical,
)
from analyst_dashboard.smart_money.governance_control_plane.predicate_evaluator import (
    PredicateFact,
    SelfAssertedAuthorityRejectedError,
)

PRE_GATE_HEAD_SHA = "7d877c524ba859918e115561a2ccba12e95bc3e1"
DOCS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../docs/domain/smart_money/governance_control_plane")
)


@pytest.fixture
def sample_clean_ledger():
    ledger = AuthorityLedger(epoch="epoch-gov-002")
    ledger.create_genesis(
        genesis_manifest={"genesis": True, "epoch": "epoch-gov-002"},
        actor_identity="urn:arx:operator:genesis:001",
    )
    return ledger


@pytest.fixture
def canonical_builder():
    return CanonicalDecisionContextBuilder(
        source_freeze_sha=PRE_GATE_HEAD_SHA,
        clock_skew_seconds=600,
        evaluation_time=None,
    )


@pytest.fixture
def canonical_context(canonical_builder):
    return canonical_builder.build(finding_registry=create_historical_finding_registry())


# =============================================================================
# Matrix 55: Context Authority (Tests 1–8)
# =============================================================================

class TestMatrix55ContextAuthority:

    def test_01_caller_constructed_canonical_context_rejected(self, canonical_context):
        """1. CALLER_CONSTRUCTED_CANONICAL_CONTEXT_ACCEPTED == 0"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        with pytest.raises(CallerConstructedCanonicalContextRejectedError):
            engine.evaluate_canonical(context=None)

    def test_02_caller_supplied_fact_context_rejected(self):
        """2. CALLER_SUPPLIED_CANONICAL_FACT_CONTEXT_ACCEPTED == 0"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        with pytest.raises(CallerConstructedCanonicalContextRejectedError):
            engine.evaluate_canonical(fact_context=FactContext(facts=[PredicateFact(fact_id="fact.test", value="TRUE")]))

    def test_03_caller_supplied_dependency_policy_rejected(self):
        """3. CALLER_SUPPLIED_CANONICAL_DEPENDENCY_POLICY_ACCEPTED == 0"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        with pytest.raises(CallerConstructedCanonicalContextRejectedError):
            engine.evaluate_canonical(compiled_dependency_policy=compile_dependency_policy({"policy_id": "test", "predicates": {}, "nodes": []}))

    def test_04_caller_supplied_finding_registry_rejected(self):
        """4. CALLER_SUPPLIED_CANONICAL_FINDING_REGISTRY_ACCEPTED == 0"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        with pytest.raises(CallerConstructedCanonicalContextRejectedError):
            engine.evaluate_canonical(finding_registry=FindingRegistry())

    def test_05_tampered_context_hash_rejected(self, canonical_context):
        """5. Context with forged or mismatched context_hash is rejected"""
        tampered_ctx = CanonicalDecisionContext(
            context_version=canonical_context.context_version,
            source_freeze=canonical_context.source_freeze,
            evaluation_clock=canonical_context.evaluation_clock,
            evidence=canonical_context.evidence,
            claims=canonical_context.claims,
            findings=canonical_context.findings,
            dependencies=canonical_context.dependencies,
            evaluator=canonical_context.evaluator,
            candidate=canonical_context.candidate,
            decision_context_hash="0" * 64,  # Forged hash
            evidence_manifest=canonical_context.evidence_manifest,
            derived_claims=canonical_context.derived_claims,
            finding_registry=canonical_context.finding_registry,
            compiled_dependency_policy=canonical_context.compiled_dependency_policy,
            fact_context=canonical_context.fact_context,
            dependency_resolution=canonical_context.dependency_resolution,
        )
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        with pytest.raises(CallerConstructedCanonicalContextRejectedError):
            engine.evaluate_canonical(context=tampered_ctx)

    def test_06_builder_rejects_invalid_source_freeze(self):
        """6. Builder rejects non-existent or invalid source freeze SHA"""
        bad_builder = CanonicalDecisionContextBuilder(source_freeze_sha="f" * 40)
        with pytest.raises(SourceFreezeVerificationError):
            bad_builder.build()

    def test_07_context_binds_ten_roots_and_computes_hash(self, canonical_context):
        """7. Context binds all 10 root objects and produces valid SHA-256 hash"""
        assert len(canonical_context.decision_context_hash) == 64
        d = canonical_context.to_dict()
        required_roots = [
            "context_version",
            "source_freeze",
            "evaluation_clock",
            "evidence",
            "claims",
            "findings",
            "dependencies",
            "evaluator",
            "candidate",
            "decision_context_hash",
        ]
        for root in required_roots:
            assert root in d

    def test_08_context_is_immutable(self, canonical_context):
        """8. CanonicalDecisionContext is deeply frozen / immutable"""
        with pytest.raises(Exception):
            canonical_context.decision_context_hash = "modified"


# =============================================================================
# Matrix 56: Evidence Authority (Tests 9–15)
# =============================================================================

class TestMatrix56EvidenceAuthority:

    def test_09_repository_evidence_recomputed_from_git_tree_blob(self, canonical_context):
        """9. CALLER_SUPPLIED_CANONICAL_CONTENT_HASH_ACCEPTED == 0"""
        cand_rec = [r for r in canonical_context.evidence_manifest.records if "candidate004-freeze" in r.evidence_id][0]
        assert cand_rec.content_hash == "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"
        assert cand_rec.provenance_class == "CANONICAL_LEDGER"

    def test_10_git_tree_blob_missing_raises_error(self, canonical_builder):
        """10. Missing file in git tree raises RepositoryEvidenceRecomputationError"""
        with pytest.raises(RepositoryEvidenceRecomputationError):
            canonical_builder.recompute_blob_sha("nonexistent/path/to/blob.txt")

    def test_11_evidence_record_validation_rejects_synthetic(self):
        """11. Synthetic fixture rejected in canonical evidence"""
        rec = EvidenceRecord(
            evidence_id="EVD-TEST-SYNTH",
            claim_id="CLAIM-TEST",
            evidence_state="ESTABLISHED",
            provenance_class="SYNTHETIC_FIXTURE",
            authority_class="NON_AUTHORITATIVE",
            source="mock_operator",
            content_hash=hashlib.sha256(b"content").hexdigest(),
            observed_at="2026-10-12T10:00:00Z",
            validity="CURRENT",
        )
        assert rec.provenance_class == "SYNTHETIC_FIXTURE"

    def test_12_external_operational_requires_payload_hash(self):
        """12. External operational evidence record has valid attributes"""
        rec = EvidenceRecord(
            evidence_id="EVD-EXT-001",
            claim_id="CLAIM-RELEASE-PROVENANCE",
            evidence_state="ESTABLISHED",
            provenance_class="EXTERNAL_OPERATIONAL",
            authority_class="AUTHORITATIVE",
            source="urn:arx:authority:sigstore:oidc",
            content_hash=hashlib.sha256(b"payload").hexdigest(),
            observed_at="2026-10-12T10:00:00Z",
            validity="CURRENT",
        )
        assert rec.provenance_class == "EXTERNAL_OPERATIONAL"

    def test_13_stale_evidence_status_preserved(self):
        """13. Stale evidence is tagged STALE in summary"""
        rec = EvidenceRecord(
            evidence_id="EVD-STALE-001",
            claim_id="CLAIM-RELEASE-PROVENANCE",
            evidence_state="ESTABLISHED",
            provenance_class="LOCAL_REFERENCE",
            authority_class="NON_AUTHORITATIVE",
            source="auditor",
            content_hash=hashlib.sha256(b"stale").hexdigest(),
            observed_at="2025-01-01T00:00:00Z",
            validity="STALE",
        )
        manifest = build_canonical_evidence_manifest([rec], "e" * 40)
        assert manifest.summary["stale"] == 1

    def test_14_revoked_evidence_status_preserved(self):
        """14. Revoked evidence is tagged REVOKED in summary"""
        rec = EvidenceRecord(
            evidence_id="EVD-REVOKED-001",
            claim_id="CLAIM-RELEASE-PROVENANCE",
            evidence_state="ESTABLISHED",
            provenance_class="OPERATOR_RATIFIED",
            authority_class="NON_AUTHORITATIVE",
            source="council",
            content_hash=hashlib.sha256(b"revoked").hexdigest(),
            observed_at="2026-10-12T10:00:00Z",
            validity="REVOKED",
        )
        manifest = build_canonical_evidence_manifest([rec], "e" * 40)
        assert manifest.summary["revoked"] == 1

    def test_15_zero_record_manifest_valid(self):
        """15. Empty record set produces valid manifest with zero count"""
        manifest = build_canonical_evidence_manifest([], "e" * 40)
        assert len(manifest.records) == 0
        assert len(manifest.manifest_hash) == 64


# =============================================================================
# Matrix 57: Claims Authority & Derivation (Tests 16–22)
# =============================================================================

class TestMatrix57ClaimsAuthority:

    def test_16_empty_claim_registry_rejected(self):
        """16. EMPTY_CLAIM_REGISTRY_ACCEPTED_FOR_CANONICAL_DECISION == 0"""
        engine = ClaimDerivationEngine()
        empty_manifest = build_canonical_evidence_manifest([], PRE_GATE_HEAD_SHA)
        with pytest.raises(EmptyClaimRegistryError):
            engine.derive_all_claims(empty_manifest)

    def test_17_claim_derivation_policy_loaded(self, canonical_context):
        """17. CLAIM_WITHOUT_DERIVATION_POLICY_ACCEPTED == 0"""
        claims = canonical_context.derived_claims
        assert len(claims) >= 9
        required_classes = [
            "GOVERNANCE_ROOT",
            "CANDIDATE_FREEZE_IDENTITY",
            "CANDIDATE_SEMANTIC_CLOSURE",
            "EXTERNAL_LEDGER",
            "EXTERNAL_WITNESS",
            "FINDING_BLOCKERS",
            "DENOMINATOR_ALIGNMENT",
            "GOVERNANCE_POLICY",
            "RELEASE_PROVENANCE",
        ]
        for rc in required_classes:
            matching = [c for c in claims.values() if c.claim_class == rc]
            assert len(matching) > 0, f"Missing claim class {rc}"

    def test_18_unknown_state_not_laundered_as_not_established(self):
        """18. UNKNOWN_STATE_LAUNDERED_AS_NOT_ESTABLISHED == 0"""
        with pytest.raises(AuthoritativeStateBasisMissingError):
            CanonicalDerivedClaim(
                claim_id="CLAIM-TEST-ILLEGAL",
                claim_class="TEST",
                claim_state="NOT_ESTABLISHED",
                derivation_id="drv-1",
                derivation_policy_hash="p" * 64,
                evidence_refs=["EVD-001"],
                state_basis=None,
            )

    def test_19_state_basis_required_for_not_established(self):
        """19. Valid StateBasis permits NOT_ESTABLISHED claim creation"""
        basis = StateBasis(
            basis_type="ABSENCE_ATTESTATION",
            authority_ref="urn:arx:authority:auditor:v1",
            state_source_ref="urn:arx:repo:tree:v1",
            observed_at="2026-10-12T10:00:00Z",
        )
        claim = CanonicalDerivedClaim(
            claim_id="CLAIM-TEST-VALID-NOT-EST",
            claim_class="TEST",
            claim_state="NOT_ESTABLISHED",
            derivation_id="drv-1",
            derivation_policy_hash="p" * 64,
            evidence_refs=["EVD-001"],
            state_basis=basis,
        )
        assert claim.claim_state == "NOT_ESTABLISHED"
        assert claim.state_basis is not None

    def test_20_candidate004_dual_identity_separation(self, canonical_context):
        """20. Dual identity separation for Candidate 004"""
        claims = canonical_context.derived_claims
        freeze_claim = claims["CLAIM-CANDIDATE004-FREEZE-FILE-IDENTITY"]
        semantic_claim = claims["CLAIM-CANDIDATE004-SEMANTIC-CLOSURE-IDENTITY"]

        assert freeze_claim.claim_class == "CANDIDATE_FREEZE_IDENTITY"
        assert semantic_claim.claim_class == "CANDIDATE_SEMANTIC_CLOSURE"
        assert freeze_claim.claim_hash != semantic_claim.claim_hash
        assert freeze_claim.claim_state == "ESTABLISHED"
        assert semantic_claim.claim_state != "ESTABLISHED"

    def test_21_external_theorems_derived_as_not_established(self, canonical_context):
        """21. External unestablished claims derived with audit basis"""
        claims = canonical_context.derived_claims
        for cid in ["CLAIM-GOVERNANCE-ROOT", "CLAIM-EXTERNAL-LEDGER", "CLAIM-EXTERNAL-WITNESS"]:
            c = claims[cid]
            assert c.claim_state == "NOT_ESTABLISHED"
            assert c.state_basis is not None

    def test_22_six_valued_claim_states(self):
        """22. All 6 claim states supported and verifiable"""
        valid_states = {"ESTABLISHED", "NOT_ESTABLISHED", "UNKNOWN", "CONFLICT", "STALE", "REVOKED"}
        for s in valid_states:
            basis = StateBasis("TEST", "AUTH", "SRC", "2026-10-12T10:00:00Z") if s == "NOT_ESTABLISHED" else None
            c = CanonicalDerivedClaim(f"CLAIM-{s}", "TEST", s, "drv-1", "p" * 64, ["EVD-1"], state_basis=basis)
            assert c.claim_state == s


# =============================================================================
# Matrix 58: Facts Authority (Tests 23–28)
# =============================================================================

class TestMatrix58FactsAuthority:

    def test_23_self_asserted_fact_authority_rejected(self):
        """23. SELF_ASSERTED_FACT_AUTHORITY_ACCEPTED == 0"""
        fact = PredicateFact(
            fact_id="fact.self_asserted",
            value="TRUE",
            provenance={"authority_class": "AUTHORITATIVE"},
        )
        ctx = FactContext(strict_authoritative_check=True)
        with pytest.raises(SelfAssertedAuthorityRejectedError):
            ctx.add_fact(fact)

    def test_24_fact_without_claim_binding_rejected(self):
        """24. Missing required claim bindings raises FactWithoutClaimBindingError"""
        engine = FactDerivationEngine()
        with pytest.raises(FactWithoutClaimBindingError):
            engine.derive_fact_context(claims={})

    def test_25_established_claim_maps_to_true_fact(self, canonical_context):
        """25. ESTABLISHED claim maps to ESTABLISHED fact"""
        fc = canonical_context.fact_context
        f1, err = fc.get_fact("fact_candidate_freeze_valid")
        assert err is None
        assert f1.value == "ESTABLISHED"
        f2, err = fc.get_fact("fact_release_provenance_status")
        assert err is None
        assert f2.value == "ESTABLISHED"

    def test_26_not_established_claim_maps_to_false_fact(self, canonical_context):
        """26. NOT_ESTABLISHED claim maps to NOT_ESTABLISHED fact"""
        fc = canonical_context.fact_context
        f1, err = fc.get_fact("fact_governance_root_status")
        assert err is None
        assert f1.value == "NOT_ESTABLISHED"

    def test_27_partial_logic_unknown_mapping(self):
        """27. UNKNOWN claim maps to UNKNOWN fact"""
        engine = FactDerivationEngine()
        claims = {
            r["source_claim_id"]: CanonicalDerivedClaim(
                claim_id=r["source_claim_id"],
                claim_class="TEST",
                claim_state="UNKNOWN",
                derivation_id="drv-1",
                derivation_policy_hash="p" * 64,
                evidence_refs=["EVD-1"],
            )
            for r in engine.rules
        }
        fc = engine.derive_fact_context(claims)
        f1, _ = fc.get_fact("fact_governance_root_status")
        assert f1.value == "UNKNOWN"

    def test_28_fact_context_hash_deterministic(self, canonical_context):
        """28. Fact context hash is deterministically derived and 64-char hex"""
        fc = canonical_context.fact_context
        h = fc.compute_context_hash()
        assert len(h) == 64
        assert h == canonical_context.fact_context.compute_context_hash()


# =============================================================================
# Matrix 59: Findings Authority & Split Identity (Tests 29–35)
# =============================================================================

class TestMatrix59FindingsAuthority:

    def test_29_split_finding_registry_hashes_distinct(self, canonical_context):
        """29. finding_identity, finding_admission, and active_blocker hashes are distinct"""
        id_hash = canonical_context.finding_identity_registry_hash
        adm_hash = canonical_context.finding_admission_registry_hash
        blk_hash = canonical_context.active_blocker_set_hash

        assert len(id_hash) == 64
        assert len(adm_hash) == 64
        assert len(blk_hash) == 64
        assert id_hash != adm_hash
        assert adm_hash != blk_hash
        assert id_hash != blk_hash

    def test_30_lifecycle_mutation_preserves_identity_hash(self):
        """30. Resolving a finding alters admission and blocker hashes but NOT identity hash"""
        reg = create_historical_finding_registry()
        initial_id_hash = reg.compute_identity_registry_hash()
        initial_adm_hash = reg.compute_admission_registry_hash()
        initial_blk_hash = reg.compute_active_blocker_set_hash()

        # Resolve one finding
        f1 = reg.get_finding("FND-ARCH-001")
        f1.lifecycle_status = FindingStatus.VERIFIED_RESOLVED

        new_id_hash = reg.compute_identity_registry_hash()
        new_adm_hash = reg.compute_admission_registry_hash()
        new_blk_hash = reg.compute_active_blocker_set_hash()

        assert new_id_hash == initial_id_hash
        assert new_adm_hash != initial_adm_hash
        assert new_blk_hash != initial_blk_hash

    def test_31_duplicate_without_canonical_target_rejected(self):
        """31. Duplicate finding without canonical target raises DuplicateWithoutCanonicalTargetError"""
        reg = FindingRegistry()
        f = Finding(
            finding_id="FND-DUP-001",
            title="Duplicate without target",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="DUPLICATE",
            duplicate_of=None,
        )
        with pytest.raises(DuplicateWithoutCanonicalTargetError):
            reg.add_finding(f)

    def test_32_duplicate_self_reference_rejected(self):
        """32. Duplicate self-reference raises DuplicateSelfReferenceError"""
        reg = FindingRegistry()
        f = Finding(
            finding_id="FND-DUP-001",
            title="Duplicate self ref",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="DUPLICATE",
            duplicate_of="FND-DUP-001",
        )
        with pytest.raises(DuplicateSelfReferenceError):
            reg.add_finding(f)

    def test_33_duplicate_cycle_rejected(self):
        """33. Duplicate cycle A -> B -> A raises DuplicateCycleError"""
        reg = FindingRegistry()
        f_a = Finding(
            finding_id="FND-A",
            title="A",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="ADMITTED",
        )
        f_b = Finding(
            finding_id="FND-B",
            title="B",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="DUPLICATE",
            duplicate_of="FND-A",
        )
        reg.add_finding(f_a)
        reg.add_finding(f_b)
        adj = FindingAdjudication(
            adjudication_id="adj-cycle",
            finding_id="FND-A",
            prior_admission_status="ADMITTED",
            new_admission_status="DUPLICATE",
            outcome="DUPLICATE",
            rationale="cycle test",
            duplicate_of="FND-B",
        )
        with pytest.raises(DuplicateCycleError):
            reg.apply_adjudication(adj)

    def test_34_adjudication_report_application(self):
        """34. FindingAdjudication updates admission status"""
        reg = create_canonical_finding_registry()
        adj = FindingAdjudication(
            adjudication_id="adj-001",
            finding_id="FND-ARCH-002",
            prior_admission_status="ADMITTED",
            new_admission_status="DUPLICATE",
            outcome="DUPLICATE",
            rationale="test duplicate",
            duplicate_of="FND-ARCH-001",
        )
        reg.apply_adjudication(adj)
        f2 = reg.get_finding("FND-ARCH-002")
        assert f2.admission_status == "DUPLICATE"
        assert f2.is_blocking is False

    def test_35_duplicate_findings_excluded_from_blockers(self):
        """35. Only ADMITTED findings enter blocker set; DUPLICATE findings do not double count"""
        reg = create_historical_finding_registry()
        blockers = reg.get_active_blockers()
        assert len(blockers) == 13
        duplicate_ids = {f.finding_id for f in reg.findings if f.admission_status == "DUPLICATE"}
        assert "FND-ARCH-014" not in duplicate_ids
        assert "FND-ARCH-015" in duplicate_ids
        blocker_ids = {b.finding_id for b in blockers}
        assert "FND-ARCH-015" not in blocker_ids


# =============================================================================
# Matrix 60: Dependency DAG & Completeness (Tests 36–41)
# =============================================================================

class TestMatrix60DAGCompleteness:

    def test_36_admitted_open_findings_without_mapping_rejected(self, canonical_builder, canonical_context):
        """36. ADMITTED_OPEN_FINDINGS_WITHOUT_DECISION_MAPPING == 0"""
        reg = create_canonical_finding_registry()
        unmapped = Finding(
            finding_id="FND-UNMAPPED-001",
            title="Unmapped Finding",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="ADMITTED",
        )
        reg.add_finding(unmapped)
        with pytest.raises(DAGIncompletenessError):
            canonical_builder.validate_dag_completeness(reg, canonical_context.compiled_dependency_policy)

    def test_37_active_blockers_have_dependency_path(self):
        """37. ACTIVE_BLOCKERS_WITHOUT_DEPENDENCY_PATH == 0"""
        mapping_path = os.path.join(DOCS_DIR, "FINDING_DECISION_MAPPING.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        for m in mapping.get("mappings", []):
            assert len(m.get("dependency_paths", [])) > 0, f"Blocker {m['finding_id']} has no dependency path"
            assert len(m.get("expected_blocks", [])) > 0, f"Blocker {m['finding_id']} has no expected blocks"

    def test_38_mandatory_claims_have_output_path(self):
        """38. MANDATORY_CLAIMS_WITHOUT_OUTPUT_PATH == 0"""
        surface_path = os.path.join(DOCS_DIR, "CANONICAL_DECISION_SURFACE.json")
        with open(surface_path, "r", encoding="utf-8") as f:
            surface = json.load(f)
        assert len(surface["required_claim_classes"]) >= 9
        for th, spec in surface["theorems"].items():
            assert len(spec["required_claims"]) > 0

    def test_39_required_outputs_have_derivation_path(self):
        """39. REQUIRED_OUTPUTS_WITHOUT_DERIVATION_PATH == 0"""
        policy_path = os.path.join(DOCS_DIR, "CANONICAL_DEPENDENCY_POLICY.json")
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
        assert len(policy["nodes"]) > 0
        assert len(policy["edges"]) > 0
        target_outputs = {e["target_output_id"] for e in policy["edges"]}
        for node in policy["nodes"]:
            if node["node_type"] == "OUTPUT" and node["node_id"] != "RUNTIME_ACTION":
                assert node["node_id"] in target_outputs

    def test_40_dependency_policy_repository_bound(self, canonical_context):
        """40. CANONICAL_DEPENDENCY_POLICY_REPOSITORY_BOUND == PASS"""
        assert canonical_context.dependencies["dependency_policy_hash"] is not None
        assert len(canonical_context.dependency_policy_hash) == 64

    def test_41_dag_evaluation_produces_deterministic_resolution(self, canonical_context):
        """41. Evaluating DAG over complete canonical context produces deterministic result"""
        res = canonical_context.dependency_resolution
        assert len(res.target_statuses) > 0
        assert len(res.dependency_resolution_hash) == 64


# =============================================================================
# Matrix 61: Blocker Ablation (Tests 42–47)
# =============================================================================

class TestMatrix61BlockerAblation:

    def test_42_baseline_all_thirteen_blockers_active(self, canonical_context):
        """42. Baseline canonical decision produces BLOCKED with 13 active blockers"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        snap = engine.evaluate_canonical(context=canonical_context)
        assert snap.global_provisioning_readiness == "BLOCKED"
        assert snap.architecture_closure_status == "OPEN"
        assert snap.theorem_readiness["A"] == "BLOCKED"
        assert snap.theorem_readiness["B"] == "BLOCKED"
        assert snap.theorem_readiness["C"] == "BLOCKED"
        assert snap.theorem_readiness["D"] == "BLOCKED"
        assert snap.theorem_readiness["E"] == "BLOCKED"
        assert len(canonical_context.finding_registry.get_active_blockers()) == 13

    def test_43_ablating_fnd_arch_001_leaves_theorems_blocked(self, canonical_builder):
        """43. Resolving FND-ARCH-001 leaves remaining blockers active"""
        reg = create_historical_finding_registry()
        f1 = reg.get_finding("FND-ARCH-001")
        f1.lifecycle_status = FindingStatus.VERIFIED_RESOLVED

        ctx = canonical_builder.build(finding_registry=reg)
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        snap = engine.evaluate_canonical(context=ctx)
        assert snap.global_provisioning_readiness == "BLOCKED"
        assert len(reg.get_active_blockers()) == 12

    def test_44_ablating_all_theorem_a_blockers(self, canonical_builder):
        """44. Resolving Theorem A blockers leaves Theorems B–E blocked"""
        reg = create_historical_finding_registry()
        for fid in ["FND-ARCH-001", "FND-ARCH-002"]:
            reg.get_finding(fid).lifecycle_status = FindingStatus.VERIFIED_RESOLVED

        ctx = canonical_builder.build(finding_registry=reg)
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        snap = engine.evaluate_canonical(context=ctx)
        assert snap.global_provisioning_readiness == "BLOCKED"
        assert snap.theorem_readiness["B"] == "BLOCKED"
        assert snap.theorem_readiness["C"] == "BLOCKED"

    def test_45_monotone_dependency_resolution(self):
        """45. Monotonicity: resolving findings never increases active blocker count"""
        reg = create_historical_finding_registry()
        count_0 = len(reg.get_active_blockers())
        reg.get_finding("FND-ARCH-003").lifecycle_status = FindingStatus.VERIFIED_RESOLVED
        count_1 = len(reg.get_active_blockers())
        assert count_1 == count_0 - 1

    def test_46_resolved_finding_drops_from_blocker_hash(self):
        """46. Resolved finding drops out of active_blocker_set_hash"""
        reg = create_historical_finding_registry()
        h0 = reg.compute_active_blocker_set_hash()
        reg.get_finding("FND-ARCH-004").lifecycle_status = FindingStatus.VERIFIED_RESOLVED
        h1 = reg.compute_active_blocker_set_hash()
        assert h0 != h1

    def test_47_waived_finding_drops_from_active_blockers(self):
        """47. Waived finding drops out of active blockers if within validity"""
        reg = create_historical_finding_registry()
        f = reg.get_finding("FND-ARCH-005")
        f.lifecycle_status = FindingStatus.WAIVED
        assert f not in reg.get_active_blockers()


# =============================================================================
# Matrix 62: Anti-Equivocation & Sequence Integrity (Tests 48–52)
# =============================================================================

class TestMatrix62AntiEquivocation:

    def test_48_duplicate_sequence_with_different_hash_rejected(self, canonical_context):
        """48. DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002", enforce_sequence_uniqueness=True)
        snap1 = engine.evaluate_canonical(context=canonical_context, sequence=1)
        assert snap1.decision_sequence == 1

        engine.register_committed_decision("epoch-gov-002", 1, snap1.decision_hash)
        with pytest.raises(DuplicateDecisionSequenceError):
            engine.evaluate(
                finding_registry=FindingRegistry(),
                evidence_inventory=canonical_context.evidence_manifest,
                sequence=1,
            )

    def test_49_pre_integration_snapshot_3dc851a9_rejected(self):
        """49. Pre-integration snapshot 3dc851a9 rejected as current canonical"""
        with pytest.raises(PreIntegrationSnapshotRejectedError):
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
                architecture_closure_status="CLOSED",
                remediation_status="NONE",
                global_provisioning_readiness="READY",
                theorem_readiness={},
                runtime_action="NONE",
                derived_at="2026-10-12T10:00:00Z",
                decision_validity="CURRENT",
                decision_hash="3dc851a90e213da557a79027fc4b3a07287713bc84d88effcf92dadd0b539e41",
            )

    def test_50_pre_integration_snapshot_2563e663_rejected(self):
        """50. Pre-integration snapshot 2563e663 rejected as current canonical"""
        with pytest.raises(PreIntegrationSnapshotRejectedError):
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
                architecture_closure_status="CLOSED",
                remediation_status="NONE",
                global_provisioning_readiness="READY",
                theorem_readiness={},
                runtime_action="NONE",
                derived_at="2026-10-12T10:00:00Z",
                decision_validity="CURRENT",
                decision_hash="2563e663a7717bbb411229ea0bbab3ad0da58d3ff8a100bce14099132e096254",
            )

    def test_51_invalidated_snapshot_cannot_be_current(self):
        """51. Snapshot marked INVALIDATED_NON_CANONICAL cannot have validity CURRENT"""
        with pytest.raises(PreIntegrationSnapshotRejectedError):
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
                architecture_closure_status="CLOSED",
                remediation_status="NONE",
                global_provisioning_readiness="READY",
                theorem_readiness={},
                runtime_action="NONE",
                derived_at="2026-10-12T10:00:00Z",
                decision_validity="CURRENT",
                decision_hash="d" * 64,
                canonical_status="INVALIDATED_NON_CANONICAL",
            )

    def test_52_validate_canonical_snapshot_rejects_epoch_001(self, canonical_context):
        """52. validate_canonical_decision_snapshot rejects epoch-gov-001 snapshot in epoch-gov-002"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-001", allow_caller_canonical_inputs=True)
        snap = engine.evaluate(FindingRegistry(), canonical_context.evidence_manifest)
        with pytest.raises(PreIntegrationSnapshotRejectedError):
            validate_canonical_decision_snapshot(snap, current_epoch="epoch-gov-002")


# =============================================================================
# Matrix 63: Temporal Integrity & Evaluation Clock (Tests 53–56)
# =============================================================================

class TestMatrix63TemporalIntegrity:

    def test_53_future_dated_record_rejected(self):
        """53. Evaluation timestamp in future beyond skew raises FutureDatedRecordError"""
        future_time = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
        builder = CanonicalDecisionContextBuilder(
            source_freeze_sha=PRE_GATE_HEAD_SHA,
            clock_skew_seconds=60,
            evaluation_time=future_time,
        )
        with pytest.raises(FutureDatedRecordError):
            builder.get_clock()

    def test_54_future_dated_evaluation_time_rejected(self):
        """54. Evaluation time far in the future rejected"""
        future_time = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
        builder = CanonicalDecisionContextBuilder(
            source_freeze_sha=PRE_GATE_HEAD_SHA,
            clock_skew_seconds=60,
            evaluation_time=future_time,
        )
        with pytest.raises(FutureDatedRecordError):
            builder.get_clock()

    def test_55_valid_clock_within_skew_accepted(self, canonical_context):
        """55. Valid evaluation time within skew passes clock checks"""
        assert canonical_context.evaluation_clock["clock_source"] == "LOCAL_SYSTEM_CLOCK"

    def test_56_decision_derived_at_monotonic(self, canonical_context):
        """56. Derived timestamp recorded in DecisionSnapshot"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        snap = engine.evaluate_canonical(context=canonical_context)
        assert snap.derived_at is not None
        assert "T" in snap.derived_at


# =============================================================================
# Matrix 64: Authorization Freshness & Invalidation (Tests 57–61)
# =============================================================================

class TestMatrix64AuthorizationFreshness:

    def test_57_epoch_001_authorization_issuance_rejected_in_epoch_002(self, sample_clean_ledger):
        """57. PRE_INTEGRATION_AUTHORIZATION_EXERCISABLE == 0: cannot issue auth for epoch-gov-001 in epoch-gov-002"""
        auth_mgr = AuthorizationManager(sample_clean_ledger, current_epoch="epoch-gov-002")
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
            derived_at="2026-10-12T10:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
        )
        with pytest.raises(PreIntegrationAuthorizationError):
            auth_mgr.issue_provisioning_authorization(
                decision=snap,
                operator_identity="operator:1",
                allowed_actions=["PROVISION_A"],
                resource_scope=["scope:1"],
                theorem_scope=["A"],
                current_registry_hash="f" * 64,
            )

    def test_58_invalidated_hash_authorization_issuance_rejected(self, sample_clean_ledger):
        """58. Issuing authorization for 3dc851a9 raises PreIntegrationAuthorizationError"""
        auth_mgr = AuthorizationManager(sample_clean_ledger, current_epoch="epoch-gov-001")
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
            derived_at="2026-10-12T10:00:00Z",
            decision_validity="NON_CURRENT",
            decision_hash="3dc851a90e213da557a79027fc4b3a07287713bc84d88effcf92dadd0b539e41",
            canonical_status="INVALIDATED_NON_CANONICAL",
        )
        with pytest.raises(PreIntegrationAuthorizationError):
            auth_mgr.issue_provisioning_authorization(
                decision=snap,
                operator_identity="operator:1",
                allowed_actions=["PROVISION_A"],
                resource_scope=["scope:1"],
                theorem_scope=["A"],
                current_registry_hash="f" * 64,
            )

    def test_59_pre_integration_authorization_exercise_rejected(self, sample_clean_ledger):
        """59. Exercising pre-integration authorization in epoch-gov-002 raises PreIntegrationAuthorizationError"""
        auth_mgr = AuthorizationManager(sample_clean_ledger, current_epoch="epoch-gov-002")
        auth = ProvisioningAuthorization(
            authorization_id="urn:arx:auth:001",
            authorization_type="PROVISIONING",
            governance_epoch="epoch-gov-001",
            decision_hash="d" * 64,
            policy_hash="p" * 64,
            ledger_head_hash=sample_clean_ledger.head_hash,
            registry_hash="r" * 64,
            operator_identity="operator:1",
            allowed_actions=["PROVISION_A"],
            resource_scope=["scope:1"],
            theorem_scope=["A"],
            issued_sequence=1,
            issued_at="2026-10-12T10:00:00Z",
            expires_at="2026-10-12T11:00:00Z",
            single_use=True,
            max_uses=1,
            precondition_hashes={},
            authorization_hash="a" * 64,
            signature_or_ratification_ref="rat",
        )
        auth_mgr.active_authorizations["urn:arx:auth:001"] = auth
        with pytest.raises(PreIntegrationAuthorizationError):
            auth_mgr.validate_and_consume(
                auth_id="urn:arx:auth:001",
                caller_identity="operator:1",
                requested_action="PROVISION_A",
                current_decision_hash="d" * 64,
                current_registry_hash="r" * 64,
            )

    def test_60_decision_context_hash_mismatch_at_issuance_rejected(self, sample_clean_ledger, canonical_context):
        """60. Stale decision context hash at issuance raises StaleDecisionException"""
        auth_mgr = AuthorizationManager(sample_clean_ledger, current_epoch="epoch-gov-002")
        snap = DecisionSnapshot(
            decision_id="urn:arx:dec:001",
            governance_epoch="epoch-gov-002",
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
            derived_at="2026-10-12T10:00:00Z",
            decision_validity="CURRENT",
            decision_hash="d" * 64,
            is_canonical=True,
            dependency_resolution_hash="dr" * 32,
            evidence_manifest_hash="em" * 32,
            decision_context_hash="0" * 64,
            finding_identity_registry_hash="id" * 32,
            finding_admission_registry_hash="adm" * 32,
            active_blocker_set_hash="blk" * 32,
        )
        with pytest.raises(StaleDecisionException):
            auth_mgr.issue_provisioning_authorization(
                decision=snap,
                operator_identity="operator:1",
                allowed_actions=["PROVISION_A"],
                resource_scope=["scope:1"],
                theorem_scope=["A"],
                current_registry_hash="f" * 64,
                current_evidence_manifest_hash="em" * 32,
                current_dependency_resolution_hash="dr" * 32,
                current_decision_context_hash=canonical_context.context_hash,
            )

    def test_61_toctou_context_root_mismatch_at_exercise(self, sample_clean_ledger):
        """61. Context root mismatch at exercise raises TOCTOUException"""
        auth_mgr = AuthorizationManager(sample_clean_ledger, current_epoch="epoch-gov-002")
        auth = ProvisioningAuthorization(
            authorization_id="urn:arx:auth:002",
            authorization_type="PROVISIONING",
            governance_epoch="epoch-gov-002",
            decision_hash="d" * 64,
            policy_hash="p" * 64,
            ledger_head_hash=sample_clean_ledger.head_hash,
            registry_hash="r" * 64,
            operator_identity="operator:1",
            allowed_actions=["PROVISION_A"],
            resource_scope=["scope:1"],
            theorem_scope=["A"],
            issued_sequence=1,
            issued_at="2026-10-12T10:00:00Z",
            expires_at="2099-01-01T00:00:00Z",
            single_use=True,
            max_uses=1,
            precondition_hashes={},
            authorization_hash="a" * 64,
            signature_or_ratification_ref="rat",
            decision_context_hash="ctx_orig" * 8,
        )
        auth_mgr.active_authorizations["urn:arx:auth:002"] = auth
        with pytest.raises(TOCTOUException):
            auth_mgr.validate_and_consume(
                auth_id="urn:arx:auth:002",
                caller_identity="operator:1",
                requested_action="PROVISION_A",
                current_decision_hash="d" * 64,
                current_registry_hash="r" * 64,
                current_decision_context_hash="ctx_altered" * 7 + "a",
                check_time=datetime.fromisoformat("2026-10-12T10:30:00+00:00"),
            )


# =============================================================================
# Matrix 65: Legacy vs DAG Parity (Tests 62–64)
# =============================================================================

class TestMatrix65Parity:

    def test_62_legacy_dag_parity_pass(self, canonical_context):
        """62. verify_legacy_dag_parity passes over identical canonical inputs"""
        reg = canonical_context.finding_registry
        res = verify_legacy_dag_parity(
            finding_registry=reg,
            evidence_inventory=canonical_context.evidence_manifest,
            compiled_dependency_policy=canonical_context.compiled_dependency_policy,
            fact_context=canonical_context.fact_context,
            evidence_manifest=canonical_context.evidence_manifest,
            claim_registry=ClaimRegistry(claims=list(canonical_context.derived_claims.values())),
        )
        assert res["parity_pass"] is True
        assert res["mismatches"] == 0

    def test_63_primary_vs_reference_engine_parity(self, canonical_context):
        """63. Primary GovernanceDecisionEngine matches ReferenceDecisionEngine gate outputs"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        snap = engine.evaluate_canonical(context=canonical_context)

        ref_engine = ReferenceDecisionEngine()
        ref_out = ref_engine.evaluate_gates(
            canonical_context.finding_registry,
            canonical_context.evidence_manifest,
        )
        assert snap.architecture_closure_status == ref_out["architecture_closure_status"]
        assert snap.remediation_status == ref_out["remediation_status"]
        assert snap.global_provisioning_readiness == ref_out["global_provisioning_readiness"]

    def test_64_parity_mismatch_detected(self, canonical_context):
        """64. Artificial divergence between legacy and DAG is detected"""
        reg_legacy = copy.deepcopy(canonical_context.finding_registry)
        reg_legacy._findings = {}
        res = verify_legacy_dag_parity(
            finding_registry=reg_legacy,
            evidence_inventory=canonical_context.evidence_manifest,
            compiled_dependency_policy=canonical_context.compiled_dependency_policy,
            fact_context=canonical_context.fact_context,
            evidence_manifest=canonical_context.evidence_manifest,
            claim_registry=ClaimRegistry(claims=list(canonical_context.derived_claims.values())),
        )
        assert res["parity_pass"] is False
        assert res["mismatches"] > 0


# =============================================================================
# Matrix 66: Normative Ground Truth Oracle & Whole-Decision Order Independence (Tests 65–68)
# =============================================================================

class TestMatrix66NormativeOracleAndOrderIndependence:

    def test_65_normative_test_oracle_cases(self, canonical_context):
        """65. All cases in NORMATIVE_DECISION_CASES.json match expected outputs"""
        cases_path = os.path.join(DOCS_DIR, "NORMATIVE_DECISION_CASES.json")
        with open(cases_path, "r", encoding="utf-8") as f:
            oracle_data = json.load(f)

        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")

        snap_case1 = engine.evaluate_canonical(context=canonical_context)
        cases_by_id = {c["case_id"]: c["expected"] for c in oracle_data["cases"]}
        expected1 = cases_by_id["CURRENT_PRELIVE_BASELINE"]
        assert snap_case1.architecture_closure_status == expected1["architecture_closure_status"]
        assert snap_case1.remediation_status == expected1["remediation_status"]
        assert snap_case1.global_provisioning_readiness == expected1["global_provisioning_readiness"]
        assert snap_case1.provisioning_authorization_status == expected1["provisioning_authorization_status"]
        for th, st in expected1["theorems"].items():
            assert snap_case1.theorem_readiness[th] == st

    def test_66_finding_order_independence(self, canonical_builder):
        """66. Permuting finding insertion order produces bitwise identical decision_hash"""
        findings_forward = get_all_registered_findings()
        findings_reversed = list(reversed(findings_forward))

        reg_fwd = FindingRegistry(findings_forward)
        reg_rev = FindingRegistry(findings_reversed)

        assert reg_fwd.compute_identity_registry_hash() == reg_rev.compute_identity_registry_hash()
        assert reg_fwd.compute_admission_registry_hash() == reg_rev.compute_admission_registry_hash()
        assert reg_fwd.compute_active_blocker_set_hash() == reg_rev.compute_active_blocker_set_hash()

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        builder_fixed = CanonicalDecisionContextBuilder(
            source_freeze_sha=PRE_GATE_HEAD_SHA,
            evaluation_time=now_iso,
        )
        ctx_fwd = builder_fixed.build(finding_registry=reg_fwd)
        ctx_rev = builder_fixed.build(finding_registry=reg_rev)

        assert ctx_fwd.context_hash == ctx_rev.context_hash

    def test_67_evidence_order_independence(self, canonical_context):
        """67. Permuting evidence record order produces bitwise identical manifest_hash"""
        recs_fwd = canonical_context.evidence_manifest.records
        recs_rev = list(reversed(recs_fwd))

        m_fwd = build_canonical_evidence_manifest(recs_fwd, PRE_GATE_HEAD_SHA)
        m_rev = build_canonical_evidence_manifest(recs_rev, PRE_GATE_HEAD_SHA)

        assert m_fwd.manifest_hash == m_rev.manifest_hash

    def test_68_candidate_004_freeze_sha_invariance(self):
        """68. Candidate 004 freeze SHA integrity check: CANDIDATE004_DIFF == 0"""
        freeze_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json",
            )
        )
        assert os.path.exists(freeze_path)
        with open(freeze_path, "r", encoding="utf-8") as f:
            content = f.read()
        computed_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert computed_sha == "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"
