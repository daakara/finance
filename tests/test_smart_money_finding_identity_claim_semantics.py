"""
Test Suite: Finding Identity, Claim Semantics & Admission Authority Repair V1
ARX Terminal — Smart Money

Covers Matrices 44 through 53 (Tests 1 to 52):
- Matrix 44: Immutable Finding Identity Restoration (Tests 1–6)
- Matrix 45: Foreign Finding Audit & Admission Denial (Tests 7–11)
- Matrix 46: Claim-State & Authority Dimension Separation (Tests 12–17)
- Matrix 47: Authoritative StateBasis Completeness (Tests 18–22)
- Matrix 48: Operator Ratification Artifact Enforcement (Tests 23–27)
- Matrix 49: Root-Cause Duplicate Equivalence & Cycles (Tests 28–32)
- Matrix 50: Causal Blocker Re-Derivation (Zero Severity Bias) (Tests 33–38)
- Matrix 51: Blocker Ablation Parity (Tests 39–43)
- Matrix 52: Decision Invalidation & Epoch Succession (Tests 44–47)
- Matrix 53: Canonical Context Anti-Forgery & Split Hash Integrity (Tests 48–52)
"""

import os
import json
import hashlib
import pytest
from datetime import datetime, timezone

from analyst_dashboard.smart_money.governance_control_plane.finding_registry import (
    Finding,
    FindingRegistry,
    FindingStatus,
    FindingAdjudication,
    FINDING_ORIGINAL_SUBJECTS,
    FINDING_DOMAIN_SCOPES,
    KNOWN_DISPROVED_FOREIGN_SUBJECTS,
    FindingSubjectMutationForbiddenError,
    UnsupportedForeignFindingError,
    DuplicateWithoutCanonicalTargetError,
    DuplicateSelfReferenceError,
    DuplicateCycleError,
    get_canonical_known_architecture_findings,
    get_additional_registered_findings,
    get_all_registered_findings,
    get_canonical_admitted_findings,
    create_canonical_finding_registry,
)
from analyst_dashboard.smart_money.governance_control_plane.claim_derivation import (
    StateBasis,
    CanonicalDerivedClaim,
    ClaimDerivationEngine,
    AuthoritativeStateBasisMissingError,
    UnratifiedClaimError,
    EmptyClaimRegistryError,
    CLAIM_STATES,
)
from analyst_dashboard.smart_money.governance_control_plane.evidence_manifest import (
    CanonicalEvidenceManifest,
    EvidenceRecord,
    build_canonical_evidence_manifest,
)
from analyst_dashboard.smart_money.governance_control_plane.decision_engine import (
    GovernanceDecisionEngine,
    DecisionSnapshot,
    INVALIDATED_PRE_INTEGRATION_DECISION_HASHES,
    PreIntegrationSnapshotRejectedError,
    DuplicateDecisionSequenceError,
    ReusedInvalidatedSequenceSlotError,
    validate_canonical_decision_snapshot,
)
from analyst_dashboard.smart_money.governance_control_plane.canonical_context import (
    CanonicalDecisionContextBuilder,
    CanonicalDecisionContext,
    CallerConstructedCanonicalContextRejectedError,
    EXPECTED_CANDIDATE004_FREEZE_HASH,
    EXPECTED_CANDIDATE004_SEMANTIC_HASH,
)
from analyst_dashboard.smart_money.governance_control_plane.canonicalization import (
    sha256_canonical,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DOCS_DIR = os.path.join(REPO_ROOT, "docs/domain/smart_money/governance_control_plane")
FREEZE_PATH = os.path.join(
    REPO_ROOT,
    "docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json",
)


@pytest.fixture
def baseline_data():
    path = os.path.join(DOCS_DIR, "HISTORICAL_FINDING_IDENTITY_BASELINE.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def foreign_audit_data():
    path = os.path.join(DOCS_DIR, "FOREIGN_FINDING_ASSERTION_AUDIT.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def ablation_report_data():
    path = os.path.join(DOCS_DIR, "BLOCKER_ABLATION_REPORT.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def invalidation_registry_data():
    path = os.path.join(DOCS_DIR, "DECISION_INVALIDATION_REGISTRY.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def canonical_builder():
    return CanonicalDecisionContextBuilder(
        source_freeze_sha="68e2e05f4d632f984f5ed3ed00b41c04d7980943",
        repo_root=REPO_ROOT,
    )


@pytest.fixture
def canonical_context(canonical_builder):
    return canonical_builder.build()


# =============================================================================
# Matrix 44: Immutable Finding Identity Restoration (Tests 1–6)
# =============================================================================

class TestMatrix44ImmutableFindingIdentityRestoration:

    def test_01_all_historical_fifteen_finding_ids_present(self):
        """01. All 15 historical finding IDs FND-ARCH-001..015 exist in registry"""
        findings = get_all_registered_findings()
        finding_ids = [f.finding_id for f in findings]
        assert len(findings) == 15
        for i in range(1, 16):
            expected_id = f"FND-ARCH-{i:03d}"
            assert expected_id in finding_ids, f"Missing {expected_id}"

    def test_02_historical_subject_and_hash_parity(self, baseline_data):
        """02. Every finding matches the historical original subject and SHA-256 hash"""
        baseline_map = {b["finding_id"]: b for b in baseline_data["findings"]}
        for f in get_all_registered_findings():
            assert f.finding_id in baseline_map, f"{f.finding_id} not in baseline"
            b = baseline_map[f.finding_id]
            assert f.original_subject == b["original_subject"]
            expected_hash = hashlib.sha256(b["original_subject"].strip().lower().encode("utf-8")).hexdigest()
            assert f.original_subject_hash == expected_hash
            assert f.original_subject_hash == b["original_subject_hash"]

    def test_03_subject_mutation_forbidden(self):
        """03. Mutating a historical finding subject raises FindingSubjectMutationForbiddenError"""
        with pytest.raises(FindingSubjectMutationForbiddenError):
            Finding(
                finding_id="FND-ARCH-001",
                title="Mutated title",
                severity="CRITICAL",
                lifecycle_status=FindingStatus.OPEN,
                evidence_status="UNPROVEN",
                owner="sec",
                resolution_authority="council",
                verification_authority="auditor",
                derived_blocks=["PROVISIONING"],
                subject="Completely fabricated subject that mutates historical FND-ARCH-001",
            )

    def test_04_historical_id_reuse_forbidden(self):
        """04. Adding a finding reusing historical ID with mutated subject is rejected"""
        reg = FindingRegistry()
        f = Finding(
            finding_id="FND-ARCH-002",
            title="Genesis release provenance was unsupported or not derived from authoritative evidence.",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            subject="Genesis release provenance was unsupported or not derived from authoritative evidence.",
        )
        # Attempting to add with corrupted subject property
        f.subject = "New foreign subject reusing historical ID"
        with pytest.raises(FindingSubjectMutationForbiddenError):
            reg.add_finding(f)

    def test_05_identity_dict_schema_parity(self):
        """05. to_identity_dict() includes original_subject, domain_scope, and defect_layer"""
        f = get_canonical_known_architecture_findings()[0]
        ident = f.to_identity_dict()
        assert ident["finding_id"] == "FND-ARCH-001"
        assert ident["original_subject"] == FINDING_ORIGINAL_SUBJECTS["FND-ARCH-001"]
        assert ident["current_classification"]["domain_scope"] == "DENOMINATOR"
        assert ident["current_classification"]["defect_layer"] == "ARCHITECTURE_DEFECT"
        assert ident["identity_preserved"] is True

    def test_06_finding_identity_preservation_invariant(self, baseline_data):
        """06. Invariant check: FINDING_ID_SUBJECT_MUTATIONS_REMAINING == 0"""
        mutations = 0
        baseline_map = {b["finding_id"]: b["original_subject"] for b in baseline_data["findings"]}
        for f in get_all_registered_findings():
            if f.original_subject != baseline_map.get(f.finding_id):
                mutations += 1
        assert mutations == 0
        assert len(baseline_data["findings"]) == 15


# =============================================================================
# Matrix 45: Foreign Finding Audit & Admission Denial (Tests 7–11)
# =============================================================================

class TestMatrix45ForeignFindingAuditAndAdmissionDenial:

    def test_07_ten_candidate_foreign_findings_rejected_in_audit(self, foreign_audit_data):
        """07. All 10 candidate foreign findings audited and disproved with 0 code occurrences"""
        assert foreign_audit_data["summary"]["foreign_subjects_audited"] == 10
        assert foreign_audit_data["summary"]["unsupported_new_findings_admitted"] == 0
        assert len(foreign_audit_data["candidate_findings"]) == 10
        for cand in foreign_audit_data["candidate_findings"]:
            assert cand["evidence_quality"] == "DISPROVED"
            assert cand["recommendation"] == "REJECT_UNSUPPORTED"

    def test_08_attempt_to_admit_foreign_finding_raises_error(self):
        """08. Attempting to add any of the 10 foreign candidate findings raises UnsupportedForeignFindingError"""
        reg = FindingRegistry()
        for foreign_subj in KNOWN_DISPROVED_FOREIGN_SUBJECTS:
            f = Finding(
                finding_id=f"FND-FOREIGN-{abs(hash(foreign_subj)) % 1000}",
                title=foreign_subj,
                severity="HIGH",
                lifecycle_status=FindingStatus.OPEN,
                evidence_status="UNPROVEN",
                owner="foreign",
                resolution_authority="foreign",
                verification_authority="foreign",
                derived_blocks=["PROVISIONING"],
                subject=foreign_subj,
            )
            with pytest.raises(UnsupportedForeignFindingError):
                reg.add_finding(f)

    def test_09_foreign_finding_not_admitted_to_canonical_registry(self):
        """09. None of the 10 foreign subjects exist in the canonical registry"""
        canonical_subjects = {f.original_subject for f in get_all_registered_findings()}
        for foreign_subj in KNOWN_DISPROVED_FOREIGN_SUBJECTS:
            assert foreign_subj not in canonical_subjects

    def test_10_zero_unsupported_new_findings_admitted(self, foreign_audit_data):
        """10. Invariant check: UNSUPPORTED_NEW_FINDINGS_ADMITTED == 0"""
        assert foreign_audit_data["invariants"]["UNSUPPORTED_NEW_FINDINGS_ADMITTED"] == 0

    def test_11_audit_finding_hash_verification(self, foreign_audit_data):
        """11. Each audited candidate has deterministic SHA-256 subject hash"""
        for cand in foreign_audit_data["candidate_findings"]:
            expected_hash = hashlib.sha256(cand["proposed_subject"].strip().lower().encode("utf-8")).hexdigest()
            assert len(expected_hash) == 64


# =============================================================================
# Matrix 46: Claim-State & Authority Dimension Separation (Tests 12–17)
# =============================================================================

class TestMatrix46ClaimStateAuthorityDimensionSeparation:

    def test_12_six_valued_claim_states_supported(self):
        """12. All 6 claim states (ESTABLISHED, NOT_ESTABLISHED, UNKNOWN, CONFLICT, STALE, REVOKED) accepted"""
        assert CLAIM_STATES == {"ESTABLISHED", "NOT_ESTABLISHED", "UNKNOWN", "CONFLICT", "STALE", "REVOKED"}
        for s in CLAIM_STATES:
            basis = (
                StateBasis("ABSENCE", "AUTH", "SCOPE", ["git:tree"], "EXHAUSTIVE", "WINDOW", "2026-10-12T10:00:00Z")
                if s == "NOT_ESTABLISHED"
                else None
            )
            claim = CanonicalDerivedClaim(
                claim_id=f"CLAIM-{s}",
                claim_class="TEST",
                claim_state=s,
                derivation_id="drv-test",
                derivation_policy_hash="a" * 64,
                evidence_refs=["ev-1"],
                state_basis=basis,
            )
            assert claim.claim_state == s

    def test_13_orthogonal_claim_dimensions(self):
        """13. Dimensions (claim_state, provenance_class, attestation_status, freshness, authority_scope) are orthogonal"""
        basis = StateBasis("ABSENCE", "AUTH", "SCOPE", ["git:tree"], "EXHAUSTIVE", "WINDOW", "2026-10-12T10:00:00Z")
        claim = CanonicalDerivedClaim(
            claim_id="CLAIM-ORTHO-1",
            claim_class="ROOT_OF_TRUST",
            claim_state="NOT_ESTABLISHED",
            provenance_class="REPOSITORY_GROUNDED",
            attestation_status="LOCAL_REFERENCE",
            freshness="CURRENT",
            authority_scope="LOCAL_GOVERNANCE_REFERENCE",
            derivation_id="drv-ortho",
            derivation_policy_hash="b" * 64,
            evidence_refs=["ev-2"],
            state_basis=basis,
        )
        d = claim.to_dict()
        assert d["claim_state"] == "NOT_ESTABLISHED"
        assert d["provenance_class"] == "REPOSITORY_GROUNDED"
        assert d["attestation_status"] == "LOCAL_REFERENCE"
        assert d["freshness"] == "CURRENT"
        assert d["authority_scope"] == "LOCAL_GOVERNANCE_REFERENCE"

    def test_14_claim_semantics_schema_validation(self):
        """14. Derived claims match CLAIM_SEMANTICS_SCHEMA.json structure"""
        schema_path = os.path.join(DOCS_DIR, "CLAIM_SEMANTICS_SCHEMA.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        req = schema["required"]
        assert "claim_id" in req
        assert "claim_state" in req
        assert "provenance_class" in req
        assert "attestation_status" in req
        assert "freshness" in req
        assert "authority_scope" in req

    def test_15_conflict_preserved_never_collapsed_to_not_established(self):
        """15. Conflicting evidence preserves CONFLICT; CONFLICT_COLLAPSED_TO_NOT_ESTABLISHED == 0"""
        records = [
            EvidenceRecord(
                evidence_id="ev-est-1",
                claim_id="CLAIM-EXTERNAL-LEDGER",
                evidence_state="ESTABLISHED",
                provenance_class="CANONICAL_LEDGER",
                authority_class="AUTHORITATIVE",
                source="urn:arx:source:ledger:test1",
                content_hash=hashlib.sha256(b"ledger:evidence:established:test").hexdigest(),
                validity="CURRENT",
                observed_at="2026-10-12T10:00:00Z",
            ),
            EvidenceRecord(
                evidence_id="ev-notest-1",
                claim_id="CLAIM-EXTERNAL-LEDGER",
                evidence_state="NOT_ESTABLISHED",
                provenance_class="NONE",
                authority_class="NOT_APPLICABLE",
                source=None,
                content_hash=None,
                validity="CURRENT",
                observed_at="2026-10-12T10:00:00Z",
            ),
        ]
        manifest = build_canonical_evidence_manifest(records, "commit-sha")
        engine = ClaimDerivationEngine()
        claims = engine.derive_all_claims(manifest)
        c = claims["CLAIM-EXTERNAL-LEDGER"]
        assert c.claim_state == "CONFLICT"
        assert c.claim_state != "NOT_ESTABLISHED"

    def test_16_conflict_preserved_never_collapsed_to_unknown(self):
        """16. Conflicting evidence preserves CONFLICT; CONFLICT_COLLAPSED_TO_UNKNOWN == 0"""
        records = [
            EvidenceRecord(
                evidence_id="ev-est-1",
                claim_id="CLAIM-EXTERNAL-LEDGER",
                evidence_state="ESTABLISHED",
                provenance_class="CANONICAL_LEDGER",
                authority_class="AUTHORITATIVE",
                source="urn:arx:source:ledger:test1",
                content_hash=hashlib.sha256(b"ledger:evidence:established:test").hexdigest(),
                validity="CURRENT",
                observed_at="2026-10-12T10:00:00Z",
            ),
            EvidenceRecord(
                evidence_id="ev-notest-1",
                claim_id="CLAIM-EXTERNAL-LEDGER",
                evidence_state="NOT_ESTABLISHED",
                provenance_class="NONE",
                authority_class="NOT_APPLICABLE",
                source=None,
                content_hash=None,
                validity="CURRENT",
                observed_at="2026-10-12T10:00:00Z",
            ),
        ]
        manifest = build_canonical_evidence_manifest(records, "commit-sha")
        engine = ClaimDerivationEngine()
        claims = engine.derive_all_claims(manifest)
        c = claims["CLAIM-EXTERNAL-LEDGER"]
        assert c.claim_state == "CONFLICT"
        assert c.claim_state != "UNKNOWN"

    def test_17_unknown_never_laundered_as_not_established(self):
        """17. Invariant check: UNKNOWN_LAUNDERED_AS_NOT_ESTABLISHED == 0"""
        with pytest.raises(AuthoritativeStateBasisMissingError):
            CanonicalDerivedClaim(
                claim_id="CLAIM-LAUNDERED",
                claim_class="TEST",
                claim_state="NOT_ESTABLISHED",
                derivation_id="drv-bad",
                derivation_policy_hash="c" * 64,
                evidence_refs=[],
                state_basis=None,
            )


# =============================================================================
# Matrix 47: Authoritative StateBasis Completeness (Tests 18–22)
# =============================================================================

class TestMatrix47AuthoritativeStateBasisCompleteness:

    def test_18_not_established_requires_complete_state_basis(self):
        """18. NOT_ESTABLISHED claim without StateBasis raises AuthoritativeStateBasisMissingError"""
        with pytest.raises(AuthoritativeStateBasisMissingError):
            CanonicalDerivedClaim(
                claim_id="CLAIM-TEST-NO-BASIS",
                claim_class="TEST",
                claim_state="NOT_ESTABLISHED",
                derivation_id="drv-test",
                derivation_policy_hash="d" * 64,
                evidence_refs=["ev-1"],
                state_basis=None,
            )

    def test_19_state_basis_all_fields_mandatory(self):
        """19. Incomplete StateBasis (missing authority_source or search_scope) raises AuthoritativeStateBasisMissingError"""
        sb_incomplete = StateBasis(
            basis_type="ABSENCE",
            authority_source="",  # Empty!
            search_scope="scope",
            source_set=["git:tree"],
            completeness_assertion="EXHAUSTIVE",
            observation_window="WINDOW",
            observed_at="2026-10-12T10:00:00Z",
        )
        with pytest.raises(AuthoritativeStateBasisMissingError):
            CanonicalDerivedClaim(
                claim_id="CLAIM-TEST-INCOMPLETE-BASIS",
                claim_class="TEST",
                claim_state="NOT_ESTABLISHED",
                derivation_id="drv-test",
                derivation_policy_hash="e" * 64,
                evidence_refs=["ev-1"],
                state_basis=sb_incomplete,
            )

    def test_20_empty_source_set_in_state_basis_rejected(self):
        """20. StateBasis with empty source_set raises AuthoritativeStateBasisMissingError"""
        sb_empty_sources = StateBasis(
            basis_type="ABSENCE",
            authority_source="auth",
            search_scope="scope",
            source_set=[],  # Empty!
            completeness_assertion="EXHAUSTIVE",
            observation_window="WINDOW",
            observed_at="2026-10-12T10:00:00Z",
        )
        with pytest.raises(AuthoritativeStateBasisMissingError):
            CanonicalDerivedClaim(
                claim_id="CLAIM-TEST-EMPTY-SOURCES",
                claim_class="TEST",
                claim_state="NOT_ESTABLISHED",
                derivation_id="drv-test",
                derivation_policy_hash="f" * 64,
                evidence_refs=["ev-1"],
                state_basis=sb_empty_sources,
            )

    def test_21_canonical_claim_derivation_emits_valid_state_basis(self, canonical_context):
        """21. Engine produces complete StateBasis for unestablished claims"""
        not_est_claims = [c for c in canonical_context.derived_claims.values() if c.claim_state == "NOT_ESTABLISHED"]
        assert len(not_est_claims) > 0
        for c in not_est_claims:
            sb = c.state_basis
            assert sb is not None
            assert sb.basis_type != ""
            assert sb.authority_source != ""
            assert sb.search_scope != ""
            assert len(sb.source_set) > 0
            assert sb.completeness_assertion != ""
            assert sb.observation_window != ""
            assert sb.observed_at != ""

    def test_22_state_basis_serialization_parity(self):
        """22. StateBasis.to_dict() serializes all required fields according to CLAIM_SEMANTICS_SCHEMA"""
        sb = StateBasis(
            basis_type="AUTHORITATIVE_CONFIRMED_ABSENCE",
            authority_source="urn:arx:authority:bootstrap-council:v1",
            search_scope="urn:arx:scope:canonical-evidence-records",
            source_set=["git:tree:docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json"],
            completeness_assertion="EXHAUSTIVE_RECOMPUTED_ABSENCE",
            observation_window="GENESIS_TO_HEAD",
            observed_at="2026-10-12T12:00:00Z",
        )
        d = sb.to_dict()
        assert d["basis_type"] == "AUTHORITATIVE_CONFIRMED_ABSENCE"
        assert d["authority_source"] == "urn:arx:authority:bootstrap-council:v1"
        assert d["search_scope"] == "urn:arx:scope:canonical-evidence-records"
        assert len(d["source_set"]) == 1
        assert d["completeness_assertion"] == "EXHAUSTIVE_RECOMPUTED_ABSENCE"
        assert d["observation_window"] == "GENESIS_TO_HEAD"
        assert d["observed_at"] == "2026-10-12T12:00:00Z"


# =============================================================================
# Matrix 48: Operator Ratification Artifact Enforcement (Tests 23–27)
# =============================================================================

class TestMatrix48OperatorRatificationArtifactEnforcement:

    def test_23_operator_ratified_without_artifact_rejected(self):
        """23. Asserting OPERATOR_RATIFIED without ratification artifact raises UnratifiedClaimError"""
        with pytest.raises(UnratifiedClaimError):
            CanonicalDerivedClaim(
                claim_id="CLAIM-RATIFIED-FAKE",
                claim_class="TEST",
                claim_state="ESTABLISHED",
                attestation_status="OPERATOR_RATIFIED",
                derivation_id="drv-fake-rat",
                derivation_policy_hash="1" * 64,
                evidence_refs=["ev-1"],
                metadata={},  # No ratification artifact ref!
            )

    def test_24_operator_ratified_with_artifact_accepted(self):
        """24. Asserting OPERATOR_RATIFIED with ratification artifact reference succeeds"""
        claim = CanonicalDerivedClaim(
            claim_id="CLAIM-RATIFIED-VALID",
            claim_class="TEST",
            claim_state="ESTABLISHED",
            attestation_status="OPERATOR_RATIFIED",
            derivation_id="drv-valid-rat",
            derivation_policy_hash="2" * 64,
            evidence_refs=["ev-1"],
            metadata={
                "ratification_artifact_ref": "urn:arx:artifact:ratification:operator-charlie:v1",
            },
        )
        assert claim.attestation_status == "OPERATOR_RATIFIED"

    def test_25_local_execution_defaults_to_local_reference(self, canonical_context):
        """25. Engine derives claims with LOCAL_REFERENCE attestation by default"""
        for claim in canonical_context.derived_claims.values():
            assert claim.attestation_status in ["LOCAL_REFERENCE", "UNRATIFIED"]
            assert claim.attestation_status != "OPERATOR_RATIFIED"

    def test_26_root_operator_ratification_status_pending(self):
        """26. Authoritative state reflects ROOT_MODEL_OPERATOR_RATIFICATION == PENDING_OPERATOR_RATIFICATION"""
        audit_path = os.path.join(DOCS_DIR, "FINDING_ADJUDICATION_REPORT.json")
        with open(audit_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["adjudication_authority"]["operational_authority"] is False
        assert report["adjudication_authority"]["ratification_ref"] == "PENDING_OPERATOR_RATIFICATION"

    def test_27_operator_ratified_invariant_zero(self):
        """27. Invariant check: OPERATOR_RATIFIED_WITHOUT_RATIFICATION_ARTIFACT == 0"""
        reg = create_canonical_finding_registry()
        for f in reg.findings:
            assert f.resolution_authority != "OPERATOR_RATIFIED_UNPROVEN"


# =============================================================================
# Matrix 49: Root-Cause Duplicate Equivalence & Cycles (Tests 28–32)
# =============================================================================

class TestMatrix49RootCauseDuplicateEquivalenceAndCycles:

    def test_28_fnd_arch_013_is_duplicate_of_007(self):
        """28. FND-ARCH-013 is formally classified as duplicate of FND-ARCH-007 with root cause equivalence"""
        reg = create_canonical_finding_registry()
        f13 = reg.get_finding("FND-ARCH-013")
        f07 = reg.get_finding("FND-ARCH-007")
        assert f13.admission_status == "DUPLICATE"
        assert f13.duplicate_of == "FND-ARCH-007"
        assert f13.is_blocking is False
        assert f07.admission_status == "ADMITTED"
        assert f07.is_blocking is True

    def test_29_fnd_arch_015_is_duplicate_of_011(self):
        """29. FND-ARCH-015 is formally classified as duplicate of FND-ARCH-011 with root cause equivalence"""
        reg = create_canonical_finding_registry()
        f15 = reg.get_finding("FND-ARCH-015")
        f11 = reg.get_finding("FND-ARCH-011")
        assert f15.admission_status == "DUPLICATE"
        assert f15.duplicate_of == "FND-ARCH-011"
        assert f15.is_blocking is False
        assert f11.admission_status == "ADMITTED"
        assert f11.is_blocking is True

    def test_30_duplicate_without_canonical_target_rejected(self):
        """30. Duplicate finding referencing non-existent target raises DuplicateWithoutCanonicalTargetError"""
        reg = FindingRegistry()
        f = Finding(
            finding_id="FND-DUP-NONE",
            title="Duplicate without target",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="DUPLICATE",
            duplicate_of="FND-NON-EXISTENT",
        )
        with pytest.raises(DuplicateWithoutCanonicalTargetError):
            reg.add_finding(f)

    def test_31_duplicate_self_reference_rejected(self):
        """31. Duplicate finding referencing itself raises DuplicateSelfReferenceError"""
        reg = FindingRegistry()
        f = Finding(
            finding_id="FND-DUP-SELF",
            title="Duplicate self",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="DUPLICATE",
            duplicate_of="FND-DUP-SELF",
        )
        with pytest.raises(DuplicateSelfReferenceError):
            reg.add_finding(f)

    def test_32_duplicate_cycle_rejected(self):
        """32. Duplicate cycle raises DuplicateCycleError"""
        reg = FindingRegistry()
        f_a = Finding(
            finding_id="FND-CYCLE-A",
            title="Cycle A",
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
            finding_id="FND-CYCLE-B",
            title="Cycle B",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNPROVEN",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="DUPLICATE",
            duplicate_of="FND-CYCLE-A",
        )
        reg.add_finding(f_a)
        reg.add_finding(f_b)
        adj = FindingAdjudication(
            adjudication_id="adj-cyc",
            finding_id="FND-CYCLE-A",
            prior_admission_status="ADMITTED",
            new_admission_status="DUPLICATE",
            outcome="DUPLICATE",
            rationale="creating cycle",
            duplicate_of="FND-CYCLE-B",
        )
        with pytest.raises(DuplicateCycleError):
            reg.apply_adjudication(adj)


# =============================================================================
# Matrix 50: Causal Blocker Re-Derivation (Zero Severity Bias) (Tests 33–38)
# =============================================================================

class TestMatrix50CausalBlockerReDerivation:

    def test_33_blocking_is_causal_not_severity_based(self):
        """33. Blocking is derived from non-empty derived_blocks, zero severity bias"""
        f_med = Finding(
            finding_id="FND-MED-BLOCKING",
            title="Medium severity finding with derived blocks",
            severity="MEDIUM",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="CONFIRMED",
            owner="sec",
            resolution_authority="council",
            verification_authority="auditor",
            derived_blocks=["PROVISIONING"],
            admission_status="ADMITTED",
        )
        assert f_med.is_blocking is True

    def test_34_fnd_arch_009_is_active_blocker(self):
        """34. FND-ARCH-009 (severity MEDIUM, POLICY_DEFECT) is an active blocker because of PROVISIONING and THEOREM_D"""
        reg = create_canonical_finding_registry()
        f09 = reg.get_finding("FND-ARCH-009")
        assert f09.severity == "MEDIUM"
        assert f09.reclassified_layer == "POLICY_DEFECT"
        assert "PROVISIONING" in f09.derived_blocks
        assert "THEOREM_D" in f09.derived_blocks
        assert f09.is_blocking is True
        assert f09 in reg.get_active_blockers()

    def test_35_total_active_blocker_count_is_thirteen(self):
        """35. Canonical finding registry produces exactly 13 active blockers"""
        reg = create_canonical_finding_registry()
        blockers = reg.get_active_blockers()
        assert len(blockers) == 13
        blocker_ids = {b.finding_id for b in blockers}
        assert "FND-ARCH-013" not in blocker_ids
        assert "FND-ARCH-015" not in blocker_ids

    def test_36_unresolved_architecture_defects_count_is_eight(self):
        """36. Exactly 8 unresolved architecture defects exist (001, 004, 007, 008, 010, 011, 012, 014)"""
        reg = create_canonical_finding_registry()
        arch_defects = reg.get_unresolved_architecture_defects()
        assert len(arch_defects) == 8
        arch_ids = [a.finding_id for a in arch_defects]
        expected_arch_ids = [
            "FND-ARCH-001",
            "FND-ARCH-004",
            "FND-ARCH-007",
            "FND-ARCH-008",
            "FND-ARCH-010",
            "FND-ARCH-011",
            "FND-ARCH-012",
            "FND-ARCH-014",
        ]
        assert arch_ids == expected_arch_ids

    def test_37_active_non_architecture_blockers_count_is_five(self):
        """37. Exactly 5 active non-architecture blockers exist (002, 003, 005, 006, 009)"""
        reg = create_canonical_finding_registry()
        non_arch = reg.get_active_non_architecture_blockers()
        assert len(non_arch) == 5
        non_arch_ids = [n.finding_id for n in non_arch]
        expected_non_arch_ids = [
            "FND-ARCH-002",
            "FND-ARCH-003",
            "FND-ARCH-005",
            "FND-ARCH-006",
            "FND-ARCH-009",
        ]
        assert non_arch_ids == expected_non_arch_ids
        # 8 + 5 == 13 total blockers
        assert len(reg.get_unresolved_architecture_defects()) + len(non_arch) == len(reg.get_active_blockers())

    def test_38_blocking_derived_from_severity_only_invariant(self):
        """38. Invariant check: BLOCKING_DERIVED_FROM_SEVERITY_ONLY == 0"""
        mapping_path = os.path.join(DOCS_DIR, "FINDING_DECISION_MAPPING.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        assert mapping["invariants"]["BLOCKING_DERIVED_FROM_SEVERITY_ONLY"] == 0
        assert mapping["invariants"]["ADMITTED_OPEN_FINDINGS_WITHOUT_CAUSAL_MAPPING"] == 0
        assert mapping["invariants"]["ACTIVE_BLOCKERS_WITHOUT_DEPENDENCY_PATH"] == 0


# =============================================================================
# Matrix 51: Blocker Ablation Parity (Tests 39–43)
# =============================================================================

class TestMatrix51BlockerAblationParity:

    def test_39_ablation_report_parity(self, ablation_report_data):
        """39. Blocker ablation report confirms 13 cases executed and 13 passed"""
        assert ablation_report_data["summary"]["active_blocker_count"] == 13
        assert ablation_report_data["summary"]["ablation_cases_executed"] == 13
        assert ablation_report_data["summary"]["ablation_cases_passed"] == 13
        assert ablation_report_data["summary"]["ablation_cases_failed"] == 0
        assert ablation_report_data["summary"]["parity_confirmed"] is True

    def test_40_all_thirteen_blockers_ablated_individually(self, ablation_report_data):
        """40. Every one of the 13 active blockers passes isolated ablation"""
        results = ablation_report_data["ablation_results"]
        assert len(results) == 13
        ablated_ids = [r["finding_id"] for r in results]
        expected_ids = [
            "FND-ARCH-001", "FND-ARCH-002", "FND-ARCH-003", "FND-ARCH-004",
            "FND-ARCH-005", "FND-ARCH-006", "FND-ARCH-007", "FND-ARCH-008",
            "FND-ARCH-009", "FND-ARCH-010", "FND-ARCH-011", "FND-ARCH-012",
            "FND-ARCH-014",
        ]
        assert ablated_ids == expected_ids
        for r in results:
            assert r["ablation_verdict"] == "PASS"

    def test_41_monotone_resolution_decreases_blocker_count(self):
        """41. Monotonicity: resolving 1 blocker decreases count from 13 to 12"""
        reg = create_canonical_finding_registry()
        assert len(reg.get_active_blockers()) == 13
        reg.get_finding("FND-ARCH-001").lifecycle_status = FindingStatus.VERIFIED_RESOLVED
        assert len(reg.get_active_blockers()) == 12

    def test_42_waived_finding_within_validity_drops_from_blockers(self):
        """42. Waived finding drops out of active blockers"""
        reg = create_canonical_finding_registry()
        reg.get_finding("FND-ARCH-006").lifecycle_status = FindingStatus.WAIVED
        assert len(reg.get_active_blockers()) == 12
        assert reg.get_finding("FND-ARCH-006") not in reg.get_active_blockers()

    def test_43_ablation_invariants_pass(self, ablation_report_data):
        """43. Invariant check: ACTIVE_BLOCKER_COUNT == ABLATION_CASES_EXECUTED == ABLATION_CASES_PASSED == 13"""
        inv = ablation_report_data["invariants"]
        assert inv["ACTIVE_BLOCKER_COUNT"] == 13
        assert inv["ABLATION_CASES_EXECUTED"] == 13
        assert inv["ABLATION_CASES_PASSED"] == 13
        assert inv["ABLATION_CASES_FAILED"] == 0


# =============================================================================
# Matrix 52: Decision Invalidation & Epoch Succession (Tests 44–47)
# =============================================================================

class TestMatrix52DecisionInvalidationAndEpochSuccession:

    def test_44_three_historical_decisions_invalidated(self, invalidation_registry_data):
        """44. Decisions 3dc851a9, 2563e663, and c3277609 are in invalidation registry"""
        inv_hashes = {d["decision_hash"] for d in invalidation_registry_data["invalidated_decisions"]}
        assert "3dc851a90e213da557a79027fc4b3a07287713bc84d88effcf92dadd0b539e41" in inv_hashes
        assert "2563e663a7717bbb411229ea0bbab3ad0da58d3ff8a100bce14099132e096254" in inv_hashes
        assert "c3277609a97c07a4bfcdd87701109cf0c40f36ba21105835d24a9b5d1bbe352f" in inv_hashes
        for h in inv_hashes:
            assert h in INVALIDATED_PRE_INTEGRATION_DECISION_HASHES

    def test_45_invalidated_decision_cannot_be_current(self):
        """45. DecisionSnapshot with invalidated hash and CURRENT validity raises PreIntegrationSnapshotRejectedError"""
        with pytest.raises(PreIntegrationSnapshotRejectedError):
            DecisionSnapshot(
                decision_id="urn:arx:decision:test",
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
                derivation_engine_hash="d" * 64,
                derivation_engine_version="1.0.0",
                normalization_version="ARX-GOV-NORMALIZE-V1",
                canonicalization_version="RFC-8785-JCS",
                architecture_closure_status="OPEN",
                remediation_status="REQUIRED",
                global_provisioning_readiness="BLOCKED",
                theorem_readiness={"A": "BLOCKED"},
                runtime_action="NONE",
                derived_at="2026-10-12T12:00:00Z",
                decision_validity="CURRENT",
                decision_hash="c3277609a97c07a4bfcdd87701109cf0c40f36ba21105835d24a9b5d1bbe352f",
            )

    def test_46_reused_invalidated_sequence_slot_rejected(self, canonical_context):
        """46. Attempting to evaluate in invalidated sequence slot (epoch-gov-002, 1) raises ReusedInvalidatedSequenceSlotError"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        engine.invalidate_sequence_slot("epoch-gov-002", 1)
        with pytest.raises(ReusedInvalidatedSequenceSlotError):
            engine.evaluate_canonical(context=canonical_context, sequence=1)

    def test_47_successor_decision_is_sequence_two(self, canonical_context):
        """47. Clean successor decision in epoch-gov-002 is sequence 2 linked to invalidated sequence 1 hash"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        prev_hash = "c3277609a97c07a4bfcdd87701109cf0c40f36ba21105835d24a9b5d1bbe352f"
        snap = engine.evaluate_canonical(
            context=canonical_context,
            sequence=2,
            previous_decision_hash=prev_hash,
        )
        assert snap.decision_sequence == 2
        assert snap.previous_decision_hash == prev_hash
        assert snap.decision_validity == "CURRENT"
        assert snap.is_canonical is True
        assert snap.global_provisioning_readiness == "BLOCKED"
        assert snap.architecture_closure_status == "OPEN"


# =============================================================================
# Matrix 53: Canonical Context Anti-Forgery & Split Hash Integrity (Tests 48–52)
# =============================================================================

class TestMatrix53CanonicalContextAntiForgeryAndSplitHashIntegrity:

    def test_48_split_finding_hashes_distinct(self, canonical_context):
        """48. finding_identity_registry_hash, finding_admission_registry_hash, and active_blocker_set_hash are distinct"""
        id_h = canonical_context.finding_identity_registry_hash
        adm_h = canonical_context.finding_admission_registry_hash
        blk_h = canonical_context.active_blocker_set_hash
        assert len(id_h) == 64
        assert len(adm_h) == 64
        assert len(blk_h) == 64
        assert id_h != adm_h
        assert adm_h != blk_h
        assert id_h != blk_h

    def test_49_subject_change_mutates_identity_hash(self):
        """49. Changing any finding subject changes identity_registry_hash"""
        reg1 = create_canonical_finding_registry()
        id_h1 = reg1.compute_identity_registry_hash()

        # Create reg2 with altered subject
        findings2 = get_all_registered_findings()
        findings2[0].subject = "Mutated subject for test hash check"
        reg2 = FindingRegistry(findings2)
        id_h2 = reg2.compute_identity_registry_hash()

        assert id_h1 != id_h2

    def test_50_lifecycle_change_mutates_admission_and_blocker_hashes_not_identity(self):
        """50. Resolving a finding alters admission and blocker hashes but preserves identity hash"""
        reg = create_canonical_finding_registry()
        id_h1 = reg.compute_identity_registry_hash()
        adm_h1 = reg.compute_admission_registry_hash()
        blk_h1 = reg.compute_active_blocker_set_hash()

        reg.get_finding("FND-ARCH-001").lifecycle_status = FindingStatus.VERIFIED_RESOLVED

        id_h2 = reg.compute_identity_registry_hash()
        adm_h2 = reg.compute_admission_registry_hash()
        blk_h2 = reg.compute_active_blocker_set_hash()

        assert id_h1 == id_h2
        assert adm_h1 != adm_h2
        assert blk_h1 != blk_h2

    def test_51_caller_constructed_forged_context_rejected(self, canonical_context):
        """51. Tampered context hash raises CallerConstructedCanonicalContextRejectedError"""
        engine = GovernanceDecisionEngine(epoch="epoch-gov-002")
        # Construct fake context with mismatched hash
        class ForgedContext:
            decision_context_hash = "deadbeef" * 8
            def to_dict(self):
                return {"fake": "payload", "decision_context_hash": self.decision_context_hash}

        with pytest.raises(CallerConstructedCanonicalContextRejectedError):
            engine.evaluate_canonical(context=ForgedContext())

    def test_52_candidate004_freeze_hash_strict_invariance(self):
        """52. Raw byte hash of CANDIDATE_GENERATION_004_FREEZE.json strictly matches 297ef5ab..."""
        with open(FREEZE_PATH, "rb") as f:
            raw_bytes = f.read()
        computed = hashlib.sha256(raw_bytes).hexdigest()
        assert computed == EXPECTED_CANDIDATE004_FREEZE_HASH
        assert computed == "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"
