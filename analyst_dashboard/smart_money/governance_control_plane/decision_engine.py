"""
Governance Decision Engine Subsystem
ARX Terminal — Smart Money

Evaluates policy against Finding Registry and Evidence Inventory to produce
deterministic, hash-bound Decision Snapshots.
Includes Primary Engine and Independent Reference Evaluator.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime, timezone

from .canonicalization import sha256_canonical, ARX_GOV_NORMALIZE_VERSION, JCS_VERSION
from .finding_registry import FindingRegistry, FindingStatus
from .evidence_inventory import EvidenceInventory
from .evidence_manifest import CanonicalEvidenceManifest, ClaimRegistry, EvidenceRecord
from .dependency_policy import (
    CompiledDependencyPolicy,
    DependencyResolutionResult,
    resolve_dependencies,
    compile_dependency_policy,
)
from .predicate_evaluator import FactContext

ENGINE_VERSION = "1.0.0"
ENGINE_SOURCE_SHA = "ab32ec91bff0dce73a35938af73e1af8d43e5c68"

RUNTIME_ACTIONS = ["NONE", "HALT_NEW_ADMISSIONS", "TERMINAL_HOLD", "INCIDENT_RESPONSE"]

INVALIDATED_PRE_INTEGRATION_DECISION_HASHES = {
    "3dc851a90e213da557a79027fc4b3a07287713bc84d88effcf92dadd0b539e41",
    "2563e663a7717bbb411229ea0bbab3ad0da58d3ff8a100bce14099132e096254",
}


class PreIntegrationSnapshotRejectedError(Exception):
    """Raised when an invalidated pre-integration snapshot is supplied as current canonical."""
    pass


class DuplicateDecisionSequenceError(Exception):
    """Raised when an attempt is made to evaluate/commit a conflicting decision with an already-used sequence."""
    pass


from .canonical_context import CallerConstructedCanonicalContextRejectedError


def compute_canonical_contract_set_hash() -> str:
    """Computes SHA-256 hash over canonical list of contract documents."""
    contract_manifest = {
        "contracts": [
            "urn:arx:contract:governance:root_of_trust:v1",
            "urn:arx:contract:governance:bootstrap_authority:v1",
            "urn:arx:contract:governance:root_identity_model:v1",
            "urn:arx:contract:governance:genesis_ceremony_protocol:v1",
            "urn:arx:contract:governance:bootstrap_extinguishment:v1",
            "urn:arx:contract:governance:normal_authority_activation:v1",
            "urn:arx:contract:governance:authority_loss_recovery:v1",
            "urn:arx:contract:governance:witness_fork_recovery:v1",
            "urn:arx:contract:governance:dr_epoch_transition:v1",
        ],
        "schema": "ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1",
    }
    return sha256_canonical(contract_manifest)


def compute_canonical_theorem_state_hash() -> str:
    """Computes SHA-256 hash over canonical mapping of theorems."""
    theorem_manifest = {
        "theorems": {
            "A": "BLOCKED",
            "B": "BLOCKED",
            "C": "BLOCKED",
            "D": "BLOCKED",
            "E": "BLOCKED",
        },
        "governance_epoch": "epoch-gov-001",
    }
    return sha256_canonical(theorem_manifest)


@dataclass
class DecisionSnapshot:
    decision_id: str
    governance_epoch: str
    decision_sequence: int
    previous_decision_hash: str
    policy_hash: str
    policy_ratification_hash: str
    finding_registry_hash: str
    evidence_inventory_hash: str
    contract_set_hash: str
    theorem_state_hash: str
    derivation_engine_source_sha: str
    derivation_engine_hash: str
    derivation_engine_version: str
    normalization_version: str
    canonicalization_version: str
    architecture_closure_status: str
    remediation_status: str
    global_provisioning_readiness: str
    theorem_readiness: Dict[str, str]
    runtime_action: str
    derived_at: str
    decision_validity: str
    decision_hash: str
    # Canonical bindings (Section 14 & 15)
    source_freeze_sha: Optional[str] = None
    evidence_manifest_hash: Optional[str] = None
    claim_registry_hash: Optional[str] = None
    dependency_policy_hash: Optional[str] = None
    predicate_set_hash: Optional[str] = None
    predicate_context_hash: Optional[str] = None
    dependency_resolution_hash: Optional[str] = None
    provisioning_authorization_status: str = "NOT_ISSUABLE"
    is_canonical: bool = False
    canonical_eligibility: str = "PASS"
    authority_eligibility: str = "AUTHORITATIVE"
    decision_context_hash: Optional[str] = None
    finding_identity_registry_hash: Optional[str] = None
    finding_admission_registry_hash: Optional[str] = None
    active_blocker_set_hash: Optional[str] = None
    evaluator_bundle_hash: Optional[str] = None
    candidate_freeze_file_sha256: Optional[str] = None
    candidate_semantic_closure_hash: Optional[str] = None
    canonical_status: str = "CANONICAL"

    def __post_init__(self):
        if self.decision_hash in INVALIDATED_PRE_INTEGRATION_DECISION_HASHES:
            self.canonical_status = "INVALIDATED_NON_CANONICAL"
        if self.canonical_status == "INVALIDATED_NON_CANONICAL" and self.decision_validity == "CURRENT":
            raise PreIntegrationSnapshotRejectedError(
                "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: "
                "Invalidated pre-integration snapshot cannot have decision_validity == 'CURRENT'."
            )
        if self.is_canonical:
            if not self.dependency_resolution_hash:
                raise ValueError(
                    "Canonical DecisionSnapshot requires dependency_resolution_hash. "
                    "DECISION_SNAPSHOT_WITHOUT_DEPENDENCY_BINDING_ACCEPTED == 0."
                )
            if not self.evidence_manifest_hash:
                raise ValueError(
                    "Canonical DecisionSnapshot requires evidence_manifest_hash. "
                    "DECISION_SNAPSHOT_WITHOUT_EVIDENCE_BINDING_ACCEPTED == 0."
                )
            if not self.finding_registry_hash:
                raise ValueError(
                    "Canonical DecisionSnapshot requires finding_registry_hash. "
                    "DECISION_SNAPSHOT_WITHOUT_FINDING_BINDING_ACCEPTED == 0."
                )
            if self.governance_epoch == "epoch-gov-002":
                if not self.decision_context_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires decision_context_hash."
                    )
                if not self.finding_identity_registry_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires finding_identity_registry_hash."
                    )
                if not self.finding_admission_registry_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires finding_admission_registry_hash."
                    )
                if not self.active_blocker_set_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires active_blocker_set_hash."
                    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_canonical_decision_snapshot(snapshot: DecisionSnapshot, current_epoch: str = "epoch-gov-002") -> bool:
    """Validates that a DecisionSnapshot satisfies all canonical requirements for the active governance epoch."""
    if snapshot.decision_hash in INVALIDATED_PRE_INTEGRATION_DECISION_HASHES:
        raise PreIntegrationSnapshotRejectedError(
            "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: Pre-integration snapshot is invalidated."
        )
    if snapshot.canonical_status == "INVALIDATED_NON_CANONICAL":
        raise PreIntegrationSnapshotRejectedError(
            "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: Snapshot is marked INVALIDATED_NON_CANONICAL."
        )
    if current_epoch == "epoch-gov-002" and snapshot.governance_epoch == "epoch-gov-001":
        raise PreIntegrationSnapshotRejectedError(
            "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: Predecessor epoch-gov-001 snapshot cannot be accepted as current canonical in epoch-gov-002."
        )
    if not snapshot.is_canonical:
        raise ValueError("Snapshot is not marked is_canonical == True.")
    if snapshot.canonical_eligibility != "PASS":
        raise ValueError(f"Canonical eligibility is {snapshot.canonical_eligibility}, expected PASS.")
    if snapshot.authority_eligibility != "AUTHORITATIVE":
        raise ValueError(f"Authority eligibility is {snapshot.authority_eligibility}, expected AUTHORITATIVE.")
    return True


class GovernanceDecisionEngine:
    """
    Primary Deterministic Decision Engine.
    Evaluates policy rules against FindingRegistry and EvidenceInventory.
    """

    def __init__(
        self,
        policy_hash: str = "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
        policy_ratification_hash: str = "8a9f0e1d2c3b4a596877869504132435465768798091a2b3c4d5e6f7a8b9c0d1",
        epoch: str = "epoch-gov-001",
        allow_caller_canonical_inputs: Optional[bool] = None,
        enforce_sequence_uniqueness: bool = False,
    ):
        self.policy_hash = policy_hash
        self.policy_ratification_hash = policy_ratification_hash
        self.epoch = epoch
        self.version = ENGINE_VERSION
        self.source_sha = ENGINE_SOURCE_SHA
        if allow_caller_canonical_inputs is None:
            self.allow_caller_canonical_inputs = (epoch == "epoch-gov-001")
        else:
            self.allow_caller_canonical_inputs = allow_caller_canonical_inputs
        self.enforce_sequence_uniqueness = enforce_sequence_uniqueness
        self._registered_committed_decisions: Dict[tuple, str] = {}
        self._committed_decisions: Dict[tuple, str] = {}

    def register_committed_decision(self, epoch: str, sequence: int, decision_hash: str) -> None:
        """Explicitly registers an authoritative decision hash into the engine's anti-equivocation ledger."""
        self._registered_committed_decisions[(epoch, sequence)] = decision_hash

    def compute_engine_hash(self) -> str:
        metadata = {
            "version": self.version,
            "source_sha": self.source_sha,
            "engine_class": "GovernanceDecisionEngine",
        }
        return sha256_canonical(metadata)

    def evaluate(
        self,
        finding_registry: FindingRegistry,
        evidence_inventory: EvidenceInventory,
        sequence: int = 1,
        previous_decision_hash: str = "0" * 64,
        contract_set_hash: Optional[str] = None,
        theorem_state_hash: Optional[str] = None,
        derived_at: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> DecisionSnapshot:
        if not derived_at:
            derived_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not contract_set_hash:
            contract_set_hash = compute_canonical_contract_set_hash()

        if not theorem_state_hash:
            theorem_state_hash = compute_canonical_theorem_state_hash()

        # 1. Evaluate Findings
        active_blockers = finding_registry.get_active_blockers()
        has_blockers = len(active_blockers) > 0
        unresolved_arch_defects = finding_registry.get_unresolved_architecture_defects()

        # Check blocked theorems from active blockers
        theorem_readiness = {"A": "READY", "B": "READY", "C": "READY", "D": "READY", "E": "BLOCKED"}
        for b in active_blockers:
            for blk in b.derived_blocks:
                if blk in ["THEOREM_A", "PROVISIONING"]:
                    theorem_readiness["A"] = "BLOCKED"
                if blk in ["THEOREM_B", "PROVISIONING"]:
                    theorem_readiness["B"] = "BLOCKED"
                if blk in ["THEOREM_C", "PROVISIONING"]:
                    theorem_readiness["C"] = "BLOCKED"
                if blk in ["THEOREM_D", "PROVISIONING"]:
                    theorem_readiness["D"] = "BLOCKED"

        # 2. Evaluate Evidence Completeness
        if hasattr(evidence_inventory, "is_complete"):
            evidence_complete = evidence_inventory.is_complete()
        elif hasattr(evidence_inventory, "summary"):
            evidence_complete = evidence_inventory.summary.get("not_established", 0) == 0
        else:
            evidence_complete = False

        # 3. Derive Gate Statuses (Section 13 & 25 Reclassification Rule)
        # Architecture closure is blocked ONLY IF unresolved architecture-level semantic defects exist
        if len(unresolved_arch_defects) > 0:
            arch_status = "OPEN"
            rem_status = "REQUIRED"
        else:
            arch_status = "CLOSED"
            rem_status = "NONE_OR_VERIFIED_COMPLETE"

        # Global provisioning readiness requires zero blockers AND complete evidence
        if has_blockers or not evidence_complete:
            global_readiness = "BLOCKED"
        else:
            global_readiness = "READY"

        # Prelive runtime action is strictly NONE (Section 14)
        runtime_action = "NONE"

        engine_hash = self.compute_engine_hash()
        finding_hash = finding_registry.compute_registry_hash()
        if hasattr(evidence_inventory, "compute_inventory_hash"):
            evidence_hash = evidence_inventory.compute_inventory_hash()
        elif hasattr(evidence_inventory, "manifest_hash"):
            evidence_hash = evidence_inventory.manifest_hash
        else:
            evidence_hash = "0" * 64

        if not decision_id:
            det_seed = {
                "epoch": self.epoch,
                "sequence": sequence,
                "previous_decision_hash": previous_decision_hash,
                "policy_hash": self.policy_hash,
                "finding_registry_hash": finding_hash,
                "evidence_inventory_hash": evidence_hash,
                "contract_set_hash": contract_set_hash,
                "theorem_state_hash": theorem_state_hash,
            }
            det_suffix = sha256_canonical(det_seed)[:12]
            decision_id = f"urn:arx:decision:{self.epoch}:{sequence}:{det_suffix}"

        body = {
            "decision_id": decision_id,
            "governance_epoch": self.epoch,
            "decision_sequence": sequence,
            "previous_decision_hash": previous_decision_hash,
            "policy_hash": self.policy_hash,
            "policy_ratification_hash": self.policy_ratification_hash,
            "finding_registry_hash": finding_hash,
            "evidence_inventory_hash": evidence_hash,
            "contract_set_hash": contract_set_hash,
            "theorem_state_hash": theorem_state_hash,
            "derivation_engine_source_sha": self.source_sha,
            "derivation_engine_hash": engine_hash,
            "derivation_engine_version": self.version,
            "normalization_version": ARX_GOV_NORMALIZE_VERSION,
            "canonicalization_version": JCS_VERSION,
            "architecture_closure_status": arch_status,
            "remediation_status": rem_status,
            "global_provisioning_readiness": global_readiness,
            "theorem_readiness": theorem_readiness,
            "runtime_action": runtime_action,
            "derived_at": derived_at,
            "decision_validity": "CURRENT",
        }
        dec_hash = sha256_canonical(body)

        # Check sequence anti-equivocation
        seq_key = (self.epoch, sequence)
        if seq_key in self._registered_committed_decisions:
            if self._registered_committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash {self._registered_committed_decisions[seq_key]}."
                )
        if self.enforce_sequence_uniqueness:
            if seq_key in self._committed_decisions and self._committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash."
                )
            self._committed_decisions[seq_key] = dec_hash

        return DecisionSnapshot(
            decision_id=decision_id,
            governance_epoch=self.epoch,
            decision_sequence=sequence,
            previous_decision_hash=previous_decision_hash,
            policy_hash=self.policy_hash,
            policy_ratification_hash=self.policy_ratification_hash,
            finding_registry_hash=finding_hash,
            evidence_inventory_hash=evidence_hash,
            contract_set_hash=contract_set_hash,
            theorem_state_hash=theorem_state_hash,
            derivation_engine_source_sha=self.source_sha,
            derivation_engine_hash=engine_hash,
            derivation_engine_version=self.version,
            normalization_version=ARX_GOV_NORMALIZE_VERSION,
            canonicalization_version=JCS_VERSION,
            architecture_closure_status=arch_status,
            remediation_status=rem_status,
            global_provisioning_readiness=global_readiness,
            theorem_readiness=theorem_readiness,
            runtime_action=runtime_action,
            derived_at=derived_at,
            decision_validity="CURRENT",
            decision_hash=dec_hash,
            # Fallback bindings for backward compatibility with legacy callers
            source_freeze_sha=self.source_sha,
            evidence_manifest_hash=evidence_hash,
            dependency_resolution_hash=sha256_canonical({"legacy": True, "readiness": global_readiness}),
            is_canonical=False,
            canonical_eligibility="PASS",
            authority_eligibility="AUTHORITATIVE",
        )

    def evaluate_canonical(
        self,
        context: Optional[Any] = None,
        *,
        finding_registry: Optional[FindingRegistry] = None,
        evidence_manifest: Optional[CanonicalEvidenceManifest] = None,
        claim_registry: Optional[ClaimRegistry] = None,
        compiled_dependency_policy: Optional[CompiledDependencyPolicy] = None,
        fact_context: Optional[FactContext] = None,
        sequence: int = 1,
        previous_decision_hash: str = "0" * 64,
        source_freeze_sha: Optional[str] = None,
        contract_set_hash: Optional[str] = None,
        theorem_state_hash: Optional[str] = None,
        derived_at: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> DecisionSnapshot:
        """
        Authoritative Canonical Decision Derivation (Sections 14, 15, 16, 35).
        Consumes: Canonical Evidence -> Claims -> Context -> Policy DAG -> Admitted Findings.
        """
        if context is not None:
            # Mechanically verify context hash
            context_dict = context.to_dict()
            ctx_body = {k: v for k, v in context_dict.items() if k != "decision_context_hash"}
            if sha256_canonical(ctx_body) != context.decision_context_hash:
                raise CallerConstructedCanonicalContextRejectedError(
                    "CALLER_CONSTRUCTED_CANONICAL_CONTEXT_ACCEPTED == 0: Canonical context hash does not match canonical recomputation."
                )

            finding_registry = context.finding_registry
            evidence_manifest = context.evidence_manifest
            claim_registry = ClaimRegistry(claims=list(context.derived_claims.values()))
            compiled_dependency_policy = context.compiled_dependency_policy
            fact_context = context.fact_context
            source_freeze_sha = context.source_freeze_sha
            decision_context_hash = context.context_hash
            finding_identity_registry_hash = context.finding_identity_registry_hash
            finding_admission_registry_hash = context.finding_admission_registry_hash
            active_blocker_set_hash = context.active_blocker_set_hash
            evaluator_bundle_hash = context.evaluator_bundle_hash
            candidate_freeze_file_sha256 = context.candidate_freeze_file_sha256
            candidate_semantic_closure_hash = context.candidate_semantic_closure_hash
            dep_res = context.dependency_resolution
        else:
            if not self.allow_caller_canonical_inputs:
                raise CallerConstructedCanonicalContextRejectedError(
                    "CALLER_CONSTRUCTED_CANONICAL_CONTEXT_ACCEPTED == 0: "
                    "evaluate_canonical requires an authoritative CanonicalDecisionContext."
                )
            if (
                finding_registry is None
                or evidence_manifest is None
                or claim_registry is None
                or compiled_dependency_policy is None
                or fact_context is None
            ):
                raise ValueError("All canonical inputs must be provided when evaluate_canonical is called without context.")
            decision_context_hash = None
            finding_identity_registry_hash = finding_registry.compute_identity_registry_hash() if hasattr(finding_registry, "compute_identity_registry_hash") else None
            finding_admission_registry_hash = finding_registry.compute_admission_registry_hash() if hasattr(finding_registry, "compute_admission_registry_hash") else None
            active_blocker_set_hash = finding_registry.compute_active_blocker_set_hash() if hasattr(finding_registry, "compute_active_blocker_set_hash") else None
            evaluator_bundle_hash = None
            candidate_freeze_file_sha256 = None
            candidate_semantic_closure_hash = None
            dep_res = resolve_dependencies(compiled_dependency_policy, fact_context)
        """
        Authoritative Canonical Decision Derivation (Sections 14, 15, 16, 35).
        Consumes: Canonical Evidence -> Claims -> Context -> Policy DAG -> Admitted Findings.
        """
        if not derived_at:
            derived_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not contract_set_hash:
            contract_set_hash = compute_canonical_contract_set_hash()

        if not theorem_state_hash:
            theorem_state_hash = compute_canonical_theorem_state_hash()

        # 1. Evidence Manifest Validation
        canonical_eligibility = "PASS"
        authority_eligibility = "AUTHORITATIVE"
        for rec in evidence_manifest.records:
            if rec.provenance_class in {"SYNTHETIC_FIXTURE", "TEST_DOUBLE"}:
                canonical_eligibility = "FAIL"
                authority_eligibility = "NON_AUTHORITATIVE"
                break
            if rec.source and any(p in rec.source.lower() for p in ["test_persona", "mock_operator", "synthetic"]):
                canonical_eligibility = "FAIL"
                authority_eligibility = "NON_AUTHORITATIVE"
                break

        # 2. Finding Registry Validation
        # Invariant: ONLY admitted findings enter decision computation
        admitted_findings = [f for f in finding_registry.findings if f.admission_status == "ADMITTED"]
        active_blockers = [f for f in admitted_findings if f.is_blocking]
        unresolved_arch_defects = [
            f for f in admitted_findings
            if f.reclassified_layer == "ARCHITECTURE_DEFECT"
            and f.lifecycle_status not in [FindingStatus.VERIFIED_RESOLVED, FindingStatus.DISPROVED]
        ]

        if len(unresolved_arch_defects) > 0:
            arch_status = "OPEN"
            rem_status = "REQUIRED"
        else:
            arch_status = "CLOSED"
            rem_status = "NONE_OR_VERIFIED_COMPLETE"

        # 3. Declarative DAG Dependency Resolution
        dep_res = resolve_dependencies(compiled_dependency_policy, fact_context)
        dep_res_hash = dep_res.dependency_resolution_hash
        pred_ctx_hash = dep_res.predicate_context_hash
        dep_pol_hash = compiled_dependency_policy.policy_hash
        pred_set_hash = sha256_canonical([p.predicate_hash for p in compiled_dependency_policy.predicates.values()])

        # Theorem readiness derivation
        theorem_readiness = {"A": "READY", "B": "READY", "C": "READY", "D": "READY", "E": "BLOCKED"}
        for b in active_blockers:
            for blk in b.derived_blocks:
                if blk in ["THEOREM_A", "PROVISIONING"]:
                    theorem_readiness["A"] = "BLOCKED"
                if blk in ["THEOREM_B", "PROVISIONING"]:
                    theorem_readiness["B"] = "BLOCKED"
                if blk in ["THEOREM_C", "PROVISIONING"]:
                    theorem_readiness["C"] = "BLOCKED"
                if blk in ["THEOREM_D", "PROVISIONING"]:
                    theorem_readiness["D"] = "BLOCKED"

        for tgt_id, status in dep_res.target_statuses.items():
            if status != "SATISFIED":
                if "theorem_a" in tgt_id.lower() or "output_a" in tgt_id.lower():
                    theorem_readiness["A"] = "BLOCKED"
                if "theorem_b" in tgt_id.lower() or "output_b" in tgt_id.lower():
                    theorem_readiness["B"] = "BLOCKED"
                if "theorem_c" in tgt_id.lower() or "output_c" in tgt_id.lower():
                    theorem_readiness["C"] = "BLOCKED"
                if "theorem_d" in tgt_id.lower() or "output_d" in tgt_id.lower():
                    theorem_readiness["D"] = "BLOCKED"

        # Global provisioning readiness from DAG + Blockers + Evidence Completeness
        has_dag_unresolved = len(dep_res.unresolved_target_ids) > 0
        has_blockers = len(active_blockers) > 0
        evidence_incomplete = (
            evidence_manifest.summary.get("not_established", 0) > 0
            or evidence_manifest.summary.get("stale", 0) > 0
            or evidence_manifest.summary.get("revoked", 0) > 0
        )

        if has_blockers or has_dag_unresolved or evidence_incomplete:
            global_readiness = "BLOCKED"
            provisioning_auth_status = "NOT_ISSUABLE"
        else:
            global_readiness = "READY"
            provisioning_auth_status = "ISSUABLE"

        runtime_action = "NONE"
        engine_hash = self.compute_engine_hash()
        finding_hash = finding_registry.compute_registry_hash()
        evidence_manifest_hash = evidence_manifest.manifest_hash
        claim_registry_hash = claim_registry.compute_registry_hash()

        freeze_sha = source_freeze_sha or self.source_sha

        if not decision_id:
            det_seed = {
                "epoch": self.epoch,
                "sequence": sequence,
                "previous_decision_hash": previous_decision_hash,
                "source_freeze_sha": freeze_sha,
                "evidence_manifest_hash": evidence_manifest_hash,
                "claim_registry_hash": claim_registry_hash,
                "finding_registry_hash": finding_hash,
                "dependency_policy_hash": dep_pol_hash,
                "dependency_resolution_hash": dep_res_hash,
            }
            if decision_context_hash:
                det_seed["decision_context_hash"] = decision_context_hash
            det_suffix = sha256_canonical(det_seed)[:12]
            decision_id = f"urn:arx:decision:{self.epoch}:{sequence}:{det_suffix}"

        canonical_body = {
            "decision_id": decision_id,
            "governance_epoch": self.epoch,
            "decision_sequence": sequence,
            "previous_decision_hash": previous_decision_hash,
            "policy_hash": self.policy_hash,
            "policy_ratification_hash": self.policy_ratification_hash,
            "source_freeze_sha": freeze_sha,
            "evidence_manifest_hash": evidence_manifest_hash,
            "claim_registry_hash": claim_registry_hash,
            "finding_registry_hash": finding_hash,
            "dependency_policy_hash": dep_pol_hash,
            "predicate_set_hash": pred_set_hash,
            "predicate_context_hash": pred_ctx_hash,
            "dependency_resolution_hash": dep_res_hash,
            "contract_set_hash": contract_set_hash,
            "theorem_state_hash": theorem_state_hash,
            "derivation_engine_source_sha": self.source_sha,
            "derivation_engine_hash": engine_hash,
            "derivation_engine_version": self.version,
            "normalization_version": ARX_GOV_NORMALIZE_VERSION,
            "canonicalization_version": JCS_VERSION,
            "architecture_closure_status": arch_status,
            "remediation_status": rem_status,
            "global_provisioning_readiness": global_readiness,
            "provisioning_authorization_status": provisioning_auth_status,
            "theorem_readiness": theorem_readiness,
            "runtime_action": runtime_action,
            "derived_at": derived_at,
            "decision_validity": "CURRENT",
            "canonical_eligibility": canonical_eligibility,
            "authority_eligibility": authority_eligibility,
        }
        if decision_context_hash:
            canonical_body["decision_context_hash"] = decision_context_hash
            canonical_body["finding_identity_registry_hash"] = finding_identity_registry_hash
            canonical_body["finding_admission_registry_hash"] = finding_admission_registry_hash
            canonical_body["active_blocker_set_hash"] = active_blocker_set_hash
            canonical_body["evaluator_bundle_hash"] = evaluator_bundle_hash
            canonical_body["candidate_freeze_file_sha256"] = candidate_freeze_file_sha256
            canonical_body["candidate_semantic_closure_hash"] = candidate_semantic_closure_hash

        dec_hash = sha256_canonical(canonical_body)

        # Check sequence anti-equivocation
        seq_key = (self.epoch, sequence)
        if seq_key in self._registered_committed_decisions:
            if self._registered_committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash {self._registered_committed_decisions[seq_key]}."
                )
        if self.enforce_sequence_uniqueness:
            if seq_key in self._committed_decisions and self._committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash."
                )
            self._committed_decisions[seq_key] = dec_hash

        return DecisionSnapshot(
            decision_id=decision_id,
            governance_epoch=self.epoch,
            decision_sequence=sequence,
            previous_decision_hash=previous_decision_hash,
            policy_hash=self.policy_hash,
            policy_ratification_hash=self.policy_ratification_hash,
            finding_registry_hash=finding_hash,
            evidence_inventory_hash=evidence_manifest_hash,
            contract_set_hash=contract_set_hash,
            theorem_state_hash=theorem_state_hash,
            derivation_engine_source_sha=self.source_sha,
            derivation_engine_hash=engine_hash,
            derivation_engine_version=self.version,
            normalization_version=ARX_GOV_NORMALIZE_VERSION,
            canonicalization_version=JCS_VERSION,
            architecture_closure_status=arch_status,
            remediation_status=rem_status,
            global_provisioning_readiness=global_readiness,
            theorem_readiness=theorem_readiness,
            runtime_action=runtime_action,
            derived_at=derived_at,
            decision_validity="CURRENT",
            decision_hash=dec_hash,
            source_freeze_sha=freeze_sha,
            evidence_manifest_hash=evidence_manifest_hash,
            claim_registry_hash=claim_registry_hash,
            dependency_policy_hash=dep_pol_hash,
            predicate_set_hash=pred_set_hash,
            predicate_context_hash=pred_ctx_hash,
            dependency_resolution_hash=dep_res_hash,
            provisioning_authorization_status=provisioning_auth_status,
            is_canonical=True,
            canonical_eligibility=canonical_eligibility,
            authority_eligibility=authority_eligibility,
            decision_context_hash=decision_context_hash,
            finding_identity_registry_hash=finding_identity_registry_hash,
            finding_admission_registry_hash=finding_admission_registry_hash,
            active_blocker_set_hash=active_blocker_set_hash,
            evaluator_bundle_hash=evaluator_bundle_hash,
            candidate_freeze_file_sha256=candidate_freeze_file_sha256,
            candidate_semantic_closure_hash=candidate_semantic_closure_hash,
        )


def verify_legacy_dag_parity(
    finding_registry: FindingRegistry,
    evidence_inventory: EvidenceInventory,
    compiled_dependency_policy: CompiledDependencyPolicy,
    fact_context: FactContext,
    evidence_manifest: CanonicalEvidenceManifest,
    claim_registry: ClaimRegistry,
) -> Dict[str, Any]:
    """
    Compares outputs between legacy imperative derivation and declarative DAG derivation
    over identical canonical inputs (Section 17 & 44).
    """
    engine = GovernanceDecisionEngine()
    legacy_snap = engine.evaluate(finding_registry, evidence_inventory)
    dag_snap = engine.evaluate_canonical(
        finding_registry=finding_registry,
        evidence_manifest=evidence_manifest,
        claim_registry=claim_registry,
        compiled_dependency_policy=compiled_dependency_policy,
        fact_context=fact_context,
    )

    mismatches = 0
    diffs = []

    fields_to_compare = [
        "architecture_closure_status",
        "remediation_status",
        "global_provisioning_readiness",
        "runtime_action",
    ]
    for f in fields_to_compare:
        val_leg = getattr(legacy_snap, f)
        val_dag = getattr(dag_snap, f)
        if val_leg != val_dag:
            mismatches += 1
            diffs.append(f"{f}: legacy={val_leg} != dag={val_dag}")

    # Compare theorem_readiness
    for th in ["A", "B", "C", "D", "E"]:
        leg_th = legacy_snap.theorem_readiness.get(th)
        dag_th = dag_snap.theorem_readiness.get(th)
        if leg_th != dag_th:
            mismatches += 1
            diffs.append(f"theorem_{th}: legacy={leg_th} != dag={dag_th}")

    return {
        "mismatches": mismatches,
        "diffs": diffs,
        "parity_pass": mismatches == 0,
        "legacy_outputs": {
            "architecture_closure_status": legacy_snap.architecture_closure_status,
            "remediation_status": legacy_snap.remediation_status,
            "global_provisioning_readiness": legacy_snap.global_provisioning_readiness,
            "runtime_action": legacy_snap.runtime_action,
            "theorem_readiness": legacy_snap.theorem_readiness,
        },
        "dag_outputs": {
            "architecture_closure_status": dag_snap.architecture_closure_status,
            "remediation_status": dag_snap.remediation_status,
            "global_provisioning_readiness": dag_snap.global_provisioning_readiness,
            "runtime_action": dag_snap.runtime_action,
            "theorem_readiness": dag_snap.theorem_readiness,
        },
    }


class ReferenceDecisionEngine:
    """
    Independent Reference Evaluator (Section 34).
    Uses alternate derivation logic (declarative filter pipeline) to verify parity:
    PRIMARY_ENGINE_RESULT == REFERENCE_ENGINE_RESULT.
    """

    def __init__(
        self,
        policy_hash: str = "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
        policy_ratification_hash: str = "8a9f0e1d2c3b4a596877869504132435465768798091a2b3c4d5e6f7a8b9c0d1",
        epoch: str = "epoch-gov-001",
    ):
        self.policy_hash = policy_hash
        self.policy_ratification_hash = policy_ratification_hash
        self.epoch = epoch

    def evaluate_gates(
        self,
        finding_registry: FindingRegistry,
        evidence_inventory: EvidenceInventory,
    ) -> Dict[str, Any]:
        # Declarative functional pipeline
        blockers = [
            f for f in finding_registry.findings
            if f.severity in ["CRITICAL", "HIGH", "BLOCKING"]
            and f.lifecycle_status not in [FindingStatus.VERIFIED_RESOLVED, FindingStatus.DISPROVED, FindingStatus.WAIVED]
        ]
        arch_defects = [
            f for f in finding_registry.findings
            if f.reclassified_layer == "ARCHITECTURE_DEFECT"
            and f.lifecycle_status not in [FindingStatus.VERIFIED_RESOLVED, FindingStatus.DISPROVED]
        ]
        if hasattr(evidence_inventory, "is_complete"):
            ev_complete = evidence_inventory.is_complete()
        elif hasattr(evidence_inventory, "summary"):
            ev_complete = evidence_inventory.summary.get("not_established", 0) == 0
        else:
            ev_complete = False

        arch_blocked = len(arch_defects) > 0
        prov_blocked = bool(blockers) or not ev_complete
        return {
            "architecture_closure_status": "OPEN" if arch_blocked else "CLOSED",
            "remediation_status": "REQUIRED" if arch_blocked else "NONE_OR_VERIFIED_COMPLETE",
            "global_provisioning_readiness": "BLOCKED" if prov_blocked else "READY",
            "runtime_action": "NONE",
            "blocker_count": len(blockers),
            "unresolved_arch_defects": len(arch_defects),
            "evidence_complete": ev_complete,
        }
