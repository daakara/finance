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

    def __post_init__(self):
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    ):
        self.policy_hash = policy_hash
        self.policy_ratification_hash = policy_ratification_hash
        self.epoch = epoch
        self.version = ENGINE_VERSION
        self.source_sha = ENGINE_SOURCE_SHA

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
        evidence_complete = evidence_inventory.is_complete()

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
        evidence_hash = evidence_inventory.compute_inventory_hash()

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
        finding_registry: FindingRegistry,
        evidence_manifest: CanonicalEvidenceManifest,
        claim_registry: ClaimRegistry,
        compiled_dependency_policy: CompiledDependencyPolicy,
        fact_context: FactContext,
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
        dec_hash = sha256_canonical(canonical_body)

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
        ev_complete = evidence_inventory.is_complete()

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
