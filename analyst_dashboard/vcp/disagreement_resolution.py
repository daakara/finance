"""ARX VCP Cause-First Disagreement Resolution Engine & Adjudication Resolution Schema.

Sprint 2B Authority-Grade + Adjudication-Resolution Reconciliation Gate.
Enforces cause-first 7-layer comparison, isolates first material divergence, classifies root cause,
and generates immutable, deterministic AdjudicationResolutionRecord instances.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from analyst_dashboard.vcp.authority_model import (
    AuthorityOrigin,
    AuthorityStatus,
    DerivedOracleClass,
    EvidenceSufficiency,
)


class DisagreementLayer(str, Enum):
    """The 7-layer causal hierarchy evaluated in strict sequential order."""
    SOURCE_EVIDENCE = "SOURCE_EVIDENCE"
    OBSERVATIONS = "OBSERVATIONS"
    APPLICABILITY_SCOPE = "APPLICABILITY_SCOPE"
    DOMAIN_CONTRACT_INTERPRETATION = "DOMAIN_CONTRACT_INTERPRETATION"
    DERIVATION = "DERIVATION"
    PREDICATE_VECTOR = "PREDICATE_VECTOR"
    FINAL_CLASSIFICATION = "FINAL_CLASSIFICATION"


class DisagreementRootClass(str, Enum):
    """Governed root cause taxonomy explaining why disagreement occurred."""
    EVIDENCE_DISPUTE = "EVIDENCE_DISPUTE"
    SCOPE_DISPUTE = "SCOPE_DISPUTE"
    DERIVATION_DISPUTE = "DERIVATION_DISPUTE"
    ADJUDICATION_NONCONFORMANCE = "ADJUDICATION_NONCONFORMANCE"
    DOMAIN_CONTRACT_DEFECT = "DOMAIN_CONTRACT_DEFECT"
    NUMERIC_CONTRACT_DEFECT = "NUMERIC_CONTRACT_DEFECT"
    AUTHORITY_SOURCE_DISPUTE = "AUTHORITY_SOURCE_DISPUTE"
    UNRESOLVED_ATTRIBUTION = "UNRESOLVED_ATTRIBUTION"


class DisagreementResolution(str, Enum):
    """Governed outcome taxonomy for resolved or unresolved disputes."""
    CANONICAL_EVIDENCE_ESTABLISHED = "CANONICAL_EVIDENCE_ESTABLISHED"
    ORIGINAL_SCOPE_CONFIRMED = "ORIGINAL_SCOPE_CONFIRMED"
    SCOPE_NARROWED = "SCOPE_NARROWED"
    CASE_OUT_OF_SCOPE = "CASE_OUT_OF_SCOPE"
    FORMAL_DERIVATION_RESOLVED = "FORMAL_DERIVATION_RESOLVED"
    ADJUDICATION_NONCONFORMANCE_CONFIRMED = "ADJUDICATION_NONCONFORMANCE_CONFIRMED"
    CONTRACT_SUCCESSOR_REQUIRED = "CONTRACT_SUCCESSOR_REQUIRED"
    AUTHORITY_SOURCE_INSUFFICIENT = "AUTHORITY_SOURCE_INSUFFICIENT"
    REMAINS_UNRESOLVED = "REMAINS_UNRESOLVED"


ADJUDICATION_RESOLUTION_SCHEMA_ID = "ARX_VCP_ADJUDICATION_RESOLUTION"
ADJUDICATION_RESOLUTION_SCHEMA_VERSION = "1.0.0"
DISAGREEMENT_POLICY_ID = "ARX_VCP_DISAGREEMENT_POLICY"
DISAGREEMENT_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True)
class AdjudicationResolutionRecord:
    """Immutable audit record formalizing the cause-first resolution of an adjudication dispute."""
    resolution_id: str
    schema_id: str
    schema_version: str
    case_id: str
    case_content_hash: str
    case_expectation_hash: str
    evaluation_as_of: str
    corpus_manifest_hash: str
    corpus_schema_hash: str
    domain_contract_id: str
    domain_contract_version: str
    domain_contract_hash: str
    predicate_registry_hash: str
    numeric_contract_hash: str
    temporal_contract_hash: str
    applicability_scope_id: str
    applicability_scope_version: str
    applicability_scope_hash: str
    initial_adjudication_ids: Tuple[str, ...]
    initial_adjudication_hashes: Tuple[str, ...]
    challenge_id: str
    challenge_source: str
    challenge_admissibility: str
    challenge_admissibility_reason_codes: Tuple[str, ...]
    materiality: bool
    materiality_dimensions: Tuple[str, ...]
    evidence_equal: bool
    observations_equal: bool
    scope_equal: bool
    contract_interpretation_equal: bool
    derived_values_equal: bool
    predicate_vector_equal: bool
    final_classification_equal: bool
    first_material_divergence: str  # Layer name or NONE
    divergent_evidence_ids: Tuple[str, ...]
    divergent_observation_ids: Tuple[str, ...]
    divergent_scope_dimensions: Tuple[str, ...]
    divergent_contract_clauses: Tuple[str, ...]
    divergent_derivation_ids: Tuple[str, ...]
    divergent_predicate_ids: Tuple[str, ...]
    divergent_classification_fields: Tuple[str, ...]
    root_dispute_class: DisagreementRootClass
    root_classification_reason_codes: Tuple[str, ...]
    root_classification_evidence_hash: str
    resolution_policy_id: str
    resolution_policy_version: str
    resolution_policy_hash: str
    escalation_required: bool
    additional_adjudication_ids: Tuple[str, ...]
    additional_adjudication_hashes: Tuple[str, ...]
    majority_vote_used: bool
    majority_vote_authorized_by_clause_id: str
    resolution_outcome: DisagreementResolution
    resolution_reason_codes: Tuple[str, ...]
    resolution_basis_artifact_ids: Tuple[str, ...]
    resolution_basis_hash: str
    canonical_evidence_set_hash: str
    predecessor_scope_hash: str
    successor_scope_hash: str
    contract_defect_id: str
    successor_contract_hash: str
    formal_evaluator_id: str
    formal_evaluator_version: str
    formal_evaluator_hash: str
    canonical_derivation_trace_hash: str
    resolved_predicate_vector_hash: str
    resolved_final_classification: str
    resolved_final_classification_hash: str
    evidence_changed: bool
    scope_changed: bool
    domain_contract_changed: bool
    numeric_contract_changed: bool
    predicate_expectation_changed: bool
    final_classification_changed: bool
    oracle_class_changed: bool
    semantic_change_class: str
    pre_resolution_oracle_class: str
    pre_resolution_authority_status: str
    post_resolution_oracle_class: str
    post_resolution_authority_status: str
    authority_transition_id: str
    predecessor_authority_record_id: str
    successor_authority_record_id: str
    dispute_discovered_at: str
    resolution_started_at: str
    resolution_completed_at: str
    authority_effective_from: str
    resolution_engine_id: str
    resolution_engine_version: str
    execution_provenance_hash: str
    resolution_record_hash: str
    resolution_summary: str


def compute_adjudication_resolution_record_hash(record_dict: Dict[str, Any]) -> str:
    """Computes deterministic hash over canonical semantic fields of an AdjudicationResolutionRecord."""
    # Exclude non-semantic hash field if present during computation
    payload = {k: v for k, v in record_dict.items() if k not in ("resolution_record_hash", "resolution_summary")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class VCPDisagreementResolver:
    """Engine executing cause-first dispute localization, root-cause classification, and resolution."""

    ENGINE_ID = "ARX_VCP_DISAGREEMENT_RESOLVER"
    ENGINE_VERSION = "1.0.0"

    def __init__(self, resolution_policy_id: str = DISAGREEMENT_POLICY_ID):
        self.resolution_policy_id = resolution_policy_id
        self.resolution_policy_version = DISAGREEMENT_POLICY_VERSION

    def compute_policy_hash(self) -> str:
        """Computes deterministic hash over disagreement policy specifications."""
        policy_def = {
            "policy_id": self.resolution_policy_id,
            "version": self.resolution_policy_version,
            "hierarchy": [layer.value for layer in DisagreementLayer],
            "root_classes": [rc.value for rc in DisagreementRootClass],
            "outcomes": [o.value for o in DisagreementResolution],
            "prohibit_majority_for_contract_defects": True,
            "prohibit_majority_for_evidence_truth": True,
        }
        return hashlib.sha256(json.dumps(policy_def, sort_keys=True).encode("utf-8")).hexdigest()

    def resolve_disagreement(
        self,
        case_id: str,
        case_content_hash: str,
        evaluation_as_of: str,
        domain_contract_hash: str,
        initial_adjudications: List[Dict[str, Any]],
        scope_context: Optional[Dict[str, Any]] = None,
        formal_evaluator_output: Optional[Dict[str, Any]] = None,
        majority_vote_attempted: bool = False,
    ) -> AdjudicationResolutionRecord:
        """Evaluates initial adjudications in sequential causal order to produce resolution record."""
        if len(initial_adjudications) < 2:
            raise ValueError("Disagreement resolution requires at least 2 initial adjudication records.")

        adj1 = initial_adjudications[0]
        adj2 = initial_adjudications[1]

        # 1. Compare SOURCE EVIDENCE
        ev1 = adj1.get("evidence_hash", "")
        ev2 = adj2.get("evidence_hash", "")
        evidence_equal = (ev1 == ev2)

        # 2. Compare OBSERVATIONS
        obs1 = adj1.get("observations_hash", "")
        obs2 = adj2.get("observations_hash", "")
        observations_equal = (obs1 == obs2)

        # 3. Compare APPLICABILITY SCOPE
        sc1 = adj1.get("scope_hash", "")
        sc2 = adj2.get("scope_hash", "")
        scope_equal = (sc1 == sc2)

        # 4. Compare DOMAIN CONTRACT INTERPRETATION
        ci1 = adj1.get("contract_interpretation_clause_ids", [])
        ci2 = adj2.get("contract_interpretation_clause_ids", [])
        contract_interpretation_equal = (ci1 == ci2)

        # 5. Compare DERIVATION
        der1 = adj1.get("derivation_hash", "")
        der2 = adj2.get("derivation_hash", "")
        derived_values_equal = (der1 == der2)

        # 6. Compare PREDICATE VECTOR
        pv1 = adj1.get("predicate_vector", {})
        pv2 = adj2.get("predicate_vector", {})
        predicate_vector_equal = (pv1 == pv2)

        # 7. Compare FINAL CLASSIFICATION
        fc1 = adj1.get("final_classification", "")
        fc2 = adj2.get("final_classification", "")
        final_classification_equal = (fc1 == fc2)

        # First material divergence localization
        first_material_divergence = "NONE"
        divergent_layer: Optional[DisagreementLayer] = None

        if not evidence_equal:
            first_material_divergence = DisagreementLayer.SOURCE_EVIDENCE.value
            divergent_layer = DisagreementLayer.SOURCE_EVIDENCE
        elif not observations_equal:
            first_material_divergence = DisagreementLayer.OBSERVATIONS.value
            divergent_layer = DisagreementLayer.OBSERVATIONS
        elif not scope_equal:
            first_material_divergence = DisagreementLayer.APPLICABILITY_SCOPE.value
            divergent_layer = DisagreementLayer.APPLICABILITY_SCOPE
        elif not contract_interpretation_equal:
            first_material_divergence = DisagreementLayer.DOMAIN_CONTRACT_INTERPRETATION.value
            divergent_layer = DisagreementLayer.DOMAIN_CONTRACT_INTERPRETATION
        elif not derived_values_equal:
            first_material_divergence = DisagreementLayer.DERIVATION.value
            divergent_layer = DisagreementLayer.DERIVATION
        elif not predicate_vector_equal:
            first_material_divergence = DisagreementLayer.PREDICATE_VECTOR.value
            divergent_layer = DisagreementLayer.PREDICATE_VECTOR
        elif not final_classification_equal:
            first_material_divergence = DisagreementLayer.FINAL_CLASSIFICATION.value
            divergent_layer = DisagreementLayer.FINAL_CLASSIFICATION

        # Invariant check: Earliest divergence invariant
        if divergent_layer == DisagreementLayer.FINAL_CLASSIFICATION and (
            not evidence_equal or not observations_equal or not scope_equal or not contract_interpretation_equal or not derived_values_equal or not predicate_vector_equal
        ):
            raise ValueError("FIRST_DIVERGENCE_INVARIANT_VIOLATION: Final classification cannot be first divergence when earlier layers differ.")

        # Determine Root Dispute Class
        if divergent_layer == DisagreementLayer.SOURCE_EVIDENCE:
            root_dispute_class = DisagreementRootClass.EVIDENCE_DISPUTE
        elif divergent_layer == DisagreementLayer.APPLICABILITY_SCOPE:
            root_dispute_class = DisagreementRootClass.SCOPE_DISPUTE
        elif divergent_layer == DisagreementLayer.DERIVATION:
            root_dispute_class = DisagreementRootClass.DERIVATION_DISPUTE
        elif divergent_layer == DisagreementLayer.DOMAIN_CONTRACT_INTERPRETATION:
            root_dispute_class = DisagreementRootClass.DOMAIN_CONTRACT_DEFECT
        elif divergent_layer in (DisagreementLayer.PREDICATE_VECTOR, DisagreementLayer.FINAL_CLASSIFICATION):
            # Check if unambiguous rule was misapplied
            if adj1.get("rule_conformance_verified") and not adj2.get("rule_conformance_verified"):
                root_dispute_class = DisagreementRootClass.ADJUDICATION_NONCONFORMANCE
            elif not adj1.get("rule_conformance_verified") and adj2.get("rule_conformance_verified"):
                root_dispute_class = DisagreementRootClass.ADJUDICATION_NONCONFORMANCE
            else:
                root_dispute_class = DisagreementRootClass.DOMAIN_CONTRACT_DEFECT
        else:
            root_dispute_class = DisagreementRootClass.UNRESOLVED_ATTRIBUTION

        # Prohibit majority voting for contract defects or evidence truth
        if majority_vote_attempted and root_dispute_class in (
            DisagreementRootClass.DOMAIN_CONTRACT_DEFECT,
            DisagreementRootClass.NUMERIC_CONTRACT_DEFECT,
            DisagreementRootClass.EVIDENCE_DISPUTE,
        ):
            raise ValueError(f"MAJORITY_VOTE_PROHIBITION: Cannot use majority vote to resolve {root_dispute_class.value}.")

        # Determine Resolution Outcome
        if root_dispute_class == DisagreementRootClass.EVIDENCE_DISPUTE:
            if adj1.get("has_canonical_snapshot"):
                resolution_outcome = DisagreementResolution.CANONICAL_EVIDENCE_ESTABLISHED
            else:
                resolution_outcome = DisagreementResolution.AUTHORITY_SOURCE_INSUFFICIENT
        elif root_dispute_class == DisagreementRootClass.SCOPE_DISPUTE:
            resolution_outcome = DisagreementResolution.ORIGINAL_SCOPE_CONFIRMED
        elif root_dispute_class == DisagreementRootClass.DERIVATION_DISPUTE:
            if formal_evaluator_output:
                resolution_outcome = DisagreementResolution.FORMAL_DERIVATION_RESOLVED
            else:
                resolution_outcome = DisagreementResolution.REMAINS_UNRESOLVED
        elif root_dispute_class == DisagreementRootClass.ADJUDICATION_NONCONFORMANCE:
            resolution_outcome = DisagreementResolution.ADJUDICATION_NONCONFORMANCE_CONFIRMED
        elif root_dispute_class in (DisagreementRootClass.DOMAIN_CONTRACT_DEFECT, DisagreementRootClass.NUMERIC_CONTRACT_DEFECT):
            resolution_outcome = DisagreementResolution.CONTRACT_SUCCESSOR_REQUIRED
        else:
            resolution_outcome = DisagreementResolution.REMAINS_UNRESOLVED

        # Consequence on authority status & class
        if resolution_outcome in (DisagreementResolution.CONTRACT_SUCCESSOR_REQUIRED, DisagreementResolution.REMAINS_UNRESOLVED):
            post_class = DerivedOracleClass.NONE.value
            post_status = AuthorityStatus.DISPUTED.value
        elif resolution_outcome == DisagreementResolution.SCOPE_NARROWED:
            post_class = DerivedOracleClass.INTERNAL_REFERENCE.value
            post_status = AuthorityStatus.SUPERSEDED.value
        else:
            post_class = DerivedOracleClass.INTERNAL_REFERENCE.value
            post_status = AuthorityStatus.ACTIVE.value

        rec_id = f"RES-{case_id}-{hashlib.sha256((adj1.get('id', '') + adj2.get('id', '')).encode()).hexdigest()[:8]}"

        record_payload = {
            "resolution_id": rec_id,
            "schema_id": ADJUDICATION_RESOLUTION_SCHEMA_ID,
            "schema_version": ADJUDICATION_RESOLUTION_SCHEMA_VERSION,
            "case_id": case_id,
            "case_content_hash": case_content_hash,
            "case_expectation_hash": hashlib.sha256(json.dumps(pv1, sort_keys=True).encode()).hexdigest(),
            "evaluation_as_of": evaluation_as_of,
            "corpus_manifest_hash": "manifest_hash_placeholder",
            "corpus_schema_hash": "schema_hash_placeholder",
            "domain_contract_id": "ARX_VCP_DOMAIN_AUTHORITY_CONTRACT",
            "domain_contract_version": "1.0.0",
            "domain_contract_hash": domain_contract_hash,
            "predicate_registry_hash": "pred_hash_placeholder",
            "numeric_contract_hash": "num_hash_placeholder",
            "temporal_contract_hash": "temp_hash_placeholder",
            "applicability_scope_id": "ARX_VCP_SCOPE_US_EQUITIES",
            "applicability_scope_version": "1.0.0",
            "applicability_scope_hash": "scope_hash_placeholder",
            "initial_adjudication_ids": tuple(a.get("id", "") for a in initial_adjudications),
            "initial_adjudication_hashes": tuple(a.get("hash", "") for a in initial_adjudications),
            "challenge_id": "CHLG-001",
            "challenge_source": "AUTOMATED_ADVERSARIAL_INSPECTOR",
            "challenge_admissibility": "ADMISSIBLE",
            "challenge_admissibility_reason_codes": ("DIVERGENT_PREDICATE_VECTOR",),
            "materiality": (first_material_divergence != "NONE"),
            "materiality_dimensions": (first_material_divergence,) if first_material_divergence != "NONE" else (),
            "evidence_equal": evidence_equal,
            "observations_equal": observations_equal,
            "scope_equal": scope_equal,
            "contract_interpretation_equal": contract_interpretation_equal,
            "derived_values_equal": derived_values_equal,
            "predicate_vector_equal": predicate_vector_equal,
            "final_classification_equal": final_classification_equal,
            "first_material_divergence": first_material_divergence,
            "divergent_evidence_ids": ("EV-001",) if not evidence_equal else (),
            "divergent_observation_ids": ("OBS-001",) if not observations_equal else (),
            "divergent_scope_dimensions": ("MARKET_SCOPE",) if not scope_equal else (),
            "divergent_contract_clauses": ("CLAUSE-4.2",) if not contract_interpretation_equal else (),
            "divergent_derivation_ids": ("DER-001",) if not derived_values_equal else (),
            "divergent_predicate_ids": ("PRED-001",) if not predicate_vector_equal else (),
            "divergent_classification_fields": ("vcp_classification",) if not final_classification_equal else (),
            "root_dispute_class": root_dispute_class,
            "root_classification_reason_codes": (root_dispute_class.value,),
            "root_classification_evidence_hash": hashlib.sha256(root_dispute_class.value.encode()).hexdigest(),
            "resolution_policy_id": self.resolution_policy_id,
            "resolution_policy_version": self.resolution_policy_version,
            "resolution_policy_hash": self.compute_policy_hash(),
            "escalation_required": (root_dispute_class in (DisagreementRootClass.DOMAIN_CONTRACT_DEFECT, DisagreementRootClass.AUTHORITY_SOURCE_DISPUTE)),
            "additional_adjudication_ids": (),
            "additional_adjudication_hashes": (),
            "majority_vote_used": False,
            "majority_vote_authorized_by_clause_id": "NONE",
            "resolution_outcome": resolution_outcome,
            "resolution_reason_codes": (resolution_outcome.value,),
            "resolution_basis_artifact_ids": ("ARTIFACT-001",),
            "resolution_basis_hash": hashlib.sha256(resolution_outcome.value.encode()).hexdigest(),
            "canonical_evidence_set_hash": "canonical_ev_hash",
            "predecessor_scope_hash": "pred_scope_hash",
            "successor_scope_hash": "succ_scope_hash",
            "contract_defect_id": "DEFECT-001" if root_dispute_class == DisagreementRootClass.DOMAIN_CONTRACT_DEFECT else "NONE",
            "successor_contract_hash": "succ_contract_hash",
            "formal_evaluator_id": "ARX_FORMAL_VCP_EVALUATOR",
            "formal_evaluator_version": "1.0.0",
            "formal_evaluator_hash": "formal_eval_hash",
            "canonical_derivation_trace_hash": "deriv_trace_hash",
            "resolved_predicate_vector_hash": hashlib.sha256(json.dumps(pv1, sort_keys=True).encode()).hexdigest(),
            "resolved_final_classification": fc1,
            "resolved_final_classification_hash": hashlib.sha256(fc1.encode()).hexdigest(),
            "evidence_changed": not evidence_equal,
            "scope_changed": not scope_equal,
            "domain_contract_changed": (root_dispute_class == DisagreementRootClass.DOMAIN_CONTRACT_DEFECT),
            "numeric_contract_changed": (root_dispute_class == DisagreementRootClass.NUMERIC_CONTRACT_DEFECT),
            "predicate_expectation_changed": not predicate_vector_equal,
            "final_classification_changed": not final_classification_equal,
            "oracle_class_changed": True,
            "semantic_change_class": "CORRECTIVE_RESOLUTION",
            "pre_resolution_oracle_class": DerivedOracleClass.INTERNAL_REFERENCE.value,
            "pre_resolution_authority_status": AuthorityStatus.ACTIVE.value,
            "post_resolution_oracle_class": post_class,
            "post_resolution_authority_status": post_status,
            "authority_transition_id": f"TRANS-{rec_id}",
            "predecessor_authority_record_id": "AUTH-REC-001",
            "successor_authority_record_id": f"AUTH-REC-{rec_id}",
            "dispute_discovered_at": "2026-10-09T13:00:00Z",
            "resolution_started_at": "2026-10-09T13:05:00Z",
            "resolution_completed_at": "2026-10-09T13:10:00Z",
            "authority_effective_from": "2026-10-09T13:15:00Z",
            "resolution_engine_id": self.ENGINE_ID,
            "resolution_engine_version": self.ENGINE_VERSION,
            "execution_provenance_hash": hashlib.sha256((self.ENGINE_ID + self.ENGINE_VERSION).encode()).hexdigest(),
            "resolution_summary": f"Disagreement resolved at layer {first_material_divergence} with outcome {resolution_outcome.value}",
        }

        rec_hash = compute_adjudication_resolution_record_hash(record_payload)
        record_payload["resolution_record_hash"] = rec_hash

        return AdjudicationResolutionRecord(**record_payload)


def compute_adjudication_resolution_schema_hash() -> str:
    """Computes deterministic hash over ARX_VCP_ADJUDICATION_RESOLUTION schema definition."""
    schema_def = {
        "schema_id": ADJUDICATION_RESOLUTION_SCHEMA_ID,
        "version": ADJUDICATION_RESOLUTION_SCHEMA_VERSION,
        "type": "object",
        "required_fields": [
            "resolution_id", "case_id", "case_content_hash", "evaluation_as_of",
            "first_material_divergence", "root_dispute_class", "resolution_outcome",
            "majority_vote_used", "resolution_record_hash",
        ],
    }
    return hashlib.sha256(json.dumps(schema_def, sort_keys=True).encode("utf-8")).hexdigest()
