"""ARX VCP Normalized Authority Model, Oracle Class Derivation & Transition Ledger.

Sprint 2B Authority-Grade + Adjudication-Resolution Reconciliation Gate.
Decouples authority origin, evidence sufficiency, adjudication status, and authority status
into orthogonal dimensions. Derives oracle classes deterministically via policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AuthorityOrigin(str, Enum):
    """Identifies the institutional or synthetic origin of the adjudication."""
    EXTERNAL_INDEPENDENT = "EXTERNAL_INDEPENDENT"
    INTERNAL_GOVERNED = "INTERNAL_GOVERNED"
    SYNTHETIC_FORMAL = "SYNTHETIC_FORMAL"
    NONE = "NONE"


class EvidenceSufficiency(str, Enum):
    """Classifies the completeness of the evidentiary corpus supporting adjudication."""
    COMPLETE = "COMPLETE"
    LIMITED = "LIMITED"
    INSUFFICIENT = "INSUFFICIENT"


class AdjudicationStatus(str, Enum):
    """Records whether adjudication has reached definitive resolution."""
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class AuthorityStatus(str, Enum):
    """Lifecycle status of the authority determination."""
    ACTIVE = "ACTIVE"
    DISPUTED = "DISPUTED"
    SUPERSEDED = "SUPERSEDED"
    PENDING_REVIEW = "PENDING_REVIEW"
    RETIRED = "RETIRED"


class DerivedOracleClass(str, Enum):
    """Epistemic authority classification derived deterministically by policy."""
    GOLD = "GOLD"
    SILVER = "SILVER"
    INTERNAL_REFERENCE = "INTERNAL_REFERENCE"
    NONE = "NONE"


class SilverLimitationCode(str, Enum):
    """Governed reason codes explaining bounded limitations preventing Gold eligibility."""
    LIMITED_DOMAIN_SCOPE = "LIMITED_DOMAIN_SCOPE"
    CROSS_MARKET_GENERALIZATION = "CROSS_MARKET_GENERALIZATION"
    PARTIAL_SOURCE_AUTHORITY = "PARTIAL_SOURCE_AUTHORITY"
    TEMPORAL_PROVENANCE_LIMITED = "TEMPORAL_PROVENANCE_LIMITED"
    ADJUDICATION_ENVIRONMENT_LIMITED = "ADJUDICATION_ENVIRONMENT_LIMITED"
    EVIDENCE_RECONSTRUCTION = "EVIDENCE_RECONSTRUCTION"
    SUPPORTING_PREDICATE_LIMITED = "SUPPORTING_PREDICATE_LIMITED"
    OTHER_GOVERNED_LIMITATION = "OTHER_GOVERNED_LIMITATION"


class AdjudicationSource(str, Enum):
    """Entity or mechanism performing adjudication."""
    EXTERNAL_INDEPENDENT_HUMAN = "EXTERNAL_INDEPENDENT_HUMAN"
    INTERNAL_HUMAN = "INTERNAL_HUMAN"
    FORMAL_POLICY_EVALUATOR = "FORMAL_POLICY_EVALUATOR"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    MIXED = "MIXED"
    NONE = "NONE"


class KnownAtProvenance(str, Enum):
    """Classifies the provenance of point-in-time known_at availability timestamps."""
    NATIVE_SOURCE_TIMESTAMP = "NATIVE_SOURCE_TIMESTAMP"
    IMMUTABLE_ARCHIVED_SNAPSHOT = "IMMUTABLE_ARCHIVED_SNAPSHOT"
    GOVERNED_INGEST_TIMESTAMP = "GOVERNED_INGEST_TIMESTAMP"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    RECONSTRUCTED = "RECONSTRUCTED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class ExpectedDomainResult(str, Enum):
    """Canonical machine enum for expected domain evaluation outcome."""
    QUALIFIED = "QUALIFIED"
    NON_QUALIFIED = "NON_QUALIFIED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNRESOLVED = "UNRESOLVED"

    @classmethod
    def from_vcp_classification(cls, vcp_status: str) -> ExpectedDomainResult:
        """Maps string classification to canonical ExpectedDomainResult enum."""
        mapping = {
            "VCP_QUALIFIED": cls.QUALIFIED,
            "QUALIFIED": cls.QUALIFIED,
            "VCP_NON_QUALIFIED": cls.NON_QUALIFIED,
            "NON_QUALIFIED": cls.NON_QUALIFIED,
            "VCP_INSUFFICIENT_DATA": cls.INSUFFICIENT_DATA,
            "INSUFFICIENT_DATA": cls.INSUFFICIENT_DATA,
            "NOT_APPLICABLE": cls.NOT_APPLICABLE,
            "VCP_UNRESOLVED": cls.UNRESOLVED,
            "UNRESOLVED": cls.UNRESOLVED,
        }
        if vcp_status not in mapping:
            raise ValueError(f"UNMAPPED_EXPECTED_RESULT_VALUE: '{vcp_status}' cannot be mapped to ExpectedDomainResult")
        return mapping[vcp_status]

    def to_pass_fail(self) -> str:
        """Explicit mapping layer converting canonical domain result to binary display indicator."""
        if self == ExpectedDomainResult.QUALIFIED:
            return "PASS"
        elif self == ExpectedDomainResult.NON_QUALIFIED:
            return "FAIL"
        elif self == ExpectedDomainResult.INSUFFICIENT_DATA:
            return "INSUFFICIENT_DATA"
        elif self == ExpectedDomainResult.NOT_APPLICABLE:
            return "NOT_APPLICABLE"
        else:
            return "UNRESOLVED"


# Policy Invariant Flags:
GOLD_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION: bool = True
SILVER_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION: bool = True
INTERNAL_REFERENCE_REQUIRES_EXTERNAL_ADJUDICATION: bool = False
SYNTHETIC_ADJUDICATION_CAN_PRODUCE_GOLD: bool = False
SYNTHETIC_ADJUDICATION_CAN_PRODUCE_SILVER: bool = False
INTERNAL_ADJUDICATION_CAN_PRODUCE_GOLD: bool = False
INTERNAL_ADJUDICATION_CAN_PRODUCE_SILVER: bool = False
SYNTHETIC_ADJUDICATION_CAN_PRODUCE_INTERNAL_REFERENCE: bool = True
INTERNAL_REFERENCE_COUNTS_AS_INDEPENDENT_ORACLE: bool = False
INTERNAL_REFERENCE_COUNTS_AS_ENGINEERING_REFERENCE: bool = True
SILVER_HARD_ORACLE_ELIGIBLE: bool = False
SILVER_SOFT_CONFORMANCE_ELIGIBLE: bool = True


@dataclass(frozen=True)
class OracleAuthorityTransition:
    """Immutable ledger record of authority grade succession."""
    transition_id: str
    case_id: str
    predecessor_class: DerivedOracleClass
    successor_class: DerivedOracleClass
    predecessor_authority_status: AuthorityStatus
    successor_authority_status: AuthorityStatus
    transition_reason: str
    authority_evidence_hash: str
    effective_at: str


def derive_oracle_class(
    adjudication_status: AdjudicationStatus,
    authority_origin: AuthorityOrigin,
    evidence_sufficiency: EvidenceSufficiency,
    authority_status: AuthorityStatus,
    adjudication_source: AdjudicationSource,
    silver_limitations: Tuple[SilverLimitationCode, ...] = (),
    external_adjudication_verified: bool = False,
    material_disagreements_count: int = 0,
    normative_predicates_resolved: bool = True,
    final_classification_resolved: bool = True,
) -> DerivedOracleClass:
    """Deterministically derives the authority class for a case under frozen policy rules.

    Never permits manual assignment or heuristic promotion without qualifying criteria.
    """
    # Rule 1: Any unresolved adjudication derives NONE
    if adjudication_status != AdjudicationStatus.RESOLVED:
        return DerivedOracleClass.NONE

    # Rule 2: Inactive or disputed authority derives NONE for active conformance
    if authority_status != AuthorityStatus.ACTIVE:
        return DerivedOracleClass.NONE

    # Rule 3: GOLD derivation criteria (Strict fail-closed external independence)
    if (
        authority_origin == AuthorityOrigin.EXTERNAL_INDEPENDENT
        and evidence_sufficiency == EvidenceSufficiency.COMPLETE
        and adjudication_source == AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN
        and external_adjudication_verified
        and material_disagreements_count == 0
        and normative_predicates_resolved
        and final_classification_resolved
    ):
        return DerivedOracleClass.GOLD

    # Rule 4: SILVER derivation criteria (External independent with governed limitations)
    if (
        authority_origin == AuthorityOrigin.EXTERNAL_INDEPENDENT
        and evidence_sufficiency == EvidenceSufficiency.LIMITED
        and adjudication_source in (AdjudicationSource.EXTERNAL_INDEPENDENT_HUMAN, AdjudicationSource.MIXED)
        and external_adjudication_verified
        and len(silver_limitations) > 0
        and material_disagreements_count == 0
        and normative_predicates_resolved
        and final_classification_resolved
    ):
        return DerivedOracleClass.SILVER

    # Rule 5: INTERNAL REFERENCE criteria (Internally governed or synthetic formal resolution)
    if (
        authority_origin in (AuthorityOrigin.INTERNAL_GOVERNED, AuthorityOrigin.SYNTHETIC_FORMAL)
        and evidence_sufficiency in (EvidenceSufficiency.COMPLETE, EvidenceSufficiency.LIMITED)
        and adjudication_source in (AdjudicationSource.SYNTHETIC_FIXTURE, AdjudicationSource.INTERNAL_HUMAN, AdjudicationSource.FORMAL_POLICY_EVALUATOR)
        and normative_predicates_resolved
        and final_classification_resolved
    ):
        return DerivedOracleClass.INTERNAL_REFERENCE

    # Fallback default: NONE
    return DerivedOracleClass.NONE


AUTHORITY_MODEL_ID = "ARX_VCP_AUTHORITY_MODEL"
AUTHORITY_MODEL_VERSION = "1.0.0"


def compute_authority_model_hash() -> str:
    """Computes deterministic hash over ARX_VCP_AUTHORITY_MODEL specification."""
    spec = {
        "model_id": AUTHORITY_MODEL_ID,
        "version": AUTHORITY_MODEL_VERSION,
        "classes": [c.value for c in DerivedOracleClass],
        "origins": [o.value for o in AuthorityOrigin],
        "sufficiencies": [s.value for s in EvidenceSufficiency],
        "statuses": [st.value for st in AuthorityStatus],
        "sources": [src.value for src in AdjudicationSource],
        "provenances": [p.value for p in KnownAtProvenance],
        "limitations": [l.value for l in SilverLimitationCode],
        "expected_results": [r.value for r in ExpectedDomainResult],
        "gold_requires_external": GOLD_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
        "silver_requires_external": SILVER_REQUIRES_EXTERNAL_INDEPENDENT_ADJUDICATION,
        "internal_reference_requires_external": INTERNAL_REFERENCE_REQUIRES_EXTERNAL_ADJUDICATION,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
