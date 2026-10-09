"""
analyst_dashboard/security_master/semantic_governance.py

Semantic Dependency Hashing, Concrete Policy Contracts, Governance Bundle,
and Decision Hash Model for ARX Terminal Radar VCP Sprint 2A Final Closure Integrity Gate.

Enforces:
- Concrete policy contracts for all required non-direct bindings (Section 6 & 7).
- Policy artifact hash vs semantic projection hash (Section 8).
- Country / currency value evidence authority as DERIVED_POLICY (Section 9).
- Unified Governance Bundle with aggregate semantic hash (Section 10).
- Dependency-scoped decision governance and evidence dependency hashing (Section 11 & 12).
- Six distinct decision hashes closing over exact semantic inputs (Section 13 & 14).
- Impact Analysis Policy & Replay Scope Governor (Section 29).
- Reason taxonomy semantic projection vs artifact hash (Section 30).
- Run ID / timestamp exclusion from semantic input hashes (Section 31).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict

from .source_governance_models import (
    canonical_hash,
    canonical_json_dumps,
    ReasonCode,
    REASON_CODE_TAXONOMY_ID,
    REASON_CODE_TAXONOMY_VERSION,
    REASON_CODE_TAXONOMY_HASH,
)
from .source_governance_policy import (
    FieldAuthorityPolicyRegistry,
    SingleFieldPolicy,
)
from .requirement_catalog import (
    RequiredGovernanceConceptCatalog,
    REQUIRED_GOVERNANCE_CONCEPT_CATALOG_ID,
    REQUIRED_GOVERNANCE_CONCEPT_CATALOG_VERSION,
    REQUIRED_GOVERNANCE_CONCEPT_CATALOG_HASH,
)
from .required_field_registry import GovernedScope


# =====================================================================
# Concrete Policy Definition & Semantic Projection (Sections 6, 7 & 8)
# =====================================================================

class ConcretePolicyContract(BaseModel):
    """
    Formal contract definition exposing both artifact identity and semantic projection.
    """
    policy_id: str
    policy_version: str
    policy_name: str
    description: str
    authorities: List[str]
    precedence: List[str]
    admissibility_rules: List[str]
    normalization_rules: List[str] = Field(default_factory=list)
    missing_behavior: str
    conflict_behavior: str
    stale_behavior: Optional[str] = None
    unknown_behavior: Optional[str] = None
    temporal_behavior: Optional[str] = None
    derivation_inputs: Optional[List[str]] = None
    reason_codes: List[str] = Field(default_factory=list)
    effective_scopes: List[GovernedScope] = Field(default_factory=list)
    # Non-semantic documentation metadata
    documentation_prose: Optional[str] = None
    display_label: Optional[str] = None
    maintainer: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    def compute_semantic_projection(self) -> Dict[str, Any]:
        """
        Extracts pure semantic projection, excluding documentation prose, comments,
        display labels, and author metadata.
        """
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "authorities": sorted(self.authorities),
            "precedence": self.precedence,
            "admissibility_rules": sorted(self.admissibility_rules),
            "normalization_rules": sorted(self.normalization_rules),
            "missing_behavior": self.missing_behavior,
            "conflict_behavior": self.conflict_behavior,
            "stale_behavior": self.stale_behavior,
            "unknown_behavior": self.unknown_behavior,
            "temporal_behavior": self.temporal_behavior,
            "derivation_inputs": sorted(self.derivation_inputs) if self.derivation_inputs else None,
            "reason_codes": sorted(self.reason_codes),
            "effective_scopes": [s.value for s in sorted(self.effective_scopes, key=lambda x: x.value)],
        }

    def compute_semantic_hash(self) -> str:
        return canonical_hash(self.compute_semantic_projection())

    def compute_artifact_hash(self) -> str:
        return canonical_hash(self.model_dump())


# All 11 required non-direct and derived policy contracts (Section 7 & 9)
POLICY_CONTRACTS: Dict[str, ConcretePolicyContract] = {
    "POL_POPULATION_V1": ConcretePolicyContract(
        policy_id="POL_POPULATION_V1",
        policy_version="1.0.0",
        policy_name="Current Population Admission Policy",
        description="Governs raw provider snapshot admission into candidate population.",
        authorities=["ALPACA_ASSET_DIRECTORY"],
        precedence=["ALPACA_ASSET_DIRECTORY"],
        admissibility_rules=["SCHEMA_VALID", "FRESHNESS_WINDOW", "US_EQUITY_CLASS"],
        missing_behavior="EXCLUDE_FROM_POPULATION",
        conflict_behavior="S3_BLOCKING",
        effective_scopes=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
        documentation_prose="Evaluates active provider assets against current snapshot admissibility.",
    ),
    "POL_PROVIDER_ID_V1": ConcretePolicyContract(
        policy_id="POL_PROVIDER_ID_V1",
        policy_version="1.0.0",
        policy_name="Provider Instrument Identity Policy",
        description="Enforces 1:<=1 mapping of provider instrument UUID to canonical listing.",
        authorities=["ALPACA_ASSET_DIRECTORY"],
        precedence=["ALPACA_ASSET_DIRECTORY"],
        admissibility_rules=["NON_EMPTY_STRING", "VALID_UUID_FORMAT"],
        missing_behavior="FAIL_CLOSED_QUARANTINE",
        conflict_behavior="COLLISION_S3_BLOCKING",
        effective_scopes=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
    ),
    "POL_ISSUER_ID_V1": ConcretePolicyContract(
        policy_id="POL_ISSUER_ID_V1",
        policy_version="1.0.0",
        policy_name="Canonical Issuer Identity Policy",
        description="Establishes corporate entity identity from SEC CIK or OpenFIGI mapping.",
        authorities=["SEC_EDGAR", "OPENFIGI_V3_MAPPING"],
        precedence=["SEC_EDGAR", "OPENFIGI_V3_MAPPING"],
        admissibility_rules=["CIK_FORMAT", "ENTITY_RESOLVED"],
        missing_behavior="RESOLVE_AS_FALLBACK_SYNTHETIC",
        conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
        effective_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
    ),
    "POL_SECURITY_ID_V1": ConcretePolicyContract(
        policy_id="POL_SECURITY_ID_V1",
        policy_version="1.0.0",
        policy_name="Canonical Security Identity Policy",
        description="Establishes canonical security identity from Share Class FIGI or synthetic fallback.",
        authorities=["OPENFIGI_V3_MAPPING"],
        precedence=["OPENFIGI_V3_MAPPING"],
        admissibility_rules=["SHARE_CLASS_FIGI_FORMAT"],
        missing_behavior="RESOLVE_AS_FALLBACK_SYNTHETIC",
        conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
        effective_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
    ),
    "POL_LISTING_ID_V1": ConcretePolicyContract(
        policy_id="POL_LISTING_ID_V1",
        policy_version="1.0.0",
        policy_name="Canonical Listing Identity Policy",
        description="Synthesizes deterministic composite listing ID: LST_{MIC}_{SYMBOL}.",
        authorities=["ALPACA_ASSET_DIRECTORY", "OPENFIGI_V3_MAPPING"],
        precedence=["ALPACA_ASSET_DIRECTORY", "OPENFIGI_V3_MAPPING"],
        admissibility_rules=["NORMALIZED_SYMBOL", "VALID_OPERATING_MIC"],
        normalization_rules=["UPPERCASE", "STRIP_WHITESPACE", "DOT_SEPARATOR"],
        missing_behavior="FAIL_CLOSED_QUARANTINE",
        conflict_behavior="COLLISION_S3_BLOCKING",
        effective_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
    ),
    "POL_HISTORICAL_MEMBERSHIP_V1": ConcretePolicyContract(
        policy_id="POL_HISTORICAL_MEMBERSHIP_V1",
        policy_version="1.0.0",
        policy_name="Historical Membership State Policy",
        description="Governs bitemporal valid time and point-in-time membership evaluation.",
        authorities=["ALPACA_ASSET_DIRECTORY"],
        precedence=["ALPACA_ASSET_DIRECTORY"],
        admissibility_rules=["OBSERVATION_TIMESTAMP_PRESENT", "NON_BACKDATED_EFFECTIVE_FROM"],
        missing_behavior="NOT_AVAILABLE",
        conflict_behavior="UNRESOLVED_REMOVAL",
        stale_behavior="STALE_EVIDENCE_REJECTED",
        unknown_behavior="NOT_AVAILABLE",
        temporal_behavior="BITEMPORAL_DECOUPLED",
        effective_scopes=[GovernedScope.TEMPORAL_MEMBERSHIP, GovernedScope.HISTORICAL_REPLAY],
    ),
    "POL_HISTORICAL_AUTHORITY_V1": ConcretePolicyContract(
        policy_id="POL_HISTORICAL_AUTHORITY_V1",
        policy_version="1.0.0",
        policy_name="Historical Membership Authority Level Policy",
        description="Classifies point-in-time universe authority (CURRENT_ONLY vs POINT_IN_TIME).",
        authorities=["ARX_GOVERNANCE_COUNCIL"],
        precedence=["ARX_GOVERNANCE_COUNCIL"],
        admissibility_rules=["ORGANIZATIONAL_APPROVAL_REQUIRED"],
        missing_behavior="CURRENT_ONLY",
        conflict_behavior="S3_BLOCKING",
        stale_behavior="FAIL_CLOSED",
        unknown_behavior="UNRESOLVED",
        effective_scopes=[GovernedScope.TEMPORAL_MEMBERSHIP, GovernedScope.HISTORICAL_REPLAY],
    ),
    "POL_PROVIDER_ASSET_CLASS_V1": ConcretePolicyContract(
        policy_id="POL_PROVIDER_ASSET_CLASS_V1",
        policy_version="1.0.0",
        policy_name="Provider Asset Class Derivation Policy",
        description="Normalizes raw provider asset class (us_equity -> US_EQUITY).",
        authorities=["ALPACA_ASSET_DIRECTORY"],
        precedence=["ALPACA_ASSET_DIRECTORY"],
        admissibility_rules=["NON_EMPTY"],
        normalization_rules=["UPPERCASE"],
        missing_behavior="DEFAULT_US_EQUITY",
        conflict_behavior="S1_WARNING",
        derivation_inputs=["class"],
        effective_scopes=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
    ),
    "POL_ENRICHMENT_STATUS_V1": ConcretePolicyContract(
        policy_id="POL_ENRICHMENT_STATUS_V1",
        policy_version="1.0.0",
        policy_name="Reference Enrichment Status Derivation Policy",
        description="Tracks subtype enrichment state (ENRICHED vs AWAITING_ENRICHMENT).",
        authorities=["OPENFIGI_V3_MAPPING"],
        precedence=["OPENFIGI_V3_MAPPING"],
        admissibility_rules=["SUBTYPE_KNOWN"],
        missing_behavior="AWAITING_ENRICHMENT",
        conflict_behavior="S1_WARNING",
        derivation_inputs=["security_type"],
        effective_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.DATA_READINESS],
    ),
    # Section 9 Country and Currency Value Evidence Derivation Policies
    "POL_COUNTRY_DERIVATION_V1": ConcretePolicyContract(
        policy_id="POL_COUNTRY_DERIVATION_V1",
        policy_version="1.0.0",
        policy_name="Listing Country Evidence Derivation Policy",
        description="Derives listing country from primary exchange operating venue (USA for US venues).",
        authorities=["ALPACA_ASSET_DIRECTORY"],
        precedence=["ALPACA_ASSET_DIRECTORY"],
        admissibility_rules=["VALID_OPERATING_MIC"],
        normalization_rules=["ISO_3166_1_ALPHA3"],
        missing_behavior="DEFAULT_USA",
        conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
        derivation_inputs=["primary_exchange"],
        effective_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
    ),
    "POL_CURRENCY_DERIVATION_V1": ConcretePolicyContract(
        policy_id="POL_CURRENCY_DERIVATION_V1",
        policy_version="1.0.0",
        policy_name="Listing Currency Evidence Derivation Policy",
        description="Derives trading currency from primary exchange operating venue (USD for US venues).",
        authorities=["ALPACA_ASSET_DIRECTORY"],
        precedence=["ALPACA_ASSET_DIRECTORY"],
        admissibility_rules=["VALID_OPERATING_MIC"],
        normalization_rules=["ISO_4217_ALPHA3"],
        missing_behavior="DEFAULT_USD",
        conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
        derivation_inputs=["primary_exchange"],
        effective_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
    ),
}


# =====================================================================
# Governance Bundle Model (Section 10)
# =====================================================================

GOVERNANCE_BUNDLE_ID: str = "ARX_SOURCE_GOVERNANCE_BUNDLE"
GOVERNANCE_BUNDLE_VERSION: str = "1.0.0"
POLICY_SEMANTIC_PROJECTION_VERSION: str = "1.0.0"


class GovernanceBundle(BaseModel):
    """
    Aggregate governance identity closing over all semantic governance contracts.
    """
    bundle_id: str = GOVERNANCE_BUNDLE_ID
    bundle_version: str = GOVERNANCE_BUNDLE_VERSION
    catalog_hash: str
    registry_hash: str
    policy_semantic_hashes: Dict[str, str]
    normalization_contract_hash: str
    reason_taxonomy_semantic_hash: str
    canonical_serialization_version: str = "1.0.0"

    model_config = ConfigDict(frozen=True)

    def compute_bundle_hash(self) -> str:
        payload = {
            "bundle_id": self.bundle_id,
            "bundle_version": self.bundle_version,
            "catalog_hash": self.catalog_hash,
            "registry_hash": self.registry_hash,
            "policy_semantic_hashes": {k: self.policy_semantic_hashes[k] for k in sorted(self.policy_semantic_hashes.keys())},
            "normalization_contract_hash": self.normalization_contract_hash,
            "reason_taxonomy_semantic_hash": self.reason_taxonomy_semantic_hash,
            "canonical_serialization_version": self.canonical_serialization_version,
        }
        return canonical_hash(payload)


def get_active_governance_bundle(
    registry_hash: str,
    catalog_hash: str = REQUIRED_GOVERNANCE_CONCEPT_CATALOG_HASH,
) -> GovernanceBundle:
    """Constructs active frozen governance bundle."""
    semantic_hashes = {}
    for pid, contract in sorted(POLICY_CONTRACTS.items()):
        semantic_hashes[pid] = contract.compute_semantic_hash()

    # Add direct field policies from FieldAuthorityPolicyRegistry
    for f, p in sorted(FieldAuthorityPolicyRegistry.POLICIES.items()):
        semantic_hashes[p.field_policy_id] = canonical_hash({
            "policy_id": p.field_policy_id,
            "version": getattr(p, "field_policy_version", FieldAuthorityPolicyRegistry.POLICY_VERSION),
            "canonical_field": p.canonical_field,
            "authority_chain": p.authority_chain,
            "resolution_mode": p.resolution_mode,
            "missing_authority_behavior": p.missing_authority_behavior,
        })

    norm_hash = canonical_hash({
        "symbol_normalization": "UPPERCASE_AND_DOTS",
        "mic_normalization": "ISO_10383_OPERATING_MIC",
    })

    return GovernanceBundle(
        catalog_hash=catalog_hash,
        registry_hash=registry_hash,
        policy_semantic_hashes=semantic_hashes,
        normalization_contract_hash=norm_hash,
        reason_taxonomy_semantic_hash=REASON_CODE_TAXONOMY_HASH,
    )


# =====================================================================
# Decision Semantic Dependencies & Six Hash Model (Sections 11, 12 & 13)
# =====================================================================

class DecisionSemanticDependencies(BaseModel):
    """
    Granular, dependency-scoped semantic inputs for exactly one governed decision.
    """
    concept_id: str
    requirement_catalog_entry_hash: str
    authority_binding_hash: str
    relevant_policy_semantic_hashes: Dict[str, str]
    normalization_contract_hash: Optional[str] = None
    temporal_policy_hash: Optional[str] = None
    reason_taxonomy_semantic_hash: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    def compute_dependency_hash(self) -> str:
        payload = {
            "concept_id": self.concept_id,
            "requirement_catalog_entry_hash": self.requirement_catalog_entry_hash,
            "authority_binding_hash": self.authority_binding_hash,
            "relevant_policy_semantic_hashes": {k: self.relevant_policy_semantic_hashes[k] for k in sorted(self.relevant_policy_semantic_hashes.keys())},
            "normalization_contract_hash": self.normalization_contract_hash,
            "temporal_policy_hash": self.temporal_policy_hash,
            "reason_taxonomy_semantic_hash": self.reason_taxonomy_semantic_hash,
        }
        return canonical_hash(payload)


class DecisionHashModel:
    """
    Frozen six-hash architecture (Section 13).
    Ensures clear separation of dependency, evidence, input, value, execution, and derivation.
    """
    @staticmethod
    def compute_evidence_dependency_hash(used_evidence: Dict[str, Any]) -> str:
        """
        Closes over ONLY the evidence fields actually utilized for this decision.
        Unrelated raw records in the snapshot do NOT alter this hash.
        """
        return canonical_hash(used_evidence)

    @staticmethod
    def compute_decision_input_hash(
        decision_dependency_hash: str,
        decision_evidence_dependency_hash: str,
        as_of: str,
        additional_semantic_inputs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Closes over all semantic governance and evidence inputs.
        Implementation SHA is STRICTLY EXCLUDED (Section 13).
        Run ID and wall-clock timestamps are STRICTLY EXCLUDED (Section 31).
        """
        payload = {
            "decision_dependency_hash": decision_dependency_hash,
            "decision_evidence_dependency_hash": decision_evidence_dependency_hash,
            "as_of": as_of,
            "additional": additional_semantic_inputs or {},
        }
        return canonical_hash(payload)

    @staticmethod
    def compute_decision_value_hash(
        canonical_value: Any,
        severity: str,
        reason_code: str,
    ) -> str:
        """Closes over canonical reconciliation outcome only."""
        payload = {
            "canonical_value": canonical_value,
            "severity": severity,
            "reason_code": reason_code,
        }
        return canonical_hash(payload)

    @staticmethod
    def compute_execution_provenance_hash(
        implementation_sha: str,
        engine_contract_version: str = "1.0.0",
        serialization_version: str = "1.0.0",
    ) -> str:
        """Closes over execution environment, implementation code, and serialization format."""
        payload = {
            "implementation_sha": implementation_sha,
            "engine_contract_version": engine_contract_version,
            "serialization_version": serialization_version,
        }
        return canonical_hash(payload)

    @staticmethod
    def compute_decision_derivation_hash(
        decision_input_hash: str,
        decision_value_hash: str,
        execution_provenance_hash: str,
    ) -> str:
        """
        Complete derivation digest binding inputs, outputs, and execution provenance.
        """
        payload = {
            "decision_input_hash": decision_input_hash,
            "decision_value_hash": decision_value_hash,
            "execution_provenance_hash": execution_provenance_hash,
        }
        return canonical_hash(payload)


# =====================================================================
# Impact Analysis Policy & Replay Scope Governor (Section 29)
# =====================================================================

IMPACT_ANALYSIS_POLICY_ID: str = "ARX_IMPACT_ANALYSIS_POLICY"
IMPACT_ANALYSIS_POLICY_VERSION: str = "1.0.0"

IMPACT_ANALYSIS_POLICY_HASH: str = canonical_hash({
    "policy_id": IMPACT_ANALYSIS_POLICY_ID,
    "version": IMPACT_ANALYSIS_POLICY_VERSION,
    "scope_rule": "DECISION_DEPENDENCY_OR_EVIDENCE_HASH_MISMATCH",
})


class ImpactAnalyzer:
    """
    Evaluates whether a decision must be included in a replay scope.
    Invariants:
    - Relevant policy change -> affected decision included.
    - Unrelated policy change -> unaffected decision excluded.
    - Relevant evidence change -> affected decision included.
    - Unrelated evidence change -> unaffected decision excluded.
    - KNOWN_AFFECTED_DECISION_OMITTED_FROM_REPLAY_SCOPE = 0.
    """
    @staticmethod
    def is_in_replay_scope(
        prior_dependency_hash: str,
        current_dependency_hash: str,
        prior_evidence_hash: str,
        current_evidence_hash: str,
    ) -> bool:
        dependency_changed = prior_dependency_hash != current_dependency_hash
        evidence_changed = prior_evidence_hash != current_evidence_hash
        return dependency_changed or evidence_changed
