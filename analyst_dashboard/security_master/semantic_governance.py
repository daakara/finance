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
from enum import Enum
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
# =====================================================================
# Governance Bundle Model (Sections 10, 16, 17, 18 & 14)
# =====================================================================

GOVERNANCE_BUNDLE_ID: str = "ARX_SOURCE_GOVERNANCE_BUNDLE"
GOVERNANCE_BUNDLE_VERSION: str = "1.0.0"

POLICY_SEMANTIC_PROJECTION_ID: str = "ARX_POLICY_SEMANTIC_PROJECTION"
POLICY_SEMANTIC_PROJECTION_VERSION: str = "1.0.0"

POLICY_SEMANTIC_PROJECTION_RULES = {
    "projection_id": POLICY_SEMANTIC_PROJECTION_ID,
    "version": POLICY_SEMANTIC_PROJECTION_VERSION,
    "semantic_attributes": [
        "policy_id",
        "policy_version",
        "canonical_field",
        "binding_type",
        "authority_chain",
        "resolution_mode",
        "missing_behavior",
        "conflict_behavior",
        "derivation_inputs",
        "effective_scopes",
    ],
    "non_semantic_attributes": [
        "description",
        "author",
        "created_at",
        "documentation_url",
        "execution_notes",
    ],
}
POLICY_SEMANTIC_PROJECTION_HASH: str = canonical_hash(POLICY_SEMANTIC_PROJECTION_RULES)

# Reason Taxonomy Artifact vs Semantic Hash (Section 18)
REASON_TAXONOMY_ARTIFACT_HASH: str = REASON_CODE_TAXONOMY_HASH
REASON_TAXONOMY_SEMANTIC_HASH: str = canonical_hash({
    "taxonomy_id": REASON_CODE_TAXONOMY_ID,
    "semantic_version": REASON_CODE_TAXONOMY_VERSION,
    "reason_codes": sorted([r.value for r in ReasonCode]),
})

# Country & Currency Canonical Semantics (Section 14)
COUNTRY_FIELD_SEMANTICS: str = "LISTING_COUNTRY"
CURRENCY_FIELD_SEMANTICS: str = "TRADING_CURRENCY"

COUNTRY_SEMANTIC_DEFINITION: str = (
    "Country associated with governed listing venue, "
    "NOT issuer domicile/incorporation country."
)

CURRENCY_SEMANTIC_DEFINITION: str = (
    "Trading/listing currency associated with governed listing venue, "
    "NOT issuer reporting currency or domicile currency."
)


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


# Constant recorded active governance bundle hash
GOVERNANCE_BUNDLE_HASH: str = "9aeed0c785d0b6737d0995192a4383ab198784d91d1525f0595d65a01a434bad"


# =====================================================================
# Decision Semantic Dependencies & Value/Outcome Hashes (Sections 3, 4, 5)
# =====================================================================

DECISION_HASH_CONTRACT_VERSION: str = "2.0.0"
DECISION_VALUE_HASH_INCLUDES_REASON_CODE: str = "NO"
DECISION_VALUE_HASH_INCLUDES_SEVERITY: str = "NO"
DECISION_OUTCOME_HASH_ESTABLISHED: str = "YES"
SAME_VALUE_AUTOMATICALLY_IMPLIES_SEMANTIC_EQUIVALENCE: str = "NO"


class DecisionEquivalenceClass(str, Enum):
    VALUE_EQUIVALENT = "VALUE_EQUIVALENT"
    SEMANTICALLY_EQUIVALENT = "SEMANTICALLY_EQUIVALENT"
    CONFORMANT_CROSS_IMPLEMENTATION_EQUIVALENT = "CONFORMANT_CROSS_IMPLEMENTATION_EQUIVALENT"
    VALUE_EQUIVALENT_ONLY = "VALUE_EQUIVALENT_ONLY"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    UNRESOLVED_EQUIVALENCE = "UNRESOLVED_EQUIVALENCE"


class DecisionEquivalenceEvaluator:
    """
    Evaluates equivalence relationships between two decisions (Section 4).
    Rules:
    - Same value_hash -> VALUE_EQUIVALENT
    - Same value_hash + changed semantic dependency -> VALUE_EQUIVALENT_ONLY (SEMANTICALLY_EQUIVALENT is NO)
    - Same semantic dependencies + same evidence + same value -> SEMANTICALLY_EQUIVALENT
    - Same dependencies + evidence + value + different implementation + conformance passed -> CONFORMANT_CROSS_IMPLEMENTATION_EQUIVALENT
    - Otherwise unresolved or not equivalent.
    """
    @staticmethod
    def evaluate(
        old_value_hash: str,
        new_value_hash: str,
        old_dependency_hash: str,
        new_dependency_hash: str,
        old_evidence_hash: str,
        new_evidence_hash: str,
        old_provenance_hash: Optional[str] = None,
        new_provenance_hash: Optional[str] = None,
        conformance_passed: bool = False,
    ) -> Set[DecisionEquivalenceClass]:
        classes: Set[DecisionEquivalenceClass] = set()
        same_value = (old_value_hash == new_value_hash)
        same_dep = (old_dependency_hash == new_dependency_hash)
        same_ev = (old_evidence_hash == new_evidence_hash)
        same_prov = (old_provenance_hash == new_provenance_hash) if (old_provenance_hash and new_provenance_hash) else True

        if not same_value:
            classes.add(DecisionEquivalenceClass.NOT_EQUIVALENT)
            return classes

        classes.add(DecisionEquivalenceClass.VALUE_EQUIVALENT)

        if not same_dep or not same_ev:
            classes.add(DecisionEquivalenceClass.VALUE_EQUIVALENT_ONLY)
        else:
            classes.add(DecisionEquivalenceClass.SEMANTICALLY_EQUIVALENT)
            if not same_prov:
                if conformance_passed:
                    classes.add(DecisionEquivalenceClass.CONFORMANT_CROSS_IMPLEMENTATION_EQUIVALENT)
                else:
                    classes.add(DecisionEquivalenceClass.UNRESOLVED_EQUIVALENCE)

        return classes


class DecisionSemanticDependencies(BaseModel):
    """
    Granular, dependency-scoped semantic inputs for exactly one governed decision.
    Contract v2.0.0 closes over all relevant semantic governance dimensions (Section 5).
    """
    concept_id: str
    requirement_catalog_entry_hash: str
    required_scopes: List[str] = Field(default_factory=list)
    required_status: bool = True
    authority_binding_hash: str
    relevant_policy_semantic_hashes: Dict[str, str] = Field(default_factory=dict)
    normalization_contract_hash: Optional[str] = None
    temporal_policy_hash: Optional[str] = None
    reason_taxonomy_semantic_hash: Optional[str] = None
    evidence_schema_semantic_hash: Optional[str] = None
    value_domain_semantic_identity: Optional[str] = None
    derivation_policy_semantic_hash: Optional[str] = None
    identity_policy_semantic_hash: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    def compute_dependency_hash(self) -> str:
        payload = {
            "concept_id": self.concept_id,
            "requirement_catalog_entry_hash": self.requirement_catalog_entry_hash,
            "required_scopes": sorted(self.required_scopes),
            "required_status": self.required_status,
            "authority_binding_hash": self.authority_binding_hash,
            "relevant_policy_semantic_hashes": {
                k: self.relevant_policy_semantic_hashes[k]
                for k in sorted(self.relevant_policy_semantic_hashes.keys())
            },
            "normalization_contract_hash": self.normalization_contract_hash,
            "temporal_policy_hash": self.temporal_policy_hash,
            "reason_taxonomy_semantic_hash": self.reason_taxonomy_semantic_hash,
            "evidence_schema_semantic_hash": self.evidence_schema_semantic_hash,
            "value_domain_semantic_identity": self.value_domain_semantic_identity,
            "derivation_policy_semantic_hash": self.derivation_policy_semantic_hash,
            "identity_policy_semantic_hash": self.identity_policy_semantic_hash,
        }
        return canonical_hash(payload)


class DecisionHashModel:
    """
    Frozen hash architecture (Contract v2.0.0, Section 3).
    Ensures strict separation of value, outcome, dependency, evidence, input, execution, and derivation.
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
        severity: Optional[str] = None,
        reason_code: Optional[str] = None,
    ) -> str:
        """
        Contract v2.0.0: Represents ONLY the canonical field value/result payload.
        STRICTLY EXCLUDES conflict severity and reason code (Section 3).
        """
        payload = {
            "canonical_value": canonical_value,
        }
        return canonical_hash(payload)

    @staticmethod
    def compute_decision_outcome_hash(
        decision_value_hash: str,
        conflict_severity: str,
        reason_code: str,
    ) -> str:
        """
        Contract v2.0.0: Closes over value hash + conflict severity + reason code semantic identity.
        """
        payload = {
            "decision_value_hash": decision_value_hash,
            "conflict_severity": str(conflict_severity),
            "reason_code": str(reason_code),
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
        decision_outcome_hash: str,
        execution_provenance_hash: str,
        derivation_contract_version: str = "2.0.0",
    ) -> str:
        """
        Contract v2.0.0: Complete derivation digest binding inputs, outcome (value+severity+reason),
        and execution provenance.
        """
        payload = {
            "derivation_contract_version": derivation_contract_version,
            "decision_input_hash": decision_input_hash,
            "decision_outcome_hash": decision_outcome_hash,
            "execution_provenance_hash": execution_provenance_hash,
        }
        return canonical_hash(payload)


# =====================================================================
# Impact Analysis Policy & Replay Scope Governor (Sections 10 & 11)
# =====================================================================

IMPACT_ANALYSIS_POLICY_ID: str = "ARX_IMPACT_ANALYSIS_POLICY"
IMPACT_ANALYSIS_POLICY_VERSION: str = "2.0.0"

IMPACT_ANALYSIS_POLICY_RULES = {
    "policy_id": IMPACT_ANALYSIS_POLICY_ID,
    "version": IMPACT_ANALYSIS_POLICY_VERSION,
    "semantic_dependency_scope_rule": "REPLAY_ON_DEPENDENCY_HASH_MISMATCH",
    "evidence_dependency_scope_rule": "REPLAY_ON_EVIDENCE_HASH_MISMATCH",
    "implementation_change_replay_rules": {
        "NON_SEMANTIC_REFACTOR": "CONFORMANCE_VALIDATION_ONLY",
        "CONFORMANCE_FIX": "AFFECTED_DECISIONS_ENTER_REPLAY_SCOPE",
        "SEMANTIC_POLICY_CHANGE": "REPLAY_VIA_SEMANTIC_DEPENDENCY_CHANGE",
        "MIGRATION_BEHAVIOR_CHANGE": "AFFECTED_PERSISTED_RECONSTRUCTED_DECISIONS_ENTER_REPLAY_SCOPE",
        "UNKNOWN": "PROHIBIT_MATERIAL_AUTHORITY_ACTIVATION",
    },
    "migration_replay_rule": "REPLAY_IF_MIGRATION_CONTRACT_REQUIRES_REPLAY",
    "explicit_validation_scope_rule": "REPLAY_IF_EXPLICIT_VALIDATION_SCOPE_REQUESTED",
}

IMPACT_ANALYSIS_POLICY_HASH: str = canonical_hash(IMPACT_ANALYSIS_POLICY_RULES)


class ImpactAnalyzer:
    """
    Evaluates whether a decision must be included in a replay scope (Sections 10, 11 & 13).
    Invariants:
    - REPLAY_REQUIRED = semantic_dependency_changed
                     OR evidence_dependency_changed
                     OR implementation_change_requires_replay
                     OR migration_contract_requires_replay
                     OR explicit_validation_scope_requires_replay
    - UNKNOWN material changes fail-closed for authority activation.
    - IMPLEMENTATION_ONLY_AFFECTED_DECISION_OMITTED_FROM_REPLAY_SCOPE = 0.
    """
    @staticmethod
    def is_in_replay_scope(
        prior_dependency_hash: str,
        current_dependency_hash: str,
        prior_evidence_hash: str,
        current_evidence_hash: str,
        implementation_change_class: Optional[Any] = None,
        is_affected_by_implementation: bool = False,
        migration_contract_requires_replay: bool = False,
        explicit_validation_scope: bool = False,
    ) -> bool:
        dependency_changed = prior_dependency_hash != current_dependency_hash
        evidence_changed = prior_evidence_hash != current_evidence_hash

        if dependency_changed or evidence_changed:
            return True

        if migration_contract_requires_replay:
            return True

        if explicit_validation_scope:
            return True

        if implementation_change_class:
            c = getattr(implementation_change_class, "value", str(implementation_change_class))
            if c == "CONFORMANCE_FIX" and is_affected_by_implementation:
                return True
            if c == "MIGRATION_BEHAVIOR_CHANGE" and is_affected_by_implementation:
                return True
            if c == "NON_SEMANTIC_REFACTOR":
                return False
            if c == "UNKNOWN":
                return False

        return False

    @staticmethod
    def can_activate_authority(
        implementation_change_class: Any,
        conformance_attribution: Optional[Any] = None,
    ) -> bool:
        """
        Sections 9, 10 & 13:
        UNKNOWN material changes fail closed for authority activation.
        CONFORMANCE_FIX requires an independent conformance basis (not UNRESOLVED).
        """
        c = getattr(implementation_change_class, "value", str(implementation_change_class))
        if c == "UNKNOWN":
            return False
        if c == "CONFORMANCE_FIX":
            attr = getattr(conformance_attribution, "value", str(conformance_attribution)) if conformance_attribution else "UNRESOLVED"
            if attr in ("UNRESOLVED", "None", ""):
                return False
        return True
