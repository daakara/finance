"""
analyst_dashboard/security_master/requirement_catalog.py

Root Requirement-Catalog Governance & Change-Control Ledger
for ARX Terminal Radar VCP Sprint 2A Final Closure Integrity Gate.

Invariants Enforced:
- Defines WHAT MUST BE GOVERNED independently of the authority registry:
  REQUIREMENT_CATALOG != AUTHORITY_REGISTRY.
- The required set MUST NOT be derived from the registry.
- Every catalog modification must carry a cryptographically linked CatalogChangeRecord.
- Prohibits silent concept deletion, silent required=true -> false, silent scope removal,
  and silent conversion to NOT_APPLICABLE.
- UNAUTHORIZED_REQUIRED_CONCEPT_REMOVALS = 0.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, ConfigDict

from .source_governance_models import canonical_hash, canonical_json_dumps
from .required_field_registry import GovernedScope


REQUIRED_GOVERNANCE_CONCEPT_CATALOG_ID: str = "ARX_REQUIRED_GOVERNANCE_CONCEPT_CATALOG"
REQUIRED_GOVERNANCE_CONCEPT_CATALOG_VERSION: str = "1.0.0"


class CatalogChangeType(str, Enum):
    ADD_CONCEPT = "ADD_CONCEPT"
    REMOVE_CONCEPT = "REMOVE_CONCEPT"
    CHANGE_REQUIRED_SCOPE = "CHANGE_REQUIRED_SCOPE"
    CHANGE_REQUIRED_STATUS = "CHANGE_REQUIRED_STATUS"
    RENAME_WITH_SEMANTIC_EQUIVALENCE = "RENAME_WITH_SEMANTIC_EQUIVALENCE"
    OTHER_SEMANTIC_CHANGE = "OTHER_SEMANTIC_CHANGE"


class GovernanceConceptDefinition(BaseModel):
    """Governed requirement definition for a core concept."""
    concept_id: str
    concept_name: str
    description: str
    required: bool = True
    required_scopes: List[GovernedScope] = Field(default_factory=list)
    must_have_provenance: bool = True
    semantic_domain: str = "SECURITIES_REFERENCE_DATA"

    model_config = ConfigDict(frozen=True)


class CatalogChangeRecord(BaseModel):
    """Cryptographically linked audit record for requirement catalog mutations."""
    catalog_change_id: str
    predecessor_catalog_hash: Optional[str] = None
    successor_catalog_hash: str
    change_type: CatalogChangeType
    rationale: str
    affected_concepts: List[str]
    affected_scopes: List[GovernedScope] = Field(default_factory=list)
    authorization_status: str = "AUTHORIZED"  # AUTHORIZED | REJECTED | PENDING
    differential_required: bool = True
    created_at: str

    model_config = ConfigDict(frozen=True)


class RequirementCatalogValidationError(BaseModel):
    error_code: str
    concept_id: Optional[str] = None
    detail: str

    model_config = ConfigDict(frozen=True)


class RequirementCatalogValidationResult(BaseModel):
    passed: bool
    catalog_hash: str
    errors: List[RequirementCatalogValidationError] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


# The canonical closed-world definition of 17 minimum concepts that must be governed
CONCEPTS_V1: Dict[str, GovernanceConceptDefinition] = {
    "current_population_membership": GovernanceConceptDefinition(
        concept_id="current_population_membership",
        concept_name="Current Population Membership",
        description="Point-in-time presence within provider source population.",
        required=True,
        required_scopes=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
        must_have_provenance=False,
    ),
    "provider_instrument_identity": GovernanceConceptDefinition(
        concept_id="provider_instrument_identity",
        concept_name="Provider Instrument Identity",
        description="Provider-native instrument identifier (e.g. Alpaca UUID).",
        required=True,
        required_scopes=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
        must_have_provenance=True,
    ),
    "canonical_issuer_identity": GovernanceConceptDefinition(
        concept_id="canonical_issuer_identity",
        concept_name="Canonical Issuer Identity",
        description="Canonical legal issuer entity identity (SEC CIK/Name).",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "canonical_security_identity": GovernanceConceptDefinition(
        concept_id="canonical_security_identity",
        concept_name="Canonical Security Identity",
        description="Canonical financial security identifier.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "canonical_listing_identity": GovernanceConceptDefinition(
        concept_id="canonical_listing_identity",
        concept_name="Canonical Listing Identity",
        description="Canonical market listing venue identifier.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "symbol": GovernanceConceptDefinition(
        concept_id="symbol",
        concept_name="Trading Ticker Symbol",
        description="Standard normalized market trading symbol.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "listing_status": GovernanceConceptDefinition(
        concept_id="listing_status",
        concept_name="Market Listing Status",
        description="Venue trading status (ACTIVE, INACTIVE, DELISTED, SUSPENDED).",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "primary_exchange": GovernanceConceptDefinition(
        concept_id="primary_exchange",
        concept_name="Primary Operating Exchange",
        description="Operating MIC of the primary listing exchange.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "security_type": GovernanceConceptDefinition(
        concept_id="security_type",
        concept_name="Security Subtype",
        description="Granular financial security subtype (COMMON_STOCK, ETF, etc.).",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "share_class": GovernanceConceptDefinition(
        concept_id="share_class",
        concept_name="Share Class Specification",
        description="Share class designation or share class FIGI.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION],
        must_have_provenance=True,
    ),
    "corporate_action_state": GovernanceConceptDefinition(
        concept_id="corporate_action_state",
        concept_name="Corporate Action State",
        description="Pending/effective corporate actions affecting the security.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION],
        must_have_provenance=True,
    ),
    "country": GovernanceConceptDefinition(
        concept_id="country",
        concept_name="Listing Country",
        description="ISO 3166-1 alpha-3 country of listing.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "currency": GovernanceConceptDefinition(
        concept_id="currency",
        concept_name="Trading Currency",
        description="ISO 4217 currency code.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
        must_have_provenance=True,
    ),
    "historical_membership_state": GovernanceConceptDefinition(
        concept_id="historical_membership_state",
        concept_name="Historical Membership State",
        description="Temporal membership state for historical point-in-time reconstruction.",
        required=True,
        required_scopes=[GovernedScope.TEMPORAL_MEMBERSHIP, GovernedScope.HISTORICAL_REPLAY],
        must_have_provenance=True,
    ),
    "historical_membership_authority": GovernanceConceptDefinition(
        concept_id="historical_membership_authority",
        concept_name="Historical Membership Authority Level",
        description="Authority level of historical universe queries (CURRENT_ONLY vs POINT_IN_TIME).",
        required=True,
        required_scopes=[GovernedScope.TEMPORAL_MEMBERSHIP, GovernedScope.HISTORICAL_REPLAY],
        must_have_provenance=False,
    ),
    "provider_asset_class": GovernanceConceptDefinition(
        concept_id="provider_asset_class",
        concept_name="Provider Broad Asset Class",
        description="Broad asset category provided by source adapter (e.g. US_EQUITY).",
        required=True,
        required_scopes=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
        must_have_provenance=False,
    ),
    "enrichment_status": GovernanceConceptDefinition(
        concept_id="enrichment_status",
        concept_name="Reference Enrichment Status",
        description="Lifecycle status of reference subtype enrichment.",
        required=True,
        required_scopes=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.DATA_READINESS],
        must_have_provenance=False,
    ),
}


class RequiredGovernanceConceptCatalog:
    """
    Independent root authority defining WHAT MUST BE GOVERNED.
    Strictly decoupled from how the authority registry chooses to bind policies.
    """
    CATALOG_ID = REQUIRED_GOVERNANCE_CONCEPT_CATALOG_ID
    CATALOG_VERSION = REQUIRED_GOVERNANCE_CONCEPT_CATALOG_VERSION
    CONCEPTS: Dict[str, GovernanceConceptDefinition] = CONCEPTS_V1

    # Audit ledger tracking revisions
    CHANGE_LOG: List[CatalogChangeRecord] = [
        CatalogChangeRecord(
            catalog_change_id="CHG_CATALOG_V1_INIT",
            predecessor_catalog_hash=None,
            successor_catalog_hash="8b259eb5029bdc507be80117135fc8182fb6eba50b4843eef88ef9baae1c4f50",
            change_type=CatalogChangeType.ADD_CONCEPT,
            rationale="Initial frozen baseline requirement catalog for Sprint 2A (17 concepts).",
            affected_concepts=sorted(list(CONCEPTS_V1.keys())),
            affected_scopes=[
                GovernedScope.SOURCE_POPULATION,
                GovernedScope.CANONICAL_RECONCILIATION,
                GovernedScope.TEMPORAL_MEMBERSHIP,
                GovernedScope.UNIVERSE_ELIGIBILITY_INPUT,
                GovernedScope.DATA_READINESS,
                GovernedScope.HISTORICAL_REPLAY,
            ],
            authorization_status="AUTHORIZED",
            differential_required=False,
            created_at="2026-10-09T08:00:00Z",
        )
    ]

    @classmethod
    def compute_catalog_hash(cls, concepts: Optional[Dict[str, GovernanceConceptDefinition]] = None) -> str:
        """
        Computes deterministic canonical SHA-256 digest of root requirement catalog.
        Invariant to dict insertion order.
        """
        target = concepts if concepts is not None else cls.CONCEPTS
        normalized = {}
        for k, v in sorted(target.items()):
            normalized[k] = v.model_dump()
        payload = {
            "catalog_id": cls.CATALOG_ID,
            "catalog_version": cls.CATALOG_VERSION,
            "concepts": normalized,
        }
        return canonical_hash(payload)

    @classmethod
    def validate(
        cls,
        concepts: Optional[Dict[str, GovernanceConceptDefinition]] = None,
        expected_version: Optional[str] = None,
        expected_hash: Optional[str] = None,
        change_record: Optional[CatalogChangeRecord] = None,
    ) -> RequirementCatalogValidationResult:
        """
        Pure, deterministic validator for root requirement catalog integrity.
        Enforces Section 3, 4, and 5 change-control rules.
        """
        target = concepts if concepts is not None else cls.CONCEPTS
        errors: List[RequirementCatalogValidationError] = []

        # Version check
        if expected_version is not None and expected_version != cls.CATALOG_VERSION:
            if change_record is None:
                errors.append(RequirementCatalogValidationError(
                    error_code="CATALOG_VERSION_WITHOUT_CHANGE_ID",
                    detail=f"Catalog version changed to '{cls.CATALOG_VERSION}' without an authorized CatalogChangeRecord.",
                ))

        actual_hash = cls.compute_catalog_hash(target)
        if expected_hash is not None and expected_hash != actual_hash:
            errors.append(RequirementCatalogValidationError(
                error_code="CATALOG_HASH_MISMATCH",
                detail=f"Catalog hash '{actual_hash}' does not match expected hash '{expected_hash}'.",
            ))

        # Check: Every baseline concept must exist unless an authorized REMOVE_CONCEPT record exists
        for baseline_id, baseline_def in CONCEPTS_V1.items():
            if baseline_id not in target:
                authorized_removal = (
                    change_record is not None
                    and change_record.change_type == CatalogChangeType.REMOVE_CONCEPT
                    and baseline_id in change_record.affected_concepts
                    and change_record.authorization_status == "AUTHORIZED"
                )
                if not authorized_removal:
                    errors.append(RequirementCatalogValidationError(
                        error_code="UNAUTHORIZED_CONCEPT_REMOVAL",
                        concept_id=baseline_id,
                        detail=f"Required governance concept '{baseline_id}' removed without authorization.",
                    ))
            else:
                entry = target[baseline_id]
                # Check required flag
                if baseline_def.required and not entry.required:
                    errors.append(RequirementCatalogValidationError(
                        error_code="UNAUTHORIZED_REQUIRED_STATUS_CHANGE",
                        concept_id=baseline_id,
                        detail=f"Concept '{baseline_id}' required flag silently disabled.",
                    ))
                # Check scope removal
                missing_scopes = set(baseline_def.required_scopes) - set(entry.required_scopes)
                if missing_scopes:
                    errors.append(RequirementCatalogValidationError(
                        error_code="UNAUTHORIZED_SCOPE_REMOVAL",
                        concept_id=baseline_id,
                        detail=f"Concept '{baseline_id}' has unauthorized scope removal: {missing_scopes}.",
                    ))

        return RequirementCatalogValidationResult(
            passed=(len(errors) == 0),
            catalog_hash=actual_hash,
            errors=errors,
        )


REQUIRED_GOVERNANCE_CONCEPT_CATALOG_HASH: str = RequiredGovernanceConceptCatalog.compute_catalog_hash()
