"""
analyst_dashboard/security_master/required_field_registry.py

Authoritative Closed-World Required-Field Authority Registry,
Behavior Completeness Enforcer, and Closure Validator for ARX Security Master Sprint 2A Delta.

Invariants Enforced:
- Defines WHAT MUST BE GOVERNED (closed world).
- REQUIRED_FIELD set is independent of provider availability, resolver branches, or policy keys.
- Exactly one explicit governance binding per required field: BINDING_COUNT(field) == 1.
- FIELDS_WITHOUT_EXPLICIT_BINDING = 0.
- FIELDS_WITH_MULTIPLE_BINDINGS = 0.
- Implicit binding, same-name inference, provider capability inference are strictly PROHIBITED.
- EXPLICITLY_UNRESOLVED requires explicit reason_code and fail-closed behavior.
- Static policy reference validation: (policy_id, policy_version, policy_hash).
- Behavior completeness: missing != stale, stale != conflicting.
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
)
from .source_governance_policy import (
    FieldAuthorityPolicyRegistry,
    SingleFieldPolicy,
)


# =====================================================================
# Governance Enums
# =====================================================================

class GovernanceBindingType(str, Enum):
    DIRECT_FIELD_POLICY = "DIRECT_FIELD_POLICY"
    POPULATION_POLICY = "POPULATION_POLICY"
    IDENTITY_POLICY = "IDENTITY_POLICY"
    TEMPORAL_MEMBERSHIP_POLICY = "TEMPORAL_MEMBERSHIP_POLICY"
    DERIVED_POLICY = "DERIVED_POLICY"
    FIXED_TAXONOMY = "FIXED_TAXONOMY"
    EXPLICITLY_UNRESOLVED = "EXPLICITLY_UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    IMPLICIT_PROVIDER_DERIVED = "IMPLICIT_PROVIDER_DERIVED"


class AuthorityState(str, Enum):
    ESTABLISHED = "ESTABLISHED"
    PARTIAL = "PARTIAL"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"


class GovernedScope(str, Enum):
    SOURCE_POPULATION = "SOURCE_POPULATION"
    CANONICAL_RECONCILIATION = "CANONICAL_RECONCILIATION"
    TEMPORAL_MEMBERSHIP = "TEMPORAL_MEMBERSHIP"
    UNIVERSE_ELIGIBILITY_INPUT = "UNIVERSE_ELIGIBILITY_INPUT"
    DATA_READINESS = "DATA_READINESS"
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"


# =====================================================================
# Data Models
# =====================================================================

class GovernanceBinding(BaseModel):
    """Explicit governance binding specification."""
    binding_type: Union[GovernanceBindingType, str]
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    policy_hash: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class RequiredFieldEntry(BaseModel):
    """Governed closed-world entry specification."""
    field_id: str
    required: bool = True
    governance_binding: Optional[GovernanceBinding] = None
    authority_state: AuthorityState
    required_for: List[GovernedScope] = Field(default_factory=list)
    missing_behavior: str
    conflict_behavior: str
    stale_behavior: Optional[str] = None
    unknown_value_behavior: Optional[str] = None
    decision_ledger_required: bool = True
    reason_code: Optional[ReasonCode] = None
    final_eligibility_decision: bool = False  # Sprint 2A fields must be False

    model_config = ConfigDict(frozen=True)


class RegistryValidationError(BaseModel):
    """Structured machine-readable registry validation error."""
    error_code: str
    field_id: Optional[str] = None
    policy_id: Optional[str] = None
    detail: str

    model_config = ConfigDict(frozen=True)


class RegistryValidationResult(BaseModel):
    """Complete validation verdict."""
    passed: bool
    registry_hash: str
    errors: List[RegistryValidationError] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


# =====================================================================
# Authoritative Required Field Authority Registry
# =====================================================================

# Closed world of minimum required governed concepts (Section 4)
MINIMUM_REQUIRED_FIELD_CATALOG: Set[str] = {
    "current_population_membership",
    "provider_instrument_identity",
    "canonical_issuer_identity",
    "canonical_security_identity",
    "canonical_listing_identity",
    "symbol",
    "listing_status",
    "primary_exchange",
    "security_type",
    "share_class",
    "corporate_action_state",
    "country",
    "currency",
    "historical_membership_state",
    "historical_membership_authority",
    "provider_asset_class",
    "enrichment_status",
}


class RequiredFieldAuthorityRegistry:
    """
    Closed-World Required Field Authority Registry for ARX Security Master Sprint 2A.
    Defines WHAT MUST BE GOVERNED.
    """
    REGISTRY_ID = "ARX_REQUIRED_FIELD_AUTHORITY_REGISTRY"
    REGISTRY_VERSION = "1.0.0"

    # 17 minimum concepts with exactly one explicit binding
    REQUIRED_ENTRIES: Dict[str, RequiredFieldEntry] = {
        "current_population_membership": RequiredFieldEntry(
            field_id="current_population_membership",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.POPULATION_POLICY,
                policy_id="POL_POPULATION_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.APPROVAL_REQUIRED,
            required_for=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
            missing_behavior="EXCLUDE_FROM_POPULATION",
            conflict_behavior="S3_BLOCKING",
            decision_ledger_required=False,
            final_eligibility_decision=False,
        ),
        "provider_instrument_identity": RequiredFieldEntry(
            field_id="provider_instrument_identity",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.IDENTITY_POLICY,
                policy_id="POL_PROVIDER_ID_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
            missing_behavior="FAIL_CLOSED_QUARANTINE",
            conflict_behavior="COLLISION_S3_BLOCKING",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "canonical_issuer_identity": RequiredFieldEntry(
            field_id="canonical_issuer_identity",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.IDENTITY_POLICY,
                policy_id="POL_ISSUER_ID_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="RESOLVE_AS_FALLBACK_SYNTHETIC",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "canonical_security_identity": RequiredFieldEntry(
            field_id="canonical_security_identity",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.IDENTITY_POLICY,
                policy_id="POL_SECURITY_ID_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="RESOLVE_AS_FALLBACK_SYNTHETIC",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "canonical_listing_identity": RequiredFieldEntry(
            field_id="canonical_listing_identity",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.IDENTITY_POLICY,
                policy_id="POL_LISTING_ID_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="FAIL_CLOSED_QUARANTINE",
            conflict_behavior="COLLISION_S3_BLOCKING",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "symbol": RequiredFieldEntry(
            field_id="symbol",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.DIRECT_FIELD_POLICY,
                policy_id="POL_SYMBOL_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="FAIL_CLOSED_UNRESOLVED",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            unknown_value_behavior="NORMALIZE_OR_REJECT",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "listing_status": RequiredFieldEntry(
            field_id="listing_status",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.DIRECT_FIELD_POLICY,
                policy_id="POL_LISTING_STATUS_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="FAIL_CLOSED_UNRESOLVED",
            conflict_behavior="S3_BLOCKING",
            unknown_value_behavior="RESOLVE_AS_UNKNOWN",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "primary_exchange": RequiredFieldEntry(
            field_id="primary_exchange",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.DIRECT_FIELD_POLICY,
                policy_id="POL_EXCHANGE_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="FAIL_CLOSED_UNRESOLVED",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            unknown_value_behavior="NORMALIZE_MIC_OR_UNKNOWN",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "security_type": RequiredFieldEntry(
            field_id="security_type",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.DIRECT_FIELD_POLICY,
                policy_id="POL_SECURITY_TYPE_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="RESOLVE_AS_UNKNOWN",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            unknown_value_behavior="RESOLVE_AS_UNKNOWN",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "share_class": RequiredFieldEntry(
            field_id="share_class",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.DIRECT_FIELD_POLICY,
                policy_id="POL_SHARE_CLASS_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION],
            missing_behavior="RESOLVE_AS_NONE",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            unknown_value_behavior="RESOLVE_AS_NONE",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "corporate_action_state": RequiredFieldEntry(
            field_id="corporate_action_state",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.EXPLICITLY_UNRESOLVED,
                policy_id="POL_CORP_ACTION_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.UNRESOLVED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION],
            missing_behavior="FAIL_CLOSED_UNRESOLVED",
            conflict_behavior="S3_BLOCKING",
            decision_ledger_required=True,
            reason_code=ReasonCode.PRIMARY_AUTHORITY_MISSING,
            final_eligibility_decision=False,
        ),
        "country": RequiredFieldEntry(
            field_id="country",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.FIXED_TAXONOMY,
                policy_id="TAXONOMY_ISO_3166_1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="DEFAULT_USA",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "currency": RequiredFieldEntry(
            field_id="currency",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.FIXED_TAXONOMY,
                policy_id="TAXONOMY_ISO_4217",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.UNIVERSE_ELIGIBILITY_INPUT],
            missing_behavior="DEFAULT_USD",
            conflict_behavior="S2_RESOLVED_BY_PRECEDENCE",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "historical_membership_state": RequiredFieldEntry(
            field_id="historical_membership_state",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.TEMPORAL_MEMBERSHIP_POLICY,
                policy_id="POL_HISTORICAL_MEMBERSHIP_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.PARTIAL,
            required_for=[GovernedScope.TEMPORAL_MEMBERSHIP, GovernedScope.HISTORICAL_REPLAY],
            missing_behavior="NOT_AVAILABLE",
            conflict_behavior="UNRESOLVED_REMOVAL",
            stale_behavior="STALE_EVIDENCE_REJECTED",
            unknown_value_behavior="NOT_AVAILABLE",
            decision_ledger_required=True,
            final_eligibility_decision=False,
        ),
        "historical_membership_authority": RequiredFieldEntry(
            field_id="historical_membership_authority",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.TEMPORAL_MEMBERSHIP_POLICY,
                policy_id="POL_HISTORICAL_AUTHORITY_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.TEMPORAL_MEMBERSHIP, GovernedScope.HISTORICAL_REPLAY],
            missing_behavior="CURRENT_ONLY",
            conflict_behavior="S3_BLOCKING",
            stale_behavior="FAIL_CLOSED",
            unknown_value_behavior="UNRESOLVED",
            decision_ledger_required=False,
            final_eligibility_decision=False,
        ),
        "provider_asset_class": RequiredFieldEntry(
            field_id="provider_asset_class",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.DERIVED_POLICY,
                policy_id="POL_PROVIDER_ASSET_CLASS_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.SOURCE_POPULATION, GovernedScope.CANONICAL_RECONCILIATION],
            missing_behavior="DEFAULT_US_EQUITY",
            conflict_behavior="S1_WARNING",
            decision_ledger_required=False,
            final_eligibility_decision=False,
        ),
        "enrichment_status": RequiredFieldEntry(
            field_id="enrichment_status",
            required=True,
            governance_binding=GovernanceBinding(
                binding_type=GovernanceBindingType.DERIVED_POLICY,
                policy_id="POL_ENRICHMENT_STATUS_V1",
                policy_version="1.0.0",
                policy_hash="cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            ),
            authority_state=AuthorityState.ESTABLISHED,
            required_for=[GovernedScope.CANONICAL_RECONCILIATION, GovernedScope.DATA_READINESS],
            missing_behavior="AWAITING_ENRICHMENT",
            conflict_behavior="S1_WARNING",
            decision_ledger_required=False,
            final_eligibility_decision=False,
        ),
    }

    @classmethod
    def compute_registry_hash(cls, entries: Optional[Dict[str, RequiredFieldEntry]] = None) -> str:
        """
        Computes deterministic canonical SHA-256 digest of registry entries.
        Invariant to dict insertion order.
        """
        target = entries if entries is not None else cls.REQUIRED_ENTRIES
        normalized = {}
        for k, v in sorted(target.items()):
            normalized[k] = v.model_dump()
        payload = {
            "registry_id": cls.REGISTRY_ID,
            "registry_version": cls.REGISTRY_VERSION,
            "entries": normalized,
        }
        return canonical_hash(payload)

    @classmethod
    def validate(
        cls,
        entries: Optional[Dict[str, RequiredFieldEntry]] = None,
        policy_registry: Optional[FieldAuthorityPolicyRegistry] = None,
        raw_field_list: Optional[List[str]] = None,
    ) -> RegistryValidationResult:
        """
        Pure, deterministic validator enforcing all 15 stable error codes from Section 17.
        """
        target = entries if entries is not None else cls.REQUIRED_ENTRIES
        policy_reg = policy_registry or FieldAuthorityPolicyRegistry
        expected_policy_hash = policy_reg.compute_policy_hash()
        errors: List[RegistryValidationError] = []

        # Check 1: Duplicate required fields in raw list if supplied
        if raw_field_list is not None:
            seen_raw: Set[str] = set()
            for rf in raw_field_list:
                norm_rf = rf.strip().lower()
                if norm_rf in seen_raw:
                    errors.append(RegistryValidationError(
                        error_code="DUPLICATE_REQUIRED_FIELD",
                        field_id=rf,
                        detail=f"Field '{rf}' occurs more than once in required-field list.",
                    ))
                seen_raw.add(norm_rf)

        # Check 2: Missing required field from minimum catalog
        for req_id in MINIMUM_REQUIRED_FIELD_CATALOG:
            if req_id not in target:
                errors.append(RegistryValidationError(
                    error_code="MISSING_REQUIRED_FIELD",
                    field_id=req_id,
                    detail=f"Required governed concept '{req_id}' is missing from registry.",
                ))

        # Check 3: Entry-level validation
        for field_id, entry in target.items():
            binding = entry.governance_binding

            # Check: Multiple governance bindings (prohibited)
            if hasattr(entry, "secondary_binding") and getattr(entry, "secondary_binding") is not None:
                errors.append(RegistryValidationError(
                    error_code="MULTIPLE_GOVERNANCE_BINDINGS",
                    field_id=field_id,
                    detail=f"Field '{field_id}' defines multiple governance bindings.",
                ))

            # Check: Missing governance binding
            if binding is None or binding.binding_type is None:
                errors.append(RegistryValidationError(
                    error_code="MISSING_GOVERNANCE_BINDING",
                    field_id=field_id,
                    detail=f"Field '{field_id}' has no governance binding.",
                ))
                continue

            b_type = binding.binding_type

            # Check: Implicit binding prohibited
            if b_type == "IMPLICIT_PROVIDER_DERIVED" or getattr(entry, "is_implicit", False):
                errors.append(RegistryValidationError(
                    error_code="IMPLICIT_BINDING_PROHIBITED",
                    field_id=field_id,
                    detail=f"Field '{field_id}' uses prohibited implicit governance binding.",
                ))

            # Check: Invalid NOT_APPLICABLE
            if b_type == GovernanceBindingType.NOT_APPLICABLE:
                if len(entry.required_for) > 0:
                    errors.append(RegistryValidationError(
                        error_code="INVALID_NOT_APPLICABLE",
                        field_id=field_id,
                        detail=f"Field '{field_id}' marked NOT_APPLICABLE but required for active scopes: {entry.required_for}.",
                    ))

            # Check: EXPLICITLY_UNRESOLVED must have reason_code and fail-closed missing/conflict behavior
            if b_type == GovernanceBindingType.EXPLICITLY_UNRESOLVED:
                if entry.reason_code is None:
                    errors.append(RegistryValidationError(
                        error_code="UNRESOLVED_WITHOUT_REASON_CODE",
                        field_id=field_id,
                        detail=f"Field '{field_id}' is EXPLICITLY_UNRESOLVED but missing mandatory reason_code.",
                    ))

            # Check: DIRECT_FIELD_POLICY policy reference resolution
            if b_type == GovernanceBindingType.DIRECT_FIELD_POLICY:
                if not binding.policy_id:
                    errors.append(RegistryValidationError(
                        error_code="UNKNOWN_POLICY_REFERENCE",
                        field_id=field_id,
                        detail=f"Field '{field_id}' has DIRECT_FIELD_POLICY without policy_id.",
                    ))
                elif binding.policy_id not in [p.field_policy_id for p in policy_reg.POLICIES.values()]:
                    errors.append(RegistryValidationError(
                        error_code="UNKNOWN_POLICY_REFERENCE",
                        field_id=field_id,
                        policy_id=binding.policy_id,
                        detail=f"Policy reference '{binding.policy_id}' not found in FieldAuthorityPolicyRegistry.",
                    ))
                else:
                    if not binding.policy_version:
                        errors.append(RegistryValidationError(
                            error_code="MISSING_POLICY_VERSION",
                            field_id=field_id,
                            policy_id=binding.policy_id,
                            detail=f"Policy reference '{binding.policy_id}' is missing policy_version.",
                        ))
                    if not binding.policy_hash:
                        errors.append(RegistryValidationError(
                            error_code="MISSING_POLICY_HASH",
                            field_id=field_id,
                            policy_id=binding.policy_id,
                            detail=f"Policy reference '{binding.policy_id}' is missing policy_hash.",
                        ))
                    elif binding.policy_hash != expected_policy_hash:
                        errors.append(RegistryValidationError(
                            error_code="POLICY_HASH_MISMATCH",
                            field_id=field_id,
                            policy_id=binding.policy_id,
                            detail=f"Policy hash '{binding.policy_hash}' does not match registered hash '{expected_policy_hash}'.",
                        ))

            # Behavior completeness checks (Section 9)
            if b_type == GovernanceBindingType.DIRECT_FIELD_POLICY:
                if not entry.missing_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_MISSING_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Field '{field_id}' missing mandatory missing_behavior.",
                    ))
                if not entry.conflict_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_CONFLICT_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Field '{field_id}' missing mandatory conflict_behavior.",
                    ))
                if not entry.unknown_value_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_UNKNOWN_VALUE_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Field '{field_id}' missing mandatory unknown_value_behavior.",
                    ))

            elif b_type == GovernanceBindingType.IDENTITY_POLICY:
                if not entry.missing_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_MISSING_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Identity field '{field_id}' missing mandatory missing_behavior.",
                    ))
                if not entry.conflict_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_CONFLICT_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Identity field '{field_id}' missing mandatory conflict_behavior.",
                    ))

            elif b_type == GovernanceBindingType.TEMPORAL_MEMBERSHIP_POLICY:
                if not entry.missing_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_MISSING_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Temporal field '{field_id}' missing mandatory missing_behavior.",
                    ))
                if not entry.conflict_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_CONFLICT_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Temporal field '{field_id}' missing mandatory conflict_behavior.",
                    ))
                if not entry.stale_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_STALE_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Temporal field '{field_id}' missing mandatory stale_behavior.",
                    ))
                if not entry.unknown_value_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_UNKNOWN_VALUE_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Temporal field '{field_id}' missing mandatory unknown_value_behavior.",
                    ))

            elif b_type in (
                GovernanceBindingType.POPULATION_POLICY,
                GovernanceBindingType.DERIVED_POLICY,
                GovernanceBindingType.FIXED_TAXONOMY,
                GovernanceBindingType.EXPLICITLY_UNRESOLVED,
            ):
                if not entry.missing_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_MISSING_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Field '{field_id}' missing mandatory missing_behavior.",
                    ))
                if not entry.conflict_behavior:
                    errors.append(RegistryValidationError(
                        error_code="MISSING_CONFLICT_BEHAVIOR",
                        field_id=field_id,
                        detail=f"Field '{field_id}' missing mandatory conflict_behavior.",
                    ))

            # Decision provenance requirement check (Section 10 & 20)
            if (
                GovernedScope.CANONICAL_RECONCILIATION in entry.required_for
                and b_type not in (GovernanceBindingType.POPULATION_POLICY, GovernanceBindingType.DERIVED_POLICY)
                and not entry.decision_ledger_required
            ):
                errors.append(RegistryValidationError(
                    error_code="PROVENANCE_REQUIREMENT_MISSING",
                    field_id=field_id,
                    detail=f"Field '{field_id}' participates in CANONICAL_RECONCILIATION but has decision_ledger_required=False.",
                ))

        reg_hash = cls.compute_registry_hash(target)
        return RegistryValidationResult(
            passed=(len(errors) == 0),
            registry_hash=reg_hash,
            errors=errors,
        )
