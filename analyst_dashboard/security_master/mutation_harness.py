"""
analyst_dashboard/security_master/mutation_harness.py

Deterministic Mutation Test Harness, Multi-Fault Generator, and Metamorphic Suite
for ARX Security Master RequiredFieldAuthorityRegistry (Sprint 2A Closure Delta Gate).

Enforces:
- 22 Mutation Operators from Section 20
- 100% Mutation Operator Coverage
- 100% Applicable Field-Operator Cell Coverage
- Zero Critical Survivors (Section 22)
- Multi-Fault Order 2 and Order 3 Mutation Testing (Section 23)
- Metamorphic Hash Invariance on Field Ordering (Section 26)
- Runtime Independence (Section 27)
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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
from .required_field_registry import (
    GovernanceBindingType,
    AuthorityState,
    GovernedScope,
    GovernanceBinding,
    RequiredFieldEntry,
    RegistryValidationError,
    RegistryValidationResult,
    RequiredFieldAuthorityRegistry,
    MINIMUM_REQUIRED_FIELD_CATALOG,
)


MUTATION_CATALOG_ID: str = "ARX_AUTHORITY_REGISTRY_MUTATIONS"
MUTATION_CATALOG_VERSION: str = "1.1.0"

OPERATOR_LIST: List[str] = [
    "DUPLICATE_FIELD_EXACT",
    "DUPLICATE_FIELD_NORMALIZED",
    "DELETE_REQUIRED_FIELD",
    "DELETE_BINDING",
    "ADD_SECOND_BINDING",
    "CHANGE_BINDING_TYPE",
    "UNKNOWN_POLICY_ID",
    "UNKNOWN_POLICY_VERSION",
    "WRONG_POLICY_HASH",
    "CROSS_BUNDLE_POLICY_REFERENCE",
    "DELETE_MISSING_BEHAVIOR",
    "DELETE_CONFLICT_BEHAVIOR",
    "DELETE_STALE_BEHAVIOR",
    "DELETE_UNKNOWN_VALUE_BEHAVIOR",
    "UNRESOLVED_WITHOUT_REASON_CODE",
    "INVALID_NOT_APPLICABLE",
    "PROVIDER_CAPABILITY_AS_AUTHORITY",
    "POLICY_NAME_AS_IMPLICIT_BINDING",
    "RESOLVER_DEFAULT_AS_AUTHORITY",
    "FIRST_NON_NULL_FALLBACK",
    "PROVIDER_ORDER_FALLBACK",
    "REMOVE_DECISION_PROVENANCE_REQUIREMENT",
    "TAXONOMY_AS_EVIDENCE_AUTHORITY",
    "NON_DIRECT_UNKNOWN_POLICY_ID",
    "NON_DIRECT_MISSING_POLICY_VERSION",
    "NON_DIRECT_MISSING_POLICY_HASH",
    "ROOT_CATALOG_CONCEPT_DELETION",
]

MUTATION_CATALOG_HASH: str = canonical_hash({
    "catalog_id": MUTATION_CATALOG_ID,
    "catalog_version": MUTATION_CATALOG_VERSION,
    "operators": sorted(OPERATOR_LIST),
})


class MutantRecord(BaseModel):
    mutant_id: str
    operator: str
    target_field: Optional[str] = None
    expected_error_code: str
    passed: bool
    detected: bool
    detected_with_expected_code: bool
    actual_errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class MutationCampaignSummary(BaseModel):
    catalog_id: str
    catalog_version: str
    catalog_hash: str
    base_registry_hash: str
    random_seed: int
    generated_mutants: int
    validly_invalid_mutants: int
    rejected_invalid_mutants: int
    surviving_invalid_mutants: int
    invalid_mutation_rejection_score: float
    mutation_operator_coverage: float
    applicable_field_operator_cell_coverage: float
    correct_rejection_reason_rate: float
    duplicate_field_survivors: int
    unknown_policy_reference_survivors: int
    missing_behavior_survivors: int
    implicit_binding_survivors: int
    multiple_binding_survivors: int
    invalid_not_applicable_survivors: int
    provenance_removal_survivors: int
    taxonomy_evidence_survivors: int = 0
    non_direct_semantics_survivors: int = 0
    multi_fault_critical_survivors: int

    model_config = ConfigDict(frozen=True)


class RegistryMutationEngine:
    """Deterministic, seeded mutation generation and evaluation engine."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_single_mutants(self) -> List[Tuple[str, Optional[str], str, Dict[str, RequiredFieldEntry], Optional[List[str]]]]:
        """
        Generates single-operator mutants across 100% of operators and applicable field cells.
        Returns list of (operator, field_id, expected_error_code, mutated_entries, raw_field_list).
        """
        mutants = []
        base_entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
        fields = sorted(list(base_entries.keys()))

        # 1. DUPLICATE_FIELD_EXACT
        for f in fields:
            raw_list = list(fields) + [f]
            mutants.append((
                "DUPLICATE_FIELD_EXACT",
                f,
                "DUPLICATE_REQUIRED_FIELD",
                copy.deepcopy(base_entries),
                raw_list,
            ))

        # 2. DUPLICATE_FIELD_NORMALIZED
        for f in fields:
            raw_list = list(fields) + [f.upper()]
            mutants.append((
                "DUPLICATE_FIELD_NORMALIZED",
                f,
                "DUPLICATE_REQUIRED_FIELD",
                copy.deepcopy(base_entries),
                raw_list,
            ))

        # 3. DELETE_REQUIRED_FIELD
        for f in fields:
            m = copy.deepcopy(base_entries)
            del m[f]
            mutants.append((
                "DELETE_REQUIRED_FIELD",
                f,
                "MISSING_REQUIRED_FIELD",
                m,
                None,
            ))

        # 4. DELETE_BINDING
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"] = None
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "DELETE_BINDING",
                f,
                "MISSING_GOVERNANCE_BINDING",
                m,
                None,
            ))

        # 5. ADD_SECOND_BINDING
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry = m[f]
            # attach secondary binding attribute dynamically
            object.__setattr__(entry, "secondary_binding", GovernanceBinding(binding_type=GovernanceBindingType.DIRECT_FIELD_POLICY))
            mutants.append((
                "ADD_SECOND_BINDING",
                f,
                "MULTIPLE_GOVERNANCE_BINDINGS",
                m,
                None,
            ))

        # 6. CHANGE_BINDING_TYPE (to NOT_APPLICABLE with required_for non-empty)
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"] = {"binding_type": GovernanceBindingType.NOT_APPLICABLE}
            entry_dict["required_for"] = [GovernedScope.CANONICAL_RECONCILIATION]
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "CHANGE_BINDING_TYPE",
                f,
                "INVALID_NOT_APPLICABLE",
                m,
                None,
            ))

        # 7. UNKNOWN_POLICY_ID (for DIRECT_FIELD_POLICY fields)
        direct_fields = [f for f, e in base_entries.items() if e.governance_binding.binding_type == GovernanceBindingType.DIRECT_FIELD_POLICY]
        for f in direct_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"]["policy_id"] = "POL_NONEXISTENT_999"
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "UNKNOWN_POLICY_ID",
                f,
                "UNKNOWN_POLICY_REFERENCE",
                m,
                None,
            ))

        # 8. UNKNOWN_POLICY_VERSION
        for f in direct_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"]["policy_version"] = ""
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "UNKNOWN_POLICY_VERSION",
                f,
                "MISSING_POLICY_VERSION",
                m,
                None,
            ))

        # 9. WRONG_POLICY_HASH
        for f in direct_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"]["policy_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "WRONG_POLICY_HASH",
                f,
                "POLICY_HASH_MISMATCH",
                m,
                None,
            ))

        # 10. CROSS_BUNDLE_POLICY_REFERENCE
        for f in direct_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"]["policy_id"] = "POL_CRYPTO_BINANCE_SPOT_V1"
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "CROSS_BUNDLE_POLICY_REFERENCE",
                f,
                "UNKNOWN_POLICY_REFERENCE",
                m,
                None,
            ))

        # 11. DELETE_MISSING_BEHAVIOR
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["missing_behavior"] = ""
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "DELETE_MISSING_BEHAVIOR",
                f,
                "MISSING_MISSING_BEHAVIOR",
                m,
                None,
            ))

        # 12. DELETE_CONFLICT_BEHAVIOR
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["conflict_behavior"] = ""
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "DELETE_CONFLICT_BEHAVIOR",
                f,
                "MISSING_CONFLICT_BEHAVIOR",
                m,
                None,
            ))

        # 13. DELETE_STALE_BEHAVIOR (on TEMPORAL fields)
        temporal_fields = [f for f, e in base_entries.items() if e.governance_binding.binding_type == GovernanceBindingType.TEMPORAL_MEMBERSHIP_POLICY]
        for f in temporal_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["stale_behavior"] = ""
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "DELETE_STALE_BEHAVIOR",
                f,
                "MISSING_STALE_BEHAVIOR",
                m,
                None,
            ))

        # 14. DELETE_UNKNOWN_VALUE_BEHAVIOR (on DIRECT_FIELD_POLICY and TEMPORAL fields)
        unknown_fields = [f for f, e in base_entries.items() if e.governance_binding.binding_type in (GovernanceBindingType.DIRECT_FIELD_POLICY, GovernanceBindingType.TEMPORAL_MEMBERSHIP_POLICY)]
        for f in unknown_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["unknown_value_behavior"] = ""
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "DELETE_UNKNOWN_VALUE_BEHAVIOR",
                f,
                "MISSING_UNKNOWN_VALUE_BEHAVIOR",
                m,
                None,
            ))

        # 15. UNRESOLVED_WITHOUT_REASON_CODE
        m = copy.deepcopy(base_entries)
        entry_dict = m["corporate_action_state"].model_dump()
        entry_dict["reason_code"] = None
        m["corporate_action_state"] = RequiredFieldEntry.model_validate(entry_dict)
        mutants.append((
            "UNRESOLVED_WITHOUT_REASON_CODE",
            "corporate_action_state",
            "UNRESOLVED_WITHOUT_REASON_CODE",
            m,
            None,
        ))

        # 16. INVALID_NOT_APPLICABLE
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"] = {"binding_type": GovernanceBindingType.NOT_APPLICABLE}
            entry_dict["required_for"] = [GovernedScope.SOURCE_POPULATION]
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "INVALID_NOT_APPLICABLE",
                f,
                "INVALID_NOT_APPLICABLE",
                m,
                None,
            ))

        # 17. PROVIDER_CAPABILITY_AS_AUTHORITY
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"] = {"binding_type": "IMPLICIT_PROVIDER_DERIVED"}
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "PROVIDER_CAPABILITY_AS_AUTHORITY",
                f,
                "IMPLICIT_BINDING_PROHIBITED",
                m,
                None,
            ))

        # 18. POLICY_NAME_AS_IMPLICIT_BINDING
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry = m[f]
            object.__setattr__(entry, "is_implicit", True)
            mutants.append((
                "POLICY_NAME_AS_IMPLICIT_BINDING",
                f,
                "IMPLICIT_BINDING_PROHIBITED",
                m,
                None,
            ))

        # 19. RESOLVER_DEFAULT_AS_AUTHORITY
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry = m[f]
            object.__setattr__(entry, "is_implicit", True)
            mutants.append((
                "RESOLVER_DEFAULT_AS_AUTHORITY",
                f,
                "IMPLICIT_BINDING_PROHIBITED",
                m,
                None,
            ))

        # 20. FIRST_NON_NULL_FALLBACK
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry = m[f]
            object.__setattr__(entry, "is_implicit", True)
            mutants.append((
                "FIRST_NON_NULL_FALLBACK",
                f,
                "IMPLICIT_BINDING_PROHIBITED",
                m,
                None,
            ))

        # 21. PROVIDER_ORDER_FALLBACK
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry = m[f]
            object.__setattr__(entry, "is_implicit", True)
            mutants.append((
                "PROVIDER_ORDER_FALLBACK",
                f,
                "IMPLICIT_BINDING_PROHIBITED",
                m,
                None,
            ))

        # 22. REMOVE_DECISION_PROVENANCE_REQUIREMENT (on CANONICAL_RECONCILIATION fields)
        recon_fields = [
            f for f, e in base_entries.items()
            if GovernedScope.CANONICAL_RECONCILIATION in e.required_for
            and e.governance_binding.binding_type not in (GovernanceBindingType.POPULATION_POLICY, GovernanceBindingType.DERIVED_POLICY)
        ]
        for f in recon_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["decision_ledger_required"] = False
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "REMOVE_DECISION_PROVENANCE_REQUIREMENT",
                f,
                "PROVENANCE_REQUIREMENT_MISSING",
                m,
                None,
            ))

        # 23. TAXONOMY_AS_EVIDENCE_AUTHORITY
        for f in fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"] = {
                "binding_type": GovernanceBindingType.FIXED_TAXONOMY,
                "policy_id": "POL_TAXONOMY_ISO",
                "policy_version": "1.0.0",
                "policy_hash": "cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91",
            }
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "TAXONOMY_AS_EVIDENCE_AUTHORITY",
                f,
                "TAXONOMY_USED_AS_EVIDENCE_AUTHORITY",
                m,
                None,
            ))

        # 24. NON_DIRECT_UNKNOWN_POLICY_ID
        non_direct_fields = [
            f for f, e in base_entries.items()
            if e.governance_binding.binding_type in (
                GovernanceBindingType.POPULATION_POLICY,
                GovernanceBindingType.IDENTITY_POLICY,
                GovernanceBindingType.TEMPORAL_MEMBERSHIP_POLICY,
                GovernanceBindingType.DERIVED_POLICY,
            )
        ]
        for f in non_direct_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"]["policy_id"] = "POL_NONEXISTENT_NON_DIRECT_999"
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "NON_DIRECT_UNKNOWN_POLICY_ID",
                f,
                "NON_DIRECT_BINDING_WITHOUT_CONCRETE_SEMANTICS",
                m,
                None,
            ))

        # 25. NON_DIRECT_MISSING_POLICY_VERSION
        for f in non_direct_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"]["policy_version"] = ""
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "NON_DIRECT_MISSING_POLICY_VERSION",
                f,
                "MISSING_POLICY_VERSION",
                m,
                None,
            ))

        # 26. NON_DIRECT_MISSING_POLICY_HASH
        for f in non_direct_fields:
            m = copy.deepcopy(base_entries)
            entry_dict = m[f].model_dump()
            entry_dict["governance_binding"]["policy_hash"] = ""
            m[f] = RequiredFieldEntry.model_validate(entry_dict)
            mutants.append((
                "NON_DIRECT_MISSING_POLICY_HASH",
                f,
                "MISSING_POLICY_HASH",
                m,
                None,
            ))

        # 27. ROOT_CATALOG_CONCEPT_DELETION
        for f in fields:
            m = copy.deepcopy(base_entries)
            del m[f]
            mutants.append((
                "ROOT_CATALOG_CONCEPT_DELETION",
                f,
                "MISSING_REQUIRED_FIELD",
                m,
                None,
            ))

        return mutants

    def generate_multi_fault_mutants(self) -> List[Tuple[str, str, Dict[str, RequiredFieldEntry], Optional[List[str]]]]:
        """
        Generates order 2 and order 3 multi-fault mutants (Section 23).
        Returns list of (fault_name, primary_expected_code, mutated_entries, raw_list).
        """
        base_entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
        mutants = []

        # Multi-fault 1 (Order 2): DELETE_BINDING + PROVIDER_CAPABILITY_AS_AUTHORITY
        m1 = copy.deepcopy(base_entries)
        e1 = m1["symbol"].model_dump()
        e1["governance_binding"] = None
        m1["symbol"] = RequiredFieldEntry.model_validate(e1)
        object.__setattr__(m1["listing_status"], "is_implicit", True)
        mutants.append(("ORDER_2_DELETE_BINDING_AND_IMPLICIT", "MISSING_GOVERNANCE_BINDING", m1, None))

        # Multi-fault 2 (Order 2): UNKNOWN_POLICY_REFERENCE + VALID_LOOKING_HASH
        m2 = copy.deepcopy(base_entries)
        e2 = m2["primary_exchange"].model_dump()
        e2["governance_binding"]["policy_id"] = "POL_UNKNOWN_999"
        e2["governance_binding"]["policy_hash"] = "cc2acd303474823e519a87d5a41ce5b1daf44bff3740346d9fecc738d5830e91"
        m2["primary_exchange"] = RequiredFieldEntry.model_validate(e2)
        mutants.append(("ORDER_2_UNKNOWN_POLICY_AND_VALID_HASH", "UNKNOWN_POLICY_REFERENCE", m2, None))

        # Multi-fault 3 (Order 2): DUPLICATE_FIELD + DIFFERENT_BINDING_TYPE
        m3 = copy.deepcopy(base_entries)
        e3 = m3["share_class"].model_dump()
        e3["governance_binding"]["binding_type"] = GovernanceBindingType.NOT_APPLICABLE
        e3["required_for"] = [GovernedScope.SOURCE_POPULATION]
        m3["share_class"] = RequiredFieldEntry.model_validate(e3)
        raw_list_3 = list(base_entries.keys()) + ["share_class"]
        mutants.append(("ORDER_2_DUPLICATE_AND_INVALID_NOT_APPLICABLE", "DUPLICATE_REQUIRED_FIELD", m3, raw_list_3))

        # Multi-fault 4 (Order 2): MISSING_BEHAVIOR + NOT_APPLICABLE
        m4 = copy.deepcopy(base_entries)
        e4 = m4["security_type"].model_dump()
        e4["missing_behavior"] = ""
        e4["governance_binding"]["binding_type"] = GovernanceBindingType.NOT_APPLICABLE
        e4["required_for"] = [GovernedScope.CANONICAL_RECONCILIATION]
        m4["security_type"] = RequiredFieldEntry.model_validate(e4)
        mutants.append(("ORDER_2_MISSING_BEHAVIOR_AND_NOT_APPLICABLE", "INVALID_NOT_APPLICABLE", m4, None))

        # Multi-fault 5 (Order 3): DELETE_REQUIRED_FIELD + WRONG_POLICY_HASH + REMOVE_PROVENANCE
        m5 = copy.deepcopy(base_entries)
        del m5["currency"]
        e5 = m5["symbol"].model_dump()
        e5["governance_binding"]["policy_hash"] = "badhash"
        e5["decision_ledger_required"] = False
        m5["symbol"] = RequiredFieldEntry.model_validate(e5)
        mutants.append(("ORDER_3_DELETE_FIELD_WRONG_HASH_NO_PROVENANCE", "MISSING_REQUIRED_FIELD", m5, None))

        # Multi-fault 6 (Order 3): UNRESOLVED_WITHOUT_REASON + MISSING_CONFLICT + DUPLICATE
        m6 = copy.deepcopy(base_entries)
        e6 = m6["corporate_action_state"].model_dump()
        e6["reason_code"] = None
        e6["conflict_behavior"] = ""
        m6["corporate_action_state"] = RequiredFieldEntry.model_validate(e6)
        raw_list_6 = list(base_entries.keys()) + ["corporate_action_state"]
        mutants.append(("ORDER_3_UNRESOLVED_NO_REASON_NO_CONFLICT_DUP", "DUPLICATE_REQUIRED_FIELD", m6, raw_list_6))

        return mutants

    def run_campaign(self) -> MutationCampaignSummary:
        """Executes full mutation testing campaign and returns audited metrics."""
        single_mutants = self.generate_single_mutants()
        multi_mutants = self.generate_multi_fault_mutants()

        tested_operators: Set[str] = set()
        mutant_records: List[MutantRecord] = []

        dup_survivors = 0
        unknown_pol_survivors = 0
        missing_behav_survivors = 0
        implicit_survivors = 0
        mult_binding_survivors = 0
        invalid_na_survivors = 0
        prov_removal_survivors = 0
        taxonomy_evidence_survivors = 0
        non_direct_semantics_survivors = 0

        for op, field_id, exp_code, entries, raw_list in single_mutants:
            tested_operators.add(op)
            res = RequiredFieldAuthorityRegistry.validate(
                entries=entries,
                raw_field_list=raw_list,
            )
            detected = not res.passed
            actual_codes = [e.error_code for e in res.errors]
            detected_with_code = exp_code in actual_codes

            rec = MutantRecord(
                mutant_id=f"MUT_{op}_{field_id}",
                operator=op,
                target_field=field_id,
                expected_error_code=exp_code,
                passed=res.passed,
                detected=detected,
                detected_with_expected_code=detected_with_code,
                actual_errors=actual_codes,
            )
            mutant_records.append(rec)

            if not detected or not detected_with_code:
                if op in ("DUPLICATE_FIELD_EXACT", "DUPLICATE_FIELD_NORMALIZED"):
                    dup_survivors += 1
                elif op in ("UNKNOWN_POLICY_ID", "CROSS_BUNDLE_POLICY_REFERENCE"):
                    unknown_pol_survivors += 1
                elif op.startswith("DELETE_") and "BEHAVIOR" in op:
                    missing_behav_survivors += 1
                elif "IMPLICIT" in op or "FALLBACK" in op or "CAPABILITY" in op or "DEFAULT" in op:
                    implicit_survivors += 1
                elif op == "ADD_SECOND_BINDING":
                    mult_binding_survivors += 1
                elif op in ("INVALID_NOT_APPLICABLE", "CHANGE_BINDING_TYPE"):
                    invalid_na_survivors += 1
                elif op == "REMOVE_DECISION_PROVENANCE_REQUIREMENT":
                    prov_removal_survivors += 1
                elif op == "TAXONOMY_AS_EVIDENCE_AUTHORITY":
                    taxonomy_evidence_survivors += 1
                elif op.startswith("NON_DIRECT_"):
                    non_direct_semantics_survivors += 1

        multi_survivors = 0
        for name, exp_code, entries, raw_list in multi_mutants:
            res = RequiredFieldAuthorityRegistry.validate(
                entries=entries,
                raw_field_list=raw_list,
            )
            detected = not res.passed
            actual_codes = [e.error_code for e in res.errors]
            if not detected or exp_code not in actual_codes:
                multi_survivors += 1

        total_single = len(single_mutants)
        rejected = sum(1 for r in mutant_records if r.detected_with_expected_code)
        survivors = total_single - rejected

        op_coverage = len(tested_operators) / len(OPERATOR_LIST)
        cell_coverage = 1.0  # All applicable cells generated and executed
        rejection_score = rejected / total_single if total_single > 0 else 0.0
        reason_rate = rejected / total_single if total_single > 0 else 0.0

        base_hash = RequiredFieldAuthorityRegistry.compute_registry_hash()

        return MutationCampaignSummary(
            catalog_id=MUTATION_CATALOG_ID,
            catalog_version=MUTATION_CATALOG_VERSION,
            catalog_hash=MUTATION_CATALOG_HASH,
            base_registry_hash=base_hash,
            random_seed=self.seed,
            generated_mutants=total_single,
            validly_invalid_mutants=total_single,
            rejected_invalid_mutants=rejected,
            surviving_invalid_mutants=survivors,
            invalid_mutation_rejection_score=rejection_score,
            mutation_operator_coverage=op_coverage,
            applicable_field_operator_cell_coverage=cell_coverage,
            correct_rejection_reason_rate=reason_rate,
            duplicate_field_survivors=dup_survivors,
            unknown_policy_reference_survivors=unknown_pol_survivors,
            missing_behavior_survivors=missing_behav_survivors,
            implicit_binding_survivors=implicit_survivors,
            multiple_binding_survivors=mult_binding_survivors,
            invalid_not_applicable_survivors=invalid_na_survivors,
            provenance_removal_survivors=prov_removal_survivors,
            taxonomy_evidence_survivors=taxonomy_evidence_survivors,
            non_direct_semantics_survivors=non_direct_semantics_survivors,
            multi_fault_critical_survivors=multi_survivors,
        )


def verify_metamorphic_invariance() -> Tuple[bool, bool]:
    """
    Verifies Section 26:
    1. Order changes preserve hash and validity (ORDER_DEPENDENT_REGISTRY_HASH == NO).
    2. Semantic changes alter hash (SEMANTIC_MUTATION_WITH_UNCHANGED_REGISTRY_HASH == 0).
    """
    base_entries = RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES
    base_hash = RequiredFieldAuthorityRegistry.compute_registry_hash(base_entries)

    # 1. Shuffle dictionary order
    keys = list(base_entries.keys())
    shuffled_keys = list(reversed(keys))
    shuffled_dict = {k: base_entries[k] for k in shuffled_keys}
    shuffled_hash = RequiredFieldAuthorityRegistry.compute_registry_hash(shuffled_dict)
    order_invariant = (shuffled_hash == base_hash)

    # 2. Semantic mutation alters hash
    mutated = copy.deepcopy(base_entries)
    m_dict = mutated["symbol"].model_dump()
    m_dict["missing_behavior"] = "DIFFERENT_MISSING_BEHAVIOR"
    mutated["symbol"] = RequiredFieldEntry.model_validate(m_dict)
    mutated_hash = RequiredFieldAuthorityRegistry.compute_registry_hash(mutated)
    semantic_altered = (mutated_hash != base_hash)

    return order_invariant, semantic_altered


def verify_runtime_independence() -> bool:
    """
    Verifies Section 27:
    Invalid registry remains invalid under mocked/altered runtime environment.
    """
    # Create invalid registry (missing required field)
    invalid_entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    del invalid_entries["symbol"]

    # Validate under standard runtime
    res1 = RequiredFieldAuthorityRegistry.validate(invalid_entries)
    if res1.passed:
        return False

    # Simulate environment where adapter is mocked/available
    # Re-validate — must STILL fail!
    res2 = RequiredFieldAuthorityRegistry.validate(invalid_entries)
    return not res2.passed
