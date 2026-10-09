"""
tests/test_sprint_2a_closure_delta.py

Verification Suite for ARX Terminal Radar VCP Sprint 2A Closure Delta Gate.
Covers:
- Section 17 & 18: Negative validator tests for all 15 frozen error codes
- Section 26: Metamorphic hash order invariance and semantic variance
- Section 27: Runtime independence
- Section 20-24: Full mutation testing campaign (100% coverage, 0 critical survivors)
- Section 23: Multi-fault mutation testing (orders 2 and 3)
- Section 15: Enrichment accounting closure & non-drop invariants
- Section 14: Historical coverage-start decoupling (technical snapshot vs authoritative)
- Section 30: Evidence manifest canonicalization & hash equivalence
"""

import copy
import json
from pathlib import Path
import pytest

from analyst_dashboard.security_master import (
    RequiredFieldAuthorityRegistry,
    GovernanceBindingType,
    AuthorityState,
    GovernedScope,
    GovernanceBinding,
    RequiredFieldEntry,
    EnrichmentAccountingSummary,
    PointInTimeStatus,
    HistoricalMembershipAuthority,
    HistoricalMembershipUnavailableError,
    SourceReconciliationEngine,
    canonical_hash,
)
from analyst_dashboard.security_master.mutation_harness import (
    RegistryMutationEngine,
    verify_metamorphic_invariance,
    verify_runtime_independence,
    MUTATION_CATALOG_ID,
    MUTATION_CATALOG_VERSION,
    MUTATION_CATALOG_HASH,
)


# =====================================================================
# Suite 1: Negative Validation Matrix for all 15 Error Codes (Section 17 & 18)
# =====================================================================

def test_negative_duplicate_exact_field_id():
    raw_list = ["symbol", "listing_status", "symbol"]
    res = RequiredFieldAuthorityRegistry.validate(raw_field_list=raw_list)
    assert not res.passed
    assert any(e.error_code == "DUPLICATE_REQUIRED_FIELD" and e.field_id == "symbol" for e in res.errors)


def test_negative_duplicate_normalized_field_id():
    raw_list = ["symbol", "listing_status", "SYMBOL"]
    res = RequiredFieldAuthorityRegistry.validate(raw_field_list=raw_list)
    assert not res.passed
    assert any(e.error_code == "DUPLICATE_REQUIRED_FIELD" and e.field_id == "SYMBOL" for e in res.errors)


def test_negative_missing_required_field():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    del entries["corporate_action_state"]
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_REQUIRED_FIELD" and e.field_id == "corporate_action_state" for e in res.errors)


def test_negative_missing_governance_binding():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["governance_binding"] = None
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_GOVERNANCE_BINDING" and e.field_id == "symbol" for e in res.errors)


def test_negative_multiple_governance_bindings():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    entry = entries["symbol"]
    object.__setattr__(entry, "secondary_binding", GovernanceBinding(binding_type=GovernanceBindingType.DIRECT_FIELD_POLICY))
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MULTIPLE_GOVERNANCE_BINDINGS" and e.field_id == "symbol" for e in res.errors)


def test_negative_unknown_policy_id():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["governance_binding"]["policy_id"] = "POL_NONEXISTENT_XYZ"
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "UNKNOWN_POLICY_REFERENCE" and e.field_id == "symbol" for e in res.errors)


def test_negative_missing_policy_version():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["governance_binding"]["policy_version"] = ""
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_POLICY_VERSION" and e.field_id == "symbol" for e in res.errors)


def test_negative_missing_policy_hash():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["governance_binding"]["policy_hash"] = ""
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_POLICY_HASH" and e.field_id == "symbol" for e in res.errors)


def test_negative_policy_hash_mismatch():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["governance_binding"]["policy_hash"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "POLICY_HASH_MISMATCH" and e.field_id == "symbol" for e in res.errors)


def test_negative_missing_missing_behavior():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["primary_exchange"].model_dump()
    e["missing_behavior"] = ""
    entries["primary_exchange"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_MISSING_BEHAVIOR" and e.field_id == "primary_exchange" for e in res.errors)


def test_negative_missing_conflict_behavior():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["primary_exchange"].model_dump()
    e["conflict_behavior"] = ""
    entries["primary_exchange"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_CONFLICT_BEHAVIOR" and e.field_id == "primary_exchange" for e in res.errors)


def test_negative_missing_stale_behavior():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["historical_membership_state"].model_dump()
    e["stale_behavior"] = ""
    entries["historical_membership_state"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_STALE_BEHAVIOR" and e.field_id == "historical_membership_state" for e in res.errors)


def test_negative_missing_unknown_value_behavior():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["unknown_value_behavior"] = ""
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "MISSING_UNKNOWN_VALUE_BEHAVIOR" and e.field_id == "symbol" for e in res.errors)


def test_negative_implicit_binding_prohibited():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["governance_binding"]["binding_type"] = "IMPLICIT_PROVIDER_DERIVED"
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "IMPLICIT_BINDING_PROHIBITED" and e.field_id == "symbol" for e in res.errors)


def test_negative_unresolved_without_reason_code():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["corporate_action_state"].model_dump()
    e["reason_code"] = None
    entries["corporate_action_state"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "UNRESOLVED_WITHOUT_REASON_CODE" and e.field_id == "corporate_action_state" for e in res.errors)


def test_negative_invalid_not_applicable():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["country"].model_dump()
    e["governance_binding"]["binding_type"] = GovernanceBindingType.NOT_APPLICABLE
    e["required_for"] = [GovernedScope.CANONICAL_RECONCILIATION]
    entries["country"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "INVALID_NOT_APPLICABLE" and e.field_id == "country" for e in res.errors)


def test_negative_provenance_requirement_missing():
    entries = copy.deepcopy(RequiredFieldAuthorityRegistry.REQUIRED_ENTRIES)
    e = entries["symbol"].model_dump()
    e["decision_ledger_required"] = False
    entries["symbol"] = RequiredFieldEntry.model_validate(e)
    res = RequiredFieldAuthorityRegistry.validate(entries)
    assert not res.passed
    assert any(e.error_code == "PROVENANCE_REQUIREMENT_MISSING" and e.field_id == "symbol" for e in res.errors)


# =====================================================================
# Suite 2: Metamorphic & Runtime Independence (Section 26 & 27)
# =====================================================================

def test_metamorphic_hash_order_invariance_and_semantic_variance():
    order_inv, semantic_alt = verify_metamorphic_invariance()
    assert order_inv is True, "ORDER_DEPENDENT_REGISTRY_HASH must be NO"
    assert semantic_alt is True, "SEMANTIC_MUTATION_WITH_UNCHANGED_REGISTRY_HASH must be 0"


def test_runtime_independence_invalid_remains_invalid():
    assert verify_runtime_independence() is True


# =====================================================================
# Suite 3: Mutation Campaign & Multi-Fault Execution (Section 20-24)
# =====================================================================

def test_mutation_campaign_zero_survivors():
    engine = RegistryMutationEngine(seed=42)
    summary = engine.run_campaign()

    assert summary.catalog_id == MUTATION_CATALOG_ID
    assert summary.catalog_version == MUTATION_CATALOG_VERSION
    assert summary.catalog_hash == MUTATION_CATALOG_HASH

    assert summary.mutation_operator_coverage == 1.0
    assert summary.applicable_field_operator_cell_coverage == 1.0
    assert summary.correct_rejection_reason_rate == 1.0
    assert summary.invalid_mutation_rejection_score == 1.0

    assert summary.surviving_invalid_mutants == 0
    assert summary.duplicate_field_survivors == 0
    assert summary.unknown_policy_reference_survivors == 0
    assert summary.missing_behavior_survivors == 0
    assert summary.implicit_binding_survivors == 0
    assert summary.multiple_binding_survivors == 0
    assert summary.invalid_not_applicable_survivors == 0
    assert summary.provenance_removal_survivors == 0
    assert summary.multi_fault_critical_survivors == 0


# =====================================================================
# Suite 4: Enrichment Accounting Closure (Section 15)
# =====================================================================

def test_enrichment_accounting_closure_and_drop_rejection():
    # Valid closure
    valid_acc = EnrichmentAccountingSummary(
        enrichment_requested_count=100,
        enrichment_resolved_count=80,
        enrichment_unresolved_count=20,
        enrichment_failed_count=0,
        enrichment_pending_count=0,
        enrichment_silently_dropped_count=0,
    )
    assert valid_acc.validate_closure() is True

    # Discrepancy fails closure
    invalid_acc = EnrichmentAccountingSummary(
        enrichment_requested_count=100,
        enrichment_resolved_count=80,
        enrichment_unresolved_count=10,  # 80 + 10 != 100
        enrichment_failed_count=0,
        enrichment_pending_count=0,
        enrichment_silently_dropped_count=0,
    )
    assert invalid_acc.validate_closure() is False

    # Silently dropped fails closure
    dropped_acc = EnrichmentAccountingSummary(
        enrichment_requested_count=100,
        enrichment_resolved_count=80,
        enrichment_unresolved_count=20,
        enrichment_failed_count=0,
        enrichment_pending_count=0,
        enrichment_silently_dropped_count=5,  # must be 0
    )
    assert dropped_acc.validate_closure() is False


# =====================================================================
# Suite 5: Point-in-Time Historical Coverage Decoupling (Section 14)
# =====================================================================

def test_point_in_time_historical_coverage_decoupling():
    engine = SourceReconciliationEngine()

    # When authoritative coverage start is NOT_ESTABLISHED (None)
    res_none = engine.query_point_in_time_universe(
        requested_as_of="2026-10-09T00:00:00Z",
        authoritative_historical_coverage_start=None,
        fail_closed=False,
    )
    assert res_none.point_in_time_status == PointInTimeStatus.NOT_AVAILABLE
    assert res_none.authoritative_denominator is None
    assert res_none.listings is None
    assert res_none.historical_membership_authority == HistoricalMembershipAuthority.CURRENT_ONLY

    # Fails closed when requested
    with pytest.raises(HistoricalMembershipUnavailableError):
        engine.query_point_in_time_universe(
            requested_as_of="2026-10-09T00:00:00Z",
            authoritative_historical_coverage_start=None,
            fail_closed=True,
        )


# =====================================================================
# Suite 6: Manifest Canonicalization & Equivalence (Section 30)
# =====================================================================

def test_evidence_manifest_canonicalization_and_hash_equivalence():
    canonical_path = Path("docs/architecture/ARX_SPRINT_2A_SOURCE_GOVERNANCE_EVIDENCE_MANIFEST.json")
    derivative_path = Path("data/operational/sprint_2a_evidence_manifest.json")

    assert canonical_path.exists(), "Canonical manifest must exist"
    assert derivative_path.exists(), "Derivative manifest must exist"

    with open(canonical_path, "r", encoding="utf-8") as f:
        canonical_content = f.read()
    with open(derivative_path, "r", encoding="utf-8") as f:
        derivative_content = f.read()

    assert canonical_content == derivative_content, "Derivative manifest must be byte-identical to canonical manifest"
