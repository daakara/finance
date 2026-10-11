"""
Test Suite: Smart Money Anchor Policy Resolution Closure Gate.
Verifies Sections 4 through 25 of the Anchor Policy Resolution specification:
- Section 4: Versioned Policy Registry & Half-Open [valid_from, valid_until) intervals
- Section 5: Canonical Epoch Registry & Ordinal Monotonicity
- Section 6: ANCHOR_POLICY_MIN_RUN_EPOCH & Ordinal replacement
- Section 7: Canonical applicability rule (ALL_PREDICATES_REQUIRED)
- Section 8: Authoritative Run-Start Interval Input
- Section 9: Clock boundary relation classification
- Section 10: Fail-closed straddle handling (INDETERMINATE_BOUNDARY)
- Section 11: Exactly-one policy selection
- Section 12: Pre-effective / No-policy semantics
- Section 13: Normal policy supersession
- Section 14: Emergency revocation semantics
- Section 15: Run-genesis immutable policy binding
- Section 16 & 17 & 18: Historical replay parity & run policy set hash
- Section 19: Comprehensive 15-case test suite
- Section 20: Clock boundary edge-case tests
- Section 21: Registry conflict tests
- Section 22: Revocation tests
- Section 23: No retroactive policy application tests
- Section 24: Downstream policy firewall enforcement
- Section 25: Policy artifact retention verification
"""

import hashlib
import unittest
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from analyst_dashboard.smart_money.anchor_finalization import (
    ANCHOR_POLICY_ID,
    ANCHOR_POLICY_VERSION,
    ANCHOR_POLICY_EFFECTIVE_AT,
    ANCHOR_POLICY_MIN_RUN_EPOCH,
    ANCHOR_POLICY_MIN_RUN_EPOCH_ORDINAL,
    ANCHOR_POLICY_APPLICABILITY_RULE,
    RUN_STARTED_AT_DEFINITION,
    STRADDLES_BOUNDARY_POLICY,
    APPLICABLE_POLICY_COUNT_REQUIRED,
    NORMAL_POLICY_SUPERSESSION_REBINDS_EXISTING_RUN,
    EMERGENCY_REVOCATION_REWRITES_ORIGINAL_BINDING,
    EMERGENCY_REVOCATION_CAN_BLOCK_AUTHORITY,
    POLICY_BINDING_IMMUTABLE,
    MID_RUN_POLICY_REBINDING,
    RevocationStatus,
    BoundaryClassification,
    PolicyResolutionResult,
    PreEffectivePolicyBehavior,
    EpochRegistry,
    DEFAULT_EPOCH_REGISTRY,
    EPOCH_REGISTRY_VERSION,
    EPOCH_REGISTRY_SHA256,
    PolicyRegistryEntry,
    PolicyRegistry,
    CANONICAL_ANCHOR_POLICY_V1_ENTRY,
    DEFAULT_POLICY_REGISTRY,
    POLICY_REGISTRY_VERSION,
    POLICY_REGISTRY_SHA256,
    RunStartInterval,
    RunPolicyBinding,
    classify_run_interval_boundary,
    classify_run_interval_policy_end,
    PolicyResolutionEngine,
    replay_policy_binding,
    DownstreamUseFirewall,
    DownstreamFirewallBlockedError,
    AnchorFinalizationReviewer,
    verify_all_policy_artifacts_retained,
    get_policy_artifact,
)


class TestSmartMoneyAnchorPolicyResolution(unittest.TestCase):
    """
    Independent Policy Resolution and Governance State Machine Test Suite.
    """

    def setUp(self):
        self.default_engine = PolicyResolutionEngine()
        self.valid_from = CANONICAL_ANCHOR_POLICY_V1_ENTRY.valid_from  # "2026-10-11T12:00:00Z"

    # =========================================================================
    # Section 19: Comprehensive 15-Case Test Suite
    # =========================================================================

    def test_case_1_run_entirely_before_valid_from(self):
        """Case 1: Run interval ends before valid_from -> NO_APPLICABLE_POLICY, BEFORE."""
        interval = RunStartInterval(
            run_genesis_event_id="gen_001",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T11:15:00+02:00",
            run_start_clock_evidence_id="clk_001",
            run_started_at_interval_start="2026-10-11T09:14:59.000Z",
            run_started_at_interval_end="2026-10-11T09:15:01.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, b_class = self.default_engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.NO_APPLICABLE_POLICY)
        self.assertIsNone(entry)
        self.assertEqual(b_class, BoundaryClassification.BEFORE)

    def test_case_2_run_interval_straddles_valid_from(self):
        """Case 2: Run interval straddles valid_from -> INDETERMINATE_BOUNDARY."""
        interval = RunStartInterval(
            run_genesis_event_id="gen_002",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:00:00+02:00",
            run_start_clock_evidence_id="clk_002",
            run_started_at_interval_start="2026-10-11T11:59:59.500Z",
            run_started_at_interval_end="2026-10-11T12:00:00.500Z",
            run_epoch="EPOCH_001",
        )
        res, entry, b_class = self.default_engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.INDETERMINATE_BOUNDARY)
        self.assertIsNone(entry)
        self.assertEqual(b_class, BoundaryClassification.STRADDLES_BOUNDARY)

    def test_case_3_run_entirely_after_valid_from_in_epoch_1(self):
        """Case 3: Run interval strictly after valid_from in EPOCH_001 -> APPLICABLE, exactly 1."""
        interval = RunStartInterval(
            run_genesis_event_id="gen_003",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:05:00+02:00",
            run_start_clock_evidence_id="clk_003",
            run_started_at_interval_start="2026-10-11T12:04:59.000Z",
            run_started_at_interval_end="2026-10-11T12:05:01.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, b_class = self.default_engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.APPLICABLE)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.policy_id, "SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1")
        self.assertEqual(b_class, BoundaryClassification.AFTER_OR_EQUAL)

    def test_case_4_run_in_epoch_0_lower_ordinal(self):
        """Case 4: Run in epoch with ordinal < min_run_epoch_ordinal -> NO_APPLICABLE_POLICY."""
        custom_epoch_reg = EpochRegistry(
            version="1.0",
            mapping={"EPOCH_000": 0, "EPOCH_001": 1, "EPOCH_002": 2},
        )
        engine = PolicyResolutionEngine(
            policy_registry=DEFAULT_POLICY_REGISTRY,
            epoch_registry=custom_epoch_reg,
        )
        interval = RunStartInterval(
            run_genesis_event_id="gen_004",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:05:00+02:00",
            run_start_clock_evidence_id="clk_004",
            run_started_at_interval_start="2026-10-11T12:05:00.000Z",
            run_started_at_interval_end="2026-10-11T12:05:02.000Z",
            run_epoch="EPOCH_000",
        )
        res, entry, b_class = engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.NO_APPLICABLE_POLICY)
        self.assertIsNone(entry)

    def test_case_5_run_in_epoch_2_greater_ordinal(self):
        """Case 5: Run in EPOCH_002 (ordinal 2 >= 1) after valid_from -> APPLICABLE."""
        interval = RunStartInterval(
            run_genesis_event_id="gen_005",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:10:00+02:00",
            run_start_clock_evidence_id="clk_005",
            run_started_at_interval_start="2026-10-11T12:10:00.000Z",
            run_started_at_interval_end="2026-10-11T12:10:02.000Z",
            run_epoch="EPOCH_002",
        )
        res, entry, b_class = self.default_engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.APPLICABLE)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.policy_id, "SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1")

    def test_case_6_run_in_unknown_epoch(self):
        """Case 6: Run in unknown epoch not present in EpochRegistry -> BLOCKED_UNKNOWN_EPOCH."""
        interval = RunStartInterval(
            run_genesis_event_id="gen_006",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:15:00+02:00",
            run_start_clock_evidence_id="clk_006",
            run_started_at_interval_start="2026-10-11T12:15:00.000Z",
            run_started_at_interval_end="2026-10-11T12:15:02.000Z",
            run_epoch="EPOCH_999_SYNTHETIC",
        )
        res, entry, b_class = self.default_engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.BLOCKED_UNKNOWN_EPOCH)
        self.assertIsNone(entry)
        self.assertEqual(b_class, BoundaryClassification.NOT_APPLICABLE)

    def test_case_7_open_valid_until_remains_active(self):
        """Case 7: Policy with valid_until = 'OPEN' remains active indefinitely."""
        interval = RunStartInterval(
            run_genesis_event_id="gen_007",
            run_genesis_sequence=1,
            run_started_at_local="2027-01-01T00:00:00Z",
            run_start_clock_evidence_id="clk_007",
            run_started_at_interval_start="2027-01-01T00:00:00.000Z",
            run_started_at_interval_end="2027-01-01T00:00:02.000Z",
            run_epoch="EPOCH_003",
        )
        res, entry, b_class = self.default_engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.APPLICABLE)
        self.assertIsNotNone(entry)

    def test_case_8_valid_until_closed_run_before_end(self):
        """Case 8: Closed valid_until, run entirely before termination -> APPLICABLE."""
        closed_entry = PolicyRegistryEntry(
            policy_id="EXPIRING_POLICY",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="2026-10-12T00:00:00Z",
            min_run_epoch_ordinal=1,
        )
        reg = PolicyRegistry(version="1.0", entries=[closed_entry])
        engine = PolicyResolutionEngine(policy_registry=reg)

        interval = RunStartInterval(
            run_genesis_event_id="gen_008",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T18:00:00Z",
            run_start_clock_evidence_id="clk_008",
            run_started_at_interval_start="2026-10-11T18:00:00.000Z",
            run_started_at_interval_end="2026-10-11T18:00:02.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, _ = engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.APPLICABLE)
        self.assertIsNotNone(entry)

    def test_case_9_valid_until_closed_run_straddles_end(self):
        """Case 9: Closed valid_until, run straddles termination -> INDETERMINATE_BOUNDARY."""
        closed_entry = PolicyRegistryEntry(
            policy_id="EXPIRING_POLICY",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="2026-10-12T00:00:00Z",
            min_run_epoch_ordinal=1,
        )
        reg = PolicyRegistry(version="1.0", entries=[closed_entry])
        engine = PolicyResolutionEngine(policy_registry=reg)

        interval = RunStartInterval(
            run_genesis_event_id="gen_009",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-12T00:00:00Z",
            run_start_clock_evidence_id="clk_009",
            run_started_at_interval_start="2026-10-11T23:59:59.000Z",
            run_started_at_interval_end="2026-10-12T00:00:01.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, b_class = engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.INDETERMINATE_BOUNDARY)
        self.assertIsNone(entry)

    def test_case_10_valid_until_closed_run_after_end(self):
        """Case 10: Closed valid_until, run starts after termination -> NO_APPLICABLE_POLICY."""
        closed_entry = PolicyRegistryEntry(
            policy_id="EXPIRING_POLICY",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="2026-10-12T00:00:00Z",
            min_run_epoch_ordinal=1,
        )
        reg = PolicyRegistry(version="1.0", entries=[closed_entry])
        engine = PolicyResolutionEngine(policy_registry=reg)

        interval = RunStartInterval(
            run_genesis_event_id="gen_010",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-12T01:00:00Z",
            run_start_clock_evidence_id="clk_010",
            run_started_at_interval_start="2026-10-12T01:00:00.000Z",
            run_started_at_interval_end="2026-10-12T01:00:02.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, _ = engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.NO_APPLICABLE_POLICY)
        self.assertIsNone(entry)

    def test_case_11_active_supersession_v1_and_v2_disjoint(self):
        """Case 11: Normal supersession with contiguous half-open intervals."""
        v1_entry = PolicyRegistryEntry(
            policy_id="SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="2026-10-15T12:00:00Z",
            min_run_epoch_ordinal=1,
        )
        v2_entry = PolicyRegistryEntry(
            policy_id="SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V2",
            policy_version="2.0",
            policy_sha256="def67890" * 8,
            valid_from="2026-10-15T12:00:00Z",
            valid_until="OPEN",
            min_run_epoch_ordinal=1,
        )
        reg = PolicyRegistry(version="2.0", entries=[v1_entry, v2_entry])
        reg.validate_integrity()  # Must not raise
        engine = PolicyResolutionEngine(policy_registry=reg)

        # Run in V1 era
        interval_v1 = RunStartInterval(
            run_genesis_event_id="gen_v1",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-12T00:00:00Z",
            run_start_clock_evidence_id="clk_v1",
            run_started_at_interval_start="2026-10-12T00:00:00.000Z",
            run_started_at_interval_end="2026-10-12T00:00:02.000Z",
            run_epoch="EPOCH_001",
        )
        res1, entry1, _ = engine.resolve_policy_for_run(interval_v1)
        self.assertEqual(res1, PolicyResolutionResult.APPLICABLE)
        self.assertEqual(entry1.policy_version, "1.0")

        # Run in V2 era
        interval_v2 = RunStartInterval(
            run_genesis_event_id="gen_v2",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-16T00:00:00Z",
            run_start_clock_evidence_id="clk_v2",
            run_started_at_interval_start="2026-10-16T00:00:00.000Z",
            run_started_at_interval_end="2026-10-16T00:00:02.000Z",
            run_epoch="EPOCH_001",
        )
        res2, entry2, _ = engine.resolve_policy_for_run(interval_v2)
        self.assertEqual(res2, PolicyResolutionResult.APPLICABLE)
        self.assertEqual(entry2.policy_version, "2.0")

    def test_case_12_registry_conflict_overlapping_intervals_rejected(self):
        """Case 12: Overlapping intervals raise ValueError on validate_integrity."""
        entry_a = PolicyRegistryEntry(
            policy_id="CONFLICT_POLICY",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="2026-10-13T12:00:00Z",
            min_run_epoch_ordinal=1,
        )
        entry_b = PolicyRegistryEntry(
            policy_id="CONFLICT_POLICY",
            policy_version="1.1",
            policy_sha256="def67890" * 8,
            valid_from="2026-10-12T12:00:00Z",  # Overlaps with entry_a!
            valid_until="2026-10-14T12:00:00Z",
            min_run_epoch_ordinal=1,
        )
        reg = PolicyRegistry(version="1.0", entries=[entry_a, entry_b])
        with self.assertRaises(ValueError):
            reg.validate_integrity()

        # If unvalidated registry is evaluated, engine detects conflict
        engine = PolicyResolutionEngine(policy_registry=reg)
        interval = RunStartInterval(
            run_genesis_event_id="gen_overlap",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-12T18:00:00Z",
            run_start_clock_evidence_id="clk_overlap",
            run_started_at_interval_start="2026-10-12T18:00:00.000Z",
            run_started_at_interval_end="2026-10-12T18:00:02.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, _ = engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.CONFLICTING_APPLICABLE_POLICIES)
        self.assertIsNone(entry)

    def test_case_13_revoked_for_future_runs_blocks_new_run_binding(self):
        """Case 13: Entry marked REVOKED_FOR_FUTURE_RUNS is not applicable to new runs."""
        revoked_entry = PolicyRegistryEntry(
            policy_id="SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="OPEN",
            min_run_epoch_ordinal=1,
            revocation_status=RevocationStatus.REVOKED_FOR_FUTURE_RUNS,
            revocation_effective_at="2026-10-11T13:00:00Z",
            revocation_reason="Emergency supersession for future runs",
        )
        reg = PolicyRegistry(version="1.0", entries=[revoked_entry])
        engine = PolicyResolutionEngine(policy_registry=reg)

        interval = RunStartInterval(
            run_genesis_event_id="gen_revoked_future",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:00:00Z",
            run_start_clock_evidence_id="clk_rev",
            run_started_at_interval_start="2026-10-11T14:00:00.000Z",
            run_started_at_interval_end="2026-10-11T14:00:02.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, _ = engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.NO_APPLICABLE_POLICY)
        self.assertIsNone(entry)

    def test_case_14_revoked_for_authority_use_blocks_downstream(self):
        """Case 14: REVOKED_FOR_AUTHORITY_USE blocks downstream firewall unconditionally."""
        revoked_entry = PolicyRegistryEntry(
            policy_id="SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="OPEN",
            min_run_epoch_ordinal=1,
            revocation_status=RevocationStatus.REVOKED_FOR_AUTHORITY_USE,
            revocation_effective_at="2026-10-11T13:00:00Z",
            revocation_reason="Critical defect: revoking all authority use",
        )
        reg = PolicyRegistry(version="1.0", entries=[revoked_entry])
        engine = PolicyResolutionEngine(policy_registry=reg)

        interval = RunStartInterval(
            run_genesis_event_id="gen_revoked_auth",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:00:00Z",
            run_start_clock_evidence_id="clk_rev_auth",
            run_started_at_interval_start="2026-10-11T14:00:00.000Z",
            run_started_at_interval_end="2026-10-11T14:00:02.000Z",
            run_epoch="EPOCH_001",
        )
        res, entry, _ = engine.resolve_policy_for_run(interval)
        self.assertEqual(res, PolicyResolutionResult.NO_APPLICABLE_POLICY)

        # Downstream firewall must reject non-APPLICABLE resolution
        with self.assertRaises(DownstreamFirewallBlockedError):
            DownstreamUseFirewall.assert_policy_resolution_allowed(res)

    def test_case_15_historical_replay_exact_parity(self):
        """Case 15: Independent historical replay yields byte/hash exact parity."""
        interval = RunStartInterval(
            run_genesis_event_id="gen_parity_001",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T14:05:00+02:00",
            run_start_clock_evidence_id="clk_parity",
            run_started_at_interval_start="2026-10-11T12:04:59.000Z",
            run_started_at_interval_end="2026-10-11T12:05:01.000Z",
            run_epoch="EPOCH_001",
        )
        raw_metadata = {
            "run_genesis_event_id": "gen_parity_001",
            "run_genesis_sequence": 1,
            "clock_evidence_id": "clk_parity",
        }
        # 1. Store binding at genesis
        binding = self.default_engine.bind_run_genesis(
            interval=interval,
            binding_event_id="evt_bind_001",
            binding_time_utc="2026-10-11T12:05:02Z",
        )
        self.assertEqual(binding.policy_applicability_result, PolicyResolutionResult.APPLICABLE)
        self.assertEqual(binding.bound_policy_id, "SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1")

        # 2. Independent Reviewer Replay
        audit_result = AnchorFinalizationReviewer.audit_policy_binding(
            binding=binding,
            interval=interval,
            raw_genesis_metadata=raw_metadata,
            historical_policy_registry=DEFAULT_POLICY_REGISTRY,
            historical_epoch_registry=DEFAULT_EPOCH_REGISTRY,
        )
        self.assertEqual(audit_result["policy_binding_replay_parity"], "PASS")

        # 3. Tamper test: if stored binding has altered policy_id, replay parity FAILS
        tampered_binding = RunPolicyBinding(
            run_policy_binding_id=binding.run_policy_binding_id,
            bound_policy_id="TAMPERED_POLICY_ID",
            bound_policy_version=binding.bound_policy_version,
            bound_policy_sha256=binding.bound_policy_sha256,
            bound_policy_registry_sha256=binding.bound_policy_registry_sha256,
            bound_epoch_registry_sha256=binding.bound_epoch_registry_sha256,
            run_epoch=binding.run_epoch,
            run_epoch_ordinal=binding.run_epoch_ordinal,
            run_start_interval_start=binding.run_start_interval_start,
            run_start_interval_end=binding.run_start_interval_end,
            policy_applicability_result=binding.policy_applicability_result,
            boundary_classification=binding.boundary_classification,
            policy_binding_at=binding.policy_binding_at,
            policy_binding_event_id=binding.policy_binding_event_id,
        )
        tampered_audit = AnchorFinalizationReviewer.audit_policy_binding(
            binding=tampered_binding,
            interval=interval,
            raw_genesis_metadata=raw_metadata,
        )
        self.assertEqual(tampered_audit["policy_binding_replay_parity"], "FAIL")

    # =========================================================================
    # Section 20: Clock Boundary Edge-Case Tests
    # =========================================================================

    def test_clock_boundary_precision_cases(self):
        """Tests exact mathematical relations around valid_from."""
        target = "2026-10-11T12:00:00.000Z"

        # Case 1: Strictly BEFORE ([T-2, T-1])
        c1 = classify_run_interval_boundary(
            "2026-10-11T11:59:58.000Z",
            "2026-10-11T11:59:59.000Z",
            target,
        )
        self.assertEqual(c1, BoundaryClassification.BEFORE)

        # Case 2: Straddles boundary ([T-0.5, T+0.5])
        c2 = classify_run_interval_boundary(
            "2026-10-11T11:59:59.500Z",
            "2026-10-11T12:00:00.500Z",
            target,
        )
        self.assertEqual(c2, BoundaryClassification.STRADDLES_BOUNDARY)

        # Case 3: Strictly AFTER_OR_EQUAL ([T, T+1])
        c3 = classify_run_interval_boundary(
            "2026-10-11T12:00:00.000Z",
            "2026-10-11T12:00:01.000Z",
            target,
        )
        self.assertEqual(c3, BoundaryClassification.AFTER_OR_EQUAL)

        # Case 4: Strictly AFTER ([T+1, T+2])
        c4 = classify_run_interval_boundary(
            "2026-10-11T12:00:01.000Z",
            "2026-10-11T12:00:02.000Z",
            target,
        )
        self.assertEqual(c4, BoundaryClassification.AFTER_OR_EQUAL)

    # =========================================================================
    # Section 21: Registry Conflict Tests
    # =========================================================================

    def test_registry_conflict_duplicate_keys_rejected(self):
        """Duplicate (policy_id, policy_version) rejected on validate_integrity."""
        entry1 = PolicyRegistryEntry(
            policy_id="DUPLICATE_POLICY",
            policy_version="1.0",
            policy_sha256="abc12345" * 8,
            valid_from="2026-10-11T12:00:00Z",
            valid_until="2026-10-12T00:00:00Z",
            min_run_epoch_ordinal=1,
        )
        entry2 = PolicyRegistryEntry(
            policy_id="DUPLICATE_POLICY",
            policy_version="1.0",
            policy_sha256="def67890" * 8,
            valid_from="2026-10-12T00:00:00Z",
            valid_until="OPEN",
            min_run_epoch_ordinal=1,
        )
        reg = PolicyRegistry(version="1.0", entries=[entry1, entry2])
        with self.assertRaises(ValueError) as ctx:
            reg.validate_integrity()
        self.assertIn("Duplicate policy entry", str(ctx.exception))

    # =========================================================================
    # Section 22: Revocation Tests
    # =========================================================================

    def test_revocation_status_enum_values(self):
        """Verifies RevocationStatus has all required states."""
        self.assertEqual(RevocationStatus.ACTIVE.value, "ACTIVE")
        self.assertEqual(RevocationStatus.REVOKED_FOR_FUTURE_RUNS.value, "REVOKED_FOR_FUTURE_RUNS")
        self.assertEqual(RevocationStatus.REVOKED_FOR_AUTHORITY_USE.value, "REVOKED_FOR_AUTHORITY_USE")

    # =========================================================================
    # Section 23: No Retroactive Policy Application Tests
    # =========================================================================

    def test_no_retroactive_policy_rebinding(self):
        """Runs bound at genesis are immutable and cannot be rebound mid-run."""
        self.assertTrue(POLICY_BINDING_IMMUTABLE)
        self.assertFalse(NORMAL_POLICY_SUPERSESSION_REBINDS_EXISTING_RUN)
        self.assertFalse(EMERGENCY_REVOCATION_REWRITES_ORIGINAL_BINDING)
        self.assertEqual(MID_RUN_POLICY_REBINDING, "PROHIBITED")

        interval = RunStartInterval(
            run_genesis_event_id="gen_noretro_001",
            run_genesis_sequence=1,
            run_started_at_local="2026-10-11T12:05:00Z",
            run_start_clock_evidence_id="clk_noretro",
            run_started_at_interval_start="2026-10-11T12:05:00.000Z",
            run_started_at_interval_end="2026-10-11T12:05:02.000Z",
            run_epoch="EPOCH_001",
        )
        binding = self.default_engine.bind_run_genesis(
            interval=interval,
            binding_event_id="evt_noretro",
            binding_time_utc="2026-10-11T12:05:03Z",
        )
        self.assertTrue(binding.is_immutable)
        self.assertEqual(binding.bound_policy_version, "1.0")

    # =========================================================================
    # Section 24: Downstream Policy Firewall Enforcement Tests
    # =========================================================================

    def test_downstream_firewall_blocks_all_non_applicable_resolutions(self):
        """DownstreamUseFirewall allows ONLY PolicyResolutionResult.APPLICABLE."""
        # Allowed
        DownstreamUseFirewall.assert_policy_resolution_allowed(PolicyResolutionResult.APPLICABLE)

        # Blocked states
        blocked_results = [
            PolicyResolutionResult.INDETERMINATE_BOUNDARY,
            PolicyResolutionResult.NO_APPLICABLE_POLICY,
            PolicyResolutionResult.CONFLICTING_APPLICABLE_POLICIES,
            PolicyResolutionResult.BLOCKED_UNKNOWN_EPOCH,
            PolicyResolutionResult.BLOCKED_REVOKED_POLICY,
            PolicyResolutionResult.BLOCKED_NO_POLICY,
        ]
        for b_res in blocked_results:
            with self.subTest(result=b_res):
                with self.assertRaises(DownstreamFirewallBlockedError):
                    DownstreamUseFirewall.assert_policy_resolution_allowed(b_res)

    # =========================================================================
    # Section 25: Policy Artifact Retention Verification
    # =========================================================================

    def test_retained_policy_artifacts_completeness(self):
        """Verifies all 11 policy artifacts are retained and readable."""
        self.assertTrue(verify_all_policy_artifacts_retained())

        # Verify specifically Section 25 new artifacts
        v1_registry_doc = get_policy_artifact("anchor_policy_registry_v1")
        self.assertIsNotNone(v1_registry_doc)
        raw_text_v1 = v1_registry_doc["policy_artifact_raw_bytes"].decode("utf-8")
        self.assertIn("SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1", raw_text_v1)

        epoch_registry_doc = get_policy_artifact("epoch_registry_v1")
        self.assertIsNotNone(epoch_registry_doc)
        raw_text_epoch = epoch_registry_doc["policy_artifact_raw_bytes"].decode("utf-8")
        self.assertIn("EPOCH_001", raw_text_epoch)

        contract_doc = get_policy_artifact("policy_resolution_contract_v1")
        self.assertIsNotNone(contract_doc)
        raw_text_contract = contract_doc["policy_artifact_raw_bytes"].decode("utf-8")
        self.assertIn("ALL_PREDICATES_REQUIRED", raw_text_contract)


if __name__ == "__main__":
    unittest.main()
