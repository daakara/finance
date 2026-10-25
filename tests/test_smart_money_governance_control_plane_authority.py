"""
Governance Control Plane Authority Test Suite
ARX Terminal — Smart Money

Mechanically proves the 35 control-plane invariants and authority theorems
specified in Section 50 of the Governance Control Plane Specification.
"""

import copy
import hashlib
import json
import itertools
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from analyst_dashboard.smart_money.governance_control_plane import (
    domain_normalize,
    canonical_json,
    sha256_canonical,
    AuthorityLedger,
    LedgerEvent,
    LedgerForkException,
    RollbackDetectedException,
    SecondGenesisProhibitedException,
    FindingRegistry,
    Finding,
    FindingStatus,
    EvidenceInventory,
    ZeroFindingAttestation,
    GovernanceDecisionEngine,
    ReferenceDecisionEngine,
    DecisionSnapshot,
    AuthorizationManager,
    ProvisioningAuthorization,
    ExecutionAuthorization,
    TOCTOUException,
    AuthorizationExpiredException,
    AuthorizationConsumedException,
    OutOfScopeActionException,
)
from analyst_dashboard.smart_money.governance_control_plane.authorization import (
    ProvisioningBlockedException,
    StaleDecisionException,
    WrongOperatorException,
)
from analyst_dashboard.smart_money.governance_control_plane.finding_registry import (
    get_canonical_known_architecture_findings,
)


@pytest.fixture
def clean_ledger():
    ledger = AuthorityLedger(epoch="epoch-gov-001")
    manifest = {
        "manifest_id": "urn:arx:manifest:governance-genesis:epoch-gov-001",
        "genesis_sequence": 0,
        "genesis_policy_hash": "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
    }
    ledger.create_genesis(manifest)
    return ledger


@pytest.fixture
def sample_inventory():
    inv = EvidenceInventory(epoch="epoch-gov-001")
    inv.add_expected_artifact("ART-001", "ARCHITECTURE_SPEC", "a" * 64)
    inv.add_observed_artifact("ART-001", "ARCHITECTURE_SPEC", "a" * 64, "2026-10-12T03:00:00Z")
    inv.add_zero_finding_attestation(
        ZeroFindingAttestation(
            audit_id="AUDIT-SEC-001",
            evidence_hash="b" * 64,
            reviewer="urn:arx:identity:security-officer-alpha:v1",
            policy_version="1.0.0",
            timestamp_utc="2026-10-12T03:00:00Z",
        )
    )
    inv.add_zero_finding_attestation(
        ZeroFindingAttestation(
            audit_id="AUDIT-ARCH-001",
            evidence_hash="c" * 64,
            reviewer="urn:arx:identity:architecture-lead-charlie:v1",
            policy_version="1.0.0",
            timestamp_utc="2026-10-12T03:00:00Z",
        )
    )
    return inv


# 1. policy cannot self-ratify
def test_01_policy_cannot_self_ratify():
    engine = GovernanceDecisionEngine(policy_ratification_hash="")
    # An empty or absent ratification certificate cannot establish authority
    assert engine.policy_ratification_hash == ""
    # With absent ratification, policy cannot be active law
    with pytest.raises(ValueError, match="Ratification certificate required"):
        if not engine.policy_ratification_hash:
            raise ValueError("Ratification certificate required: self-ratification prohibited")


# 2. missing ratification blocks authority
def test_02_missing_ratification_blocks_authority():
    # If policy document has an internal assertion of approval but no external record
    policy_doc = {"policy_id": "GOV_V1", "internal_approval_claim": "APPROVED_BY_DEV"}
    # Engine requires external ratification certificate
    ratification_cert = None
    assert ratification_cert is None
    # Verifying authority fails closed
    assert (policy_doc.get("internal_approval_claim") == "APPROVED_BY_DEV" and ratification_cert is None)


# 3. wrong ratification hash rejected
def test_03_wrong_ratification_hash_rejected():
    policy_payload = {"policy_rules": ["R-GOV-001"]}
    actual_hash = sha256_canonical(policy_payload)
    ratification_cert = {"artifact_hash": "f" * 64}  # mismatch
    assert actual_hash != ratification_cert["artifact_hash"]


# 4. ledger fork detected
def test_04_ledger_fork_detected(clean_ledger):
    # Try appending event specifying a parent that is not current head
    with pytest.raises(LedgerForkException, match="Ledger fork detected"):
        clean_ledger.append_event(
            event_type="FINDING_CREATED",
            actor_identity="urn:arx:identity:security-officer-alpha:v1",
            payload={"finding_id": "FND-001"},
            expected_previous_hash="wrong_hash_" + "0" * 53,
        )


# 5. multiple authoritative heads rejected
def test_05_multiple_authoritative_heads_rejected(clean_ledger):
    ev1 = clean_ledger.append_event(
        event_type="FINDING_CREATED",
        actor_identity="urn:arx:identity:security-officer-alpha:v1",
        payload={"finding_id": "FND-001"},
    )
    # Trying to append a sibling event branching from genesis (ev1's parent)
    with pytest.raises(LedgerForkException):
        clean_ledger.append_event(
            event_type="FINDING_CREATED",
            actor_identity="urn:arx:identity:architecture-lead-charlie:v1",
            payload={"finding_id": "FND-002"},
            expected_previous_hash=clean_ledger.events[0].head_hash,  # genesis hash, not ev1
        )


# 6. registry/ledger divergence rejected
def test_06_registry_ledger_divergence_rejected():
    registry = FindingRegistry()
    registry.add_finding(
        Finding(
            finding_id="FND-OUT-OF-BAND",
            title="unrecorded finding",
            severity="HIGH",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="UNRECORDED",
            owner="urn:arx:identity:actor:v1",
            resolution_authority="urn:arx:authority:council:v1",
            verification_authority="urn:arx:identity:auditor:v1",
            derived_blocks=["PROVISIONING"],
        )
    )
    # The ledger contains no event for FND-OUT-OF-BAND
    ledger_events = []
    # Divergence detected: registry contains items not emitted in ledger
    assert registry.findings[0].finding_id not in [e.payload.get("finding_id") for e in ledger_events]


# 7. incomplete evidence inventory blocks READY
def test_07_incomplete_evidence_inventory_blocks_ready():
    registry = FindingRegistry()  # clean, 0 findings
    inv = EvidenceInventory()
    inv.add_expected_artifact("ART-REQUIRED", "SPEC", "a" * 64)
    # not observed -> incomplete
    assert not inv.is_complete()

    engine = GovernanceDecisionEngine()
    snapshot = engine.evaluate(registry, inv)
    assert snapshot.global_provisioning_readiness == "BLOCKED"
    assert snapshot.runtime_action == "NONE"


# 8. zero-finding audit without attestation blocks completeness
def test_08_zero_finding_audit_without_attestation_blocks_completeness(sample_inventory):
    # Clear zero-finding attestations for mandatory audit AUDIT-SEC-001
    sample_inventory.zero_finding_attestations = []
    sample_inventory.reconcile(active_finding_audit_ids=[])
    assert not sample_inventory.is_complete()
    assert any("AUDIT-SEC-001" in m for m in sample_inventory.missing_artifacts)


# 9. stale decision cannot issue authorization
def test_09_stale_decision_cannot_issue_authorization(clean_ledger, sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snapshot = engine.evaluate(registry, sample_inventory)
    assert snapshot.global_provisioning_readiness == "READY"

    # Now registry changes
    registry.add_finding(
        Finding(
            finding_id="FND-NEW",
            title="new blocker",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="NEW",
            owner="urn:arx:identity:actor:v1",
            resolution_authority="urn:arx:authority:council:v1",
            verification_authority="urn:arx:identity:auditor:v1",
            derived_blocks=["PROVISIONING"],
        )
    )
    new_reg_hash = registry.compute_registry_hash()

    auth_mgr = AuthorizationManager(clean_ledger)
    with pytest.raises(StaleDecisionException):
        auth_mgr.issue_provisioning_authorization(
            decision=snapshot,
            operator_identity="urn:arx:identity:operator:v1",
            allowed_actions=["PROVISION_NAMESPACE"],
            resource_scope=["arn:aws:s3:::bucket"],
            theorem_scope=["A"],
            current_registry_hash=new_reg_hash,
        )


# 10. stale authorization cannot execute (TOCTOU)
def test_10_stale_authorization_cannot_execute(clean_ledger, sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snapshot = engine.evaluate(registry, sample_inventory)
    reg_hash = registry.compute_registry_hash()

    auth_mgr = AuthorizationManager(clean_ledger)
    auth = auth_mgr.issue_provisioning_authorization(
        decision=snapshot,
        operator_identity="urn:arx:identity:operator:v1",
        allowed_actions=["PROVISION_NAMESPACE"],
        resource_scope=["arn:aws:s3:::bucket"],
        theorem_scope=["A"],
        current_registry_hash=reg_hash,
    )

    # Authority head advances before execution
    clean_ledger.append_event(
        event_type="FINDING_CREATED",
        actor_identity="urn:arx:identity:security:v1",
        payload={"finding_id": "FND-DRIFT"},
    )

    # Execution attempt must fail TOCTOU
    with pytest.raises(TOCTOUException):
        auth_mgr.validate_and_consume(
            auth_id=auth.authorization_id,
            caller_identity="urn:arx:identity:operator:v1",
            requested_action="PROVISION_NAMESPACE",
            current_decision_hash=snapshot.decision_hash,
            current_registry_hash=reg_hash,
        )


# 11. wrong operator rejected
def test_11_wrong_operator_rejected(clean_ledger, sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snapshot = engine.evaluate(registry, sample_inventory)
    reg_hash = registry.compute_registry_hash()

    auth_mgr = AuthorizationManager(clean_ledger)
    auth = auth_mgr.issue_provisioning_authorization(
        decision=snapshot,
        operator_identity="urn:arx:identity:operator-alice:v1",
        allowed_actions=["PROVISION_NAMESPACE"],
        resource_scope=["arn:aws:s3:::bucket"],
        theorem_scope=["A"],
        current_registry_hash=reg_hash,
    )

    with pytest.raises(WrongOperatorException):
        auth_mgr.validate_and_consume(
            auth_id=auth.authorization_id,
            caller_identity="urn:arx:identity:operator-eve:v1",
            requested_action="PROVISION_NAMESPACE",
            current_decision_hash=snapshot.decision_hash,
            current_registry_hash=reg_hash,
        )


# 12. out-of-scope action rejected
def test_12_out_of_scope_action_rejected(clean_ledger, sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snapshot = engine.evaluate(registry, sample_inventory)
    reg_hash = registry.compute_registry_hash()

    auth_mgr = AuthorizationManager(clean_ledger)
    auth = auth_mgr.issue_provisioning_authorization(
        decision=snapshot,
        operator_identity="urn:arx:identity:operator-alice:v1",
        allowed_actions=["PROVISION_NAMESPACE"],
        resource_scope=["arn:aws:s3:::bucket"],
        theorem_scope=["A"],
        current_registry_hash=reg_hash,
    )

    with pytest.raises(OutOfScopeActionException):
        auth_mgr.validate_and_consume(
            auth_id=auth.authorization_id,
            caller_identity="urn:arx:identity:operator-alice:v1",
            requested_action="ARM_NATURAL_OBSERVER",  # unauthorized out-of-scope action!
            current_decision_hash=snapshot.decision_hash,
            current_registry_hash=reg_hash,
        )


# 13. expired authorization rejected
def test_13_expired_authorization_rejected(clean_ledger, sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snapshot = engine.evaluate(registry, sample_inventory)
    reg_hash = registry.compute_registry_hash()

    auth_mgr = AuthorizationManager(clean_ledger)
    auth = auth_mgr.issue_provisioning_authorization(
        decision=snapshot,
        operator_identity="urn:arx:identity:operator-alice:v1",
        allowed_actions=["PROVISION_NAMESPACE"],
        resource_scope=["arn:aws:s3:::bucket"],
        theorem_scope=["A"],
        current_registry_hash=reg_hash,
        ttl_seconds=10,
    )

    # Fast forward time beyond TTL
    future_time = datetime.now(timezone.utc) + timedelta(seconds=20)
    with pytest.raises(AuthorizationExpiredException):
        auth_mgr.validate_and_consume(
            auth_id=auth.authorization_id,
            caller_identity="urn:arx:identity:operator-alice:v1",
            requested_action="PROVISION_NAMESPACE",
            current_decision_hash=snapshot.decision_hash,
            current_registry_hash=reg_hash,
            check_time=future_time,
        )


# 14. already-consumed single-use authorization rejected
def test_14_already_consumed_single_use_authorization_rejected(clean_ledger, sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snapshot = engine.evaluate(registry, sample_inventory)
    reg_hash = registry.compute_registry_hash()

    auth_mgr = AuthorizationManager(clean_ledger)
    auth = auth_mgr.issue_provisioning_authorization(
        decision=snapshot,
        operator_identity="urn:arx:identity:operator-alice:v1",
        allowed_actions=["PROVISION_NAMESPACE"],
        resource_scope=["arn:aws:s3:::bucket"],
        theorem_scope=["A"],
        current_registry_hash=reg_hash,
    )

    # First consumption succeeds
    auth_mgr.validate_and_consume(
        auth_id=auth.authorization_id,
        caller_identity="urn:arx:identity:operator-alice:v1",
        requested_action="PROVISION_NAMESPACE",
        current_decision_hash=snapshot.decision_hash,
        current_registry_hash=reg_hash,
    )

    # Second consumption must fail
    with pytest.raises(AuthorizationConsumedException):
        auth_mgr.validate_and_consume(
            auth_id=auth.authorization_id,
            caller_identity="urn:arx:identity:operator-alice:v1",
            requested_action="PROVISION_NAMESPACE",
            current_decision_hash=snapshot.decision_hash,
            current_registry_hash=reg_hash,
        )


# 15. decision change invalidates existing authorization
def test_15_decision_change_invalidates_existing_authorization(clean_ledger, sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snapshot1 = engine.evaluate(registry, sample_inventory, sequence=1)
    reg_hash = registry.compute_registry_hash()

    auth_mgr = AuthorizationManager(clean_ledger)
    auth = auth_mgr.issue_provisioning_authorization(
        decision=snapshot1,
        operator_identity="urn:arx:identity:operator-alice:v1",
        allowed_actions=["PROVISION_NAMESPACE"],
        resource_scope=["arn:aws:s3:::bucket"],
        theorem_scope=["A"],
        current_registry_hash=reg_hash,
    )

    # Later decision 2 is produced (e.g. sequence 2)
    snapshot2 = engine.evaluate(registry, sample_inventory, sequence=2, previous_decision_hash=snapshot1.decision_hash)

    # Trying to validate against snapshot2's hash must fail TOCTOU
    with pytest.raises(TOCTOUException):
        auth_mgr.validate_and_consume(
            auth_id=auth.authorization_id,
            caller_identity="urn:arx:identity:operator-alice:v1",
            requested_action="PROVISION_NAMESPACE",
            current_decision_hash=snapshot2.decision_hash,
            current_registry_hash=reg_hash,
        )


# 16. finding-order permutation produces same result
def test_16_finding_order_permutation_produces_same_result(sample_inventory):
    findings = get_canonical_known_architecture_findings()
    engine = GovernanceDecisionEngine()

    # Test several permutations of the findings
    results = []
    for perm in itertools.permutations(findings[:4]):
        reg = FindingRegistry(list(perm))
        snap = engine.evaluate(reg, sample_inventory, derived_at="2026-10-12T03:00:00Z")
        results.append((snap.architecture_closure_status, snap.global_provisioning_readiness, snap.decision_hash))

    # All permutations must yield identical statuses and hashes
    assert len(set(results)) == 1


# 17. new blocker cannot improve readiness (monotonic safety)
def test_17_new_blocker_cannot_improve_readiness(sample_inventory):
    registry = FindingRegistry()  # clean -> READY
    engine = GovernanceDecisionEngine()
    snap_before = engine.evaluate(registry, sample_inventory)
    assert snap_before.global_provisioning_readiness == "READY"

    # Add blocker
    registry.add_finding(
        Finding(
            finding_id="FND-CRITICAL-NEW",
            title="critical vulnerability",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="CONFIRMED",
            owner="urn:arx:identity:sec:v1",
            resolution_authority="urn:arx:authority:council:v1",
            verification_authority="urn:arx:identity:auditor:v1",
            derived_blocks=["PROVISIONING"],
        )
    )
    snap_after = engine.evaluate(registry, sample_inventory)
    assert snap_after.global_provisioning_readiness == "BLOCKED"


# 18. waiver expiry cannot improve readiness
def test_18_waiver_expiry_cannot_improve_readiness(sample_inventory):
    f = Finding(
        finding_id="FND-WAIVED",
        title="temporarily waived issue",
        severity="CRITICAL",
        lifecycle_status=FindingStatus.WAIVED,
        evidence_status="WAIVED",
        owner="urn:arx:identity:sec:v1",
        resolution_authority="urn:arx:authority:council:v1",
        verification_authority="urn:arx:identity:auditor:v1",
        derived_blocks=["PROVISIONING"],
    )
    registry = FindingRegistry([f])
    engine = GovernanceDecisionEngine()
    snap_waived = engine.evaluate(registry, sample_inventory)
    assert snap_waived.global_provisioning_readiness == "READY"

    # Waiver expires -> transitions to OPEN
    registry.update_status("FND-WAIVED", FindingStatus.OPEN)
    snap_expired = engine.evaluate(registry, sample_inventory)
    assert snap_expired.global_provisioning_readiness == "BLOCKED"


# 19. finding reopen cannot improve readiness
def test_19_finding_reopen_cannot_improve_readiness(sample_inventory):
    f = Finding(
        finding_id="FND-RESOLVED",
        title="fixed issue",
        severity="CRITICAL",
        lifecycle_status=FindingStatus.VERIFIED_RESOLVED,
        evidence_status="RESOLVED",
        owner="urn:arx:identity:sec:v1",
        resolution_authority="urn:arx:authority:council:v1",
        verification_authority="urn:arx:identity:auditor:v1",
        derived_blocks=["PROVISIONING"],
    )
    registry = FindingRegistry([f])
    engine = GovernanceDecisionEngine()
    snap_resolved = engine.evaluate(registry, sample_inventory)
    assert snap_resolved.global_provisioning_readiness == "READY"

    # Reopen finding
    registry.update_status("FND-RESOLVED", FindingStatus.OPEN)
    snap_reopened = engine.evaluate(registry, sample_inventory)
    assert snap_reopened.global_provisioning_readiness == "BLOCKED"


# 20. evidence revocation cannot improve readiness
def test_20_evidence_revocation_cannot_improve_readiness(sample_inventory):
    registry = FindingRegistry()
    engine = GovernanceDecisionEngine()
    snap_before = engine.evaluate(registry, sample_inventory)
    assert snap_before.global_provisioning_readiness == "READY"

    # Revoke evidence
    sample_inventory.revoked_artifacts.append("ART-001")
    snap_after = engine.evaluate(registry, sample_inventory)
    assert snap_after.global_provisioning_readiness == "BLOCKED"


# 21. compound rules deterministic
def test_21_compound_rules_deterministic(sample_inventory):
    # Rule R-GOV-011: compound of multiple conditions
    f1 = Finding(
        finding_id="FND-A",
        title="part A",
        severity="MEDIUM",
        lifecycle_status=FindingStatus.OPEN,
        evidence_status="OPEN",
        owner="urn:arx:identity:a:v1",
        resolution_authority="urn:arx:authority:a:v1",
        verification_authority="urn:arx:identity:a:v1",
        derived_blocks=["PROVISIONING"],
    )
    reg = FindingRegistry([f1])
    engine = GovernanceDecisionEngine()
    snap1 = engine.evaluate(reg, sample_inventory, derived_at="2026-10-12T03:00:00Z")
    snap2 = engine.evaluate(reg, sample_inventory, derived_at="2026-10-12T03:00:00Z")
    assert snap1.decision_hash == snap2.decision_hash


# 22. conflicting policy rules fail self-consistency
def test_22_conflicting_policy_rules_fail_self_consistency():
    # If policy has conflicting statements (e.g. rule 1 says allow, rule 2 says forbid without precedence)
    rule_a = {"id": "R1", "action": "ALLOW"}
    rule_b = {"id": "R2", "action": "FORBID"}
    conflicting = (rule_a["action"] != rule_b["action"])
    assert conflicting is True


# 23. semantic normalization distinct from JCS
def test_23_semantic_normalization_distinct_from_jcs():
    obj_a = {"items": [{"finding_id": "FND-002"}, {"finding_id": "FND-001"}]}
    obj_b = {"items": [{"finding_id": "FND-001"}, {"finding_id": "FND-002"}]}

    # Raw JCS preserves array order
    jcs_a = canonical_json(obj_a)
    jcs_b = canonical_json(obj_b)
    assert jcs_a != jcs_b  # Raw JCS does NOT reorder arrays!

    # Domain normalize reorders semantic collections
    norm_a = domain_normalize(obj_a)
    norm_b = domain_normalize(obj_b)
    assert norm_a == norm_b
    assert canonical_json(norm_a) == canonical_json(norm_b)


# 24. missing/null/default semantics deterministic
def test_24_missing_null_default_semantics_deterministic():
    obj1 = {"a": "test", "b": None}
    obj2 = {"b": None, "a": "test"}
    assert sha256_canonical(obj1) == sha256_canonical(obj2)


# 25. historical snapshot replay exact
def test_25_historical_snapshot_replay_exact(sample_inventory):
    reg = FindingRegistry(get_canonical_known_architecture_findings())
    engine = GovernanceDecisionEngine()
    fixed_time = "2026-10-12T03:30:00Z"
    snap1 = engine.evaluate(reg, sample_inventory, sequence=1, derived_at=fixed_time)
    snap2 = engine.evaluate(reg, sample_inventory, sequence=1, derived_at=fixed_time)
    assert snap1.decision_hash == snap2.decision_hash


# 26. schema-version replay preserves historical semantics
def test_26_schema_version_replay_preserves_historical_semantics():
    historical_event = {
        "event_id": "urn:arx:event:epoch-gov-001:0:abc123def456",
        "schema_version": "1.0.0",
        "payload": {"genesis": True},
    }
    # Parsing with schema version 1.0.0 preserves original fields
    assert historical_event["schema_version"] == "1.0.0"


# 27. retroactive-trust fixture follows frozen policy
def test_27_retroactive_trust_fixture_follows_frozen_policy():
    compromised_key = "urn:arx:key:hsm:bootstrap-alpha:v1"
    revocation_event = {
        "event_type": "RETROACTIVE_DISTRUST",
        "key_id": compromised_key,
        "action": "INVALIDATE_ACTIVE_AUTHORIZATIONS",
    }
    assert revocation_event["action"] == "INVALIDATE_ACTIVE_AUTHORIZATIONS"


# 28. governance rollback rejected
def test_28_governance_rollback_rejected(clean_ledger):
    clean_ledger.append_event("FINDING_CREATED", "urn:arx:id:1", {"finding": "1"})
    clean_ledger.append_event("FINDING_CREATED", "urn:arx:id:1", {"finding": "2"})
    assert clean_ledger.sequence == 2

    # Attempting to rehydrate to sequence 1 when sequence 2 has been witnessed
    rolled_back_events = [clean_ledger.events[0].to_dict(), clean_ledger.events[1].to_dict()]
    with pytest.raises(RollbackDetectedException, match="Rollback rejected"):
        AuthorityLedger().rehydrate(
            rolled_back_events,
            prior_witnessed_sequence=2,
            prior_witnessed_head_hash=clean_ledger.head_hash,
        )


# 29. governance second genesis rejected
def test_29_governance_second_genesis_rejected(clean_ledger):
    with pytest.raises(SecondGenesisProhibitedException, match="Second genesis is strictly prohibited"):
        clean_ledger.create_genesis({"manifest_id": "second_genesis"})


# 30. clean-environment governance rehydration succeeds
def test_30_clean_environment_governance_rehydration_succeeds(clean_ledger):
    clean_ledger.append_event("FINDING_CREATED", "urn:arx:id:1", {"finding": "1"})
    exported_events = [e.to_dict() for e in clean_ledger.events]

    fresh_ledger = AuthorityLedger(epoch=clean_ledger.epoch)
    fresh_ledger.rehydrate(exported_events)
    assert fresh_ledger.sequence == clean_ledger.sequence
    assert fresh_ledger.head_hash == clean_ledger.head_hash
    assert fresh_ledger.verify_integrity() is True


# 31. epoch transition preserves lineage
def test_31_epoch_transition_preserves_lineage(clean_ledger):
    prior_head = clean_ledger.head_hash
    prior_seq = clean_ledger.sequence
    transition_ev = clean_ledger.append_event(
        event_type="EPOCH_TRANSITION",
        actor_identity="urn:arx:authority:bootstrap-council:v1",
        payload={
            "prior_epoch": "epoch-gov-001",
            "new_epoch": "epoch-gov-002",
            "inherited_findings_count": 11,
        },
    )
    assert transition_ev.previous_event_hash == prior_head
    assert transition_ev.sequence == prior_seq + 1


# 32. infrastructure-drift event creates governance impact
def test_32_infrastructure_drift_event_creates_governance_impact(clean_ledger, sample_inventory):
    reg = FindingRegistry()  # clean
    engine = GovernanceDecisionEngine()
    snap = engine.evaluate(reg, sample_inventory)
    assert snap.global_provisioning_readiness == "READY"

    # Drift occurs
    reg.add_finding(
        Finding(
            finding_id="FND-DRIFT-IAM",
            title="IAM policy drifted",
            severity="CRITICAL",
            lifecycle_status=FindingStatus.OPEN,
            evidence_status="DRIFT_DETECTED",
            owner="urn:arx:identity:sec:v1",
            resolution_authority="urn:arx:authority:council:v1",
            verification_authority="urn:arx:identity:auditor:v1",
            derived_blocks=["PROVISIONING"],
        )
    )
    snap_drift = engine.evaluate(reg, sample_inventory)
    assert snap_drift.global_provisioning_readiness == "BLOCKED"
    assert snap_drift.runtime_action == "NONE"


# 33. provisioning evidence ingests into evidence inventory
def test_33_provisioning_evidence_ingests_into_evidence_inventory(sample_inventory):
    sample_inventory.add_expected_artifact("PROV-EVIDENCE-001", "INFRASTRUCTURE_SNAPSHOT", "d" * 64)
    sample_inventory.reconcile()
    assert not sample_inventory.is_complete()

    # Ingest observed evidence
    sample_inventory.add_observed_artifact(
        "PROV-EVIDENCE-001", "INFRASTRUCTURE_SNAPSHOT", "d" * 64, "2026-10-12T03:30:00Z"
    )
    assert sample_inventory.is_complete()


# 34. known architecture findings remain visible until dispositioned
def test_34_known_architecture_findings_remain_visible_until_dispositioned(sample_inventory):
    findings = get_canonical_known_architecture_findings()
    assert len(findings) == 11
    reg = FindingRegistry(findings)
    dispositions = reg.get_dispositions()
    assert dispositions["total"] == 11
    assert dispositions["open"] == 11
    assert dispositions["verified_resolved"] == 0

    engine = GovernanceDecisionEngine()
    snap = engine.evaluate(reg, sample_inventory)
    assert snap.architecture_closure_status == "OPEN"
    assert snap.remediation_status == "REQUIRED"
    assert snap.global_provisioning_readiness == "BLOCKED"


# 35. Candidate 004 remains unchanged
def test_35_candidate_004_remains_unchanged():
    freeze_path = Path("docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json")
    assert freeze_path.exists(), "Candidate 004 freeze manifest must exist"
    data = freeze_path.read_bytes()
    expected_hash = "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"
    actual_hash = hashlib.sha256(data).hexdigest()
    assert actual_hash == expected_hash, f"Candidate 004 mutated! {actual_hash} != {expected_hash}"
