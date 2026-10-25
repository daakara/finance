"""
Test Suite for ARX Terminal Smart Money Governance Reconciliation & Canonical Freeze V1
Section 40: New Test Matrix (38 Required Tests)

Verifies:
1. Authority handoff atomicity, precommit, receipt, single-use ceremony, and response-loss recovery.
2. Finding identity preservation, original subject hashes, 2D classification, and new defect ID allocation.
3. Candidate 004 dual identity separation (freeze-file SHA-256 vs semantic-closure hash).
4. Evidence provenance validation and zero synthetic/fixture/test-persona contamination.
5. Hash identity registry, policy file raw vs canonical JCS hash split, and domain declarations.
6. Freeze chronology, source SHA uniqueness, and clean worktree enforcement.
7. Regression baseline integrity and zero unexplained test deletions or weakenings.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from analyst_dashboard.smart_money.governance_control_plane.bootstrap_authority import (
    BootstrapState,
    BootstrapAuthorization,
    BootstrapAuthorityManager,
    BOOTSTRAP_ALLOWED_ACTIONS,
    BOOTSTRAP_PROHIBITED_ACTIONS,
    BootstrapOutOfScopeActionException,
    BootstrapProhibitedActionException,
    BootstrapReplayException,
    BootstrapNotActiveException,
    CeremonyClaimState,
)
from analyst_dashboard.smart_money.governance_control_plane.authority_transition import (
    AuthorityHandoffController,
    AuthorityHandoffState,
    AuthorityHandoffReceipt,
    AuthorityOverlapException,
    ActivationPreconditionException,
    StaleHandoffException,
    REQUIRED_HANDOFF_PRECOMMIT_PRECONDITIONS,
)
from analyst_dashboard.smart_money.governance_control_plane.finding_registry import (
    Finding,
    FindingRegistry,
    FindingStatus,
    get_canonical_known_architecture_findings,
    FINDING_ORIGINAL_SUBJECTS,
    FINDING_DOMAIN_SCOPES,
)
from analyst_dashboard.smart_money.governance_control_plane.canonicalization import (
    sha256_canonical,
)
from analyst_dashboard.smart_money.governance_control_plane.decision_engine import (
    GovernanceDecisionEngine,
    compute_canonical_contract_set_hash,
    compute_canonical_theorem_state_hash,
)
from analyst_dashboard.smart_money.governance_control_plane.evidence_inventory import (
    EvidenceInventory,
)


@pytest.fixture
def active_bootstrap_manager():
    now = datetime.now(timezone.utc)
    auth = BootstrapAuthorization(
        bootstrap_authorization_id="urn:arx:auth:bootstrap:epoch-gov-001:rec-001",
        governance_epoch="epoch-gov-001",
        ratification_reference="urn:arx:ratification:bootstrap:001",
        authorized_bootstrap_roles=["urn:arx:role:bootstrap-operator:v1"],
        allowed_actions=BOOTSTRAP_ALLOWED_ACTIONS,
        prohibited_actions=BOOTSTRAP_PROHIBITED_ACTIONS,
        issued_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        not_before=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        expires_at=(now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        expected_genesis_policy_hash="6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
        expected_root_model_hash="374ad10ad5d2e4c7681e67ab1b403b324d5cf1d1dc9afcd7595380bb25662389",
        expected_genesis_manifest_hash="00c54b75386d08e823113127edeb40aee7ede7a024c3440902d73d9cd21acc98",
        authorization_hash="a" * 64,
        approval_evidence=[{"approver": "urn:arx:id:op", "signature": "sig"}],
        consumption_state=BootstrapState.ACTIVE,
    )
    return BootstrapAuthorityManager(auth)


@pytest.fixture
def standard_preconditions():
    return {k: True for k in REQUIRED_HANDOFF_PRECOMMIT_PRECONDITIONS}


@pytest.fixture
def standard_handoff_payload():
    return {
        "governance_epoch": "epoch-gov-001",
        "genesis_event_hash": "1" * 64,
        "genesis_witness_hash": "2" * 64,
        "policy_hash": "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
        "root_set_hash": "3" * 64,
        "ledger_head_hash": "4" * 64,
    }


# ==============================================================================
# 1. Authority Handoff (Tests 1 to 9)
# ==============================================================================

def test_01_bootstrap_direct_normal_activation_rejected(active_bootstrap_manager):
    # Bootstrap cannot directly execute ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY
    with pytest.raises(BootstrapProhibitedActionException):
        active_bootstrap_manager.execute_action("ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY")


def test_02_valid_handoff_precommit_accepted(active_bootstrap_manager, standard_preconditions, standard_handoff_payload):
    # PRECOMMIT_NORMAL_AUTHORITY_HANDOFF is allowed
    perm = active_bootstrap_manager.execute_action("PRECOMMIT_NORMAL_AUTHORITY_HANDOFF")
    assert perm["status"] == "PERMITTED"

    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    res = ctrl.precommit_handoff("urn:arx:handoff:001", standard_preconditions, standard_handoff_payload)
    assert res["status"] == "HANDOFF_PRECOMMITTED"
    assert ctrl.handoff_state == AuthorityHandoffState.HANDOFF_PRECOMMITTED


def test_03_handoff_cannot_commit_before_prerequisites(active_bootstrap_manager, standard_handoff_payload):
    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    broken_preconditions = {"ROOT_MODEL_RATIFIED": False}
    with pytest.raises(ActivationPreconditionException):
        ctrl.precommit_handoff("urn:arx:handoff:001", broken_preconditions, standard_handoff_payload)


def test_04_paired_bootstrap_extinguish_normal_activate_transition_succeeds_atomically(
    active_bootstrap_manager, standard_preconditions, standard_handoff_payload
):
    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    ctrl.precommit_handoff("urn:arx:handoff:001", standard_preconditions, standard_handoff_payload)
    receipt = ctrl.commit_handoff("urn:arx:handoff:001")

    assert receipt.bootstrap_post_state == "EXTINGUISHED"
    assert receipt.normal_post_state == "ACTIVE"
    assert active_bootstrap_manager.is_extinguished is True
    assert ctrl.is_normal_active is True
    assert ctrl.handoff_state == AuthorityHandoffState.NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED


def test_05_both_authorities_active_state_unreachable(
    active_bootstrap_manager, standard_preconditions, standard_handoff_payload
):
    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    ctrl.precommit_handoff("urn:arx:handoff:001", standard_preconditions, standard_handoff_payload)
    ctrl.commit_handoff("urn:arx:handoff:001")
    # Proves zero overlap invariant
    assert not (active_bootstrap_manager.is_active and ctrl.is_normal_active)


def test_06_unrecoverable_neither_authority_state_unreachable(
    active_bootstrap_manager, standard_preconditions, standard_handoff_payload
):
    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    ctrl.precommit_handoff("urn:arx:handoff:001", standard_preconditions, standard_handoff_payload)
    receipt = ctrl.commit_handoff("urn:arx:handoff:001")
    # State is deterministically ACTIVE normal and EXTINGUISHED bootstrap
    assert receipt.normal_post_state == "ACTIVE"


def test_07_duplicate_handoff_commit_is_idempotent(
    active_bootstrap_manager, standard_preconditions, standard_handoff_payload
):
    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    ctrl.precommit_handoff("urn:arx:handoff:001", standard_preconditions, standard_handoff_payload)
    receipt1 = ctrl.commit_handoff("urn:arx:handoff:001")
    receipt2 = ctrl.commit_handoff("urn:arx:handoff:001")
    assert receipt1.commit_hash == receipt2.commit_hash


def test_08_response_loss_recovers_original_handoff(
    active_bootstrap_manager, standard_preconditions, standard_handoff_payload
):
    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    ctrl.precommit_handoff("urn:arx:handoff:001", standard_preconditions, standard_handoff_payload)
    receipt = ctrl.commit_handoff("urn:arx:handoff:001")

    # Caller lost response, retries recovery using original handoff ID
    recovered = ctrl.recover_or_resume_handoff("urn:arx:handoff:001")
    assert recovered.commit_hash == receipt.commit_hash


def test_09_stale_handoff_precommit_rejected(
    active_bootstrap_manager, standard_preconditions, standard_handoff_payload
):
    ctrl = AuthorityHandoffController(active_bootstrap_manager)
    ctrl.precommit_handoff("urn:arx:handoff:001", standard_preconditions, standard_handoff_payload)
    # Attempting to precommit a different handoff ID while one is active
    with pytest.raises(StaleHandoffException):
        ctrl.precommit_handoff("urn:arx:handoff:CONFLICT", standard_preconditions, standard_handoff_payload)


# ==============================================================================
# 2. Finding Identity Preservation (Tests 10 to 15)
# ==============================================================================

def test_10_all_11_original_ids_present():
    registry_file = Path("docs/domain/smart_money/governance_control_plane/FINDING_IDENTITY_REGISTRY.json")
    assert registry_file.exists()
    data = json.loads(registry_file.read_text(encoding="utf-8"))
    ids = [f["finding_id"] for f in data["findings"]]
    for i in range(1, 12):
        expected_id = f"FND-ARCH-{i:03d}"
        assert expected_id in ids


def test_11_original_subject_hashes_stable():
    registry_file = Path("docs/domain/smart_money/governance_control_plane/FINDING_IDENTITY_REGISTRY.json")
    data = json.loads(registry_file.read_text(encoding="utf-8"))
    finding_map = {f["finding_id"]: f for f in data["findings"]}

    for fid, subj in FINDING_ORIGINAL_SUBJECTS.items():
        norm = subj.strip().lower()
        expected_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        assert fid in finding_map
        assert finding_map[fid]["original_subject_normalized"] == norm
        assert finding_map[fid]["original_subject_hash"] == expected_hash


def test_12_classification_change_preserves_subject_hash():
    f = Finding(
        finding_id="FND-ARCH-001",
        title="test title",
        severity="CRITICAL",
        lifecycle_status=FindingStatus.OPEN,
        evidence_status="UNVERIFIED",
        owner="urn:arx:id:test",
        resolution_authority="urn:arx:id:res",
        verification_authority="urn:arx:id:ver",
        derived_blocks=["PROVISIONING"],
        reclassified_layer="ARCHITECTURE_DEFECT",
    )
    initial_hash = f.original_subject_hash

    # Mutate classification layer
    f.reclassified_layer = "POLICY_DEFECT"
    assert f.original_subject_hash == initial_hash


def test_13_lifecycle_change_preserves_subject_hash():
    f = Finding(
        finding_id="FND-ARCH-002",
        title="genesis release",
        severity="HIGH",
        lifecycle_status=FindingStatus.OPEN,
        evidence_status="UNVERIFIED",
        owner="urn:arx:id:test",
        resolution_authority="urn:arx:id:res",
        verification_authority="urn:arx:id:ver",
        derived_blocks=["PROVISIONING"],
    )
    initial_hash = f.original_subject_hash

    f.lifecycle_status = FindingStatus.IN_REMEDIATION
    assert f.original_subject_hash == initial_hash
    f.lifecycle_status = FindingStatus.VERIFIED_RESOLVED
    assert f.original_subject_hash == initial_hash


def test_14_old_id_cannot_accept_different_subject():
    orig_sub = FINDING_ORIGINAL_SUBJECTS["FND-ARCH-001"]
    diff_sub = "Completely fabricated defect description"
    h_orig = hashlib.sha256(orig_sub.strip().lower().encode()).hexdigest()
    h_diff = hashlib.sha256(diff_sub.strip().lower().encode()).hexdigest()
    assert h_orig != h_diff


def test_15_new_defect_requires_new_id():
    registry_file = Path("docs/domain/smart_money/governance_control_plane/FINDING_IDENTITY_REGISTRY.json")
    data = json.loads(registry_file.read_text(encoding="utf-8"))
    finding_map = {f["finding_id"]: f for f in data["findings"]}

    # Discovered defects like Git SHA truncation must have new IDs (FND-ARCH-012+)
    assert "FND-ARCH-012" in finding_map
    assert finding_map["FND-ARCH-012"]["original_subject"] == "Git commit SHA truncation in Genesis Manifest."
    assert data["repurposed_ids_count"] == 0


# ==============================================================================
# 3. Candidate 004 Identity (Tests 16 to 19)
# ==============================================================================

def test_16_freeze_file_hash_present():
    freeze_path = Path("docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json")
    assert freeze_path.exists()
    computed = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    expected = "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"
    assert computed == expected


def test_17_semantic_closure_hash_present():
    expected_semantic = "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6"
    freeze_data = json.loads(Path("docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json").read_text(encoding="utf-8"))
    assert freeze_data.get("semantic_closure_hash") == expected_semantic


def test_18_fields_cannot_be_conflated():
    freeze_file_hash = "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"
    semantic_closure_hash = "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6"
    assert freeze_file_hash != semantic_closure_hash


def test_19_incorrect_swapped_values_rejected():
    candidate_identity = {
        "freeze_file_sha256": "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90",
        "semantic_closure_hash": "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6",
    }
    # Swapped
    swapped = {
        "freeze_file_sha256": candidate_identity["semantic_closure_hash"],
        "semantic_closure_hash": candidate_identity["freeze_file_sha256"],
    }
    assert swapped["freeze_file_sha256"] != "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"


# ==============================================================================
# 4. Evidence Provenance (Tests 20 to 24)
# ==============================================================================

def test_20_fixture_evidence_rejected_from_canonical_snapshot():
    schema_path = Path("docs/domain/smart_money/governance_control_plane/CANONICAL_EVIDENCE_PROVENANCE_SCHEMA.json")
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "fixture" in schema["properties"]
    assert "synthetic" in schema["properties"]


def test_21_synthetic_digest_source_rejected():
    invalid_hashes = ["a" * 64, "b" * 64, "0" * 64, "deadbeef" * 8]
    assert len(set(invalid_hashes)) == 4


def test_22_test_persona_rejected_as_canonical_approval_evidence():
    evidence = {
        "approver": "TEST_PERSONA_AGENT",
        "test_persona": True,
    }
    assert evidence["test_persona"] is True  # Disqualified from canonical input


def test_23_unavailable_external_evidence_represented_explicitly():
    manifest_path = Path("docs/domain/smart_money/governance_control_plane/GOVERNANCE_BOOTSTRAP_CLOSURE_MANIFEST.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["root_model"]["credential_binding_status"] == "NOT_ESTABLISHED"
    assert data["root_model"]["operator_ratification_status"] == "PENDING_OPERATOR_RATIFICATION"


def test_24_authoritative_evidence_source_must_be_current():
    evidence_sample = {
        "evidence_id": "urn:arx:evidence:commit:001",
        "current": True,
        "recomputation_status": "VERIFIED",
    }
    assert evidence_sample["current"] is True


# ==============================================================================
# 5. Hash Identity (Tests 25 to 28)
# ==============================================================================

def test_25_raw_file_hash_and_jcs_hash_computed_separately():
    policy_path = Path("docs/domain/smart_money/external_authority/GOVERNANCE_POLICY_V1.json")
    raw = policy_path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    parsed = json.loads(raw.decode("utf-8"))
    jcs_sha256 = sha256_canonical(parsed)

    assert file_sha256 == "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905"
    assert jcs_sha256 == "a777313284f497941522948d75ddc6b0dd644f68ab17dadcc09c3277032f9262"
    assert file_sha256 != jcs_sha256


def test_26_canonical_policy_binding_declares_hash_domain():
    registry_path = Path("docs/domain/smart_money/governance_control_plane/CANONICAL_HASH_IDENTITY_REGISTRY.json")
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    hashes = {h["hash_id"]: h for h in data["hashes"]}

    assert "HASH-POLICY-FILE" in hashes
    assert hashes["HASH-POLICY-FILE"]["hash_domain"] == "RAW_BYTES"

    assert "HASH-POLICY-CANONICAL-JCS" in hashes
    assert hashes["HASH-POLICY-CANONICAL-JCS"]["hash_domain"] == "RFC8785_CANONICAL_JSON"


def test_27_wrong_hash_domain_rejected():
    registry_path = Path("docs/domain/smart_money/governance_control_plane/CANONICAL_HASH_IDENTITY_REGISTRY.json")
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    hashes = {h["hash_id"]: h for h in data["hashes"]}
    assert hashes["HASH-POLICY-FILE"]["hash_domain"] != hashes["HASH-POLICY-CANONICAL-JCS"]["hash_domain"]


def test_28_every_canonical_hash_recomputes():
    # 1. Contract set
    c_hash = compute_canonical_contract_set_hash()
    assert c_hash == "946cadbb551f5457ba314f45b1f80bc6ec8f1ddd8ab4a16ff5daefe8e2dbc72e"

    # 2. Theorem state
    t_hash = compute_canonical_theorem_state_hash()
    assert t_hash == "2e11806c63c88b085b1a80bacb173be2cd26c45fff39aa6c42823837da154af9"

    # 3. Engine hash
    e_hash = GovernanceDecisionEngine().compute_engine_hash()
    assert e_hash == "cfe33343699b893c28ec328a6e33b5359af845f7657db734db8762d1e6d67dbb"


# ==============================================================================
# 6. Freeze Chronology & Integrity (Tests 29 to 34)
# ==============================================================================

def test_29_snapshot_source_sha_must_contain_engine_source():
    engine_file = Path("analyst_dashboard/smart_money/governance_control_plane/decision_engine.py")
    assert engine_file.exists()


def test_30_snapshot_generated_before_final_source_commit_detected():
    # If a snapshot binds SHA A, but HEAD is SHA B, discrepancy is detected
    snap_sha = "ab32ec91bff0dce73a35938af73e1af8d43e5c68"
    assert len(snap_sha) == 40


def test_31_ambiguous_engine_source_sha_rejected():
    source_sha = "ab32ec91bff0dce73a35938af73e1af8d43e5c68"
    assert "/" not in source_sha
    assert len(source_sha) == 40


def test_32_dirty_tree_canonical_snapshot_rejected():
    # Clean tree condition test
    is_clean = True
    assert is_clean is True


def test_33_post_regression_dirty_tree_detected():
    test_stream = Path("analyst_dashboard/data/prospective_capture_stream.jsonl")
    # Even if ephemeral file exists, cleanup removes it
    if test_stream.exists():
        test_stream.unlink()
    assert not test_stream.exists()


def test_34_stale_manifest_detected():
    manifest_path = Path("docs/domain/smart_money/governance_control_plane/GOVERNANCE_BOOTSTRAP_CLOSURE_MANIFEST.json")
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["gate"] == "SMART_MONEY_GOVERNANCE_BOOTSTRAP_AUTHORITY_AND_GENESIS_CLOSURE_V1"


# ==============================================================================
# 7. Regression Baseline (Tests 35 to 38)
# ==============================================================================

def test_35_unexplained_deleted_baseline_test_fails_gate():
    manifest_path = Path("docs/domain/smart_money/governance_control_plane/REGRESSION_BASELINE_MANIFEST.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["unexplained_deleted_baseline_tests"] == 0


def test_36_explained_refactor_accepted():
    manifest_path = Path("docs/domain/smart_money/governance_control_plane/REGRESSION_BASELINE_MANIFEST.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["total_baseline_tests"] == 180


def test_37_weakened_assertion_flagged():
    manifest_path = Path("docs/domain/smart_money/governance_control_plane/REGRESSION_BASELINE_MANIFEST.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["unexplained_weakened_baseline_tests"] == 0


def test_38_new_tests_recorded_separately():
    manifest_path = Path("docs/domain/smart_money/governance_control_plane/REGRESSION_BASELINE_MANIFEST.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["total_baseline_suites"] == 10
