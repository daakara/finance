"""
ARX Terminal — Epoch 2 Activation Architecture & Boundary Test Suite

Tests all scenarios and invariants specified in Sections 5, 7, and 8 of the
Epoch 2 Activation Architecture Specification:
1. Activation Record Contract Schema & Validation
2. Production Activation Record Non-Existence in Repository (Pre-Activation State)
3. Boundary 1: No activation record -> PRE_ACTIVATION_RECORD_ELIGIBLE = NO
4. Boundary 2: Failed deployment status -> ELIGIBLE = NO
5. Boundary 3: Wrong release SHA -> WRONG_RELEASE_RECORD_ELIGIBLE = NO
6. Boundary 4: Wrong epoch ID -> ELIGIBLE = NO
7. Boundary 5: Malformed activation timestamp -> ELIGIBLE = NO
8. Boundary 6: Record before activation (rec_dt < act_dt) -> ELIGIBLE = NO
9. Boundary 7: Record exactly at activation (rec_dt == act_dt) -> BOUNDARY_EQUAL_RECORD_ELIGIBLE = YES
10. Boundary 8: Record after activation (rec_dt > act_dt) -> POST_ACTIVATION_RECORD_ELIGIBLE = YES
11. Boundary 9: Historical Epoch-1 record after activation -> Separated; NOT counted in Epoch 2
12. Boundary 10: Future-dated record -> ELIGIBLE = NO
13. Boundary 11: Missing runtime identity -> ELIGIBLE = NO
14. Passive Capture Pre-Activation Suppression
15. Retroactive Reclassification Impossibility
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

from analyst_dashboard.governance.experiment_ledger import (
    ExperimentLedger,
    ProvenanceCohort,
)
from analyst_dashboard.governance.passive_capture import PassiveCaptureHook


VALID_RELEASE_SHA = "cd655c49b593b1acaf952c85623a133400a1e4a3"
DECISION_ENGINE_SHA = ExperimentLedger.DECISION_ENGINE_SHA
ACTIVATION_TIME_STR = "2026-09-23T10:00:00.000000Z"


def _make_temp_activation_record(
    tmp_path: str,
    epoch_id: str = "ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
    release_sha: str = VALID_RELEASE_SHA,
    deployment_id: str = "render-deploy-uuid-12345",
    deployment_status: str = "SUCCESS",
    activated_at_utc: str = ACTIVATION_TIME_STR,
    prospective_observation_authorized: bool = True,
    runtime_identity_attestation: str = "CONTAINER_ENTRYPOINT_MANIFEST_VERIFIED",
) -> str:
    record = {
        "epochId": epoch_id,
        "releaseSha": release_sha,
        "deploymentId": deployment_id,
        "deploymentStatus": deployment_status,
        "activatedAtUtc": activated_at_utc,
        "runtimeIdentityAttestation": runtime_identity_attestation,
        "prospectiveObservationAuthorized": prospective_observation_authorized,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return tmp_path


def _build_test_signal(
    symbol: str = "AAPL",
    rec_time: str = ACTIVATION_TIME_STR,
    epoch_id: str = "ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
    release_sha: str = VALID_RELEASE_SHA,
    cohort: str = ProvenanceCohort.PROSPECTIVE_CLEAN,
) -> dict:
    return {
        "symbol": symbol,
        "signalDate": rec_time[:10],
        "recommended_at": rec_time,
        "signalTimestamp": rec_time,
        "timestamp": rec_time,
        "status": "VALID_SETUP",
        "decisionState": "ACTIONABLE_SETUP",
        "provenanceCohort": cohort,
        "epochId": epoch_id,
        "releaseSha": release_sha,
        "engineVersion": DECISION_ENGINE_SHA,
        "engineCommit": release_sha,
        "entryPrice": 220.0,
        "inputs": {
            "marketSnapshotObservedAt": rec_time,
            "macroObservationAvailableAt": rec_time,
        },
    }


# ---------------------------------------------------------------------------
# Section 5 & Pre-Activation Verification
# ---------------------------------------------------------------------------

def test_01_production_activation_record_does_not_exist_in_repo():
    """Verify that no real production activation record exists in the repository prior to deployment."""
    default_path = ExperimentLedger.DEFAULT_ACTIVATION_RECORD_PATH
    assert not os.path.exists(default_path), f"Production activation record must NOT exist pre-deployment: {default_path}"
    assert ExperimentLedger.get_activation_record() is None
    assert ExperimentLedger.is_epoch2_observation_authorized() is False
    assert ExperimentLedger.get_epoch2_clean_prospective_count() == 0


def test_02_activation_record_contract_schema_and_validation():
    """Verify schema, fields, and validation for valid and invalid activation records."""
    # Valid record
    rec = {
        "epochId": "ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
        "releaseSha": VALID_RELEASE_SHA,
        "deploymentId": "deploy-uuid-001",
        "deploymentStatus": "SUCCESS",
        "activatedAtUtc": "2026-09-23T10:00:00Z",
        "runtimeIdentityAttestation": "CONTAINER_ENTRYPOINT_MANIFEST_VERIFIED",
        "prospectiveObservationAuthorized": True,
    }
    valid, reason = ExperimentLedger.validate_activation_record(rec)
    assert valid is True
    assert reason is None

    # Invalid epochId
    rec_bad_epoch = dict(rec, epochId="WRONG_EPOCH")
    valid, reason = ExperimentLedger.validate_activation_record(rec_bad_epoch)
    assert valid is False
    assert "EPOCH_ID_MISMATCH" in reason

    # Invalid deploymentStatus
    rec_failed_deploy = dict(rec, deploymentStatus="FAILED")
    valid, reason = ExperimentLedger.validate_activation_record(rec_failed_deploy)
    assert valid is False
    assert "DEPLOYMENT_STATUS_NOT_SUCCESS" in reason

    # Not authorized
    rec_unauth = dict(rec, prospectiveObservationAuthorized=False)
    valid, reason = ExperimentLedger.validate_activation_record(rec_unauth)
    assert valid is False
    assert "PROSPECTIVE_OBSERVATION_NOT_AUTHORIZED" in reason

    # Missing fields
    rec_no_sha = dict(rec, releaseSha="")
    valid, reason = ExperimentLedger.validate_activation_record(rec_no_sha)
    assert valid is False
    assert "MISSING_RELEASE_SHA" in reason


# ---------------------------------------------------------------------------
# Section 8: Boundary Tests
# ---------------------------------------------------------------------------

def test_03_boundary_no_activation_record():
    """Boundary 1: No activation record exists -> PRE_ACTIVATION_RECORD_ELIGIBLE = NO."""
    sig = _build_test_signal(rec_time="2026-09-23T10:05:00Z")
    eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path="/nonexistent/act.json")
    assert eligible is False
    assert reason == "NO_VALID_ACTIVATION_RECORD"


def test_04_boundary_failed_deployment_status():
    """Boundary 2: Activation record indicates deployment failure -> ELIGIBLE = NO."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path, deployment_status="FAILED")
        sig = _build_test_signal(rec_time="2026-09-23T10:05:00Z")
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is False
        assert reason == "NO_VALID_ACTIVATION_RECORD"
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_05_boundary_wrong_release_sha():
    """Boundary 3: Signal recorded by different release SHA than activated -> WRONG_RELEASE_RECORD_ELIGIBLE = NO."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path, release_sha=VALID_RELEASE_SHA)
        sig = _build_test_signal(rec_time="2026-09-23T10:05:00Z", release_sha="different_unauthorized_sha_1234567")
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is False
        assert "RELEASE_ATTRIBUTION_MISMATCH" in reason
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_06_boundary_wrong_epoch_id():
    """Boundary 4: Signal tagged with different epoch ID -> ELIGIBLE = NO."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path)
        sig = _build_test_signal(rec_time="2026-09-23T10:05:00Z", epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_1")
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is False
        assert reason == "EPOCH_ID_MISMATCH"
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_07_boundary_malformed_activation_timestamp():
    """Boundary 5: Malformed activation timestamp -> ELIGIBLE = NO."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path, activated_at_utc="not_a_valid_date")
        sig = _build_test_signal(rec_time="2026-09-23T10:05:00Z")
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is False
        assert reason == "NO_VALID_ACTIVATION_RECORD"
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_08_boundary_record_before_activation():
    """Boundary 6: Record strictly precedes activation timestamp -> ELIGIBLE = NO."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path, activated_at_utc="2026-09-23T10:00:00Z")
        # Record 1 second before activation
        sig = _build_test_signal(rec_time="2026-09-23T09:59:59Z")
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is False
        assert "RECORD_PRECEDES_ACTIVATION" in reason
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_09_boundary_record_exactly_at_activation():
    """Boundary 7: Record timestamp exactly equals activation timestamp -> BOUNDARY_EQUAL_RECORD_ELIGIBLE = YES."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path, activated_at_utc="2026-09-23T10:00:00Z")
        # Record exactly at activation boundary
        sig = _build_test_signal(rec_time="2026-09-23T10:00:00Z")
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is True
        assert reason is None
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_10_boundary_record_post_activation():
    """Boundary 8: Record timestamp after activation -> POST_ACTIVATION_RECORD_ELIGIBLE = YES."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path, activated_at_utc="2026-09-23T10:00:00.000000Z")
        # Record 1 microsecond after activation
        sig = _build_test_signal(rec_time="2026-09-23T10:00:00.000001Z")
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is True
        assert reason is None
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_11_boundary_historical_epoch1_record_after_activation():
    """Boundary 9: Historical Epoch-1 record created after activation -> Separated; NOT counted in Epoch 2."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f_act, \
         tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f_led:
        act_path = f_act.name
        led_path = f_led.name
    try:
        _make_temp_activation_record(act_path, activated_at_utc="2026-09-23T10:00:00Z")
        # Epoch 1 record in ledger
        ep1_sig = _build_test_signal(
            rec_time="2026-09-23T10:05:00Z",
            epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_1"
        )
        ExperimentLedger.save_ledger({"version": "1.1.0", "signals": [ep1_sig]}, led_path)

        # Counted in Epoch 1 (if clean), but strictly 0 in Epoch 2
        assert ExperimentLedger.get_epoch2_clean_prospective_count(ledger_path=led_path, activation_record_path=act_path) == 0
    finally:
        if os.path.exists(act_path): os.remove(act_path)
        if os.path.exists(led_path): os.remove(led_path)


def test_12_boundary_future_dated_record():
    """Boundary 10: Record with timestamp far in the future -> ELIGIBLE = NO."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path, activated_at_utc="2026-09-23T10:00:00Z")
        far_future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        sig = _build_test_signal(rec_time=far_future)
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is False
        assert reason == "FUTURE_DATED_RECORD"
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_13_boundary_missing_runtime_identity():
    """Boundary 11: Record missing runtime identity / release SHA -> ELIGIBLE = NO."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        _make_temp_activation_record(act_path)
        sig = _build_test_signal(rec_time="2026-09-23T10:05:00Z", release_sha="")
        sig.pop("engineCommit", None)
        sig.pop("releaseSha", None)
        eligible, reason = ExperimentLedger.is_record_epoch2_eligible(sig, activation_record_path=act_path)
        assert eligible is False
        assert reason == "MISSING_RUNTIME_IDENTITY"
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_14_passive_capture_pre_activation_suppression():
    """Verify that PassiveCaptureHook suppresses natural observation capture when no activation record exists."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        act_path = f.name
    try:
        # Before activation: is_temporal_gate_satisfied is False
        assert PassiveCaptureHook.is_temporal_gate_satisfied(activation_record_path=act_path) is False

        # After creating valid activation record: is_temporal_gate_satisfied is True
        _make_temp_activation_record(act_path)
        assert PassiveCaptureHook.is_temporal_gate_satisfied(activation_record_path=act_path) is True
    finally:
        if os.path.exists(act_path):
            os.remove(act_path)


def test_15_retroactive_reclassification_impossible():
    """Verify that retroactive reclassification is impossible: Epoch 1 records cannot be counted by Epoch 2."""
    assert ExperimentLedger.get_epoch1_clean_prospective_count() == 0
    assert ExperimentLedger.get_epoch2_clean_prospective_count() == 0


# ---------------------------------------------------------------------------
# Phase 3, 4, 5 Verification: Activation Record Validation & Deployment Boundary
# ---------------------------------------------------------------------------

def test_16_activation_record_future_timestamp_rejected():
    """Phase 4: Explicit test of the activation record itself rejecting future timestamps."""
    far_future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    rec = {
        "epochId": "ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
        "releaseSha": VALID_RELEASE_SHA,
        "deploymentId": "deploy-uuid-001",
        "deploymentStatus": "SUCCESS",
        "activatedAtUtc": far_future,
        "runtimeIdentityAttestation": "CONTAINER_ENTRYPOINT_MANIFEST_VERIFIED",
        "prospectiveObservationAuthorized": True,
    }
    valid, reason = ExperimentLedger.validate_activation_record(rec)
    assert valid is False
    assert reason == "FUTURE_ACTIVATION_TIMESTAMP"


def test_17_activation_vs_deployment_boundary_scenarios():
    """Phase 3: Verify the 6 deployment boundary vs activation timestamp scenarios."""
    dep_boundary = "2026-09-23T10:00:00Z"
    base_rec = {
        "epochId": "ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
        "releaseSha": VALID_RELEASE_SHA,
        "deploymentId": "deploy-uuid-001",
        "deploymentStatus": "SUCCESS",
        "activatedAtUtc": "2026-09-23T10:00:00Z",
        "deploymentFinishedAtUtc": dep_boundary,
        "runtimeIdentityAttestation": "CONTAINER_ENTRYPOINT_MANIFEST_VERIFIED",
        "prospectiveObservationAuthorized": True,
    }

    # Scenario 1: activation timestamp before successful deployment/runtime activation -> REJECT
    rec_pre_deploy = dict(base_rec, activatedAtUtc="2026-09-23T09:59:59Z")
    valid, reason = ExperimentLedger.validate_activation_record(rec_pre_deploy)
    assert valid is False
    assert "ACTIVATION_PREDATES_DEPLOYMENT_BOUNDARY" in reason

    # Scenario 2: activation timestamp equal to successful activation boundary -> ACCEPT
    rec_equal = dict(base_rec, activatedAtUtc=dep_boundary)
    valid, reason = ExperimentLedger.validate_activation_record(rec_equal)
    assert valid is True
    assert reason is None

    # Scenario 3: activation timestamp after successful activation boundary -> ACCEPT
    rec_post_deploy = dict(base_rec, activatedAtUtc="2026-09-23T10:00:01Z")
    valid, reason = ExperimentLedger.validate_activation_record(rec_post_deploy)
    assert valid is True
    assert reason is None

    # Scenario 4: deployment status != SUCCESS -> REJECT
    rec_failed = dict(base_rec, deploymentStatus="FAILED")
    valid, reason = ExperimentLedger.validate_activation_record(rec_failed)
    assert valid is False
    assert "DEPLOYMENT_STATUS_NOT_SUCCESS" in reason

    # Scenario 5: deployment identity absent -> REJECT
    rec_no_deploy_id = dict(base_rec, deploymentId="")
    valid, reason = ExperimentLedger.validate_activation_record(rec_no_deploy_id)
    assert valid is False
    assert "MISSING_DEPLOYMENT_ID" in reason

    # Scenario 6: deployment identity mismatches release identity -> REJECT
    valid, reason = ExperimentLedger.validate_activation_record(
        base_rec, expected_release_sha="different_release_sha_12345"
    )
    assert valid is False
    assert "RELEASE_SHA_MISMATCH" in reason


def test_18_activation_record_cannot_overwrite_prior_immutable_record():
    """Phase 5: Activation record creation cannot overwrite an existing record silently."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name
    try:
        # Create initial record
        ExperimentLedger.create_activation_record(
            epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
            release_sha=VALID_RELEASE_SHA,
            deployment_id="deploy-1",
            activated_at_utc="2026-09-23T10:00:00Z",
            output_path=temp_path,
            overwrite=True,
        )
        assert os.path.exists(temp_path)

        # Attempt to overwrite without overwrite=True must raise FileExistsError
        with pytest.raises(FileExistsError) as exc_info:
            ExperimentLedger.create_activation_record(
                epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
                release_sha=VALID_RELEASE_SHA,
                deployment_id="deploy-2",
                activated_at_utc="2026-09-23T10:05:00Z",
                output_path=temp_path,
                overwrite=False,
            )
        assert "already exists and is immutable" in str(exc_info.value)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
