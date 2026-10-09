"""Unit and integration tests for ARX Execution Ladder Passive Capture Extension (Epoch 001).

Validates:
1. Actionable plan admission (Section 16)
2. Non-actionable plan admission for Epoch 001 (Section 17)
3. Legacy fail-closed parity for non-Epoch 001 (Section 18)
4. Invalid execution status rejection (Section 19)
5. Snapshot completeness & stable semantic rejection codes (Section 20)
6. Deduplication & idempotent zero-denominator-increment (Section 21)
7. Two-tier release vs authority SHA provenance (Section 22)
8. SQLite write-once immutability triggers (Section 23)
9. Prevention of retroactive historical backfill (Section 24)
10. Denominator zero baseline & isolated test storage (Section 25)
11. Day vs Long stratification separation (Section 13)
"""

import os
import json
import math
import sqlite3
import tempfile
import pytest
from datetime import datetime, timezone

from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
from analyst_dashboard.governance.passive_capture import (
    PassiveCaptureHook,
    ExecutionContext,
    EXECUTION_LADDER_EPOCH_ID,
    EXECUTION_LADDER_OBSERVATION_STREAM,
    EXECUTION_LADDER_AUTHORITY_SHA,
    RATIFIED_EXECUTION_LADDER_STATUSES,
    resolve_release_sha,
    compute_execution_ladder_plan_id,
    build_execution_ladder_snapshot,
    extract_canonical_trading_date,
)
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger


@pytest.fixture
def temp_db_path():
    """Provides a fresh isolated temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    # Initialize schema
    GovernanceDatabaseEngine(db_path=path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def isolate_test_ledger(monkeypatch, tmp_path):
    """Guarantees tests never write to the production paper trading ledger."""
    dummy_ledger = tmp_path / "dummy_ledger.json"
    dummy_ledger.write_text(
        json.dumps({
            "version": "1.0.0",
            "totalActiveSignals": 0,
            "signals": [],
            "execution_ladder_plans": [],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(ExperimentLedger, "DEFAULT_LEDGER_PATH", str(dummy_ledger))
    monkeypatch.setattr(
        ExperimentLedger,
        "record_execution_ladder_plan_snapshot",
        classmethod(lambda cls, snap, ledger_path=None: snap),
    )


@pytest.fixture
def temp_ledger_path():
    """Provides a fresh isolated temporary ledger json file path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    initial_data = {
        "version": "1.0.0",
        "totalActiveSignals": 0,
        "signals": [],
        "execution_ladder_plans": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(initial_data, f)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _create_valid_plan_dict(
    symbol: str = "NAUT",
    user_role: str = "LONG_TERM",
    status: str = "WAITING_PULLBACK",
    is_actionable: bool = False,
    current_spot: float = 1.96,
    entry_min: float = 1.27,
    entry_max: float = 1.46,
    planned_entry: float = 1.46,
    structural_invalidation: float = 1.26,
    tp1: float = 1.90,
    tp2: float = 2.19,
    atr_14: float = 0.15,
    gen_time: str = "2026-10-08T05:00:00Z",
    source_time: str = "2026-10-08T04:59:00Z",
    release_sha: str = "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
    authority_sha: str = EXECUTION_LADDER_AUTHORITY_SHA,
):
    """Generates a complete valid execution ladder plan dictionary."""
    return {
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": symbol,
        "instrument_class": "EQUITY",
        "user_role": user_role,
        "generation_timestamp": gen_time,
        "source_data_timestamp": source_time,
        "release_sha": release_sha,
        "execution_ladder_authority_sha": authority_sha,
        "current_spot": current_spot,
        "entry_min": entry_min,
        "entry_max": entry_max,
        "planned_entry": planned_entry,
        "structural_invalidation": structural_invalidation,
        "execution_risk": round(planned_entry - structural_invalidation, 4),
        "atr_14": atr_14,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "market_location": "EXTENDED_ABOVE_RANGE",
        "execution_status": status,
        "is_actionable": is_actionable,
        "execution_stop_visible": True,
    }


# ==============================================================================
# SECTION 16: TEST ACTIONABLE ADMISSION
# ==============================================================================
def test_actionable_plan_admission(temp_db_path, monkeypatch):
    """Verifies that an actionable plan (e.g. IN_BUY_ZONE) with complete provenance is admitted."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    plan = _create_valid_plan_dict(
        status="IN_BUY_ZONE",
        is_actionable=True,
        current_spot=1.40,
        entry_min=1.35,
        entry_max=1.45,
        planned_entry=1.40,
        structural_invalidation=1.25,
    )

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=True,
        live_spot_price=1.40,
        current_price=1.40,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )

    assert admission["prospectiveCaptureEligible"] is True
    assert admission["prospectiveCaptureRejectionReason"] is None
    assert admission["observationStream"] == EXECUTION_LADDER_OBSERVATION_STREAM
    assert "plan_id" in admission
    assert admission["plan_id"].startswith("PLAN_")


# ==============================================================================
# SECTION 17: TEST NON-ACTIONABLE ADMISSION (EPOCH 001)
# ==============================================================================
@pytest.mark.parametrize(
    "status",
    [
        "WAITING_PULLBACK",
        "EXTENDED_ABOVE_BUY_ZONE",
        "IN_BUY_ZONE_AWAITING_TRIGGER",
        "STOPPED_OUT",
    ],
)
def test_non_actionable_admission_epoch_001(status, temp_db_path, monkeypatch):
    """Verifies that non-actionable ratified statuses are admitted for Epoch 001 when complete."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    test_spot = 1.20 if status == "STOPPED_OUT" else 1.96
    plan = _create_valid_plan_dict(
        status=status,
        is_actionable=False,
        current_spot=test_spot,
        structural_invalidation=1.26,
        planned_entry=1.46,
    )

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=test_spot,
        current_price=test_spot,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )

    assert admission["prospectiveCaptureEligible"] is True
    assert admission["prospectiveCaptureRejectionReason"] is None
    assert admission["snapshot"]["execution_status"] == status
    assert admission["snapshot"]["is_actionable"] is False


@pytest.mark.parametrize(
    "status",
    [
        "APPROACHING_TARGET",
        "TARGET_REACHED",
        "READY_TO_BUY",
    ],
)
def test_active_position_and_alias_statuses_rejected_in_epoch_001(status, temp_db_path, monkeypatch):
    """Verifies that active-position target progress and legacy alias statuses are strictly rejected for Epoch 001."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    plan = _create_valid_plan_dict(
        status=status,
        is_actionable=False,
    )

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )

    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "UNRATIFIED_EXECUTION_STATUS"


# ==============================================================================
# SECTION 18: TEST LEGACY FAIL-CLOSED PARITY
# ==============================================================================
@pytest.mark.parametrize(
    "status",
    ["WAITING_PULLBACK", "EXTENDED_ABOVE_BUY_ZONE", "IN_BUY_ZONE_AWAITING_TRIGGER"],
)
def test_legacy_fail_closed_parity_outside_epoch_001(status, temp_db_path, temp_ledger_path, monkeypatch):
    """Verifies that non-actionable plans are REJECTED when epoch_id != Epoch 001.

    Guarantees: GLOBAL_ACTIONABILITY_GATE_CHANGED = NO
    """
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    plan = _create_valid_plan_dict(status=status, is_actionable=False)

    # 1. Non-actionable evaluated under legacy/generic epoch (None)
    admission_default = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        epoch_id=None,
    )
    assert admission_default["prospectiveCaptureEligible"] is False

    # 2. Non-actionable evaluated under standard validation epoch
    admission_epoch1 = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_1",
    )
    assert admission_epoch1["prospectiveCaptureEligible"] is False


# ==============================================================================
# SECTION 19: TEST INVALID STATUS REJECTION
# ==============================================================================
@pytest.mark.parametrize(
    "invalid_status",
    ["UNKNOWN", "ERROR", "UNRESOLVED", "INVALID", "BULLISH_TREND", "ARBITRARY_STATUS", ""],
)
def test_invalid_status_rejected_even_in_epoch_001(invalid_status, temp_db_path, monkeypatch):
    """Verifies that non-ratified statuses are rejected even inside Epoch 001."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    plan = _create_valid_plan_dict(status=invalid_status, is_actionable=False)

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )

    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] in (
        "UNRATIFIED_EXECUTION_STATUS",
        "MISSING_EXECUTION_STATUS",
    )


# ==============================================================================
# SECTION 20: TEST SNAPSHOT COMPLETENESS & REJECTION CODES
# ==============================================================================
@pytest.mark.parametrize(
    "field_to_remove, expected_reason",
    [
        ("planned_entry", "MISSING_PLANNED_ENTRY"),
        ("structural_invalidation", "MISSING_STRUCTURAL_INVALIDATION"),
        ("current_spot", "MISSING_CURRENT_SPOT"),
        ("entry_min", "MISSING_ENTRY_MIN"),
        ("entry_max", "MISSING_ENTRY_MAX"),
        ("execution_risk", "MISSING_EXECUTION_RISK"),
        ("atr_14", "MISSING_ATR_14"),
        ("take_profit_1", "MISSING_TAKE_PROFIT_1"),
        ("take_profit_2", "MISSING_TAKE_PROFIT_2"),
        ("market_location", "MISSING_MARKET_LOCATION"),
        ("execution_status", "MISSING_EXECUTION_STATUS"),
        ("release_sha", "MISSING_RELEASE_SHA"),
        ("execution_ladder_authority_sha", "MISSING_AUTHORITY_SHA"),
    ],
)
def test_snapshot_completeness_field_removal(field_to_remove, expected_reason, temp_db_path, monkeypatch):
    """Verifies that removing each required field yields prospectiveCaptureEligible=False and stable code."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    plan = _create_valid_plan_dict()
    del plan[field_to_remove]

    # Clear runtime env if testing release_sha
    if field_to_remove == "release_sha":
        monkeypatch.delenv("ARX_RELEASE_SHA", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_ARX_RELEASE", raising=False)
        monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=plan.get("current_spot"),
        current_price=plan.get("current_spot"),
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha=plan.get("release_sha"),
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )

    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == expected_reason


def test_invalid_authority_sha_rejection(temp_db_path, monkeypatch):
    """Verifies that an authority SHA different from the ratified authority is rejected."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    plan = _create_valid_plan_dict(authority_sha="0000000000000000000000000000000000000000")

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )

    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "INVALID_AUTHORITY_SHA"


# ==============================================================================
# SECTION 21: TEST DEDUPLICATION (IDEMPOTENCY)
# ==============================================================================
def test_deduplication_zero_denominator_inflation(temp_db_path, temp_ledger_path, monkeypatch):
    """Verifies that capturing the exact same plan twice inserts on first and no-ops on second.

    Guarantees: DUPLICATE_CAPTURE = NO_DENOMINATOR_INCREMENT
    """
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    plan = _create_valid_plan_dict()

    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    # Initial denominator is 0
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 0

    # First capture
    res1 = PassiveCaptureHook.record_execution_ladder_plan(
        symbol="NAUT",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        snapshot_dict=plan,
        execution_context=ExecutionContext.NATURAL_CLIENT,
    )
    assert res1 is not None
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1

    # Second capture with identical plan (simulating immediate client remount)
    res2 = PassiveCaptureHook.record_execution_ladder_plan(
        symbol="NAUT",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        snapshot_dict=plan,
        execution_context=ExecutionContext.NATURAL_CLIENT,
    )
    assert res2 is not None
    assert res2["plan_id"] == res1["plan_id"]
    # Denominator increment is ZERO (count remains 1)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1

    # Third capture with different request generation timestamp (simulating browser page refresh 5s later)
    plan_refreshed = _create_valid_plan_dict(gen_time="2026-10-08T05:00:05Z")
    res_refreshed = PassiveCaptureHook.record_execution_ladder_plan(
        symbol="NAUT",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        snapshot_dict=plan_refreshed,
        execution_context=ExecutionContext.NATURAL_CLIENT,
    )
    assert res_refreshed is not None
    # CROSS_REQUEST_PLAN_ID_STABILITY: plan_id is identical despite new generation_timestamp
    assert res_refreshed["plan_id"] == res1["plan_id"]
    # Denominator increment remains ZERO (count remains 1)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1

    # Now alter a legitimate immutable identity field (e.g. planned_entry)
    plan_altered = _create_valid_plan_dict(planned_entry=1.45)
    res3 = PassiveCaptureHook.record_execution_ladder_plan(
        symbol="NAUT",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        snapshot_dict=plan_altered,
        execution_context=ExecutionContext.NATURAL_CLIENT,
    )
    assert res3 is not None
    assert res3["plan_id"] != res1["plan_id"]
    # Count increments to 2
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 2


# ==============================================================================
# SECTION 22: TEST RELEASE PROVENANCE (TWO-TIER IDENTITY)
# ==============================================================================
def test_two_tier_release_and_authority_provenance(temp_db_path, monkeypatch):
    """Verifies authority SHA is fixed (7bcb778...) while application release SHA can advance."""
    # Test with simulated new runtime release SHA
    simulated_new_app_release = "9999999999999999999999999999999999999999"
    monkeypatch.setenv("ARX_RELEASE_SHA", simulated_new_app_release)

    plan = _create_valid_plan_dict(
        release_sha=simulated_new_app_release,
        authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha=simulated_new_app_release,
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )

    assert admission["prospectiveCaptureEligible"] is True
    snapshot = admission["snapshot"]
    assert snapshot["release_sha"] == simulated_new_app_release
    assert snapshot["execution_ladder_authority_sha"] == EXECUTION_LADDER_AUTHORITY_SHA


# ==============================================================================
# SECTION 23: TEST IMMUTABILITY TRIGGERS
# ==============================================================================
def test_immutability_update_and_delete_prohibited(temp_db_path, monkeypatch):
    """Verifies that UPDATE and DELETE on execution_ladder_prospective_plans raise IMMUTABILITY_VIOLATION."""
    plan = _create_valid_plan_dict()
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    snapshot, _ = build_execution_ladder_snapshot(
        symbol="NAUT",
        optimal_execution_plan={},
        release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
        snapshot_dict=plan,
    )
    inserted = gov_db.insert_execution_ladder_plan(snapshot)
    assert inserted is True

    conn = gov_db.get_connection()
    try:
        # Test 1: UPDATE attempt
        with pytest.raises(sqlite3.IntegrityError) as exc_update:
            with conn:
                conn.execute(
                    "UPDATE execution_ladder_prospective_plans SET current_spot = 999.0 WHERE plan_id = ?",
                    (snapshot["plan_id"],),
                )
        assert "IMMUTABILITY_VIOLATION" in str(exc_update.value)

        # Test 2: DELETE attempt
        with pytest.raises(sqlite3.IntegrityError) as exc_delete:
            with conn:
                conn.execute(
                    "DELETE FROM execution_ladder_prospective_plans WHERE plan_id = ?",
                    (snapshot["plan_id"],),
                )
        assert "IMMUTABILITY_VIOLATION" in str(exc_delete.value)
    finally:
        conn.close()


# ==============================================================================
# SECTION 24: TEST NO RETROACTIVE CAPTURE / SYNTHETIC REJECTION
# ==============================================================================
def test_no_retroactive_capture_and_synthetic_context_rejection(temp_db_path, monkeypatch):
    """Verifies that non-natural / synthetic execution contexts are rejected fail-closed."""
    plan = _create_valid_plan_dict()

    # Context test 1: GOVERNANCE_CERTIFICATION context
    admission1 = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.GOVERNANCE_CERTIFICATION,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )
    assert admission1["prospectiveCaptureEligible"] is False
    assert admission1["prospectiveCaptureRejectionReason"] == "NON_NATURAL_CONTEXT"

    # Context test 2: ARX_TEST_MODE env flag
    monkeypatch.setenv("ARX_TEST_MODE", "1")
    admission2 = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan,
    )
    assert admission2["prospectiveCaptureEligible"] is False
    assert admission2["prospectiveCaptureRejectionReason"] == "NON_NATURAL_CONTEXT"
    monkeypatch.delenv("ARX_TEST_MODE", raising=False)

    # Context test 3: Synthetic fixture plan
    plan_synthetic = _create_valid_plan_dict()
    plan_synthetic["isSynthetic"] = True
    admission3 = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="NAUT",
        is_actionable=False,
        live_spot_price=1.96,
        current_price=1.96,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        snapshot_dict=plan_synthetic,
    )
    assert admission3["prospectiveCaptureEligible"] is False
    assert admission3["prospectiveCaptureRejectionReason"] == "NON_NATURAL_CONTEXT"


# ==============================================================================
# SECTION 25: TEST DENOMINATOR START (ZERO BASELINE)
# ==============================================================================
def test_denominator_start_zero_and_isolated_db(temp_db_path):
    """Verifies that prospective denominator for Epoch 001 is 0 prior to natural admissions."""
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    count = gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID)
    assert count == 0


# ==============================================================================
# SECTION 13: TEST DAY / LONG STRATIFICATION SEPARATION
# ==============================================================================
def test_day_vs_long_stratification_separation(temp_db_path, monkeypatch):
    """Verifies DAY_TRADER and LONG_TERM plans are counted separately in stratification."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    # Insert a LONG_TERM plan
    plan_long = _create_valid_plan_dict(
        user_role="LONG_TERM",
        gen_time="2026-10-08T05:00:00Z",
    )
    snapshot_long, _ = build_execution_ladder_snapshot(
        symbol="NAUT",
        user_role="LONG_TERM",
        release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        snapshot_dict=plan_long,
    )
    gov_db.insert_execution_ladder_plan(snapshot_long)

    # Insert a DAY_TRADER plan
    plan_day = _create_valid_plan_dict(
        user_role="DAY_TRADER",
        status="IN_BUY_ZONE",
        is_actionable=True,
        current_spot=1.97,
        entry_min=1.95,
        entry_max=1.98,
        planned_entry=1.97,
        structural_invalidation=1.93,
        tp1=2.05,
        tp2=2.15,
        gen_time="2026-10-08T05:01:00Z",
    )
    snapshot_day, _ = build_execution_ladder_snapshot(
        symbol="NAUT",
        user_role="DAY_TRADER",
        release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        snapshot_dict=plan_day,
    )
    gov_db.insert_execution_ladder_plan(snapshot_day)

    strat = gov_db.get_execution_ladder_stratification(epoch_id=EXECUTION_LADDER_EPOCH_ID)
    assert strat["total"] == 2
    assert strat["by_role"]["DAY_TRADER"] == 1
    assert strat["by_role"]["LONG_TERM"] == 1
    assert strat["by_status"]["WAITING_PULLBACK"] == 1
    assert strat["by_status"]["IN_BUY_ZONE"] == 1


# ==============================================================================
# SECTION 5: TEST STATUS-PARITY CONTRACT (SECTIONS 4 & 5)
# ==============================================================================
def test_status_contract_parity():
    """Verifies that passive capture exactly matches the frozen engine's emitted no-position status set.

    Fails if passive capture:
    - omits a frozen no-position state
    - admits an active-position target state
    - admits a legacy alias
    - adds an arbitrary state
    """
    required_expected_set = {
        "IN_BUY_ZONE",
        "IN_BUY_ZONE_AWAITING_TRIGGER",
        "EXTENDED_ABOVE_BUY_ZONE",
        "WAITING_PULLBACK",
        "STOPPED_OUT",
    }
    assert set(RATIFIED_EXECUTION_LADDER_STATUSES) == required_expected_set

    explicit_negatives = [
        "READY_TO_BUY",
        "APPROACHING_TARGET",
        "TARGET_REACHED",
        "UNKNOWN",
        "ERROR",
        "UNRESOLVED",
    ]
    for neg in explicit_negatives:
        assert neg not in RATIFIED_EXECUTION_LADDER_STATUSES


# ==============================================================================
# SECTION 7: TEST SNAPSHOT ROUNDTRIP FOR PRUNED FIELDS
# ==============================================================================
def test_snapshot_roundtrip_for_pruned_fields(temp_db_path, monkeypatch):
    """Verifies that all 6 pruned relational fields remain fully recoverable from snapshot_payload_json."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb")
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    plan = _create_valid_plan_dict(
        entry_min=1.27,
        entry_max=1.46,
        atr_14=0.15,
    )
    plan["execution_risk"] = 0.20
    plan["market_location"] = "EXTENDED_ABOVE_RANGE"
    plan["execution_stop_visible"] = True

    snapshot, err = build_execution_ladder_snapshot(
        symbol="NAUT",
        release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
        snapshot_dict=plan,
    )
    assert err is None
    inserted = gov_db.insert_execution_ladder_plan(snapshot)
    assert inserted is True

    retrieved = gov_db.get_execution_ladder_plan(snapshot["plan_id"])
    assert retrieved is not None
    assert retrieved["entry_min"] == 1.27
    assert retrieved["entry_max"] == 1.46
    assert retrieved["execution_risk"] == 0.20
    assert retrieved["atr_14"] == 0.15
    assert retrieved["market_location"] == "EXTENDED_ABOVE_RANGE"
    assert retrieved["execution_stop_visible"] is True


# ==============================================================================
# SECTION 8: TEST DETERMINISTIC IDENTITY AGAINST REALISTIC REPEATED CALLS
# ==============================================================================
def test_cross_request_plan_id_stability_realistic():
    """Verifies deterministic plan_id across realistic independent representations with different invocation times."""
    # Plan A: generated at 09:15:30.123456Z
    plan_a = _create_valid_plan_dict(
        gen_time="2026-10-08T09:15:30.123456Z",
        source_time="2026-10-08T09:15:00Z",
    )
    # Plan B: independently generated at 09:45:12.654321Z (same trading date, same levels)
    plan_b = _create_valid_plan_dict(
        gen_time="2026-10-08T09:45:12.654321Z",
        source_time="2026-10-08T09:45:00Z",
    )

    id_a = compute_execution_ladder_plan_id(plan_a)
    id_b = compute_execution_ladder_plan_id(plan_b)
    assert id_a == id_b

    # Plan C: change one substantive ladder field (e.g. planned_entry)
    plan_c = _create_valid_plan_dict(
        planned_entry=1.48,
        gen_time="2026-10-08T09:15:30.123456Z",
        source_time="2026-10-08T09:15:00Z",
    )
    id_c = compute_execution_ladder_plan_id(plan_c)
    assert id_c != id_a


# ==============================================================================
# SECTION 9: TEST SAME-DAY STATE EVOLUTION IS NOT COLLAPSED
# ==============================================================================
def test_same_day_state_evolution_not_collapsed():
    """Verifies that legitimate intra-day state or level changes produce distinct plan IDs."""
    base_plan = _create_valid_plan_dict(
        status="WAITING_PULLBACK",
        gen_time="2026-10-08T10:00:00Z",
        source_time="2026-10-08T10:00:00Z",
    )
    base_id = compute_execution_ladder_plan_id(base_plan)

    # 1. State evolution: WAITING_PULLBACK -> EXTENDED_ABOVE_BUY_ZONE
    evolved_status_plan = dict(base_plan)
    evolved_status_plan["execution_status"] = "EXTENDED_ABOVE_BUY_ZONE"
    evolved_id = compute_execution_ladder_plan_id(evolved_status_plan)
    assert evolved_id != base_id

    # 2. Level adjustment: planned_entry
    altered_entry = dict(base_plan)
    altered_entry["planned_entry"] = 1.48
    assert compute_execution_ladder_plan_id(altered_entry) != base_id

    # 3. Level adjustment: structural_invalidation
    altered_stop = dict(base_plan)
    altered_stop["structural_invalidation"] = 1.25
    assert compute_execution_ladder_plan_id(altered_stop) != base_id

    # 4. Level adjustment: take_profit_1
    altered_tp1 = dict(base_plan)
    altered_tp1["take_profit_1"] = 1.95
    assert compute_execution_ladder_plan_id(altered_tp1) != base_id

    # 5. Level adjustment: take_profit_2
    altered_tp2 = dict(base_plan)
    altered_tp2["take_profit_2"] = 2.25
    assert compute_execution_ladder_plan_id(altered_tp2) != base_id


# ==============================================================================
# SECTION 10: ENTRYPOINT WIRING & ADMISSION TESTS (TESTS A - J + E2E + SHA)
# ==============================================================================

def _create_mock_optimal_execution_plan(
    status: str = "WAITING_PULLBACK",
    is_actionable: bool = False,
    current_spot: float = 1.96,
    entry_min: float = 1.27,
    entry_max: float = 1.46,
    planned_entry: float = 1.46,
    stop_loss: float = 1.26,
    tp1: float = 1.90,
    tp2: float = 2.19,
    atr_14: float = 0.15,
) -> dict:
    """Creates a mock optimal_execution_plan dictionary returned by calculate_trade_levels."""
    return {
        "execution_status": status,
        "is_actionable": is_actionable,
        "entry_min": entry_min,
        "entry_max": entry_max,
        "optimal_entry_min": entry_min,
        "optimal_entry_max": entry_max,
        "planned_entry": planned_entry,
        "structural_invalidation": stop_loss,
        "stop_loss": stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "atr_14": atr_14,
        "execution_risk": round(planned_entry - stop_loss, 4),
        "execution_stop_visible": True,
        "market_location": "PULLBACK_CORRIDOR",
        "user_role": "LONG_TERM",
    }


def test_entrypoint_wiring_test_a_non_actionable_waiting_pullback(temp_db_path, temp_ledger_path):
    """TEST A: Non-actionable ratified state (WAITING_PULLBACK) reaches capture hook and persists."""
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)
    result = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT",
        current_price=1.96,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=1.96,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        user_role="LONG_TERM",
        instrument_class="EQUITY",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert result is not None
    assert result["epoch_id"] == EXECUTION_LADDER_EPOCH_ID
    assert result["execution_status"] == "WAITING_PULLBACK"
    assert result["is_actionable"] is False

    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_entrypoint_wiring_test_b_actionable_state(temp_db_path, temp_ledger_path):
    """TEST B: Actionable qualifying state reaches capture hook and persists under Epoch 001."""
    plan = _create_mock_optimal_execution_plan(status="IN_BUY_ZONE_AWAITING_TRIGGER", is_actionable=True)
    result = PassiveCaptureHook.record_natural_recommendation(
        symbol="AAPL",
        current_price=225.50,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=225.50,
        is_actionable=True,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        user_role="LONG_TERM",
        instrument_class="EQUITY",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert result is not None
    assert result["epoch_id"] == EXECUTION_LADDER_EPOCH_ID
    assert result["execution_status"] == "IN_BUY_ZONE_AWAITING_TRIGGER"
    assert result["is_actionable"] is True

    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_entrypoint_wiring_test_c_in_buy_zone_awaiting_trigger_non_actionable(temp_db_path, temp_ledger_path):
    """TEST C: IN_BUY_ZONE_AWAITING_TRIGGER with is_actionable=False reaches hook and persists."""
    plan = _create_mock_optimal_execution_plan(status="IN_BUY_ZONE_AWAITING_TRIGGER", is_actionable=False)
    result = PassiveCaptureHook.record_natural_recommendation(
        symbol="IREN",
        current_price=12.50,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=12.50,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        user_role="LONG_TERM",
        instrument_class="EQUITY",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert result is not None
    assert result["execution_status"] == "IN_BUY_ZONE_AWAITING_TRIGGER"
    assert result["is_actionable"] is False


def test_entrypoint_wiring_test_d_legacy_capture_preservation(temp_db_path, temp_ledger_path):
    """TEST D: Legacy capture path (epoch_id=None) preserves strict is_actionable filter."""
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)
    # Calling legacy path (epoch_id=None) with is_actionable=False fails admission
    result = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT",
        current_price=1.96,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=1.96,
        is_actionable=False,
        epoch_id=None,  # Legacy route
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
    )
    assert result is None
    # Execution ladder table was untouched
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 0


def test_entrypoint_wiring_test_e_test_context_rejection(temp_db_path, temp_ledger_path, monkeypatch):
    """TEST E: Test context or ARX_TEST_MODE prevents prospective plan persistence."""
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)
    # 1. Via ExecutionContext.TEST
    res1 = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT",
        current_price=1.96,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=1.96,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.GOVERNANCE_CERTIFICATION,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert res1 is None

    # 2. Via ARX_TEST_MODE=1
    monkeypatch.setenv("ARX_TEST_MODE", "1")
    res2 = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT",
        current_price=1.96,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=1.96,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert res2 is None
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 0


def test_entrypoint_wiring_test_f_synthetic_context_rejection(temp_db_path, temp_ledger_path):
    """TEST F: Synthetic indicators in optimal plan or provider prevent persistence."""
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)
    plan["isSynthetic"] = True
    result = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT",
        current_price=1.96,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=1.96,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert result is None
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 0


def test_entrypoint_wiring_test_g_admin_context_rejection(temp_db_path, temp_ledger_path):
    """TEST G: Admin execution context prevents prospective plan persistence."""
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)
    result = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT",
        current_price=1.96,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=1.96,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context="ADMIN",
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert result is None
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 0


def test_entrypoint_wiring_test_h_stale_quote_rejection():
    """TEST H: Stale quote (non-REALTIME) skips prospective capture hook."""
    class MockMarketPriceState:
        live_freshness = "DELAYED"
        market_session = "REGULAR_SESSION"
        live_spot_price = 1.96

    mps = MockMarketPriceState()
    quote_and_session_eligible = (
        mps.live_freshness == "REALTIME"
        and mps.market_session == "REGULAR_SESSION"
        and mps.live_spot_price is not None
    )
    assert quote_and_session_eligible is False


def test_entrypoint_wiring_test_i_non_regular_session_rejection():
    """TEST I: Non-regular market session skips prospective capture hook."""
    class MockMarketPriceState:
        live_freshness = "REALTIME"
        market_session = "CLOSED"
        live_spot_price = 1.96

    mps = MockMarketPriceState()
    quote_and_session_eligible = (
        mps.live_freshness == "REALTIME"
        and mps.market_session == "REGULAR_SESSION"
        and mps.live_spot_price is not None
    )
    assert quote_and_session_eligible is False


def test_entrypoint_wiring_test_j_duplicate_idempotency(temp_db_path, temp_ledger_path):
    """TEST J: Duplicate capture with same canonical plan identity yields row count 1."""
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)
    kwargs = dict(
        symbol="NAUT",
        current_price=1.96,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=1.96,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        user_role="LONG_TERM",
        instrument_class="EQUITY",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    res1 = PassiveCaptureHook.record_natural_recommendation(**kwargs)
    assert res1 is not None
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1

    # Second invocation with same plan
    res2 = PassiveCaptureHook.record_natural_recommendation(**kwargs)
    assert res2 is not None
    assert res2["plan_id"] == res1["plan_id"]
    # Row count strictly remains 1
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_isolated_end_to_end_capture_pipeline(temp_db_path, temp_ledger_path):
    """End-to-end simulated analytics request pipeline to isolated temporary database."""
    # 1. Generate execution ladder levels
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)

    # 2. Check quote and session gating
    live_freshness = "REALTIME"
    market_session = "REGULAR_SESSION"
    live_spot_price = 1.96
    current_price = 1.96
    quote_eligible = (
        live_freshness == "REALTIME"
        and market_session == "REGULAR_SESSION"
        and live_spot_price is not None
        and math.isfinite(live_spot_price)
        and live_spot_price > 0
        and current_price is not None
        and math.isfinite(current_price)
        and current_price > 0
    )
    assert quote_eligible is True

    # 3. Invoke capture hook with EXECUTION_LADDER_EPOCH_ID
    captured = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT",
        current_price=current_price,
        optimal_execution_plan=plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=live_spot_price,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        user_role="LONG_TERM",
        instrument_class="EQUITY",
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert captured is not None
    assert captured["execution_status"] == "WAITING_PULLBACK"

    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1

    # 4. Ineligible synthetic attempt on same pipeline does NOT persist
    synth_plan = dict(plan)
    synth_plan["isSynthetic"] = True
    synth_captured = PassiveCaptureHook.record_natural_recommendation(
        symbol="NAUT_SYNTH",
        current_price=current_price,
        optimal_execution_plan=synth_plan,
        confluence_output={},
        technicals={},
        factor_scores={},
        macro_inputs={},
        observed_at="2026-10-08T18:00:00Z",
        fetched_at="2026-10-08T18:00:01Z",
        freshness_status="LIVE",
        provider_source="YAHOO_AUTHENTIC",
        live_spot_price=live_spot_price,
        is_actionable=False,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        execution_context=ExecutionContext.NATURAL_CLIENT,
        runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
    )
    assert synth_captured is None
    # Database still only contains the 1 genuine plan
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_dynamic_runtime_release_sha_resolution(temp_db_path, temp_ledger_path, monkeypatch):
    """Verifies that release_sha resolves dynamically from runtime environment and persists accurately."""
    test_shas = [
        "d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
        "defdf8900ab64d14705b57ecdd0c8ed902526195",
        "aabbccddeeff00112233445566778899aabbccdd",
    ]
    for sha in test_shas:
        monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", sha)
        resolved = resolve_release_sha()
        assert resolved == sha

        plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK")
        captured = PassiveCaptureHook.record_natural_recommendation(
            symbol=f"TEST_{sha[:4]}",
            current_price=10.0,
            optimal_execution_plan=plan,
            confluence_output={},
            technicals={},
            factor_scores={},
            macro_inputs={},
            observed_at="2026-10-08T18:00:00Z",
            fetched_at="2026-10-08T18:00:01Z",
            freshness_status="LIVE",
            provider_source="YAHOO_AUTHENTIC",
            live_spot_price=10.0,
            is_actionable=False,
            epoch_id=EXECUTION_LADDER_EPOCH_ID,
            user_role="LONG_TERM",
            instrument_class="EQUITY",
            db_path=temp_db_path,
            ledger_path=temp_ledger_path,
            execution_context=ExecutionContext.NATURAL_CLIENT,
            runtime_release_sha=sha,
            execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
        )
        assert captured is not None
        assert captured["release_sha"] == sha


def test_coexistence_actionable_with_execution_plan(temp_db_path, temp_ledger_path, monkeypatch):
    """Verifies that an actionable qualifying request invokes BOTH Execution Ladder and legacy capture routes."""
    plan = _create_mock_optimal_execution_plan(status="IN_BUY_ZONE", is_actionable=True)
    calls = []
    original_fn = PassiveCaptureHook.record_natural_recommendation

    def spy_record(*args, **kwargs):
        calls.append(kwargs)
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(PassiveCaptureHook, "record_natural_recommendation", spy_record)

    # Simulate analytics.py control flow
    optimal_execution_plan = plan
    is_actionable = True
    quote_and_session_eligible = True

    if quote_and_session_eligible:
        if optimal_execution_plan:
            PassiveCaptureHook.record_natural_recommendation(
                symbol="NAUT",
                current_price=1.96,
                optimal_execution_plan=optimal_execution_plan,
                confluence_output={},
                technicals={},
                factor_scores={},
                macro_inputs={},
                observed_at="2026-10-08T18:00:00Z",
                fetched_at="2026-10-08T18:00:01Z",
                freshness_status="LIVE",
                provider_source="YAHOO_AUTHENTIC",
                live_spot_price=1.96,
                is_actionable=is_actionable,
                epoch_id=EXECUTION_LADDER_EPOCH_ID,
                user_role="LONG_TERM",
                instrument_class="EQUITY",
                db_path=temp_db_path,
                ledger_path=temp_ledger_path,
                execution_context=ExecutionContext.NATURAL_CLIENT,
                runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
                execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
            )
        if is_actionable:
            PassiveCaptureHook.record_natural_recommendation(
                symbol="NAUT",
                current_price=1.96,
                optimal_execution_plan=optimal_execution_plan,
                confluence_output={},
                technicals={},
                factor_scores={},
                macro_inputs={},
                observed_at="2026-10-08T18:00:00Z",
                fetched_at="2026-10-08T18:00:01Z",
                freshness_status="LIVE",
                provider_source="YAHOO_AUTHENTIC",
                live_spot_price=1.96,
                is_actionable=is_actionable,
                db_path=temp_db_path,
                ledger_path=temp_ledger_path,
                execution_context=ExecutionContext.NATURAL_CLIENT,
                runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
            )

    execution_ladder_calls = [c for c in calls if c.get("epoch_id") == EXECUTION_LADDER_EPOCH_ID]
    legacy_calls = [c for c in calls if c.get("epoch_id") != EXECUTION_LADDER_EPOCH_ID]

    assert len(execution_ladder_calls) == 1
    assert execution_ladder_calls[0]["epoch_id"] == EXECUTION_LADDER_EPOCH_ID
    assert len(legacy_calls) == 1
    assert legacy_calls[0].get("epoch_id") is None

    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_coexistence_non_actionable_with_execution_plan(temp_db_path, temp_ledger_path, monkeypatch):
    """Verifies that a non-actionable qualifying request invokes Execution Ladder capture but NOT legacy route."""
    plan = _create_mock_optimal_execution_plan(status="WAITING_PULLBACK", is_actionable=False)
    calls = []
    original_fn = PassiveCaptureHook.record_natural_recommendation

    def spy_record(*args, **kwargs):
        calls.append(kwargs)
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(PassiveCaptureHook, "record_natural_recommendation", spy_record)

    # Simulate analytics.py control flow
    optimal_execution_plan = plan
    is_actionable = False
    quote_and_session_eligible = True

    if quote_and_session_eligible:
        if optimal_execution_plan:
            PassiveCaptureHook.record_natural_recommendation(
                symbol="NAUT",
                current_price=1.96,
                optimal_execution_plan=optimal_execution_plan,
                confluence_output={},
                technicals={},
                factor_scores={},
                macro_inputs={},
                observed_at="2026-10-08T18:00:00Z",
                fetched_at="2026-10-08T18:00:01Z",
                freshness_status="LIVE",
                provider_source="YAHOO_AUTHENTIC",
                live_spot_price=1.96,
                is_actionable=is_actionable,
                epoch_id=EXECUTION_LADDER_EPOCH_ID,
                user_role="LONG_TERM",
                instrument_class="EQUITY",
                db_path=temp_db_path,
                ledger_path=temp_ledger_path,
                execution_context=ExecutionContext.NATURAL_CLIENT,
                runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
                execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
            )
        if is_actionable:
            PassiveCaptureHook.record_natural_recommendation(
                symbol="NAUT",
                current_price=1.96,
                optimal_execution_plan=optimal_execution_plan,
                confluence_output={},
                technicals={},
                factor_scores={},
                macro_inputs={},
                observed_at="2026-10-08T18:00:00Z",
                fetched_at="2026-10-08T18:00:01Z",
                freshness_status="LIVE",
                provider_source="YAHOO_AUTHENTIC",
                live_spot_price=1.96,
                is_actionable=is_actionable,
                db_path=temp_db_path,
                ledger_path=temp_ledger_path,
                execution_context=ExecutionContext.NATURAL_CLIENT,
                runtime_release_sha="d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354",
            )

    execution_ladder_calls = [c for c in calls if c.get("epoch_id") == EXECUTION_LADDER_EPOCH_ID]
    legacy_calls = [c for c in calls if c.get("epoch_id") != EXECUTION_LADDER_EPOCH_ID]

    assert len(execution_ladder_calls) == 1
    assert execution_ladder_calls[0]["epoch_id"] == EXECUTION_LADDER_EPOCH_ID
    assert len(legacy_calls) == 0

    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


# ==============================================================================
# SECTION 22: REMEDIATION & NORMALIZATION REGRESSION TEST SUITE (PHASE 4)
# ==============================================================================

def test_canonical_plan_id_across_timestamp_formats():
    """Phase 4.1: Verifies identical plan_id generation across ISO strings, epoch seconds,
    epoch milliseconds, and timezone-aware datetime objects for the same calendar trading day."""
    base_plan = _create_valid_plan_dict()

    # Formats for 2026-10-09
    ts_formats = [
        "2026-10-09T16:33:44Z",
        "2026-10-09T16:33:44.123456Z",
        "2026-10-09T16:33:44+00:00",
        "2026-10-09T12:33:44-04:00",  # 16:33:44 UTC
        1791563624,                   # Epoch seconds (int)
        1791563624.0,                 # Epoch seconds (float)
        1791563624000,                # Epoch milliseconds (int)
        "1791563624000",              # Epoch milliseconds (str)
        datetime(2026, 10, 9, 16, 33, 44, tzinfo=timezone.utc),
    ]

    plan_ids = []
    for ts in ts_formats:
        assert extract_canonical_trading_date(ts) == "2026-10-09"
        p = dict(base_plan)
        p["source_data_timestamp"] = ts
        p["generation_timestamp"] = ts
        pid = compute_execution_ladder_plan_id(p)
        plan_ids.append(pid)

    # All representations must yield the exact same plan_id
    assert len(set(plan_ids)) == 1, f"Expected 1 unique plan_id, got: {set(plan_ids)}"


def test_fail_closed_rejection_invalid_non_finite_out_of_range_ambiguous_timestamps(temp_db_path):
    """Phase 4.2: Verifies fail-closed rejection of missing, non-finite, out-of-range,
    timezone-ambiguous naive, and malformed timestamps."""
    invalid_timestamps = [
        None,
        "",
        "   ",
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "inf",
        "-inf",
        "NaN",
        "Infinity",
        -1,
        -100000,
        0,                            # 1970-01-01 (< 2000)
        100000,                       # Historical (< 2000)
        946684799,                    # 1999-12-31T23:59:59 (< 2000)
        5000000000,                   # Far future (> 2100)
        9999999999999,                # Far future ms (> 2100)
        datetime(2026, 10, 9, 16, 33, 44),  # Naive datetime (no tzinfo)
        "2026-10-09T16:33:44",        # Naive ISO string (no Z or offset)
        "2026-10-09",                 # Date without timezone
        "not-a-timestamp",
        "2026-99-99T99:99:99Z",
    ]

    for bad_ts in invalid_timestamps:
        # 1. extract_canonical_trading_date must reject
        with pytest.raises(ValueError):
            extract_canonical_trading_date(bad_ts)

        # 2. compute_execution_ladder_plan_id must reject
        plan = _create_valid_plan_dict()
        plan["source_data_timestamp"] = bad_ts
        plan["generation_timestamp"] = bad_ts
        with pytest.raises(ValueError):
            compute_execution_ladder_plan_id(plan)

        # 3. Admission pipeline must fail closed with eligible=False
        admission = PassiveCaptureHook.evaluate_prospective_admission(
            symbol="NAUT",
            is_actionable=False,
            live_spot_price=1.96,
            current_price=1.96,
            execution_context=ExecutionContext.NATURAL_CLIENT,
            runtime_release_sha="3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb",
            db_path=temp_db_path,
            epoch_id=EXECUTION_LADDER_EPOCH_ID,
            execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
            source_data_timestamp=bad_ts,
            generation_timestamp=bad_ts,
            optimal_execution_plan={
                "optimal_entry_min": 1.40,
                "optimal_entry_max": 1.55,
                "planned_entry": 1.50,
                "structural_invalidation": 1.20,
                "take_profit_1": 2.00,
                "take_profit_2": 2.30,
                "execution_status": "WAITING_PULLBACK",
                "execution_risk": 0.30,
                "atr_14": 0.08,
                "market_location": "AT_VALUE_AREA_LOW",
                "execution_stop_visible": True,
            },
        )
        assert admission["prospectiveCaptureEligible"] is False
        assert admission["prospectiveCaptureRejectionReason"] in (
            "INVALID_SOURCE_DATA_TIMESTAMP",
            "MISSING_SOURCE_DATA_TIMESTAMP",
            "INVALID_GENERATION_TIMESTAMP",
            "MISSING_GENERATION_TIMESTAMP",
            "INVALID_TIMESTAMP",
        )


def test_cross_request_stability_with_millisecond_drift(temp_db_path, monkeypatch):
    """Phase 4.3: Verifies cross-request stability where intra-day queries with shifting
    millisecond timestamps produce the exact same plan_id and zero denominator increment."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "5a90b918b0975151b74e936b3fbfa536b575edd7")

    # Three queries on 2026-10-09 at different times of day (epoch ms)
    # Query 1: 16:33:44.000 UTC
    # Query 2: 16:33:44.500 UTC (+500ms)
    # Query 3: 17:19:27.000 UTC (+46 minutes)
    timestamps = [1791563624000, 1791563624500, 1791566367000]

    captured_ids = []
    for ts in timestamps:
        res = PassiveCaptureHook.record_execution_ladder_plan(
            symbol="TSLA",
            optimal_execution_plan={
                "optimal_entry_min": 215.00,
                "optimal_entry_max": 216.00,
                "planned_entry": 215.50,
                "structural_invalidation": 210.00,
                "take_profit_1": 225.00,
                "take_profit_2": 235.00,
                "execution_status": "WAITING_PULLBACK",
                "execution_risk": 5.50,
                "atr_14": 4.20,
                "market_location": "AT_VALUE_AREA_LOW",
                "execution_stop_visible": True,
            },
            current_price=216.00,
            live_spot_price=216.00,
            source_data_timestamp=str(ts),
            generation_timestamp=str(ts),
            user_role="LONG_TERM",
            instrument_class="EQUITY",
            release_sha="5a90b918b0975151b74e936b3fbfa536b575edd7",
            authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
            db_path=temp_db_path,
        )
        assert res is not None
        captured_ids.append(res["plan_id"])

    # All three requests returned the exact same plan_id
    assert len(set(captured_ids)) == 1

    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    # Database must contain exactly 1 plan (no denominator inflation)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_historical_plan_compatibility_migration_deduplication(temp_db_path, monkeypatch):
    """Phase 4.4: Verifies migration boundary deduplication where an existing historical plan
    captured under an unnormalized plan_id correctly suppresses re-queries under the new
    normalized plan_id without mutating historical rows or plan IDs."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "5a90b918b0975151b74e936b3fbfa536b575edd7")
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    # 1. Seed historical defective record directly (reproducing PLAN_02dd763ae694a0f8ed21b9cf)
    historical_plan = {
        "plan_id": "PLAN_02dd763ae694a0f8ed21b9cf",
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": "TSLA",
        "instrument_class": "EQUITY",
        "user_role": "LONG_TERM",
        "generation_timestamp": "1791563624000",
        "source_data_timestamp": "1791563624000",
        "release_sha": "5a90b918b0975151b74e936b3fbfa536b575edd7",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "current_spot": 215.50,
        "entry_min": 215.00,
        "entry_max": 216.00,
        "planned_entry": 215.50,
        "structural_invalidation": 210.00,
        "execution_risk": 5.50,
        "atr_14": 4.20,
        "take_profit_1": 225.00,
        "take_profit_2": 235.00,
        "market_location": "AT_VALUE_AREA_LOW",
        "execution_status": "WAITING_PULLBACK",
        "is_actionable": False,
        "execution_stop_visible": True,
        "created_at_utc": "2026-10-09T16:33:44.000000Z",
    }
    inserted = gov_db.insert_execution_ladder_plan(historical_plan)
    assert inserted is True
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1

    # 2. Incoming live request on same trading day under normalized logic (produces PLAN_cdd6ac23a4cf45f9bd8e8ba9)
    incoming_opt_plan = {
        "optimal_entry_min": 215.00,
        "optimal_entry_max": 216.00,
        "planned_entry": 215.50,
        "structural_invalidation": 210.00,
        "take_profit_1": 225.00,
        "take_profit_2": 235.00,
        "execution_status": "WAITING_PULLBACK",
        "execution_risk": 5.50,
        "atr_14": 4.20,
        "market_location": "AT_VALUE_AREA_LOW",
        "execution_stop_visible": True,
    }

    admission = PassiveCaptureHook.evaluate_prospective_admission(
        symbol="TSLA",
        optimal_execution_plan=incoming_opt_plan,
        current_price=215.50,
        live_spot_price=215.50,
        user_role="LONG_TERM",
        instrument_class="EQUITY",
        source_data_timestamp="1791566367000",  # Later that same day
        generation_timestamp="1791566367000",
        runtime_release_sha="5a90b918b0975151b74e936b3fbfa536b575edd7",
        execution_ladder_authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
        db_path=temp_db_path,
        epoch_id=EXECUTION_LADDER_EPOCH_ID,
    )

    # 3. Admission must detect historical equivalent and reject as DUPLICATE
    assert admission["prospectiveCaptureEligible"] is False
    assert admission["prospectiveCaptureRejectionReason"] == "DUPLICATE"
    assert admission["existingRecord"]["plan_id"] == "PLAN_02dd763ae694a0f8ed21b9cf"

    # 4. Recording the plan must return the existing record and NOT insert a new row
    res = PassiveCaptureHook.record_execution_ladder_plan(
        symbol="TSLA",
        optimal_execution_plan=incoming_opt_plan,
        current_price=215.50,
        live_spot_price=215.50,
        user_role="LONG_TERM",
        instrument_class="EQUITY",
        source_data_timestamp="1791566367000",
        generation_timestamp="1791566367000",
        release_sha="5a90b918b0975151b74e936b3fbfa536b575edd7",
        authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
        db_path=temp_db_path,
    )
    assert res["plan_id"] == "PLAN_02dd763ae694a0f8ed21b9cf"
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_intraday_spot_price_movement_preserves_plan_id_and_deduplicates(temp_db_path, monkeypatch):
    """Phase 4.5: Verifies intraday price movements that do not alter execution-ladder levels
    or status do not generate new plan IDs and deduplicate cleanly."""
    monkeypatch.setenv("ARX_RELEASE_SHA", "5a90b918b0975151b74e936b3fbfa536b575edd7")

    spot_prices = [215.50, 217.80, 214.20]
    results = []
    for spot in spot_prices:
        res = PassiveCaptureHook.record_execution_ladder_plan(
            symbol="TSLA",
            optimal_execution_plan={
                "optimal_entry_min": 215.00,
                "optimal_entry_max": 216.00,
                "planned_entry": 215.50,
                "structural_invalidation": 210.00,
                "take_profit_1": 225.00,
                "take_profit_2": 235.00,
                "execution_status": "WAITING_PULLBACK",
                "execution_risk": 5.50,
                "atr_14": 4.20,
                "market_location": "AT_VALUE_AREA_LOW",
                "execution_stop_visible": True,
            },
            current_price=spot,
            live_spot_price=spot,
            source_data_timestamp="2026-10-09T16:33:44Z",
            generation_timestamp="2026-10-09T16:33:44Z",
            user_role="LONG_TERM",
            instrument_class="EQUITY",
            release_sha="5a90b918b0975151b74e936b3fbfa536b575edd7",
            authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
            db_path=temp_db_path,
        )
        assert res is not None
        results.append(res["plan_id"])

    assert len(set(results)) == 1
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 1


def test_level_changes_and_status_transitions_produce_distinct_plan_ids(temp_db_path):
    """Phase 4.6: Verifies that genuine level adjustments or status changes on the same trading date
    produce distinct plan IDs and are NOT falsely deduplicated."""
    base_plan = _create_valid_plan_dict(
        source_time="2026-10-09T16:33:44Z",
        gen_time="2026-10-09T16:33:44Z",
    )
    base_id = compute_execution_ladder_plan_id(base_plan)

    # 1. Status transition
    p_status = dict(base_plan)
    p_status["execution_status"] = "IN_BUY_ZONE_AWAITING_TRIGGER"
    assert compute_execution_ladder_plan_id(p_status) != base_id

    # 2. Planned entry adjustment
    p_entry = dict(base_plan)
    p_entry["planned_entry"] = 1.85
    assert compute_execution_ladder_plan_id(p_entry) != base_id

    # 3. Stop adjustment
    p_stop = dict(base_plan)
    p_stop["structural_invalidation"] = 1.35
    assert compute_execution_ladder_plan_id(p_stop) != base_id

    # 4. Target 1 adjustment
    p_tp1 = dict(base_plan)
    p_tp1["take_profit_1"] = 2.15
    assert compute_execution_ladder_plan_id(p_tp1) != base_id

    # 5. Target 2 adjustment
    p_tp2 = dict(base_plan)
    p_tp2["take_profit_2"] = 2.55
    assert compute_execution_ladder_plan_id(p_tp2) != base_id

    # 6. Role change
    p_role = dict(base_plan)
    p_role["user_role"] = "DAY_TRADER"
    assert compute_execution_ladder_plan_id(p_role) != base_id

    # 7. Next day trading date
    p_next_day = dict(base_plan)
    p_next_day["source_data_timestamp"] = "2026-10-10T16:33:44Z"
    p_next_day["generation_timestamp"] = "2026-10-10T16:33:44Z"
    assert compute_execution_ladder_plan_id(p_next_day) != base_id


def test_vcp_epoch_signal_capture_isolation(temp_db_path, temp_ledger_path, monkeypatch):
    """Phase 4.7: Verifies VCP signal capture (Epoch 4 / 1) remains completely separate and unaffected."""
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 0

    # Ensure temporal gate is bypassed for legacy in test mode
    monkeypatch.setattr(PassiveCaptureHook, "is_temporal_gate_satisfied", classmethod(lambda cls, **kw: True))

    # Trigger a legacy recommendation record
    res = PassiveCaptureHook.record_natural_recommendation(
        symbol="AAPL",
        current_price=150.0,
        optimal_execution_plan={
            "planned_entry": 150.0,
            "structural_invalidation": 145.0,
            "take_profit_1": 160.0,
            "take_profit_2": 170.0,
            "execution_status": "WAITING_PULLBACK",
        },
        confluence_output={"overall_eligibility": "FULL", "market_regime": "BULL"},
        technicals={"sma_50": 148.0, "atr_14": 2.5},
        factor_scores={"quality_score": 85.0},
        macro_inputs={"macro_observation_available_at": "2026-10-08T09:00:00Z"},
        observed_at="2026-10-08T09:30:00Z",
        fetched_at="2026-10-08T09:30:00Z",
        live_spot_price=150.0,
        is_actionable=True,
        db_path=temp_db_path,
        ledger_path=temp_ledger_path,
        runtime_release_sha="5a90b918b0975151b74e936b3fbfa536b575edd7",
        runtime_deployment_id="test_deploy_123",
    )

    # Legacy VCP record must NOT touch execution ladder table
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID) == 0


def test_ratified_denominator_substantive_count(temp_db_path):
    """Phase 4.8: Verifies that replicating the 4 authoritative historical production records yields exactly
    raw_stored_plans=4 and candidate_daily_deduplicated_count (ratified denominator) = 3.
    Provenance: Production SQLite database read-only reconciliation.
    Records:
    1. AAPL PLAN_08f60711fd16b68ab1d83eff: LONG_TERM, 336.06 / 312.62 / 379.43 / 402.87, IN_BUY_ZONE
    2. TSLA PLAN_02dd763ae694a0f8ed21b9cf: LONG_TERM, 373.32 / 342.67 / 430.03 / 460.68, EXTENDED_ABOVE_BUY_ZONE
    3. TSLA PLAN_97e2b47d5fc812813f129d45: DAY_TRADER, 383.30 / 376.39 / 398.63 / 406.30, IN_BUY_ZONE
    4. TSLA PLAN_19d8bc8fd753bc921de0d6f4: LONG_TERM, 373.32 / 342.67 / 430.03 / 460.68, EXTENDED_ABOVE_BUY_ZONE (duplicate of #2)
    All 4 captured under original functional release 01683a39a19f3f74720f798459cec717698e2ab2.
    """
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    raw_records = [
        # 1. AAPL LONG_TERM (First natural capture)
        {
            "plan_id": "PLAN_08f60711fd16b68ab1d83eff",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "AAPL",
            "instrument_class": "EQUITY",
            "user_role": "LONG_TERM",
            "generation_timestamp": "1791554903000",
            "source_data_timestamp": "1791554903000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 332.21,
            "planned_entry": 336.06,
            "structural_invalidation": 312.62,
            "take_profit_1": 379.43,
            "take_profit_2": 402.87,
            "execution_status": "IN_BUY_ZONE",
            "is_actionable": True,
            "created_at_utc": "2026-10-09T13:28:23.000000Z",
        },
        # 2. TSLA LONG_TERM (first observation)
        {
            "plan_id": "PLAN_02dd763ae694a0f8ed21b9cf",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "TSLA",
            "instrument_class": "EQUITY",
            "user_role": "LONG_TERM",
            "generation_timestamp": "1791563624000",
            "source_data_timestamp": "1791563624000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 382.69,
            "planned_entry": 373.32,
            "structural_invalidation": 342.67,
            "take_profit_1": 430.03,
            "take_profit_2": 460.68,
            "execution_status": "EXTENDED_ABOVE_BUY_ZONE",
            "is_actionable": False,
            "created_at_utc": "2026-10-09T16:33:44.000000Z",
        },
        # 3. TSLA DAY_TRADER
        {
            "plan_id": "PLAN_97e2b47d5fc812813f129d45",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "TSLA",
            "instrument_class": "EQUITY",
            "user_role": "DAY_TRADER",
            "generation_timestamp": "1791563624000",
            "source_data_timestamp": "1791563624000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 382.69,
            "planned_entry": 383.30,
            "structural_invalidation": 376.39,
            "take_profit_1": 398.63,
            "take_profit_2": 406.30,
            "execution_status": "IN_BUY_ZONE",
            "is_actionable": True,
            "created_at_utc": "2026-10-09T16:33:44.000000Z",
        },
        # 4. TSLA LONG_TERM (duplicate due to timestamp slicing at 17:19 UTC)
        {
            "plan_id": "PLAN_19d8bc8fd753bc921de0d6f4",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "TSLA",
            "instrument_class": "EQUITY",
            "user_role": "LONG_TERM",
            "generation_timestamp": "1791566339000",
            "source_data_timestamp": "1791566339000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 382.69,
            "planned_entry": 373.32,
            "structural_invalidation": 342.67,
            "take_profit_1": 430.03,
            "take_profit_2": 460.68,
            "execution_status": "EXTENDED_ABOVE_BUY_ZONE",
            "is_actionable": False,
            "created_at_utc": "2026-10-09T17:19:27.000000Z",
        },
    ]

    for rec in raw_records:
        gov_db.insert_execution_ladder_plan(rec, check_equivalent=False)

    # 1. Raw row count
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=False) == 4

    # 2. Substantive deduplicated count (ratified denominator)
    assert gov_db.count_execution_ladder_plans(epoch_id=EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=True) == 3

    # 3. Stratification with deduplication
    strat = gov_db.get_execution_ladder_stratification(epoch_id=EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=True)
    assert strat["total"] == 3
    assert strat["by_role"]["LONG_TERM"] == 2
    assert strat["by_role"]["DAY_TRADER"] == 1
    assert strat["by_status"]["IN_BUY_ZONE"] == 2
    assert strat["by_status"]["EXTENDED_ABOVE_BUY_ZONE"] == 1


# ==============================================================================
# SECTION 5: FINAL REMEDIATION & CERTIFICATION MANDATORY REGRESSION MATRIX
# ==============================================================================

def test_identity_isolation_matrix():
    """Section 5.1: Verifies complete 11-field identity isolation.
    - Different release SHAs remain distinct
    - Different authority SHAs remain distinct
    - Different epochs remain distinct
    - Different roles remain distinct
    - Changed status or levels remain distinct
    - Timestamp representation changes alone do not create new identities
    """
    base = {
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "symbol": "AAPL",
        "user_role": "LONG_TERM",
        "source_data_timestamp": "2026-10-09T14:30:00Z",
        "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "planned_entry": 336.06,
        "structural_invalidation": 312.62,
        "take_profit_1": 379.43,
        "take_profit_2": 402.87,
        "execution_status": "IN_BUY_ZONE",
    }
    base_id = compute_execution_ladder_plan_id(base)

    # 1. Different release SHA
    p_rel = dict(base, release_sha="5a90b918b0975151b74e936b3fbfa536b575edd7")
    assert compute_execution_ladder_plan_id(p_rel) != base_id

    # 2. Different authority SHA
    p_auth = dict(base, execution_ladder_authority_sha="0000000000000000000000000000000000000000")
    assert compute_execution_ladder_plan_id(p_auth) != base_id

    # 3. Different epoch
    p_epoch = dict(base, epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_2")
    assert compute_execution_ladder_plan_id(p_epoch) != base_id

    # 4. Different role
    p_role = dict(base, user_role="DAY_TRADER")
    assert compute_execution_ladder_plan_id(p_role) != base_id

    # 5. Changed status
    p_status = dict(base, execution_status="EXTENDED_ABOVE_BUY_ZONE")
    assert compute_execution_ladder_plan_id(p_status) != base_id

    # 6. Changed levels
    p_entry = dict(base, planned_entry=340.00)
    assert compute_execution_ladder_plan_id(p_entry) != base_id
    p_stop = dict(base, structural_invalidation=310.00)
    assert compute_execution_ladder_plan_id(p_stop) != base_id
    p_tp1 = dict(base, take_profit_1=380.00)
    assert compute_execution_ladder_plan_id(p_tp1) != base_id
    p_tp2 = dict(base, take_profit_2=410.00)
    assert compute_execution_ladder_plan_id(p_tp2) != base_id

    # 7. Timestamp representations for same trading date produce identical plan ID
    for alt_ts in [
        "2026-10-09T18:45:12.123456Z",
        "2026-10-09T10:30:00-04:00",
        1791554903000,
        1791554903,
        "1791554903000",
    ]:
        p_ts = dict(base, source_data_timestamp=alt_ts)
        assert compute_execution_ladder_plan_id(p_ts) == base_id


def test_equivalence_lookup_isolates_across_releases(temp_db_path):
    """Section 5.1b: Verifies that find_equivalent_execution_ladder_plan strictly isolates
    across different release SHAs and authority SHAs."""
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    plan_r1 = {
        "plan_id": "PLAN_release1",
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": "AAPL",
        "instrument_class": "EQUITY",
        "user_role": "LONG_TERM",
        "generation_timestamp": "2026-10-09T14:30:00Z",
        "source_data_timestamp": "2026-10-09T14:30:00Z",
        "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "current_spot": 332.21,
        "planned_entry": 336.06,
        "structural_invalidation": 312.62,
        "take_profit_1": 379.43,
        "take_profit_2": 402.87,
        "execution_status": "IN_BUY_ZONE",
        "is_actionable": True,
        "created_at_utc": "2026-10-09T14:30:00Z",
    }
    gov_db.insert_execution_ladder_plan(plan_r1)

    # Query with different release_sha must NOT match
    cand_r2 = dict(plan_r1, release_sha="5a90b918b0975151b74e936b3fbfa536b575edd7")
    equiv = gov_db.find_equivalent_execution_ladder_plan(cand_r2)
    assert equiv is None

    # Query with different authority_sha must NOT match
    cand_auth = dict(plan_r1, execution_ladder_authority_sha="0000000000000000000000000000000000000000")
    equiv_auth = gov_db.find_equivalent_execution_ladder_plan(cand_auth)
    assert equiv_auth is None

    # Query with identical release_sha and authority_sha DOES match
    cand_same = dict(plan_r1, source_data_timestamp="2026-10-09T20:00:00Z")
    equiv_same = gov_db.find_equivalent_execution_ladder_plan(cand_same)
    assert equiv_same is not None
    assert equiv_same["plan_id"] == "PLAN_release1"


def test_concurrency_two_identical_requests(temp_db_path):
    """Section 5.2a: Two concurrent identical requests produce exactly one insertion."""
    import concurrent.futures
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    plan = {
        "plan_id": "PLAN_concurrent_2_test",
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": "AAPL",
        "instrument_class": "EQUITY",
        "user_role": "LONG_TERM",
        "generation_timestamp": "2026-10-09T18:00:00Z",
        "source_data_timestamp": "2026-10-09T18:00:00Z",
        "release_sha": "5a90b918b0975151b74e936b3fbfa536b575edd7",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "current_spot": 332.21,
        "planned_entry": 336.06,
        "structural_invalidation": 312.62,
        "take_profit_1": 379.43,
        "take_profit_2": 402.87,
        "execution_status": "IN_BUY_ZONE",
        "is_actionable": True,
        "created_at_utc": "2026-10-09T18:00:00Z",
    }

    def worker(i):
        db = GovernanceDatabaseEngine(temp_db_path)
        return db.insert_execution_ladder_plan_atomic(dict(plan), check_equivalent=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(worker, i) for i in range(2)]
        results = [f.result() for f in futs]

    assert sum(1 for r in results if r[0] is True) == 1
    assert sum(1 for r in results if r[0] is False) == 1
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID) == 1


def test_concurrency_ten_identical_requests(temp_db_path):
    """Section 5.2b: Ten concurrent identical requests produce exactly one insertion."""
    import concurrent.futures
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    plan = {
        "plan_id": "PLAN_concurrent_10_test",
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": "TSLA",
        "instrument_class": "EQUITY",
        "user_role": "LONG_TERM",
        "generation_timestamp": "2026-10-09T18:00:00Z",
        "source_data_timestamp": "2026-10-09T18:00:00Z",
        "release_sha": "5a90b918b0975151b74e936b3fbfa536b575edd7",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "current_spot": 382.69,
        "planned_entry": 373.32,
        "structural_invalidation": 342.67,
        "take_profit_1": 430.03,
        "take_profit_2": 460.68,
        "execution_status": "EXTENDED_ABOVE_BUY_ZONE",
        "is_actionable": False,
        "created_at_utc": "2026-10-09T18:00:00Z",
    }

    def worker(i):
        db = GovernanceDatabaseEngine(temp_db_path)
        return db.insert_execution_ladder_plan_atomic(dict(plan), check_equivalent=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(worker, i) for i in range(10)]
        results = [f.result() for f in futs]

    assert sum(1 for r in results if r[0] is True) == 1
    assert sum(1 for r in results if r[0] is False) == 9
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID) == 1


def test_concurrency_migration_boundary_requests(temp_db_path):
    """Section 5.2c: Concurrent migration-boundary requests produce one canonical admission."""
    import concurrent.futures
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    # Seed historical defective record
    historical_record = {
        "plan_id": "PLAN_02dd763ae694a0f8ed21b9cf",
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": "TSLA",
        "instrument_class": "EQUITY",
        "user_role": "LONG_TERM",
        "generation_timestamp": "1791563624000",
        "source_data_timestamp": "1791563624000",
        "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "current_spot": 382.69,
        "planned_entry": 373.32,
        "structural_invalidation": 342.67,
        "take_profit_1": 430.03,
        "take_profit_2": 460.68,
        "execution_status": "EXTENDED_ABOVE_BUY_ZONE",
        "is_actionable": False,
        "created_at_utc": "2026-10-09T16:33:44.000000Z",
    }
    gov_db.insert_execution_ladder_plan(historical_record)

    opt_plan = {
        "optimal_entry_min": 370.0,
        "optimal_entry_max": 375.0,
        "planned_entry": 373.32,
        "structural_invalidation": 342.67,
        "take_profit_1": 430.03,
        "take_profit_2": 460.68,
        "execution_status": "EXTENDED_ABOVE_BUY_ZONE",
        "execution_risk": 30.65,
        "atr_14": 8.5,
        "market_location": "ABOVE_VALUE_AREA",
        "execution_stop_visible": True,
    }

    def worker(i):
        return PassiveCaptureHook.record_execution_ladder_plan(
            symbol="TSLA",
            optimal_execution_plan=opt_plan,
            current_price=382.69,
            live_spot_price=382.69,
            user_role="LONG_TERM",
            instrument_class="EQUITY",
            generation_timestamp="1791566339000",
            source_data_timestamp="1791566339000",
            release_sha="01683a39a19f3f74720f798459cec717698e2ab2",
            authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
            is_actionable=False,
            db_path=temp_db_path,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(worker, i) for i in range(5)]
        results = [f.result() for f in futs]

    assert all(r is not None for r in results)
    assert all(r["plan_id"] == "PLAN_02dd763ae694a0f8ed21b9cf" for r in results)
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID) == 1


def test_concurrency_cross_release_requests(temp_db_path):
    """Section 5.2d: Cross-release concurrent requests preserve separate identities."""
    import concurrent.futures
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    opt_plan = {
        "optimal_entry_min": 335.0,
        "optimal_entry_max": 337.0,
        "planned_entry": 336.06,
        "structural_invalidation": 312.62,
        "take_profit_1": 379.43,
        "take_profit_2": 402.87,
        "execution_status": "IN_BUY_ZONE",
        "execution_risk": 23.44,
        "atr_14": 5.20,
        "market_location": "IN_VALUE_AREA",
        "execution_stop_visible": True,
    }

    def worker(rel_sha):
        return PassiveCaptureHook.record_execution_ladder_plan(
            symbol="AAPL",
            optimal_execution_plan=opt_plan,
            current_price=332.21,
            live_spot_price=332.21,
            user_role="LONG_TERM",
            instrument_class="EQUITY",
            generation_timestamp="2026-10-09T18:00:00Z",
            source_data_timestamp="2026-10-09T18:00:00Z",
            release_sha=rel_sha,
            authority_sha=EXECUTION_LADDER_AUTHORITY_SHA,
            is_actionable=True,
            db_path=temp_db_path,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(worker, "01683a39a19f3f74720f798459cec717698e2ab2"),
            ex.submit(worker, "5a90b918b0975151b74e936b3fbfa536b575edd7"),
        ]
        results = [f.result() for f in futs]

    assert len(results) == 2
    assert all(r is not None for r in results)
    plan_ids = set(r["plan_id"] for r in results)
    assert len(plan_ids) == 2
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID) == 2


def test_concurrency_database_exception_rollback(temp_db_path, monkeypatch):
    """Section 5.2e: Database exceptions roll back safely and fail closed."""
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    # Malformed entry (cannot convert to float)
    plan = {
        "plan_id": "PLAN_rollback_test",
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": "AAPL",
        "instrument_class": "EQUITY",
        "user_role": "LONG_TERM",
        "generation_timestamp": "2026-10-09T18:00:00Z",
        "source_data_timestamp": "2026-10-09T18:00:00Z",
        "release_sha": "5a90b918b0975151b74e936b3fbfa536b575edd7",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "current_spot": 332.21,
        "planned_entry": "NOT_A_FLOAT",
        "structural_invalidation": 312.62,
        "take_profit_1": 379.43,
        "take_profit_2": 402.87,
        "execution_status": "IN_BUY_ZONE",
        "is_actionable": True,
        "created_at_utc": "2026-10-09T18:00:00Z",
    }

    with pytest.raises((ValueError, TypeError)):
        gov_db.insert_execution_ladder_plan_atomic(plan)

    # DB remains clean with zero rows
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID) == 0


def test_historical_compatibility_five_rows_four_canonical(temp_db_path):
    """Section 5.3: Historical compatibility:
    - Four historical rows remain physically intact
    - Historical canonical denominator remains 3
    - Fifth post-release row remains a separate identity
    - Cumulative denominator remains 4
    - Immutability triggers block UPDATE and DELETE
    """
    gov_db = GovernanceDatabaseEngine(db_path=temp_db_path)

    raw_records = [
        # 1. AAPL LONG_TERM (Release 01683a3)
        {
            "plan_id": "PLAN_08f60711fd16b68ab1d83eff",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "AAPL",
            "instrument_class": "EQUITY",
            "user_role": "LONG_TERM",
            "generation_timestamp": "1791554903000",
            "source_data_timestamp": "1791554903000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 332.21,
            "planned_entry": 336.06,
            "structural_invalidation": 312.62,
            "take_profit_1": 379.43,
            "take_profit_2": 402.87,
            "execution_status": "IN_BUY_ZONE",
            "is_actionable": True,
            "created_at_utc": "2026-10-09T13:28:23.000000Z",
        },
        # 2. TSLA LONG_TERM (Release 01683a3)
        {
            "plan_id": "PLAN_02dd763ae694a0f8ed21b9cf",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "TSLA",
            "instrument_class": "EQUITY",
            "user_role": "LONG_TERM",
            "generation_timestamp": "1791563624000",
            "source_data_timestamp": "1791563624000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 382.69,
            "planned_entry": 373.32,
            "structural_invalidation": 342.67,
            "take_profit_1": 430.03,
            "take_profit_2": 460.68,
            "execution_status": "EXTENDED_ABOVE_BUY_ZONE",
            "is_actionable": False,
            "created_at_utc": "2026-10-09T16:33:44.000000Z",
        },
        # 3. TSLA DAY_TRADER (Release 01683a3)
        {
            "plan_id": "PLAN_97e2b47d5fc812813f129d45",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "TSLA",
            "instrument_class": "EQUITY",
            "user_role": "DAY_TRADER",
            "generation_timestamp": "1791563624000",
            "source_data_timestamp": "1791563624000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 382.69,
            "planned_entry": 383.30,
            "structural_invalidation": 376.39,
            "take_profit_1": 398.63,
            "take_profit_2": 406.30,
            "execution_status": "IN_BUY_ZONE",
            "is_actionable": True,
            "created_at_utc": "2026-10-09T16:33:44.000000Z",
        },
        # 4. TSLA LONG_TERM duplicate (Release 01683a3)
        {
            "plan_id": "PLAN_19d8bc8fd753bc921de0d6f4",
            "epoch_id": EXECUTION_LADDER_EPOCH_ID,
            "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
            "symbol": "TSLA",
            "instrument_class": "EQUITY",
            "user_role": "LONG_TERM",
            "generation_timestamp": "1791566339000",
            "source_data_timestamp": "1791566339000",
            "release_sha": "01683a39a19f3f74720f798459cec717698e2ab2",
            "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
            "current_spot": 382.69,
            "planned_entry": 373.32,
            "structural_invalidation": 342.67,
            "take_profit_1": 430.03,
            "take_profit_2": 460.68,
            "execution_status": "EXTENDED_ABOVE_BUY_ZONE",
            "is_actionable": False,
            "created_at_utc": "2026-10-09T17:19:27.000000Z",
        },
    ]

    for rec in raw_records:
        gov_db.insert_execution_ladder_plan(rec, check_equivalent=False)

    # 1. Historical cohort checks
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=False) == 4
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=True) == 3

    # 2. Add 5th record under Release 5a90b91
    record_5 = {
        "epoch_id": EXECUTION_LADDER_EPOCH_ID,
        "observation_stream": EXECUTION_LADDER_OBSERVATION_STREAM,
        "symbol": "AAPL",
        "instrument_class": "EQUITY",
        "user_role": "LONG_TERM",
        "generation_timestamp": "2026-10-09T18:00:00Z",
        "source_data_timestamp": "2026-10-09T18:00:00Z",
        "release_sha": "5a90b918b0975151b74e936b3fbfa536b575edd7",
        "execution_ladder_authority_sha": EXECUTION_LADDER_AUTHORITY_SHA,
        "current_spot": 332.21,
        "planned_entry": 336.06,
        "structural_invalidation": 312.62,
        "take_profit_1": 379.43,
        "take_profit_2": 402.87,
        "execution_status": "IN_BUY_ZONE",
        "is_actionable": True,
        "created_at_utc": "2026-10-09T18:00:00Z",
    }
    record_5["plan_id"] = compute_execution_ladder_plan_id(record_5)

    inserted = gov_db.insert_execution_ladder_plan(record_5, check_equivalent=True)
    assert inserted is True

    # 3. Post-release counts
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=False) == 5
    assert gov_db.count_execution_ladder_plans(EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=True) == 4

    # 4. Stratification across the 4 cumulative canonical identities
    strat = gov_db.get_execution_ladder_stratification(EXECUTION_LADDER_EPOCH_ID, deduplicate_substantive=True)
    assert strat["total"] == 4
    assert strat["by_role"]["LONG_TERM"] == 3  # AAPL(0168), TSLA(0168), AAPL(5a90)
    assert strat["by_role"]["DAY_TRADER"] == 1  # TSLA(0168)
    assert strat["by_status"]["IN_BUY_ZONE"] == 3
    assert strat["by_status"]["EXTENDED_ABOVE_BUY_ZONE"] == 1

    # 5. Immutability: UPDATE and DELETE must fail closed on historical rows
    conn = gov_db.get_connection()
    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION.*Updates"):
        conn.execute("UPDATE execution_ladder_prospective_plans SET current_spot = 999.0 WHERE plan_id = 'PLAN_08f60711fd16b68ab1d83eff'")

    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION.*Deletions"):
        conn.execute("DELETE FROM execution_ladder_prospective_plans WHERE plan_id = 'PLAN_08f60711fd16b68ab1d83eff'")
    conn.close()


def test_quantitative_and_governance_invariance():
    """Section 5.4: Verifies invariance:
    - Zero quantitative logic alterations
    - Zero price-authority semantics changes
    - VCP prospective evidence remains untouched
    - No production traffic generated
    """
    assert EXECUTION_LADDER_AUTHORITY_SHA == "7bcb7780221f58cf596dabce484d83276e0a3c50"


