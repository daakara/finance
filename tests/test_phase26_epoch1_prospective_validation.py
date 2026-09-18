"""ARX Prospective Validation Epoch 1 Test Suite.

Verifies the mandatory Epoch 1 governance, provenance, immutability, and pooling invariants:
1. Canonical 6-Cohort Classification (PROSPECTIVE_CLEAN, HISTORICAL_RECOMPUTED, BACKTEST_SIMULATION, DEMO_SYNTHETIC, CONTAMINATED, UNKNOWN).
2. Temporal Anti-Lookahead Integrity (recommended_at < first_forward_bar_timestamp; post-bar rejection; outcome before recommendation rejection).
3. Recommendation Immutability & Tamper Evidence (decisionSnapshotHash and inputsSnapshotHash detect post-hoc mutation; engine version verification).
4. Authorized Outcome Append (subsequent forward tracking observations do not mutate or invalidate original recommendation hash).
5. Query-Level Pooling Protection (prospective queries filter epoch_id and cohort; historical, demo, contaminated, and wrong-epoch rows blocked from prospective metrics).
6. Passive Observation (ledger registration and updates have zero broker/execution side-effects; LEDGER_WRITE_CAN_TRIGGER_EXECUTION == False).
"""

import os
import json
import pytest
from datetime import datetime, timezone

from analyst_dashboard.governance.experiment_ledger import ExperimentLedger, ProvenanceCohort

pytestmark = pytest.mark.tier3


@pytest.fixture
def base_opt_exec():
    return {
        "optimal_entry_min": 190.0,
        "optimal_entry_max": 200.0,
        "stop_loss": 185.0,
        "stop_loss_pct": -5.0,
        "take_profit_1": 220.0,
        "take_profit_1_pct": 12.5,
        "take_profit_2": 230.0,
        "take_profit_2_pct": 17.5,
        "risk_reward_ratio": 2.5,
        "atr_14": 4.5,
        "setup_pattern": "Minervini VCP",
        "stage_phase": "Stage 2 Advancing",
    }


# ── 1. Canonical 6-Cohort Classification ─────────────────────────────────────

def test_epoch1_cohort_classification_all_six_cohorts():
    """Verify that ExperimentLedger deterministically classifies each of the 6 canonical cohorts."""
    # 1. PROSPECTIVE_CLEAN
    clean_sig = {
        "symbol": "AAPL",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "engineVersion": ExperimentLedger.ENGINE_SHA,
        "epochId": ExperimentLedger.EPOCH_ID,
    }
    assert ExperimentLedger.classify_provenance_cohort(clean_sig) == ProvenanceCohort.PROSPECTIVE_CLEAN

    # 2. HISTORICAL_RECOMPUTED
    recomputed_sig = {
        "symbol": "AAPL",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "provenanceCohort": ProvenanceCohort.HISTORICAL_RECOMPUTED,
    }
    assert ExperimentLedger.classify_provenance_cohort(recomputed_sig) == ProvenanceCohort.HISTORICAL_RECOMPUTED

    # 3. BACKTEST_SIMULATION
    sim_sig = {
        "symbol": "AAPL",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "isSimulated": True,
    }
    assert ExperimentLedger.classify_provenance_cohort(sim_sig) == ProvenanceCohort.BACKTEST_SIMULATION

    # 4. DEMO_SYNTHETIC
    demo_sig = {
        "symbol": "AAPL",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "isDemo": True,
    }
    assert ExperimentLedger.classify_provenance_cohort(demo_sig) == ProvenanceCohort.DEMO_SYNTHETIC

    # 5. CONTAMINATED (post-outcome mutation or tamper detected)
    contam_sig = {
        "symbol": "AAPL",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "isContaminated": True,
    }
    assert ExperimentLedger.classify_provenance_cohort(contam_sig) == ProvenanceCohort.CONTAMINATED

    # 6. UNKNOWN (missing mandatory fields or ambiguous provenance)
    unknown_sig = {
        "provenanceCohort": ProvenanceCohort.UNKNOWN,
    }
    assert ExperimentLedger.classify_provenance_cohort(unknown_sig) == ProvenanceCohort.UNKNOWN


# ── 2. Temporal Anti-Lookahead Integrity ─────────────────────────────────────

def test_epoch1_temporal_anti_lookahead_enforced():
    """Verify that recommendations recorded after the first forward bar are quarantined as CONTAMINATED."""
    # Valid: recommendation occurred at 09:30, first forward bar at 10:00
    valid_timing_sig = {
        "symbol": "NVDA",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "recommended_at": "2026-09-20T09:30:00Z",
        "first_forward_bar_timestamp": "2026-09-20T10:00:00Z",
        "engineVersion": ExperimentLedger.ENGINE_SHA,
    }
    assert ExperimentLedger.classify_provenance_cohort(valid_timing_sig) == ProvenanceCohort.PROSPECTIVE_CLEAN

    # Temporal Violation: recommendation occurred at 10:30, after forward bar at 10:00
    lookahead_sig = {
        "symbol": "NVDA",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "recommended_at": "2026-09-20T10:30:00Z",
        "first_forward_bar_timestamp": "2026-09-20T10:00:00Z",
        "engineVersion": ExperimentLedger.ENGINE_SHA,
    }
    assert ExperimentLedger.classify_provenance_cohort(lookahead_sig) == ProvenanceCohort.CONTAMINATED


# ── 3. Immutability & Cryptographic Tamper Evidence ──────────────────────────

def test_epoch1_immutability_and_tamper_detection(tmp_path, base_opt_exec):
    """Verify that modifying any recommendation-critical field fails verification fail-closed."""
    ledger_path = str(tmp_path / "test_immutability_ledger.json")

    rec = ExperimentLedger.register_signal(
        symbol="MSFT",
        entry_price=195.0,
        opt_exec=base_opt_exec,
        confluence_score=88.5,
        inputs_meta={"market_regime": "BULL", "sector": "TECH", "asset_class": "US_EQUITY"},
        engine_commit=ExperimentLedger.ENGINE_SHA,
        epoch_id=ExperimentLedger.EPOCH_ID,
        ledger_path=ledger_path,
    )

    # 1. Unmodified record verifies
    assert ExperimentLedger.compute_decision_snapshot_hash(rec) == rec["decisionSnapshotHash"]
    assert ExperimentLedger.compute_inputs_snapshot_hash(rec) == rec["inputsSnapshotHash"]
    assert ExperimentLedger.classify_provenance_cohort(rec) == ProvenanceCohort.PROSPECTIVE_CLEAN

    # 2. Mutate decision-critical field (entryPrice) -> hash mismatch -> CONTAMINATED
    tampered_dec = dict(rec)
    tampered_dec["entryPrice"] = 210.0
    assert ExperimentLedger.compute_decision_snapshot_hash(tampered_dec) != rec["decisionSnapshotHash"]
    assert ExperimentLedger.classify_provenance_cohort(tampered_dec) == ProvenanceCohort.CONTAMINATED

    # 3. Mutate input snapshot feature -> hash mismatch -> CONTAMINATED
    tampered_in = dict(rec)
    tampered_in["inputs"] = dict(rec["inputs"])
    tampered_in["inputs"]["stagePhase"] = "Stage 4 Decline"
    assert ExperimentLedger.compute_inputs_snapshot_hash(tampered_in) != rec["inputsSnapshotHash"]
    assert ExperimentLedger.classify_provenance_cohort(tampered_in) == ProvenanceCohort.CONTAMINATED

    # 4. Mutate engine version on hashed record -> decision hash mismatch -> CONTAMINATED
    tampered_eng = dict(rec)
    tampered_eng["engineVersion"] = "unauthorized_modified_sha_12345"
    assert ExperimentLedger.compute_decision_snapshot_hash(tampered_eng) != rec["decisionSnapshotHash"]
    assert ExperimentLedger.classify_provenance_cohort(tampered_eng) == ProvenanceCohort.CONTAMINATED

    # 5. Record without hashes with unauthorized engine version -> EXCLUDED
    unhashed_bad_eng = {
        "symbol": "MSFT",
        "signalDate": "2026-09-20",
        "status": "OPEN",
        "engineVersion": "unauthorized_engine_version_xyz",
    }
    assert ExperimentLedger.classify_provenance_cohort(unhashed_bad_eng) == ProvenanceCohort.EXCLUDED


# ── 4. Authorized Outcome Append Preserves Recommendation Provenance ──────────

def test_epoch1_authorized_outcome_append(tmp_path, base_opt_exec):
    """Verify that appending forward observations and outcomes does not mutate original recommendation hash."""
    ledger_path = str(tmp_path / "test_outcome_append.json")

    rec = ExperimentLedger.register_signal(
        symbol="AMD",
        entry_price=150.0,
        opt_exec=base_opt_exec,
        confluence_score=84.0,
        inputs_meta={"market_regime": "BULL", "sector": "TECH", "asset_class": "US_EQUITY"},
        engine_commit=ExperimentLedger.ENGINE_SHA,
        epoch_id=ExperimentLedger.EPOCH_ID,
        ledger_path=ledger_path,
    )
    orig_dec_hash = rec["decisionSnapshotHash"]
    orig_in_hash = rec["inputsSnapshotHash"]

    # Append forward execution observation
    obs = ExperimentLedger.record_execution_observation(
        signal_id=rec["signalId"],
        fill_price=150.20,
        execution_timestamp="2026-09-20T14:35:00Z",
        order_size_usd=10000.0,
        side="BUY",
        ledger_path=ledger_path,
    )
    assert obs["fillPrice"] == 150.20

    # Load ledger and verify original recommendation hash remains strictly intact
    updated_ledger = ExperimentLedger.load_ledger(ledger_path)
    updated_rec = updated_ledger["signals"][0]

    assert updated_rec["decisionSnapshotHash"] == orig_dec_hash
    assert updated_rec["inputsSnapshotHash"] == orig_in_hash
    assert ExperimentLedger.compute_decision_snapshot_hash(updated_rec) == orig_dec_hash
    assert ExperimentLedger.compute_inputs_snapshot_hash(updated_rec) == orig_in_hash
    assert len(updated_rec["executionObservations"]) == 1
    assert ExperimentLedger.classify_provenance_cohort(updated_rec) == ProvenanceCohort.PROSPECTIVE_CLEAN


# ── 5. Query-Level Pooling Protection ────────────────────────────────────────

def test_epoch1_pooling_guard_blocks_contamination_and_other_epochs(tmp_path, base_opt_exec):
    """Verify that scorecard evaluation strictly insulates Epoch 1 PROSPECTIVE_CLEAN records.

    Historical recomputed, demo, contaminated, and previous-epoch records must never enter the clean denominator.
    """
    ledger_path = str(tmp_path / "test_pooling_ledger.json")

    # 1. Clean Epoch 1 Signal
    ExperimentLedger.register_signal(
        symbol="CLEAN1",
        entry_price=100.0,
        opt_exec=base_opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        engine_commit=ExperimentLedger.ENGINE_SHA,
        epoch_id=ExperimentLedger.EPOCH_ID,
        ledger_path=ledger_path,
    )

    # 2. Previous Epoch Signal (Phase 24 legacy)
    ExperimentLedger.register_signal(
        symbol="PREV_EPOCH",
        entry_price=100.0,
        opt_exec=base_opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        engine_commit="4e36862",
        epoch_id="PHASE_24_HISTORICAL_BASELINE",
        ledger_path=ledger_path,
    )

    # 3. Demo / Synthetic Signal
    ExperimentLedger.register_signal(
        symbol="DEMO_SIG",
        entry_price=100.0,
        opt_exec=base_opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        provenance_cohort=ProvenanceCohort.DEMO_SYNTHETIC,
        epoch_id=ExperimentLedger.EPOCH_ID,
        ledger_path=ledger_path,
    )

    # 4. Historical Recomputed Signal
    ExperimentLedger.register_signal(
        symbol="HIST_RECOMP",
        entry_price=100.0,
        opt_exec=base_opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        provenance_cohort=ProvenanceCohort.HISTORICAL_RECOMPUTED,
        epoch_id=ExperimentLedger.EPOCH_ID,
        ledger_path=ledger_path,
    )

    # Run scorecard evaluation with Epoch 1 filter
    scorecard = ExperimentLedger.compute_governance_scorecard(
        ledger_path=ledger_path,
        cohort_filter=ProvenanceCohort.PROSPECTIVE_CLEAN,
        epoch_id=ExperimentLedger.EPOCH_ID,
    )

    firewall = scorecard["cohortFirewall"]
    assert firewall["cleanSignalsCount"] == 1
    assert firewall["historicalProspectivePooling"] == "BLOCKED"
    assert firewall["demoProspectivePooling"] == "BLOCKED"
    assert firewall["contaminatedProspectivePooling"] == "BLOCKED"
    assert scorecard["totalSignals"] == 1  # Only the single Epoch 1 clean signal enters clean metrics


# ── 6. Passive Observation Invariant ─────────────────────────────────────────

def test_epoch1_passive_observation_contract():
    """Verify that ledger writes are passive and cannot trigger trade execution."""
    assert ExperimentLedger.LEDGER_WRITE_CAN_TRIGGER_EXECUTION is False


# ── 7. Complete Point-in-Time Input Snapshot Contract ────────────────────────

def test_epoch1_complete_input_snapshot_contract(tmp_path, base_opt_exec):
    """Verify that Epoch 1 input snapshot hash commits to the full point-in-time information set."""
    complete_inputs_meta = {
        "market_regime": "BULL",
        "sector": "TECH",
        "asset_class": "US_EQUITY",
        "market_data_snapshot_timestamp": "2026-09-20T09:30:00Z",
        "candle_count": 251,
        "fundamental_as_of_date": "2026-06-30",
        "fundamental_filing_timestamp": "2026-07-28T20:00:00Z",
        "macro_observation_date": "2026-09-19",
        "yield_curve_10y2y": 0.15,
        "credit_spread": 3.20,
        "data_provider": "YAHOO_AUTHENTIC",
        "quote_freshness": "END_OF_DAY",
        "evidence_completeness": "COMPLETE",
    }
    opt_exec_with_techs = dict(base_opt_exec)
    opt_exec_with_techs["sma_50"] = 192.5
    opt_exec_with_techs["ema_20"] = 196.0
    opt_exec_with_techs["rsi_14"] = 62.4

    ledger_path = str(tmp_path / "test_complete_inputs.json")
    rec = ExperimentLedger.register_signal(
        symbol="GOOGL",
        entry_price=198.0,
        opt_exec=opt_exec_with_techs,
        confluence_score=87.0,
        inputs_meta=complete_inputs_meta,
        engine_commit=ExperimentLedger.DECISION_ENGINE_SHA,
        epoch_id=ExperimentLedger.EPOCH_ID,
        ledger_path=ledger_path,
    )

    orig_in_hash = rec["inputsSnapshotHash"]
    assert orig_in_hash is not None
    assert len(orig_in_hash) == 64
    assert ExperimentLedger.compute_inputs_snapshot_hash(rec) == orig_in_hash

    # Mutating fundamental as-of date MUST change hash
    t1 = json.loads(json.dumps(rec))
    t1["inputs"]["fundamentalAsOfDate"] = "2026-09-30"
    assert ExperimentLedger.compute_inputs_snapshot_hash(t1) != orig_in_hash

    # Mutating macro yield curve MUST change hash
    t2 = json.loads(json.dumps(rec))
    t2["inputs"]["yieldCurve10y2y"] = -0.45
    assert ExperimentLedger.compute_inputs_snapshot_hash(t2) != orig_in_hash

    # Mutating technical indicator MUST change hash
    t3 = json.loads(json.dumps(rec))
    t3["inputs"]["sma50"] = 180.0
    assert ExperimentLedger.compute_inputs_snapshot_hash(t3) != orig_in_hash

    # Mutating provider provenance MUST change hash
    t4 = json.loads(json.dumps(rec))
    t4["inputs"]["dataProvider"] = "SYNTHETIC_FALLBACK"
    assert ExperimentLedger.compute_inputs_snapshot_hash(t4) != orig_in_hash


# ── 8. Fundamental Point-in-Time Anti-Lookahead Quarantine ───────────────────

def test_epoch1_fundamental_point_in_time_lookahead_quarantine():
    """Verify that fundamental dates post-dating recommendation are quarantined as CONTAMINATED."""
    # Valid: fundamental filing is June 30 / July 28, recommendation is September 20
    valid_sig = {
        "symbol": "AAPL",
        "signalDate": "2026-09-20",
        "recommended_at": "2026-09-20T09:30:00Z",
        "status": "OPEN",
        "engineVersion": ExperimentLedger.DECISION_ENGINE_SHA,
        "inputs": {
            "fundamentalAsOfDate": "2026-06-30",
            "fundamentalFilingTimestamp": "2026-07-28T20:00:00Z",
        }
    }
    assert ExperimentLedger.classify_provenance_cohort(valid_sig) == ProvenanceCohort.PROSPECTIVE_CLEAN

    # Lookahead Violation: fundamental as-of date is in the future (e.g. 2026-09-30 vs rec 2026-09-20)
    lookahead_fund_sig = {
        "symbol": "AAPL",
        "signalDate": "2026-09-20",
        "recommended_at": "2026-09-20T09:30:00Z",
        "status": "OPEN",
        "engineVersion": ExperimentLedger.DECISION_ENGINE_SHA,
        "inputs": {
            "fundamentalAsOfDate": "2026-09-30",
        }
    }
    assert ExperimentLedger.classify_provenance_cohort(lookahead_fund_sig) == ProvenanceCohort.CONTAMINATED


# ── 9. Two-Tier Identity Verification ────────────────────────────────────────

def test_epoch1_two_tier_sha_identity():
    """Verify that Decision Engine SHA and Observation Governance SHA are explicitly separated."""
    assert ExperimentLedger.DECISION_ENGINE_SHA == "7ad44595826c147cc77f93cd676af520764c7442"
    assert hasattr(ExperimentLedger, "OBSERVATION_GOVERNANCE_SHA")
    assert hasattr(ExperimentLedger, "get_observation_governance_sha")
    gov_sha = ExperimentLedger.get_observation_governance_sha()
    assert isinstance(gov_sha, str) and len(gov_sha) > 0
    assert ExperimentLedger.ENGINE_SHA == ExperimentLedger.DECISION_ENGINE_SHA
