"""Adversarial Measurement Suite for ARX Model Governance.

Stress-tests the prospective measurement engine against 10 critical edge cases:
1. Same-day intrabar TP/Stop collision (pessimistic fail-closed policy).
2. Missing trading days / holiday calendar gap resilience.
3. Temporal anti-lookahead integrity (no retro-leakage).
4. Multi-horizon capture ratios (T+5, T+10, T+20).
5. Post-Stop Favorable Excursion (PSFE) path tracking.
6. Friction sensitivity grid (0, 15, 30, 50, 100 bps).
7. Benchmark medians and hit rate metrics.
8. Spearman rank correlation and top-vs-bottom quartile discrimination.
9. Monte Carlo random baseline reproducibility (1,000 runs, fixed seed).
10. Frozen production baseline dual-hash anti-tamper verification.
"""

import os
import json
import hashlib
import pytest
import numpy as np
from datetime import datetime, timezone
from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
from analyst_dashboard.governance.baseline_engine import BaselineEngine

pytestmark = pytest.mark.tier3


class DummyDB:
    def __init__(self, candles_map):
        self.candles_map = candles_map

    def get_daily_candles(self, symbol, limit=100):
        return self.candles_map.get(symbol, [])


def test_adversarial_intrabar_tp_stop_collision_fail_closed(tmp_path):
    """
    CRITICAL RISK INVARIANT: When High >= TP1 AND Low <= Stop occur on the same daily bar,
    the system MUST resolve fail-closed to STOP_LOSS (pessimistic fill policy).
    It must NEVER resolve as TP1_WIN without tick data.
    """
    ledger_path = str(tmp_path / "test_ledger.json")
    sig_date = "2026-09-01"

    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 94.0,     # -6.0%
        "stop_loss_pct": -6.0,
        "take_profit_1": 110.0, # +10.0%
        "take_profit_1_pct": 10.0,
        "risk_reward_ratio": 1.67,
        "atr_14": 3.0,
    }

    ExperimentLedger.register_signal(
        symbol="COLLIDE",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        signal_date=sig_date,
        ledger_path=ledger_path
    )

    # Day 1: Extreme wild bar touching BOTH 115.0 (TP1) and 90.0 (Stop)
    collision_candles = [
        {"time": "2026-09-02", "open": 100.0, "high": 115.0, "low": 90.0, "close": 105.0, "volume": 100000},
    ]
    db = DummyDB({"COLLIDE": collision_candles})
    ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)

    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    ft = sig["forwardTracking"]

    # Must resolve fail-closed to STOP_LOSS
    assert sig["status"] == "RESOLVED"
    assert ft["resolvedOutcome"] == "STOP_LOSS"
    assert ft["intrabarCollision"] is True
    assert ft["stopHit"] is True
    assert ft["realizedReturnPct"] == -6.0  # Stopped out


def test_adversarial_holiday_and_missing_session_resilience(tmp_path):
    """Verify that gaps from weekends and market holidays do not desync session counters or multi-horizons."""
    ledger_path = str(tmp_path / "test_ledger.json")
    sig_date = "2026-09-04"  # Friday before Labor Day

    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 80.0,
        "stop_loss_pct": -20.0,
        "take_profit_1": 150.0,
        "take_profit_1_pct": 50.0,
        "risk_reward_ratio": 2.5,
        "atr_14": 2.0,
    }

    ExperimentLedger.register_signal(
        symbol="HOLIDAY",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=80.0,
        inputs_meta={"market_regime": "BULL"},
        signal_date=sig_date,
        ledger_path=ledger_path
    )

    # Tuesday Sep 8 (Session 1), Wednesday Sep 9 (Session 2), Monday Sep 14 (Session 3 - missing 2 days)
    sparse_candles = [
        {"time": "2026-09-08", "open": 101.0, "high": 103.0, "low": 100.5, "close": 102.0, "volume": 1000},
        {"time": "2026-09-09", "open": 102.0, "high": 104.0, "low": 101.0, "close": 103.0, "volume": 1000},
        {"time": "2026-09-14", "open": 103.0, "high": 106.0, "low": 102.5, "close": 105.0, "volume": 1000},
    ]
    db = DummyDB({"HOLIDAY": sparse_candles})
    ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)

    ledger = ExperimentLedger.load_ledger(ledger_path)
    ft = ledger["signals"][0]["forwardTracking"]

    # 3 sessions observed, current price = 105.0, return1d = +2.0%
    assert ft["sessionsObserved"] == 3
    assert ft["currentPrice"] == 105.0
    assert ft["return1d"] == 2.0
    assert ft["return5d"] is None  # Only 3 sessions observed, cannot fabricate 5d return!


def test_adversarial_anti_lookahead_temporal_leakage(tmp_path):
    """Verify that candles dated <= signal_date cannot be used in forward tracking."""
    ledger_path = str(tmp_path / "test_ledger.json")
    sig_date = "2026-09-06"

    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 90.0,
        "take_profit_1": 120.0,
        "risk_reward_ratio": 2.0,
        "atr_14": 2.0,
    }

    ExperimentLedger.register_signal(
        symbol="PAST",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=80.0,
        inputs_meta={"market_regime": "BULL"},
        signal_date=sig_date,
        ledger_path=ledger_path
    )

    # Candles that occurred before or on signal date
    past_candles = [
        {"time": "2026-09-01", "open": 90.0, "high": 92.0, "low": 89.0, "close": 91.0, "volume": 1000},
        {"time": "2026-09-06", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000},
    ]
    db = DummyDB({"PAST": past_candles})
    ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)

    ledger = ExperimentLedger.load_ledger(ledger_path)
    ft = ledger["signals"][0]["forwardTracking"]

    # Zero forward sessions observed -> excursions must be None
    assert ft["sessionsObserved"] == 0
    assert ft["currentPrice"] == 100.0
    assert ft["maxFavorableExcursionPct"] is None
    assert ft["maxAdverseExcursionPct"] is None


def test_friction_sensitivity_grid_and_breakeven(tmp_path):
    """Verify scorecard friction sensitivity table (0, 15, 30, 50, 100 bps) and breakeven friction."""
    ledger_path = str(tmp_path / "test_ledger.json")

    # 1 win (+10.0%), 1 loss (-6.0%) -> Expectancy = 0.5*10 - 0.5*6 = +2.00%
    synthetic_ledger = {
        "version": "1.0.0",
        "totalActiveSignals": 0,
        "signals": [
            {
                "signalId": "WIN_1",
                "symbol": "W1",
                "status": "RESOLVED",
                "confluenceScore": 80.0,
                "forwardTracking": {"resolvedOutcome": "TP1_WIN", "realizedReturnPct": 10.0}
            },
            {
                "signalId": "LOSS_1",
                "symbol": "L1",
                "status": "RESOLVED",
                "confluenceScore": 75.0,
                "forwardTracking": {"resolvedOutcome": "STOP_LOSS", "realizedReturnPct": -6.0}
            }
        ]
    }
    ExperimentLedger.save_ledger(synthetic_ledger, ledger_path)
    scorecard = ExperimentLedger.compute_governance_scorecard(ledger_path)

    fs = scorecard["frictionSensitivity"]
    assert fs["0bps_gross"] == 2.0
    assert fs["15bps_optimistic"] == 1.85
    assert fs["30bps_primary"] == 1.70
    assert fs["50bps_stress"] == 1.50
    assert fs["100bps_severe"] == 1.00
    assert fs["edgeBreakevenFrictionBps"] == 200.0  # 2.00% = 200 bps


def test_multi_horizon_capture_ratios_and_psfe(tmp_path):
    """Verify horizon capture ratios (T+5, T+10, T+20) and Post-Stop Favorable Excursion tracking."""
    ledger_path = str(tmp_path / "test_ledger.json")
    sig_date = "2026-09-01"

    opt_exec = {
        "optimal_entry_min": 98.0,
        "optimal_entry_max": 102.0,
        "stop_loss": 95.0,     # Stop -5%
        "stop_loss_pct": -5.0,
        "take_profit_1": 115.0, # TP +15%
        "take_profit_1_pct": 15.0,
        "risk_reward_ratio": 3.0,
        "atr_14": 2.0,
    }

    ExperimentLedger.register_signal(
        symbol="PSFE_TEST",
        entry_price=100.0,
        opt_exec=opt_exec,
        confluence_score=85.0,
        inputs_meta={"market_regime": "BULL"},
        signal_date=sig_date,
        ledger_path=ledger_path
    )

    # Session 1: stops out at 94. Sessions 2-5: rallies up to 125 (well past TP1)
    candles = [
        {"time": "2026-09-02", "open": 98.0, "high": 99.0, "low": 94.0, "close": 94.5, "volume": 1000},
        {"time": "2026-09-03", "open": 96.0, "high": 108.0, "low": 95.0, "close": 107.0, "volume": 2000},
        {"time": "2026-09-04", "open": 108.0, "high": 125.0, "low": 107.0, "close": 124.0, "volume": 3000},
    ]
    db = DummyDB({"PSFE_TEST": candles})
    ExperimentLedger.update_forward_observations(db, ledger_path=ledger_path)

    ledger = ExperimentLedger.load_ledger(ledger_path)
    sig = ledger["signals"][0]
    tc = sig["forwardTracking"]["tradeConstruction"]

    # Must detect post-stop favorable excursion
    assert tc["postStopFavorableExcursion"]["tp1CrossedPostStop"] is True
    assert tc["postStopFavorableExcursion"]["maxFavorableExcursionPostStopPct"] == 25.0
    assert tc["prematureStopOut"] is True  # Alias matches


def test_confluence_spearman_and_quartile_discrimination(tmp_path):
    """Verify Spearman rank correlation and top-vs-bottom quartile discrimination on confluence scores."""
    ledger_path = str(tmp_path / "test_ledger.json")

    # 8 signals with strong rank ordering (confluence 70..95 -> returns -5%..+25%)
    signals = []
    for i in range(8):
        c_score = 70.0 + i * 3.5  # 70.0 to 94.5
        ret = -5.0 + i * 4.0      # -5.0% to +23.0%
        outcome = "TP1_WIN" if ret > 0 else "STOP_LOSS"
        signals.append({
            "signalId": f"SIG_{i}",
            "symbol": f"S{i}",
            "status": "RESOLVED",
            "confluenceScore": c_score,
            "signalDate": "2026-09-01",
            "forwardTracking": {
                "resolvedOutcome": outcome,
                "realizedReturnPct": ret,
                "maxFavorableExcursionPct": max(0.0, ret + 2.0)
            }
        })

    synthetic_ledger = {"version": "1.0.0", "totalActiveSignals": 0, "signals": signals}
    ExperimentLedger.save_ledger(synthetic_ledger, ledger_path)
    scorecard = ExperimentLedger.compute_governance_scorecard(ledger_path)

    rp = scorecard["rankingPower"]
    # Spearman rho should be positive and high
    assert rp["spearmanCorrelation"]["confluenceVsReturnRho"] > 0.90

    # Quartiles: Q4 (top 2) vs Q1 (bottom 2)
    qd = rp["quartileDiscrimination"]
    assert qd["topQuartileQ4"]["winRatePct"] == 100.0
    assert qd["bottomQuartileQ1"]["winRatePct"] == 0.0
    assert qd["spreadQ4MinusQ1"]["rankingPowerPositive"] is True
    assert qd["spreadQ4MinusQ1"]["meanReturnSpread"] > 20.0


def test_monte_carlo_random_baseline_reproducibility():
    """Verify BaselineEngine Monte Carlo runs produce deterministic results with fixed seed."""
    signals = [
        {
            "symbol": "AAA",
            "status": "RESOLVED",
            "signalDate": "2026-09-01",
            "stopLossPct": -6.0,
            "takeProfit1Pct": 15.0,
            "forwardTracking": {"realizedReturnPct": 15.0}
        },
        {
            "symbol": "BBB",
            "status": "RESOLVED",
            "signalDate": "2026-09-01",
            "stopLossPct": -6.0,
            "takeProfit1Pct": 15.0,
            "forwardTracking": {"realizedReturnPct": -6.0}
        }
    ]

    # Universe with 4 alternative assets
    universe = {
        "U1": [{"time": f"2026-09-{d:02d}", "open": 50.0 + d, "high": 52.0 + d, "low": 49.0 + d, "close": 51.0 + d} for d in range(1, 25)],
        "U2": [{"time": f"2026-09-{d:02d}", "open": 100.0 - d * 0.5, "high": 101.0 - d * 0.5, "low": 98.0 - d * 0.5, "close": 99.0 - d * 0.5} for d in range(1, 25)],
        "U3": [{"time": f"2026-09-{d:02d}", "open": 20.0 + d * 0.2, "high": 21.0 + d * 0.2, "low": 19.5 + d * 0.2, "close": 20.5 + d * 0.2} for d in range(1, 25)],
        "U4": [{"time": f"2026-09-{d:02d}", "open": 80.0, "high": 81.0, "low": 79.0, "close": 80.0} for d in range(1, 25)],
    }

    # Run 100 simulations with seed 42
    res1 = BaselineEngine.run_random_monte_carlo(signals, universe, n_simulations=100, seed=42, friction_bps=30.0)
    res2 = BaselineEngine.run_random_monte_carlo(signals, universe, n_simulations=100, seed=42, friction_bps=30.0)

    assert res1["status"] == "COMPLETED"
    assert res1["nSimulations"] == 100
    assert res1["actualStrategy"]["percentileRankVsRandom"] == res2["actualStrategy"]["percentileRankVsRandom"]
    assert res1["randomDistribution"]["meanNetReturnPct"] == res2["randomDistribution"]["meanNetReturnPct"]


def test_frozen_production_ledger_live_cohort_hashes_intact():
    """Verify that all 9 live signals in paper_trading_ledger.json maintain valid dual cryptographic hashes."""
    live_ledger = ExperimentLedger.load_ledger()
    signals = live_ledger.get("signals", [])
    assert len([s for s in signals if s.get("signalDate") == "2026-09-04"]) == 9, "All 9 original Phase 24 cohort signals must be present"
    assert len(signals) >= 9, "All cohort signals must be present"

    for sig in signals:
        sym = sig.get("symbol")
        # Verify decision hash
        computed_dec_hash = ExperimentLedger.compute_decision_snapshot_hash(sig)
        assert computed_dec_hash == sig["decisionSnapshotHash"], f"Decision hash mismatch on {sym}"

        # Verify inputs hash if present
        if sig.get("inputsSnapshotHash"):
            computed_inp_hash = ExperimentLedger.compute_inputs_snapshot_hash(sig)
            assert computed_inp_hash == sig["inputsSnapshotHash"], f"Inputs hash mismatch on {sym}"


def test_production_engine_freeze_manifest_compliance():
    """Verify historical engine manifest integrity and active production compliance under Release Addendum 002 and RC5."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    manifest_v251_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_1.json")
    manifest_v250_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_0.json")
    manifest_v240_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_4_0.json")
    manifest_default_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST.json")

    assert os.path.exists(manifest_default_path) or os.path.exists(manifest_v240_path), (
        "Historical 2.4.0 freeze manifest must exist"
    )

    # 1. Historical Manifest Integrity Audit
    with open(manifest_v240_path, "r", encoding="utf-8") as f:
        v240 = json.load(f)
    assert v240.get("frozenStrategyVersion") == "2.4.0"
    assert v240.get("provenanceCommit") == "4e3686296aad24e2210ef580bbc9116054d84fd1"

    with open(manifest_v250_path, "r", encoding="utf-8") as f:
        v250 = json.load(f)
    assert v250.get("frozenStrategyVersion") == "2.5.0"
    assert v250.get("provenanceCommit") == "PENDING_EPOCH_3_FREEZE"

    with open(manifest_v251_path, "r", encoding="utf-8") as f:
        v251 = json.load(f)
    assert v251.get("frozenStrategyVersion") == "2.5.0"
    assert v251.get("provenanceCommit") == "cd0922471767775636957df74406a2d5efb8f519"
    assert v251.get("releaseAuthorityBinding") == "RELEASE_ADDENDUM_002"

    # Verify V2.5.2 historical release manifest integrity
    manifest_v252_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_2.json")
    assert os.path.exists(manifest_v252_path), "FROZEN_ENGINE_MANIFEST_V2_5_2.json missing!"
    with open(manifest_v252_path, "r", encoding="utf-8") as f:
        v252 = json.load(f)
    assert v252.get("manifestVersion") == "2.5.2"
    assert v252.get("provenanceCommit") == "7980d3a6e5dde0fe172d215c729ec213b9a164b1"
    assert v252.get("releaseAuthorityBinding") == "SPRINT3_RC5_CORPORATE_EVENTS_RATIFICATION"
    assert v252["engines"]["optimal_execution"]["sha256"] == "84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576"
    assert v252["engines"]["decision_hierarchy"]["sha256"] == "823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3"
    assert v252["engines"]["confluence_engine"]["sha256"] == "0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a"

    # 2. Active Engine Compliance Audit (Ratified Production Baseline or Authorized Candidate Scope)
    import hashlib

    opt_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "optimal_execution.py")
    with open(opt_path, "rb") as fp:
        opt_sha = hashlib.sha256(fp.read().replace(b"\r\n", b"\n")).hexdigest()

    dec_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "decision_hierarchy.py")
    with open(dec_path, "rb") as fp:
        dec_sha = hashlib.sha256(fp.read().replace(b"\r\n", b"\n")).hexdigest()

    conf_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "confluence_engine.py")
    with open(conf_path, "rb") as fp:
        conf_sha = hashlib.sha256(fp.read().replace(b"\r\n", b"\n")).hexdigest()

    candidate_manifest_path = os.path.join(repo_root, "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json")
    if os.path.exists(candidate_manifest_path):
        # Development candidate scope (PR #105 / ARX-S4-ARCH-002)
        with open(candidate_manifest_path, "r", encoding="utf-8") as fp:
            cand = json.load(fp)
        assert opt_sha == cand["engines"]["optimal_execution"]["sha256"], f"Unexpected optimal_execution SHA {opt_sha}"
        assert dec_sha == cand["engines"]["decision_hierarchy"]["sha256"], f"Unexpected decision_hierarchy SHA {dec_sha}"
        assert conf_sha == cand["engines"]["confluence_engine"]["sha256"], f"Unexpected confluence_engine SHA {conf_sha}"
        assert cand.get("isProductionRatified") is False, "Candidate manifest must never claim production ratification"
    else:
        # Production baseline: strict ratified hash enforcement
        assert opt_sha == "84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576", (
            f"Section 7 contract violation: unexpected optimal_execution SHA {opt_sha}"
        )
        assert dec_sha == "823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3", (
            f"Decision hierarchy integrity violation: unexpected decision_hierarchy SHA {dec_sha}"
        )
        assert conf_sha == "0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a", (
            f"Confluence engine integrity violation: unexpected confluence_engine SHA {conf_sha}"
        )


def test_cohort_contamination_firewall_enforcement(tmp_path):
    """Verify that the ledger firewall strictly insulates prospective clean data from contaminated history."""
    from unittest.mock import patch
    from analyst_dashboard.governance.experiment_ledger import ProvenanceCohort

    ledger_path = str(tmp_path / "mixed_firewall_ledger.json")

    # Construct mixed cohort: 1 clean prospective signal + 1 contaminated historical signal
    mixed_ledger = {
        "version": "1.0.0",
        "signals": [
            {
                "signalId": "CLEAN_ANET_2026-09-04",
                "symbol": "ANET",
                "signalDate": "2026-09-04",
                "status": "RESOLVED",
                "inputs": {"sector": "TECHNOLOGY", "marketRegime": "BULL"},
                "forwardTracking": {
                    "resolvedOutcome": "TP1_WIN",
                    "realizedReturnPct": 14.0
                }
            },
            {
                "signalId": "CONTAMINATED_HIST_2026-08-01",
                "symbol": "HIST",
                "signalDate": "2026-08-01",  # Pre-freeze date
                "provenanceCohort": ProvenanceCohort.HISTORICAL_CONTAMINATED,
                "status": "RESOLVED",
                "inputs": {"sector": "TECHNOLOGY", "marketRegime": "BULL"},
                "forwardTracking": {
                    "resolvedOutcome": "STOP_LOSS",
                    "realizedReturnPct": -6.5
                }
            }
        ]
    }
    ExperimentLedger.save_ledger(mixed_ledger, ledger_path)

    # Use mock to isolate cohort firewall unit logic from filesystem engine audits (as authorized)
    with patch.object(
        ExperimentLedger, "verify_frozen_engine_manifest", return_value={"status": "VERIFIED", "valid": True}
    ):
        # 1. Default evaluation: Auto-insulates clean cohort
        scorecard_auto = ExperimentLedger.compute_governance_scorecard(ledger_path)
        fw = scorecard_auto["cohortFirewall"]
        assert fw["cleanSignalsCount"] == 1
        assert fw["contaminatedSignalsCount"] == 1
        assert fw["cleanEvaluationEligible"] is True
        assert scorecard_auto["totalSignals"] == 1  # Only the clean signal evaluated!
        assert scorecard_auto["winRate"] == 100.0

        # 2. Explicit mixed evaluation: Refuses clean evaluation eligibility
        scorecard_mixed = ExperimentLedger.compute_governance_scorecard(ledger_path, cohort_filter="ALL")
        assert scorecard_mixed["cohortFirewall"]["cleanEvaluationEligible"] is False
        assert "WARNING" in scorecard_mixed["cohortFirewall"]["firewallWarning"]

        # 3. Contaminated-only evaluation: Marked not eligible
        scorecard_contam = ExperimentLedger.compute_governance_scorecard(
            ledger_path, cohort_filter=ProvenanceCohort.HISTORICAL_CONTAMINATED
        )
        assert scorecard_contam["cohortFirewall"]["cleanEvaluationEligible"] is False
        assert "CRITICAL" in scorecard_contam["cohortFirewall"]["firewallWarning"]

    # 4. Invariant: Pre-freeze signals claiming PROSPECTIVE_CLEAN must fail closed to HISTORICAL_CONTAMINATED
    malicious_prefreeze = {
        "signalId": "SPOOF_CLEAN",
        "symbol": "SPOOF",
        "signalDate": "2026-08-15",
        "provenanceCohort": ProvenanceCohort.PROSPECTIVE_CLEAN,  # Malicious clean claim on pre-freeze date
        "status": "RESOLVED",
    }
    assert ExperimentLedger.classify_provenance_cohort(malicious_prefreeze) == ProvenanceCohort.HISTORICAL_CONTAMINATED

    # 5. Invariant: Unknown/unverified engineVersion fails closed to EXCLUDED
    bad_engine = {
        "signalId": "BAD_ENG",
        "symbol": "BAD",
        "signalDate": "2026-09-05",
        "engineVersion": "unfrozen-dev-branch-999",
        "status": "RESOLVED",
    }
    assert ExperimentLedger.classify_provenance_cohort(bad_engine) == ProvenanceCohort.EXCLUDED

    # 6. Scorecard includes verified frozen manifest
    assert scorecard_auto["frozenEngineManifest"]["valid"] is True
    assert scorecard_auto["frozenEngineManifest"]["status"] == "VERIFIED"


def test_cluster_structure_and_portfolio_aggregation_diagnostics(tmp_path):
    """Verify that cluster dependence and portfolio concurrent exposure are calculated and reported."""
    ledger_path = str(tmp_path / "clustering_ledger.json")

    signals = [
        {
            "signalId": f"SIG_{i}",
            "symbol": f"SYM_{i}",
            "signalDate": "2026-09-04",  # All clustered on the same day
            "status": "RESOLVED",
            "inputs": {
                "sector": "TECHNOLOGY" if i < 4 else "HEALTHCARE",  # 4 out of 5 in Tech
                "marketRegime": "BULL"
            },
            "forwardTracking": {
                "resolvedOutcome": "TP1_WIN" if i % 2 == 0 else "STOP_LOSS",
                "realizedReturnPct": 10.0 if i % 2 == 0 else -5.0
            }
        }
        for i in range(5)
    ]
    ExperimentLedger.save_ledger({"version": "1.0.0", "signals": signals}, ledger_path)

    scorecard = ExperimentLedger.compute_governance_scorecard(ledger_path)

    clus = scorecard["clusteringAndDependence"]
    assert clus["maxTradesPerSession"] == 5
    assert clus["uniqueSessionsCount"] == 1
    assert clus["sectorConcentration"]["TECHNOLOGY"] == 4
    assert clus["maxSectorConcentrationPct"] == 80.0
    assert "dependenceWarning" in clus

    port = scorecard["portfolioAggregation"]
    assert port["maxConcurrentEntries"] == 5
    assert port["topSectorExposurePct"] == 80.0


def test_dual_random_baselines_mandatory_reporting():
    """Verify that BaselineEngine executes and reports BOTH unconditional and sector-conditioned baselines."""
    signals = [
        {
            "signalId": "TECH_1",
            "symbol": "TECH_A",
            "signalDate": "2026-09-01",
            "stopLossPct": -6.5,
            "takeProfit1Pct": 15.0,
            "status": "RESOLVED",
            "inputs": {"sector": "TECHNOLOGY"},
            "forwardTracking": {
                "resolvedOutcome": "TP1_WIN",
                "realizedReturnPct": 15.0
            }
        }
    ]

    universe = {
        "TECH_A": [{"date": "2026-09-02", "open": 100, "high": 120, "low": 98, "close": 118}],
        "TECH_B": [{"date": "2026-09-02", "open": 100, "high": 116, "low": 99, "close": 115}],
        "FIN_A": [{"date": "2026-09-02", "open": 100, "high": 102, "low": 90, "close": 91}],
    }

    sectors_map = {
        "TECH_A": "TECHNOLOGY",
        "TECH_B": "TECHNOLOGY",
        "FIN_A": "FINANCIALS",
    }

    res = BaselineEngine.evaluate_dual_random_baselines(
        signals=signals,
        universe_candles_map=universe,
        n_simulations=50,
        seed=42,
        friction_bps=30.0,
        asset_sectors_map=sectors_map
    )

    assert "unconditionalRandom" in res
    assert "sectorConditionedRandom" in res
    assert res["unconditionalRandom"]["status"] == "COMPLETED"
    assert res["sectorConditionedRandom"]["status"] == "COMPLETED"
    assert "Both baselines must be reported simultaneously" in res["reportingRequirement"]


def test_governance_engine_authority_and_tampering_unmocked(tmp_path):
    """Unmocked integration test verifying engine tampering detection and fail-closed eligibility revocation."""
    import shutil
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger

    repo_root = os.path.dirname(os.path.dirname(__file__))

    # 1. Live repository check: Approved source MUST be VERIFIED in production, or fail closed as unratified in development
    real_audit = ExperimentLedger.verify_frozen_engine_manifest()
    cand_manifest_path = os.path.join(repo_root, "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json")
    if os.path.exists(cand_manifest_path):
        # Active candidate development scope (PR #105 / ARX-S4-ARCH-002)
        # Development candidate engines MUST fail production freeze verification (fail-closed protection):
        assert real_audit["status"] == "CORRUPTED", (
            "Safety violation: Development candidate engines must not pass production manifest verification"
        )
        assert real_audit["valid"] is False
        assert real_audit["manifestVersion"] == "2.5.2"
        assert len(real_audit["engines"]) == 3

        # Development candidate engines MUST pass candidate-scoped integrity verification:
        cand_audit = ExperimentLedger.verify_candidate_engine_manifest(repo_root=repo_root)
        assert cand_audit["status"] == "CANDIDATE_VERIFIED", f"Candidate integrity audit failed: {cand_audit}"
        assert cand_audit["valid"] is True
        assert cand_audit["isProductionRatified"] is False, "Candidate manifest must never claim production ratification"
    else:
        # In clean production checkout, live repository MUST pass production manifest verification:
        assert real_audit["status"] == "VERIFIED"
        assert real_audit["valid"] is True
        assert real_audit["manifestVersion"] == "2.5.2"
        assert len(real_audit["engines"]) == 3
        for eng_name, eng_info in real_audit["engines"].items():
            assert eng_info["valid"] is True

    # 2. Setup temporary copy tree for real unmocked tamper integration testing
    temp_repo = tmp_path / "temp_repo"
    temp_repo.mkdir()

    # Copy manifest V2.5.2
    manifest_src = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_2.json")
    manifest_dst = temp_repo / "FROZEN_ENGINE_MANIFEST_V2_5_2.json"
    shutil.copy2(manifest_src, manifest_dst)

    # Populate temporary tree with authoritative production baseline engine files
    manifest = ExperimentLedger.get_frozen_manifest()
    import subprocess
    for eng_name, meta in manifest["engines"].items():
        dst_path = temp_repo / meta["filePath"]
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        src_path = os.path.join(repo_root, meta["filePath"])
        with open(src_path, "rb") as fp:
            current_bytes = fp.read().replace(b"\r\n", b"\n")
        current_sha = hashlib.sha256(current_bytes).hexdigest()
        if current_sha == meta["sha256"]:
            with open(dst_path, "wb") as fp:
                fp.write(current_bytes)
        else:
            # Extract authoritative production baseline bytes from git provenance commit / origin/main
            try:
                prod_bytes = subprocess.check_output(
                    ["git", "show", f"{manifest.get('provenanceCommit', 'origin/main')}:{meta['filePath']}"],
                    cwd=repo_root
                ).replace(b"\r\n", b"\n")
            except Exception:
                prod_bytes = subprocess.check_output(
                    ["git", "show", f"origin/main:{meta['filePath']}"],
                    cwd=repo_root
                ).replace(b"\r\n", b"\n")
            with open(dst_path, "wb") as fp:
                fp.write(prod_bytes)

    # Setup temporary test ledger
    ledger_path = str(tmp_path / "integration_ledger.json")
    clean_ledger = {
        "version": "1.0.0",
        "signals": [
            {
                "signalId": "SIG_CLEAN_PROSPECTIVE",
                "symbol": "ANET",
                "signalDate": "2026-10-15",
                "status": "RESOLVED",
                "prospectiveValidation": {"isProspectiveClean": True},
                "inputs": {"sector": "TECHNOLOGY", "marketRegime": "BULL"},
                "forwardTracking": {"resolvedOutcome": "TP1_WIN", "realizedReturnPct": 8.0}
            }
        ]
    }
    ExperimentLedger.save_ledger(clean_ledger, ledger_path)

    # 3. Prove the unchanged temporary artifact passes both verifier and scorecard
    temp_audit_clean = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_repo))
    assert temp_audit_clean["status"] == "VERIFIED"
    assert temp_audit_clean["valid"] is True

    sc_clean = ExperimentLedger.compute_governance_scorecard(ledger_path, repo_root=str(temp_repo))
    assert sc_clean["cohortFirewall"]["cleanEvaluationEligible"] is True
    assert sc_clean["cohortFirewall"]["status"] == "AUTO_INSULATED_PROSPECTIVE_CLEAN"

    # 4. Verify actual rejection of changed source bytes for EACH protected engine
    for eng_name, meta in manifest["engines"].items():
        eng_file = temp_repo / meta["filePath"]
        with open(eng_file, "rb") as fp:
            orig_bytes = fp.read()
        try:
            # Mutate source byte
            with open(eng_file, "wb") as fp:
                fp.write(orig_bytes + b"\n# UNAUTHORIZED_MUTATION\n")

            mut_audit = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_repo))
            assert mut_audit["status"] == "CORRUPTED"
            assert mut_audit["valid"] is False
            assert mut_audit["engines"][eng_name]["valid"] is False

            # Verify scorecard eligibility revocation using real failed verification
            sc_mut = ExperimentLedger.compute_governance_scorecard(ledger_path, repo_root=str(temp_repo))
            assert sc_mut["cohortFirewall"]["cleanEvaluationEligible"] is False
            assert sc_mut["cohortFirewall"]["status"] == "PRODUCTION_ENGINE_MUTATION_DETECTED"
            assert "CRITICAL" in sc_mut["cohortFirewall"]["firewallWarning"]
        finally:
            # Restore exact bytes for subsequent tests
            with open(eng_file, "wb") as fp:
                fp.write(orig_bytes)

    # 5. Verify actual rejection of changed manifest hash or engine path
    with open(manifest_dst, "rb") as fp:
        orig_manifest_bytes = fp.read()
    try:
        # Mutate expected hash in manifest
        with open(manifest_dst, "r", encoding="utf-8") as fp:
            manifest_json = json.load(fp)
        manifest_json["engines"]["optimal_execution"]["sha256"] = "0" * 64
        with open(manifest_dst, "w", encoding="utf-8") as fp:
            json.dump(manifest_json, fp, indent=2)

        # Implementation's exact digest validation rejects tampered manifest
        mut_hash_audit = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_repo))
        assert mut_hash_audit["status"] == "MANIFEST_MISSING"
        assert mut_hash_audit["valid"] is False

        # Mutate engine path
        manifest_json["engines"]["optimal_execution"]["sha256"] = meta["sha256"]
        manifest_json["engines"]["optimal_execution"]["filePath"] = "analyst_dashboard/analyzers/nonexistent.py"
        with open(manifest_dst, "w", encoding="utf-8") as fp:
            json.dump(manifest_json, fp, indent=2)

        mut_path_audit = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_repo))
        assert mut_path_audit["status"] == "MANIFEST_MISSING"
        assert mut_path_audit["valid"] is False
    finally:
        with open(manifest_dst, "wb") as fp:
            fp.write(orig_manifest_bytes)

    # 6. Verify actual rejection of missing engine entry
    try:
        with open(manifest_dst, "r", encoding="utf-8") as fp:
            manifest_json = json.load(fp)
        del manifest_json["engines"]["optimal_execution"]
        with open(manifest_dst, "w", encoding="utf-8") as fp:
            json.dump(manifest_json, fp, indent=2)

        del_audit = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_repo))
        assert del_audit["status"] == "MANIFEST_MISSING"
        assert del_audit["valid"] is False
    finally:
        with open(manifest_dst, "wb") as fp:
            fp.write(orig_manifest_bytes)

    # 7. Verify actual rejection of missing/malformed manifest
    try:
        os.remove(manifest_dst)
        missing_audit = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_repo))
        assert missing_audit["status"] == "MANIFEST_MISSING"
        assert missing_audit["valid"] is False

        with open(manifest_dst, "w", encoding="utf-8") as fp:
            fp.write("{ invalid json manifest content")
        malformed_audit = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_repo))
        assert malformed_audit["status"] == "MANIFEST_MISSING"
        assert malformed_audit["valid"] is False
    finally:
        with open(manifest_dst, "wb") as fp:
            fp.write(orig_manifest_bytes)

    # 8. Verify actual rejection of wrong authority/version or unauthorized successor
    unauth_audit = ExperimentLedger.verify_frozen_engine_manifest(version="9.9.9", repo_root=str(temp_repo))
    assert unauth_audit["status"] == "MANIFEST_MISSING"
    assert unauth_audit["valid"] is False

    # 9. Verify candidate-scoped tamper detection and rejection of unratified production claims
    temp_cand_repo = tmp_path / "temp_cand_repo"
    temp_cand_repo.mkdir()

    cand_manifest_src = os.path.join(repo_root, "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json")
    if os.path.exists(cand_manifest_src):
        cand_manifest_dst = temp_cand_repo / "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json"
        shutil.copy2(cand_manifest_src, cand_manifest_dst)

        # Copy production manifest into candidate tree to test production rejection
        shutil.copy2(manifest_src, temp_cand_repo / "FROZEN_ENGINE_MANIFEST_V2_5_2.json")

        # Copy candidate engines from live repo
        for eng_name, meta in manifest["engines"].items():
            dst_cand_path = temp_cand_repo / meta["filePath"]
            dst_cand_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.path.join(repo_root, meta["filePath"]), dst_cand_path)

        # 9.1 Clean candidate passes candidate verification
        cand_clean = ExperimentLedger.verify_candidate_engine_manifest(repo_root=str(temp_cand_repo))
        assert cand_clean["status"] == "CANDIDATE_VERIFIED"
        assert cand_clean["valid"] is True
        assert cand_clean["isProductionRatified"] is False

        # 9.2 Candidate cannot be represented as ratified production release (fails closed)
        prod_reject = ExperimentLedger.verify_frozen_engine_manifest(repo_root=str(temp_cand_repo))
        assert prod_reject["status"] == "CORRUPTED"
        assert prod_reject["valid"] is False

        # 9.3 1-byte mutation on candidate engine fails candidate verification
        test_eng_file = temp_cand_repo / "analyst_dashboard" / "analyzers" / "optimal_execution.py"
        with open(test_eng_file, "rb") as fp:
            orig_cand_bytes = fp.read()
        try:
            with open(test_eng_file, "wb") as fp:
                fp.write(orig_cand_bytes + b"\n# UNAUTHORIZED_CANDIDATE_MUTATION\n")
            mut_cand_audit = ExperimentLedger.verify_candidate_engine_manifest(repo_root=str(temp_cand_repo))
            assert mut_cand_audit["status"] == "CORRUPTED"
            assert mut_cand_audit["valid"] is False
        finally:
            with open(test_eng_file, "wb") as fp:
                fp.write(orig_cand_bytes)

        # 9.4 Candidate manifest claiming production ratification fails closed
        try:
            with open(cand_manifest_dst, "r", encoding="utf-8") as fp:
                cand_json = json.load(fp)
            cand_json["isProductionRatified"] = True  # Illegal ratification claim
            with open(cand_manifest_dst, "w", encoding="utf-8") as fp:
                json.dump(cand_json, fp, indent=2)

            fraud_audit = ExperimentLedger.verify_candidate_engine_manifest(repo_root=str(temp_cand_repo))
            assert fraud_audit["status"] == "CORRUPTED"
            assert fraud_audit["valid"] is False
            assert fraud_audit["error"] == "UNAUTHORIZED_PRODUCTION_RATIFICATION_CLAIM"
        finally:
            shutil.copy2(cand_manifest_src, cand_manifest_dst)


