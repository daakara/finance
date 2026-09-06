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

    # Zero forward sessions observed
    assert ft["sessionsObserved"] == 0
    assert ft["currentPrice"] == 100.0
    assert ft["maxFavorableExcursionPct"] == 0.0


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
    assert len(signals) == 9, "All 9 live cohort signals must be present"

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
    """Verify that all 3 production engines match the frozen cryptographic manifest."""
    manifest_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FROZEN_ENGINE_MANIFEST.json")
    assert os.path.exists(manifest_path), "FROZEN_ENGINE_MANIFEST.json must exist in repository root"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    repo_root = os.path.dirname(os.path.dirname(__file__))
    import hashlib

    for engine_name, meta in manifest["engines"].items():
        full_path = os.path.join(repo_root, meta["filePath"])
        assert os.path.exists(full_path), f"Engine file {meta['filePath']} does not exist"

        with open(full_path, "rb") as fp:
            content = fp.read()
            # Normalize LF line endings to ensure platform-independent verification
            normalized = content.replace(b"\r\n", b"\n")
            computed_sha = hashlib.sha256(normalized).hexdigest()

        expected_sha = meta["sha256"]
        assert computed_sha == expected_sha, (
            f"PRODUCTION_FREEZE_VIOLATION: Engine {engine_name} ({meta['filePath']}) was modified! "
            f"Expected {expected_sha}, got {computed_sha}."
        )


def test_cohort_contamination_firewall_enforcement(tmp_path):
    """Verify that the ledger firewall strictly insulates prospective clean data from contaminated history."""
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
