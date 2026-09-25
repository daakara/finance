"""Targeted Offline Certification Test Suite for ARX Canonical ETF Research Specification v1.0.1.

Verifies:
1. Canonical JSON loads cleanly and matches schema requirements.
2. All feature IDs referenced by supported subtypes are defined in feature_definitions.
3. Feature formula parameter metadata matches declared lookbacks and bounds.
4. Production ETF actionability is strictly NOT_AUTHORIZED.
5. Historical snapshot data (AUM, holdings, expense ratio) is explicitly prohibited in backtests.
6. 2025 holdout partition is strictly labeled HISTORICAL_HOLDOUT (never blind).
7. Stage A (ranking) and Stage B (execution) are separated.
8. Two-time macro release timing is explicitly modeled without post-close leakage; unverified same-day macro is prohibited.
9. Confirmatory hypothesis count is derived dynamically from subtype registry (equals 15), with f4 excluded.
10. Few-cluster adequacy check is formally required under dependence correction.
"""

import json
from pathlib import Path
import pytest

SPEC_PATH = Path("docs/research/ETF_RESEARCH_SPEC_V1.json")


@pytest.fixture(scope="module")
def spec():
    assert SPEC_PATH.exists(), f"Canonical specification file not found at {SPEC_PATH}"
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_spec_canonical_load_and_keys(spec):
    assert spec["spec_version"] == "1.0.1"
    assert spec["governance_status"] == "FROZEN_RESEARCH_SPECIFICATION"
    assert spec["production_actionability_status"] == "NOT_AUTHORIZED"

    required_keys = [
        "spec_version", "created_at", "governance_status", "production_actionability_status",
        "instrument_scope", "subtype_support", "universe_construction", "macro_information_timing",
        "historical_data_sources", "point_in_time_rules", "survivorship_bias_status",
        "feature_definitions", "feature_preprocessing", "target_preprocessing", "feature_hypotheses",
        "feature_source_requirements", "observation_generator", "duplicate_suppression",
        "ranking_outcomes", "risk_outcomes", "execution_outcomes", "benchmark_definitions",
        "primary_execution_model", "trade_simulation_contract", "stage_b_execution_policy",
        "transaction_cost_policy", "validation_protocol", "dependence_correction",
        "multiple_testing_policy", "primary_model_family", "baseline_models",
        "hyperparameter_search_space", "exploratory_partition", "confirmatory_partition",
        "historical_holdout_policy", "experiment_ledger_contract", "blind_test_policy",
        "dataset_schema", "dataset_lineage", "data_quality_gates", "lookahead_tests"
    ]
    for key in required_keys:
        assert key in spec, f"Missing required top-level key: {key}"


def test_subtype_feature_references_exist(spec):
    feature_defs = spec["feature_definitions"]
    for subtype, sub_data in spec["subtype_support"].items():
        features = sub_data.get("features", [])
        for feat in features:
            assert feat in feature_defs, f"Subtype {subtype} references undefined feature {feat}"


def test_confirmatory_hypothesis_count_derived(spec):
    """Derive confirmatory hypothesis count dynamically from subtype registry and assert match."""
    subtypes = spec["subtype_support"]
    total_confirmatory_pairs = 0
    all_referenced_features = []

    for sub_name, sub_info in subtypes.items():
        feats = sub_info.get("features", [])
        total_confirmatory_pairs += len(feats)
        all_referenced_features.extend(feats)

    # Assert exactly 15 pairs across the 5 active subtypes (3 each)
    assert total_confirmatory_pairs == 15, f"Expected 15 confirmatory pairs, got {total_confirmatory_pairs}"

    # Verify declared count in multiple_testing_policy matches derived count
    mt = spec["multiple_testing_policy"]
    assert mt["confirmatory_hypothesis_count"] == total_confirmatory_pairs
    assert mt["hypothesis_unit"] == "SUBTYPE_FEATURE_PAIR"
    assert mt["fdr_family"] == "GLOBAL_ACROSS_ALL_CONFIRMATORY_SUBTYPE_FEATURE_PAIRS"

    # f4_vol_compression must NEVER be in any subtype confirmatory list
    assert "f4_vol_compression" not in all_referenced_features
    assert "EXPLORATORY_ONLY" in spec["feature_definitions"]["f4_vol_compression"]["expected_sign"]


def test_feature_formula_metadata_and_lookbacks(spec):
    feature_defs = spec["feature_definitions"]
    assert len(feature_defs) == 11

    # Check bullion breakout is exactly 50 bars
    f11 = feature_defs["f11_bullion_breakout"]
    assert f11["lookback_sessions"] == 50
    assert f11["parameters"]["breakout_period"] == 50
    assert "0..49" in f11["formula"]

    # Check RSI regime distance
    f3 = feature_defs["f3_rsi_regime_distance"]
    assert f3["parameters"]["rsi_period"] == 14
    assert f3["parameters"]["regime_center"] == 60.0
    assert f3["parameters"]["regime_width"] == 15.0

    # Check SMA200
    f1 = feature_defs["f1_trend_ratio"]
    assert f1["lookback_sessions"] == 200
    assert f1["parameters"]["sma_period"] == 200

    # Ensure all features have parameters dict
    for f_id, f_val in feature_defs.items():
        assert "parameters" in f_val, f"Feature {f_id} lacks explicit parameters metadata"
        assert f_val["lookback_sessions"] > 0


def test_universe_and_minimum_history_consistency(spec):
    u = spec["universe_construction"]
    assert u["minimum_eligible_history"] == 250
    assert u["max_feature_lookback"] == 200
    assert u["warmup_buffer"] == 50
    assert u["warmup_buffer_class"] == "PREREGISTERED_POLICY"
    assert u["universe_membership"] == "POINT_IN_TIME_WITHIN_OBSERVABLE_SURVIVOR_UNIVERSE"
    assert u["point_in_time_cross_section_complete"] is False

    obs_gen = spec["observation_generator"]["stage_a_ranking"]
    assert "history_sessions >= 250" in obs_gen["predicate"]


def test_macro_release_timing_and_anti_leakage(spec):
    pit = spec["point_in_time_rules"]
    assert "HOLDINGS_FEATURES_IN_BACKTEST = PROHIBITED" in pit["holdings_data_prohibition"]
    assert "PROHIBITED" in pit["snapshot_metadata_prohibition"]
    assert "PROHIBITED" in pit["normalization_parameters_from_future_data"]

    macro_timing = spec["macro_information_timing"]
    assert macro_timing["historical_macro_policy"] == "LAGGED_KNOWABLE_INFORMATION"
    assert macro_timing["unverified_same_day_macro"] == "PROHIBITED"
    assert macro_timing["same_day_macro_used_without_release_timestamp_proof"] == "NO"
    assert "macro_source_observation_date" in macro_timing["macro_as_of_join"]["audit_columns"]


def test_dependence_correction_reclassification(spec):
    dc = spec["dependence_correction"]
    assert dc["primary_inference_candidate"] == "TWO_WAY_CLUSTERED_COVARIANCE"
    assert dc["few_cluster_adequacy_check_required"] is True
    assert dc["final_inference_method_status"] == "TO_BE_CONFIRMED_AFTER_DATASET_ADEQUACY_AUDIT"


def test_holdout_and_blind_testing_policy(spec):
    hh = spec["historical_holdout_policy"]
    assert hh["2025_holdout_status"] == "HISTORICAL_HOLDOUT"
    # Ensure word blind is not attached to 2025 holdout
    assert "blind" not in hh["2025_holdout_status"].lower()

    bp = spec["blind_test_policy"]
    assert bp["true_blind_validation"] == "FUTURE_PROSPECTIVE_ONLY (Epoch-5 prospective ledger)"


def test_ranking_and_execution_separation(spec):
    obs = spec["observation_generator"]
    assert obs["stage_a_ranking"]["position_suppression"] is False
    assert obs["stage_b_execution"]["position_suppression"] is True

    stage_b = spec["stage_b_execution_policy"]
    assert stage_b["stage_b_stop_policy"] == "UNCALIBRATED"
    assert stage_b["stage_b_target_policy"] == "UNCALIBRATED"
    assert stage_b["stage_b_actionability_threshold"] == "UNCALIBRATED"
    assert stage_b["stage_b_uncalibrated_acceptance_concept"] is True
