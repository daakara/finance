"""Targeted Offline Certification Test Suite for ARX Canonical ETF Research Specification v1.0.2.

Verifies:
1. spec_version == 1.0.2.
2. f7 is deprecated with precise public-source limitation wording; absent from active vectors.
3. f12 exists, uses BAA10Y, and is explicitly not high-yield OAS.
4. f12 aligns to ETF trading sessions (20 sessions, not calendar days) with max 5-session forward fill and fail-closed handling.
5. FIXED_INCOME_CREDIT == [f1, f8, f12].
6. Confirmatory hypothesis unit is FEATURE_X_SUBTYPE_PAIR with exact 15-pair manifest.
7. Unknown vehicle structures are not research eligible.
8. Nasdaq ETF flag does not independently establish eligibility.
9. Historical universe estimand is explicit (CURRENT_ELIGIBLE_UNIVERSE_RETROSPECTIVE_HISTORY with survivorship bias declared).
10. Classification authority matrix distinguishes CURRENT_ONLY vs POINT_IN_TIME_CAPABLE vs RETROSPECTIVE_STATIC_LABEL_ONLY.
11. Universe snapshot lineage fields and pre-generation builder identity are mandatory.
12. Storage contract strictly separates raw diagnostic store from applicability-masked model-ready view (fail closed).
13. Rank IC remains unchanged as primary validation metric.
14. Final inference method remains pending compliant-dataset adequacy evaluation.
15. Three-way hypothesis consistency verified across feature_definitions, feature_hypotheses, and confirmatory_hypothesis_manifest (11 POSITIVE, 4 NEGATIVE).
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


def test_spec_canonical_load_and_version(spec):
    assert spec["spec_version"] == "1.0.2"
    assert spec["production_actionability_status"] == "NOT_AUTHORIZED"
    assert "amendment_history" in spec
    assert len(spec["amendment_history"]) == 3
    v102 = spec["amendment_history"][2]
    assert v102["version"] == "1.0.2"
    assert v102["parent_spec_sha256"] == "736cec7da411adf08af4326e136312da2ffd9eb3499c60a2d878ed0735805fd7"


def test_f7_deprecated_and_public_source_wording(spec):
    f7 = spec["feature_definitions"]["f7_credit_spread_trend"]
    assert f7["status"] == "DEPRECATED_IN_V1_0_2"
    assert f7["original_source"] == "BAMLH0A0HYM2"
    assert "FRED HOSTED BAMLH0A0HYM2 LIMITED TO THREE YEARS" in f7["deprecation_reason"]
    assert "OLDER HISTORY NOT ASSUMED AVAILABLE" in f7["deprecation_reason"]
    assert "public_source_limitation" in f7
    assert f7["public_source_limitation"]["limitation_statement"] == "FRED HOSTED BAMLH0A0HYM2 LIMITED TO THREE YEARS STARTING APRIL 2026"

    # f7 must NOT appear in any active subtype feature vector
    for sub, data in spec["subtype_support"].items():
        assert "f7_credit_spread_trend" not in data.get("features", []), f"f7 leaked into active subtype {sub}"


def test_f12_definition_and_session_alignment_semantics(spec):
    assert "f12_baa_corporate_spread_trend" in spec["feature_definitions"]
    f12 = spec["feature_definitions"]["f12_baa_corporate_spread_trend"]
    assert f12["status"] == "PREREGISTERED_CONFIRMATORY_CANDIDATE"
    assert f12["source"] == "FRED BAA10Y"
    assert f12["is_high_yield_oas"] is False
    assert f12["is_semantically_equivalent_to_f7"] is False
    assert f12["expected_sign"] == "NEGATIVE"
    assert f12["lookback_sessions"] == 20

    # Session alignment semantics
    sas = f12["session_alignment_semantics"]
    assert sas["observation_calendar"] == "ETF_PRIMARY_TRADING_SESSION_CALENDAR"
    assert "20 ETF TRADING SESSIONS" in sas["lookback_nature"]
    assert "EXPLICITLY NOT 20 CALENDAR DAYS" in sas["lookback_nature"]
    assert sas["forward_fill"] == "MAX 5 ETF TRADING SESSIONS"
    assert "FAIL_CLOSED" in sas["gap_gt_5_sessions_policy"]


def test_fixed_income_credit_feature_vector(spec):
    credit_feats = spec["subtype_support"]["FIXED_INCOME_CREDIT"]["features"]
    assert credit_feats == [
        "f1_trend_ratio",
        "f8_relative_strength_vs_lqd",
        "f12_baa_corporate_spread_trend"
    ]


def test_confirmatory_hypothesis_unit_and_exact_manifest(spec):
    mt = spec["multiple_testing_policy"]
    assert mt["confirmatory_hypothesis_unit"] == "FEATURE_X_SUBTYPE_PAIR"
    assert mt["confirmatory_family"] == "ALL PREREGISTERED ACTIVE FEATURE_X_SUBTYPE PAIRS"
    assert mt["confirmatory_hypothesis_count"] == 15
    assert mt["bh_family_size"] == 15
    assert mt["bh_family_membership"] == "FROZEN"
    assert mt["bh_q_level"] == 0.05
    assert mt["family_redefinition_after_results"] == "PROHIBITED"

    manifest = mt["confirmatory_hypothesis_manifest"]
    assert len(manifest) == 15

    # Check each pair has all required attributes and no prematurely frozen engine
    required_pair_fields = [
        "subtype", "feature", "pair_id", "economic_hypothesis",
        "expected_sign", "validation_metric", "inference_policy",
        "p_value_inference_method_status", "bh_family_membership"
    ]
    for pair in manifest:
        for f in required_pair_fields:
            assert f in pair, f"Missing {f} in hypothesis pair {pair.get('pair_id')}"
        assert "p_value_producing_test" not in pair, (
            f"Prematurely frozen p_value_producing_test found in {pair.get('pair_id')}"
        )
        assert pair["validation_metric"] == "CROSS_SECTIONAL_SPEARMAN_RANK_IC"
        assert pair["inference_policy"] == "DEPENDENCE_CORRECTION_POLICY_V1"
        assert pair["p_value_inference_method_status"] == "PENDING_COMPLIANT_DATASET_ADEQUACY_AUDIT"
        assert pair["expected_sign"] in ["POSITIVE", "NEGATIVE"]
        assert pair["bh_family_membership"] == "GLOBAL_ACROSS_ALL_CONFIRMATORY_SUBTYPE_FEATURE_PAIRS"


def test_confirmatory_three_way_hypothesis_consistency(spec):
    fd = spec["feature_definitions"]
    fh = spec["feature_hypotheses"]
    mt = spec["multiple_testing_policy"]
    manifest = mt["confirmatory_hypothesis_manifest"]

    assert len(manifest) == 15, f"Expected 15 confirmatory pairs, got {len(manifest)}"

    # 1. Verify every distinct feature in manifest has explicit, unambiguous hypothesis mapping
    seen_features = set()
    for pair in manifest:
        feat_id = pair["feature"]
        h_id = pair.get("hypothesis_id")
        assert h_id, f"Manifest pair {pair.get('pair_id')} lacks hypothesis_id"
        assert feat_id in fd, f"Feature {feat_id} not found in feature_definitions"
        assert h_id in fh, f"Hypothesis {h_id} not found in feature_hypotheses"

        # Explicit mapping in feature_definition
        assert fd[feat_id].get("hypothesis_id") == h_id, (
            f"Mapping mismatch for feature {feat_id}: "
            f"feature_def hypothesis_id={fd[feat_id].get('hypothesis_id')} vs pair hypothesis_id={h_id}"
        )
        assert fh[h_id]["feature"] == feat_id, (
            f"Reverse mapping mismatch for hypothesis {h_id}: "
            f"feature_hypotheses points to {fh[h_id].get('feature')} vs expected {feat_id}"
        )
        seen_features.add(feat_id)

    assert len(seen_features) == 10, f"Expected 10 distinct features across 15 pairs, got {len(seen_features)}"

    # 2. Verify three-way sign consistency across all 15 pairs
    pos_count = 0
    neg_count = 0
    negative_features = {"f6_rate_momentum_20d", "f9_real_yield_regime", "f10_dxy_trend", "f12_baa_corporate_spread_trend"}

    for pair in manifest:
        pair_id = pair["pair_id"]
        feat_id = pair["feature"]
        h_id = pair["hypothesis_id"]

        fd_sign = fd[feat_id]["expected_sign"]
        fh_sign = fh[h_id]["expected_sign"]
        m_sign = pair["expected_sign"]

        # Feature definition vs Economic hypothesis
        assert fd_sign == fh_sign, (
            f"Feature definition sign ({fd_sign}) != Hypothesis sign ({fh_sign}) for {pair_id}"
        )
        # Economic hypothesis vs Manifest
        assert fh_sign == m_sign, (
            f"Hypothesis sign ({fh_sign}) != Manifest sign ({m_sign}) for {pair_id}"
        )
        # Feature definition vs Manifest
        assert fd_sign == m_sign, (
            f"Feature definition sign ({fd_sign}) != Manifest sign ({m_sign}) for {pair_id}"
        )

        if m_sign == "POSITIVE":
            pos_count += 1
            assert feat_id not in negative_features, f"Feature {feat_id} expected to be negative"
        elif m_sign == "NEGATIVE":
            neg_count += 1
            assert feat_id in negative_features, f"Feature {feat_id} unexpected negative"
        else:
            pytest.fail(f"Invalid sign {m_sign} for pair {pair_id}")

    assert pos_count == 11, f"Expected 11 positive hypotheses, derived {pos_count}"
    assert neg_count == 4, f"Expected 4 negative hypotheses, derived {neg_count}"
    assert pos_count + neg_count == 15


def test_historical_universe_policy_and_survivorship(spec):
    u = spec["universe_construction"]
    hup = u["historical_universe_policy"]
    assert hup["universe_estimand"] == "CURRENT_ELIGIBLE_UNIVERSE_RETROSPECTIVE_HISTORY"
    assert hup["survivorship_bias"] == "PRESENT_BY_DESIGN"
    assert hup["permitted_claim_scope"] == "HISTORICAL_BEHAVIOR_OF_CURRENT_ELIGIBLE_UNIVERSE_ONLY"
    assert "PROHIBITED" in hup["generalization_restriction"]


def test_current_vs_historical_classification_authority(spec):
    u = spec["universe_construction"]
    cam = u["classification_authority_matrix"]
    assert cam["current_snapshot_classification"] == "MAY_USE CURRENT VERIFIED STRUCTURED METADATA"
    assert "REQUIRES POINT_IN_TIME EVIDENCE" in cam["historical_point_in_time_classification"]

    st = cam["source_temporality"]
    assert st["nasdaq_traded_directory"] == "CURRENT_ONLY"
    assert st["sec_edgar_filings"] == "POINT_IN_TIME_CAPABLE"
    assert st["morningstar_category"] == "RETROSPECTIVE_STATIC_LABEL_ONLY"
    assert st["yfinance_metadata"] == "RETROSPECTIVE_STATIC_LABEL_ONLY"
    assert st["verified_registries"] == "POINT_IN_TIME_CAPABLE"
    assert st["defensive_heuristics"] == "RETROSPECTIVE_STATIC_LABEL_ONLY"


def test_unknown_vehicle_structures_and_nasdaq_semantics(spec):
    u = spec["universe_construction"]
    assert u["nasdaq_semantics"]["declared_field"] == "ETF"
    assert u["nasdaq_semantics"]["discovered_etf_flag_implies_eligibility"] is False
    assert u["unknown_structure_defaults_to_research_eligible"] is False
    assert "FAIL_CLOSED" in u["unknown_structure_policy"]

    assert "UNKNOWN" in spec["instrument_scope"]["allowed_structure_classes"]
    assert "UNKNOWN" not in spec["instrument_scope"]["research_eligible_structures"]
    assert "UNKNOWN" in spec["instrument_scope"]["excluded_structures"]


def test_universe_snapshot_and_builder_identity_lineage(spec):
    u = spec["universe_construction"]
    snap = u["universe_snapshot_contract"]
    assert snap["artifact_path"] == "docs/research/ETF_SURVIVING_UNIVERSE_V1.parquet"
    required = [
        "symbol", "security_name", "listing_exchange", "nasdaq_etf_flag",
        "vehicle_structure", "vehicle_structure_state", "research_subtype",
        "research_subtype_state", "classification_source", "classification_evidence",
        "classification_timestamp", "is_research_eligible", "exclusion_reason"
    ]
    for r in required:
        assert r in snap["required_fields"], f"Missing field in universe snapshot contract: {r}"

    dl = spec["dataset_lineage"]
    b_req = dl["pre_generation_builder_identity_requirement"]
    assert b_req["builder_git_commit_mandatory"] is True
    assert b_req["builder_file_sha256_mandatory"] is True


def test_storage_contract_and_model_boundary_fail_closed(spec):
    sc = spec["feature_storage_contract"]
    assert sc["raw_feature_store"]["nature"] == "PERMISSIVE_POINT_IN_TIME_FEATURE_STORE"
    assert sc["model_ready_feature_view"]["nature"] == "APPLICABILITY_MASKED_ESTIMATOR_VIEW"
    assert "HARD_FAILURE" in sc["model_ready_feature_view"]["unauthorized_feature_handling"]

    # Contamination test
    active_credit_features = set(spec["subtype_support"]["FIXED_INCOME_CREDIT"]["features"])
    unauthorized_feature = "f11_bullion_breakout"
    assert unauthorized_feature not in active_credit_features

    def project_model_matrix(features_row: dict, subtype: str):
        authorized = set(spec["subtype_support"][subtype]["features"])
        for k in features_row.keys():
            if k not in authorized:
                raise ValueError(f"UNAUTHORIZED_FEATURE_ENTERING_MODEL_MATRIX: {k} not allowed for {subtype}")
        return [features_row[k] for k in sorted(authorized)]

    clean_row = {"f1_trend_ratio": 1.2, "f8_relative_strength_vs_lqd": 0.5, "f12_baa_corporate_spread_trend": -0.1}
    assert len(project_model_matrix(clean_row, "FIXED_INCOME_CREDIT")) == 3

    contaminated_row = {"f1_trend_ratio": 1.2, "f8_relative_strength_vs_lqd": 0.5, "f12_baa_corporate_spread_trend": -0.1, "f11_bullion_breakout": 0.9}
    with pytest.raises(ValueError, match="UNAUTHORIZED_FEATURE_ENTERING_MODEL_MATRIX"):
        project_model_matrix(contaminated_row, "FIXED_INCOME_CREDIT")


def test_rank_ic_and_inference_method_status(spec):
    vp = spec["validation_protocol"]
    assert vp["primary_validation_metric"] == "CROSS_SECTIONAL_SPEARMAN_RANK_IC"
    assert vp["validation_metric_status"] == "RETAINED_PENDING_BROAD_DATASET_ADEQUACY_AUDIT"

    dc = spec["dependence_correction"]
    assert dc["policy_id"] == "DEPENDENCE_CORRECTION_POLICY_V1"
    assert dc["final_inference_method_status"] == "PENDING_COMPLIANT_DATASET_ADEQUACY_AUDIT"
    assert dc["cluster_dimensions"]["date_dimension"].startswith("LARGE")
    assert dc["cluster_dimensions"]["subtype_dimension"].startswith("SMALL")
    assert dc["two_way_cluster_asymptotics_status"] == "REQUIRES_ADEQUACY_REVIEW"

    # Candidate remedies declared
    assert "candidate_remedies" in dc and len(dc["candidate_remedies"]) > 0
    assert "DATE_BASED_BLOCK_BOOTSTRAP" in dc["candidate_remedies"]
    assert "WILD_CLUSTER_BOOTSTRAP_OVER_SUBTYPES" in dc["candidate_remedies"]
    assert "SMALL_CLUSTER_FINITE_SAMPLE_CORRECTIONS" in dc["candidate_remedies"]

    # Candidate inference methods declared
    assert "candidate_inference_methods" in dc and len(dc["candidate_inference_methods"]) > 0
    assert "TWO_WAY_CLUSTER_ROBUST_INFERENCE" in dc["candidate_inference_methods"]

    # Inference selection gate invariants
    assert "inference_selection_gate" in dc
    isg = dc["inference_selection_gate"]
    assert isg["inference_selection_requires"] == "COMPLIANT_DATASET_ADEQUACY_AUDIT"
    assert isg["inference_selection_may_use_outcome_effect_size"] is False
    assert isg["inference_method_selection_after_seeing_p_values"] == "PROHIBITED"
    assert isg["selection_basis"] == "DEPENDENCE_STRUCTURE_AND_ESTIMATOR_VALIDITY_ONLY"

    required_eval_criteria = [
        "number of dates",
        "number of subtype clusters",
        "observations per subtype",
        "cross-sectional density",
        "missingness",
        "serial dependence",
        "cross-sectional dependence",
        "cluster-size imbalance"
    ]
    for crit in required_eval_criteria:
        assert crit in isg["evaluation_criteria"], f"Missing evaluation criterion: {crit}"
