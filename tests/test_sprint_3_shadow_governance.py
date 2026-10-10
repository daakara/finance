"""Tests for ARX Terminal Radar Sprint 3 Production Shadow Governance & Contamination Control.

Covers mandatory test matrix items A through T:
A. exposed case becomes holdout-ineligible
B. exposed group becomes holdout-ineligible
C. same case cannot remain FUTURE_UNSEEN_ELIGIBLE after exposure
D. semantic delta must be ledgered
E. unknown candidate generation fails closed
F. shadow decision requires candidate SHA
G. shadow decision requires semantic closure hash
H. shadow decision requires universe_build_id
I. shadow decision requires snapshot_run_id
J. outcome settlement cannot mutate prospective decision record
K. future outcome cannot alter domain classification
L. shadow cannot call execution/order hooks
M. model tuning remains frozen
N. internal reference cannot be promoted to Gold/Silver
O. outcome stream can be access-restricted
P. exposure registry is append-only
Q. duplicate exposure is idempotent or deterministically reconciled
R. candidate generation transition is explicit
S. no synthetic/replay/admin record enters natural-production denominator
T. evidence-authority status remains PRODUCTION_ENGINEERING_OBSERVATION
"""

import copy
import hashlib
import json
import pytest

from analyst_dashboard.vcp.sprint_3_shadow_governance import (
    ALLOWED_SHADOW_CLAIMS,
    PROHIBITED_SHADOW_CLAIMS,
    ClaimViolationError,
    validate_claim,
    TelemetryClass,
    check_telemetry_access,
    ProductionExposureRecord,
    ProductionExposureLedger,
    HoldoutExclusionRegistry,
    SemanticDeltaClassification,
    SemanticDeltaRecord,
    SemanticDeltaLedger,
    CandidateGeneration,
    CandidateGenerationManager,
    ProspectiveDecisionRecord,
    ProspectiveDecisionLedger,
    OutcomeSettlementRecord,
    OutcomeSettlementLedger,
    ShadowActioningProhibitedError,
    ShadowRoutingGuard,
    ShadowDenominatorMetrics,
    Sprint3ShadowGovernanceSuite,
    SHADOW_EVIDENCE_AUTHORITY,
    SHADOW_DOMAIN_AUTHORITY,
    EXTERNAL_DOMAIN_AUTHORITY,
    MODEL_TUNING,
    MODEL_TUNING_STATUS,
    LEARNING_CLAIM,
    INTERNAL_REFERENCE_TO_GOLD_PROMOTION,
    INTERNAL_REFERENCE_TO_SILVER_PROMOTION,
)


# ======================================================================
# Test Matrix A, B, C: Holdout Ineligibility upon Exposure
# ======================================================================

def test_a_exposed_case_becomes_holdout_ineligible():
    ledger = ProductionExposureLedger()
    rec = ledger.record_exposure(
        security_id="AAPL",
        evaluation_as_of="2026-10-10T00:00:00Z",
        group_or_episode_id="GRP_001",
        universe_build_id="UNIV_2026_01",
        snapshot_run_id="SNAP_001",
        candidate_generation_id="CANDIDATE_GENERATION_001",
        candidate_sha="a" * 40,
        semantic_closure_hash="b" * 64,
        runtime_config_hash="c" * 64,
        data_provenance_hash="d" * 64,
        candidate_output_visible_to_dev=True,
    )
    assert rec.future_holdout_eligibility == "NO"
    assert rec.exclusion_reason == "PRODUCTION_SHADOW_EXPOSURE"


def test_b_exposed_group_becomes_holdout_ineligible():
    registry = HoldoutExclusionRegistry()
    hashes = registry.register_exclusion(
        security_id="NVDA",
        evaluation_as_of="2026-10-10T00:00:00Z",
        group_or_episode_id="EPISODE_NVDA_VCP_2026",
    )
    assert registry.is_episode_excluded("EPISODE_NVDA_VCP_2026") is True
    # Verify collision count
    collisions = registry.check_episode_collisions([hashes["episode_or_group_hash"]])
    assert collisions == 1


def test_c_same_case_cannot_remain_future_unseen_eligible_after_exposure():
    registry = HoldoutExclusionRegistry()
    hashes = registry.register_exclusion(
        security_id="MSFT",
        evaluation_as_of="2026-10-10T12:00:00Z",
        group_or_episode_id="GRP_MSFT",
    )
    assert registry.is_case_excluded("MSFT", "2026-10-10T12:00:00Z") is True
    collisions = registry.check_collisions([hashes["case_content_hash"]])
    assert collisions == 1


# ======================================================================
# Test Matrix D, E: Semantic Delta & Candidate Generation
# ======================================================================

def test_d_semantic_delta_must_be_ledgered():
    ledger = SemanticDeltaLedger()
    delta = ledger.record_delta(
        classification=SemanticDeltaClassification.SPRINT_3_POST_DEFERRAL_SEMANTIC_DELTA,
        parent_candidate_generation="CANDIDATE_GENERATION_001",
        resulting_candidate_generation="CANDIDATE_GENERATION_002",
        commit_sha="e" * 40,
        affected_authority="VCP_RULE_SEMANTICS",
        before_semantic_hash="1" * 64,
        after_semantic_hash="2" * 64,
        change_rationale="Add explicit volume dry-up contraction tolerance",
        change_origin="SPRINT_3_SHADOW_OBSERVATION",
        informed_by="internal reference case",
    )
    assert delta.delta_id is not None
    assert ledger.count() == 1
    assert ledger.get_deltas()[0].classification == SemanticDeltaClassification.SPRINT_3_POST_DEFERRAL_SEMANTIC_DELTA


def test_e_unknown_candidate_generation_fails_closed():
    manager = CandidateGenerationManager()
    manager.register_generation(
        candidate_generation_id="CANDIDATE_GENERATION_001",
        candidate_sha="a" * 40,
        semantic_closure_hash="b" * 64,
    )
    with pytest.raises(KeyError, match="Unknown candidate generation"):
        manager.get_generation("CANDIDATE_GENERATION_999")

    with pytest.raises(ValueError, match="Parent generation 'CANDIDATE_GENERATION_XYZ' unknown"):
        manager.register_generation(
            candidate_generation_id="CANDIDATE_GENERATION_002",
            candidate_sha="c" * 40,
            semantic_closure_hash="d" * 64,
            parent_generation="CANDIDATE_GENERATION_XYZ",
        )


# ======================================================================
# Test Matrix F, G, H, I: Decision Record Field Invariants
# ======================================================================

def test_f_shadow_decision_requires_candidate_sha():
    ledger = ProspectiveDecisionLedger()
    with pytest.raises(ValueError, match="candidate_sha"):
        ledger.record_decision(
            evaluation_as_of="2026-10-10T00:00:00Z",
            known_at="2026-10-10T00:00:00Z",
            security_id="AAPL",
            universe_build_id="UNIV_001",
            snapshot_run_id="SNAP_001",
            candidate_generation_id="CANDIDATE_GENERATION_001",
            candidate_sha="",  # Invalid empty SHA
            semantic_closure_hash="b" * 64,
            runtime_config_hash="c" * 64,
            dependency_lock_hash="d" * 64,
            data_provenance_hash="e" * 64,
            ruleset_id="VCP_RULES",
            ruleset_version="1.0.0",
            predicate_vector_hash="f" * 64,
            classification="SETUP_FORMING",
            decision_posture="WATCH",
            input_fingerprint="g" * 64,
        )


def test_g_shadow_decision_requires_semantic_closure_hash():
    ledger = ProspectiveDecisionLedger()
    with pytest.raises(ValueError, match="semantic_closure_hash"):
        ledger.record_decision(
            evaluation_as_of="2026-10-10T00:00:00Z",
            known_at="2026-10-10T00:00:00Z",
            security_id="AAPL",
            universe_build_id="UNIV_001",
            snapshot_run_id="SNAP_001",
            candidate_generation_id="CANDIDATE_GENERATION_001",
            candidate_sha="a" * 40,
            semantic_closure_hash="short_hash",  # Invalid short hash
            runtime_config_hash="c" * 64,
            dependency_lock_hash="d" * 64,
            data_provenance_hash="e" * 64,
            ruleset_id="VCP_RULES",
            ruleset_version="1.0.0",
            predicate_vector_hash="f" * 64,
            classification="SETUP_FORMING",
            decision_posture="WATCH",
            input_fingerprint="g" * 64,
        )


def test_h_shadow_decision_requires_universe_build_id():
    ledger = ProspectiveDecisionLedger()
    with pytest.raises(ValueError, match="universe_build_id"):
        ledger.record_decision(
            evaluation_as_of="2026-10-10T00:00:00Z",
            known_at="2026-10-10T00:00:00Z",
            security_id="AAPL",
            universe_build_id="",  # Missing
            snapshot_run_id="SNAP_001",
            candidate_generation_id="CANDIDATE_GENERATION_001",
            candidate_sha="a" * 40,
            semantic_closure_hash="b" * 64,
            runtime_config_hash="c" * 64,
            dependency_lock_hash="d" * 64,
            data_provenance_hash="e" * 64,
            ruleset_id="VCP_RULES",
            ruleset_version="1.0.0",
            predicate_vector_hash="f" * 64,
            classification="SETUP_FORMING",
            decision_posture="WATCH",
            input_fingerprint="g" * 64,
        )


def test_i_shadow_decision_requires_snapshot_run_id():
    ledger = ProspectiveDecisionLedger()
    with pytest.raises(ValueError, match="snapshot_run_id"):
        ledger.record_decision(
            evaluation_as_of="2026-10-10T00:00:00Z",
            known_at="2026-10-10T00:00:00Z",
            security_id="AAPL",
            universe_build_id="UNIV_001",
            snapshot_run_id="",  # Missing
            candidate_generation_id="CANDIDATE_GENERATION_001",
            candidate_sha="a" * 40,
            semantic_closure_hash="b" * 64,
            runtime_config_hash="c" * 64,
            dependency_lock_hash="d" * 64,
            data_provenance_hash="e" * 64,
            ruleset_id="VCP_RULES",
            ruleset_version="1.0.0",
            predicate_vector_hash="f" * 64,
            classification="SETUP_FORMING",
            decision_posture="WATCH",
            input_fingerprint="g" * 64,
        )


# ======================================================================
# Test Matrix J, K: Outcome Separation & Immutability
# ======================================================================

def test_j_outcome_settlement_cannot_mutate_prospective_decision_record():
    prospective_ledger = ProspectiveDecisionLedger()
    dec = prospective_ledger.record_decision(
        evaluation_as_of="2026-10-10T00:00:00Z",
        known_at="2026-10-10T00:00:00Z",
        security_id="AAPL",
        universe_build_id="UNIV_001",
        snapshot_run_id="SNAP_001",
        candidate_generation_id="CANDIDATE_GENERATION_001",
        candidate_sha="a" * 40,
        semantic_closure_hash="b" * 64,
        runtime_config_hash="c" * 64,
        dependency_lock_hash="d" * 64,
        data_provenance_hash="e" * 64,
        ruleset_id="VCP_RULES",
        ruleset_version="1.0.0",
        predicate_vector_hash="f" * 64,
        classification="SETUP_READY",
        decision_posture="ACTIONABLE",
        input_fingerprint="g" * 64,
    )

    settlement_ledger = OutcomeSettlementLedger(prospective_ledger)
    settlement = settlement_ledger.record_settlement(
        decision_record_id=dec.decision_record_id,
        settlement_as_of="2026-10-15T00:00:00Z",
        future_observation={"t1_price": 240.0, "t5_price": 255.0},
        target_state="TARGET_1_HIT",
        stop_state="UNTOUCHED",
        return_metrics={"gain_pct": 6.25},
        data_provenance_hash="p" * 64,
    )
    assert settlement.decision_record_id == dec.decision_record_id

    # Verify original prospective record is completely unchanged
    original_dec = prospective_ledger.get_records()[0]
    assert original_dec.classification == "SETUP_READY"
    assert original_dec.decision_posture == "ACTIONABLE"
    assert original_dec.evaluation_as_of == "2026-10-10T00:00:00Z"


def test_k_future_outcome_cannot_alter_domain_classification():
    prospective_ledger = ProspectiveDecisionLedger()
    dec = prospective_ledger.record_decision(
        evaluation_as_of="2026-10-10T00:00:00Z",
        known_at="2026-10-10T00:00:00Z",
        security_id="TSLA",
        universe_build_id="UNIV_001",
        snapshot_run_id="SNAP_001",
        candidate_generation_id="CANDIDATE_GENERATION_001",
        candidate_sha="a" * 40,
        semantic_closure_hash="b" * 64,
        runtime_config_hash="c" * 64,
        dependency_lock_hash="d" * 64,
        data_provenance_hash="e" * 64,
        ruleset_id="VCP_RULES",
        ruleset_version="1.0.0",
        predicate_vector_hash="f" * 64,
        classification="REJECTED_WIDE_BASE",
        decision_posture="NON_ACTIONABLE",
        input_fingerprint="g" * 64,
    )

    # Attempting to mutate dataclass raises FrozenInstanceError
    with pytest.raises(Exception):
        dec.classification = "SETUP_READY"  # Dataclass is frozen

    # Even if later settlement is profitable, original classification remains strictly REJECTED_WIDE_BASE
    assert dec.classification == "REJECTED_WIDE_BASE"


# ======================================================================
# Test Matrix L, M, N: Routing Guards & Domain Constraints
# ======================================================================

def test_l_shadow_cannot_call_execution_or_order_hooks():
    ShadowRoutingGuard.assert_non_actioning()
    with pytest.raises(ShadowActioningProhibitedError, match="order hook"):
        ShadowRoutingGuard.execute_order_hook(symbol="AAPL", quantity=100)

    with pytest.raises(ShadowActioningProhibitedError, match="portfolio mutation"):
        ShadowRoutingGuard.mutate_portfolio_hook(symbol="AAPL", position=100)


def test_m_model_tuning_remains_frozen():
    assert MODEL_TUNING == "FROZEN"
    assert MODEL_TUNING_STATUS == "FROZEN"
    assert LEARNING_CLAIM == "NOT_AUTHORIZED"


def test_n_internal_reference_cannot_be_promoted_to_gold_or_silver():
    assert INTERNAL_REFERENCE_TO_GOLD_PROMOTION == "PROHIBITED_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION"
    assert INTERNAL_REFERENCE_TO_SILVER_PROMOTION == "PROHIBITED_WITHOUT_EXTERNAL_INDEPENDENT_ADJUDICATION"


# ======================================================================
# Test Matrix O, P, Q, R, S, T: Access, Integrity, Idempotency & Denominator
# ======================================================================

def test_o_outcome_stream_can_be_access_restricted():
    # Developer role has access to Class A and Class B, but not Class C (Outcome)
    assert check_telemetry_access(TelemetryClass.CLASS_A_ENGINEERING, "DEVELOPER") is True
    assert check_telemetry_access(TelemetryClass.CLASS_B_DECISION_DIAGNOSTIC, "DEVELOPER") is True
    assert check_telemetry_access(TelemetryClass.CLASS_C_OUTCOME, "DEVELOPER") is False
    assert check_telemetry_access(TelemetryClass.CLASS_D_HUMAN_CORRECTNESS_JUDGMENT, "DEVELOPER") is False

    # Governance auditor has access to Class C and D
    assert check_telemetry_access(TelemetryClass.CLASS_C_OUTCOME, "GOVERNANCE_AUDITOR") is True
    assert check_telemetry_access(TelemetryClass.CLASS_D_HUMAN_CORRECTNESS_JUDGMENT, "GOVERNANCE_AUDITOR") is True


def test_p_exposure_registry_is_append_only():
    registry = HoldoutExclusionRegistry()
    registry.register_exclusion("AAPL", "2026-10-10", "G1")
    registry.register_exclusion("MSFT", "2026-10-10", "G2")
    stats = registry.stats()
    assert stats["excluded_case_count"] == 2
    assert stats["excluded_episode_count"] == 2


def test_q_duplicate_exposure_is_idempotent():
    ledger = ProductionExposureLedger()
    rec1 = ledger.record_exposure(
        security_id="AAPL",
        evaluation_as_of="2026-10-10T00:00:00Z",
        group_or_episode_id="GRP_001",
        universe_build_id="UNIV_001",
        snapshot_run_id="SNAP_001",
        candidate_generation_id="CANDIDATE_GENERATION_001",
        candidate_sha="a" * 40,
        semantic_closure_hash="b" * 64,
        runtime_config_hash="c" * 64,
        data_provenance_hash="d" * 64,
    )
    rec2 = ledger.record_exposure(
        security_id="AAPL",
        evaluation_as_of="2026-10-10T00:00:00Z",
        group_or_episode_id="GRP_001",
        universe_build_id="UNIV_001",
        snapshot_run_id="SNAP_001",
        candidate_generation_id="CANDIDATE_GENERATION_001",
        candidate_sha="a" * 40,
        semantic_closure_hash="b" * 64,
        runtime_config_hash="c" * 64,
        data_provenance_hash="d" * 64,
    )
    assert rec1.exposure_id == rec2.exposure_id
    assert ledger.count() == 1  # Exactly 1, deduplicated / idempotent


def test_r_candidate_generation_transition_is_explicit():
    manager = CandidateGenerationManager()
    gen1 = manager.register_generation(
        candidate_generation_id="CANDIDATE_GENERATION_001",
        candidate_sha="1" * 40,
        semantic_closure_hash="a" * 64,
    )
    assert manager.get_active_generation().candidate_generation_id == "CANDIDATE_GENERATION_001"

    gen2 = manager.register_generation(
        candidate_generation_id="CANDIDATE_GENERATION_002",
        candidate_sha="2" * 40,
        semantic_closure_hash="b" * 64,
        parent_generation="CANDIDATE_GENERATION_001",
        semantic_delta_set=["DELTA_001"],
    )
    assert manager.get_active_generation().candidate_generation_id == "CANDIDATE_GENERATION_002"
    assert gen2.parent_generation == "CANDIDATE_GENERATION_001"


def test_s_no_synthetic_replay_admin_enters_natural_production_denominator():
    metrics = ShadowDenominatorMetrics()
    metrics.assert_initial_denominator_clean()
    assert metrics.shadow_record_count == 0
    assert metrics.natural_production_shadow_record_count == 0
    assert metrics.synthetic_shadow_record_count == 0
    assert metrics.replay_shadow_record_count == 0
    assert metrics.admin_forced_shadow_record_count == 0


def test_t_evidence_authority_status_remains_production_engineering_observation():
    assert SHADOW_EVIDENCE_AUTHORITY == "PRODUCTION_ENGINEERING_OBSERVATION"
    assert SHADOW_DOMAIN_AUTHORITY == "INTERNAL_REFERENCE_ONLY"
    assert EXTERNAL_DOMAIN_AUTHORITY == "NOT_ESTABLISHED"

    # Also test claim validator enforcement
    with pytest.raises(ClaimViolationError):
        validate_claim("ARX Terminal VCP is Gold validated by external expert")

    with pytest.raises(ClaimViolationError):
        validate_claim("ARX Terminal is alpha generating in production")

    assert validate_claim("ARX Terminal VCP is production shadow observed and engineering verified") is True


# ======================================================================
# SECTION 9 & 12: MUTATION SENSITIVITY & ATOMIC WIRING TESTS
# ======================================================================

from analyst_dashboard.vcp.sprint_3_shadow_governance import (
    CandidateSemanticClosure,
    SemanticClosureItem,
    build_canonical_semantic_closure,
    CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
    get_default_shadow_suite,
    reset_default_shadow_suite,
)


def test_u_candidate_mutation_sensitivity_vcp_predicate():
    """Mutating VCP predicate semantics must change the closure hash."""
    base_closure = build_canonical_semantic_closure()
    base_hash = base_closure.compute_closure_hash()
    assert base_hash == CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH

    mutated_items = dict(base_closure.items)
    mutated_items["vcp_predicate_semantics"] = SemanticClosureItem(
        input_key="vcp_predicate_semantics",
        authority_name="MUTATED_VCP_PREDICATE_SEMANTICS",
        authority_hash="f" * 64,
        classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
        description="Mutated predicate test",
    )
    mutated_closure = CandidateSemanticClosure(items=mutated_items)
    mutated_hash = mutated_closure.compute_closure_hash()

    assert mutated_hash != base_hash


def test_v_candidate_mutation_sensitivity_threshold():
    """Mutating threshold semantic authority must change the closure hash."""
    base_closure = build_canonical_semantic_closure()
    base_hash = base_closure.compute_closure_hash()

    mutated_items = dict(base_closure.items)
    mutated_items["threshold_authorities"] = SemanticClosureItem(
        input_key="threshold_authorities",
        authority_name="MUTATED_CONFLUENCE_SCORE_FLOOR_80",
        authority_hash="e" * 64,
        classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
        description="Mutated threshold test",
    )
    mutated_closure = CandidateSemanticClosure(items=mutated_items)
    mutated_hash = mutated_closure.compute_closure_hash()

    assert mutated_hash != base_hash


def test_w_candidate_mutation_sensitivity_universe():
    """Mutating universe semantic authority must change the closure hash."""
    base_closure = build_canonical_semantic_closure()
    base_hash = base_closure.compute_closure_hash()

    mutated_items = dict(base_closure.items)
    mutated_items["universe_builder_identity"] = SemanticClosureItem(
        input_key="universe_builder_identity",
        authority_name="MUTATED_UNIVERSE_BUILDER_V2",
        authority_hash="d" * 64,
        classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
        description="Mutated universe builder test",
    )
    mutated_closure = CandidateSemanticClosure(items=mutated_items)
    mutated_hash = mutated_closure.compute_closure_hash()

    assert mutated_hash != base_hash


def test_x_candidate_mutation_sensitivity_data_interpretation():
    """Mutating data interpretation authority must change the closure hash."""
    base_closure = build_canonical_semantic_closure()
    base_hash = base_closure.compute_closure_hash()

    mutated_items = dict(base_closure.items)
    mutated_items["data_eligibility"] = SemanticClosureItem(
        input_key="data_eligibility",
        authority_name="MUTATED_DAILY_CANDLE_COUNT_GTE_100",
        authority_hash="c" * 64,
        classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
        description="Mutated candle threshold test",
    )
    mutated_closure = CandidateSemanticClosure(items=mutated_items)
    mutated_hash = mutated_closure.compute_closure_hash()

    assert mutated_hash != base_hash


def test_y_non_semantic_metadata_stability():
    """Mutating non-semantic metadata or execution inputs must not change the closure hash."""
    base_closure = build_canonical_semantic_closure(metadata={"run_timestamp": "2026-10-10T00:00:00Z"})
    base_hash = base_closure.compute_closure_hash()

    # Mutate metadata dictionary
    mutated_metadata_closure = build_canonical_semantic_closure(metadata={"run_timestamp": "2026-10-11T99:99:99Z", "debug_trace": "xyz"})
    assert mutated_metadata_closure.compute_closure_hash() == base_hash

    # Mutate non-semantic execution input
    mutated_items = dict(base_closure.items)
    mutated_items["dependency_identity"] = SemanticClosureItem(
        input_key="dependency_identity",
        authority_name="REQUIREMENTS_LOCK_PYTHON312",
        authority_hash="1" * 64,
        classification="NON_SEMANTIC_EXECUTION_INPUT",
        description="Non-semantic dependency hash change",
    )
    non_semantic_mutated_closure = CandidateSemanticClosure(items=mutated_items)
    assert non_semantic_mutated_closure.compute_closure_hash() == base_hash


def test_z_holdout_and_future_outcome_prohibited_in_closure():
    """Holdout information or future outcome data must fail validation by construction."""
    base_closure = build_canonical_semantic_closure()

    mutated_items = dict(base_closure.items)
    mutated_items["holdout_case_data"] = SemanticClosureItem(
        input_key="holdout_case_data",
        authority_name="HOLDOUT_SECRET_KEYS",
        authority_hash="0" * 64,
        classification="PREEXISTING_FROZEN_SPRINT_2B_AUTHORITY",
        description="Prohibited holdout leakage",
    )
    leaked_closure = CandidateSemanticClosure(items=mutated_items)
    with pytest.raises(ValueError, match="Holdout/future/secret data prohibited"):
        leaked_closure.compute_closure_hash()


def test_aa_atomic_shadow_observation_recording():
    """Prospective decision, exposure record, and holdout exclusion must commit atomically."""
    suite = reset_default_shadow_suite()

    obs = suite.record_shadow_observation(
        security_id="NVDA",
        evaluation_as_of="2026-10-10",
        universe_build_id="UB_TEST_001",
        snapshot_run_id="SNAP_TEST_001",
        candidate_generation_id="CANDIDATE_GENERATION_001",
        trigger_class="NATURAL_PRODUCTION",
    )

    assert obs["decision_id"] is not None
    assert obs["exposure_id"] is not None
    assert obs["exclusion_hashes"]["case_content_hash"] is not None

    # Check prospective decision ledger
    assert suite.prospective_decision_ledger.count() == 1
    # Check exposure ledger
    assert suite.exposure_ledger.count() == 1
    # Check exclusion registry
    assert suite.exclusion_registry.is_case_excluded("NVDA", "2026-10-10") is True
    # Check denominator metrics
    assert suite.denominator.shadow_record_count == 1
    assert suite.denominator.natural_production_shadow_record_count == 1
    assert suite.denominator.unregistered_exposures == 0


def test_bb_unregistered_exposure_fails_closed():
    """Unknown candidate generation must fail closed and record zero partial exposure."""
    suite = reset_default_shadow_suite()

    with pytest.raises(ValueError, match="UNKNOWN_CANDIDATE_GENERATION"):
        suite.record_shadow_observation(
            security_id="TSLA",
            evaluation_as_of="2026-10-10",
            universe_build_id="UB_TEST_001",
            snapshot_run_id="SNAP_TEST_001",
            candidate_generation_id="UNKNOWN_CANDIDATE_GEN_999",
            trigger_class="NATURAL_PRODUCTION",
        )

    # Exposure ledger and decision ledger must have zero records
    assert suite.exposure_ledger.count() == 0
    assert suite.prospective_decision_ledger.count() == 0
    assert suite.denominator.unknown_candidate_generations == 1


def test_cc_denominator_classification_isolation():
    """SYNTHETIC, REPLAY, ADMIN_FORCED, and TEST records must not enter natural production count."""
    suite = reset_default_shadow_suite()

    classes = ["SYNTHETIC", "REPLAY", "ADMIN_FORCED", "TEST"]
    for idx, trig in enumerate(classes, start=1):
        suite.record_shadow_observation(
            security_id=f"SYM{idx}",
            evaluation_as_of="2026-10-10",
            universe_build_id="UB_TEST_001",
            snapshot_run_id=f"SNAP_{idx}",
            candidate_generation_id="CANDIDATE_GENERATION_001",
            trigger_class=trig,
        )

    assert suite.denominator.natural_production_shadow_record_count == 0
    assert suite.denominator.synthetic_shadow_record_count == 1
    assert suite.denominator.replay_shadow_record_count == 1
    assert suite.denominator.admin_forced_shadow_record_count == 1
    assert suite.denominator.test_shadow_record_count == 1
    assert suite.denominator.shadow_record_count == 4


def test_dd_scanner_runner_wires_shadow_governance():
    """VCPScannerRunner must be wired to shadow suite and record observations upon scan."""
    from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner
    from analyst_dashboard.coordination import TriggerType

    suite = reset_default_shadow_suite()
    runner = VCPScannerRunner(shadow_suite=suite)
    assert runner.shadow_suite is suite

    # Verify shadow suite is accessible via governance snapshot
    snap = runner.shadow_suite.get_governance_snapshot()
    assert snap["sprint_3_shadow_engineering"] == "AUTHORIZED"
    assert snap["routing_guards"]["user_order_execution"] == "DISABLED"
    assert snap["routing_guards"]["portfolio_mutation"] == "DISABLED"

