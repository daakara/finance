"""
tests/test_publication_state_machine_and_foundation.py

Sprint 2 Track C: Publication State Machine & Isolated Foundation Test Suite.
Tests the fail-closed lifecycle, canonical association IDs (Step 3),
governance authority separation (Step 4), accounting closure, population isolation,
rollback invariants, and anti-leakage consumer protection.
"""

from datetime import datetime, timezone
import pytest

from analyst_dashboard.universe import (
    PublicationState,
    PublicationStateMachine,
    PublicationStateTransitionError,
    EvidenceMissingError,
    EvidenceAmbiguityError,
    StateOrderViolationError,
    PolicyViolationError,
    IdentityRegistrationEvidence,
    ProviderResolutionEvidence,
    MarketDataQualityEvidence,
    EligibilityEvidence,
    PublicationAttestationEvidence,
    StrategyAdmissionEvidence,
    ScannerDispatchEvidence,
    validate_canonical_association_id,
    IsolatedUniverseCandidate,
    AccountingClosureSummary,
    CandidateEvaluationResult,
    UniversePublicationPointer,
    IsolatedPublicationEngine,
    DEFAULT_EXPANSION_NAMESPACE,
)
from analyst_dashboard.universe.fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE

pytestmark = pytest.mark.tier1


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_aapl_candidate() -> IsolatedUniverseCandidate:
    return IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        security_type="COMMON_STOCK",
        asset_class="EQUITY",
        listing_status="ACTIVE",
        currency="USD",
        primary_listing=True,
        alpaca_symbol="AAPL",
        polygon_symbol="AAPL",
        figi_composite="BBG000B9XRY4",
        candle_count=287,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )


@pytest.fixture
def sample_goog_candidate() -> IsolatedUniverseCandidate:
    return IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_A11A24D442E2",
        symbol="GOOG",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17",
        security_type="COMMON_STOCK",
        asset_class="EQUITY",
        listing_status="ACTIVE",
        currency="USD",
        primary_listing=True,
        alpaca_symbol="GOOG",
        polygon_symbol="GOOG",
        figi_composite="BBG009S39JX6",
        candle_count=287,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )


@pytest.fixture
def sample_googl_candidate() -> IsolatedUniverseCandidate:
    return IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_B7EF9ED7A35A",
        symbol="GOOGL",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17",
        security_type="COMMON_STOCK",
        asset_class="EQUITY",
        listing_status="ACTIVE",
        currency="USD",
        primary_listing=True,
        alpaca_symbol="GOOGL",
        polygon_symbol="GOOGL",
        figi_composite="BBG009S3NB30",
        candle_count=287,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )


# ─── 1. Canonical Association ID Tests (Step 3) ──────────────────────────────

def test_canonical_association_id_validation():
    """Step 3: Validates accepted Package A association ID formats and rejects malformed IDs."""
    # 1. Accepted durable format
    assert validate_canonical_association_id("ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17") is True
    assert validate_canonical_association_id("ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17") is True
    assert validate_canonical_association_id("ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17") is True
    assert validate_canonical_association_id("ASSOC_ARX_SEC_442CE0B25527_XNAS_NVDA_2026-10-19") is True
    assert validate_canonical_association_id("ASSOC_ARX_SEC_123456ABCDEF_XNYS_BRK.B_2026-10-19") is True

    # 2. Accepted hex format
    assert validate_canonical_association_id("ASSOC_A1B2C3D4E5F6") is True

    # 3. Accepted provisional format
    assert validate_canonical_association_id("PROV_ASSOC_PROV_SEC_A562B55CDF_XNAS_AAPL_20261017") is True

    # 4. Rejected formats
    assert validate_canonical_association_id("") is False
    assert validate_canonical_association_id("RANDOM_ID_123") is False
    assert validate_canonical_association_id("ASSOC_INVALID_FORMAT_WITHOUT_DATES") is False
    assert validate_canonical_association_id("ARX_SEC_0B4C728FE0BE") is False  # Security ID, not association ID


def test_goog_googl_distinct_association_records(sample_goog_candidate, sample_googl_candidate):
    """Step 3: Alphabet Inc. Class C (GOOG) and Class A (GOOGL) must preserve distinct association records."""
    assert sample_goog_candidate.internal_security_id != sample_googl_candidate.internal_security_id
    assert sample_goog_candidate.association_id != sample_googl_candidate.association_id
    assert sample_goog_candidate.association_id == "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17"
    assert sample_googl_candidate.association_id == "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17"

    # Engine normalization preserves both distinctly without collision
    deduped, exclusions = IsolatedPublicationEngine.normalize_and_deduplicate([
        sample_goog_candidate,
        sample_googl_candidate,
    ])
    assert len(deduped) == 2
    assert len(exclusions) == 0


def test_invalid_association_id_fails_closed_in_evidence():
    """Step 3: Supplying malformed association_id to evidence fails closed."""
    with pytest.raises(EvidenceAmbiguityError, match="Invalid association_id format"):
        IdentityRegistrationEvidence(
            internal_security_id="ARX_SEC_0B4C728FE0BE",
            primary_symbol="AAPL",
            operating_mic="XNAS",
            association_id="MALFORMED_ASSOC_ID",
        )


# ─── 2. Correct Governance Authority Tests (Step 4) ──────────────────────────

def test_governance_authority_separation_vcp_012_vs_s2_act():
    """Step 4: G-VCP-012 governs strategy admission; G-S2-ACT governs operational dispatch."""
    # 1. State 5 -> State 6: Strategy Admission governed by G-VCP-012
    # Missing G-VCP-012 fails closed
    vcp_locked_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_STANDARD",
        g_vcp_012_certified=False,  # Uncertified VCP Holdout Epoch 002
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    assert vcp_locked_ev.governance_authority == "G-VCP-012"
    with pytest.raises(PolicyViolationError, match="G-VCP-012 certification absent"):
        PublicationStateMachine.transition(
            PublicationState.PUBLISHED_UNIVERSE,
            PublicationState.STRATEGY_ADMITTED,
            vcp_locked_ev,
        )

    # Valid G-VCP-012 strategy admission
    vcp_certified_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_STANDARD",
        g_vcp_012_certified=True,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    st6 = PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        vcp_certified_ev,
    )
    assert st6 == PublicationState.STRATEGY_ADMITTED

    # 2. State 6 -> State 7: Scanner Dispatch governed by G-S2-ACT
    # Attempting scanner dispatch without G-S2-ACT authorization fails closed
    s2_unauth_ev = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="VCP_SCANNER_RUNNER",
        g_s2_act_authorized=False,  # Unauthorized Sprint 2 operational activation
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    assert s2_unauth_ev.governance_authority == "G-S2-ACT"
    with pytest.raises(PolicyViolationError, match="G-S2-ACT absent"):
        PublicationStateMachine.transition(
            PublicationState.STRATEGY_ADMITTED,
            PublicationState.ACTIVELY_SCANNED,
            s2_unauth_ev,
        )

    # Valid G-S2-ACT operational dispatch
    s2_auth_ev = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="VCP_SCANNER_RUNNER",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    st7 = PublicationStateMachine.transition(
        PublicationState.STRATEGY_ADMITTED,
        PublicationState.ACTIVELY_SCANNED,
        s2_auth_ev,
    )
    assert st7 == PublicationState.ACTIVELY_SCANNED


def test_scanner_non_enrollment_firewall():
    """Step 4: G-VCP-012 does NOT authorize scanner enrollment under Sprint 2 without G-S2-ACT."""
    published_symbols = ["AAPL", "GOOG", "GOOGL"]
    active_scanner_symbols = {"AAPL", "MSFT"}  # Leaked symbol

    with pytest.raises(PolicyViolationError, match="Actionability leakage detected"):
        IsolatedPublicationEngine.assert_no_scanner_enrollment(
            published_symbols,
            active_scanner_symbols,
        )

    # Clean un-enrolled state
    clean_scanner_symbols = {"NVDA"}
    IsolatedPublicationEngine.assert_no_scanner_enrollment(
        published_symbols,
        clean_scanner_symbols,
    )


def test_vcp_strategy_admission_requires_g_vcp_012_fails_closed():
    """Task 1.1: VCP strategy admission strictly requires G-VCP-012 certification and fails closed without it."""
    # VCP strategy with g_vcp_012_certified=False
    vcp_false_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_BREAKOUT_SCANNER",
        g_vcp_012_certified=False,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    assert vcp_false_ev.is_vcp_strategy() is True
    assert vcp_false_ev.governance_authority == "G-VCP-012"
    with pytest.raises(PolicyViolationError, match="Governance authority G-VCP-012 certification absent"):
        PublicationStateMachine.transition(
            PublicationState.PUBLISHED_UNIVERSE,
            PublicationState.STRATEGY_ADMITTED,
            vcp_false_ev,
        )

    # VCP strategy with g_vcp_012_certified omitted (None)
    vcp_none_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_SWING_PILOT",
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    with pytest.raises(PolicyViolationError, match="Governance authority G-VCP-012 certification absent"):
        PublicationStateMachine.transition(
            PublicationState.PUBLISHED_UNIVERSE,
            PublicationState.STRATEGY_ADMITTED,
            vcp_none_ev,
        )

    # VCP strategy with valid g_vcp_012_certified=True succeeds
    vcp_valid_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_BREAKOUT_SCANNER",
        g_vcp_012_certified=True,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    st6 = PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        vcp_valid_ev,
    )
    assert st6 == PublicationState.STRATEGY_ADMITTED


def test_non_vcp_strategy_isolated_from_g_vcp_012():
    """Task 1.2: Non-VCP strategies admit under their own qualification authority without requiring G-VCP-012."""
    # Non-VCP strategy with g_vcp_012_certified=None/False admits successfully when strategy_certified=True
    non_vcp_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MOMENTUM_TREND_PILOT",
        g_vcp_012_certified=False,  # G-VCP-012 absent/un-certified
        strategy_certified=True,    # Own qualification authority verified
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="MOMENTUM_LEAD",
    )
    assert non_vcp_ev.is_vcp_strategy() is False
    assert non_vcp_ev.governance_authority == "G-STRAT-MOMENTUM_TREND_PILOT"
    st6 = PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        non_vcp_ev,
    )
    assert st6 == PublicationState.STRATEGY_ADMITTED

    # Explicit governance authority can also be assigned for non-VCP strategy
    non_vcp_custom_auth = StrategyAdmissionEvidence(
        symbol="GOOG",
        strategy_id="SMART_MONEY_CANDIDATE_006",
        governance_authority="G-SMC-006",
        strategy_certified=True,
        min_price_floor=100.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="RESEARCH_LEAD",
    )
    assert non_vcp_custom_auth.governance_authority == "G-SMC-006"
    st6_custom = PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        non_vcp_custom_auth,
    )
    assert st6_custom == PublicationState.STRATEGY_ADMITTED

    # Non-VCP strategy with strategy_certified=False fails closed under general policy
    uncertified_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MOMENTUM_TREND_PILOT",
        strategy_certified=False,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="MOMENTUM_LEAD",
    )
    with pytest.raises(PolicyViolationError, match="Strategy certification absent for non-VCP strategy"):
        PublicationStateMachine.transition(
            PublicationState.PUBLISHED_UNIVERSE,
            PublicationState.STRATEGY_ADMITTED,
            uncertified_ev,
        )


def test_neither_strategy_admission_path_allows_scanner_without_g_s2_act():
    """Task 1.3: Neither VCP nor non-VCP strategy admission authorizes scanner dispatch without explicit G-S2-ACT."""
    # 1. Non-VCP strategy in State 6 (STRATEGY_ADMITTED)
    non_vcp_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MOMENTUM_TREND_PILOT",
        strategy_certified=True,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="MOMENTUM_LEAD",
    )
    PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        non_vcp_ev,
    )

    # Attempting scanner activation without G-S2-ACT fails closed
    s2_missing_auth = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="MOMENTUM_SCANNER",
        g_s2_act_authorized=False,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    with pytest.raises(PolicyViolationError, match="G-S2-ACT absent"):
        PublicationStateMachine.transition(
            PublicationState.STRATEGY_ADMITTED,
            PublicationState.ACTIVELY_SCANNED,
            s2_missing_auth,
        )

    # 2. VCP strategy in State 6 (STRATEGY_ADMITTED) under G-VCP-012
    vcp_ev = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_BREAKOUT_SCANNER",
        g_vcp_012_certified=True,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        vcp_ev,
    )

    # Even with G-VCP-012 certified, scanner dispatch still fails closed without G-S2-ACT
    vcp_scanner_missing_s2 = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="VCP_SCANNER",
        g_s2_act_authorized=False,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    with pytest.raises(PolicyViolationError, match="G-S2-ACT absent"):
        PublicationStateMachine.transition(
            PublicationState.STRATEGY_ADMITTED,
            PublicationState.ACTIVELY_SCANNED,
            vcp_scanner_missing_s2,
        )

    # Both succeed when explicit G-S2-ACT authorization is provided
    s2_valid_dispatch = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="VCP_SCANNER",
        g_s2_act_authorized=True,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    st7 = PublicationStateMachine.transition(
        PublicationState.STRATEGY_ADMITTED,
        PublicationState.ACTIVELY_SCANNED,
        s2_valid_dispatch,
    )
    assert st7 == PublicationState.ACTIVELY_SCANNED


# ─── 3. State Machine Monotonicity & Precondition Tests ──────────────────────

def test_state_machine_monotonicity_and_fail_closed():
    """State transitions must strictly progress +1 step forward; state skipping fails closed."""
    # State 1 -> State 3 (skipping State 2)
    with pytest.raises(StateOrderViolationError, match="strictly incremental"):
        PublicationStateMachine.transition(
            PublicationState.REGISTERED_IDENTITY,
            PublicationState.DATA_QUALIFIED,
            None,
        )

    # State 1 -> State 5 (skipping 2, 3, 4)
    with pytest.raises(StateOrderViolationError, match="strictly incremental"):
        PublicationStateMachine.transition(
            PublicationState.REGISTERED_IDENTITY,
            PublicationState.PUBLISHED_UNIVERSE,
            None,
        )

    # State 5 -> State 7 (skipping 6)
    with pytest.raises(StateOrderViolationError, match="strictly incremental"):
        PublicationStateMachine.transition(
            PublicationState.PUBLISHED_UNIVERSE,
            PublicationState.ACTIVELY_SCANNED,
            None,
        )

    # Missing evidence
    with pytest.raises(EvidenceMissingError, match="Missing mandatory evidence"):
        PublicationStateMachine.transition(
            PublicationState.REGISTERED_IDENTITY,
            PublicationState.PROVIDER_RESOLVABLE,
            None,
        )


def test_state_machine_provider_resolution_evidence():
    """State 1 -> 2: Verifies ProviderResolutionEvidence validation."""
    # Ambiguous resolution
    ambig_ev = ProviderResolutionEvidence(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        operating_mic="XNAS",
        alpaca_symbol="AAPL",
        is_ambiguous=True,
        dispute_reason="Conflicting MIC reported by provider",
    )
    with pytest.raises(EvidenceAmbiguityError, match="is ambiguous"):
        PublicationStateMachine.transition(
            PublicationState.REGISTERED_IDENTITY,
            PublicationState.PROVIDER_RESOLVABLE,
            ambig_ev,
        )

    # Missing symbols
    empty_sym_ev = ProviderResolutionEvidence(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="",
        operating_mic="XNAS",
    )
    with pytest.raises(EvidenceMissingError, match="symbol must not be empty"):
        PublicationStateMachine.transition(
            PublicationState.REGISTERED_IDENTITY,
            PublicationState.PROVIDER_RESOLVABLE,
            empty_sym_ev,
        )

    # Missing provider feeds
    no_feed_ev = ProviderResolutionEvidence(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        operating_mic="XNAS",
        alpaca_symbol=None,
        polygon_symbol=None,
    )
    with pytest.raises(EvidenceMissingError, match="No provider symbols resolved"):
        PublicationStateMachine.transition(
            PublicationState.REGISTERED_IDENTITY,
            PublicationState.PROVIDER_RESOLVABLE,
            no_feed_ev,
        )

    # Valid resolution
    valid_ev = ProviderResolutionEvidence(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        operating_mic="XNAS",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        alpaca_symbol="AAPL",
        polygon_symbol="AAPL",
    )
    next_st = PublicationStateMachine.transition(
        PublicationState.REGISTERED_IDENTITY,
        PublicationState.PROVIDER_RESOLVABLE,
        valid_ev,
    )
    assert next_st == PublicationState.PROVIDER_RESOLVABLE


def test_state_machine_market_data_quality_evidence():
    """State 2 -> 3: Verifies MarketDataQualityEvidence validation."""
    # Insufficient history (< 50 bars floor)
    short_ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=45,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
    )
    with pytest.raises(PolicyViolationError, match="Insufficient historical coverage"):
        PublicationStateMachine.transition(
            PublicationState.PROVIDER_RESOLVABLE,
            PublicationState.DATA_QUALIFIED,
            short_ev,
        )

    # Missing live price
    no_price_ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=60,
        has_live_price=False,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
    )
    with pytest.raises(PolicyViolationError, match="Missing live price"):
        PublicationStateMachine.transition(
            PublicationState.PROVIDER_RESOLVABLE,
            PublicationState.DATA_QUALIFIED,
            no_price_ev,
        )

    # Incompatible dividend adjustment
    div_adj_ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=60,
        has_live_price=True,
        adjustment_state="DIVIDEND_AND_SPLIT_ADJUSTED",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
    )
    with pytest.raises(PolicyViolationError, match="Invalid adjustment state"):
        PublicationStateMachine.transition(
            PublicationState.PROVIDER_RESOLVABLE,
            PublicationState.DATA_QUALIFIED,
            div_adj_ev,
        )

    # Degraded structural quality
    degraded_ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=60,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CORRUPT",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
    )
    with pytest.raises(PolicyViolationError, match="Structural data quality degraded or corrupt"):
        PublicationStateMachine.transition(
            PublicationState.PROVIDER_RESOLVABLE,
            PublicationState.DATA_QUALIFIED,
            degraded_ev,
        )

    # Non-SIP feed (e.g. IEX)
    iex_ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=60,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="IEX",
        data_as_of="2026-10-19",
    )
    with pytest.raises(PolicyViolationError, match="Invalid market-data provenance feed"):
        PublicationStateMachine.transition(
            PublicationState.PROVIDER_RESOLVABLE,
            PublicationState.DATA_QUALIFIED,
            iex_ev,
        )

    # Valid quality
    valid_ev = MarketDataQualityEvidence(
        symbol="AAPL",
        candle_count=60,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
    )
    next_st = PublicationStateMachine.transition(
        PublicationState.PROVIDER_RESOLVABLE,
        PublicationState.DATA_QUALIFIED,
        valid_ev,
    )
    assert next_st == PublicationState.DATA_QUALIFIED


def test_state_machine_eligibility_evidence():
    """State 3 -> 4: Verifies EligibilityEvidence validation against canonical R01-R07."""
    # Incomplete rules evaluation
    incomplete_ev = EligibilityEvidence(
        security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        rule_results={"R01_ASSET_CLASS": True, "R02_SECURITY_TYPE": True},
        is_eligible=True,
        reason_code="PASSED",
    )
    with pytest.raises(EvidenceMissingError, match="Eligibility evaluation incomplete"):
        PublicationStateMachine.transition(
            PublicationState.DATA_QUALIFIED,
            PublicationState.PUBLICATION_ELIGIBLE,
            incomplete_ev,
        )

    # Failing R02 (e.g. ETF)
    failing_r02_ev = EligibilityEvidence(
        security_id="ARX_SEC_0B4C728FE0BE",
        symbol="SPY",
        rule_results={
            "R01_ASSET_CLASS": True,
            "R02_SECURITY_TYPE": False,
            "R03_LISTING_STATUS": True,
            "R04_PRIMARY_EXCHANGE": True,
            "R05_CURRENCY": True,
            "R06_PRIMARY_LISTING": True,
            "R07_NOT_DELISTED": True,
        },
        is_eligible=False,
        reason_code="INELIGIBLE_SECURITY_TYPE",
    )
    with pytest.raises(PolicyViolationError, match="failed eligibility rules"):
        PublicationStateMachine.transition(
            PublicationState.DATA_QUALIFIED,
            PublicationState.PUBLICATION_ELIGIBLE,
            failing_r02_ev,
        )

    # Valid eligibility
    valid_ev = EligibilityEvidence(
        security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        rule_results={
            "R01_ASSET_CLASS": True,
            "R02_SECURITY_TYPE": True,
            "R03_LISTING_STATUS": True,
            "R04_PRIMARY_EXCHANGE": True,
            "R05_CURRENCY": True,
            "R06_PRIMARY_LISTING": True,
            "R07_NOT_DELISTED": True,
        },
        is_eligible=True,
        reason_code="PASSED",
    )
    next_st = PublicationStateMachine.transition(
        PublicationState.DATA_QUALIFIED,
        PublicationState.PUBLICATION_ELIGIBLE,
        valid_ev,
    )
    assert next_st == PublicationState.PUBLICATION_ELIGIBLE


def test_state_machine_publication_attestation():
    """State 4 -> 5: Verifies PublicationAttestationEvidence validation."""
    # Quarantine decision
    quarantine_ev = PublicationAttestationEvidence(
        universe_build_id="build-001",
        universe_id=DEFAULT_EXPANSION_NAMESPACE,
        snapshot_id="snap-001",
        publication_digest="a" * 64,
        accounting_closure_verified=True,
        publication_decision="QUARANTINE",
        authorized_by="PO_RELEASE_LEAD",
    )
    with pytest.raises(PolicyViolationError, match="Publication decision is 'QUARANTINE'"):
        PublicationStateMachine.transition(
            PublicationState.PUBLICATION_ELIGIBLE,
            PublicationState.PUBLISHED_UNIVERSE,
            quarantine_ev,
        )

    # Accounting closure unverified
    unclosed_ev = PublicationAttestationEvidence(
        universe_build_id="build-001",
        universe_id=DEFAULT_EXPANSION_NAMESPACE,
        snapshot_id="snap-001",
        publication_digest="a" * 64,
        accounting_closure_verified=False,
        publication_decision="PUBLISH",
        authorized_by="PO_RELEASE_LEAD",
    )
    with pytest.raises(PolicyViolationError, match="Accounting closure not verified"):
        PublicationStateMachine.transition(
            PublicationState.PUBLICATION_ELIGIBLE,
            PublicationState.PUBLISHED_UNIVERSE,
            unclosed_ev,
        )

    # Valid attestation
    valid_ev = PublicationAttestationEvidence(
        universe_build_id="build-001",
        universe_id=DEFAULT_EXPANSION_NAMESPACE,
        snapshot_id="snap-001",
        publication_digest="a" * 64,
        accounting_closure_verified=True,
        publication_decision="PUBLISH",
        authorized_by="PO_RELEASE_LEAD",
    )
    next_st = PublicationStateMachine.transition(
        PublicationState.PUBLICATION_ELIGIBLE,
        PublicationState.PUBLISHED_UNIVERSE,
        valid_ev,
    )
    assert next_st == PublicationState.PUBLISHED_UNIVERSE


# ─── 4. Isolated Publication Foundation Core Tests ───────────────────────────

def test_deterministic_output_under_reordered_input(
    sample_aapl_candidate,
    sample_goog_candidate,
    sample_googl_candidate,
):
    """Publication content digest and sorted order must be bit-identical regardless of input order."""
    list_order_1 = [sample_aapl_candidate, sample_goog_candidate, sample_googl_candidate]
    list_order_2 = [sample_googl_candidate, sample_aapl_candidate, sample_goog_candidate]
    list_order_3 = [sample_goog_candidate, sample_googl_candidate, sample_aapl_candidate]

    digest_1 = IsolatedPublicationEngine.compute_publication_digest(list_order_1)
    digest_2 = IsolatedPublicationEngine.compute_publication_digest(list_order_2)
    digest_3 = IsolatedPublicationEngine.compute_publication_digest(list_order_3)

    assert digest_1 == digest_2 == digest_3
    assert len(digest_1) == 64


def test_duplicate_securities_and_listings(sample_aapl_candidate):
    """Duplicate internal security IDs or active listing locators must be quarantined into exclusion ledger."""
    duplicate_sec_id = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_0B4C728FE0BE",  # Duplicate ID
        symbol="AAPL2",
        operating_mic="XNAS",
    )
    duplicate_locator = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_111111111111",
        symbol="AAPL",  # Duplicate (AAPL, XNAS)
        operating_mic="XNAS",
    )

    candidates = [sample_aapl_candidate, duplicate_sec_id, duplicate_locator]
    deduped, exclusions = IsolatedPublicationEngine.normalize_and_deduplicate(candidates)

    assert len(deduped) == 1
    assert deduped[0].symbol == "AAPL"
    assert len(exclusions) == 2
    assert exclusions[0]["reason"] == "DUPLICATE_INTERNAL_SECURITY_ID"
    assert exclusions[1]["reason"] == "DUPLICATE_LISTING_LOCATOR"


def test_missing_identity_mapping_and_ambiguous_provider_symbols():
    """Missing or ambiguous provider mappings must fail closed to UNRESOLVED state."""
    ambig_cand = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_999999999999",
        symbol="AMBIG",
        operating_mic="XNAS",
        is_ambiguous_identity=True,
    )
    res = IsolatedPublicationEngine.evaluate_candidate(ambig_cand, as_of_date="2026-10-19")
    assert res.eligibility_decision == "UNRESOLVED"
    assert "PROVIDER_RESOLUTION_FAILED" in res.eligibility_reason
    assert res.current_publication_state == PublicationState.REGISTERED_IDENTITY


def test_missing_provenance_and_insufficient_historical_coverage():
    """Unqualified or short historical data must evaluate to DATA_UNAVAILABLE."""
    short_cand = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_888888888888",
        symbol="SHORT",
        operating_mic="XNAS",
        alpaca_symbol="SHORT",
        candle_count=40,  # Below 50 floor
        has_live_price=True,
    )
    res = IsolatedPublicationEngine.evaluate_candidate(short_cand, as_of_date="2026-10-19")
    assert res.data_readiness_decision == "DATA_UNAVAILABLE"
    assert res.current_publication_state == PublicationState.PROVIDER_RESOLVABLE


def test_ineligible_listing_evaluation():
    """Candidates violating rules R01-R07 must evaluate to INELIGIBLE."""
    etf_cand = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_777777777777",
        symbol="SPY",
        operating_mic="XNAS",
        security_type="ETF",
        asset_class="ETF",
        alpaca_symbol="SPY",
        candle_count=287,
        has_live_price=True,
    )
    res = IsolatedPublicationEngine.evaluate_candidate(etf_cand, as_of_date="2026-10-19")
    assert res.eligibility_decision == "INELIGIBLE"
    assert "FAILED_RULES: R01_ASSET_CLASS, R02_SECURITY_TYPE" in res.eligibility_reason
    assert res.current_publication_state == PublicationState.DATA_QUALIFIED


def test_accounting_closure_enforcement(
    sample_aapl_candidate,
    sample_goog_candidate,
    sample_googl_candidate,
):
    """Accounting closure invariants must strictly verify: source == eligible + ineligible + unresolved."""
    engine = IsolatedPublicationEngine()
    candidates = [sample_aapl_candidate, sample_goog_candidate, sample_googl_candidate]

    dry_run = engine.execute_dry_run(candidates, as_of_date="2026-10-19")
    accounting = dry_run["accounting"]

    assert accounting["is_closed"] is True
    assert accounting["source_count"] == 3
    assert accounting["eligible_count"] == 3
    assert accounting["ineligible_count"] == 0
    assert accounting["unresolved_count"] == 0
    assert accounting["data_ready_count"] == 3
    assert accounting["scannable_count"] == 3

    # Tampered accounting closure test
    tampered_summary = AccountingClosureSummary(
        source_count=3,
        eligible_count=2,  # Missing 1
        ineligible_count=0,
        unresolved_count=0,
        data_ready_count=2,
        data_unavailable_count=0,
        data_unresolved_count=0,
        scannable_count=2,
        is_closed=False,
        closure_error="Source closure violation",
    )
    assert tampered_summary.is_closed is False


def test_atomic_publication_pointer_and_rollback_invariants(
    sample_aapl_candidate,
    sample_goog_candidate,
    sample_googl_candidate,
):
    """Publication pointer must be atomic and reversible via rollback_publication."""
    engine = IsolatedPublicationEngine()

    # Initial publication (AAPL only)
    ptr_1 = engine.publish_universe(
        candidates=[sample_aapl_candidate],
        as_of_date="2026-10-19",
        authorized_by="PO_LEAD",
        build_id="build-iso-001",
    )
    assert ptr_1.build_id == "build-iso-001"
    assert ptr_1.published_symbols == ["AAPL"]
    assert ptr_1.previous_build_id is None

    # Second publication (AAPL + GOOG + GOOGL)
    ptr_2 = engine.publish_universe(
        candidates=[sample_aapl_candidate, sample_goog_candidate, sample_googl_candidate],
        as_of_date="2026-10-19",
        authorized_by="PO_LEAD",
        build_id="build-iso-002",
    )
    assert ptr_2.build_id == "build-iso-002"
    assert ptr_2.published_symbols == ["AAPL", "GOOG", "GOOGL"]
    assert ptr_2.previous_build_id == "build-iso-001"

    # Rollback to build-iso-001
    rolled_ptr = engine.rollback_publication(reason="Simulated regression quarantine")
    assert rolled_ptr.build_id == "build-iso-001"
    assert rolled_ptr.published_symbols == ["AAPL"]
    assert rolled_ptr.is_rollback is True
    assert rolled_ptr.rollback_reason == "Simulated regression quarantine"

    # Attempt rollback when history exhausted
    with pytest.raises(PolicyViolationError, match="No historical publication pointer exists"):
        engine.rollback_publication(reason="Exhausted rollback")


def test_population_isolation_pilots_vs_additional_candidates(
    sample_aapl_candidate,
    sample_goog_candidate,
    sample_googl_candidate,
):
    """
    Population Isolation: The 3 pilots qualify, while the 5 additional candidates
    (FIX, DUOL, ULTA, KLAC, DECK) remain unqualified pending Package B data.
    """
    engine = IsolatedPublicationEngine()

    # 5 additional uncertified candidates
    additional_5 = [
        IsolatedUniverseCandidate(
            internal_security_id=f"ARX_SEC_{sym}_TEST",
            symbol=sym,
            operating_mic="XNAS",
            is_qualified_candidate=False,  # Unqualified pending Package B
        )
        for sym in ["FIX", "DUOL", "ULTA", "KLAC", "DECK"]
    ]

    combined_population = [
        sample_aapl_candidate,
        sample_goog_candidate,
        sample_googl_candidate,
        *additional_5,
    ]

    dry_run = engine.execute_dry_run(combined_population, as_of_date="2026-10-19")
    accounting = dry_run["accounting"]

    # 3 pilots eligible and data ready, 5 additional candidates unresolved
    assert accounting["source_count"] == 8
    assert accounting["eligible_count"] == 3
    assert accounting["unresolved_count"] == 5
    assert accounting["data_ready_count"] == 3
    assert dry_run["published_symbols"] == ["AAPL", "GOOG", "GOOGL"]


def test_historical_fixture_isolation():
    """Historical 35 regression fixture cannot be silently substituted for live universe authority."""
    # Direct substitution by universe_id
    with pytest.raises(PolicyViolationError, match="Historical regression fixture cannot be used"):
        IsolatedPublicationEngine.verify_historical_fixture_isolation(
            "HISTORICAL_VCP_35_REGRESSION_FIXTURE",
            HISTORICAL_VCP_35_REGRESSION_FIXTURE,
        )

    # Silent symbol clone outside isolated expansion namespace
    with pytest.raises(PolicyViolationError, match="Silent substitution of 35-symbol regression fixture detected"):
        IsolatedPublicationEngine.verify_historical_fixture_isolation(
            "UNAUTHORIZED_LIVE_NAMESPACE",
            HISTORICAL_VCP_35_REGRESSION_FIXTURE,
        )
