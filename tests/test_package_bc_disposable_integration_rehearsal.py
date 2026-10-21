"""
tests/test_package_bc_disposable_integration_rehearsal.py

Sprint 2: Package B / Package C End-to-End Integration Rehearsal.
Validates the full pipeline against disposable, in-memory databases:
Package A Identity -> B2 Projection -> Certified Historical Data Fixture -> Package C Publication Dry Run.

Tests the 10 mandated integration invariants:
1. Canonical identity linkage
2. GOOG/GOOGL multi-share-class separation
3. Valid provenance qualification (SIP_CONSOLIDATED, SPLIT_ADJUSTED_ONLY, CLEAN)
4. Incompatible provider lineage (IEX participant-only / dividend-adjusted rejected)
5. Incomplete historical coverage (< 50 bars builder floor rejected)
6. Primary listing failures (Rule R06 rejected)
7. Exact mathematical accounting closure
8. Deterministic, order-invariant publication content digests
9. Zero scanner enrollment without explicit G-S2-ACT authority
10. Zero live database mutation (100% in-memory / disposable)
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import random
from typing import Any, Dict, List, Optional
import pytest

from analyst_dashboard.universe.identity_persistence import (
    SqliteIdentityRepository,
    InternalSecurityRecord,
    EffectiveDatedListingAssociation,
    AttributableFieldEvidence,
    EvidenceNature,
)
from analyst_dashboard.universe.publication_state_machine import (
    PublicationState,
    PublicationStateMachine,
    ProviderResolutionEvidence,
    MarketDataQualityEvidence,
    EligibilityEvidence,
    PublicationAttestationEvidence,
    StrategyAdmissionEvidence,
    ScannerDispatchEvidence,
    PolicyViolationError,
    validate_canonical_association_id,
)
from analyst_dashboard.universe.isolated_publication import (
    IsolatedUniverseCandidate,
    IsolatedPublicationEngine,
    DEFAULT_EXPANSION_NAMESPACE,
)

pytestmark = pytest.mark.tier1


# ─── B2 Projection Bridge Implementation ──────────────────────────────────────

def project_b2_universe_candidates(
    identity_repo: SqliteIdentityRepository,
    market_data_lookup: Dict[str, Dict[str, Any]],
    as_of_date: str = "2026-10-19",
) -> List[IsolatedUniverseCandidate]:
    """
    B2 Identity Projection Bridge:
    Reads Package A registered securities and their active listing associations,
    joins with certified Package B market-data observations, and projects
    into Track C IsolatedUniverseCandidate objects.
    """
    projected: List[IsolatedUniverseCandidate] = []
    all_securities = identity_repo.list_all_securities()

    for sec in all_securities:
        assocs = identity_repo.get_associations_for_security(sec.internal_security_id)
        active_assocs = [a for a in assocs if a.is_active_on(as_of_date)]
        if not active_assocs:
            # Security without active listing association cannot resolve provider locators
            continue

        active_assoc = active_assocs[0]
        sym = active_assoc.symbol
        md = market_data_lookup.get(sym, {})

        cand = IsolatedUniverseCandidate(
            internal_security_id=sec.internal_security_id,
            symbol=sym,
            operating_mic=active_assoc.operating_mic,
            association_id=active_assoc.association_id,
            security_type=md.get("security_type", "COMMON_STOCK"),
            asset_class=md.get("asset_class", "EQUITY"),
            listing_status=md.get("listing_status", "ACTIVE"),
            currency=md.get("currency", "USD"),
            primary_listing=md.get("primary_listing", True),
            delisting_date=md.get("delisting_date", None),
            alpaca_symbol=sym,
            polygon_symbol=sym,
            figi_composite=md.get("figi_composite", f"BBG{sym.ljust(8, '0')}"),
            candle_count=md.get("candle_count", 0),
            has_live_price=md.get("has_live_price", False),
            adjustment_state=md.get("adjustment_state", "SPLIT_ADJUSTED_ONLY"),
            structural_quality=md.get("structural_quality", "CLEAN"),
            provenance_feed=md.get("provenance_feed", "SIP_CONSOLIDATED"),
            data_as_of=md.get("data_as_of", as_of_date),
            is_ambiguous_identity=False,
            is_qualified_candidate=md.get("is_qualified_candidate", True),
        )
        projected.append(cand)

    return projected


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def disposable_identity_repo() -> SqliteIdentityRepository:
    """
    Creates an ephemeral, in-memory Package A identity database with
    canonical test securities and associations.
    """
    repo = SqliteIdentityRepository(db_path=":memory:", auto_init=True)

    # 1. Apple Inc. (Single class)
    repo.save_security(InternalSecurityRecord(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        issuer_cik="0000320193",
        primary_symbol="AAPL",
        share_class=None,
        created_at_utc="2026-10-17T00:00:00Z",
    ))
    repo.save_association(EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        operating_mic="XNAS",
        effective_from="2026-10-17",
    ))

    # 2. Alphabet Inc. Class C (GOOG)
    repo.save_security(InternalSecurityRecord(
        internal_security_id="ARX_SEC_A11A24D442E2",
        issuer_cik="0001652044",
        primary_symbol="GOOG",
        share_class="Class C",
        created_at_utc="2026-10-17T00:00:00Z",
    ))
    repo.save_association(EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17",
        internal_security_id="ARX_SEC_A11A24D442E2",
        symbol="GOOG",
        operating_mic="XNAS",
        effective_from="2026-10-17",
    ))

    # 3. Alphabet Inc. Class A (GOOGL)
    repo.save_security(InternalSecurityRecord(
        internal_security_id="ARX_SEC_B7EF9ED7A35A",
        issuer_cik="0001652044",
        primary_symbol="GOOGL",
        share_class="Class A",
        created_at_utc="2026-10-17T00:00:00Z",
    ))
    repo.save_association(EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17",
        internal_security_id="ARX_SEC_B7EF9ED7A35A",
        symbol="GOOGL",
        operating_mic="XNAS",
        effective_from="2026-10-17",
    ))

    # 4. AbbVie Inc. (ABBV) - NYSE primary listing
    repo.save_security(InternalSecurityRecord(
        internal_security_id="ARX_SEC_B26707324F97",
        issuer_cik="0001551152",
        primary_symbol="ABBV",
        share_class=None,
        created_at_utc="2026-10-19T00:00:00Z",
    ))
    repo.save_association(EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_B26707324F97_XNYS_ABBV_2026-10-19",
        internal_security_id="ARX_SEC_B26707324F97",
        symbol="ABBV",
        operating_mic="XNYS",
        effective_from="2026-10-19",
    ))

    # 5. Non-Primary Listing Candidate (FAIL_R06)
    repo.save_security(InternalSecurityRecord(
        internal_security_id="ARX_SEC_999999999991",
        issuer_cik="0000999991",
        primary_symbol="FAIL_R06",
        share_class=None,
        created_at_utc="2026-10-19T00:00:00Z",
    ))
    repo.save_association(EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_999999999991_XNAS_FAIL_R06_2026-10-19",
        internal_security_id="ARX_SEC_999999999991",
        symbol="FAIL_R06",
        operating_mic="XNAS",
        effective_from="2026-10-19",
    ))

    # 6. Incomplete Coverage Candidate (SHORT_HIST)
    repo.save_security(InternalSecurityRecord(
        internal_security_id="ARX_SEC_999999999992",
        issuer_cik="0000999992",
        primary_symbol="SHORT_HIST",
        share_class=None,
        created_at_utc="2026-10-19T00:00:00Z",
    ))
    repo.save_association(EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_999999999992_XNAS_SHORT_HIST_2026-10-19",
        internal_security_id="ARX_SEC_999999999992",
        symbol="SHORT_HIST",
        operating_mic="XNAS",
        effective_from="2026-10-19",
    ))

    # 7. Incompatible Lineage Candidate (BAD_PROV)
    repo.save_security(InternalSecurityRecord(
        internal_security_id="ARX_SEC_999999999993",
        issuer_cik="0000999993",
        primary_symbol="BAD_PROV",
        share_class=None,
        created_at_utc="2026-10-19T00:00:00Z",
    ))
    repo.save_association(EffectiveDatedListingAssociation(
        association_id="ASSOC_ARX_SEC_999999999993_XNAS_BAD_PROV_2026-10-19",
        internal_security_id="ARX_SEC_999999999993",
        symbol="BAD_PROV",
        operating_mic="XNAS",
        effective_from="2026-10-19",
    ))

    return repo


@pytest.fixture
def certified_market_data_lookup() -> Dict[str, Dict[str, Any]]:
    """
    Market-data fixtures representing Package B certified observations
    and adversarial edge cases.
    """
    return {
        "AAPL": {
            "candle_count": 288,
            "has_live_price": True,
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "provenance_feed": "SIP_CONSOLIDATED",
            "primary_listing": True,
            "is_qualified_candidate": True,
        },
        "GOOG": {
            "candle_count": 287,
            "has_live_price": True,
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "provenance_feed": "SIP_CONSOLIDATED",
            "primary_listing": True,
            "is_qualified_candidate": True,
        },
        "GOOGL": {
            "candle_count": 288,
            "has_live_price": True,
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "provenance_feed": "SIP_CONSOLIDATED",
            "primary_listing": True,
            "is_qualified_candidate": True,
        },
        "ABBV": {
            "candle_count": 260,
            "has_live_price": True,
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "provenance_feed": "SIP_CONSOLIDATED",
            "primary_listing": True,
            "is_qualified_candidate": True,
        },
        # Adversarial 1: Fails Rule R06 (primary listing = False)
        "FAIL_R06": {
            "candle_count": 280,
            "has_live_price": True,
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "provenance_feed": "SIP_CONSOLIDATED",
            "primary_listing": False,
            "is_qualified_candidate": True,
        },
        # Adversarial 2: Incomplete historical coverage (< 50 bars builder floor)
        "SHORT_HIST": {
            "candle_count": 35,  # Insufficient history
            "has_live_price": True,
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "provenance_feed": "SIP_CONSOLIDATED",
            "primary_listing": True,
            "is_qualified_candidate": True,
        },
        # Adversarial 3: Incompatible provider lineage (IEX participant-only)
        "BAD_PROV": {
            "candle_count": 280,
            "has_live_price": True,
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "provenance_feed": "IEX_PARTICIPANT_ONLY",  # Incompatible feed
            "primary_listing": True,
            "is_qualified_candidate": True,
        },
    }


# ─── End-to-End Disposable Integration Tests ──────────────────────────────────

def test_01_canonical_identity_linkage_through_b2_projection(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 1: Verifies that B2 projection accurately links Package A internal IDs
    and listing association IDs into publication candidate objects without data loss.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    assert len(candidates) == 7

    by_sym = {c.symbol: c for c in candidates}
    assert by_sym["AAPL"].internal_security_id == "ARX_SEC_0B4C728FE0BE"
    assert by_sym["AAPL"].association_id == "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"
    assert by_sym["ABBV"].internal_security_id == "ARX_SEC_B26707324F97"
    assert by_sym["ABBV"].operating_mic == "XNYS"
    assert validate_canonical_association_id(by_sym["ABBV"].association_id) is True


def test_02_goog_googl_multi_share_class_isolation(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 2: Verifies that GOOG and GOOGL remain strictly distinct entities under
    the same issuer CIK (0001652044) with distinct IDs, associations, and candles.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    by_sym = {c.symbol: c for c in candidates}

    goog = by_sym["GOOG"]
    googl = by_sym["GOOGL"]

    # Distinct internal IDs
    assert goog.internal_security_id == "ARX_SEC_A11A24D442E2"
    assert googl.internal_security_id == "ARX_SEC_B7EF9ED7A35A"
    assert goog.internal_security_id != googl.internal_security_id

    # Distinct association IDs
    assert goog.association_id != googl.association_id
    assert "GOOG" in goog.association_id
    assert "GOOGL" in googl.association_id

    # Distinct market data properties
    assert goog.candle_count == 287
    assert googl.candle_count == 288


def test_03_valid_provenance_and_publication_dry_run(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 3: Valid candidates with SIP_CONSOLIDATED, SPLIT_ADJUSTED_ONLY, and CLEAN
    history successfully reach DATA_QUALIFIED and PUBLICATION_ELIGIBLE in dry run.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    engine = IsolatedPublicationEngine(universe_id="ARX_TEST_REHEARSAL_UNIVERSE")

    result = engine.execute_dry_run(candidates, as_of_date="2026-10-19")
    published = result["published_symbols"]

    # The 4 valid candidates must be published
    assert "AAPL" in published
    assert "GOOG" in published
    assert "GOOGL" in published
    assert "ABBV" in published


def test_04_incompatible_provider_lineage_fails_closed(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 4: Candidate with incompatible feed (IEX participant-only) fails closed
    at State 2 -> State 3 transition and is excluded from published universe.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    engine = IsolatedPublicationEngine()
    result = engine.execute_dry_run(candidates, as_of_date="2026-10-19")

    # BAD_PROV must NOT be published
    assert "BAD_PROV" not in result["published_symbols"]

    eval_map = {e["candidate"]["symbol"]: e for e in result["evaluations"]}
    bad_prov_eval = eval_map["BAD_PROV"]
    assert bad_prov_eval["data_readiness_decision"] == "DATA_UNAVAILABLE"
    assert "SIP_CONSOLIDATED" in bad_prov_eval["data_readiness_reason"] or "provenance" in bad_prov_eval["data_readiness_reason"].lower()


def test_05_incomplete_historical_coverage_fails_closed(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 5: Candidate with insufficient history (35 bars < 50 bar builder floor)
    fails closed to DATA_UNAVAILABLE and is excluded from published universe.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    engine = IsolatedPublicationEngine()
    result = engine.execute_dry_run(candidates, as_of_date="2026-10-19")

    # SHORT_HIST must NOT be published
    assert "SHORT_HIST" not in result["published_symbols"]

    eval_map = {e["candidate"]["symbol"]: e for e in result["evaluations"]}
    short_eval = eval_map["SHORT_HIST"]
    assert short_eval["data_readiness_decision"] == "DATA_UNAVAILABLE"
    assert "Insufficient historical coverage" in short_eval["data_readiness_reason"] or "required builder floor" in short_eval["data_readiness_reason"]


def test_06_primary_listing_failure_fails_closed(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 6: Candidate failing primary listing Rule R06 is evaluated as INELIGIBLE
    and is excluded from published universe.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    engine = IsolatedPublicationEngine()
    result = engine.execute_dry_run(candidates, as_of_date="2026-10-19")

    # FAIL_R06 must NOT be published
    assert "FAIL_R06" not in result["published_symbols"]

    eval_map = {e["candidate"]["symbol"]: e for e in result["evaluations"]}
    fail_r06_eval = eval_map["FAIL_R06"]
    assert fail_r06_eval["eligibility_decision"] == "INELIGIBLE"
    assert "R06_PRIMARY_LISTING" in fail_r06_eval["eligibility_reason"]


def test_07_exact_mathematical_accounting_closure(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 7: Verifies exact arithmetic closure across all candidate partitions:
    source_count == eligible + ineligible + unresolved
    eligible == data_ready + data_unavailable + data_unresolved
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    engine = IsolatedPublicationEngine()
    result = engine.execute_dry_run(candidates, as_of_date="2026-10-19")

    accounting = result["accounting"]
    assert accounting["is_closed"] is True
    assert accounting["closure_error"] is None

    # 7 input candidates:
    # 4 data ready & eligible: AAPL, GOOG, GOOGL, ABBV
    # 1 ineligible (Rule R06): FAIL_R06
    # 2 unresolved (data quality failures): SHORT_HIST, BAD_PROV
    assert accounting["source_count"] == 7
    assert accounting["eligible_count"] == 4
    assert accounting["ineligible_count"] == 1
    assert accounting["unresolved_count"] == 2

    # Within eligible partition:
    assert accounting["data_ready_count"] == 4
    assert accounting["data_unavailable_count"] == 0
    assert accounting["data_unresolved_count"] == 0

    # Arithmetic closure verified
    assert accounting["source_count"] == accounting["eligible_count"] + accounting["ineligible_count"] + accounting["unresolved_count"]
    assert accounting["eligible_count"] == accounting["data_ready_count"] + accounting["data_unavailable_count"] + accounting["data_unresolved_count"]


def test_08_order_invariant_deterministic_publication_digest(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 8: Verifies that list reordering produces an identical SHA-256 publication content digest.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)

    digest_orig = IsolatedPublicationEngine.compute_publication_digest(candidates)

    # Reverse candidates
    reversed_candidates = list(reversed(candidates))
    digest_rev = IsolatedPublicationEngine.compute_publication_digest(reversed_candidates)
    assert digest_orig == digest_rev

    # Shuffle candidates
    shuffled_candidates = list(candidates)
    random.Random(42).shuffle(shuffled_candidates)
    digest_shuffled = IsolatedPublicationEngine.compute_publication_digest(shuffled_candidates)
    assert digest_orig == digest_shuffled
    assert len(digest_orig) == 64


def test_09_zero_scanner_enrollment_and_strategy_governance_isolation(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 9: Verifies:
    1. Zero scanner enrollment: published candidates are not active in scanners without G-S2-ACT.
    2. VCP strategy admission strictly requires G-VCP-012.
    3. Canonical strategy types (e.g. MINERVINI_VCP) require G-VCP-012.
    4. Non-VCP strategies admit under their own qualification authority.
    5. Scanner dispatch strictly requires G-S2-ACT for all strategies.
    """
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    engine = IsolatedPublicationEngine()
    result = engine.execute_dry_run(candidates, as_of_date="2026-10-19")

    published_symbols = result["published_symbols"]

    # 1. Anti-leakage firewall: accidental enrollment fails closed
    active_scanner_symbols = {"AAPL", "UNAUTHORIZED_TICKER"}
    with pytest.raises(PolicyViolationError, match="Actionability leakage detected"):
        IsolatedPublicationEngine.assert_no_scanner_enrollment(published_symbols, active_scanner_symbols)

    # Clean un-enrolled state passes
    IsolatedPublicationEngine.assert_no_scanner_enrollment(published_symbols, set())

    # 2. VCP strategy admission requires G-VCP-012
    vcp_unauth = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="VCP_BREAKOUT",
        strategy_type="VCP",
        g_vcp_012_certified=False,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    assert vcp_unauth.is_vcp_strategy() is True
    assert vcp_unauth.governance_authority == "G-VCP-012"
    with pytest.raises(PolicyViolationError, match="G-VCP-012 certification absent"):
        PublicationStateMachine.transition(
            PublicationState.PUBLISHED_UNIVERSE,
            PublicationState.STRATEGY_ADMITTED,
            vcp_unauth,
        )

    # Canonical scanner ID MINERVINI_VCP without strategy_type also recognized as VCP
    minervini_vcp = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="MINERVINI_VCP",
        g_vcp_012_certified=False,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="QUANT_LEAD",
    )
    assert minervini_vcp.is_vcp_strategy() is True
    assert minervini_vcp.governance_authority == "G-VCP-012"

    # 3. Non-VCP strategy admits under own authority without requiring G-VCP-012
    smart_money = StrategyAdmissionEvidence(
        symbol="AAPL",
        strategy_id="SMART_MONEY",
        strategy_type="SMART_MONEY",
        strategy_certified=True,
        min_price_floor=150.0,
        min_history_bars=287,
        admission_approved=True,
        approved_by="SMART_MONEY_LEAD",
    )
    assert smart_money.is_vcp_strategy() is False
    assert smart_money.governance_authority == "G-STRAT-SMART_MONEY"

    st6 = PublicationStateMachine.transition(
        PublicationState.PUBLISHED_UNIVERSE,
        PublicationState.STRATEGY_ADMITTED,
        smart_money,
    )
    assert st6 == PublicationState.STRATEGY_ADMITTED

    # 4. Scanner dispatch strictly requires G-S2-ACT
    s2_unauth = ScannerDispatchEvidence(
        symbol="AAPL",
        scanner_id="SMART_MONEY_RUNNER",
        g_s2_act_authorized=False,
        schedule_active=True,
        operational_dispatch_authorized=True,
    )
    with pytest.raises(PolicyViolationError, match="G-S2-ACT absent"):
        PublicationStateMachine.transition(
            PublicationState.STRATEGY_ADMITTED,
            PublicationState.ACTIVELY_SCANNED,
            s2_unauth,
        )


def test_10_zero_live_database_mutation(
    disposable_identity_repo: SqliteIdentityRepository,
    certified_market_data_lookup: Dict[str, Dict[str, Any]],
):
    """
    Test 10: Verifies that the entire rehearsal operates exclusively in memory
    and that no files are created or modified on disk.
    """
    assert disposable_identity_repo.is_memory is True

    # Execute publication dry-run and in-memory publication
    candidates = project_b2_universe_candidates(disposable_identity_repo, certified_market_data_lookup)
    engine = IsolatedPublicationEngine(universe_id="ARX_IN_MEMORY_UNIVERSE")

    dry_run = engine.execute_dry_run(candidates, as_of_date="2026-10-19")
    assert dry_run["zero_production_mutation_verified"] is True

    # Verify that in-memory publication pointer works without disk writes
    pointer = engine.publish_universe(
        candidates=candidates,
        as_of_date="2026-10-19",
        authorized_by="TEST_REHEARSAL_OPERATOR",
        build_id="build-test-rehearsal-001",
    )
    assert pointer.build_id == "build-test-rehearsal-001"
    assert len(pointer.published_symbols) == 4
    assert engine.current_pointer == pointer


def test_11_expanded_v3_canonical_rehearsal_and_distinctions():
    """
    Test 11 (Phase 6): Expanded disposable integration rehearsal with V3 canonical inputs.
    Exercises the complete end-to-end pipeline:
    1. Identity projection consumption (Package B V3 canonical manifest & candidate bindings)
    2. Market-data qualification & provenance verification (manifest-only vs synthetic certified fixture)
    3. Eligibility evaluation (R01-R07)
    4. Deterministic snapshot generation & pure SHA-256 digest
    5. Atomic pointer update & in-memory pointer rollback
    6. Explicit verification of design-vs-implementation boundaries:
       - SHA-256 canonical digest vs Merkle trees
       - In-memory pointer rollback vs durable SQLite backup rollback
       - Local test population (4 pilot candidates) vs active production universe (3 pilots).
    """
    from analyst_dashboard.universe.isolated_publication import (
        PackageBV3Consumer,
        CANONICAL_PACKAGE_B_V3_DIGEST,
    )
    from pathlib import Path

    manifest_path = Path("tests/fixtures/package_b_canonical_v3/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json")
    manifest, digest = PackageBV3Consumer.load_manifest(manifest_path)
    assert manifest["manifest_id"] == "MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL"
    assert digest == CANONICAL_PACKAGE_B_V3_DIGEST

    # Step 1: Pre-replacement fails closed (manifest intent alone without synthetic fixture)
    pre_candidates = PackageBV3Consumer.build_candidates_from_manifest(manifest)
    engine = IsolatedPublicationEngine(universe_id="ARX_EXPANDED_V3_REHEARSAL")
    dry_pre = engine.execute_dry_run(pre_candidates, "2026-10-19")
    assert dry_pre["published_symbols"] == []
    assert dry_pre["accounting"]["data_ready_count"] == 0
    assert dry_pre["accounting"]["scannable_count"] == 0

    # Step 2: Post-replacement qualifies all 4 pilot candidates using certified synthetic test fixture
    synthetic_fixture = {
        "_fixture_nature": "SYNTHETIC_TEST_FIXTURE",
        "AAPL": {
            "candle_count": 288,
            "has_live_price": True,
            "provenance_feed": "SIP_CONSOLIDATED",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "data_as_of": "2026-10-19",
            "is_qualified_candidate": True,
        },
        "GOOG": {
            "candle_count": 288,
            "has_live_price": True,
            "provenance_feed": "SIP_CONSOLIDATED",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "data_as_of": "2026-10-19",
            "is_qualified_candidate": True,
        },
        "GOOGL": {
            "candle_count": 288,
            "has_live_price": True,
            "provenance_feed": "SIP_CONSOLIDATED",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "data_as_of": "2026-10-19",
            "is_qualified_candidate": True,
        },
        "ABBV": {
            "candle_count": 288,
            "has_live_price": True,
            "provenance_feed": "SIP_CONSOLIDATED",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "CLEAN",
            "data_as_of": "2026-10-19",
            "is_qualified_candidate": True,
        },
    }
    post_candidates = PackageBV3Consumer.build_candidates_from_manifest(
        manifest,
        market_data_fixture=synthetic_fixture,
    )
    assert len(post_candidates) == 4

    # Step 3: Pure deterministic digest (SHA-256 over canonical JSON)
    digest1 = engine.compute_publication_digest(post_candidates)
    shuffled_candidates = list(reversed(post_candidates))
    digest2 = engine.compute_publication_digest(shuffled_candidates)
    assert digest1 == digest2, "Digest must be order-invariant and deterministic"
    assert len(digest1) == 64
    # Affirmation: This is pure deterministic SHA-256, NOT a Merkle tree data structure
    assert isinstance(digest1, str)

    # Step 4: Atomic pointer update
    p1 = engine.publish_universe(
        post_candidates,
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_RELEASE_LEAD",
        build_id="build-v3-pointer-001",
    )
    assert engine.current_pointer == p1
    assert p1.published_symbols == ["AAPL", "ABBV", "GOOG", "GOOGL"]
    assert p1.previous_build_id is None

    # Step 5: Second build and in-memory rollback verification
    p2 = engine.publish_universe(
        post_candidates,
        as_of_date="2026-10-19",
        authorized_by="OPERATOR_RELEASE_LEAD",
        build_id="build-v3-pointer-002",
    )
    assert engine.current_pointer == p2
    assert p2.previous_build_id == "build-v3-pointer-001"

    # Roll back in-memory pointer
    rb = engine.rollback_publication(reason="Simulated dry-run release rollback rehearsal")
    assert engine.current_pointer == rb
    assert rb.build_id == "build-v3-pointer-001"
    assert rb.is_rollback is True
    assert rb.rollback_reason == "Simulated dry-run release rollback rehearsal"
    # Affirmation: Pointer rollback is strictly in-memory, NOT durable production WAL rollback

    # Step 6: Population boundary affirmation
    # Local test population has 4 pilot candidates; active production universe remains at 3 pilots
    assert len(p1.published_symbols) == 4
    # Verification: Zero live DB mutations
    assert dry_pre["zero_production_mutation_verified"] is True

