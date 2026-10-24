"""
tests/test_package_b_v3_canonical_consumer.py

Sprint 2 Track C: Direct Package B V3 Canonical Consumer Integration Tests.
Verifies the 14 mandatory contract and fail-closed requirements:
1. Canonical V3 manifest digest accepted (0F54C702..., 9,972 bytes).
2. Superseded predecessor Workstream A digest rejected (65F5D9AB...).
3. Superseded predecessor Workstream B digest rejected (8923DAD9...).
4. Tampered manifest fails closed on altered bytes/digest.
5. Canonical identity associations and bindings validated.
6. Manifest alone does NOT imply ingestion (candle_count=0, unprovenanced).
7. Manifest alone does NOT imply data certification (fails State 2->3).
8. Synthetic test fixture requires explicit _fixture_nature='SYNTHETIC_TEST_FIXTURE'.
9. Uncertified production-like data fails closed (DATA_UNAVAILABLE).
10. AAPL October 20 unprovenanced tail fails closed and cannot qualify.
11. GOOG/GOOGL share-class separation verified via identity; price equality does not trigger failure.
12. Published-only candidates have zero scanner enrollment (assert_no_scanner_enrollment).
13. Lookback threshold boundaries remain 50/60/260.
14. Deterministic publication digest and arithmetic accounting closure.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pytest

from analyst_dashboard.universe.isolated_publication import (
    IsolatedUniverseCandidate,
    IsolatedPublicationEngine,
    PackageBV3Consumer,
    CANONICAL_PACKAGE_B_V3_DIGEST,
    CANONICAL_PACKAGE_B_V3_BYTE_LENGTH,
    SUPERSEDED_PREDECESSOR_DIGESTS,
)
from analyst_dashboard.universe.publication_state_machine import (
    PublicationState,
    MarketDataQualityEvidence,
    PolicyViolationError,
    validate_canonical_association_id,
)

MANIFEST_FIXTURE_PATH = Path("tests/fixtures/package_b_canonical_v3/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json")


def test_01_canonical_v3_manifest_digest_accepted():
    """
    Test 1: Canonical V3 manifest digest accepted.
    Authoritative Option B Full-History ruling:
    - Path: tests/fixtures/package_b_canonical_v3/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json
    - Digest: 0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF
    - Byte length: 9,972 bytes
    - Manifest ID: MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL
    - Authorization state: PROPOSED_NOT_AUTHORIZED
    """
    assert MANIFEST_FIXTURE_PATH.is_file(), f"Canonical manifest not found at {MANIFEST_FIXTURE_PATH}"

    raw_bytes = MANIFEST_FIXTURE_PATH.read_bytes()
    assert len(raw_bytes) == CANONICAL_PACKAGE_B_V3_BYTE_LENGTH == 9972

    manifest, computed_digest = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)
    assert computed_digest == CANONICAL_PACKAGE_B_V3_DIGEST
    assert manifest["manifest_id"] == "MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL"

    gov = manifest["governance_status"]
    assert gov["authorization_state"] == "PROPOSED_NOT_AUTHORIZED"
    assert gov["is_authorized"] is False
    assert gov["authorized_by"] is None
    assert gov["operational_authority"] == "G-S2-ACT"
    assert gov["deferred_gate_reference"] == "G-VCP-012"

    window = manifest["acquisition_window"]
    assert window["requested_start_date"] == "2025-08-27"
    assert window["requested_end_date"] == "2026-10-19"
    assert window["expected_nyse_sessions_count"] == 288

    provider_cfg = manifest["provider_configuration"]
    assert provider_cfg["tape_feed"] == "sip"
    assert provider_cfg["adjustment_convention"] == "split"


def test_02_superseded_predecessor_workstream_a_digest_rejected(tmp_path):
    """
    Test 2: Superseded predecessor Workstream A digest (65F5D9AB..., 3,489 bytes)
    is formally invalidated and must fail closed.
    """
    superseded_digest_a = "65f5d9ab185056fd8c092ad1601a6db195494063b5ccfce562a11aad8aa063cd"
    assert superseded_digest_a in SUPERSEDED_PREDECESSOR_DIGESTS

    # Rejection when passed as expected_digest
    mock_file = tmp_path / "predecessor_a.json"
    mock_file.write_text('{"manifest_id": "PREDECESSOR_A"}', encoding="utf-8")

    with pytest.raises(PolicyViolationError, match="superseded predecessor digest"):
        PackageBV3Consumer.load_manifest(
            mock_file,
            expected_digest=superseded_digest_a,
        )


def test_03_superseded_predecessor_workstream_b_digest_rejected(tmp_path):
    """
    Test 3: Superseded predecessor Workstream B digest (8923DAD9..., 13,833 bytes)
    is formally invalidated due to historical boundary defect and must fail closed.
    """
    superseded_digest_b = "8923dad988a10ae07a2b7bda1044127130bf9c90539424d98f25676a9a59f07b"
    assert superseded_digest_b in SUPERSEDED_PREDECESSOR_DIGESTS

    mock_file = tmp_path / "predecessor_b.json"
    mock_file.write_text('{"manifest_id": "PREDECESSOR_B"}', encoding="utf-8")

    with pytest.raises(PolicyViolationError, match="superseded predecessor digest"):
        PackageBV3Consumer.load_manifest(
            mock_file,
            expected_digest=superseded_digest_b,
        )


def test_04_tampered_manifest_fails_closed(tmp_path):
    """
    Test 4: Tampered manifest (altered bytes or altered fields) fails closed.
    """
    manifest_raw = MANIFEST_FIXTURE_PATH.read_text(encoding="utf-8")
    # Tamper with the window date
    tampered_json = manifest_raw.replace('"requested_start_date": "2025-08-27"', '"requested_start_date": "2025-09-01"')
    tampered_file = tmp_path / "tampered_manifest.json"
    tampered_file.write_text(tampered_json, encoding="utf-8")

    # Digest mismatch fails closed
    with pytest.raises(PolicyViolationError, match="Package B V3 manifest digest mismatch"):
        PackageBV3Consumer.load_manifest(tampered_file)


def test_05_identity_association_validation():
    """
    Test 5: Validates candidate bindings from canonical manifest:
    - AAPL: ARX_SEC_0B4C728FE0BE, ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17, XNAS
    - GOOG: ARX_SEC_A11A24D442E2, ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17, XNAS
    - GOOGL: ARX_SEC_B7EF9ED7A35A, ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17, XNAS
    - ABBV: ARX_SEC_BFA51AC4C0B5, ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19, XNYS
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)
    pilot_candidates = {c["symbol"]: c for c in manifest["pilot_candidates"]}

    expected_bindings = {
        "AAPL": {
            "durable_security_id": "ARX_SEC_0B4C728FE0BE",
            "operating_mic": "XNAS",
            "listing_association_id": "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
            "mandated_action": "REPLACE",
        },
        "GOOG": {
            "durable_security_id": "ARX_SEC_A11A24D442E2",
            "operating_mic": "XNAS",
            "listing_association_id": "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17",
            "mandated_action": "REPLACE",
        },
        "GOOGL": {
            "durable_security_id": "ARX_SEC_B7EF9ED7A35A",
            "operating_mic": "XNAS",
            "listing_association_id": "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17",
            "mandated_action": "REPLACE",
        },
        "ABBV": {
            "durable_security_id": "ARX_SEC_BFA51AC4C0B5",
            "operating_mic": "XNYS",
            "listing_association_id": "ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19",
            "mandated_action": "BACKFILL",
        },
    }

    assert set(pilot_candidates.keys()) == set(expected_bindings.keys())

    for sym, exp in expected_bindings.items():
        cand = pilot_candidates[sym]
        assert cand["durable_security_id"] == exp["durable_security_id"]
        assert cand["operating_mic"] == exp["operating_mic"]
        assert cand["listing_association_id"] == exp["listing_association_id"]
        assert cand["mandated_action"] == exp["mandated_action"]
        assert validate_canonical_association_id(cand["listing_association_id"]) is True

    # Corrupted association format fails validation
    assert validate_canonical_association_id("ASSOC_CORRUPTED_IDENTIFIER") is False
    assert validate_canonical_association_id("ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL") is False
    assert validate_canonical_association_id("ASSOC_ARX_SEC_TOOLONGHEX00000_XNAS_AAPL_2026-10-17") is False


def test_06_manifest_alone_does_not_imply_ingestion():
    """
    Test 6: Valid manifest alone does NOT imply ingestion.
    Default candidate build yields candle_count=0, has_live_price=False, unprovenanced.
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)
    candidates = PackageBV3Consumer.build_candidates_from_manifest(manifest)
    assert len(candidates) == 4

    for c in candidates:
        assert c.candle_count == 0
        assert c.has_live_price is False
        assert c.provenance_feed == "UNVERIFIED"
        assert c.adjustment_state == "UNVERIFIED"
        assert c.structural_quality == "UNVERIFIED"
        assert c.data_as_of is None
        assert c.is_qualified_candidate is False


def test_07_manifest_alone_does_not_imply_certification():
    """
    Test 7: Valid manifest alone does NOT imply certification.
    Candidates cannot pass State 2 -> State 3 (DATA_QUALIFIED),
    resulting in DATA_UNAVAILABLE, zero scannable, and zero published symbols.
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)
    candidates = PackageBV3Consumer.build_candidates_from_manifest(manifest)

    engine = IsolatedPublicationEngine()
    for c in candidates:
        ev = engine.evaluate_candidate(c, "2026-10-19")
        assert ev.data_readiness_decision == "DATA_UNAVAILABLE"
        assert ev.current_publication_state == PublicationState.PROVIDER_RESOLVABLE

    dry_run = engine.execute_dry_run(candidates, "2026-10-19")
    assert dry_run["published_symbols"] == []
    assert dry_run["accounting"]["data_ready_count"] == 0
    assert dry_run["accounting"]["scannable_count"] == 0


def test_08_synthetic_test_fixture_requires_explicit_label():
    """
    Test 8: Synthetic test fixture requires explicit declaration:
    _fixture_nature='SYNTHETIC_TEST_FIXTURE'.
    Unlabeled market data fixtures fail closed.
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)

    # 1. Unlabeled fixture is rejected
    unlabeled_fixture = {
        "AAPL": {"candle_count": 288, "has_live_price": True},
    }
    with pytest.raises(PolicyViolationError, match="Market data fixture must be explicitly declared"):
        PackageBV3Consumer.build_candidates_from_manifest(manifest, market_data_fixture=unlabeled_fixture)

    # 2. Labeled fixture succeeds
    labeled_fixture = {
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
    }
    cands = PackageBV3Consumer.build_candidates_from_manifest(manifest, market_data_fixture=labeled_fixture)
    cand_by_sym = {c.symbol: c for c in cands}
    assert cand_by_sym["AAPL"].candle_count == 288
    assert cand_by_sym["AAPL"].has_live_price is True
    assert cand_by_sym["AAPL"].is_qualified_candidate is True


def test_09_uncertified_production_like_data_fails_closed():
    """
    Test 9: Prior to Option B replacement, production-like data
    (288 unprovenanced AAPL, 287 Polygon GOOG, 288 unprovenanced GOOGL, 0-bar ABBV)
    fails closed in data qualification.
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)

    pre_replacement_fixture = {
        "_fixture_nature": "SYNTHETIC_TEST_FIXTURE",
        "AAPL": {
            "candle_count": 288,
            "has_live_price": True,
            "provenance_feed": "UNPROVENANCED",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "UNPROVENANCED",
            "data_as_of": "2026-10-16",
            "is_qualified_candidate": True,
        },
        "GOOG": {
            "candle_count": 287,
            "has_live_price": True,
            "provenance_feed": "MASSIVE_POLYGON",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "FOREIGN_PROVIDER",
            "data_as_of": "2026-10-16",
            "is_qualified_candidate": True,
        },
        "GOOGL": {
            "candle_count": 288,
            "has_live_price": True,
            "provenance_feed": "UNPROVENANCED",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "UNPROVENANCED",
            "data_as_of": "2026-10-16",
            "is_qualified_candidate": True,
        },
        "ABBV": {
            "candle_count": 0,
            "has_live_price": False,
            "provenance_feed": "UNVERIFIED",
            "adjustment_state": "UNVERIFIED",
            "structural_quality": "MISSING",
            "data_as_of": None,
            "is_qualified_candidate": False,
        },
    }

    candidates = PackageBV3Consumer.build_candidates_from_manifest(
        manifest,
        market_data_fixture=pre_replacement_fixture,
    )
    engine = IsolatedPublicationEngine()
    evaluations = [engine.evaluate_candidate(c, "2026-10-19") for c in candidates]

    for ev in evaluations:
        assert ev.data_readiness_decision == "DATA_UNAVAILABLE"
        assert ev.current_publication_state == PublicationState.PROVIDER_RESOLVABLE

    dry_run = engine.execute_dry_run(candidates, "2026-10-19")
    assert dry_run["published_symbols"] == []
    assert dry_run["accounting"]["data_ready_count"] == 0
    assert dry_run["accounting"]["scannable_count"] == 0


def test_10_aapl_october_20_unprovenanced_tail_fails_closed():
    """
    Test 10: AAPL October 20 unprovenanced tail fails closed and cannot qualify.
    Simulates production condition where 289 rows exist through 2026-10-20, leaving
    the 289th session outside the certified window (2025-08-27 to 2026-10-19).
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)

    tail_fixture = {
        "_fixture_nature": "SYNTHETIC_TEST_FIXTURE",
        "AAPL": {
            "candle_count": 289,
            "has_live_price": True,
            "provenance_feed": "UNPROVENANCED_TAIL_DETECTED",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "structural_quality": "DEFECTIVE_OUTSIDE_WINDOW",
            "data_as_of": "2026-10-20",
            "is_qualified_candidate": False,
        },
    }

    candidates = PackageBV3Consumer.build_candidates_from_manifest(
        manifest,
        market_data_fixture=tail_fixture,
    )
    cand_by_sym = {c.symbol: c for c in candidates}
    aapl_cand = cand_by_sym["AAPL"]

    engine = IsolatedPublicationEngine()
    ev = engine.evaluate_candidate(aapl_cand, "2026-10-19")
    assert ev.data_readiness_decision == "DATA_UNAVAILABLE"
    assert ev.data_readiness_reason == "UNQUALIFIED_PENDING_PACKAGE_B_DATA"
    assert ev.current_publication_state == PublicationState.PROVIDER_RESOLVABLE


def test_11_goog_googl_share_class_separation_and_pricing():
    """
    Test 11: Alphabet Inc. dual-class securities GOOG (Class C) and GOOGL (Class A)
    maintain strict separation based on canonical identity and associations.
    Identical price series do NOT trigger a false failure.
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)
    pilot_candidates = {c["symbol"]: c for c in manifest["pilot_candidates"]}

    goog_meta = pilot_candidates["GOOG"]
    googl_meta = pilot_candidates["GOOGL"]

    # 1. Distinct internal IDs and associations
    assert goog_meta["durable_security_id"] == "ARX_SEC_A11A24D442E2"
    assert googl_meta["durable_security_id"] == "ARX_SEC_B7EF9ED7A35A"
    assert goog_meta["listing_association_id"] != googl_meta["listing_association_id"]

    goog_cand = IsolatedUniverseCandidate(
        internal_security_id=goog_meta["durable_security_id"],
        symbol="GOOG",
        association_id=goog_meta["listing_association_id"],
        alpaca_symbol="GOOG",
    )
    googl_cand = IsolatedUniverseCandidate(
        internal_security_id=googl_meta["durable_security_id"],
        symbol="GOOGL",
        association_id=googl_meta["listing_association_id"],
        alpaca_symbol="GOOGL",
    )

    # 2. Identical prices do NOT trigger false failure
    identical_prices = [180.25, 180.50, 181.00]
    PackageBV3Consumer.verify_multi_share_class_isolation(
        goog_cand,
        googl_cand,
        goog_prices=identical_prices,
        googl_prices=identical_prices,
    )

    # 3. Genuine identity collision DOES fail closed
    colliding_cand = IsolatedUniverseCandidate(
        internal_security_id=goog_cand.internal_security_id,
        symbol="GOOGL",
        association_id=goog_cand.association_id,
        alpaca_symbol="GOOG",
    )
    with pytest.raises(PolicyViolationError, match="Share class collision"):
        PackageBV3Consumer.verify_multi_share_class_isolation(goog_cand, colliding_cand)


def test_12_published_only_zero_scanner_enrollment():
    """
    Test 12: Published-only candidates have ZERO scanner enrollment.
    Guarantees State 5 publication does NOT leak into State 7 scanner dispatch.
    """
    published_symbols = ["AAPL", "ABBV", "GOOG", "GOOGL"]

    # 1. No active scanners passes
    IsolatedPublicationEngine.assert_no_scanner_enrollment(
        published_symbols=published_symbols,
        active_scanner_symbols=set(),
    )

    # 2. Active scanner symbol triggers fail-closed PolicyViolationError
    with pytest.raises(PolicyViolationError, match="Actionability leakage detected"):
        IsolatedPublicationEngine.assert_no_scanner_enrollment(
            published_symbols=published_symbols,
            active_scanner_symbols={"AAPL"},
        )


def test_13_lookback_threshold_boundaries():
    """
    Test 13: Verifies that lookback thresholds remain 50/60/260:
    - 50: Builder Floor (State 2 -> State 3 failure below 50 bars)
    - 60: Operational Builder Readiness Contract (DATA_UNAVAILABLE below 60 bars)
    - 260: Annual Momentum Lookback Window (satisfied by 288 bars)
    """
    # 1. < 50 bars: Fails in MarketDataQualityEvidence validation
    with pytest.raises(PolicyViolationError, match="less than required builder floor"):
        evidence = MarketDataQualityEvidence(
            symbol="AAPL",
            candle_count=49,
            has_live_price=True,
            adjustment_state="SPLIT_ADJUSTED_ONLY",
            structural_quality="CLEAN",
            provenance_feed="SIP_CONSOLIDATED",
            data_as_of="2026-10-19",
        )
        evidence.validate()

    # 2. 50-59 bars: Passes State 2->3 transition, but fails operational builder contract (< 60 bars)
    cand_55 = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        alpaca_symbol="AAPL",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        candle_count=55,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )
    engine = IsolatedPublicationEngine()
    ev_55 = engine.evaluate_candidate(cand_55, "2026-10-19")
    assert ev_55.data_readiness_decision == "DATA_UNAVAILABLE"
    assert "INSUFFICIENT_BUILDER_READINESS_HISTORY" in ev_55.data_readiness_reason
    assert "55 bars < 60 bars" in ev_55.data_readiness_reason

    # 3. 288 bars: Satisfies 260-bar annual window and receives DATA_READY
    cand_288 = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        alpaca_symbol="AAPL",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        candle_count=288,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )
    ev_288 = engine.evaluate_candidate(cand_288, "2026-10-19")
    assert ev_288.data_readiness_decision == "DATA_READY"


def test_14_deterministic_digest_and_accounting_closure():
    """
    Test 14: Verifies deterministic publication digest and exact arithmetic accounting closure.
    """
    manifest, _ = PackageBV3Consumer.load_manifest(MANIFEST_FIXTURE_PATH)

    qualified_fixture = {
        "_fixture_nature": "SYNTHETIC_TEST_FIXTURE",
        "AAPL": {"candle_count": 288, "has_live_price": True, "provenance_feed": "SIP_CONSOLIDATED", "adjustment_state": "SPLIT_ADJUSTED_ONLY", "structural_quality": "CLEAN", "data_as_of": "2026-10-19", "is_qualified_candidate": True},
        "GOOG": {"candle_count": 288, "has_live_price": True, "provenance_feed": "SIP_CONSOLIDATED", "adjustment_state": "SPLIT_ADJUSTED_ONLY", "structural_quality": "CLEAN", "data_as_of": "2026-10-19", "is_qualified_candidate": True},
        "GOOGL": {"candle_count": 288, "has_live_price": True, "provenance_feed": "SIP_CONSOLIDATED", "adjustment_state": "SPLIT_ADJUSTED_ONLY", "structural_quality": "CLEAN", "data_as_of": "2026-10-19", "is_qualified_candidate": True},
        "ABBV": {"candle_count": 288, "has_live_price": True, "provenance_feed": "SIP_CONSOLIDATED", "adjustment_state": "SPLIT_ADJUSTED_ONLY", "structural_quality": "CLEAN", "data_as_of": "2026-10-19", "is_qualified_candidate": True},
    }

    candidates = PackageBV3Consumer.build_candidates_from_manifest(
        manifest,
        market_data_fixture=qualified_fixture,
    )
    engine = IsolatedPublicationEngine()
    dry_run = engine.execute_dry_run(candidates, "2026-10-19")

    # Accounting closure verification
    acct = dry_run["accounting"]
    assert acct["source_count"] == 4
    assert acct["eligible_count"] == 4
    assert acct["ineligible_count"] == 0
    assert acct["unresolved_count"] == 0
    assert acct["data_ready_count"] == 4
    assert acct["data_unavailable_count"] == 0
    assert acct["data_unresolved_count"] == 0
    assert acct["scannable_count"] == 4
    assert acct["strategy_admitted_count"] == 0
    assert acct["scanner_enrolled_count"] == 0
    assert acct["is_closed"] is True

    # Order-invariant digest
    digest1 = engine.compute_publication_digest(candidates)
    digest2 = engine.compute_publication_digest(list(reversed(candidates)))
    assert digest1 == digest2
    assert len(digest1) == 64


def test_15_data_ready_but_r01_r07_ineligible_cannot_publish_or_scan():
    """
    Test 15 (Accounting Semantics): Verifies that a candidate with fully certified market data
    (288 daily bars, live price, clean SIP split-adjusted) that fails R01-R07 eligibility rules
    (e.g., non-USD currency or foreign exchange):
    - Receives data_readiness_decision == "DATA_READY"
    - Receives eligibility_decision == "INELIGIBLE"
    - Remains at State 3 (DATA_QUALIFIED), failing transition to State 4 (PUBLICATION_ELIGIBLE)
    - Is NOT counted in scannable_count, published_count, or published_symbols
    - Preserves mathematical accounting closure:
      source_count (1) == eligible (0) + ineligible (1) + unresolved (0)
      eligible (0) == data_ready (0) + data_unavailable (0) + data_unresolved (0)
    """
    ineligible_cand = IsolatedUniverseCandidate(
        internal_security_id="ARX_SEC_0B4C728FE0BE",
        symbol="AAPL",
        alpaca_symbol="AAPL",
        association_id="ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17",
        currency="EUR",  # Violates R05
        primary_listing=False,  # Violates R06
        candle_count=288,
        has_live_price=True,
        adjustment_state="SPLIT_ADJUSTED_ONLY",
        structural_quality="CLEAN",
        provenance_feed="SIP_CONSOLIDATED",
        data_as_of="2026-10-19",
        is_qualified_candidate=True,
    )

    engine = IsolatedPublicationEngine()
    ev = engine.evaluate_candidate(ineligible_cand, "2026-10-19")

    # Verify candidate-level decisions
    assert ev.data_readiness_decision == "DATA_READY"
    assert ev.eligibility_decision == "INELIGIBLE"
    assert "FAILED_RULES: R05_CURRENCY, R06_PRIMARY_LISTING" in ev.eligibility_reason
    assert ev.current_publication_state == PublicationState.DATA_QUALIFIED
    assert ev.current_publication_state < PublicationState.PUBLICATION_ELIGIBLE

    # Verify accounting closure
    acct = engine.evaluate_accounting_closure([ev], source_count=1)
    assert acct.source_count == 1
    assert acct.eligible_count == 0
    assert acct.ineligible_count == 1
    assert acct.unresolved_count == 0
    assert acct.data_ready_count == 0
    assert acct.scannable_count == 0
    assert acct.published_count == 0
    assert acct.strategy_admitted_count == 0
    assert acct.scanner_enrolled_count == 0
    assert acct.is_closed is True

    # Verify dry-run execution excludes candidate from published symbols
    dry_run = engine.execute_dry_run([ineligible_cand], "2026-10-19")
    assert dry_run["published_symbols"] == []
    assert dry_run["accounting"]["scannable_count"] == 0
    assert dry_run["accounting"]["data_ready_count"] == 0
    assert dry_run["accounting"]["ineligible_count"] == 1
    assert dry_run["accounting"]["is_closed"] is True

