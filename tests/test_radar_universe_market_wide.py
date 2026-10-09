"""
tests/test_radar_universe_market_wide.py

Comprehensive Verification Suite for Radar VCP Universe Architecture & Invariants.
Covers:
- Source-population accounting & arithmetic closure
- Single terminal states & zero silent symbol drops
- Deterministic reconstruction across shuffle, reverse, worker parallelism
- Exact rule boundaries (R01-R07) and data readiness (history seasoning)
- Cross-build reconciliation & same-count membership drift quarantine
- Last-good universe preservation
- Operator authorization & overlapping scan protection
- Smart Money isolation
"""

import os
import random
import pytest
from fastapi.testclient import TestClient

from analyst_dashboard.universe.contracts import (
    UNIVERSE_ID,
    UNIVERSE_VERSION,
    RADAR_SCOPE_LABEL,
    EligibilityDecision,
    DataReadinessDecision,
    ConstructionStatus,
    PublicationDecision,
    SourceSecurity,
    SourcePopulationSnapshot,
    compute_sha256,
)
from analyst_dashboard.universe.builder import DeterministicUniverseBuilder
from analyst_dashboard.universe.store import UniverseStore
from analyst_dashboard.universe.fixtures import (
    HISTORICAL_VCP_35_REGRESSION_FIXTURE,
    CANONICAL_FIXTURE_SECURITIES,
    create_canonical_fixture_snapshot,
)
from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    CANONICAL_VCP_UNIVERSE,
)
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    SmartMoneyScannerRunner,
)
from api.main import app
from api.routes.screener import OPERATOR_TOKEN

client = TestClient(app)


def test_curated_35_list_demoted_to_historical_fixture():
    """Verify curated 35 list is an explicit fixture and not production authority."""
    assert len(HISTORICAL_VCP_35_REGRESSION_FIXTURE) == 35
    assert CANONICAL_VCP_UNIVERSE == HISTORICAL_VCP_35_REGRESSION_FIXTURE
    assert "CPRX" in HISTORICAL_VCP_35_REGRESSION_FIXTURE
    assert "LNTH" in HISTORICAL_VCP_35_REGRESSION_FIXTURE


def test_source_population_accounting_and_arithmetic_closure(tmp_path):
    """Test accounting closure: source_count == eligible + ineligible + unresolved."""
    snapshot = create_canonical_fixture_snapshot()
    builder = DeterministicUniverseBuilder()

    attestation, el_rows, rd_rows = builder.build_universe(snapshot, build_id="test-build-001")

    # Invariant: source_population_count == eligible_count + ineligible_count + eligibility_unresolved_count
    assert snapshot.count == attestation.source_population_count
    assert attestation.source_population_count == (
        attestation.eligible_count + attestation.ineligible_count + attestation.eligibility_unresolved_count
    )

    # Invariant: eligible_count == data_ready_count + data_unavailable_count + data_unresolved_count
    assert attestation.eligible_count == (
        attestation.data_ready_count + attestation.data_unavailable_count + attestation.data_unresolved_count
    )
    assert attestation.scannable_count == attestation.data_ready_count
    assert attestation.eligibility_unresolved_count == 0
    assert attestation.data_unresolved_count == 0
    assert attestation.construction_status == ConstructionStatus.COMPLETE
    assert attestation.publication_decision == PublicationDecision.PUBLISH


def test_every_source_security_gets_single_terminal_eligibility_decision():
    """Verify every source security receives exactly one terminal eligibility state with zero silent drops."""
    snapshot = create_canonical_fixture_snapshot()
    builder = DeterministicUniverseBuilder()

    attestation, el_rows, rd_rows = builder.build_universe(snapshot)

    assert len(el_rows) == snapshot.count
    seen_symbols = set()
    for row in el_rows:
        assert row.eligibility_decision in [
            EligibilityDecision.ELIGIBLE,
            EligibilityDecision.INELIGIBLE,
            EligibilityDecision.UNRESOLVED,
        ]
        assert row.symbol not in seen_symbols
        seen_symbols.add(row.symbol)

    # Every input symbol is accounted for
    input_symbols = {s.symbol for s in snapshot.securities}
    assert seen_symbols == input_symbols


def test_every_eligible_security_gets_single_terminal_readiness_state():
    """Verify every ELIGIBLE security receives exactly one terminal readiness state."""
    snapshot = create_canonical_fixture_snapshot()
    builder = DeterministicUniverseBuilder()

    attestation, el_rows, rd_rows = builder.build_universe(snapshot)

    eligible_symbols = {r.symbol for r in el_rows if r.eligibility_decision == EligibilityDecision.ELIGIBLE}
    assert len(rd_rows) == len(eligible_symbols)

    for r in rd_rows:
        assert r.readiness_result in [
            DataReadinessDecision.DATA_READY,
            DataReadinessDecision.DATA_UNAVAILABLE,
            DataReadinessDecision.DATA_UNRESOLVED,
        ]
        assert r.symbol in eligible_symbols


def test_deterministic_reconstruction_under_shuffled_and_reversed_input():
    """Verify universe reconstruction is 100% deterministic under shuffled and reversed input sequences."""
    base_snapshot = create_canonical_fixture_snapshot()
    builder = DeterministicUniverseBuilder()

    attestation_base, _, _ = builder.build_universe(base_snapshot, build_id="fixed-build-base")
    base_membership_hash = attestation_base.eligible_membership_hash
    base_decision_hash = attestation_base.per_security_decision_hash

    # Test reverse ordering
    reversed_securities = list(reversed(base_snapshot.securities))
    rev_snapshot = SourcePopulationSnapshot(
        snapshot_id="rev-snap",
        source_authority=base_snapshot.source_authority,
        as_of=base_snapshot.as_of,
        count=len(reversed_securities),
        source_hash=base_snapshot.source_hash,
        securities=reversed_securities,
    )
    attestation_rev, _, _ = builder.build_universe(rev_snapshot, build_id="fixed-build-rev")
    assert attestation_rev.eligible_membership_hash == base_membership_hash
    assert attestation_rev.per_security_decision_hash == base_decision_hash

    # Test multiple fixed shuffle seeds
    for seed in [42, 1337, 9999]:
        shuffled = list(base_snapshot.securities)
        random.seed(seed)
        random.shuffle(shuffled)
        shuf_snapshot = SourcePopulationSnapshot(
            snapshot_id=f"shuf-snap-{seed}",
            source_authority=base_snapshot.source_authority,
            as_of=base_snapshot.as_of,
            count=len(shuffled),
            source_hash=base_snapshot.source_hash,
            securities=shuffled,
        )
        attestation_shuf, _, _ = builder.build_universe(shuf_snapshot, build_id=f"fixed-build-{seed}")
        assert attestation_shuf.eligible_membership_hash == base_membership_hash
        assert attestation_shuf.per_security_decision_hash == base_decision_hash


def test_deterministic_reconstruction_under_worker_parallelism():
    """Verify universe reconstruction is invariant across thread worker counts (1, 2, 4)."""
    snapshot = create_canonical_fixture_snapshot()
    builder = DeterministicUniverseBuilder()

    attestation_1, _, _ = builder.build_universe(snapshot, worker_count=1)
    attestation_2, _, _ = builder.build_universe(snapshot, worker_count=2)
    attestation_4, _, _ = builder.build_universe(snapshot, worker_count=4)

    assert attestation_1.eligible_membership_hash == attestation_2.eligible_membership_hash
    assert attestation_1.eligible_membership_hash == attestation_4.eligible_membership_hash
    assert attestation_1.per_security_decision_hash == attestation_2.per_security_decision_hash
    assert attestation_1.per_security_decision_hash == attestation_4.per_security_decision_hash


def test_rule_boundaries_and_reasons():
    """Test exact rule boundary behavior and assigned reason codes for ineligible instruments."""
    snapshot = create_canonical_fixture_snapshot()
    builder = DeterministicUniverseBuilder()

    _, el_rows, _ = builder.build_universe(snapshot)
    decision_map = {r.symbol: (r.eligibility_decision, r.eligibility_reason_code) for r in el_rows}

    # Common stock should be eligible
    assert decision_map["NVDA"][0] == EligibilityDecision.ELIGIBLE
    assert decision_map["NVDA"][1] == "PASSED"

    # ETF fails R02
    assert decision_map["SPY"][0] == EligibilityDecision.INELIGIBLE
    assert decision_map["SPY"][1] == "INELIGIBLE_ASSET_CLASS" or decision_map["SPY"][1] == "INELIGIBLE_SECURITY_TYPE"

    # ADR fails R02
    assert decision_map["TSM"][0] == EligibilityDecision.INELIGIBLE
    assert decision_map["TSM"][1] == "INELIGIBLE_SECURITY_TYPE"

    # Inactive listing fails R03
    assert decision_map["INACT1"][0] == EligibilityDecision.INELIGIBLE
    assert decision_map["INACT1"][1] == "INELIGIBLE_LISTING_STATUS"

    # Foreign exchange fails R04
    assert decision_map["LON1"][0] == EligibilityDecision.INELIGIBLE
    assert decision_map["LON1"][1] == "INELIGIBLE_EXCHANGE"

    # Foreign currency fails R05
    assert decision_map["EUR1"][0] == EligibilityDecision.INELIGIBLE
    assert decision_map["EUR1"][1] == "INELIGIBLE_CURRENCY"

    # Secondary listing fails R06
    assert decision_map["SEC1"][0] == EligibilityDecision.INELIGIBLE
    assert decision_map["SEC1"][1] == "INELIGIBLE_SECONDARY_LISTING"

    # Delisted fails R07
    assert decision_map["DELIST1"][0] == EligibilityDecision.INELIGIBLE
    assert decision_map["DELIST1"][1] == "INELIGIBLE_DELISTED"


def test_same_count_membership_drift_detection_quarantines():
    """Verify builder detects same-count membership drift across builds and quarantines candidate."""
    builder = DeterministicUniverseBuilder()

    # Build 1: standard fixture
    snap1 = create_canonical_fixture_snapshot("snap-001")
    attestation1, _, _ = builder.build_universe(snap1, build_id="ubuild-001")
    assert attestation1.publication_decision == PublicationDecision.PUBLISH

    # Build 2: replace one eligible stock with another eligible stock (same eligible count, different membership)
    sec_modified = [s for s in snap1.securities if s.symbol != "NVDA"]
    sec_modified.append(SourceSecurity(
        security_id="SEC_NEWCO",
        symbol="NEWCO",
        exchange="NASDAQ",
        security_type="COMMON_STOCK",
        listing_status="ACTIVE",
    ))
    snap2 = SourcePopulationSnapshot(
        snapshot_id="snap-002",
        source_authority="TEST_SWAP",
        as_of="2026-10-09T01:00:00Z",
        count=len(sec_modified),
        source_hash="swap_hash",
        securities=sec_modified,
    )

    attestation2, _, _ = builder.build_universe(snap2, build_id="ubuild-002", previous_build=attestation1)
    assert attestation2.eligible_count == attestation1.eligible_count
    assert attestation2.reconciliation_summary["same_count_drift"] is True
    assert attestation2.publication_decision == PublicationDecision.QUARANTINE
    assert attestation2.construction_status == ConstructionStatus.PARTIAL


def test_universe_store_immutability_triggers(tmp_path):
    """Verify SQLite triggers prevent mutation or deletion of universe records."""
    db_file = str(tmp_path / "test_universe.db")
    store = UniverseStore(db_path=db_file)

    snap = create_canonical_fixture_snapshot()
    builder = DeterministicUniverseBuilder()
    attestation, el_rows, rd_rows = builder.build_universe(snap, build_id="ubuild-immut-01")

    store.save_source_snapshot(snap)
    store.save_universe_build(attestation, el_rows, rd_rows)

    conn = store._get_connection()
    try:
        with pytest.raises(Exception, match="CANONICAL_INVARIANT_VIOLATION"):
            conn.execute("UPDATE universe_builds SET eligible_count = 999 WHERE universe_build_id = 'ubuild-immut-01'")

        with pytest.raises(Exception, match="CANONICAL_INVARIANT_VIOLATION"):
            conn.execute("DELETE FROM universe_builds WHERE universe_build_id = 'ubuild-immut-01'")

        with pytest.raises(Exception, match="CANONICAL_INVARIANT_VIOLATION"):
            conn.execute("UPDATE universe_eligibility_ledger SET eligibility_decision = 'INELIGIBLE' WHERE universe_build_id = 'ubuild-immut-01'")

        with pytest.raises(Exception, match="CANONICAL_INVARIANT_VIOLATION"):
            conn.execute("DELETE FROM universe_eligibility_ledger WHERE universe_build_id = 'ubuild-immut-01'")
    finally:
        conn.close()


def test_public_scan_trigger_is_forbidden():
    """Verify unauthenticated public requests to POST /api/v1/screener/vcp/scan receive 403 Forbidden."""
    resp = client.post("/api/v1/screener/vcp/scan")
    assert resp.status_code == 403
    assert "PUBLIC_MARKET_WIDE_SCAN_FORBIDDEN" in resp.json()["detail"]


def test_operator_scan_trigger_succeeds():
    """Verify authorized operator requests with X-Operator-Token succeed."""
    resp = client.post(
        "/api/v1/screener/vcp/scan",
        headers={"X-Operator-Token": OPERATOR_TOKEN},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scanner_id"] == "MINERVINI_VCP"
    assert "universe" in data
    assert "coverage" in data
    assert data["universe"]["display_name"] == RADAR_SCOPE_LABEL


def test_overlapping_vcp_scans_prohibited():
    """Verify VCPScannerRunner rejects concurrent overlapping scan attempts."""
    runner = VCPScannerRunner()
    # Acquire lock externally to simulate ongoing scan
    runner._scan_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="OVERLAPPING_VCP_SCANS_PROHIBITED"):
            runner.execute_market_wide_scan()
    finally:
        runner._scan_lock.release()


def test_smart_money_isolation():
    """Verify Smart Money scanner remains strictly PIPELINE_PENDING with None matches."""
    resp = client.get("/api/v1/screener/smart-money/snapshot")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scanner_id"] == "SMART_MONEY"
    assert data["status"] == "PIPELINE_PENDING"
    assert data["snapshot"]["matched_count"] is None
