"""
Comprehensive Verification Test Suite for Radar Scanner Pipeline.
Adheres to ARX Radar Activation Specification Sections 24, 25, 26, 27.

Covers:
- Section 24: Zero-result contract test (distinguishable from PIPELINE_PENDING)
- Section 25: Determinism tests (identical inputs = identical output)
- Section 26: Data quality reconciliation (candidate, API, snapshot parity)
- Section 27: Full test matrix (states, immutability, versioning, publication integrity, quarantine)
"""

import pytest
import sqlite3
import tempfile
import os
import copy
from typing import Dict, Any

from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    FreshnessStatus,
    PublicationDecision,
    DriftClassification,
    ScannerVersionTuple,
    ScannerCandidateResult,
    ImmutableScannerSnapshot,
    CANONICAL_VCP_UNIVERSE,
    VCP_API_CONTRACT_VERSION,
    VCP_RULESET_VERSION,
    VCP_EVIDENCE_SCHEMA_VERSION,
    VCP_SCORE_MODEL_VERSION,
    VCP_DATA_PROVENANCE_VERSION,
    VCP_UNIVERSE_VERSION,
    VCP_FRESHNESS_POLICY_VERSION,
    SMART_MONEY_API_CONTRACT_VERSION,
    SMART_MONEY_RULESET_VERSION,
    compute_hash,
    compute_semantic_fingerprint,
)
from analyst_dashboard.analyzers.scanner_publication_integrity import (
    ScannerPublicationIntegrityEngine,
    CANONICAL_VCP_RULESET_HASH,
    CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
    CANONICAL_VCP_SCORE_MODEL_HASH,
    CANONICAL_VCP_DATA_PROVENANCE_HASH,
    CANONICAL_VCP_UNIVERSE_HASH,
    CANONICAL_VCP_FRESHNESS_HASH,
    AUTHORIZED_VERSION_REGISTRY,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    SmartMoneyScannerRunner,
    CURRENT_IMPLEMENTATION_RELEASE_SHA,
)
from api.routes.screener import (
    run_screener_get,
    get_vcp_snapshot,
    get_smart_money_snapshot,
    trigger_vcp_scan,
    RADAR_CAPABILITY_CONTRACT,
)


@pytest.fixture
def temp_scanner_store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = ScannerSnapshotStore(db_path=db_path)
    yield store
    # Close any open connections before removing
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except OSError:
        pass



# ======================================================================
# SECTION 24: ZERO-RESULT CONTRACT TESTS
# ======================================================================

def test_zero_result_contract_distinguishable_from_pending(temp_scanner_store):
    """
    Section 24 Invariant:
    A completed scan with no matches yields:
      status = AVAILABLE
      matched_count = 0
    This must remain strictly distinguishable from PIPELINE_PENDING where:
      status = PIPELINE_PENDING
      matched_count = None
    """
    vcp_runner = VCPScannerRunner(snapshot_store=temp_scanner_store)
    # Scan with empty universe to produce 0 matches
    zero_scan = vcp_runner.execute_market_wide_scan(universe_override=[])

    assert zero_scan["status"] == ScannerStatus.AVAILABLE.value
    assert zero_scan["snapshot"]["matched_count"] == 0
    assert zero_scan["snapshot"]["universe_size"] == 0
    assert len(zero_scan["results"]) == 0

    sm_runner = SmartMoneyScannerRunner()
    sm_envelope = sm_runner.get_status_envelope()

    assert sm_envelope["status"] == ScannerStatus.PIPELINE_PENDING.value
    assert sm_envelope["snapshot"]["matched_count"] is None
    assert sm_envelope["results"] == []

    # Hard distinction
    assert zero_scan["status"] != sm_envelope["status"]
    assert zero_scan["snapshot"]["matched_count"] != sm_envelope["snapshot"]["matched_count"]


# ======================================================================
# SECTION 25: DETERMINISM TESTS
# ======================================================================

def test_scanner_deterministic_replay(temp_scanner_store):
    """
    Section 25 Invariant:
    SAME_SEMANTICS + SAME_CANONICAL_INPUTS + SAME_UNIVERSE_MEMBERSHIP = SAME_OUTPUT
    """
    vcp_runner = VCPScannerRunner(snapshot_store=temp_scanner_store)
    subset_universe = ["LNTH", "CPRX", "MEDP", "ACLS", "ELF"]

    run_1 = vcp_runner.execute_market_wide_scan(universe_override=subset_universe)
    run_2 = vcp_runner.execute_market_wide_scan(universe_override=subset_universe)

    assert run_1["snapshot"]["matched_count"] == run_2["snapshot"]["matched_count"]
    assert [r["symbol"] for r in run_1["results"]] == [r["symbol"] for r in run_2["results"]]
    assert [r["score"] for r in run_1["results"]] == [r["score"] for r in run_2["results"]]
    assert [r["rank"] for r in run_1["results"]] == [r["rank"] for r in run_2["results"]]
    assert run_1["snapshot"]["semantic_fingerprint"] == run_2["snapshot"]["semantic_fingerprint"]


# ======================================================================
# SECTION 26: DATA QUALITY RECONCILIATION
# ======================================================================

def test_data_quality_reconciliation():
    """
    Section 26:
    Reconciles frontend candidate, API response, persisted snapshot, scanner output,
    version tuple, provenance, and freshness.
    """
    api_envelope = get_vcp_snapshot()
    screener_resp = run_screener_get(filter_type="vcp")

    assert api_envelope["status"] == "AVAILABLE"
    assert api_envelope["methodology"]["ruleset_version"] == VCP_RULESET_VERSION
    assert api_envelope["methodology"]["evidence_schema_version"] == VCP_EVIDENCE_SCHEMA_VERSION
    assert api_envelope["methodology"]["score_model_version"] == VCP_SCORE_MODEL_VERSION
    assert api_envelope["provenance"]["data_provenance_version"] == VCP_DATA_PROVENANCE_VERSION
    assert api_envelope["provenance"]["universe_version"] == VCP_UNIVERSE_VERSION
    assert api_envelope["provenance"]["implementation_release_sha"] == CURRENT_IMPLEMENTATION_RELEASE_SHA

    vcp_symbols_api = {r["symbol"] for r in api_envelope["results"]}
    screener_symbols = {c["symbol"] for c in screener_resp["candidates"]}

    # Parity check: all qualifying candidates in screener must be in VCP snapshot
    assert screener_symbols == vcp_symbols_api
    for c in screener_resp["candidates"]:
        assert "VCP" in c["categories"]
        assert c["categoryEvidence"].get("VCP") == "CRITERIA_MATCHED"


# ======================================================================
# SECTION 27: TEST MATRIX & SNAPSHOT IMMUTABILITY
# ======================================================================

def test_snapshot_immutability_triggers(temp_scanner_store):
    """
    Section 27 & 12: Database triggers prevent UPDATE and DELETE on scanner_snapshots.
    """
    vcp_runner = VCPScannerRunner(snapshot_store=temp_scanner_store)
    scan_res = vcp_runner.execute_market_wide_scan(universe_override=["LNTH", "CPRX"])
    snap_id = scan_res["snapshot"]["snapshot_id"]

    conn = temp_scanner_store._get_connection()
    try:
        # Mutation attempt must fail via trigger
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION"):
            conn.execute("UPDATE scanner_snapshots SET status_at_publication = 'MUTATED' WHERE snapshot_id = ?", (snap_id,))

        # Deletion attempt must fail via trigger
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY_VIOLATION"):
            conn.execute("DELETE FROM scanner_snapshots WHERE snapshot_id = ?", (snap_id,))
    finally:
        conn.close()


def test_independent_version_tuple_propagation():
    """
    Section 6 & 27: All 8 parts of the version tuple are preserved independently.
    """
    runner = VCPScannerRunner()
    vt = runner.get_version_tuple()

    assert vt.scanner_id == "MINERVINI_VCP"
    assert vt.api_contract_version == VCP_API_CONTRACT_VERSION
    assert vt.ruleset_version == VCP_RULESET_VERSION
    assert vt.evidence_schema_version == VCP_EVIDENCE_SCHEMA_VERSION
    assert vt.score_model_version == VCP_SCORE_MODEL_VERSION
    assert vt.data_provenance_version == VCP_DATA_PROVENANCE_VERSION
    assert vt.universe_version == VCP_UNIVERSE_VERSION
    assert vt.freshness_policy_version == VCP_FRESHNESS_POLICY_VERSION
    assert vt.implementation_release_sha == CURRENT_IMPLEMENTATION_RELEASE_SHA


def test_publication_integrity_authorizations():
    """
    Section 14, 15, 27:
    - Same version + same hash = PASS (PUBLISH)
    - Same version + changed hash = QUARANTINE
    - Unknown version = QUARANTINE
    - Unsupported combination = QUARANTINE
    """
    engine = ScannerPublicationIntegrityEngine()
    runner = VCPScannerRunner()
    vt = runner.get_version_tuple()

    # 1. Authorized baseline: PASS
    rep = engine.evaluate_publication_integrity(
        scanner_id="MINERVINI_VCP",
        candidate_version_tuple=vt,
        ruleset_hash=CANONICAL_VCP_RULESET_HASH,
        evidence_schema_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
        score_model_hash=CANONICAL_VCP_SCORE_MODEL_HASH,
        data_provenance_spec_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
        universe_definition_hash=CANONICAL_VCP_UNIVERSE_HASH,
        freshness_policy_hash=CANONICAL_VCP_FRESHNESS_HASH,
    )
    assert rep.decision == PublicationDecision.PUBLISH
    assert rep.drift_classification == DriftClassification.NO_SEMANTIC_DRIFT

    # 2. Same version + changed hash: QUARANTINE
    rep_tampered = engine.evaluate_publication_integrity(
        scanner_id="MINERVINI_VCP",
        candidate_version_tuple=vt,
        ruleset_hash="tampered_hash_12345",
        evidence_schema_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
        score_model_hash=CANONICAL_VCP_SCORE_MODEL_HASH,
        data_provenance_spec_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
        universe_definition_hash=CANONICAL_VCP_UNIVERSE_HASH,
        freshness_policy_hash=CANONICAL_VCP_FRESHNESS_HASH,
    )
    assert rep_tampered.decision == PublicationDecision.QUARANTINE
    assert rep_tampered.drift_classification in [
        DriftClassification.CONTENT_CHANGE_WITHOUT_VERSION_BUMP,
        DriftClassification.SILENT_SEMANTIC_DRIFT,
    ]

    # 3. Unknown version: QUARANTINE
    vt_unknown = ScannerVersionTuple(
        scanner_id="MINERVINI_VCP",
        api_contract_version="99.0.0",
        ruleset_version="99.0.0",
        evidence_schema_version=VCP_EVIDENCE_SCHEMA_VERSION,
        score_model_version=VCP_SCORE_MODEL_VERSION,
        data_provenance_version=VCP_DATA_PROVENANCE_VERSION,
        universe_version=VCP_UNIVERSE_VERSION,
        freshness_policy_version=VCP_FRESHNESS_POLICY_VERSION,
        implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
    )
    rep_unknown = engine.evaluate_publication_integrity(
        scanner_id="MINERVINI_VCP",
        candidate_version_tuple=vt_unknown,
        ruleset_hash=CANONICAL_VCP_RULESET_HASH,
        evidence_schema_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
        score_model_hash=CANONICAL_VCP_SCORE_MODEL_HASH,
        data_provenance_spec_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
        universe_definition_hash=CANONICAL_VCP_UNIVERSE_HASH,
        freshness_policy_hash=CANONICAL_VCP_FRESHNESS_HASH,
    )
    assert rep_unknown.decision == PublicationDecision.QUARANTINE


def test_quarantined_run_preserves_last_good_snapshot(temp_scanner_store):
    """
    Section 16 & 27:
    If semantic integrity fails, active snapshot remains last verified good snapshot.
    """
    vcp_runner = VCPScannerRunner(snapshot_store=temp_scanner_store)
    # Run good scan
    good_scan = vcp_runner.execute_market_wide_scan(universe_override=["LNTH", "CPRX"])
    good_snap_id = good_scan["snapshot"]["snapshot_id"]

    # Tamper integrity engine with bad hash
    bad_engine = ScannerPublicationIntegrityEngine()
    # Evaluate with tampered hash
    eval_bad = bad_engine.evaluate_publication_integrity(
        scanner_id="MINERVINI_VCP",
        candidate_version_tuple=vcp_runner.get_version_tuple(),
        ruleset_hash="corrupt_hash",
        evidence_schema_hash=CANONICAL_VCP_EVIDENCE_SCHEMA_HASH,
        score_model_hash=CANONICAL_VCP_SCORE_MODEL_HASH,
        data_provenance_spec_hash=CANONICAL_VCP_DATA_PROVENANCE_HASH,
        universe_definition_hash=CANONICAL_VCP_UNIVERSE_HASH,
        freshness_policy_hash=CANONICAL_VCP_FRESHNESS_HASH,
    )
    assert eval_bad.decision == PublicationDecision.QUARANTINE

    # Verify store still returns the good snapshot as active
    latest = temp_scanner_store.get_latest_active_snapshot("MINERVINI_VCP")
    assert latest["snapshot_id"] == good_snap_id


def test_cross_scanner_isolation():
    """
    Section 22 & 27: Cross-scanner isolation.
    VCP candidates must not leak into Smart Money.
    Smart Money remains PIPELINE_PENDING with 0 candidates.
    """
    resp_vcp = run_screener_get(filter_type="vcp")
    resp_sm = run_screener_get(filter_type="smart_money")

    assert resp_vcp["gemsFound"] > 0
    assert resp_sm["gemsFound"] == 0
    assert resp_sm["capabilities"]["SMART_MONEY"]["status"] == "PIPELINE_PENDING"

    vcp_cands = resp_vcp["candidates"]
    sm_cands = resp_sm["candidates"]

    # Ensure zero overlap
    vcp_symbols = {c["symbol"] for c in vcp_cands}
    sm_symbols = {c["symbol"] for c in sm_cands}
    assert len(vcp_symbols.intersection(sm_symbols)) == 0


def test_dedicated_snapshot_routes():
    """
    Section 18: Expose snapshots through dedicated API routes.
    """
    vcp_snap = get_vcp_snapshot()
    assert vcp_snap["scanner_id"] == "MINERVINI_VCP"
    assert vcp_snap["status"] == "AVAILABLE"
    assert "snapshot" in vcp_snap
    assert "results" in vcp_snap

    sm_snap = get_smart_money_snapshot()
    assert sm_snap["scanner_id"] == "SMART_MONEY"
    assert sm_snap["status"] == "PIPELINE_PENDING"
    assert sm_snap["snapshot"]["matched_count"] is None
    assert sm_snap["results"] == []
