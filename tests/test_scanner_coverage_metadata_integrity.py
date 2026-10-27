import pytest
import sqlite3
import tempfile
import os
from unittest.mock import MagicMock

from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    PublicationDecision,
    resolve_coverage_envelope,
    ImmutableScannerSnapshot,
    ScannerVersionTuple,
    VCP_API_CONTRACT_VERSION,
    VCP_RULESET_VERSION,
    VCP_EVIDENCE_SCHEMA_VERSION,
    VCP_SCORE_MODEL_VERSION,
    VCP_DATA_PROVENANCE_VERSION,
    VCP_UNIVERSE_VERSION,
    VCP_FRESHNESS_POLICY_VERSION,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    CURRENT_IMPLEMENTATION_RELEASE_SHA,
)
from api.routes.screener import get_radar_capabilities


def test_error_snapshot_with_zero_symbols_clamps_to_unavailable():
    """Gate 1: An ERROR snapshot with 0 scanned symbols reports UNAVAILABLE, never COMPLETE."""
    c_meta = {
        "coverage_status": "PARTIAL",
        "data_complete_count": 0,
        "scanned_successfully_count": 0,
        "matched_count": None,
        "data_completeness_pct": 0.0,
        "scan_coverage_pct": 0.0,
        "failure_reason": "ZERO_SYMBOLS_SCANNED",
    }
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.ERROR,
        universe_size=40,
        matched_count=None,
        c_meta=c_meta,
    )
    assert env_cov["coverage_status"] == "UNAVAILABLE"
    assert env_cov["scan_coverage_pct"] == 0.0
    assert env_cov["data_completeness_pct"] == 0.0
    assert env_cov["matched_count"] is None
    assert env_cov["failure_reason"] == "ZERO_SYMBOLS_SCANNED"


def test_corrupted_error_snapshot_claiming_complete_is_clamped():
    """Gate 1 & Invariant: Any ERROR snapshot claiming COMPLETE is clamped to UNAVAILABLE."""
    c_meta = {
        "coverage_status": "COMPLETE",
        "data_complete_count": 40,
        "scanned_successfully_count": 40,
        "matched_count": 5,  # Invalid on ERROR
        "data_completeness_pct": 100.0,
        "scan_coverage_pct": 100.0,
    }
    env_cov = resolve_coverage_envelope(
        status="ERROR",
        universe_size=40,
        matched_count=5,
        c_meta=c_meta,
        failure_reason_fallback="TEST_ERROR",
    )
    assert env_cov["coverage_status"] == "UNAVAILABLE"
    assert env_cov["matched_count"] is None  # Invariant: matched_count None on ERROR
    assert env_cov["failure_reason"] == "TEST_ERROR"


def test_historical_available_snapshot_without_metadata_reports_unknown():
    """Historical AVAILABLE snapshot must report UNKNOWN and null percentages, never invented 100%."""
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.AVAILABLE,
        universe_size=40,
        matched_count=2,
        c_meta={},
    )
    assert env_cov["coverage_status"] == "UNKNOWN"
    assert env_cov["data_completeness_pct"] is None
    assert env_cov["scan_coverage_pct"] is None
    assert env_cov["matched_count"] == 2


def test_historical_error_snapshot_without_metadata_reports_unavailable():
    """Historical ERROR snapshot without coverage metadata reports UNAVAILABLE and null percentages."""
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.ERROR,
        universe_size=40,
        matched_count=None,
        c_meta=None,
    )
    assert env_cov["coverage_status"] == "UNAVAILABLE"
    assert env_cov["data_completeness_pct"] is None
    assert env_cov["scan_coverage_pct"] is None
    assert env_cov["matched_count"] is None
    assert env_cov["failure_reason"] == "SCANNER_EXECUTION_ERROR"


def test_empty_universe_reports_empty_universe_state():
    """Empty universe (universe_size=0) reports EMPTY_UNIVERSE with 0.0 percentages."""
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.ERROR,
        universe_size=0,
        matched_count=None,
        c_meta=None,
    )
    assert env_cov["coverage_status"] == "EMPTY_UNIVERSE"
    assert env_cov["data_completeness_pct"] == 0.0
    assert env_cov["scan_coverage_pct"] == 0.0
    assert env_cov["matched_count"] is None


def test_pipeline_pending_reports_pending():
    """PIPELINE_PENDING snapshot reports PENDING coverage with None percentages."""
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.PIPELINE_PENDING,
        universe_size=40,
        matched_count=None,
        c_meta=None,
    )
    assert env_cov["coverage_status"] == "PENDING"
    assert env_cov["data_completeness_pct"] is None
    assert env_cov["scan_coverage_pct"] is None
    assert env_cov["matched_count"] is None


def test_sqlite_store_roundtrip_preserves_coverage_metadata():
    """Coverage metadata roundtrips through SQLite provenance_json without schema migration."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        store = ScannerSnapshotStore(db_path=db_path)
        snapshot_id = "vcp-test-cov-001"
        cov_meta = {
            "coverage_status": "PARTIAL",
            "data_complete_count": 30,
            "scanned_successfully_count": 30,
            "matched_count": 1,
            "unavailable_symbol_count": 10,
            "unresolved_symbol_count": 0,
            "data_completeness_pct": 75.0,
            "scan_coverage_pct": 75.0,
            "unavailable_reasons": {"DATA_MISSING": 10},
            "failure_reason": None,
            "adjustment_state_distribution": {"SPLIT_ADJUSTED": 30},
        }
        univ_meta = {
            "scope_class": "US_EQUITIES",
            "display_name": "US Equities",
            "source_population_count": 40,
            "eligible_universe_count": 40,
            "universe_version": VCP_UNIVERSE_VERSION,
            "universe_build_id": "test-build-1",
            "membership_hash": "hash123",
            "construction_status": "COMPLETE",
        }
        snap_data = {
            "snapshot_id": snapshot_id,
            "scanner_id": "MINERVINI_VCP",
            "run_id": "run-001",
            "api_contract_version": VCP_API_CONTRACT_VERSION,
            "ruleset_version": VCP_RULESET_VERSION,
            "evidence_schema_version": VCP_EVIDENCE_SCHEMA_VERSION,
            "score_model_version": VCP_SCORE_MODEL_VERSION,
            "data_provenance_version": VCP_DATA_PROVENANCE_VERSION,
            "universe_version": VCP_UNIVERSE_VERSION,
            "freshness_policy_version": VCP_FRESHNESS_POLICY_VERSION,
            "implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA,
            "semantic_fingerprint": "sem-fp-001",
            "generated_at": "2026-10-27T00:00:00Z",
            "data_as_of": "2026-10-27T00:00:00Z",
            "status_at_publication": "AVAILABLE",
            "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
            "universe_size": 40,
            "matched_count": 1,
            "results": [{"symbol": "NVDA", "score": 92.0}],
            "provenance": {"role": "PRICE_VOLUME_HISTORY"},
            "freshness": {"status": "LIVE"},
            "publication_decision": "PUBLISH",
            "coverage_metadata": cov_meta,
            "universe_metadata": univ_meta,
        }

        saved_id = store.save_snapshot(snap_data)
        assert saved_id == snapshot_id

        # Reload from store
        loaded = store.get_snapshot_by_id(snapshot_id)
        assert loaded is not None
        assert loaded["coverage_metadata"]["coverage_status"] == "PARTIAL"
        assert loaded["coverage_metadata"]["scan_coverage_pct"] == 75.0
        assert loaded["universe_metadata"]["universe_build_id"] == "test-build-1"

        # Verify envelope conversion preserves exact values
        runner = VCPScannerRunner(snapshot_store=store)
        env = runner._to_envelope(loaded)
        assert env["coverage"]["coverage_status"] == "PARTIAL"
        assert env["coverage"]["scan_coverage_pct"] == 75.0
        assert env["coverage"]["matched_count"] == 1
        assert env["universe"]["universe_build_id"] == "test-build-1"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_radar_capabilities_clamps_corrupted_error_coverage():
    """get_radar_capabilities clamps any ERROR snapshot claiming COMPLETE to UNAVAILABLE."""
    corrupted_error_snapshot = {
        "status": "ERROR",
        "snapshot_id": "snap-err-corrupt",
        "matched_count": None,
        "data_as_of": None,
        "coverage": {
            "coverage_status": "COMPLETE",  # Corrupted claim
            "scanned_successfully_count": 0,
            "unavailable_symbol_count": 40,
            "scan_coverage_pct": 100.0,
            "failure_reason": "SCANNER_EXECUTION_ERROR",
        },
        "universe": {
            "eligible_universe_count": 40,
            "universe_build_id": "build-40",
        },
    }

    caps = get_radar_capabilities(vcp_snapshot=corrupted_error_snapshot)
    vcp_cap = caps["VCP"]
    assert vcp_cap["status"] == "ERROR"
    assert vcp_cap["universeScreening"] == "ERROR"
    assert vcp_cap["matchedCount"] is None
    assert vcp_cap["coverageStatus"] == "UNAVAILABLE"  # Clamped!
    assert vcp_cap["scannedSuccessfullyCount"] == 0


def test_immutable_scanner_snapshot_to_envelope_resolves_coverage():
    """ImmutableScannerSnapshot.to_envelope delegates to canonical resolver."""
    snap = ImmutableScannerSnapshot(
        snapshot_id="snap-imm-1",
        scanner_id="MINERVINI_VCP",
        run_id="run-imm-1",
        version_tuple=ScannerVersionTuple(
            scanner_id="MINERVINI_VCP",
            api_contract_version=VCP_API_CONTRACT_VERSION,
            ruleset_version=VCP_RULESET_VERSION,
            evidence_schema_version=VCP_EVIDENCE_SCHEMA_VERSION,
            score_model_version=VCP_SCORE_MODEL_VERSION,
            data_provenance_version=VCP_DATA_PROVENANCE_VERSION,
            universe_version=VCP_UNIVERSE_VERSION,
            freshness_policy_version=VCP_FRESHNESS_POLICY_VERSION,
            implementation_release_sha=CURRENT_IMPLEMENTATION_RELEASE_SHA,
        ),
        semantic_fingerprint="fp-imm-1",
        generated_at="2026-10-27T00:00:00Z",
        data_as_of="2026-10-27T00:00:00Z",
        status_at_publication=ScannerStatus.ERROR,
        universe_id="ARX_CANONICAL_LONG_TERM_V1",
        universe_size=40,
        matched_count=None,
        results=[],
        provenance={},
        freshness={},
        publication_decision=PublicationDecision.PUBLISH,
        coverage_metadata={
            "coverage_status": "COMPLETE",  # Corrupted input
            "scanned_successfully_count": 0,
            "scan_coverage_pct": 0.0,
        },
    )
    env = snap.to_envelope()
    assert env["coverage"]["coverage_status"] == "UNAVAILABLE"
    assert env["coverage"]["matched_count"] is None
    assert env["coverage"]["scan_coverage_pct"] == 0.0


# ==============================================================================
# FOCUSED ADVERSARIAL TESTS (DIRECTIVE 028/029 FINAL HARDENING)
# ==============================================================================

def test_adversarial_incomplete_metadata_does_not_infer_universe_size():
    """Directive 029: Eliminate unsupported fallbacks inferring scanned counts from universe size."""
    c_meta = {
        "coverage_status": "AVAILABLE",
        # scanned_successfully_count omitted
        # data_complete_count omitted
        # scan_coverage_pct omitted
    }
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.AVAILABLE,
        universe_size=40,
        matched_count=1,
        c_meta=c_meta,
    )
    # Must NOT infer 40 or 100.0% from universe_size
    assert env_cov["scanned_successfully_count"] is None
    assert env_cov["data_complete_count"] is None
    assert env_cov["scan_coverage_pct"] is None
    assert env_cov["data_completeness_pct"] is None
    assert env_cov["matched_count"] == 1


def test_adversarial_contradictory_counts_rectified_from_measurements():
    """Directive 029: Contradictory COMPLETE claim with partial measured counts is corrected."""
    c_meta = {
        "coverage_status": "COMPLETE",  # Contradictory claim
        "scanned_successfully_count": 15,  # 15 / 40 = 37.5%
        "data_complete_count": 15,
        "scan_coverage_pct": None,  # Must be derived from authoritative measurements
    }
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.AVAILABLE,
        universe_size=40,
        matched_count=0,
        c_meta=c_meta,
    )
    assert env_cov["coverage_status"] == "PARTIAL"
    assert env_cov["scanned_successfully_count"] == 15
    assert env_cov["scan_coverage_pct"] == 37.5
    assert env_cov["matched_count"] == 0


def test_adversarial_failed_publication_preserves_valid_partial_coverage_with_disclosure():
    """Directive 029: Preserve genuinely measured partial coverage on ERROR with publication-state disclosure."""
    c_meta = {
        "coverage_status": "PARTIAL",
        "scanned_successfully_count": 20,
        "data_complete_count": 20,
        "scan_coverage_pct": 50.0,
        "failure_reason": "PUBLICATION_INTEGRITY_REJECTED",
    }
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.ERROR,
        universe_size=40,
        matched_count=2,  # Must be nulled out on ERROR
        c_meta=c_meta,
    )
    assert env_cov["coverage_status"] == "PARTIAL"
    assert env_cov["scanned_successfully_count"] == 20
    assert env_cov["scan_coverage_pct"] == 50.0
    assert env_cov["matched_count"] is None  # Candidate eligibility nulled out
    assert env_cov["failure_reason"] == "PUBLICATION_INTEGRITY_REJECTED"


def test_adversarial_error_with_full_scanned_count_fails_closed():
    """Directive 029: ERROR snapshot can NEVER advertise false COMPLETE coverage even if scanned_count == universe_size."""
    c_meta = {
        "coverage_status": "COMPLETE",
        "scanned_successfully_count": 40,
        "data_complete_count": 40,
        "scan_coverage_pct": 100.0,
        "failure_reason": "DOWNSTREAM_STORAGE_ABORT",
    }
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.ERROR,
        universe_size=40,
        matched_count=5,
        c_meta=c_meta,
    )
    assert env_cov["coverage_status"] == "UNAVAILABLE"
    assert env_cov["matched_count"] is None
    assert env_cov["failure_reason"] == "DOWNSTREAM_STORAGE_ABORT"


def test_adversarial_unknown_historical_coverage_preserves_null_metrics():
    """Directive 029: Unmeasured historical snapshots preserve UNKNOWN semantics and null metrics."""
    env_cov = resolve_coverage_envelope(
        status=ScannerStatus.AVAILABLE,
        universe_size=40,
        matched_count=3,
        c_meta=None,
    )
    assert env_cov["coverage_status"] == "UNKNOWN"
    assert env_cov["scanned_successfully_count"] is None
    assert env_cov["data_complete_count"] is None
    assert env_cov["scan_coverage_pct"] is None
    assert env_cov["data_completeness_pct"] is None
    assert env_cov["matched_count"] == 3

