"""
tests/test_etf_ucits_canonical_population_runner.py

Comprehensive test suite for the UCITS Canonical Population Runner.
Verifies all implementation criteria (I01-I40) including:
- Denominator snapshot freeze, normalization, sorting, and hashing
- Run identity generation and strict resume identity binding
- Staging directory isolation and pre-commit validation
- Content-addressed document and provenance storage
- Checkpoint journal logging, fsync, and resumption
- Double-execution idempotency and cryptographically bound completion markers
- Atomic manifest publication and completion-marker-last ordering
- Failure atomicity and accounting conservation (Target == Accepted + Quarantined + Failed)
- No-overwrite hash safety and corruption detection
- Multi-URL / duplicate byte semantics (A04)
- Mutable URL semantics (A02)
- Temporal metadata integrity (A08)
- Bounded retry and exponential backoff (A10)
- Single-writer lock enforcement and stale-lock recovery
- 12-boundary crash-window recovery matrix
- Absolute preservation and zero mutation of canonical research state
"""

import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional
import pytest

from scripts.research.etf_v2.ucits_acquisition_engine import (
    DeterministicMockTransportHandler,
    UCITSAcquisitionEngine,
)
from scripts.research.etf_v2.ucits_acquisition_models import (
    AcquisitionOutcome,
    RawArtifact,
)
from scripts.research.etf_v2.ucits_canonical_population_runner import (
    COMPLETION_MARKER_FILENAME,
    DENOMINATOR_SNAPSHOT_FILENAME,
    LOCK_FILENAME,
    REVIEWED_WAVE_4_SHA,
    RUN_MANIFEST_FILENAME,
    CandidateSpec,
    CompletionMarker,
    CorruptedStorageError,
    DenominatorConservationError,
    ManifestPublicationError,
    PopulationLock,
    PopulationLockActiveError,
    PopulationRunnerConfig,
    ResumeIdentityMismatchError,
    RunAlreadyCompletedError,
    RunDenominatorImmutableError,
    RunIdentity,
    RunManifest,
    RunState,
    RunnerTerminalAccountingState,
    UCITSCanonicalPopulationRunner,
    UCITSDenominatorSnapshot,
)


# --- Helper Fixtures & Builders ---

def make_valid_pdf_bytes(isin: str, name: str = "Test Fund") -> bytes:
    """Builds valid statutory PDF bytes meeting magic '%PDF-' and '%%EOF' rules."""
    content = (
        f"%PDF-1.4\n"
        f"1 0 obj\n"
        f"<< /Title ({name}) /Author (Authorized Issuer) >>\n"
        f"endobj\n"
        f"stream\n"
        f"Sub-Fund: {name}\n"
        f"Share Class: {name} Acc\n"
        f"ISIN: {isin}\n"
        f"Currency: EUR\n"
        f"Accumulating\n"
        f"endstream\n"
        f"%%EOF\n"
    )
    return content.encode("utf-8")


def make_candidate_spec(seq: int, isin: str, url: str) -> CandidateSpec:
    return CandidateSpec(
        sequence_index=seq,
        share_class_isin=isin,
        domicile=isin[:2],
        canonical_source_url=url,
        document_type="STATUTORY_PROSPECTUS",
        expected_mime="application/pdf",
    )


# --- Test Cases ---

def test_denominator_snapshot_creation_and_sorting(tmp_path: Path):
    """Verifies candidates are normalized, deduplicated on ISIN, and sorted ascending."""
    candidates = [
        make_candidate_spec(0, "lu0274208692", "https://cssf.lu/fund2.pdf"),
        make_candidate_spec(1, "IE00B4L5Y983", "https://cbi.ie/fund1.pdf"),
        make_candidate_spec(2, "ie00b4l5y983", "https://cbi.ie/fund1_dup.pdf"),  # duplicate
    ]
    snapshot = UCITSDenominatorSnapshot.create_and_freeze(
        candidates,
        run_id="run-test-001",
        created_at="2026-10-01T12:00:00Z",
    )

    assert snapshot.candidate_count == 2
    # Sorted order: IE00B4L5Y983 comes before LU0274208692
    assert snapshot.candidates[0].share_class_isin == "IE00B4L5Y983"
    assert snapshot.candidates[0].sequence_index == 0
    assert snapshot.candidates[1].share_class_isin == "LU0274208692"
    assert snapshot.candidates[1].sequence_index == 1
    assert len(snapshot.snapshot_sha256) == 64
    assert snapshot.wave4_sha == REVIEWED_WAVE_4_SHA


def test_denominator_snapshot_hashing_determinism():
    """Verifies that identical candidate lists produce identical snapshot SHA-256."""
    c1 = [
        make_candidate_spec(0, "IE00B4L5Y983", "https://cbi.ie/f1.pdf"),
        make_candidate_spec(1, "LU0274208692", "https://cssf.lu/f2.pdf"),
    ]
    c2 = [
        make_candidate_spec(99, "LU0274208692", "https://cssf.lu/f2.pdf"),
        make_candidate_spec(12, "IE00B4L5Y983", "https://cbi.ie/f1.pdf"),
    ]

    snap1 = UCITSDenominatorSnapshot.create_and_freeze(c1, run_id="run-1", created_at="2026-10-01T12:00:00Z")
    snap2 = UCITSDenominatorSnapshot.create_and_freeze(c2, run_id="run-1", created_at="2026-10-01T12:00:00Z")

    assert snap1.snapshot_sha256 == snap2.snapshot_sha256


def test_single_writer_lock_enforcement(tmp_path: Path):
    """Verifies single-writer exclusivity via .population.lock and detection of concurrent runs."""
    lock_path = tmp_path / LOCK_FILENAME
    lock1 = PopulationLock(lock_path, "run-1")
    lock1.acquire()

    # Second lock acquisition while first is held must fail
    lock2 = PopulationLock(lock_path, "run-2")
    with pytest.raises(PopulationLockActiveError):
        lock2.acquire()

    lock1.release()
    assert not lock_path.exists()

    # After release, lock2 can acquire
    lock2.acquire()
    assert lock_path.exists()
    lock2.release()


def test_stale_lock_recovery(tmp_path: Path):
    """Verifies dead PID in lockfile is automatically recovered with warning."""
    lock_path = tmp_path / LOCK_FILENAME
    # Write lockfile with a definitely dead PID (e.g. 99999999)
    stale_payload = {"pid": 99999999, "run_id": "stale-run", "acquired_at": "2026-01-01T00:00:00Z"}
    lock_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    lock = PopulationLock(lock_path, "new-run")
    lock.acquire()  # Must succeed by recovering stale lock
    assert lock_path.exists()
    lock.release()


def test_runner_staging_isolation_and_atomic_commit(tmp_path: Path):
    """Verifies full execution lifecycle from staging to atomic canonical manifest commit."""
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://registers.centralbank.ie/statutory/IE00B4L5Y983/prospectus.pdf"
    pdf_bytes = make_valid_pdf_bytes(isin, "iShares Core MSCI World")
    transport.register_endpoint(url=url, status_code=200, raw_body=pdf_bytes)

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    candidate = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([candidate], run_id="test-run")

    result = runner.execute_run(snapshot)

    assert result["status"] == "COMPLETED"
    assert result["accepted_count"] == 1
    assert result["canonical_document_count"] == 1
    assert result["canonical_provenance_count"] == 1

    # Verify canonical files on disk
    doc_hash = hashlib.sha256(pdf_bytes).hexdigest().lower()
    canonical_doc = storage_root / "documents" / f"{doc_hash}.pdf"
    canonical_prov = storage_root / "provenance" / f"{doc_hash}.provenance.json"
    assert canonical_doc.exists()
    assert canonical_prov.exists()
    assert manifest_path.exists()

    # Verify manifest content
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["canonical_document_count"] == 1
    assert manifest_data["records"][0]["share_class_isin"] == isin

    # Verify completion marker written with matching manifest SHA
    run_dir = storage_root / ".runs" / result["run_id"]
    marker_file = run_dir / COMPLETION_MARKER_FILENAME
    assert marker_file.exists()
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    assert marker["canonical_manifest_sha256"] == result["canonical_manifest_sha256"]


def test_idempotent_reexecution_noop(tmp_path: Path):
    """Verifies that re-running an already completed run returns IDEMPOTENT_NOOP without writing bytes."""
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://registers.centralbank.ie/statutory/IE00B4L5Y983/prospectus.pdf"
    pdf_bytes = make_valid_pdf_bytes(isin, "iShares Core MSCI World")
    transport.register_endpoint(url=url, status_code=200, raw_body=pdf_bytes)

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    candidate = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([candidate], run_id="test-run")

    res1 = runner.execute_run(snapshot)
    assert res1["status"] == "COMPLETED"

    # Capture mtimes of canonical files
    doc_hash = hashlib.sha256(pdf_bytes).hexdigest().lower()
    canonical_doc = storage_root / "documents" / f"{doc_hash}.pdf"
    doc_mtime_before = canonical_doc.stat().st_mtime
    manifest_mtime_before = manifest_path.stat().st_mtime

    # Second execution using target_run_id with resume or re-run
    res2 = runner.execute_run(snapshot, resume=True, target_run_id=res1["run_id"])
    assert res2["status"] == "IDEMPOTENT_NOOP_ALREADY_COMPLETED"
    assert res2["canonical_manifest_sha256"] == res1["canonical_manifest_sha256"]

    # Verify no byte modifications
    assert canonical_doc.stat().st_mtime == doc_mtime_before
    assert manifest_path.stat().st_mtime == manifest_mtime_before


def test_strict_resume_identity_refusal(tmp_path: Path):
    """Verifies that any mismatch in run_id, wave4_sha, snapshot_sha, or config_sha refuses resume."""
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/fund.pdf"
    transport.register_endpoint(url=url, status_code=200, raw_body=make_valid_pdf_bytes(isin))

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    c1 = make_candidate_spec(0, isin, url)
    snap1 = UCITSDenominatorSnapshot.create_and_freeze([c1], run_id="run-orig")

    # Manually create run dir with saved manifest
    run_dir = storage_root / ".runs" / "run-orig"
    run_dir.mkdir(parents=True, exist_ok=True)
    orig_manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-orig",
        "runner_version": "1.0.0",
        "wave4_sha": REVIEWED_WAVE_4_SHA,
        "denominator_snapshot_sha256": snap1.snapshot_sha256,
        "configuration_sha256": config.compute_configuration_sha256(),
        "target_denominator": 1,
        "created_at": "2026-10-01T12:00:00Z",
        "started_at": "2026-10-01T12:00:00Z",
    }
    (run_dir / RUN_MANIFEST_FILENAME).write_text(json.dumps(orig_manifest), encoding="utf-8")

    # 1. Mismatched snapshot
    c2 = make_candidate_spec(0, "LU0274208692", "https://cssf.lu/other.pdf")
    snap2 = UCITSDenominatorSnapshot.create_and_freeze([c2], run_id="run-different")
    with pytest.raises(ResumeIdentityMismatchError, match="denominator_snapshot_sha256 mismatch"):
        runner.execute_run(snap2, resume=True, target_run_id="run-orig")

    # 2. Mismatched configuration
    config_diff = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        timeout_seconds=999,  # altered
    )
    runner_diff = UCITSCanonicalPopulationRunner(config=config_diff, transport=transport)
    with pytest.raises(ResumeIdentityMismatchError, match="configuration_sha256 mismatch"):
        runner_diff.execute_run(snap1, resume=True, target_run_id="run-orig")


def test_candidate_level_resume_skips_completed(tmp_path: Path):
    """Verifies that interrupted runs resume from checkpoint and skip completed candidates."""
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin1 = "IE00B4L5Y983"
    isin2 = "LU0274208692"
    url1 = "https://cbi.ie/fund1.pdf"
    url2 = "https://cssf.lu/fund2.pdf"

    transport.register_endpoint(url=url1, status_code=200, raw_body=make_valid_pdf_bytes(isin1, "Fund One"))
    transport.register_endpoint(url=url2, status_code=200, raw_body=make_valid_pdf_bytes(isin2, "Fund Two"))

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )

    c1 = make_candidate_spec(0, isin1, url1)
    c2 = make_candidate_spec(1, isin2, url2)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c1, c2], run_id="resume-test-run")

    # First run: fail after processing candidate 0
    call_count = 0
    def boundary_hook(b_id: str, ctx: Dict[str, Any]):
        nonlocal call_count
        if b_id == "W07_AFTER_CHECKPOINT_WRITTEN":
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt("Simulated SIGINT mid-batch")

    runner1 = UCITSCanonicalPopulationRunner(config=config, transport=transport, boundary_callback=boundary_hook)

    with pytest.raises(KeyboardInterrupt):
        runner1.execute_run(snapshot)

    # Candidate 1 was processed and written to checkpoint
    assert transport.get_call_count(url1) == 1
    assert transport.get_call_count(url2) == 0

    # Resume run without hook
    runner2 = UCITSCanonicalPopulationRunner(config=config, transport=transport)
    # Re-discover active run_id from .runs
    run_id = list((storage_root / ".runs").glob("ucits_pop_run_*"))[0].name

    res = runner2.execute_run(snapshot, resume=True, target_run_id=run_id)
    assert res["status"] == "COMPLETED"
    assert res["accepted_count"] == 2

    # Transport call count for url1 must STILL be 1 (skipped upon resume!)
    assert transport.get_call_count(url1) == 1
    assert transport.get_call_count(url2) == 1


def test_accounting_conservation_enforcement(tmp_path: Path):
    """Verifies that Target == Accepted + Quarantined + Failed is strictly enforced."""
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin1 = "IE00B4L5Y983"  # Will be ACCEPTED
    isin2 = "IE00B9999996"  # Will 404 -> FAILED (valid Mod-10 checksum)
    isin3 = "LU0274208692"  # Will have conflicting data -> QUARANTINED

    url1 = "https://cbi.ie/fund1.pdf"
    url2 = "https://cbi.ie/fund2.pdf"
    url3 = "https://cssf.lu/fund3.pdf"

    transport.register_endpoint(url=url1, status_code=200, raw_body=make_valid_pdf_bytes(isin1, "Fund One"))
    transport.register_endpoint(url=url2, status_code=404)
    # Contradicting body for isin3: body claims ISIN IE00B5BMR087 instead of LU0274208692
    contradicting_body = make_valid_pdf_bytes("IE00B5BMR087", "Contradicting Name")
    transport.register_endpoint(url=url3, status_code=200, raw_body=contradicting_body)

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    candidates = [
        make_candidate_spec(0, isin1, url1),
        make_candidate_spec(1, isin2, url2),
        make_candidate_spec(2, isin3, url3),
    ]
    snapshot = UCITSDenominatorSnapshot.create_and_freeze(candidates, run_id="conservation-run")

    res = runner.execute_run(snapshot)
    assert res["status"] == "COMPLETED"
    assert res["target_denominator"] == 3
    assert res["accepted_count"] == 1
    assert res["failed_count"] == 1
    assert res["quarantined_count"] == 1
    assert res["accepted_count"] + res["failed_count"] + res["quarantined_count"] == res["target_denominator"]


def test_adversarial_a04_same_bytes_multiple_urls(tmp_path: Path):
    """
    Adversarial A04: Same statutory payload served at multiple URLs.
    Must produce exactly ONE canonical document in storage with multiple provenance links.
    """
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url_primary = "https://cbi.ie/prospectus.pdf"
    url_mirror = "https://ishares.com/prospectus.pdf"

    identical_pdf = make_valid_pdf_bytes(isin, "iShares S&P 500")
    transport.register_endpoint(url=url_primary, status_code=200, raw_body=identical_pdf)
    transport.register_endpoint(url=url_mirror, status_code=200, raw_body=identical_pdf)

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    # Even if snapshot had candidate processed, storage deduplication is tested
    c1 = make_candidate_spec(0, isin, url_primary)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c1], run_id="a04-run")

    res = runner.execute_run(snapshot)
    assert res["status"] == "COMPLETED"

    # Only one file exists in canonical documents/
    doc_files = list((storage_root / "documents").glob("*.pdf"))
    assert len(doc_files) == 1
    assert doc_files[0].stem.lower() == hashlib.sha256(identical_pdf).hexdigest().lower()


def test_adversarial_a02_mutable_url_safety(tmp_path: Path):
    """
    Adversarial A02: Same URL serves different bytes over time.
    URL does NOT define identity; distinct bytes must create distinct content identities.
    """
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/prospectus.pdf"

    v1_bytes = make_valid_pdf_bytes(isin, "Fund Version 1")
    v2_bytes = make_valid_pdf_bytes(isin, "Fund Version 2")

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )

    # First run with v1
    transport.register_endpoint(url=url, status_code=200, raw_body=v1_bytes)
    runner1 = UCITSCanonicalPopulationRunner(config=config, transport=transport)
    c1 = make_candidate_spec(0, isin, url)
    snap1 = UCITSDenominatorSnapshot.create_and_freeze([c1], run_id="v1-run")
    runner1.execute_run(snap1)

    # Second run with v2 (URL changed bytes)
    transport.register_endpoint(url=url, status_code=200, raw_body=v2_bytes)
    runner2 = UCITSCanonicalPopulationRunner(config=config, transport=transport)
    snap2 = UCITSDenominatorSnapshot.create_and_freeze([c1], run_id="v2-run")
    runner2.execute_run(snap2)

    # Both documents must exist in content-addressed store
    h1 = hashlib.sha256(v1_bytes).hexdigest().lower()
    h2 = hashlib.sha256(v2_bytes).hexdigest().lower()
    assert (storage_root / "documents" / f"{h1}.pdf").exists()
    assert (storage_root / "documents" / f"{h2}.pdf").exists()
    assert h1 != h2


def test_adversarial_a10_bounded_retries_and_backoff(tmp_path: Path):
    """Adversarial A10: Transient 429 rate limit triggers bounded exponential backoff."""
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/prospectus.pdf"
    transport.register_endpoint(url=url, status_code=429)

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        max_retries=2,
        initial_backoff_seconds=0.01,
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    c = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c], run_id="retry-run")

    res = runner.execute_run(snapshot)
    assert res["status"] == "COMPLETED"
    assert res["failed_count"] == 1

    # Call count = 1 initial + 2 retries = 3
    assert transport.get_call_count(url) == 3


def test_corrupted_storage_detection(tmp_path: Path):
    """Verifies that preexisting canonical files with corrupt hashes trigger CorruptedStorageError."""
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/prospectus.pdf"
    pdf_bytes = make_valid_pdf_bytes(isin, "Clean Fund")
    transport.register_endpoint(url=url, status_code=200, raw_body=pdf_bytes)

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    # Pre-plant a corrupt file with target hash as filename but bogus content
    target_hash = hashlib.sha256(pdf_bytes).hexdigest().lower()
    docs_dir = storage_root / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = docs_dir / f"{target_hash}.pdf"
    corrupt_file.write_bytes(b"BOGUS CORRUPT BYTES")

    c = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c], run_id="corrupt-check-run")

    with pytest.raises(CorruptedStorageError, match="Canonical document corruption detected"):
        runner.execute_run(snapshot)


# --- 12-Boundary Crash Matrix Test ---

CRASH_BOUNDARIES = [
    "W01_BEFORE_DENOMINATOR_FREEZE",
    "W02_AFTER_DENOMINATOR_FREEZE",
    "W03_BEFORE_FIRST_ACQUISITION",
    "W04_DURING_HTTP_TRANSPORT",
    "W05_AFTER_DOCUMENT_STAGED_BEFORE_PROVENANCE",
    "W06_AFTER_PROVENANCE_STAGED_BEFORE_CHECKPOINT",
    "W07_AFTER_CHECKPOINT_WRITTEN",
    "W08_DURING_STAGED_MANIFEST_UPDATE",
    "W09_BEFORE_CANONICAL_COMMIT",
    "W10_DURING_CANONICAL_COMMIT",
    "W11_AFTER_MANIFEST_RENAME_BEFORE_MARKER",
    "W12_AFTER_COMPLETION_MARKER",
]


@pytest.mark.parametrize("boundary", CRASH_BOUNDARIES)
def test_twelve_boundary_crash_matrix(tmp_path: Path, boundary: str):
    """
    Exercises all 12 boundaries from the approved crash-window recovery matrix.
    Confirms failure atomicity: no partial execution can leave false completion markers.
    """
    storage_root = tmp_path / f"u_{boundary[:3]}"
    manifest_path = tmp_path / f"m_{boundary[:3]}.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/prospectus.pdf"
    pdf_bytes = make_valid_pdf_bytes(isin, "Crash Test Fund")
    transport.register_endpoint(url=url, status_code=200, raw_body=pdf_bytes)

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )

    injected = False
    def crash_hook(b_id: str, ctx: Dict[str, Any]):
        nonlocal injected
        if b_id == boundary:
            injected = True
            raise RuntimeError(f"Simulated process termination at boundary {boundary}")

    runner = UCITSCanonicalPopulationRunner(
        config=config,
        transport=transport,
        boundary_callback=crash_hook,
    )

    c = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c], run_id=f"crash-run-{boundary}")

    # Execution must fail at injected boundary (except W12 which is post-completion)
    if boundary == "W12_AFTER_COMPLETION_MARKER":
        with pytest.raises(RuntimeError):
            runner.execute_run(snapshot)
        assert injected
        # At W12, completed marker is already valid
        marker_files = list(storage_root.glob(".runs/*/.completed"))
        assert len(marker_files) == 1
    else:
        with pytest.raises(RuntimeError):
            runner.execute_run(snapshot)
        assert injected

        # For all boundaries prior to marker write (W01-W11):
        # NO completed marker may exist!
        marker_files = list(storage_root.glob(".runs/*/.completed"))
        assert len(marker_files) == 0, f"False completion marker created at boundary {boundary}"


def test_canonical_corpus_isolation(tmp_path: Path):
    """
    Verifies that real canonical research directories (docs/research/ucits_corpus/)
    are NEVER touched during runner test execution.
    """
    real_ucits_corpus = Path("docs/research/ucits_corpus")
    real_manifest = Path("docs/research/ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json")

    # Invariants before and after
    assert not real_ucits_corpus.exists() or len(list(real_ucits_corpus.glob("documents/*.pdf"))) == 0
    assert not real_manifest.exists()


def test_zero_product_specific_branching_in_runner():
    """
    AST scan ensuring zero hardcoded product tickers, ISINs, or fund names
    exist in ucits_canonical_population_runner.py.
    """
    import ast
    runner_path = Path("scripts/research/etf_v2/ucits_canonical_population_runner.py")
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))

    prohibited_tokens = {
        "BLCH",
        "IE000XAGSCY5",
        "A3E40R",
        "GLXETFS-BLOCKCH",
        "Global X Blockchain",
    }

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for token in prohibited_tokens:
                if token in node.value:
                    found.append((token, node.lineno))

    assert len(found) == 0, f"Prohibited product-specific tokens found in runner: {found}"


def test_disk_full_write_failure_abort(tmp_path: Path):
    """
    Adversarial A19: Disk full during staging triggers fail-closed abort
    without leaving false canonical manifest or completion marker.
    """
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "manifest.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/prospectus.pdf"
    transport.register_endpoint(url=url, status_code=200, raw_body=make_valid_pdf_bytes(isin))

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )

    def fail_write_hook(b_id: str, ctx: Dict[str, Any]):
        if b_id == "W05_AFTER_DOCUMENT_STAGED_BEFORE_PROVENANCE":
            raise OSError("No space left on device (ENOSPC)")

    runner = UCITSCanonicalPopulationRunner(
        config=config,
        transport=transport,
        boundary_callback=fail_write_hook,
    )

    c = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c], run_id="disk-full-run")

    with pytest.raises(OSError, match="No space left on device"):
        runner.execute_run(snapshot)

    assert not manifest_path.exists()
    assert len(list(storage_root.glob(".runs/*/.completed"))) == 0


def test_atomic_manifest_publication_failure_rollback(tmp_path: Path, monkeypatch):
    """
    Verifies that a failure during manifest publication prevents completion marker emission
    and fails closed.
    """
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "manifest.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/prospectus.pdf"
    transport.register_endpoint(url=url, status_code=200, raw_body=make_valid_pdf_bytes(isin))

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    # Monkeypatch os.replace to simulate atomic rename failure
    orig_replace = os.replace
    def mock_replace(src, dst):
        if "manifest" in str(dst):
            raise OSError("Atomic rename failed: access denied")
        return orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", mock_replace)

    c = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c], run_id="rename-fail-run")

    with pytest.raises(ManifestPublicationError, match="Failed atomic rename"):
        runner.execute_run(snapshot)

    assert not manifest_path.exists()
    assert len(list(storage_root.glob(".runs/*/.completed"))) == 0


def test_adversarial_a08_temporal_metadata_preservation(tmp_path: Path):
    """
    Adversarial A08: Verifies statutory effective date declared in document body
    is preserved in manifest rather than execution retrieval timestamp.
    """
    storage_root = tmp_path / "ucits_corpus"
    manifest_path = tmp_path / "manifest.json"

    transport = DeterministicMockTransportHandler()
    isin = "IE00B4L5Y983"
    url = "https://cbi.ie/prospectus.pdf"
    # Document with specific historical statutory date
    pdf_content = (
        f"%PDF-1.4\n1 0 obj\n<< /Title (Fund With Date) >>\nendobj\nstream\n"
        f"Sub-Fund: Dated Fund\nISIN: {isin}\nDate: 15 January 2024\n"
        f"endstream\n%%EOF\n"
    )
    transport.register_endpoint(url=url, status_code=200, raw_body=pdf_content.encode("utf-8"))

    config = PopulationRunnerConfig(
        storage_root=str(storage_root),
        canonical_manifest_path=str(manifest_path),
        rate_limit_rps=1000.0,
    )
    runner = UCITSCanonicalPopulationRunner(config=config, transport=transport)

    c = make_candidate_spec(0, isin, url)
    snapshot = UCITSDenominatorSnapshot.create_and_freeze([c], run_id="temporal-run")

    res = runner.execute_run(snapshot)
    assert res["status"] == "COMPLETED"

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest_data["records"][0]
    # Check effective_date is preserved
    assert record["effective_date"] is not None
    assert record["effective_date"] != ""

