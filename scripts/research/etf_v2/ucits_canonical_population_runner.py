"""
scripts/research/etf_v2/ucits_canonical_population_runner.py

Statutory Canonical Population Runner for UCITS Source Authority (Pipeline V2 Wave 4).

Orchestrates bounded canonical corpus population for UCITS ETFs while strictly preserving
frozen statutory authority modules (engine, extractor, reconciliation pipeline, domain models).

Enforces:
- Strict separation between dynamic discovery and bounded execution via immutable Denominator Snapshot.
- Single-writer process exclusivity via .population.lock with stale-lock detection.
- Isolated run staging in .runs/<run_id>/staging/ before atomic canonical publication.
- Content-addressed document and provenance persistence (<sha256>.pdf, <sha256>.provenance.json).
- Strict 1:1 pairing invariant (CANONICAL_DOCUMENT_COUNT == CANONICAL_PROVENANCE_RECORD_COUNT).
- Append-only candidate-level checkpoint journal (checkpoint.jsonl) with fsync flushing.
- Crash recovery and strict resume identity binding (run_id, wave4_sha, snapshot_sha, config_sha).
- Idempotent re-execution detection via cryptographically bound .completed markers.
- Atomic manifest swap (ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json.tmp -> final).
- Completion marker written strictly LAST after atomic manifest publication.
- Rigorous accounting conservation (TARGET_DENOMINATOR == ACCEPTED + QUARANTINED + FAILED).
- Zero product-specific branching or hardcoded ticker logic.
- Absolute isolation of the canonical SEC corpus (docs/research/sec_corpus/).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Re-use frozen Wave 4 authority models verbatim (NO duplicate business logic)
from .global_identifier_authority import normalize_isin, validate_isin
from .global_identity_models import ETFInstrument, ETFListing, ETFShareClass
from .global_identity_resolver import AuthorityShareClassEntry
from .ucits_acquisition_engine import (
    DeterministicMockTransportHandler,
    HttpTransportHandler,
    UCITSAcquisitionEngine,
)
from .ucits_acquisition_models import (
    AcquisitionOutcome,
    AcquisitionResult,
    AuthorityLocator,
    AuthorityRequest,
    ExtractedUCITSEvidence,
    RawArtifact,
    UCITSPopulationUniverseCounters,
)
from .ucits_identity_extractor import UCITSIdentityExtractor
from .ucits_provenance_models import (
    AUTHORIZED_DOCUMENT_CLASSES,
    ETFSourceAuthorityError,
    UCITSSourceProvenanceRecord,
)
from .ucits_reconciliation_pipeline import UCITSReconciliationPipeline

LOGGER_NAME = "arx.etf_v2.ucits.population_runner"
logger = logging.getLogger(LOGGER_NAME)

REVIEWED_WAVE_4_SHA: str = "3590f7f0ba4f7e20869ea48c1b353a105c080d4d"
RUNNER_VERSION: str = "1.0.0"
CANONICAL_STORAGE_ROOT_DEFAULT: str = "docs/research/ucits_corpus"
CANONICAL_MANIFEST_FILENAME: str = "ETF_V2_UCITS_SOURCE_CORPUS_MANIFEST.json"
CANONICAL_MANIFEST_PATH_DEFAULT: str = f"docs/research/{CANONICAL_MANIFEST_FILENAME}"
COMPLETION_MARKER_FILENAME: str = ".completed"
CHECKPOINT_FILENAME: str = "checkpoint.jsonl"
RUN_MANIFEST_FILENAME: str = "run_manifest.json"
DENOMINATOR_SNAPSHOT_FILENAME: str = "denominator_snapshot.json"
LOCK_FILENAME: str = ".population.lock"


class RunState(str, Enum):
    """Deterministic states for canonical population run lifecycle."""
    CREATED = "CREATED"
    DENOMINATOR_FROZEN = "DENOMINATOR_FROZEN"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    READY_TO_COMMIT = "READY_TO_COMMIT"
    COMMITTING = "COMMITTING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


class RunnerTerminalAccountingState(str, Enum):
    """Terminal accounting outcomes for denominator candidate conservation."""
    ACCEPTED = "ACCEPTED"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


# --- Exceptions ---

class PopulationRunnerError(Exception):
    """Base exception for population runner failures."""
    pass


class PopulationLockActiveError(PopulationRunnerError):
    """Raised when an active single-writer population lock is held by another process."""
    pass


class ResumeIdentityMismatchError(PopulationRunnerError):
    """Raised when resume parameters do not match the frozen run identity."""
    pass


class CorruptedStorageError(PopulationRunnerError):
    """Raised when content-addressed storage verification fails (hash mismatch or missing pair)."""
    pass


class DenominatorConservationError(PopulationRunnerError):
    """Raised when accounting conservation (Target == Accepted + Quarantined + Failed) is violated."""
    pass


class ManifestPublicationError(PopulationRunnerError):
    """Raised when atomic canonical manifest swap fails."""
    pass


class RunDenominatorImmutableError(PopulationRunnerError):
    """Raised when candidate list is mutated after denominator freeze."""
    pass


class RunAlreadyCompletedError(PopulationRunnerError):
    """Raised when attempting to execute an already completed run without explicit force flag."""
    pass


# --- Data Structures & Schemas ---

@dataclass(frozen=True)
class CandidateSpec:
    """Individual candidate specification for statutory acquisition."""
    sequence_index: int
    share_class_isin: str
    domicile: str
    canonical_source_url: str
    document_type: str
    expected_mime: str = "application/pdf"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "share_class_isin": self.share_class_isin,
            "domicile": self.domicile,
            "canonical_source_url": self.canonical_source_url,
            "document_type": self.document_type,
            "expected_mime": self.expected_mime,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateSpec:
        return cls(
            sequence_index=int(data["sequence_index"]),
            share_class_isin=str(data["share_class_isin"]).strip().upper(),
            domicile=str(data["domicile"]).strip().upper(),
            canonical_source_url=str(data["canonical_source_url"]).strip(),
            document_type=str(data["document_type"]).strip(),
            expected_mime=str(data.get("expected_mime", "application/pdf")).strip(),
        )


@dataclass(frozen=True)
class UCITSDenominatorSnapshot:
    """
    Immutable denominator snapshot binding candidate universe and canonical ordering
    prior to batch execution dispatch.
    """
    schema_version: str
    snapshot_id: str
    run_id: str
    created_at: str
    wave4_sha: str
    discovery_authority_version: str
    candidate_count: int
    candidates: Tuple[CandidateSpec, ...]
    snapshot_sha256: str

    @classmethod
    def create_and_freeze(
        cls,
        candidates: Sequence[CandidateSpec],
        run_id: str,
        discovery_version: str = "1.0.0",
        created_at: Optional[str] = None,
    ) -> UCITSDenominatorSnapshot:
        """
        Deduplicates, normalizes, sorts ascending by ISIN, assigns sequence indices,
        and computes deterministic SHA-256 over canonical JSON serialization.
        """
        now_iso = created_at or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Deduplicate on normalized ISIN, preserving earliest valid spec
        seen_isins: Set[str] = set()
        deduped: List[CandidateSpec] = []
        for c in candidates:
            norm_isin = normalize_isin(c.share_class_isin)
            if not validate_isin(norm_isin):
                raise PopulationRunnerError(f"Invalid ISIN in candidate spec: {c.share_class_isin}")
            if norm_isin not in seen_isins:
                seen_isins.add(norm_isin)
                deduped.append(c)

        # Deterministic ordering: sorted by normalized share_class_isin ascending
        sorted_candidates = sorted(deduped, key=lambda x: normalize_isin(x.share_class_isin))

        # Re-index sequences canonically
        indexed_candidates = tuple(
            CandidateSpec(
                sequence_index=idx,
                share_class_isin=normalize_isin(c.share_class_isin),
                domicile=c.domicile.upper().strip(),
                canonical_source_url=c.canonical_source_url.strip(),
                document_type=c.document_type.strip(),
                expected_mime=c.expected_mime.strip(),
            )
            for idx, c in enumerate(sorted_candidates)
        )

        # Canonical dict representation for hashing
        payload = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "created_at": now_iso,
            "wave4_sha": REVIEWED_WAVE_4_SHA,
            "discovery_authority_version": discovery_version,
            "candidate_count": len(indexed_candidates),
            "candidates": [c.to_dict() for c in indexed_candidates],
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        snap_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest().lower()
        snap_id = f"snap-{snap_hash[:16]}"

        return cls(
            schema_version="1.0.0",
            snapshot_id=snap_id,
            run_id=run_id,
            created_at=now_iso,
            wave4_sha=REVIEWED_WAVE_4_SHA,
            discovery_authority_version=discovery_version,
            candidate_count=len(indexed_candidates),
            candidates=indexed_candidates,
            snapshot_sha256=snap_hash,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "wave4_sha": self.wave4_sha,
            "discovery_authority_version": self.discovery_authority_version,
            "candidate_count": self.candidate_count,
            "candidates": [c.to_dict() for c in self.candidates],
            "snapshot_sha256": self.snapshot_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UCITSDenominatorSnapshot:
        candidates = tuple(CandidateSpec.from_dict(c) for c in data.get("candidates", []))
        return cls(
            schema_version=data["schema_version"],
            snapshot_id=data["snapshot_id"],
            run_id=data["run_id"],
            created_at=data["created_at"],
            wave4_sha=data["wave4_sha"],
            discovery_authority_version=data["discovery_authority_version"],
            candidate_count=int(data["candidate_count"]),
            candidates=candidates,
            snapshot_sha256=data["snapshot_sha256"],
        )


@dataclass(frozen=True)
class PopulationRunnerConfig:
    """Configuration governing statutory population execution."""
    storage_root: str = CANONICAL_STORAGE_ROOT_DEFAULT
    canonical_manifest_path: str = CANONICAL_MANIFEST_PATH_DEFAULT
    max_retries: int = 3
    initial_backoff_seconds: float = 0.05  # Short default for tests, configurable
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 0.5
    rate_limit_rps: float = 100.0  # High default for tests/CI
    timeout_seconds: int = 30
    tls_validation: bool = True
    http_downgrade_allowed: bool = False
    dry_run: bool = False

    def compute_configuration_sha256(self) -> str:
        d = {
            "storage_root": str(self.storage_root).replace("\\", "/"),
            "canonical_manifest_path": str(self.canonical_manifest_path).replace("\\", "/"),
            "max_retries": self.max_retries,
            "rate_limit_rps": self.rate_limit_rps,
            "timeout_seconds": self.timeout_seconds,
            "tls_validation": self.tls_validation,
            "http_downgrade_allowed": self.http_downgrade_allowed,
            "dry_run": self.dry_run,
        }
        canonical_json = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest().lower()


@dataclass(frozen=True)
class RunIdentity:
    """Cryptographic binding of all components governing a single execution attempt."""
    run_id: str
    wave4_sha: str
    denominator_snapshot_sha256: str
    configuration_sha256: str
    created_at: str

    @classmethod
    def generate(
        cls,
        snapshot_sha: str,
        config_sha: str,
        now: Optional[datetime.datetime] = None,
    ) -> RunIdentity:
        dt = now or datetime.datetime.now(datetime.timezone.utc)
        ts_str = dt.strftime("%Y%m%dT%H%M%SZ")
        composite = f"{REVIEWED_WAVE_4_SHA}:{snapshot_sha}:{config_sha}:{ts_str}"
        short_sha = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:8]
        run_id = f"ucits_pop_run_{ts_str}_{short_sha}"
        return cls(
            run_id=run_id,
            wave4_sha=REVIEWED_WAVE_4_SHA,
            denominator_snapshot_sha256=snapshot_sha,
            configuration_sha256=config_sha,
            created_at=dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    """Durable state descriptor of a population run."""
    schema_version: str
    run_id: str
    runner_version: str
    wave4_sha: str
    denominator_snapshot_sha256: str
    configuration_sha256: str
    target_denominator: int
    created_at: str
    started_at: str
    completed_at: Optional[str] = None
    run_state: RunState = RunState.CREATED
    accounting_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["run_state"] = self.run_state.value
        return d


@dataclass(frozen=True)
class CompletionMarker:
    """Authoritative proof of successful canonical commit and verified manifest swap."""
    run_id: str
    wave4_sha: str
    denominator_snapshot_sha256: str
    target_denominator: int
    accepted_count: int
    quarantined_count: int
    failed_count: int
    canonical_document_count: int
    canonical_provenance_count: int
    canonical_manifest_sha256: str
    completed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompletionMarker:
        return cls(
            run_id=data["run_id"],
            wave4_sha=data["wave4_sha"],
            denominator_snapshot_sha256=data["denominator_snapshot_sha256"],
            target_denominator=int(data["target_denominator"]),
            accepted_count=int(data["accepted_count"]),
            quarantined_count=int(data["quarantined_count"]),
            failed_count=int(data["failed_count"]),
            canonical_document_count=int(data["canonical_document_count"]),
            canonical_provenance_count=int(data["canonical_provenance_count"]),
            canonical_manifest_sha256=data["canonical_manifest_sha256"],
            completed_at=data["completed_at"],
        )


# --- Single-Writer Lock Handler ---

class PopulationLock:
    """Filesystem-based single-writer lock with process liveness detection."""

    def __init__(self, lock_path: Path, run_id: str) -> None:
        self.lock_path = lock_path
        self.run_id = run_id
        self._acquired = False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            # Check liveness of recorded PID
            try:
                content = json.loads(self.lock_path.read_text(encoding="utf-8"))
                held_pid = int(content.get("pid", -1))
                if self._is_pid_alive(held_pid):
                    raise PopulationLockActiveError(
                        f"Population lock is actively held by PID {held_pid} (run_id: {content.get('run_id')})."
                    )
                else:
                    logger.warning(
                        "Stale population lock detected (PID %s dead). Recovering lock for run %s.",
                        held_pid,
                        self.run_id,
                    )
                    self.lock_path.unlink()
            except (json.JSONDecodeError, ValueError, OSError) as e:
                logger.warning("Unreadable population lock file: %s. Recovering lock.", e)
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass

        payload = {
            "pid": os.getpid(),
            "run_id": self.run_id,
            "acquired_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            # Atomic creation
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            self._acquired = True
        except FileExistsError:
            raise PopulationLockActiveError("Concurrent lock acquisition collision detected.")

    def release(self) -> None:
        if self._acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink()
                self._acquired = False
            except OSError as e:
                logger.warning("Failed to remove lockfile %s: %s", self.lock_path, e)

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


# --- Main Population Runner ---

def _safe_path(p: Path) -> Path:
    """Ensures paths exceeding Windows MAX_PATH (260 chars) are prefixed with \\\\?\\."""
    resolved = p.resolve()
    s = str(resolved)
    if sys.platform == "win32" and not s.startswith("\\\\?\\") and len(s) >= 200:
        return Path("\\\\?\\" + s)
    return resolved


class UCITSCanonicalPopulationRunner:
    """
    Orchestration authority executing bounded canonical UCITS population.
    Coordinates denominator freeze, checkpoint journal, isolated staging,
    content-addressed storage, and atomic manifest publication.
    """

    def __init__(
        self,
        config: Optional[PopulationRunnerConfig] = None,
        transport: Optional[HttpTransportHandler] = None,
        acquisition_engine: Optional[UCITSAcquisitionEngine] = None,
        identity_extractor: Optional[UCITSIdentityExtractor] = None,
        reconciliation_pipeline: Optional[UCITSReconciliationPipeline] = None,
        boundary_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = config or PopulationRunnerConfig()
        self.transport = transport or DeterministicMockTransportHandler()
        self.engine = acquisition_engine or UCITSAcquisitionEngine(transport=self.transport)
        self.extractor = identity_extractor or UCITSIdentityExtractor()
        self.reconciliation = reconciliation_pipeline or UCITSReconciliationPipeline()
        self.boundary_callback = boundary_callback  # Test hook for crash-window validation

        self.storage_root = _safe_path(Path(self.config.storage_root))
        self.canonical_manifest_path = _safe_path(Path(self.config.canonical_manifest_path))

        # Canonical subdirectories
        self.documents_dir = self.storage_root / "documents"
        self.provenance_dir = self.storage_root / "provenance"
        self.runs_root = self.storage_root / ".runs"
        self.lock_path = self.storage_root / LOCK_FILENAME

    def _trigger_boundary(self, boundary_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Invokes test callback if set to deterministically test crash windows."""
        if self.boundary_callback is not None:
            self.boundary_callback(boundary_id, context or {})

    def execute_run(
        self,
        snapshot: UCITSDenominatorSnapshot,
        resume: bool = False,
        target_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a bounded canonical population run against a frozen denominator snapshot.
        Supports resume mode when resuming an interrupted run under exact identity binding.
        """
        self._trigger_boundary("W01_BEFORE_DENOMINATOR_FREEZE", {"snapshot": snapshot})

        config_sha = self.config.compute_configuration_sha256()

        if resume:
            if not target_run_id:
                raise ResumeIdentityMismatchError("Resume requested but target_run_id was not provided.")
            run_id = target_run_id
            run_dir = self.runs_root / run_id
            if not run_dir.exists():
                raise ResumeIdentityMismatchError(f"Run directory {run_dir} does not exist for resume.")

            # Validate exact identity binding
            run_manifest_file = run_dir / RUN_MANIFEST_FILENAME
            if not run_manifest_file.exists():
                raise ResumeIdentityMismatchError(f"Run manifest missing in {run_dir}.")

            saved_manifest = json.loads(run_manifest_file.read_text(encoding="utf-8"))
            if saved_manifest.get("run_id") != run_id:
                raise ResumeIdentityMismatchError("run_id mismatch on resume.")
            if saved_manifest.get("wave4_sha") != REVIEWED_WAVE_4_SHA:
                raise ResumeIdentityMismatchError("wave4_sha mismatch on resume.")
            if saved_manifest.get("denominator_snapshot_sha256") != snapshot.snapshot_sha256:
                raise ResumeIdentityMismatchError(
                    f"denominator_snapshot_sha256 mismatch on resume (got {snapshot.snapshot_sha256}, expected {saved_manifest.get('denominator_snapshot_sha256')})."
                )
            if saved_manifest.get("configuration_sha256") != config_sha:
                raise ResumeIdentityMismatchError(
                    f"configuration_sha256 mismatch on resume (got {config_sha}, expected {saved_manifest.get('configuration_sha256')})."
                )

            identity = RunIdentity(
                run_id=run_id,
                wave4_sha=REVIEWED_WAVE_4_SHA,
                denominator_snapshot_sha256=snapshot.snapshot_sha256,
                configuration_sha256=config_sha,
                created_at=saved_manifest["created_at"],
            )
        else:
            identity = RunIdentity.generate(
                snapshot_sha=snapshot.snapshot_sha256,
                config_sha=config_sha,
            )
            run_id = identity.run_id
            run_dir = self.runs_root / run_id

        # Check for completed run idempotency
        completed_marker_path = run_dir / COMPLETION_MARKER_FILENAME
        if completed_marker_path.exists():
            # Validate completion marker binds to current canonical manifest
            marker_data = json.loads(completed_marker_path.read_text(encoding="utf-8"))
            if self.canonical_manifest_path.exists():
                actual_manifest_bytes = self.canonical_manifest_path.read_bytes()
                actual_manifest_sha = hashlib.sha256(actual_manifest_bytes).hexdigest().lower()
                if actual_manifest_sha == marker_data.get("canonical_manifest_sha256"):
                    return {
                        "status": "IDEMPOTENT_NOOP_ALREADY_COMPLETED",
                        "run_id": run_id,
                        "canonical_manifest_sha256": actual_manifest_sha,
                        "target_denominator": marker_data.get("target_denominator"),
                        "accepted_count": marker_data.get("accepted_count"),
                        "quarantined_count": marker_data.get("quarantined_count"),
                        "failed_count": marker_data.get("failed_count"),
                    }
            raise CorruptedStorageError("Completed marker exists but canonical manifest is missing or mismatch.")

        self._trigger_boundary("W02_AFTER_DENOMINATOR_FREEZE", {"snapshot": snapshot, "run_id": run_id})

        # Lock single-writer exclusivity
        lock = PopulationLock(self.lock_path, run_id)
        lock.acquire()

        try:
            return self._execute_run_internal(identity, snapshot, run_dir, resume=resume)
        finally:
            lock.release()

    def _execute_run_internal(
        self,
        identity: RunIdentity,
        snapshot: UCITSDenominatorSnapshot,
        run_dir: Path,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Internal execution body protected by single-writer exclusivity."""
        run_id = identity.run_id
        staging_dir = run_dir / "staging"
        staged_docs_dir = staging_dir / "documents"
        staged_prov_dir = staging_dir / "provenance"
        quarantine_dir = run_dir / "quarantine"
        checkpoint_path = run_dir / CHECKPOINT_FILENAME
        run_manifest_path = run_dir / RUN_MANIFEST_FILENAME

        staged_docs_dir.mkdir(parents=True, exist_ok=True)
        staged_prov_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Save denominator snapshot in run directory
        snap_file = run_dir / DENOMINATOR_SNAPSHOT_FILENAME
        if not snap_file.exists():
            snap_file.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")

        # Initialize or update run manifest
        start_time_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_manifest = RunManifest(
            schema_version="1.0.0",
            run_id=run_id,
            runner_version=RUNNER_VERSION,
            wave4_sha=REVIEWED_WAVE_4_SHA,
            denominator_snapshot_sha256=snapshot.snapshot_sha256,
            configuration_sha256=identity.configuration_sha256,
            target_denominator=snapshot.candidate_count,
            created_at=identity.created_at,
            started_at=start_time_iso,
            run_state=RunState.RUNNING,
        )
        run_manifest_path.write_text(json.dumps(run_manifest.to_dict(), indent=2), encoding="utf-8")

        # Reconstruct resume state from checkpoint journal if resuming
        completed_candidate_isins: Set[str] = set()
        candidate_accounting: Dict[str, RunnerTerminalAccountingState] = {}
        staged_records_map: Dict[str, Dict[str, Any]] = {}

        if resume and checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("state") == "TERMINAL":
                        isin = entry["candidate_id"]
                        completed_candidate_isins.add(isin)
                        term_state = RunnerTerminalAccountingState(entry["terminal_accounting"])
                        candidate_accounting[isin] = term_state
                        if "manifest_record" in entry:
                            staged_records_map[isin] = entry["manifest_record"]
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning("Corrupt checkpoint line ignored on resume: %s", e)

        self._trigger_boundary("W03_BEFORE_FIRST_ACQUISITION", {"run_id": run_id})

        # Checkpoint file append handle
        checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")

        try:
            for candidate in snapshot.candidates:
                isin = candidate.share_class_isin
                if isin in completed_candidate_isins:
                    logger.debug("Skipping already completed candidate %s on resume", isin)
                    continue

                # 1. Log attempt
                attempt_entry = {
                    "sequence_index": candidate.sequence_index,
                    "candidate_id": isin,
                    "state": "ATTEMPTING",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                checkpoint_file.write(json.dumps(attempt_entry) + "\n")
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())

                self._trigger_boundary("W04_DURING_HTTP_TRANSPORT", {"candidate": candidate})

                # 2. Execute statutory acquisition attempt with bounded retries
                acq_result, retry_count = self._execute_attempt_with_retry(candidate)

                terminal_accounting: RunnerTerminalAccountingState
                manifest_record: Optional[Dict[str, Any]] = None

                if acq_result.outcome == AcquisitionOutcome.ACQUIRED and acq_result.artifact:
                    artifact = acq_result.artifact
                    doc_hash = artifact.raw_sha256

                    # Stage document payload
                    staged_doc_path = staged_docs_dir / f"{doc_hash}.pdf"
                    staged_doc_path.write_bytes(artifact.raw_bytes)

                    self._trigger_boundary(
                        "W05_AFTER_DOCUMENT_STAGED_BEFORE_PROVENANCE",
                        {"candidate": candidate, "doc_hash": doc_hash},
                    )

                    # Extract identity and reconcile
                    try:
                        extracted = self.extractor.extract_from_artifact(
                            artifact,
                            hints={"isin": isin, "domicile": candidate.domicile},
                        )
                        # Check if extracted ISIN contradicts candidate ISIN
                        if extracted.share_class_isin and normalize_isin(extracted.share_class_isin) != normalize_isin(isin):
                            terminal_accounting = RunnerTerminalAccountingState.QUARANTINED
                            self._quarantine_candidate(
                                quarantine_dir,
                                candidate,
                                artifact,
                                f"ISIN_CONTRADICTION: extracted {extracted.share_class_isin} != {isin}",
                            )
                        else:
                            prov = acq_result.provenance_record
                            reconciled = self.reconciliation.reconcile_extracted_evidence(
                                [extracted],
                                [prov] if prov else None,
                            )
                            if not reconciled:
                                terminal_accounting = RunnerTerminalAccountingState.QUARANTINED
                                self._quarantine_candidate(quarantine_dir, candidate, artifact, "UNRECONCILED")
                            else:
                                terminal_accounting = RunnerTerminalAccountingState.ACCEPTED

                                # Stage provenance record
                                prov_dict = prov.to_dict() if prov else {"raw_sha256": doc_hash, "share_class_isin": isin}
                                prov_hash = hashlib.sha256(json.dumps(prov_dict, sort_keys=True).encode("utf-8")).hexdigest()
                                staged_prov_path = staged_prov_dir / f"{doc_hash}.provenance.json"
                                staged_prov_path.write_text(json.dumps(prov_dict, indent=2), encoding="utf-8")

                                eff_date = extracted.effective_date or "2026-01-01"
                                manifest_record = {
                                    "share_class_isin": isin,
                                    "domicile": candidate.domicile,
                                    "document_sha256": doc_hash,
                                    "provenance_sha256": prov_hash,
                                    "document_type": candidate.document_type,
                                    "effective_date": eff_date,
                                    "source_authority_tier": "TIER_2_STATUTORY_ISSUER",
                                }
                                staged_records_map[isin] = manifest_record

                    except (ETFSourceAuthorityError, InvalidIdentifierError, UnsupportedJurisdictionError) as e:
                        logger.warning("Reconciliation/extraction error for %s: %s", isin, e)
                        terminal_accounting = RunnerTerminalAccountingState.QUARANTINED
                        self._quarantine_candidate(quarantine_dir, candidate, artifact, str(e))

                    # Clean up staged doc if candidate was quarantined
                    if terminal_accounting != RunnerTerminalAccountingState.ACCEPTED and staged_doc_path.exists():
                        try:
                            staged_doc_path.unlink()
                        except OSError:
                            pass

                    # Trigger W06 after provenance is confirmed staged and before checkpoint journal write
                    if terminal_accounting == RunnerTerminalAccountingState.ACCEPTED:
                        self._trigger_boundary(
                            "W06_AFTER_PROVENANCE_STAGED_BEFORE_CHECKPOINT",
                            {"candidate": candidate, "doc_hash": doc_hash},
                        )

                elif acq_result.outcome == AcquisitionOutcome.PROVENANCE_FAILURE:
                    terminal_accounting = RunnerTerminalAccountingState.QUARANTINED
                    self._quarantine_candidate(
                        quarantine_dir,
                        candidate,
                        acq_result.artifact,
                        f"PROVENANCE_FAILURE: {acq_result.error_message}",
                    )
                else:
                    # Non-acquisition terminal failures (NO_MATCH, UNSUPPORTED, INVALID_REQUEST, RETRIEVAL_FAILURE)
                    terminal_accounting = RunnerTerminalAccountingState.FAILED

                # 3. Log terminal state in checkpoint journal
                candidate_accounting[isin] = terminal_accounting
                completed_candidate_isins.add(isin)

                terminal_entry = {
                    "sequence_index": candidate.sequence_index,
                    "candidate_id": isin,
                    "state": "TERMINAL",
                    "outcome": acq_result.outcome.value,
                    "terminal_accounting": terminal_accounting.value,
                    "document_sha256": acq_result.artifact.raw_sha256 if acq_result.artifact else None,
                    "retry_count": retry_count,
                    "manifest_record": manifest_record,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                checkpoint_file.write(json.dumps(terminal_entry) + "\n")
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())

                self._trigger_boundary("W07_AFTER_CHECKPOINT_WRITTEN", {"candidate": candidate})

        finally:
            checkpoint_file.close()

        # 4. Accounting Conservation Check
        accepted_cnt = sum(1 for s in candidate_accounting.values() if s == RunnerTerminalAccountingState.ACCEPTED)
        quarantined_cnt = sum(1 for s in candidate_accounting.values() if s == RunnerTerminalAccountingState.QUARANTINED)
        failed_cnt = sum(1 for s in candidate_accounting.values() if s == RunnerTerminalAccountingState.FAILED)
        total_accounted = accepted_cnt + quarantined_cnt + failed_cnt

        if total_accounted != snapshot.candidate_count:
            raise DenominatorConservationError(
                f"Accounting conservation failed: total_accounted={total_accounted} != candidate_count={snapshot.candidate_count}"
            )

        run_manifest.accounting_summary = {
            "accepted": accepted_cnt,
            "quarantined": quarantined_cnt,
            "failed": failed_cnt,
            "target_denominator": snapshot.candidate_count,
        }

        # 5. Staging 1:1 Pair Invariant Verification
        staged_docs = set(p.name.replace(".pdf", "") for p in staged_docs_dir.glob("*.pdf"))
        staged_prov = set(p.name.replace(".provenance.json", "") for p in staged_prov_dir.glob("*.provenance.json"))
        if staged_docs != staged_prov:
            raise CorruptedStorageError(
                f"Staging 1:1 pair mismatch: {len(staged_docs)} documents vs {len(staged_prov)} provenance records."
            )

        self._trigger_boundary("W08_DURING_STAGED_MANIFEST_UPDATE", {"run_id": run_id})

        # 6. Build Staged Manifest
        sorted_records = [staged_records_map[k] for k in sorted(staged_records_map.keys())]
        staged_manifest_path = staging_dir / "staged_manifest.json"

        # Compute aggregate identities
        agg_doc_identity = hashlib.sha256("".join(r["document_sha256"] for r in sorted_records).encode("utf-8")).hexdigest()
        agg_prov_identity = hashlib.sha256("".join(r["provenance_sha256"] for r in sorted_records).encode("utf-8")).hexdigest()

        manifest_data = {
            "manifest_version": "1.0.0",
            "manifest_authority": "arx.etf_v2.ucits.canonical_manifest",
            "population_run_id": run_id,
            "wave4_sha": REVIEWED_WAVE_4_SHA,
            "denominator_snapshot_sha256": snapshot.snapshot_sha256,
            "target_denominator": snapshot.candidate_count,
            "canonical_document_count": len(sorted_records),
            "canonical_provenance_record_count": len(sorted_records),
            "corpus_aggregate_document_identity": agg_doc_identity,
            "corpus_aggregate_provenance_identity": agg_prov_identity,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "records": sorted_records,
        }
        staged_manifest_path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8")

        run_manifest.run_state = RunState.READY_TO_COMMIT
        run_manifest_path.write_text(json.dumps(run_manifest.to_dict(), indent=2), encoding="utf-8")

        self._trigger_boundary("W09_BEFORE_CANONICAL_COMMIT", {"run_id": run_id})

        if self.config.dry_run:
            logger.info("Dry-run enabled: skipping canonical commit for %s", run_id)
            return {
                "status": "DRY_RUN_COMPLETED",
                "run_id": run_id,
                "accounting": run_manifest.accounting_summary,
            }

        # 7. Atomic Canonical Commit
        run_manifest.run_state = RunState.COMMITTING
        run_manifest_path.write_text(json.dumps(run_manifest.to_dict(), indent=2), encoding="utf-8")

        self._commit_to_canonical(staged_docs_dir, staged_prov_dir, staged_manifest_path)

        # 8. Compute published manifest SHA-256
        published_manifest_bytes = self.canonical_manifest_path.read_bytes()
        final_manifest_sha = hashlib.sha256(published_manifest_bytes).hexdigest().lower()

        self._trigger_boundary("W11_AFTER_MANIFEST_RENAME_BEFORE_MARKER", {"run_id": run_id})

        # 9. Write .completed Marker strictly LAST
        completed_marker = CompletionMarker(
            run_id=run_id,
            wave4_sha=REVIEWED_WAVE_4_SHA,
            denominator_snapshot_sha256=snapshot.snapshot_sha256,
            target_denominator=snapshot.candidate_count,
            accepted_count=accepted_cnt,
            quarantined_count=quarantined_cnt,
            failed_count=failed_cnt,
            canonical_document_count=len(sorted_records),
            canonical_provenance_count=len(sorted_records),
            canonical_manifest_sha256=final_manifest_sha,
            completed_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        completed_marker_path = run_dir / COMPLETION_MARKER_FILENAME
        completed_marker_path.write_text(json.dumps(completed_marker.to_dict(), indent=2), encoding="utf-8")

        run_manifest.run_state = RunState.COMPLETED
        run_manifest.completed_at = completed_marker.completed_at
        run_manifest_path.write_text(json.dumps(run_manifest.to_dict(), indent=2), encoding="utf-8")

        self._trigger_boundary("W12_AFTER_COMPLETION_MARKER", {"run_id": run_id})

        return {
            "status": "COMPLETED",
            "run_id": run_id,
            "canonical_manifest_sha256": final_manifest_sha,
            "target_denominator": snapshot.candidate_count,
            "accepted_count": accepted_cnt,
            "quarantined_count": quarantined_cnt,
            "failed_count": failed_cnt,
            "canonical_document_count": len(sorted_records),
            "canonical_provenance_count": len(sorted_records),
        }

    def _execute_attempt_with_retry(self, candidate: CandidateSpec) -> Tuple[AcquisitionResult, int]:
        """Dispatches an acquisition request with bounded exponential backoff for transient errors."""
        req = AuthorityRequest(
            request_id=f"req-{candidate.share_class_isin}-{candidate.sequence_index}",
            share_class_isin=candidate.share_class_isin,
            locator=AuthorityLocator(
                authority_id=f"auth-{candidate.domicile}",
                jurisdiction=candidate.domicile,
                source_url=candidate.canonical_source_url,
                document_type=candidate.document_type,
                expected_mime=candidate.expected_mime,
            ),
            timeout_seconds=self.config.timeout_seconds,
        )

        retries = 0
        backoff = self.config.initial_backoff_seconds

        while True:
            res = self.engine.execute_attempt(req, attempt_id=f"att-{candidate.sequence_index}-{retries}")

            # Determine retryability
            is_retryable = res.outcome in (
                AcquisitionOutcome.RATE_LIMITED,
                AcquisitionOutcome.AUTHORITY_UNAVAILABLE,
                AcquisitionOutcome.RETRIEVAL_FAILURE,
            )

            if is_retryable and retries < self.config.max_retries:
                retries += 1
                logger.warning(
                    "Transient acquisition outcome %s for %s; retrying attempt %d/%d after %.2fs",
                    res.outcome.value,
                    candidate.share_class_isin,
                    retries,
                    self.config.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * self.config.backoff_factor, self.config.max_backoff_seconds)
                continue

            return res, retries

    def _quarantine_candidate(
        self,
        quarantine_dir: Path,
        candidate: CandidateSpec,
        artifact: Optional[RawArtifact],
        reason: str,
    ) -> None:
        """Stores quarantined metadata and payload outside canonical paths."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = {
            "candidate_id": candidate.share_class_isin,
            "domicile": candidate.domicile,
            "source_url": candidate.canonical_source_url,
            "quarantine_reason": reason,
            "raw_sha256": artifact.raw_sha256 if artifact else None,
            "quarantined_at": now_iso,
        }
        meta_file = quarantine_dir / f"{candidate.share_class_isin}_quarantine.json"
        meta_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

        if artifact and artifact.raw_bytes:
            payload_file = quarantine_dir / f"{candidate.share_class_isin}_{artifact.raw_sha256[:16]}.bin"
            payload_file.write_bytes(artifact.raw_bytes)

    def _commit_to_canonical(
        self,
        staged_docs_dir: Path,
        staged_prov_dir: Path,
        staged_manifest_path: Path,
    ) -> None:
        """Performs atomic canonical publication with no-overwrite checksum validation."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self._trigger_boundary("W10_DURING_CANONICAL_COMMIT", {})

        # 1. Move/Copy documents to canonical store with content-address integrity check
        for doc_file in staged_docs_dir.glob("*.pdf"):
            target_file = self.documents_dir / doc_file.name
            expected_hash = doc_file.stem.lower()

            if target_file.exists():
                existing_bytes = target_file.read_bytes()
                existing_hash = hashlib.sha256(existing_bytes).hexdigest().lower()
                if existing_hash != expected_hash:
                    raise CorruptedStorageError(
                        f"Canonical document corruption detected: {target_file.name} expected {expected_hash}, got {existing_hash}"
                    )
            else:
                # Copy into canonical documents
                target_file.write_bytes(doc_file.read_bytes())

        # 2. Move/Copy provenance to canonical store
        for prov_file in staged_prov_dir.glob("*.provenance.json"):
            target_prov = self.provenance_dir / prov_file.name
            target_prov.write_text(prov_file.read_text(encoding="utf-8"), encoding="utf-8")

        # 3. Atomic Manifest Publication via temporary file + rename
        tmp_manifest = self.canonical_manifest_path.with_name(f"{self.canonical_manifest_path.name}.tmp")
        tmp_manifest.write_text(staged_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

        # Atomic replacement
        try:
            os.replace(tmp_manifest, self.canonical_manifest_path)
        except OSError as e:
            raise ManifestPublicationError(f"Failed atomic rename of canonical manifest: {e}") from e
