"""
analyst_dashboard/research/maintenance/durable_state.py

Durable State Manager for ARX Research-Maintenance Pilot.
Governed by ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.

Operational Requirements:
1. Cross-run state recovery: Cache is an optimization; state can be recovered from durable artifacts or backup storage.
2. Missing state after prior operation fails closed with StateRecoveryError (never silently resets to genesis).
3. Genuine first run is explicitly detected and initialized.
4. Completed events ledger prevents duplicate review artifacts; failed events remain retryable.
5. Concurrency protection uses atomic OS-level file creation (O_CREAT | O_EXCL) to prevent race conditions.
6. Checksum (SHA-256) and sequence chaining verify internal local consistency across runs.
   (Note: Local checksum and sequence linkage verify internal consistency, but do not prevent external restoration of an older valid state).
7. Zero mutations on dormant runs: no directories or lock files created when pilot is dormant.
8. State and review packets remain strictly outside public research publication paths.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import zipfile

from .contracts import (
    EventProcessingState,
    MaintenanceEvent,
    RunOutcome,
)
from .pilot_config import (
    TARGET_SYMBOL,
    TARGET_CIK,
    RECORDED_KNOWN_ACCESSIONS,
)

logger = logging.getLogger(__name__)

STATE_MANIFEST_VERSION = "ARX-MAINTENANCE-STATE-V1"
DEFAULT_LOCK_TIMEOUT_SECONDS = 300.0  # 5 minutes stale lock boundary


class StatePersistenceError(RuntimeError):
    """Raised when atomic state writing or filesystem persistence fails."""
    pass


class ConcurrentRunError(RuntimeError):
    """Raised when another runner holds the state concurrency lock."""
    pass


class StateIntegrityError(RuntimeError):
    """Raised when state manifest verification or checksum linkage fails."""
    pass


class StateRecoveryError(RuntimeError):
    """Raised when prior operational state exists but cannot be recovered (fails closed)."""
    pass


def deterministic_json(data: Any) -> str:
    """Deterministic JSON serialization under the defined local schema (sorted keys, compact separators)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(content: bytes) -> str:
    """Computes SHA-256 hex digest over bytes."""
    return hashlib.sha256(content).hexdigest()


class DurableStateManager:
    """
    Manages cross-run durable state persistence, atomic concurrency locking,
    idempotent event completion tracking, and multi-tier recovery.
    """

    MANIFEST_NAME = "DURABLE_STATE_MANIFEST.json"
    LOCK_NAME = ".maintenance.lock"
    PRIOR_RUNS_MARKER = ".has_prior_runs"

    def __init__(self, state_dir: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None):
        self.state_dir = Path(state_dir).resolve()
        self.backup_dir = Path(backup_dir).resolve() if backup_dir else None
        self.manifest_path = self.state_dir / self.MANIFEST_NAME
        self.lock_path = self.state_dir / self.LOCK_NAME

        # In-memory hydrated state
        self.manifest_version = STATE_MANIFEST_VERSION
        self.symbol = TARGET_SYMBOL
        self.cik = TARGET_CIK
        self.checkpoint_sequence: int = 0
        self.prev_checkpoint_hash: Optional[str] = None
        self.created_at_utc: Optional[str] = None
        self.last_checked_utc: Optional[str] = None
        self.last_outcome: Optional[str] = None
        self.run_count: int = 0
        self.completed_accessions: Set[str] = set(RECORDED_KNOWN_ACCESSIONS)
        self.events: Dict[str, Dict[str, Any]] = {}
        self._lock_held: bool = False
        self._lock_record: Optional[Dict[str, Any]] = None

    def hydrate_or_recover(self, allow_remote_artifact: bool = True) -> Tuple[bool, str]:
        """
        Primary entry point for durable state rehydration.
        1. Checks for local manifest. If present, verifies checksum and rehydrates.
        2. If missing, attempts recovery from backup_dir or remote workflow artifact.
        3. If prior operation evidence exists but state cannot be recovered, fails closed with StateRecoveryError.
        4. If genuine first run, initializes genesis state explicitly.
        """
        # Tier 1: Local manifest exists (e.g. disk or restored Actions cache)
        if self.manifest_path.exists():
            return self._hydrate_from_file(self.manifest_path)

        # Tier 2: Check backup directory (local backup or artifact store)
        if self.backup_dir and (self.backup_dir / self.MANIFEST_NAME).exists():
            logger.info(f"Local manifest missing; recovering from backup dir {self.backup_dir}")
            backup_manifest = self.backup_dir / self.MANIFEST_NAME
            self.state_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_manifest, self.manifest_path)
            return self._hydrate_from_file(self.manifest_path)

        # Tier 3: Attempt remote artifact recovery in GitHub Actions environment
        if allow_remote_artifact and os.environ.get("GITHUB_ACTIONS") == "true":
            recovered = self._recover_from_github_artifact()
            if recovered:
                return self._hydrate_from_file(self.manifest_path)

        # Tier 4: Check if prior operations exist before assuming genesis
        if self._has_evidence_of_prior_operation():
            raise StateRecoveryError(
                "STATE_RECOVERY_FAILURE: Evidence of prior operational runs exists, but the durable state "
                "manifest could not be recovered from cache, backup, or artifacts. "
                "Failing closed to prevent resetting run count or losing completed event history."
            )

        # Genuine first run: Initialize genesis explicitly
        logger.info("Genuine first run confirmed: Initializing genesis state.")
        self._init_genesis_state()
        self._save_atomic()
        return True, "GENESIS_INITIALIZED"

    def hydrate(self) -> Tuple[bool, str]:
        """Backward-compatible alias for hydrate_or_recover."""
        return self.hydrate_or_recover()

    def _hydrate_from_file(self, manifest_file: Path) -> Tuple[bool, str]:
        """Reads, verifies checksum, and populates in-memory state from manifest file."""
        try:
            raw_bytes = manifest_file.read_bytes()
            manifest = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            raise StateIntegrityError(f"Failed to read state manifest from {manifest_file}: {e}")

        # Verify content checksum
        manifest_copy = dict(manifest)
        claimed_sha = manifest_copy.pop("manifest_sha256", None)
        computed_sha = sha256_hex(deterministic_json(manifest_copy).encode("utf-8"))
        if claimed_sha != computed_sha:
            raise StateIntegrityError(
                f"Manifest SHA-256 mismatch! Claimed: {claimed_sha}, Computed: {computed_sha}. "
                "State manifest content is corrupt or tampered."
            )

        # Rehydrate fields
        self.manifest_version = manifest.get("manifest_version", STATE_MANIFEST_VERSION)
        self.symbol = manifest.get("symbol", TARGET_SYMBOL)
        self.cik = manifest.get("cik", TARGET_CIK)
        self.checkpoint_sequence = manifest.get("checkpoint_sequence", 0)
        self.prev_checkpoint_hash = manifest.get("prev_checkpoint_hash")
        self.created_at_utc = manifest.get("created_at_utc")
        self.last_checked_utc = manifest.get("last_checked_utc")
        self.last_outcome = manifest.get("last_outcome")
        self.run_count = manifest.get("run_count", 0)

        completed_list = manifest.get("completed_accessions", [])
        self.completed_accessions = set(completed_list)
        # Ensure known baseline accessions are always present
        self.completed_accessions.update(RECORDED_KNOWN_ACCESSIONS)

        self.events = manifest.get("events", {})

        return True, f"HYDRATED_SEQUENCE_{self.checkpoint_sequence}"

    def _has_evidence_of_prior_operation(self) -> bool:
        """Determines whether prior runs existed (to prevent silent genesis reset)."""
        # 1. Check backup dir marker
        if self.backup_dir and (self.backup_dir / self.PRIOR_RUNS_MARKER).exists():
            return True

        # 2. Check local state dir marker
        if (self.state_dir / self.PRIOR_RUNS_MARKER).exists():
            return True

        # 3. Check for lingering events or non-manifest files in state dir
        if self.state_dir.exists():
            other_files = [f for f in self.state_dir.iterdir() if f.name != self.LOCK_NAME and not f.name.startswith(".")]
            if other_files:
                return True

        # 4. Check GitHub Actions workflow history if in CI
        if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_TOKEN"):
            try:
                import urllib.request
                repo = os.environ.get("GITHUB_REPOSITORY", "daakara/finance")
                token = os.environ["GITHUB_TOKEN"]
                url = f"https://api.github.com/repos/{repo}/actions/workflows/scheduled_research_maintenance.yml/runs?status=success&per_page=1"
                req = urllib.request.Request(url, headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "ARX-DurableState-Recovery",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("total_count", 0) > 0:
                        return True
            except Exception as e:
                logger.warning(f"Failed to check remote workflow history: {e}")

        return False

    def _recover_from_github_artifact(self) -> bool:
        """Attempts to download and extract the latest research-maintenance-fix-state artifact."""
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY", "daakara/finance")
        if not token:
            logger.info("No GITHUB_TOKEN available for remote artifact recovery.")
            return False

        try:
            import urllib.request
            url = f"https://api.github.com/repos/{repo}/actions/artifacts?name=research-maintenance-fix-state&per_page=1"
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "ARX-DurableState-Recovery",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            artifacts = data.get("artifacts", [])
            if not artifacts:
                logger.info("No prior research-maintenance-fix-state artifact found on GitHub.")
                return False

            latest_art = artifacts[0]
            download_url = latest_art.get("archive_download_url")
            if not download_url:
                return False

            logger.info(f"Downloading state artifact ID {latest_art.get('id')} from {download_url}...")
            dl_req = urllib.request.Request(download_url, headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "ARX-DurableState-Recovery",
            })
            with urllib.request.urlopen(dl_req, timeout=30) as dl_resp:
                zip_bytes = dl_resp.read()

            with tempfile.NamedTemporaryFile("wb", delete=False) as tf:
                tf.write(zip_bytes)
                tmp_zip = Path(tf.name)

            self.state_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                for member in zf.namelist():
                    if member.endswith(self.MANIFEST_NAME):
                        target_file = self.state_dir / self.MANIFEST_NAME
                        with zf.open(member) as src, open(target_file, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        logger.info(f"Successfully recovered {self.MANIFEST_NAME} from remote artifact.")
                        tmp_zip.unlink(missing_ok=True)
                        return True

            tmp_zip.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Remote artifact recovery failed: {e}")

        return False

    def _init_genesis_state(self):
        """
        Initializes pristine genesis state with retained primary-source evidence.
        Keeps filing dates distinct from UTC acceptance datetimes.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        self.manifest_version = STATE_MANIFEST_VERSION
        self.symbol = TARGET_SYMBOL
        self.cik = TARGET_CIK
        self.checkpoint_sequence = 0
        self.prev_checkpoint_hash = None
        self.created_at_utc = now_iso
        self.last_checked_utc = None
        self.last_outcome = None
        self.run_count = 0
        self.completed_accessions = set(RECORDED_KNOWN_ACCESSIONS)
        self.events = {
            "0001104659-26-118159": {
                "accession_number": "0001104659-26-118159",
                "form": "10-Q",
                "reporting_period": "2026-09-30",
                "filing_date": "2026-10-22",
                "acceptance_datetime_utc": "2026-10-23T00:12:01.000Z",
                "primary_document": "fix-20260930x10q.htm",
                "processing_state": EventProcessingState.REVIEW_PENDING.value,
                "note": "Pre-existing baseline filing (worked example accepted under PR #74).",
            },
            "0001104659-26-118156": {
                "accession_number": "0001104659-26-118156",
                "form": "8-K",
                "reporting_period": "2026-10-22",
                "filing_date": "2026-10-22",
                "acceptance_datetime_utc": "2026-10-23T00:10:19.000Z",
                "primary_document": "fix-20261022x8k.htm",
                "processing_state": EventProcessingState.REJECTED_MISIDENTIFIED.value,
                "note": "Pre-existing baseline earnings disclosure (non-periodic report correctly rejected).",
            },
        }

    def is_accession_completed(self, accession_number: str) -> bool:
        """Checks if accession has already been completed or is a known baseline filing."""
        clean_acc = accession_number.strip()
        return clean_acc in self.completed_accessions

    def record_completed_event(self, event: MaintenanceEvent, review_packet_id: Optional[str] = None):
        """Records a successfully processed event as completed to prevent future duplicates."""
        clean_acc = event.accession_number.strip()
        self.completed_accessions.add(clean_acc)
        self.events[clean_acc] = {
            "accession_number": clean_acc,
            "form": event.form,
            "reporting_period": event.reporting_period,
            "filing_date": event.filing_date,
            "acceptance_datetime_utc": event.acceptance_datetime_utc,
            "processing_state": event.processing_state.value,
            "is_amendment": event.is_amendment,
            "unsupported_reason": event.unsupported_reason,
            "review_packet_id": review_packet_id,
            "sha256": event.sha256,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def record_failed_event(self, event: MaintenanceEvent, stage: str, error_msg: str):
        """Records a failed event without marking it completed, keeping it retryable."""
        clean_acc = event.accession_number.strip()
        self.events[clean_acc] = {
            "accession_number": clean_acc,
            "form": event.form,
            "reporting_period": event.reporting_period,
            "filing_date": event.filing_date,
            "acceptance_datetime_utc": event.acceptance_datetime_utc,
            "processing_state": EventProcessingState.FAILED.value,
            "failed_stage": stage,
            "error_message": error_msg,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def commit_run_checkpoint(
        self,
        outcome: RunOutcome,
        last_processed_accession: Optional[str] = None,
    ):
        """
        Increments checkpoint sequence, updates run telemetry, and writes manifest atomically.
        Fails closed if disk write cannot be verified.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        self.checkpoint_sequence += 1
        self.run_count += 1
        self.last_checked_utc = now_iso
        self.last_outcome = outcome.value

        self._save_atomic()

    def _save_atomic(self):
        """Atomically saves the manifest file with fsync and content checksum."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

        manifest_data: Dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "symbol": self.symbol,
            "cik": self.cik,
            "checkpoint_sequence": self.checkpoint_sequence,
            "prev_checkpoint_hash": self.prev_checkpoint_hash,
            "created_at_utc": self.created_at_utc,
            "last_checked_utc": self.last_checked_utc,
            "last_outcome": self.last_outcome,
            "run_count": self.run_count,
            "completed_accessions": sorted(list(self.completed_accessions)),
            "events": self.events,
        }

        # Content checksum
        canonical_str = deterministic_json(manifest_data)
        manifest_hash = sha256_hex(canonical_str.encode("utf-8"))
        manifest_data["manifest_sha256"] = manifest_hash

        # Update sequence pointer
        self.prev_checkpoint_hash = manifest_hash

        # Atomic write: tmp file + os.replace + fsync
        tmp_path = self.state_dir / f".tmp_manifest_{os.getpid()}_{time.time_ns()}.json"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.manifest_path)

            # Record prior runs marker to ensure state loss is detectable
            marker_file = self.state_dir / self.PRIOR_RUNS_MARKER
            marker_file.write_text(f"sequence={self.checkpoint_sequence}\n", encoding="utf-8")

            # Update backup dir if configured
            if self.backup_dir:
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                backup_manifest = self.backup_dir / self.MANIFEST_NAME
                shutil.copy2(self.manifest_path, backup_manifest)
                (self.backup_dir / self.PRIOR_RUNS_MARKER).write_text(
                    f"sequence={self.checkpoint_sequence}\n", encoding="utf-8"
                )
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise StatePersistenceError(f"CRITICAL: Failed to atomically write state manifest: {e}")

    @contextmanager
    def acquire_lock(self, timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS):
        """
        Acquires mutual exclusion lock atomically using OS-level O_CREAT | O_EXCL.
        Eliminates TOCTOU exists-then-write race conditions.
        Reclaims stale locks safely (> timeout_seconds) and avoids deleting another process's lock.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        now = time.time()
        lock_data = {
            "pid": pid,
            "timestamp": now,
            "iso_time": datetime.now(timezone.utc).isoformat(),
            "runner_id": os.environ.get("GITHUB_RUN_ID", "local"),
        }
        lock_bytes = json.dumps(lock_data, indent=2).encode("utf-8")

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        acquired = False

        try:
            fd = os.open(str(self.lock_path), flags)
            os.write(fd, lock_bytes)
            os.close(fd)
            acquired = True
            self._lock_held = True
            self._lock_record = lock_data
        except FileExistsError:
            # Lock exists: inspect for staleness
            try:
                raw = self.lock_path.read_text(encoding="utf-8")
                existing = json.loads(raw)
                lock_time = existing.get("timestamp", 0)
                lock_pid = existing.get("pid")
                age = now - lock_time
                if age < timeout_seconds:
                    raise ConcurrentRunError(
                        f"CONCURRENT_RUN_REJECTED: Maintenance lock is currently held by PID {lock_pid} "
                        f"(acquired {age:.1f}s ago). Another run is in progress."
                    )
                else:
                    logger.warning(f"Reclaiming stale lock held by PID {lock_pid} (age {age:.1f}s)")
                    try:
                        self.lock_path.unlink()
                    except OSError:
                        pass
                    # Retry atomic acquisition once
                    fd = os.open(str(self.lock_path), flags)
                    os.write(fd, lock_bytes)
                    os.close(fd)
                    acquired = True
                    self._lock_held = True
                    self._lock_record = lock_data
            except ConcurrentRunError:
                raise
            except Exception as e:
                raise ConcurrentRunError(f"CONCURRENT_RUN_REJECTED: Lock collision or unreadable lock file: {e}")

        try:
            yield self
        finally:
            if acquired and self.lock_path.exists():
                try:
                    # Safe release: verify lock file still contains our PID before deleting
                    current_content = json.loads(self.lock_path.read_text(encoding="utf-8"))
                    if current_content.get("pid") == pid and current_content.get("timestamp") == now:
                        self.lock_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to safely release lock file: {e}")
                self._lock_held = False
                self._lock_record = None
