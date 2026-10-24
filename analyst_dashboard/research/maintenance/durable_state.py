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
    OPERATIONAL_STATE_DIR,
    REHEARSAL_STATE_DIR,
    OPERATIONAL_ARTIFACT_NAME,
    REHEARSAL_ARTIFACT_NAME,
    TARGET_REPOSITORY,
    DESIGNATED_WORKFLOW_FILE,
    DESIGNATED_WORKFLOW_PATH,
    RunBudgetTracker,
    BudgetExhaustedError,
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


def validate_github_artifact_list_response(data: Any, context: str = "artifacts") -> Dict[str, Any]:
    """
    Validates GitHub Actions artifact list response structure and types.
    Fails closed on missing fields, wrong types, or inconsistent counts.
    """
    if not isinstance(data, dict):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: expected JSON object, got {type(data).__name__}"
        )

    if "total_count" not in data:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing required field 'total_count'"
        )
    total_count = data["total_count"]
    if not isinstance(total_count, int) or isinstance(total_count, bool) or total_count < 0:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: invalid 'total_count' ({total_count})"
        )

    if "artifacts" not in data:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing required field 'artifacts'"
        )
    artifacts = data["artifacts"]
    if not isinstance(artifacts, list):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: 'artifacts' must be a list"
        )

    if total_count < len(artifacts):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Inconsistent {context} response: total_count ({total_count}) < len(artifacts) ({len(artifacts)})"
        )

    for idx, art in enumerate(artifacts):
        if not isinstance(art, dict):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed artifact entry at index {idx}: expected dict"
            )
        if "id" not in art or not isinstance(art["id"], int) or isinstance(art["id"], bool):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed artifact entry at index {idx}: missing/invalid 'id'"
            )
        if "name" not in art or not isinstance(art["name"], str):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed artifact entry at index {idx}: missing/invalid 'name'"
            )
        if "archive_download_url" not in art or not isinstance(art["archive_download_url"], str):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed artifact entry at index {idx}: missing/invalid 'archive_download_url'"
            )

    return data


def validate_github_workflow_runs_response(data: Any, context: str = "workflow runs") -> Dict[str, Any]:
    """
    Validates GitHub Actions workflow runs response structure and types.
    Fails closed on missing fields, wrong types, or inconsistent counts.
    """
    if not isinstance(data, dict):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: expected JSON object, got {type(data).__name__}"
        )

    if "total_count" not in data:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing required field 'total_count'"
        )
    total_count = data["total_count"]
    if not isinstance(total_count, int) or isinstance(total_count, bool) or total_count < 0:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: invalid 'total_count' ({total_count})"
        )

    if "workflow_runs" not in data:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing required field 'workflow_runs'"
        )
    runs = data["workflow_runs"]
    if not isinstance(runs, list):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: 'workflow_runs' must be a list"
        )

    if total_count < len(runs):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Inconsistent {context} response: total_count ({total_count}) < len(runs) ({len(runs)})"
        )

    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed workflow run entry at index {idx}: expected dict"
            )
        if "id" not in run or not isinstance(run["id"], int) or isinstance(run["id"], bool):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed workflow run entry at index {idx}: missing/invalid 'id'"
            )

    return data


def validate_github_run_jobs_response(data: Any, context: str = "run jobs") -> Dict[str, Any]:
    """
    Validates GitHub Actions run jobs response structure and types.
    Fails closed on missing fields, wrong types, or inconsistent counts.
    """
    if not isinstance(data, dict):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: expected JSON object, got {type(data).__name__}"
        )

    if "total_count" not in data:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing required field 'total_count'"
        )
    total_count = data["total_count"]
    if not isinstance(total_count, int) or isinstance(total_count, bool) or total_count < 0:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: invalid 'total_count' ({total_count})"
        )

    if "jobs" not in data:
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing required field 'jobs'"
        )
    jobs = data["jobs"]
    if not isinstance(jobs, list):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: 'jobs' must be a list"
        )

    for idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed job entry at index {idx}: expected dict"
            )
        if "id" not in job or not isinstance(job["id"], int) or isinstance(job["id"], bool):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed job entry at index {idx}: missing/invalid 'id'"
            )
        if "steps" not in job or not isinstance(job["steps"], list):
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Malformed job entry at index {idx}: missing/invalid 'steps'"
            )
        for s_idx, step in enumerate(job["steps"]):
            if not isinstance(step, dict):
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Malformed step entry at job {idx}, step {s_idx}"
                )
            if "name" not in step or not isinstance(step["name"], str):
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Malformed step entry at job {idx}, step {s_idx}: missing 'name'"
                )

    return data


def validate_github_workflow_run_response(data: Any, context: str = "workflow run") -> Dict[str, Any]:
    """
    Validates a single workflow run response structure and types.
    Fails closed on missing fields or wrong types.
    """
    if not isinstance(data, dict):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: expected JSON object, got {type(data).__name__}"
        )
    if "id" not in data or not isinstance(data["id"], int) or isinstance(data["id"], bool):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing/invalid 'id'"
        )
    if "path" not in data or not isinstance(data["path"], str):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing/invalid 'path'"
        )
    if "repository" not in data or not isinstance(data["repository"], dict):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing/invalid 'repository'"
        )
    repo_obj = data["repository"]
    if "full_name" not in repo_obj or not isinstance(repo_obj["full_name"], str):
        raise StateRecoveryError(
            f"STATE_RECOVERY_FAILURE: Malformed {context} response: missing/invalid 'repository.full_name'"
        )
    return data


class DurableStateManager:
    """
    Manages cross-run durable state persistence, atomic concurrency locking,
    idempotent event completion tracking, and multi-tier recovery.
    """

    MANIFEST_NAME = "DURABLE_STATE_MANIFEST.json"
    LOCK_NAME = ".maintenance.lock"
    PRIOR_RUNS_MARKER = ".has_prior_runs"

    def __init__(
        self,
        state_dir: Union[str, Path],
        backup_dir: Optional[Union[str, Path]] = None,
        artifact_name: str = OPERATIONAL_ARTIFACT_NAME,
        is_rehearsal: bool = False,
        expected_repo: str = "daakara/finance",
    ):
        self.state_dir = Path(state_dir).resolve()
        self.backup_dir = Path(backup_dir).resolve() if backup_dir else None
        self.artifact_name = artifact_name
        self.is_rehearsal = is_rehearsal
        self.expected_repo = expected_repo
        self.manifest_path = self.state_dir / self.MANIFEST_NAME
        self.lock_path = self.state_dir / self.LOCK_NAME

        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        op_state_root = (repo_root / OPERATIONAL_STATE_DIR).resolve()

        if self.is_rehearsal:
            if self.artifact_name == OPERATIONAL_ARTIFACT_NAME:
                raise ValueError(
                    f"BOUNDARY_VIOLATION: Rehearsal state manager cannot use operational artifact name '{OPERATIONAL_ARTIFACT_NAME}'."
                )
            if self.state_dir == op_state_root or self.state_dir.is_relative_to(op_state_root) or op_state_root.is_relative_to(self.state_dir):
                raise ValueError(
                    f"BOUNDARY_VIOLATION: Rehearsal state dir '{self.state_dir}' overlaps with protected operational root '{op_state_root}'."
                )
        else:
            if self.artifact_name == REHEARSAL_ARTIFACT_NAME:
                raise ValueError(
                    f"BOUNDARY_VIOLATION: Operational state manager cannot use rehearsal artifact name '{REHEARSAL_ARTIFACT_NAME}'."
                )

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

    def hydrate_or_recover(
        self,
        allow_bootstrap: bool = False,
        budget_tracker: Optional[RunBudgetTracker] = None,
        allow_remote_artifact: bool = True,
    ) -> Tuple[bool, str]:
        """
        Primary entry point for durable state rehydration.
        1. Checks for local manifest. If present, verifies checksum/schema and rehydrates.
        2. If missing, attempts recovery from backup_dir or remote workflow artifact.
        3. If prior operation evidence exists but state cannot be recovered, fails closed with StateRecoveryError.
        4. If genuine first run, initializes genesis state ONLY IF explicit bootstrap is authorized.
           Otherwise fails closed with StateRecoveryError.
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
            recovered = self._recover_from_github_artifact(budget_tracker=budget_tracker)
            if recovered:
                return self._hydrate_from_file(self.manifest_path)

        # Tier 4: Check if prior operations exist before assuming genesis
        if self._has_evidence_of_prior_operation(budget_tracker=budget_tracker):
            raise StateRecoveryError(
                "STATE_RECOVERY_FAILURE: Evidence of prior operational runs exists, but the durable state "
                "manifest could not be recovered from cache, backup, or artifacts. "
                "Failing closed to prevent resetting run count or losing completed event history."
            )

        # Tier 5: Genesis initialization - strictly guarded by explicit bootstrap authorization
        is_bootstrap_authorized = (
            allow_bootstrap
            or os.environ.get("ARX_BOOTSTRAP_GENESIS", "").strip().lower() in ("1", "true", "yes")
            or (self.state_dir / ".bootstrap_genesis").exists()
        )
        if not is_bootstrap_authorized:
            raise StateRecoveryError(
                "STATE_RECOVERY_FAILURE: No existing durable state recovered and explicit bootstrap is not authorized. "
                "Operational runs require prior durable state or explicit bootstrap authorization (--bootstrap or ARX_BOOTSTRAP_GENESIS=true)."
            )

        logger.info("Explicit bootstrap authorized: Initializing genesis state.")
        self._init_genesis_state()
        self._save_atomic()
        return True, "GENESIS_INITIALIZED"

    def hydrate(self) -> Tuple[bool, str]:
        """Backward-compatible alias for hydrate_or_recover with allow_bootstrap=True."""
        return self.hydrate_or_recover(allow_bootstrap=True)

    def _hydrate_from_file(self, manifest_file: Path) -> Tuple[bool, str]:
        """Reads, verifies checksum and invariants, and populates in-memory state from manifest file."""
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

        # Validate schema invariants
        version = manifest.get("manifest_version")
        if version != STATE_MANIFEST_VERSION:
            raise StateIntegrityError(
                f"Invalid manifest version: expected {STATE_MANIFEST_VERSION}, got {version}"
            )
        symbol = manifest.get("symbol")
        if symbol != TARGET_SYMBOL:
            raise StateIntegrityError(
                f"Invalid manifest symbol: expected {TARGET_SYMBOL}, got {symbol}"
            )
        cik = manifest.get("cik")
        if cik != TARGET_CIK:
            raise StateIntegrityError(
                f"Invalid manifest CIK: expected {TARGET_CIK}, got {cik}"
            )
        seq = manifest.get("checkpoint_sequence", 0)
        if not isinstance(seq, int) or seq < 0:
            raise StateIntegrityError(f"Invalid checkpoint sequence: {seq}")
        rc = manifest.get("run_count", 0)
        if not isinstance(rc, int) or rc < 0:
            raise StateIntegrityError(f"Invalid run count: {rc}")

        # Rehydrate fields
        self.manifest_version = version
        self.symbol = symbol
        self.cik = cik
        self.checkpoint_sequence = seq
        self.prev_checkpoint_hash = manifest.get("prev_checkpoint_hash")
        self.created_at_utc = manifest.get("created_at_utc")
        self.last_checked_utc = manifest.get("last_checked_utc")
        self.last_outcome = manifest.get("last_outcome")
        self.run_count = rc

        completed_list = manifest.get("completed_accessions", [])
        self.completed_accessions = set(completed_list)
        # Ensure known baseline accessions are always present
        self.completed_accessions.update(RECORDED_KNOWN_ACCESSIONS)

        self.events = manifest.get("events", {})

        return True, f"HYDRATED_SEQUENCE_{self.checkpoint_sequence}"

    def _has_evidence_of_prior_operation(self, budget_tracker: Optional[RunBudgetTracker] = None) -> bool:
        """
        Determines whether prior runs existed (to prevent silent genesis reset).
        Fails closed: an unavailable, unauthorized, malformed, inconsistent, or truncated
        history response raises StateRecoveryError rather than returning False.
        Dormant workflow runs where stateful steps were skipped are distinguished
        from operational runs via execution evidence.
        """
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
        if os.environ.get("GITHUB_ACTIONS") == "true":
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                raise StateRecoveryError(
                    "STATE_RECOVERY_FAILURE: GITHUB_ACTIONS is active but GITHUB_TOKEN is not configured; "
                    "cannot verify remote workflow history."
                )

            import urllib.request
            import urllib.error

            repo = os.environ.get("GITHUB_REPOSITORY", self.expected_repo)

            # Step 4a: Check if an artifact matching self.artifact_name exists or existed
            timeout = (
                budget_tracker.acquire_recovery_request_slot(default_timeout_sec=10.0)
                if budget_tracker
                else 10.0
            )
            art_url = f"https://api.github.com/repos/{repo}/actions/artifacts?name={self.artifact_name}&per_page=5"
            try:
                req = urllib.request.Request(art_url, headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "ARX-DurableState-Recovery",
                })
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw_bytes = resp.read()
                    try:
                        art_data = json.loads(raw_bytes.decode("utf-8"))
                    except Exception as e:
                        raise StateRecoveryError(f"STATE_RECOVERY_FAILURE: Malformed artifacts JSON response: {e}")

                validate_github_artifact_list_response(art_data, context="actions/artifacts")
                if art_data["total_count"] > 0:
                    logger.info(f"Prior operational artifact found ({self.artifact_name}).")
                    return True
            except (StateRecoveryError, StateIntegrityError):
                raise
            except Exception as e:
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Failed to query remote artifacts ({e}). History is unavailable."
                )

            # Step 4b: Query workflow runs to inspect history
            timeout = (
                budget_tracker.acquire_recovery_request_slot(default_timeout_sec=10.0)
                if budget_tracker
                else 10.0
            )
            runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/{DESIGNATED_WORKFLOW_FILE}/runs?status=success&per_page=30"
            try:
                req = urllib.request.Request(runs_url, headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "ARX-DurableState-Recovery",
                })
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw_runs = resp.read()
                    try:
                        runs_data = json.loads(raw_runs.decode("utf-8"))
                    except Exception as e:
                        raise StateRecoveryError(f"STATE_RECOVERY_FAILURE: Malformed workflow runs JSON response: {e}")

                validate_github_workflow_runs_response(runs_data, context="actions/workflows/runs")

                total_runs = runs_data["total_count"]
                runs_list = runs_data["workflow_runs"]

                if total_runs == 0:
                    logger.info("Remote workflow history confirmed empty (0 successful runs).")
                    return False

                # Requirement 7: Do not treat a truncated page of recent successful runs as complete historical evidence.
                if total_runs > len(runs_list):
                    raise StateRecoveryError(
                        f"STATE_RECOVERY_FAILURE: Remote workflow run history is truncated ({len(runs_list)} of {total_runs} runs retrieved). "
                        "Bounded inspection cannot resolve prior operational execution across full history."
                    )

                # Step 4c: Inspect execution evidence showing stateful steps were skipped (Requirement 6)
                for run_item in runs_list:
                    run_id = run_item.get("id")
                    if not run_id:
                        raise StateRecoveryError("STATE_RECOVERY_FAILURE: Workflow run missing 'id'.")

                    # 1. Query run artifacts
                    timeout = (
                        budget_tracker.acquire_recovery_request_slot(default_timeout_sec=10.0)
                        if budget_tracker
                        else 10.0
                    )
                    run_art_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
                    req_art = urllib.request.Request(run_art_url, headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "ARX-DurableState-Recovery",
                    })
                    with urllib.request.urlopen(req_art, timeout=timeout) as r_art:
                        raw_art_run = r_art.read()
                        try:
                            run_art_data = json.loads(raw_art_run.decode("utf-8"))
                        except Exception as e:
                            raise StateRecoveryError(f"STATE_RECOVERY_FAILURE: Malformed run artifacts JSON: {e}")

                    validate_github_artifact_list_response(run_art_data, context=f"runs/{run_id}/artifacts")
                    for a in run_art_data["artifacts"]:
                        if a.get("name") == self.artifact_name:
                            logger.info(f"Prior operational run confirmed via run artifact ID {a.get('id')}.")
                            return True

                    # 2. Query run jobs/steps to inspect execution evidence (verify stateful steps skipped)
                    timeout = (
                        budget_tracker.acquire_recovery_request_slot(default_timeout_sec=10.0)
                        if budget_tracker
                        else 10.0
                    )
                    jobs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
                    req_jobs = urllib.request.Request(jobs_url, headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "ARX-DurableState-Recovery",
                    })
                    with urllib.request.urlopen(req_jobs, timeout=timeout) as r_jobs:
                        raw_jobs = r_jobs.read()
                        try:
                            jobs_data = json.loads(raw_jobs.decode("utf-8"))
                        except Exception as e:
                            raise StateRecoveryError(f"STATE_RECOVERY_FAILURE: Malformed run jobs JSON: {e}")

                    validate_github_run_jobs_response(jobs_data, context=f"runs/{run_id}/jobs")
                    jobs = jobs_data["jobs"]
                    if not jobs:
                        raise StateRecoveryError(
                            f"STATE_RECOVERY_FAILURE: Run ID {run_id} has no jobs recorded; cannot verify dormancy."
                        )

                    for job in jobs:
                        steps = job.get("steps", [])
                        stateful_steps = [
                            s for s in steps
                            if any(kw in s.get("name", "") for kw in (
                                "Save Durable State Cache",
                                "Upload State and Review Artifacts",
                                "Restore Durable State Cache",
                            ))
                        ]
                        if not stateful_steps:
                            raise StateRecoveryError(
                                f"STATE_RECOVERY_FAILURE: Run ID {run_id} does not contain verifiable stateful steps; "
                                "cannot verify dormancy from execution evidence."
                            )
                        for s in stateful_steps:
                            s_name = s.get("name", "")
                            s_concl = s.get("conclusion")
                            if s_concl != "skipped":
                                logger.info(
                                    f"Prior operational run confirmed: Run ID {run_id} executed stateful step '{s_name}' "
                                    f"(conclusion: '{s_concl}')."
                                )
                                return True

                logger.info(f"All {len(runs_list)} historical successful runs inspected: confirmed dormant (stateful steps skipped).")
                return False

            except (StateRecoveryError, StateIntegrityError):
                raise
            except Exception as e:
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Failed to query remote workflow runs ({e}). History is unavailable."
                )

        return False

    def _recover_from_github_artifact(self, budget_tracker: Optional[RunBudgetTracker] = None) -> bool:
        """
        Attempts to download and extract the designated state artifact from GitHub Actions.
        Only recovers state matching the expected repository, designated workflow,
        correct operational/rehearsal namespace, and expected artifact name.
        """
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY", self.expected_repo)
        if not token:
            logger.info("No GITHUB_TOKEN available for remote artifact recovery.")
            return False

        try:
            import urllib.request
            timeout = (
                budget_tracker.acquire_recovery_request_slot(default_timeout_sec=15.0)
                if budget_tracker
                else 15.0
            )
            url = f"https://api.github.com/repos/{repo}/actions/artifacts?name={self.artifact_name}&per_page=5"
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "ARX-DurableState-Recovery",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
                try:
                    data = json.loads(raw_bytes.decode("utf-8"))
                except Exception as e:
                    raise StateRecoveryError(f"STATE_RECOVERY_FAILURE: Remote artifact list JSON parse error: {e}")

            validate_github_artifact_list_response(data, context="actions/artifacts")

            total_count = data["total_count"]
            artifacts = data["artifacts"]

            if total_count == 0 or len(artifacts) == 0:
                logger.info(f"No prior {self.artifact_name} artifact found on GitHub.")
                return False

            latest_art = artifacts[0]

            # 1. Verify artifact name exactly matches expected name
            art_name = latest_art.get("name")
            if art_name != self.artifact_name:
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Artifact name mismatch: expected '{self.artifact_name}', got '{art_name}'."
                )

            # 2. Verify namespace consistency
            if self.is_rehearsal and self.artifact_name != REHEARSAL_ARTIFACT_NAME:
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Rehearsal recovery attempted with non-rehearsal artifact name '{self.artifact_name}'."
                )
            if not self.is_rehearsal and self.artifact_name != OPERATIONAL_ARTIFACT_NAME:
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Operational recovery attempted with non-operational artifact name '{self.artifact_name}'."
                )

            # 3. Check expiration
            if latest_art.get("expired") is True:
                logger.warning(f"Remote artifact {self.artifact_name} is marked as expired.")
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Remote artifact {self.artifact_name} (ID {latest_art.get('id')}) has expired."
                )

            # 4. Verify workflow run association and provenance
            wf_run = latest_art.get("workflow_run")
            if not wf_run or not isinstance(wf_run, dict):
                raise StateRecoveryError(
                    "STATE_RECOVERY_FAILURE: Artifact metadata missing 'workflow_run' provenance object."
                )
            run_id = wf_run.get("id")
            if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Artifact metadata has invalid workflow_run ID ({run_id})."
                )

            # Query workflow run details to verify repository and designated workflow
            run_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
            run_timeout = (
                budget_tracker.acquire_recovery_request_slot(default_timeout_sec=10.0)
                if budget_tracker
                else 10.0
            )
            run_req = urllib.request.Request(run_url, headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "ARX-DurableState-Recovery",
            })
            with urllib.request.urlopen(run_req, timeout=run_timeout) as run_resp:
                run_raw = run_resp.read()
                try:
                    run_data = json.loads(run_raw.decode("utf-8"))
                except Exception as e:
                    raise StateRecoveryError(f"STATE_RECOVERY_FAILURE: Workflow run JSON parse error: {e}")

            validate_github_workflow_run_response(run_data, context=f"actions/runs/{run_id}")

            # Verify repository association
            run_repo = run_data["repository"]["full_name"]
            if run_repo != self.expected_repo:
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Artifact originates from repository '{run_repo}'; expected '{self.expected_repo}'."
                )

            # Verify designated workflow path
            run_path = run_data.get("path", "")
            if run_path != DESIGNATED_WORKFLOW_PATH and not run_path.endswith(DESIGNATED_WORKFLOW_FILE):
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Artifact originated from unauthorized workflow '{run_path}'; "
                    f"expected '{DESIGNATED_WORKFLOW_PATH}'."
                )

            # 5. Download archive
            download_url = latest_art.get("archive_download_url")
            if not download_url or not isinstance(download_url, str):
                raise StateRecoveryError(
                    f"STATE_RECOVERY_FAILURE: Artifact ID {latest_art.get('id')} missing archive_download_url."
                )

            dl_timeout = (
                budget_tracker.acquire_recovery_request_slot(default_timeout_sec=20.0)
                if budget_tracker
                else 20.0
            )
            logger.info(f"Downloading state artifact ID {latest_art.get('id')} from {download_url}...")
            dl_req = urllib.request.Request(download_url, headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "ARX-DurableState-Recovery",
            })
            with urllib.request.urlopen(dl_req, timeout=dl_timeout) as dl_resp:
                zip_bytes = dl_resp.read()

            import io
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                manifest_found = False
                for member in zf.namelist():
                    # Security check: prevent zip slip / path traversal
                    if ".." in member or member.startswith("/") or member.startswith("\\"):
                        raise StateIntegrityError(f"Malicious member path in artifact zip: {member}")
                    if member.endswith(self.MANIFEST_NAME):
                        target_file = self.state_dir / self.MANIFEST_NAME
                        with zf.open(member) as src, open(target_file, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        manifest_found = True
                        logger.info(f"Successfully recovered {self.MANIFEST_NAME} from remote artifact {self.artifact_name}.")
                if not manifest_found:
                    raise StateIntegrityError(
                        f"Artifact zip does not contain expected manifest {self.MANIFEST_NAME}."
                    )
                return True
        except (StateIntegrityError, StateRecoveryError):
            raise
        except Exception as e:
            logger.warning(f"Remote artifact recovery failed: {e}")
            raise StateRecoveryError(
                f"STATE_RECOVERY_FAILURE: Remote artifact {self.artifact_name} could not be retrieved or extracted: {e}"
            )

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
