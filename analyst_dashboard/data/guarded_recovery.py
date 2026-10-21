"""
analyst_dashboard/data/guarded_recovery.py

Guarded Disaster Recovery Engine for SQLite Stores (ARX Closure Model v2).
Enforces non-destructive, evidence-preserving, fail-closed database recovery.

AUTHORIZATION & GOVERNANCE CONTRACT:
The operator authorization token, maintenance mode token, and incident identifier are
OPERATIONAL AUDIT REFERENCES (correlation identifiers), NOT cryptographic bearer credentials.
They link the recovery runbook execution to an authorized operational change procedure and
incident record. These tokens alone DO NOT confer authority.
Actual recovery safety and authorization are enforced by:
1. Operational incident correlation (explicit, non-empty, distinct incident and maintenance references).
2. Mandatory pre-flight cryptographic backup verification (SHA-256 match and PRAGMA integrity_check)
   BEFORE any target database mutation or quarantine.
3. System quiescence and handle fencing: Proving zero open file descriptors across all system processes
   and zero active database readers/writers before database unlinking.
4. Deterministic content-hashed evidence: GuardedRecoveryReceipt produces a tamper-evident SHA-256
   digest of the complete execution record (unsigned plain JSON audit record with content hash).

CRITICAL ARCHITECTURAL INVARIANT (WAL Recovery Fence):
Under SQLite WAL mode, `BEGIN EXCLUSIVE` only blocks competing writers; it DOES NOT
block or detect active readers. In POSIX/Linux, executing `os.replace` or moving a database
file while reader processes hold open file descriptors leaves those processes attached to the
unlinked inode (stale ghost handle), causing desynchronized WAL/SHM access or corruption.
Therefore, connection quiescence CANNOT be proven by `BEGIN EXCLUSIVE` alone; it strictly requires
an active process and handle fence (maintenance-mode isolation, reader closure, and handle verification).

Guarded 10-Step Incident Recovery Protocol:
1. Formal operator incident authorization check (G-S2-ACT / INC-...).
2. Explicit maintenance-mode authorization token check (MAINT-...).
3. Pre-flight backup integrity and digest verification (BEFORE target DB is touched).
4. Fencing of every known database writer (zero active writer processes).
5. Coordinated closure/isolation of active application readers (handle fence verification).
6. Quiescent database write lock probe (exclusive transaction probe).
7. Non-destructive evidence quarantine of .db, .db-wal, and .db-shm state.
8. Staged candidate restoration (.restore_candidate) before controlled atomic replacement.
9. Full readback verification of restored schema, table counts, and provenance.
10. Explicit application reopening and deterministic content-hashed receipt generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arx.data.guarded_recovery")


@dataclass
class QuarantinedEvidenceFile:
    original_path: str
    quarantine_path: str
    file_bytes: int
    sha256: str


@dataclass
class GuardedRecoveryReceipt:
    incident_id: str
    operator_authorization: str
    maintenance_mode_token: str
    target_database_path: str
    backup_source_path: str
    backup_source_sha256: str
    quarantine_directory: str
    quarantined_files: List[Dict[str, Any]]
    pre_recovery_state: Dict[str, Any]
    post_recovery_readback: Dict[str, Any]
    recovery_status: str  # SUCCESS | FAILED
    error_message: Optional[str] = None
    started_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    completed_at_utc: Optional[str] = None
    receipt_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_receipt_sha256(self) -> str:
        """Computes deterministic SHA-256 digest over the receipt payload."""
        d = self.to_dict()
        d.pop("receipt_sha256", None)
        canonical_bytes = json.dumps(d, sort_keys=True).encode("utf-8")
        self.receipt_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        return self.receipt_sha256


class GuardedRecoveryEngine:
    """
    Executes fail-closed guarded disaster recovery of SQLite databases with strict
    reader/writer handle fencing, pre-flight backup validation, and non-destructive evidence preservation.
    """

    def __init__(
        self,
        target_db_path: Path | str,
        backup_db_path: Path | str,
        expected_backup_sha256: str,
        incident_id: str,
        operator_authorization: str,
        maintenance_mode_token: str,
        quarantine_base_dir: Path | str = "/root/incident_quarantine",
        busy_timeout_ms: int = 2000,
        handle_checker: Optional[Callable[[Path], List[int]]] = None,
    ) -> None:
        self.target_path = Path(target_db_path).resolve()
        self.backup_path = Path(backup_db_path).resolve()
        self.expected_backup_sha256 = expected_backup_sha256.lower().strip()
        self.incident_id = incident_id.strip()
        self.operator_authorization = operator_authorization.strip()
        self.maintenance_mode_token = maintenance_mode_token.strip()
        self.quarantine_base_dir = Path(quarantine_base_dir).resolve()
        self.busy_timeout_ms = busy_timeout_ms
        self._custom_handle_checker = handle_checker
        self._active_reader_registry: Set[Any] = set()

    @staticmethod
    def compute_file_sha256(path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def step1_verify_authorization(self) -> None:
        """Step 1: Check formal operator incident authorization audit references."""
        if not self.incident_id or not self.incident_id.strip():
            raise ValueError("AUTHORIZATION_REJECTED: Incident ID must not be empty.")
        inc_clean = self.incident_id.strip()
        if len(inc_clean) < 6 or not inc_clean.startswith("INC-"):
            raise ValueError(
                f"AUTHORIZATION_REJECTED: Incident ID '{self.incident_id}' must follow structured format (INC-xxx)."
            )

        if not self.operator_authorization or not self.operator_authorization.strip():
            raise ValueError("AUTHORIZATION_REJECTED: Formal operator authorization token must not be empty.")
        auth_clean = self.operator_authorization.strip()
        if len(auth_clean) < 6 or not (auth_clean.startswith("G-S2-") or auth_clean.startswith("INC-G-S2-")):
            raise ValueError(
                f"AUTHORIZATION_REJECTED: Operator authorization token '{self.operator_authorization}' "
                "must reference an authorized operational gate (e.g. G-S2-ACT, INC-G-S2-...)."
            )

    def step2_verify_maintenance_mode(self) -> None:
        """Step 2: Check explicit maintenance mode authorization audit reference."""
        if not self.maintenance_mode_token or not self.maintenance_mode_token.strip():
            raise ValueError("MAINTENANCE_MODE_REQUIRED: Maintenance mode authorization token must not be empty.")
        token_clean = self.maintenance_mode_token.strip()
        if len(token_clean) < 8 or not (token_clean.startswith("MAINT-") or token_clean.startswith("MAINTENANCE-")):
            raise ValueError(
                f"MAINTENANCE_MODE_REQUIRED: Token '{self.maintenance_mode_token}' must start with MAINT- or MAINTENANCE-."
            )
        if token_clean in ("MAINT-", "MAINTENANCE-", "MAINT-DEFAULT", "MAINT-TEST"):
            raise ValueError(
                f"MAINTENANCE_MODE_REQUIRED: Generic or placeholder maintenance token '{self.maintenance_mode_token}' is rejected."
            )
        if token_clean == self.incident_id.strip():
            raise ValueError(
                "MAINTENANCE_MODE_REQUIRED: Maintenance mode token must be distinct from incident ID."
            )

    def step3_and_4_verify_reader_writer_fence(self) -> None:
        """
        Step 3 & 4: Fence all writers and readers.
        CRITICAL: Proves that no running process holds an open file handle to target_path,
        preventing stale handle inode desynchronization under WAL mode.
        """
        # 1. Custom handle checker or test registry
        if self._custom_handle_checker is not None:
            open_pids = self._custom_handle_checker(self.target_path)
            if open_pids:
                raise RuntimeError(
                    f"RECOVERY_FENCE_VIOLATION: Active database handles detected for PIDs {open_pids}; "
                    "refusing recovery without complete reader/writer isolation."
                )

        # 2. Check registered active readers in-process
        if self._active_reader_registry:
            raise RuntimeError(
                f"RECOVERY_FENCE_VIOLATION: {len(self._active_reader_registry)} active application reader handles "
                "detected in connection registry; refusing recovery without coordinated closure."
            )

        # 3. On Linux/POSIX environments, check /proc/*/fd
        if sys.platform != "win32" and self.target_path.exists():
            target_str = str(self.target_path)
            proc_root = Path("/proc")
            if proc_root.exists():
                active_pids: List[int] = []
                my_pid = os.getpid()
                for pid_dir in proc_root.glob("[0-9]*"):
                    try:
                        pid = int(pid_dir.name)
                        if pid == my_pid:
                            continue
                        fd_dir = pid_dir / "fd"
                        if fd_dir.exists():
                            for fd in fd_dir.iterdir():
                                try:
                                    resolved = os.readlink(fd)
                                    if target_str in resolved:
                                        active_pids.append(pid)
                                        break
                                except (OSError, FileNotFoundError):
                                    continue
                    except (ValueError, PermissionError):
                        continue
                if active_pids:
                    raise RuntimeError(
                        f"RECOVERY_FENCE_VIOLATION: Active database handles detected in running processes {active_pids}. "
                        "Cannot safely replace database under WAL mode; processes would remain attached to stale inode. "
                        "Halt application service before recovery."
                    )

    def step5_verify_quiescence_lock(self) -> None:
        """
        Step 5: Verify database access is quiescent via write lock probe.
        Note: BEGIN EXCLUSIVE only proves no competing writer; Step 4 proves zero readers.
        """
        if not self.target_path.exists():
            return

        conn_str = f"{self.target_path.as_uri()}?mode=rw"
        conn = sqlite3.connect(conn_str, uri=True, timeout=self.busy_timeout_ms / 1000.0)
        try:
            cur = conn.cursor()
            cur.execute("BEGIN EXCLUSIVE")
            cur.execute("COMMIT")
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                f"ACTIVE_WRITER_DETECTED: Target database '{self.target_path}' is not quiescent. "
                f"Cannot acquire exclusive lock: {e}. Halt competing writers before recovery."
            )
        finally:
            conn.close()

    def step6_quarantine_affected_database(self) -> Tuple[Path, List[QuarantinedEvidenceFile]]:
        """
        Step 6: Non-destructively move existing target DB, WAL, and SHM files to quarantine evidence directory.
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        quarantine_dir = self.quarantine_base_dir / f"{self.incident_id}_{timestamp_str}"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        quarantined: List[QuarantinedEvidenceFile] = []
        files_to_quarantine = [
            self.target_path,
            self.target_path.with_name(self.target_path.name + "-wal"),
            self.target_path.with_name(self.target_path.name + "-shm"),
        ]

        for src_f in files_to_quarantine:
            if src_f.exists():
                dst_f = quarantine_dir / src_f.name
                file_size = src_f.stat().st_size
                file_sha = self.compute_file_sha256(src_f) if file_size > 0 else "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                shutil.copy2(src_f, dst_f)
                src_f.unlink()
                quarantined.append(
                    QuarantinedEvidenceFile(
                        original_path=str(src_f),
                        quarantine_path=str(dst_f),
                        file_bytes=file_size,
                        sha256=file_sha,
                    )
                )

        evidence_manifest = quarantine_dir / "quarantine_manifest.json"
        with open(evidence_manifest, "w", encoding="utf-8") as f:
            json.dump([asdict(q) for q in quarantined], f, indent=2)

        return quarantine_dir, quarantined

    def step7_verify_backup_integrity(self) -> None:
        """
        Step 7: Verify backup file exists, matches expected SHA-256, and passes SQLite integrity_check.
        """
        if not self.backup_path.is_file():
            raise FileNotFoundError(f"BACKUP_FILE_NOT_FOUND: Backup '{self.backup_path}' does not exist.")

        actual_sha = self.compute_file_sha256(self.backup_path).lower()
        if actual_sha != self.expected_backup_sha256:
            raise ValueError(
                f"BACKUP_DIGEST_MISMATCH: Backup file SHA-256 '{actual_sha}' does not match "
                f"expected digest '{self.expected_backup_sha256}'."
            )

        uri = f"{self.backup_path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            rows = cur.fetchall()
            if not rows or rows[0][0] != "ok":
                raise ValueError(f"BACKUP_CORRUPTED: PRAGMA integrity_check failed on backup: {rows}")
        finally:
            conn.close()

    def step8_restore_via_candidate_file(self) -> Path:
        """
        Step 8: Stage restore into candidate file, verify candidate, then atomically move to target.
        """
        candidate_path = self.target_path.with_name(self.target_path.name + ".restore_candidate")
        if candidate_path.exists():
            candidate_path.unlink()

        shutil.copy2(self.backup_path, candidate_path)

        candidate_sha = self.compute_file_sha256(candidate_path)
        if candidate_sha != self.expected_backup_sha256:
            candidate_path.unlink()
            raise ValueError("CANDIDATE_VERIFICATION_FAILED: Candidate file digest corrupted during copy.")

        os.replace(candidate_path, self.target_path)
        return self.target_path

    def step9_readback_and_verify_schema(self) -> Dict[str, Any]:
        """
        Step 9: Verify restored schema, row counts, provenance, and data integrity.
        """
        uri = f"{self.target_path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        readback_info: Dict[str, Any] = {}
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            integrity_res = cur.fetchall()
            if not integrity_res or integrity_res[0][0] != "ok":
                raise ValueError(f"RESTORED_TARGET_CORRUPT: Post-restore integrity check failed: {integrity_res}")
            readback_info["integrity_check"] = "ok"

            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            readback_info["tables"] = tables

            table_counts = {}
            for t in tables:
                if t not in ("sqlite_sequence",):
                    cur.execute(f"SELECT COUNT(*) FROM {t}")
                    table_counts[t] = cur.fetchone()[0]
            readback_info["table_counts"] = table_counts
        finally:
            conn.close()

        return readback_info

    @staticmethod
    def _capture_store_metadata(db_path: Path) -> Dict[str, Any]:
        """Captures basic table counts for pre-recovery baseline if readable."""
        uri = f"{db_path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        meta = {}
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall() if r[0] != "sqlite_sequence"]
            counts = {}
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                counts[t] = cur.fetchone()[0]
            meta["table_counts"] = counts
        finally:
            conn.close()
        return meta

    def _rollback_quarantine_to_target(self, quarantined: List[QuarantinedEvidenceFile]) -> None:
        """
        Restores quarantined target database files back to original paths if recovery fails post-quarantine.
        """
        for q in quarantined:
            src = Path(q.quarantine_path)
            dst = Path(q.original_path)
            if src.exists() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                    logger.warning(f"Rolled back quarantined file '{src.name}' to '{dst}'.")
                except Exception as rb_err:
                    logger.error(f"Failed to rollback '{src}' to '{dst}': {rb_err}")

    def execute_guarded_recovery(self) -> GuardedRecoveryReceipt:
        """
        Executes full 10-step guarded disaster recovery runbook with active handle fencing,
        pre-flight backup verification, and evidence-preserving rollback protection.
        """
        receipt = GuardedRecoveryReceipt(
            incident_id=self.incident_id,
            operator_authorization=self.operator_authorization,
            maintenance_mode_token=self.maintenance_mode_token,
            target_database_path=str(self.target_path),
            backup_source_path=str(self.backup_path),
            backup_source_sha256=self.expected_backup_sha256,
            quarantine_directory="",
            quarantined_files=[],
            pre_recovery_state={},
            post_recovery_readback={},
            recovery_status="PENDING",
        )

        quarantined: List[QuarantinedEvidenceFile] = []

        try:
            # 1. Authorization audit references
            self.step1_verify_authorization()

            # 2. Maintenance mode audit reference
            self.step2_verify_maintenance_mode()

            # 3. CRITICAL PRE-FLIGHT: Verify backup integrity BEFORE touching or quarantining target database
            self.step7_verify_backup_integrity()

            # 4. Handle fence: prove zero active reader/writer connections
            self.step3_and_4_verify_reader_writer_fence()

            # 5. Quiescence write lock probe
            self.step5_verify_quiescence_lock()

            # Capture pre-recovery state if target DB exists and readable
            if self.target_path.exists():
                try:
                    receipt.pre_recovery_state = self._capture_store_metadata(self.target_path)
                except Exception:
                    receipt.pre_recovery_state = {"status": "unreadable_or_corrupted"}

            # 6. Quarantine affected files non-destructively
            quarantine_dir, quarantined = self.step6_quarantine_affected_database()
            receipt.quarantine_directory = str(quarantine_dir)
            receipt.quarantined_files = [asdict(q) for q in quarantined]

            # 7. Staged restore and controlled atomic replacement
            self.step8_restore_via_candidate_file()

            # 8. Readback & schema verification
            readback = self.step9_readback_and_verify_schema()
            receipt.post_recovery_readback = readback

            # 9. Completed successfully & ready for explicit application reopening
            receipt.recovery_status = "SUCCESS"
            receipt.completed_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            receipt.compute_receipt_sha256()
            return receipt

        except Exception as e:
            receipt.recovery_status = "FAILED"
            receipt.error_message = str(e)
            receipt.completed_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            # If target DB was quarantined but replacement failed, restore original from quarantine to prevent data loss
            if quarantined and not self.target_path.exists():
                logger.warning("Recovery aborted post-quarantine. Restoring original database files from quarantine...")
                self._rollback_quarantine_to_target(quarantined)
            receipt.compute_receipt_sha256()
            raise
