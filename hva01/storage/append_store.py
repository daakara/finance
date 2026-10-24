"""
Transactional Append-Only Storage Engine for HVA-01.
Implements write-ahead staging, atomic append, fsync durability, crash recovery,
and cross-platform non-blocking file locking.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..engine.fsm import AuditEvent
from ..engine.hasher import CanonicalHasher, HashChainVerificationError
from ..models.response import RatingsPayload, ResponseRecord, ResponseValidationError
from ..models.session import SessionModel, SessionState, ValidationError


class StorageCorruptionError(ValueError):
    """Raised when on-disk data fails verification or exhibits corruption."""
    pass


class ConflictingDuplicateSubmissionError(ValueError):
    """Raised when an idempotency conflict occurs (different payload for existing sequence/case)."""
    pass


class FileLockError(IOError):
    """Raised when file lock cannot be acquired within the designated timeout."""
    pass


class FileLock:
    """
    Cross-platform file lock using msvcrt (Windows) and fcntl (POSIX).
    Guarantees single-writer exclusion across multiple processes or threads.
    """

    def __init__(
        self,
        lock_file_path: Union[str, Path],
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.lock_file_path = str(lock_file_path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        start_time = time.time()
        while True:
            try:
                self._fd = os.open(self.lock_file_path, os.O_CREAT | os.O_RDWR)
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except (BlockingIOError, PermissionError, OSError) as exc:
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except Exception:
                        pass
                    self._fd = None

                if time.time() - start_time >= self.timeout:
                    raise FileLockError(
                        f"Could not acquire lock on {self.lock_file_path} within {self.timeout:.2f}s"
                    ) from exc
                time.sleep(self.poll_interval)

    def release(self) -> None:
        if self._fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


@dataclass(frozen=True)
class CorrectionRecord:
    petition_id: str
    session_id: str
    target_case_id: str
    target_sequence_index: int
    target_record_hash: str
    revised_ratings: RatingsPayload
    justification: str
    timestamp_utc: str
    petition_hash: str

    def __post_init__(self) -> None:
        if not self.justification or len(self.justification.strip()) < 10:
            raise ResponseValidationError("Correction justification must be at least 10 characters.")

    @classmethod
    def create(
        cls,
        petition_id: str,
        session_id: str,
        target_case_id: str,
        target_sequence_index: int,
        target_record_hash: str,
        revised_ratings: RatingsPayload,
        justification: str,
        timestamp_utc: str,
    ) -> CorrectionRecord:
        ratings_json = CanonicalHasher.canonical_json(revised_ratings.to_canonical_dict())
        payload = (
            f"CORRECTION:{petition_id}:{session_id}:{target_case_id}:{target_sequence_index}:"
            f"{target_record_hash}:{justification.strip()}:{ratings_json}"
        ).encode("utf-8")
        petition_hash = hashlib.sha256(payload).hexdigest()
        return cls(
            petition_id=petition_id,
            session_id=session_id,
            target_case_id=target_case_id,
            target_sequence_index=target_sequence_index,
            target_record_hash=target_record_hash,
            revised_ratings=revised_ratings,
            justification=justification.strip(),
            timestamp_utc=timestamp_utc,
            petition_hash=petition_hash,
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "justification": self.justification,
            "petition_hash": self.petition_hash,
            "petition_id": self.petition_id,
            "revised_ratings": self.revised_ratings.to_canonical_dict(),
            "session_id": self.session_id,
            "target_case_id": self.target_case_id,
            "target_record_hash": self.target_record_hash,
            "target_sequence_index": self.target_sequence_index,
            "timestamp_utc": self.timestamp_utc,
        }

    def to_canonical_json(self) -> str:
        return CanonicalHasher.canonical_json(self.to_canonical_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CorrectionRecord:
        ratings_dict = d["revised_ratings"]
        revised_ratings = (
            ratings_dict
            if isinstance(ratings_dict, RatingsPayload)
            else RatingsPayload.from_dict(ratings_dict)
        )
        return cls(
            petition_id=d["petition_id"],
            session_id=d["session_id"],
            target_case_id=d["target_case_id"],
            target_sequence_index=d["target_sequence_index"],
            target_record_hash=d["target_record_hash"],
            revised_ratings=revised_ratings,
            justification=d["justification"],
            timestamp_utc=d["timestamp_utc"],
            petition_hash=d["petition_hash"],
        )


class AppendOnlyStore:
    """
    Append-only transactional storage engine managing session records, audit events,
    and correction exceptions on disk.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self, session_id: str) -> Path:
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _lock_path(self, session_id: str) -> Path:
        return self.get_session_dir(session_id) / ".session.lock"

    def _session_file(self, session_id: str) -> Path:
        return self.get_session_dir(session_id) / "session.json"

    def _master_file(self, session_id: str) -> Path:
        return self.get_session_dir(session_id) / "master.jsonl"

    def _audit_file(self, session_id: str) -> Path:
        return self.get_session_dir(session_id) / "audit.jsonl"

    def _exceptions_file(self, session_id: str) -> Path:
        return self.get_session_dir(session_id) / "exceptions.jsonl"

    def init_session(self, session: SessionModel) -> Tuple[Path, str]:
        """
        Initialize session directory, write session.json, and compute genesis hash H_0.
        """
        session_dir = self.get_session_dir(session.session_id)
        lock_path = self._lock_path(session.session_id)

        with FileLock(lock_path):
            session_file = self._session_file(session.session_id)
            if session_file.exists():
                raise ValidationError(f"Session {session.session_id} already exists.")

            # Compute genesis hash
            genesis_hash = CanonicalHasher.compute_genesis_hash(
                protocol_document_sha256=session.protocol_document_sha256,
                package_sha256=session.package_sha256,
                reviewer_pseudonym=session.reviewer_pseudonym,
                session_id=session.session_id,
            )

            # Enrich metadata with genesis_hash
            session_data = session.to_canonical_dict()
            session_data["genesis_hash"] = genesis_hash

            # Atomic write of session.json via tmp file
            tmp_file = session_dir / "session.json.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(CanonicalHasher.canonical_json(session_data))
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_file, session_file)

            # Ensure master, audit, and exceptions files exist
            master_file = self._master_file(session.session_id)
            if not master_file.exists():
                master_file.touch()

            audit_file = self._audit_file(session.session_id)
            if not audit_file.exists():
                audit_file.touch()

            exceptions_file = self._exceptions_file(session.session_id)
            if not exceptions_file.exists():
                exceptions_file.touch()

            return session_dir, genesis_hash

    def save_session(self, session: SessionModel) -> None:
        """
        Persist updated SessionModel to session.json atomically.
        """
        session_dir = self.get_session_dir(session.session_id)
        lock_path = self._lock_path(session.session_id)

        with FileLock(lock_path):
            session_file = self._session_file(session.session_id)
            genesis_hash = None
            if session_file.exists():
                with open(session_file, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    genesis_hash = old_data.get("genesis_hash")

            session_data = session.to_canonical_dict()
            if genesis_hash:
                session_data["genesis_hash"] = genesis_hash

            tmp_file = session_dir / f"session_{session.session_id}.json.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(CanonicalHasher.canonical_json(session_data))
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_file, session_file)

    def load_session(self, session_id: str) -> SessionModel:
        """
        Load SessionModel from disk.
        """
        session_file = self._session_file(session_id)
        if not session_file.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionModel.from_dict(data)

    def get_genesis_hash(self, session_id: str) -> str:
        session = self.load_session(session_id)
        return CanonicalHasher.compute_genesis_hash(
            protocol_document_sha256=session.protocol_document_sha256,
            package_sha256=session.package_sha256,
            reviewer_pseudonym=session.reviewer_pseudonym,
            session_id=session.session_id,
        )

    def cleanup_uncommitted_staging(self, session_id: str) -> int:
        """
        Scan session directory and remove any uncommitted .tmp write files left by an interrupted process.
        Returns the number of discarded staging files.
        """
        session_dir = self.get_session_dir(session_id)
        tmp_patterns = [str(session_dir / "*.tmp")]
        cleaned = 0
        for pattern in tmp_patterns:
            for p in glob.glob(pattern):
                try:
                    os.remove(p)
                    cleaned += 1
                except OSError:
                    pass
        return cleaned

    def load_master_records(
        self, session_id: str, verify: bool = True
    ) -> Tuple[List[ResponseRecord], str]:
        """
        Load all committed ResponseRecords from master.jsonl.
        Cleans up any uncommitted staging files and verifies cryptographic hash chain.
        Returns (records, latest_hash).
        """
        self.cleanup_uncommitted_staging(session_id)
        master_file = self._master_file(session_id)
        if not master_file.exists():
            genesis_hash = self.get_genesis_hash(session_id)
            return [], genesis_hash

        records: List[ResponseRecord] = []
        with open(master_file, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record_dict = json.loads(line)
                    record = ResponseRecord.from_dict(record_dict)
                    records.append(record)
                except Exception as exc:
                    raise StorageCorruptionError(
                        f"Unparseable master log record at line {line_no} in {master_file}: {exc}"
                    ) from exc

        genesis_hash = self.get_genesis_hash(session_id)
        if verify:
            valid, err = CanonicalHasher.verify_chain(records, genesis_hash)
            if not valid:
                raise StorageCorruptionError(f"Master log verification failed for session {session_id}: {err}")

        latest_hash = records[-1].record_hash if records else genesis_hash
        return records, latest_hash

    def append_response(self, session_id: str, record: ResponseRecord) -> ResponseRecord:
        """
        Append a ResponseRecord to master.jsonl using write-ahead staging, fsync durability,
        and strict idempotency/continuity validation under file lock.
        """
        session_dir = self.get_session_dir(session_id)
        lock_path = self._lock_path(session_id)

        with FileLock(lock_path):
            # Step 1: Discard stale staging files
            self.cleanup_uncommitted_staging(session_id)

            # Step 2: Read current records & verify chain
            records, latest_hash = self.load_master_records(session_id, verify=True)

            # Step 3: Check idempotency & duplicates
            for existing in records:
                if (
                    existing.sequence_index == record.sequence_index
                    or existing.case_id == record.case_id
                    or existing.idempotency_key == record.idempotency_key
                ):
                    # Check exact match
                    if existing.to_canonical_dict() == record.to_canonical_dict():
                        # Idempotent replay: return existing locked record
                        return existing
                    else:
                        raise ConflictingDuplicateSubmissionError(
                            f"Conflicting submission for sequence {record.sequence_index} / case {record.case_id}. "
                            f"Case is already permanently locked with a differing payload."
                        )

            # Step 4: Verify sequence continuity
            expected_sequence = len(records) + 1
            if record.sequence_index != expected_sequence:
                raise ValueError(
                    f"Sequence discontinuity: expected sequence {expected_sequence}, "
                    f"got {record.sequence_index}"
                )

            # Step 5: Verify previous hash linkage
            if record.previous_hash != latest_hash:
                raise ValueError(
                    f"Hash link mismatch: record previous_hash {record.previous_hash} "
                    f"does not match latest chain hash {latest_hash}"
                )

            # Step 6: Verify record hash computation
            ratings_json = CanonicalHasher.canonical_json(record.ratings.to_canonical_dict())
            computed_hash = CanonicalHasher.compute_record_hash(
                previous_hash=record.previous_hash,
                sequence_index=record.sequence_index,
                case_id=record.case_id,
                timestamp_utc=record.timestamp_utc,
                ratings_canonical_json=ratings_json,
            )
            if record.record_hash != computed_hash:
                raise ValueError(
                    f"Cryptographic hash mismatch: record_hash {record.record_hash} "
                    f"!= computed {computed_hash}"
                )

            # Step 7: Write-Ahead Staging Buffer
            tmp_file = session_dir / f"write_{record.sequence_index}_{record.case_id}.tmp"
            record_canonical_line = record.to_canonical_json() + "\n"

            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(record_canonical_line)
                f.flush()
                os.fsync(f.fileno())

            # Step 8: Atomic Append to master.jsonl
            master_file = self._master_file(session_id)
            with open(master_file, "a", encoding="utf-8") as f:
                f.write(record_canonical_line)
                f.flush()
                os.fsync(f.fileno())

            # Step 9: Remove temporary staging buffer
            try:
                os.remove(tmp_file)
            except OSError:
                pass

            return record

    def append_audit_event(self, session_id: str, event: AuditEvent) -> None:
        """
        Append an AuditEvent to audit.jsonl under lock with fsync.
        """
        lock_path = self._lock_path(session_id)
        audit_file = self._audit_file(session_id)

        with FileLock(lock_path):
            event_line = CanonicalHasher.canonical_json(event.to_dict()) + "\n"
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(event_line)
                f.flush()
                os.fsync(f.fileno())

    def load_audit_events(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load all audit events from audit.jsonl.
        """
        audit_file = self._audit_file(session_id)
        if not audit_file.exists():
            return []

        events: List[Dict[str, Any]] = []
        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    events.append(json.loads(stripped))
        return events

    def append_correction(self, session_id: str, correction: CorrectionRecord) -> CorrectionRecord:
        """
        Append a CorrectionRecord to exceptions.jsonl.
        Validates target case exists and matches target_record_hash in master.jsonl.
        Master log is NEVER modified.
        """
        lock_path = self._lock_path(session_id)
        exceptions_file = self._exceptions_file(session_id)

        with FileLock(lock_path):
            # Validate target record exists in master
            records, _ = self.load_master_records(session_id, verify=True)
            matched = False
            for rec in records:
                if (
                    rec.sequence_index == correction.target_sequence_index
                    and rec.case_id == correction.target_case_id
                ):
                    if rec.record_hash != correction.target_record_hash:
                        raise ValueError(
                            f"Target record hash mismatch: expected {rec.record_hash}, "
                            f"got {correction.target_record_hash}"
                        )
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Target record for correction not found in master: sequence {correction.target_sequence_index}, "
                    f"case {correction.target_case_id}"
                )

            # Append to exceptions.jsonl
            correction_line = correction.to_canonical_json() + "\n"
            with open(exceptions_file, "a", encoding="utf-8") as f:
                f.write(correction_line)
                f.flush()
                os.fsync(f.fileno())

            return correction

    def load_corrections(self, session_id: str) -> List[CorrectionRecord]:
        """
        Load all correction petitions from exceptions.jsonl.
        """
        exceptions_file = self._exceptions_file(session_id)
        if not exceptions_file.exists():
            return []

        corrections: List[CorrectionRecord] = []
        with open(exceptions_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    corrections.append(CorrectionRecord.from_dict(json.loads(stripped)))
        return corrections

    def get_crash_recovery_status(self, session_id: str) -> Dict[str, Any]:
        """
        Inspect session directory, report committed cases, and check for uncommitted staging files.
        """
        session_dir = self.get_session_dir(session_id)
        tmp_files = glob.glob(str(session_dir / "*.tmp"))
        records, latest_hash = self.load_master_records(session_id, verify=True)
        session = self.load_session(session_id)

        return {
            "session_id": session_id,
            "current_state": session.current_state.value,
            "committed_cases_count": len(records),
            "next_expected_sequence": len(records) + 1,
            "latest_record_hash": latest_hash,
            "uncommitted_staging_files_found": len(tmp_files),
            "is_sealed": session.current_state == SessionState.SESSION_CLOSED,
        }
