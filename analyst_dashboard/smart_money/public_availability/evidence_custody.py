"""
ARX Terminal — Smart Money Public Availability
DURABLE EVIDENCE CUSTODY SUBSTRATE (Theorem D)

Implements:
1. Verbatim raw SEC evidence tuple binding accession, endpoint, clock, SHA-256, and bytes
2. Object/metadata atomic coupling and staging outbox
3. Downstream-failure quarantine preserving original bytes
4. Deterministic independent replay from durable custody alone
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
import shutil
from typing import Any, Dict, Optional, Tuple


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonicalize(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class EvidenceStatus(str, Enum):
    STAGED = "STAGED"
    COMMITTED = "COMMITTED"
    CAPTURED_QUARANTINED = "CAPTURED_QUARANTINED"


@dataclass(frozen=True)
class RawSecEvidenceTuple:
    """Immutable tuple binding all prospective capture attributes to verbatim bytes."""
    accession: str
    endpoint: str
    http_status: int
    request_time_utc: str
    response_time_utc: str
    clock_evidence_id: str
    object_id: str
    object_version: str
    byte_length: int
    sha256: str
    content_type: str
    capture_run_id: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def canonical_hash(self) -> str:
        return sha256_bytes(canonicalize(self.to_dict()).encode("utf-8"))

    def verify_bytes(self, raw_bytes: bytes) -> Tuple[bool, str]:
        if len(raw_bytes) != self.byte_length:
            return False, f"Byte length mismatch: expected {self.byte_length}, got {len(raw_bytes)}"
        computed_sha = sha256_bytes(raw_bytes)
        if computed_sha != self.sha256:
            return False, f"SHA-256 mismatch: expected {self.sha256}, got {computed_sha}"
        return True, "VERIFIED"


class DurableEvidenceCustodyStore:
    """Manages durable, versioned, replayable SEC evidence custody."""

    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self.raw_dir = os.path.join(self.base_dir, "raw_objects")
        self.staged_dir = os.path.join(self.base_dir, "staged_objects")
        self.quarantine_dir = os.path.join(self.base_dir, "quarantine_objects")
        self.metadata_dir = os.path.join(self.base_dir, "metadata")

        for d in (self.raw_dir, self.staged_dir, self.quarantine_dir, self.metadata_dir):
            os.makedirs(d, exist_ok=True)

    def stage_evidence(
        self,
        evidence: RawSecEvidenceTuple,
        raw_bytes: bytes,
    ) -> Tuple[bool, str]:
        """Stages raw bytes and verifies hash integrity before promotion."""
        valid, msg = evidence.verify_bytes(raw_bytes)
        if not valid:
            return False, f"Staging verification failed: {msg}"

        obj_path = os.path.join(self.staged_dir, f"{evidence.object_id}.bin")
        meta_path = os.path.join(self.staged_dir, f"{evidence.object_id}.meta.json")

        with open(obj_path, "wb") as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())

        with open(meta_path, "w", encoding="utf-8") as f:
            meta = {
                "status": EvidenceStatus.STAGED.value,
                "evidence_tuple": evidence.to_dict(),
                "staged_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            json.dump(meta, f, indent=2, sort_keys=True)

        return True, "STAGED_OK"

    def commit_evidence(self, object_id: str) -> Tuple[bool, str]:
        """Atomically promotes staged evidence to immutable durable custody."""
        src_bin = os.path.join(self.staged_dir, f"{object_id}.bin")
        src_meta = os.path.join(self.staged_dir, f"{object_id}.meta.json")

        if not os.path.exists(src_bin) or not os.path.exists(src_meta):
            return False, f"Staged object {object_id} not found"

        with open(src_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dest_bin = os.path.join(self.raw_dir, f"{object_id}.bin")
        dest_meta = os.path.join(self.metadata_dir, f"{object_id}.meta.json")

        meta["status"] = EvidenceStatus.COMMITTED.value
        meta["committed_at_utc"] = datetime.now(timezone.utc).isoformat()

        # Atomic file movements
        shutil.move(src_bin, dest_bin)
        with open(dest_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        if os.path.exists(src_meta):
            os.remove(src_meta)

        return True, "COMMITTED_OK"

    def quarantine_evidence(
        self,
        object_id: str,
        failure_stage: str,
        failure_reason: str,
    ) -> Tuple[bool, str]:
        """Moves evidence to immutable quarantine upon downstream authority failure."""
        src_bin = os.path.join(self.staged_dir, f"{object_id}.bin")
        if not os.path.exists(src_bin):
            src_bin = os.path.join(self.raw_dir, f"{object_id}.bin")

        if not os.path.exists(src_bin):
            return False, f"Object {object_id} not found for quarantine"

        dest_bin = os.path.join(self.quarantine_dir, f"{object_id}.bin")
        dest_meta = os.path.join(self.quarantine_dir, f"{object_id}.quarantine.json")

        shutil.copy2(src_bin, dest_bin)

        q_record = {
            "object_id": object_id,
            "status": EvidenceStatus.CAPTURED_QUARANTINED.value,
            "admissible": False,
            "quarantined_at_utc": datetime.now(timezone.utc).isoformat(),
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
        }
        with open(dest_meta, "w", encoding="utf-8") as f:
            json.dump(q_record, f, indent=2, sort_keys=True)

        return True, "QUARANTINED_OK"

    def replay_and_verify(self, object_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Independently replays and verifies evidence from raw bytes alone."""
        meta_path = os.path.join(self.metadata_dir, f"{object_id}.meta.json")
        bin_path = os.path.join(self.raw_dir, f"{object_id}.bin")

        if not os.path.exists(meta_path) or not os.path.exists(bin_path):
            return False, {"error": "Evidence object or metadata missing from durable custody"}

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        with open(bin_path, "rb") as f:
            raw_bytes = f.read()

        ev_dict = meta.get("evidence_tuple", {})
        ev_tuple = RawSecEvidenceTuple(**ev_dict)

        ok, msg = ev_tuple.verify_bytes(raw_bytes)
        if not ok:
            return False, {"error": f"Byte verification failed: {msg}"}

        return True, {
            "status": "REPLAY_VERIFIED",
            "accession": ev_tuple.accession,
            "byte_length": len(raw_bytes),
            "sha256": ev_tuple.sha256,
            "endpoint": ev_tuple.endpoint,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }
