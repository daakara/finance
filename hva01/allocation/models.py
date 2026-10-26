"""
HVA-01 Reviewer Allocation Models.
Provides immutable allocation records and cryptographic verification under ARX Closure Model v2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class AllocationStateError(Exception):
    """Raised when an illegal operation is attempted on an allocation record (e.g. mutating locked allocation)."""
    pass


@dataclass(frozen=True)
class AllocationRecord:
    """
    Canonical pre-session allocation record binding a pseudonymous reviewer
    to an isolated evaluation package and seed.
    """
    allocation_id: str
    protocol_version: str
    reviewer_pseudonym: str
    assigned_package: str
    assigned_package_sha256: str
    chart_directory: str
    case_id_range: Tuple[str, str]
    randomization_seed: int
    attestation_status: str = "SIGNED_AND_VERIFIED"
    created_at_utc: str = ""
    is_locked: bool = False
    locked_at_utc: Optional[str] = None
    is_revoked: bool = False
    revoked_at_utc: Optional[str] = None
    revocation_reason: Optional[str] = None
    allocation_hash: str = ""

    def __post_init__(self) -> None:
        if not self.allocation_id:
            raise ValueError("Allocation ID cannot be empty.")
        if not self.reviewer_pseudonym:
            raise ValueError("Reviewer pseudonym cannot be empty.")
        if not self.created_at_utc:
            object.__setattr__(self, "created_at_utc", datetime.now(timezone.utc).isoformat())

    @classmethod
    def create(
        cls,
        allocation_id: str,
        protocol_version: str,
        reviewer_pseudonym: str,
        assigned_package: str,
        assigned_package_sha256: str,
        chart_directory: str,
        case_id_range: Tuple[str, str],
        randomization_seed: int,
        attestation_status: str = "SIGNED_AND_VERIFIED",
    ) -> AllocationRecord:
        """Create a new allocation record with calculated hash."""
        now_utc = datetime.now(timezone.utc).isoformat()
        temp_record = cls(
            allocation_id=allocation_id,
            protocol_version=protocol_version,
            reviewer_pseudonym=reviewer_pseudonym,
            assigned_package=assigned_package,
            assigned_package_sha256=assigned_package_sha256,
            chart_directory=chart_directory,
            case_id_range=case_id_range,
            randomization_seed=randomization_seed,
            attestation_status=attestation_status,
            created_at_utc=now_utc,
            is_locked=False,
            locked_at_utc=None,
            is_revoked=False,
            revoked_at_utc=None,
            revocation_reason=None,
            allocation_hash="",
        )
        calculated_hash = temp_record.compute_hash()
        return cls(
            allocation_id=allocation_id,
            protocol_version=protocol_version,
            reviewer_pseudonym=reviewer_pseudonym,
            assigned_package=assigned_package,
            assigned_package_sha256=assigned_package_sha256,
            chart_directory=chart_directory,
            case_id_range=case_id_range,
            randomization_seed=randomization_seed,
            attestation_status=attestation_status,
            created_at_utc=now_utc,
            is_locked=False,
            locked_at_utc=None,
            is_revoked=False,
            revoked_at_utc=None,
            revocation_reason=None,
            allocation_hash=calculated_hash,
        )

    def compute_hash(self) -> str:
        """Compute canonical RFC 8785 SHA-256 digest of allocation fields."""
        data = {
            "allocation_id": self.allocation_id,
            "protocol_version": self.protocol_version,
            "reviewer_pseudonym": self.reviewer_pseudonym,
            "assigned_package": self.assigned_package,
            "assigned_package_sha256": self.assigned_package_sha256,
            "chart_directory": self.chart_directory,
            "case_id_range": list(self.case_id_range),
            "randomization_seed": self.randomization_seed,
            "attestation_status": self.attestation_status,
            "created_at_utc": self.created_at_utc,
            "is_locked": self.is_locked,
            "locked_at_utc": self.locked_at_utc,
            "is_revoked": self.is_revoked,
            "revoked_at_utc": self.revoked_at_utc,
            "revocation_reason": self.revocation_reason,
        }
        canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Verify record integrity against its stored allocation_hash."""
        return self.allocation_hash == self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["case_id_range"] = list(self.case_id_range)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AllocationRecord:
        case_range = tuple(data["case_id_range"])
        return cls(
            allocation_id=data["allocation_id"],
            protocol_version=data["protocol_version"],
            reviewer_pseudonym=data["reviewer_pseudonym"],
            assigned_package=data["assigned_package"],
            assigned_package_sha256=data["assigned_package_sha256"],
            chart_directory=data["chart_directory"],
            case_id_range=case_range,  # type: ignore
            randomization_seed=data["randomization_seed"],
            attestation_status=data.get("attestation_status", "SIGNED_AND_VERIFIED"),
            created_at_utc=data.get("created_at_utc", ""),
            is_locked=data.get("is_locked", False),
            locked_at_utc=data.get("locked_at_utc"),
            is_revoked=data.get("is_revoked", False),
            revoked_at_utc=data.get("revoked_at_utc"),
            revocation_reason=data.get("revocation_reason"),
            allocation_hash=data.get("allocation_hash", ""),
        )

    def with_lock(self, locked_at_utc: Optional[str] = None) -> AllocationRecord:
        """Return a new copy marked as locked."""
        now_utc = locked_at_utc or datetime.now(timezone.utc).isoformat()
        new_record = AllocationRecord(
            allocation_id=self.allocation_id,
            protocol_version=self.protocol_version,
            reviewer_pseudonym=self.reviewer_pseudonym,
            assigned_package=self.assigned_package,
            assigned_package_sha256=self.assigned_package_sha256,
            chart_directory=self.chart_directory,
            case_id_range=self.case_id_range,
            randomization_seed=self.randomization_seed,
            attestation_status=self.attestation_status,
            created_at_utc=self.created_at_utc,
            is_locked=True,
            locked_at_utc=now_utc,
            is_revoked=self.is_revoked,
            revoked_at_utc=self.revoked_at_utc,
            revocation_reason=self.revocation_reason,
            allocation_hash="",
        )
        return AllocationRecord(
            allocation_id=self.allocation_id,
            protocol_version=self.protocol_version,
            reviewer_pseudonym=self.reviewer_pseudonym,
            assigned_package=self.assigned_package,
            assigned_package_sha256=self.assigned_package_sha256,
            chart_directory=self.chart_directory,
            case_id_range=self.case_id_range,
            randomization_seed=self.randomization_seed,
            attestation_status=self.attestation_status,
            created_at_utc=self.created_at_utc,
            is_locked=True,
            locked_at_utc=now_utc,
            is_revoked=self.is_revoked,
            revoked_at_utc=self.revoked_at_utc,
            revocation_reason=self.revocation_reason,
            allocation_hash=new_record.compute_hash(),
        )

    def with_revocation(self, reason: str, revoked_at_utc: Optional[str] = None) -> AllocationRecord:
        """Return a new copy marked as revoked."""
        now_utc = revoked_at_utc or datetime.now(timezone.utc).isoformat()
        new_record = AllocationRecord(
            allocation_id=self.allocation_id,
            protocol_version=self.protocol_version,
            reviewer_pseudonym=self.reviewer_pseudonym,
            assigned_package=self.assigned_package,
            assigned_package_sha256=self.assigned_package_sha256,
            chart_directory=self.chart_directory,
            case_id_range=self.case_id_range,
            randomization_seed=self.randomization_seed,
            attestation_status="REVOKED",
            created_at_utc=self.created_at_utc,
            is_locked=self.is_locked,
            locked_at_utc=self.locked_at_utc,
            is_revoked=True,
            revoked_at_utc=now_utc,
            revocation_reason=reason,
            allocation_hash="",
        )
        return AllocationRecord(
            allocation_id=self.allocation_id,
            protocol_version=self.protocol_version,
            reviewer_pseudonym=self.reviewer_pseudonym,
            assigned_package=self.assigned_package,
            assigned_package_sha256=self.assigned_package_sha256,
            chart_directory=self.chart_directory,
            case_id_range=self.case_id_range,
            randomization_seed=self.randomization_seed,
            attestation_status="REVOKED",
            created_at_utc=self.created_at_utc,
            is_locked=self.is_locked,
            locked_at_utc=self.locked_at_utc,
            is_revoked=True,
            revoked_at_utc=now_utc,
            revocation_reason=reason,
            allocation_hash=new_record.compute_hash(),
        )
