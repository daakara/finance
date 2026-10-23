"""
Session Data Model for HVA-01 Intake Runner.
Enforces strict schema validation, canonical JSON serialization, and zero-PII constraints.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class SessionState(str, Enum):
    """The 8 canonical states of the reviewer evaluation lifecycle."""
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_AUTHORIZED = "SESSION_AUTHORIZED"
    RESPONSE_DRAFT = "RESPONSE_DRAFT"
    RESPONSE_SUBMITTED = "RESPONSE_SUBMITTED"
    RESPONSE_LOCKED = "RESPONSE_LOCKED"
    CORRECTION_REQUESTED = "CORRECTION_REQUESTED"
    CORRECTION_RECORDED = "CORRECTION_RECORDED"
    SESSION_CLOSED = "SESSION_CLOSED"


SESSION_ID_PATTERN = re.compile(r"^HVA01-SESS-[A-Z0-9_-]{4,64}$")
REVIEWER_PSEUDONYM_PATTERN = re.compile(r"^(SYNTH-REV|REV)-[A-Z0-9_-]{2,32}$")
SHA256_HEX_PATTERN = re.compile(r"^[a-f0-9]{64}$")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


class ValidationError(ValueError):
    """Raised when session data violates schema or governance rules."""
    pass


@dataclass(frozen=True)
class SessionModel:
    session_id: str
    protocol_version: str
    protocol_document_sha256: str
    assigned_package: str
    package_sha256: str
    reviewer_pseudonym: str
    created_at_utc: str
    current_state: SessionState = SessionState.SESSION_CREATED
    current_sequence: int = 1
    total_cases: int = 59
    schema_version: str = "1.0.0"
    expiry_utc: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate session_id
        if not SESSION_ID_PATTERN.match(self.session_id):
            raise ValidationError(
                f"Invalid session_id: {self.session_id}. Must match {SESSION_ID_PATTERN.pattern}"
            )

        # Validate reviewer_pseudonym (Synthetic or Pseudonymized, Zero PII)
        if not REVIEWER_PSEUDONYM_PATTERN.match(self.reviewer_pseudonym):
            raise ValidationError(
                f"Invalid reviewer_pseudonym: {self.reviewer_pseudonym}. "
                f"Must match {REVIEWER_PSEUDONYM_PATTERN.pattern}"
            )
        if EMAIL_PATTERN.search(self.reviewer_pseudonym):
            raise ValidationError("Personal email address detected in reviewer_pseudonym; strict PII violation.")

        # Validate package
        if self.assigned_package not in ("PACKAGE_A", "PACKAGE_B"):
            raise ValidationError(f"assigned_package must be PACKAGE_A or PACKAGE_B, got {self.assigned_package}")

        # Validate hashes
        if not SHA256_HEX_PATTERN.match(self.protocol_document_sha256):
            raise ValidationError("protocol_document_sha256 must be a 64-character lowercase hex SHA-256")
        if not SHA256_HEX_PATTERN.match(self.package_sha256):
            raise ValidationError("package_sha256 must be a 64-character lowercase hex SHA-256")

        # Validate sequence bounds
        if not (1 <= self.current_sequence <= self.total_cases + 1):
            raise ValidationError(
                f"current_sequence {self.current_sequence} out of range [1, {self.total_cases + 1}]"
            )

        # Validate timestamps format
        try:
            datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"Invalid created_at_utc timestamp: {self.created_at_utc}") from exc

        if self.expiry_utc is not None:
            try:
                datetime.fromisoformat(self.expiry_utc.replace("Z", "+00:00"))
            except (ValueError, TypeError) as exc:
                raise ValidationError(f"Invalid expiry_utc timestamp: {self.expiry_utc}") from exc

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return a sorted-key dictionary representation suitable for canonical hashing."""
        return {
            "assigned_package": self.assigned_package,
            "created_at_utc": self.created_at_utc,
            "current_sequence": self.current_sequence,
            "current_state": self.current_state.value,
            "expiry_utc": self.expiry_utc,
            "metadata": dict(sorted(self.metadata.items())),
            "package_sha256": self.package_sha256,
            "protocol_document_sha256": self.protocol_document_sha256,
            "protocol_version": self.protocol_version,
            "reviewer_pseudonym": self.reviewer_pseudonym,
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "total_cases": self.total_cases,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionModel:
        """Create a validated SessionModel from a dictionary."""
        current_state_val = data.get("current_state", SessionState.SESSION_CREATED)
        if isinstance(current_state_val, str):
            try:
                current_state = SessionState(current_state_val)
            except ValueError as exc:
                raise ValidationError(f"Unsupported session state: {current_state_val}") from exc
        else:
            current_state = current_state_val

        return cls(
            session_id=data["session_id"],
            protocol_version=data["protocol_version"],
            protocol_document_sha256=data["protocol_document_sha256"],
            assigned_package=data["assigned_package"],
            package_sha256=data["package_sha256"],
            reviewer_pseudonym=data["reviewer_pseudonym"],
            created_at_utc=data["created_at_utc"],
            current_state=current_state,
            current_sequence=data.get("current_sequence", 1),
            total_cases=data.get("total_cases", 59),
            schema_version=data.get("schema_version", "1.0.0"),
            expiry_utc=data.get("expiry_utc"),
            metadata=data.get("metadata", {}),
        )
