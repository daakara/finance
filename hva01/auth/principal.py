"""
HVA-01 Synthetic Security Principal and Token Context.
Represents verified server-side caller identity and access context.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .roles import Role


@dataclass(frozen=True)
class Principal:
    """
    Immutable synthetic identity principal for server-side policy enforcement.
    Zero-PII compliant: Reviewers must use pseudonymous IDs.
    """
    principal_id: str
    role: Role
    assigned_package: Optional[str] = None
    assigned_session_id: Optional[str] = None
    issued_at_utc: str = ""
    expires_at_utc: Optional[str] = None
    is_revoked: bool = False

    def __post_init__(self) -> None:
        if not self.principal_id or not self.principal_id.strip():
            raise ValueError("Principal ID cannot be empty.")
        if not isinstance(self.role, Role):
            raise ValueError(f"Invalid role: {self.role}")
        if not self.issued_at_utc:
            object.__setattr__(self, "issued_at_utc", datetime.now(timezone.utc).isoformat())

    @classmethod
    def create_reviewer(
        cls,
        pseudonym: str,
        assigned_package: str,
        assigned_session_id: Optional[str] = None,
        session_id: Optional[str] = None,
        expires_at_utc: Optional[str] = None,
        is_revoked: bool = False,
    ) -> Principal:
        """Create a pseudonymous Reviewer principal bound to a single package."""
        eff_session_id = assigned_session_id or session_id
        return cls(
            principal_id=pseudonym,
            role=Role.REVIEWER,
            assigned_package=assigned_package,
            assigned_session_id=eff_session_id,
            expires_at_utc=expires_at_utc,
            is_revoked=is_revoked,
        )

    @classmethod
    def create_owner(cls, owner_id: str = "RESEARCH-OWNER-01") -> Principal:
        """Create a Research Owner principal."""
        return cls(principal_id=owner_id, role=Role.RESEARCH_OWNER)

    @classmethod
    def create_allocation_admin(cls, admin_id: str = "ALLOC-ADMIN-01") -> Principal:
        """Create an Allocation Administrator principal."""
        return cls(principal_id=admin_id, role=Role.ALLOCATION_ADMINISTRATOR)

    @classmethod
    def create_custodian(cls, custodian_id: str = "CUSTODIAN-ESCROW-01") -> Principal:
        """Create a Custodian principal."""
        return cls(principal_id=custodian_id, role=Role.CUSTODIAN)

    @classmethod
    def create_witness(cls, witness_id: str = "WITNESS-INDEPENDENT-01") -> Principal:
        """Create an Independent Witness principal."""
        return cls(principal_id=witness_id, role=Role.INDEPENDENT_WITNESS)

    @classmethod
    def create_operator(cls, operator_id: str = "TECH-OPERATOR-01") -> Principal:
        """Create a Technical Operator principal."""
        return cls(principal_id=operator_id, role=Role.TECHNICAL_OPERATOR)

    def to_dict(self) -> Dict[str, Any]:
        """Convert principal to dictionary."""
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Principal:
        """Deserialize principal from dictionary."""
        return cls(
            principal_id=data["principal_id"],
            role=Role(data["role"]),
            assigned_package=data.get("assigned_package"),
            assigned_session_id=data.get("assigned_session_id"),
            issued_at_utc=data.get("issued_at_utc", ""),
            expires_at_utc=data.get("expires_at_utc"),
            is_revoked=data.get("is_revoked", False),
        )

    def generate_token(self, secret: str = "ARX_HVA01_LOCAL_SYNTHETIC_SECRET") -> str:
        """Generate a deterministic synthetic HMAC token for local authentication."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    @classmethod
    def verify_token(cls, token: str, secret: str = "ARX_HVA01_LOCAL_SYNTHETIC_SECRET") -> Principal:
        """Verify and decode a synthetic bearer token."""
        try:
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                raise ValueError("Malformed token format.")
            payload_str, sig = parts
            expected_sig = hmac.new(
                secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                raise ValueError("Invalid token signature.")
            data = json.loads(payload_str)
            return cls.from_dict(data)
        except Exception as exc:
            raise ValueError(f"Token verification failed: {exc}") from exc
