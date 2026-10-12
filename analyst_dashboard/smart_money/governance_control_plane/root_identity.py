"""
Root Identity and Quorum Subsystem
ARX Terminal — Smart Money

Implements the non-collapsible identity status hierarchy, 3-of-4 root quorum,
prohibition on automatic quorum downgrade, and quorum recovery lineage.
"""

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

from .canonicalization import sha256_canonical


def validate_git_oid(oid: str) -> str:
    """
    Validates that a Git object identifier is an unabbreviated 40-character hexadecimal string.
    Truncation, short display SHAs, and prefix representations are strictly prohibited in canonical contexts.
    """
    if not isinstance(oid, str):
        raise ValueError(f"Git OID must be str, got {type(oid).__name__}")
    clean = oid.strip().lower()
    if not re.match(r"^[0-9a-f]{40}$", clean):
        raise ValueError(
            f"CANONICAL_GIT_COMMIT_IDENTITY_TRUNCATED == 0: Git OID must be full 40-character hexadecimal string, got '{clean}'."
        )
    return clean


@dataclass(frozen=True)
class DisplayShortSha:
    """
    Explicitly non-authoritative short SHA for UI and logging display only.
    Strictly forbidden from being used as a canonical authority identifier.
    """
    display_value: str
    is_canonical_identity: bool = False

    def __post_init__(self):
        if self.is_canonical_identity:
            raise ValueError(
                "DISPLAY_SHA_USED_AS_AUTHORITY_IDENTITY == 0: Display short SHA cannot be used as canonical identity."
            )


class IdentityStatus(str, Enum):
    ROLE_TEMPLATE = "ROLE_TEMPLATE"
    NOMINATED = "NOMINATED"
    OPERATOR_RATIFIED = "OPERATOR_RATIFIED"
    CREDENTIAL_BOUND = "CREDENTIAL_BOUND"
    OPERATIONAL = "OPERATIONAL"
    REVOKED = "REVOKED"


class RootIdentityException(Exception):
    """Base exception for root identity and quorum errors."""
    pass


class RevokedIdentityException(RootIdentityException):
    """Raised when an action is signed by a revoked identity."""
    pass


class QuorumDowngradeException(RootIdentityException):
    """Raised when an attempt is made to automatically downgrade root quorum."""
    pass


class InvalidIdentityTransitionException(RootIdentityException):
    """Raised when an illegal identity status transition is requested."""
    pass


@dataclass
class RootIdentity:
    identity_id: str
    role_id: str
    status: IdentityStatus
    public_identifier: Optional[str] = None
    fingerprint: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class RootQuorumManager:
    """
    Manages root identities, 3-of-4 quorum enforcement,
    and lineage-preserving quorum recovery.
    """

    def __init__(self, threshold: int = 3, total_population: int = 4):
        self.threshold = threshold
        self.total_population = total_population
        self.identities: Dict[str, RootIdentity] = {}

    def register_identity(self, identity: RootIdentity) -> None:
        self.identities[identity.identity_id] = identity

    def transition_status(self, identity_id: str, new_status: IdentityStatus) -> None:
        if identity_id not in self.identities:
            raise RootIdentityException(f"Identity {identity_id} not registered.")

        current = self.identities[identity_id].status

        # Enforce non-collapsible hierarchy:
        # ROLE_TEMPLATE -> NOMINATED -> OPERATOR_RATIFIED -> CREDENTIAL_BOUND -> OPERATIONAL -> REVOKED
        valid_transitions = {
            IdentityStatus.ROLE_TEMPLATE: [IdentityStatus.NOMINATED, IdentityStatus.REVOKED],
            IdentityStatus.NOMINATED: [IdentityStatus.OPERATOR_RATIFIED, IdentityStatus.REVOKED],
            IdentityStatus.OPERATOR_RATIFIED: [IdentityStatus.CREDENTIAL_BOUND, IdentityStatus.REVOKED],
            IdentityStatus.CREDENTIAL_BOUND: [IdentityStatus.OPERATIONAL, IdentityStatus.REVOKED],
            IdentityStatus.OPERATIONAL: [IdentityStatus.REVOKED],
            IdentityStatus.REVOKED: [],
        }

        if new_status not in valid_transitions.get(current, []):
            raise InvalidIdentityTransitionException(
                f"Illegal identity transition from {current.value} to {new_status.value}."
            )

        self.identities[identity_id].status = new_status

    def verify_quorum(self, signing_identities: List[str]) -> bool:
        operational_signatures = 0
        for i_id in signing_identities:
            if i_id not in self.identities:
                continue
            identity = self.identities[i_id]
            if identity.status == IdentityStatus.REVOKED:
                raise RevokedIdentityException(f"Signing identity {i_id} is REVOKED.")
            if identity.status == IdentityStatus.OPERATIONAL:
                operational_signatures += 1

        if operational_signatures < self.threshold:
            return False
        return True

    def attempt_quorum_downgrade(self, proposed_threshold: int) -> None:
        if proposed_threshold < self.threshold:
            raise QuorumDowngradeException(
                f"Automatic quorum downgrade from {self.threshold} to {proposed_threshold} is strictly rejected."
            )

    def compute_root_set_hash(self) -> str:
        identities_list = [i.to_dict() for i in sorted(self.identities.values(), key=lambda x: x.identity_id)]
        return sha256_canonical({"identities": identities_list, "threshold": self.threshold})

    def execute_quorum_recovery(
        self,
        compromised_identity_id: str,
        replacement_identity: RootIdentity,
        authorizing_signers: List[str],
    ) -> Dict[str, Any]:
        # 1. Immediately revoke compromised member
        if compromised_identity_id in self.identities:
            self.identities[compromised_identity_id].status = IdentityStatus.REVOKED

        # 2. Verify remaining signers meet surviving quorum
        valid_survivors = [
            s for s in authorizing_signers
            if s != compromised_identity_id
            and self.identities.get(s)
            and self.identities[s].status != IdentityStatus.REVOKED
        ]
        if len(valid_survivors) < (self.threshold - 1):
            raise RootIdentityException("Insufficient surviving members to authorize quorum recovery.")

        # 3. Register replacement
        self.register_identity(replacement_identity)

        recovery_record = {
            "revoked_identity": compromised_identity_id,
            "replacement_identity": replacement_identity.to_dict(),
            "authorizing_signers": valid_survivors,
            "new_root_set_hash": self.compute_root_set_hash(),
            "second_genesis": False,
        }
        return recovery_record
