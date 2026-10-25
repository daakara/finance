"""
Bootstrap Authority Subsystem
ARX Terminal — Smart Money

Implements one-time exceptional bootstrap authority, strict allowlist/denylist boundaries,
single-use lifecycle, and permanent extinguishment.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .canonicalization import sha256_canonical


class BootstrapState(str, Enum):
    NOT_RATIFIED = "NOT_RATIFIED"
    RATIFIED_NOT_ACTIVE = "RATIFIED_NOT_ACTIVE"
    ACTIVE = "ACTIVE"
    CONSUMED_PENDING_EXTINGUISHMENT = "CONSUMED_PENDING_EXTINGUISHMENT"
    EXTINGUISHED = "EXTINGUISHED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


BOOTSTRAP_ALLOWED_ACTIONS = [
    "RATIFY_ROOT_IDENTITY_MODEL",
    "REGISTER_INITIAL_ROOT_IDENTITIES",
    "REGISTER_INITIAL_ROOT_KEYS",
    "CREATE_GOVERNANCE_LEDGER",
    "CREATE_GOVERNANCE_WITNESS",
    "RATIFY_GENESIS_POLICY",
    "PUBLISH_GENESIS_MANIFEST",
    "PUBLISH_GENESIS_LEDGER_EVENT",
    "VERIFY_GENESIS_WITNESS",
    "PRECOMMIT_NORMAL_AUTHORITY_HANDOFF",
    "EXTINGUISH_BOOTSTRAP_AUTHORITY",
]

BOOTSTRAP_PROHIBITED_ACTIONS = [
    "ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY",
    "PROVISION_THEOREM_A",
    "PROVISION_THEOREM_B",
    "PROVISION_THEOREM_C",
    "PROVISION_THEOREM_D",
    "QUALIFY_THEOREM_E",
    "ARM_NATURAL_OBSERVER",
    "ALTER_NATURAL_DENOMINATOR",
    "MODIFY_CANDIDATE_GENERATION_004",
    "SUPPRESS_FINDING",
    "DELETE_FINDING",
    "WAIVE_NON_WAIVABLE_FINDING",
    "ALTER_MODEL_OUTPUT",
    "TUNE_MODEL",
    "AUTHORIZE_PRODUCTION",
    "DELETE_LEDGER_HISTORY",
    "REWRITE_GENESIS",
]


class BootstrapException(Exception):
    """Base exception for bootstrap authority violations."""
    pass


class BootstrapOutOfScopeActionException(BootstrapException):
    """Raised when an action is not present in the bootstrap allowlist."""
    pass


class BootstrapProhibitedActionException(BootstrapException):
    """Raised when an action matches the bootstrap denylist."""
    pass


class BootstrapExtinguishedException(BootstrapException):
    """Raised when an extinguished bootstrap authority is re-invoked."""
    pass


class BootstrapReplayException(BootstrapException):
    """Raised when bootstrap authority is replayed after single use."""
    pass


class BootstrapNotActiveException(BootstrapException):
    """Raised when bootstrap authority is invoked before activation or without ratification."""
    pass


class BootstrapExpiredException(BootstrapException):
    """Raised when bootstrap authorization has expired."""
    pass


class CeremonyClaimState(str, Enum):
    UNCLAIMED = "UNCLAIMED"
    CLAIMED = "CLAIMED"
    ACTIVE = "ACTIVE"
    TERMINAL = "TERMINAL"


@dataclass
class BootstrapCeremony:
    ceremony_id: str
    bootstrap_authorization_id: str
    claim_state: CeremonyClaimState
    started_at: str
    current_step: str
    irreversible_boundary_crossed: bool = False
    terminal_state: Optional[str] = None


@dataclass
class BootstrapAuthorization:
    bootstrap_authorization_id: str
    governance_epoch: str
    ratification_reference: str
    authorized_bootstrap_roles: List[str]
    allowed_actions: List[str]
    prohibited_actions: List[str]
    issued_at: str
    not_before: str
    expires_at: str
    expected_genesis_policy_hash: str
    expected_root_model_hash: str
    expected_genesis_manifest_hash: str
    authorization_hash: str
    approval_evidence: List[Dict[str, Any]]
    purpose: str = "ESTABLISH_OPERATIONAL_GOVERNANCE_CONTROL_PLANE"
    single_use: bool = True
    extinguishment_required: bool = True
    consumption_state: BootstrapState = BootstrapState.NOT_RATIFIED
    reuse_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["consumption_state"] = self.consumption_state.value
        return d


class BootstrapAuthorityManager:
    """
    Manages the lifecycle, enforcement, and extinguishment of the
    one-time exceptional Bootstrap Authority.
    """

    MAX_CEREMONIES_PER_BOOTSTRAP_AUTHORIZATION: int = 1

    def __init__(self, auth: BootstrapAuthorization):
        self.auth = auth
        self.claimed_ceremony: Optional[BootstrapCeremony] = None

    @property
    def state(self) -> BootstrapState:
        return self.auth.consumption_state

    @property
    def is_active(self) -> bool:
        return self.auth.consumption_state == BootstrapState.ACTIVE

    @property
    def is_extinguished(self) -> bool:
        return self.auth.consumption_state == BootstrapState.EXTINGUISHED

    def claim_ceremony(self, ceremony_id: str) -> BootstrapCeremony:
        if self.claimed_ceremony is not None:
            raise BootstrapReplayException(
                f"Ceremony claim rejected: Bootstrap authorization allows at most 1 ceremony (already claimed: {self.claimed_ceremony.ceremony_id})."
            )
        if self.auth.consumption_state != BootstrapState.ACTIVE:
            raise BootstrapNotActiveException(
                f"Cannot claim ceremony from non-ACTIVE state {self.auth.consumption_state.value}."
            )
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ceremony = BootstrapCeremony(
            ceremony_id=ceremony_id,
            bootstrap_authorization_id=self.auth.bootstrap_authorization_id,
            claim_state=CeremonyClaimState.CLAIMED,
            started_at=now_str,
            current_step="GENESIS_01_PREPARATION",
            irreversible_boundary_crossed=False,
            terminal_state=None,
        )
        self.claimed_ceremony = ceremony
        return ceremony

    def ratify(self, operator_approval_evidence: List[Dict[str, Any]]) -> None:
        if self.auth.consumption_state != BootstrapState.NOT_RATIFIED:
            raise BootstrapException(
                f"Cannot ratify bootstrap authority in state {self.auth.consumption_state}."
            )
        if not operator_approval_evidence:
            raise BootstrapException("Missing operator approval evidence for bootstrap ratification.")
        self.auth.approval_evidence = operator_approval_evidence
        self.auth.consumption_state = BootstrapState.RATIFIED_NOT_ACTIVE

    def activate(self) -> None:
        if self.auth.consumption_state == BootstrapState.EXTINGUISHED:
            raise BootstrapExtinguishedException("Cannot reactivate extinguished bootstrap authority.")
        if self.auth.consumption_state != BootstrapState.RATIFIED_NOT_ACTIVE:
            raise BootstrapNotActiveException(
                f"Cannot activate bootstrap authority from state {self.auth.consumption_state}."
            )
        self.auth.consumption_state = BootstrapState.ACTIVE

    def execute_action(self, action: str, check_time: Optional[datetime] = None) -> Dict[str, Any]:
        # 1. State check
        if self.auth.consumption_state == BootstrapState.EXTINGUISHED:
            raise BootstrapExtinguishedException("Bootstrap authority has been permanently extinguished.")
        if self.auth.consumption_state in [BootstrapState.EXPIRED, BootstrapState.REVOKED]:
            raise BootstrapException(f"Bootstrap authority is invalid ({self.auth.consumption_state.value}).")
        if self.auth.consumption_state != BootstrapState.ACTIVE:
            raise BootstrapNotActiveException(f"Bootstrap authority is not ACTIVE (current: {self.auth.consumption_state.value}).")

        # 2. Expiry check
        now = check_time or datetime.now(timezone.utc)
        exp_dt = datetime.fromisoformat(self.auth.expires_at.replace("Z", "+00:00"))
        if now > exp_dt:
            self.auth.consumption_state = BootstrapState.EXPIRED
            raise BootstrapExpiredException(f"Bootstrap authorization expired at {self.auth.expires_at}.")

        # 3. Single-use and replay check
        if self.auth.reuse_count > 1:
            raise BootstrapReplayException("Bootstrap authorization replay rejected: maximum uses exceeded.")

        # 4. Denylist check
        if action in BOOTSTRAP_PROHIBITED_ACTIONS:
            raise BootstrapProhibitedActionException(
                f"Action '{action}' is strictly prohibited by bootstrap denylist."
            )

        # 5. Allowlist check
        if action not in BOOTSTRAP_ALLOWED_ACTIONS:
            raise BootstrapOutOfScopeActionException(
                f"Action '{action}' is outside the authorized bootstrap allowlist."
            )

        return {
            "status": "PERMITTED",
            "action": action,
            "bootstrap_authorization_id": self.auth.bootstrap_authorization_id,
            "executed_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def consume_and_extinguish(self) -> str:
        if self.auth.consumption_state == BootstrapState.EXTINGUISHED:
            raise BootstrapExtinguishedException("Bootstrap authority is already extinguished.")
        if self.auth.consumption_state != BootstrapState.ACTIVE:
            raise BootstrapNotActiveException("Can only extinguish from ACTIVE state.")

        self.auth.consumption_state = BootstrapState.CONSUMED_PENDING_EXTINGUISHMENT
        self.auth.reuse_count += 1

        extinguishment_manifest = {
            "bootstrap_authorization_id": self.auth.bootstrap_authorization_id,
            "governance_epoch": self.auth.governance_epoch,
            "status": "EXTINGUISHED",
            "extinguished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        extinguish_hash = sha256_canonical(extinguishment_manifest)
        self.auth.consumption_state = BootstrapState.EXTINGUISHED
        return extinguish_hash
