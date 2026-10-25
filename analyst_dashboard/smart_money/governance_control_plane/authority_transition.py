"""
Authority Transition & Emergency Authority Subsystem
ARX Terminal — Smart Money

Manages the transition from Bootstrap Authority to Normal Governance Authority,
enforces atomic handoff, zero authority overlap, and models restricted emergency halt powers.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .canonicalization import sha256_canonical
from .bootstrap_authority import BootstrapAuthorityManager, BootstrapState


class AuthorityTransitionException(Exception):
    """Base exception for authority transition violations."""
    pass


class AuthorityOverlapException(AuthorityTransitionException):
    """Raised when bootstrap authority and normal authority attempt to overlap."""
    pass


class ActivationPreconditionException(AuthorityTransitionException):
    """Raised when a mandatory activation precondition is not satisfied."""
    pass


class StaleHandoffException(AuthorityTransitionException):
    """Raised when a stale or conflicting handoff ID is submitted."""
    pass


class HandoffDoubleCommitException(AuthorityTransitionException):
    """Raised when an already-committed handoff is re-committed with different parameters."""
    pass


class EmergencyAuthorityException(Exception):
    """Base exception for emergency authority violations."""
    pass


class EmergencyCapabilityMintingException(EmergencyAuthorityException):
    """Raised when emergency authority attempts to create new capabilities."""
    pass


class AuthorityHandoffState(str, Enum):
    BOOTSTRAP_ACTIVE_NORMAL_INACTIVE = "BOOTSTRAP_ACTIVE_NORMAL_INACTIVE"
    HANDOFF_PRECOMMITTED = "HANDOFF_PRECOMMITTED"
    NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED = "NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED"
    HANDOFF_RECOVERY_REQUIRED = "HANDOFF_RECOVERY_REQUIRED"


REQUIRED_ACTIVATION_PRECONDITIONS = [
    "ROOT_MODEL_RATIFIED",
    "ROOT_CREDENTIAL_BINDING",
    "BOOTSTRAP_AUTHORIZATION",
    "EXTERNAL_LEDGER",
    "EXTERNAL_WITNESS",
    "GENESIS_EVENT",
    "GENESIS_WITNESS",
    "BOOTSTRAP_EXTINGUISHMENT",
]

REQUIRED_HANDOFF_PRECOMMIT_PRECONDITIONS = [
    "ROOT_MODEL_RATIFIED",
    "ROOT_CREDENTIAL_BINDING",
    "BOOTSTRAP_AUTHORIZATION",
    "EXTERNAL_LEDGER",
    "EXTERNAL_WITNESS",
    "GENESIS_EVENT",
    "GENESIS_WITNESS",
]


@dataclass
class AuthorityHandoffReceipt:
    handoff_id: str
    governance_epoch: str
    bootstrap_authorization_id: str
    bootstrap_pre_state: str
    normal_pre_state: str
    genesis_event_hash: str
    genesis_witness_hash: str
    policy_hash: str
    root_set_hash: str
    ledger_head_hash: str
    handoff_precommit_hash: str
    bootstrap_post_state: str
    normal_post_state: str
    commit_sequence: int
    commit_hash: str
    committed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuthorityHandoffController:
    """
    Deterministic transition controller that executes atomic authority handoff.
    Guarantees:
    1. Zero overlap: Bootstrap and Normal are never concurrently active.
    2. Zero crash-deadlock: Transitions from PRECOMMITTED to COMMITTED atomically.
    3. Idempotent recovery on response loss using handoff_id.
    """

    def __init__(self, bootstrap_mgr: BootstrapAuthorityManager):
        self.bootstrap_mgr = bootstrap_mgr
        self.handoff_state = AuthorityHandoffState.BOOTSTRAP_ACTIVE_NORMAL_INACTIVE
        self.current_handoff_id: Optional[str] = None
        self.precommit_payload: Optional[Dict[str, Any]] = None
        self.precommit_hash: Optional[str] = None
        self.committed_receipt: Optional[AuthorityHandoffReceipt] = None
        self.normal_authority_active = False

    @property
    def is_normal_active(self) -> bool:
        return self.normal_authority_active

    @property
    def is_bootstrap_active(self) -> bool:
        return self.bootstrap_mgr.is_active

    def precommit_handoff(
        self,
        handoff_id: str,
        preconditions: Dict[str, bool],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 1. State check
        if self.handoff_state == AuthorityHandoffState.NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED:
            if handoff_id == self.current_handoff_id:
                return {"status": "ALREADY_COMMITTED", "receipt": self.committed_receipt.to_dict()}
            raise StaleHandoffException("Authority handoff has already been committed.")

        if self.handoff_state == AuthorityHandoffState.HANDOFF_PRECOMMITTED:
            if handoff_id == self.current_handoff_id:
                return {"status": "ALREADY_PRECOMMITTED", "precommit_hash": self.precommit_hash}
            raise StaleHandoffException(
                f"Conflicting handoff '{handoff_id}' rejected; '{self.current_handoff_id}' is already precommitted."
            )

        # 2. Bootstrap must be ACTIVE
        if not self.bootstrap_mgr.is_active:
            raise AuthorityOverlapException(
                f"Cannot precommit handoff: Bootstrap authority is not ACTIVE (current: {self.bootstrap_mgr.state.value})."
            )

        # 3. Verify all precommit prerequisites
        for req in REQUIRED_HANDOFF_PRECOMMIT_PRECONDITIONS:
            if not preconditions.get(req):
                raise ActivationPreconditionException(
                    f"Mandatory handoff precommit precondition '{req}' is not satisfied."
                )

        precommit_body = {
            "handoff_id": handoff_id,
            "governance_epoch": payload.get("governance_epoch", "epoch-gov-001"),
            "bootstrap_authorization_id": self.bootstrap_mgr.auth.bootstrap_authorization_id,
            "genesis_event_hash": payload.get("genesis_event_hash"),
            "genesis_witness_hash": payload.get("genesis_witness_hash"),
            "policy_hash": payload.get("policy_hash"),
            "root_set_hash": payload.get("root_set_hash"),
            "ledger_head_hash": payload.get("ledger_head_hash"),
        }
        self.precommit_hash = sha256_canonical(precommit_body)
        self.current_handoff_id = handoff_id
        self.precommit_payload = payload
        self.handoff_state = AuthorityHandoffState.HANDOFF_PRECOMMITTED

        return {
            "status": "HANDOFF_PRECOMMITTED",
            "handoff_id": handoff_id,
            "precommit_hash": self.precommit_hash,
        }

    def commit_handoff(self, handoff_id: str) -> AuthorityHandoffReceipt:
        # Idempotent response-loss check
        if self.handoff_state == AuthorityHandoffState.NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED:
            if handoff_id == self.current_handoff_id:
                return self.committed_receipt
            raise StaleHandoffException(
                f"Stale handoff ID '{handoff_id}' rejected; transition committed under '{self.current_handoff_id}'."
            )

        if self.handoff_state != AuthorityHandoffState.HANDOFF_PRECOMMITTED:
            raise AuthorityTransitionException(
                f"Cannot commit handoff from state '{self.handoff_state.value}': must be HANDOFF_PRECOMMITTED."
            )

        if handoff_id != self.current_handoff_id:
            raise StaleHandoffException(
                f"Handoff ID mismatch: expected '{self.current_handoff_id}', got '{handoff_id}'."
            )

        # ATOMIC PAIRED COMMIT:
        # 1. Extinguish Bootstrap
        self.bootstrap_mgr.auth.consumption_state = BootstrapState.EXTINGUISHED
        self.bootstrap_mgr.auth.reuse_count += 1

        # 2. Activate Normal Authority
        self.normal_authority_active = True
        self.handoff_state = AuthorityHandoffState.NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        receipt_body = {
            "handoff_id": handoff_id,
            "governance_epoch": self.precommit_payload.get("governance_epoch", "epoch-gov-001"),
            "bootstrap_authorization_id": self.bootstrap_mgr.auth.bootstrap_authorization_id,
            "bootstrap_pre_state": "ACTIVE",
            "normal_pre_state": "INACTIVE",
            "genesis_event_hash": self.precommit_payload.get("genesis_event_hash", ""),
            "genesis_witness_hash": self.precommit_payload.get("genesis_witness_hash", ""),
            "policy_hash": self.precommit_payload.get("policy_hash", ""),
            "root_set_hash": self.precommit_payload.get("root_set_hash", ""),
            "ledger_head_hash": self.precommit_payload.get("ledger_head_hash", ""),
            "handoff_precommit_hash": self.precommit_hash,
            "bootstrap_post_state": "EXTINGUISHED",
            "normal_post_state": "ACTIVE",
            "commit_sequence": 1,
            "committed_at": now_str,
        }
        commit_hash = sha256_canonical(receipt_body)

        receipt = AuthorityHandoffReceipt(
            handoff_id=handoff_id,
            governance_epoch=self.precommit_payload.get("governance_epoch", "epoch-gov-001"),
            bootstrap_authorization_id=self.bootstrap_mgr.auth.bootstrap_authorization_id,
            bootstrap_pre_state="ACTIVE",
            normal_pre_state="INACTIVE",
            genesis_event_hash=self.precommit_payload.get("genesis_event_hash", ""),
            genesis_witness_hash=self.precommit_payload.get("genesis_witness_hash", ""),
            policy_hash=self.precommit_payload.get("policy_hash", ""),
            root_set_hash=self.precommit_payload.get("root_set_hash", ""),
            ledger_head_hash=self.precommit_payload.get("ledger_head_hash", ""),
            handoff_precommit_hash=self.precommit_hash,
            bootstrap_post_state="EXTINGUISHED",
            normal_post_state="ACTIVE",
            commit_sequence=1,
            commit_hash=commit_hash,
            committed_at=now_str,
        )
        self.committed_receipt = receipt
        return receipt

    def recover_or_resume_handoff(self, handoff_id: str) -> AuthorityHandoffReceipt:
        """
        Handles response-loss recovery using original handoff_id.
        """
        if self.handoff_state == AuthorityHandoffState.NORMAL_ACTIVE_BOOTSTRAP_EXTINGUISHED:
            if handoff_id == self.current_handoff_id:
                return self.committed_receipt
            raise StaleHandoffException(
                f"Recovery rejected: handoff '{handoff_id}' does not match committed handoff '{self.current_handoff_id}'."
            )
        if self.handoff_state == AuthorityHandoffState.HANDOFF_PRECOMMITTED:
            if handoff_id == self.current_handoff_id:
                return self.commit_handoff(handoff_id)
            raise StaleHandoffException(
                f"Recovery rejected: handoff '{handoff_id}' does not match precommitted handoff '{self.current_handoff_id}'."
            )
        raise AuthorityTransitionException("No committed or precommitted handoff available for recovery.")


class AuthorityTransitionManager:
    """
    Backwards-compatible interface for validating preconditions and
    executing normal authority activation.
    """

    def __init__(self, bootstrap_mgr: BootstrapAuthorityManager):
        self.bootstrap_mgr = bootstrap_mgr
        self.normal_authority_active = False
        self.controller = AuthorityHandoffController(bootstrap_mgr)

    def activate_normal_authority(
        self,
        preconditions: Dict[str, bool],
        activation_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 1. Authority Overlap Check: Bootstrap must be extinguished
        if self.bootstrap_mgr.is_active:
            raise AuthorityOverlapException(
                "Authority overlap strictly forbidden: bootstrap authority is still ACTIVE."
            )
        if not self.bootstrap_mgr.is_extinguished:
            raise AuthorityOverlapException(
                f"Bootstrap authority must be EXTINGUISHED before normal activation (current: {self.bootstrap_mgr.state.value})."
            )

        # 2. Check all 8 preconditions
        for pre in REQUIRED_ACTIVATION_PRECONDITIONS:
            if not preconditions.get(pre):
                raise ActivationPreconditionException(
                    f"Mandatory activation precondition '{pre}' is not satisfied."
                )

        # 3. Compute deterministic activation hash
        activation_hash = sha256_canonical(activation_payload)
        self.normal_authority_active = True

        event = {
            "event_type": "NORMAL_GOVERNANCE_AUTHORITY_ACTIVATED",
            "governance_epoch": activation_payload.get("governance_epoch", "epoch-gov-001"),
            "genesis_event_hash": activation_payload.get("genesis_event_hash"),
            "genesis_witness_hash": activation_payload.get("genesis_witness_hash"),
            "root_set_hash": activation_payload.get("root_set_hash"),
            "policy_hash": activation_payload.get("policy_hash"),
            "ledger_head_hash": activation_payload.get("ledger_head_hash"),
            "bootstrap_authorization_id": activation_payload.get("bootstrap_authorization_id"),
            "bootstrap_consumption_hash": activation_payload.get("bootstrap_consumption_hash"),
            "bootstrap_extinguishment_hash": activation_payload.get("bootstrap_extinguishment_hash"),
            "activation_hash": activation_hash,
            "activated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return event


class EmergencyAuthorityManager:
    """
    Enforces asymmetric emergency powers:
    Emergency authority may remove capability, but may NEVER create capability.
    """

    ALLOWED_EMERGENCY_ACTIONS = [
        "HALT_NEW_AUTHORIZATIONS",
        "HALT_NEW_ADMISSIONS",
        "REVOKE_EXISTING_UNCONSUMED_AUTHORIZATIONS",
    ]

    PROHIBITED_EMERGENCY_ACTIONS = [
        "CREATE_NEW_AUTHORIZATION",
        "WAIVE_FINDINGS",
        "CHANGE_POLICY",
        "CHANGE_ROOT_QUORUM",
        "ARM_NATURAL_OBSERVER",
    ]

    def __init__(self):
        self.halts_active = set()

    def execute_emergency_action(self, action: str) -> Dict[str, Any]:
        if action in self.PROHIBITED_EMERGENCY_ACTIONS:
            raise EmergencyCapabilityMintingException(
                f"Emergency action '{action}' is strictly prohibited: emergency authority cannot create capabilities."
            )

        if action not in self.ALLOWED_EMERGENCY_ACTIONS:
            raise EmergencyAuthorityException(f"Unrecognized emergency action '{action}'.")

        self.halts_active.add(action)
        return {
            "status": "EMERGENCY_HALT_APPLIED",
            "action": action,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
