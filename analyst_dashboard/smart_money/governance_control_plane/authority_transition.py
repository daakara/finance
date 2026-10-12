"""
Authority Transition & Emergency Authority Subsystem
ARX Terminal — Smart Money

Manages the transition from Bootstrap Authority to Normal Governance Authority,
enforces zero authority overlap, and models restricted emergency halt powers.
"""

from dataclasses import dataclass, asdict
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


class EmergencyAuthorityException(Exception):
    """Base exception for emergency authority violations."""
    pass


class EmergencyCapabilityMintingException(EmergencyAuthorityException):
    """Raised when emergency authority attempts to create new capabilities."""
    pass


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


class AuthorityTransitionManager:
    """
    Validates preconditions and executes the formal activation of Normal Governance Authority.
    """

    def __init__(self, bootstrap_mgr: BootstrapAuthorityManager):
        self.bootstrap_mgr = bootstrap_mgr
        self.normal_authority_active = False

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
