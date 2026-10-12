"""
Genesis Ceremony Subsystem
ARX Terminal — Smart Money

Executes the 14 ordered ceremony steps, failure recovery states,
and duplicate genesis prevention.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from .ledger import AuthorityLedger, SecondGenesisProhibitedException
from .bootstrap_authority import BootstrapAuthorityManager


class CeremonyFailureAction(str, Enum):
    RETRY = "RETRY"
    RESUME = "RESUME"
    ROLLBACK_NON_AUTHORITATIVE_PREPARATION = "ROLLBACK_NON_AUTHORITATIVE_PREPARATION"
    TERMINAL_HOLD = "TERMINAL_HOLD"


CEREMONY_STEPS = [
    (1, "RATIFY_ROOT_MODEL"),
    (2, "BIND_ROOT_IDENTITIES_AND_CREDENTIALS"),
    (3, "RATIFY_BOOTSTRAP_AUTHORIZATION"),
    (4, "PROVISION_AUTHORITATIVE_GOVERNANCE_LEDGER"),
    (5, "PROVISION_INDEPENDENT_WITNESS"),
    (6, "VERIFY_LEDGER_AND_WITNESS_CONFIGURATION"),
    (7, "RATIFY_GENESIS_POLICY_HASH"),
    (8, "PUBLISH_GENESIS_MANIFEST"),
    (9, "APPEND_GENESIS_LEDGER_EVENT"),
    (10, "OBTAIN_GENESIS_WITNESS_PROOF"),
    (11, "VERIFY_GENESIS_CHAIN"),
    (12, "ACTIVATE_NORMAL_GOVERNANCE_AUTHORITY"),
    (13, "CONSUME_AND_EXTINGUISH_BOOTSTRAP_AUTHORITY"),
    (14, "PRODUCE_FIRST_NORMAL_AUTHORITY_SNAPSHOT"),
]


class GenesisCeremonyException(Exception):
    """Base exception for genesis ceremony failures."""
    pass


class SkippedStepException(GenesisCeremonyException):
    """Raised when an attempt is made to execute a step out of chronological order."""
    pass


class PreconditionNotMetException(GenesisCeremonyException):
    """Raised when an external prerequisite is not met."""
    pass


class GenesisCeremonyRunner:
    """
    Executes the 14-step Genesis Ceremony. Enforces strict ordering,
    unambiguous failure state machine, and second-genesis prohibition.
    """

    def __init__(
        self,
        ledger: AuthorityLedger,
        bootstrap_mgr: BootstrapAuthorityManager,
    ):
        self.ledger = ledger
        self.bootstrap_mgr = bootstrap_mgr
        self.completed_steps: List[int] = []
        self.step_receipts: Dict[int, Any] = {}
        self.genesis_executed = False

    @property
    def current_step(self) -> int:
        return len(self.completed_steps)

    def execute_step(self, step_id: int, step_payload: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Duplicate genesis check
        if step_id == 9 and (self.genesis_executed or self.ledger.sequence >= 0):
            raise SecondGenesisProhibitedException("Duplicate genesis event is strictly rejected.")

        # 2. Strict ordering check: step_id must be exactly current_step + 1
        expected_step = len(self.completed_steps) + 1
        if step_id != expected_step:
            raise SkippedStepException(
                f"Cannot execute step {step_id}: expected step {expected_step} (step {expected_step-1} completed)."
            )

        step_name = CEREMONY_STEPS[step_id - 1][1]

        # 3. Validate specific preconditions
        if step_id == 1:
            if not step_payload.get("root_model_ratified"):
                raise PreconditionNotMetException("Root model must be ratified by operator.")
        elif step_id == 2:
            if not step_payload.get("credentials_bound"):
                raise PreconditionNotMetException("Root identity credentials must be bound.")
        elif step_id == 3:
            if not self.bootstrap_mgr.is_active:
                raise PreconditionNotMetException("Bootstrap authority must be active.")
        elif step_id == 4:
            if not step_payload.get("external_ledger_provisioned"):
                raise PreconditionNotMetException("Authoritative external ledger must be provisioned.")
        elif step_id == 5:
            if not step_payload.get("independent_witness_provisioned"):
                raise PreconditionNotMetException("Independent witness must be provisioned.")
        elif step_id == 9:
            # Append genesis event to ledger
            manifest = step_payload.get("manifest", {})
            self.ledger.create_genesis(manifest)
            self.genesis_executed = True
        elif step_id == 13:
            # Extinguish bootstrap authority
            self.bootstrap_mgr.consume_and_extinguish()

        receipt = {
            "step_id": step_id,
            "step_name": step_name,
            "status": "COMPLETED",
            "payload": step_payload,
        }
        self.completed_steps.append(step_id)
        self.step_receipts[step_id] = receipt
        return receipt

    def handle_failure(self, failure_case: str) -> CeremonyFailureAction:
        mapping = {
            "ROOT_RATIFICATION_INCOMPLETE": CeremonyFailureAction.TERMINAL_HOLD,
            "CREDENTIAL_BINDING_INCOMPLETE": CeremonyFailureAction.TERMINAL_HOLD,
            "LEDGER_CREATION_FAILED": CeremonyFailureAction.RETRY,
            "WITNESS_UNAVAILABLE": CeremonyFailureAction.TERMINAL_HOLD,
            "GENESIS_APPEND_RESPONSE_LOST": CeremonyFailureAction.RESUME,
            "LEDGER_APPEND_SUCCEEDED_WITNESS_FAILED": CeremonyFailureAction.RETRY,
            "WITNESS_SUCCEEDED_LOCAL_RESPONSE_LOST": CeremonyFailureAction.RESUME,
            "ROOT_MEMBER_REVOKED_MID_CEREMONY": CeremonyFailureAction.ROLLBACK_NON_AUTHORITATIVE_PREPARATION,
            "BOOTSTRAP_EXPIRED_MID_CEREMONY": CeremonyFailureAction.TERMINAL_HOLD,
        }
        if failure_case not in mapping:
            raise GenesisCeremonyException(f"Unknown or ambiguous failure state: {failure_case}")
        return mapping[failure_case]
