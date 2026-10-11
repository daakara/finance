"""
Finalization State Machine Architecture (Sections 7, 8).
Enforces the explicit transition contract among states A0..A11, validates allowable edges,
and separates authority eligibility from authority effectiveness.
"""

from typing import Set, Tuple, Optional, Final
from datetime import datetime

from analyst_dashboard.smart_money.anchor_finalization.models import (
    FinalizationState,
    AuthorityEligibility,
    AuthorityEffectiveStatus,
    InvalidTransitionError,
    AuthorityTimingRecord,
)

# ---------------------------------------------------------------------------
# Permitted Transition Graph (Section 7)
# ---------------------------------------------------------------------------
VALID_TRANSITIONS: Final[Set[Tuple[FinalizationState, FinalizationState]]] = {
    # Initial trigger
    (FinalizationState.A0_ANCHOR_NOT_REQUIRED, FinalizationState.A1_ANCHOR_REQUIRED),
    # Sealing payload
    (FinalizationState.A1_ANCHOR_REQUIRED, FinalizationState.A2_ANCHOR_PAYLOAD_SEALED),
    # Submission pending
    (FinalizationState.A2_ANCHOR_PAYLOAD_SEALED, FinalizationState.A3_ANCHOR_SUBMISSION_PENDING),
    # From A3:
    (FinalizationState.A3_ANCHOR_SUBMISSION_PENDING, FinalizationState.A6_ANCHOR_CONFIRMED),
    (FinalizationState.A3_ANCHOR_SUBMISSION_PENDING, FinalizationState.A4_ANCHOR_RETRY_WAIT),
    (FinalizationState.A3_ANCHOR_SUBMISSION_PENDING, FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE),
    (FinalizationState.A3_ANCHOR_SUBMISSION_PENDING, FinalizationState.A8_ANCHOR_CONFLICT),
    (FinalizationState.A3_ANCHOR_SUBMISSION_PENDING, FinalizationState.A7_ANCHOR_PENDING_EXPIRED),
    # From A4:
    (FinalizationState.A4_ANCHOR_RETRY_WAIT, FinalizationState.A3_ANCHOR_SUBMISSION_PENDING),
    (FinalizationState.A4_ANCHOR_RETRY_WAIT, FinalizationState.A7_ANCHOR_PENDING_EXPIRED),
    # From A5 (Ambiguous commit lookup):
    (FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE, FinalizationState.A6_ANCHOR_CONFIRMED),
    (FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE, FinalizationState.A4_ANCHOR_RETRY_WAIT),
    (FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE, FinalizationState.A3_ANCHOR_SUBMISSION_PENDING),
    (FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE, FinalizationState.A8_ANCHOR_CONFLICT),
    (FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE, FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE),
    (FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE, FinalizationState.A7_ANCHOR_PENDING_EXPIRED),
    # From A6:
    (FinalizationState.A6_ANCHOR_CONFIRMED, FinalizationState.A10_AUTHORITY_EFFECTIVE),
    (FinalizationState.A6_ANCHOR_CONFIRMED, FinalizationState.A3_ANCHOR_SUBMISSION_PENDING),
    (FinalizationState.A6_ANCHOR_CONFIRMED, FinalizationState.A6_ANCHOR_CONFIRMED),
    # From A2:
    (FinalizationState.A2_ANCHOR_PAYLOAD_SEALED, FinalizationState.A7_ANCHOR_PENDING_EXPIRED),
    # From A7:
    (FinalizationState.A7_ANCHOR_PENDING_EXPIRED, FinalizationState.A9_GOVERNANCE_RECOVERY_REVIEW),
    # From A8:
    (FinalizationState.A8_ANCHOR_CONFLICT, FinalizationState.A11_TERMINAL_HOLD),
    # From A9:
    (FinalizationState.A9_GOVERNANCE_RECOVERY_REVIEW, FinalizationState.A10_AUTHORITY_EFFECTIVE),
    (FinalizationState.A9_GOVERNANCE_RECOVERY_REVIEW, FinalizationState.A11_TERMINAL_HOLD),
}


class FinalizationStateMachine:
    """
    State machine controlling external anchor finalization and authority progression.
    """

    def __init__(
        self,
        initial_state: FinalizationState = FinalizationState.A1_ANCHOR_REQUIRED,
        initial_eligibility: AuthorityEligibility = AuthorityEligibility.PASS,
        initial_effective_status: AuthorityEffectiveStatus = AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
        evidence_observed_at: Optional[str] = None,
        eligibility_established_at: Optional[str] = None,
    ):
        self.state = initial_state
        self.eligibility = initial_eligibility
        self.effective_status = initial_effective_status
        self.evidence_observed_at = evidence_observed_at
        self.eligibility_established_at = eligibility_established_at
        self.authority_became_effective_at: Optional[str] = None

    def validate_transition(
        self,
        to_state: FinalizationState,
        quorum_satisfied: bool = False,
        recovery_authorized: bool = False,
    ) -> None:
        """
        Validates transition from current state to `to_state`.
        Enforces Section 7 transition contract.
        """
        pair = (self.state, to_state)
        if pair not in VALID_TRANSITIONS:
            raise InvalidTransitionError(
                f"INVALID_FINALIZATION_TRANSITION: Cannot transition from {self.state} to {to_state}."
            )

        # Transitioning to A10 requires either A6 with quorum or A9 with recovery
        if to_state == FinalizationState.A10_AUTHORITY_EFFECTIVE:
            if self.state == FinalizationState.A6_ANCHOR_CONFIRMED and not quorum_satisfied:
                raise InvalidTransitionError(
                    "CANNOT_TRANSITION_TO_EFFECTIVE: Quorum must be satisfied to transition from A6 to A10."
                )
            if self.state == FinalizationState.A9_GOVERNANCE_RECOVERY_REVIEW and not recovery_authorized:
                raise InvalidTransitionError(
                    "CANNOT_TRANSITION_TO_EFFECTIVE: Explicit governance recovery authorization required from A9."
                )

    def transition(
        self,
        to_state: FinalizationState,
        effective_time_utc: Optional[str] = None,
        quorum_satisfied: bool = False,
        recovery_authorized: bool = False,
    ) -> None:
        """Executes valid state transition and manages effective authority status."""
        self.validate_transition(to_state, quorum_satisfied, recovery_authorized)

        self.state = to_state

        if to_state == FinalizationState.A10_AUTHORITY_EFFECTIVE:
            self.effective_status = AuthorityEffectiveStatus.EFFECTIVE
            if effective_time_utc is None:
                raise ValueError("effective_time_utc must be provided when transitioning to A10.")
            self._record_effective_time(effective_time_utc)
        elif to_state in (FinalizationState.A8_ANCHOR_CONFLICT, FinalizationState.A11_TERMINAL_HOLD):
            self.effective_status = AuthorityEffectiveStatus.TERMINAL_HOLD
        elif to_state in (FinalizationState.A7_ANCHOR_PENDING_EXPIRED, FinalizationState.A9_GOVERNANCE_RECOVERY_REVIEW):
            self.effective_status = AuthorityEffectiveStatus.BLOCKED_PENDING_GOVERNANCE_RECOVERY
        else:
            self.effective_status = AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR

    def _record_effective_time(self, effective_time_utc: str) -> None:
        """Enforces T1 <= T2 <= T3 (Sections 8, 29) and blocks backdating."""
        self.authority_became_effective_at = effective_time_utc
        if self.evidence_observed_at and self.eligibility_established_at:
            t1 = datetime.fromisoformat(self.evidence_observed_at.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(self.eligibility_established_at.replace("Z", "+00:00"))
            t3 = datetime.fromisoformat(effective_time_utc.replace("Z", "+00:00"))
            if not (t1 <= t2 <= t3):
                raise ValueError(
                    f"AUTHORITY_TIMING_VIOLATION: Timings violate T1 <= T2 <= T3. "
                    f"T1={self.evidence_observed_at}, T2={self.eligibility_established_at}, T3={effective_time_utc}"
                )

    def get_timing_record(self) -> AuthorityTimingRecord:
        return AuthorityTimingRecord(
            evidence_observed_at=self.evidence_observed_at or "",
            authority_eligibility_established_at=self.eligibility_established_at or "",
            authority_became_effective_at=self.authority_became_effective_at,
        )
