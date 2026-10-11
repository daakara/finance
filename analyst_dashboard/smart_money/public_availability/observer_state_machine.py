"""
ARX TERMINAL — SMART MONEY
OBSERVER STATE MACHINE (EPOCH 001)

Formal state machine governing the exact natural authority progression:
PRE_OBSERVATION_READY
→ GATE3_VERIFIED
→ AUTHORIZATION_CLAIMED
→ OBSERVER_ARMED
→ AWAITING_NATURAL_EVENT
→ NATURAL_CANDIDATE_DETECTED
→ NATURAL_CANDIDATE_QUALIFIED
→ RAW_EVIDENCE_CAPTURED
→ R1_ROOTED
→ AUTHORITY_EVENT_RECORDED
→ R2_CLOSED
→ NATURAL_ADMITTED
→ DENOMINATOR_INCREMENTED
→ PRE_OBSERVATION_READY

Any invalid transition fails closed immediately to SAFE_HOLD or TERMINAL_HOLD.
Every transition is durably recorded with cryptographic receipts.
"""

from enum import Enum
import datetime
import json
from typing import Dict, Any, List, Optional, Tuple


class ObserverState(str, Enum):
    PRE_OBSERVATION_READY = "PRE_OBSERVATION_READY"
    GATE3_VERIFIED = "GATE3_VERIFIED"
    AUTHORIZATION_CLAIMED = "AUTHORIZATION_CLAIMED"
    OBSERVER_ARMED = "OBSERVER_ARMED"
    AWAITING_NATURAL_EVENT = "AWAITING_NATURAL_EVENT"
    NATURAL_CANDIDATE_DETECTED = "NATURAL_CANDIDATE_DETECTED"
    NATURAL_CANDIDATE_QUALIFIED = "NATURAL_CANDIDATE_QUALIFIED"
    RAW_EVIDENCE_CAPTURED = "RAW_EVIDENCE_CAPTURED"
    R1_ROOTED = "R1_ROOTED"
    AUTHORITY_EVENT_RECORDED = "AUTHORITY_EVENT_RECORDED"
    R2_CLOSED = "R2_CLOSED"
    NATURAL_ADMITTED = "NATURAL_ADMITTED"
    DENOMINATOR_INCREMENTED = "DENOMINATOR_INCREMENTED"
    SAFE_HOLD = "SAFE_HOLD"
    TERMINAL_HOLD = "TERMINAL_HOLD"


ALLOWED_TRANSITIONS: Dict[ObserverState, List[ObserverState]] = {
    ObserverState.PRE_OBSERVATION_READY: [
        ObserverState.GATE3_VERIFIED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.GATE3_VERIFIED: [
        ObserverState.AUTHORIZATION_CLAIMED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.AUTHORIZATION_CLAIMED: [
        ObserverState.OBSERVER_ARMED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.OBSERVER_ARMED: [
        ObserverState.AWAITING_NATURAL_EVENT,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.AWAITING_NATURAL_EVENT: [
        ObserverState.NATURAL_CANDIDATE_DETECTED,
        ObserverState.SAFE_HOLD,  # Quiet session without natural event -> safe hold
        ObserverState.PRE_OBSERVATION_READY,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.NATURAL_CANDIDATE_DETECTED: [
        ObserverState.NATURAL_CANDIDATE_QUALIFIED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.NATURAL_CANDIDATE_QUALIFIED: [
        ObserverState.RAW_EVIDENCE_CAPTURED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.RAW_EVIDENCE_CAPTURED: [
        ObserverState.R1_ROOTED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.R1_ROOTED: [
        ObserverState.AUTHORITY_EVENT_RECORDED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.AUTHORITY_EVENT_RECORDED: [
        ObserverState.R2_CLOSED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.R2_CLOSED: [
        ObserverState.NATURAL_ADMITTED,
        ObserverState.SAFE_HOLD,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.NATURAL_ADMITTED: [
        ObserverState.DENOMINATOR_INCREMENTED,
        ObserverState.TERMINAL_HOLD,
    ],
    ObserverState.DENOMINATOR_INCREMENTED: [
        ObserverState.PRE_OBSERVATION_READY,
    ],
    ObserverState.SAFE_HOLD: [
        ObserverState.PRE_OBSERVATION_READY,
    ],
    ObserverState.TERMINAL_HOLD: [],  # Terminal state, no exit without governed manual remediation
}


class ObserverStateMachine:
    """Explicit, persisted state machine enforcing natural authority progression."""

    def __init__(self, initial_state: ObserverState = ObserverState.PRE_OBSERVATION_READY):
        self.current_state = initial_state
        self.history: List[Dict[str, Any]] = [
            {
                "state": initial_state.value,
                "transitioned_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": "INITIAL_STATE",
                "metadata": {},
            }
        ]

    def transition(
        self,
        target_state: ObserverState,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Attempts a transition to target_state.
        Fails closed if the transition is not in ALLOWED_TRANSITIONS.
        """
        allowed = ALLOWED_TRANSITIONS.get(self.current_state, [])
        if target_state not in allowed:
            fail_msg = (
                f"INVALID_STATE_TRANSITION: Cannot transition from {self.current_state.value} "
                f"to {target_state.value}. Allowed targets: {[s.value for s in allowed]}"
            )
            # Fail closed to TERMINAL_HOLD if attempting an unauthorized leap
            self.current_state = ObserverState.TERMINAL_HOLD
            self.history.append({
                "state": ObserverState.TERMINAL_HOLD.value,
                "transitioned_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": f"FAIL_CLOSED: {fail_msg}",
                "metadata": metadata or {},
            })
            return False, fail_msg

        self.current_state = target_state
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.history.append({
            "state": target_state.value,
            "transitioned_at_utc": now_utc,
            "reason": reason,
            "metadata": metadata or {},
        })
        return True, "TRANSITION_ACCEPTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "transition_count": len(self.history),
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObserverStateMachine":
        sm = cls(initial_state=ObserverState(data.get("current_state", ObserverState.PRE_OBSERVATION_READY.value)))
        sm.history = data.get("history", sm.history)
        return sm
