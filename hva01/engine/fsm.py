"""
Deterministic Response-Locking Finite State Machine (FSM) for HVA-01.
Enforces the 8-state evaluation lifecycle, rejects backward transitions, and emits audit events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.session import SessionModel, SessionState


class StateTransitionError(ValueError):
    """Raised when an illegal state transition is requested."""
    pass


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_code: str
    timestamp_utc: str
    actor: str
    from_state: SessionState
    to_state: SessionState
    sequence_index: int
    case_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor,
            "case_id": self.case_id,
            "details": self.details,
            "event_code": self.event_code,
            "event_id": self.event_id,
            "from_state": self.from_state.value,
            "sequence_index": self.sequence_index,
            "timestamp_utc": self.timestamp_utc,
            "to_state": self.to_state.value,
        }


class ResponseLockingFSM:
    """
    State machine governing session and case evaluation lifecycle.
    
    Valid transition map:
    - SESSION_CREATED     -> SESSION_AUTHORIZED
    - SESSION_AUTHORIZED  -> RESPONSE_DRAFT
    - RESPONSE_DRAFT      -> RESPONSE_DRAFT (in-memory update)
    - RESPONSE_DRAFT      -> RESPONSE_SUBMITTED (in-flight validation)
    - RESPONSE_SUBMITTED  -> RESPONSE_LOCKED (atomic durable commit)
    - RESPONSE_SUBMITTED  -> RESPONSE_DRAFT (validation failure rollback)
    - RESPONSE_LOCKED     -> RESPONSE_DRAFT (sequence increment k -> k+1, where k < 59)
    - RESPONSE_LOCKED     -> SESSION_CLOSED (when k == 59 completed)
    - RESPONSE_LOCKED     -> CORRECTION_REQUESTED
    - CORRECTION_REQUESTED-> CORRECTION_RECORDED
    - CORRECTION_REQUESTED-> RESPONSE_LOCKED (petition denied)
    - CORRECTION_RECORDED -> RESPONSE_LOCKED
    """

    ALLOWED_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
        SessionState.SESSION_CREATED: {SessionState.SESSION_AUTHORIZED},
        SessionState.SESSION_AUTHORIZED: {SessionState.RESPONSE_DRAFT},
        SessionState.RESPONSE_DRAFT: {
            SessionState.RESPONSE_DRAFT,
            SessionState.RESPONSE_SUBMITTED,
            SessionState.CORRECTION_REQUESTED,
        },
        SessionState.RESPONSE_SUBMITTED: {SessionState.RESPONSE_LOCKED, SessionState.RESPONSE_DRAFT},
        SessionState.RESPONSE_LOCKED: {
            SessionState.RESPONSE_DRAFT,
            SessionState.SESSION_CLOSED,
            SessionState.CORRECTION_REQUESTED,
        },
        SessionState.CORRECTION_REQUESTED: {
            SessionState.CORRECTION_RECORDED,
            SessionState.RESPONSE_LOCKED,
            SessionState.RESPONSE_DRAFT,
        },
        SessionState.CORRECTION_RECORDED: {
            SessionState.RESPONSE_LOCKED,
            SessionState.RESPONSE_DRAFT,
        },
        SessionState.SESSION_CLOSED: set(),  # Terminal state: zero allowed transitions
    }

    def __init__(self, session: SessionModel) -> None:
        self.session_id = session.session_id
        self.reviewer_pseudonym = session.reviewer_pseudonym
        self.total_cases = session.total_cases
        self._current_state = session.current_state
        self._current_sequence = session.current_sequence
        self._locked_cases: Set[int] = set()
        self._audit_log: List[AuditEvent] = []
        self._event_counter = 0

    @property
    def current_state(self) -> SessionState:
        return self._current_state

    @property
    def current_sequence(self) -> int:
        return self._current_sequence

    @property
    def locked_cases(self) -> Set[int]:
        return set(self._locked_cases)

    @property
    def audit_log(self) -> List[AuditEvent]:
        return list(self._audit_log)

    def _emit_event(
        self,
        event_code: str,
        actor: str,
        from_state: SessionState,
        to_state: SessionState,
        case_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        self._event_counter += 1
        now_utc = datetime.now(timezone.utc).isoformat()
        evt = AuditEvent(
            event_id=f"EVT-{self.session_id}-{self._event_counter:04d}",
            event_code=event_code,
            timestamp_utc=now_utc,
            actor=actor,
            from_state=from_state,
            to_state=to_state,
            sequence_index=self._current_sequence,
            case_id=case_id,
            details=details or {},
        )
        self._audit_log.append(evt)
        return evt

    def transition_to(
        self,
        target_state: SessionState,
        actor: str,
        case_id: Optional[str] = None,
        event_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """
        Attempt a state transition. Raises StateTransitionError on any forbidden or backward step.
        """
        # Check terminal state
        if self._current_state == SessionState.SESSION_CLOSED:
            self._emit_event(
                event_code="ERR_SESSION_ALREADY_CLOSED",
                actor=actor,
                from_state=self._current_state,
                to_state=self._current_state,
                case_id=case_id,
                details={"target_state": target_state.value, "reason": "Session is closed and sealed."},
            )
            raise StateTransitionError(f"Cannot transition from terminal state {self._current_state.value}")

        # Check allowed transition table
        allowed = self.ALLOWED_TRANSITIONS.get(self._current_state, set())
        if target_state not in allowed:
            self._emit_event(
                event_code="ERR_ILLEGAL_STATE_TRANSITION",
                actor=actor,
                from_state=self._current_state,
                to_state=self._current_state,
                case_id=case_id,
                details={
                    "target_state": target_state.value,
                    "allowed_states": [s.value for s in allowed],
                },
            )
            raise StateTransitionError(
                f"Illegal state transition from {self._current_state.value} to {target_state.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )

        # Transition logic
        from_state = self._current_state
        code = event_code or f"EVT_TRANSITION_{target_state.value}"

        # If transitioning to RESPONSE_LOCKED, register the case as locked
        if target_state == SessionState.RESPONSE_LOCKED:
            self._locked_cases.add(self._current_sequence)
            code = "EVT_RESPONSE_LOCK"

        # If advancing to next case from RESPONSE_LOCKED -> RESPONSE_DRAFT
        if from_state == SessionState.RESPONSE_LOCKED and target_state == SessionState.RESPONSE_DRAFT:
            if self._current_sequence < self.total_cases:
                self._current_sequence += 1
                code = "EVT_ADVANCE_NEXT_CASE"
            else:
                raise StateTransitionError(
                    f"Cannot advance past final case {self.total_cases}; session must close."
                )

        # If closing session
        if target_state == SessionState.SESSION_CLOSED:
            if len(self._locked_cases) < self.total_cases:
                raise StateTransitionError(
                    f"Cannot close session: only {len(self._locked_cases)} of {self.total_cases} cases locked."
                )
            code = "EVT_SESSION_SEALED"

        self._current_state = target_state
        return self._emit_event(
            event_code=code,
            actor=actor,
            from_state=from_state,
            to_state=target_state,
            case_id=case_id,
            details=details,
        )

    def attempt_backtrack_to_case(self, target_sequence: int, actor: str) -> None:
        """
        Explicitly rejects any attempt to navigate or edit a previously locked case.
        """
        if target_sequence in self._locked_cases:
            self._emit_event(
                event_code="EVT_BACKTRACKING_REJECTED",
                actor=actor,
                from_state=self._current_state,
                to_state=self._current_state,
                details={
                    "attempted_sequence": target_sequence,
                    "current_sequence": self._current_sequence,
                    "reason": "Hard-lock policy forbids re-editing locked cases.",
                },
            )
            raise StateTransitionError(
                f"Backtracking prohibited: Case sequence {target_sequence} is permanently locked."
            )

    def restore_state(
        self,
        locked_cases: Set[int],
        current_sequence: int,
        current_state: SessionState,
    ) -> None:
        """
        Restore state machine state during crash recovery or session reload.
        """
        self._locked_cases = set(locked_cases)
        self._current_sequence = current_sequence
        self._current_state = current_state

