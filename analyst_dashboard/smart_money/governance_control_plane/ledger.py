"""
Authority Ledger Subsystem
ARX Terminal — Smart Money

Implements the external append-only Governance Authority Ledger (Option B model)
with single-head enforcement, anti-fork detection, rollback protection, and hash chaining.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone

from .canonicalization import sha256_canonical, domain_normalize


class LedgerForkException(Exception):
    """Raised when an event attempts to branch or fork from non-head state."""
    pass


class RollbackDetectedException(Exception):
    """Raised when a candidate state has a lower sequence or diverges from witnessed head."""
    pass


class SecondGenesisProhibitedException(Exception):
    """Raised when an attempt is made to create a second genesis event."""
    pass


class InvalidLedgerEventException(Exception):
    """Raised when an event fails schema, sequence, or cryptographic validation."""
    pass


@dataclass
class LedgerEvent:
    event_id: str
    governance_epoch: str
    sequence: int
    event_type: str
    timestamp_utc: str
    actor_identity: str
    payload: Dict[str, Any]
    previous_event_hash: str
    event_hash: str
    head_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuthorityLedger:
    """
    Append-only cryptographic authority ledger for Governance Control Plane.
    Enforces exactly one authoritative head per epoch and rejects silent forks.
    """

    GENESIS_PREVIOUS_HASH = "0" * 64

    def __init__(self, epoch: str = "epoch-gov-001"):
        self.epoch = epoch
        self.events: List[LedgerEvent] = []
        self._head: Optional[LedgerEvent] = None
        self._witnessed_sequences: List[int] = []

    @property
    def head(self) -> Optional[LedgerEvent]:
        return self._head

    @property
    def sequence(self) -> int:
        return self._head.sequence if self._head else -1

    @property
    def head_hash(self) -> Optional[str]:
        return self._head.head_hash if self._head else None

    def create_genesis(
        self,
        genesis_manifest: Dict[str, Any],
        actor_identity: str = "urn:arx:root:governance-control-plane:v1",
        timestamp_utc: Optional[str] = None,
    ) -> LedgerEvent:
        if self.events or self._head is not None:
            raise SecondGenesisProhibitedException(
                "Second genesis is strictly prohibited: an authoritative epoch already exists."
            )

        if not timestamp_utc:
            timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        seq = 0
        event_id = f"urn:arx:event:{self.epoch}:{seq}:{uuid.uuid4().hex[:12]}"
        prev_hash = self.GENESIS_PREVIOUS_HASH

        body = {
            "event_id": event_id,
            "governance_epoch": self.epoch,
            "sequence": seq,
            "event_type": "POLICY_RATIFIED",
            "timestamp_utc": timestamp_utc,
            "actor_identity": actor_identity,
            "payload": genesis_manifest,
            "previous_event_hash": prev_hash,
        }
        event_hash = sha256_canonical(body)
        head_hash = event_hash

        event = LedgerEvent(
            event_id=event_id,
            governance_epoch=self.epoch,
            sequence=seq,
            event_type="POLICY_RATIFIED",
            timestamp_utc=timestamp_utc,
            actor_identity=actor_identity,
            payload=genesis_manifest,
            previous_event_hash=prev_hash,
            event_hash=event_hash,
            head_hash=head_hash,
        )

        self.events.append(event)
        self._head = event
        self._witnessed_sequences.append(seq)
        return event

    def append_event(
        self,
        event_type: str,
        actor_identity: str,
        payload: Dict[str, Any],
        expected_previous_hash: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
    ) -> LedgerEvent:
        if self._head is None:
            raise InvalidLedgerEventException("Cannot append event before genesis is created.")

        current_head_hash = self._head.head_hash

        # Anti-equivocation / anti-fork check:
        if expected_previous_hash is not None and expected_previous_hash != current_head_hash:
            raise LedgerForkException(
                f"Ledger fork detected: expected previous hash {expected_previous_hash} "
                f"does not match current head hash {current_head_hash}."
            )

        if not timestamp_utc:
            timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        seq = self._head.sequence + 1
        event_id = f"urn:arx:event:{self.epoch}:{seq}:{uuid.uuid4().hex[:12]}"

        body = {
            "event_id": event_id,
            "governance_epoch": self.epoch,
            "sequence": seq,
            "event_type": event_type,
            "timestamp_utc": timestamp_utc,
            "actor_identity": actor_identity,
            "payload": payload,
            "previous_event_hash": current_head_hash,
        }
        event_hash = sha256_canonical(body)
        head_hash = event_hash

        event = LedgerEvent(
            event_id=event_id,
            governance_epoch=self.epoch,
            sequence=seq,
            event_type=event_type,
            timestamp_utc=timestamp_utc,
            actor_identity=actor_identity,
            payload=payload,
            previous_event_hash=current_head_hash,
            event_hash=event_hash,
            head_hash=head_hash,
        )

        self.events.append(event)
        self._head = event
        self._witnessed_sequences.append(seq)
        return event

    def verify_integrity(self) -> bool:
        """Verifies full cryptographic chain from genesis to head."""
        if not self.events:
            return True

        for i, ev in enumerate(self.events):
            if i == 0:
                if ev.sequence != 0 or ev.previous_event_hash != self.GENESIS_PREVIOUS_HASH:
                    return False
            else:
                prev_ev = self.events[i - 1]
                if ev.sequence != prev_ev.sequence + 1:
                    return False
                if ev.previous_event_hash != prev_ev.head_hash:
                    return False

            body = {
                "event_id": ev.event_id,
                "governance_epoch": ev.governance_epoch,
                "sequence": ev.sequence,
                "event_type": ev.event_type,
                "timestamp_utc": ev.timestamp_utc,
                "actor_identity": ev.actor_identity,
                "payload": ev.payload,
                "previous_event_hash": ev.previous_event_hash,
            }
            computed_hash = sha256_canonical(body)
            if ev.event_hash != computed_hash or ev.head_hash != computed_hash:
                return False

        return True

    def rehydrate(
        self,
        event_dicts: List[Dict[str, Any]],
        prior_witnessed_sequence: Optional[int] = None,
        prior_witnessed_head_hash: Optional[str] = None,
    ) -> None:
        """
        Rehydrates ledger from an external event list.
        Validates anti-rollback: restored sequence >= witnessed sequence and proves ancestry.
        """
        if not event_dicts:
            raise InvalidLedgerEventException("Cannot rehydrate from empty event list.")

        rehydrated_events = []
        for d in event_dicts:
            ev = LedgerEvent(**d)
            rehydrated_events.append(ev)

        candidate_head = rehydrated_events[-1]

        # Anti-rollback check
        if prior_witnessed_sequence is not None:
            if candidate_head.sequence < prior_witnessed_sequence:
                raise RollbackDetectedException(
                    f"Rollback rejected: candidate sequence {candidate_head.sequence} "
                    f"is less than prior witnessed sequence {prior_witnessed_sequence}."
                )

        if prior_witnessed_head_hash is not None:
            # Check if prior witnessed head hash is in the candidate ancestry
            hashes = {e.head_hash for e in rehydrated_events}
            if prior_witnessed_head_hash not in hashes:
                raise RollbackDetectedException(
                    f"Rollback rejected: prior witnessed head hash {prior_witnessed_head_hash} "
                    f"not found in candidate event ancestry."
                )

        temp_ledger = AuthorityLedger(epoch=self.epoch)
        temp_ledger.events = rehydrated_events
        temp_ledger._head = candidate_head
        if not temp_ledger.verify_integrity():
            raise InvalidLedgerEventException("Rehydrated event list failed cryptographic integrity check.")

        self.events = rehydrated_events
        self._head = candidate_head
        self._witnessed_sequences.append(candidate_head.sequence)
