"""
HVA-01 Reviewer Allocation Ledger.
Provides transactional, append-only persistence and audit logging for reviewer allocations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..auth.enforcer import (
    AuthorizationError,
    CrossSessionAccessError,
    SecurityEnforcer,
)
from ..auth.principal import Principal
from ..auth.roles import Permission, Role
from ..storage.append_store import FileLock
from .models import AllocationRecord, AllocationStateError


class AllocationLedger:
    """
    Append-only ledger for reviewer allocations and pre-session registrations.
    Thread-safe and multi-process safe using local file locking.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir = Path(base_dir)
        self.alloc_dir = self.base_dir / "allocations"
        self.alloc_dir.mkdir(parents=True, exist_ok=True)
        self.alloc_file = self.alloc_dir / "allocations.jsonl"
        self.audit_file = self.alloc_dir / "allocations_audit.jsonl"
        self.lock_file = self.alloc_dir / "allocations.lock"
        self.enforcer = SecurityEnforcer()

    def _append_audit(self, event_code: str, actor: str, details: Dict[str, Any]) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp_utc": now_utc,
            "event_code": event_code,
            "actor": actor,
            "details": details,
        }
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()

    def register_allocation(self, principal: Principal, record: AllocationRecord) -> AllocationRecord:
        """Register a new pre-session allocation."""
        self.enforcer.check_authorization(principal, Permission.ALLOCATION_GENERATE)

        if not record.verify_hash():
            raise AllocationStateError("Allocation record integrity check failed: hash mismatch.")

        with FileLock(str(self.lock_file)):
            active_allocations = self._load_active_records_unlocked()

            # Check for existing allocation ID
            if record.allocation_id in active_allocations:
                existing = active_allocations[record.allocation_id]
                if existing.is_locked:
                    raise AllocationStateError(
                        f"ALLOCATION_LOCKED: Cannot modify already locked allocation '{record.allocation_id}'."
                    )
                raise AllocationStateError(
                    f"DUPLICATE_ALLOCATION_ID: Allocation '{record.allocation_id}' already registered."
                )

            # Check between-subjects invariants against other active allocations
            for existing in active_allocations.values():
                if existing.is_revoked:
                    continue
                if existing.reviewer_pseudonym == record.reviewer_pseudonym:
                    raise AllocationStateError(
                        f"BETWEEN_SUBJECTS_VIOLATION: Reviewer '{record.reviewer_pseudonym}' "
                        f"is already allocated to package '{existing.assigned_package}'."
                    )
                if existing.assigned_package == record.assigned_package:
                    raise AllocationStateError(
                        f"DUPLICATE_PACKAGE_ALLOCATION: Package '{record.assigned_package}' "
                        f"is already allocated to '{existing.reviewer_pseudonym}'."
                    )

            # Append to allocations.jsonl
            with open(self.alloc_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                f.flush()

            self._append_audit(
                event_code="EVT_ALLOCATION_REGISTERED",
                actor=principal.principal_id,
                details={
                    "allocation_id": record.allocation_id,
                    "reviewer_pseudonym": record.reviewer_pseudonym,
                    "assigned_package": record.assigned_package,
                    "allocation_hash": record.allocation_hash,
                },
            )

        return record

    def lock_allocation(self, principal: Principal, allocation_id: str) -> AllocationRecord:
        """Formally lock an allocation, preventing any future reassignment or modification."""
        self.enforcer.check_authorization(principal, Permission.ALLOCATION_LOCK)

        with FileLock(str(self.lock_file)):
            active_allocations = self._load_active_records_unlocked()
            if allocation_id not in active_allocations:
                raise AllocationStateError(f"Allocation '{allocation_id}' not found.")

            current = active_allocations[allocation_id]
            if current.is_locked:
                return current  # Idempotent lock

            locked_record = current.with_lock()

            with open(self.alloc_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(locked_record.to_dict(), sort_keys=True) + "\n")
                f.flush()

            self._append_audit(
                event_code="EVT_ALLOCATION_LOCKED",
                actor=principal.principal_id,
                details={
                    "allocation_id": allocation_id,
                    "locked_at_utc": locked_record.locked_at_utc,
                    "allocation_hash": locked_record.allocation_hash,
                },
            )

        return locked_record

    def revoke_allocation(self, principal: Principal, allocation_id: str, reason: str) -> AllocationRecord:
        """Revoke an allocation. Preserves historical entry in audit log."""
        self.enforcer.check_authorization(principal, Permission.ALLOCATION_REVOKE)

        with FileLock(str(self.lock_file)):
            active_allocations = self._load_active_records_unlocked()
            if allocation_id not in active_allocations:
                raise AllocationStateError(f"Allocation '{allocation_id}' not found.")

            current = active_allocations[allocation_id]
            revoked_record = current.with_revocation(reason=reason)

            with open(self.alloc_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(revoked_record.to_dict(), sort_keys=True) + "\n")
                f.flush()

            self._append_audit(
                event_code="EVT_ALLOCATION_REVOKED",
                actor=principal.principal_id,
                details={
                    "allocation_id": allocation_id,
                    "reason": reason,
                    "revoked_at_utc": revoked_record.revoked_at_utc,
                },
            )

        return revoked_record

    def get_allocation_by_id(self, principal: Principal, allocation_id: str) -> AllocationRecord:
        """Fetch allocation by ID with permission enforcement."""
        with FileLock(str(self.lock_file)):
            active_allocations = self._load_active_records_unlocked()
            if allocation_id not in active_allocations:
                raise AllocationStateError(f"Allocation '{allocation_id}' not found.")
            rec = active_allocations[allocation_id]

            if principal.role == Role.REVIEWER:
                if rec.reviewer_pseudonym != principal.principal_id:
                    raise CrossSessionAccessError(
                        f"CROSS_ALLOCATION_ACCESS_BLOCKED: Reviewer '{principal.principal_id}' "
                        f"cannot inspect allocation '{allocation_id}' belonging to '{rec.reviewer_pseudonym}'."
                    )
            else:
                self.enforcer.check_authorization(principal, Permission.ALLOCATION_READ_ALL)

            return rec

    def get_allocation_for_reviewer(self, principal: Principal, pseudonym: str) -> AllocationRecord:
        """Fetch active allocation for a given reviewer pseudonym."""
        if principal.role == Role.REVIEWER:
            if principal.principal_id != pseudonym:
                raise CrossSessionAccessError(
                    f"CROSS_REVIEWER_ACCESS_BLOCKED: Reviewer '{principal.principal_id}' "
                    f"cannot inspect allocations for '{pseudonym}'."
                )
        else:
            self.enforcer.check_authorization(principal, Permission.ALLOCATION_READ_ALL)

        with FileLock(str(self.lock_file)):
            active_allocations = self._load_active_records_unlocked()
            for rec in active_allocations.values():
                if rec.reviewer_pseudonym == pseudonym:
                    return rec

            raise AllocationStateError(f"No active allocation found for reviewer '{pseudonym}'.")

    def list_allocations(self, principal: Principal) -> List[AllocationRecord]:
        """List allocations based on role permissions."""
        with FileLock(str(self.lock_file)):
            active = self._load_active_records_unlocked()
            if principal.role == Role.REVIEWER:
                return [r for r in active.values() if r.reviewer_pseudonym == principal.principal_id]
            else:
                self.enforcer.check_authorization(principal, Permission.ALLOCATION_READ_ALL)
                return list(active.values())

    def _load_active_records_unlocked(self) -> Dict[str, AllocationRecord]:
        """Load latest state for each allocation ID from allocations.jsonl."""
        if not self.alloc_file.exists():
            return {}

        records: Dict[str, AllocationRecord] = {}
        with open(self.alloc_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rec = AllocationRecord.from_dict(data)
                records[rec.allocation_id] = rec
        return records

    def verify_ledger_integrity(self) -> bool:
        """Verify hash integrity of every allocation entry in the ledger."""
        if not self.alloc_file.exists():
            return True

        with FileLock(str(self.lock_file)):
            with open(self.alloc_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    rec = AllocationRecord.from_dict(data)
                    if not rec.verify_hash():
                        raise AllocationStateError(
                            f"ALLOCATION_LEDGER_CORRUPTION: Hash verification failed on line {line_idx} "
                            f"for allocation '{rec.allocation_id}'."
                        )
        return True
