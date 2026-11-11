"""
HVA-01 Canonical Role and Permission Model.
Establishes deny-by-default role-based access control (RBAC) and explicit permissions
for HVA-00 V2.1 blinded human evaluation under ARX Closure Model v2.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class Role(str, Enum):
    """
    Ratified participant roles for HVA-00 V2.1 / HVA-01 evaluation.
    Enforces separation-of-duties and conflict-of-interest firewalls.
    """
    RESEARCH_OWNER = "RESEARCH_OWNER"
    ALLOCATION_ADMINISTRATOR = "ALLOCATION_ADMINISTRATOR"
    REVIEWER = "REVIEWER"
    CUSTODIAN = "CUSTODIAN"
    INDEPENDENT_WITNESS = "INDEPENDENT_WITNESS"
    TECHNICAL_OPERATOR = "TECHNICAL_OPERATOR"


class Permission(str, Enum):
    """
    Atomic permissions for protected evaluation operations.
    """
    # Session lifecycle
    SESSION_CREATE = "SESSION_CREATE"
    SESSION_AUTHORIZE = "SESSION_AUTHORIZE"
    SESSION_READ_OWN = "SESSION_READ_OWN"
    SESSION_READ_ALL = "SESSION_READ_ALL"
    SESSION_WITHDRAW_OWN = "SESSION_WITHDRAW_OWN"
    SESSION_CLOSE = "SESSION_CLOSE"

    # Evaluation & Response locking
    RESPONSE_SUBMIT_OWN = "RESPONSE_SUBMIT_OWN"
    RESPONSE_LOCK_OWN = "RESPONSE_LOCK_OWN"
    CORRECTION_PETITION_OWN = "CORRECTION_PETITION_OWN"

    # Asset & Case access
    ASSET_READ_ASSIGNED = "ASSET_READ_ASSIGNED"
    ASSET_READ_ALL = "ASSET_READ_ALL"

    # Allocation management
    ALLOCATION_GENERATE = "ALLOCATION_GENERATE"
    ALLOCATION_LOCK = "ALLOCATION_LOCK"
    ALLOCATION_REVOKE = "ALLOCATION_REVOKE"
    ALLOCATION_READ_OWN = "ALLOCATION_READ_OWN"
    ALLOCATION_READ_ALL = "ALLOCATION_READ_ALL"

    # Audit & Witness custody
    AUDIT_VERIFY = "AUDIT_VERIFY"
    WITNESS_ANCHOR = "WITNESS_ANCHOR"

    # Restricted Custody
    UNBLIND_KEYS_ACCESS = "UNBLIND_KEYS_ACCESS"


# Canonical deny-by-default role-permission matrix
ROLE_PERMISSIONS: Dict[Role, FrozenSet[Permission]] = {
    Role.RESEARCH_OWNER: frozenset({
        Permission.SESSION_CREATE,
        Permission.SESSION_READ_ALL,
        Permission.SESSION_CLOSE,
        Permission.ASSET_READ_ALL,
        Permission.ALLOCATION_GENERATE,
        Permission.ALLOCATION_LOCK,
        Permission.ALLOCATION_REVOKE,
        Permission.ALLOCATION_READ_ALL,
        Permission.AUDIT_VERIFY,
        Permission.UNBLIND_KEYS_ACCESS,
    }),
    Role.ALLOCATION_ADMINISTRATOR: frozenset({
        Permission.SESSION_CREATE,
        Permission.ALLOCATION_GENERATE,
        Permission.ALLOCATION_LOCK,
        Permission.ALLOCATION_REVOKE,
        Permission.ALLOCATION_READ_ALL,
    }),
    Role.REVIEWER: frozenset({
        Permission.SESSION_AUTHORIZE,
        Permission.SESSION_READ_OWN,
        Permission.SESSION_WITHDRAW_OWN,
        Permission.SESSION_CLOSE,
        Permission.RESPONSE_SUBMIT_OWN,
        Permission.RESPONSE_LOCK_OWN,
        Permission.CORRECTION_PETITION_OWN,
        Permission.ASSET_READ_ASSIGNED,
        Permission.ALLOCATION_READ_OWN,
    }),
    Role.CUSTODIAN: frozenset({
        Permission.SESSION_READ_ALL,
        Permission.ASSET_READ_ALL,
        Permission.ALLOCATION_READ_ALL,
        Permission.AUDIT_VERIFY,
        Permission.UNBLIND_KEYS_ACCESS,
    }),
    Role.INDEPENDENT_WITNESS: frozenset({
        Permission.AUDIT_VERIFY,
        Permission.WITNESS_ANCHOR,
    }),
    Role.TECHNICAL_OPERATOR: frozenset({
        Permission.SESSION_CREATE,
        Permission.ASSET_READ_ALL,
        Permission.AUDIT_VERIFY,
    }),
}
