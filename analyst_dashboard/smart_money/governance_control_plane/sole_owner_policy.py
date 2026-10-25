"""
ARX Terminal — Smart Money Governance Control Plane
Sole Owner Temporary FIDO2 Suspension Policy Implementation

Implements:
- ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1 policy validation.
- Single-owner release approval verification (exact SHA binding, environment check, expiration, MFA).
- Reversible FIDO2 reactivation lifecycle (TEMPORARILY_SUSPENDED -> ENROLLMENT_READY -> FIDO2_REQUIRED).
- Non-collapsible audit logging.
"""

from __future__ import annotations

import re
import uuid
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, List


class Fido2RequirementStatus(str, Enum):
    TEMPORARILY_SUSPENDED = "TEMPORARILY_SUSPENDED"
    ENROLLMENT_READY = "ENROLLMENT_READY"
    FIDO2_REQUIRED = "FIDO2_REQUIRED"


class ReleaseApprovalException(Exception):
    """Base exception for release authorization failures."""
    pass


class ReleaseApprovalExpiredException(ReleaseApprovalException):
    """Raised when release approval validity timestamp has lapsed."""
    pass


class ReleaseApprovalMismatchedShaException(ReleaseApprovalException):
    """Raised when approval commit SHA does not match target release SHA."""
    pass


class ReleaseApprovalMismatchedEnvironmentException(ReleaseApprovalException):
    """Raised when approval target environment does not match requested deployment environment."""
    pass


class ReleaseApprovalMissingMfaException(ReleaseApprovalException):
    """Raised when operator account MFA is unverified."""
    pass


class ReleaseApprovalTechnicalPrerequisiteException(ReleaseApprovalException):
    """Raised when technical, security, or recovery verifications are incomplete."""
    pass


@dataclass(frozen=True)
class SoleOwnerReleaseApproval:
    approval_id: str
    policy_id: str
    owner_identity: str
    target_environment: str
    release_sha: str
    approved_at: str
    valid_until: str
    account_mfa_verified: bool
    technical_verification_passed: bool
    security_verification_passed: bool
    recovery_verification_passed: bool
    fido2_assertion: Optional[str] = None
    approval_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SoleOwnerGovernanceValidator:
    """
    Validates deployment requests under ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1.
    """

    POLICY_ID = "ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1"

    def __init__(self, status: Fido2RequirementStatus = Fido2RequirementStatus.TEMPORARILY_SUSPENDED):
        self.status = status

    def validate_release_authorization(
        self,
        approval: SoleOwnerReleaseApproval,
        target_sha: str,
        target_environment: str,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Validates that a release approval satisfies all criteria under the active policy.
        Fails closed on any violation.
        """
        now = current_time or datetime.now(timezone.utc)

        # 1. Policy Identity Check
        if approval.policy_id != self.POLICY_ID:
            raise ReleaseApprovalException(
                f"POLICY_MISMATCH: Expected {self.POLICY_ID}, got {approval.policy_id}"
            )

        # 2. Exact SHA Binding Check
        clean_target_sha = target_sha.strip().lower()
        clean_approval_sha = approval.release_sha.strip().lower()
        if clean_target_sha != clean_approval_sha:
            raise ReleaseApprovalMismatchedShaException(
                f"RELEASE_SHA_MISMATCH: Target SHA {clean_target_sha} does not match approval SHA {clean_approval_sha}"
            )

        # 3. Target Environment Check
        if approval.target_environment != target_environment:
            raise ReleaseApprovalMismatchedEnvironmentException(
                f"ENVIRONMENT_MISMATCH: Target {target_environment} does not match approval {approval.target_environment}"
            )

        # 4. Validity Window (Expiration) Check
        valid_until_dt = datetime.fromisoformat(approval.valid_until.replace("Z", "+00:00"))
        if now > valid_until_dt:
            raise ReleaseApprovalExpiredException(
                f"APPROVAL_EXPIRED: Approval expired at {approval.valid_until}, current time is {now.isoformat()}"
            )

        # 5. Account MFA Check
        if not approval.account_mfa_verified:
            raise ReleaseApprovalMissingMfaException(
                "MFA_UNVERIFIED: Account MFA must be verified for sole owner release authorization."
            )

        # 6. Technical, Security & Recovery Verification Checks
        if not approval.technical_verification_passed:
            raise ReleaseApprovalTechnicalPrerequisiteException(
                "TECHNICAL_VERIFICATION_MISSING: Automated test suite verification must pass."
            )
        if not approval.security_verification_passed:
            raise ReleaseApprovalTechnicalPrerequisiteException(
                "SECURITY_VERIFICATION_MISSING: Pre-deployment security checklist must pass."
            )
        if not approval.recovery_verification_passed:
            raise ReleaseApprovalTechnicalPrerequisiteException(
                "RECOVERY_VERIFICATION_MISSING: Database restore verification must pass."
            )

        # 7. FIDO2 Status Check
        if self.status == Fido2RequirementStatus.FIDO2_REQUIRED:
            if not approval.fido2_assertion:
                raise ReleaseApprovalException(
                    "FIDO2_ASSERTION_REQUIRED: Active policy requires hardware FIDO2 assertion."
                )

        audit_record = {
            "validation_id": str(uuid.uuid4()),
            "status": "APPROVED",
            "policy_id": self.POLICY_ID,
            "fido2_status": self.status.value,
            "release_sha": clean_target_sha,
            "target_environment": target_environment,
            "owner_identity": approval.owner_identity,
            "validated_at": now.isoformat(),
            "audit_hash": hashlib.sha256(
                f"{self.POLICY_ID}:{clean_target_sha}:{target_environment}:{now.isoformat()}".encode()
            ).hexdigest(),
        }
        return audit_record
