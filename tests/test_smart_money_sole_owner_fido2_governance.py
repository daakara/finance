"""
Governance Regression Suite: ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1
Mechanically proves all Section 21 Governance Regression Invariants.
"""

import uuid
from datetime import datetime, timezone, timedelta
import pytest

from analyst_dashboard.smart_money.governance_control_plane.sole_owner_policy import (
    SoleOwnerGovernanceValidator,
    SoleOwnerReleaseApproval,
    Fido2RequirementStatus,
    ReleaseApprovalException,
    ReleaseApprovalExpiredException,
    ReleaseApprovalMismatchedShaException,
    ReleaseApprovalMismatchedEnvironmentException,
    ReleaseApprovalMissingMfaException,
    ReleaseApprovalTechnicalPrerequisiteException,
)


@pytest.fixture
def valid_approval():
    now = datetime.now(timezone.utc)
    return SoleOwnerReleaseApproval(
        approval_id=str(uuid.uuid4()),
        policy_id="ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1",
        owner_identity="lead-maintainer-daakara",
        target_environment="production",
        release_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
        approved_at=now.isoformat(),
        valid_until=(now + timedelta(hours=4)).isoformat(),
        account_mfa_verified=True,
        technical_verification_passed=True,
        security_verification_passed=True,
        recovery_verification_passed=True,
    )


class TestSoleOwnerGovernanceRegressionMatrix:
    """Verifies all Section 21 regression invariants."""

    def test_valid_approval_succeeds_without_fido2_device(self, valid_approval):
        validator = SoleOwnerGovernanceValidator(status=Fido2RequirementStatus.TEMPORARILY_SUSPENDED)
        record = validator.validate_release_authorization(
            approval=valid_approval,
            target_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
            target_environment="production",
        )
        assert record["status"] == "APPROVED"
        assert record["fido2_status"] == "TEMPORARILY_SUSPENDED"
        assert record["audit_hash"] is not None

    def test_wrong_release_sha_blocks_release(self, valid_approval):
        validator = SoleOwnerGovernanceValidator()
        with pytest.raises(ReleaseApprovalMismatchedShaException):
            validator.validate_release_authorization(
                approval=valid_approval,
                target_sha="09f183de52e275ae543615747dff2f9ae00ef9ca",  # Mismatched SHA
                target_environment="production",
            )

    def test_wrong_target_environment_blocks_release(self, valid_approval):
        validator = SoleOwnerGovernanceValidator()
        with pytest.raises(ReleaseApprovalMismatchedEnvironmentException):
            validator.validate_release_authorization(
                approval=valid_approval,
                target_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
                target_environment="staging",  # Mismatched env
            )

    def test_expired_approval_blocks_release(self, valid_approval):
        validator = SoleOwnerGovernanceValidator()
        future_time = datetime.now(timezone.utc) + timedelta(hours=5)
        with pytest.raises(ReleaseApprovalExpiredException):
            validator.validate_release_authorization(
                approval=valid_approval,
                target_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
                target_environment="production",
                current_time=future_time,
            )

    def test_unverified_mfa_blocks_release(self, valid_approval):
        validator = SoleOwnerGovernanceValidator()
        unverified_mfa = SoleOwnerReleaseApproval(
            **{**valid_approval.to_dict(), "account_mfa_verified": False}
        )
        with pytest.raises(ReleaseApprovalMissingMfaException):
            validator.validate_release_authorization(
                approval=unverified_mfa,
                target_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
                target_environment="production",
            )

    def test_failed_technical_or_recovery_verification_blocks_release(self, valid_approval):
        validator = SoleOwnerGovernanceValidator()
        
        # Test failed technical verification
        failed_tech = SoleOwnerReleaseApproval(
            **{**valid_approval.to_dict(), "technical_verification_passed": False}
        )
        with pytest.raises(ReleaseApprovalTechnicalPrerequisiteException):
            validator.validate_release_authorization(
                approval=failed_tech,
                target_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
                target_environment="production",
            )

        # Test failed recovery verification
        failed_recovery = SoleOwnerReleaseApproval(
            **{**valid_approval.to_dict(), "recovery_verification_passed": False}
        )
        with pytest.raises(ReleaseApprovalTechnicalPrerequisiteException):
            validator.validate_release_authorization(
                approval=failed_recovery,
                target_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
                target_environment="production",
            )

    def test_reactivated_fido2_enforcement_blocks_unauthenticated_release(self, valid_approval):
        # When status transitions to FIDO2_REQUIRED, missing hardware assertion fails closed
        validator = SoleOwnerGovernanceValidator(status=Fido2RequirementStatus.FIDO2_REQUIRED)
        with pytest.raises(ReleaseApprovalException) as exc_info:
            validator.validate_release_authorization(
                approval=valid_approval,
                target_sha="da7f6ff52b8d79658e041f935424c0fa57f42984",
                target_environment="production",
            )
        assert "FIDO2_ASSERTION_REQUIRED" in str(exc_info.value)
