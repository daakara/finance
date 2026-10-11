"""
Downstream-Use Firewall Architecture (Sections 22, 23).
Mandatory fail-closed consumer gate that enforces authority status boundaries.
Rejects any downstream consumption (production decisions, empirical denominators,
model tuning, learning claims, backfills, retroactive justifications)
unless AUTHORITY_EFFECTIVE_STATUS == EFFECTIVE.
"""

from typing import Dict, Any, Optional

from analyst_dashboard.smart_money.anchor_finalization.models import (
    AuthorityEffectiveStatus,
    FinalizationState,
    DownstreamFirewallBlockedError,
)
from analyst_dashboard.smart_money.anchor_finalization.policy_registry import (
    PolicyResolutionResult,
)
from analyst_dashboard.smart_money.anchor_finalization.root_models import (
    FinalizationRootStatus,
)


class DownstreamUseFirewall:
    """
    Consumer-side fail-closed firewall.
    Every governed downstream consumer MUST invoke this firewall before accepting evidence.
    """

    @staticmethod
    def assert_finalization_root_allowed(root_status: FinalizationRootStatus) -> None:
        """Enforces: Authority is blocked if finalization root is not CONFIRMED (Section 28)."""
        if root_status != FinalizationRootStatus.CONFIRMED:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: Finalization root status '{root_status.value}' "
                f"prohibits authority effectiveness. Required 'CONFIRMED'."
            )

    @staticmethod
    def assert_policy_resolution_allowed(resolution: PolicyResolutionResult) -> None:
        """Enforces: Authority is blocked if policy resolution is not APPLICABLE (Section 24)."""
        if resolution != PolicyResolutionResult.APPLICABLE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: Policy resolution '{resolution.value}' "
                f"prohibits authority effectiveness."
            )

    @staticmethod
    def assert_production_decision_allowed(status: AuthorityEffectiveStatus) -> None:
        """Enforces: CAN_FEED_PRODUCTION_DECISION = NO unless status == EFFECTIVE."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_FEED_PRODUCTION_DECISION = NO. "
                f"Current authority status is '{status}', required 'EFFECTIVE'."
            )

    @staticmethod
    def assert_empirical_denominator_allowed(status: AuthorityEffectiveStatus) -> None:
        """Enforces: CAN_ENTER_EMPIRICAL_DENOMINATOR = NO unless status == EFFECTIVE."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_ENTER_EMPIRICAL_DENOMINATOR = NO. "
                f"Current authority status is '{status}', required 'EFFECTIVE'."
            )

    @staticmethod
    def assert_model_tuning_allowed(status: AuthorityEffectiveStatus) -> None:
        """Enforces: CAN_ENTER_MODEL_TUNING = NO unless status == EFFECTIVE."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_ENTER_MODEL_TUNING = NO. "
                f"Current authority status is '{status}', required 'EFFECTIVE'."
            )

    @staticmethod
    def assert_learning_claim_allowed(status: AuthorityEffectiveStatus) -> None:
        """Enforces: CAN_BE_USED_FOR_LEARNING_CLAIM = NO unless status == EFFECTIVE."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_BE_USED_FOR_LEARNING_CLAIM = NO. "
                f"Current authority status is '{status}', required 'EFFECTIVE'."
            )

    @staticmethod
    def assert_historical_backfill_allowed(status: AuthorityEffectiveStatus) -> None:
        """Enforces: CAN_BE_USED_FOR_HISTORICAL_BACKFILL = NO unless status == EFFECTIVE."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_BE_USED_FOR_HISTORICAL_BACKFILL = NO. "
                f"Current authority status is '{status}', required 'EFFECTIVE'."
            )

    @staticmethod
    def assert_retroactive_decision_justification_allowed(status: AuthorityEffectiveStatus) -> None:
        """Enforces: CAN_BE_USED_FOR_RETROACTIVE_DECISION_JUSTIFICATION = NO (ALWAYS PROHIBITED)."""
        raise DownstreamFirewallBlockedError(
            "DOWNSTREAM_FIREWALL_BLOCKED: CAN_BE_USED_FOR_RETROACTIVE_DECISION_JUSTIFICATION = NO. "
            "Retroactive justification of past decisions is strictly prohibited by policy."
        )

    @classmethod
    def evaluate_downstream_permissions(
        cls, status: AuthorityEffectiveStatus
    ) -> Dict[str, bool]:
        """
        Returns boolean flags representing downstream permissions.
        """
        is_effective = status == AuthorityEffectiveStatus.EFFECTIVE
        return {
            "can_feed_production_decision": is_effective,
            "can_enter_empirical_denominator": is_effective,
            "can_enter_model_tuning": is_effective,
            "can_be_used_for_learning_claim": is_effective,
            "can_be_used_for_historical_backfill": is_effective,
            "can_be_used_for_retroactive_decision_justification": False,
        }
