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
    PostAuthorityClosureStatus,
)


class DownstreamUseFirewall:
    """
    Consumer-side fail-closed firewall (Section 27).
    Every governed downstream consumer MUST invoke this firewall before accepting evidence.
    Enforces authority effectiveness and post-authority closure lifecycle boundaries.
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
    def assert_closure_status_allowed(closure: Any) -> None:
        """Enforces: Authority use is blocked if post-authority closure is EXPIRED or CONFLICT (Section 16, 17, 27)."""
        closure_status = getattr(closure, "closure_status", closure)
        if closure_status == PostAuthorityClosureStatus.CONFLICT:
            raise DownstreamFirewallBlockedError(
                "DOWNSTREAM_FIREWALL_BLOCKED: CLOSURE_CONFLICT prohibits downstream authority use."
            )
        if closure_status == PostAuthorityClosureStatus.EXPIRED:
            raise DownstreamFirewallBlockedError(
                "DOWNSTREAM_FIREWALL_BLOCKED: CLOSURE_EXPIRED prohibits downstream authority use."
            )
        if closure_status not in (PostAuthorityClosureStatus.CONFIRMED, PostAuthorityClosureStatus.PENDING):
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: Post-authority closure status '{closure_status}' prohibits downstream authority use."
            )

    @staticmethod
    def assert_authority_effective(
        authority_effective_status: AuthorityEffectiveStatus,
        closure_status: Optional[Any] = None,
    ) -> None:
        """Enforces: Authority is effective only if status is EFFECTIVE and closure not blocked."""
        if authority_effective_status == AuthorityEffectiveStatus.BLOCKED_INTEGRITY_HOLD:
            raise DownstreamFirewallBlockedError(
                "DOWNSTREAM_FIREWALL_BLOCKED: AUTHORITY_BLOCKED_INTEGRITY_HOLD prohibits downstream authority use."
            )
        if authority_effective_status == AuthorityEffectiveStatus.SUSPENDED_INTEGRITY_HOLD:
            raise DownstreamFirewallBlockedError(
                "DOWNSTREAM_FIREWALL_BLOCKED: AUTHORITY_SUSPENDED_INTEGRITY_HOLD prohibits downstream authority use."
            )
        if authority_effective_status == AuthorityEffectiveStatus.SUSPENDED_PENDING_CLOSURE:
            raise DownstreamFirewallBlockedError(
                "DOWNSTREAM_FIREWALL_BLOCKED: AUTHORITY_SUSPENDED_PENDING_CLOSURE prohibits downstream authority use."
            )
        if authority_effective_status == AuthorityEffectiveStatus.BLOCKED_PENDING_FINALIZATION_ROOT:
            raise DownstreamFirewallBlockedError(
                "DOWNSTREAM_FIREWALL_BLOCKED: AUTHORITY_BLOCKED_PENDING_FINALIZATION_ROOT prohibits downstream authority use."
            )
        if authority_effective_status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: Authority status '{getattr(authority_effective_status, 'value', authority_effective_status)}' is not EFFECTIVE."
            )
        if closure_status is not None:
            DownstreamUseFirewall.assert_closure_status_allowed(closure_status)

    @staticmethod
    def assert_policy_resolution_allowed(resolution: PolicyResolutionResult) -> None:
        """Enforces: Authority is blocked if policy resolution is not APPLICABLE (Section 24)."""
        if resolution != PolicyResolutionResult.APPLICABLE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: Policy resolution '{resolution.value}' "
                f"prohibits authority effectiveness."
            )

    @staticmethod
    def assert_production_decision_allowed(
        status: AuthorityEffectiveStatus,
        closure_status: Optional[PostAuthorityClosureStatus] = None,
    ) -> None:
        """Enforces: CAN_FEED_PRODUCTION_DECISION = NO unless status == EFFECTIVE and closure not expired/conflict."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_FEED_PRODUCTION_DECISION = NO. "
                f"Current authority status is '{status.value}', required 'EFFECTIVE'."
            )
        if closure_status is not None:
            DownstreamUseFirewall.assert_closure_status_allowed(closure_status)

    @staticmethod
    def assert_empirical_denominator_allowed(
        status: AuthorityEffectiveStatus,
        closure_status: Optional[PostAuthorityClosureStatus] = None,
    ) -> None:
        """Enforces: CAN_ENTER_EMPIRICAL_DENOMINATOR = NO unless status == EFFECTIVE and closure not expired/conflict."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_ENTER_EMPIRICAL_DENOMINATOR = NO. "
                f"Current authority status is '{status.value}', required 'EFFECTIVE'."
            )
        if closure_status is not None:
            DownstreamUseFirewall.assert_closure_status_allowed(closure_status)

    @staticmethod
    def assert_model_tuning_allowed(
        status: AuthorityEffectiveStatus,
        closure_status: Optional[PostAuthorityClosureStatus] = None,
    ) -> None:
        """Enforces: CAN_ENTER_MODEL_TUNING = NO unless status == EFFECTIVE and closure not expired/conflict."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_ENTER_MODEL_TUNING = NO. "
                f"Current authority status is '{status.value}', required 'EFFECTIVE'."
            )
        if closure_status is not None:
            DownstreamUseFirewall.assert_closure_status_allowed(closure_status)

    @staticmethod
    def assert_learning_claim_allowed(
        status: AuthorityEffectiveStatus,
        closure_status: Optional[PostAuthorityClosureStatus] = None,
    ) -> None:
        """Enforces: CAN_BE_USED_FOR_LEARNING_CLAIM = NO unless status == EFFECTIVE and closure not expired/conflict."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_BE_USED_FOR_LEARNING_CLAIM = NO. "
                f"Current authority status is '{status.value}', required 'EFFECTIVE'."
            )
        if closure_status is not None:
            DownstreamUseFirewall.assert_closure_status_allowed(closure_status)

    @staticmethod
    def assert_historical_backfill_allowed(
        status: AuthorityEffectiveStatus,
        closure_status: Optional[PostAuthorityClosureStatus] = None,
    ) -> None:
        """Enforces: CAN_BE_USED_FOR_HISTORICAL_BACKFILL = NO unless status == EFFECTIVE and closure not expired/conflict."""
        if status != AuthorityEffectiveStatus.EFFECTIVE:
            raise DownstreamFirewallBlockedError(
                f"DOWNSTREAM_FIREWALL_BLOCKED: CAN_BE_USED_FOR_HISTORICAL_BACKFILL = NO. "
                f"Current authority status is '{status.value}', required 'EFFECTIVE'."
            )
        if closure_status is not None:
            DownstreamUseFirewall.assert_closure_status_allowed(closure_status)

    @staticmethod
    def assert_retroactive_decision_justification_allowed(status: AuthorityEffectiveStatus) -> None:
        """Enforces: CAN_BE_USED_FOR_RETROACTIVE_DECISION_JUSTIFICATION = NO (ALWAYS PROHIBITED)."""
        raise DownstreamFirewallBlockedError(
            "DOWNSTREAM_FIREWALL_BLOCKED: CAN_BE_USED_FOR_RETROACTIVE_DECISION_JUSTIFICATION = NO. "
            "Retroactive justification of past decisions is strictly prohibited by policy."
        )

    @classmethod
    def evaluate_downstream_permissions(
        cls,
        status: AuthorityEffectiveStatus,
        closure_status: Optional[PostAuthorityClosureStatus] = None,
    ) -> Dict[str, bool]:
        """
        Returns boolean flags representing downstream permissions.
        """
        closure_blocked = closure_status in (PostAuthorityClosureStatus.EXPIRED, PostAuthorityClosureStatus.CONFLICT)
        is_effective = (status == AuthorityEffectiveStatus.EFFECTIVE) and not closure_blocked
        return {
            "can_feed_production_decision": is_effective,
            "can_enter_empirical_denominator": is_effective,
            "can_enter_model_tuning": is_effective,
            "can_be_used_for_learning_claim": is_effective,
            "can_be_used_for_historical_backfill": is_effective,
            "can_be_used_for_retroactive_decision_justification": False,
        }
