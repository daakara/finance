"""
Scope-Safe Smart Money Product State Composition for ARX Terminal.
Adheres strictly to Sections 42-52:
- Formal state tuple: (SCOPE, PIPELINE, COMPLETENESS, FRESHNESS, GAP, SCAN_RESULT, DECISION_ELIGIBILITY)
- Prohibits ambiguous global booleans (available, fresh, no_matches) without explicit scope qualification.
- Enforces Market-Wide Readiness Theorem: MARKET_WIDE_CAPABILITY_STATE is strictly NOT_AUTHORIZED
  whenever SCOPE_KIND == GOVERNED_SUBSET.
"""

from typing import Dict, Any, Optional

from analyst_dashboard.smart_money.certification_models import (
    ScopeKind,
    PipelineState,
    CompletenessState,
    FreshnessState,
    ScopedGapState,
    ScanResultState,
    DecisionEligibilityState,
    ScopedCapabilityState,
    MarketWideCapabilityState,
    SmartMoneyCompositeState,
)
from analyst_dashboard.smart_money.partition_set_authority import (
    GOVERNED_UNIVERSE_BUILD_ID,
    REQUIRED_PARTITION_COUNT,
)


class SmartMoneyStateComposer:
    """
    Evaluates and composes the formal scope-bound Smart Money product state.
    """

    @classmethod
    def compose_state(
        cls,
        pipeline_state: PipelineState,
        completeness_state: CompletenessState,
        freshness_state: FreshnessState,
        gap_state: ScopedGapState,
        scan_result_state: ScanResultState,
        certified_through: Optional[str] = None,
        certification_receipt_id: Optional[str] = None,
        active_breach_count: int = 0,
        open_gap_count: int = 0,
        qualifying_matches: Optional[int] = None,
        scope_kind: ScopeKind = ScopeKind.GOVERNED_SUBSET,
        market_wide_classification_coverage: str = "INCOMPLETE",
        market_wide_authority: str = "NOT_ESTABLISHED",
    ) -> SmartMoneyCompositeState:
        """
        Builds the formal composite state while enforcing all scope and readiness invariants.
        """
        # Scope Definition
        scope_info = {
            "scope_kind": scope_kind.value,
            "universe_build_id": GOVERNED_UNIVERSE_BUILD_ID,
            "required_partition_count": REQUIRED_PARTITION_COUNT,
            "market_wide_classification_coverage": market_wide_classification_coverage,
            "market_wide_authority": market_wide_authority,
        }

        # Invariant: Integrity breach forces freshness to UNKNOWN and decision to NOT_ELIGIBLE
        effective_freshness = freshness_state
        if completeness_state == CompletenessState.INTEGRITY_BREACH or active_breach_count > 0:
            effective_freshness = FreshnessState.UNKNOWN

        # Invariant: Open gaps or incomplete scans force matched_count to None
        effective_matches = qualifying_matches
        if gap_state != ScopedGapState.NONE or open_gap_count > 0 or completeness_state != CompletenessState.CERTIFIED:
            effective_matches = None

        # Derive Scoped Capability State
        if pipeline_state in (PipelineState.NOT_INITIALIZED, PipelineState.BOOTSTRAPPING):
            scoped_cap = ScopedCapabilityState.PIPELINE_PENDING
        elif completeness_state in (CompletenessState.BLOCKED_BY_GAPS, CompletenessState.INTEGRITY_BREACH) or gap_state != ScopedGapState.NONE:
            scoped_cap = ScopedCapabilityState.DATA_INCOMPLETE
        elif effective_freshness == FreshnessState.STALE:
            scoped_cap = ScopedCapabilityState.STALE
        elif effective_freshness == FreshnessState.EXPIRED or effective_freshness == FreshnessState.UNKNOWN:
            scoped_cap = ScopedCapabilityState.UNAVAILABLE
        elif completeness_state == CompletenessState.CERTIFIED and effective_freshness == FreshnessState.FRESH:
            scoped_cap = ScopedCapabilityState.AVAILABLE_WITHIN_SCOPE
        else:
            scoped_cap = ScopedCapabilityState.UNAVAILABLE

        # Enforce Market-Wide Readiness Theorem:
        # MARKET_WIDE_CAPABILITY_STATE may equal AVAILABLE only if:
        # scope_kind == MARKET_WIDE_AUTHORITATIVE AND market_wide_classification_coverage == COMPLETE
        # AND market_wide_authority == ESTABLISHED AND pipeline == OPERATIONAL AND completeness == CERTIFIED ...
        if (
            scope_kind == ScopeKind.MARKET_WIDE_AUTHORITATIVE
            and market_wide_classification_coverage == "COMPLETE"
            and market_wide_authority == "ESTABLISHED"
            and pipeline_state == PipelineState.OPERATIONAL
            and completeness_state == CompletenessState.CERTIFIED
            and effective_freshness == FreshnessState.FRESH
            and gap_state == ScopedGapState.NONE
        ):
            market_wide_cap = MarketWideCapabilityState.AVAILABLE
        else:
            market_wide_cap = MarketWideCapabilityState.NOT_AUTHORIZED

        # Derive Decision Eligibility
        if (
            pipeline_state == PipelineState.OPERATIONAL
            and completeness_state == CompletenessState.CERTIFIED
            and effective_freshness == FreshnessState.FRESH
            and gap_state == ScopedGapState.NONE
            and scan_result_state != ScanResultState.NOT_EVALUATED
            and certification_receipt_id is not None
            and active_breach_count == 0
            and open_gap_count == 0
        ):
            decision_elig = DecisionEligibilityState.ELIGIBLE_WITHIN_SCOPE
        else:
            decision_elig = DecisionEligibilityState.NOT_ELIGIBLE

        return SmartMoneyCompositeState(
            scope=scope_info,
            pipeline=pipeline_state.value,
            completeness=completeness_state.value,
            freshness=effective_freshness.value,
            gaps=gap_state.value,
            scan_result=scan_result_state.value,
            decision_eligibility=decision_elig.value,
            scoped_capability=scoped_cap.value,
            market_wide_capability=market_wide_cap.value,
            certified_through=certified_through,
            certification_receipt_id=certification_receipt_id,
            active_breach_count=active_breach_count,
            open_gap_count=open_gap_count,
            matched_count=effective_matches,
        )

    @classmethod
    def reject_ambiguous_global_booleans(cls, payload: Dict[str, Any]):
        """
        Enforces Section 51 invariant:
        GLOBAL_AVAILABLE_BOOLEAN = PROHIBITED
        GLOBAL_FRESH_BOOLEAN = PROHIBITED
        GLOBAL_NO_MATCHES_BOOLEAN = PROHIBITED
        """
        prohibited_keys = {"available", "fresh", "no_matches", "is_available", "is_fresh"}
        for k in prohibited_keys:
            if k in payload:
                raise ValueError(
                    f"PROHIBITED_GLOBAL_BOOLEAN: Key '{k}' is prohibited. Must use scope-qualified state."
                )
