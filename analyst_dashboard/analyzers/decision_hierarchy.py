"""ARX Decision-State Hierarchy & Precedence Engine (Phase 20A).

Establishes the non-negotiable 6-state decision hierarchy with mutually exclusive precedence:
    UNVERIFIED (Precedence 1 - Highest)
        ↓
    INSUFFICIENT_DATA (Precedence 2)
        ↓
    STALE_DATA (Precedence 3)
        ↓
    EVIDENCE_INCOMPLETE (Precedence 4)
        ↓
    VALID_SETUP (Precedence 5)
        ↓
    ACTIONABLE_SETUP (Precedence 6 - Lowest Precedence / Highest Criteria)

Guarantees that contradictory states (e.g. INSUFFICIENT_DATA + ACTIONABLE_SETUP)
are mathematically and structurally impossible.
"""

from enum import Enum
from typing import Dict, Any, Optional, List


class DecisionState(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    STALE_DATA = "STALE_DATA"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    VALID_SETUP = "VALID_SETUP"
    ACTIONABLE_SETUP = "ACTIONABLE_SETUP"


class DecisionHierarchyEngine:
    """Deterministic, pure evaluator of institutional decision states."""

    @staticmethod
    def normalize_stage(stage_val: Optional[Any]) -> Optional[int]:
        """Normalize integer, string, or descriptive stage phases into standard Minervini 1-4 integers."""
        if stage_val is None:
            return None
        if isinstance(stage_val, int):
            return stage_val
        if isinstance(stage_val, str):
            clean = stage_val.strip().lower()
            if "stage 4" in clean or "markdown" in clean or "correction" in clean:
                return 4
            elif "stage 2" in clean or "advancing" in clean or "growth" in clean or "vcp" in clean:
                return 2
            elif "stage 1" in clean or "basing" in clean or "accumulation" in clean:
                return 1
            elif "stage 3" in clean or "topping" in clean or "distribution" in clean:
                return 3
            elif clean.isdigit():
                return int(clean)
        return None

    @staticmethod
    def is_valid_intraday_stage(stage_val: Any) -> bool:
        """Verify if stage represents confirmed intraday momentum trend expansion."""
        if not stage_val or not isinstance(stage_val, str):
            return False
        clean = stage_val.strip().lower()
        return "intraday momentum" in clean or "trend expansion" in clean

    @staticmethod
    def resolve_decision_state(
        symbol: str,
        current_price: float,
        candle_count: int,
        freshness_status: str,
        has_fundamentals: bool,
        confluence_score: float,
        stage_phase: Optional[Any] = None,
        is_in_buy_zone: bool = False,
        risk_reward_ratio: Optional[float] = None,
        is_cataloged: bool = True,
        is_confirmed: bool = True,
        user_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve the active decision state and execution eligibility following strict precedence."""
        clean_sym = symbol.upper().strip()

        # ── Precedence 1: UNVERIFIED ──────────────────────────────────────────
        if current_price <= 0 or candle_count == 0 or not is_cataloged or freshness_status == "UNAVAILABLE":
            return {
                "symbol": clean_sym,
                "state": DecisionState.UNVERIFIED.value,
                "label": "Unverified Asset — Disclosures Required",
                "isActionable": False,
                "canSizeTrade": False,
                "allowedActions": ["RESEARCH_PROFILE"],
                "disqualificationReason": "No verified real-time or historical exchange tape on record.",
            }

        # ── Precedence 2: INSUFFICIENT_DATA ───────────────────────────────────
        if candle_count < 50:
            return {
                "symbol": clean_sym,
                "state": DecisionState.INSUFFICIENT_DATA.value,
                "label": f"Insufficient History ({candle_count}/50 Sessions)",
                "isActionable": False,
                "canSizeTrade": False,
                "allowedActions": ["RESEARCH_PROFILE", "ADD_WATCHLIST"],
                "disqualificationReason": f"Requires minimum 50 daily trading sessions for trend validation; {candle_count} provided.",
            }

        # ── Precedence 3: STALE_DATA ──────────────────────────────────────────
        if freshness_status == "STALE_HISTORICAL":
            return {
                "symbol": clean_sym,
                "state": DecisionState.STALE_DATA.value,
                "label": "Stale Historical Tape (> 4 Days)",
                "isActionable": False,
                "canSizeTrade": False,
                "allowedActions": ["RESEARCH_PROFILE", "ADD_WATCHLIST"],
                "disqualificationReason": "Market data is historical/stale; live trade triggers are suspended.",
            }

        # ── Precedence 4: EVIDENCE_INCOMPLETE ─────────────────────────────────
        if not has_fundamentals:
            return {
                "symbol": clean_sym,
                "state": DecisionState.EVIDENCE_INCOMPLETE.value,
                "label": "Evidence Incomplete — Fundamentals Missing",
                "isActionable": False,
                "canSizeTrade": False,
                "allowedActions": ["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"],
                "disqualificationReason": "Audited SEC EDGAR 10-K/10-Q financial filings are unverified.",
            }

        # ── Precedence 6: ACTIONABLE_SETUP (Highest criteria) ─────────────────
        # Requires: Full evidence + Horizon-specific trend qualification + Confluence >= 75 + in buy zone + confirmed trigger + R:R >= 2.0
        norm_stage = DecisionHierarchyEngine.normalize_stage(stage_phase)
        is_intraday_expansion = DecisionHierarchyEngine.is_valid_intraday_stage(stage_phase)
        clean_role = user_role.upper().strip() if isinstance(user_role, str) else None

        # Explicit horizon-specific eligibility:
        # Day Trader evaluates confirmed intraday momentum trend expansion or Stage 2.
        # Swing / Long-term strictly requires Minervini Stage 2 advancing growth phase.
        is_day_trader = (clean_role == "DAY_TRADER") or (clean_role is None and is_intraday_expansion)
        if is_day_trader:
            is_stage_eligible = is_intraday_expansion or (norm_stage == 2)
        else:
            is_stage_eligible = (norm_stage == 2)

        rr = risk_reward_ratio if risk_reward_ratio is not None else 0.0
        if (
            confluence_score >= 75.0
            and is_in_buy_zone
            and is_confirmed
            and is_stage_eligible
            and rr >= 2.0
        ):
            return {
                "symbol": clean_sym,
                "state": DecisionState.ACTIONABLE_SETUP.value,
                "label": "Actionable Setup — Buy Zone Confirmed",
                "isActionable": True,
                "canSizeTrade": True,
                "allowedActions": ["SIZE_TRADE", "SET_ALERT", "REVIEW_THESIS", "ADD_WATCHLIST"],
                "disqualificationReason": None,
            }

        # ── Precedence 5: VALID_SETUP (Default when verified data is sound) ───
        # Sound verified data, but currently awaiting breakout, in Stage 4, or outside buy zone
        reason = "Awaiting volume breakout confirmation."
        if norm_stage == 4:
            reason = "Stage 4 distribution: price below 50-day SMA; wait for floor formation."
        elif norm_stage == 1:
            reason = "Stage 1 structural basing phase: price establishing floor; awaiting Stage 2 breakout."
        elif norm_stage == 3:
            reason = "Stage 3 distribution phase: topping pattern detected; protect capital."
        elif is_day_trader and not is_stage_eligible:
            reason = "Intraday momentum unconfirmed: Active intraday trend expansion required for trade approval."
        elif not is_day_trader and norm_stage is None:
            reason = "Stage unconfirmed: Minervini Stage 2 advancing growth phase required for trade approval."
        elif not is_in_buy_zone:
            reason = "Price is outside the optimal entry corridor; awaiting pullback to buy zone."
        elif not is_confirmed:
            reason = "Price is in optimal buy zone; awaiting reversal/stabilization confirmation candle."
        elif rr < 2.0:
            reason = f"Risk/Reward ratio ({rr:.1f}:1) is below the institutional 2.0:1 minimum threshold."
        elif confluence_score < 75.0:
            reason = f"Confluence conviction ({confluence_score:.1f}/100) is below the actionable 75.0 floor."

        return {
            "symbol": clean_sym,
            "state": DecisionState.VALID_SETUP.value,
            "label": "Valid Setup — Awaiting Trigger",
            "isActionable": False,
            "canSizeTrade": False,
            "allowedActions": ["SET_ALERT", "ADD_WATCHLIST", "RESEARCH_PROFILE"],
            "disqualificationReason": reason,
        }
