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
    def resolve_decision_state(
        symbol: str,
        current_price: float,
        candle_count: int,
        freshness_status: str,
        has_fundamentals: bool,
        confluence_score: float,
        stage_phase: Optional[int] = None,
        is_in_buy_zone: bool = False,
        risk_reward_ratio: Optional[float] = None,
        is_cataloged: bool = True,
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
        # Requires: Full evidence + Stage 2 accumulation + Confluence >= 75 + in buy zone + R:R >= 2.0
        rr = risk_reward_ratio if risk_reward_ratio is not None else 0.0
        is_stage_2 = stage_phase == 2 or stage_phase is None
        if (
            confluence_score >= 75.0
            and is_in_buy_zone
            and is_stage_2
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
        if stage_phase == 4:
            reason = "Stage 4 distribution: price below 50-day SMA; wait for floor formation."
        elif not is_in_buy_zone:
            reason = "Price is outside the optimal entry corridor; awaiting pullback to buy zone."
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
