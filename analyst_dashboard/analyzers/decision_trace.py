"""ARX Institutional Decision Trace & Explainability Engine (Phase 20D).

Generates a machine-readable, auditable trace answering:
    "Why did the system produce this decision?"
for every ticker and analytical recommendation.
"""

from typing import Dict, Any, Optional, List
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState


class DecisionTraceEngine:
    """Builds full institutional decision traces from underlying analytical pillars."""

    @staticmethod
    def build_decision_trace(
        symbol: str,
        current_price: float,
        candles: List[Dict[str, Any]],
        freshness: Dict[str, Any],
        technicals: Dict[str, Any],
        confluence: Dict[str, Any],
        factor_scores: Dict[str, Any],
        optimal_execution: Dict[str, Any],
        smart_money: Optional[Dict[str, Any]] = None,
        macro_difficulty: Optional[Dict[str, Any]] = None,
        catalyst_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct the comprehensive Decision Trace tree."""
        clean_sym = symbol.upper().strip()
        candle_count = len(candles) if candles else 0
        freshness_status = freshness.get("status", "UNAVAILABLE") if freshness else "UNAVAILABLE"

        has_fundamentals = bool(
            factor_scores and any(factor_scores.get(k) is not None for k in ["qualityScore", "piotroskiFScore", "piotroski_f"])
        )

        confluence_score = float(confluence.get("confluenceScore", 0.0)) if confluence else 0.0
        exec_status = optimal_execution.get("execution_status", "UNKNOWN") if optimal_execution else "UNKNOWN"
        is_in_buy_zone = exec_status in ("IN_BUY_ZONE", "IN_BUY_ZONE_CONFIRMED", "IN_BUY_ZONE_AWAITING_TRIGGER")
        is_confirmed = exec_status in ("IN_BUY_ZONE", "IN_BUY_ZONE_CONFIRMED")
        rr = optimal_execution.get("risk_reward_ratio") if optimal_execution else None

        stage = None
        if technicals and "setup_pattern" in optimal_execution:
            pattern = optimal_execution.get("setup_pattern", "")
            if "Stage 4" in pattern:
                stage = 4
            elif "Stage 2" in pattern:
                stage = 2

        # 1. Resolve State via Canonical Precedence Hierarchy
        decision_state = DecisionHierarchyEngine.resolve_decision_state(
            symbol=clean_sym,
            current_price=current_price,
            candle_count=candle_count,
            freshness_status=freshness_status,
            has_fundamentals=has_fundamentals,
            confluence_score=confluence_score,
            stage_phase=stage,
            is_in_buy_zone=is_in_buy_zone,
            risk_reward_ratio=rr,
            is_cataloged=True,
            is_confirmed=is_confirmed,
        )

        # 2. Extract Pillar Statuses
        pillars = confluence.get("pillars", []) if confluence else []
        tech_pillar = next((p for p in pillars if p.get("pillar") == "TECHNICAL_STRUCTURE"), {})
        fund_pillar = next((p for p in pillars if p.get("pillar") == "FUNDAMENTAL_SOLVENCY"), {})
        smart_pillar = next((p for p in pillars if p.get("pillar") == "SMART_MONEY_FLOW"), {})
        macro_pillar = next((p for p in pillars if p.get("pillar") == "MACRO_SAFETY_FLOOR"), {})

        # 3. Penalties & Warnings
        penalties = confluence.get("warnings", []) if confluence else []

        # 4. Formulate Plain English Explainability Narrative
        state_str = decision_state["state"]
        if state_str == DecisionState.UNVERIFIED.value:
            explanation = "Trade decision withheld: No verified real-time or historical market tape on record."
        elif state_str == DecisionState.INSUFFICIENT_DATA.value:
            explanation = f"Trade decision withheld: Asset has {candle_count} daily sessions, below the 50 sessions required for trend confirmation."
        elif state_str == DecisionState.STALE_DATA.value:
            explanation = "Live orders suspended: Historical data is older than 4 calendar days. Awaiting fresh market tape."
        elif state_str == DecisionState.EVIDENCE_INCOMPLETE.value:
            explanation = "Execution withheld: Core SEC Form 10-Q/10-K financial filings are unverified. Full due diligence required."
        elif state_str == DecisionState.ACTIONABLE_SETUP.value:
            explanation = f"High conviction buy setup: Multi-factor confluence ({confluence_score:.1f}/100) confirmed in buy zone with {rr:.1f}:1 R:R."
        else:
            explanation = f"Valid asset structure: Confluence is {confluence_score:.1f}/100. {decision_state.get('disqualificationReason', 'Awaiting directional breakout.')}"

        return {
            "symbol": clean_sym,
            "decisionState": state_str,
            "decisionStateLabel": decision_state["label"],
            "isActionable": decision_state["isActionable"],
            "canSizeTrade": decision_state["canSizeTrade"],
            "allowedActions": decision_state["allowedActions"],
            "disqualificationReason": decision_state["disqualificationReason"],
            "confluenceScore": confluence_score,
            "trace": {
                "marketData": {
                    "currentPrice": current_price,
                    "freshness": freshness_status,
                    "providerSource": freshness.get("providerSource", "unknown"),
                    "candleCount": candle_count,
                    "lastTradeDate": freshness.get("lastTradeDate"),
                },
                "trend": {
                    "stage": stage or 2,
                    "rsi14": technicals.get("rsi_14") if technicals else None,
                    "ema20": technicals.get("ema_20") if technicals else None,
                    "sma50": technicals.get("sma_50") if technicals else None,
                    "score": tech_pillar.get("score"),
                    "status": tech_pillar.get("status", "neutral"),
                },
                "fundamentals": {
                    "piotroski": factor_scores.get("piotroskiFScore") if factor_scores else None,
                    "qualityScore": factor_scores.get("qualityScore") if factor_scores else None,
                    "growthScore": factor_scores.get("growthScore") if factor_scores else None,
                    "valuationScore": factor_scores.get("valuationScore") if factor_scores else None,
                    "score": fund_pillar.get("score"),
                    "status": fund_pillar.get("status", "unavailable"),
                },
                "smartMoney": {
                    "hasCongressBuy": smart_money.get("has_congress_buy", False) if smart_money else False,
                    "optionsFlow": smart_money.get("optionsFlow", []) if smart_money else [],
                    "score": smart_pillar.get("score"),
                    "status": smart_pillar.get("status", "unavailable"),
                },
                "macro": {
                    "yieldCurve": macro_difficulty.get("yield_curve_10y2y") if isinstance(macro_difficulty, dict) else None,
                    "creditSpread": macro_difficulty.get("high_yield_credit_spread") if isinstance(macro_difficulty, dict) else None,
                    "score": macro_pillar.get("score"),
                    "status": macro_pillar.get("status", "neutral"),
                },
                "executionPlan": {
                    "executionStatus": exec_status,
                    "optimalEntryMin": optimal_execution.get("optimal_entry_min") if optimal_execution else None,
                    "optimalEntryMax": optimal_execution.get("optimal_entry_max") if optimal_execution else None,
                    "stopLoss": optimal_execution.get("stop_loss") if optimal_execution else None,
                    "takeProfit1": optimal_execution.get("take_profit_1") if optimal_execution else None,
                    "riskRewardRatio": rr,
                },
                "penalties": penalties,
            },
            "explanation": explanation,
        }
