"""Multi-Factor Confluence & Dynamic Position Sizing Engine.
Fuses Technical Setups (VCP/ATR), Regulatory Filings (SEC Form 4 / Capitol Hill),
Fundamental Moats (ROIC / PEG), and Catalyst Risk Runways into a unified Confluence Score.
"""

from typing import Dict, Any, Optional
import math


class ConfluenceEngine:
    """Calculates multi-factor trade conviction scores and institutional Kelly risk allocations."""

    @staticmethod
    def calculate_confluence(
        symbol: str,
        technical_data: Optional[Dict[str, Any]] = None,
        smart_money_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        catalyst_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        score = 50.0  # Baseline
        reasons = []
        warnings = []

        # 1. Technical Alignment (Max +25 pts)
        if technical_data:
            exec_status = technical_data.get("executionStatus", "")
            rr = float(technical_data.get("riskRewardRatio", 1.5))
            if exec_status == "IN_BUY_ZONE":
                score += 15.0
                reasons.append("Optimal VCP / ATR Accumulation Discount Active (+15)")
            elif exec_status == "APPROACHING_TARGET":
                score += 5.0
                reasons.append("Target Expansion Momentum Confirmed (+5)")
            
            if rr >= 2.5:
                score += 10.0
                reasons.append(f"Asymmetric Risk-Reward ({rr}:1 R:R) (+10)")
            elif rr >= 1.8:
                score += 5.0

        # 2. Smart Money & Regulatory Filings (Max +25 pts)
        if smart_money_data:
            has_insider_buy = smart_money_data.get("has_insider_buy", False)
            has_congress_buy = smart_money_data.get("has_congress_buy", False)
            if has_insider_buy:
                score += 15.0
                reasons.append("SEC Form 4 C-Suite Insider Accumulation (+15)")
            if has_congress_buy:
                score += 10.0
                reasons.append("Congressional STOCK Act Buy Filing (+10)")

        # 3. Fundamental Quality & Moat (Max +25 pts)
        if fundamental_data:
            roic = float(fundamental_data.get("roic", 20.0))
            peg = float(fundamental_data.get("peg", 1.2))
            piotroski = int(fundamental_data.get("piotroski_f", 7))

            if roic >= 25.0:
                score += 10.0
                reasons.append(f"High ROIC ({roic}%) Moat Compounder (+10)")
            if peg <= 1.0:
                score += 10.0
                reasons.append(f"Peter Lynch GARP Valuation (PEG {peg}) (+10)")
            if piotroski >= 8:
                score += 5.0
                reasons.append(f"Pristine Balance Sheet (F-Score {piotroski}/9) (+5)")

        # 4. Catalyst Runway & Binary Event Risk
        if catalyst_data:
            days_to_earnings = catalyst_data.get("days_to_earnings", 30)
            if days_to_earnings is not None:
                if days_to_earnings < 2:
                    score -= 25.0
                    warnings.append(f"⚠️ HIGH BINARY GAP RISK: Earnings in <{days_to_earnings*24}h! Recommend 50% max sizing.")
                elif days_to_earnings < 7:
                    score -= 10.0
                    warnings.append(f"Caution: Earnings in {days_to_earnings} days.")
                elif days_to_earnings >= 21:
                    score += 5.0
                    reasons.append(f"Safe Catalyst Runway ({days_to_earnings}d to earnings) (+5)")

        score = max(5.0, min(99.0, round(score, 1)))

        if score >= 85.0:
            rating = "⭐ EXCEPTIONAL HIGH-CONVICTION"
            badge_color = "emerald"
        elif score >= 70.0:
            rating = "🔥 HIGH CONFLUENCE"
            badge_color = "cyan"
        elif score >= 50.0:
            rating = "⚖️ MODERATE CONFLUENCE"
            badge_color = "amber"
        else:
            rating = "⚠️ LOW CONFLUENCE / HIGH RISK"
            badge_color = "rose"

        return {
            "symbol": symbol,
            "confluenceScore": score,
            "confluenceRating": rating,
            "badgeColor": badge_color,
            "reasons": reasons,
            "warnings": warnings,
        }

    @staticmethod
    def calculate_position_size(
        account_equity: float,
        risk_pct: float,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        win_rate_est: float = 0.55,
    ) -> Dict[str, Any]:
        """Calculate exact share allocation based on dollar risk constraint & Fractional Kelly."""
        if entry_price <= 0 or stop_loss <= 0 or entry_price <= stop_loss:
            return {
                "error": "Invalid price levels: Entry must be greater than Stop Loss.",
                "shares": 0,
                "position_value": 0,
            }

        risk_per_share = entry_price - stop_loss
        max_dollar_risk = (account_equity * (risk_pct / 100.0))
        shares = math.floor(max_dollar_risk / risk_per_share)
        shares = max(1, shares) if account_equity >= entry_price else 0

        position_value = round(shares * entry_price, 2)
        portfolio_allocation_pct = round((position_value / max(1.0, account_equity)) * 100, 2)
        actual_dollar_risk = round(shares * risk_per_share, 2)

        # Reward projections
        reward_per_share_tp1 = max(0.01, take_profit_1 - entry_price)
        projected_profit_tp1 = round(shares * reward_per_share_tp1, 2)
        rr_actual = round(reward_per_share_tp1 / risk_per_share, 2)

        # Fractional Kelly (Half-Kelly for drawdown protection)
        b_ratio = max(0.5, reward_per_share_tp1 / risk_per_share)
        p = max(0.1, min(0.9, win_rate_est))
        q = 1.0 - p
        full_kelly = max(0.0, (b_ratio * p - q) / b_ratio)
        half_kelly_pct = round((full_kelly / 2.0) * 100, 2)
        recommended_kelly_allocation = min(25.0, half_kelly_pct)  # Institutional 25% max cap per position

        return {
            "accountEquity": account_equity,
            "riskPct": risk_pct,
            "shares": shares,
            "positionValue": position_value,
            "portfolioAllocationPct": portfolio_allocation_pct,
            "actualDollarRisk": actual_dollar_risk,
            "riskPerShare": round(risk_per_share, 2),
            "projectedProfitTp1": projected_profit_tp1,
            "riskRewardRatio": rr_actual,
            "halfKellyOptimalPct": recommended_kelly_allocation,
            "guidance": f"Buy {shares} shares at ${entry_price:.2f}. Total Risk: ${actual_dollar_risk:.2f} ({risk_pct}% of portfolio) with hard stop at ${stop_loss:.2f}.",
        }
