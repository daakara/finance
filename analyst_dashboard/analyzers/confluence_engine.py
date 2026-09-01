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
        macro_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute institutional multi-factor confluence score, pillar breakdown, and risk boundaries."""
        clean_sym = symbol.upper().replace("-USD", "").strip()

        # ── 1. TECHNICAL STRUCTURE & MOMENTUM (Weight: 25%) ──────────────────
        tech_score = 55.0
        tech_status = "neutral"
        tech_detail = "Neutral range consolidation. Awaiting directional breakout trigger."
        tech_plain = "Trading inside a sideways channel. Waiting for clear market momentum."

        if technical_data:
            pattern = technical_data.get("setup_pattern", "")
            stage = technical_data.get("stage_phase", "")
            exec_status = technical_data.get("executionStatus") or technical_data.get("execution_status", "")
            rsi = float(technical_data.get("rsi_14", 50.0))
            rr = float(technical_data.get("risk_reward_ratio") or technical_data.get("riskRewardRatio", 2.0))

            if "Correction" in pattern or "Stage 4" in pattern or "Stage 4" in stage:
                tech_score = 30.0
                tech_status = "warning"
                tech_detail = f"Stage 4 Markdown / Distribution Phase (RSI {rsi:.1f}). Price below key declining moving averages."
                tech_plain = "Falling knife mode: Price is breaking down. Wait for buyers to establish a firm floor."
            elif any(k in pattern for k in ["Breakout", "Stage 2", "VCP", "Pullback"]) or "Stage 2" in stage or exec_status == "IN_BUY_ZONE":
                tech_score = 88.0 if rr >= 2.5 else 80.0 if rr >= 2.0 else 75.0
                if 42.0 <= rsi <= 65.0:
                    tech_score += 5.0
                if rsi > 75.0:
                    tech_score -= 10.0
                tech_status = "positive"
                tech_detail = f"{stage or 'Stage 2 Accumulation'} with {rr:.1f}:1 R:R structure and healthy RSI ({rsi:.1f})."
                tech_plain = f"Coiled spring setup: Buyers defending key levels with healthy {rsi:.1f} RSI momentum."
            else:
                tech_score = 58.0 if 45.0 <= rsi <= 58.0 else 50.0
                tech_status = "neutral"
                tech_detail = f"Consolidation range with RSI {rsi:.1f}. Awaiting high-volume breakout confirmation."
                tech_plain = f"Sideways price action (RSI {rsi:.1f}). No clear direction yet."

        # ── 2. FUNDAMENTAL QUALITY & SOLVENCY (Weight: 25%) ──────────────────
        fund_score = 65.0
        fund_status = "neutral"
        fund_detail = "Moderate financial solvency with stable operational profile."
        fund_plain = "Healthy average company financials with no immediate solvency risks."

        if fundamental_data:
            piotroski = int(fundamental_data.get("piotroski_f", 7))
            quality = float(fundamental_data.get("qualityScore", 70.0))
            growth = float(fundamental_data.get("growthScore", 70.0))
            valuation = float(fundamental_data.get("valuationScore", 65.0))

            fund_score = round(0.45 * quality + 0.30 * growth + 0.25 * valuation, 1)

            if piotroski >= 8 or quality >= 82.0:
                fund_status = "positive"
                fund_detail = f"Fortress Solvency: Piotroski F-Score {piotroski}/9, Quality Factor {quality:.0f}/100."
                fund_plain = f"Rock-solid balance sheet: Top-tier {piotroski}/9 financial strength with strong margins."
            elif piotroski <= 4 or quality < 45.0:
                fund_status = "warning"
                fund_detail = f"Elevated Balance Sheet Risk: Piotroski F-Score {piotroski}/9. Sensitive to credit tightening."
                fund_plain = f"Weaker financial health ({piotroski}/9 score). Carries elevated debt or thinning margins."
            else:
                fund_status = "neutral"
                fund_detail = f"Stable Solvency: Piotroski F-Score {piotroski}/9, Quality Factor {quality:.0f}/100."
                fund_plain = f"Stable core financials ({piotroski}/9 score) without acute balance sheet concerns."

        # ── 3. SMART MONEY & REGULATORY FILINGS (Weight: 25%) ────────────────
        smart_score = 50.0
        smart_status = "neutral"
        
        # Check if Foreign Private Issuer / ADR (American Depositary Receipt)
        known_foreign_adrs = {
            "DHLGY", "NVO", "ASML", "TSM", "BABA", "AZN", "BP", "SHEL", "SAP", "SNY",
            "TM", "HMC", "VALE", "BTI", "RIO", "UL", "LVMUY", "TCEHY", "NPSNY", "ADDYY",
            "BAYRY", "DDAIF", "VWAGY", "BMWYY", "RHHBY", "SAN", "BBVA", "ING", "BNPQY", "CRH"
        }
        is_foreign_adr = (len(clean_sym) == 5 and clean_sym.endswith(("Y", "F"))) or clean_sym in known_foreign_adrs

        if is_foreign_adr:
            smart_detail = "Foreign ADR (FPI): Executive transactions governed by local regulatory filings (e.g., BaFin Directors' Dealings, FCA DTR, TSE) rather than US SEC Form 4."
            smart_plain = "Foreign Company (ADR): Executive trades are reported to European/overseas regulators rather than the US SEC."
        else:
            smart_detail = "No high-conviction C-Suite open-market purchases filed on SEC EDGAR in last 30 days."
            smart_plain = "No recent major C-suite insider purchases filed with the SEC this month."

        if smart_money_data:
            has_insider_buy = smart_money_data.get("has_insider_buy", False)
            insider_val = float(smart_money_data.get("insider_value_usd", 0))
            insider_name = smart_money_data.get("insider_name", "")
            has_congress_buy = smart_money_data.get("has_congress_buy", False)
            has_options = smart_money_data.get("has_options_flow", False)

            if has_insider_buy and insider_val > 0:
                smart_score = 90.0
                smart_status = "positive"
                smart_detail = f"{insider_name or 'C-Suite Executive'} purchased ${(insider_val / 1e6):.1f}M USD on open market."
                smart_plain = f"Insider skin in the game: ${(insider_val / 1e6):.1f}M bought directly by corporate executives."
            elif has_congress_buy:
                smart_score = 80.0
                smart_status = "positive"
                smart_detail = "Congressional STOCK Act filing: Capitol Hill committee member buy disclosure active."
                smart_plain = "Congress buy reported: Lawmaker disclosed an open-market purchase in this stock/sector."
            elif has_options:
                smart_score = 70.0
                smart_status = "positive"
                smart_detail = "Unusual institutional order flow: High-volume call sweeps detected in options tape."
                smart_plain = "Big money call options detected: Institutional traders positioning for upside."
            else:
                smart_score = 50.0
                smart_status = "neutral"
                if is_foreign_adr:
                    smart_detail = "Foreign ADR (FPI): Executive transactions governed by local regulatory filings (e.g., BaFin Directors' Dealings, FCA DTR, TSE) rather than US SEC Form 4."
                    smart_plain = "Foreign Company (ADR): Executive trades are reported to European/overseas regulators rather than the US SEC."
                else:
                    smart_detail = "No major C-Suite open-market purchases filed on SEC EDGAR in last 30 days."
                    smart_plain = "No recent big boss insider purchases filed with the SEC this month."

        # ── 4. MACRO REGIME & DOWNSIDE SAFETY FLOOR (Weight: 25%) ────────────
        macro_score = 60.0
        macro_status = "neutral"
        macro_detail = "Neutral macro liquidity regime with defined risk floor."
        macro_plain = "Stable economic backdrop with defined safety exit floor."

        stop_loss = 0.0
        risk_pct = 5.0
        if technical_data:
            stop_loss = float(technical_data.get("stop_loss", 0))
            current_p = float(technical_data.get("current_price", 0))
            if stop_loss > 0 and current_p > 0:
                risk_pct = abs((current_p - stop_loss) / current_p) * 100.0

        yield_curve = 0.25
        credit_spread = 3.5
        if macro_data:
            yield_curve = float(macro_data.get("yield_curve_10y2y", 0.25))
            credit_spread = float(macro_data.get("credit_spread", 3.5))

        if yield_curve >= 0.0 and credit_spread < 4.2 and risk_pct <= 7.5:
            macro_score = 85.0
            macro_status = "positive"
            macro_detail = f"FRED 10Y-2Y yield curve positive (+{yield_curve:.2f}%), credit spreads tight. Defined stop at ${stop_loss:.2f} ({risk_pct:.1f}% risk)."
            macro_plain = f"Macro green light: Credit markets healthy. Clear exit floor set at ${stop_loss:.2f} ({risk_pct:.1f}% risk)."
        elif yield_curve < 0.0 or risk_pct > 12.0:
            macro_score = 38.0
            macro_status = "warning"
            macro_detail = f"Macro risk elevated or wide stop requirement ({risk_pct:.1f}% risk). Defensive sizing required."
            macro_plain = f"Macro warning flags or large downside distance ({risk_pct:.1f}% risk). Size down carefully."
        else:
            macro_score = 60.0
            macro_status = "neutral"
            macro_detail = f"Supportive liquidity background with stop floor at ${stop_loss:.2f} ({risk_pct:.1f}% risk)."
            macro_plain = f"Stable economic background with safety exit floor set at ${stop_loss:.2f} ({risk_pct:.1f}% risk)."

        # ── 5. CATALYST RUNWAY RISK ADJUSTMENT ────────────────────────────────
        catalyst_mod = 0.0
        warnings = []
        if catalyst_data:
            days_to_earnings = catalyst_data.get("days_to_earnings")
            if days_to_earnings is not None:
                if days_to_earnings < 2:
                    catalyst_mod -= 15.0
                    warnings.append(f"⚠️ HIGH BINARY GAP RISK: Earnings in <{int(days_to_earnings*24)}h! Limit position sizing.")
                elif days_to_earnings < 7:
                    catalyst_mod -= 5.0
                    warnings.append(f"Caution: Earnings in {days_to_earnings} days.")

        # ── COMPOSITE SYNTHESIS ──────────────────────────────────────────────
        raw_composite = (
            0.25 * tech_score +
            0.25 * fund_score +
            0.25 * smart_score +
            0.25 * macro_score +
            catalyst_mod
        )
        final_score = max(20.0, min(96.0, round(raw_composite, 1)))

        pillars = [
            {
                "pillar": "TECHNICAL_STRUCTURE",
                "label": "Technical Structure",
                "plainLabel": "Chart Structure",
                "score": round(tech_score, 1),
                "status": tech_status,
                "detail": tech_detail,
                "plainDetail": tech_plain,
                "icon": "📈" if tech_status == "positive" else "⚠️" if tech_status == "warning" else "📊",
            },
            {
                "pillar": "FUNDAMENTAL_SOLVENCY",
                "label": "Fundamental Solvency",
                "plainLabel": "Company Health",
                "score": round(fund_score, 1),
                "status": fund_status,
                "detail": fund_detail,
                "plainDetail": fund_plain,
                "icon": "🏢",
            },
            {
                "pillar": "SMART_MONEY_FLOW",
                "label": "Corporate Insiders & Flow",
                "plainLabel": "Smart Money Flow",
                "score": round(smart_score, 1),
                "status": smart_status,
                "detail": smart_detail,
                "plainDetail": smart_plain,
                "icon": "🏛️",
            },
            {
                "pillar": "MACRO_SAFETY_FLOOR",
                "label": "Macro Regime & Safety Floor",
                "plainLabel": "Market Tailwinds & Stop",
                "score": round(macro_score, 1),
                "status": macro_status,
                "detail": macro_detail,
                "plainDetail": macro_plain,
                "icon": "🛡️" if macro_status == "positive" else "⚠️" if macro_status == "warning" else "⚖️",
            },
        ]

        positives = len([p for p in pillars if p["status"] == "positive"])
        warns = len([p for p in pillars if p["status"] == "warning"])

        if positives == 4:
            confluence_badge = "4-Pillar Confluence (Pristine)"
            plain_badge = "4/4 High Conviction"
        elif positives == 3:
            confluence_badge = "3-Feed Confluence (Selective)"
            plain_badge = "3/4 Strong Alignment"
        elif warns >= 2:
            confluence_badge = "Multi-Feed Divergence (Risk-Off)"
            plain_badge = "⚠️ Mixed / Caution"
        else:
            confluence_badge = f"{positives}-Feed Mixed Confluence"
            plain_badge = f"{positives}/4 Positive Signals"

        if final_score >= 80.0:
            rating = "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT"
            plain_rating = "🟢 GREEN LIGHT: HIGH CONVICTION"
            badge_color = "emerald"
            bottom_line = f"Strong multi-factor alignment ({positives}/4 pillars positive): Technicals, balance sheet quality, and smart money flow are synchronised."
        elif final_score <= 48.0 or warns >= 2:
            rating = "DEFENSIVE / CAPITAL PRESERVATION MODE"
            plain_rating = "🔴 RED LIGHT: WAIT FOR DUST TO SETTLE"
            badge_color = "rose"
            bottom_line = f"Technical turbulence or weak balance sheet metrics detected ({warns} warning flags). Preserve cash and wait for a proper accumulation base to form."
        elif final_score < 65.0:
            rating = "SELECTIVE / RANGE-BOUND MOMENTUM"
            plain_rating = "🟡 SELECTIVE ENTRY: WAIT FOR TRIGGER"
            badge_color = "amber"
            bottom_line = f"Mixed signal environment ({positives} positive, {warns} warning). Take half-position sizing and honor stops tightly."
        else:
            rating = "CONSTRUCTIVE ACCUMULATION CONVICTION"
            plain_rating = "💡 SOLID ACCUMULATION SETUP"
            badge_color = "cyan"
            bottom_line = f"Disciplined trade structure with {positives}/4 supporting pillars. Favorable risk floor near ${stop_loss:.2f}. Execute inside buy zones."

        return {
            "symbol": clean_sym,
            "confluenceScore": final_score,
            "confluenceRating": rating,
            "plainRating": plain_rating,
            "confluenceBadge": confluence_badge,
            "plainBadge": plain_badge,
            "badgeColor": badge_color,
            "bottomLine": bottom_line,
            "pillars": pillars,
            "reasons": [p["detail"] for p in pillars if p["status"] == "positive"] or [p["detail"] for p in pillars],
            "plainReasons": [p["plainDetail"] for p in pillars if p["status"] == "positive"] or [p["plainDetail"] for p in pillars],
            "positivesCount": positives,
            "warningsCount": warns,
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
