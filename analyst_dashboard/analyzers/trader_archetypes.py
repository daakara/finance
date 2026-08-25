"""Elite Trader Strategy Models & Smart-Money Quantitative Analysis Engine."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class TraderArchetypeAnalyzer:
    """Analyzes any asset against proven institutional and iconic trader mental models."""

    CONGRESSIONAL_POLICY_TICKERS = {
        "NVDA": "Direct beneficiary of federal CHIPS Act subsidies and hyperscale AI computing demand.",
        "AAPL": "High consumer hardware loyalty and strong enterprise services ecosystem.",
        "MSFT": "Key provider of US defense cloud infrastructure and corporate AI software.",
        "GOOGL": "Major defense and government cloud computing partner.",
        "TSLA": "Supported by Inflation Reduction Act clean energy and battery tax credits.",
        "PLTR": "Primary contractor for US Department of Defense data and battlefield AI systems.",
        "CRWD": "Essential beneficiary of federal cybersecurity mandates for government and enterprise.",
        "ENPH": "Boosted by federal solar investment tax credits and clean power incentives.",
        "BTC-USD": "Supported by legislative discussions on national digital asset reserves.",
        "ETH-USD": "Approved for US spot exchange-traded funds (ETFs) and institutional settlement.",
        "SOL-USD": "High-speed blockchain adoption for global financial payment networks.",
    }

    def analyze_asset(
        self,
        symbol: str,
        info: Dict[str, Any],
        price_df: pd.DataFrame,
        risk_metrics: Dict[str, Any],
        macro_indicators: Dict[str, Any],
        factor_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all 4 iconic trader archetype models against the asset."""
        sym_clean = symbol.upper().replace("-USD", "")
        is_crypto = "-USD" in symbol.upper() or sym_clean in ["BTC", "ETH", "SOL"]

        # 1. The Oracle (Warren Buffett / Berkshire Hathaway)
        buffett = self._evaluate_buffett_moat(sym_clean, is_crypto, info, factor_scores)

        # 2. The Capitol Whale (Nancy Pelosi / Congressional Policy Flows)
        pelosi = self._evaluate_congressional_whale(symbol.upper(), factor_scores, is_crypto)

        # 3. The Macro Sorcerer (Stanley Druckenmiller / Macro Trends)
        druckenmiller = self._evaluate_druckenmiller_macro(macro_indicators, factor_scores, price_df)

        # 4. The Medallion Quant (Jim Simons / Quantitative Risk)
        simons = self._evaluate_simons_quant(risk_metrics, price_df, factor_scores)

        archetypes = [buffett, pelosi, druckenmiller, simons]
        consensus_score = round(sum(a["alignmentScore"] for a in archetypes) / len(archetypes))

        if consensus_score >= 85:
            verdict = "Strong Buy / Core Accumulation"
        elif consensus_score >= 75:
            verdict = "Favorable Multi-Strategy Buy"
        elif consensus_score >= 65:
            verdict = "Moderate Growth Hold"
        else:
            verdict = "High Volatility Speculative"

        return {
            "consensusScore": consensus_score,
            "verdict": verdict,
            "archetypes": archetypes,
        }

    def _evaluate_buffett_moat(
        self, sym: str, is_crypto: bool, info: Dict[str, Any], factor_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Warren Buffett Value, Quality, Moat & Free Cash Flow model."""
        if is_crypto:
            return {
                "name": "Warren Buffett (Value & Moat)",
                "archetype": "High Cash Flow & Wide Moats",
                "alignmentScore": 30,
                "status": "Not Buffett Style",
                "thesis": "Buffett avoids cryptocurrencies because they do not produce cash flows or physical goods.",
                "catalyst": "Prefers companies with strong pricing power and predictable cash dividends.",
            }

        quality = factor_scores.get("qualityScore", 80)
        valuation = factor_scores.get("valuationScore", 75)
        piotroski = factor_scores.get("piotroskiFScore", 8)

        score = min(96, max(40, int(quality * 0.45 + valuation * 0.35 + (piotroski * 10) * 0.20)))
        if sym in ["AAPL", "BAC", "KO", "AXP", "OXY"]:
            score = max(score, 92)

        return {
            "name": "Warren Buffett (Value & Moat)",
            "archetype": "High Cash Flow & Wide Moats",
            "alignmentScore": score,
            "status": "High Moat Alignment" if score >= 80 else "Moderate Moat",
            "thesis": "High cash generation with strong pricing power, low corporate debt, and consistent share buybacks.",
            "catalyst": "Durable competitive advantage and steady profit margins across economic cycles.",
        }

    def _evaluate_congressional_whale(
        self, symbol: str, factor_scores: Dict[str, Any], is_crypto: bool
    ) -> Dict[str, Any]:
        """Nancy Pelosi / Congressional Policy Catalyst model."""
        policy_catalyst = self.CONGRESSIONAL_POLICY_TICKERS.get(
            symbol,
            self.CONGRESSIONAL_POLICY_TICKERS.get(symbol.replace("-USD", ""), None),
        )

        momentum = factor_scores.get("momentumScore", 70)
        growth = factor_scores.get("growthScore", 75)

        if policy_catalyst:
            score = min(98, max(75, int(82 + (momentum * 0.1) + (growth * 0.08))))
            return {
                "name": "Nancy Pelosi (Policy & Government Catalysts)",
                "archetype": "Government Spending & High-Conviction Tech",
                "alignmentScore": score,
                "status": "Strong Policy Support",
                "thesis": policy_catalyst,
                "catalyst": "Beneficiary of federal industrial policy, technology subsidies, and government contracts.",
            }

        score = min(82, max(45, int((growth * 0.6) + (momentum * 0.4))))
        return {
            "name": "Nancy Pelosi (Policy & Government Catalysts)",
            "archetype": "Government Spending & High-Conviction Tech",
            "alignmentScore": score,
            "status": "Neutral Policy Exposure",
            "thesis": "Moderate policy alignment without major direct federal government spending programs.",
            "catalyst": "General growth in business technology adoption.",
        }

    def _evaluate_druckenmiller_macro(
        self, macro_indicators: Dict[str, Any], factor_scores: Dict[str, Any], price_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Stanley Druckenmiller / Macro Trends & Reflexivity model."""
        yield_curve = macro_indicators.get("yield_curve_spread", 0.47)
        credit_spread = macro_indicators.get("credit_spread_oas", 2.69)
        momentum = factor_scores.get("momentumScore", 75)
        growth = factor_scores.get("growthScore", 75)

        base = 70
        if yield_curve > 0.20:
            base += 12
        if credit_spread < 3.2:
            base += 8

        score = min(97, max(45, int(base * 0.5 + momentum * 0.3 + growth * 0.2)))

        return {
            "name": "Stanley Druckenmiller (Macro Trends)",
            "archetype": "Interest Rate Trends & Market Momentum",
            "alignmentScore": score,
            "status": "Positive Macro Trend" if score >= 80 else "Neutral Macro",
            "thesis": "The lower interest rate environment and upward price momentum favor holding this asset.",
            "catalyst": "Central bank rate cuts and strong institutional buying momentum.",
        }

    def _evaluate_simons_quant(
        self, risk_metrics: Dict[str, Any], price_df: pd.DataFrame, factor_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Jim Simons / Renaissance Quantitative Risk model."""
        sortino = risk_metrics.get("Sortino_Ratio", 1.84)
        skew = risk_metrics.get("Skewness", -0.15)
        tail_risk = factor_scores.get("tailRiskScore", 80)
        momentum = factor_scores.get("momentumScore", 75)

        skew_bonus = 10 if skew > -0.3 else -5
        score = min(96, max(40, int(tail_risk * 0.45 + momentum * 0.35 + sortino * 8 + skew_bonus)))

        return {
            "name": "Jim Simons (Quantitative Risk)",
            "archetype": "Statistical Stability & Crash Protection",
            "alignmentScore": score,
            "status": "Low Downside Risk" if score >= 80 else "Normal Volatility",
            "thesis": "Solid risk-adjusted returns with limited crash risk in down markets.",
            "catalyst": "Low downside volatility and steady historical recovery during market pullbacks.",
        }

