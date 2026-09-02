import math
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def _safe_num(d: Any, key: str, default: float) -> float:
    """Safely extracts a numeric float value from dictionary d, coalescing None, NaN, Inf, booleans, and invalid types."""
    if not isinstance(d, dict):
        return default
    v = d.get(key)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return default
    return float(v)


class TraderArchetypeAnalyzer:
    """Analyzes any asset against proven institutional and iconic trader mental models:
    1. Warren Buffett (Value, Moat, Quality & FCF Yield)
    2. Nancy Pelosi (Congressional Policy Catalysts & Subsidies)
    3. Stanley Druckenmiller (Macro Trends, Interest Rates & Momentum)
    4. Jim Simons (Quantitative Tail-Risk & Statistical Stability)
    5. David Gardner / The Motley Fool (Rule Breaker Hyper-Growth & Secular Disruptors)
    """

    CONGRESSIONAL_POLICY_TICKERS = {
        "NVDA": "Direct beneficiary of federal CHIPS Act subsidies and hyperscale AI computing demand.",
        "AAPL": "High consumer hardware loyalty and strong enterprise services ecosystem.",
        "MSFT": "Key provider of US defense cloud infrastructure and corporate AI software.",
        "GOOGL": "Major defense and government cloud computing partner.",
        "TSLA": "Supported by Inflation Reduction Act clean energy and battery tax credits.",
        "PLTR": "Primary contractor for US Department of Defense data and battlefield AI systems.",
        "CRWD": "Essential beneficiary of federal cybersecurity mandates for government and enterprise.",
        "ENPH": "Boosted by federal solar investment tax credits and clean power incentives.",
        "AMD": "Key supplier for open-source AI computing and federal supercomputing initiatives.",
        "AVGO": "Custom AI ASIC accelerator buildouts and federal high-speed telecom connectivity mandates.",
        "TSM": "Core beneficiary of $6.6B direct CHIPS Act funding and Arizona advanced fab construction.",
        "LMT": "Prime contractor for DoD Next-Gen Air Dominance, F-35 fighter program, and foreign military sales.",
        "RTX": "Critical defense missile defense systems, radar systems, and commercial aerospace aftermarket support.",
        "VRT": "High-density liquid cooling and power infrastructure standards for federal compute facilities.",
        "LLY": "Major beneficiary of Medicare Part D coverage expansion discussions and domestic biotech manufacturing.",
        "NVO": "Pioneer in metabolic GLP-1 therapies with bipartisan discussions on federal healthcare reimbursement.",
        "GE": "Commercial aerospace propulsion aftermarket and military combat aircraft engine programs.",
        "ASML": "Monopoly supplier of EUV lithography systems backed by Western strategic export control alliances.",
        "BTC-USD": "Supported by legislative discussions on national digital asset reserves.",
        "ETH-USD": "Approved for US spot exchange-traded funds (ETFs) and institutional settlement.",
        "SOL-USD": "High-speed blockchain adoption for global financial payment networks.",
    }

    CRYPTO_MOATS = {
        "BTC": {"score": 78, "thesis": "Digital gold monetary premium and spot ETF institutional custody baseline.", "catalyst": "Fixed 21M supply cap and dominant store-of-value network effects."},
        "ETH": {"score": 75, "thesis": "Yield-generating base protocol ($3B+ annual fee burn & 3.2% validator staking yield).", "catalyst": "Layer-2 settlement growth and institutional tokenized real-world assets."},
        "SOL": {"score": 72, "thesis": "High-velocity decentralized exchange volume and ultra-low cost transaction moat.", "catalyst": "Global consumer payments integration and developer ecosystem expansion."},
        "BNB": {"score": 70, "thesis": "Continuous quarterly token burns financed by global exchange trading revenues.", "catalyst": "Ecosystem utility demand and automated supply deflation."},
    }

    MOTLEY_FOOL_DISRUPTORS = {
        "NVDA": {"score": 96, "thesis": "Top-dog GPU computing architecture with 75% gross margins and unmatched developer lock-in.", "catalyst": "Massive secular migration from CPU to accelerated AI datacenter compute."},
        "PLTR": {"score": 94, "thesis": "Founder-led enterprise AI operating system with 81% gross margins and accelerating commercial revenue.", "catalyst": "AIP platform viral adoption and defense data ontology network effects."},
        "CRWD": {"score": 90, "thesis": "First-mover cloud security platform with 76% subscription margins and strong net revenue retention.", "catalyst": "Single-agent Falcon platform module expansion and mandatory cybersecurity insurance."},
        "TSLA": {"score": 88, "thesis": "Visionary founder-led clean transport and humanoid robotics pioneer with vertical manufacturing moats.", "catalyst": "Full Self-Driving (FSD) commercial robotaxi scaling and Megapack energy storage growth."},
        "BTC-USD": {"score": 86, "thesis": "First-mover digital monetary network with unassailable brand dominance and institutional adoption.", "catalyst": "Global fiat debasement hedge and institutional custodial inflows."},
        "SOL-USD": {"score": 88, "thesis": "High-throughput consumer blockchain capturing market share in decentralized trading and payments.", "catalyst": "Sub-second transaction finality and emerging retail application ecosystem."},
    }

    def analyze_asset(
        self,
        symbol: str,
        info: Dict[str, Any],
        price_df: Any,
        risk_metrics: Dict[str, Any],
        macro_indicators: Dict[str, Any],
        factor_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all 5 iconic trader archetype models against the asset."""
        sym_str = str(symbol or "").strip().upper()
        sym_clean = sym_str.replace("-USD", "")
        is_crypto = "-USD" in sym_str or sym_clean in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LTC"]

        # 1. The Oracle (Warren Buffett / Berkshire Hathaway Value & Cash Flow)
        buffett = self._evaluate_buffett_moat(sym_clean, is_crypto, info, factor_scores)

        # 2. The Capitol Whale (Nancy Pelosi / Congressional Policy Flows)
        pelosi = self._evaluate_congressional_whale(sym_str, sym_clean, factor_scores, is_crypto, info)

        # 3. The Macro Sorcerer (Stanley Druckenmiller / Macro Trends)
        druckenmiller = self._evaluate_druckenmiller_macro(macro_indicators, factor_scores, price_df)

        # 4. The Medallion Quant (Jim Simons / Quantitative Risk)
        simons = self._evaluate_simons_quant(risk_metrics, price_df, factor_scores)

        # 5. The Growth Disruptor (David Gardner / Motley Fool Rule Breakers)
        gardner = self._evaluate_motley_fool_growth(sym_str, sym_clean, is_crypto, factor_scores, info)

        archetypes = [buffett, pelosi, druckenmiller, simons, gardner]
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
        """Warren Buffett Value, Quality, Moat & Free Cash Flow model (with Crypto Protocol Fee Proxy)."""
        if is_crypto:
            crypto_data = self.CRYPTO_MOATS.get(sym, None)
            if crypto_data:
                return {
                    "name": "Warren Buffett (Value & Moat)",
                    "archetype": "Network Moat & Protocol Cash Flows",
                    "alignmentScore": crypto_data["score"],
                    "status": "Tier-1 Network Moat",
                    "thesis": crypto_data["thesis"],
                    "catalyst": crypto_data["catalyst"],
                }
            return {
                "name": "Warren Buffett (Value & Moat)",
                "archetype": "High Cash Flow & Wide Moats",
                "alignmentScore": 38,
                "status": "Speculative Altcoin",
                "thesis": "Lacks consistent protocol fee generation or store-of-value monetary premium.",
                "catalyst": "Prefers assets with sustainable economic utility and clear cash dividends.",
            }

        quality = _safe_num(factor_scores, "qualityScore", 80.0)
        valuation = _safe_num(factor_scores, "valuationScore", 75.0)
        piotroski = _safe_num(factor_scores, "piotroskiFScore", 8.0)

        sector = (info.get("sector") if (isinstance(info, dict) and isinstance(info.get("sector"), str)) else "").lower()
        industry = (info.get("industry") if (isinstance(info, dict) and isinstance(info.get("industry"), str)) else "").lower()

        # Hardware Server Integrator & Low Gross-Margin ODM check (e.g. SMCI, DELL, HPE, VRT, CIEN, FLEX, CLS, JBL, WST)
        is_hardware_odm = (
            sym in {"SMCI", "DELL", "HPE", "VRT", "CIEN", "FLEX", "CLS", "JBL", "WST"}
            or any(k in industry for k in ["computer hardware", "server", "electronic manufacturing", "contract electronics", "chassis", "liquid cooling"])
        )
        if is_hardware_odm:
            score = min(62, max(42, int(quality * 0.30 + valuation * 0.35 + (piotroski * 10) * 0.15)))
            return {
                "name": "Warren Buffett (Value & Moat)",
                "archetype": "High Cash Flow & Wide Moats",
                "alignmentScore": score,
                "status": "Competitive Commodity Risk",
                "thesis": "Capital-intensive server hardware integration with thin gross margins (~11-14%) and customer concentration risk.",
                "catalyst": "Prefers wide-moat pricing power and tollbooth businesses over cyclical hardware assembly.",
            }

        # Logistics & Freight check (DHLGY, FDX, UPS, EXPD, JBHT, CHRW)
        is_logistics = (
            sym in {"DHLGY", "FDX", "UPS", "EXPD", "JBHT", "CHRW", "ZTO", "GXO", "XPO"}
            or (sector != "real estate" and "reit" not in industry and any(k in industry for k in ["freight", "logistics", "shipping", "courier", "trucking"]))
        )
        if is_logistics:
            score = min(72, max(48, int(quality * 0.40 + valuation * 0.40 + (piotroski * 10) * 0.15)))
            return {
                "name": "Warren Buffett (Value & Moat)",
                "archetype": "High Cash Flow & Wide Moats",
                "alignmentScore": score,
                "status": "Capital-Intensive Network Moat",
                "thesis": "Extensive global delivery sorting infrastructure provides physical barrier to entry, but requires heavy recurring fleet CapEx and is exposed to union labor and fuel cycles.",
                "catalyst": "Focuses on owner-earnings yield and free cash flow generation across global freight volume cycles.",
            }

        # Clinical Biopharma (High R&D / Binary clinical risk)
        is_biopharma = (
            sym in {"ARWR", "CPRX", "MRNA", "CRSP", "BEAM", "BIIB", "LLY", "NVO", "VRTX", "REGN", "AMGN", "GILD", "BMY", "PFE", "INCY"}
            or (sector == "healthcare" and any(k in industry for k in ["biotechnology", "drug manufacturers", "pharmaceutical", "therapeutics", "therapies", "biopharmaceutical", "drug discovery", "gene editing"]))
        )
        if is_biopharma and quality < 75:
            score = min(58, max(38, int(quality * 0.30 + valuation * 0.30 + (piotroski * 10) * 0.15)))
            return {
                "name": "Warren Buffett (Value & Moat)",
                "archetype": "High Cash Flow & Wide Moats",
                "alignmentScore": score,
                "status": "Outside Circle of Competence",
                "thesis": "Binary clinical trial risk and unpredictable cash flows fall outside classic franchise predictability.",
                "catalyst": "Prefers predictable consumer and industrial monopolies with demonstrated century-long durability.",
            }

        score = min(96, max(40, int(quality * 0.45 + valuation * 0.35 + (piotroski * 10) * 0.20)))
        if sym in ["AAPL", "BAC", "KO", "AXP", "OXY", "SPY", "QQQ"]:
            score = max(score, 90)

        if score >= 80:
            status = "High Moat Alignment"
            thesis = "High cash generation with strong pricing power, low corporate debt, and consistent share buybacks."
            catalyst = "Durable competitive advantage and steady profit margins across economic cycles."
        elif score >= 65:
            status = "Moderate Moat"
            thesis = "Solid core business with moderate pricing power; requires monitoring capital allocation and valuation entry multiple."
            catalyst = "Steady operational cash flow and disciplined balance sheet management."
        else:
            status = "Weak Moat / Capital Intensive"
            thesis = "Lacks durable pricing power or high return on invested capital (ROIC); susceptible to margin compression in downturns."
            catalyst = "Buffett discipline avoids low-margin commodity businesses without durable tollbooth moats."

        return {
            "name": "Warren Buffett (Value & Moat)",
            "archetype": "High Cash Flow & Wide Moats",
            "alignmentScore": score,
            "status": status,
            "thesis": thesis,
            "catalyst": catalyst,
        }

    def _evaluate_congressional_whale(
        self, symbol: str, sym_clean: str, factor_scores: Dict[str, Any], is_crypto: bool, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Nancy Pelosi / Congressional Policy Catalyst model."""
        policy_catalyst = self.CONGRESSIONAL_POLICY_TICKERS.get(
            symbol,
            self.CONGRESSIONAL_POLICY_TICKERS.get(sym_clean, None),
        )

        momentum = _safe_num(factor_scores, "momentumScore", 70.0)
        growth = _safe_num(factor_scores, "growthScore", 75.0)

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

        sector = (info.get("sector") if (isinstance(info, dict) and isinstance(info.get("sector"), str)) else "").lower()
        industry = (info.get("industry") if (isinstance(info, dict) and isinstance(info.get("industry"), str)) else "").lower()

        score = min(82, max(45, int((growth * 0.6) + (momentum * 0.4))))

        if any(term in sector or term in industry for term in ["defense", "aerospace"]):
            status = "Defense Appropriations Exposure"
            thesis = "Supported by national defense appropriations (NDAA) and foreign military sales (FMS) allocations."
            catalyst = "DoD multi-year modernization budgets and allied defense spending expansion."
        elif any(term in sector or term in industry for term in ["health", "biotech", "pharma"]):
            status = "Healthcare Policy Exposure"
            thesis = "Subject to federal Medicare reimbursement policy, FDA approval pathways, and public health funding."
            catalyst = "Bipartisan healthcare reform discussions and accelerated therapeutic review milestones."
        elif any(term in sector or term in industry for term in ["energy", "utilities", "industrial"]):
            status = "Infrastructure & Energy Policy"
            thesis = "Beneficiary of federal infrastructure legislation, clean energy incentives, and domestic supply chain reshoring."
            catalyst = "Federal grant disbursements and domestic manufacturing tax incentives."
        elif any(term in sector or term in industry for term in ["tech", "semiconductor", "software"]):
            status = "Tech & Innovation Policy"
            thesis = "Beneficiary of federal semiconductor incentives, AI research grants, and enterprise digital modernization."
            catalyst = "Federal technology competitiveness initiatives and government IT modernization budgets."
        else:
            status = "Neutral Policy Exposure"
            thesis = "Moderate policy alignment without major direct federal government spending programs."
            catalyst = "Broader macroeconomic policy, corporate tax rates, and regulatory oversight."

        return {
            "name": "Nancy Pelosi (Policy & Government Catalysts)",
            "archetype": "Government Spending & High-Conviction Tech",
            "alignmentScore": score,
            "status": status,
            "thesis": thesis,
            "catalyst": catalyst,
        }

    def _evaluate_druckenmiller_macro(
        self, macro_indicators: Dict[str, Any], factor_scores: Dict[str, Any], price_df: Any
    ) -> Dict[str, Any]:
        """Stanley Druckenmiller / Macro Trends & Reflexivity model (Dynamic Regime-Aware)."""
        yield_curve = _safe_num(macro_indicators, "yield_curve_spread", 0.47)
        credit_spread = _safe_num(macro_indicators, "credit_spread_oas", 2.69)
        momentum = _safe_num(factor_scores, "momentumScore", 75.0)
        growth = _safe_num(factor_scores, "growthScore", 75.0)

        # 1. Inverted Yield Curve Regime (Late-cycle / Tightening)
        if yield_curve < 0.0:
            score = min(82, max(42, int(52 * 0.4 + momentum * 0.35 + growth * 0.25)))
            status = "Inverted Yield Curve / Late-Cycle Warning"
            thesis = "Inverted yield curve signals late-cycle macroeconomic tightening; warrants tight trailing stops and tactical risk management."
            catalyst = "Flight to balance-sheet liquidity, defensive cash flows, and defensive secular alpha."
        # 2. Widening Credit Spread Regime (Credit Stress)
        elif credit_spread >= 4.0:
            score = min(84, max(45, int(58 * 0.4 + momentum * 0.35 + growth * 0.25)))
            status = "Credit Spread Widening / Macro Caution"
            thesis = "Elevated credit spreads indicate tightening financial conditions and increased discount rate risk on high-multiple equities."
            catalyst = "Federal Reserve liquidity management and corporate debt refinancing stability."
        # 3. Steepening Yield Curve / Bullish Expansion
        elif yield_curve > 0.20 and credit_spread < 3.2:
            base = 90
            score = min(97, max(45, int(base * 0.5 + momentum * 0.3 + growth * 0.2)))
            status = "Positive Macro Trend" if score >= 80 else "Neutral Macro"
            thesis = "Accommodative monetary liquidity, steepening yield curve, and strong price momentum create high-conviction macro tailwinds."
            catalyst = "Central bank easing trajectory and institutional trend-following capital inflows."
        # 4. Transitional / Normalizing Macro Regime
        else:
            base = 70 + (8 if credit_spread < 3.5 else 0)
            score = min(90, max(45, int(base * 0.5 + momentum * 0.3 + growth * 0.2)))
            status = "Neutral Macro Transition"
            thesis = "Transitional interest rate regime with flat yield curve dynamics; macro sizing favors selective momentum leaders with balance sheet resilience."
            catalyst = "Selective corporate earnings resilience amidst macroeconomic policy shifts."

        return {
            "name": "Stanley Druckenmiller (Macro Trends)",
            "archetype": "Interest Rate Trends & Market Momentum",
            "alignmentScore": score,
            "status": status,
            "thesis": thesis,
            "catalyst": catalyst,
        }

    def _evaluate_simons_quant(
        self, risk_metrics: Dict[str, Any], price_df: Any, factor_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Jim Simons / Renaissance Quantitative Risk model (Distribution & Tail-Risk Aware)."""
        sortino = _safe_num(risk_metrics, "Sortino_Ratio", 1.84)
        skew = _safe_num(risk_metrics, "Skewness", -0.15)
        tail_risk = _safe_num(factor_scores, "tailRiskScore", 80.0)
        momentum = _safe_num(factor_scores, "momentumScore", 75.0)

        skew_bonus = 10 if skew > -0.3 else (-10 if skew < -0.6 else -5)
        score = min(96, max(40, int(tail_risk * 0.45 + momentum * 0.35 + sortino * 8 + skew_bonus)))

        if score >= 80 and skew > -0.4:
            status = "Low Downside Risk"
            thesis = "Superior Sortino risk-adjusted returns and favorable return distribution with minimal left-tail downside asymmetry."
            catalyst = "Statistical edge supported by low downside semi-variance and disciplined statistical stop bounds."
        elif score >= 65 and skew > -0.6:
            status = "Normal Volatility"
            thesis = "Acceptable risk-adjusted profile with normal volatility distributions; requires quantitative position sizing."
            catalyst = "Mean-reversion tendencies and balanced reward-to-downside variance."
        else:
            status = "Elevated Tail Risk / Asymmetric Downside"
            thesis = "Asymmetric left-tail crash risk, negative skewness, or sub-optimal Sortino ratio penalize quantitative signal conviction."
            catalyst = "Heightened downside semi-variance requires strict fractional position sizing and tail-risk hedging."

        return {
            "name": "Jim Simons (Quantitative Risk)",
            "archetype": "Statistical Stability & Crash Protection",
            "alignmentScore": score,
            "status": status,
            "thesis": thesis,
            "catalyst": catalyst,
        }

    def _evaluate_motley_fool_growth(
        self, symbol: str, sym_clean: str, is_crypto: bool, factor_scores: Dict[str, Any], info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """David Gardner / Motley Fool Rule Breakers (Sector & Economics-Aware)."""
        disruptor = self.MOTLEY_FOOL_DISRUPTORS.get(symbol) or self.MOTLEY_FOOL_DISRUPTORS.get(sym_clean)
        if disruptor:
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": disruptor["score"],
                "status": "High-Conviction Rule Breaker",
                "thesis": disruptor["thesis"],
                "catalyst": disruptor["catalyst"],
            }

        growth = _safe_num(factor_scores, "growthScore", 75.0)
        momentum = _safe_num(factor_scores, "momentumScore", 70.0)
        sector = (info.get("sector") if (isinstance(info, dict) and isinstance(info.get("sector"), str)) else "").lower()
        industry = (info.get("industry") if (isinstance(info, dict) and isinstance(info.get("industry"), str)) else "").lower()

        # 1. Hardware ODM / Server Integrators (SMCI, DELL, HPE, VRT, CIEN, FLEX, CLS, JBL, WST)
        is_hardware_odm = (
            sym_clean in {"SMCI", "DELL", "HPE", "VRT", "CIEN", "FLEX", "CLS", "JBL", "WST"}
            or any(k in industry for k in ["computer hardware", "server", "electronic manufacturing", "contract electronics", "chassis", "liquid cooling"])
        )
        if is_hardware_odm:
            score = min(95, max(40, int(growth * 0.65 + momentum * 0.35)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "AI Hardware Supercycle",
                "thesis": "High-velocity AI datacenter rack deployment and direct liquid cooling integration expanding market share.",
                "catalyst": "Hyperscaler liquid-cooled GPU cluster buildouts and modular compute architecture adoption.",
            }

        # 2. Logistics & Freight (DHLGY, FDX, UPS, EXPD, JBHT, CHRW) - Exclude REITs
        is_logistics = (
            sym_clean in {"DHLGY", "FDX", "UPS", "EXPD", "JBHT", "CHRW", "ZTO", "GXO", "XPO"}
            or (sector != "real estate" and "reit" not in industry and any(k in industry for k in ["freight", "logistics", "shipping", "courier", "trucking"]))
        )
        if is_logistics:
            score = min(78, max(42, int(growth * 0.50 + momentum * 0.35 + 10)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Supply Chain & Logistics Network",
                "thesis": "Asset-heavy global supply chain and logistics network with capital-intensive delivery infrastructure and operational leverage.",
                "catalyst": "E-commerce volume expansion, automated sorting hub efficiency, and cross-border freight rate realization.",
            }

        # 3. Biopharma & Therapeutics (LLY, NVO, VRTX, ARWR, CPRX, AMGN, GILD, BIIB, MRNA, REGN, CRSP, BEAM)
        is_biopharma = (
            sym_clean in {"LLY", "NVO", "VRTX", "ARWR", "CPRX", "AMGN", "GILD", "BIIB", "MRNA", "REGN", "BMY", "PFE", "INCY", "CRSP", "BEAM"}
            or (sector == "healthcare" and any(k in industry for k in ["biotechnology", "drug manufacturers", "pharmaceutical", "therapeutics", "therapies", "biopharmaceutical", "drug discovery", "gene editing"]))
            or any(k in industry for k in ["biotechnology", "drug discovery", "biopharmaceutical"])
        )
        if is_biopharma:
            score = min(96, max(45, int(growth * 0.65 + momentum * 0.35)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Biopharma Innovation & Pipeline",
                "thesis": "Pioneering therapeutic drug pipeline with high gross margins (~75-85%), proprietary intellectual property, and blockbuster market potential.",
                "catalyst": "Phase 3 clinical trial readouts, FDA accelerated approvals, and global commercial formulary expansion.",
            }

        # 4. Utilities & Regulated Infrastructure (Evaluated before Commodities to prevent 'gas' utilities collision)
        is_utility = (
            sym_clean in {"NEE", "DUK", "SO", "AEP", "SRE", "D", "EXC", "XEL"}
            or sector in ["utilities"]
            or any(k in industry for k in ["utilities", "electric utility", "power", "water utilities", "regulated gas"])
        )
        if is_utility:
            score = min(76, max(40, int(growth * 0.45 + momentum * 0.35 + 12)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Regulated Utility / Infrastructure Asset Base",
                "thesis": "Capital-intensive regulated asset base with contracted utility rate structures and steady cash distribution.",
                "catalyst": "Grid electrification, data center power interconnection demand, and rate base expansion.",
            }

        # 5. Commodities / Energy / Materials (XOM, CVX, COP, OXY, CLF, NUE, FCX, SLB)
        is_commodity = (
            sym_clean in {"XOM", "CVX", "COP", "OXY", "CLF", "NUE", "FCX", "SLB"}
            or sector in ["energy", "basic materials"]
            or (sector != "utilities" and any(k in industry for k in ["oil", "gas", "mining", "metals", "steel", "chemical"]))
        )
        if is_commodity:
            score = min(74, max(40, int(growth * 0.45 + momentum * 0.40 + 10)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Commodity Resource / Cyclical",
                "thesis": "Cyclical resource producer dependent on global commodity pricing and capital expenditure cycles rather than proprietary software moat.",
                "catalyst": "Global upstream demand cycles, refining crack spreads, and disciplined capital return programs.",
            }

        # 6. Financial Services & Banking
        is_financial = (
            sym_clean in {"JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW"}
            or sector in ["financials", "financial services"]
            or any(k in industry for k in ["banking", "bank", "financial services", "capital markets", "insurance", "asset management"])
        )
        if is_financial:
            score = min(80, max(40, int(growth * 0.50 + momentum * 0.35 + 10)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Financial Institution & Capital Allocator",
                "thesis": "Regulated financial services provider benefiting from net interest margins, loan book growth, and institutional asset management.",
                "catalyst": "Credit expansion, capital markets activity, and return on tangible equity (ROTE) optimization.",
            }

        # 7. Real Estate & REITs
        is_real_estate = (
            sym_clean in {"O", "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "DLR"}
            or sector in ["real estate"]
            or any(k in industry for k in ["reit", "real estate"])
        )
        if is_real_estate:
            score = min(75, max(40, int(growth * 0.45 + momentum * 0.35 + 10)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Real Estate Asset Portfolio",
                "thesis": "Income-generating property portfolio with contracted tenant lease cash flows and asset appreciation.",
                "catalyst": "Occupancy rate expansion, rental rate escalations, and property acquisition pipeline.",
            }

        # 8. Consumer Defensive & Retail Networks
        is_consumer_retail = (
            sym_clean in {"WMT", "COST", "TGT", "HD", "LOW", "PG", "KO", "PEP"}
            or sector in ["consumer defensive", "consumer staples"]
            or any(k in industry for k in ["retail", "grocery", "discount stores", "household products", "beverages", "food products"])
        )
        if is_consumer_retail:
            score = min(78, max(40, int(growth * 0.50 + momentum * 0.35 + 10)))
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Consumer Distribution & Retail Network",
                "thesis": "High-volume consumer retail distribution network with strong omnichannel foot-traffic and supply chain scale.",
                "catalyst": "Same-store sales growth, private label expansion, and supply chain automation.",
            }

        # 9. Turnaround / Mature / Low Growth Candidate (e.g. ULTA, LULU, KO or growth < 60)
        score = min(95, max(40, int(growth * 0.65 + momentum * 0.35)))
        if growth < 60 or score < 60:
            return {
                "name": "David Gardner (Motley Fool Rule Breakers)",
                "archetype": "First-Mover Disruptors & Hyper-Growth",
                "alignmentScore": score,
                "status": "Maturing Business / Low Secular Growth",
                "thesis": "Moderate or decelerating top-line revenue growth; lacks the hyper-growth velocity sought in early-stage Rule Breaker candidates.",
                "catalyst": "Brand revitalization, operational turnaround, and margin stabilization.",
            }

        # 10. Default High-Margin Tech / Software / Secular Growth
        return {
            "name": "David Gardner (Motley Fool Rule Breakers)",
            "archetype": "First-Mover Disruptors & Hyper-Growth",
            "alignmentScore": score,
            "status": "Growth Compounder" if score >= 80 else "Moderate Growth",
            "thesis": "High gross margin secular growth candidate with expanding industry market share.",
            "catalyst": "Emerging product adoption and industry transition toward digital/cloud architecture.",
        }

