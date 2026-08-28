/**
 * Shared Single Source of Truth for Asset Data, Price Fallbacks, and Factor Baselines.
 * Enforces state parity across Watchlist, Radar, Chart, and Terminal components.
 */

import { AssetFactorScores, MacroDifficultyRating, ExpectedReturnForecast } from "./api";

export interface WatchlistDefinition {
  symbol: string;
  name: string;
  price: string;
  change: string;
  isUp: boolean;
  type: "Stock" | "ETF" | "Crypto";
}

export const SHARED_WATCHLIST_ITEMS: WatchlistDefinition[] = [
  // Mega-Cap Tech & Global Pharma Equities
  { symbol: "NVO", name: "Novo Nordisk", price: "$138.50", change: "+1.85%", isUp: true, type: "Stock" },
  { symbol: "LLY", name: "Eli Lilly", price: "$920.40", change: "+2.10%", isUp: true, type: "Stock" },
  { symbol: "AAPL", name: "Apple Inc.", price: "$309.90", change: "-0.45%", isUp: false, type: "Stock" },
  { symbol: "NVDA", name: "NVIDIA Corp.", price: "$213.05", change: "+3.14%", isUp: true, type: "Stock" },
  { symbol: "MSFT", name: "Microsoft Corp.", price: "$491.71", change: "+0.85%", isUp: true, type: "Stock" },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: "$346.96", change: "+1.40%", isUp: true, type: "Stock" },
  { symbol: "TSLA", name: "Tesla Inc.", price: "$350.25", change: "+2.15%", isUp: true, type: "Stock" },
  { symbol: "PLTR", name: "Palantir Tech", price: "$142.80", change: "+4.12%", isUp: true, type: "Stock" },
  { symbol: "CIEN", name: "Ciena Corp.", price: "$417.00", change: "+2.65%", isUp: true, type: "Stock" },

  // Institutional Index, Sector, Commodity & Bond ETFs
  { symbol: "SPY", name: "S&P 500 ETF", price: "$765.91", change: "+0.65%", isUp: true, type: "ETF" },
  { symbol: "QQQ", name: "Invesco QQQ", price: "$710.72", change: "+1.10%", isUp: true, type: "ETF" },
  { symbol: "SMH", name: "VanEck Semi ETF", price: "$288.40", change: "+2.45%", isUp: true, type: "ETF" },
  { symbol: "XLK", name: "Tech Select SPDR", price: "$246.15", change: "+1.30%", isUp: true, type: "ETF" },
  { symbol: "IWM", name: "Russell 2000 ETF", price: "$224.50", change: "+0.95%", isUp: true, type: "ETF" },
  { symbol: "GLD", name: "SPDR Gold Shares", price: "$264.20", change: "+0.40%", isUp: true, type: "ETF" },
  { symbol: "TLT", name: "20+ Yr Treasury", price: "$88.65", change: "-0.30%", isUp: false, type: "ETF" },
  { symbol: "XLE", name: "Energy Select ETF", price: "$86.10", change: "+1.05%", isUp: true, type: "ETF" },

  // Digital Assets / Crypto
  { symbol: "BTC-USD", name: "Bitcoin", price: "$78,213.00", change: "+2.80%", isUp: true, type: "Crypto" },
  { symbol: "ETH-USD", name: "Ethereum", price: "$2,438.00", change: "+1.65%", isUp: true, type: "Crypto" },
  { symbol: "SOL-USD", name: "Solana", price: "$96.73", change: "+0.24%", isUp: true, type: "Crypto" },
];

export const SHARED_FACTOR_SCORES: Record<string, { price: number; changePct: number; scores: AssetFactorScores }> = {
  "AAPL": {
    price: 309.90,
    changePct: -0.45,
    scores: { growthScore: 84, qualityScore: 90, valuationScore: 72, momentumScore: 78, tailRiskScore: 82, compositeFactorScore: 82, verdict: "Strong Buy / Core Hold", piotroskiFScore: 8 }
  },
  "NVDA": {
    price: 213.05,
    changePct: 3.14,
    scores: { growthScore: 96, qualityScore: 95, valuationScore: 68, momentumScore: 94, tailRiskScore: 76, compositeFactorScore: 91, verdict: "Exceptional Growth Leader", piotroskiFScore: 8 }
  },
  "NVO": {
    price: 138.50,
    changePct: 1.85,
    scores: { growthScore: 89, qualityScore: 94, valuationScore: 74, momentumScore: 86, tailRiskScore: 85, compositeFactorScore: 88, verdict: "High Quality Compounder", piotroskiFScore: 8 }
  },
  "LLY": {
    price: 920.40,
    changePct: 2.10,
    scores: { growthScore: 92, qualityScore: 91, valuationScore: 65, momentumScore: 90, tailRiskScore: 81, compositeFactorScore: 86, verdict: "Secular Pharma Leader", piotroskiFScore: 7 }
  },
  "MSFT": {
    price: 491.71,
    changePct: 0.85,
    scores: { growthScore: 86, qualityScore: 96, valuationScore: 70, momentumScore: 80, tailRiskScore: 88, compositeFactorScore: 87, verdict: "Fortress Balance Sheet", piotroskiFScore: 8 }
  },
  "GOOGL": {
    price: 346.96,
    changePct: 1.40,
    scores: { growthScore: 85, qualityScore: 93, valuationScore: 78, momentumScore: 82, tailRiskScore: 86, compositeFactorScore: 86, verdict: "Deep Value & AI Moat", piotroskiFScore: 8 }
  },
  "TSLA": {
    price: 350.25,
    changePct: 2.15,
    scores: { growthScore: 80, qualityScore: 82, valuationScore: 55, momentumScore: 88, tailRiskScore: 68, compositeFactorScore: 77, verdict: "High Beta Autonomy Speculation", piotroskiFScore: 6 }
  },
  "PLTR": {
    price: 142.80,
    changePct: 4.12,
    scores: { growthScore: 94, qualityScore: 90, valuationScore: 62, momentumScore: 95, tailRiskScore: 74, compositeFactorScore: 87, verdict: "Commercial AI Breakout", piotroskiFScore: 8 }
  },
  "SPY": {
    price: 765.91,
    changePct: 0.65,
    scores: { growthScore: 75, qualityScore: 88, valuationScore: 75, momentumScore: 80, tailRiskScore: 90, compositeFactorScore: 82, verdict: "Core Index Benchmark", piotroskiFScore: 8 }
  },
  "QQQ": {
    price: 710.72,
    changePct: 1.10,
    scores: { growthScore: 85, qualityScore: 90, valuationScore: 72, momentumScore: 86, tailRiskScore: 84, compositeFactorScore: 85, verdict: "Nasdaq-100 Growth Engine", piotroskiFScore: 8 }
  },
  "SMH": {
    price: 288.40,
    changePct: 2.45,
    scores: { growthScore: 92, qualityScore: 90, valuationScore: 68, momentumScore: 92, tailRiskScore: 78, compositeFactorScore: 88, verdict: "Semiconductor Supercycle", piotroskiFScore: 8 }
  },
  "XLK": {
    price: 246.15,
    changePct: 1.30,
    scores: { growthScore: 88, qualityScore: 92, valuationScore: 70, momentumScore: 88, tailRiskScore: 82, compositeFactorScore: 86, verdict: "Broad Technology Sector", piotroskiFScore: 8 }
  },
  "IWM": {
    price: 224.50,
    changePct: 0.95,
    scores: { growthScore: 70, qualityScore: 75, valuationScore: 80, momentumScore: 75, tailRiskScore: 75, compositeFactorScore: 75, verdict: "Small-Cap Value Rotation", piotroskiFScore: 6 }
  },
  "GLD": {
    price: 264.20,
    changePct: 0.40,
    scores: { growthScore: 50, qualityScore: 95, valuationScore: 70, momentumScore: 78, tailRiskScore: 95, compositeFactorScore: 80, verdict: "Macro Hedge & Safe Haven", piotroskiFScore: 8 }
  },
  "TLT": {
    price: 88.65,
    changePct: -0.30,
    scores: { growthScore: 40, qualityScore: 98, valuationScore: 85, momentumScore: 60, tailRiskScore: 88, compositeFactorScore: 74, verdict: "Duration Rate Hedge", piotroskiFScore: 8 }
  },
  "XLE": {
    price: 86.10,
    changePct: 1.05,
    scores: { growthScore: 68, qualityScore: 85, valuationScore: 82, momentumScore: 72, tailRiskScore: 80, compositeFactorScore: 78, verdict: "Energy Cash Flow Dividend", piotroskiFScore: 7 }
  },
  "BTC": {
    price: 78213.0,
    changePct: 2.80,
    scores: { growthScore: 90, qualityScore: 80, valuationScore: 60, momentumScore: 92, tailRiskScore: 65, compositeFactorScore: 81, verdict: "Digital Gold & Liquidity Beta", piotroskiFScore: 7 }
  },
  "ETH": {
    price: 2438.0,
    changePct: 1.65,
    scores: { growthScore: 85, qualityScore: 82, valuationScore: 65, momentumScore: 86, tailRiskScore: 70, compositeFactorScore: 80, verdict: "Smart Contract Settlement Layer", piotroskiFScore: 7 }
  },
  "SOL": {
    price: 96.73,
    changePct: 0.24,
    scores: { growthScore: 92, qualityScore: 78, valuationScore: 62, momentumScore: 90, tailRiskScore: 62, compositeFactorScore: 80, verdict: "High Throughput DeFi Beta", piotroskiFScore: 6 }
  },
    // Authentic Small-Cap Discovery Gems (from Screener)
  "CPRX": {
    price: 21.40,
    changePct: 1.90,
    scores: { growthScore: 88, qualityScore: 95, valuationScore: 85, momentumScore: 86, tailRiskScore: 90, compositeFactorScore: 89, verdict: "Greenblatt Magic Formula", piotroskiFScore: 9 }
  },
  "ACLS": {
    price: 94.20,
    changePct: 2.30,
    scores: { growthScore: 91, qualityScore: 93, valuationScore: 82, momentumScore: 88, tailRiskScore: 85, compositeFactorScore: 88, verdict: "Peter Lynch GARP Compounder", piotroskiFScore: 8 }
  },
  "TMDX": {
    price: 128.50,
    changePct: 3.15,
    scores: { growthScore: 94, qualityScore: 89, valuationScore: 72, momentumScore: 92, tailRiskScore: 81, compositeFactorScore: 86, verdict: "Disruptive Rule Breaker", piotroskiFScore: 8 }
  },
  "LNTH": {
    price: 100.78,
    changePct: -4.09,
    scores: { growthScore: 89, qualityScore: 94, valuationScore: 86, momentumScore: 85, tailRiskScore: 88, compositeFactorScore: 88, verdict: "Greenblatt Magic Formula", piotroskiFScore: 9 }
  },
  "POWI": {
    price: 72.50,
    changePct: 0.85,
    scores: { growthScore: 85, qualityScore: 92, valuationScore: 80, momentumScore: 82, tailRiskScore: 86, compositeFactorScore: 85, verdict: "Peter Lynch GARP Compounder", piotroskiFScore: 8 }
  },
  "MEDP": {
    price: 395.10,
    changePct: 1.45,
    scores: { growthScore: 90, qualityScore: 94, valuationScore: 76, momentumScore: 85, tailRiskScore: 88, compositeFactorScore: 87, verdict: "Greenblatt Magic Formula", piotroskiFScore: 9 }
  },
  "ELF": {
    price: 182.40,
    changePct: 2.15,
    scores: { growthScore: 92, qualityScore: 90, valuationScore: 78, momentumScore: 88, tailRiskScore: 84, compositeFactorScore: 88, verdict: "Peter Lynch GARP Compounder", piotroskiFScore: 8 }
  },
  "DUOL": {
    price: 312.80,
    changePct: 3.20,
    scores: { growthScore: 94, qualityScore: 88, valuationScore: 70, momentumScore: 92, tailRiskScore: 80, compositeFactorScore: 86, verdict: "Disruptive Rule Breaker", piotroskiFScore: 8 }
  },
  "DECK": {
    price: 86.33,
    changePct: -1.25,
    scores: { growthScore: 82, qualityScore: 94, valuationScore: 80, momentumScore: 45, tailRiskScore: 78, compositeFactorScore: 76, verdict: "Stage 4 Correction / Base Building Required", piotroskiFScore: 8 }
  },
  "PODD": {
    price: 143.41,
    changePct: -0.95,
    scores: { growthScore: 84, qualityScore: 91, valuationScore: 76, momentumScore: 42, tailRiskScore: 76, compositeFactorScore: 74, verdict: "Stage 4 Correction / Base Building Required", piotroskiFScore: 8 }
  },
  "SNPS": {
    price: 464.89,
    changePct: 0.85,
    scores: { growthScore: 88, qualityScore: 94, valuationScore: 78, momentumScore: 72, tailRiskScore: 84, compositeFactorScore: 84, verdict: "EDA Software Moat & AI Chip Design", piotroskiFScore: 8 }
  },
  "CDNS": {
    price: 254.20,
    changePct: 1.10,
    scores: { growthScore: 86, qualityScore: 92, valuationScore: 76, momentumScore: 74, tailRiskScore: 82, compositeFactorScore: 83, verdict: "EDA Semiconductor IP Leader", piotroskiFScore: 8 }
  },
  "NOW": {
    price: 785.40,
    changePct: 1.45,
    scores: { growthScore: 90, qualityScore: 95, valuationScore: 70, momentumScore: 82, tailRiskScore: 85, compositeFactorScore: 86, verdict: "Enterprise Workflow AI Monopoly", piotroskiFScore: 8 }
  },
  "ANET": {
    price: 324.50,
    changePct: 2.10,
    scores: { growthScore: 93, qualityScore: 91, valuationScore: 74, momentumScore: 89, tailRiskScore: 82, compositeFactorScore: 87, verdict: "AI Networking & Cloud Switching", piotroskiFScore: 8 }
  },
  "VRT": {
    price: 88.40,
    changePct: 2.85,
    scores: { growthScore: 95, qualityScore: 86, valuationScore: 68, momentumScore: 96, tailRiskScore: 78, compositeFactorScore: 86, verdict: "High Momentum AI Infrastructure", piotroskiFScore: 7 }
  },
  "CRWD": {
    price: 272.50,
    changePct: 3.40,
    scores: { growthScore: 91, qualityScore: 89, valuationScore: 65, momentumScore: 94, tailRiskScore: 79, compositeFactorScore: 85, verdict: "Secular Cybersecurity Leader", piotroskiFScore: 8 }
  },
  "TSM": {
    price: 188.60,
    changePct: 2.15,
    scores: { growthScore: 92, qualityScore: 96, valuationScore: 75, momentumScore: 90, tailRiskScore: 86, compositeFactorScore: 89, verdict: "Global Foundry Monopoly", piotroskiFScore: 9 }
  },
  "AMD": {
    price: 154.30,
    changePct: 2.70,
    scores: { growthScore: 90, qualityScore: 88, valuationScore: 68, momentumScore: 89, tailRiskScore: 80, compositeFactorScore: 84, verdict: "AI Compute Challenger", piotroskiFScore: 8 }
  },
  "CIEN": {
    price: 417.00,
    changePct: 2.65,
    scores: { growthScore: 86, qualityScore: 89, valuationScore: 74, momentumScore: 84, tailRiskScore: 82, compositeFactorScore: 84, verdict: "Optical Networking AI Beneficiary", piotroskiFScore: 8 }
  },
};
export const DEFAULT_MACRO_DIFFICULTY: MacroDifficultyRating = {
  rating: 1,
  regime: "Optimal Expansionary Goldilocks",
  interestRateImpact: "Steepening curve (+0.47%) and tight credit spreads fuel strong risk-on alpha",
  inflationImpact: "CPI (2.4% YoY) moderation reduces discount rate pressure on valuations",
  yield_curve_spread: 0.47,
  fed_funds_rate: 3.63,
  credit_spread_oas: 2.69,
  cpi_yoy: 2.4,
};

export const DEFAULT_EXPECTED_RETURN: ExpectedReturnForecast = {
  p10Pessimistic: -8.4,
  p50Expected: +18.6,
  p90Optimistic: +38.2,
  annualizedVolatility: 22.4,
  forecastHorizonDays: 90,
};