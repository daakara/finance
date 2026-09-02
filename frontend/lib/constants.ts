/**
 * Shared Single Source of Truth for Asset Data, Price Fallbacks, and Factor Baselines.
 * Enforces state parity across Watchlist, Radar, Chart, and Terminal components.
 */

import { AssetFactorScores, MacroDifficultyRating, ExpectedReturnForecast } from "./api";

export interface WatchlistDefinition {
  symbol: string;
  name: string;
  type: "Stock" | "ETF" | "Crypto";
}

export const SHARED_WATCHLIST_ITEMS: WatchlistDefinition[] = [
  // Mega-Cap Tech & Global Pharma Equities
  { symbol: "NVO", name: "Novo Nordisk", type: "Stock" },
  { symbol: "LLY", name: "Eli Lilly", type: "Stock" },
  { symbol: "AAPL", name: "Apple Inc.", type: "Stock" },
  { symbol: "NVDA", name: "NVIDIA Corp.", type: "Stock" },
  { symbol: "MSFT", name: "Microsoft Corp.", type: "Stock" },
  { symbol: "GOOGL", name: "Alphabet Inc.", type: "Stock" },
  { symbol: "TSLA", name: "Tesla Inc.", type: "Stock" },
  { symbol: "PLTR", name: "Palantir Tech", type: "Stock" },
  { symbol: "CIEN", name: "Ciena Corp.", type: "Stock" },

  // Institutional Index, Sector, Commodity & Bond ETFs
  { symbol: "SPY", name: "S&P 500 ETF", type: "ETF" },
  { symbol: "QQQ", name: "Invesco QQQ", type: "ETF" },
  { symbol: "SMH", name: "VanEck Semi ETF", type: "ETF" },
  { symbol: "XLK", name: "Tech Select SPDR", type: "ETF" },
  { symbol: "IWM", name: "Russell 2000 ETF", type: "ETF" },
  { symbol: "GLD", name: "SPDR Gold Shares", type: "ETF" },
  { symbol: "TLT", name: "20+ Yr Treasury", type: "ETF" },
  { symbol: "XLE", name: "Energy Select ETF", type: "ETF" },

  // Digital Assets / Crypto
  { symbol: "BTC-USD", name: "Bitcoin", type: "Crypto" },
  { symbol: "ETH-USD", name: "Ethereum", type: "Crypto" },
  { symbol: "SOL-USD", name: "Solana", type: "Crypto" },
];

export const SHARED_FACTOR_SCORES: Record<string, { scores: AssetFactorScores }> = {
  "AAPL": {
    scores: { growthScore: 84, qualityScore: 90, valuationScore: 72, momentumScore: 78, tailRiskScore: 82, compositeFactorScore: 82, verdict: "Strong Accumulation Candidate", piotroskiFScore: 8 }
  },
  "NVDA": {
    scores: { growthScore: 96, qualityScore: 95, valuationScore: 68, momentumScore: 94, tailRiskScore: 76, compositeFactorScore: 91, verdict: "Exceptional Growth Leader", piotroskiFScore: 8 }
  },
  "NVO": {
    scores: { growthScore: 89, qualityScore: 94, valuationScore: 74, momentumScore: 86, tailRiskScore: 85, compositeFactorScore: 88, verdict: "High Quality Compounder", piotroskiFScore: 8 }
  },
  "LLY": {
    scores: { growthScore: 92, qualityScore: 91, valuationScore: 65, momentumScore: 90, tailRiskScore: 81, compositeFactorScore: 86, verdict: "Secular Pharma Leader", piotroskiFScore: 7 }
  },
  "MSFT": {
    scores: { growthScore: 86, qualityScore: 96, valuationScore: 70, momentumScore: 80, tailRiskScore: 88, compositeFactorScore: 87, verdict: "Fortress Balance Sheet", piotroskiFScore: 8 }
  },
  "GOOGL": {
    scores: { growthScore: 85, qualityScore: 93, valuationScore: 78, momentumScore: 82, tailRiskScore: 86, compositeFactorScore: 86, verdict: "Deep Value & AI Moat", piotroskiFScore: 8 }
  },
  "TSLA": {
    scores: { growthScore: 80, qualityScore: 82, valuationScore: 55, momentumScore: 88, tailRiskScore: 68, compositeFactorScore: 77, verdict: "High Beta Autonomy Speculation", piotroskiFScore: 6 }
  },
  "PLTR": {
    scores: { growthScore: 94, qualityScore: 90, valuationScore: 62, momentumScore: 95, tailRiskScore: 74, compositeFactorScore: 87, verdict: "Commercial AI Breakout", piotroskiFScore: 8 }
  },
  "SPY": {
    scores: { growthScore: 75, qualityScore: 88, valuationScore: 75, momentumScore: 80, tailRiskScore: 90, compositeFactorScore: 82, verdict: "Core Index Benchmark", piotroskiFScore: 8 }
  },
  "QQQ": {
    scores: { growthScore: 85, qualityScore: 90, valuationScore: 72, momentumScore: 86, tailRiskScore: 84, compositeFactorScore: 85, verdict: "Nasdaq-100 Growth Engine", piotroskiFScore: 8 }
  },
  "SMH": {
    scores: { growthScore: 92, qualityScore: 90, valuationScore: 68, momentumScore: 92, tailRiskScore: 78, compositeFactorScore: 88, verdict: "Semiconductor Supercycle", piotroskiFScore: 8 }
  },
  "XLK": {
    scores: { growthScore: 88, qualityScore: 92, valuationScore: 70, momentumScore: 88, tailRiskScore: 82, compositeFactorScore: 86, verdict: "Broad Technology Sector", piotroskiFScore: 8 }
  },
  "IWM": {
    scores: { growthScore: 70, qualityScore: 75, valuationScore: 80, momentumScore: 75, tailRiskScore: 75, compositeFactorScore: 75, verdict: "Small-Cap Value Rotation", piotroskiFScore: 6 }
  },
  "GLD": {
    scores: { growthScore: 50, qualityScore: 95, valuationScore: 70, momentumScore: 78, tailRiskScore: 95, compositeFactorScore: 80, verdict: "Macro Hedge & Safe Haven", piotroskiFScore: 8 }
  },
  "TLT": {
    scores: { growthScore: 40, qualityScore: 98, valuationScore: 85, momentumScore: 60, tailRiskScore: 88, compositeFactorScore: 74, verdict: "Duration Rate Hedge", piotroskiFScore: 8 }
  },
  "XLE": {
    scores: { growthScore: 68, qualityScore: 85, valuationScore: 82, momentumScore: 72, tailRiskScore: 80, compositeFactorScore: 78, verdict: "Energy Cash Flow Dividend", piotroskiFScore: 7 }
  },
  "BTC": {
    scores: { growthScore: 90, qualityScore: 80, valuationScore: 60, momentumScore: 92, tailRiskScore: 65, compositeFactorScore: 81, verdict: "Digital Gold & Liquidity Beta", piotroskiFScore: 7 }
  },
  "ETH": {
    scores: { growthScore: 85, qualityScore: 82, valuationScore: 65, momentumScore: 86, tailRiskScore: 70, compositeFactorScore: 80, verdict: "Smart Contract Settlement Layer", piotroskiFScore: 7 }
  },
  "SOL": {
    scores: { growthScore: 92, qualityScore: 78, valuationScore: 62, momentumScore: 90, tailRiskScore: 62, compositeFactorScore: 80, verdict: "High Throughput DeFi Beta", piotroskiFScore: 6 }
  },
    // Authentic Small-Cap Discovery Gems (from Screener)
  "CPRX": {
    scores: { growthScore: 88, qualityScore: 95, valuationScore: 85, momentumScore: 86, tailRiskScore: 90, compositeFactorScore: 89, verdict: "Greenblatt Magic Formula", piotroskiFScore: 9 }
  },
  "ACLS": {
    scores: { growthScore: 91, qualityScore: 93, valuationScore: 82, momentumScore: 88, tailRiskScore: 85, compositeFactorScore: 88, verdict: "Peter Lynch GARP Compounder", piotroskiFScore: 8 }
  },
  "TMDX": {
    scores: { growthScore: 94, qualityScore: 89, valuationScore: 72, momentumScore: 92, tailRiskScore: 81, compositeFactorScore: 86, verdict: "Disruptive Rule Breaker", piotroskiFScore: 8 }
  },
  "LNTH": {
    scores: { growthScore: 89, qualityScore: 94, valuationScore: 86, momentumScore: 85, tailRiskScore: 88, compositeFactorScore: 88, verdict: "Greenblatt Magic Formula", piotroskiFScore: 9 }
  },
  "POWI": {
    scores: { growthScore: 85, qualityScore: 92, valuationScore: 80, momentumScore: 82, tailRiskScore: 86, compositeFactorScore: 85, verdict: "Peter Lynch GARP Compounder", piotroskiFScore: 8 }
  },
  "MEDP": {
    scores: { growthScore: 90, qualityScore: 94, valuationScore: 76, momentumScore: 85, tailRiskScore: 88, compositeFactorScore: 87, verdict: "Greenblatt Magic Formula", piotroskiFScore: 9 }
  },
  "ELF": {
    scores: { growthScore: 92, qualityScore: 90, valuationScore: 78, momentumScore: 88, tailRiskScore: 84, compositeFactorScore: 88, verdict: "Peter Lynch GARP Compounder", piotroskiFScore: 8 }
  },
  "DUOL": {
    scores: { growthScore: 94, qualityScore: 88, valuationScore: 70, momentumScore: 92, tailRiskScore: 80, compositeFactorScore: 86, verdict: "Disruptive Rule Breaker", piotroskiFScore: 8 }
  },
  "DECK": {
    scores: { growthScore: 82, qualityScore: 94, valuationScore: 80, momentumScore: 45, tailRiskScore: 78, compositeFactorScore: 76, verdict: "Stage 4 Correction / Base Building Required", piotroskiFScore: 8 }
  },
  "PODD": {
    scores: { growthScore: 84, qualityScore: 91, valuationScore: 76, momentumScore: 42, tailRiskScore: 76, compositeFactorScore: 74, verdict: "Stage 4 Correction / Base Building Required", piotroskiFScore: 8 }
  },
  "DXCM": {
    scores: { growthScore: 82, qualityScore: 88, valuationScore: 72, momentumScore: 68, tailRiskScore: 80, compositeFactorScore: 78, verdict: "Stage 1 Bottoming Base / MedTech Re-Accumulation", piotroskiFScore: 8 }
  },
  "SNPS": {
    scores: { growthScore: 88, qualityScore: 94, valuationScore: 78, momentumScore: 72, tailRiskScore: 84, compositeFactorScore: 84, verdict: "EDA Software Moat & AI Chip Design", piotroskiFScore: 8 }
  },
  "CDNS": {
    scores: { growthScore: 86, qualityScore: 92, valuationScore: 76, momentumScore: 74, tailRiskScore: 82, compositeFactorScore: 83, verdict: "EDA Semiconductor IP Leader", piotroskiFScore: 8 }
  },
  "NOW": {
    scores: { growthScore: 90, qualityScore: 95, valuationScore: 70, momentumScore: 82, tailRiskScore: 85, compositeFactorScore: 86, verdict: "Enterprise Workflow AI Monopoly", piotroskiFScore: 8 }
  },
  "ANET": {
    scores: { growthScore: 93, qualityScore: 91, valuationScore: 74, momentumScore: 89, tailRiskScore: 82, compositeFactorScore: 87, verdict: "AI Networking & Cloud Switching", piotroskiFScore: 8 }
  },
  "VRT": {
    scores: { growthScore: 95, qualityScore: 86, valuationScore: 68, momentumScore: 96, tailRiskScore: 78, compositeFactorScore: 86, verdict: "High Momentum AI Infrastructure", piotroskiFScore: 7 }
  },
  "CRWD": {
    scores: { growthScore: 91, qualityScore: 89, valuationScore: 65, momentumScore: 94, tailRiskScore: 79, compositeFactorScore: 85, verdict: "Secular Cybersecurity Leader", piotroskiFScore: 8 }
  },
  "TSM": {
    scores: { growthScore: 92, qualityScore: 96, valuationScore: 75, momentumScore: 90, tailRiskScore: 86, compositeFactorScore: 89, verdict: "Global Foundry Monopoly", piotroskiFScore: 9 }
  },
  "AMD": {
    scores: { growthScore: 90, qualityScore: 88, valuationScore: 68, momentumScore: 89, tailRiskScore: 80, compositeFactorScore: 84, verdict: "AI Compute Challenger", piotroskiFScore: 8 }
  },
  "CIEN": {
    scores: { growthScore: 86, qualityScore: 89, valuationScore: 74, momentumScore: 84, tailRiskScore: 82, compositeFactorScore: 84, verdict: "Optical Networking AI Beneficiary", piotroskiFScore: 8 }
  },
  "DHL": {
    scores: { growthScore: 82, qualityScore: 91, valuationScore: 84, momentumScore: 76, tailRiskScore: 86, compositeFactorScore: 85, verdict: "Global Express Logistics Leader", piotroskiFScore: 8 }
  },
  "DHLGY": {
    scores: { growthScore: 82, qualityScore: 91, valuationScore: 84, momentumScore: 76, tailRiskScore: 86, compositeFactorScore: 85, verdict: "Global Express Logistics Leader", piotroskiFScore: 8 }
  },
  "FDX": {
    scores: { growthScore: 80, qualityScore: 88, valuationScore: 80, momentumScore: 82, tailRiskScore: 82, compositeFactorScore: 83, verdict: "Integrated Express Network", piotroskiFScore: 7 }
  },
  "UPS": {
    scores: { growthScore: 76, qualityScore: 90, valuationScore: 82, momentumScore: 74, tailRiskScore: 85, compositeFactorScore: 82, verdict: "High-Yield Domestic Logistics Moat", piotroskiFScore: 8 }
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