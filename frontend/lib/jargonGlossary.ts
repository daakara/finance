/**
 * ARX Terminal — Quantitative & Institutional Financial Jargon Glossary
 * Canonical dictionary of quantitative, statistical, and technical terminology
 * mapped to institutional plain-English definitions and progressive disclosure notes.
 */

export interface JargonDefinition {
  term: string;
  shortName: string;
  category: "TECHNICAL" | "FUNDAMENTAL" | "RISK" | "EXECUTION" | "MACRO";
  plainEnglishSummary: string;
  detailedDefinition: string;
  institutionalContext?: string;
}

export const JARGON_GLOSSARY: Record<string, JargonDefinition> = {
  VCP: {
    term: "Volatility Contraction Pattern",
    shortName: "VCP",
    category: "TECHNICAL",
    plainEnglishSummary: "A tightening price consolidation with drying volume, showing sellers are exhausted before a breakout.",
    detailedDefinition: "Mark Minervini's structural consolidation pattern characterized by 2 to 4 progressive pullbacks of decreasing percentage depth on declining volume, reflecting institutional supply absorption.",
    institutionalContext: "Used in Radar Screener and Confluence Engine as a primary Stage 2 continuation pattern.",
  },
  SMA50: {
    term: "50-Day Simple Moving Average",
    shortName: "50 SMA",
    category: "TECHNICAL",
    plainEnglishSummary: "The average closing price over the last 10 trading weeks, used as a key benchmark for institutional support.",
    detailedDefinition: "Arithmetic average of the past 50 daily closing prices. Leading institutional funds use this level to defend core positions during pullbacks.",
    institutionalContext: "Constitutes the structural baseline for Stage 2 trend qualification.",
  },
  EMA20: {
    term: "20-Day Exponential Moving Average",
    shortName: "20 EMA",
    category: "TECHNICAL",
    plainEnglishSummary: "A responsive short-term trendline tracking price momentum over roughly one month.",
    detailedDefinition: "Exponentially weighted moving average giving higher weight to recent prices. Serves as dynamic pullback support in active momentum regimes.",
    institutionalContext: "Used in Raschke 20 EMA Pullback intraday trade setups.",
  },
  VWAP: {
    term: "Volume-Weighted Average Price",
    shortName: "VWAP",
    category: "EXECUTION",
    plainEnglishSummary: "The true average price paid by all market participants today, weighted by trading volume.",
    detailedDefinition: "Cumulative dollar value traded divided by cumulative volume throughout the market session. Institutions measure execution efficiency against VWAP.",
    institutionalContext: "Actionable intraday setups require price holding above session VWAP.",
  },
  PIOTROSKI: {
    term: "Piotroski F-Score",
    shortName: "Piotroski F",
    category: "FUNDAMENTAL",
    plainEnglishSummary: "A 0-to-9 score grading a company's financial health across profitability, debt, liquidity, and efficiency.",
    detailedDefinition: "Joseph Piotroski's 9-factor fundamental scoring framework. Scores 8–9 indicate strong financial solvency; scores 0–2 indicate distress risk.",
    institutionalContext: "Core solvency gate in the 4-pillar Confluence Engine.",
  },
  ALTMAN_Z: {
    term: "Altman Z-Score",
    shortName: "Altman Z",
    category: "FUNDAMENTAL",
    plainEnglishSummary: "A formula assessing corporate credit health and bankruptcy probability within two years.",
    detailedDefinition: "Edward Altman's multivariate credit risk model combining working capital, retained earnings, EBIT, market value, and asset turnover.",
    institutionalContext: "Scores > 2.99 indicate Safe Zone; < 1.81 indicate Distress Zone.",
  },
  CORNISH_FISHER_VAR: {
    term: "Cornish-Fisher Value at Risk",
    shortName: "M-VaR",
    category: "RISK",
    plainEnglishSummary: "An advanced risk estimate of worst-case portfolio loss that accounts for sudden market crashes and fat tails.",
    detailedDefinition: "Expansion adjusting parametric VaR for higher-order moments (skewness and kurtosis), correcting standard normal distribution underestimations during market dislocations.",
    institutionalContext: "Primary portfolio tail-risk constraint in the Portfolio Hub.",
  },
  R_MULTIPLE: {
    term: "Risk-Reward Multiple",
    shortName: "R-Multiple",
    category: "EXECUTION",
    plainEnglishSummary: "The ratio of potential profit to initial capital risked at the stop loss (e.g. 2.5R = $2.50 profit for every $1.00 risked).",
    detailedDefinition: "Dr. Van Tharp's standardized trade evaluation unit. 1R represents the distance between entry price and initial protective stop loss.",
    institutionalContext: "Minimum 1.85R planned reward-to-risk required for actionable trade plan qualification.",
  },
  SORTINO: {
    term: "Sortino Ratio",
    shortName: "Sortino",
    category: "RISK",
    plainEnglishSummary: "A return-per-risk score that penalizes only downside losses, ignoring upside volatility.",
    detailedDefinition: "Variation of the Sharpe ratio differentiating harmful volatility from total volatility by dividing excess return by downside deviation.",
    institutionalContext: "Used in historical performance evaluation and governor calibration.",
  },
  AMIHUD: {
    term: "Amihud Illiquidity Measure",
    shortName: "Amihud Ratio",
    category: "RISK",
    plainEnglishSummary: "A measure of how much price moves per dollar traded; lower numbers mean smoother execution with less slippage.",
    detailedDefinition: "Average ratio of absolute daily return to daily dollar volume, quantifying price impact per unit of volume traded.",
    institutionalContext: "Execution safety filter guarding against illiquid micro-cap market impact.",
  },
  RVOL: {
    term: "Relative Volume",
    shortName: "RVOL",
    category: "TECHNICAL",
    plainEnglishSummary: "Today's trading volume compared to average volume for the same time of day; above 1.5x shows institutional buying.",
    detailedDefinition: "Ratio of current intraday or daily volume to the 50-day moving average volume for the matching time window.",
    institutionalContext: "Confirming condition for Stage 2 breakouts and institutional accumulation.",
  },
  PIVOT: {
    term: "Entry Pivot Level",
    shortName: "Pivot",
    category: "EXECUTION",
    plainEnglishSummary: "The exact price trigger point where a stock breaks out of its consolidation base on heavy volume.",
    detailedDefinition: "The optimal, low-risk buy point at the high of the final contraction in a base. Buying within 1–2% of the pivot minimizes drawdowns.",
    institutionalContext: "Base model execution anchor in Trade Plan and Confluence Engine.",
  },
  STOP_LOSS: {
    term: "Protective Stop Loss",
    shortName: "Stop Loss",
    category: "EXECUTION",
    plainEnglishSummary: "The predetermined price floor where you sell to cut losses and protect capital if the trade fails.",
    detailedDefinition: "Deterministic risk-mitigation floor placed immediately upon entry, structurally anchored to recent swing lows or max 7–8% risk floor.",
    institutionalContext: "Non-negotiable requirement for position sizing calculation.",
  },
  STAGE_2: {
    term: "Stage 2 Advancing Phase",
    shortName: "Stage 2",
    category: "TECHNICAL",
    plainEnglishSummary: "A sustained structural uptrend where institutional funds are actively accumulating shares.",
    detailedDefinition: "Stan Weinstein's cyclical trend classification: price trades above rising 50-day and 200-day moving averages with expanding volume on up days.",
    institutionalContext: "Prerequisite market condition for long growth equity positions.",
  },
  CONFLUENCE: {
    term: "Multi-Model Confluence Score",
    shortName: "Confluence",
    category: "RISK",
    plainEnglishSummary: "A 0-to-100 rating measuring agreement across technical, fundamental, ownership, and macro models.",
    detailedDefinition: "Composite conviction metric synthesizing 4 pillars: Price Structure (35%), Valuation/Solvency (25%), Institutional Ownership (20%), and Macro Regime (20%). Requires 4 verified pillars for high-conviction labels.",
    institutionalContext: "Primary decision-support metric across Radar, Analysis, and Trade Plan.",
  },
};

/**
 * Helper to retrieve a jargon definition by key or normalized acronym
 */
export function getJargonDefinition(key: string): JargonDefinition | undefined {
  if (!key) return undefined;
  if (JARGON_GLOSSARY[key]) return JARGON_GLOSSARY[key];

  const normKey = key.toUpperCase().replace(/[\s\-_]/g, "");

  for (const [k, v] of Object.entries(JARGON_GLOSSARY)) {
    const normK = k.toUpperCase().replace(/[\s\-_]/g, "");
    const normShort = v.shortName.toUpperCase().replace(/[\s\-_]/g, "");
    const normTerm = v.term.toUpperCase().replace(/[\s\-_]/g, "");
    if (normK === normKey || normShort === normKey || normTerm === normKey) {
      return v;
    }
  }
  return undefined;
}
