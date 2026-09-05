export interface GlossaryTerm {
  slug: string;
  name: string;
  shortDefinition: string;
  category: "Econometric & Mathematical Modeling" | "Algorithmic Setups & Execution" | "Statutory & Smart Money Forensics";
  latexFormula?: string;
  detailedExplanation: string[];
  arxApplication: string;
  keyTakeaway: string;
  relatedTerms: string[];
  externalReferences?: { title: string; url: string }[];
  relatedRoute?: { title: string; url: string };
}

export const GLOSSARY_CATALOG: GlossaryTerm[] = [
  {
    slug: "arx-model",
    name: "Autoregressive with Exogenous Inputs (ARX Model)",
    shortDefinition: "A foundational econometric time-series model that predicts a target financial variable using both its own historical lagged values and external (exogenous) market drivers.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "y_t = c + \\sum_{i=1}^p \\phi_i y_{t-i} + \\sum_{j=1}^m \\beta_j x_{t-j} + \\epsilon_t",
    detailedExplanation: [
      "In quantitative finance and econometrics, an Autoregressive with Exogenous Inputs (ARX) model expands classical autoregression by incorporating external covariate time-series (such as interest rate yields, oil prices, or market volatility) to improve forecast accuracy.",
      "The parameter 'p' denotes the number of autoregressive lags (how many past values of y affect the current state), while 'm' represents the exogenous delay order (how external signals 'x' transfer momentum into the system).",
      "Unlike pure black-box deep learning models, ARX models provide deterministic, statistically auditable coefficients that allow risk managers to trace exact causal contributions without latency or hallucination."
    ],
    arxApplication: "ARX Terminal uses autoregressive exogenous principles to dynamically condition equity candidate volatility bands upon macro exogenous drivers, including the Federal Reserve 10Y-2Y yield curve spread and high-yield OAS credit default spreads.",
    keyTakeaway: "ARX models bridge historical asset momentum with external macroeconomic conditions, delivering auditable mathematical forecasts without black-box opacity.",
    relatedTerms: ["cornish-fisher-var", "amihud-illiquidity", "kupiec-pof-test"],
    externalReferences: [
      { title: "Box & Jenkins Time Series Analysis", url: "https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model" }
    ],
    relatedRoute: { title: "Explore ARX Multi-Factor Screener", url: "/screener/" }
  },
  {
    slug: "minervini-vcp",
    name: "Mark Minervini Volatility Contraction Pattern (VCP)",
    shortDefinition: "An institutional swing accumulation pattern characterized by progressive contractions in price volatility paired with volume dry-up prior to an asymmetric pivot breakout.",
    category: "Algorithmic Setups & Execution",
    latexFormula: "\\text{Contraction Ratio: } \\Delta_k = \\frac{\\text{High}_k - \\text{Low}_k}{\\text{High}_k} \\quad \\text{where } \\Delta_1 > \\Delta_2 > \\dots > \\Delta_n",
    detailedExplanation: [
      "Pioneered by U.S. Investing Champion Mark Minervini, the Volatility Contraction Pattern (VCP) visually represents the absorption of supply by institutional buyers in an advancing Stage 2 uptrend.",
      "As a stock consolidates, each successive contraction wave exhibits a smaller percentage drawdown (e.g., -16% -> -8% -> -3%), indicating that motivated sellers are exhausted and shares are migrating into strong hands.",
      "The final contraction wave creates a tight pivot point where risk can be mathematically defined with tight stop-loss invalidation (typically within 3% to 6%)."
    ],
    arxApplication: "ARX Terminal scans equities daily for Stage 2 Trend Template alignment and identifies algorithmic VCP pivots, providing traders with precise Buy Zone corridors and Turtle ATR invalidation stops.",
    keyTakeaway: "VCP setups eliminate guessing by identifying the exact moment institutional supply dry-up creates asymmetric risk-reward breakout geometry.",
    relatedTerms: ["twenty-ema-pullback", "turtle-atr-trailing-stop", "piotroski-f-score"],
    relatedRoute: { title: "View Active Minervini VCP Candidates", url: "/strategy/minervini-vcp/" }
  },
  {
    slug: "cornish-fisher-var",
    name: "Cornish-Fisher Modified Value-at-Risk (M-VaR)",
    shortDefinition: "An advanced statistical risk metric that calculates downside Value-at-Risk by adjusting standard Gaussian quantiles for skewness and fat-tailed excess kurtosis.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "\\tilde{z}_\\alpha = z_\\alpha + \\frac{S}{6}(z_\\alpha^2 - 1) + \\frac{K}{24}(z_\\alpha^3 - 3z_\\alpha) - \\frac{S^2}{36}(2z_\\alpha^3 - 5z_\\alpha)",
    detailedExplanation: [
      "Traditional Value-at-Risk (VaR) assumes financial returns follow a standard bell-curve normal distribution. In reality, financial markets exhibit pronounced negative skewness (crash hazard) and fat-tailed leptokurtosis (black swan events).",
      "The Cornish-Fisher expansion applies a polynomial adjustment to the normal critical value z_alpha using sample skewness (S) and excess kurtosis (K), capturing true tail vulnerability without requiring computationally intensive Monte Carlo simulations.",
      "This modified quantile allows portfolio managers to estimate downside capital at risk with institutional precision during market shocks."
    ],
    arxApplication: "ARX Terminal computes 95% and 99% Cornish-Fisher Modified VaR across every individual stock and multi-asset portfolio, continuously auditing forecast accuracy through automated Kupiec exception tests.",
    keyTakeaway: "Cornish-Fisher M-VaR prevents catastrophic underestimation of downside risk by explicitly accounting for market fat tails and asymmetric crash skewness.",
    relatedTerms: ["arx-model", "kupiec-pof-test", "amihud-illiquidity"],
    relatedRoute: { title: "Analyze Portfolio Cornish-Fisher VaR", url: "/portfolio/" }
  },
  {
    slug: "stock-act",
    name: "Stop Trading on Congressional Knowledge (STOCK) Act of 2012",
    shortDefinition: "A U.S. federal statute (Public Law 112-105) prohibiting members of Congress and legislative staff from using non-public information for private securities trading.",
    category: "Statutory & Smart Money Forensics",
    latexFormula: "\\text{Statutory Window: } t_{\\text{filing}} - t_{\\text{transaction}} \\le 45 \\text{ days}",
    detailedExplanation: [
      "Enacted in April 2012, the STOCK Act affirmed that members of Congress, judicial officers, and executive branch officials are subject to insider trading prohibitions under the Securities Exchange Act of 1934.",
      "The law requires lawmakers to file Periodic Transaction Reports (PTRs) within 30 to 45 days of any securities transaction exceeding $1,000 made by themselves, their spouses, or dependent children.",
      "While intended to deter conflicts of interest, delays in filing and modest late-filing fines ($200) have led to persistent late disclosures, providing critical forensic signals for retail market observers."
    ],
    arxApplication: "ARX Terminal ingests statutory Senate and House PTR filings in real time, scoring legislative committee jurisdiction overlap (+16 to +32 points) and decaying stale signals via an automated time-decay algorithm.",
    keyTakeaway: "STOCK Act disclosures provide an unprecedented public window into legislative capital movement, but require forensic decay analysis to account for filing latency.",
    relatedTerms: ["late-filer-decay", "amihud-illiquidity"],
    relatedRoute: { title: "Explore Congressional Smart Money Radar", url: "/smart-money/" }
  },
  {
    slug: "amihud-illiquidity",
    name: "Amihud Illiquidity Ratio (Price Impact)",
    shortDefinition: "A classic microstructure econometric measure that calculates the absolute price change per dollar of daily trading volume, measuring liquidity depth.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "\\text{ILLIQ}_t = \\frac{1}{N} \\sum_{d=1}^N \\frac{|R_d|}{\\text{Volume}_d \\times \\text{Price}_d}",
    detailedExplanation: [
      "Introduced by Yakov Amihud in 2002, the Amihud illiquidity metric assesses how easily an asset can be absorbed by the market without causing adverse slippage.",
      "A high Amihud ratio indicates an illiquid asset where even modest institutional orders will move the market against the buyer, while a low ratio signifies deep liquidity capable of handling institutional block sweeps.",
      "Unlike simple volume tallies, the Amihud ratio directly connects trading activity with realized price distortion."
    ],
    arxApplication: "ARX Terminal's LiquidityGuard operates as a shadow observer evaluating 20-day scaled Amihud illiquidity and dollar volume, warning traders when order size threatens execution slippage.",
    keyTakeaway: "The Amihud ratio measures true execution friction, ensuring traders do not mistake high volatility for actionable liquidity.",
    relatedTerms: ["arx-model", "turtle-atr-trailing-stop"],
    relatedRoute: { title: "Check Execution Friction on Screener", url: "/screener/" }
  },
  {
    slug: "turtle-atr-trailing-stop",
    name: "Turtle Trading Average True Range (ATR) Trailing Stop",
    shortDefinition: "A dynamic risk management framework that calibrates stop-loss distances to the underlying volatility of an asset using the 14-period Average True Range.",
    category: "Algorithmic Setups & Execution",
    latexFormula: "\\text{Stop Loss} = \\text{Entry} - (k \\times \\text{ATR}_{14}) \\quad \\text{where } k \\in [1.5, 2.5]",
    detailedExplanation: [
      "Originating from Richard Dennis and William Eckhardt's legendary 1983 Turtle Trading experiment, ATR-based volatility stops adjust trade invalidation levels to market noise.",
      "Fixed percentage stops (e.g. always -7%) fail because low-beta utility stocks get stopped out too easily, while high-beta tech stocks have their stops set too loose.",
      "By anchoring stop losses to a multiple of ATR (typically 1.5x to 2.5x), the stop allows normal statistical breathing room while protecting capital against trend reversals."
    ],
    arxApplication: "ARX Terminal dynamically derives 14-day ATR corridors across all actionable setups, calculating exact dollar stops, risk-to-reward ratios (minimum 2.0:1), and dual profit targets.",
    keyTakeaway: "ATR trailing stops eliminate arbitrary percentage rules by tailoring capital defense directly to each stock's empirical volatility signature.",
    relatedTerms: ["minervini-vcp", "twenty-ema-pullback"],
    relatedRoute: { title: "Inspect Live Execution Risk Ladders", url: "/screener/" }
  },
  {
    slug: "piotroski-f-score",
    name: "Piotroski 9-Point Fundamental Accounting Score",
    shortDefinition: "A discrete score between 0 and 9 based on nine accounting criteria that evaluates the financial health, profitability, and operational efficiency of a company.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "\\text{F-Score} = \\sum_{i=1}^9 C_i \\quad \\text{where } C_i \\in \\{0, 1\\}",
    detailedExplanation: [
      "Developed in 2000 by Stanford accounting professor Joseph Piotroski, the F-Score evaluates companies across three critical dimensions: Profitability (positive ROA, CFO > Net Income), Leverage/Liquidity (decreasing debt, higher current ratio), and Operating Efficiency (improving gross margin, asset turnover).",
      "Stocks with scores of 8 or 9 represent fundamentally bulletproof operations, while companies scoring 0 to 3 carry high risks of structural decay or distress.",
      "In quantitative screening, pairing technical breakouts with high Piotroski scores dramatically filters out value traps."
    ],
    arxApplication: "ARX Terminal displays Piotroski ratings across all screened assets, ensuring technical momentum setups possess institutional balance-sheet confirmation.",
    keyTakeaway: "The Piotroski F-Score provides a rapid, statistically robust test that verifies whether a rising stock possesses authentic earnings quality.",
    relatedTerms: ["minervini-vcp", "arx-model"],
    relatedRoute: { title: "Filter Candidates by Fundamental Health", url: "/screener/" }
  },
  {
    slug: "twenty-ema-pullback",
    name: "Linda Raschke 20-Period EMA Pullback Model",
    shortDefinition: "A high-probability swing trading setup that identifies shallow counter-trend pullbacks into an advancing 20-day Exponential Moving Average in strong trends.",
    category: "Algorithmic Setups & Execution",
    latexFormula: "\\text{Buy Trigger: } \\text{Price} \\in [\\text{EMA}_{20} - 0.5\\times\\text{ATR}, \\text{EMA}_{20} + 0.5\\times\\text{ATR}] \\quad \\text{and } \\text{Slope}(\\text{EMA}_{20}) > 0",
    detailedExplanation: [
      "Popularized by Market Wizard Linda Bradford Raschke, the 20 EMA pullback strategy exploits institutional trend persistence.",
      "When an asset enters an aggressive markup phase, institutions use shallow dips toward the rising 20 EMA to accumulate shares without bidding prices higher.",
      "The setup provides a mathematically defined entry zone with an immediate invalidation level if the moving average slope flattens or breaks down."
    ],
    arxApplication: "ARX Terminal tracks 20 EMA pullback candidates daily, classifying assets into four execution states: IN_BUY_ZONE, APPROACHING_TARGET, WAITING_PULLBACK, or STOPPED_OUT.",
    keyTakeaway: "The 20 EMA pullback avoids chasing extended runners by catching institutional re-accumulation at prime structural support.",
    relatedTerms: ["minervini-vcp", "turtle-atr-trailing-stop"],
    relatedRoute: { title: "View Active 20 EMA Pullback Setups", url: "/strategy/minervini-vcp/" }
  },
  {
    slug: "kupiec-pof-test",
    name: "Kupiec Proportion of Failures (POF) Test",
    shortDefinition: "A formal statistical likelihood ratio test used by financial regulators and risk managers to evaluate whether a Value-at-Risk (VaR) model is calibrated accurately.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "LR_{\\text{POF}} = -2 \\ln \\left( \\frac{p^x (1-p)^{N-x}}{(x/N)^x (1 - x/N)^{N-x}} \\right) \\sim \\chi^2(1)",
    detailedExplanation: [
      "Introduced by Paul Kupiec in 1995, the POF test compares the observed number of VaR breaches (x) over a sample period (N) against the model's nominal failure rate (p).",
      "Under the null hypothesis, the model is perfectly calibrated. If the likelihood ratio statistic exceeds the critical chi-square value (3.84 at the 5% significance level), the VaR model is rejected for either underestimating risk (dangerous) or overestimating risk (capital-inefficient).",
      "This test forms the backbone of the Basel Committee's regulatory traffic-light system for internal market risk models."
    ],
    arxApplication: "ARX Terminal features an autonomous Self-Healing Forecast Auditor that runs Kupiec POF tests on historical returns, expanding confidence intervals whenever volatility regimes shift.",
    keyTakeaway: "Kupiec tests ensure risk models remain statistically honest and self-calibrating rather than relying on unverified assumptions.",
    relatedTerms: ["cornish-fisher-var", "arx-model"],
    relatedRoute: { title: "Review Model Governance & Calibration", url: "/guide/" }
  },
  {
    slug: "late-filer-decay",
    name: "Congressional Disclosure Staleness Decay Function",
    shortDefinition: "A quantitative decay formula that progressively reduces the actionable weighting of political insider filings as the latency between trade execution and public filing widens.",
    category: "Statutory & Smart Money Forensics",
    latexFormula: "W(t) = \\max\\left(0, 1 - \\lambda \\cdot (t_{\\text{filed}} - t_{\\text{trade}})\\right) \\quad \\text{where } \\lambda = \\begin{cases} 0.01 & \\Delta t \\le 15 \\\\ 0.03 & 15 < \\Delta t \\le 30 \\\\ 0.06 & \\Delta t > 30 \\end{cases}",
    detailedExplanation: [
      "Under the 2012 STOCK Act, politicians frequently report transactions weeks or months after execution. While a fresh filing may offer actionable market signal, disclosures filed 60+ days late often reflect mean-reverting or obsolete theses.",
      "A quantitative staleness decay function applies a tiered discount factor that penalizes aged disclosures, preventing traders from acting on stale information.",
      "Late filers are also audited for pattern violations, identifying lawmakers who systematically withhold disclosures until after pivotal earnings or regulatory announcements."
    ],
    arxApplication: "ARX Terminal scores politician filings on a 0-100 Legislative Alignment Index and routes persistent late-reporters to the Congressional Late-Filer Hall of Shame.",
    keyTakeaway: "Filing latency destroys informational edge; quantitative time decay prevents retail traders from falling into the late-filing trap.",
    relatedTerms: ["stock-act", "amihud-illiquidity"],
    relatedRoute: { title: "Inspect the Late-Filer Hall of Shame", url: "/smart-money/late-filers/" }
  }
];
