export interface GlossaryTerm {
  slug: string;
  name: string;
  shortDefinition: string;
  metaTitle?: string;
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
    metaTitle: "Autoregressive with Exogenous Inputs (ARX Model) | Quantitative Dictionary",
    shortDefinition: "A foundational econometric time-series model that forecasts a financial variable using its own historical values alongside observable external macroeconomic drivers.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "y_t = c + \\sum_{i=1}^p \\phi_i y_{t-i} + \\sum_{j=1}^m \\beta_j x_{t-j} + \\epsilon_t",
    detailedExplanation: [
      "In quantitative finance, an Autoregressive with Exogenous Inputs (ARX) model expands classical autoregression by incorporating observable external covariates ($x_t$), such as benchmark interest rate yields, oil prices, or systemic credit spreads. Rather than assuming an asset moves in isolation, ARX models quantify how macroeconomic liquidity conditions transmit momentum into asset prices.",
      "The parameter $p$ specifies the number of lagged asset observations, while $m$ denotes the exogenous delay order—measuring the speed at which external macroeconomic shifts affect the target variable. The error term $\\epsilon_t$ represents white-noise residual variance.",
      "Unlike opaque deep-learning models, ARX models produce deterministic, statistically auditable coefficients ($\\phi_i, \\beta_j$). This mathematical transparency allows risk managers to verify exact causal drivers, run parameter stability tests, and satisfy institutional governance audits without latency or model hallucination."
    ],
    arxApplication: "ARX Terminal utilizes autoregressive exogenous principles to dynamically condition individual equity volatility bands upon systemic macro drivers, including the Federal Reserve 10Y-2Y Treasury yield curve spread and high-yield OAS credit default spreads.",
    keyTakeaway: "ARX models bridge asset-level price history with macro credit conditions, delivering transparent, auditable forecasts without black-box complexity.",
    relatedTerms: ["cornish-fisher-var", "amihud-illiquidity", "kupiec-pof-test"],
    externalReferences: [
      { title: "Box & Jenkins Time Series Analysis", url: "https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model" }
    ],
    relatedRoute: { title: "Explore ARX Opportunity Radar", url: "/radar/" }
  },
  {
    slug: "minervini-vcp",
    name: "Mark Minervini Volatility Contraction Pattern (VCP)",
    metaTitle: "Mark Minervini Volatility Contraction Pattern (VCP) | ARX Strategy Guide",
    shortDefinition: "An institutional swing accumulation pattern where successive price pullbacks narrow within an advancing Stage 2 uptrend, signaling supply exhaustion prior to a pivot breakout.",
    category: "Algorithmic Setups & Execution",
    latexFormula: "\\Delta_k = \\frac{\\text{High}_k - \\text{Low}_k}{\\text{High}_k} \\quad \\text{where } \\Delta_1 > \\Delta_2 > \\dots > \\Delta_n \\text{ and } \\text{Vol}(\\Delta_n) < 0.50 \\times \\overline{\\text{Vol}}_{50}",
    detailedExplanation: [
      "Developed by U.S. Investing Champion Mark Minervini, the Volatility Contraction Pattern (VCP) reflects institutional supply absorption within an established Stage 2 uptrend. As a stock consolidates, motivated sellers liquidate their positions during early, wider corrections. With each subsequent wave, the magnitude of selling diminishes as shares migrate into institutional hands.",
      "A classic VCP base forms through two to four successive contraction waves. For example, an initial pullback of 18% ($T_1$) is followed by a recovery, then a shallower 9% pullback ($T_2$), and finally a tight 3.5% consolidation ($T_3$). Daily trading volume systematically dries up during the final contraction, confirming that overhead floating supply is exhausted.",
      "The setup culminates at an actionable pivot point. Because volatility has contracted, a stop-loss order can be placed directly beneath the final pivot low (typically within 3% to 5%), providing favorable asymmetric reward-to-risk geometry."
    ],
    arxApplication: "ARX Terminal continuously scans the equity universe for Stage 2 Trend Template alignment and identifies algorithmic VCP pivots, providing traders with calibrated Buy Zone corridors and Turtle ATR invalidation stops.",
    keyTakeaway: "VCP setups define precise entry corridors by identifying the exact moment overhead selling dries up in an established uptrend.",
    relatedTerms: ["twenty-ema-pullback", "turtle-atr-trailing-stop", "piotroski-f-score"],
    relatedRoute: { title: "View Active Minervini VCP Candidates", url: "/radar/" }
  },
  {
    slug: "cornish-fisher-var",
    name: "Cornish-Fisher Modified Value-at-Risk (M-VaR)",
    metaTitle: "Cornish-Fisher Modified Value-at-Risk (M-VaR) | Risk Analytics",
    shortDefinition: "An econometric risk metric that calculates downside Value-at-Risk by adjusting standard Gaussian quantiles for empirical skewness and excess kurtosis.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "\\tilde{z}_\\alpha = z_\\alpha + \\frac{S}{6}(z_\\alpha^2 - 1) + \\frac{K}{24}(z_\\alpha^3 - 3z_\\alpha) - \\frac{S^2}{36}(2z_\\alpha^3 - 5z_\\alpha)",
    detailedExplanation: [
      "Standard Gaussian Value-at-Risk (VaR) assumes that financial asset returns follow a symmetric normal distribution. In real-world capital markets, equity distributions exhibit severe negative skewness (crash hazard) and high excess kurtosis (fat tails). Standard VaR systematically underestimates potential downside losses during market distress.",
      "The Cornish-Fisher expansion applies a polynomial adjustment to the standard normal critical value $z_\\alpha$ using sample skewness ($S$) and excess kurtosis ($K$). For example, in a portfolio with $S = -0.85$ and $K = 4.20$, the standard 99% Gaussian quantile ($z_{0.01} = -2.326$) adjusts to a modified quantile of $\\tilde{z} = -3.412$, expanding estimated 1-day potential losses from -$4,650 to -$6,825.",
      "This modified critical value allows portfolio managers to price tail-risk exposure accurately without relying on computationally demanding Monte Carlo simulations. However, risk managers must note that the polynomial expansion can become non-monotonic under extreme skewness ($|S| > 1.5$), requiring bounded boundary validation."
    ],
    arxApplication: "ARX Terminal computes 95% and 99% Cornish-Fisher Modified VaR across individual assets and multi-asset portfolios, incorporating Kupiec POF backtest validation for historical verification.",
    keyTakeaway: "Cornish-Fisher M-VaR prevents dangerous underestimation of downside losses by explicitly incorporating market crash skewness and fat-tailed kurtosis.",
    relatedTerms: ["arx-model", "kupiec-pof-test", "amihud-illiquidity"],
    relatedRoute: { title: "Analyze Portfolio Cornish-Fisher VaR", url: "/portfolio/" }
  },
  {
    slug: "stock-act",
    name: "Stop Trading on Congressional Knowledge (STOCK) Act of 2012",
    metaTitle: "STOCK Act Legal Disclosures & Tracking | Smart Money Forensics",
    shortDefinition: "A U.S. federal statute (Public Law 112-105) requiring members of Congress to publicly disclose securities transactions within 45 days of execution.",
    category: "Statutory & Smart Money Forensics",
    latexFormula: "\\Delta t_{\\text{disclosure}} = t_{\\text{filing}} - t_{\\text{execution}} \\le 45 \\text{ days}",
    detailedExplanation: [
      "Enacted in April 2012, the Stop Trading on Congressional Knowledge (STOCK) Act established that members of Congress, judicial officers, and executive branch officials are subject to insider trading laws under Section 10(b) of the Securities Exchange Act of 1934.",
      "Lawmakers are legally required to file Periodic Transaction Reports (PTRs) within 30 days of receiving notice of a trade, and no later than 45 days after the transaction date, for any transaction exceeding $1,000 made by themselves, their spouses, or dependent children.",
      "Because the 45-day statutory window represents a significant delay in modern financial markets, raw disclosures cannot be treated as real-time copy signals. ARX evaluates disclosures by measuring the lawmaker's committee jurisdiction (such as Armed Services members purchasing defense contractors) and applies an automated time-decay discount to aging filings."
    ],
    arxApplication: "ARX Terminal tracks Senate and House PTR filings, evaluates legislative alignment scores (0–100), and penalizes aging filings via an automated exponential staleness decay function.",
    keyTakeaway: "Congressional disclosures provide valuable forensic insight into legislative alignment, but must be adjusted for filing latency to avoid chasing obsolete news.",
    relatedTerms: ["late-filer-decay", "amihud-illiquidity"],
    relatedRoute: { title: "Explore Congressional Smart Money Radar", url: "/radar/" }
  },
  {
    slug: "amihud-illiquidity",
    name: "Amihud Illiquidity Ratio (Price Impact)",
    metaTitle: "Amihud Illiquidity Ratio & Market Microstructure | ARX Dictionary",
    shortDefinition: "An econometric microstructure metric that measures the absolute percentage price movement per dollar of daily trading volume.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "\\text{ILLIQ}_t = \\frac{1}{N} \\sum_{d=1}^N \\frac{|R_d|}{\\text{Volume}_d \\times \\text{Price}_d} \\times 10^6",
    detailedExplanation: [
      "Introduced by Yakov Amihud in 2002, the Amihud illiquidity ratio measures market depth and execution friction. It quantifies how much an asset's price moves in response to a given dollar amount of trading volume.",
      "A high Amihud ratio indicates an illiquid market where modest institutional transactions cause large price swings, resulting in severe execution slippage. A low ratio indicates deep, resilient liquidity where substantial block trades are absorbed with minimal price impact.",
      "Unlike simple volume tallies, the Amihud ratio directly connects trading volume with observed price distortion, helping quantitative analysts distinguish genuine institutional accumulation from low-volume speculative noise."
    ],
    arxApplication: "ARX Terminal's LiquidityGuard continuously monitors 20-day scaled Amihud ratios, alerting traders when proposed position sizes risk causing adverse market impact.",
    keyTakeaway: "The Amihud ratio measures true execution friction, ensuring traders do not mistake low-volume volatility for actionable liquidity.",
    relatedTerms: ["arx-model", "turtle-atr-trailing-stop"],
    relatedRoute: { title: "Check Execution Friction on Radar", url: "/radar/" }
  },
  {
    slug: "turtle-atr-trailing-stop",
    name: "Turtle Trading Average True Range (ATR) Trailing Stop",
    metaTitle: "Turtle Trading ATR Volatility Stop-Loss | ARX Execution Models",
    shortDefinition: "A dynamic risk management framework that calibrates stop-loss distances to an asset's empirical volatility using the 14-period Average True Range.",
    category: "Algorithmic Setups & Execution",
    latexFormula: "\\text{Stop Loss} = \\text{Entry} - (k \\times \\text{ATR}_{14}) \\quad \\text{where } k \\in [1.5, 2.5]",
    detailedExplanation: [
      "Originating from Richard Dennis and William Eckhardt's 1983 Turtle Trading experiment, ATR-based stops replace arbitrary percentage thresholds with volatility-adjusted boundaries. Average True Range measures the true price variance over 14 sessions, accounting for overnight gaps.",
      "Static percentage stops (such as always risking 7%) create structural vulnerabilities: they stop out high-volatility growth stocks prematurely during normal market noise, while giving low-volatility defensive stocks far too much room to decline before triggering an exit.",
      "By setting the stop-loss at a multiple of ATR (typically 1.5x to 2.5x), the stop expands during turbulent periods and tightens during quiet consolidations, preserving capital while giving winning trends room to develop."
    ],
    arxApplication: "ARX Terminal derives 14-day ATR corridors across all actionable setups, calculating exact dollar invalidation levels, minimum 2:1 reward-to-risk ratios, and stepped profit targets.",
    keyTakeaway: "ATR trailing stops eliminate arbitrary percentage rules by tailoring capital defense directly to each stock's empirical volatility signature.",
    relatedTerms: ["minervini-vcp", "twenty-ema-pullback"],
    relatedRoute: { title: "Inspect Live Execution Risk Ladders", url: "/radar/" }
  },
  {
    slug: "piotroski-f-score",
    name: "Piotroski 9-Point Fundamental Accounting Score",
    metaTitle: "Piotroski 9-Point Fundamental Score | Balance Sheet Forensics",
    shortDefinition: "A discrete score between 0 and 9 based on nine accounting criteria that evaluates the financial health, earnings quality, and operational efficiency of a company.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "\\text{F-Score} = \\sum_{i=1}^9 C_i \\quad \\text{where } C_i \\in \\{0, 1\\}",
    detailedExplanation: [
      "Developed in 2000 by Stanford accounting professor Joseph Piotroski, the F-Score evaluates companies across three critical dimensions: Profitability (positive ROA, positive Operating Cash Flow, cash flow exceeding net income), Leverage and Liquidity (decreasing long-term debt, improving current ratio, zero share dilution), and Operating Efficiency (improving gross margin, higher asset turnover).",
      "Each satisfied test awards exactly 1 point. Companies scoring 8 or 9 demonstrate robust balance-sheet health and authentic cash generation. Companies scoring 0 to 3 carry high risks of accounting distortion, structural margin decay, or financial distress.",
      "In quantitative screening, requiring an F-Score of $\\ge 7$ filters out speculative value traps that appear statistically cheap based on traditional price-to-earnings multiples alone."
    ],
    arxApplication: "ARX Terminal displays Piotroski ratings across all screened assets, ensuring technical momentum setups possess institutional balance-sheet confirmation.",
    keyTakeaway: "The Piotroski F-Score provides a transparent, nine-point financial screen that separates authentic corporate profitability from accounting manipulation.",
    relatedTerms: ["minervini-vcp", "arx-model"],
    relatedRoute: { title: "Filter Candidates by Fundamental Health", url: "/radar/" }
  },
  {
    slug: "twenty-ema-pullback",
    name: "Linda Raschke 20-Period EMA Pullback Model",
    metaTitle: "20-Day EMA Pullback Swing Model | Algorithmic Trend Following",
    shortDefinition: "A trend-following swing setup that identifies shallow, low-risk pullbacks to an advancing 20-day Exponential Moving Average within an established uptrend.",
    category: "Algorithmic Setups & Execution",
    latexFormula: "\\text{Buy Corridor: } \\text{Spot} \\in [\\text{EMA}_{20} - 0.5\\times\\text{ATR}_{14}, \\text{EMA}_{20} + 0.5\\times\\text{ATR}_{14}] \\quad \\text{with } \\text{Slope}(\\text{EMA}_{20}) > 0",
    detailedExplanation: [
      "Popularized by Market Wizard Linda Bradford Raschke, the 20 EMA pullback strategy capitalizes on institutional trend continuation. During an established markup phase, institutions routinely use shallow dips toward the 20-day EMA to add to core positions without chasing market highs.",
      "The setup requires a confirmed Stage 2 uptrend with an upward-sloping 20-day EMA. The entry corridor is defined as price touching or testing the moving average. Invalidation occurs immediately if price closes beneath the recent swing low or if the moving average slope flattens.",
      "By waiting for price to retrace to the 20 EMA rather than buying breakout extensions, traders achieve favorable asymmetric entry geometry while minimizing downside risk."
    ],
    arxApplication: "ARX Terminal classifies setups daily into four discrete execution states: IN_BUY_ZONE, APPROACHING_TARGET, WAITING_PULLBACK, and STOPPED_OUT based on proximity to the 20 EMA corridor.",
    keyTakeaway: "The 20 EMA pullback model prevents chasing extended runners by catching institutional re-accumulation at prime structural support.",
    relatedTerms: ["minervini-vcp", "turtle-atr-trailing-stop"],
    relatedRoute: { title: "View Active 20 EMA Pullback Setups", url: "/radar/" }
  },
  {
    slug: "kupiec-pof-test",
    name: "Kupiec Proportion of Failures (POF) Test",
    metaTitle: "Kupiec POF Statistical Risk Backtest | Risk Model Governance",
    shortDefinition: "A formal likelihood-ratio test used by financial regulators and risk managers to evaluate whether a Value-at-Risk (VaR) model accurately forecasts real-world portfolio losses.",
    category: "Econometric & Mathematical Modeling",
    latexFormula: "LR_{\\text{POF}} = -2 \\ln \\left( \\frac{p^x (1-p)^{N-x}}{(x/N)^x (1 - x/N)^{N-x}} \\right) \\sim \\chi^2(1)",
    detailedExplanation: [
      "Introduced by Paul Kupiec in 1995, the Proportion of Failures (POF) test evaluates whether a Value-at-Risk model's observed exception rate matches its theoretical confidence level. For example, a 95% VaR model should experience losses exceeding the threshold on exactly 5% of trading days.",
      "The test constructs a likelihood-ratio statistic ($LR_{\\text{POF}}$) comparing the observed number of breaches ($x$) over sample period ($N$) against the expected failure probability ($p$). Under the null hypothesis that the model is accurately calibrated, the test statistic follows a chi-square distribution with 1 degree of freedom (critical value 3.841 at the 5% significance level).",
      "If $LR_{\\text{POF}} > 3.841$, the model is rejected. Too many breaches indicate that the model understates risk, exposing the firm to unhedged losses. Too few breaches indicate the model is overly conservative, tying up excessive defensive capital."
    ],
    arxApplication: "ARX Terminal performs automated Kupiec POF calibration on historical return series when sample depth allows ($N \\ge 35$), validating that downside risk bounds remain statistically honest.",
    keyTakeaway: "Kupiec tests ensure risk models remain statistically honest and self-calibrating rather than relying on unverified assumptions.",
    relatedTerms: ["cornish-fisher-var", "arx-model"],
    relatedRoute: { title: "Review Model Governance & Calibration", url: "/guide/" }
  },
  {
    slug: "late-filer-decay",
    name: "Congressional Disclosure Staleness Decay Function",
    metaTitle: "Disclosure Staleness Decay Function | Smart Money Forensics",
    shortDefinition: "A mathematical discounting function that progressively reduces the actionable weighting of political insider filings as the latency between trade execution and public disclosure widens.",
    category: "Statutory & Smart Money Forensics",
    latexFormula: "W(t) = \\max\\left(0, 1 - \\lambda \\cdot (t_{\\text{filed}} - t_{\\text{execution}})\\right) \\quad \\text{where } \\lambda = \\begin{cases} 0.00 & \\Delta t \\le 15 \\text{ days} \\\\ 0.05 & 16 < \\Delta t \\le 30 \\text{ days} \\\\ 0.16 & 31 < \\Delta t \\le 45 \\text{ days} \\\\ 0.32 & \\Delta t > 45 \\text{ days} \\end{cases}",
    detailedExplanation: [
      "Under the 2012 STOCK Act, lawmakers frequently disclose securities transactions weeks or months after execution. While a fresh filing may provide an actionable market signal, disclosures filed 45+ days late typically reflect price movements that have already occurred.",
      "A quantitative staleness decay function applies a tiered discount penalty to aged filings: disclosures within 15 days retain full weight (0 pt penalty), filings between 16 and 30 days receive a 5-point deduction, filings between 31 and 45 days receive a 16-point deduction, and filings past the statutory 45-day deadline receive a severe 32-point penalty.",
      "This mathematical decay protects investors from buying extended stocks on stale disclosures that larger institutional market participants have already priced in."
    ],
    arxApplication: "ARX Terminal scores politician filings on a 0-100 Legislative Alignment Index and routes persistent late-reporters to the Congressional Late-Filer Registry.",
    keyTakeaway: "Filing latency destroys informational edge; quantitative time decay prevents investors from acting on obsolete disclosures.",
    relatedTerms: ["stock-act", "amihud-illiquidity"],
    relatedRoute: { title: "Inspect the Late-Filer Hall of Fame", url: "/radar/" }
  }
];
