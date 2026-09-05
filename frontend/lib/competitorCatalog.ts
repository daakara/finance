export interface ComparisonFeature {
  featureName: string;
  arxTerminal: string | boolean;
  competitor: string | boolean;
  notes: string;
}

export interface CompetitorComparison {
  slug: string;
  competitorName: string;
  competitorDomain: string;
  tagline: string;
  summary: string;
  pricingComparison: {
    arx: string;
    competitor: string;
  };
  keyAdvantagesArx: string[];
  keyAdvantagesCompetitor: string[];
  featuresMatrix: ComparisonFeature[];
  verdict: string;
}

export const COMPETITOR_CATALOG: CompetitorComparison[] = [
  {
    slug: "quiver-quantitative",
    competitorName: "Quiver Quantitative",
    competitorDomain: "quiverquant.com",
    tagline: "Alternative Data & Congressional Insider Trading Tracking",
    summary: "While Quiver Quantitative focuses primarily on scraping political disclosures and government contract awards for retail awareness, ARX Terminal combines statutory STOCK Act filings with mathematical execution geometry (Minervini VCP, Cornish-Fisher VaR, and Turtle ATR stops) to deliver actionable institutional trading entries rather than passive news feeds.",
    pricingComparison: {
      arx: "Free Institutional Access (Open Terminal)",
      competitor: "$35 – $250 / month subscription tiers"
    },
    keyAdvantagesArx: [
      "Dynamic Execution Geometry (In-Buy-Zone, 20 EMA pullbacks, ATR stops)",
      "Cornish-Fisher Modified Value-at-Risk (M-VaR) downside modeling",
      "Automated Filing Staleness Time-Decay Engine",
      "Piotroski 9-point fundamental accounting confirmation",
      "Open client-side terminal with zero paywalls"
    ],
    keyAdvantagesCompetitor: [
      "Broader non-financial alternative data (government contracts, corporate private jet tracking)",
      "Established community and retail social media syndication",
      "Historical data export API for backtesting researchers"
    ],
    featuresMatrix: [
      {
        featureName: "Congressional STOCK Act Disclosures",
        arxTerminal: "Real-Time PTR Ingestion with Committee Jurisdictional Scoring",
        competitor: "Real-Time Tracking & Politician Portfolios",
        notes: "Both platforms monitor Senate and House disclosures; ARX scores committee jurisdiction overlap (+16 to +32 pts)."
      },
      {
        featureName: "Algorithmic Buy/Sell Execution Corridors",
        arxTerminal: true,
        competitor: false,
        notes: "ARX calculates 3-Stage VCP pivots, 14-day ATR stops, and 2.0:1 minimum risk-reward targets."
      },
      {
        featureName: "Downside Risk Modeling (VaR)",
        arxTerminal: "Cornish-Fisher Modified VaR (95% & 99%) with Kupiec Auditing",
        competitor: false,
        notes: "Quiver provides raw holding returns without statistical tail-risk or kurtosis adjustments."
      },
      {
        featureName: "Filing Staleness Penalty (Anti-Late-Filing Trap)",
        arxTerminal: "Automated Time-Decay Function & Late-Filer Hall of Shame",
        competitor: "Static Filing Dates",
        notes: "ARX actively penalizes transactions filed >30 days after execution to avoid mean-reversion traps."
      },
      {
        featureName: "Core Focus",
        arxTerminal: "Institutional Execution & Risk Mitigation",
        competitor: "Alternative Data Aggregation & News Awareness",
        notes: "ARX converts alternative data into actionable execution levels."
      }
    ],
    verdict: "Choose Quiver Quantitative if you want broad alternative datasets (patents, lobbying, flights); choose ARX Terminal if you want to turn legislative and institutional smart money signals into mathematically defined trading setups with institutional risk corridors."
  },
  {
    slug: "unusual-whales",
    competitorName: "Unusual Whales",
    competitorDomain: "unusualwhales.com",
    tagline: "Options Flow Forensics & Retail Flow Tracking",
    summary: "Unusual Whales is built around high-frequency options order flow, dark pools, and social retail sentiment. ARX Terminal complements order flow by providing structural swing architecture: filtering out options market noise through Minervini Volatility Contraction, Turtle ATR volatility stops, and macroeconomic yield-curve risk conditioning.",
    pricingComparison: {
      arx: "Free Institutional Access (Open Terminal)",
      competitor: "$50 – $100+ / month subscription tiers"
    },
    keyAdvantagesArx: [
      "No options noise or theta decay traps — focuses on high-conviction underlying equity setups",
      "Cornish-Fisher tail risk modeling adjusting for extreme market skewness",
      "Piotroski balance sheet health scores filtering speculative liquidity traps",
      "Permanent free access without subscription tiers"
    ],
    keyAdvantagesCompetitor: [
      "Direct streaming options tape and sweep alerts",
      "Interactive options profit calculators and Greeks surfaces",
      "Large active Discord trading community"
    ],
    featuresMatrix: [
      {
        featureName: "Congressional Trading Alerts",
        arxTerminal: "Yes (with Jurisdictional Overlay & Decay Penalties)",
        competitor: "Yes (with Social Media Alerts)",
        notes: "Both platforms track politician stock trading."
      },
      {
        featureName: "Primary Trading Asset Focus",
        arxTerminal: "Equities, Swing Setups & Risk-Balanced Portfolios",
        competitor: "Short-Dated Options, Volatility & Day Trading",
        notes: "ARX is optimized for swing and position traders; Unusual Whales targets active options scalpers."
      },
      {
        featureName: "Risk Invalidation Corridors",
        arxTerminal: "Mathematical 14-period Turtle ATR trailing stop ladders",
        competitor: "Discretionary (User-Defined)",
        notes: "ARX provides automated stop-loss prices and risk-to-reward boundaries."
      },
      {
        featureName: "Fundamental Balance Sheet Screening",
        arxTerminal: "Piotroski F-Score, ROIC, and Debt Quality Ratios",
        competitor: "Basic Fundamental Overlays",
        notes: "ARX pairs technical setups with fundamental accounting solvency checks."
      }
    ],
    verdict: "Choose Unusual Whales for intra-day options flow and gamma scalp setups; choose ARX Terminal for institutional-grade swing execution, statutory congressional forensics, and rigorous downside risk modeling."
  },
  {
    slug: "koyfin",
    competitorName: "Koyfin",
    competitorDomain: "koyfin.com",
    tagline: "Modern Macro Financial Analytics & Charting Workstation",
    summary: "Koyfin offers an elegant, data-dense charting terminal designed as a cost-effective alternative to Bloomberg and FactSet. ARX Terminal distinguishes itself by offering automated decision intelligence: rather than forcing traders to construct custom indicator screens, ARX continuously classifies assets into algorithmic buy zones, downside risk quantiles, and legislative insider alignments.",
    pricingComparison: {
      arx: "Free Institutional Access (Open Terminal)",
      competitor: "Free limited tier; $45 – $110 / month for full features"
    },
    keyAdvantagesArx: [
      "Automated Algorithmic Decision States (In-Buy-Zone, Waiting Pullback, Stopped Out)",
      "Statutory Congressional STOCK Act integration with committee oversight matching",
      "Cornish-Fisher M-VaR modeling natively computed on every asset",
      "100% free institutional functionality"
    ],
    keyAdvantagesCompetitor: [
      "Extensive customizable multi-chart layouts and historical financial statement time-series",
      "Consensus analyst estimate revisions and earnings call transcripts",
      "Global coverage across international exchanges"
    ],
    featuresMatrix: [
      {
        featureName: "Charting & Financial Visualizations",
        arxTerminal: "Lightweight Institutional Execution Charts (TradingView/Lightweight-Charts)",
        competitor: "Deep Customizable Macro & Graphing Canvas",
        notes: "Koyfin excels in flexible graphic charting; ARX focuses on execution geometry."
      },
      {
        featureName: "Automated Setup Scoring (0-100)",
        arxTerminal: "Multi-Factor Confluence Engine (Minervini, Piotroski, Raschke)",
        competitor: "Manual Custom Screener Building",
        notes: "ARX provides pre-computed quantitative edge scores on every catalog asset."
      },
      {
        featureName: "Congressional Insider Tracking",
        arxTerminal: "Native STOCK Act Radar with Committee Jurisdictional Scoring",
        competitor: "Not Natively Integrated",
        notes: "ARX integrates legislative insider disclosures directly into asset scoring."
      },
      {
        featureName: "Tail Risk Modeling",
        arxTerminal: "Cornish-Fisher M-VaR (95% & 99%) with Kupiec Auditing",
        competitor: "Standard Historical Volatility & Beta",
        notes: "ARX adjusts for real-world fat tails and negative return skewness."
      }
    ],
    verdict: "Choose Koyfin for deep macro research, analyst estimate models, and flexible multi-window charting; choose ARX Terminal for automated swing setups, statutory insider tracking, and mathematical downside risk discipline."
  },
  {
    slug: "bloomberg-terminal",
    competitorName: "Bloomberg Professional Terminal",
    competitorDomain: "bloomberg.com/professional",
    tagline: "The Legacy Wall Street Institutional Workstation",
    summary: "The Bloomberg Terminal is the undisputed gold standard of Wall Street trading floors, priced at over $30,000 per year per seat. ARX Terminal delivers institutional-grade quantitative discipline (Minervini VCP setups, Cornish-Fisher downside risk modeling, and STOCK Act intelligence) directly to modern browsers at zero cost, removing the proprietary hardware and financial barriers of legacy finance.",
    pricingComparison: {
      arx: "$0 (Free Institutional Web Terminal)",
      competitor: "$30,000+ per user / year (requires 2-year lease agreement)"
    },
    keyAdvantagesArx: [
      "Instant web access on any device with zero software installation or dedicated keyboards",
      "Transparent, auditable quantitative formulas with complete open documentation",
      "Bespoke Congressional STOCK Act Radar with committee jurisdiction analysis",
      "100% free access for independent traders, quants, and students"
    ],
    keyAdvantagesCompetitor: [
      "Instantaneous broker execution routing, fixed income pricing, and interbank messaging (IB)",
      "Unrivaled global newsroom, real-time economic data, and supply-chain matrices",
      "Decades of institutional legacy dominance across every asset class"
    ],
    featuresMatrix: [
      {
        featureName: "Annual Subscription Cost",
        arxTerminal: "$0 (Free)",
        competitor: "$30,000+ / year",
        notes: "ARX levels the playing field for retail and quantitative researchers."
      },
      {
        featureName: "Deployment & Accessibility",
        arxTerminal: "Web-native (Next.js, PWA, Cloudflare Edge)",
        competitor: "Desktop Client / Dedicated Terminal Hardware",
        notes: "ARX loads in under 10ms with offline caching."
      },
      {
        featureName: "Quantitative Execution Geometry",
        arxTerminal: "Automated Minervini VCP, 20 EMA, and Turtle ATR Ladders",
        competitor: "Extensive Bloomberg Query Language (BQL) & Custom Scripting",
        notes: "Bloomberg offers infinite customizability via BQL; ARX provides pre-built, fail-closed quantitative setups."
      },
      {
        featureName: "Congressional Intelligence",
        arxTerminal: "Dedicated Legislative Overlap Index & Late-Filer Decay",
        competitor: "Raw Disclosure News Feeds",
        notes: "ARX provides specialized legislative scoring designed for retail traders."
      }
    ],
    verdict: "For institutional investment banks requiring broker execution, private chats, and fixed-income syndicate underwriting, Bloomberg remains irreplaceable. For independent quantitative traders seeking algorithmic setups, statutory insider tracking, and downside risk models without a $30,000 fee, ARX Terminal provides an unmatched modern alternative."
  }
];
