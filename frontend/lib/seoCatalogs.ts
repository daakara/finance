/**
 * ARX Terminal Programmatic SEO Catalogs (Single Source of Truth)
 *
 * Consolidates static data catalogs for Strategies, Politicians, Committees,
 * and Head-to-Head Comparisons used across Next.js dynamic routes and sitemap generation.
 */

// 1. STRATEGIES CATALOG
export interface StrategyCandidate {
  symbol: string;
  name: string;
  price: number;
  changePct: number;
  piotroski: number;
  roic: string;
  pegOrShort: string;
  state: "IN_BUY_ZONE" | "APPROACHING_TARGET" | "WAITING_PULLBACK";
  stateBadge: string;
  entryRange: string;
  target1: string;
  stopLoss: string;
  thesis: string;
}

export interface StrategyDefinition {
  slug: string;
  name: string;
  author: string;
  tagline: string;
  description: string;
  screeningRules: string[];
  candidates: StrategyCandidate[];
}

export const STRATEGY_DATABASE: StrategyDefinition[] = [
  {
    slug: "minervini-vcp",
    name: "Mark Minervini Volatility Contraction Pattern (VCP)",
    author: "Mark Minervini (U.S. Investing Champion)",
    tagline: "Institutional swing accumulation setups with progressive contraction cycles and volume dry-up.",
    description: "The VCP setup identifies institutional accumulation where selling pressure dries up across 2 to 4 distinct contractions (e.g. 15% -> 8% -> 3%), creating an asymmetric pivot entry with tight volatility risk invalidation.",
    screeningRules: [
      "Stage 2 Structural Uptrend: Price > 50-day EMA > 200-day SMA.",
      "Progressive Volatility Contraction: Range narrowing across consecutive swing pullbacks.",
      "Volume Dry-Up (VDU): Volume drops >= 40% below 50-day average on final consolidation handle.",
      "Tight Risk Invalidation: Stop loss strictly placed 1.25x ATR below optimal accumulation pivot."
    ],
    candidates: [
      {
        symbol: "NVDA",
        name: "NVIDIA Corporation",
        price: 128.50,
        changePct: 3.14,
        piotroski: 8,
        roic: "58.4%",
        pegOrShort: "PEG 0.85",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$124.80 - $128.50",
        target1: "$142.00",
        stopLoss: "$118.20",
        thesis: "3-Stage contraction handle resting above 20 EMA with Blackwell datacenter ramp."
      },
      {
        symbol: "PLTR",
        name: "Palantir Technologies",
        price: 31.20,
        changePct: 4.12,
        piotroski: 8,
        roic: "32.1%",
        pegOrShort: "PEG 1.10",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$29.80 - $31.20",
        target1: "$35.50",
        stopLoss: "$28.40",
        thesis: "High-density institutional accumulation handle following TITAN contract award."
      },
      {
        symbol: "VRT",
        name: "Vertiv Holdings",
        price: 114.20,
        changePct: 2.85,
        piotroski: 7,
        roic: "28.6%",
        pegOrShort: "PEG 0.92",
        state: "APPROACHING_TARGET",
        stateBadge: "🔵 APPROACHING_TARGET",
        entryRange: "$109.50 - $112.00",
        target1: "$124.50",
        stopLoss: "$105.80",
        thesis: "Liquid cooling AI datacenter infrastructure breakout expanding toward Target 1."
      }
    ]
  },
  {
    slug: "magic-formula",
    name: "Joel Greenblatt Magic Formula",
    author: "Joel Greenblatt (Gotham Capital)",
    tagline: "High Return on Invested Capital (ROIC) combined with deep Earnings Yield discounts.",
    description: "Greenblatt's Magic Formula ranks equities by two mathematically rigorous factors: high capital efficiency (ROIC > 20%) and high enterprise earnings yield (EBIT/EV), identifying outstanding businesses selling at bargain valuations.",
    screeningRules: [
      "Top-Decile Return on Capital: ROIC >= 25% indicating wide economic moat.",
      "High Earnings Yield: EBIT / Enterprise Value >= 8.5%.",
      "Piotroski F-Score >= 8: Pristine accounting quality with zero debt dilution risk.",
      "Minimum Liquidity Tier: Market Cap >= $1B with robust institutional float."
    ],
    candidates: [
      {
        symbol: "CPRX",
        name: "Catalyst Pharmaceuticals",
        price: 23.40,
        changePct: 1.90,
        piotroski: 9,
        roic: "42.8%",
        pegOrShort: "P/E 9.4x",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$22.50 - $23.40",
        target1: "$26.50",
        stopLoss: "$21.60",
        thesis: "Orphan disease franchise with 88%+ gross margins and pristine 9/9 Piotroski balance sheet."
      },
      {
        symbol: "LNTH",
        name: "Lantheus Holdings",
        price: 100.78,
        changePct: -4.09,
        piotroski: 9,
        roic: "38.5%",
        pegOrShort: "P/E 11.2x",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$97.50 - $100.78",
        target1: "$112.40",
        stopLoss: "$93.20",
        thesis: "Radiopharmaceutical diagnostic monopoly trading at deep valuation discount."
      },
      {
        symbol: "MEDP",
        name: "Medpace Holdings",
        price: 342.10,
        changePct: 1.45,
        piotroski: 9,
        roic: "34.2%",
        pegOrShort: "P/E 19.8x",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$332.00 - $342.10",
        target1: "$385.00",
        stopLoss: "$315.00",
        thesis: "Full-service clinical CRO compounder with zero long-term debt."
      }
    ]
  },
  {
    slug: "peter-lynch-garp",
    name: "Peter Lynch Growth at a Reasonable Price (GARP)",
    author: "Peter Lynch (Fidelity Magellan Fund)",
    tagline: "High EPS growth compounders trading at PEG ratios <= 1.0 with zero hype inflation.",
    description: "Peter Lynch's classic strategy searches for steady earnings compounders where the Price/Earnings-to-Growth (PEG) ratio is <= 1.0, ensuring investors do not overpay for future secular cash flow growth.",
    screeningRules: [
      "Valuation Hygiene: PEG Ratio <= 1.0 (P/E / 3-Year EPS CAGR).",
      "Historical Growth: 3-Year Revenue CAGR >= 20%.",
      "Clean Balance Sheet: Debt-to-Equity < 0.5 with expanding operating margins.",
      "Institutional Sweet Spot: Under-the-radar mid-caps ignored by passive mega-cap funds."
    ],
    candidates: [
      {
        symbol: "ACLS",
        name: "Axcelis Technologies",
        price: 94.20,
        changePct: 2.30,
        piotroski: 8,
        roic: "31.4%",
        pegOrShort: "PEG 0.78",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$91.50 - $94.20",
        target1: "$104.50",
        stopLoss: "$87.80",
        thesis: "Ion implantation semiconductor equipment leader priced at PEG 0.78."
      },
      {
        symbol: "POWI",
        name: "Power Integrations",
        price: 72.50,
        changePct: 0.85,
        piotroski: 8,
        roic: "24.8%",
        pegOrShort: "PEG 0.88",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$70.20 - $72.50",
        target1: "$80.40",
        stopLoss: "$67.30",
        thesis: "GaN power semiconductor efficiency chips with automotive and datacenter tailwinds."
      },
      {
        symbol: "ELF",
        name: "e.l.f. Beauty Inc.",
        price: 182.40,
        changePct: 2.15,
        piotroski: 8,
        roic: "26.5%",
        pegOrShort: "PEG 0.94",
        state: "APPROACHING_TARGET",
        stateBadge: "🔵 APPROACHING_TARGET",
        entryRange: "$175.00 - $180.00",
        target1: "$198.50",
        stopLoss: "$168.00",
        thesis: "High market-share cosmetics compounder with international digital expansion."
      }
    ]
  },
  {
    slug: "short-squeeze",
    name: "High Short Float & Volume Expansion Asymmetric Setups",
    author: "Quantitative Market Microstructure Engine",
    tagline: "Short float >= 6.0% combined with Relative Volume (RVOL >= 2.5x) and tight base.",
    description: "Identifies heavily shorted equities undergoing aggressive volume expansion. When institutional buyers step in, short sellers are forced into rapid market buying to cover, causing explosive multi-day upside squeezes.",
    screeningRules: [
      "Elevated Short Interest: Short Float >= 6.0% of free float.",
      "Relative Volume (RVOL): Today's volume >= 2.5x 30-day average volume.",
      "Piotroski Health: F-Score >= 6 to filter out structurally bankrupt debt traps.",
      "ATR Volatility Boundary: Stop loss strictly enforced at recent handle low."
    ],
    candidates: [
      {
        symbol: "SMCI",
        name: "Super Micro Computer",
        price: 48.20,
        changePct: 5.40,
        piotroski: 7,
        roic: "22.4%",
        pegOrShort: "Short 14.8%",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$46.50 - $48.20",
        target1: "$56.80",
        stopLoss: "$42.50",
        thesis: "14.8% Short Float with heavy liquid-cooling server rack demand."
      },
      {
        symbol: "CELH",
        name: "Celsius Holdings",
        price: 36.80,
        changePct: 3.85,
        piotroski: 7,
        roic: "21.0%",
        pegOrShort: "Short 9.2%",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$35.40 - $36.80",
        target1: "$42.50",
        stopLoss: "$33.10",
        thesis: "9.2% Short Float with PepsiCo international distribution expansion."
      }
    ]
  },
  {
    slug: "rule-breakers",
    name: "David Gardner Disruptive Rule Breakers",
    author: "David Gardner (Motley Fool Rule Breakers)",
    tagline: "First-mover innovators in high-growth industries with strong consumer brand moats.",
    description: "Targets disruptive growth pioneers reshaping massive global industries with accelerating top-line revenue growth and visionary management.",
    screeningRules: [
      "Top-Dog First-Mover: Dominant brand and technology leadership in emerging sectors.",
      "Accelerating Revenue CAGR: 3-Year Top-Line Revenue Growth >= 30%.",
      "Massive Gross Margin Moat: Gross Margin >= 65% allowing self-funded R&D expansion.",
      "Consumer Mindshare: High organic virality and network-effect switching costs."
    ],
    candidates: [
      {
        symbol: "DUOL",
        name: "Duolingo Inc.",
        price: 312.80,
        changePct: 3.20,
        piotroski: 8,
        roic: "29.4%",
        pegOrShort: "Gross 73%",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$302.00 - $312.80",
        target1: "$345.00",
        stopLoss: "$288.50",
        thesis: "Gamified generative AI language & math learning with 73% gross margins."
      },
      {
        symbol: "TMDX",
        name: "TransMedics Group",
        price: 92.60,
        changePct: 3.15,
        piotroski: 8,
        roic: "27.8%",
        pegOrShort: "Gross 68%",
        state: "IN_BUY_ZONE",
        stateBadge: "🟢 IN_BUY_ZONE",
        entryRange: "$89.50 - $92.60",
        target1: "$104.00",
        stopLoss: "$85.20",
        thesis: "Organ Care System (OCS) warm-perfusion donor organ transport monopoly."
      }
    ]
  }
];

// 2. POLITICIAN CATALOG
export interface PoliticianTrade {
  ticker: string;
  assetName: string;
  type: string;
  amount: string;
  date: string;
  lagDays: number;
  stalenessStatus: "FRESH" | "STANDARD" | "AGING" | "LATE_FILER";
  stalenessBadge: string;
  alignmentScore: number;
  thesis: string;
}

export interface PoliticianProfile {
  slug: string;
  name: string;
  chamber: "House" | "Senate";
  party: "Democrat" | "Republican";
  stateDistrict: string;
  committees: string[];
  recentTrades: PoliticianTrade[];
}

export const POLITICIAN_DATABASE: PoliticianProfile[] = [
  {
    slug: "nancy-pelosi",
    name: "Nancy Pelosi",
    chamber: "House",
    party: "Democrat",
    stateDistrict: "CA-11 (San Francisco)",
    committees: ["Former Speaker of the House", "Democratic Leadership", "Appropriations (Prior)"],
    recentTrades: [
      {
        ticker: "NVDA",
        assetName: "NVIDIA Corporation",
        type: "Purchase (Deep ITM Calls)",
        amount: "$1,000,000 - $5,000,000",
        date: "2026-07-28",
        lagDays: 17,
        stalenessStatus: "STANDARD",
        stalenessBadge: "⏳ Standard (17d lag)",
        alignmentScore: 94,
        thesis: "Strategic timing ahead of federal AI compute export rule revisions and next-generation datacenter infrastructure appropriations."
      },
      {
        ticker: "MSFT",
        assetName: "Microsoft Corporation",
        type: "Purchase (LEAPS Calls)",
        amount: "$500,000 - $1,000,000",
        date: "2026-06-15",
        lagDays: 24,
        stalenessStatus: "STANDARD",
        stalenessBadge: "⏳ Standard (24d lag)",
        alignmentScore: 88,
        thesis: "Enterprise cloud software expansion and federal defense generative AI procurement contracts."
      }
    ]
  },
  {
    slug: "dan-crenshaw",
    name: "Dan Crenshaw",
    chamber: "House",
    party: "Republican",
    stateDistrict: "TX-02 (Houston)",
    committees: ["Energy & Commerce", "House Permanent Select Committee on Intelligence"],
    recentTrades: [
      {
        ticker: "PLTR",
        assetName: "Palantir Technologies",
        type: "Purchase (Common Stock)",
        amount: "$50,000 - $100,000",
        date: "2026-08-10",
        lagDays: 12,
        stalenessStatus: "FRESH",
        stalenessBadge: "⚡ Fresh (<15d lag)",
        alignmentScore: 95,
        thesis: "Direct oversight of intelligence community software procurement and defense AI telemetry systems."
      }
    ]
  },
  {
    slug: "tommy-tuberville",
    name: "Tommy Tuberville",
    chamber: "Senate",
    party: "Republican",
    stateDistrict: "Alabama (Senior Senator)",
    committees: ["Senate Armed Services Committee", "Agriculture, Nutrition & Forestry", "Veterans' Affairs"],
    recentTrades: [
      {
        ticker: "CELH",
        assetName: "Celsius Holdings",
        type: "Purchase (Common Stock)",
        amount: "$100,000 - $250,000",
        date: "2026-06-25",
        lagDays: 58,
        stalenessStatus: "LATE_FILER",
        stalenessBadge: "🛑 Late Filer (58d lag)",
        alignmentScore: 48,
        thesis: "Consumer staples and distribution expansion; non-compliant disclosure with severe time-decay penalty."
      }
    ]
  },
  {
    slug: "michael-mccaul",
    name: "Michael McCaul",
    chamber: "House",
    party: "Republican",
    stateDistrict: "TX-10 (Austin/Houston)",
    committees: ["Foreign Affairs Committee (Chairman)", "Homeland Security"],
    recentTrades: [
      {
        ticker: "NVO",
        assetName: "Novo Nordisk A/S",
        type: "Purchase (Common Stock)",
        amount: "$250,000 - $500,000",
        date: "2026-08-02",
        lagDays: 16,
        stalenessStatus: "STANDARD",
        stalenessBadge: "⏳ Standard (16d lag)",
        alignmentScore: 92,
        thesis: "Transatlantic pharmaceutical supply chain discussions and federal healthcare Medicare GLP-1 reimbursement expansion deliberations."
      }
    ]
  },
  {
    slug: "mark-green",
    name: "Mark Green",
    chamber: "House",
    party: "Republican",
    stateDistrict: "TX-07 (Clarksville)",
    committees: ["Homeland Security (Chairman)", "Foreign Affairs"],
    recentTrades: [
      {
        ticker: "TSM",
        assetName: "Taiwan Semiconductor Mfg",
        type: "Purchase (Common Stock)",
        amount: "$500,000 - $1,000,000",
        date: "2026-08-04",
        lagDays: 15,
        stalenessStatus: "FRESH",
        stalenessBadge: "⚡ Fresh (<15d lag)",
        alignmentScore: 91,
        thesis: "Direct involvement in CHIPS Act national security defense allocations and Indo-Pacific supply-chain resilience."
      }
    ]
  },
  {
    slug: "ro-khanna",
    name: "Ro Khanna",
    chamber: "House",
    party: "Democrat",
    stateDistrict: "CA-17 (Silicon Valley)",
    committees: ["Armed Services (Cyber, Innovative Tech)", "Oversight & Accountability"],
    recentTrades: [
      {
        ticker: "IONQ",
        assetName: "IonQ Inc.",
        type: "Purchase (Common Stock)",
        amount: "$50,000 - $100,000",
        date: "2026-08-05",
        lagDays: 16,
        stalenessStatus: "STANDARD",
        stalenessBadge: "⏳ Standard (16d lag)",
        alignmentScore: 89,
        thesis: "Oversight of federal quantum computing appropriations and DoD cryptographic transition initiatives."
      }
    ]
  },
  {
    slug: "josh-gottheimer",
    name: "Josh Gottheimer",
    chamber: "House",
    party: "Democrat",
    stateDistrict: "NJ-05",
    committees: ["Financial Services (Capital Markets)", "Permanent Select Committee on Intelligence"],
    recentTrades: [
      {
        ticker: "COIN",
        assetName: "Coinbase Global",
        type: "Purchase (Common Stock)",
        amount: "$100,000 - $250,000",
        date: "2026-08-08",
        lagDays: 14,
        stalenessStatus: "FRESH",
        stalenessBadge: "⚡ Fresh (<15d lag)",
        alignmentScore: 93,
        thesis: "Deliberations on market structure reform legislation and digital asset regulatory clarity bills."
      }
    ]
  },
  {
    slug: "sheldon-whitehouse",
    name: "Sheldon Whitehouse",
    chamber: "Senate",
    party: "Democrat",
    stateDistrict: "Rhode Island (Senior Senator)",
    committees: ["Senate Budget Committee (Chairman)", "Finance", "Environment & Public Works"],
    recentTrades: [
      {
        ticker: "VRT",
        assetName: "Vertiv Holdings",
        type: "Purchase (Common Stock)",
        amount: "$50,000 - $100,000",
        date: "2026-08-11",
        lagDays: 13,
        stalenessStatus: "FRESH",
        stalenessBadge: "⚡ Fresh (<15d lag)",
        alignmentScore: 86,
        thesis: "Federal grid modernization and green energy cooling infrastructure tax incentive alignment."
      }
    ]
  }
];

// 3. COMMITTEE CATALOG
export interface CommitteeTrade {
  politician: string;
  politicianSlug: string;
  ticker: string;
  assetName: string;
  type: string;
  amount: string;
  date: string;
  alignmentScore: number;
  stalenessBadge: string;
  thesis: string;
}

export interface CommitteeDefinition {
  slug: string;
  name: string;
  chamber: "House" | "Senate" | "Joint";
  jurisdictionSummary: string;
  regulatedSectors: string[];
  keyMembers: { name: string; slug: string; party: string }[];
  trades: CommitteeTrade[];
}

export const COMMITTEE_DATABASE: CommitteeDefinition[] = [
  {
    slug: "armed-services",
    name: "House & Senate Armed Services Committees",
    chamber: "Joint",
    jurisdictionSummary: "Direct statutory oversight and annual National Defense Authorization Act (NDAA) budget allocations for Department of Defense procurement, military AI telemetry, cybersecurity, and aerospace defense contracting.",
    regulatedSectors: ["Aerospace & Defense", "Military AI & Telemetry", "Autonomous Drone Swarms", "Defense Cybersecurity"],
    keyMembers: [
      { name: "Sen. Tommy Tuberville", slug: "tommy-tuberville", party: "R-AL" },
      { name: "Rep. Ro Khanna", slug: "ro-khanna", party: "D-CA" }
    ],
    trades: [
      {
        politician: "Rep. Ro Khanna (D-CA)",
        politicianSlug: "ro-khanna",
        ticker: "IONQ",
        assetName: "IonQ Inc.",
        type: "Purchase (Common Stock)",
        amount: "$50,000 - $100,000",
        date: "2026-08-05",
        alignmentScore: 89,
        stalenessBadge: "⏳ Standard (16d lag)",
        thesis: "Oversight of federal quantum computing appropriations and DoD cryptographic transition initiatives."
      }
    ]
  },
  {
    slug: "energy-commerce",
    name: "House Energy & Commerce Committee",
    chamber: "House",
    jurisdictionSummary: "Broadest legislative jurisdiction over telecommunications, semiconductor supply chains, energy grid modernization, pharmaceutical drug manufacturing, and interstate commerce regulations.",
    regulatedSectors: ["Semiconductors", "Datacenter Power Infrastructure", "Telecommunications", "Biotechnology"],
    keyMembers: [
      { name: "Rep. Dan Crenshaw", slug: "dan-crenshaw", party: "R-TX" },
      { name: "Rep. Nancy Pelosi (Leadership)", slug: "nancy-pelosi", party: "D-CA" }
    ],
    trades: [
      {
        politician: "Rep. Nancy Pelosi (D-CA)",
        politicianSlug: "nancy-pelosi",
        ticker: "NVDA",
        assetName: "NVIDIA Corporation",
        type: "Purchase (Call Options)",
        amount: "$1,000,000 - $5,000,000",
        date: "2026-07-28",
        alignmentScore: 94,
        stalenessBadge: "⏳ Standard (17d lag)",
        thesis: "Strategic timing ahead of federal AI compute export rule revisions and next-generation datacenter infrastructure appropriations."
      },
      {
        politician: "Rep. Dan Crenshaw (R-TX)",
        politicianSlug: "dan-crenshaw",
        ticker: "PLTR",
        assetName: "Palantir Technologies",
        type: "Purchase (Common Stock)",
        amount: "$50,000 - $100,000",
        date: "2026-08-10",
        alignmentScore: 95,
        stalenessBadge: "⚡ Fresh (<15d lag)",
        thesis: "Direct oversight of intelligence community software procurement and defense AI telemetry systems."
      }
    ]
  },
  {
    slug: "intelligence",
    name: "House Permanent Select Committee on Intelligence",
    chamber: "House",
    jurisdictionSummary: "Classified oversight of the 18 United States intelligence agencies (CIA, NSA, DIA, NGA, NRO), cyber warfare capabilities, and sovereign national security software platforms.",
    regulatedSectors: ["Sovereign Enterprise Software", "Classified Cloud Hosting", "Signals Intelligence", "Satellite Reconnaissance"],
    keyMembers: [
      { name: "Rep. Dan Crenshaw", slug: "dan-crenshaw", party: "R-TX" },
      { name: "Rep. Josh Gottheimer", slug: "josh-gottheimer", party: "D-NJ" }
    ],
    trades: [
      {
        politician: "Rep. Dan Crenshaw (R-TX)",
        politicianSlug: "dan-crenshaw",
        ticker: "PLTR",
        assetName: "Palantir Technologies",
        type: "Purchase (Common Stock)",
        amount: "$50,000 - $100,000",
        date: "2026-08-10",
        alignmentScore: 95,
        stalenessBadge: "⚡ Fresh (<15d lag)",
        thesis: "Direct oversight of intelligence community software procurement and defense AI telemetry systems."
      }
    ]
  },
  {
    slug: "foreign-affairs",
    name: "House Foreign Affairs & Senate Foreign Relations",
    chamber: "Joint",
    jurisdictionSummary: "Oversight of international treaties, pharmaceutical import/export supply-chain agreements, foreign military sales, and geopolitical tech export restrictions.",
    regulatedSectors: ["Global Pharmaceutical Supply Chains", "Semiconductor Foundry Exports", "Cross-Border Energy Infrastructure"],
    keyMembers: [
      { name: "Rep. Michael McCaul (Chairman)", slug: "michael-mccaul", party: "R-TX" },
      { name: "Rep. Mark Green", slug: "mark-green", party: "R-TN" }
    ],
    trades: [
      {
        politician: "Rep. Michael McCaul (R-TX)",
        politicianSlug: "michael-mccaul",
        ticker: "NVO",
        assetName: "Novo Nordisk A/S",
        type: "Purchase (Common Stock)",
        amount: "$250,000 - $500,000",
        date: "2026-08-02",
        alignmentScore: 92,
        stalenessBadge: "⏳ Standard (16d lag)",
        thesis: "Transatlantic pharmaceutical supply chain discussions and federal healthcare Medicare GLP-1 reimbursement expansion deliberations."
      },
      {
        politician: "Rep. Mark Green (R-TN)",
        politicianSlug: "mark-green",
        ticker: "TSM",
        assetName: "Taiwan Semiconductor Mfg",
        type: "Purchase (Common Stock)",
        amount: "$500,000 - $1,000,000",
        date: "2026-08-04",
        alignmentScore: 91,
        stalenessBadge: "⚡ Fresh (<15d lag)",
        thesis: "Direct involvement in CHIPS Act national security defense allocations and Indo-Pacific supply-chain resilience."
      }
    ]
  },
  {
    slug: "financial-services",
    name: "House Financial Services & Senate Banking",
    chamber: "Joint",
    jurisdictionSummary: "Regulatory oversight of the SEC, Federal Reserve, CFTC, digital asset market structure legislation, public company reporting standards, and banking capital liquidity ratios.",
    regulatedSectors: ["Digital Asset Exchanges", "Commercial Banking", "Asset Management", "Payment Processors"],
    keyMembers: [
      { name: "Rep. Josh Gottheimer", slug: "josh-gottheimer", party: "D-NJ" }
    ],
    trades: [
      {
        politician: "Rep. Josh Gottheimer (D-NJ)",
        politicianSlug: "josh-gottheimer",
        ticker: "COIN",
        assetName: "Coinbase Global",
        type: "Purchase (Common Stock)",
        amount: "$100,000 - $250,000",
        date: "2026-08-08",
        alignmentScore: 93,
        stalenessBadge: "⚡ Fresh (<15d lag)",
        thesis: "Deliberations on market structure reform legislation and digital asset regulatory clarity bills."
      }
    ]
  }
];

// 4. COMPARISON PAIRS CATALOG
export interface ComparisonPair {
  pair: string;
  a: string;
  b: string;
  label: string;
}

export const COMPARISON_PAIRS: ComparisonPair[] = [
  { pair: "nvo-vs-lly", a: "NVO", b: "LLY", label: "Novo Nordisk (NVO) vs. Eli Lilly (LLY)" },
  { pair: "spy-vs-qqq", a: "SPY", b: "QQQ", label: "S&P 500 (SPY) vs. Nasdaq-100 (QQQ)" },
  { pair: "nvda-vs-aapl", a: "NVDA", b: "AAPL", label: "NVIDIA (NVDA) vs. Apple (AAPL)" },
  { pair: "tsla-vs-pltr", a: "TSLA", b: "PLTR", label: "Tesla (TSLA) vs. Palantir (PLTR)" },
  { pair: "amd-vs-nvda", a: "AMD", b: "NVDA", label: "AMD (AMD) vs. NVIDIA (NVDA)" },
  { pair: "msft-vs-aapl", a: "MSFT", b: "AAPL", label: "Microsoft (MSFT) vs. Apple (AAPL)" },
  { pair: "cprx-vs-powi", a: "CPRX", b: "POWI", label: "Catalyst Pharma (CPRX) vs. Power Integrations (POWI)" },
];
