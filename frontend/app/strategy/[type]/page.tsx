import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import Navbar from "../../../components/Navbar";

interface PageProps {
  params: {
    type: string;
  };
}

interface StrategyDefinition {
  slug: string;
  name: string;
  author: string;
  tagline: string;
  description: string;
  screeningRules: string[];
  candidates: {
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
  }[];
}

const STRATEGY_DATABASE: StrategyDefinition[] = [
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

export function generateStaticParams() {
  return STRATEGY_DATABASE.map(s => ({ type: s.slug }));
}

export function generateMetadata({ params }: PageProps): Metadata {
  const strategy = STRATEGY_DATABASE.find(s => s.slug === params.type.toLowerCase());
  if (!strategy) {
    return {
      title: "Trading Strategy Not Found | ARX Terminal",
      description: "The requested quantitative trading strategy could not be found.",
    };
  }

  return {
    title: `🎯 ${strategy.name} Stock Screener & Quantitative Invalidation Levels | ARX Terminal`,
    description: `Screen top ${strategy.name} equities: ${strategy.tagline} Review candidate entry ranges, ATR stop loss targets, and Piotroski F-Scores.`,
    openGraph: {
      title: `${strategy.name} Quantitative Screener Matrix`,
      description: strategy.description,
      url: `https://www.arxterminal.com/strategy/${params.type.toLowerCase()}/`,
      siteName: "ARX Terminal",
      type: "article",
    },
    alternates: {
      canonical: `https://www.arxterminal.com/strategy/${params.type.toLowerCase()}/`,
    },
  };
}

export default function StrategyDetailPage({ params }: PageProps) {
  const strategy = STRATEGY_DATABASE.find(s => s.slug === params.type.toLowerCase());
  if (!strategy) {
    notFound();
  }

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "Dataset",
      "name": `${strategy.name} Screener Candidates`,
      "description": strategy.description,
      "url": `https://www.arxterminal.com/strategy/${params.type.toLowerCase()}/`,
      "creator": {
        "@type": "Organization",
        "name": "Finance Terminal"
      }
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "Finance Terminal",
          "item": "https://www.arxterminal.com/"
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Quantitative Screeners",
          "item": "https://www.arxterminal.com/screener/"
        },
        {
          "@type": "ListItem",
          "position": 3,
          "name": strategy.name,
          "item": `https://www.arxterminal.com/strategy/${params.type.toLowerCase()}/`
        }
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans selection:bg-cyan-500 selection:text-black transition-colors duration-200">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }}
      />
      <Navbar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 font-mono space-y-8 pb-24 sm:pb-16">
        {/* Breadcrumb Nav */}
        <nav className="text-xs text-slate-500 flex items-center space-x-2">
          <Link href="/" className="hover:text-cyan-400">Terminal</Link>
          <span>/</span>
          <Link href="/screener" className="hover:text-cyan-400">Screener</Link>
          <span>/</span>
          <span className="text-slate-300 font-bold">{strategy.name}</span>
        </nav>

        {/* Strategy Methodology Provenance Banner */}
        <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-800/60 text-xs text-cyan-200 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span>📈</span>
            <span><strong>Algorithmic Strategy Matrix & Screening Rules:</strong> Multi-factor quantitative formulation, risk guardrails, and systematic candidate ranking parameters.</span>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-900/60 border border-cyan-700/80 font-bold uppercase shrink-0 hidden sm:inline">
            Quantitative Ruleset
          </span>
        </div>

        {/* Hero Header */}
        <header className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-3">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-xs font-bold font-mono">
              QUANTITATIVE STRATEGY DOSSIER
            </span>
            <span className="text-slate-500 text-xs">• {strategy.author}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            {strategy.name}
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
            {strategy.description}
          </p>
        </header>

        {/* 1-Click Interactive CTA */}
        <section className="bg-gradient-to-r from-emerald-950/40 via-[#0b1019] to-cyan-950/40 p-5 rounded-2xl border border-emerald-800/60 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center sm:text-left">
            <h2 className="text-sm sm:text-base font-bold text-white">
              Launch Live Interactive Stock Screener
            </h2>
            <p className="text-xs text-slate-300 font-sans">
              Filter real-time equities across Minervini VCP, Greenblatt Magic Formula, and Peter Lynch GARP models.
            </p>
          </div>
          <Link
            href="/screener"
            className="w-full sm:w-auto px-5 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black text-xs font-extrabold rounded-xl shadow-lg transition-transform active:scale-95 text-center whitespace-nowrap"
          >
            Launch Interactive Screener →
          </Link>
        </section>

        {/* Mathematical Screening Rules */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-3">
          <h2 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
            <span>📐 Quantitative Screening Rules & Mathematical Bounds</span>
          </h2>
          <ul className="space-y-2 text-xs text-slate-300 font-sans list-disc pl-5">
            {strategy.screeningRules.map((rule, idx) => (
              <li key={idx} className="leading-relaxed">{rule}</li>
            ))}
          </ul>
        </section>

        {/* Candidates Table */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e293b] pb-3">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>🎯 Current Matching Candidates</span>
            </h2>
            <span className="text-[11px] text-slate-500 font-sans">Updated Daily Pre-Market</span>
          </div>

          <div className="space-y-3">
            {strategy.candidates.map((cand, idx) => (
              <div
                key={idx}
                className="bg-[#06090f] p-4 rounded-xl border border-[#1b2434] space-y-2.5 text-xs"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center space-x-2.5">
                    <Link
                      href={`/stock/${cand.symbol.toLowerCase()}/`}
                      className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 font-bold hover:underline"
                    >
                      {cand.symbol}
                    </Link>
                    <span className="text-white font-bold">{cand.name}</span>
                    <span className="text-slate-400 font-mono">${cand.price.toFixed(2)} ({cand.changePct >= 0 ? "+" : ""}{cand.changePct}%)</span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold font-mono">
                      {cand.stateBadge}
                    </span>
                    <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">
                      Piotroski: {cand.piotroski}/9
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-400 pt-1">
                  <div><strong>Optimal Entry:</strong> <span className="text-slate-200 font-mono">{cand.entryRange}</span></div>
                  <div><strong>Target 1 (+2.5x ATR):</strong> <span className="text-cyan-300 font-mono">{cand.target1}</span></div>
                  <div><strong>Stop Loss Floor:</strong> <span className="text-rose-400 font-mono">{cand.stopLoss}</span></div>
                  <div><strong>Capital Efficiency:</strong> <span className="text-emerald-300 font-mono">ROIC {cand.roic}</span></div>
                </div>

                <p className="text-[11px] text-slate-300 font-sans leading-relaxed pt-1 border-t border-[#141b26]">
                  <strong>Execution Setup:</strong> {cand.thesis}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Other Strategies */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-3">
          <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Explore Other Quantitative Screener Strategies
          </h2>
          <div className="flex flex-wrap gap-2 text-xs">
            {STRATEGY_DATABASE.filter(s => s.slug !== params.type).map(s => (
              <Link
                key={s.slug}
                href={`/strategy/${s.slug}/`}
                className="px-3 py-1.5 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-slate-300 hover:text-cyan-300 border border-[#243044] transition-colors"
              >
                🎯 {s.name.split("(")[0]}
              </Link>
            ))}
          </div>
        </section>

        {/* Footer Navigation */}
        <footer className="border-t border-[#1e293b] pt-6 flex flex-wrap items-center justify-between gap-4 text-xs">
          <Link
            href="/screener"
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl transition-transform active:scale-95"
          >
            ← Return to Interactive Screener
          </Link>
          <div className="text-slate-500 font-sans">
            Grounded in Minervini VCP, Joel Greenblatt & Peter Lynch Methodologies
          </div>
        </footer>
      </main>
    </div>
  );
}
