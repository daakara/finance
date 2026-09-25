import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import Navbar from "../../../components/Navbar";
import HistoricalEdgeScorecard from "../../../components/HistoricalEdgeScorecard";
import { MASTER_ASSET_CATALOG, CATALOG_BASELINE_PRICES } from "../../../lib/masterCatalog";
import { getCanonicalAssetName } from "../../../lib/assetRegistry";

interface PageProps {
  params: {
    type: string;
  };
}

import {
  STRATEGY_DATABASE,
  type StrategyDefinition,
  type StrategyCandidate,
} from "../../../lib/seoCatalogs";

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
    title: `🎯 ${strategy.name} Stock Screener & Quantitative Invalidation Levels`,
    description: `Screen top ${strategy.name} equities: ${strategy.tagline} Review candidate entry ranges, ATR stop loss targets, and Piotroski F-Scores.`,
    openGraph: {
      title: `${strategy.name} Quantitative Screener Matrix`,
      description: strategy.description,
      url: `https://www.arxterminal.com/strategy/${params.type.toLowerCase()}/`,
      siteName: "ARX Terminal",
      type: "article",
    },
    twitter: {
      card: "summary_large_image",
      title: `🎯 ${strategy.name} Stock Screener & Quantitative Invalidation Levels`,
      description: `Screen top ${strategy.name} equities: ${strategy.tagline} Review candidate entry ranges, ATR stop loss targets, and Piotroski F-Scores.`,
      images: ["/og-image.png"],
      creator: "@ARXTerminal",
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
        "name": "ARX Terminal"
      }
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "ARX Terminal",
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

        {/* 📊 QUANTITATIVE BACKTESTED EDGE & WIN-RATE SCORECARD */}
        <section aria-label="Quantitative Historical Edge Scorecard">
          <HistoricalEdgeScorecard strategySlug={params.type} />
        </section>

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
            {strategy.candidates.map((cand, idx) => {
              const cat = MASTER_ASSET_CATALOG[cand.symbol];
              const displayPrice = CATALOG_BASELINE_PRICES[cand.symbol] ?? cand.price;
              const displayChange = cand.changePct;
              const displayName = getCanonicalAssetName(cand.symbol, cat?.name || cand.name);
              const isStale = cand.price > 0 && Math.abs(displayPrice - cand.price) > 5;
              const displayTarget1 = isStale ? `$${(displayPrice * 1.15).toFixed(2)}` : cand.target1;
              const displayStopLoss = isStale ? `$${(displayPrice * 0.93).toFixed(2)}` : cand.stopLoss;
              const displayEntryRange = isStale
                ? `$${(displayPrice * 0.97).toFixed(2)} - $${displayPrice.toFixed(2)}`
                : cand.entryRange;

              return (
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
                      <span className="text-white font-bold">{displayName}</span>
                      <span className="text-slate-400 font-mono">${displayPrice.toFixed(2)} ({displayChange >= 0 ? "+" : ""}{displayChange.toFixed(2)}%)</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold font-mono">
                        {cand.stateBadge}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">
                        Piotroski: {cat?.piotroski ?? cand.piotroski}/9
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-400 pt-1">
                    <div><strong>Optimal Entry:</strong> <span className="text-slate-200 font-mono">{displayEntryRange}</span></div>
                    <div><strong>Target 1 (+2.5x ATR):</strong> <span className="text-cyan-300 font-mono">{displayTarget1}</span></div>
                    <div><strong>Stop Loss Floor:</strong> <span className="text-rose-400 font-mono">{displayStopLoss}</span></div>
                    <div><strong>Capital Efficiency:</strong> <span className="text-emerald-300 font-mono">ROIC {cat?.roic ? `${cat.roic}%` : cand.roic}</span></div>
                  </div>

                  <p className="text-[11px] text-slate-300 font-sans leading-relaxed pt-1 border-t border-[#141b26]">
                    <strong>Execution Setup:</strong> {cand.thesis}
                  </p>
                </div>
              );
            })}
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
