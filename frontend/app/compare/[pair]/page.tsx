import type { Metadata } from "next";
import Link from "next/link";
import Navbar from "../../../components/Navbar";
import { SHARED_WATCHLIST_ITEMS } from "../../../lib/constants";
import ComparePairMatrix from "../../../components/ComparePairMatrix";

interface PageProps {
  params: {
    pair: string;
  };
}

import {
  COMPARISON_PAIRS,
  type ComparisonPair,
} from "../../../lib/seoCatalogs";

export function generateStaticParams() {
  return COMPARISON_PAIRS.map(p => ({ pair: p.pair }));
}

export function generateMetadata({ params }: PageProps): Metadata {
  const match = COMPARISON_PAIRS.find(p => p.pair === params.pair.toLowerCase()) || {
    pair: params.pair,
    a: params.pair.split("-vs-")[0]?.toUpperCase() || "ASSET_A",
    b: params.pair.split("-vs-")[1]?.toUpperCase() || "ASSET_B",
    label: `${params.pair.toUpperCase()} Comparison`
  };

  const nameA = SHARED_WATCHLIST_ITEMS.find(i => i.symbol.toUpperCase() === match.a)?.name || match.a;
  const nameB = SHARED_WATCHLIST_ITEMS.find(i => i.symbol.toUpperCase() === match.b)?.name || match.b;

  return {
    title: `${nameA} (${match.a}) vs. ${nameB} (${match.b}) Quantitative Comparison | ARX Terminal`,
    description: `Head-to-head quantitative comparison of ${nameA} (${match.a}) vs. ${nameB} (${match.b}): Valuation multiples, 5-Factor radar scores, Piotroski F-Scores, ATR volatility targets, and institutional conviction.`,
    openGraph: {
      title: `${nameA} (${match.a}) vs. ${nameB} (${match.b}) Comparison Matrix`,
      description: `Compare valuation, growth CAGR, Piotroski F-Score, and volatility invalidation levels between ${match.a} and ${match.b}.`,
      url: `https://www.arxterminal.com/compare/${params.pair.toLowerCase()}/`,
      siteName: "ARX Terminal",
      type: "article",
    },
    twitter: {
      card: "summary_large_image",
      title: `${nameA} (${match.a}) vs. ${nameB} (${match.b}) Comparison Matrix`,
      description: `Compare valuation, growth CAGR, Piotroski F-Score, and volatility invalidation levels between ${match.a} and ${match.b}.`,
      images: ["/og-image.png"],
      creator: "@ARXTerminal",
    },
    alternates: {
      canonical: `https://www.arxterminal.com/compare/${params.pair.toLowerCase()}/`,
    },
  };
}

export default function ComparisonPairPage({ params }: PageProps) {
  const match = COMPARISON_PAIRS.find(p => p.pair === params.pair.toLowerCase()) || {
    pair: params.pair,
    a: params.pair.split("-vs-")[0]?.toUpperCase() || "NVO",
    b: params.pair.split("-vs-")[1]?.toUpperCase() || "LLY",
    label: `${params.pair.toUpperCase()} Comparison`
  };

  const symA = match.a;
  const symB = match.b;

  const itemA = SHARED_WATCHLIST_ITEMS.find(i => i.symbol.toUpperCase() === symA);
  const itemB = SHARED_WATCHLIST_ITEMS.find(i => i.symbol.toUpperCase() === symB);

  const nameA = itemA?.name || symA;
  const nameB = itemB?.name || symB;

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebPage",
      "name": `${match.label} - Quantitative Analysis Matrix`,
      "description": `Side-by-side financial comparison between ${symA} and ${symB}.`,
      "url": `https://www.arxterminal.com/compare/${params.pair.toLowerCase()}/`,
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
          "name": "Comparison Matrix",
          "item": "https://www.arxterminal.com/compare/"
        },
        {
          "@type": "ListItem",
          "position": 3,
          "name": `${symA} vs. ${symB}`,
          "item": `https://www.arxterminal.com/compare/${params.pair.toLowerCase()}/`
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
          <Link href="/compare" className="hover:text-cyan-400">Comparison Matrix</Link>
          <span>/</span>
          <span className="text-slate-300 font-bold">{symA} vs. {symB}</span>
        </nav>

        {/* Hero Header */}
        <header className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-3">
          <span className="px-2.5 py-1 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-xs font-bold font-mono">
            HEAD-TO-HEAD QUANTITATIVE MATRIX
          </span>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            {itemA?.name || symA} ({symA}) vs. {itemB?.name || symB} ({symB})
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
            Side-by-side institutional breakdown evaluating fundamental balance sheet health, growth trajectory, Piotroski F-Scores, and volatility invalidation levels.
          </p>
        </header>

        {/* 1-Click Interactive Matrix CTA */}
        <section className="bg-gradient-to-r from-cyan-950/40 via-[#0b1019] to-purple-950/40 p-5 rounded-2xl border border-cyan-800/60 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center sm:text-left">
            <h2 className="text-sm sm:text-base font-bold text-white">
              Launch Live Interactive Comparison Matrix
            </h2>
            <p className="text-xs text-slate-300 font-sans">
              Filter by Day Trader scalping metrics, macro correlation coefficients, and institutional holdings.
            </p>
          </div>
          <Link
            href={`/compare?a=${symA}&b=${symB}`}
            className="w-full sm:w-auto px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-black text-xs font-extrabold rounded-xl shadow-lg transition-transform active:scale-95 text-center whitespace-nowrap"
          >
            Launch Interactive Comparison →
          </Link>
        </section>

        {/* Side-by-Side Comparison Table */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-4">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">
            Quantitative Scorecard Comparison
          </h2>

          <ComparePairMatrix
            symA={symA}
            symB={symB}
            nameA={nameA}
            nameB={nameB}
          />
        </section>

        {/* More Preset Comparisons */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-3">
          <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Explore Other High-Alpha Comparisons
          </h2>
          <div className="flex flex-wrap gap-2 text-xs">
            {COMPARISON_PAIRS.filter(p => p.pair !== params.pair).map(p => (
              <Link
                key={p.pair}
                href={`/compare/${p.pair}/`}
                className="px-3 py-1.5 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-slate-300 hover:text-cyan-300 border border-[#243044] transition-colors"
              >
                {p.label}
              </Link>
            ))}
          </div>
        </section>

        {/* Footer Navigation */}
        <footer className="border-t border-[#1e293b] pt-6 flex flex-wrap items-center justify-between gap-4 text-xs">
          <Link
            href="/"
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl transition-transform active:scale-95"
          >
            ← Return to Live Terminal
          </Link>
          <div className="text-slate-500 font-sans">
            Grounded in SEC EDGAR, Capitol Hill STOCK Act & Federal Reserve FRED Data
          </div>
        </footer>
      </main>
    </div>
  );
}
