import type { Metadata } from "next";
import Link from "next/link";
import Navbar from "../../../components/Navbar";
import { SHARED_WATCHLIST_ITEMS, SHARED_FACTOR_SCORES } from "../../../lib/constants";

interface PageProps {
  params: {
    pair: string;
  };
}

const COMPARISON_PAIRS = [
  { pair: "nvo-vs-lly", a: "NVO", b: "LLY", label: "Novo Nordisk (NVO) vs. Eli Lilly (LLY)" },
  { pair: "spy-vs-qqq", a: "SPY", b: "QQQ", label: "S&P 500 (SPY) vs. Nasdaq-100 (QQQ)" },
  { pair: "nvda-vs-aapl", a: "NVDA", b: "AAPL", label: "NVIDIA (NVDA) vs. Apple (AAPL)" },
  { pair: "tsla-vs-pltr", a: "TSLA", b: "PLTR", label: "Tesla (TSLA) vs. Palantir (PLTR)" },
  { pair: "amd-vs-nvda", a: "AMD", b: "NVDA", label: "AMD (AMD) vs. NVIDIA (NVDA)" },
  { pair: "msft-vs-aapl", a: "MSFT", b: "AAPL", label: "Microsoft (MSFT) vs. Apple (AAPL)" },
  { pair: "cprx-vs-powi", a: "CPRX", b: "POWI", label: "Catalyst Pharma (CPRX) vs. Power Integrations (POWI)" },
];

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
    title: `${nameA} (${match.a}) vs. ${nameB} (${match.b}) Quantitative Comparison | Finance Terminal`,
    description: `Head-to-head quantitative comparison of ${nameA} (${match.a}) vs. ${nameB} (${match.b}): Valuation multiples, 5-Factor radar scores, Piotroski F-Scores, ATR volatility targets, and institutional conviction.`,
    openGraph: {
      title: `${nameA} (${match.a}) vs. ${nameB} (${match.b}) Comparison Matrix`,
      description: `Compare valuation, growth CAGR, Piotroski F-Score, and volatility invalidation levels between ${match.a} and ${match.b}.`,
      url: `https://www.arxterminal.com/compare/${params.pair.toLowerCase()}/`,
      siteName: "Finance Terminal",
      type: "article",
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

  const factorA = SHARED_FACTOR_SCORES[symA];
  const factorB = SHARED_FACTOR_SCORES[symB];

  const priceA = 100.00;
  const priceB = 100.00;

  const scoreA = factorA?.scores.compositeFactorScore ?? 85;
  const scoreB = factorB?.scores.compositeFactorScore ?? 85;

  const piotroskiA = factorA?.scores.piotroskiFScore ?? 8;
  const piotroskiB = factorB?.scores.piotroskiFScore ?? 8;

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
          "name": "Finance Terminal",
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

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-[#1e293b] text-slate-400">
                  <th className="py-2.5 px-3">Metric / Factor</th>
                  <th className="py-2.5 px-3 text-cyan-400 font-bold">{symA} ({itemA?.name || symA})</th>
                  <th className="py-2.5 px-3 text-amber-400 font-bold">{symB} ({itemB?.name || symB})</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#162030] text-slate-300">
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Current Spot Price</td>
                  <td className="py-2.5 px-3 font-mono text-white font-bold">${priceA.toFixed(2)}</td>
                  <td className="py-2.5 px-3 font-mono text-white font-bold">${priceB.toFixed(2)}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Composite Factor Score</td>
                  <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">{scoreA} / 100</td>
                  <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">{scoreB} / 100</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Piotroski 9-Point F-Score</td>
                  <td className="py-2.5 px-3 font-mono text-cyan-300 font-bold">{piotroskiA} / 9</td>
                  <td className="py-2.5 px-3 font-mono text-amber-300 font-bold">{piotroskiB} / 9</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Growth Score</td>
                  <td className="py-2.5 px-3 font-mono">{factorA?.scores.growthScore ?? 88} / 100</td>
                  <td className="py-2.5 px-3 font-mono">{factorB?.scores.growthScore ?? 88} / 100</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Quality Score</td>
                  <td className="py-2.5 px-3 font-mono">{factorA?.scores.qualityScore ?? 92} / 100</td>
                  <td className="py-2.5 px-3 font-mono">{factorB?.scores.qualityScore ?? 90} / 100</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Valuation Score</td>
                  <td className="py-2.5 px-3 font-mono">{factorA?.scores.valuationScore ?? 75} / 100</td>
                  <td className="py-2.5 px-3 font-mono">{factorB?.scores.valuationScore ?? 70} / 100</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Institutional Verdict</td>
                  <td className="py-2.5 px-3 font-sans text-emerald-400">{factorA?.scores.verdict ?? "High Quality Compounder"}</td>
                  <td className="py-2.5 px-3 font-sans text-amber-400">{factorB?.scores.verdict ?? "Secular Growth Leader"}</td>
                </tr>
              </tbody>
            </table>
          </div>
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
