import type { Metadata } from "next";
import Link from "next/link";
import Navbar from "../../components/Navbar";

export const metadata: Metadata = {
  title: "Quantitative Terminal User Guide & Algorithmic Methodologies",
  description:
    "Comprehensive guide to Congressional STOCK Act tracking, Mark Minervini VCP algorithmic entry points, Cornish-Fisher Modified VaR risk modeling, and FRED macroeconomic regimes.",
  openGraph: {
    title: "Finance Terminal: Quantitative User Guide & Methodologies",
    description: "Master institutional quantitative trading, legislative STOCK Act signals, and risk invalidation frameworks.",
    url: "https://finance-xp8.pages.dev/guide",
    siteName: "Finance Terminal",
    type: "article",
  },
  alternates: {
    canonical: "https://finance-xp8.pages.dev/guide",
  },
};

export default function GuidePage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    "headline": "Quantitative Terminal User Guide & Algorithmic Methodologies",
    "description": "Comprehensive guide to Congressional STOCK Act tracking, Mark Minervini VCP algorithmic entry points, Cornish-Fisher Modified VaR risk modeling, and FRED macroeconomic regimes.",
    "author": {
      "@type": "Organization",
      "name": "Finance Terminal Quantitative Intelligence"
    },
    "publisher": {
      "@type": "Organization",
      "name": "Finance Terminal",
      "logo": {
        "@type": "ImageObject",
        "url": "https://finance-xp8.pages.dev/icons/icon-512x512.png"
      }
    },
    "datePublished": "2026-08-26",
    "dateModified": "2026-08-26"
  };

  return (
    <div className="min-h-screen bg-[#070a10] text-slate-100 font-sans selection:bg-cyan-500 selection:text-black">
      {/* Schema.org TechArticle Structured Data for SEO / GEO Rich Results */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <Navbar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 font-mono space-y-8 sm:space-y-12">
        {/* Hero Header */}
        <header className="border-b border-[#243044] pb-6 sm:pb-8 space-y-3">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-[11px] font-bold">
              DOCUMENTATION & SPECIFICATION
            </span>
            <span className="text-slate-500 text-xs">• 100% Grounded Provenance</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight">
            Quantitative Platform User Guide & Algorithmic Blueprint
          </h1>
          <p className="text-sm sm:text-base text-slate-400 leading-relaxed font-sans">
            A comprehensive reference manual detailing how our mathematical models calculate entry ladders, stop-loss invalidation thresholds, Congressional STOCK Act disclosure conviction, and macroeconomic regimes.
          </p>
        </header>

        {/* Section 1: In-Terminal Modular Workspaces */}
        <section id="workspaces" className="space-y-4">
          <h2 className="text-lg sm:text-xl font-bold text-cyan-400 flex items-center gap-2 border-b border-[#1b2434] pb-2">
            <span>🗂️ 1. The 4 Modular In-Terminal Workspaces</span>
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            To eliminate cognitive overload, the Terminal groups deep analytical tools into 4 dedicated workspace tabs directly beneath the live candlestick chart:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-1.5">
              <strong className="text-cyan-300 text-sm flex items-center gap-1.5">
                <span>🎯</span>
                <span>Execution & Levels</span>
              </strong>
              <p className="text-xs text-slate-400">
                Calculates live optimal accumulation ranges, 14-period Turtle ATR stop-losses, and multi-tier take-profit targets with a minimum 2:1 reward-to-risk requirement.
              </p>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-1.5">
              <strong className="text-amber-300 text-sm flex items-center gap-1.5">
                <span>🏛️</span>
                <span>Smart Money & Insiders</span>
              </strong>
              <p className="text-xs text-slate-400">
                Tracks official US House & Senate STOCK Act disclosures, SEC Form 4 insider transactions, and unusual options flow call sweeps with dark pool volume context.
              </p>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-1.5">
              <strong className="text-emerald-300 text-sm flex items-center gap-1.5">
                <span>📊</span>
                <span>Factors & Macro</span>
              </strong>
              <p className="text-xs text-slate-400">
                Displays 5-factor radar scores (Growth, Quality, Valuation, Momentum, Tail Risk), 9-point Piotroski F-Scores, and Federal Reserve yield curve regime spreads.
              </p>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-1.5">
              <strong className="text-purple-300 text-sm flex items-center gap-1.5">
                <span>🛡️</span>
                <span>Risk & Contagion</span>
              </strong>
              <p className="text-xs text-slate-400">
                Maps directed supply-chain and peer contagion network graphs alongside continuous self-healing forecast accuracy calibration audits.
              </p>
            </div>
          </div>
        </section>

        {/* Section 2: Algorithmic Execution Playbook */}
        <section id="algorithmic-playbook" className="space-y-4">
          <h2 className="text-lg sm:text-xl font-bold text-cyan-400 flex items-center gap-2 border-b border-[#1b2434] pb-2">
            <span>🧮 2. Mathematical Execution Formulas & Risk Rules</span>
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            Our execution engine replaces discretionary guesswork with rigorous algorithmic boundaries:
          </p>

          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              A. Dual Take-Profit Ladder (TP1 & TP2)
            </h3>
            <ul className="text-xs text-slate-300 space-y-2 list-disc pl-5">
              <li>
                <strong>Target 1 (Scale 50% Position)</strong>: Computed as Spot + 2.5x ATR14 (Long-Term) or Spot + 1.75x ATR14 (Day Trader). Once reached, stop-loss is raised to breakeven.
              </li>
              <li>
                <strong>Target 2 (Extended Runner Exit)</strong>: Computed as Spot + 4.5x ATR14 (Long-Term) or Spot + 3.0x ATR14 (Day Trader) at channel exhaustion.
              </li>
            </ul>

            <h3 className="text-sm font-bold text-rose-400 uppercase tracking-wider pt-2">
              B. Volatility Invalidation Stop-Loss
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Calculated dynamically via 14-period Average True Range: <code>Stop Loss = Entry_Min - 1.8 * ATR14</code>. If daily price closes below this floor, the thesis is structurally invalidated.
            </p>
          </div>
        </section>

        {/* Section 3: Congressional STOCK Act Disclosures */}
        <section id="stock-act" className="space-y-4">
          <h2 className="text-lg sm:text-xl font-bold text-amber-400 flex items-center gap-2 border-b border-[#1b2434] pb-2">
            <span>🏛️ 3. Congressional STOCK Act & Smart Money Tracking</span>
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            Under Public Law 112-105 (2012 STOCK Act), members of the US Congress must disclose securities transactions within 45 days. Our system evaluates:
          </p>
          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2 text-xs text-slate-300">
            <div>• <strong>Filing Delay Penalty</strong>: Discounts trades reported near the 45-day statutory deadline to avoid copycat top-buying.</div>
            <div>• <strong>Jurisdiction Overlap</strong>: Cross-references member committee assignments (e.g. Armed Services, Energy & Commerce) against ticker sector classification.</div>
            <div>• <strong>Direct Statutory Source Linking</strong>: Every trade includes a direct link to the Office of the Clerk (US House) or Electronic Financial Disclosure (US Senate).</div>
          </div>
        </section>

        {/* Section 4: Quantitative Risk Modeling */}
        <section id="risk-framework" className="space-y-4">
          <h2 className="text-lg sm:text-xl font-bold text-purple-400 flex items-center gap-2 border-b border-[#1b2434] pb-2">
            <span>🛡️ 4. Cornish-Fisher Modified Value-at-Risk (M-VaR)</span>
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            Standard Gaussian VaR assumes normal distribution, dangerously underestimating real-world crash risk. Our engine applies Cornish-Fisher expansion to calibrate for asset skewness (S) and excess kurtosis (K):
          </p>
          <div className="bg-[#090d14] p-4 rounded-xl border border-[#1e293b] font-mono text-xs text-cyan-300">
            Z_cf = z_alpha + (z_alpha^2 - 1)*S / 6 + (z_alpha^3 - 3*z_alpha)*K / 24 - (2*z_alpha^3 - 5*z_alpha)*S^2 / 36
          </div>
        </section>

        {/* Footer Navigation */}
        <footer className="border-t border-[#243044] pt-6 flex flex-wrap items-center justify-between gap-4">
          <Link
            href="/"
            className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-xs font-bold rounded-xl shadow transition-transform active:scale-95"
          >
            ← Return to Quantitative Terminal
          </Link>
          <div className="text-xs text-slate-500">
            Grounded in SEC EDGAR, Capitol Hill STOCK Act & Federal Reserve FRED Data
          </div>
        </footer>
      </main>
    </div>
  );
}