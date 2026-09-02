import type { Metadata } from "next";
import Link from "next/link";
import Navbar from "../../../components/Navbar";

export const metadata: Metadata = {
  title: "🛑 Congressional Late-Filer Hall of Shame: STOCK Act Violations & Decay | ARX Terminal",
  description: "Audit US House & Senate politicians violating the 45-day statutory STOCK Act disclosure deadline. Review filing latency, -32 pt staleness decay penalties, and mean-reversion risks.",
  openGraph: {
    title: "Congressional Late-Filer Hall of Shame & STOCK Act Violations",
    description: "Track delayed Congressional stock disclosures, statutory compliance violations, and mathematical staleness decay penalties.",
    url: "https://www.arxterminal.com/smart-money/late-filers/",
    siteName: "ARX Terminal",
    type: "article",
  },
  alternates: {
    canonical: "https://www.arxterminal.com/smart-money/late-filers/",
  },
};

export default function LateFilersPage() {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "Dataset",
      "name": "Congressional STOCK Act Late Filers Ledger",
      "description": "Comprehensive audit of US Congressional stock transactions disclosed after the statutory 45-day deadline.",
      "url": "https://www.arxterminal.com/smart-money/late-filers/",
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
          "name": "Smart Money",
          "item": "https://www.arxterminal.com/smart-money/"
        },
        {
          "@type": "ListItem",
          "position": 3,
          "name": "Late-Filer Hall of Shame",
          "item": "https://www.arxterminal.com/smart-money/late-filers/"
        }
      ]
    }
  ];

  const LATE_TRADES = [
    {
      politician: "Sen. Tommy Tuberville (R-AL)",
      politicianSlug: "tommy-tuberville",
      ticker: "CELH",
      assetName: "Celsius Holdings",
      amount: "$100,000 - $250,000",
      txDate: "2026-06-25",
      filingDate: "2026-08-22",
      lagDays: 58,
      penalty: "-32 Pts (Late Filer)",
      adjustedScore: 48,
      thesis: "58-day filing lag severely violates 45-day STOCK Act statutory limit; alpha is already fully priced into public markets."
    },
    {
      politician: "Rep. Mark Green (R-TN)",
      politicianSlug: "mark-green",
      ticker: "PFE",
      assetName: "Pfizer Inc.",
      amount: "$50,000 - $100,000",
      txDate: "2026-06-18",
      filingDate: "2026-08-06",
      lagDays: 49,
      penalty: "-32 Pts (Late Filer)",
      adjustedScore: 45,
      thesis: "Disclosed 4 days past statutory limit. Time-decay penalty triggers high mean-reversion risk."
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
          <Link href="/smart-money" className="hover:text-cyan-400">Smart Money</Link>
          <span>/</span>
          <span className="text-slate-300 font-bold">Late Filers</span>
        </nav>

        {/* Hero Header */}
        <header className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-rose-900/60 space-y-3">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded bg-rose-950/80 text-rose-400 border border-rose-800 text-xs font-bold font-mono">
              🛑 STATUTORY COMPLIANCE SURVEILLANCE
            </span>
            <span className="text-slate-500 text-xs">• Public Law 112-105</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Congressional Late-Filer Hall of Shame
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
            Under the Stop Trading on Congressional Knowledge (STOCK) Act of 2012, members of Congress are legally mandated to disclose securities transactions within <strong>45 days</strong> of execution. When politicians file late, any strategic informational advantage has already decayed into public market noise.
          </p>
        </header>

        {/* 1-Click Interactive CTA */}
        <section className="bg-gradient-to-r from-rose-950/40 via-[#0b1019] to-purple-950/40 p-5 rounded-2xl border border-rose-800/60 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center sm:text-left">
            <h2 className="text-sm sm:text-base font-bold text-white">
              Filter High-Latency Disclosures in Live Scanner
            </h2>
            <p className="text-xs text-slate-300 font-sans">
              Isolate fresh trades (&lt;15d lag) and automatically discount delayed filings.
            </p>
          </div>
          <Link
            href="/smart-money"
            className="w-full sm:w-auto px-5 py-2.5 bg-rose-500 hover:bg-rose-400 text-black text-xs font-extrabold rounded-xl shadow-lg transition-transform active:scale-95 text-center whitespace-nowrap"
          >
            Launch Live Smart Money Scanner →
          </Link>
        </section>

        {/* Mathematical Decay Tier Explanation */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-4">
          <h2 className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
            <span>⏱️ Mathematical Staleness Decay Penalty Formula</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="bg-[#06090f] p-3 rounded-xl border border-emerald-900/50 space-y-1">
              <strong className="text-emerald-400 block font-mono">⚡ Fresh (&lt;15 Days)</strong>
              <span className="text-slate-300 block font-mono">0 Pt Penalty</span>
              <span className="text-[10px] text-slate-500 font-sans block">High Informational Alpha</span>
            </div>
            <div className="bg-[#06090f] p-3 rounded-xl border border-cyan-900/50 space-y-1">
              <strong className="text-cyan-400 block font-mono">⏳ Standard (16–30 Days)</strong>
              <span className="text-slate-300 block font-mono">-5 Pt Penalty</span>
              <span className="text-[10px] text-slate-500 font-sans block">Normal Statutory Processing</span>
            </div>
            <div className="bg-[#06090f] p-3 rounded-xl border border-amber-900/50 space-y-1">
              <strong className="text-amber-400 block font-mono">⚠️ Aging (31–45 Days)</strong>
              <span className="text-slate-300 block font-mono">-15 Pt Penalty</span>
              <span className="text-[10px] text-slate-500 font-sans block">Severe Time Decay</span>
            </div>
            <div className="bg-[#06090f] p-3 rounded-xl border border-rose-900/50 space-y-1">
              <strong className="text-rose-400 block font-mono">🛑 Late Filer (&gt;45 Days)</strong>
              <span className="text-slate-300 block font-mono">-32 Pt Penalty</span>
              <span className="text-[10px] text-slate-500 font-sans block">Statutory Breach / Noise</span>
            </div>
          </div>
        </section>

        {/* Late Filers Table */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e293b] pb-3">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>📋 Audited Statutory Violations Ledger</span>
            </h2>
            <span className="text-[11px] text-rose-400 font-sans">Late Filing Penalties Applied</span>
          </div>

          <div className="space-y-3">
            {LATE_TRADES.map((trade, idx) => (
              <div
                key={idx}
                className="bg-[#06090f] p-4 rounded-xl border border-rose-950/60 space-y-2.5 text-xs"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center space-x-2.5">
                    <Link
                      href={`/politician/${trade.politicianSlug}/`}
                      className="text-white font-bold hover:text-cyan-400"
                    >
                      {trade.politician}
                    </Link>
                    <Link
                      href={`/stock/${trade.ticker.toLowerCase()}/`}
                      className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 font-bold hover:underline"
                    >
                      {trade.ticker}
                    </Link>
                    <span className="text-slate-400">({trade.assetName})</span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800 font-bold">
                      🛑 {trade.lagDays} Days Lag ({trade.penalty})
                    </span>
                    <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">
                      Score: {trade.adjustedScore}/100
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] text-slate-400 pt-1">
                  <div><strong>Transaction Amount:</strong> <span className="text-slate-200 font-mono">{trade.amount}</span></div>
                  <div><strong>Executed:</strong> <span className="text-slate-200 font-mono">{trade.txDate}</span></div>
                  <div><strong>Filed:</strong> <span className="text-rose-300 font-mono">{trade.filingDate}</span></div>
                </div>

                <p className="text-[11px] text-slate-300 font-sans leading-relaxed pt-1 border-t border-[#141b26]">
                  <strong>Risk Assessment:</strong> {trade.thesis}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Footer Navigation */}
        <footer className="border-t border-[#1e293b] pt-6 flex flex-wrap items-center justify-between gap-4 text-xs">
          <Link
            href="/smart-money"
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl transition-transform active:scale-95"
          >
            ← Return to Smart Money Feeds
          </Link>
          <div className="text-slate-500 font-sans">
            Audited Against Public Law 112-105 Statutory Mandates
          </div>
        </footer>
      </main>
    </div>
  );
}
