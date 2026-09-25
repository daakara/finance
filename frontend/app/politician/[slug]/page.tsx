import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import Navbar from "../../../components/Navbar";

interface PageProps {
  params: {
    slug: string;
  };
}

import {
  POLITICIAN_DATABASE,
  type PoliticianProfile,
  type PoliticianTrade,
} from "../../../lib/seoCatalogs";

export function generateStaticParams() {
  return POLITICIAN_DATABASE.map(p => ({ slug: p.slug }));
}

export function generateMetadata({ params }: PageProps): Metadata {
  const profile = POLITICIAN_DATABASE.find(p => p.slug === params.slug.toLowerCase());
  if (!profile) {
    return {
      title: "Politician Profile Not Found | ARX Terminal",
      description: "The requested politician STOCK Act profile could not be found.",
    };
  }

  return {
    title: `🏛️ ${profile.name} Portfolio: Congressional STOCK Act Disclosures`,
    description: `Disclosures portfolio, committee assignments, and recent STOCK Act disclosures for ${profile.name}. Review Legislative Alignment scores and committee oversight conflicts.`,
    openGraph: {
      title: `🏛️ ${profile.name} Congressional Stock Trading Profile`,
      description: `Track securities transactions, committee oversight overlaps, and Legislative Alignment Index for ${profile.name}.`,
      url: `https://www.arxterminal.com/politician/${params.slug.toLowerCase()}/`,
      siteName: "ARX Terminal",
      type: "profile",
    },
    twitter: {
      card: "summary_large_image",
      title: `🏛️ ${profile.name} Portfolio: Congressional STOCK Act Disclosures`,
      description: `Track securities transactions, committee oversight overlaps, and Legislative Alignment Index for ${profile.name}.`,
      images: ["/og-image.png"],
      creator: "@ARXTerminal",
    },
    alternates: {
      canonical: `https://www.arxterminal.com/politician/${params.slug.toLowerCase()}/`,
    },
  };
}

export default function PoliticianProfilePage({ params }: PageProps) {
  const profile = POLITICIAN_DATABASE.find(p => p.slug === params.slug.toLowerCase());
  if (!profile) {
    notFound();
  }

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "ProfilePage",
      "mainEntity": {
        "@type": "Person",
        "name": profile.name,
        "jobTitle": `Member of the US ${profile.chamber}`,
        "description": `${profile.name} is a ${profile.party} representing ${profile.stateDistrict} in the United States ${profile.chamber}.`,
        "url": `https://www.arxterminal.com/politician/${params.slug.toLowerCase()}/`,
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
          "name": "Smart Money & Congressional Disclosures",
          "item": "https://www.arxterminal.com/smart-money/"
        },
        {
          "@type": "ListItem",
          "position": 3,
          "name": profile.name,
          "item": `https://www.arxterminal.com/politician/${params.slug.toLowerCase()}/`
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
          <Link href="/smart-money" className="hover:text-cyan-400">Smart Money</Link>
          <span>/</span>
          <span className="text-slate-300 font-bold">{profile.name}</span>
        </nav>

        {/* Forensic Research Provenance Banner */}
        <div className="p-3 rounded-xl bg-purple-950/40 border border-purple-800/60 text-xs text-purple-200 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span>📜</span>
            <span><strong>Verified Forensic Investigation Dossier:</strong> Audited STOCK Act disclosures, committee conflict analysis, and legislative timeline alignment.</span>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded bg-purple-900/60 border border-purple-700/80 font-bold uppercase shrink-0 hidden sm:inline">
            Public Law 112-105
          </span>
        </div>

        {/* Hero Header */}
        <header className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-3">
                <span className={`px-2.5 py-1 rounded text-xs font-bold font-mono ${
                  profile.party === "Democrat"
                    ? "bg-blue-950/80 text-blue-400 border border-blue-800"
                    : "bg-red-950/80 text-red-400 border border-red-800"
                }`}>
                  {profile.chamber.toUpperCase()} • {profile.party.toUpperCase()}
                </span>
                <span className="text-slate-400 text-xs font-sans">• {profile.stateDistrict}</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mt-1">
                {profile.name}
              </h1>
            </div>

            <div className="flex items-center space-x-4 text-right">
              <div className="bg-[#06090f] p-3 rounded-xl border border-[#1b2434]">
                <span className="text-[10px] text-slate-500 uppercase block">Curated Filings</span>
                <strong className="text-emerald-400 font-mono text-base">{profile.recentTrades.length}</strong>
              </div>
              <div className="bg-[#06090f] p-3 rounded-xl border border-[#1b2434]">
                <span className="text-[10px] text-slate-500 uppercase block">Committees</span>
                <strong className="text-cyan-400 font-mono text-base">{profile.committees.length}</strong>
              </div>
            </div>
          </div>

          {/* Committee Oversight */}
          <div className="pt-3 border-t border-[#1e293b] space-y-2 text-xs">
            <span className="text-slate-500 uppercase font-bold text-[11px] block">Committee Jurisdiction Oversight:</span>
            <div className="flex flex-wrap gap-1.5">
              {profile.committees.map((comm, idx) => (
                <span
                  key={idx}
                  className="px-2.5 py-1 rounded-lg bg-[#111722] text-slate-300 border border-[#243044] text-[11px]"
                >
                  ⚖️ {comm}
                </span>
              ))}
            </div>
          </div>
        </header>

        {/* 1-Click Interactive CTA */}
        <section className="bg-gradient-to-r from-purple-950/40 via-[#0b1019] to-cyan-950/40 p-5 rounded-2xl border border-purple-800/60 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center sm:text-left">
            <h2 className="text-sm sm:text-base font-bold text-white">
              Launch Live Congressional Smart Money Scanner
            </h2>
            <p className="text-xs text-slate-300 font-sans">
              Filter trades by fresh vs. late filers, committee jurisdiction conflicts, and unusual options flow.
            </p>
          </div>
          <Link
            href="/smart-money"
            className="w-full sm:w-auto px-5 py-2.5 bg-purple-500 hover:bg-purple-400 text-black text-xs font-extrabold rounded-xl shadow-lg transition-transform active:scale-95 text-center whitespace-nowrap"
          >
            Launch Smart Money Scanner →
          </Link>
        </section>

        {/* Recent Disclosures Ledger */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e293b] pb-3">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>🏛️ Audited STOCK Act Disclosures & Conviction Scores</span>
            </h2>
            <span className="text-[11px] text-slate-500 font-sans">Public Law 112-105 Compliant</span>
          </div>

          <div className="space-y-3">
            {profile.recentTrades.map((trade, idx) => (
              <div
                key={idx}
                className="bg-[#06090f] p-4 rounded-xl border border-[#1b2434] space-y-2.5 text-xs"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center space-x-2.5">
                    <Link
                      href={`/stock/${trade.ticker.toLowerCase()}/`}
                      className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 font-bold hover:underline"
                    >
                      {trade.ticker}
                    </Link>
                    <span className="text-white font-bold">{trade.assetName}</span>
                    <span className="text-slate-500 text-[11px]">({trade.type})</span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">
                      ⚖️ Alignment: {trade.alignmentScore}/100
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      trade.stalenessStatus === "FRESH"
                        ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                        : trade.stalenessStatus === "LATE_FILER"
                        ? "bg-rose-950 text-rose-400 border border-rose-800"
                        : "bg-cyan-950 text-cyan-400 border border-cyan-800"
                    }`}>
                      {trade.stalenessBadge}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] text-slate-400 pt-1">
                  <div><strong>Transaction Size:</strong> <span className="text-slate-200 font-mono">{trade.amount}</span></div>
                  <div><strong>Execution Date:</strong> <span className="text-slate-200 font-mono">{trade.date}</span></div>
                  <div><strong>Filing Latency:</strong> <span className="text-slate-200 font-mono">{trade.lagDays} Days</span></div>
                </div>

                <p className="text-[11px] text-slate-300 font-sans leading-relaxed pt-1 border-t border-[#141b26]">
                  <strong>Strategic Conflict Thesis:</strong> {trade.thesis}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Other Active Congressional Traders */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-3">
          <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Explore Other Active Congressional Portfolios
          </h2>
          <div className="flex flex-wrap gap-2 text-xs">
            {POLITICIAN_DATABASE.filter(p => p.slug !== params.slug).map(p => (
              <Link
                key={p.slug}
                href={`/politician/${p.slug}/`}
                className="px-3 py-1.5 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-slate-300 hover:text-cyan-300 border border-[#243044] transition-colors"
              >
                🏛️ {p.name} ({p.party[0]}-{p.stateDistrict.slice(0, 2)})
              </Link>
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
            Grounded in Office of the Clerk of the US House & Senate Office of Public Records
          </div>
        </footer>
      </main>
    </div>
  );
}
