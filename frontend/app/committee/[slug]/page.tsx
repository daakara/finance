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
  COMMITTEE_DATABASE,
  type CommitteeDefinition,
  type CommitteeTrade,
} from "../../../lib/seoCatalogs";

export function generateStaticParams() {
  return COMMITTEE_DATABASE.map(c => ({ slug: c.slug }));
}

export function generateMetadata({ params }: PageProps): Metadata {
  const committee = COMMITTEE_DATABASE.find(c => c.slug === params.slug.toLowerCase());
  if (!committee) {
    return {
      title: "Congressional Committee Not Found | ARX Terminal",
      description: "The requested Congressional Committee profile could not be found.",
    };
  }

  return {
    title: `🏛️ ${committee.name} Stock Trades & Legislative Conflict Tracking`,
    description: `Track securities transactions and STOCK Act disclosures by members of the ${committee.name}. Review Legislative Alignment Index (0-100) and regulatory oversight conflicts.`,
    openGraph: {
      title: `${committee.name} Congressional Stock Trading Hub`,
      description: committee.jurisdictionSummary,
      url: `https://www.arxterminal.com/committee/${params.slug.toLowerCase()}/`,
      siteName: "ARX Terminal",
      type: "article",
    },
    twitter: {
      card: "summary_large_image",
      title: `🏛️ ${committee.name} Stock Trades & Legislative Conflict Tracking`,
      description: `Track securities transactions and STOCK Act disclosures by members of the ${committee.name}.`,
      images: ["/og-image.png"],
      creator: "@ARXTerminal",
    },
    alternates: {
      canonical: `https://www.arxterminal.com/committee/${params.slug.toLowerCase()}/`,
    },
  };
}

export default function CommitteeHubPage({ params }: PageProps) {
  const committee = COMMITTEE_DATABASE.find(c => c.slug === params.slug.toLowerCase());
  if (!committee) {
    notFound();
  }

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "Dataset",
      "name": `${committee.name} Stock Disclosures`,
      "description": committee.jurisdictionSummary,
      "url": `https://www.arxterminal.com/committee/${params.slug.toLowerCase()}/`,
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
          "name": "Smart Money & Congressional Disclosures",
          "item": "https://www.arxterminal.com/smart-money/"
        },
        {
          "@type": "ListItem",
          "position": 3,
          "name": committee.name,
          "item": `https://www.arxterminal.com/committee/${params.slug.toLowerCase()}/`
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
          <span className="text-slate-300 font-bold">{committee.name}</span>
        </nav>

        {/* Forensic Research Provenance Banner */}
        <div className="p-3 rounded-xl bg-purple-950/40 border border-purple-800/60 text-xs text-purple-200 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span>🏛️</span>
            <span><strong>Legislative Oversight & Conflict Forensic Dossier:</strong> Audited statutory oversight areas, committee membership securities transactions, and conflict-of-interest alignment.</span>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded bg-purple-900/60 border border-purple-700/80 font-bold uppercase shrink-0 hidden sm:inline">
            Forensic Audit
          </span>
        </div>

        {/* Hero Header */}
        <header className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-3">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded bg-purple-950/80 text-purple-300 border border-purple-800 text-xs font-bold font-mono">
              CONGRESSIONAL COMMITTEE JURISDICTION
            </span>
            <span className="text-slate-500 text-xs">• {committee.chamber.toUpperCase()}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            {committee.name}
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
            {committee.jurisdictionSummary}
          </p>
          <div className="flex flex-wrap gap-1.5 pt-2">
            {committee.regulatedSectors.map((sector, idx) => (
              <span key={idx} className="px-2 py-0.5 rounded bg-[#111722] text-slate-300 border border-[#243044] text-[11px]">
                🎯 {sector}
              </span>
            ))}
          </div>
        </header>

        {/* 1-Click Interactive CTA */}
        <section className="bg-gradient-to-r from-purple-950/40 via-[#0b1019] to-cyan-950/40 p-5 rounded-2xl border border-purple-800/60 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center sm:text-left">
            <h2 className="text-sm sm:text-base font-bold text-white">
              Launch Live Smart Money & Legislative Scanner
            </h2>
            <p className="text-xs text-slate-300 font-sans">
              Filter real-time disclosures by committee jurisdiction, dollar brackets, and filing latency.
            </p>
          </div>
          <Link
            href="/smart-money"
            className="w-full sm:w-auto px-5 py-2.5 bg-purple-500 hover:bg-purple-400 text-black text-xs font-extrabold rounded-xl shadow-lg transition-transform active:scale-95 text-center whitespace-nowrap"
          >
            Launch Smart Money Scanner →
          </Link>
        </section>

        {/* Active Trading Committee Members */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-3">
          <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Key Active Trading Committee Members
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {committee.keyMembers.map((member, idx) => (
              <Link
                key={idx}
                href={`/politician/${member.slug}/`}
                className="bg-[#111722] hover:bg-[#1a2332] p-3.5 rounded-xl border border-[#243044] flex items-center justify-between transition-colors group"
              >
                <div>
                  <strong className="text-white block font-bold">{member.name}</strong>
                  <span className="text-slate-500 text-[11px] font-sans">{member.party}</span>
                </div>
                <div className="text-right">
                  <span className="text-xs text-cyan-400 font-mono group-hover:translate-x-0.5 transition-transform">View Dossier →</span>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Relevant Trades Table */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-4">
          <div className="flex items-center justify-between border-b border-[#1e293b] pb-3">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>🏛️ High-Alignment Committee Disclosures</span>
            </h2>
            <span className="text-[11px] text-purple-400 font-sans">Jurisdiction Overlap Verified</span>
          </div>

          <div className="space-y-3">
            {committee.trades.map((trade, idx) => (
              <div
                key={idx}
                className="bg-[#06090f] p-4 rounded-xl border border-[#1b2434] space-y-2.5 text-xs"
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
                    <span className="px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">
                      ⚖️ Alignment: {trade.alignmentScore}/100
                    </span>
                    <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 font-bold text-[10px]">
                      {trade.stalenessBadge}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] text-slate-400 pt-1">
                  <div><strong>Transaction Size:</strong> <span className="text-slate-200 font-mono">{trade.amount}</span></div>
                  <div><strong>Execution Date:</strong> <span className="text-slate-200 font-mono">{trade.date}</span></div>
                  <div><strong>Trade Type:</strong> <span className="text-slate-200 font-mono">{trade.type}</span></div>
                </div>

                <p className="text-[11px] text-slate-300 font-sans leading-relaxed pt-1 border-t border-[#141b26]">
                  <strong>Jurisdiction Thesis:</strong> {trade.thesis}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Other Committee Hubs */}
        <section className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-3">
          <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Explore Other Congressional Committees
          </h2>
          <div className="flex flex-wrap gap-2 text-xs">
            {COMMITTEE_DATABASE.filter(c => c.slug !== params.slug).map(c => (
              <Link
                key={c.slug}
                href={`/committee/${c.slug}/`}
                className="px-3 py-1.5 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-slate-300 hover:text-cyan-300 border border-[#243044] transition-colors"
              >
                🏛️ {c.name.split("&")[0]}
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
            Grounded in House & Senate Committee Records and STOCK Act Disclosures
          </div>
        </footer>
      </main>
    </div>
  );
}
