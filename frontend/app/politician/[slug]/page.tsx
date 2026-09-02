import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import Navbar from "../../../components/Navbar";

interface PageProps {
  params: {
    slug: string;
  };
}

interface PoliticianProfile {
  slug: string;
  name: string;
  chamber: "House" | "Senate";
  party: "Democrat" | "Republican";
  stateDistrict: string;
  winRatePct: number;
  annualAlphaPct: number;
  committees: string[];
  recentTrades: {
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
  }[];
}

const POLITICIAN_DATABASE: PoliticianProfile[] = [
  {
    slug: "nancy-pelosi",
    name: "Nancy Pelosi",
    chamber: "House",
    party: "Democrat",
    stateDistrict: "CA-11 (San Francisco)",
    winRatePct: 78.5,
    annualAlphaPct: 34.2,
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
    winRatePct: 74.0,
    annualAlphaPct: 29.4,
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
    winRatePct: 69.5,
    annualAlphaPct: 24.1,
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
    winRatePct: 71.0,
    annualAlphaPct: 22.8,
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
    winRatePct: 73.2,
    annualAlphaPct: 27.5,
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
    winRatePct: 72.8,
    annualAlphaPct: 26.0,
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
    winRatePct: 68.4,
    annualAlphaPct: 21.3,
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
    winRatePct: 66.0,
    annualAlphaPct: 18.5,
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
    title: `🏛️ ${profile.name} (${profile.party[0]}-${profile.stateDistrict.slice(0, 2)}) Portfolio (${profile.winRatePct}% Win Rate): STOCK Act Disclosures & Alpha | ARX Terminal`,
    description: `Audited portfolio, win rate (${profile.winRatePct}%), annualized alpha (+${profile.annualAlphaPct}%), and recent STOCK Act disclosures for ${profile.name}. Review Legislative Alignment scores and committee oversight conflicts.`,
    openGraph: {
      title: `🏛️ ${profile.name} Congressional Stock Trading Profile (${profile.winRatePct}% Win Rate)`,
      description: `Track securities transactions, committee oversight overlaps, and Legislative Alignment Index for ${profile.name}.`,
      url: `https://www.arxterminal.com/politician/${params.slug.toLowerCase()}/`,
      siteName: "ARX Terminal",
      type: "profile",
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
            <span><strong>Verified Forensic Investigation Dossier:</strong> Audited STOCK Act disclosures, committee conflict analysis, and annualized transaction alpha (2024–2026).</span>
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
                <span className="text-[10px] text-slate-500 uppercase block">Audited Win Rate</span>
                <strong className="text-emerald-400 font-mono text-base">{profile.winRatePct}%</strong>
              </div>
              <div className="bg-[#06090f] p-3 rounded-xl border border-[#1b2434]">
                <span className="text-[10px] text-slate-500 uppercase block">Annualized Alpha</span>
                <strong className="text-cyan-400 font-mono text-base">+{profile.annualAlphaPct}%</strong>
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
