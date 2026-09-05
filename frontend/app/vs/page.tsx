import type { Metadata } from "next";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import AuthorEeatBadge from "../../components/AuthorEeatBadge";
import { COMPETITOR_CATALOG } from "../../lib/competitorCatalog";

export const metadata: Metadata = {
  title: "Trading Terminal Comparisons & Platform Alternatives",
  description:
    "Objective, feature-by-feature comparisons between ARX Terminal and alternative financial data platforms: Quiver Quantitative, Unusual Whales, Koyfin, and Bloomberg Terminal.",
  alternates: {
    canonical: "https://www.arxterminal.com/vs/",
  },
  openGraph: {
    title: "Financial Terminal & Trading Platform Comparisons | ARX Terminal",
    description:
      "Compare ARX Terminal with Quiver Quantitative, Unusual Whales, Koyfin, and Bloomberg. See feature matrices, pricing differences, and algorithmic execution capabilities.",
    url: "https://www.arxterminal.com/vs/",
    siteName: "ARX Terminal",
    type: "website",
  },
};

export default function ComparisonIndexPage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    "name": "ARX Terminal Platform Comparisons & Alternatives",
    "description": "Comprehensive comparative analyses between ARX Terminal and leading financial research and alternative data platforms.",
    "url": "https://www.arxterminal.com/vs/",
    "mainEntity": {
      "@type": "ItemList",
      "itemListElement": COMPETITOR_CATALOG.map((comp, idx) => ({
        "@type": "ListItem",
        "position": idx + 1,
        "name": `ARX Terminal vs ${comp.competitorName}`,
        "url": `https://www.arxterminal.com/vs/${comp.slug}/`
      }))
    }
  };

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans antialiased">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Header Breadcrumb & Title */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <Link href="/" className="hover:text-cyan-400 transition-colors">Terminal</Link>
            <span>/</span>
            <span className="text-cyan-400">Platform Comparisons</span>
          </div>

          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1b273d] pb-6">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-3">
                <span className="text-cyan-400 font-mono">⚔️</span>
                <span>Platform Comparisons & Market Alternatives</span>
              </h1>
              <p className="mt-2 text-sm text-slate-400 max-w-3xl leading-relaxed">
                Objective, feature-by-feature technical breakdowns evaluating how ARX Terminal compares with legacy terminals, retail alternative data trackers, and modern charting suites.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/glossary/"
                className="px-3.5 py-2 rounded-lg bg-[#111a2c] hover:bg-[#18263f] border border-[#223554] text-xs font-mono text-cyan-300 transition-all shadow-sm"
              >
                Quant Glossary 📚
              </Link>
              <Link
                href="/screener/"
                className="px-3.5 py-2 rounded-lg bg-[#111a2c] hover:bg-[#18263f] border border-[#223554] text-xs font-mono text-slate-200 transition-all shadow-sm"
              >
                Live Screener ⚡
              </Link>
            </div>
          </div>
        </div>

        {/* E-E-A-T Credibility Badge */}
        <AuthorEeatBadge
          topic="Financial Platform Architecture & Market Intelligence"
          lastUpdated="September 2026"
        />

        {/* Comparison Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {COMPETITOR_CATALOG.map((comp) => (
            <div
              key={comp.slug}
              className="flex flex-col justify-between p-6 rounded-2xl bg-[#090f1d] border border-[#18263d] hover:border-cyan-500/50 transition-all duration-200 space-y-6 shadow-sm"
            >
              <div className="space-y-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <span className="text-[11px] font-mono text-cyan-400 uppercase tracking-wider">
                      Comparison Analysis
                    </span>
                    <h2 className="text-xl font-bold text-white mt-0.5">
                      ARX Terminal <span className="text-slate-500 font-normal">vs</span> {comp.competitorName}
                    </h2>
                    <p className="text-xs font-mono text-slate-400 mt-1">
                      {comp.tagline}
                    </p>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-mono bg-[#111c30] border border-[#203352] text-slate-300">
                    {comp.competitorDomain}
                  </span>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">
                  {comp.summary}
                </p>

                {/* Pricing Teaser */}
                <div className="p-3.5 rounded-xl bg-[#050912] border border-[#141f33] grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div className="text-[10px] font-mono text-slate-400">ARX TERMINAL</div>
                    <div className="text-cyan-400 font-semibold mt-0.5">{comp.pricingComparison.arx}</div>
                  </div>
                  <div className="border-l border-[#1b2b46] pl-3">
                    <div className="text-[10px] font-mono text-slate-400">{comp.competitorName.toUpperCase()}</div>
                    <div className="text-slate-300 font-semibold mt-0.5">{comp.pricingComparison.competitor}</div>
                  </div>
                </div>

                {/* Feature Highlights */}
                <div className="space-y-1.5 pt-1">
                  <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider font-semibold">
                    Core ARX Edge:
                  </div>
                  <ul className="space-y-1 text-xs text-slate-300">
                    {comp.keyAdvantagesArx.slice(0, 3).map((adv, idx) => (
                      <li key={idx} className="flex items-center gap-2">
                        <span className="text-cyan-400 font-bold">✓</span>
                        <span>{adv}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <Link
                href={`/vs/${comp.slug}/`}
                className="w-full text-center py-2.5 rounded-xl bg-[#101b2e] hover:bg-cyan-500 hover:text-slate-950 text-cyan-300 text-xs font-mono font-semibold transition-all duration-200 border border-[#213554] hover:border-cyan-400 shadow-sm"
              >
                Read Full Technical Comparison →
              </Link>
            </div>
          ))}
        </div>

        {/* Evaluation Methodology Note */}
        <div className="p-6 rounded-2xl bg-[#070b14] border border-[#17243c] space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span className="text-cyan-400">⚖️</span> Objective Evaluation Standard
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Our comparison benchmarks evaluate platforms strictly on empirical functional capabilities: data provenance, execution geometry, downside tail-risk accounting, statutory disclosure integrity, and latency. ARX Terminal does not accept affiliate sponsorships or promotional placement fees from any financial data vendor.
          </p>
        </div>
      </main>

      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }}
      />
    </div>
  );
}
