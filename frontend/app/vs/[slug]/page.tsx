import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import Navbar from "../../../components/Navbar";
import AuthorEeatBadge from "../../../components/AuthorEeatBadge";
import { COMPETITOR_CATALOG, CompetitorComparison } from "../../../lib/competitorCatalog";

interface PageProps {
  params: {
    slug: string;
  };
}

export async function generateStaticParams() {
  return COMPETITOR_CATALOG.map((comp) => ({
    slug: comp.slug,
  }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const comp = COMPETITOR_CATALOG.find((c) => c.slug === params.slug);
  if (!comp) {
    return {
      title: "Comparison Not Found | ARX Terminal",
    };
  }

  return {
    title: `ARX Terminal vs ${comp.competitorName} | Feature & Pricing Comparison`,
    description: `Comprehensive comparison of ARX Terminal vs ${comp.competitorName}. Compare pricing, STOCK Act tracking, algorithmic execution corridors, and risk modeling.`,
    alternates: {
      canonical: `https://www.arxterminal.com/vs/${comp.slug}/`,
    },
    openGraph: {
      title: `ARX Terminal vs ${comp.competitorName} (${comp.competitorDomain})`,
      description: comp.summary,
      url: `https://www.arxterminal.com/vs/${comp.slug}/`,
      siteName: "ARX Terminal",
      type: "article",
    },
  };
}

export default function CompetitorComparisonPage({ params }: PageProps) {
  const comp = COMPETITOR_CATALOG.find((c) => c.slug === params.slug);
  if (!comp) {
    notFound();
  }

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    "headline": `ARX Terminal vs ${comp.competitorName} - Platform Comparison`,
    "description": comp.summary,
    "url": `https://www.arxterminal.com/vs/${comp.slug}/`,
    "mainEntity": {
      "@type": "Table",
      "about": `Comparison between ARX Terminal and ${comp.competitorName}`
    }
  };

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans antialiased">
      <Navbar />

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
        {/* Breadcrumb Navigation */}
        <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <Link href="/" className="hover:text-cyan-400 transition-colors">Terminal</Link>
          <span>/</span>
          <Link href="/vs/" className="hover:text-cyan-400 transition-colors">Comparisons</Link>
          <span>/</span>
          <span className="text-cyan-400 truncate">vs {comp.competitorName}</span>
        </nav>

        {/* Header Title & Summary */}
        <header className="space-y-4 border-b border-[#1b273d] pb-6">
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-cyan-950/60 border border-cyan-700/50 text-cyan-300">
              Platform Benchmark
            </span>
            <span className="text-xs font-mono text-slate-400">
              Target Competitor: <strong className="text-slate-200">{comp.competitorDomain}</strong>
            </span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-black tracking-tight text-white">
            ARX Terminal <span className="text-cyan-400 font-normal">vs</span> {comp.competitorName}
          </h1>

          <p className="text-base text-slate-300 leading-relaxed font-sans font-normal">
            {comp.summary}
          </p>
        </header>

        {/* E-E-A-T Quantitative Credibility Badge */}
        <AuthorEeatBadge
          topic={`Competitive Benchmark: ARX vs ${comp.competitorName}`}
          lastUpdated="September 2026"
        />

        {/* Pricing Comparison Hero Banner */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-6 rounded-2xl bg-[#091120] border border-cyan-500/40 relative overflow-hidden shadow-lg shadow-cyan-950/20">
            <div className="absolute top-3 right-3 text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-cyan-400 text-slate-950 uppercase">
              Free Institutional
            </div>
            <div className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">
              ARX Terminal
            </div>
            <div className="text-2xl font-black text-white mt-1">
              $0 <span className="text-xs font-normal text-slate-400">/ forever</span>
            </div>
            <p className="text-xs text-slate-300 mt-2 leading-relaxed">
              {comp.pricingComparison.arx}. Institutional decision suites, algorithmic screener, and downside risk models without paywalls.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-[#080d18] border border-[#1a2942]">
            <div className="text-xs font-mono text-slate-400 font-bold uppercase tracking-wider">
              {comp.competitorName}
            </div>
            <div className="text-2xl font-black text-slate-200 mt-1">
              {comp.pricingComparison.competitor.split(" ")[0]}{" "}
              <span className="text-xs font-normal text-slate-400">/ recurring</span>
            </div>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              {comp.pricingComparison.competitor}. Access tier depends on paid subscription level.
            </p>
          </div>
        </section>

        {/* Side-by-Side Architectural Advantages */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-6 rounded-2xl bg-[#090f1d] border border-[#17253d] space-y-3">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="text-cyan-400">✓</span> Where ARX Terminal Excels
            </h2>
            <ul className="space-y-2 text-xs text-slate-300 leading-relaxed">
              {comp.keyAdvantagesArx.map((adv, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-cyan-400 mt-0.5">•</span>
                  <span>{adv}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="p-6 rounded-2xl bg-[#080d18] border border-[#152136] space-y-3">
            <h2 className="text-sm font-bold text-slate-300 flex items-center gap-2">
              <span>★</span> Where {comp.competitorName} Has Focus
            </h2>
            <ul className="space-y-2 text-xs text-slate-400 leading-relaxed">
              {comp.keyAdvantagesCompetitor.map((adv, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-slate-400 mt-0.5">•</span>
                  <span>{adv}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* Detailed Feature-by-Feature Matrix */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-white tracking-wide">
            Detailed Feature Comparison Matrix
          </h2>

          <div className="border border-[#17253d] rounded-2xl overflow-hidden bg-[#070c17]">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-[#0b1424] text-slate-300 font-mono text-[11px] border-b border-[#18263d]">
                  <tr>
                    <th className="py-3 px-4 w-1/4">Functional Capability</th>
                    <th className="py-3 px-4 w-1/3 text-cyan-400">ARX Terminal</th>
                    <th className="py-3 px-4 w-1/3 text-slate-300">{comp.competitorName}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#131e33] text-slate-300">
                  {comp.featuresMatrix.map((row, idx) => (
                    <tr key={idx} className="hover:bg-[#0c1527] transition-colors">
                      <td className="py-3.5 px-4 font-semibold text-white">
                        {row.featureName}
                        <div className="text-[10px] font-normal text-slate-400 mt-0.5 font-mono">
                          {row.notes}
                        </div>
                      </td>
                      <td className="py-3.5 px-4 font-medium text-cyan-300">
                        {typeof row.arxTerminal === "boolean" ? (
                          row.arxTerminal ? "✓ Native" : "✕ Not Supported"
                        ) : (
                          row.arxTerminal
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-slate-400">
                        {typeof row.competitor === "boolean" ? (
                          row.competitor ? "✓ Native" : "✕ Not Supported"
                        ) : (
                          row.competitor
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Definitive Verdict Box */}
        <section className="p-6 rounded-2xl bg-gradient-to-br from-[#0c1424] to-[#080d1a] border border-[#223554] space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-cyan-400 text-base">🎯</span>
            <h2 className="text-base font-bold text-white">The Bottom Line: Which Should You Use?</h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-normal">
            {comp.verdict}
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Link
              href="/screener/"
              className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono font-semibold text-xs transition-all shadow-md shadow-cyan-500/20"
            >
              Launch Live ARX Screener →
            </Link>
            <Link
              href="/vs/"
              className="px-4 py-2 rounded-lg bg-[#111a2c] hover:bg-[#18263f] border border-[#223554] text-xs font-mono text-slate-300 transition-all"
            >
              View All Platform Comparisons
            </Link>
          </div>
        </section>

        {/* Back Link */}
        <div className="pt-4 flex justify-between items-center text-xs font-mono text-slate-400 border-t border-[#1b273d]">
          <Link href="/vs/" className="text-cyan-400 hover:underline flex items-center gap-1">
            <span>←</span>
            <span>All Terminal Comparisons</span>
          </Link>
          <Link href="/glossary/" className="hover:text-slate-200 transition-colors">
            Explore Quantitative Glossary →
          </Link>
        </div>
      </main>

      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }}
      />
    </div>
  );
}
