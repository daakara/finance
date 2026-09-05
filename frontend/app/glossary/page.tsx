import type { Metadata } from "next";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import AuthorEeatBadge from "../../components/AuthorEeatBadge";
import { GLOSSARY_CATALOG } from "../../lib/glossaryCatalog";

export const metadata: Metadata = {
  title: "Quantitative Finance Glossary & Econometric Dictionary",
  description:
    "Authoritative reference guide to institutional quantitative trading terminology: Autoregressive Exogenous (ARX) models, Mark Minervini VCP patterns, Cornish-Fisher Modified VaR, STOCK Act forensics, and Amihud illiquidity.",
  alternates: {
    canonical: "https://www.arxterminal.com/glossary/",
  },
  openGraph: {
    title: "Quantitative Finance & Econometrics Glossary | ARX Terminal",
    description:
      "Master institutional trading vocabulary: ARX econometric modeling, Minervini VCP setups, Cornish-Fisher downside risk, and Congressional STOCK Act tracking.",
    url: "https://www.arxterminal.com/glossary/",
    siteName: "ARX Terminal",
    type: "website",
  },
};

export default function GlossaryIndexPage() {
  const categories = [
    "Econometric & Mathematical Modeling",
    "Algorithmic Setups & Execution",
    "Statutory & Smart Money Forensics",
  ] as const;

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "DefinedTermSet",
    "name": "ARX Terminal Quantitative Finance & Econometric Dictionary",
    "description": "Comprehensive reference glossary for institutional algorithmic models, statutory insider tracking, and market microstructure metrics.",
    "url": "https://www.arxterminal.com/glossary/",
    "hasDefinedTerm": GLOSSARY_CATALOG.map((term) => ({
      "@type": "DefinedTerm",
      "name": term.name,
      "description": term.shortDefinition,
      "url": `https://www.arxterminal.com/glossary/${term.slug}/`,
      "inDefinedTermSet": "https://www.arxterminal.com/glossary/"
    }))
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
            <span className="text-cyan-400">Quantitative Glossary</span>
          </div>

          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1b273d] pb-6">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-3">
                <span className="text-cyan-400 font-mono">📚</span>
                <span>Quantitative Finance & Econometrics Dictionary</span>
              </h1>
              <p className="mt-2 text-sm text-slate-400 max-w-3xl leading-relaxed">
                Institutional methodology guide defining the mathematical frameworks, statistical risk metrics, and statutory insider tracking models powering ARX Terminal.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/guide/"
                className="px-3.5 py-2 rounded-lg bg-[#111a2c] hover:bg-[#18263f] border border-[#223554] text-xs font-mono text-cyan-300 transition-all shadow-sm"
              >
                Field Manual 📖
              </Link>
              <Link
                href="/vs/"
                className="px-3.5 py-2 rounded-lg bg-[#111a2c] hover:bg-[#18263f] border border-[#223554] text-xs font-mono text-slate-200 transition-all shadow-sm"
              >
                Compare Terminals ⚔️
              </Link>
            </div>
          </div>
        </div>

        {/* E-E-A-T Quantitative Credibility Badge */}
        <AuthorEeatBadge
          topic="Mathematical Modeling, Econometrics & Execution Geometry"
          lastUpdated="September 2026"
        />

        {/* Categories & Term Cards */}
        <div className="space-y-10">
          {categories.map((cat) => {
            const termsInCat = GLOSSARY_CATALOG.filter((t) => t.category === cat);
            return (
              <section key={cat} className="space-y-4">
                <div className="flex items-center gap-2.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400/50" />
                  <h2 className="text-lg font-bold text-white tracking-wide">
                    {cat}
                  </h2>
                  <span className="text-xs font-mono text-slate-400 bg-[#0c1424] px-2 py-0.5 rounded border border-[#1b2b48]">
                    {termsInCat.length} Terms
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {termsInCat.map((term) => (
                    <Link
                      key={term.slug}
                      href={`/glossary/${term.slug}/`}
                      className="group flex flex-col justify-between p-5 rounded-xl bg-[#090f1d] hover:bg-[#0e172a] border border-[#162238] hover:border-cyan-500/50 transition-all duration-200 shadow-sm"
                    >
                      <div className="space-y-2.5">
                        <div className="flex items-start justify-between gap-2">
                          <h3 className="font-semibold text-white text-sm group-hover:text-cyan-400 transition-colors">
                            {term.name}
                          </h3>
                          <span className="text-xs font-mono text-slate-600 group-hover:text-cyan-400 transition-colors">
                            →
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 leading-relaxed line-clamp-3">
                          {term.shortDefinition}
                        </p>
                      </div>

                      {term.latexFormula && (
                        <div className="mt-4 pt-3 border-t border-[#141f33] font-mono text-[11px] text-cyan-300/80 truncate">
                          <code>{term.latexFormula.split("\\quad")[0]}</code>
                        </div>
                      )}
                    </Link>
                  ))}
                </div>
              </section>
            );
          })}
        </div>

        {/* SEO Citability Summary Box */}
        <div className="p-6 rounded-2xl bg-[#070b14] border border-[#17243c] space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span className="text-cyan-400">💡</span> Why Quantitative Terminology Matters
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Unlike retail indicators that rely on subjective visual chart overlays, institutional quantitative finance relies on mathematically auditable definitions. ARX Terminal enforces strict epistemic standards: every factor score, execution corridor, and risk parameter is grounded in peer-reviewed econometric literature (from Box-Jenkins time series to Amihud microstructure).
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
