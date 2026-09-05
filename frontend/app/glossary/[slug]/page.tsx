import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import Navbar from "../../../components/Navbar";
import AuthorEeatBadge from "../../../components/AuthorEeatBadge";
import { GLOSSARY_CATALOG, GlossaryTerm } from "../../../lib/glossaryCatalog";

interface PageProps {
  params: {
    slug: string;
  };
}

export async function generateStaticParams() {
  return GLOSSARY_CATALOG.map((term) => ({
    slug: term.slug,
  }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const term = GLOSSARY_CATALOG.find((t) => t.slug === params.slug);
  if (!term) {
    return {
      title: "Term Not Found | ARX Quantitative Glossary",
    };
  }

  return {
    title: `${term.name} | Quantitative Finance Glossary`,
    description: `${term.shortDefinition} Learn mathematical formulation, practical trading application, and algorithmic implementation in ARX Terminal.`,
    alternates: {
      canonical: `https://www.arxterminal.com/glossary/${term.slug}/`,
    },
    openGraph: {
      title: `${term.name} | ARX Quantitative Glossary`,
      description: term.shortDefinition,
      url: `https://www.arxterminal.com/glossary/${term.slug}/`,
      siteName: "ARX Terminal",
      type: "article",
    },
  };
}

export default function GlossaryTermPage({ params }: PageProps) {
  const term = GLOSSARY_CATALOG.find((t) => t.slug === params.slug);
  if (!term) {
    notFound();
  }

  const related = GLOSSARY_CATALOG.filter((t) => term.relatedTerms.includes(t.slug));

  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "DefinedTerm",
      "name": term.name,
      "description": term.shortDefinition,
      "inDefinedTermSet": "https://www.arxterminal.com/glossary/",
      "url": `https://www.arxterminal.com/glossary/${term.slug}/`,
    },
    {
      "@context": "https://schema.org",
      "@type": "TechArticle",
      "headline": `${term.name} - Quantitative Definition & Mathematical Formulation`,
      "description": term.shortDefinition,
      "author": {
        "@type": "Organization",
        "name": "ARX Quantitative Research Group",
        "url": "https://www.arxterminal.com"
      },
      "publisher": {
        "@type": "Organization",
        "name": "ARX Terminal",
        "logo": {
          "@type": "ImageObject",
          "url": "https://www.arxterminal.com/icon.png"
        }
      },
      "datePublished": "2026-09-01",
      "dateModified": "2026-09-05"
    }
  ];

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans antialiased">
      <Navbar />

      <main className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        {/* Breadcrumb Navigation */}
        <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <Link href="/" className="hover:text-cyan-400 transition-colors">Terminal</Link>
          <span>/</span>
          <Link href="/glossary/" className="hover:text-cyan-400 transition-colors">Glossary</Link>
          <span>/</span>
          <span className="text-cyan-400 truncate">{term.name}</span>
        </nav>

        {/* Title & Tagging */}
        <header className="space-y-4 border-b border-[#1b273d] pb-6">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-mono bg-cyan-950/60 border border-cyan-700/50 text-cyan-300">
            <span>🏷️</span>
            <span>{term.category}</span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-black tracking-tight text-white">
            {term.name}
          </h1>

          <p className="text-base text-slate-300 leading-relaxed font-sans font-normal">
            {term.shortDefinition}
          </p>
        </header>

        {/* E-E-A-T Author Credibility Badge */}
        <AuthorEeatBadge
          topic={term.category}
          lastUpdated="September 2026"
        />

        {/* Key Takeaway / AI Overview Citability Block */}
        <div className="p-5 rounded-xl bg-cyan-950/20 border border-cyan-500/30 space-y-2">
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-cyan-300 uppercase tracking-wider">
            <span>⚡</span>
            <span>Key Takeaway for Quantitative Analysts</span>
          </div>
          <p className="text-xs sm:text-sm text-slate-200 leading-relaxed font-medium">
            {term.keyTakeaway}
          </p>
        </div>

        {/* Mathematical Formulation (if applicable) */}
        {term.latexFormula && (
          <section className="p-6 rounded-2xl bg-[#090f1d] border border-[#1b2b46] space-y-3">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <span className="text-cyan-400">∑</span> Mathematical Formulation
            </h2>
            <div className="p-4 rounded-xl bg-[#050912] border border-[#131e33] overflow-x-auto text-sm sm:text-base font-mono text-cyan-300">
              <code>{term.latexFormula}</code>
            </div>
            <p className="text-[11px] text-slate-400 font-mono">
              Formula rendered in standardized econometric syntax for automated algorithmic execution.
            </p>
          </section>
        )}

        {/* In-Depth Explanation */}
        <section className="space-y-4">
          <h2 className="text-lg font-bold text-white tracking-wide">
            Detailed Quantitative Explanation
          </h2>
          <div className="space-y-3 text-sm text-slate-300 leading-relaxed">
            {term.detailedExplanation.map((paragraph, idx) => (
              <p key={idx}>{paragraph}</p>
            ))}
          </div>
        </section>

        {/* Practical Application in ARX Terminal */}
        <section className="p-6 rounded-2xl bg-gradient-to-br from-[#0c1424] to-[#080d1a] border border-[#223554] space-y-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-mono font-bold text-xs">
              ARX
            </div>
            <h2 className="text-base font-bold text-white">
              Application in ARX Terminal Architecture
            </h2>
          </div>

          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            {term.arxApplication}
          </p>

          {term.relatedRoute && (
            <div className="pt-2">
              <Link
                href={term.relatedRoute.url}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono font-semibold text-xs transition-all shadow-md shadow-cyan-500/20"
              >
                <span>{term.relatedRoute.title}</span>
                <span>→</span>
              </Link>
            </div>
          )}
        </section>

        {/* Related Terms */}
        {related.length > 0 && (
          <section className="space-y-4 pt-4 border-t border-[#1b273d]">
            <h2 className="text-sm font-mono font-bold uppercase tracking-wider text-slate-400">
              Related Quantitative Terms
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {related.map((rel) => (
                <Link
                  key={rel.slug}
                  href={`/glossary/${rel.slug}/`}
                  className="p-4 rounded-xl bg-[#090f1d] hover:bg-[#0e172a] border border-[#162238] hover:border-cyan-500/40 transition-all text-xs group"
                >
                  <div className="font-semibold text-white group-hover:text-cyan-400 transition-colors">
                    {rel.name}
                  </div>
                  <div className="text-slate-400 line-clamp-2 mt-1">
                    {rel.shortDefinition}
                  </div>
                </Link>
              ))}
            </div>
          </section>
        )}

        {/* Back to Glossary Index */}
        <div className="pt-4 flex justify-between items-center text-xs font-mono text-slate-400">
          <Link href="/glossary/" className="text-cyan-400 hover:underline flex items-center gap-1">
            <span>←</span>
            <span>Back to Full Glossary</span>
          </Link>
          <Link href="/guide/" className="hover:text-slate-200 transition-colors">
            View Field Manual →
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
