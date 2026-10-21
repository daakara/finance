"use client";

/**
 * frontend/app/research/page.tsx
 *
 * ARX Read-Only Opportunity Dossier surface (Alternative A - Integrated Assessment).
 * Renders the 5-question central investment assessment grounded in the accepted ODP-05 research candidate.
 *
 * Key Behaviors:
 * - Single bundled data path via frontend/lib/opportunityDossiers.ts (statically imported).
 * - Suspense boundary wrapping useSearchParams() for static-export and hydration safety.
 * - Responsive 2-column layout on desktop, responsive tab-rail on mobile.
 * - Explicit fail-closed handling for unsupported/uncataloged symbols.
 * - Identity-preserving deep-links to Live Terminal (/), Radar (/radar), and Fundamentals (/stock/[ticker]).
 */

import React, { Suspense, useMemo } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import Navbar from "../../components/Navbar";
import OpportunityDossierCard from "../../components/OpportunityDossierCard";
import {
  getOpportunityDossier,
  listOpportunityCohort,
  listPilotDraftCohort,
  normalizeSymbol,
  isCohortSymbol,
  isPilotDraftSymbol,
  getOpportunitySnapshotMetadata,
  getPilotDraftSnapshotMetadata,
  getDossierProvenance,
  getCollectionProvenance,
  OpportunityCohortItem,
  IS_PILOT_DRAFTS_BUILD_ENABLED,
} from "../../lib/opportunityDossiers";
import {
  Microscope,
  ArrowRight,
  AlertCircle,
  ShieldCheck,
  Calendar,
  Layers,
  Search,
  ExternalLink,
  ChevronRight,
  Info,
} from "lucide-react";

function DispositionPill({ disposition }: { disposition: string }) {
  switch (disposition) {
    case "WATCH_CONDITIONS":
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-950/80 text-amber-300 border border-amber-800">
          WATCH
        </span>
      );
    case "NO_SUPPORTED_OPPORTUNITY":
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-950/80 text-rose-300 border border-rose-800">
          REJECT
        </span>
      );
    default:
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-900 text-slate-400 border border-slate-700">
          {disposition}
        </span>
      );
  }
}

function ResearchGovernanceBox({
  dossierProvenance,
  collectionProvenance,
  normalizedSymbol,
}: {
  dossierProvenance: ReturnType<typeof getDossierProvenance>;
  collectionProvenance: ReturnType<typeof getCollectionProvenance>;
  normalizedSymbol: string;
}) {
  return (
    <div className="bg-[#090d14] rounded-2xl border border-[#172030] p-4 space-y-2 text-xs font-mono text-slate-400">
      <div className="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
        <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
        {dossierProvenance ? `${normalizedSymbol} Provenance` : "Collection Governance"}
      </div>
      <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
        {dossierProvenance
          ? `Primary filings authenticated via SEC EDGAR accessions. Governed by ${dossierProvenance.governingContract}.`
          : collectionProvenance.disclaimer}
      </p>
      {dossierProvenance ? (
        <div className="pt-2 border-t border-[#151e2e] space-y-1 text-[10px] text-slate-500">
          <div>Authority: <span className="text-slate-300">{dossierProvenance.sourceAuthority}</span></div>
          <div>Status: <span className="text-cyan-400">{dossierProvenance.lifecycleStatusLabel}</span></div>
          <div>Review: <span className="text-slate-300">{dossierProvenance.reviewStatusLabel}</span></div>
          <div>Authored UTC: <span className="text-slate-300">{dossierProvenance.authoringTimestampUtc}</span></div>
          <div>Filing Cutoff: <span className="text-slate-300">{dossierProvenance.filingCutoffUtc}</span></div>
          <div>{dossierProvenance.digestLabel}: <span className="text-slate-300 font-mono">{dossierProvenance.contentDigest}</span></div>
        </div>
      ) : (
        <div className="pt-2 border-t border-[#151e2e] space-y-1 text-[10px] text-slate-500">
          <div>Standard: <span className="text-slate-300">{collectionProvenance.governingStandard}</span></div>
          <div>Collection: <span className="text-slate-300">{collectionProvenance.publishedCount} Published Assets</span></div>
          <div>Market Date: <span className="text-slate-300">{collectionProvenance.marketSessionDate} Close</span></div>
        </div>
      )}
    </div>
  );
}

function ResearchContent() {
  const searchParams = useSearchParams();
  const rawSymbol = searchParams.get("symbol") || searchParams.get("ticker");
  const experienceMode = searchParams.get("mode");
  const previewParam = searchParams.get("preview");
  const isDraftPreviewActive =
    IS_PILOT_DRAFTS_BUILD_ENABLED &&
    (previewParam === "pilot" || searchParams.get("draft") === "true");
  const displaySymbol = (rawSymbol || "").trim().toUpperCase();
  const hasSymbolParam = displaySymbol.length > 0;
  const normalizedSymbol = normalizeSymbol(displaySymbol || "FIX");

  const cohort = useMemo(
    () => (isDraftPreviewActive ? listPilotDraftCohort() : listOpportunityCohort()),
    [isDraftPreviewActive]
  );
  const metadata = useMemo(
    () => (isDraftPreviewActive ? getPilotDraftSnapshotMetadata() : getOpportunitySnapshotMetadata()),
    [isDraftPreviewActive]
  );
  const isKnown = isDraftPreviewActive ? isPilotDraftSymbol(normalizedSymbol) : isCohortSymbol(normalizedSymbol);
  const dossier = isKnown ? getOpportunityDossier(normalizedSymbol, { previewDrafts: isDraftPreviewActive }) : null;

  const dossierProvenance = useMemo(() => {
    if (isKnown && dossier) {
      return getDossierProvenance(normalizedSymbol, { previewDrafts: isDraftPreviewActive });
    }
    return null;
  }, [isKnown, dossier, normalizedSymbol, isDraftPreviewActive]);

  const collectionProvenance = useMemo(() => {
    return getCollectionProvenance(hasSymbolParam && !isKnown ? displaySymbol : undefined);
  }, [hasSymbolParam, isKnown, displaySymbol]);

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 pb-24">
      {/* Breadcrumb Navigation */}
      <nav aria-label="Breadcrumb" className="text-xs font-mono text-slate-500 flex items-center space-x-2">
        <Link href="/" className="hover:text-cyan-400 transition-colors">
          Terminal
        </Link>
        <span>/</span>
        <span className="text-slate-400">Research</span>
        <span>/</span>
        <span className="text-cyan-400 font-bold">Opportunity Dossier</span>
        {isKnown && (
          <>
            <span>/</span>
            <span className="text-white font-bold">{normalizedSymbol}</span>
          </>
        )}
      </nav>

      {/* Surface Header Banner */}
      <header className="bg-gradient-to-r from-[#0b1019] via-[#0e1624] to-[#0b1019] p-5 sm:p-6 rounded-2xl border border-[#1e293b] space-y-4 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-xs font-mono font-bold">
                <Microscope className="w-3.5 h-3.5" />
                ARX QUALITY WATCH RESEARCH
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-900 text-slate-300 border border-slate-700 text-xs font-mono">
                <Calendar className="w-3 h-3 text-cyan-400" />
                Reference Date: {metadata.market_session_date || "2026-10-16"} Close
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-900 text-slate-300 border border-slate-700 text-xs font-mono">
                <ShieldCheck className="w-3 h-3 text-emerald-400" />
                {isDraftPreviewActive ? "Scaling Pilot Draft Evidence" : "Curated Quality Watch Selection"}
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight font-sans">
              Curated Quality Watch Research & Invalidation Dossiers
            </h1>
            <p className="text-xs sm:text-sm text-slate-300 font-sans max-w-3xl leading-relaxed">
              Why does this business deserve continuous monitoring at the stated reference price?
              Examines primary SEC filings, business quality moats, reverse-DCF valuation conditions, balance sheet resilience, and observable invalidation triggers.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row lg:flex-col gap-2 shrink-0">
            <div className="p-3 rounded-xl bg-[#080c13] border border-[#1f293d] space-y-1 text-right">
              <div className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Research Collection</div>
              <div className="text-sm font-mono font-bold text-cyan-300">{cohort.length} Quality Watch Assets</div>
              <div className="text-[11px] font-mono text-slate-400">
                {isDraftPreviewActive ? "Pilot Draft Staging" : "Curated Published Collection"}
              </div>
              {IS_PILOT_DRAFTS_BUILD_ENABLED && !isDraftPreviewActive ? (
                <Link
                  href={`/research?symbol=${normalizedSymbol}&preview=pilot${experienceMode ? `&mode=${experienceMode}` : ""}`}
                  className="inline-block mt-1 text-[10px] font-mono text-amber-400 hover:text-amber-300 underline"
                >
                  Preview Draft Expansion ({cohort.length > 5 ? cohort.length : 18} Assets)
                </Link>
              ) : null}
            </div>
          </div>
        </div>

        {/* Global Qualification Note */}
        <div className="pt-3 border-t border-[#1a2538] flex items-center justify-between gap-3 text-[11px] font-mono text-slate-400">
          <div className="flex items-center gap-2">
            <Info className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <span>
              Read-only research artifact. Model reference prices represent valuation anchors, not price targets.
            </span>
          </div>
          <Link
            href="/radar"
            className="hidden sm:inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 transition-colors shrink-0"
          >
            <span>Launch Live Radar</span>
            <ExternalLink className="w-3 h-3" />
          </Link>
        </div>
      </header>

      {/* Draft Staging Preview Mode Banner */}
      {isDraftPreviewActive && (
        <div className="p-4 rounded-2xl bg-amber-950/40 border border-amber-600/70 text-amber-200 text-xs font-mono flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-lg">
          <div className="flex items-start sm:items-center gap-2.5">
            <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5 sm:mt-0" />
            <div className="space-y-0.5">
              <div className="font-bold text-amber-300 uppercase tracking-wider flex items-center gap-2">
                <span>DRAFT STAGING PREVIEW — 20-COMPANY SCALING PILOT</span>
                <span className="px-1.5 py-0.5 rounded bg-amber-900 text-[10px] text-amber-200 border border-amber-700">
                  PEER REVIEW ONLY
                </span>
              </div>
              <p className="text-[11px] text-amber-200/80 font-sans">
                Reviewing draft dossiers generated for the Scaling Pilot ({cohort.length} assets). These records are strictly isolated from production publication and pending independent research acceptance.
              </p>
            </div>
          </div>
          <Link
            href={`/research?symbol=${isCohortSymbol(normalizedSymbol) ? normalizedSymbol : "FIX"}${experienceMode ? `&mode=${experienceMode}` : ""}`}
            className="shrink-0 px-3 py-1.5 rounded-lg bg-amber-900/80 hover:bg-amber-800 text-amber-200 font-bold border border-amber-700 text-center transition-colors"
          >
            Exit Draft Preview
          </Link>
        </div>
      )}

      {/* Mobile Cohort Selector Rail (Horizontal Scrollable) */}
      <div className="lg:hidden space-y-2">
        <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider px-1">
          Select Candidate Asset:
        </div>
        <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin">
          {cohort.map((item) => {
            const active = normalizedSymbol === item.symbol;
            return (
              <Link
                key={item.symbol}
                href={`/research?symbol=${item.symbol}${isDraftPreviewActive ? "&preview=pilot" : ""}${experienceMode ? `&mode=${experienceMode}` : ""}`}
                scroll={false}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-xs font-mono shrink-0 transition-all min-h-[44px] ${
                  active
                    ? "bg-[#162030] border-cyan-500 text-white shadow-lg shadow-cyan-950/50"
                    : "bg-[#0b1019] border-[#1e293b] text-slate-400 hover:text-slate-200 hover:bg-[#111722]"
                }`}
              >
                <div className="flex flex-col text-left">
                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-cyan-400">{item.symbol}</span>
                    <DispositionPill disposition={item.disposition} />
                  </div>
                  <span className="text-[10px] text-slate-400">${item.reference_price.toFixed(2)}</span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Desktop Left Rail: Cohort Selector */}
        <aside className="hidden lg:block lg:col-span-3 space-y-4 sticky top-6">
          <div className="bg-[#0b1019] rounded-2xl border border-[#1e293b] p-3.5 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-[#1b2537]">
              <span className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                {isDraftPreviewActive ? "Draft Expansion" : "Quality Watch"} ({cohort.length})
              </span>
              <span className="text-[10px] font-mono text-slate-500">2026-10-16</span>
            </div>

            <div className="space-y-1.5">
              {cohort.map((item) => {
                const active = isKnown && normalizedSymbol === item.symbol;
                return (
                  <Link
                    key={item.symbol}
                    href={`/research?symbol=${item.symbol}${isDraftPreviewActive ? "&preview=pilot" : ""}${experienceMode ? `&mode=${experienceMode}` : ""}`}
                    scroll={false}
                    className={`w-full flex items-center justify-between p-2.5 rounded-xl border transition-all text-left min-h-[44px] group focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                      active
                        ? "bg-[#162030] border-cyan-500 shadow-md shadow-cyan-950/40"
                        : "bg-[#0a0f18] border-[#182234] hover:bg-[#121927] hover:border-[#24334f]"
                    }`}
                  >
                    <div className="space-y-0.5 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`font-mono text-sm font-extrabold ${active ? "text-cyan-300" : "text-white"}`}>
                          {item.symbol}
                        </span>
                        <DispositionPill disposition={item.disposition} />
                      </div>
                      <div className="text-[11px] text-slate-400 truncate max-w-[150px]">
                        {item.company_name}
                      </div>
                    </div>

                    <div className="text-right shrink-0">
                      <div className="font-mono text-xs font-bold text-slate-200">
                        ${item.reference_price.toFixed(2)}
                      </div>
                      <div className="text-[10px] font-mono text-slate-500">Ref Close</div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>

          {/* Research Cohort Governance & Provenance */}
          <ResearchGovernanceBox
            dossierProvenance={dossierProvenance}
            collectionProvenance={collectionProvenance}
            normalizedSymbol={normalizedSymbol}
          />
        </aside>

        {/* Right Main Column */}
        <section className="lg:col-span-9 space-y-6">
          {isKnown && dossier ? (
            <OpportunityDossierCard dossier={dossier} />
          ) : isKnown && !dossier ? (
            /* Explicit Unavailable State for Recognized Cohort Member */
            <div className="bg-[#0b1019] rounded-2xl border border-rose-900/50 p-6 sm:p-8 space-y-4 shadow-2xl">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-400 shrink-0">
                  <AlertCircle className="w-6 h-6" />
                </div>
                <div className="space-y-2">
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-rose-950/80 text-rose-300 border border-rose-800 text-xs font-mono font-bold">
                    DOSSIER RECORD UNAVAILABLE
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold text-white font-sans">
                    Dossier Unavailable for &ldquo;{normalizedSymbol}&rdquo;
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
                    Although {normalizedSymbol} is a recognized ODP-05 research candidate asset, its detailed assessment record could not be loaded from the bundled snapshot.
                  </p>
                </div>
              </div>
            </div>
          ) : hasSymbolParam && !isKnown ? (
            /* Explicit Neutral Unsupported State */
            <div className="bg-[#0b1019] rounded-2xl border border-amber-900/50 p-6 sm:p-8 space-y-6 shadow-2xl">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-amber-950/60 border border-amber-800 text-amber-400 shrink-0">
                  <AlertCircle className="w-6 h-6" />
                </div>
                <div className="space-y-2">
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-800 text-xs font-mono font-bold">
                    NO PUBLISHED WATCH DOSSIER
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold text-white font-sans">
                    No Published Watch Dossier for &ldquo;{displaySymbol}&rdquo;
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
                    ARX Research focuses strictly on a curated collection of high-conviction Quality Watch companies meeting explicit business resilience, observable entry triggers, and valuation discipline standards. &ldquo;{displaySymbol}&rdquo; does not currently have an active published watch dossier.
                  </p>
                </div>
              </div>

              {/* Quick Jump to Valid Cohort Assets */}
              <div className="bg-[#090d14] rounded-xl border border-[#1b2537] p-4 space-y-3">
                <div className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider">
                  Available Published Quality Watch Dossiers:
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                  {cohort.map((item) => (
                    <Link
                      key={item.symbol}
                      href={`/research?symbol=${item.symbol}${isDraftPreviewActive ? "&preview=pilot" : ""}${experienceMode ? `&mode=${experienceMode}` : ""}`}
                      className="p-3 rounded-lg bg-[#0e1624] hover:bg-[#162236] border border-[#1f2c42] hover:border-cyan-500/60 transition-colors flex items-center justify-between text-xs font-mono group"
                    >
                      <div className="flex flex-col">
                        <span className="font-bold text-cyan-400 group-hover:text-cyan-300">
                          {item.symbol}
                        </span>
                        <span className="text-[10px] text-slate-400 truncate max-w-[120px]">
                          {item.company_name}
                        </span>
                      </div>
                      <div className="text-right">
                        <DispositionPill disposition={item.disposition} />
                      </div>
                    </Link>
                  ))}
                </div>
              </div>

              {/* Cross-Surface Deep Links for the requested symbol */}
              <div className="pt-4 border-t border-[#1b2537] space-y-3">
                <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
                  Explore Live Terminal Surfaces for {displaySymbol}:
                </div>
                <div className="flex flex-wrap gap-2 text-xs font-mono">
                  <Link
                    href={`/stock/${encodeURIComponent(displaySymbol.toLowerCase())}`}
                    className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold transition-transform active:scale-95 inline-flex items-center gap-1.5"
                  >
                    <span>View Fundamentals Page ({displaySymbol})</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </Link>
                  <Link
                    href={`/?symbol=${encodeURIComponent(displaySymbol)}`}
                    className="px-4 py-2 rounded-xl bg-[#111722] hover:bg-[#182232] text-slate-300 border border-[#243044] transition-colors inline-flex items-center gap-1.5"
                  >
                    <span>Launch Terminal for {displaySymbol}</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                  <Link
                    href={`/radar?symbol=${encodeURIComponent(displaySymbol)}`}
                    className="px-4 py-2 rounded-xl bg-[#111722] hover:bg-[#182232] text-cyan-300 border border-[#243044] transition-colors inline-flex items-center gap-1.5"
                  >
                    <span>Inspect Radar for {displaySymbol}</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </Link>
                </div>
              </div>
            </div>
          ) : (
            /* Fallback or Default View */
            <div className="p-8 text-center text-slate-400 font-mono">
              Loading dossier…
            </div>
          )}
          {/* Mobile Provenance & Collection Governance Section */}
          <div className="lg:hidden pt-2">
            <ResearchGovernanceBox
              dossierProvenance={dossierProvenance}
              collectionProvenance={collectionProvenance}
              normalizedSymbol={normalizedSymbol}
            />
          </div>
        </section>
      </div>
    </main>
  );
}

function ResearchLoadingSkeleton() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div className="h-4 w-48 bg-slate-800 rounded animate-pulse" />
      <div className="h-32 bg-[#0b1019] border border-[#1e293b] rounded-2xl animate-pulse" />
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="hidden lg:block lg:col-span-3 h-80 bg-[#0b1019] border border-[#1e293b] rounded-2xl animate-pulse" />
        <div className="lg:col-span-9 h-[600px] bg-[#0b1019] border border-[#1e293b] rounded-2xl animate-pulse" />
      </div>
    </div>
  );
}

export default function ResearchPage() {
  return (
    <div className="min-h-screen bg-[#070a0f] text-slate-100 font-sans selection:bg-cyan-500 selection:text-black">
      <Navbar />
      <Suspense fallback={<ResearchLoadingSkeleton />}>
        <ResearchContent />
      </Suspense>
    </div>
  );
}
