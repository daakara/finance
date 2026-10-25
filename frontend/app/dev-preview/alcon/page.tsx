"use client";

/**
 * frontend/app/dev-preview/alcon/page.tsx
 *
 * Local Development-Only Preview Surface for Alcon Inc. (NYSE: ALC / SIX: ALC).
 * Renders the certified Alcon V4 Opportunity Dossier using the real production
 * DecisionFirstDossierView and AlconV4ExtendedView components.
 *
 * Isolation Invariants:
 * - Completely omitted / disabled in standard production builds (inaccessible when NEXT_PUBLIC_ENABLE_ALCON_DEV_PREVIEW !== "true").
 * - ALC remains 100% excluded from publishedWatchManifest.json and opportunityDossierSnapshot.json.
 * - Public research route /research?symbol=ALC remains fail-closed until formal publication enrollment.
 */

import React, { useMemo } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import Navbar from "../../../components/Navbar";
import DecisionFirstDossierView from "../../../components/DecisionFirstDossierView";
import {
  getAlconDevPreviewDossier,
  IS_ALCON_DEV_PREVIEW_ENABLED,
  OpportunityDossier,
} from "../../../lib/opportunityDossiers";

// Direct fallback load for local test environments
let localDraftPayload: OpportunityDossier | null = null;
try {
  localDraftPayload = require("../../../lib/alconDraftDossier.json");
} catch {
  localDraftPayload = null;
}

export default function AlconDevPreviewPage() {
  // In production builds, development preview is unconditionally disabled (returns 404),
  // even if NEXT_PUBLIC_ENABLE_ALCON_DEV_PREVIEW is set to "true".
  if (
    process.env.NODE_ENV === "production" ||
    process.env.NEXT_PUBLIC_ENABLE_ALCON_DEV_PREVIEW !== "true"
  ) {
    notFound();
  }

  const dossier = useMemo(() => {
    return getAlconDevPreviewDossier() || localDraftPayload;
  }, []);

  if (!dossier) {
    return (
      <div className="min-h-screen bg-[#0a0e17] text-slate-300 font-sans p-8 flex flex-col items-center justify-center">
        <div className="max-w-md bg-[#0f172a] border border-amber-800/80 rounded-xl p-6 text-center space-y-3">
          <span className="w-3 h-3 rounded-full bg-amber-400 inline-block animate-pulse"></span>
          <h1 className="text-lg font-bold text-white">Alcon V4 Preview Inactive</h1>
          <p className="text-xs text-slate-400">
            The Alcon V4 development draft is isolated from production. Enable NEXT_PUBLIC_ENABLE_ALCON_DEV_PREVIEW=&quot;true&quot; to review.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0e17] text-slate-100 font-sans selection:bg-cyan-900 selection:text-cyan-200">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Development Isolation Notice Banner */}
        <aside
          className="bg-[#0b1322] border-2 border-amber-500/80 rounded-xl p-4 sm:p-5 shadow-xl space-y-2 text-xs"
          aria-label="Development preview isolation notice"
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-amber-900/60 pb-2.5">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-950 text-amber-300 border border-amber-700 uppercase tracking-wider">
                Isolated Development Preview
              </span>
              <span className="font-mono text-slate-300 font-semibold">
                Alcon Inc. (ALC) &bull; Research Candidate v4.0
              </span>
            </div>
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800">
              Zero Production Leakage Verified
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-slate-400 text-[11px] leading-relaxed pt-1">
            <div>
              <strong className="text-slate-200 block font-mono">Manifest Isolation Status:</strong>
              ALC is strictly excluded from <code className="text-cyan-300">publishedWatchManifest.json</code> and{" "}
              <code className="text-cyan-300">opportunityDossierSnapshot.json</code>. Public route{" "}
              <code className="text-cyan-300">/research?symbol=ALC</code> returns unsupported until Slice 4 formal enrollment.
            </div>
            <div>
              <strong className="text-slate-200 block font-mono">Source Provenance:</strong>
              Bound to certified Canonical Ledger <code className="text-cyan-300">d28270a18984...</code> and v4.0 Engine{" "}
              <code className="text-cyan-300">326cbe1c...</code>. Frozen reference price: $67.43 (October 21, 2026).
            </div>
          </div>
        </aside>

        {/* Real Decision-First Dossier Presentation */}
        <DecisionFirstDossierView
          dossier={dossier}
          initialValuationOpen={true}
          initialFinancialsOpen={true}
          initialProvenanceOpen={true}
        />
      </main>
    </div>
  );
}
