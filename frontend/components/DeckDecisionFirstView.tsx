"use client";

/**
 * frontend/components/DeckDecisionFirstView.tsx
 *
 * Decision-First Research Page Prototype for DECK (Deckers Outdoor Corp.).
 * Designed under ARX Closure Model v2 to enable an investor to understand the
 * WATCH decision within 60–90 seconds.
 *
 * Key Architecture:
 * - Decision Surface (Immediate 60-90s understanding):
 *   A. Verdict & Decisive Reason: Lead with WATCH, clear reason for monitoring, reference price.
 *   B. One-Sentence Thesis: Core investment hypothesis connecting brand durability to valuation.
 *   C. What Is Priced In?: Model-implied expectations (-2.34% CAGR), Bear / Base / Bull comparisons.
 *   D. What Must Be True?: Up to 3 evidence-backed drivers (DTC velocity, gross margin defense, net cash).
 *   E. What Breaks The Thesis?: Up to 3 failure conditions (margin breach <50%, DTC deceleration <10%, wholesale slowing).
 *   F. Potential Expectation Gap: Model implication vs hypothesis vs reality (explicitly notes no consensus gap established).
 *   G. What Would Change The Decision?: Strengthening conditions, weakening conditions, next review trigger, decisive unknown.
 *   H. Optional Section ("What could this become?"): Omitted per standard (unsupported by current dossier).
 * - Progressive Disclosure (Collapsible supporting details below decision surface):
 *   1. Valuation Mechanics & Scenario Assumptions (Reverse-DCF solve vs Bear / Base / Bull full comparative table).
 *   2. Financial Resilience, Sourced Facts & Net Debt Bridge (Cash $1.60B, zero debt, ASC 842 lease convention, SEC XBRL).
 *   3. Full Dossier Baseline & Research Provenance (Quality Watch Selection Record A-F, governance metadata).
 * - Footer: On-demand Technical Radar navigation (/radar?symbol=DECK).
 *
 * Epistemic & Behavioral Invariants:
 * - Isolated exclusively to DECK; other published dossiers retain accepted layout.
 * - Pure read-only presentation grounded in the frozen 2026-10-16 market close ledger ($85.72 reference price).
 * - Zero BUY / INVEST recommendations; preserves strict WATCH status.
 * - Zero invented consensus estimates, price targets, shareholder returns, or synthetic scores.
 * - Full accessibility: min 44px touch targets, visible focus rings, zero horizontal overflow across 360px-1280px.
 */

import React, { useState } from "react";
import Link from "next/link";
import {
  OpportunityDossier,
  calculateScenarioPriceDiffPct,
} from "../lib/opportunityDossiers";

interface DeckDecisionFirstViewProps {
  dossier: OpportunityDossier;
}

export default function DeckDecisionFirstView({ dossier }: DeckDecisionFirstViewProps) {
  const { identity, central_assessment, valuation, financial_profile, monitoring, navigation } = dossier;
  const rationale = dossier.selection_rationale;

  // Track accordion disclosure states for progressive disclosure sections
  const [valuationOpen, setValuationOpen] = useState(false);
  const [financialsOpen, setFinancialsOpen] = useState(false);
  const [provenanceOpen, setProvenanceOpen] = useState(false);

  // Authoritative scenarios and differences
  const conservativeScenario = valuation.scenarios?.conservative;
  const baseScenario = valuation.scenarios?.base;
  const optimisticScenario = valuation.scenarios?.optimistic;

  const bearDiff = calculateScenarioPriceDiffPct(
    conservativeScenario?.per_share_value,
    valuation.reference_price
  );
  const baseDiff = calculateScenarioPriceDiffPct(
    baseScenario?.per_share_value,
    valuation.reference_price
  );
  const bullDiff = calculateScenarioPriceDiffPct(
    optimisticScenario?.per_share_value,
    valuation.reference_price
  );

  return (
    <article
      className="bg-[#0b1019] border border-[#1e293b] rounded-2xl p-4 sm:p-6 shadow-xl space-y-6 text-slate-100 transition-all"
      aria-labelledby={`decision-first-heading-${identity.symbol}`}
    >
      {/* =========================================================================
          1. Header & Identity Context
         ========================================================================= */}
      <header className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-[#1e293b]">
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            <h1
              id={`decision-first-heading-${identity.symbol}`}
              className="text-2xl sm:text-3xl font-bold font-mono text-white tracking-wide"
            >
              {identity.symbol}
            </h1>
            <span className="text-base sm:text-lg text-slate-200 font-medium">
              {identity.company_name}
            </span>
            <span className="text-xs text-slate-500 font-mono">
              ({identity.exchange} • CIK {identity.cik})
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2 pt-0.5">
            <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-medium bg-cyan-950 text-cyan-300 border border-cyan-800">
              {identity.opportunity_route}
            </span>
            <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-medium bg-[#141c2e] text-amber-300 border border-amber-900/60">
              Decision-First Research Prototype (DECK)
            </span>
            {identity.sec_edgar_browse_url && (
              <a
                href={identity.sec_edgar_browse_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 hover:underline inline-flex items-center gap-1 focus:outline-none focus:ring-2 focus:ring-cyan-400 rounded px-2 min-h-[44px]"
                title={`Open SEC EDGAR Browse profile for CIK ${identity.cik}`}
              >
                <span>SEC EDGAR Profile</span>
                <span aria-hidden="true">↗</span>
              </a>
            )}
          </div>
        </div>

        {/* Reference Price & WATCH Disposition */}
        <div className="text-right flex flex-col items-end gap-1.5">
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl sm:text-3xl font-mono font-bold text-white tracking-tight">
              ${valuation.reference_price.toFixed(2)}
            </span>
            <span className="text-xs text-slate-400 font-mono">USD</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono">
            Reference Close: {valuation.reference_date}
          </div>
          <span className="px-3 py-1 rounded text-xs font-mono font-bold bg-amber-950/90 text-amber-300 border border-amber-600/80 shadow-sm inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
            WATCH
          </span>
        </div>
      </header>

      {/* =========================================================================
          DECISION SURFACE (Fast 60-90s Understanding)
         ========================================================================= */}

      {/* A. Verdict & Decisive Reason */}
      <section
        className="bg-[#0e172a] border border-amber-700/60 rounded-xl p-4 sm:p-5 space-y-3 shadow-md"
        aria-label="Verdict and Decisive Reason"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-amber-900/40 pb-2">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400" aria-hidden="true"></span>
            <h2 className="text-xs font-bold uppercase tracking-wider text-amber-300">
              A. Verdict &amp; Decisive Reason
            </h2>
          </div>
          <span className="text-[10px] font-mono text-amber-400/90 bg-amber-950/70 border border-amber-800 px-2 py-0.5 rounded">
            Primary Research Disposition
          </span>
        </div>

        <div className="space-y-2">
          <p className="text-sm text-slate-100 leading-relaxed font-sans font-medium">
            Deckers Outdoor displays exceptional balance sheet strength ($1.60B net cash, zero funded senior debt) and continued brand momentum in HOKA direct-to-consumer (+17.3% YoY in Q1 FY27), but high seasonal concentration in UGG winter demand and wholesale channel growth moderation (HOKA wholesale slowed to +2.9% YoY vs +27.9% in FY26) prevent a stronger conclusion. Retained on WATCH for evidence of international DTC acceleration and sustained gross margin defense.
          </p>
          <div className="pt-2 border-t border-amber-950/60 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-amber-300/90">
            <span>Status: WATCH (Conditions required for entry • Not a BUY recommendation)</span>
            <span className="text-slate-400 font-sans text-[10px]">
              Grounded in Form 10-K FY26 &amp; Form 10-Q Q1 FY27 filings
            </span>
          </div>
        </div>
      </section>

      {/* B. One-Sentence Thesis */}
      <section
        className="bg-[#080d16] border border-cyan-900/60 rounded-xl p-4 sm:p-5 space-y-2.5 shadow-sm"
        aria-label="One-Sentence Thesis"
      >
        <div className="flex items-center gap-2 border-b border-cyan-950/80 pb-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400" aria-hidden="true"></span>
          <h2 className="text-xs font-bold uppercase tracking-wider text-cyan-300">
            B. One-Sentence Thesis
          </h2>
        </div>
        <div className="p-3 bg-[#0d1422] rounded-lg border border-cyan-950">
          <p className="text-sm sm:text-base text-cyan-100 font-medium italic leading-relaxed">
            &ldquo;Can HOKA and UGG sustain direct-to-consumer momentum and operating margins above 20% without aggressive promotional discounting as consumer footwear spending normalizes?&rdquo;
          </p>
        </div>
        <p className="text-[11px] text-slate-400 font-sans">
          Connects the durability of brand pricing power directly to the reverse-DCF valuation boundary.
        </p>
      </section>

      {/* C. What Is Priced In? (Model Expectations & Scenarios) */}
      <section
        className="bg-[#0f172a]/95 border border-cyan-800/60 rounded-xl p-4 sm:p-5 space-y-4 shadow-md"
        aria-label="What Is Priced In"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" aria-hidden="true"></span>
            <h2 className="text-xs font-bold uppercase tracking-wider text-cyan-300">
              C. What Is Priced In? (Model Expectations &amp; Scenario Values)
            </h2>
          </div>
          <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/80 border border-cyan-800 px-2 py-0.5 rounded">
            Reverse-DCF &amp; DCF Present Values
          </span>
        </div>

        {/* Reverse-DCF Implication Banner */}
        <div className="bg-[#080d16] rounded-xl p-4 border border-cyan-950/80 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs text-slate-300 font-medium">
              Reference Price Solved 5-Year Revenue CAGR:
            </span>
            <span className="font-mono font-bold text-cyan-400 text-xl sm:text-2xl">
              {valuation.solved_5y_cagr_pct !== null ? `${valuation.solved_5y_cagr_pct.toFixed(2)}%` : "-2.34%"}
            </span>
          </div>
          <div className="text-xs text-slate-400 space-y-1">
            <div>
              <strong>Model Inputs:</strong> Solved at reference price of{" "}
              <span className="text-slate-200 font-mono font-semibold">${valuation.reference_price.toFixed(2)}</span>{" "}
              as of {valuation.reference_date} (22.0% operating margin, 9.0% WACC, 2.5% terminal growth, 22.0% tax rate, 15.0% reinvestment rate).
            </div>
            <p className="text-[11px] text-amber-300/90 italic font-sans pt-0.5">
              The implied growth rate depends on the model assumptions. A negative solved growth rate reflects conditional model mechanics rather than a verified market expectation of contraction or definitive proof of undervaluation.
            </p>
          </div>
        </div>

        {/* Compact Scenario Comparison: Bear -> Base -> Bull */}
        <div className="space-y-2">
          <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
            Conditional Intrinsic Value Comparison:
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-left">
            {/* Bear case */}
            <div
              className="bg-[#080d16] border border-rose-900/50 rounded-xl p-3.5 space-y-2 flex flex-col justify-between"
              aria-label="Bear case scenario card"
            >
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-rose-950 text-rose-300 border border-rose-800">
                    Bear case
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">
                    Downside Stress
                  </span>
                </div>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">
                  Modeled Value per Share
                </span>
                <div className="font-mono font-bold text-rose-400 text-xl sm:text-2xl tracking-tight">
                  ${conservativeScenario?.per_share_value.toFixed(2) || "73.22"}
                </div>
                <div className="bg-[#0d1422] rounded-md p-2 border border-slate-800/80 space-y-0.5 text-xs">
                  <div className="text-[10px] text-slate-400">Difference from reference price:</div>
                  <div className="font-mono font-semibold text-rose-400">
                    {bearDiff !== null ? `${bearDiff > 0 ? "+" : ""}${bearDiff.toFixed(1)}%` : "-14.6%"}
                  </div>
                </div>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 space-y-0.5">
                <div className="flex justify-between">
                  <span>5Y Rev CAGR:</span>
                  <span className="text-slate-200">2.0%</span>
                </div>
                <div className="flex justify-between">
                  <span>Op Margin:</span>
                  <span className="text-slate-200">18.0%</span>
                </div>
                <div className="flex justify-between">
                  <span>WACC:</span>
                  <span className="text-slate-200">9.5%</span>
                </div>
              </div>
            </div>

            {/* Base case */}
            <div
              className="bg-[#080d16] border border-cyan-700/80 ring-1 ring-cyan-700/40 rounded-xl p-3.5 space-y-2 flex flex-col justify-between shadow-md"
              aria-label="Base case scenario card"
            >
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-cyan-950 text-cyan-300 border border-cyan-800">
                    Base case
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400">
                    Central Case
                  </span>
                </div>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">
                  Modeled Value per Share
                </span>
                <div className="font-mono font-bold text-white text-xl sm:text-2xl tracking-tight">
                  ${baseScenario?.per_share_value.toFixed(2) || "127.49"}
                </div>
                <div className="bg-[#0d1422] rounded-md p-2 border border-slate-800/80 space-y-0.5 text-xs">
                  <div className="text-[10px] text-slate-400">Difference from reference price:</div>
                  <div className="font-mono font-semibold text-cyan-300">
                    {baseDiff !== null ? `${baseDiff > 0 ? "+" : ""}${baseDiff.toFixed(1)}%` : "+48.7%"}
                  </div>
                </div>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 space-y-0.5">
                <div className="flex justify-between">
                  <span>5Y Rev CAGR:</span>
                  <span className="text-slate-200">8.0%</span>
                </div>
                <div className="flex justify-between">
                  <span>Op Margin:</span>
                  <span className="text-slate-200">22.0%</span>
                </div>
                <div className="flex justify-between">
                  <span>WACC:</span>
                  <span className="text-slate-200">9.0%</span>
                </div>
              </div>
            </div>

            {/* Bull case */}
            <div
              className="bg-[#080d16] border border-emerald-900/50 rounded-xl p-3.5 space-y-2 flex flex-col justify-between"
              aria-label="Bull case scenario card"
            >
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800">
                    Bull case
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">
                    Upside Expansion
                  </span>
                </div>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">
                  Modeled Value per Share
                </span>
                <div className="font-mono font-bold text-emerald-400 text-xl sm:text-2xl tracking-tight">
                  ${optimisticScenario?.per_share_value.toFixed(2) || "209.87"}
                </div>
                <div className="bg-[#0d1422] rounded-md p-2 border border-slate-800/80 space-y-0.5 text-xs">
                  <div className="text-[10px] text-slate-400">Difference from reference price:</div>
                  <div className="font-mono font-semibold text-emerald-400">
                    {bullDiff !== null ? `${bullDiff > 0 ? "+" : ""}${bullDiff.toFixed(1)}%` : "+144.8%"}
                  </div>
                </div>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 space-y-0.5">
                <div className="flex justify-between">
                  <span>5Y Rev CAGR:</span>
                  <span className="text-slate-200">14.0%</span>
                </div>
                <div className="flex justify-between">
                  <span>Op Margin:</span>
                  <span className="text-slate-200">24.0%</span>
                </div>
                <div className="flex justify-between">
                  <span>WACC:</span>
                  <span className="text-slate-200">8.5%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="text-[10px] text-slate-400 italic font-mono pt-1">
            Modeled values represent discounted cash flow intrinsic present equity values per share based on conditional operating assumptions, not future price targets, guaranteed returns, or assigned probabilities.
          </div>
        </div>
      </section>

      {/* D. What Must Be True? — Up to Three Drivers */}
      <section
        className="bg-[#0e1726] border border-[#233149] rounded-xl p-4 sm:p-5 space-y-3.5"
        aria-label="What Must Be True"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" aria-hidden="true"></span>
            <h2 className="text-xs font-bold uppercase tracking-wider text-emerald-300">
              D. What Must Be True? (Key Drivers That Could Support Entry)
            </h2>
          </div>
          <span className="text-[10px] font-mono text-slate-400">3 Evidence-Supported Drivers</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 text-xs">
          {/* Driver 1 */}
          <div className="bg-[#080d16] border border-emerald-950/80 rounded-lg p-3.5 space-y-2 flex flex-col justify-between">
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-emerald-300 font-semibold text-[11px] uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" aria-hidden="true"></span>
                <h3>1. DTC Velocity &amp; International Expansion</h3>
              </div>
              <p className="text-slate-300 leading-relaxed">
                <strong className="text-slate-200">Condition Required:</strong> HOKA direct-to-consumer sales must sustain double-digit growth and overseas adoption expand without margin-diluting promotional spend.
              </p>
              <div className="text-slate-400 text-[11px] pt-1">
                <strong className="text-emerald-400">Current Evidence:</strong> In Q1 FY27, HOKA DTC net sales grew 17.3% YoY to $256.8M, outpacing wholesale (+2.9%) and driving brand pull.
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 font-sans">
              <strong>Uncertainty:</strong> Sustainability of premium DTC full-price sell-through rates across non-running categories.
            </div>
          </div>

          {/* Driver 2 */}
          <div className="bg-[#080d16] border border-emerald-950/80 rounded-lg p-3.5 space-y-2 flex flex-col justify-between">
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-emerald-300 font-semibold text-[11px] uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" aria-hidden="true"></span>
                <h3>2. Gross Margin &amp; Operating Defense</h3>
              </div>
              <p className="text-slate-300 leading-relaxed">
                <strong className="text-slate-200">Condition Required:</strong> Consolidated gross margin must hold above 50.0%–55.0% and operating margin hold near ~22% despite wholesale moderation.
              </p>
              <div className="text-slate-400 text-[11px] pt-1">
                <strong className="text-emerald-400">Current Evidence:</strong> Reported TTM operating margin was 22.67% ($1.25B EBIT on $5.53B revenue) with disciplined pricing.
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 font-sans">
              <strong>Uncertainty:</strong> Wholesale channel inventory aging and retail promotional discounting pressure.
            </div>
          </div>

          {/* Driver 3 */}
          <div className="bg-[#080d16] border border-emerald-950/80 rounded-lg p-3.5 space-y-2 flex flex-col justify-between">
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-emerald-300 font-semibold text-[11px] uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" aria-hidden="true"></span>
                <h3>3. Zero-Debt Balance Sheet Strength</h3>
              </div>
              <p className="text-slate-300 leading-relaxed">
                <strong className="text-slate-200">Condition Required:</strong> Deckers must preserve its net cash cushion to self-fund international footprint expansion without debt or equity dilution.
              </p>
              <div className="text-slate-400 text-[11px] pt-1">
                <strong className="text-emerald-400">Current Evidence:</strong> $1.60B in cash reserves, zero funded senior debt, and undrawn credit facility under clean 6:1 split structure.
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 font-sans">
              <strong>Uncertainty:</strong> Capital allocation discipline between share buybacks and brand infrastructure.
            </div>
          </div>
        </div>
      </section>

      {/* E. What Breaks The Thesis? — Up to Three Conditions */}
      <section
        className="bg-[#140f1a] border border-rose-900/60 rounded-xl p-4 sm:p-5 space-y-3.5"
        aria-label="What Breaks The Thesis"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-rose-900/40 pb-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-400" aria-hidden="true"></span>
            <h2 className="text-xs font-bold uppercase tracking-wider text-rose-300">
              E. What Breaks The Thesis? (Invalidation &amp; Thesis Killers)
            </h2>
          </div>
          <span className="text-[10px] font-mono text-rose-400 bg-rose-950/80 border border-rose-800 px-2 py-0.5 rounded">
            Observable Invalidation Triggers
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 text-xs">
          {/* Invalidation 1 */}
          <div className="bg-[#080d16] border border-rose-950/80 rounded-lg p-3.5 space-y-1.5">
            <div className="flex items-center gap-1.5 text-rose-300 font-semibold text-[11px] uppercase tracking-wider">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400" aria-hidden="true"></span>
              <h3>1. Gross Margin Breach Below 50.0%</h3>
            </div>
            <p className="text-slate-300 leading-relaxed">
              Consolidated gross margin falling below 50.0% due to wholesale inventory accumulation or retail discounting would indicate brand equity erosion and loss of pricing power.
            </p>
            <div className="text-[10px] text-rose-400/90 font-mono pt-1">
              Threshold: &lt; 50.0% consolidated gross margin
            </div>
          </div>

          {/* Invalidation 2 */}
          <div className="bg-[#080d16] border border-rose-950/80 rounded-lg p-3.5 space-y-1.5">
            <div className="flex items-center gap-1.5 text-rose-300 font-semibold text-[11px] uppercase tracking-wider">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400" aria-hidden="true"></span>
              <h3>2. HOKA DTC Deceleration Below 10.0%</h3>
            </div>
            <p className="text-slate-300 leading-relaxed">
              Quarterly HOKA DTC net sales growth slowing below 10.0% YoY would signal consumer brand fatigue or market saturation in core running and lifestyle franchises.
            </p>
            <div className="text-[10px] text-rose-400/90 font-mono pt-1">
              Threshold: &lt; 10.0% YoY quarterly DTC growth
            </div>
          </div>

          {/* Invalidation 3 */}
          <div className="bg-[#080d16] border border-rose-950/80 rounded-lg p-3.5 space-y-1.5">
            <div className="flex items-center gap-1.5 text-rose-300 font-semibold text-[11px] uppercase tracking-wider">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400" aria-hidden="true"></span>
              <h3>3. Wholesale Channel Contraction</h3>
            </div>
            <p className="text-slate-300 leading-relaxed">
              Wholesale net sales growth (which moderated to +2.9% in Q1 FY27 vs 27.9% in FY26) turning negative across major accounts, indicating channel inventory destocking.
            </p>
            <div className="text-[10px] text-rose-400/90 font-mono pt-1">
              Threshold: Wholesale revenue turning negative YoY
            </div>
          </div>
        </div>
      </section>

      {/* F. Potential Expectation Gap */}
      <section
        className="bg-[#0c121e] border border-purple-900/50 rounded-xl p-4 sm:p-5 space-y-3"
        aria-label="Potential Expectation Gap"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-purple-900/40 pb-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-purple-400" aria-hidden="true"></span>
            <h2 className="text-xs font-bold uppercase tracking-wider text-purple-300">
              F. Potential Expectation Gap (What Might The Prevailing View Be Missing?)
            </h2>
          </div>
          <span className="text-[10px] font-mono text-purple-400">Disciplined Expectation Check</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 text-xs">
          <div className="bg-[#080d16] border border-purple-950 p-3.5 rounded-lg space-y-1.5">
            <h3 className="text-purple-300 font-semibold text-[11px] uppercase tracking-wider">
              1. What The Model Implies
            </h3>
            <p className="text-slate-300 leading-relaxed">
              At $85.72, the Reverse-DCF solves for an implied 5-year revenue CAGR of -2.34% (under 22.0% margin and 9.0% WACC), indicating modest baseline operational expectations.
            </p>
          </div>

          <div className="bg-[#080d16] border border-purple-950 p-3.5 rounded-lg space-y-1.5">
            <h3 className="text-purple-300 font-semibold text-[11px] uppercase tracking-wider">
              2. The Alternative Hypothesis
            </h3>
            <p className="text-slate-300 leading-relaxed">
              DTC channel momentum (+17.3% YoY in Q1 FY27) indicates enduring brand pull that may outlast wholesale channel destocking, potentially supporting consolidated revenue expansion above the negative reverse-DCF solve.
            </p>
          </div>

          <div className="bg-[#080d16] border border-purple-950 p-3.5 rounded-lg space-y-1.5">
            <h3 className="text-purple-300 font-semibold text-[11px] uppercase tracking-wider">
              3. Supporting vs. Challenging Evidence
            </h3>
            <p className="text-slate-300 leading-relaxed">
              <strong>Supporting:</strong> HOKA DTC revenue expanded to $256.8M with strong gross margin contribution. <strong className="text-rose-300">Challenging:</strong> HOKA wholesale growth slowed sharply to +2.9% and seasonal UGG concentration persists.
            </p>
          </div>

          <div className="bg-[#080d16] border border-purple-950 p-3.5 rounded-lg space-y-1.5">
            <h3 className="text-purple-300 font-semibold text-[11px] uppercase tracking-wider">
              4. Unresolved Boundary &amp; Consensus Status
            </h3>
            <p className="text-slate-300 leading-relaxed font-sans">
              <em className="text-amber-300 font-normal">An expectation gap has not yet been established beyond observed brand channel divergence.</em> No dated external consensus estimate is modeled.
            </p>
          </div>
        </div>
      </section>

      {/* G. What Would Change The Decision? */}
      <section
        className="bg-[#101726] border border-[#233149] rounded-xl p-4 sm:p-5 space-y-3.5"
        aria-label="What Would Change The Decision"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" aria-hidden="true"></span>
            <h2 className="text-xs font-bold uppercase tracking-wider text-cyan-300">
              G. What Would Change The Decision?
            </h2>
          </div>
          <span className="text-[10px] font-mono text-slate-400">Monitoring &amp; Review Triggers</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
          {/* Strengthening conditions */}
          <div className="bg-emerald-950/20 border border-emerald-900/50 p-3.5 rounded-lg space-y-1.5">
            <h3 className="text-emerald-400 font-semibold text-xs flex items-center gap-1.5">
              <span aria-hidden="true">&#10003;</span>
              Evidence That Would Strengthen The Case:
            </h3>
            <p className="text-slate-300 leading-relaxed">
              Confirmation of international HOKA DTC net sales growth exceeding 20% YoY alongside consolidated gross margins sustaining above 55.0%.
            </p>
          </div>

          {/* Weakening conditions */}
          <div className="bg-rose-950/20 border border-rose-900/50 p-3.5 rounded-lg space-y-1.5">
            <h3 className="text-rose-400 font-semibold text-xs flex items-center gap-1.5">
              <span aria-hidden="true">&#9888;</span>
              Evidence That Would Weaken / Invalidate:
            </h3>
            <p className="text-slate-300 leading-relaxed">
              Consolidated gross margin breach below 50.0% or HOKA DTC quarterly net sales growth falling below 10.0% YoY.
            </p>
          </div>

          {/* Next review trigger */}
          <div className="bg-[#0b1019] border border-slate-800 p-3.5 rounded-lg space-y-1 sm:col-span-1">
            <div className="text-[11px] font-semibold text-cyan-300 uppercase tracking-wider">
              Next Review Trigger:
            </div>
            <p className="text-slate-300 leading-relaxed">
              Form 10-Q for Q2 FY27 earnings release (expected October/November 2026) for holiday pre-orders, international expansion metrics, and inventory aging.
            </p>
          </div>

          {/* Decisive unknown */}
          <div className="bg-[#0b1019] border border-slate-800 p-3.5 rounded-lg space-y-1 sm:col-span-1">
            <div className="text-[11px] font-semibold text-amber-300 uppercase tracking-wider">
              Decisive Unknown:
            </div>
            <p className="text-slate-300 leading-relaxed">
              Sustainability of premium DTC full-price sell-through rates as HOKA expands into broader lifestyle footwear categories outside core running.
            </p>
          </div>
        </div>
      </section>

      {/* Note: Section H ("What could this become?") is intentionally omitted because
          the current accepted DECK dossier does not support an approved transformation hypothesis. */}

      {/* =========================================================================
          PROGRESSIVE DISCLOSURE (Detailed Supporting Evidence Below Decision Surface)
         ========================================================================= */}
      <section className="pt-2 border-t border-[#1e293b] space-y-4" aria-label="Progressive Disclosures">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              Supporting Evidence &amp; Model Mechanics (Progressive Disclosure)
            </h2>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">
              Expandable verification records, capital structure details, and primary SEC filing locators.
            </p>
          </div>
          <span className="text-[10px] font-mono text-slate-500">
            Click section headers to expand
          </span>
        </div>

        {/* 1. Valuation Assumptions, Sensitivities & Full Comparison */}
        <div className="border border-slate-800 rounded-xl overflow-hidden bg-[#0c121e]">
          <button
            type="button"
            onClick={() => setValuationOpen(!valuationOpen)}
            className="w-full flex items-center justify-between p-4 bg-[#0a0f19] hover:bg-[#0e1626] font-semibold text-xs text-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-400 min-h-[44px]"
            aria-expanded={valuationOpen}
            aria-controls="disclosure-valuation-deck"
          >
            <div className="flex items-center gap-2">
              <span className="text-amber-400 font-mono">1.</span>
              <span>Valuation Mechanics, Parameters &amp; Full Scenario Comparison</span>
            </div>
            <span className="text-slate-400 font-mono text-xs">{valuationOpen ? "▲ Collapse" : "▼ Expand"}</span>
          </button>

          {valuationOpen && (
            <div id="disclosure-valuation-deck" className="p-4 sm:p-5 border-t border-slate-800 space-y-4 text-xs font-sans">
              <div className="space-y-1.5 text-slate-300 leading-relaxed">
                <p>
                  <strong>Model Specification:</strong> {valuation.model_type || "Single-stage Reverse-DCF FCFF"}. Reverse-DCF solves for the 5-year revenue CAGR implied by the reference close tape ($85.72), holding margin, WACC, and reinvestment rate fixed. Forward scenarios project explicit cash flows to compute conditional intrinsic equity values.
                </p>
                <div className="p-3 bg-[#080d16] rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300">
                  <div>• Model Anchor: ${valuation.reference_price.toFixed(2)} USD (ARX MarketStore regular close: {valuation.reference_date})</div>
                  <div>• Base Revenue: ${(valuation.reverse_dcf_parameters?.base_revenue ? valuation.reverse_dcf_parameters.base_revenue / 1e9 : 5.53).toFixed(2)}B</div>
                  <div>• Diluted Share Count: {(valuation.reverse_dcf_parameters?.current_shares ? valuation.reverse_dcf_parameters.current_shares / 1e6 : 138.56).toFixed(2)}M (6:1 split-adjusted)</div>
                  <div>• Net Debt Bridge: -$1,602.6M (Net cash buffer added to Enterprise Value)</div>
                  <div>• Solved 5Y Revenue CAGR: {valuation.solved_5y_cagr_pct?.toFixed(2)}% (Roundtrip value: ${valuation.roundtrip_value?.toFixed(2)})</div>
                </div>
              </div>

              {/* Full Comparative Table */}
              <div className="overflow-x-auto rounded-lg border border-slate-800">
                <table className="w-full text-left text-[11px] font-mono border-collapse">
                  <thead>
                    <tr className="bg-[#080d16] text-slate-400 border-b border-slate-800">
                      <th className="p-2.5 font-semibold">Assumption / Parameter</th>
                      <th className="p-2.5 font-semibold text-cyan-300">Reverse-DCF Solve</th>
                      <th className="p-2.5 font-semibold text-rose-300">Bear case</th>
                      <th className="p-2.5 font-semibold text-slate-200">Base case</th>
                      <th className="p-2.5 font-semibold text-emerald-300">Bull case</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-[#0c121e]">
                    <tr>
                      <td className="p-2 text-slate-400">Role / Case Description</td>
                      <td className="p-2 text-cyan-400">Price Solve Anchor</td>
                      <td className="p-2 text-slate-300">Downside Stress</td>
                      <td className="p-2 text-slate-200">Central Case</td>
                      <td className="p-2 text-emerald-400">Upside Expansion</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-slate-400">5Y Revenue CAGR</td>
                      <td className="p-2 font-bold text-cyan-400">{valuation.solved_5y_cagr_pct?.toFixed(2)}%</td>
                      <td className="p-2 text-slate-300">2.0%</td>
                      <td className="p-2 text-slate-200">8.0%</td>
                      <td className="p-2 text-emerald-400">14.0%</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-slate-400">Target Operating Margin</td>
                      <td className="p-2 text-cyan-400">22.0%</td>
                      <td className="p-2 text-slate-300">18.0%</td>
                      <td className="p-2 text-slate-200">22.0%</td>
                      <td className="p-2 text-emerald-400">24.0%</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-slate-400">Discount Rate (WACC)</td>
                      <td className="p-2 text-slate-300">9.0%</td>
                      <td className="p-2 text-slate-300">9.5%</td>
                      <td className="p-2 text-slate-300">9.0%</td>
                      <td className="p-2 text-slate-300">8.5%</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-slate-400">Terminal Growth Rate (g)</td>
                      <td className="p-2 text-slate-300">2.5%</td>
                      <td className="p-2 text-slate-300">2.0%</td>
                      <td className="p-2 text-slate-300">2.5%</td>
                      <td className="p-2 text-slate-300">3.0%</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-slate-400">Effective Tax Rate</td>
                      <td className="p-2 text-slate-300">22.0%</td>
                      <td className="p-2 text-slate-300">23.0%</td>
                      <td className="p-2 text-slate-300">22.0%</td>
                      <td className="p-2 text-slate-300">21.0%</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-slate-400">Reinvestment Rate</td>
                      <td className="p-2 text-slate-300">15.0%</td>
                      <td className="p-2 text-slate-300">18.0%</td>
                      <td className="p-2 text-slate-300">15.0%</td>
                      <td className="p-2 text-slate-300">12.0%</td>
                    </tr>
                    <tr className="bg-[#090f1a]">
                      <td className="p-2 text-slate-300 font-semibold">Modeled Value per Share</td>
                      <td className="p-2 font-bold text-cyan-300">${valuation.reference_price.toFixed(2)}</td>
                      <td className="p-2 font-bold text-rose-400">${conservativeScenario?.per_share_value.toFixed(2) || "73.22"}</td>
                      <td className="p-2 font-bold text-white">${baseScenario?.per_share_value.toFixed(2) || "127.49"}</td>
                      <td className="p-2 font-bold text-emerald-400">${optimisticScenario?.per_share_value.toFixed(2) || "209.87"}</td>
                    </tr>
                    <tr className="bg-[#080d16]">
                      <td className="p-2 text-slate-400">Difference from Reference Price</td>
                      <td className="p-2 text-slate-500 font-mono">Anchor (0.0%)</td>
                      <td className="p-2 text-rose-400 font-semibold">
                        {bearDiff !== null ? `${bearDiff > 0 ? "+" : ""}${bearDiff.toFixed(1)}%` : "-14.6%"}
                      </td>
                      <td className="p-2 text-slate-200 font-semibold">
                        {baseDiff !== null ? `${baseDiff > 0 ? "+" : ""}${baseDiff.toFixed(1)}%` : "+48.7%"}
                      </td>
                      <td className="p-2 text-emerald-400 font-semibold">
                        {bullDiff !== null ? `${bullDiff > 0 ? "+" : ""}${bullDiff.toFixed(1)}%` : "+144.8%"}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* 2. Financial Resilience, Sourced Facts & Net Debt Bridge */}
        <div className="border border-slate-800 rounded-xl overflow-hidden bg-[#0c121e]">
          <button
            type="button"
            onClick={() => setFinancialsOpen(!financialsOpen)}
            className="w-full flex items-center justify-between p-4 bg-[#0a0f19] hover:bg-[#0e1626] font-semibold text-xs text-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-400 min-h-[44px]"
            aria-expanded={financialsOpen}
            aria-controls="disclosure-financials-deck"
          >
            <div className="flex items-center gap-2">
              <span className="text-emerald-400 font-mono">2.</span>
              <span>Financial Resilience, Capital Structure &amp; Sourced Net Debt Bridge</span>
            </div>
            <span className="text-slate-400 font-mono text-xs">{financialsOpen ? "▲ Collapse" : "▼ Expand"}</span>
          </button>

          {financialsOpen && (
            <div id="disclosure-financials-deck" className="p-4 sm:p-5 border-t border-slate-800 space-y-4 text-xs font-sans">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div className="bg-[#080d16] p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-500 text-[10px] block">Reported Revenue</span>
                  <span className="font-mono text-slate-100 font-semibold text-sm">
                    {financial_profile.revenue_display}
                  </span>
                  <span className="text-[9px] text-slate-400 font-mono block">
                    {financial_profile.revenue_period}
                  </span>
                </div>

                <div className="bg-[#080d16] p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-500 text-[10px] block">Operating Margin</span>
                  <span className="font-mono text-slate-100 font-semibold text-sm">
                    {financial_profile.operating_margin_pct?.toFixed(1)}%
                  </span>
                  <span className="text-[9px] text-slate-400 font-mono block">Reported EBIT / Rev</span>
                </div>

                <div className="bg-[#080d16] p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-500 text-[10px] block">Total Cash Buffer</span>
                  <span className="font-mono text-slate-100 font-semibold text-sm">
                    ${((financial_profile.total_cash_and_investments || 1602589000) / 1e6).toFixed(1)}M
                  </span>
                  <span className="text-[9px] text-slate-400 font-mono block">Cash &amp; Equivalents</span>
                </div>

                <div className="bg-[#080d16] p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-500 text-[10px] block">Total Financial Debt</span>
                  <span className="font-mono text-emerald-400 font-semibold text-sm">
                    $0.00
                  </span>
                  <span className="text-[9px] text-slate-400 font-mono block">Zero Senior Debt</span>
                </div>
              </div>

              {/* Net Debt Bridge & Sourced Capital Structure Breakdown */}
              <div className="bg-[#080d16] rounded-xl border border-slate-800 p-3.5 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800">
                  <span className="text-[11px] font-bold text-slate-200 uppercase tracking-wider">
                    Sourced Capital Structure &amp; Net Debt Bridge Breakdown
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400">
                    Net Cash Bridge: -$1,602.6M
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-xs font-mono">
                  <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                    <span className="text-slate-500 text-[10px] block">Cash &amp; Equivalents</span>
                    <span className="text-slate-200 font-semibold">$1,602.6M</span>
                  </div>
                  <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                    <span className="text-slate-500 text-[10px] block">Short-Term Investments</span>
                    <span className="text-slate-200 font-semibold">$0.0M</span>
                  </div>
                  <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                    <span className="text-slate-500 text-[10px] block">Restricted Cash</span>
                    <span className="text-slate-200 font-semibold">$0.0M</span>
                  </div>
                  <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                    <span className="text-slate-500 text-[10px] block">Total Financial Debt</span>
                    <span className="text-emerald-400 font-semibold">$0.00 (Zero Funded Debt)</span>
                  </div>
                  <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                    <span className="text-slate-500 text-[10px] block">Operating Leases (ASC 842)</span>
                    <span className="text-slate-200 font-semibold">$472.3M</span>
                  </div>
                  <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                    <span className="text-slate-500 text-[10px] block">Net Debt Bridge</span>
                    <span className="text-cyan-400 font-semibold">-$1,602.6M (Net Cash)</span>
                  </div>
                </div>

                <div className="p-2.5 rounded bg-[#101726] border border-slate-800 text-[11px] text-slate-300 font-sans leading-relaxed">
                  <strong>Accounting Bridge Convention:</strong> Total cash of $1.60B against zero funded debt yields $1.60B net cash bridge added to enterprise value in DCF equity bridge. Store and distribution leases ($472.33M) are treated as operating expenses under ASC 842 and excluded from the net debt bridge to avoid double-counting against operating income.
                </div>

                <div className="text-[11px] text-slate-400 pt-1">
                  <strong>Selected Share Basis:</strong> <span className="font-mono text-slate-200">138.56M shares</span> (SPLIT_ADJUSTED_DILUTED) • Period: Three months ended June 30, 2026 (Form 10-Q Q1 FY27).
                </div>
              </div>

              {/* Primary SEC Filings */}
              <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400 flex flex-wrap items-center gap-2">
                <span className="text-slate-500 font-medium">Attributable SEC XBRL Filings:</span>
                {financial_profile.primary_filings && financial_profile.primary_filings.length > 0 ? (
                  financial_profile.primary_filings.map((f, idx) => (
                    <a
                      key={idx}
                      href={f.url || `https://www.sec.gov/edgar/browse/?CIK=${identity.cik}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 font-mono text-[10px] transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-400 inline-flex items-center gap-1 min-h-[44px]"
                      title={`Open SEC EDGAR Filing Form ${f.form} for period ended ${f.period_end}`}
                    >
                      <span>{f.form} ({f.period_end}) • Accn: {f.accession_number}</span>
                      <span aria-hidden="true">↗</span>
                    </a>
                  ))
                ) : (
                  <span className="text-slate-500 italic">No primary XBRL filings</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* 3. Full Dossier Baseline & Research Provenance */}
        <div className="border border-slate-800 rounded-xl overflow-hidden bg-[#0c121e]">
          <button
            type="button"
            onClick={() => setProvenanceOpen(!provenanceOpen)}
            className="w-full flex items-center justify-between p-4 bg-[#0a0f19] hover:bg-[#0e1626] font-semibold text-xs text-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-400 min-h-[44px]"
            aria-expanded={provenanceOpen}
            aria-controls="disclosure-provenance-deck"
          >
            <div className="flex items-center gap-2">
              <span className="text-purple-400 font-mono">3.</span>
              <span>Full Dossier Baseline &amp; Quality Watch Selection Record (Questions A–F)</span>
            </div>
            <span className="text-slate-400 font-mono text-xs">{provenanceOpen ? "▲ Collapse" : "▼ Expand"}</span>
          </button>

          {provenanceOpen && (
            <div id="disclosure-provenance-deck" className="p-4 sm:p-5 border-t border-slate-800 space-y-4 text-xs font-sans">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="bg-[#080d16] border border-slate-800 p-3 rounded-lg space-y-1">
                  <h3 className="text-slate-200 font-semibold text-[11px] flex items-center gap-1">
                    <span className="text-amber-400 font-mono">A.</span> Why This Business?
                  </h3>
                  <p className="text-slate-300 leading-relaxed">{rationale?.why_this_business}</p>
                </div>

                <div className="bg-[#080d16] border border-slate-800 p-3 rounded-lg space-y-1">
                  <h3 className="text-slate-200 font-semibold text-[11px] flex items-center gap-1">
                    <span className="text-amber-400 font-mono">B.</span> Why Research It Now?
                  </h3>
                  <p className="text-slate-300 leading-relaxed">{rationale?.why_research_now}</p>
                </div>

                <div className="bg-[#080d16] border border-slate-800 p-3 rounded-lg space-y-1">
                  <h3 className="text-slate-200 font-semibold text-[11px] flex items-center gap-1">
                    <span className="text-amber-400 font-mono">C.</span> What Does The Price Require?
                  </h3>
                  <p className="text-slate-300 leading-relaxed">{rationale?.price_requirements}</p>
                </div>

                <div className="bg-[#080d16] border border-slate-800 p-3 rounded-lg space-y-1">
                  <h3 className="text-slate-200 font-semibold text-[11px] flex items-center gap-1">
                    <span className="text-amber-400 font-mono">D.</span> What Might ARX See Differently?
                  </h3>
                  <p className="text-slate-300 leading-relaxed">{rationale?.differentiated_view}</p>
                </div>

                <div className="bg-[#080d16] border border-slate-800 p-3 rounded-lg space-y-1">
                  <h3 className="text-slate-200 font-semibold text-[11px] flex items-center gap-1">
                    <span className="text-amber-400 font-mono">E.</span> What Would Change The Assessment?
                  </h3>
                  <p className="text-slate-300 leading-relaxed">{rationale?.assessment_changes}</p>
                </div>

                <div className="bg-[#080d16] border border-slate-800 p-3 rounded-lg space-y-1">
                  <h3 className="text-slate-200 font-semibold text-[11px] flex items-center gap-1">
                    <span className="text-amber-400 font-mono">F.</span> Why Prioritize Research Attention?
                  </h3>
                  <p className="text-slate-300 leading-relaxed">{rationale?.queue_priority}</p>
                </div>
              </div>

              {/* Publication Governance Metadata */}
              <div className="p-3 bg-[#080d16] rounded-lg border border-slate-800 text-[11px] font-mono text-slate-400 space-y-1">
                <div className="text-slate-300 font-bold uppercase tracking-wider">
                  Governance Record:
                </div>
                <div>Standard: {dossier.publication_metadata?.governing_standard}</div>
                <div>Contract: {dossier.publication_metadata?.governing_contract}</div>
                <div>Lifecycle Status: {dossier.publication_metadata?.lifecycle_status}</div>
                <div>Review Status: {dossier.publication_metadata?.review_status}</div>
                <div>Technical Verification: {dossier.publication_metadata?.technical_verification}</div>
                <div>Authoring Date: {dossier.publication_metadata?.authoring_date}</div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* =========================================================================
          Footer: Technical Radar Navigation
         ========================================================================= */}
      <footer className="pt-2 border-t border-[#1e293b] flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="text-slate-400 text-[11px] max-w-lg">
          <strong>Technical Analysis Notice:</strong> {navigation.radar_note} Strategy timing confirmation is distinct from multi-year fundamental compounder thesis.
        </div>
        <div className="flex items-center gap-2">
          <Link
            href={navigation.radar_url}
            className="px-4 py-2 rounded-xl text-xs font-mono font-medium bg-cyan-950 text-cyan-300 border border-cyan-800 hover:bg-cyan-900 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-400 inline-flex items-center gap-1.5 min-h-[44px]"
            title={`Open ${identity.symbol} in Technical Radar console`}
          >
            <span>Inspect in Radar ({navigation.radar_url})</span>
            <span aria-hidden="true">&rarr;</span>
          </Link>
        </div>
      </footer>
    </article>
  );
}
