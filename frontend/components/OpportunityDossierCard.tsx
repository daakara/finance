"use client";

/**
 * frontend/components/OpportunityDossierCard.tsx
 *
 * Institutional presentation surface for the ARX Read-Only Opportunity Dossier.
 * Renders the 5-question central investment assessment based on the accepted ODP-05 research candidate.
 *
 * Invariants:
 * - Pure read-only presentation grounded in the frozen 2026-10-16 market close ledger.
 * - Reverse-DCF reference price explicitly labeled as model input, never an analyst target.
 * - Suppressed valuations (IREN, CORZ) expose zero numeric fair values and display exact resume conditions.
 * - Qualified valuation (MU) visibly renders its cyclical recovery qualification banner.
 * - Zero funded debt (DUOL) explicitly distinguished from missing debt.
 * - Excludes simulated technical overlays (RVOL, VCP, Confluence); provides optional Radar deep-links.
 * - Sourced capital structure & net debt bridge explicitly reconciles liquidity and lease conventions.
 * - Full scenario assumptions disclosure compares Reverse-DCF solve with forward scenarios.
 * - Accessible controls: touch targets >= 44px, keyboard navigation with cyan focus rings.
 */

import React from "react";
import Link from "next/link";
import { OpportunityDossier } from "../lib/opportunityDossiers";

interface OpportunityDossierCardProps {
  dossier: OpportunityDossier;
}

export default function OpportunityDossierCard({ dossier }: OpportunityDossierCardProps) {
  const { identity, central_assessment, valuation, financial_profile, monitoring, navigation } = dossier;

  const isSuppressed = valuation.valuation_status === "VALUATION_SUPPRESSED";
  const isQualified = valuation.valuation_status === "COMPUTED_QUALIFIED";

  // Disposition badge
  const getDispositionBadge = (disp: string) => {
    switch (disp) {
      case "WATCH_CONDITIONS":
        return (
          <span className="px-2.5 py-1 rounded text-xs font-mono font-semibold bg-amber-950/80 text-amber-300 border border-amber-700/80 shadow-sm inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
            WATCH CONDITIONS
          </span>
        );
      case "NO_SUPPORTED_OPPORTUNITY":
        return (
          <span className="px-2.5 py-1 rounded text-xs font-mono font-semibold bg-rose-950/80 text-rose-300 border border-rose-700/80 shadow-sm inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span>
            NO SUPPORTED OPPORTUNITY
          </span>
        );
      default:
        return (
          <span className="px-2.5 py-1 rounded text-xs font-mono font-semibold bg-slate-800 text-slate-300 border border-slate-700">
            {disp}
          </span>
        );
    }
  };

  // Opportunity route badge
  const getRouteBadge = (route: string) => {
    switch (route) {
      case "UNDERVALUED_GROWTH":
        return <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">Undervalued Growth</span>;
      case "BUSINESS_INFLECTION":
        return <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-950 text-purple-300 border border-purple-800">Business Inflection</span>;
      case "CYCLICAL_RECOVERY":
        return <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-950 text-amber-300 border border-amber-800">Cyclical Recovery</span>;
      default:
        return <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-400 border border-slate-700">{route}</span>;
    }
  };

  return (
    <article
      className="bg-[#0b1019] border border-[#1e293b] rounded-2xl p-4 sm:p-6 shadow-xl space-y-6 text-slate-100 transition-all"
      aria-labelledby={`dossier-heading-${identity.symbol}`}
    >
      {/* 1. Header & Identity Context */}
      <header className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-[#1e293b]">
        <div>
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            <h1 id={`dossier-heading-${identity.symbol}`} className="text-2xl font-bold font-mono text-white tracking-wide">
              {identity.symbol}
            </h1>
            <span className="text-base sm:text-lg text-slate-200 font-medium">{identity.company_name}</span>
            <span className="text-xs text-slate-500 font-mono">
              ({identity.exchange} • CIK {identity.cik})
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2 mt-2">
            {getRouteBadge(identity.opportunity_route)}
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#141c2e] text-slate-300 border border-slate-700">
              Frozen Research Publication
            </span>
            {identity.sec_edgar_browse_url && (
              <a
                href={identity.sec_edgar_browse_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 hover:underline inline-flex items-center gap-1 focus:outline-none focus:ring-2 focus:ring-cyan-400 rounded px-2 min-h-[44px]"
                title={`Open SEC EDGAR Browse page for CIK ${identity.cik}`}
              >
                <span>SEC EDGAR Profile</span>
                <span aria-hidden="true">↗</span>
              </a>
            )}
          </div>
        </div>

        {/* Reference Price & Disposition */}
        <div className="text-right flex flex-col items-end gap-1.5">
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-mono font-bold text-white tracking-tight">
              ${valuation.reference_price.toFixed(2)}
            </span>
            <span className="text-xs text-slate-400 font-mono">USD</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono">
            Regular Close: {valuation.reference_date}
          </div>
          <div>{getDispositionBadge(central_assessment.disposition)}</div>
        </div>
      </header>

      {/* 2. Central Assessment Task */}
      <section className="bg-[#101726] border border-[#233149] rounded-xl p-4 sm:p-5 space-y-3" aria-label="Central Assessment">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <h2 className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" aria-hidden="true"></span>
            Central Assessment: Merits, Opposing Evidence &amp; Invalidation
          </h2>
          <span className="text-[11px] font-mono text-slate-400">{central_assessment.disposition_label}</span>
        </div>

        {/* Derived Presentation Summary */}
        <p className="text-sm text-slate-200 leading-relaxed font-normal">
          {central_assessment.presentation_summary}
        </p>

        {/* Opposing vs Supporting Evidence Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
          {/* Supporting Evidence */}
          <div className="bg-emerald-950/20 border border-emerald-900/40 rounded-lg p-3 space-y-1.5">
            <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <span aria-hidden="true">&#10003;</span>
              Strongest Supporting Evidence
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              {central_assessment.strongest_supporting_evidence}
            </p>
          </div>

          {/* Opposing Evidence */}
          <div className="bg-rose-950/20 border border-rose-900/40 rounded-lg p-3 space-y-1.5">
            <h3 className="text-xs font-semibold text-rose-400 uppercase tracking-wider flex items-center gap-1.5">
              <span aria-hidden="true">&#10007;</span>
              Strongest Opposing Evidence
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              {central_assessment.strongest_opposing_evidence}
            </p>
          </div>
        </div>

        {/* Price Watch & Synthesis Callout */}
        <div className="bg-[#0b1019] border border-slate-800 rounded-lg p-3 text-xs text-slate-300 space-y-1">
          <div>
            <strong className="text-amber-300">Price Watch Condition:</strong>{" "}
            <span className="text-slate-200">{central_assessment.price_watch_condition}</span>
          </div>
          <div className="text-slate-400 text-[11px]">
            <strong>Synthesis Rationale:</strong> {central_assessment.confidence_assessment}
          </div>
          {central_assessment.financing_dilution_exposure && (
            <div className="text-slate-400 text-[11px]">
              <strong>Financing &amp; Dilution Exposure:</strong> {central_assessment.financing_dilution_exposure}
            </div>
          )}
        </div>
      </section>

      {/* 3. Valuation Support & Priced-in Expectations */}
      <section className="space-y-2.5" aria-label="Valuation Support">
        <h2 className="text-xs font-semibold text-amber-300 uppercase tracking-wider flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-amber-400" aria-hidden="true"></span>
          Valuation Support &amp; Reverse-DCF Expectations
        </h2>

        {isSuppressed ? (
          /* Suppressed Valuation Block (Fail-Closed) */
          <div className="bg-amber-950/30 border border-amber-800/80 rounded-xl p-4 sm:p-5 text-xs space-y-3">
            <div className="flex items-center gap-2 text-amber-400 font-semibold">
              <span className="px-2 py-0.5 rounded bg-amber-900/80 text-[10px] font-mono border border-amber-700">
                VALUATION SUPPRESSED
              </span>
              <span>Fail-Closed Valuation Protection Active</span>
            </div>
            <p className="text-slate-300 text-xs leading-relaxed">
              {valuation.suppression_reason}
            </p>
            {valuation.resume_conditions && valuation.resume_conditions.length > 0 && (
              <div className="pt-2 border-t border-amber-900/60 space-y-1.5">
                <div className="text-[11px] font-semibold text-amber-300 uppercase tracking-wider">
                  Company-Specific Conditions Required to Resume Valuation (Fail-Closed Gates):
                </div>
                <ul className="list-disc list-inside space-y-1 text-slate-300 text-xs">
                  {valuation.resume_conditions.map((cond, idx) => (
                    <li key={idx} className="leading-relaxed">
                      {cond}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          /* Computed Valuation Block */
          <div className="bg-[#101726] border border-[#233149] rounded-xl p-4 sm:p-5 space-y-4">
            <div className="flex flex-wrap justify-between items-center gap-2">
              <span className="text-xs text-slate-300">Reverse-DCF Solved 5Y Revenue CAGR:</span>
              <span className="font-mono font-bold text-cyan-400 text-lg">
                {valuation.solved_5y_cagr_pct !== null ? `${valuation.solved_5y_cagr_pct.toFixed(2)}%` : "N/A"}
              </span>
            </div>

            <div className="text-xs text-slate-400">
              {valuation.reference_price_label}:{" "}
              <span className="font-mono text-slate-200">${valuation.reference_price.toFixed(2)}</span>
              {valuation.roundtrip_value !== null && (
                <>
                  {" "}| Roundtrip Solved Value:{" "}
                  <span className="font-mono text-slate-200">${valuation.roundtrip_value.toFixed(2)}</span>
                </>
              )}
            </div>

            {/* Model Qualification Notice (e.g. MU) */}
            {isQualified && valuation.qualification_notice && (
              <div className="p-3 rounded-lg bg-amber-950/40 border border-amber-800 text-xs text-amber-200 leading-relaxed space-y-0.5">
                <strong className="text-amber-300 uppercase tracking-wider text-[11px] block">
                  Model Qualification Notice:
                </strong>
                <p>{valuation.qualification_notice}</p>
              </div>
            )}

            {/* Multi-Scenario Bracket with Conditional Wording */}
            {valuation.scenarios && Object.keys(valuation.scenarios).length > 0 && (
              <div className="mt-3 pt-3 border-t border-slate-800 space-y-3">
                <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                  Conditional Modeled Valuation Range:
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-center text-xs">
                  {/* Conservative */}
                  {valuation.scenarios.conservative && (
                    <div className="bg-[#0b1019] border border-slate-800 p-3 rounded-lg space-y-1">
                      <div className="text-slate-400 text-[10px] uppercase font-semibold">
                        {valuation.scenarios.conservative.name}
                      </div>
                      <div className="font-mono font-bold text-rose-400 text-base">
                        ${valuation.scenarios.conservative.per_share_value.toFixed(2)}
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono">
                        {valuation.scenarios.conservative.potential_return_pct > 0 ? "+" : ""}
                        {valuation.scenarios.conservative.potential_return_pct.toFixed(1)}% modeled return
                      </div>
                    </div>
                  )}

                  {/* Base */}
                  {valuation.scenarios.base && (
                    <div className="bg-[#0b1019] border border-cyan-800/80 p-3 rounded-lg space-y-1 shadow-sm ring-1 ring-cyan-800/40">
                      <div className="text-cyan-300 text-[10px] uppercase font-semibold">
                        {valuation.scenarios.base.name}
                      </div>
                      <div className="font-mono font-bold text-white text-base">
                        ${valuation.scenarios.base.per_share_value.toFixed(2)}
                      </div>
                      <div className="text-[10px] text-slate-400 font-mono">
                        {valuation.scenarios.base.potential_return_pct > 0 ? "+" : ""}
                        {valuation.scenarios.base.potential_return_pct.toFixed(1)}% modeled return
                      </div>
                    </div>
                  )}

                  {/* Optimistic */}
                  {valuation.scenarios.optimistic && (
                    <div className="bg-[#0b1019] border border-slate-800 p-3 rounded-lg space-y-1">
                      <div className="text-slate-400 text-[10px] uppercase font-semibold">
                        {valuation.scenarios.optimistic.name}
                      </div>
                      <div className="font-mono font-bold text-emerald-400 text-base">
                        ${valuation.scenarios.optimistic.per_share_value.toFixed(2)}
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono">
                        {valuation.scenarios.optimistic.potential_return_pct > 0 ? "+" : ""}
                        {valuation.scenarios.optimistic.potential_return_pct.toFixed(1)}% modeled return
                      </div>
                    </div>
                  )}
                </div>

                <div className="text-[11px] text-slate-400 space-y-1">
                  {valuation.scenarios.conservative && (
                    <div>• {valuation.scenarios.conservative.conditional_wording}</div>
                  )}
                  {valuation.scenarios.base && (
                    <div>• {valuation.scenarios.base.conditional_wording}</div>
                  )}
                </div>

                {/* Scenario Assumptions & Model Mechanics Disclosure (Finding B) */}
                <div className="pt-3 border-t border-slate-800/80 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
                      Scenario Assumptions &amp; Model Mechanics
                    </span>
                    <span className="text-[10px] font-mono text-cyan-400">Reverse-DCF Solve vs. Forward Scenarios</span>
                  </div>

                  <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
                    <strong>Model Distinction:</strong> The Reverse-DCF solves for the 5-year revenue CAGR embedded in the reference close price (${valuation.reference_price.toFixed(2)}), holding margin, WACC, and reinvestment rate fixed. The forward scenarios project explicit cash flows under specified operating assumptions to compute conditional intrinsic equity values.
                  </p>

                  <div className="overflow-x-auto rounded-lg border border-slate-800">
                    <table className="w-full text-left text-[11px] font-mono border-collapse">
                      <thead>
                        <tr className="bg-[#080d16] text-slate-400 border-b border-slate-800">
                          <th className="p-2 font-semibold">Assumption / Metric</th>
                          <th className="p-2 font-semibold text-cyan-300">Reverse-DCF Solve</th>
                          <th className="p-2 font-semibold text-rose-300">Conservative</th>
                          <th className="p-2 font-semibold text-slate-200">Base Case</th>
                          <th className="p-2 font-semibold text-emerald-300">Optimistic</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 bg-[#0c121e]">
                        <tr>
                          <td className="p-2 text-slate-400">Model Role / Purpose</td>
                          <td className="p-2 text-cyan-400">Price Solve Anchor</td>
                          <td className="p-2 text-slate-300">Downside Stress</td>
                          <td className="p-2 text-slate-200">Central Case</td>
                          <td className="p-2 text-emerald-400">Upside Expansion</td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-400">5Y Revenue CAGR</td>
                          <td className="p-2 font-bold text-cyan-400">
                            {valuation.reverse_dcf_parameters?.solved_revenue_cagr_pct !== undefined
                              ? `${valuation.reverse_dcf_parameters.solved_revenue_cagr_pct.toFixed(2)}%`
                              : valuation.solved_5y_cagr_pct !== null
                              ? `${valuation.solved_5y_cagr_pct.toFixed(2)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.conservative?.revenue_cagr_pct !== undefined
                              ? `${valuation.scenarios.conservative.revenue_cagr_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-200">
                            {valuation.scenarios.base?.revenue_cagr_pct !== undefined
                              ? `${valuation.scenarios.base.revenue_cagr_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-emerald-400">
                            {valuation.scenarios.optimistic?.revenue_cagr_pct !== undefined
                              ? `${valuation.scenarios.optimistic.revenue_cagr_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-400">Target Operating Margin</td>
                          <td className="p-2 text-cyan-400">
                            {valuation.reverse_dcf_parameters?.operating_margin_pct !== undefined
                              ? `${valuation.reverse_dcf_parameters.operating_margin_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.conservative?.operating_margin_pct !== undefined
                              ? `${valuation.scenarios.conservative.operating_margin_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-200">
                            {valuation.scenarios.base?.operating_margin_pct !== undefined
                              ? `${valuation.scenarios.base.operating_margin_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-emerald-400">
                            {valuation.scenarios.optimistic?.operating_margin_pct !== undefined
                              ? `${valuation.scenarios.optimistic.operating_margin_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-400">Discount Rate (WACC)</td>
                          <td className="p-2 text-slate-300">
                            {valuation.reverse_dcf_parameters?.wacc_pct !== undefined
                              ? `${valuation.reverse_dcf_parameters.wacc_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.conservative?.wacc_pct !== undefined
                              ? `${valuation.scenarios.conservative.wacc_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.base?.wacc_pct !== undefined
                              ? `${valuation.scenarios.base.wacc_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.optimistic?.wacc_pct !== undefined
                              ? `${valuation.scenarios.optimistic.wacc_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-400">Terminal Growth Rate (g)</td>
                          <td className="p-2 text-slate-300">
                            {valuation.reverse_dcf_parameters?.terminal_growth_pct !== undefined
                              ? `${valuation.reverse_dcf_parameters.terminal_growth_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.conservative?.terminal_growth_pct !== undefined
                              ? `${valuation.scenarios.conservative.terminal_growth_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.base?.terminal_growth_pct !== undefined
                              ? `${valuation.scenarios.base.terminal_growth_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.optimistic?.terminal_growth_pct !== undefined
                              ? `${valuation.scenarios.optimistic.terminal_growth_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-400">Effective Tax Rate</td>
                          <td className="p-2 text-slate-300">
                            {valuation.reverse_dcf_parameters?.tax_rate_pct !== undefined
                              ? `${valuation.reverse_dcf_parameters.tax_rate_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.conservative?.tax_rate_pct !== undefined
                              ? `${valuation.scenarios.conservative.tax_rate_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.base?.tax_rate_pct !== undefined
                              ? `${valuation.scenarios.base.tax_rate_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.optimistic?.tax_rate_pct !== undefined
                              ? `${valuation.scenarios.optimistic.tax_rate_pct.toFixed(1)}%`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-400">Reinvestment (Sales/Cap)</td>
                          <td className="p-2 text-slate-300">
                            {valuation.reverse_dcf_parameters?.reinvestment_rate_pct !== undefined
                              ? `${valuation.reverse_dcf_parameters.reinvestment_rate_pct.toFixed(2)}x`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.conservative?.reinvestment_rate_pct !== undefined
                              ? `${valuation.scenarios.conservative.reinvestment_rate_pct.toFixed(2)}x`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.base?.reinvestment_rate_pct !== undefined
                              ? `${valuation.scenarios.base.reinvestment_rate_pct.toFixed(2)}x`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.optimistic?.reinvestment_rate_pct !== undefined
                              ? `${valuation.scenarios.optimistic.reinvestment_rate_pct.toFixed(2)}x`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td className="p-2 text-slate-400">Implied Equity Value</td>
                          <td className="p-2 text-cyan-400 font-bold">
                            {valuation.reverse_dcf_parameters?.current_shares
                              ? `$${((valuation.reference_price * valuation.reverse_dcf_parameters.current_shares) / 1e6).toFixed(0)}M`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-300">
                            {valuation.scenarios.conservative?.equity_value !== undefined
                              ? `$${(valuation.scenarios.conservative.equity_value / 1e6).toFixed(0)}M`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-slate-200">
                            {valuation.scenarios.base?.equity_value !== undefined
                              ? `$${(valuation.scenarios.base.equity_value / 1e6).toFixed(0)}M`
                              : "N/A"}
                          </td>
                          <td className="p-2 text-emerald-400">
                            {valuation.scenarios.optimistic?.equity_value !== undefined
                              ? `$${(valuation.scenarios.optimistic.equity_value / 1e6).toFixed(0)}M`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr className="bg-[#090f1a]">
                          <td className="p-2 text-slate-300 font-semibold">Modeled Value / Share</td>
                          <td className="p-2 font-bold text-cyan-300">
                            ${valuation.reference_price.toFixed(2)} (Anchor)
                          </td>
                          <td className="p-2 font-bold text-rose-400">
                            {valuation.scenarios.conservative?.per_share_value !== undefined
                              ? `$${valuation.scenarios.conservative.per_share_value.toFixed(2)}`
                              : "N/A"}
                          </td>
                          <td className="p-2 font-bold text-white">
                            {valuation.scenarios.base?.per_share_value !== undefined
                              ? `$${valuation.scenarios.base.per_share_value.toFixed(2)}`
                              : "N/A"}
                          </td>
                          <td className="p-2 font-bold text-emerald-400">
                            {valuation.scenarios.optimistic?.per_share_value !== undefined
                              ? `$${valuation.scenarios.optimistic.per_share_value.toFixed(2)}`
                              : "N/A"}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            <div className="text-[10px] text-slate-500 pt-1 italic">
              <strong>Inputs &amp; Convention:</strong> {valuation.convention_summary}
            </div>
          </div>
        )}
      </section>

      {/* 4. Sourced Financial Facts & Accounting Bridge */}
      <section className="space-y-2.5" aria-label="Sourced Financial Facts">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-emerald-300 uppercase tracking-wider flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" aria-hidden="true"></span>
            Sourced Financial Facts &amp; Balance Sheet Bridge
          </h2>
          <span className="text-[10px] text-slate-400 font-mono">US-GAAP Authority</span>
        </div>

        <div className="bg-[#101726] border border-[#233149] rounded-xl p-4 sm:p-5 space-y-4 text-xs">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <div className="bg-[#0b1019] p-2.5 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 text-[10px] block">Reported Revenue</span>
              <span className="font-mono text-slate-100 font-semibold text-sm">
                {financial_profile.revenue_display}
              </span>
              <span className="text-[9px] text-slate-400 font-mono block truncate" title={financial_profile.revenue_period}>
                {financial_profile.revenue_period}
              </span>
            </div>

            <div className="bg-[#0b1019] p-2.5 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 text-[10px] block">Operating Margin</span>
              <span className="font-mono text-slate-100 font-semibold text-sm">
                {financial_profile.operating_margin_pct !== null
                  ? `${financial_profile.operating_margin_pct.toFixed(1)}%`
                  : "Unavailable"}
              </span>
              <span className="text-[9px] text-slate-400 font-mono block">Reported EBIT / Rev</span>
            </div>

            <div className="bg-[#0b1019] p-2.5 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 text-[10px] block">Total Cash Buffer</span>
              <span className="font-mono text-slate-100 font-semibold text-sm">
                {financial_profile.total_cash_and_investments !== null
                  ? `$${(financial_profile.total_cash_and_investments / 1e6).toFixed(1)}M`
                  : "Unavailable"}
              </span>
              <span className="text-[9px] text-slate-400 font-mono block">Cash + Investments</span>
            </div>

            <div className="bg-[#0b1019] p-2.5 rounded-lg border border-slate-800/80">
              <span className="text-slate-500 text-[10px] block">Total Financial Debt</span>
              <span className="font-mono text-slate-100 font-semibold text-sm">
                {financial_profile.debt_display}
              </span>
              <span className="text-[9px] text-slate-400 font-mono block">
                {financial_profile.is_zero_debt ? "Zero Funded Credit" : "Carrying Debt"}
              </span>
            </div>
          </div>

          {/* Sourced Capital Structure & Net Debt Bridge (Finding C) */}
          <div className="bg-[#0b1019] rounded-xl border border-slate-800/90 p-3.5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800">
              <span className="text-[11px] font-bold text-slate-200 uppercase tracking-wider">
                Capital Structure &amp; Net Debt Bridge Breakdown
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                Bridge: {financial_profile.net_debt_bridge !== null && financial_profile.net_debt_bridge !== undefined
                  ? `$${(financial_profile.net_debt_bridge / 1e6).toFixed(1)}M`
                  : "N/A"}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-xs font-mono">
              <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                <span className="text-slate-500 text-[10px] block">Cash &amp; Equivalents</span>
                <span className="text-slate-200 font-semibold">
                  {financial_profile.cash_and_equivalents !== null && financial_profile.cash_and_equivalents !== undefined
                    ? `$${(financial_profile.cash_and_equivalents / 1e6).toFixed(1)}M`
                    : "N/A"}
                </span>
              </div>

              <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                <span className="text-slate-500 text-[10px] block">Short-Term Investments</span>
                <span className="text-slate-200 font-semibold">
                  {financial_profile.short_term_investments !== null && financial_profile.short_term_investments !== undefined
                    ? `$${(financial_profile.short_term_investments / 1e6).toFixed(1)}M`
                    : "None / Excluded"}
                </span>
              </div>

              <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                <span className="text-slate-500 text-[10px] block">Restricted Cash</span>
                <span className="text-slate-200 font-semibold">
                  {financial_profile.restricted_cash !== null && financial_profile.restricted_cash !== undefined
                    ? `$${(financial_profile.restricted_cash / 1e6).toFixed(1)}M`
                    : "None / Excluded"}
                </span>
                {financial_profile.restriction_treatment && (
                  <span className="text-[9px] text-amber-400 block font-sans truncate" title={financial_profile.restriction_treatment}>
                    {financial_profile.restriction_treatment}
                  </span>
                )}
              </div>

              {financial_profile.digital_assets !== null && financial_profile.digital_assets !== undefined && (
                <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                  <span className="text-slate-500 text-[10px] block">Digital Assets (Bitcoin)</span>
                  <span className="text-slate-200 font-semibold">
                    ${(financial_profile.digital_assets / 1e6).toFixed(1)}M
                  </span>
                  <span className="text-[9px] text-cyan-400 block font-sans">
                    Carried at fair value / impairment basis
                  </span>
                </div>
              )}

              <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                <span className="text-slate-500 text-[10px] block">Total Financial Debt</span>
                <span className="text-slate-200 font-semibold">
                  {financial_profile.is_zero_debt
                    ? "$0.00"
                    : financial_profile.total_financial_debt !== null && financial_profile.total_financial_debt !== undefined
                    ? `$${(financial_profile.total_financial_debt / 1e6).toFixed(1)}M`
                    : financial_profile.debt_display}
                </span>
              </div>

              <div className="p-2 rounded bg-[#0f1624] border border-slate-800">
                <span className="text-slate-500 text-[10px] block">Operating Leases (ASC 842)</span>
                <span className="text-slate-200 font-semibold">
                  {financial_profile.operating_lease_liabilities_asc842 !== null && financial_profile.operating_lease_liabilities_asc842 !== undefined
                    ? `$${(financial_profile.operating_lease_liabilities_asc842 / 1e6).toFixed(1)}M`
                    : "None / N/A"}
                </span>
                {financial_profile.lease_convention_treatment && (
                  <span className="text-[9px] text-slate-400 block font-sans truncate" title={financial_profile.lease_convention_treatment}>
                    {financial_profile.lease_convention_treatment}
                  </span>
                )}
              </div>
            </div>

            {/* Accounting Bridge Notes */}
            {financial_profile.bridge_note && (
              <div className="p-2.5 rounded bg-[#101726] border border-slate-800 text-[11px] text-slate-300 font-sans leading-relaxed">
                <strong className="text-slate-200 font-semibold">Accounting Bridge Convention:</strong>{" "}
                {financial_profile.bridge_note}
              </div>
            )}

            {financial_profile.is_zero_debt && financial_profile.zero_debt_support && (
              <div className="p-2.5 rounded bg-emerald-950/30 border border-emerald-900/50 text-[11px] text-emerald-200 font-sans">
                <strong>Zero Debt Verification:</strong> {financial_profile.zero_debt_support}
              </div>
            )}
          </div>

          {/* Selected Share Basis */}
          {financial_profile.selected_share_basis && (
            <div className="text-[11px] text-slate-400 pt-1">
              <strong>Selected Share Basis:</strong>{" "}
              <span className="font-mono text-slate-200">
                {(financial_profile.selected_share_basis.value / 1e6).toFixed(2)}M shares
              </span>{" "}
              ({financial_profile.selected_share_basis.basis_type}) • Source:{" "}
              <span className="italic">{financial_profile.selected_share_basis.source}</span>
            </div>
          )}

          {/* Attributable SEC XBRL Filings */}
          <div className="pt-2.5 border-t border-slate-800 text-[11px] text-slate-400 flex flex-wrap items-center gap-2">
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
              <span className="text-slate-500 italic">No primary XBRL filing bindings</span>
            )}
          </div>
        </div>
      </section>

      {/* 5. Monitoring Conditions & Invalidation Triggers */}
      <section className="space-y-2.5" aria-label="Monitoring Conditions">
        <h2 className="text-xs font-semibold text-rose-300 uppercase tracking-wider flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-rose-400" aria-hidden="true"></span>
          Monitoring Conditions &amp; Thesis Invalidation
        </h2>

        <div className="bg-[#101726] border border-[#233149] rounded-xl p-4 sm:p-5 space-y-3 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* Observable Strengthening Conditions */}
            <div className="bg-emerald-950/20 border border-emerald-900/40 p-3 rounded-lg space-y-1">
              <h3 className="text-emerald-400 font-semibold text-xs flex items-center gap-1">
                <span>&#10003;</span>
                Observable Conditions That Would Support Entry:
              </h3>
              <p className="text-slate-300 text-xs leading-relaxed">
                {monitoring.price_watch_condition}
              </p>
            </div>

            {/* Invalidation Trigger */}
            <div className="bg-rose-950/20 border border-rose-900/40 p-3 rounded-lg space-y-1">
              <h3 className="text-rose-400 font-semibold text-xs flex items-center gap-1">
                <span>&#9888;</span>
                Thesis Invalidation Triggers:
              </h3>
              <p className="text-slate-300 text-xs leading-relaxed">
                {monitoring.observable_invalidation_trigger}
              </p>
            </div>
          </div>

          <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400 flex flex-wrap items-center justify-between gap-2">
            <div>
              <strong>Next Verification Action:</strong>{" "}
              <span className="text-slate-300">{monitoring.next_monitoring_trigger}</span>
            </div>
          </div>
        </div>
      </section>

      {/* 6. On-Demand Technical Radar Navigation */}
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
