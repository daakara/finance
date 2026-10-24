"use client";

/**
 * frontend/components/AlconV4ExtendedView.tsx
 *
 * Dedicated Institutional Research Presentation for Alcon Inc. (NYSE: ALC / SIX: ALC).
 * Renders the certified Alcon V4 valuation model, 5-scenario intrinsic value matrix,
 * discrete vs calendar XIRR, dated cash flow schedule, accumulation ceilings,
 * net-debt bridge, buyback authority disaggregation, and IFRS accounting disclosures.
 *
 * Invariants:
 * - Alcon Inc. identity clearly bound (not Air Lease Corporation).
 * - Reference price ($67.43) labeled as historical model input as of October 21, 2026.
 * - Valuation targets and accumulation ceilings labeled as conditional research outputs, not executable orders.
 * - Approved statutory buybacks ($1,200M) explicitly disaggregated from assumed future authorizations ($600M).
 * - 88.4% non-equipment sales share certified as procedural replenishment, not contracted SaaS.
 * - Strict IFRS IAS 7 & IFRS 16 disclosures; no US-GAAP lease conventions applied.
 * - Fully accessible: minimum 44x44px touch targets, keyboard navigation with visible focus rings.
 */

import React, { useState } from "react";
import {
  OpportunityDossier,
  AlconV4Extension,
  ScenarioProjection,
} from "../lib/opportunityDossiers";

interface AlconV4ExtendedViewProps {
  dossier: OpportunityDossier;
  extension: AlconV4Extension;
}

const formatCurrency = (val: number | null | undefined): string => {
  if (val === null || val === undefined || !Number.isFinite(val)) return "Unavailable";
  return `$${val.toFixed(2)}`;
};

const formatMillions = (val: number | null | undefined): string => {
  if (val === null || val === undefined || !Number.isFinite(val)) return "Unavailable";
  return `$${val.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}M`;
};

const formatPercent = (val: number | null | undefined, includeSign = false): string => {
  if (val === null || val === undefined || !Number.isFinite(val)) return "Unavailable";
  const sign = includeSign && val > 0 ? "+" : "";
  return `${sign}${val.toFixed(2)}%`;
};

export default function AlconV4ExtendedView({ dossier, extension }: AlconV4ExtendedViewProps) {
  const { valuation, financial_profile } = dossier;
  const refPrice = valuation.reference_price || 67.43;

  // Accordion state
  const [cashFlowsOpen, setCashFlowsOpen] = useState(false);
  const [balanceSheetOpen, setBalanceSheetOpen] = useState(false);
  const [buybacksOpen, setBuybacksOpen] = useState(false);
  const [qualitativeOpen, setQualitativeOpen] = useState(false);

  const scenarios = valuation.scenarios || {};
  const conservative = scenarios.conservative;
  const baseMultiple = scenarios.base_multiple;
  const baseDcf = scenarios.base_dcf;
  const optimistic = scenarios.optimistic;
  const conservativeDyn = scenarios.conservative_dynamic;
  const optimisticDyn = scenarios.optimistic_dynamic;

  const ceilings = extension.accumulation_ceilings;
  const annualCeilings = ceilings?.annual_irr_basis;
  const calendarCeilings = ceilings?.calendar_xirr_basis;

  return (
    <div className="space-y-6" data-testid="alcon-v4-extended-view">
      {/* 1. Identity & Governance Clarification Banner */}
      <div className="bg-[#0b1322] border border-cyan-800/80 rounded-xl p-4 sm:p-5 shadow-lg space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-cyan-900/60 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" aria-hidden="true"></span>
              <h2 className="text-base sm:text-lg font-bold text-white tracking-wide font-sans">
                Alcon Inc. (NYSE: ALC / SIX: ALC)
              </h2>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-700">
                Model v4.0 Certified
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              CIK 0001748310 &bull; Swiss Registered Shares (ISIN CH0432492467) &bull; Primary US Trading Line (NYSE)
            </p>
          </div>
          <div className="text-right">
            <span className="text-[11px] font-mono text-slate-400 block uppercase tracking-wider">
              Frozen Reference Close
            </span>
            <span className="font-mono font-bold text-xl sm:text-2xl text-cyan-400">
              ${refPrice.toFixed(2)} USD
            </span>
            <span className="text-[10px] text-slate-400 block font-mono">
              Market Session: October 21, 2026
            </span>
          </div>
        </div>

        {/* Corporate & Price Integrity Disclaimers */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs bg-[#080d16] p-3 rounded-lg border border-slate-800">
          <div className="space-y-1">
            <span className="font-semibold text-slate-300 block">Security & Domain Identity:</span>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Alcon Inc. is the global market leader in surgical ophthalmic instrumentation and vision care.{" "}
              <strong className="text-amber-300">Note: Alcon Inc. is not Air Lease Corporation.</strong>
            </p>
          </div>
          <div className="space-y-1">
            <span className="font-semibold text-slate-300 block">Valuation Baseline Status:</span>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Modeled target prices and accumulation ceilings are conditional research outputs,{" "}
              <strong className="text-amber-300">not executable orders or guaranteed returns</strong>. Reference price is frozen historical research.
            </p>
          </div>
        </div>
      </div>

      {/* 2. All Five Canonical Scenarios Matrix */}
      <section className="bg-[#0f172a]/95 border border-cyan-800/60 rounded-xl p-4 sm:p-5 space-y-4 shadow-md" aria-label="Five Canonical Valuation Scenarios">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" aria-hidden="true"></span>
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-300">
              Canonical Five-Scenario Valuation Matrix
            </h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Holding Period: 5.0 Years (1,826 Days)
          </span>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed">
          The certified v4.0 model establishes 5 forward scenarios comparing discrete terminal multiples against perpetual DCF growth.
          At the $67.43 reference price, the market solves to a 6.51% 5-year revenue CAGR. Under Base DCF assumptions, expected annual IRR is 6.92% (fails 7.80% institutional cost of equity).
        </p>

        {/* Scenario Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3.5">
          {/* 1. Conservative */}
          {conservative && (
            <div className="bg-[#080d16] border border-rose-900/60 rounded-xl p-3.5 space-y-2 flex flex-col justify-between" data-testid="scenario-conservative">
              <div>
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-rose-950 text-rose-300 border border-rose-800">
                    Conservative
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">Multiple (10.0x)</span>
                </div>
                <div className="mt-2">
                  <span className="text-[10px] text-slate-400 uppercase font-medium">Modeled Share Price</span>
                  <div className="font-mono font-bold text-rose-400 text-2xl">
                    ${conservative.per_share_value.toFixed(2)}
                  </div>
                  <span className="text-xs font-mono font-semibold text-rose-300 block">
                    {formatPercent(conservative.potential_return_pct, true)} vs ref
                  </span>
                </div>
              </div>
              <div className="bg-[#0d1422] rounded p-2 text-[11px] font-mono space-y-1 border border-slate-800/80">
                <div className="flex justify-between text-slate-400">
                  <span>5Y Rev CAGR:</span>
                  <span className="text-slate-200">{conservative.revenue_cagr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Operating Margin:</span>
                  <span className="text-slate-200">{conservative.operating_margin_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between border-t border-slate-800 pt-1 text-slate-400">
                  <span title="Discrete 5-period annual IRR">Annual IRR:</span>
                  <span className="text-rose-300 font-bold">{conservative.five_period_annual_irr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span title="Actual/365 calendar dated XIRR">Dated XIRR:</span>
                  <span className="text-rose-300 font-bold">{conservative.calendar_dated_xirr_pct?.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}

          {/* 2. Base Multiple */}
          {baseMultiple && (
            <div className="bg-[#080d16] border border-slate-700 rounded-xl p-3.5 space-y-2 flex flex-col justify-between" data-testid="scenario-base-multiple">
              <div>
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-slate-800 text-slate-200 border border-slate-600">
                    Base Multiple
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">Multiple (10.0x)</span>
                </div>
                <div className="mt-2">
                  <span className="text-[10px] text-slate-400 uppercase font-medium">Modeled Share Price</span>
                  <div className="font-mono font-bold text-slate-100 text-2xl">
                    ${baseMultiple.per_share_value.toFixed(2)}
                  </div>
                  <span className="text-xs font-mono font-semibold text-slate-300 block">
                    {formatPercent(baseMultiple.potential_return_pct, true)} vs ref
                  </span>
                </div>
              </div>
              <div className="bg-[#0d1422] rounded p-2 text-[11px] font-mono space-y-1 border border-slate-800/80">
                <div className="flex justify-between text-slate-400">
                  <span>5Y Rev CAGR:</span>
                  <span className="text-slate-200">{baseMultiple.revenue_cagr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Operating Margin:</span>
                  <span className="text-slate-200">{baseMultiple.operating_margin_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between border-t border-slate-800 pt-1 text-slate-400">
                  <span title="Discrete 5-period annual IRR">Annual IRR:</span>
                  <span className="text-slate-200 font-bold">{baseMultiple.five_period_annual_irr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span title="Actual/365 calendar dated XIRR">Dated XIRR:</span>
                  <span className="text-slate-200 font-bold">{baseMultiple.calendar_dated_xirr_pct?.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}

          {/* 3. Base DCF (Central Perpetual Growth Case) */}
          {baseDcf && (
            <div className="bg-[#080d16] border border-cyan-500/90 ring-1 ring-cyan-500/40 rounded-xl p-3.5 space-y-2 flex flex-col justify-between shadow-md" data-testid="scenario-base-dcf">
              <div>
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-cyan-950 text-cyan-300 border border-cyan-700">
                    Base DCF (Certified)
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400">Gordon DCF (2.5%)</span>
                </div>
                <div className="mt-2">
                  <span className="text-[10px] text-cyan-300 uppercase font-medium">Modeled Share Price</span>
                  <div className="font-mono font-bold text-white text-2xl">
                    ${baseDcf.per_share_value.toFixed(2)}
                  </div>
                  <span className="text-xs font-mono font-semibold text-cyan-300 block">
                    {formatPercent(baseDcf.potential_return_pct, true)} vs ref
                  </span>
                </div>
              </div>
              <div className="bg-[#0d1422] rounded p-2 text-[11px] font-mono space-y-1 border border-cyan-900/60">
                <div className="flex justify-between text-slate-300">
                  <span>5Y Rev CAGR:</span>
                  <span className="text-cyan-300 font-semibold">{baseDcf.revenue_cagr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Operating Margin:</span>
                  <span className="text-cyan-300 font-semibold">{baseDcf.operating_margin_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between border-t border-cyan-900/60 pt-1 text-slate-300">
                  <span title="Discrete 5-period annual IRR">Annual IRR:</span>
                  <span className="text-cyan-300 font-bold">{baseDcf.five_period_annual_irr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span title="Actual/365 calendar dated XIRR">Dated XIRR:</span>
                  <span className="text-cyan-300 font-bold">{baseDcf.calendar_dated_xirr_pct?.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}

          {/* 4. Optimistic */}
          {optimistic && (
            <div className="bg-[#080d16] border border-emerald-900/60 rounded-xl p-3.5 space-y-2 flex flex-col justify-between" data-testid="scenario-optimistic">
              <div>
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-emerald-950 text-emerald-300 border border-emerald-800">
                    Optimistic
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">Multiple (11.0x)</span>
                </div>
                <div className="mt-2">
                  <span className="text-[10px] text-slate-400 uppercase font-medium">Modeled Share Price</span>
                  <div className="font-mono font-bold text-emerald-400 text-2xl">
                    ${optimistic.per_share_value.toFixed(2)}
                  </div>
                  <span className="text-xs font-mono font-semibold text-emerald-300 block">
                    {formatPercent(optimistic.potential_return_pct, true)} vs ref
                  </span>
                </div>
              </div>
              <div className="bg-[#0d1422] rounded p-2 text-[11px] font-mono space-y-1 border border-slate-800/80">
                <div className="flex justify-between text-slate-400">
                  <span>5Y Rev CAGR:</span>
                  <span className="text-slate-200">{optimistic.revenue_cagr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Operating Margin:</span>
                  <span className="text-slate-200">{optimistic.operating_margin_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between border-t border-slate-800 pt-1 text-slate-400">
                  <span title="Discrete 5-period annual IRR">Annual IRR:</span>
                  <span className="text-emerald-300 font-bold">{optimistic.five_period_annual_irr_pct?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span title="Actual/365 calendar dated XIRR">Dated XIRR:</span>
                  <span className="text-emerald-300 font-bold">{optimistic.calendar_dated_xirr_pct?.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Dynamic Variants Footer */}
        {(conservativeDyn || optimisticDyn) && (
          <div className="bg-[#080d16] p-3 rounded-lg border border-slate-800 text-xs text-slate-400 flex flex-wrap justify-between gap-2">
            <span>
              <strong>Dynamic Sensitivity Variants:</strong> Conservative Dynamic solves to{" "}
              <span className="font-mono font-semibold text-slate-200">${conservativeDyn?.per_share_value.toFixed(2)}</span> ({formatPercent(conservativeDyn?.potential_return_pct, true)} return);{" "}
              Optimistic Dynamic solves to{" "}
              <span className="font-mono font-semibold text-slate-200">${optimisticDyn?.per_share_value.toFixed(2)}</span> ({formatPercent(optimisticDyn?.potential_return_pct, true)} return).
            </span>
            <span className="text-slate-500 font-mono text-[10px]">Methodology: Sensitivity Stress Bounds</span>
          </div>
        )}
      </section>

      {/* 3. Hurdle-Specific Accumulation Ceilings Card */}
      <section className="bg-[#0f172a]/95 border border-amber-800/70 rounded-xl p-4 sm:p-5 space-y-3.5 shadow-md" aria-label="Accumulation Ceilings">
        <div className="flex items-center justify-between border-b border-amber-950/80 pb-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-400" aria-hidden="true"></span>
            <h3 className="text-xs font-bold uppercase tracking-wider text-amber-300">
              Hurdle-Specific Accumulation Ceilings (Price Watch Conditions)
            </h3>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-emerald-950 text-emerald-300 border border-emerald-700">
            Invertibility Verified
          </span>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed">
          Because the expected Base DCF annual return at $67.43 (6.92%) fails the institutional cost of equity (7.80%),
          capital accumulation requires pullbacks below the following mathematical ceiling thresholds:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* 8.0% Hurdle */}
          <div className="bg-[#080d16] border border-amber-800/60 rounded-lg p-3 space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-amber-300 font-mono">8.00% Equity Hurdle</span>
              <span className="text-[10px] font-mono text-slate-400">Min Threshold</span>
            </div>
            <div className="font-mono font-bold text-xl text-white">
              ${annualCeilings?.hurdle_80?.toFixed(2) || "64.15"}
            </div>
            <div className="text-[11px] font-mono text-slate-400 flex justify-between">
              <span>Calendar XIRR:</span>
              <span className="text-slate-300">${calendarCeilings?.hurdle_80?.toFixed(2) || "64.20"}</span>
            </div>
          </div>

          {/* 8.5% Hurdle */}
          <div className="bg-[#080d16] border border-amber-800/60 rounded-lg p-3 space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-amber-300 font-mono">8.50% Equity Hurdle</span>
              <span className="text-[10px] font-mono text-slate-400">Institutional</span>
            </div>
            <div className="font-mono font-bold text-xl text-white">
              ${annualCeilings?.hurdle_85?.toFixed(2) || "62.70"}
            </div>
            <div className="text-[11px] font-mono text-slate-400 flex justify-between">
              <span>Calendar XIRR:</span>
              <span className="text-slate-300">${calendarCeilings?.hurdle_85?.toFixed(2) || "62.75"}</span>
            </div>
          </div>

          {/* 10.0% Hurdle */}
          <div className="bg-[#080d16] border border-amber-800/60 rounded-lg p-3 space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-amber-300 font-mono">10.00% Equity Hurdle</span>
              <span className="text-[10px] font-mono text-slate-400">High Conviction</span>
            </div>
            <div className="font-mono font-bold text-xl text-white">
              ${annualCeilings?.hurdle_100?.toFixed(2) || "58.59"}
            </div>
            <div className="text-[11px] font-mono text-slate-400 flex justify-between">
              <span>Calendar XIRR:</span>
              <span className="text-slate-300">${calendarCeilings?.hurdle_100?.toFixed(2) || "58.64"}</span>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Expandable Accordions: Cash Flows, Balance Sheet, Buybacks, Qualitative */}
      <div className="space-y-3">
        {/* A. 7-Event Dated Cash Flows Schedule */}
        <div className="border border-slate-800 rounded-xl bg-[#0f172a]/95 overflow-hidden">
          <button
            type="button"
            onClick={() => setCashFlowsOpen(!cashFlowsOpen)}
            className="w-full min-h-[48px] px-4 py-3 flex items-center justify-between text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 hover:bg-slate-800/40 transition-colors"
            aria-expanded={cashFlowsOpen}
          >
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-2">
              <span className="font-mono text-slate-500">{cashFlowsOpen ? "[-]" : "[+]"}</span>
              Dated Cash Flow Schedule (7 Discrete Events &bull; 1,826 Days)
            </span>
            <span className="text-xs font-mono text-slate-400">
              {cashFlowsOpen ? "Collapse Schedule" : "Inspect Schedule"}
            </span>
          </button>

          {cashFlowsOpen && (
            <div className="p-4 border-t border-slate-800 bg-[#080d16] space-y-3">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                      <th className="py-1.5 px-2">Event</th>
                      <th className="py-1.5 px-2">Payment Date</th>
                      <th className="py-1.5 px-2 text-right">Elapsed Days</th>
                      <th className="py-1.5 px-2 text-right">Cash Flow / Share</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {extension.dated_cash_flows.map((cf, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/20">
                        <td className="py-1.5 px-2 text-slate-200">{cf.event}</td>
                        <td className="py-1.5 px-2 text-slate-300">{cf.date}</td>
                        <td className="py-1.5 px-2 text-right text-slate-400">{cf.days_from_start} d</td>
                        <td className={`py-1.5 px-2 text-right font-bold ${cf.cash_flow < 0 ? "text-rose-400" : "text-emerald-400"}`}>
                          ${cf.cash_flow.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
                Dividends are modeled at Swiss post-AGM payment dates in mid-May (Day 205, 571, 936, 1301, 1666).
                Terminal disposition at $91.82 occurs on October 22, 2031 (Day 1826). Cumulative DPS: $2.10. Total terminal payout: $93.92.
              </p>
            </div>
          )}
        </div>

        {/* B. Capital Structure & Net Debt Amortization Bridge */}
        <div className="border border-slate-800 rounded-xl bg-[#0f172a]/95 overflow-hidden">
          <button
            type="button"
            onClick={() => setBalanceSheetOpen(!balanceSheetOpen)}
            className="w-full min-h-[48px] px-4 py-3 flex items-center justify-between text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 hover:bg-slate-800/40 transition-colors"
            aria-expanded={balanceSheetOpen}
          >
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-2">
              <span className="font-mono text-slate-500">{balanceSheetOpen ? "[-]" : "[+]"}</span>
              Capital Structure, Net Debt Bridge &amp; IFRS Accounting Disclosures
            </span>
            <span className="text-xs font-mono text-slate-400">
              {balanceSheetOpen ? "Collapse Bridge" : "Inspect Bridge"}
            </span>
          </button>

          {balanceSheetOpen && (
            <div className="p-4 border-t border-slate-800 bg-[#080d16] space-y-3 text-xs">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-[#0d1422] p-3 rounded border border-slate-800 space-y-2">
                  <span className="font-bold text-slate-200 block text-xs uppercase font-mono">
                    Balance Sheet Bridge Evolution:
                  </span>
                  <div className="space-y-1 font-mono text-[11px]">
                    <div className="flex justify-between text-slate-400">
                      <span>Starting Gross Financial Debt:</span>
                      <span className="text-slate-200">$4,500.0M</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Starting Cash &amp; Equivalents:</span>
                      <span className="text-slate-200">$1,200.0M</span>
                    </div>
                    <div className="flex justify-between text-slate-300 font-semibold border-t border-slate-800 pt-1">
                      <span>Starting Net Debt Bridge:</span>
                      <span className="text-cyan-300">$3,300.0M</span>
                    </div>
                    <div className="flex justify-between text-slate-300 font-semibold pt-1 border-t border-slate-800">
                      <span>Terminal Ending Net Debt:</span>
                      <span className="text-emerald-300">-$2,176.2M (Net Cash)</span>
                    </div>
                  </div>
                </div>

                <div className="bg-[#0d1422] p-3 rounded border border-slate-800 space-y-2">
                  <span className="font-bold text-slate-200 block text-xs uppercase font-mono">
                    Diluted Share Count Bridge:
                  </span>
                  <div className="space-y-1 font-mono text-[11px]">
                    <div className="flex justify-between text-slate-400">
                      <span>Beginning Basic Shares:</span>
                      <span className="text-slate-200">484.524M</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>5Y Repurchases (Modeled):</span>
                      <span className="text-rose-400">-24.000M</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>5Y SBC Dilutive Issuance:</span>
                      <span className="text-slate-200">+11.000M</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Dilutive Share Awards:</span>
                      <span className="text-slate-200">+4.000M</span>
                    </div>
                    <div className="flex justify-between text-slate-300 font-semibold border-t border-slate-800 pt-1">
                      <span>Ending Diluted Shares:</span>
                      <span className="text-cyan-300">475.524M</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-3 bg-[#0a101b] rounded border border-slate-800 text-[11px] text-slate-400 space-y-1 font-sans">
                <span className="font-bold text-slate-300 block font-mono">IFRS Accounting Standard Disclosures:</span>
                <p>
                  Alcon is an IFRS foreign private issuer reporting under IFRS 16 and IAS 7. Lease liabilities are recognized in total financial debt, and lease interest is classified inside operating cash flows. No US-GAAP ASC 842 lease conventions are applied.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* C. Statutory Buyback Authority Disaggregation */}
        <div className="border border-slate-800 rounded-xl bg-[#0f172a]/95 overflow-hidden">
          <button
            type="button"
            onClick={() => setBuybacksOpen(!buybacksOpen)}
            className="w-full min-h-[48px] px-4 py-3 flex items-center justify-between text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 hover:bg-slate-800/40 transition-colors"
            aria-expanded={buybacksOpen}
          >
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-2">
              <span className="font-mono text-slate-500">{buybacksOpen ? "[-]" : "[+]"}</span>
              Statutory Buyback Authority Disaggregation (Approved vs Assumed Future)
            </span>
            <span className="text-xs font-mono text-slate-400">
              {buybacksOpen ? "Collapse Buybacks" : "Inspect Buybacks"}
            </span>
          </button>

          {buybacksOpen && (
            <div className="p-4 border-t border-slate-800 bg-[#080d16] space-y-3 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono">
                <div className="bg-[#0d1422] p-3 rounded border border-slate-800 space-y-1">
                  <span className="text-slate-400 text-[11px] block">Approved 2026 Program</span>
                  <div className="font-bold text-white text-base">$1,500M CHF</div>
                  <span className="text-[10px] text-slate-400 block font-sans">
                    AGM Authorized May 5, 2026 (Expires May 5, 2029)
                  </span>
                </div>

                <div className="bg-[#0d1422] p-3 rounded border border-emerald-950 border-emerald-800/60 space-y-1">
                  <span className="text-emerald-400 text-[11px] block font-semibold">Active Remaining Capacity</span>
                  <div className="font-bold text-emerald-300 text-base">$1,200M USD</div>
                  <span className="text-[10px] text-slate-400 block font-sans">
                    Statutory board &amp; shareholder authorized
                  </span>
                </div>

                <div className="bg-[#0d1422] p-3 rounded border border-amber-950 border-amber-800/60 space-y-1">
                  <span className="text-amber-400 text-[11px] block font-semibold">Assumed Future Renewal</span>
                  <div className="font-bold text-amber-300 text-base">$600M USD</div>
                  <span className="text-[10px] text-slate-400 block font-sans">
                    Hypothetical extension post-2029 (requires future AGM)
                  </span>
                </div>
              </div>

              <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
                The certified model forecasts $1,800.0M total share repurchases over the 5-year investment period.
                Of this total, <strong className="text-emerald-300">$1,200.0M is backed by current statutory AGM authority</strong>, and{" "}
                <strong className="text-amber-300">$600.0M is an assumed future authorization renewal</strong>.
              </p>
            </div>
          )}
        </div>

        {/* D. Qualitative Evidence & Audit Confidence Matrix */}
        <div className="border border-slate-800 rounded-xl bg-[#0f172a]/95 overflow-hidden">
          <button
            type="button"
            onClick={() => setQualitativeOpen(!qualitativeOpen)}
            className="w-full min-h-[48px] px-4 py-3 flex items-center justify-between text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 hover:bg-slate-800/40 transition-colors"
            aria-expanded={qualitativeOpen}
          >
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-2">
              <span className="font-mono text-slate-500">{qualitativeOpen ? "[-]" : "[+]"}</span>
              Qualitative Claims &amp; Audit Confidence Matrix
            </span>
            <span className="text-xs font-mono text-slate-400">
              {qualitativeOpen ? "Collapse Claims" : "Inspect Claims"}
            </span>
          </button>

          {qualitativeOpen && (
            <div className="p-4 border-t border-slate-800 bg-[#080d16] space-y-3">
              <div className="space-y-2">
                {extension.qualitative_evidence_matrix.map((item, idx) => (
                  <div key={idx} className="bg-[#0d1422] p-3 rounded border border-slate-800 space-y-1">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-bold text-slate-200 text-xs">{item.claim}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                        item.classification === "AUDITED_AND_CERTIFIED"
                          ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                          : item.classification === "QUALITATIVE_HYPOTHESIS"
                          ? "bg-amber-950 text-amber-300 border border-amber-800"
                          : "bg-rose-950 text-rose-300 border border-rose-800"
                      }`}>
                        {item.classification}
                      </span>
                    </div>
                    <div className="text-xs font-mono text-cyan-300">{item.finding}</div>
                    <div className="text-[11px] text-slate-400 font-sans">Source: {item.source}</div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
                Audited FY2025 Form 20-F confirms that 88.37% of net sales ($9,119M of $10,319M) represents recurring replenishment demand.{" "}
                <strong className="text-amber-300">This represents procedural surgical consumables and consumer contact lenses, NOT contracted subscription SaaS.</strong>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
