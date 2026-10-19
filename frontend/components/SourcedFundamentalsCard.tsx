/**
 * frontend/components/SourcedFundamentalsCard.tsx
 *
 * Dedicated institutional view for inspecting SEC EDGAR-sourced company fundamentals,
 * derived cash flows, SBC dilution, balance sheet debt completeness, and evidence gaps.
 *
 * Enforces strict facts-only scope & quality gates:
 * - Pure factual presentation: zero DCF scenario models, fair values, or investment rankings.
 * - Accurate units: shares formatted as shares, dollars as dollars, percents as percents.
 * - Meaningful evidence statuses: reported XBRL vs derived formulas clearly distinguished.
 * - Dynamic, truthful aggregate debt completeness notices reflecting exact reported components.
 * - Missing-value states rendered with neutral slate styling, never false positive/negative colors.
 * - Direct links to primary SEC EDGAR accession index documents.
 */

import React from "react";
import Link from "next/link";
import {
  SourcedCompanySnapshot,
  getSourcedFundamentals,
} from "../lib/sourcedFundamentals";

interface SourcedFundamentalsCardProps {
  symbol: string;
}

function formatUSD(value: number | null | undefined, abbreviate = true): string {
  if (value === null || value === undefined) return "UNAVAILABLE";
  if (!abbreviate) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(value);
  }
  const abs = Math.abs(value);
  if (abs >= 1e9) {
    return `$${(value / 1e9).toFixed(2)}B`;
  }
  if (abs >= 1e6) {
    return `$${(value / 1e6).toFixed(1)}M`;
  }
  if (abs >= 1e3) {
    return `$${(value / 1e3).toFixed(0)}K`;
  }
  return `$${value.toFixed(2)}`;
}

function formatShares(value: number | null | undefined): string {
  if (value === null || value === undefined) return "UNAVAILABLE";
  const abs = Math.abs(value);
  if (abs >= 1e9) {
    return `${(value / 1e9).toFixed(2)}B shares`;
  }
  if (abs >= 1e6) {
    return `${(value / 1e6).toFixed(2)}M shares`;
  }
  if (abs >= 1e3) {
    return `${(value / 1e3).toFixed(1)}K shares`;
  }
  return `${value.toLocaleString("en-US")} shares`;
}

function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "UNAVAILABLE";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

function formatFactValue(value: number | null | undefined, units: string | null | undefined): string {
  if (value === null || value === undefined) return "UNAVAILABLE";
  const cleanUnits = (units || "USD").toLowerCase();
  if (cleanUnits === "shares") {
    return formatShares(value);
  }
  if (cleanUnits === "percent") {
    return formatPct(value);
  }
  if (cleanUnits === "usd/share") {
    return `$${value.toFixed(2)}/share`;
  }
  return formatUSD(value);
}

export default function SourcedFundamentalsCard({ symbol }: SourcedFundamentalsCardProps) {
  const cleanSym = symbol ? symbol.trim().toUpperCase() : "";
  const snapshot: SourcedCompanySnapshot | null = getSourcedFundamentals(cleanSym);

  if (!snapshot) {
    return (
      <section
        id="sourced-fundamentals-section"
        data-testid="sourced-fundamentals-card"
        className="bg-[#0b1019] p-5 rounded-2xl border border-[#1e293b] space-y-4"
      >
        <div className="flex items-center justify-between border-b border-[#162030] pb-3">
          <div className="flex items-center gap-2">
            <span className="text-base">📑</span>
            <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              SEC EDGAR Sourced Fundamentals & Evidence
            </h2>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 text-slate-400 border border-slate-700">
            UNAVAILABLE
          </span>
        </div>
        <p className="text-xs text-slate-400 font-sans leading-relaxed">
          Primary filing XBRL extraction is currently available for the 8 deep-dive pilot companies.
          Sourced financial evidence has not yet been cataloged for <strong className="text-white font-mono">{cleanSym || "Unknown"}</strong>.
        </p>
        <div className="pt-2 border-t border-[#162030] flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500 text-[11px]">Inspect cataloged pilot evidence:</span>
          <Link
            href="/stock/fix"
            className="px-2.5 py-1 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-cyan-300 border border-cyan-900/50 text-[11px] font-mono"
          >
            FIX (Comfort Systems)
          </Link>
          <Link
            href="/stock/duol"
            className="px-2.5 py-1 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-amber-300 border border-amber-900/50 text-[11px] font-mono"
          >
            DUOL (Duolingo)
          </Link>
          <Link
            href="/stock/ulta"
            className="px-2.5 py-1 rounded-lg bg-[#111722] hover:bg-[#1a2332] text-purple-300 border border-purple-900/50 text-[11px] font-mono"
          >
            ULTA (Ulta Beauty)
          </Link>
        </div>
      </section>
    );
  }

  const {
    company_identity: ident,
    derived_calculations: dm,
    research_evidence_notes: evidenceNotes,
    reported_facts: reportedFacts,
  } = snapshot;

  const revGrowth = dm.revenue_growth_yoy_pct?.value;
  const opMargin = dm.operating_margin_pct?.value;
  const cfoMinusCapex = dm.cfo_minus_capex_ttm?.value;
  const sbcPct = dm.sbc_to_revenue_pct?.value;
  const shortDebt = dm.short_term_debt?.value;
  const longDebt = dm.long_term_debt?.value;

  // Strict completeness check: must be explicitly non-null and defined
  const isDebtEvidenced = dm.total_financial_debt?.value !== null && dm.total_financial_debt?.value !== undefined;

  // Construct dynamic, evidence-responsive debt completeness warning
  let debtNoticeText = "";
  if (!isDebtEvidenced) {
    if (shortDebt !== null && shortDebt !== undefined && (longDebt === null || longDebt === undefined)) {
      debtNoticeText = `Short-term borrowings (${formatUSD(shortDebt)}) reported under credit facility; long-term debt concept is unavailable in primary XBRL taxonomy. Under aggregate completeness rules, unevidenced long-term debt is not assumed to be $0.00 without primary source confirmation.`;
    } else if (longDebt !== null && longDebt !== undefined && (shortDebt === null || shortDebt === undefined)) {
      debtNoticeText = `Long-term debt obligations (${formatUSD(longDebt)}) reported; short-term debt concept is unavailable in primary XBRL taxonomy. Complete debt structure requires primary footnote verification.`;
    } else {
      debtNoticeText = `Financial debt concepts (short-term, long-term, and combined debt) are unavailable in primary XBRL tags. Under aggregate completeness rules, unevidenced debt is not assumed to be $0.00.`;
    }
  }

  return (
    <section
      id="sourced-fundamentals-section"
      data-testid="sourced-fundamentals-card"
      className="bg-[#0b1019] p-5 sm:p-6 rounded-2xl border border-cyan-900/50 space-y-6"
    >
      {/* Header & SEC Metadata */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#1e293b] pb-4 gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-base">📑</span>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              SEC EDGAR Sourced Fundamentals & Evidence
            </h2>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
              SEC EDGAR Disclosures (Form 10-Q / 10-K)
            </span>
          </div>
          <p className="text-[11px] text-slate-400 font-sans mt-1">
            Grounded in SEC EDGAR filings through reported period ending{" "}
            <span className="text-slate-200 font-mono">{ident.latest_reported_period}</span> (Filed:{" "}
            {ident.latest_filing_date}). CIK: <span className="font-mono text-slate-300">{ident.cik}</span>.
          </p>
        </div>

        {ident.sec_edgar_company_url && (
          <a
            href={ident.sec_edgar_company_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#111722] hover:bg-[#1a2332] text-cyan-400 border border-cyan-800/60 rounded-xl text-xs font-mono transition-colors whitespace-nowrap self-start sm:self-auto"
          >
            <span>SEC EDGAR Filings ↗</span>
          </a>
        )}
      </div>

      {/* Grid: 4 Core Fundamental Pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1. Revenue & Growth */}
        <div className="bg-[#06090f] p-4 rounded-xl border border-[#162030] space-y-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">TTM Revenue</div>
          <div className="text-lg font-bold font-mono text-white">
            {formatUSD(dm.revenue_ttm?.value)}
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-[#111722]">
            <span className="text-slate-500 text-[11px]">YoY Growth:</span>
            <span
              className={`font-mono font-bold ${
                revGrowth === null || revGrowth === undefined
                  ? "text-slate-400"
                  : revGrowth >= 0
                  ? "text-emerald-400"
                  : "text-rose-400"
              }`}
            >
              {formatPct(revGrowth)}
            </span>
          </div>
          <div className="text-[10px] text-slate-500 font-sans">
            Prior TTM: {formatUSD(dm.revenue_prior_ttm?.value)}
          </div>
        </div>

        {/* 2. Profitability & Margins */}
        <div className="bg-[#06090f] p-4 rounded-xl border border-[#162030] space-y-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">TTM Operating Income</div>
          <div className="text-lg font-bold font-mono text-white">
            {formatUSD(dm.operating_income_ttm?.value)}
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-[#111722]">
            <span className="text-slate-500 text-[11px]">Operating Margin:</span>
            <span
              className={`font-mono font-bold ${
                opMargin === null || opMargin === undefined ? "text-slate-400" : "text-cyan-400"
              }`}
            >
              {formatPct(opMargin)}
            </span>
          </div>
          <div className="text-[10px] text-slate-500 font-sans">
            Net Income: {formatUSD(dm.net_income_ttm?.value)}
          </div>
        </div>

        {/* 3. Cash Flow Conversion */}
        <div className="bg-[#06090f] p-4 rounded-xl border border-[#162030] space-y-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">CFO Less Capex (TTM)</div>
          <div
            className={`text-lg font-bold font-mono ${
              cfoMinusCapex === null || cfoMinusCapex === undefined ? "text-slate-400" : "text-emerald-400"
            }`}
          >
            {formatUSD(cfoMinusCapex)}
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-[#111722]">
            <span className="text-slate-500 text-[11px]">CFO TTM:</span>
            <span className="font-mono text-slate-300">{formatUSD(dm.cfo_ttm?.value)}</span>
          </div>
          <div className="flex items-center justify-between text-[10px] text-slate-500">
            <span>Capex TTM:</span>
            <span className="font-mono text-rose-400">{formatUSD(dm.capex_ttm?.value)}</span>
          </div>
        </div>

        {/* 4. Dilution & SBC */}
        <div className="bg-[#06090f] p-4 rounded-xl border border-[#162030] space-y-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">SBC & Share Dilution</div>
          <div
            className={`text-lg font-bold font-mono ${
              dm.sbc_ttm?.value === null || dm.sbc_ttm?.value === undefined ? "text-slate-400" : "text-purple-400"
            }`}
          >
            {formatUSD(dm.sbc_ttm?.value)}
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-[#111722]">
            <span className="text-slate-500 text-[11px]">SBC / Revenue:</span>
            <span className="font-mono font-bold text-purple-300">
              {sbcPct !== null && sbcPct !== undefined ? `${sbcPct.toFixed(1)}%` : "UNAVAILABLE"}
            </span>
          </div>
          <div className="flex items-center justify-between text-[10px] text-slate-500">
            <span>1Y / 3Y Dilution:</span>
            <span className="font-mono text-slate-300">
              {formatPct(dm.share_dilution_1y_pct?.value)} / {formatPct(dm.share_dilution_3y_pct?.value)}
            </span>
          </div>
        </div>
      </div>

      {/* Balance Sheet & Debt Completeness Status */}
      <div className="bg-[#06090f] p-4 rounded-xl border border-[#162030] space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#111722] pb-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-white uppercase tracking-wider">
              Balance Sheet & Debt Structure
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                isDebtEvidenced
                  ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                  : "bg-amber-950 text-amber-300 border-amber-800"
              }`}
            >
              {isDebtEvidenced ? "TOTAL DEBT FULLY EVIDENCED" : "TOTAL DEBT UNAVAILABLE IN XBRL"}
            </span>
          </div>
          <div className="text-[11px] text-slate-400 font-sans">
            As of period ending <span className="font-mono text-slate-200">{ident.latest_reported_period}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div>
            <span className="text-[10px] text-slate-500 block uppercase">Cash & Liquid Reserves</span>
            <span className="font-mono font-bold text-slate-200 text-sm">
              {formatUSD(dm.cash_and_equivalents?.value)}
            </span>
          </div>
          <div>
            <span className="text-[10px] text-slate-500 block uppercase">Short-Term Debt</span>
            <span className="font-mono font-bold text-slate-200 text-sm">
              {formatUSD(dm.short_term_debt?.value)}
            </span>
          </div>
          <div>
            <span className="text-[10px] text-slate-500 block uppercase">Long-Term Debt</span>
            <span className="font-mono font-bold text-slate-200 text-sm">
              {formatUSD(dm.long_term_debt?.value)}
            </span>
          </div>
          <div>
            <span className="text-[10px] text-slate-500 block uppercase">Total Financial Debt</span>
            <span
              className={`font-mono font-bold text-sm ${
                isDebtEvidenced ? "text-slate-200" : "text-amber-400"
              }`}
            >
              {isDebtEvidenced ? formatUSD(dm.total_financial_debt?.value) : "UNAVAILABLE"}
            </span>
          </div>
        </div>

        {/* Operating Leases Note */}
        <div className="text-[11px] text-slate-400 font-sans pt-1 border-t border-[#111722] flex flex-wrap items-center justify-between gap-2">
          <span>
            Operating Lease Liabilities (ASC 842):{" "}
            <strong className="text-slate-200 font-mono">
              {formatUSD(dm.operating_lease_liabilities?.value)}
            </strong>{" "}
            (Contractual operating lease commitments treated as operating expenses rather than financial debt).
          </span>
          {dm.net_debt?.value !== null && dm.net_debt?.value !== undefined && (
            <span>
              Net Debt: <strong className="text-slate-200 font-mono">{formatUSD(dm.net_debt?.value)}</strong>
            </span>
          )}
        </div>

        {/* Aggregate Completeness Alert if Debt Unavailable */}
        {!isDebtEvidenced && (
          <div className="bg-amber-950/30 border border-amber-800/60 p-3 rounded-lg text-xs space-y-1">
            <strong className="text-amber-300 block font-sans">
              ⚠️ Invariant Notice: Financial Debt Incomplete in Primary XBRL Tags
            </strong>
            <p className="text-amber-200/80 text-[11px] font-sans leading-relaxed">
              {debtNoticeText}
            </p>
          </div>
        )}
      </div>

      {/* Primary SEC EDGAR Facts Table */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <span>🔍 Primary Filing SEC Disclosures (Sample)</span>
          </h3>
          <span className="text-[11px] text-slate-500 font-mono">
            {reportedFacts.length} metric items tracked
          </span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-[#1e293b]">
          <table className="w-full text-left text-xs font-sans">
            <thead className="bg-[#0e1624] text-[10px] font-mono text-slate-400 uppercase tracking-wider border-b border-[#1e293b]">
              <tr>
                <th className="py-2.5 px-3">Metric Concept / Basis</th>
                <th className="py-2.5 px-3 text-right">Disclosed / Calculated Value</th>
                <th className="py-2.5 px-3">Period / Duration</th>
                <th className="py-2.5 px-3">Source Classification</th>
                <th className="py-2.5 px-3">Primary SEC Archive Link</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#162030] font-mono text-[11px]">
              {reportedFacts.slice(0, 10).map((fact, idx) => (
                <tr key={idx} className="hover:bg-[#111722] transition-colors">
                  <td className="py-2 px-3 text-slate-200">
                    <span className="font-semibold block">{fact.metric_key}</span>
                    <span className="text-[10px] text-slate-500 font-mono">
                      {fact.concept || fact.formula || "Derived"}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-right font-bold text-white">
                    {formatFactValue(fact.value, fact.units)}
                  </td>
                  <td className="py-2 px-3 text-slate-400">
                    {fact.period_end || "Instant"}
                    <span className="text-[10px] text-slate-500 block">{fact.duration_type}</span>
                  </td>
                  <td className="py-2 px-3">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                        fact.fact_type === "PRIMARY_REPORTED"
                          ? "bg-cyan-950 text-cyan-300 border border-cyan-800"
                          : "bg-slate-900 text-slate-400 border border-slate-700"
                      }`}
                    >
                      {fact.fact_type === "PRIMARY_REPORTED" ? "REPORTED (XBRL)" : "DERIVED FORMULA"}
                    </span>
                  </td>
                  <td className="py-2 px-3">
                    {fact.source_url ? (
                      <a
                        href={fact.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-400 hover:underline"
                        title={fact.accession_number || "Direct SEC link"}
                      >
                        {fact.accession_number ? fact.accession_number.slice(0, 18) + "..." : "SEC Archive ↗"}
                      </a>
                    ) : (
                      <span className="text-slate-500 text-[10px]">Formula Derivation</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Evidence Gaps & Invalidation Framework (Facts only - No investment rankings or targets) */}
      <div className="bg-[#0b1320] p-4 rounded-xl border border-[#1e293b] space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#162030] pb-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-white uppercase tracking-wider">
              Research Evidence & Invalidation Framework
            </span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">
            Sufficiency: {evidenceNotes.evidence_sufficiency || "Monitored"}
          </span>
        </div>

        {evidenceNotes.remaining_evidence_gaps && evidenceNotes.remaining_evidence_gaps.length > 0 && (
          <div className="space-y-1">
            <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider block">
              ⚠️ Remaining Evidence Gaps:
            </span>
            <ul className="list-disc list-inside text-[11px] text-slate-300 font-sans space-y-0.5">
              {evidenceNotes.remaining_evidence_gaps.map((gap, i) => (
                <li key={i}>{gap}</li>
              ))}
            </ul>
          </div>
        )}

        {evidenceNotes.thesis_invalidation_conditions && evidenceNotes.thesis_invalidation_conditions.length > 0 && (
          <div className="space-y-1.5 pt-2 border-t border-[#162030]">
            <span className="text-[10px] font-bold text-rose-400 uppercase tracking-wider block">
              🛑 Explicit Invalidation Conditions to Track:
            </span>
            <ul className="list-disc list-inside text-[11px] text-slate-400 font-sans space-y-1">
              {evidenceNotes.thesis_invalidation_conditions.map((cond, i) => (
                <li key={i}>{cond}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}
