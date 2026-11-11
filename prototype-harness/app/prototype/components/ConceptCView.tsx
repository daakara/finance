"use client";

import React, { useState } from "react";
import { ScenarioFixture } from "@/lib/prototype/scenarioLoader";

interface ConceptViewProps {
  fixture: ScenarioFixture;
  onInteraction: (eventType: string, details: any) => void;
  onOrderAction: () => void;
  copyFeedback: string | null;
}

export function ConceptCView({ fixture, onInteraction, onOrderAction, copyFeedback }: ConceptViewProps) {
  const [openMemoItem, setOpenMemoItem] = useState<number | null>(2); // Question 2 (Bear Case) open by default for epistemic priority
  const [techConsoleOpen, setTechConsoleOpen] = useState<boolean>(false);
  const [budgetInput, setBudgetInput] = useState<number>(1000);

  const spot = fixture.market_data.spot_price;
  const plannedEntry = fixture.technical_context.planned_entry_pivot;
  const hasPlannedEntry = plannedEntry !== undefined && plannedEntry !== null && plannedEntry > 0;
  const stopFloor = fixture.technical_context.structural_invalidation_floor || 0;

  // Sizing requires an authorized entry pivot to prevent inferring entry from spot
  const riskPerShare = hasPlannedEntry && stopFloor > 0 && plannedEntry > stopFloor ? plannedEntry - stopFloor : null;
  const shares = riskPerShare !== null && riskPerShare > 0 ? Math.floor(budgetInput / riskPerShare) : null;
  const capitalCommitted = shares !== null && plannedEntry !== undefined && plannedEntry !== null ? shares * plannedEntry : null;

  const toggleMemo = (itemIndex: number) => {
    const next = openMemoItem === itemIndex ? null : itemIndex;
    setOpenMemoItem(next);
    onInteraction("MEMO_ITEM_TOGGLED", { itemIndex, open: next === itemIndex });
  };

  const toggleTechConsole = () => {
    const next = !techConsoleOpen;
    setTechConsoleOpen(next);
    onInteraction("TECH_CONSOLE_TOGGLED", { open: next });
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 shadow-xl flex flex-col gap-6">
      {/* Workbench Header */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold bg-sky-950/80 text-sky-400 border border-sky-800/60 px-2 py-0.5 rounded">
              EVIDENCE-FIRST WORKSPACE
            </span>
            <span className="text-2xl font-bold font-mono text-slate-100">{fixture.ticker}</span>
            <span className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono">
              {fixture.primary_exchange}
            </span>
            <span className="text-sm text-slate-300">{fixture.company_name}</span>
          </div>
          <div className="text-xs text-slate-400 mt-1 font-mono flex items-center gap-3">
            <span>SPOT: <strong className="text-slate-200">${spot.toFixed(2)}</strong></span>
            <span>TAPE: <strong className={fixture.market_data.is_stale ? "text-rose-400" : "text-emerald-400"}>
              {fixture.market_data.is_stale ? "DELAYED [⚠]" : "VERIFIED [✓]"}
            </strong></span>
            <span>AS OF: <strong>{fixture.market_data.quote_timestamp}</strong></span>
          </div>
        </div>

        <div className="text-right font-mono text-xs">
          <span className="text-slate-400">PROVENANCE: </span>
          <span className="text-slate-300">{fixture.market_data_provenance}</span>
        </div>
      </div>

      {/* Epistemic Orientation Alert */}
      <div className="bg-slate-950 border border-slate-800 rounded p-3 text-xs text-slate-300 flex items-center justify-between font-mono">
        <div>
          <strong className="text-sky-400">RESEARCH MANDATE: </strong>
          Engage with primary SEC disclosures, valuation bounds, and dissenting counter-arguments before examining technical timing.
        </div>
        <div className="text-slate-400 text-[11px]">
          Authority: ARX Closure Model v2
        </div>
      </div>

      {/* 5-Question Central Memo Accordion */}
      <div className="flex flex-col gap-3">
        <h3 className="text-xs font-semibold text-slate-400 font-mono tracking-wider uppercase">
          5-Question Investment Memo
        </h3>

        {/* Q1: Business Model & Economic Moat */}
        <div className="border border-slate-800 rounded bg-slate-950 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleMemo(1)}
            className="w-full p-3 text-left text-xs font-mono font-semibold flex items-center justify-between text-slate-200 hover:bg-slate-900"
          >
            <span className="flex items-center gap-2">
              <span className="text-sky-400">1.</span>
              <span>Business Model, Revenue Durability & Balance Sheet</span>
            </span>
            <span>{openMemoItem === 1 ? "▲" : "▼"}</span>
          </button>
          {openMemoItem === 1 && (
            <div className="p-4 border-t border-slate-800 text-xs font-mono text-slate-300 flex flex-col gap-3">
              <p className="text-slate-300 leading-relaxed">
                {fixture.research_dossier?.core_thesis || "Company operates with high operational leverage and market expansion tailwinds."}
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 border-t border-slate-800/80">
                <div className="bg-slate-900 p-2.5 rounded">
                  <div className="text-[10px] text-slate-400">YoY REVENUE GROWTH</div>
                  <div className="text-sm font-bold text-emerald-400 mt-1">
                    {fixture.fundamental_context ? `+${fixture.fundamental_context.revenue_growth_yoy_pct}%` : "N/A"}
                  </div>
                </div>
                <div className="bg-slate-900 p-2.5 rounded">
                  <div className="text-[10px] text-slate-400">NET CASH BALANCE SHEET</div>
                  <div className="text-sm font-bold text-slate-200 mt-1">
                    {fixture.fundamental_context?.net_cash_balance_sheet ? "YES (Zero Debt)" : "LEVERAGED"}
                  </div>
                </div>
                <div className="bg-slate-900 p-2.5 rounded">
                  <div className="text-[10px] text-slate-400">SBC % REVENUE</div>
                  <div className="text-sm font-bold text-amber-400 mt-1">
                    {fixture.fundamental_context ? `${fixture.fundamental_context.stock_based_compensation_pct_rev}%` : "N/A"}
                  </div>
                </div>
                <div className="bg-slate-900 p-2.5 rounded">
                  <div className="text-[10px] text-slate-400">FORWARD P/E</div>
                  <div className="text-sm font-bold text-slate-200 mt-1">
                    {fixture.fundamental_context ? `${fixture.fundamental_context.forward_pe}x` : "N/A"}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Q2: Dissenting Counter-Arguments & Bear Case (First-Class Visual Priority) */}
        <div className="border border-rose-900/40 rounded bg-slate-950 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleMemo(2)}
            className="w-full p-3 text-left text-xs font-mono font-semibold flex items-center justify-between text-rose-300 bg-rose-950/20 hover:bg-rose-950/40"
          >
            <span className="flex items-center gap-2">
              <span className="text-rose-400 font-bold">2.</span>
              <span>Dissenting Counter-Arguments & Bear Case [PRIORITY AUDIT]</span>
            </span>
            <span>{openMemoItem === 2 ? "▲" : "▼"}</span>
          </button>
          {openMemoItem === 2 && (
            <div className="p-4 border-t border-rose-900/40 text-xs font-mono text-rose-200 bg-rose-950/10 flex flex-col gap-3">
              <div className="bg-rose-950/60 border border-rose-800/60 p-3 rounded leading-relaxed">
                <div className="font-bold text-rose-300 mb-1">MATERIAL BEARISH DISSENT:</div>
                {fixture.research_dossier?.bearish_dissent || "Commoditization pressure and multiple compression risk under higher terminal discount rates."}
              </div>
              {fixture.heuristic_context?.corporate_event_alert && (
                <div className="bg-amber-950/50 border border-amber-800/60 p-2.5 rounded text-amber-200">
                  <strong>CORPORATE EVENT ALERT:</strong> {fixture.heuristic_context.corporate_event_alert}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Q3: Valuation Bounds & Implied Expectations */}
        <div className="border border-slate-800 rounded bg-slate-950 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleMemo(3)}
            className="w-full p-3 text-left text-xs font-mono font-semibold flex items-center justify-between text-slate-200 hover:bg-slate-900"
          >
            <span className="flex items-center gap-2">
              <span className="text-sky-400">3.</span>
              <span>Reverse-DCF Bounds & Implied Market Expectations</span>
            </span>
            <span>{openMemoItem === 3 ? "▲" : "▼"}</span>
          </button>
          {openMemoItem === 3 && (
            <div className="p-4 border-t border-slate-800 text-xs font-mono text-slate-300 flex flex-col gap-2">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Priced-in Terminal Growth Rate:</span>
                <span className="font-bold text-slate-200">3.50%</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Required 5-Yr FCF CAGR at Current Price:</span>
                <span className="font-bold text-emerald-400">22.4%</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Downside Fair Value under Stagnation Scenario:</span>
                <span className="font-bold text-rose-400">${(spot * 0.72).toFixed(2)} (-28.0%)</span>
              </div>
            </div>
          )}
        </div>

        {/* Q4: Capital Structure & Governance */}
        <div className="border border-slate-800 rounded bg-slate-950 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleMemo(4)}
            className="w-full p-3 text-left text-xs font-mono font-semibold flex items-center justify-between text-slate-200 hover:bg-slate-900"
          >
            <span className="flex items-center gap-2">
              <span className="text-sky-400">4.</span>
              <span>Capital Allocation Discipline & Insider Ownership</span>
            </span>
            <span>{openMemoItem === 4 ? "▲" : "▼"}</span>
          </button>
          {openMemoItem === 4 && (
            <div className="p-4 border-t border-slate-800 text-xs font-mono text-slate-300 flex flex-col gap-2">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Dilution Pace (Annual Share Creep):</span>
                <span>{fixture.fundamental_context ? `${(fixture.fundamental_context.stock_based_compensation_pct_rev * 0.15).toFixed(1)}%` : "1.2%"}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">SEC EDGAR Filing Accession:</span>
                <span className="text-slate-400">{fixture.provenance_reference}</span>
              </div>
            </div>
          )}
        </div>

        {/* Q5: Observable Verification Milestones & Invalidation Triggers */}
        <div className="border border-slate-800 rounded bg-slate-950 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleMemo(5)}
            className="w-full p-3 text-left text-xs font-mono font-semibold flex items-center justify-between text-slate-200 hover:bg-slate-900"
          >
            <span className="flex items-center gap-2">
              <span className="text-sky-400">5.</span>
              <span>Empirical Invalidation Triggers & Milestones</span>
            </span>
            <span>{openMemoItem === 5 ? "▲" : "▼"}</span>
          </button>
          {openMemoItem === 5 && (
            <div className="p-4 border-t border-slate-800 text-xs font-mono text-slate-300 flex flex-col gap-2">
              <div className="flex justify-between py-1 border-b border-slate-800">
                <span className="text-slate-400">Structural Invalidation Price Floor:</span>
                <span className="font-bold text-rose-400">${stopFloor > 0 ? stopFloor.toFixed(2) : "N/A"}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Actionability Threshold:</span>
                <span>{fixture.technical_context.is_actionable ? "Active setup confirmed" : "Setup suppressed / waiting"}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Subordinated Technical Timing Console */}
      <div className="border border-slate-800 rounded bg-slate-950 overflow-hidden">
        <button
          type="button"
          onClick={toggleTechConsole}
          className="w-full p-3 text-left text-xs font-mono font-semibold flex items-center justify-between text-slate-400 hover:text-slate-200 bg-slate-900/60"
        >
          <span>[Subordinated Technical Timing & Execution Console]</span>
          <span>{techConsoleOpen ? "▲" : "▼"}</span>
        </button>

        {techConsoleOpen && (
          <div className="p-4 border-t border-slate-800 flex flex-col gap-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
              <div className="bg-slate-900 p-3 rounded">
                <div className="text-slate-400">Planned Entry:</div>
                <div className="text-sm font-bold text-slate-100 mt-1">
                  {hasPlannedEntry ? `$${plannedEntry.toFixed(2)}` : (
                    <span className="text-amber-400 text-xs font-semibold">UNAVAILABLE (No Technical Entry Pivot)</span>
                  )}
                </div>
              </div>
              <div className="bg-slate-900 p-3 rounded">
                <div className="text-slate-400">Invalidation Floor:</div>
                <div className="text-sm font-bold text-rose-400 mt-1">
                  ${stopFloor > 0 ? stopFloor.toFixed(2) : "N/A"}
                </div>
              </div>
              <div className="bg-slate-900 p-3 rounded">
                <div className="text-slate-400">Take Profit 1:</div>
                <div className="text-sm font-bold text-emerald-400 mt-1">
                  ${fixture.technical_context.take_profit_1?.toFixed(2) || "N/A"}
                </div>
              </div>
            </div>

            {/* Governed Position Sizing & Order Action */}
            <div className="bg-slate-900 p-4 rounded border border-slate-800 flex flex-col gap-3">
              <div className="flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
                <div>
                  <label htmlFor="budgetC" className="text-slate-400 block mb-1">Risk Budget ($):</label>
                  <input
                    id="budgetC"
                    type="number"
                    value={budgetInput}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value) || 0;
                      setBudgetInput(val);
                      onInteraction("INPUT_CHANGED", { budgetInput: val });
                    }}
                    disabled={!hasPlannedEntry}
                    className={`w-28 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-100 font-bold ${!hasPlannedEntry ? 'opacity-50 cursor-not-allowed' : ''}`}
                  />
                </div>
                <div>
                  <span className="text-slate-400 block mb-1">Risk Per Share:</span>
                  {riskPerShare !== null ? (
                    <span className="font-bold text-rose-400">${riskPerShare.toFixed(2)}</span>
                  ) : (
                    <span className="font-bold text-slate-500">UNAVAILABLE</span>
                  )}
                </div>
                <div>
                  <span className="text-slate-400 block mb-1">Calculated Shares:</span>
                  {shares !== null ? (
                    <span className="font-bold text-emerald-400 text-sm">{shares} Shares</span>
                  ) : (
                    <span className="font-bold text-amber-400 text-xs">UNAVAILABLE (Requires Entry Pivot)</span>
                  )}
                </div>
                <div>
                  <span className="text-slate-400 block mb-1">Committed Capital:</span>
                  {capitalCommitted !== null ? (
                    <span className="font-bold text-slate-200 text-sm">${capitalCommitted.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                  ) : (
                    <span className="font-bold text-slate-500">N/A</span>
                  )}
                </div>
              </div>

              <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
                <div className="text-[11px] text-slate-500 font-mono">
                  {hasPlannedEntry
                    ? "Offline order string copied to clipboard. Zero broker routing."
                    : "Technical entry pivot not defined in fundamental underwriting fixture. Direct order drafting suppressed."}
                </div>
                <div className="flex items-center gap-3">
                  {copyFeedback && (
                    <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-950/80 border border-emerald-800/60 px-2.5 py-1 rounded">
                      {copyFeedback}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={hasPlannedEntry ? onOrderAction : undefined}
                    disabled={!hasPlannedEntry}
                    className={`font-mono font-bold text-xs py-2 px-4 rounded shadow transition-colors ${
                      hasPlannedEntry
                        ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                        : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                    }`}
                  >
                    {hasPlannedEntry
                      ? (fixture.action_contract?.button_text || `Copy Order: ${shares} Shares ${fixture.ticker}`)
                      : "Order Action Unavailable: Missing Entry Pivot"}
                  </button>
                </div>
              </div>
            </div>

          </div>
        )}
      </div>

    </div>
  );
}
