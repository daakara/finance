"use client";

import React, { useState } from "react";
import { ScenarioFixture } from "@/lib/prototype/scenarioLoader";

interface ConceptViewProps {
  fixture: ScenarioFixture;
  onInteraction: (eventType: string, details: any) => void;
  onOrderAction: () => void;
  copyFeedback: string | null;
}

export function ConceptBView({ fixture, onInteraction, onOrderAction, copyFeedback }: ConceptViewProps) {
  const [budgetInput, setBudgetInput] = useState<number>(1000);
  const [expandedSection, setExpandedSection] = useState<"NONE" | "TECHNICAL_PROOF" | "DISSENT" | "GOVERNANCE">("TECHNICAL_PROOF");

  const spot = fixture.market_data.spot_price;
  const entryPivot = fixture.technical_context.planned_entry_pivot || spot;
  const breakoutTrigger = fixture.technical_context.breakout_pivot_trigger || entryPivot;
  const stopFloor = fixture.technical_context.structural_invalidation_floor || 0;

  // Canonical sizing strictly uses the structural invalidation floor across all concepts
  const riskPerShare = stopFloor > 0 && spot > stopFloor ? spot - stopFloor : 0;
  const shares = riskPerShare > 0 ? Math.floor(budgetInput / riskPerShare) : 0;
  const capitalCommitted = shares * spot;

  const tp1 = fixture.technical_context.take_profit_1 || 0;
  const rrRatio = riskPerShare > 0 && tp1 > spot ? ((tp1 - spot) / riskPerShare).toFixed(2) : "N/A";

  const isStale = fixture.market_data.is_stale;
  const isActionable = fixture.technical_context.is_actionable;
  const readiness = fixture.technical_context.execution_readiness || (isActionable ? "ACTIONABLE_SETUP" : "SETUP_PAUSED");
  const badge = fixture.technical_context.radar_screening_badge;

  const toggleSection = (section: "TECHNICAL_PROOF" | "DISSENT" | "GOVERNANCE") => {
    const next = expandedSection === section ? "NONE" : section;
    setExpandedSection(next);
    onInteraction("SECTION_TOGGLED", { section, expanded: next === section });
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 shadow-xl flex flex-col gap-6">
      {/* Chrome Header */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold font-mono text-emerald-400">{fixture.ticker}</span>
          <span className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono">
            {fixture.primary_exchange}
          </span>
          <span className="text-sm text-slate-300 font-medium">{fixture.company_name}</span>
          {badge && (
            <span className="text-xs font-mono font-bold bg-amber-950/80 text-amber-300 border border-amber-800/60 px-2 py-0.5 rounded">
              {badge}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div className={`px-2.5 py-1 rounded font-mono font-bold flex items-center gap-1.5 ${
            isStale ? "bg-rose-950/70 text-rose-400 border border-rose-800/50" : "bg-emerald-950/70 text-emerald-400 border border-emerald-800/50"
          }`}>
            <span>●</span>
            <span>{isStale ? "STALE / DELAYED DATA [⚠]" : "LIVE TAPE [✓]"}</span>
          </div>
          <span className="text-slate-400 font-mono text-[11px]">
            {fixture.market_data.display_label}
          </span>
        </div>
      </div>

      {/* Asymmetric Split: 40% Verdict Hero | 60% Execution Context */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Left Column: Asymmetric Verdict Hero (40% / 5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-5 flex flex-col gap-4 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-teal-500 to-sky-500" />

            <div>
              <div className="text-[11px] font-mono tracking-wider text-slate-400 uppercase">Decision Posture & Verdict</div>
              <div className="flex items-center gap-2 mt-1">
                <span className={`text-xl font-bold font-mono px-3 py-1 rounded ${
                  isActionable && !isStale && !badge
                    ? "bg-emerald-900/80 text-emerald-200 border border-emerald-700/60"
                    : "bg-amber-950/80 text-amber-300 border border-amber-800/60"
                }`}>
                  {readiness}
                </span>
              </div>
            </div>

            {fixture.technical_context.reason_suppressed && (
              <div className="bg-rose-950/50 border border-rose-900/60 rounded p-2.5 text-xs text-rose-300 font-mono">
                <strong>SUPPRESSION REASON:</strong> {fixture.technical_context.reason_suppressed}
              </div>
            )}

            {/* Execution Preconditions */}
            <div className="flex flex-col gap-2 pt-2 border-t border-slate-800/80">
              <span className="text-xs font-semibold text-slate-300">EXECUTION PRECONDITIONS</span>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className={`p-2 rounded border flex items-center justify-between ${
                  !isStale ? "bg-slate-900 border-slate-800 text-emerald-400" : "bg-rose-950/40 border-rose-900/40 text-rose-400"
                }`}>
                  <span>Tape Freshness</span>
                  <span>{!isStale ? "MET [✓]" : "UNMET [✕]"}</span>
                </div>
                <div className={`p-2 rounded border flex items-center justify-between ${
                  !badge ? "bg-slate-900 border-slate-800 text-emerald-400" : "bg-amber-950/40 border-amber-900/40 text-amber-300"
                }`}>
                  <span>Pipeline Auth</span>
                  <span>{!badge ? "MET [✓]" : "UNMET [✕]"}</span>
                </div>
                <div className={`p-2 rounded border flex items-center justify-between ${
                  stopFloor > 0 ? "bg-slate-900 border-slate-800 text-emerald-400" : "bg-rose-950/40 border-rose-900/40 text-rose-400"
                }`}>
                  <span>Invalidation Stop</span>
                  <span>{stopFloor > 0 ? "MET [✓]" : "UNMET [✕]"}</span>
                </div>
                <div className={`p-2 rounded border flex items-center justify-between ${
                  rrRatio !== "N/A" && parseFloat(rrRatio) >= 2.0 ? "bg-slate-900 border-slate-800 text-emerald-400" : "bg-slate-900 border-slate-800 text-slate-300"
                }`}>
                  <span>R:R ≥ 2.0x</span>
                  <span>{rrRatio !== "N/A" && parseFloat(rrRatio) >= 2.0 ? "MET [✓]" : `${rrRatio}x`}</span>
                </div>
              </div>
            </div>

            {/* Governed Position Sizing Plan */}
            <div className="bg-slate-900/80 border border-slate-800 rounded p-3 flex flex-col gap-2.5 mt-1">
              <span className="text-xs font-semibold text-slate-300">GOVERNED TRADE LADDER</span>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Risk Allocation ($):</span>
                <input
                  id="budgetB"
                  type="number"
                  value={budgetInput}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value) || 0;
                    setBudgetInput(val);
                    onInteraction("INPUT_CHANGED", { budgetInput: val });
                  }}
                  className="w-24 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-right text-xs font-mono text-slate-100"
                />
              </div>

              <div className="flex justify-between text-xs border-t border-slate-800 pt-1.5">
                <span className="text-slate-400">Calculated Shares:</span>
                <span className="font-mono font-bold text-emerald-400">{shares} Shares</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Committed Capital:</span>
                <span className="font-mono text-slate-200">${capitalCommitted.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Risk Per Share:</span>
                <span className="font-mono text-rose-400">${riskPerShare.toFixed(2)}</span>
              </div>

              {/* Action Dispatcher */}
              <div className="pt-2 border-t border-slate-800 flex flex-col gap-2">
                <button
                  type="button"
                  onClick={onOrderAction}
                  className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold text-xs py-2.5 px-4 rounded shadow transition-colors flex items-center justify-center gap-2"
                >
                  <span>📋</span>
                  <span>{fixture.action_contract?.button_text || `Stage Plan (${shares} Shares)`}</span>
                </button>
                {copyFeedback && (
                  <div className="text-center text-xs font-mono font-bold text-emerald-400 bg-emerald-950/80 border border-emerald-800/60 py-1 px-2 rounded">
                    {copyFeedback}
                  </div>
                )}
                <div className="text-[10px] text-slate-500 text-center">
                  Offline order string copied to clipboard. No broker routing.
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* Right Column: Price Chart & Technical Corridor (60% / 7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-5 flex flex-col gap-4">
            <div className="flex justify-between items-center border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs text-slate-400 uppercase font-mono">Market Spot Price</span>
                <div className="text-3xl font-mono font-bold text-slate-100 mt-0.5">
                  ${spot.toFixed(2)}
                </div>
              </div>
              <div className="text-right">
                <span className="text-xs text-slate-400 font-mono">Quote Timestamp</span>
                <div className="text-xs font-mono text-slate-300 mt-0.5">{fixture.market_data.quote_timestamp}</div>
              </div>
            </div>

            {/* Technical Corridor Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="bg-slate-900 p-3 rounded border border-slate-800 flex flex-col gap-1">
                <span className="text-[11px] text-slate-400">ENTRY PIVOT</span>
                <span className="text-base font-mono font-bold text-sky-400">
                  ${entryPivot.toFixed(2)}
                </span>
                <span className="text-[10px] text-slate-500">
                  Trigger: ${breakoutTrigger.toFixed(2)}
                </span>
              </div>

              <div className="bg-slate-900 p-3 rounded border border-slate-800 flex flex-col gap-1">
                <span className="text-[11px] text-slate-400">INVALIDATION FLOOR</span>
                <span className="text-base font-mono font-bold text-rose-400">
                  ${stopFloor.toFixed(2)}
                </span>
                <span className="text-[10px] text-rose-300/80 font-mono">
                  {fixture.technical_context.distance_to_invalidation_pct ? `-${fixture.technical_context.distance_to_invalidation_pct.toFixed(2)}% risk` : "Structural stop"}
                </span>
              </div>

              <div className="bg-slate-900 p-3 rounded border border-slate-800 flex flex-col gap-1">
                <span className="text-[11px] text-slate-400">TAKE PROFIT 1</span>
                <span className="text-base font-mono font-bold text-emerald-400">
                  ${tp1 > 0 ? tp1.toFixed(2) : "N/A"}
                </span>
                <span className="text-[10px] text-emerald-300/80 font-mono">
                  Target R:R: {rrRatio}x
                </span>
              </div>
            </div>

            {/* Pattern & Geometry Bar */}
            <div className="bg-slate-900/60 p-3 rounded border border-slate-800 text-xs flex flex-wrap items-center justify-between gap-3 font-mono">
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Pattern:</span>
                <span className="text-slate-200 font-bold">{fixture.technical_context.pattern || "Minervini VCP"}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Contractions:</span>
                <span className="text-slate-200 font-bold">{fixture.technical_context.contractions || 3}T</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Setup State:</span>
                <span className="text-slate-200 font-bold">{fixture.technical_context.setup_state || "READY"}</span>
              </div>
            </div>

          </div>

          {/* Level 3: Progressive Evidence Drawers */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 flex flex-col gap-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Level 3: Progressive Evidence Tier
            </span>

            {/* Drawer 1: Technical & Statistical Proof */}
            <div className="border border-slate-800 rounded overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection("TECHNICAL_PROOF")}
                className="w-full bg-slate-900 px-4 py-2.5 text-xs font-mono font-semibold flex items-center justify-between text-slate-300 hover:text-white"
              >
                <span>[1] Technical Proof & Quantitative Sizing Model</span>
                <span>{expandedSection === "TECHNICAL_PROOF" ? "▲" : "▼"}</span>
              </button>
              {expandedSection === "TECHNICAL_PROOF" && (
                <div className="p-3 bg-slate-950 text-xs font-mono text-slate-300 flex flex-col gap-2 border-t border-slate-800">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Canonical Sizing Engine:</span>
                    <span className="text-emerald-400 font-bold">{fixture.sizing_context?.authorized_engine || "governorSizingEngine.ts"} ({shares} Shares @ ${stopFloor.toFixed(2)})</span>
                  </div>
                  {fixture.statistical_context && (
                    <>
                      <div className="flex justify-between border-t border-slate-900 pt-1">
                        <span className="text-amber-400/90 font-bold">Quarantined Intraday Sizer:</span>
                        <span className="text-amber-400/90">{fixture.statistical_context.engine} [QUARANTINED]</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Raw mVaR 95%:</span>
                        <span>{fixture.statistical_context.raw_mvar_95_pct.toFixed(4)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Engine Multiplier:</span>
                        <span>{fixture.statistical_context.engine_multiplier}x</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Quarantined Bracket Stop:</span>
                        <span className="text-amber-400">${fixture.statistical_context.effective_stop_price.toFixed(2)} (-{fixture.statistical_context.effective_stop_distance_pct.toFixed(2)}%)</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Quarantined Share Output:</span>
                        <span className="text-amber-400">{fixture.statistical_context.statistical_shares_per_1000_risk} Shares (Statistical Volatility Model — Quarantined for Multi-Day Execution)</span>
                      </div>
                    </>
                  )}
                  <div className="flex justify-between">
                    <span className="text-slate-400">Market Provenance:</span>
                    <span>{fixture.market_data_provenance}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Drawer 2: First-Class Dissent */}
            <div className="border border-slate-800 rounded overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection("DISSENT")}
                className="w-full bg-slate-900 px-4 py-2.5 text-xs font-mono font-semibold flex items-center justify-between text-slate-300 hover:text-white"
              >
                <span>[2] Bearish Dissent & Material Risks</span>
                <span>{expandedSection === "DISSENT" ? "▲" : "▼"}</span>
              </button>
              {expandedSection === "DISSENT" && (
                <div className="p-3 bg-slate-950 text-xs text-slate-300 flex flex-col gap-2 border-t border-slate-800">
                  {fixture.research_dossier?.bearish_dissent ? (
                    <div className="bg-amber-950/40 border border-amber-900/60 p-2.5 rounded text-amber-200 font-mono">
                      {fixture.research_dossier.bearish_dissent}
                    </div>
                  ) : (
                    <div className="text-slate-400 italic font-mono">No material bearish dissent noted in dossier.</div>
                  )}
                  {fixture.heuristic_context?.corporate_event_alert && (
                    <div className="bg-rose-950/40 border border-rose-900/60 p-2 rounded text-rose-300 font-mono">
                      <strong>EVENT ALERT:</strong> {fixture.heuristic_context.corporate_event_alert}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Drawer 3: Governance & Pipeline Health */}
            <div className="border border-slate-800 rounded overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection("GOVERNANCE")}
                className="w-full bg-slate-900 px-4 py-2.5 text-xs font-mono font-semibold flex items-center justify-between text-slate-300 hover:text-white"
              >
                <span>[3] Governance & Pipeline Telemetry</span>
                <span>{expandedSection === "GOVERNANCE" ? "▲" : "▼"}</span>
              </button>
              {expandedSection === "GOVERNANCE" && (
                <div className="p-3 bg-slate-950 text-xs font-mono text-slate-300 flex flex-col gap-2 border-t border-slate-800">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Pipeline Safe for Trading:</span>
                    <span className={fixture.pipeline_health?.is_safe_for_trading !== false ? "text-emerald-400" : "text-rose-400"}>
                      {fixture.pipeline_health?.is_safe_for_trading !== false ? "YES [✓]" : "NO [✕]"}
                    </span>
                  </div>
                  {fixture.pipeline_health?.failure_reason && (
                    <div className="text-rose-300">
                      <strong>FAILURE:</strong> {fixture.pipeline_health.failure_reason}
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-slate-400">Reference:</span>
                    <span className="text-slate-400">{fixture.provenance_reference}</span>
                  </div>
                </div>
              )}
            </div>

          </div>

        </div>

      </div>
    </div>
  );
}
