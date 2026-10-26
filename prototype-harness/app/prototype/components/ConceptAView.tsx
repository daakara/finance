"use client";

import React, { useState } from "react";
import { ScenarioFixture } from "@/lib/prototype/scenarioLoader";

interface ConceptViewProps {
  fixture: ScenarioFixture;
  onInteraction: (eventType: string, details: any) => void;
  onOrderAction: () => void;
  copyFeedback: string | null;
}

export function ConceptAView({ fixture, onInteraction, onOrderAction, copyFeedback }: ConceptViewProps) {
  const [activeTab, setActiveTab] = useState<"OVERVIEW" | "TECHNICAL" | "SIZING">("OVERVIEW");
  const [budgetInput, setBudgetInput] = useState<number>(1000);

  const handleTabChange = (tab: "OVERVIEW" | "TECHNICAL" | "SIZING") => {
    setActiveTab(tab);
    onInteraction("TAB_CHANGED", { tab });
  };

  const spot = fixture.market_data.spot_price;
  const stopFloor = fixture.technical_context.structural_invalidation_floor || 0;
  const riskPerShare = stopFloor > 0 && spot > stopFloor ? spot - stopFloor : 0;
  const shares = riskPerShare > 0 ? Math.floor(budgetInput / riskPerShare) : 0;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 shadow-xl flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold font-mono text-emerald-400">{fixture.ticker}</span>
            <span className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono">
              {fixture.primary_exchange}
            </span>
            <span className="text-xs text-slate-400">{fixture.company_name}</span>
          </div>
          <div className="text-xs text-slate-400 mt-1 flex items-center gap-3">
            <span>SPOT: <strong className="font-mono text-slate-200">${spot.toFixed(2)}</strong></span>
            <span>DATA: <strong className="font-mono text-slate-300">{fixture.market_data.display_label}</strong></span>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex bg-slate-950 p-1 rounded border border-slate-800" role="tablist">
          {(["OVERVIEW", "TECHNICAL", "SIZING"] as const).map((tab) => (
            <button
              key={tab}
              role="tab"
              aria-selected={activeTab === tab}
              onClick={() => handleTabChange(tab)}
              className={`px-4 py-1.5 rounded text-xs font-semibold font-mono transition-colors ${
                activeTab === tab
                  ? "bg-emerald-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Tab 1: Overview */}
      {activeTab === "OVERVIEW" && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-950 p-4 rounded border border-slate-800">
            <div className="text-xs text-slate-400">MARKET QUOTE</div>
            <div className="text-xl font-mono font-bold text-slate-100 mt-1">${spot.toFixed(2)}</div>
            <div className="text-[11px] text-slate-500 mt-2">Timestamp: {fixture.market_data.quote_timestamp}</div>
          </div>
          <div className="bg-slate-950 p-4 rounded border border-slate-800">
            <div className="text-xs text-slate-400">EXECUTION READINESS</div>
            <div className="text-base font-bold text-emerald-400 mt-1">
              {fixture.technical_context.execution_readiness || (fixture.technical_context.is_actionable ? "ACTIONABLE" : "SUSPENDED")}
            </div>
            <div className="text-[11px] text-slate-500 mt-2">
              Actionable: {fixture.technical_context.is_actionable ? "YES [✓]" : "NO [✕]"}
            </div>
          </div>
          <div className="bg-slate-950 p-4 rounded border border-slate-800">
            <div className="text-xs text-slate-400">DATA FRESHNESS</div>
            <div className={`text-base font-bold mt-1 ${fixture.market_data.is_stale ? "text-rose-400" : "text-emerald-400"}`}>
              {fixture.market_data.is_stale ? "STALE / DELAYED [⚠]" : "LIVE VERIFIED [✓]"}
            </div>
            <div className="text-[11px] text-slate-500 mt-2">
              {fixture.market_data.latency_seconds ? `${fixture.market_data.latency_seconds}s latency` : "Standard feed"}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Technical */}
      {activeTab === "TECHNICAL" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-950 p-4 rounded border border-slate-800 flex flex-col gap-2">
            <span className="text-xs text-slate-400 font-semibold">STRUCTURAL BASE LEVELS</span>
            <div className="flex justify-between border-b border-slate-800 py-1.5 text-xs">
              <span className="text-slate-400">Planned Entry Pivot:</span>
              <span className="font-mono text-slate-200">
                ${fixture.technical_context.planned_entry_pivot?.toFixed(2) || "N/A"}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800 py-1.5 text-xs">
              <span className="text-slate-400">Structural Invalidation Floor:</span>
              <span className="font-mono text-rose-400 font-bold">
                ${fixture.technical_context.structural_invalidation_floor?.toFixed(2) || "N/A"}
              </span>
            </div>
            <div className="flex justify-between py-1.5 text-xs">
              <span className="text-slate-400">Stop Distance to Invalidation:</span>
              <span className="font-mono text-slate-300">
                {fixture.technical_context.distance_to_invalidation_pct?.toFixed(2) || "N/A"}%
              </span>
            </div>
          </div>

          <div className="bg-slate-950 p-4 rounded border border-slate-800 flex flex-col gap-2">
            <span className="text-xs text-slate-400 font-semibold">TARGET PROJECTIONS</span>
            <div className="flex justify-between border-b border-slate-800 py-1.5 text-xs">
              <span className="text-slate-400">Take Profit 1 (TP1):</span>
              <span className="font-mono text-emerald-400">
                ${fixture.technical_context.take_profit_1?.toFixed(2) || "N/A"}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-800 py-1.5 text-xs">
              <span className="text-slate-400">Take Profit 2 (TP2):</span>
              <span className="font-mono text-emerald-300">
                ${fixture.technical_context.take_profit_2?.toFixed(2) || "N/A"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Sizing & Order */}
      {activeTab === "SIZING" && (
        <div className="flex flex-col gap-4">
          <div className="bg-slate-950 p-4 rounded border border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="budgetA" className="text-xs text-slate-400 block mb-1">Standard Risk Budget ($):</label>
              <input
                id="budgetA"
                type="number"
                value={budgetInput}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0;
                  setBudgetInput(val);
                  onInteraction("INPUT_CHANGED", { budgetInput: val });
                }}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-xs font-mono text-slate-100"
              />
            </div>
            <div>
              <span className="text-xs text-slate-400 block mb-1">Structural Risk Per Share:</span>
              <div className="text-sm font-mono font-bold text-slate-200 py-1.5">
                ${riskPerShare.toFixed(2)}
              </div>
            </div>
            <div>
              <span className="text-xs text-slate-400 block mb-1">Recommended Shares:</span>
              <div className="text-lg font-mono font-bold text-emerald-400 py-1">
                {shares} Shares
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between bg-slate-950 p-4 rounded border border-slate-800">
            <div>
              <div className="text-xs font-semibold text-slate-200">Execution Action Dispatcher</div>
              <div className="text-[11px] text-slate-500">Action copies parameters to system clipboard.</div>
            </div>
            <div className="flex items-center gap-3">
              {copyFeedback && (
                <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2.5 py-1 rounded">
                  {copyFeedback}
                </span>
              )}
              <button
                type="button"
                onClick={onOrderAction}
                className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-bold px-4 py-2 rounded shadow transition-colors"
              >
                {fixture.action_contract?.button_text || `Authorize Order: ${shares} Shares ${fixture.ticker}`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
