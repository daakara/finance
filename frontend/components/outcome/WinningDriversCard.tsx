"use client";

import React, { useState } from "react";

export interface WinningDriverItem {
  id: string;
  name: string;
  winRate: number;
  avgReturn: number;
  tradesCount: number;
  badge: string;
  description: string;
  mechanism: string;
}

export interface WinningDriversCardProps {
  drivers?: WinningDriverItem[];
  className?: string;
}

export default function WinningDriversCard({
  drivers = [
    {
      id: "inst-acc",
      name: "Institutional Accumulation",
      winRate: 72.0,
      avgReturn: 14.2,
      tradesCount: 842,
      badge: "PRIMARY ALPHA EDGE",
      description: "Dark-pool & block volume absorption (+2.1σ) prior to price breakout",
      mechanism: "Institutional accumulation absorbs floating supply, providing strong structural support.",
    },
    {
      id: "sector-rot",
      name: "Sector Rotation Alignment",
      winRate: 69.4,
      avgReturn: 11.8,
      tradesCount: 512,
      badge: "STRONG CONVICTION",
      description: "Capital migration into leading industry groups (Semiconductors / Software)",
      mechanism: "Cross-asset sector breadth confirms institutional rotation and tailwind momentum.",
    },
    {
      id: "rs-breakout",
      name: "Relative Strength Breakout",
      winRate: 66.1,
      avgReturn: 9.4,
      tradesCount: 394,
      badge: "RELIABLE MOMENTUM",
      description: "New 52w relative highs emerging from multi-week low-volatility bases",
      mechanism: "Outperforming peers during broader market consolidations signals superior sponsorship.",
    },
    {
      id: "regime-conf",
      name: "Regime Confluence (Expansion)",
      winRate: 64.5,
      avgReturn: 8.1,
      tradesCount: 436,
      badge: "MACRO TAILWIND",
      description: "VIX < 18 and positive credit spread momentum validating risk-on exposure",
      mechanism: "Macro tailwinds suppress tail-risk stopouts and allow trends to reach full targets.",
    },
  ],
  className = "",
}: WinningDriversCardProps) {
  const [selectedDriverId, setSelectedDriverId] = useState<string | null>(null);
  const [metricMode, setMetricMode] = useState<"winRate" | "avgReturn">("winRate");

  const sortedDrivers = [...drivers].sort((a, b) =>
    metricMode === "winRate" ? b.winRate - a.winRate : b.avgReturn - a.avgReturn
  );

  return (
    <div
      role="region"
      aria-label="Winning Drivers Card"
      className={`rounded-2xl border border-border-subtle bg-surface-card p-6 shadow-sm ${className}`}
    >
      {/* Header with Title & Sort Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border-subtle pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-primary">
              What Works · Persistent Alpha Drivers
            </h3>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Institutional signals with highest empirical win rates and positive realized payoff.
          </p>
        </div>

        <div className="flex items-center bg-surface-base border border-border-subtle rounded-lg p-0.5 text-xs font-mono">
          <button
            type="button"
            onClick={() => setMetricMode("winRate")}
            className={`px-2.5 py-1 rounded-md transition-colors ${
              metricMode === "winRate"
                ? "bg-surface-raised text-emerald-400 font-semibold"
                : "text-text-muted hover:text-text-secondary"
            }`}
          >
            Win Rate
          </button>
          <button
            type="button"
            onClick={() => setMetricMode("avgReturn")}
            className={`px-2.5 py-1 rounded-md transition-colors ${
              metricMode === "avgReturn"
                ? "bg-surface-raised text-emerald-400 font-semibold"
                : "text-text-muted hover:text-text-secondary"
            }`}
          >
            Avg Return
          </button>
        </div>
      </div>

      {/* Driver List */}
      <div className="mt-5 space-y-3">
        {sortedDrivers.map((driver) => {
          const isSelected = selectedDriverId === driver.id;

          return (
            <div
              key={driver.id}
              onClick={() => setSelectedDriverId(isSelected ? null : driver.id)}
              className={`group cursor-pointer rounded-xl border p-4 transition-all duration-200 ${
                isSelected
                  ? "border-emerald-500/50 bg-emerald-500/5 shadow-xs"
                  : "border-border-subtle/80 bg-surface-subtle/40 hover:border-border-subtle hover:bg-surface-subtle"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-sm font-bold text-text-primary group-hover:text-emerald-300 transition-colors">
                      {driver.name}
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      {driver.badge}
                    </span>
                  </div>
                  <p className="text-xs text-text-secondary">{driver.description}</p>
                </div>

                <div className="text-right shrink-0">
                  <div className="font-mono text-base font-bold text-emerald-400">
                    {metricMode === "winRate" ? `${driver.winRate.toFixed(1)}%` : `+${driver.avgReturn.toFixed(1)}%`}
                  </div>
                  <span className="font-mono text-[11px] text-text-muted">
                    {driver.tradesCount} resolved trades
                  </span>
                </div>
              </div>

              {/* Visual Progress Bar */}
              <div className="mt-3 w-full bg-surface-base h-2 rounded-full overflow-hidden border border-border-subtle/40">
                <div
                  className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${Math.min(100, metricMode === "winRate" ? driver.winRate : driver.avgReturn * 5)}%` }}
                />
              </div>

              {/* Expandable Mechanism Explanation */}
              {isSelected && (
                <div className="mt-3 pt-3 border-t border-emerald-500/20 text-xs text-text-secondary space-y-1 animate-fadeIn">
                  <div className="flex items-center gap-1.5 font-mono text-[11px] text-emerald-400 font-semibold">
                    <span>⚡ Causal Mechanism:</span>
                  </div>
                  <p className="leading-relaxed">{driver.mechanism}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Insight */}
      <div className="mt-5 pt-4 border-t border-border-subtle/60 flex items-center justify-between text-xs font-mono">
        <span className="text-text-muted">Cumulative Alpha Contribution:</span>
        <span className="font-bold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20">
          +43.5% Excess Return vs SPY
        </span>
      </div>
    </div>
  );
}
