"use client";

import React, { useState, useEffect } from "react";
import { Stage4Explanation, Stage4Driver } from "../../types/workstation";
import { ExperienceMode } from "../../types/insight";
import { trackTelemetryEvent } from "../../telemetry/tracker";
import ConfluenceTraceModal from "./ConfluenceTraceModal";

export interface WhyARXCardProps {
  explanation: Stage4Explanation;
  ticker: string;
  setupScore: number;
  mode?: ExperienceMode;
  className?: string;
}

export default function WhyARXCard({
  explanation,
  ticker,
  setupScore,
  mode = "STANDARD",
  className = "",
}: WhyARXCardProps) {
  const [isTraceOpen, setIsTraceOpen] = useState(false);
  const { drivers = [], headline, confluenceScore, factors = [], modelVer, decisionHash } = explanation;

  // Track telemetry on mount
  useEffect(() => {
    trackTelemetryEvent(
      "DECISION",
      "why_arx_card_viewed",
      {
        ticker,
        setupScore,
        mode,
        driverCount: drivers.length,
        topDriver: drivers[0]?.id || "none",
      },
      ticker
    );
  }, [ticker, setupScore, mode, drivers]);

  // Anti-Cyan Directional Badges
  const getDirectionPill = (dir: Stage4Driver["direction"], pts?: number) => {
    switch (dir) {
      case "BULLISH":
        return (
          <span className="flex items-center gap-1 text-[11px] font-mono font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800/80 px-2 py-0.5 rounded shrink-0">
            <span>↑ Bullish</span>
            {pts !== undefined && <span>(+{pts} pts)</span>}
          </span>
        );
      case "BEARISH":
        return (
          <span className="flex items-center gap-1 text-[11px] font-mono font-bold text-rose-400 bg-rose-950/60 border border-rose-800/80 px-2 py-0.5 rounded shrink-0">
            <span>↓ Drag</span>
            {pts !== undefined && <span>({pts} pts)</span>}
          </span>
        );
      case "NEUTRAL":
      default:
        return (
          <span className="flex items-center gap-1 text-[11px] font-mono font-bold text-amber-400 bg-amber-950/60 border border-amber-800/80 px-2 py-0.5 rounded shrink-0">
            <span>→ Neutral</span>
            {pts !== undefined && <span>({pts >= 0 ? `+${pts}` : pts} pts)</span>}
          </span>
        );
    }
  };

  // Setup score color
  const scoreColor =
    setupScore >= 70
      ? "text-emerald-400 bg-emerald-950/40 border-emerald-800/60"
      : setupScore >= 50
      ? "text-amber-400 bg-amber-950/40 border-amber-800/60"
      : "text-rose-400 bg-rose-950/40 border-rose-800/60";

  return (
    <section
      data-testid="why-arx-card"
      aria-labelledby="why-arx-heading"
      className={`p-5 rounded-2xl bg-bg-surface border border-border-subtle shadow-xl space-y-4 font-sans ${className}`}
    >
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-border-subtle">
        <div className="flex items-center gap-2.5">
          <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase tracking-wider bg-bg-surface-raised border border-border-subtle text-accent-info rounded">
            Stage 4 · Synthesis
          </span>
          <h3 id="why-arx-heading" className="text-header-1 text-text-primary">
            Why ARX Thinks This
          </h3>
        </div>

        <div className="flex items-center gap-3">
          <div className={`px-3 py-1 rounded-lg border flex items-center gap-2 font-mono text-xs ${scoreColor}`}>
            <span className="text-text-muted">Setup Score:</span>
            <span className="text-base font-black">{setupScore}/100</span>
          </div>

          <button
            type="button"
            onClick={() => setIsTraceOpen(true)}
            className="px-3 py-1.5 rounded-lg bg-bg-surface-raised hover:bg-bg-surface-elevated border border-border-subtle text-accent-info hover:text-text-primary text-xs font-mono font-bold transition-all cursor-pointer flex items-center gap-1.5 active:scale-95"
          >
            <span>View Full Confluence Trace</span>
            <span>→</span>
          </button>
        </div>
      </div>

      {/* Synthesis Headline */}
      {headline && (
        <p className="text-body-ui text-text-secondary leading-relaxed font-medium">
          {headline}
        </p>
      )}

      {/* Top 3 Server-Authoritative Drivers (Strictly Explaining Model Output) */}
      <div className="space-y-2.5">
        <span className="text-[11px] font-mono uppercase text-text-muted font-bold tracking-wider block">
          Top Deterministic Drivers (Server-Authoritative):
        </span>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {drivers.slice(0, 3).map((driver, index) => (
            <div
              key={driver.id || index}
              className="p-3.5 rounded-xl bg-bg-surface-raised border border-border-subtle flex flex-col justify-between space-y-2"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="text-[10px] font-mono font-bold text-text-muted">
                  #{index + 1} · {driver.category}
                </span>
                {getDirectionPill(driver.direction, driver.contributionPoints)}
              </div>

              <div>
                <h4 className="text-body-ui font-bold text-text-primary leading-snug">
                  {driver.headline}
                </h4>
                <p className="text-caption-mono text-text-secondary text-[11px] mt-1 leading-relaxed">
                  {driver.detail}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Confluence Trace Modal (Level 3 Evidence) */}
      <ConfluenceTraceModal
        isOpen={isTraceOpen}
        onClose={() => setIsTraceOpen(false)}
        ticker={ticker}
        setupScore={setupScore}
        confluenceScore={confluenceScore || setupScore}
        factors={factors}
        modelVer={modelVer}
        decisionHash={decisionHash}
      />
    </section>
  );
}
