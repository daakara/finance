"use client";

import React, { useEffect } from "react";
import { ConvictionItem } from "../../types/workstation";
import { ExperienceMode } from "../../types/insight";
import { trackTelemetryEvent } from "../../telemetry/tracker";
import ConvictionPillDetailPopover from "./ConvictionPillDetailPopover";
import InstitutionalTooltip from "../tooltips/InstitutionalTooltip";

export interface ConvictionMatrixProps {
  items: ConvictionItem[];
  ticker: string;
  mode?: ExperienceMode;
  className?: string;
}

export default function ConvictionMatrix({
  items,
  ticker,
  mode = "STANDARD",
  className = "",
}: ConvictionMatrixProps) {
  useEffect(() => {
    trackTelemetryEvent(
      "DECISION",
      "conviction_matrix_viewed",
      { ticker, mode, count: items.length },
      ticker
    );
  }, [ticker, mode, items.length]);

  return (
    <section
      data-testid="conviction-matrix"
      aria-labelledby="conviction-matrix-heading"
      className={`p-4 sm:p-5 rounded-2xl bg-bg-surface border border-border-subtle shadow-lg space-y-3 font-sans ${className}`}
    >
      {/* Header with Progressive Disclosure Rule info */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-caption-mono uppercase px-2 py-0.5 rounded bg-bg-surface-raised border border-border-subtle text-accent-positive font-bold text-[10px]">
            Stage 3
          </span>
          <h3
            id="conviction-matrix-heading"
            className="text-header-2 text-text-primary"
          >
            Institutional Conviction Matrix
          </h3>
          <InstitutionalTooltip
            title="Bayesian Conviction Synthesis"
            formula="P(Thesis|E) = \frac{P(E|Thesis) \cdot P(Thesis)}{P(E)}"
            explanation="Synthesizes fundamentals, institutional flow, market regime, chart structure, and historical win-rate into an integrated conviction assessment."
            provenance="ARX Multimodal Quantitative Engine"
          />
        </div>

        <span className="text-caption-mono text-text-muted text-xs">
          Interactive Details · Click any pillar to inspect causal reasons
        </span>
      </div>

      {/* 5-Column Responsive Decision Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {items.map((item) => (
          <ConvictionPillDetailPopover
            key={item.dimension}
            item={item}
            ticker={ticker}
          />
        ))}
      </div>
    </section>
  );
}
