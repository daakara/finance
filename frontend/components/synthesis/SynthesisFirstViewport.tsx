"use client";

import React from "react";
import SynthesisVerdictCard, { SynthesisVerdictCardProps } from "./SynthesisVerdictCard";

interface SynthesisFirstViewportProps {
  verdictProps: SynthesisVerdictCardProps;
  chartSlot: React.ReactNode;
  preflightSlot: React.ReactNode;
}

export default function SynthesisFirstViewport({
  verdictProps,
  chartSlot,
  preflightSlot,
}: SynthesisFirstViewportProps) {
  return (
    <section
      className="first-viewport-row grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-5 items-start"
      aria-label="Decision Cockpit and Chart Workspace"
    >
      {/* Left Column (5 cols): Decision Cockpit */}
      <div className="lg:col-span-5 w-full min-w-0 order-1">
        <SynthesisVerdictCard {...verdictProps} />
      </div>

      {/* Right Column (7 cols): Candlestick Chart Card & Pre-Flight Execution Bar */}
      <div className="lg:col-span-7 w-full min-w-0 order-2 space-y-4">
        {/* Chart Card */}
        <article
          id="chart-card"
          className="chart-card w-full bg-[var(--surface-1)] border border-[var(--border-strong)] rounded-2xl p-3 sm:p-4 shadow-xl space-y-3"
          aria-label="Interactive Candlestick and Technical Indicators"
        >
          {chartSlot}
        </article>

        {/* Pre-Flight Readiness Bar mounted directly beneath chart */}
        <div className="w-full">
          {preflightSlot}
        </div>
      </div>
    </section>
  );
}
