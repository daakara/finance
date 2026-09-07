"use client";

import React, { useEffect } from "react";
import { WorkstationPayload } from "../../types/workstation";
import { ExperienceMode } from "../../types/insight";
import { trackTelemetryEvent } from "../../telemetry/tracker";
import { useExperienceStore } from "../../state/experience-store";
import TickerCommandStrip from "../command-strip/TickerCommandStrip";
import WorkstationGrid from "../layout/WorkstationGrid";
import PriceChartWorkspace from "./PriceChartWorkspace";
import ExecutionCorridor from "./ExecutionCorridor";

import dynamic from "next/dynamic";
import ConvictionMatrix from "../conviction/ConvictionMatrix";
import WhyARXCard from "../explanation/WhyARXCard";
import DeltaBanner from "../delta/DeltaBanner";
import { DeltaReport } from "../../types/change-intelligence";

const ResearchAccordionStack = dynamic(
  () => import("../research/ResearchAccordionStack"),
  {
    loading: () => (
      <div className="h-32 rounded-2xl bg-bg-surface/40 border border-border-subtle animate-pulse p-4 flex items-center justify-center text-text-muted font-mono text-xs">
        Loading Progressive Research Modules...
      </div>
    ),
    ssr: false,
  }
);

export interface WorkstationCanvasProps {
  payload: WorkstationPayload;
  onOpenPositionSizer: () => void;
  mode?: ExperienceMode;
  className?: string;
  deltaReport?: DeltaReport;
  onAcknowledgeDelta?: () => void | Promise<void>;
  onViewDiagnostics?: () => void;
}

export default function WorkstationCanvas({
  payload,
  onOpenPositionSizer,
  mode: propMode,
  className = "",
  deltaReport,
  onAcknowledgeDelta,
  onViewDiagnostics,
}: WorkstationCanvasProps) {
  const storeMode = useExperienceStore((state) => state.mode);
  const activeMode = propMode || storeMode || "STANDARD";

  const {
    ticker,
    identity,
    marketData,
    stage1_orientation,
    stage2_geometry,
    stage3_conviction = [],
    stage4_explanation,
  } = payload;

  // Emit workspace_loaded and ttc_started on mount
  useEffect(() => {
    trackTelemetryEvent(
      "ORIENTATION",
      "workspace_loaded",
      {
        ticker,
        mode: activeMode,
        viewport: typeof window !== "undefined" && window.innerWidth >= 1024 ? "desktop" : "mobile",
      },
      ticker
    );

    trackTelemetryEvent(
      "DECISION",
      "ttc_started",
      { ticker, mode: activeMode },
      ticker
    );
  }, [ticker, activeMode]);

  return (
    <div
      data-testid="workstation-canvas"
      className={`flex flex-col w-full space-y-6 ${className}`}
    >
      {/* Stage 6: Contextual Delta Banner (Mounted when unacknowledged material delta exists) */}
      {deltaReport && onAcknowledgeDelta && (
        <DeltaBanner
          report={deltaReport}
          onAcknowledge={onAcknowledgeDelta}
          onViewDiagnostics={onViewDiagnostics}
        />
      )}

      {/* Stage 1: Pinned Ticker Command Strip Orientation Header (110px) */}
      <TickerCommandStrip
        ticker={ticker}
        companyName={identity.name}
        spotPrice={marketData.spotPrice}
        priceChange={marketData.change}
        priceChangePct={marketData.changePct}
        setupScore={stage1_orientation.setupScore}
        domainConfidence={stage1_orientation.domainConfidence}
        executionState={stage1_orientation.executionState}
        liquidityTier={stage1_orientation.liquidityTier}
        marketRegime="RISK_ON"
        isSettlementPinned={marketData.settlementPinned}
        marketSession={marketData.marketSession}
        exchange={identity.exchange}
        sector={identity.sector}
        amihudScore={stage1_orientation.amihudScore}
        mode={activeMode}
      />

      {/* Stage 2: 65/35 Primary Decision Workspace Grid (620px min-height desktop) */}
      <WorkstationGrid
        chart={
          <PriceChartWorkspace
            ticker={ticker}
            spotPrice={marketData.spotPrice}
            priceChangePct={marketData.changePct}
            entryLow={stage2_geometry.entryZone.low}
            entryHigh={stage2_geometry.entryZone.high}
            stopLoss={stage2_geometry.stopLossFloor}
            target1={stage2_geometry.takeProfit1}
            atr={stage2_geometry.volatility?.atr ?? 0.74}
            mode={activeMode}
          />
        }
        execution={
          <ExecutionCorridor
            ticker={ticker}
            spotPrice={marketData.spotPrice}
            entryLow={stage2_geometry.entryZone.low}
            entryHigh={stage2_geometry.entryZone.high}
            stopLoss={stage2_geometry.stopLossFloor}
            target1={stage2_geometry.takeProfit1}
            target2={stage2_geometry.takeProfit2}
            riskRewardRatio={stage2_geometry.riskRewardRatio}
            executionState={stage1_orientation.executionState}
            advShareLimit={stage2_geometry.maxAdvShareLimit}
            onOpenPositionSizer={onOpenPositionSizer}
            mode={activeMode}
          />
        }
        minHeightDesktop="lg:min-h-[620px]"
      />

      {/* Stage 3: Conviction Matrix (Progressive Disclosure Level 2) */}
      {stage3_conviction.length > 0 && (
        <ConvictionMatrix
          items={stage3_conviction}
          ticker={ticker}
          mode={activeMode}
        />
      )}

      {/* Stage 4: Why ARX Thinks This (Top 3 Deterministic Drivers) */}
      {stage4_explanation && (
        <WhyARXCard
          explanation={stage4_explanation}
          ticker={ticker}
          setupScore={stage1_orientation.setupScore}
          mode={activeMode}
        />
      )}

      {/* Stage 5: Progressive Research Accordion Stack (Lazy Hydrated) */}
      <ResearchAccordionStack
        ticker={ticker}
      />
    </div>
  );
}
