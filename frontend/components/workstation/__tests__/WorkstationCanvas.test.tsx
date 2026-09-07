import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ExecutionCorridor from "../ExecutionCorridor";
import PriceChartWorkspace from "../PriceChartWorkspace";
import WorkstationCanvas from "../WorkstationCanvas";
import { WorkstationPayload } from "../../../types/workstation";

const mockPayload: WorkstationPayload = {
  ticker: "CPRX",
  identity: {
    name: "Catalyst Pharmaceuticals Inc.",
    exchange: "NASDAQ",
    sector: "Healthcare",
  },
  marketData: {
    spotPrice: 18.42,
    change: 0.38,
    changePct: 2.11,
    marketSession: "CLOSED",
    settlementPinned: true,
  },
  stage1_orientation: {
    setupScore: 71,
    domainConfidence: "HIGH",
    executionState: "IN_BUY_ZONE",
    liquidityTier: "HIGH",
    amihudScore: 0.0014,
  },
  stage2_geometry: {
    entryZone: {
      low: 18.1,
      high: 18.55,
    },
    stopLossFloor: 17.2,
    takeProfit1: 20.4,
    takeProfit2: 22.5,
    riskRewardRatio: 1.62,
    maxAdvShareLimit: 12400,
    volatility: {
      atr: 0.74,
      upperBand: 19.88,
      lowerBand: 16.96,
    },
  },
  stage3_conviction: [],
  stage4_explanation: {
    confluenceScore: 74.2,
    drivers: [],
  },
};

describe("WorkstationCanvas & ExecutionCorridor (Milestone W1.7)", () => {
  describe("ExecutionCorridor (35% Actionable Ladder)", () => {
    it("renders corridor with decision levels and anti-cyan color compliance", () => {
      const handleSizer = vi.fn();

      render(
        <ExecutionCorridor
          ticker="CPRX"
          spotPrice={18.42}
          entryLow={18.1}
          entryHigh={18.55}
          stopLoss={17.2}
          target1={20.4}
          target2={22.5}
          riskRewardRatio={1.62}
          executionState="IN_BUY_ZONE"
          advShareLimit={12400}
          onOpenPositionSizer={handleSizer}
        />
      );

      // Verify corridor card
      const corridor = screen.getByTestId("execution-corridor");
      expect(corridor).toBeDefined();

      // Entry Zone
      const entry = screen.getByTestId("corridor-entry");
      expect(entry.textContent).toContain("$18.10 – $18.55");
      expect(entry.textContent).toContain("In Buy Zone");

      // Targets
      const target1 = screen.getByTestId("corridor-target-1");
      expect(target1.textContent).toContain("$20.40");
      expect(target1.className).toContain("bg-emerald-500/10");

      const target2 = screen.getByTestId("corridor-target-2");
      expect(target2.textContent).toContain("$22.50");

      // Risk-Reward
      const rr = screen.getByTestId("corridor-risk-reward");
      expect(rr.textContent).toContain("R/R Ratio: 1.62x");

      // Stop Loss
      const stop = screen.getByTestId("corridor-stop");
      expect(stop.textContent).toContain("$17.20");
      expect(stop.className).toContain("bg-rose-500/10");

      // ADV Limit
      const adv = screen.getByTestId("corridor-adv-limit");
      expect(adv.textContent).toContain("12,400 shares");

      // Position Sizer CTA
      const cta = screen.getByTestId("corridor-sizer-cta");
      fireEvent.click(cta);
      expect(handleSizer).toHaveBeenCalledTimes(1);
    });
  });

  describe("PriceChartWorkspace (65% Chart Canvas)", () => {
    it("renders chart workspace with timeframe switcher and min-height constraint", () => {
      render(
        <PriceChartWorkspace
          ticker="CPRX"
          spotPrice={18.42}
          priceChangePct={2.11}
          atr={0.74}
        />
      );

      const workspace = screen.getByTestId("price-chart-workspace");
      expect(workspace).toBeDefined();
      expect(workspace.className).toContain("min-h-[420px]");
      expect(workspace.textContent).toContain("CPRX");
      expect(workspace.textContent).toContain("ATR(14):");
      expect(workspace.textContent).toContain("$0.74");
    });
  });

  describe("WorkstationCanvas (Integrated 65/35 Above-the-Fold Assembly)", () => {
    it("renders Stage 1 orientation and Stage 2 65/35 grid layout", () => {
      const handleSizer = vi.fn();

      render(
        <WorkstationCanvas
          payload={mockPayload}
          onOpenPositionSizer={handleSizer}
        />
      );

      expect(screen.getByTestId("workstation-canvas")).toBeDefined();
      expect(screen.getByTestId("ticker-command-strip")).toBeDefined();
      expect(screen.getByTestId("workstation-grid")).toBeDefined();
      expect(screen.getByTestId("price-chart-workspace")).toBeDefined();
      expect(screen.getByTestId("execution-corridor")).toBeDefined();
    });
  });
});
