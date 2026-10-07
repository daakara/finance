import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import StandardTerminalView from "../terminal/StandardTerminalView";
import GuidedTerminalView from "../terminal/GuidedTerminalView";
import AdvancedTerminalView from "../terminal/AdvancedTerminalView";
import PriceChart from "../PriceChart";
import AdaptiveTerminal from "../AdaptiveTerminal";
import { QuantitativeInsight } from "../../types/insight";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

// Mock matomo
vi.mock("../../lib/matomo", () => ({
  trackAnalysisToSetup: vi.fn(),
  trackWorkspaceSwitch: vi.fn(),
  trackRoleSwitch: vi.fn(),
  trackSymbolSearch: vi.fn(),
}));

// Mock lightweight-charts
const mockApplyOptions = vi.fn();
const mockFitContent = vi.fn();
const mockRemove = vi.fn();

vi.mock("lightweight-charts", () => ({
  createChart: vi.fn(() => ({
    applyOptions: mockApplyOptions,
    remove: mockRemove,
    addCandlestickSeries: vi.fn(() => ({
      setData: vi.fn(),
      setMarkers: vi.fn(),
      applyOptions: vi.fn(),
    })),
    addLineSeries: vi.fn(() => ({
      setData: vi.fn(),
      applyOptions: vi.fn(),
    })),
    timeScale: vi.fn(() => ({
      fitContent: mockFitContent,
    })),
  })),
  LineStyle: { Solid: 0 },
}));

function createFixtureInsight(overrides: Partial<QuantitativeInsight> = {}): QuantitativeInsight {
  return {
    id: "NVDA_SWING_20261007",
    symbol: "NVDA",
    companyName: "NVIDIA Corp",
    price: 184.2,
    changePct: 1.45,
    setupScore: 82,
    horizon: "SWING",
    assessment: "FAVORABLE",
    posture: "WATCH",
    postureLabel: "Wait for Pullback",
    ownership: "NOT_OWNED",
    verdict: "WAIT_FOR_TRIGGER",
    verdictLabel: "WAIT — CONFIRMATION PENDING",
    terminalState: {
      symbol: "NVDA",
      companyName: "NVIDIA Corp",
      currentPrice: 184.2,
      changePct: 1.45,
      horizon: "SWING",
      ownership: { state: "NOT_OWNED", source: "USER_DECLARED" },
      modelProvenance: {
        modelId: "ARX_QUANT_V1",
        modelVersion: "1.2.0",
        rulesetVersion: "2026_Q4",
        calculatedAt: "2026-10-07T00:00:00Z",
      },
      overallEligibility: "ELIGIBLE",
      decisionState: "VALID_SETUP",
      isActionable: false,
      canSizeTrade: false,
      assessment: "FAVORABLE",
      factorAgreement: {
        favorable: 4,
        mixed: 1,
        unfavorable: 0,
        unavailable: 0,
        evaluated: 5,
        displayLabel: "4 of 5 Favorable",
      },
      domains: [],
      posture: "WATCH",
      uiStateLabel: "Awaiting Trigger Confluence",
      headlineExplanation: "Setup structure is constructive but waiting for volume confirmation candle.",
      whatWouldChangeAssessment: "Decisive close above pivot with expanding volume.",
      primaryAction: { label: "SET ALERT", actionType: "SET_ALERT", enabled: true },
      availableActions: [],
    },
    human: {
      assessmentHeadline: "Constructive Consolidation",
      assessmentDescription: "Orderly pullback awaiting confirmation.",
      whyPills: [],
      reclaimMilestone: "Needs volume breakout above $186.00.",
      watchLevels: { watchZone: "$182.50 – $185.00", keyLevel: "$180.00", riskStop: "$174.00" },
      actionCallout: { action: "WATCH", guidance: "Do not chase; wait for confirmation." },
    },
    standard: {
      bottomLine: "Wait for volume-confirmed reversal at key moving average support.",
      signalsRatio: "4/5 Models Favorable",
      confluenceBreakdown: [
        { dimension: "Technical Structure", score: 85 },
        { dimension: "Smart Money Flow", score: 80 },
        { dimension: "Fundamental Solvency", score: 88 },
      ],
      setupSummary: "Minervini Volatility Contraction (VCP)",
      keyLevels: {
        currentPrice: 184.2,
        watchZone: "$182.50 – $185.00",
        sma50: 178.0,
        stopLoss: 174.0,
        target1: 202.0,
        profitRiskRatio: 2.75,
      },
      tradeSetup: {
        pattern: "Minervini VCP",
        timeframe: "Daily",
        keyTrigger: "Breakout above $188.00 on 1.5x volume",
        riskStop: 174.0,
        target1: 202.0,
        target2: 215.0,
      },
    },
    advanced: {
      vcpStage: 2,
      relativeStrengthScore: 92,
      accumulationGrade: "A-",
      rsRating: 92,
      compositeConviction: "HIGH",
      liquidityRisk: "LOW",
    },
    scoreAttribution: {
      items: [
        { label: "Technical Moving Average Alignment", delta: 18, direction: "POSITIVE" },
        { label: "Volume Contraction Characteristics", delta: 15, direction: "POSITIVE" },
      ],
    },
    ...overrides,
  };
}

describe("Synthesis E Wave 2 — First-Viewport Layout & Chart Refit", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ── 1. FIRST-VIEWPORT TWO-COLUMN COMPOSITION TESTS ───────────────────────
  describe("First-Viewport Grid Composition (Standard, Guided, Advanced)", () => {
    it("StandardTerminalView wraps verdict (col-5) and chart (col-7) in xl:grid-cols-12", () => {
      const insight = createFixtureInsight();
      const chartNode = <div data-testid="test-chart-slot">Chart Content</div>;

      const { container } = render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          chartSlot={chartNode}
          isDemo={false}
        />
      );

      // Verify the two-column grid container
      const gridContainer = container.querySelector(".grid.grid-cols-1.xl\\:grid-cols-12");
      expect(gridContainer).not.toBeNull();
      expect(gridContainer?.classList.contains("gap-4")).toBe(true);
      expect(gridContainer?.classList.contains("items-start")).toBe(true);

      // Left column: xl:col-span-5
      const leftCol = gridContainer?.querySelector(".xl\\:col-span-5");
      expect(leftCol).not.toBeNull();
      expect(leftCol?.querySelector('[data-testid="decision-verdict"]')).not.toBeNull();
      expect(leftCol?.querySelector('[data-testid="unmet-condition"]')).not.toBeNull();

      // Right column: xl:col-span-7
      const rightCol = gridContainer?.querySelector(".xl\\:col-span-7");
      expect(rightCol).not.toBeNull();
      expect(rightCol?.querySelector('[data-testid="market-workspace-chart"]')).not.toBeNull();
      expect(rightCol?.querySelector('[data-testid="test-chart-slot"]')).not.toBeNull();
    });

    it("GuidedTerminalView wraps guidance (col-5) and chart (col-7) in xl:grid-cols-12", () => {
      const insight = createFixtureInsight();
      const chartNode = <div data-testid="test-chart-slot">Guided Chart</div>;

      const { container } = render(
        <GuidedTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          chartSlot={chartNode}
        />
      );

      const gridContainer = container.querySelector(".grid.grid-cols-1.xl\\:grid-cols-12");
      expect(gridContainer).not.toBeNull();

      const leftCol = gridContainer?.querySelector(".xl\\:col-span-5");
      expect(leftCol?.querySelector('[data-testid="decision-verdict"]')).not.toBeNull();

      const rightCol = gridContainer?.querySelector(".xl\\:col-span-7");
      expect(rightCol?.querySelector('[data-testid="market-workspace-chart"]')).not.toBeNull();
    });

    it("AdvancedTerminalView wraps quant verdict (col-5) and chart (col-7) in xl:grid-cols-12", () => {
      const insight = createFixtureInsight();
      const chartNode = <div data-testid="test-chart-slot">Advanced Chart</div>;

      const { container } = render(
        <AdvancedTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          chartSlot={chartNode}
        />
      );

      const gridContainer = container.querySelector(".grid.grid-cols-1.xl\\:grid-cols-12");
      expect(gridContainer).not.toBeNull();

      const leftCol = gridContainer?.querySelector(".xl\\:col-span-5");
      expect(leftCol?.querySelector('[data-testid="decision-verdict"]')).not.toBeNull();

      const rightCol = gridContainer?.querySelector(".xl\\:col-span-7");
      expect(rightCol?.querySelector('[data-testid="market-workspace-chart"]')).not.toBeNull();
    });
  });

  // ── 2. CONTAINER-AWARE RESIZEOBSERVER REFIT TESTS ────────────────────────
  describe("AC-CHART-REFIT-RUNTIME: PriceChart ResizeObserver Lifecycle", () => {
    let originalResizeObserver: any;
    let observerCallback: any = null;
    let observeMock = vi.fn();
    let disconnectMock = vi.fn();

    beforeEach(() => {
      originalResizeObserver = global.ResizeObserver;
      observeMock = vi.fn();
      disconnectMock = vi.fn();
      observerCallback = null;

      global.ResizeObserver = class {
        constructor(cb: any) {
          observerCallback = cb;
        }
        observe = observeMock;
        disconnect = disconnectMock;
        unobserve = vi.fn();
      } as any;
    });

    afterEach(() => {
      global.ResizeObserver = originalResizeObserver;
    });

    it("attaches ResizeObserver to container element on mount", () => {
      const candles = [
        { time: "2026-10-01", open: 180, high: 185, low: 179, close: 184, volume: 1000000 },
      ];

      render(
        <PriceChart
          symbol="NVDA"
          candles={candles}
          currentPrice={184}
          userRole="LONG_TERM"
        />
      );

      expect(observeMock).toHaveBeenCalledTimes(1);
      expect(typeof observerCallback).toBe("function");
    });

    it("triggers chart.applyOptions and chart.timeScale().fitContent when container resizes WITHOUT window resize", () => {
      const candles = [
        { time: "2026-10-01", open: 180, high: 185, low: 179, close: 184, volume: 1000000 },
      ];

      render(
        <PriceChart
          symbol="NVDA"
          candles={candles}
          currentPrice={184}
          userRole="LONG_TERM"
        />
      );

      mockApplyOptions.mockClear();
      mockFitContent.mockClear();

      // Simulate ResizeObserver container dimension change event (e.g. sidebar collapse / grid refit)
      observerCallback([
        {
          contentRect: {
            width: 720,
            height: 440,
          },
        },
      ]);

      expect(mockApplyOptions).toHaveBeenCalledWith({
        width: 720,
        height: 440,
      });
      expect(mockFitContent).toHaveBeenCalledTimes(1);
    });

    it("disconnects ResizeObserver on component unmount", () => {
      const candles = [
        { time: "2026-10-01", open: 180, high: 185, low: 179, close: 184, volume: 1000000 },
      ];

      const { unmount } = render(
        <PriceChart
          symbol="NVDA"
          candles={candles}
          currentPrice={184}
          userRole="LONG_TERM"
        />
      );

      unmount();
      expect(disconnectMock).toHaveBeenCalledTimes(1);
      expect(mockRemove).toHaveBeenCalledTimes(1);
    });
  });

  // ── 3. PRESERVATION OF WAVE 1 INVARIANTS ──────────────────────────────────
  describe("Wave 1 Invariants Preservation in Wave 2 Composition", () => {
    it("preserves compact #demo-asset-tag in Verdict Card when isDemo=true", () => {
      const insight = createFixtureInsight();
      render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          isDemo={true}
        />
      );

      const demoTag = screen.getByText("Demo Asset");
      expect(demoTag).toBeDefined();
      expect(demoTag.id).toBe("demo-asset-tag");
      expect(demoTag.closest('[data-testid="decision-verdict"]')).not.toBeNull();
    });

    it("preserves subordinated #why-score-btn in Confluence Breakdown", () => {
      const onOpenWhy = vi.fn();
      const insight = createFixtureInsight({ setupScore: 82 });

      render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={onOpenWhy}
        />
      );

      const whyBtn = screen.getByRole("button", { name: /Why Score 82\?/i });
      expect(whyBtn).toBeDefined();
      expect(whyBtn.id).toBe("why-score-btn");

      fireEvent.click(whyBtn);
      expect(onOpenWhy).toHaveBeenCalledTimes(1);
    });

    it("preserves AdaptiveTerminal complete composition without runtime crashes", () => {
      const chartNode = <div data-testid="adaptive-chart">Chart Slot Node</div>;
      const planNode = <div data-testid="adaptive-plan">Plan Slot Node</div>;

      const { container } = render(
        <AdaptiveTerminal
          symbol="NVDA"
          companyName="NVIDIA Corp"
          currentPrice={184.2}
          changePct={1.45}
          setupScore={82}
          chartSlot={chartNode}
          planSlot={planNode}
          userRole="LONG_TERM"
          isDemo={false}
        />
      );

      expect(screen.getByTestId("adaptive-chart")).toBeDefined();
      expect(screen.getByTestId("adaptive-plan")).toBeDefined();
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("supporting-evidence")).toBeDefined();
      expect(container.textContent).toContain("NVDA");
    });
  });
});
