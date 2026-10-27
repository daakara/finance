import React from "react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import StandardTerminalView from "../terminal/StandardTerminalView";
import GuidedTerminalView from "../terminal/GuidedTerminalView";
import AdvancedTerminalView from "../terminal/AdvancedTerminalView";
import AdaptiveTerminal from "../AdaptiveTerminal";
import { useExperienceStore } from "../../state/experience-store";
import { QuantitativeInsight } from "../../types/insight";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => "/",
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

// Mock matomo
vi.mock("../../lib/matomo", () => ({
  trackFirstHubNavigation: vi.fn(),
  trackOnboardingCompleted: vi.fn(),
  trackAnalysisToSetup: vi.fn(),
  trackRoleSwitch: vi.fn(),
  trackSymbolSearch: vi.fn(),
  trackWorkspaceSwitch: vi.fn(),
}));

// Mock lightweight-charts
vi.mock("lightweight-charts", () => ({
  createChart: vi.fn(() => ({
    applyOptions: vi.fn(),
    remove: vi.fn(),
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
      fitContent: vi.fn(),
    })),
  })),
}));

const mockCanonicalInsight: QuantitativeInsight = {
  symbol: "NVDA",
  companyName: "NVIDIA Corporation",
  exchange: "NASDAQ",
  price: 135.25,
  changePct: 2.45,
  setupScore: 88,
  verdictLabel: "FAVORABLE SETUP",
  bias: "BULLISH",
  confidence: "HIGH",
  actionable: true,
  standard: {
    bottomLine: "Institutional accumulation confirmed with multi-stage VCP breakout.",
    signalsRatio: "5/5 models agree",
    setupSummary: "Minervini Stage 2 Continuation",
    confluenceBreakdown: [
      { dimension: "Momentum", score: 92 },
      { dimension: "Volume", score: 85 },
      { dimension: "Regime", score: 88 },
    ],
    keyLevels: {
      currentPrice: 135.25,
      watchZone: "$134.00 - $137.00",
      entry: 135.0,
      entryMin: 134.0,
      entryMax: 137.0,
      breakoutPivot: 134.5,
      stopLoss: 129.5,
      target1: 148.0,
      target1Pct: 9.4,
      profitRiskRatio: 2.6,
      sma50: 125.0,
      sma200: 110.0,
    },
  },
  human: {
    assessmentDescription: "Leader in AI semiconductor design showing clean structural base.",
    reclaimMilestone: "Decisively cleared resistance at $134.50 on high volume.",
    executionReadiness: "Actionable within entry corridor.",
    plainBottomLine: "NVIDIA is displaying textbook institutional demand.",
    whyPills: [
      { category: "Price Trend", sentiment: "positive", description: "Consistently above rising 50-day average." },
      { category: "Smart Money", sentiment: "positive", description: "Above-average institutional accumulation." },
    ],
    watchLevels: {
      watchZone: "$134.00 - $137.00",
      keyLevel: "$134.50",
      stopLoss: "$129.50",
    },
  },
  advanced: {
    rsi: 64.2,
    ema20: 131.1,
    sma50: 125.0,
    atr: 4.85,
    rvol: 1.82,
    beta: 1.65,
    relativeStrengthScore: 94,
    volatilityRegime: "NORMAL",
    vcpStage: "Stage 2 Continuation",
  },
  terminalState: {
    isActionable: true,
    decisionState: "ACTIONABLE_SETUP",
    overallEligibility: "ELIGIBLE",
    confidence: "HIGH",
    bias: "BULLISH",
    freshnessStatus: "FRESH",
    freshness: {
      status: "FRESH",
      asOfTimestamp: "2026-10-27T14:00:00Z",
    },
  },
};

describe("Cross-Mode Decision Parity Suite (Directive 051)", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    cleanup();
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  describe("CMP-01: Canonical Decision Parity Across All 3 Modes", () => {
    it("renders identical core trading decision invariants across Guided, Standard, and Quant", () => {
      // 1. Standard View
      const { unmount: unmountStd } = render(
        <StandardTerminalView
          insight={mockCanonicalInsight}
          preflightSlot={<div data-testid="preflight-slot-std">CLEARANCE_OK</div>}
        />
      );

      expect(screen.getByTestId("card-symbol").textContent).toBe("NVDA");
      expect(screen.getByTestId("card-price").textContent).toContain("135.25");
      expect(screen.getByTestId("card-delta").textContent).toContain("+2.45%");
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
      expect(screen.getByTestId("cell-stop").textContent).toContain("129.50");
      expect(screen.getByTestId("cell-target").textContent).toContain("148.00");
      expect(screen.getByTestId("preflight-slot-std")).toBeDefined();
      unmountStd();

      // 2. Guided View
      const { unmount: unmountGdd } = render(
        <GuidedTerminalView
          insight={mockCanonicalInsight}
          preflightSlot={<div data-testid="preflight-slot-gdd">CLEARANCE_OK</div>}
        />
      );

      expect(screen.getByTestId("card-symbol").textContent).toBe("NVDA");
      expect(screen.getByTestId("card-price").textContent).toContain("135.25");
      expect(screen.getByTestId("card-delta").textContent).toContain("+2.45%");
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
      expect(screen.getByTestId("cell-stop").textContent).toContain("129.50");
      expect(screen.getByTestId("cell-target").textContent).toContain("148.00");
      expect(screen.getByTestId("preflight-slot-gdd")).toBeDefined();
      unmountGdd();

      // 3. Quant / Advanced View
      const { unmount: unmountAdv } = render(
        <AdvancedTerminalView
          insight={mockCanonicalInsight}
          preflightSlot={<div data-testid="preflight-slot-adv">CLEARANCE_OK</div>}
        />
      );

      expect(screen.getByTestId("card-symbol").textContent).toBe("NVDA");
      expect(screen.getByTestId("card-price").textContent).toContain("135.25");
      expect(screen.getByTestId("card-delta").textContent).toContain("+2.45%");
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
      expect(screen.getByTestId("cell-stop").textContent).toContain("129.50");
      expect(screen.getByTestId("cell-target").textContent).toContain("148.00");
      expect(screen.getByTestId("preflight-slot-adv")).toBeDefined();
      unmountAdv();
    });
  });

  describe("CMP-02: Adaptive Terminal Mode Switching Invariants", () => {
    it("preserves preflightSlot and execution ladder across seamless mode transitions", () => {
      useExperienceStore.setState({ mode: "GUIDED" });
      const { rerender } = render(
        <AdaptiveTerminal
          symbol="NVDA"
          companyName="NVIDIA Corporation"
          currentPrice={135.25}
          changePct={2.45}
          setupScore={88}
          preflightSlot={<div data-testid="shared-preflight-container">MANDATORY PREFLIGHT CLEARANCE</div>}
        />
      );

      // In Guided Mode
      expect(screen.getByTestId("shared-preflight-container")).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
      expect(screen.getByText(/NVIDIA Corporation/i)).toBeDefined();

      // Switch to Standard Mode
      useExperienceStore.setState({ mode: "STANDARD" });
      rerender(
        <AdaptiveTerminal
          symbol="NVDA"
          companyName="NVIDIA Corporation"
          currentPrice={135.25}
          changePct={2.45}
          setupScore={88}
          preflightSlot={<div data-testid="shared-preflight-container">MANDATORY PREFLIGHT CLEARANCE</div>}
        />
      );

      expect(screen.getByTestId("shared-preflight-container")).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
      expect(screen.getByTestId("card-symbol").textContent).toBe("NVDA");

      // Switch to Quant Mode
      useExperienceStore.setState({ mode: "QUANT" });
      rerender(
        <AdaptiveTerminal
          symbol="NVDA"
          companyName="NVIDIA Corporation"
          currentPrice={135.25}
          changePct={2.45}
          setupScore={88}
          preflightSlot={<div data-testid="shared-preflight-container">MANDATORY PREFLIGHT CLEARANCE</div>}
        />
      );

      expect(screen.getByTestId("shared-preflight-container")).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
      expect(screen.getByTestId("card-symbol").textContent).toBe("NVDA");
    });
  });

  describe("CMP-03: Presentation Distinction & Value Preservation", () => {
    it("Guided provides plain-language guidance while preserving decision parity", () => {
      render(<GuidedTerminalView insight={mockCanonicalInsight} />);
      expect(screen.getByText(/Leader in AI semiconductor design showing clean structural base/i)).toBeDefined();
      expect(screen.getByText(/Step-by-Step Stock Analysis Walkthrough/i)).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
    });

    it("Standard provides confluence breakdown while preserving decision parity", () => {
      render(<StandardTerminalView insight={mockCanonicalInsight} />);
      expect(screen.getByText(/Confluence Breakdown/i)).toBeDefined();
      expect(screen.getByText(/Institutional accumulation confirmed/i)).toBeDefined();
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
    });

    it("Quant provides quantitative metrics ribbon while preserving decision parity", () => {
      render(<AdvancedTerminalView insight={mockCanonicalInsight} />);
      expect(screen.getByText(/Quant Confluence Factors/i)).toBeDefined();
      expect(screen.getByText(/64.2/i)).toBeDefined(); // RSI
      expect(screen.getByTestId("cockpit-execution-ladder")).toBeDefined();
    });
  });

  describe("CMP-04: Actionability Fail-Closed Invariant Across Modes", () => {
    it("all 3 modes display NOT ACTIONABLE (STALE TAPE) and disable execution when data is stale", () => {
      const staleInsight: QuantitativeInsight = {
        ...mockCanonicalInsight,
        terminalState: {
          isActionable: false,
          decisionState: "STALE_DATA",
          overallEligibility: "SUSPENDED_UNAVAILABLE",
          freshnessStatus: "STALE_HISTORICAL",
          freshness: {
            status: "STALE_HISTORICAL",
            stalenessDays: 14,
          },
        },
      };

      // 1. Standard
      const { unmount: u1 } = render(<StandardTerminalView insight={staleInsight} />);
      expect(screen.getByText(/HISTORICAL DATA ONLY/i)).toBeDefined();
      expect(screen.getByTestId("actionability-badge").textContent).toContain("NOT ACTIONABLE (STALE TAPE)");
      u1();

      // 2. Guided
      const { unmount: u2 } = render(<GuidedTerminalView insight={staleInsight} />);
      expect(screen.getByText(/HISTORICAL DATA ONLY/i)).toBeDefined();
      expect(screen.getByTestId("actionability-badge").textContent).toContain("NOT ACTIONABLE (STALE TAPE)");
      u2();

      // 3. Quant
      const { unmount: u3 } = render(<AdvancedTerminalView insight={staleInsight} />);
      expect(screen.getByText(/HISTORICAL DATA ONLY/i)).toBeDefined();
      expect(screen.getByTestId("actionability-badge").textContent).toContain("NOT ACTIONABLE (STALE TAPE)");
      u3();
    });
  });
});
