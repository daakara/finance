import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import fs from "fs";
import path from "path";
import StandardTerminalView from "../terminal/StandardTerminalView";
import AdaptiveTerminal from "../AdaptiveTerminal";
import { QuantitativeInsight } from "../../types/insight";
import { isDecisionActionable, DecisionState } from "../../types/decisionContract";

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
        stopLossPct: "-5.1%",
        target1: 198.0,
        target1Pct: "+7.9%",
        target2: 210.0,
        target2Pct: "+14.4%",
        profitRiskRatio: 2.15,
      },
    },
    advanced: {
      rsi: 54.2,
      ema20: 183.1,
      sma50: 178.0,
      atr: 4.8,
      rvol: 1.25,
      beta: 1.42,
      marketCap: "$3.1T",
      peRatio: 42.1,
      roic: 48.5,
      debtToEquity: 0.28,
      relativeStrengthScore: 92,
      var95Pct: "2.1",
      vcpStage: 3,
    },
    ...overrides,
  };
}

describe("Wave 1 Declutter and Preservation Acceptance Suite", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ── 1. DECLUTTER CHECKS ───────────────────────────────────────────────────
  describe("Wave 1 Declutter Requirements", () => {
    it("AdaptiveTerminal does NOT render duplicate body horizon indicator or explanatory text", () => {
      render(
        <AdaptiveTerminal
          symbol="NVDA"
          currentPrice={184.2}
          changePct={1.45}
          userRole="LONG_TERM"
        />
      );

      // Redundant body horizon indicators must be absent
      expect(screen.queryByText(/Governed by Trading Horizon switch/i)).toBeNull();
      expect(screen.queryByText(/Horizon: SWING/i)).toBeNull();
      expect(screen.queryByText(/Horizon: INTRADAY/i)).toBeNull();
    });

    it("AdaptiveTerminal does NOT render duplicate body depth indicators", () => {
      render(
        <AdaptiveTerminal
          symbol="NVDA"
          currentPrice={184.2}
          changePct={1.45}
        />
      );

      // Redundant body depth strip must be absent
      expect(screen.queryByText(/STANDARD EXPERIENCE/i)).toBeNull();
      expect(screen.queryByText(/GUIDED EXPERIENCE/i)).toBeNull();
      expect(screen.queryByText(/QUANT EXPERIENCE/i)).toBeNull();
    });

    it("page.tsx relocates PageIntro, IntentHero, and WeeklyConfluenceSpotlight below detailed tabs", () => {
      const pageSrc = fs.readFileSync(
        path.join(__dirname, "../../app/page.tsx"),
        "utf-8"
      );

      // Verify PageIntro and IntentHero are NOT rendered before detailed-domain-content
      const detailedTabsIdx = pageSrc.indexOf('data-testid="detailed-domain-content"');
      const pageIntroIdx = pageSrc.indexOf("<PageIntro");
      const intentHeroIdx = pageSrc.indexOf("<IntentHero");
      const spotlightIdx = pageSrc.indexOf("<WeeklyConfluenceSpotlight");

      expect(detailedTabsIdx).toBeGreaterThan(0);
      expect(pageIntroIdx).toBeGreaterThan(detailedTabsIdx);
      expect(intentHeroIdx).toBeGreaterThan(detailedTabsIdx);
      expect(spotlightIdx).toBeGreaterThan(detailedTabsIdx);

      // Verify all callbacks remain intact
      expect(pageSrc).toContain("trackAnalysisToSetup(selectedSymbol)");
      expect(pageSrc).toContain("onSelectSymbol={handleSelectSymbol}");
      expect(pageSrc).toContain('href: "/radar"');
    });
  });

  // ── 2. PRESERVATION & COMPACTION CHECKS ───────────────────────────────────
  describe("Wave 1 Preservation and Compaction Requirements", () => {
    it("AdaptiveTerminal preserves evidence provenance badge", () => {
      render(
        <AdaptiveTerminal
          symbol="NVDA"
          currentPrice={184.2}
          changePct={1.45}
          dataSource="live"
        />
      );

      // Provenance badge remains present and accessible
      const provenanceBadges = screen.getAllByText(/Live|Cataloged|Verified|Delayed/i);
      expect(provenanceBadges.length).toBeGreaterThan(0);
    });

    it("StandardTerminalView renders compact #demo-asset-tag only when isDemo is true", () => {
      const insight = createFixtureInsight();

      // Case A: isDemo = true
      const { rerender, container } = render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          isDemo={true}
        />
      );

      const tag = container.querySelector("#demo-asset-tag");
      expect(tag).not.toBeNull();
      expect(tag?.textContent).toContain("Demo Asset");

      // Case B: isDemo = false
      rerender(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          isDemo={false}
        />
      );

      expect(container.querySelector("#demo-asset-tag")).toBeNull();
    });

    it("AdaptiveTerminal propagates isDemo prop to StandardTerminalView", () => {
      const { container } = render(
        <AdaptiveTerminal
          symbol="NVDA"
          currentPrice={184.2}
          changePct={1.45}
          isDemo={true}
        />
      );

      const tag = container.querySelector("#demo-asset-tag");
      expect(tag).not.toBeNull();
      expect(tag?.textContent).toContain("Demo Asset");
    });

    it("StandardTerminalView retains accessible #why-score-btn and invokes onOpenWhy", () => {
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

    it("StandardTerminalView score badge also triggers onOpenWhy", () => {
      const onOpenWhy = vi.fn();
      const insight = createFixtureInsight({ setupScore: 91 });

      render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={onOpenWhy}
        />
      );

      const scoreBadge = screen.getByText("91/100");
      fireEvent.click(scoreBadge);
      expect(onOpenWhy).toHaveBeenCalledTimes(1);
    });

    it("StandardTerminalView execution CTA triggers onOpenSizer when posture is ACQUIRE", () => {
      const onOpenSizer = vi.fn();
      const insight = createFixtureInsight({
        terminalState: {
          ...createFixtureInsight().terminalState,
          posture: "ACQUIRE",
          isActionable: true,
        },
      });

      render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={onOpenSizer}
          onOpenWhy={vi.fn()}
        />
      );

      const actionBtn = screen.getByText(/Size & Execute Position/i);
      fireEvent.click(actionBtn);
      expect(onOpenSizer).toHaveBeenCalledTimes(1);
    });
  });

  // ── 3. DOMAIN SAFETY & FAIL-CLOSED CHECKS ────────────────────────────────
  describe("Wave 1 Domain Safety and Semantic Invariants", () => {
    it("decisionContract isDecisionActionable remains fail-closed", () => {
      // Incomplete data -> strictly non-actionable
      expect(isDecisionActionable(DecisionState.INSUFFICIENT_DATA, "WAITING_PULLBACK")).toBe(false);

      // Non-actionable status -> strictly false even if state is actionable
      expect(isDecisionActionable(DecisionState.ACTIONABLE_SETUP, "WAITING_PULLBACK")).toBe(false);

      // Actionable state + actionable status -> true
      expect(isDecisionActionable(DecisionState.ACTIONABLE_SETUP, "IN_BUY_ZONE")).toBe(true);
    });
  });

  // ── 4. VISUAL SMOKE VERIFICATION (1440x900 & 390x844) ────────────────────
  describe("Wave 1 Visual Smoke Verification", () => {
    it("WAVE_1_DESKTOP_SMOKE: Primary Analysis content, chart, and plan render cleanly at 1440x900 desktop", () => {
      window.innerWidth = 1440;
      window.innerHeight = 900;

      const chartNode = <div data-testid="smoke-chart">Interactive Lightweight Chart</div>;
      const planNode = <div data-testid="smoke-plan">Optimal Execution Ladder</div>;

      const { container } = render(
        <AdaptiveTerminal
          symbol="NVDA"
          companyName="NVIDIA Corp"
          currentPrice={184.2}
          changePct={1.45}
          setupScore={82}
          dataSource="live"
          chartSlot={chartNode}
          planSlot={planNode}
          userRole="LONG_TERM"
          isDemo={false}
        />
      );

      // Primary analysis content is present
      expect(screen.getByTestId("smoke-chart")).toBeDefined();
      expect(screen.getByTestId("smoke-plan")).toBeDefined();
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("supporting-evidence")).toBeDefined();
      expect(container.querySelector("#why-score-btn")).not.toBeNull();
      // Nothing is visually unmounted or crashing
      expect(container.textContent).toContain("NVDA");
    });

    it("WAVE_1_MOBILE_SMOKE: Primary Analysis content renders and remains accessible at 390x844 mobile", () => {
      window.innerWidth = 390;
      window.innerHeight = 844;

      const chartNode = <div data-testid="mobile-smoke-chart">Mobile Chart</div>;

      const { container } = render(
        <AdaptiveTerminal
          symbol="NVDA"
          companyName="NVIDIA Corp"
          currentPrice={184.2}
          changePct={1.45}
          setupScore={82}
          chartSlot={chartNode}
          userRole="DAY_TRADER"
          isDemo={true}
        />
      );

      // Primary analysis content is present
      expect(screen.getByTestId("mobile-smoke-chart")).toBeDefined();
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(container.querySelector("#demo-asset-tag")).not.toBeNull();
      expect(container.querySelector("#why-score-btn")).not.toBeNull();
      expect(container.textContent).toContain("NVDA");
    });
  });
});
