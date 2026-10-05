import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import StandardTerminalView from "../terminal/StandardTerminalView";
import GuidedTerminalView from "../terminal/GuidedTerminalView";
import AdvancedTerminalView from "../terminal/AdvancedTerminalView";
import OptimalEntryExitCard from "../OptimalEntryExitCard";
import DayTraderPositionSizer from "../DayTraderPositionSizer";
import ExecutionCorridor from "../workstation/ExecutionCorridor";
import { QuantitativeInsight } from "../../types/insight";
import { OptimalExecutionPlan } from "../../lib/api";
import { deriveUnmetConditions } from "../../lib/decisionHierarchyUtils";

const mockAnalyticsData: any = {
  symbol: "NVDA",
  currentPrice: 184.2,
  technicals: {
    rsi14: 55.4,
    vwap: 183.5,
    ema20: 182.0,
    sma50: 178.0,
    atr: 4.8,
  },
  analytics: {
    advanced_metrics: {
      daily_volatility: 0.025,
      intraday_atr: 4.8,
    },
  },
};

// Sample canonical execution plan
const mockPlan: OptimalExecutionPlan = {
  optimal_entry_min: 182.5,
  optimal_entry_max: 185.0,
  stop_loss: 174.0,
  stop_loss_pct: -5.1,
  take_profit_1: 198.0,
  take_profit_1_pct: 7.9,
  take_profit_2: 210.0,
  take_profit_2_pct: 14.4,
  risk_reward_ratio: 2.15,
  setup_pattern: "Minervini VCP",
  entry_thesis: "20 EMA pullback test with declining volume",
  invalidation_condition: "Close below 174.00 invalidates base structure",
  stage_phase: "Stage 2 Breakout Base",
  current_price: 184.2,
};

// Builder for deterministic QuantitativeInsight fixtures
function createFixtureInsight(overrides: Partial<QuantitativeInsight> = {}): QuantitativeInsight {
  return {
    id: "NVDA_SWING_20261004",
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
        calculatedAt: "2026-10-04T00:00:00Z",
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
      assessmentHeadline: "Constructive Consolidation Awaiting Pivot",
      assessmentDescription: "NVDA is forming an orderly pullback within its base. Wait for confirmation before committing fresh capital.",
      whyPills: [
        { category: "Price Trend", status: "Healthy", description: "Constructive 20 EMA test", sentiment: "positive" },
        { category: "Company Health", status: "Healthy", description: "Robust free cash flow", sentiment: "positive" },
      ],
      reclaimMilestone: "Needs volume breakout above $186.00 with expansion.",
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

describe("ARX Terminal Analysis Decision Hierarchy Acceptance Suite", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    vi.clearAllMocks();
  });

  // ── TEST-001 & TEST-002: Decision Hierarchy Viewport Order ───────────────────
  describe("TEST-001 & TEST-002: Strict DOM and Viewport Order", () => {
    it("renders strictly: verdict -> reason -> unmet condition -> chart -> conditional plan -> supporting evidence", () => {
      const insight = createFixtureInsight();
      const chartNode = <div id="chart-inner">Chart Content</div>;
      const planNode = <OptimalEntryExitCard symbol="NVDA" executionPlan={mockPlan} isActionable={false} />;

      const { container } = render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          chartSlot={chartNode}
          planSlot={planNode}
        />
      );

      const verdictEl = screen.getByTestId("decision-verdict");
      const reasonEl = screen.getByTestId("decision-reason");
      const unmetEl = screen.getByTestId("unmet-condition");
      const chartEl = screen.getByTestId("market-workspace-chart");
      const planEl = screen.getByTestId("conditional-trade-plan");
      const evidenceEl = screen.getByTestId("supporting-evidence");

      // Verify DOM document positions (preceding < following)
      expect(verdictEl.compareDocumentPosition(reasonEl) & Node.DOCUMENT_POSITION_CONTAINED_BY ||
             verdictEl.compareDocumentPosition(reasonEl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

      expect(verdictEl.compareDocumentPosition(unmetEl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(unmetEl.compareDocumentPosition(chartEl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(chartEl.compareDocumentPosition(planEl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(planEl.compareDocumentPosition(evidenceEl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });

    it("ensures verdict dominates visual hierarchy over setup score", () => {
      const insight = createFixtureInsight();
      render(
        <StandardTerminalView
          insight={insight}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
        />
      );

      const verdictEl = screen.getByTestId("decision-verdict");
      const evidenceEl = screen.getByTestId("supporting-evidence");

      // Verdict appears before supporting evidence containing setup score
      expect(verdictEl.compareDocumentPosition(evidenceEl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(screen.getByText(/WAIT — CONFIRMATION PENDING/i)).toBeDefined();
    });
  });

  // ── TEST-003: Score Subordination (Score Cannot Override Verdict) ───────────
  describe("TEST-003: Score Subordination", () => {
    it("high score does not render actionable state when canonical assessment is WAIT", () => {
      // Fixture: high score 94/100 but non-actionable
      const highPending = createFixtureInsight({
        setupScore: 94,
        verdict: "WAIT_FOR_TRIGGER",
        verdictLabel: "WAIT — HIGH CONFLUENCE AWAITING PIVOT",
        terminalState: {
          ...createFixtureInsight().terminalState,
          isActionable: false,
          decisionState: "VALID_SETUP",
        },
      });

      render(
        <StandardTerminalView
          insight={highPending}
          onOpenSizer={vi.fn()}
          onOpenWhy={vi.fn()}
          planSlot={<OptimalEntryExitCard symbol="NVDA" executionPlan={mockPlan} isActionable={false} />}
        />
      );

      // Verdict remains WAIT
      expect(screen.getByText(/WAIT FOR TRIGGER/i)).toBeDefined();
      expect(screen.queryByText(/^ACTIONABLE$/i)).toBeNull();

      // Conditional trade plan renders CURRENT ACTION: Wait
      expect(screen.getByText(/CURRENT ACTION: Wait/i)).toBeDefined();
      expect(screen.queryByText(/CURRENT ACTION: Execute Setup/i)).toBeNull();
    });
  });

  // ── TEST-004: Conditional Trade Plan (Wait vs Actionable States) ──────────────
  describe("TEST-004: Conditional Trade Plan Rendering", () => {
    it("renders CURRENT ACTION: Wait and IF CONFIRMATION OCCURS for non-actionable fixture", () => {
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={mockPlan}
          isActionable={false}
          decisionState="WAIT_FOR_TRIGGER"
        />
      );

      expect(screen.getByTestId("current-action").textContent).toContain("CURRENT ACTION: Wait");
      expect(screen.getByTestId("plan-status").textContent).toContain("Status: WAIT_FOR_TRIGGER");
      expect(screen.getByTestId("conditional-confirmation-block")).toBeDefined();
      expect(screen.getByText(/IF CONFIRMATION OCCURS:/i)).toBeDefined();
    });

    it("renders CURRENT ACTION: Execute Setup and suppresses confirmation block when actionable", () => {
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={mockPlan}
          isActionable={true}
          decisionState="ACTIONABLE_SETUP"
        />
      );

      expect(screen.getByTestId("current-action").textContent).toContain("CURRENT ACTION: Execute Setup");
      expect(screen.queryByTestId("conditional-confirmation-block")).toBeNull();
    });
  });

  // ── TEST-005: Spatial Corridor vs Confirmation Trigger Separation ───────────
  describe("TEST-005: Spatial Corridor vs Event Trigger Distinction", () => {
    it("semantically separates spatial accumulation corridor from market event trigger", () => {
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={mockPlan}
          isActionable={false}
        />
      );

      expect(screen.getByText(/1. Spatial Location \(Price Corridor\):/i)).toBeDefined();
      expect(screen.getAllByText(/\$182.50 – \$185.00/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/2. Market Event Trigger:/i)).toBeDefined();
      expect(screen.getByText(/Awaiting volume breakout \/ confirmation candle/i)).toBeDefined();
    });

    it("deriveUnmetConditions creates distinct CORRIDOR and TRIGGER conditions", () => {
      const insight = createFixtureInsight();
      const conditions = deriveUnmetConditions(insight);

      const corridorCond = conditions.find((c) => c.category === "CORRIDOR");
      const triggerCond = conditions.find((c) => c.category === "TRIGGER");

      expect(corridorCond).toBeDefined();
      expect(triggerCond).toBeDefined();
      expect(corridorCond?.title).toContain("Corridor");
      expect(triggerCond?.title).toContain("Trigger");
    });
  });

  // ── TEST-006: Presentation Mode Invariance (MODE-001 through MODE-008) ──────
  describe("TEST-006: Presentation Mode Invariance", () => {
    it("Guided, Standard, and Quant modes consume identical canonical fields", () => {
      const insight = createFixtureInsight();

      // Render Guided
      const { unmount: unmountGuided } = render(
        <GuidedTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("decision-reason")).toBeDefined();
      expect(screen.getByTestId("unmet-condition")).toBeDefined();
      expect(screen.getByText(/Canonical Verdict: WAIT — CONFIRMATION PENDING/i)).toBeDefined();
      unmountGuided();

      // Render Standard
      const { unmount: unmountStandard } = render(
        <StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("decision-reason")).toBeDefined();
      expect(screen.getByTestId("unmet-condition")).toBeDefined();
      expect(screen.getByText(/WAIT — CONFIRMATION PENDING/i)).toBeDefined();
      unmountStandard();

      // Render Quant
      render(
        <AdvancedTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );
      expect(screen.getByTestId("decision-verdict")).toBeDefined();
      expect(screen.getByTestId("decision-reason")).toBeDefined();
      expect(screen.getByTestId("unmet-condition")).toBeDefined();
      expect(screen.getByText(/WAIT — CONFIRMATION PENDING/i)).toBeDefined();
    });
  });

  // ── TEST-007 & TEST-008: Execution Levels Parity & Missing Data ─────────────
  describe("TEST-007 & TEST-008: Level Parity and Missing Data Safety", () => {
    it("renders optimal execution values accurately without mutation", () => {
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={mockPlan}
          isActionable={false}
        />
      );

      expect(screen.getByText(/\$174.00/i)).toBeDefined(); // stop loss
      expect(screen.getAllByText(/\$198.00/i).length).toBeGreaterThan(0); // take profit 1
      expect(screen.getAllByText(/\$210.00/i).length).toBeGreaterThan(0); // take profit 2
      expect(screen.getByText(/2.15 : 1.0/i)).toBeDefined(); // R:R
    });

    it("renders safe unresolved state when plan values are missing", () => {
      const emptyPlan: OptimalExecutionPlan = {
        optimal_entry_min: null as any,
        optimal_entry_max: null as any,
        stop_loss: null as any,
        take_profit_1: null as any,
        take_profit_2: null as any,
        current_price: 0,
        risk_reward_ratio: null as any,
        setup_pattern: "Unverified",
      };

      render(
        <OptimalEntryExitCard
          symbol="UNKNOWN"
          executionPlan={emptyPlan}
        />
      );

      expect(screen.getByText(/Execution Setup Unavailable/i)).toBeDefined();
      expect(screen.queryByText(/CURRENT ACTION: Execute Setup/i)).toBeNull();
    });
  });

  // ── Semantic Regression Matrix Across 6 Canonical Fixture States ───────────
  describe("Semantic Regression: 6 Canonical Fixture States", () => {
    it("State 1: UNVERIFIED renders safe unverified state", () => {
      const insight = createFixtureInsight({
        verdictLabel: "UNVERIFIED ASSET",
        terminalState: {
          ...createFixtureInsight().terminalState,
          decisionState: "UNVERIFIED",
          overallEligibility: "INELIGIBLE",
          isActionable: false,
          canSizeTrade: false,
          uiStateLabel: "Unverified Asset Structure",
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getAllByText(/UNVERIFIED ASSET/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/WAIT FOR TRIGGER/i)).toBeDefined();
    });

    it("State 2: INSUFFICIENT_DATA omits synthesized values", () => {
      const insight = createFixtureInsight({
        verdictLabel: "INSUFFICIENT HISTORICAL DATA",
        terminalState: {
          ...createFixtureInsight().terminalState,
          decisionState: "INSUFFICIENT_DATA",
          overallEligibility: "INELIGIBLE",
          isActionable: false,
          canSizeTrade: false,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getByText(/INSUFFICIENT HISTORICAL DATA/i)).toBeDefined();
    });

    it("State 3: STALE_DATA displays stale state without presenting as live", () => {
      const insight = createFixtureInsight({
        verdictLabel: "STALE HISTORICAL REGIME",
        terminalState: {
          ...createFixtureInsight().terminalState,
          decisionState: "STALE_DATA",
          overallEligibility: "LIMITED",
          isActionable: false,
          canSizeTrade: false,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getByText(/STALE HISTORICAL REGIME/i)).toBeDefined();
    });

    it("State 4: EVIDENCE_INCOMPLETE renders partial evidence disclaimer", () => {
      const insight = createFixtureInsight({
        verdictLabel: "PARTIAL CONFLUENCE",
        terminalState: {
          ...createFixtureInsight().terminalState,
          decisionState: "EVIDENCE_INCOMPLETE",
          overallEligibility: "LIMITED",
          isActionable: false,
          canSizeTrade: false,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getAllByText(/PARTIAL CONFLUENCE/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/Partial/i).length).toBeGreaterThan(0);
    });

    it("State 5: VALID_SETUP renders constructive non-actionable setup", () => {
      const insight = createFixtureInsight({
        verdictLabel: "CONSTRUCTIVE BASE FORMATION",
        terminalState: {
          ...createFixtureInsight().terminalState,
          decisionState: "VALID_SETUP",
          overallEligibility: "ELIGIBLE",
          isActionable: false,
          canSizeTrade: false,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getByText(/CONSTRUCTIVE BASE FORMATION/i)).toBeDefined();
      expect(screen.getByText(/WAIT FOR TRIGGER/i)).toBeDefined();
    });

    it("State 6: ACTIONABLE_SETUP renders ACTIONABLE badge and execution state", () => {
      const insight = createFixtureInsight({
        verdictLabel: "CONFIRMED ACCUMULATION BREAKOUT",
        terminalState: {
          ...createFixtureInsight().terminalState,
          decisionState: "ACTIONABLE_SETUP",
          overallEligibility: "ELIGIBLE",
          isActionable: true,
          canSizeTrade: true,
          posture: "ACQUIRE",
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getByText(/CONFIRMED ACCUMULATION BREAKOUT/i)).toBeDefined();
      expect(screen.getByText(/^ACTIONABLE$/i)).toBeDefined();
      expect(screen.getByText(/Size & Execute Position/i)).toBeDefined();
    });
  });

  // ── ANALYSIS CORRIDOR & TRIGGER SEMANTICS INVARIANTS ─────────────────────────
  describe("ARX Analysis Corridor & Trigger Semantics Invariants (Gate Re-Attestation)", () => {
    it("INV-ANALYSIS-01 & 02: deriveUnmetConditions rejects corridor reconstruction from stopLoss/SMA50 and consumes canonical bounds", () => {
      // Case A: Insight with canonical entryMin/Max
      const canonicalInsight = createFixtureInsight({
        price: 184.2,
        standard: {
          ...createFixtureInsight().standard,
          keyLevels: {
            ...createFixtureInsight().standard.keyLevels,
            entryMin: 182.5,
            entryMax: 185.0,
            watchZone: "$182.50 – $185.00",
            stopLoss: 174.0,
            sma50: 178.0,
          },
        },
      });

      const conditionsA = deriveUnmetConditions(canonicalInsight);
      const corridorCondA = conditionsA.find((c) => c.id === "corridor");
      expect(corridorCondA).toBeDefined();
      expect(corridorCondA?.status).toBe("MET");
      expect(corridorCondA?.description).toContain("$182.50 – $185.00");

      // Case B: Price above canonical entryMax
      const extendedInsight = createFixtureInsight({
        price: 190.0,
        standard: {
          ...createFixtureInsight().standard,
          keyLevels: {
            ...createFixtureInsight().standard.keyLevels,
            entryMin: 182.5,
            entryMax: 185.0,
            watchZone: "$182.50 – $185.00",
            stopLoss: 174.0,
            sma50: 178.0,
          },
        },
      });
      const conditionsB = deriveUnmetConditions(extendedInsight);
      const corridorCondB = conditionsB.find((c) => c.id === "corridor");
      expect(corridorCondB?.status).toBe("UNMET");
      expect(corridorCondB?.title).toContain("Pullback into Accumulation Corridor");

      // Case C: Missing canonical corridor does NOT synthesize bounds from stopLoss * 1.02 or sma50
      const missingCorridorInsight = createFixtureInsight({
        price: 184.2,
        standard: {
          ...createFixtureInsight().standard,
          keyLevels: {
            ...createFixtureInsight().standard.keyLevels,
            entryMin: undefined,
            entryMax: undefined,
            watchZone: "Unavailable",
            stopLoss: 174.0,
            sma50: 178.0,
          },
        },
      });
      const conditionsC = deriveUnmetConditions(missingCorridorInsight);
      const corridorCondC = conditionsC.find((c) => c.id === "corridor");
      expect(corridorCondC?.status).toBe("UNMET");
      expect(corridorCondC?.title).toBe("Accumulation Corridor Unavailable");
      expect(corridorCondC?.description).toContain("unavailable for this asset");
    });

    it("INV-ANALYSIS-01: OptimalEntryExitCard renders canonical backend corridor min/max unchanged", () => {
      render(
        <OptimalEntryExitCard
          symbol="NVDA"
          executionPlan={mockPlan}
          isActionable={false}
        />
      );

      // Verify canonical corridor bounds render
      expect(screen.getAllByText(/\$182\.50/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/\$185\.00/).length).toBeGreaterThan(0);
    });

    it("INV-ANALYSIS-03: Noncanonical numeric trigger price is absent across terminal views", () => {
      const insight = createFixtureInsight();
      const { container } = render(
        <StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );

      // Verify no element declares a numeric "Trigger: $..." price
      const allText = container.textContent || "";
      expect(allText).not.toMatch(/Trigger Price:\s*\$\d+/i);
      expect(allText).not.toMatch(/Numeric Trigger:\s*\$\d+/i);
      expect(allText).not.toMatch(/Trigger:\s*\$18/i);
    });

    it("INV-ANALYSIS-05: Canonical awaiting-readiness state renders correctly without synthetic domain trigger", () => {
      const insight = createFixtureInsight({
        terminalState: {
          ...createFixtureInsight().terminalState,
          isActionable: false,
          uiStateLabel: "Awaiting Trigger Confluence",
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getByText(/WAIT FOR TRIGGER/i)).toBeDefined();
    });

    it("INV-ANALYSIS-06: Actionability CTA follows canonical decision contract strictly", () => {
      // Non-actionable setup
      const nonActionableInsight = createFixtureInsight({
        terminalState: {
          ...createFixtureInsight().terminalState,
          isActionable: false,
          canSizeTrade: false,
        },
      });
      const { rerender } = render(
        <StandardTerminalView insight={nonActionableInsight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );
      expect(screen.queryByText(/Size & Execute Position/i)).toBeNull();

      // Actionable setup
      const actionableInsight = createFixtureInsight({
        verdictLabel: "CONFIRMED ACCUMULATION BREAKOUT",
        terminalState: {
          ...createFixtureInsight().terminalState,
          decisionState: "ACTIONABLE_SETUP",
          isActionable: true,
          canSizeTrade: true,
          posture: "ACQUIRE",
        },
      });
      rerender(<StandardTerminalView insight={actionableInsight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);
      expect(screen.getByText(/Size & Execute Position/i)).toBeDefined();
    });

    it("INV-ANALYSIS-07: SHORT control cannot activate unsupported calculations in DayTraderPositionSizer", () => {
      render(
        <DayTraderPositionSizer
          symbol="NVDA"
          data={mockAnalyticsData}
        />
      );

      const shortBtn = screen.getByRole("radio", { name: /SELL \/ SHORT/i });
      expect(shortBtn).toBeDefined();
      expect(shortBtn.getAttribute("disabled")).not.toBeNull();
      expect(shortBtn.getAttribute("aria-disabled")).toBe("true");
      expect(shortBtn.getAttribute("title")).toContain("Short analysis is not supported");
      expect(screen.getByText(/BUY \/ LONG/i).getAttribute("aria-checked")).toBe("true");
    });

    it("INV-ANALYSIS-08: Distance wording agrees with numerical sign without double negatives or pivot conflation", () => {
      // Case 1: Below corridor (96.8 spot vs 100.0 entryLow) -> positive distance, no "-3.2% below pivot"
      const { rerender, container } = render(
        <ExecutionCorridor
          ticker="NVDA"
          spotPrice={96.8}
          entryLow={100.0}
          entryHigh={105.0}
          stopLoss={92.0}
          target1={115.0}
          target2={125.0}
          riskRewardRatio={2.5}
          executionState="WAITING_PULLBACK"
          advShareLimit={5000}
          onOpenPositionSizer={vi.fn()}
        />
      );

      const textBelow = container.textContent || "";
      expect(textBelow).toContain("3.2% below corridor");
      expect(textBelow).not.toContain("-3.2%");
      expect(textBelow).not.toContain("below pivot");

      // Case 2: Above corridor (110.0 spot vs 105.0 entryHigh)
      rerender(
        <ExecutionCorridor
          ticker="NVDA"
          spotPrice={110.0}
          entryLow={100.0}
          entryHigh={105.0}
          stopLoss={92.0}
          target1={115.0}
          target2={125.0}
          riskRewardRatio={2.5}
          executionState="WAITING_PULLBACK"
          advShareLimit={5000}
          onOpenPositionSizer={vi.fn()}
        />
      );
      const textAbove = container.textContent || "";
      expect(textAbove).toContain("+4.8% above corridor");

      // Case 3: Inside corridor (102.0 spot)
      rerender(
        <ExecutionCorridor
          ticker="NVDA"
          spotPrice={102.0}
          entryLow={100.0}
          entryHigh={105.0}
          stopLoss={92.0}
          target1={115.0}
          target2={125.0}
          riskRewardRatio={2.5}
          executionState="IN_BUY_ZONE"
          advShareLimit={5000}
          onOpenPositionSizer={vi.fn()}
        />
      );
      const textInside = container.textContent || "";
      expect(textInside).toContain("In Buy Zone");
    });
  });
});
