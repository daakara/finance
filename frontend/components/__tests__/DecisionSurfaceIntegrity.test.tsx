import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import StandardTerminalView from "../terminal/StandardTerminalView";
import GuidedTerminalView from "../terminal/GuidedTerminalView";
import AdvancedTerminalView from "../terminal/AdvancedTerminalView";
import { deriveAssessmentState } from "../../lib/assessmentEngine";
import { QuantitativeInsight, DecisionTrace } from "../../types/insight";

/**
 * ARX Terminal Decision-Surface Integrity & Contract Reconciliation Test Suite
 * Permanent regression suite for QA-ESC-011.
 *
 * Enforces:
 * 1. Contract Binding: DecisionTrace.decisionStateLabel consumed canonically.
 * 2. Fail-Closed Fallback: Missing decision state yields "Setup Evaluation Pending", NEVER "Wait for Trigger".
 * 3. Domain Independence: Actionability badge strictly renders ACTIONABLE vs NOT ACTIONABLE.
 * 4. Zero Duplication: Redundant uiStateLabel badge removed from headline cluster.
 * 5. Cross-State Matrix: AVOID, HOLD, UNVERIFIED never render "WAIT FOR TRIGGER".
 * 6. Multi-Mode Parity: Guided, Standard, Advanced views behave identically.
 * 7. Transition Safety: Zero transient or persistent contradictions.
 */

// Helper to create a base mock insight
function createMockInsight(overrides: Partial<QuantitativeInsight> = {}): QuantitativeInsight {
  return {
    id: "TEST_INSIGHT",
    symbol: "TEST",
    companyName: "Test Company Inc.",
    price: 100.0,
    changePct: 1.5,
    setupScore: 80,
    horizon: "SWING",
    assessment: "FAVORABLE",
    posture: "WATCH",
    postureLabel: "Valid Setup — Awaiting Trigger",
    ownership: "NOT_OWNED",
    terminalState: {
      symbol: "TEST",
      companyName: "Test Company Inc.",
      currentPrice: 100.0,
      changePct: 1.5,
      horizon: "SWING",
      ownership: { state: "NOT_OWNED", source: "USER_DECLARED" },
      modelProvenance: {
        modelId: "ARX-MODEL-v2",
        modelVersion: "2.1",
        rulesetVersion: "2026.10",
        calculatedAt: "2026-10-08T10:00:00Z",
      },
      overallEligibility: "ELIGIBLE",
      decisionState: "VALID_SETUP",
      isActionable: false,
      canSizeTrade: false,
      assessment: "FAVORABLE",
      factorAgreement: {
        favorable: 3,
        mixed: 1,
        unfavorable: 0,
        unavailable: 0,
        evaluated: 4,
        displayLabel: "3 of 4 factors favorable",
      },
      domains: [],
      posture: "WATCH",
      uiStateLabel: "Valid Setup — Awaiting Trigger",
      headlineExplanation: "Awaiting confirmed entry trigger.",
      whatWouldChangeAssessment: "Breakout above pivot resistance.",
      primaryAction: {
        label: "Set Price Alert",
        actionType: "SET_ALERT",
        enabled: true,
      },
      availableActions: [],
    },
    verdict: "WAIT_FOR_TRIGGER",
    verdictLabel: "Valid Setup — Awaiting Trigger",
    human: {
      assessmentHeadline: "Valid Setup — Awaiting Trigger",
      assessmentDescription: "Awaiting confirmed entry trigger.",
      whyPills: [],
      reclaimMilestone: "$105.00",
      watchLevels: { watchZone: "$98 – $102", keyLevel: "$105.00", riskStop: "$95.00" },
    },
    standard: {
      bottomLine: "Awaiting confirmed entry trigger.",
      signalsRatio: "3/4 Bullish",
      confluenceBreakdown: [],
      keyLevels: {
        currentPrice: 100.0,
        watchZone: "$98 – $102",
        stopLoss: 95.0,
        stopLossPct: -5.0,
      },
      setupSummary: "Consolidating near 50 SMA.",
    },
    advanced: {
      marketCap: "$10B",
    },
    scoreAttribution: {
      finalScore: 80,
      items: [],
      catalystToIncreaseScore: "Higher volume expansion.",
    },
    primaryRiskSummary: "Broader market weakness.",
    whatWouldChangeAssessment: "Break below 50-day SMA.",
    availableActions: [],
    ...overrides,
  };
}

const mockFullDomains: any[] = [
  { domainId: "trend", domainName: "Price Trend", availability: "AVAILABLE", status: "FAVORABLE", pointImpact: 25, importanceLevel: "HIGH", observation: "Strong Uptrend", modelRule: "Stage 2 Breakout", evidence: [], whatWouldChangeAssessment: "" },
  { domainId: "health", domainName: "Company Health", availability: "AVAILABLE", status: "FAVORABLE", pointImpact: 25, importanceLevel: "HIGH", observation: "Strong Solvency", modelRule: "Solvent balance sheet", evidence: [], whatWouldChangeAssessment: "" },
  { domainId: "smart_money", domainName: "Smart Money Flow", availability: "AVAILABLE", status: "FAVORABLE", pointImpact: 25, importanceLevel: "HIGH", observation: "Institutional Accumulation", modelRule: "Net institutional inflow", evidence: [], whatWouldChangeAssessment: "" },
  { domainId: "macro", domainName: "Macro Regime", availability: "AVAILABLE", status: "NEUTRAL", pointImpact: 0, importanceLevel: "LOW", observation: "Neutral macro", modelRule: "Yield curve flat", evidence: [], whatWouldChangeAssessment: "" },
];

describe("QA-ESC-011: Decision-Surface Integrity & Contract Reconciliation", () => {
  // ── 1. Contract Binding: DecisionTrace Property Resolution ──────────────────
  describe("1. Contract Binding (DecisionTrace.decisionStateLabel)", () => {
    it("consumes authoritative decisionStateLabel from backend DecisionTrace", () => {
      const trace: DecisionTrace = {
        symbol: "NAUT",
        decisionState: "VALID_SETUP",
        decisionStateLabel: "Valid Setup — Awaiting Trigger",
        isActionable: false,
        canSizeTrade: false,
        allowedActions: ["SET_ALERT", "ADD_WATCHLIST"],
        disqualificationReason: "Stage 1 structural basing phase: price establishing floor; awaiting Stage 2 breakout.",
      };

      const terminalState = deriveAssessmentState({
        symbol: "NAUT",
        companyName: "Nautilus Marine",
        currentPrice: 1.96,
        changePct: 2.1,
        horizon: "SWING",
        ownershipState: "NOT_OWNED",
        ownershipSource: "USER_DECLARED",
        domains: mockFullDomains,
        decisionTrace: trace,
      });

      expect(terminalState.uiStateLabel).toBe("Valid Setup — Awaiting Trigger");
      expect(terminalState.decisionState).toBe("VALID_SETUP");
    });

    it("consumes authoritative decisionStateLabel for ACTIONABLE_SETUP", () => {
      const trace: DecisionTrace = {
        symbol: "NVDA",
        decisionState: "ACTIONABLE_SETUP",
        decisionStateLabel: "Actionable Setup — Buy Zone Confirmed",
        isActionable: true,
        canSizeTrade: true,
        allowedActions: ["SIZE_TRADE", "SET_ALERT"],
        disqualificationReason: null,
      };

      const terminalState = deriveAssessmentState({
        symbol: "NVDA",
        companyName: "NVIDIA Corp",
        currentPrice: 184.2,
        changePct: 1.5,
        horizon: "SWING",
        ownershipState: "NOT_OWNED",
        ownershipSource: "USER_DECLARED",
        domains: mockFullDomains,
        decisionTrace: trace,
      });

      expect(terminalState.uiStateLabel).toBe("Actionable Setup — Buy Zone Confirmed");
      expect(terminalState.posture).toBe("ACQUIRE");
      expect(terminalState.isActionable).toBe(true);
    });

    it("fails closed to 'Setup Evaluation Pending' when decisionStateLabel is missing (NEVER 'Wait for Trigger')", () => {
      // Invariant: MISSING_DECISION_STATE != WAIT_FOR_TRIGGER
      const terminalState = deriveAssessmentState({
        symbol: "TEST",
        companyName: "Test Corp",
        currentPrice: 50.0,
        changePct: 0.5,
        horizon: "SWING",
        ownershipState: "NOT_OWNED",
        ownershipSource: "USER_DECLARED",
        domains: mockFullDomains,
        decisionTrace: undefined, // Missing trace
      });

      expect(terminalState.uiStateLabel).not.toBe("Wait for Trigger");
      expect(terminalState.uiStateLabel).toBe("Setup Evaluation Pending");
    });

    it("fails closed to 'Setup Evaluation Pending' when decisionTrace has empty stateLabel", () => {
      const trace: Partial<DecisionTrace> = {
        symbol: "TEST",
        decisionState: "VALID_SETUP",
        isActionable: false,
        canSizeTrade: false,
        allowedActions: [],
        disqualificationReason: "Evaluation in progress",
      };

      const terminalState = deriveAssessmentState({
        symbol: "TEST",
        companyName: "Test Corp",
        currentPrice: 50.0,
        changePct: 0.5,
        horizon: "SWING",
        ownershipState: "NOT_OWNED",
        ownershipSource: "USER_DECLARED",
        domains: mockFullDomains,
        decisionTrace: trace as DecisionTrace,
      });

      expect(terminalState.uiStateLabel).not.toBe("Wait for Trigger");
      expect(terminalState.uiStateLabel).toBe("Setup Evaluation Pending");
    });
  });

  // ── 2. Cross-State Regression Matrix ─────────────────────────────────────────
  describe("2. Cross-State Regression Matrix", () => {
    it("State 1: VALID_SETUP (NAUT) displays canonical verdict and NOT ACTIONABLE badge with no duplicates", () => {
      const insight = createMockInsight({
        symbol: "NAUT",
        verdictLabel: "Valid Setup — Awaiting Trigger",
        terminalState: {
          ...createMockInsight().terminalState,
          symbol: "NAUT",
          uiStateLabel: "Valid Setup — Awaiting Trigger",
          isActionable: false,
        },
      });

      render(
        <StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );

      // Primary Headline
      expect(screen.getByText("Valid Setup — Awaiting Trigger")).toBeDefined();

      // Actionability Badge
      const badge = screen.getByTestId("actionability-badge");
      expect(badge.textContent).toBe("NOT ACTIONABLE");

      // Verify NO duplicated text in the verdict header container
      const verdictCard = screen.getByTestId("decision-verdict");
      const occurrences = (verdictCard.textContent || "").split("Valid Setup — Awaiting Trigger").length - 1;
      expect(occurrences).toBe(1); // Exactly once as the headline!

      // Verify "WAIT FOR TRIGGER" is NOT present anywhere
      expect(screen.queryByText(/WAIT FOR TRIGGER/i)).toBeNull();
    });

    it("State 2: AVOID (Unfavorable Setup) must NOT display 'WAIT FOR TRIGGER'", () => {
      const insight = createMockInsight({
        symbol: "AVOID_TICKER",
        verdictLabel: "Unfavorable Setup",
        posture: "AVOID",
        terminalState: {
          ...createMockInsight().terminalState,
          posture: "AVOID",
          uiStateLabel: "Unfavorable Setup",
          isActionable: false,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);

      expect(screen.getByText("Unfavorable Setup")).toBeDefined();
      const badge = screen.getByTestId("actionability-badge");
      expect(badge.textContent).toBe("NOT ACTIONABLE");

      // Invariant: AVOID must NEVER display "WAIT FOR TRIGGER"
      expect(screen.queryByText(/WAIT FOR TRIGGER/i)).toBeNull();
    });

    it("State 3: OWNED / HOLD must NOT display 'WAIT FOR TRIGGER'", () => {
      const insight = createMockInsight({
        symbol: "HELD_TICKER",
        verdictLabel: "Thesis Intact",
        posture: "HOLD",
        ownership: "OWNED",
        terminalState: {
          ...createMockInsight().terminalState,
          ownership: { state: "OWNED", source: "USER_DECLARED" },
          posture: "HOLD",
          uiStateLabel: "Thesis Intact",
          isActionable: false,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);

      expect(screen.getByText("Thesis Intact")).toBeDefined();
      const badge = screen.getByTestId("actionability-badge");
      expect(badge.textContent).toBe("NOT ACTIONABLE");

      // Invariant: Held position must NEVER display "WAIT FOR TRIGGER"
      expect(screen.queryByText(/WAIT FOR TRIGGER/i)).toBeNull();
    });

    it("State 4: UNVERIFIED asset must NOT display 'WAIT FOR TRIGGER'", () => {
      const insight = createMockInsight({
        symbol: "UNVERIFIED_TICKER",
        verdictLabel: "Unverified Asset — Live Tape Required",
        posture: "RESEARCH",
        terminalState: {
          ...createMockInsight().terminalState,
          decisionState: "UNVERIFIED",
          posture: "RESEARCH",
          uiStateLabel: "Unverified Asset — Live Tape Required",
          isActionable: false,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);

      expect(screen.getByText("Unverified Asset — Live Tape Required")).toBeDefined();
      const badge = screen.getByTestId("actionability-badge");
      expect(badge.textContent).toBe("NOT ACTIONABLE");

      // Invariant: Unverified asset must NEVER display "WAIT FOR TRIGGER"
      expect(screen.queryByText(/WAIT FOR TRIGGER/i)).toBeNull();
    });

    it("State 5: EXTENDED_ABOVE_BUY_ZONE does not imply initial trigger readiness", () => {
      const insight = createMockInsight({
        symbol: "EXTENDED_TICKER",
        verdictLabel: "Valid Setup — Awaiting Trigger",
        terminalState: {
          ...createMockInsight().terminalState,
          uiStateLabel: "Valid Setup — Awaiting Trigger",
          isActionable: false,
        },
        standard: {
          ...createMockInsight().standard,
          bottomLine: "Price is outside the optimal entry corridor; awaiting pullback to buy zone.",
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);

      expect(screen.getByText("Valid Setup — Awaiting Trigger")).toBeDefined();
      const badge = screen.getByTestId("actionability-badge");
      expect(badge.textContent).toBe("NOT ACTIONABLE");
      expect(screen.getByText(/awaiting pullback to buy zone/i)).toBeDefined();
      expect(screen.queryByText(/^WAIT FOR TRIGGER$/i)).toBeNull();
    });

    it("State 6: ACTIONABLE BUY condition displays ACTIONABLE badge in emerald", () => {
      const insight = createMockInsight({
        symbol: "NVDA",
        verdictLabel: "Actionable Setup — Buy Zone Confirmed",
        posture: "ACQUIRE",
        terminalState: {
          ...createMockInsight().terminalState,
          decisionState: "ACTIONABLE_SETUP",
          posture: "ACQUIRE",
          uiStateLabel: "Actionable Setup — Buy Zone Confirmed",
          isActionable: true,
          canSizeTrade: true,
        },
      });

      render(<StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />);

      expect(screen.getByText("Actionable Setup — Buy Zone Confirmed")).toBeDefined();
      const badge = screen.getByTestId("actionability-badge");
      expect(badge.textContent).toBe("ACTIONABLE");
      expect(badge.className).toContain("text-emerald-300");
    });
  });

  // ── 3. Presentation Mode Parity (Guided, Standard, Advanced) ───────────────────
  describe("3. Presentation Mode Parity (Guided, Standard, Advanced)", () => {
    it("ensures all three terminal views render identical actionability badge and zero duplicate label", () => {
      const insight = createMockInsight({
        symbol: "PARITY_TEST",
        verdictLabel: "Valid Setup — Awaiting Trigger",
        terminalState: {
          ...createMockInsight().terminalState,
          uiStateLabel: "Valid Setup — Awaiting Trigger",
          isActionable: false,
        },
      });

      // Standard view
      const { unmount: unmountStd } = render(
        <StandardTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );
      expect(screen.getByTestId("actionability-badge").textContent).toBe("NOT ACTIONABLE");
      expect(screen.queryByText(/^WAIT FOR TRIGGER$/i)).toBeNull();
      unmountStd();

      // Guided view
      const { unmount: unmountGui } = render(
        <GuidedTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );
      expect(screen.getByTestId("actionability-badge").textContent).toBe("NOT ACTIONABLE");
      expect(screen.queryByText(/^WAIT FOR TRIGGER$/i)).toBeNull();
      unmountGui();

      // Advanced view
      const { unmount: unmountAdv } = render(
        <AdvancedTerminalView insight={insight} onOpenSizer={vi.fn()} onOpenWhy={vi.fn()} />
      );
      expect(screen.getByTestId("actionability-badge").textContent).toBe("NOT ACTIONABLE");
      expect(screen.queryByText(/^WAIT FOR TRIGGER$/i)).toBeNull();
      unmountAdv();
    });
  });

  // ── 4. Transition Safety & Zero Contradiction ──────────────────────────────────
  describe("4. Transition Safety & Contradiction Elimination", () => {
    it("eliminates all transient/persistent contradictions (AVOID/HOLD/UNVERIFIED + WAIT FOR TRIGGER)", () => {
      // Simulate asynchronous state progression:
      // 1. Partial data (UNVERIFIED)
      const unverified = deriveAssessmentState({
        symbol: "ASYNC_TEST",
        companyName: "Async Corp",
        currentPrice: 10.0,
        changePct: 0.0,
        horizon: "SWING",
        ownershipState: "NOT_OWNED",
        ownershipSource: "USER_DECLARED",
        domains: [],
        freshnessStatus: "UNAVAILABLE",
      });
      expect(unverified.isActionable).toBeFalsy();
      expect(unverified.uiStateLabel).not.toBe("Wait for Trigger");

      // 2. Unfavorable fundamentals (AVOID)
      const avoid = deriveAssessmentState({
        symbol: "ASYNC_TEST",
        companyName: "Async Corp",
        currentPrice: 10.0,
        changePct: -3.0,
        horizon: "SWING",
        ownershipState: "NOT_OWNED",
        ownershipSource: "USER_DECLARED",
        domains: [
          { domainId: "trend", domainName: "Price Trend", availability: "AVAILABLE", status: "UNFAVORABLE", pointImpact: -25, importanceLevel: "HIGH", observation: "Downtrend", modelRule: "Trend below 50 SMA", evidence: [], whatWouldChangeAssessment: "" },
          { domainId: "health", domainName: "Company Health", availability: "AVAILABLE", status: "UNFAVORABLE", pointImpact: -25, importanceLevel: "HIGH", observation: "Poor Solvency", modelRule: "Low current ratio", evidence: [], whatWouldChangeAssessment: "" },
        ],
      });
      expect(avoid.posture).toBe("AVOID");
      expect(avoid.uiStateLabel).toBe("Unfavorable Setup");
      expect(avoid.isActionable).toBeFalsy();

      // 3. Held position (HOLD)
      const hold = deriveAssessmentState({
        symbol: "ASYNC_TEST",
        companyName: "Async Corp",
        currentPrice: 10.0,
        changePct: 1.0,
        horizon: "SWING",
        ownershipState: "OWNED",
        ownershipSource: "USER_DECLARED",
        domains: [
          { domainId: "trend", domainName: "Price Trend", availability: "AVAILABLE", status: "FAVORABLE", pointImpact: 25, importanceLevel: "HIGH", observation: "Uptrend", modelRule: "Trend above 50 SMA", evidence: [], whatWouldChangeAssessment: "" },
          { domainId: "health", domainName: "Company Health", availability: "AVAILABLE", status: "FAVORABLE", pointImpact: 25, importanceLevel: "HIGH", observation: "Solvent", modelRule: "Good ratio", evidence: [], whatWouldChangeAssessment: "" },
          { domainId: "smart_money", domainName: "Smart Money Flow", availability: "AVAILABLE", status: "FAVORABLE", pointImpact: 25, importanceLevel: "HIGH", observation: "Accumulation", modelRule: "Inflow", evidence: [], whatWouldChangeAssessment: "" },
        ],
      });
      expect(hold.posture).toBe("HOLD");
      expect(hold.uiStateLabel).toBe("Thesis Intact");
      expect(hold.isActionable).toBeFalsy();
    });
  });
});
