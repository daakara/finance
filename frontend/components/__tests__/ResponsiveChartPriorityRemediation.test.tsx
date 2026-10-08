import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import StandardTerminalView from "../terminal/StandardTerminalView";
import { QuantitativeInsight } from "../../types/insight";

function createMockInsight(): QuantitativeInsight {
  return {
    id: "test-naut-1",
    symbol: "NAUT",
    companyName: "Nautilus Marine Acquisition Corp",
    price: 28.42,
    changePct: 1.85,
    setupScore: 78,
    horizon: "SWING",
    assessment: "FAVORABLE",
    posture: "WATCH",
    postureLabel: "Watch Pullback",
    ownership: "NOT_OWNED",
    terminalState: {
      symbol: "NAUT",
      companyName: "Nautilus Marine Acquisition Corp",
      currentPrice: 28.42,
      changePct: 1.85,
      horizon: "SWING",
      ownership: {
        state: "NOT_OWNED",
        source: "USER_DECLARED",
      },
      modelProvenance: {
        modelId: "confluence-v2",
        modelVersion: "2.4.0",
        rulesetVersion: "2026.1",
        calculatedAt: "2026-10-08T14:00:00Z",
      },
      overallEligibility: "ELIGIBLE",
      decisionState: "VALID_SETUP",
      canSizeTrade: true,
      isActionable: false,
      assessment: "FAVORABLE",
      factorAgreement: {
        favorable: 4,
        mixed: 1,
        unfavorable: 0,
        unavailable: 0,
        evaluated: 5,
        displayLabel: "Strong Agreement",
      },
      domains: [],
      posture: "WATCH",
      uiStateLabel: "Awaiting Trigger",
      headlineExplanation: "Setup is structurally valid, awaiting pullback to entry corridor.",
      whatWouldChangeAssessment: "Pullback into buy zone with volume confirmation",
      primaryAction: {
        label: "Set Pullback Alert",
        actionType: "SET_ALERT",
        enabled: true,
      },
      availableActions: [],
    },
    verdict: "WAIT_FOR_TRIGGER",
    verdictLabel: "Valid Setup — Awaiting Trigger",
    human: {
      assessmentHeadline: "Valid Setup — Awaiting Trigger",
      assessmentDescription: "Price is hovering above optimal accumulation zone.",
      whyPills: [],
      reclaimMilestone: "Pullback to $27.80",
      watchLevels: {
        watchZone: "$27.50 - $28.20",
        keyLevel: "$28.20",
        riskStop: "$26.40",
      },
      actionCallout: {
        action: "WATCH",
        guidance: "Do not chase extended price.",
      },
    },
    standard: {
      bottomLine: "Stage 2 accumulation structure confirmed; volume drying up on pullback.",
      signalsRatio: "4 of 5 Pillars Aligned",
      confluenceBreakdown: [
        { dimension: "Technical Trend", score: 85, label: "Favorable" },
        { dimension: "Fundamental Health", score: 75, label: "Solid" },
      ],
      setupSummary: "VCP Pullback towards 20 EMA",
      keyLevels: {
        currentPrice: 28.42,
        watchZone: "$27.50 - $28.20",
        profitRiskRatio: 2.85,
        stopLoss: 26.4,
        target1: 32.5,
      },
    },
  };
}

describe("Responsive Chart Priority Remediation Acceptance Suite", () => {
  it("A & G & J: Enforces responsive grid classes (lg:grid-cols-12 and xl:grid-cols-12) for side-by-side first viewport", () => {
    const insight = createMockInsight();
    const chartSlot = <div data-testid="test-chart-slot">Chart Content</div>;
    const planSlot = <div data-testid="test-plan-slot">Plan Content</div>;

    const { container } = render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={chartSlot}
        planSlot={planSlot}
      />
    );

    // Verify 12-column grid container supports both lg: and xl:
    const gridContainer = container.querySelector(".grid.grid-cols-1.lg\\:grid-cols-12.xl\\:grid-cols-12");
    expect(gridContainer).not.toBeNull();
    expect(gridContainer?.classList.contains("gap-4")).toBe(true);
    expect(gridContainer?.classList.contains("items-start")).toBe(true);

    // Left column (Verdict): order-1 lg:order-1 xl:order-1, col-span-5
    const leftCol = gridContainer?.querySelector(".lg\\:col-span-5.xl\\:col-span-5");
    expect(leftCol).not.toBeNull();
    expect(leftCol?.classList.contains("order-1")).toBe(true);
    expect(leftCol?.classList.contains("lg:order-1")).toBe(true);
    expect(leftCol?.classList.contains("xl:order-1")).toBe(true);

    // Right column (Chart): order-2 lg:order-2 xl:order-2, col-span-7
    const chartCol = gridContainer?.querySelector(".lg\\:col-span-7.xl\\:col-span-7");
    expect(chartCol).not.toBeNull();
    expect(chartCol?.classList.contains("order-2")).toBe(true);
    expect(chartCol?.classList.contains("lg:order-2")).toBe(true);
    expect(chartCol?.classList.contains("xl:order-2")).toBe(true);

    // Plan slot: order-3 lg:order-3 xl:order-3, col-span-12
    const planCol = gridContainer?.querySelector(".order-3.lg\\:order-3.xl\\:order-3");
    expect(planCol).not.toBeNull();
    expect(planCol?.classList.contains("lg:col-span-12")).toBe(true);
    expect(planCol?.classList.contains("xl:col-span-12")).toBe(true);
  });

  it("H & I: Guarantees Chart/Preview (order-2) always precedes Execution Plan (order-3) on mobile and tablet", () => {
    const insight = createMockInsight();
    const { container } = render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div>Chart</div>}
        planSlot={<div>Plan</div>}
      />
    );

    const grid = container.querySelector(".grid");
    const directChildren = Array.from(grid?.children || []);

    // Child 0: Left Column (Verdict) has order-1
    expect(directChildren[0].classList.contains("order-1")).toBe(true);

    // Child 1: Chart Column has order-2
    expect(directChildren[1].classList.contains("order-2")).toBe(true);

    // Child 2: Plan Slot has order-3
    expect(directChildren[2].classList.contains("order-3")).toBe(true);
  });

  it("B: Renders mobile chart preview in default collapsed state with truthful data and hidden canvas on mobile", () => {
    const insight = createMockInsight();
    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div data-testid="inner-chart">Canvas Component</div>}
        planSlot={<div>Plan</div>}
      />
    );

    // Preview affordance is present
    const preview = screen.getByTestId("mobile-chart-preview");
    expect(preview).toBeDefined();
    expect(preview.classList.contains("md:hidden")).toBe(true);

    // Truthful data present in preview
    expect(within(preview).getByText("NAUT")).toBeDefined();
    expect(within(preview).getByText("$28.42")).toBeDefined();
    expect(within(preview).getByText("$27.50 - $28.20")).toBeDefined();

    // Default button state
    const toggleBtn = screen.getByRole("button", { name: /expand candlestick price chart/i });
    expect(toggleBtn).toBeDefined();
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("false");
    expect(toggleBtn.textContent).toContain("View Candlestick Chart ▾");

    // Full chart container has hidden md:block in collapsed state
    const chartContainer = document.querySelector("#market-workspace-chart");
    expect(chartContainer).not.toBeNull();
    expect(chartContainer?.className).toContain("hidden md:block");
  });

  it("C & D: Expands canonical full chart on user tap and restores collapsed preview on subsequent tap", () => {
    const insight = createMockInsight();
    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div data-testid="inner-chart">Canvas Component</div>}
        planSlot={<div>Plan</div>}
      />
    );

    const toggleBtn = screen.getByRole("button", { name: /expand candlestick price chart/i });
    const chartContainer = document.querySelector("#market-workspace-chart");

    // 1. Initial State: Collapsed
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("false");
    expect(chartContainer?.className).toContain("hidden md:block");

    // 2. Expand: Click toggle
    fireEvent.click(toggleBtn);

    expect(toggleBtn.getAttribute("aria-expanded")).toBe("true");
    expect(toggleBtn.getAttribute("aria-label")).toBe("Collapse candlestick price chart");
    expect(toggleBtn.textContent).toContain("Hide Chart ▲");
    expect(chartContainer?.className).toContain("block space-y-2");

    // 3. Collapse: Click toggle again
    fireEvent.click(toggleBtn);

    expect(toggleBtn.getAttribute("aria-expanded")).toBe("false");
    expect(toggleBtn.getAttribute("aria-label")).toBe("Expand candlestick price chart");
    expect(toggleBtn.textContent).toContain("View Candlestick Chart ▾");
    expect(chartContainer?.className).toContain("hidden md:block");
  });

  it("E & F: Satisfies accessibility contracts: min 44x44px touch targets, aria-controls, and visible focus", () => {
    const insight = createMockInsight();
    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div>Chart</div>}
        planSlot={<div>Plan</div>}
      />
    );

    const toggleBtn = screen.getByRole("button", { name: /expand candlestick price chart/i });

    // Touch target: min-h-[44px] min-w-[44px]
    expect(toggleBtn.classList.contains("min-h-[44px]")).toBe(true);
    expect(toggleBtn.classList.contains("min-w-[44px]")).toBe(true);

    // ARIA relationship
    expect(toggleBtn.getAttribute("aria-controls")).toBe("market-workspace-chart");

    // Focus indicators
    expect(toggleBtn.classList.contains("focus-visible:ring-2")).toBe(true);
    expect(toggleBtn.classList.contains("focus-visible:ring-cyan-400")).toBe(true);
  });

  it("K: QA-ESC-011 Invariant Preservation: Verdict and Actionability badges remain uncollapsed and strictly fail-closed", () => {
    const insight = createMockInsight();
    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div>Chart</div>}
        planSlot={<div>Plan</div>}
      />
    );

    // Heading matches authoritative verdict label
    const verdictHeading = screen.getByRole("heading", { level: 2 });
    expect(verdictHeading.textContent).toBe("Valid Setup — Awaiting Trigger");

    // Actionability badge renders NOT ACTIONABLE (not generic 'Wait for Trigger')
    const actionBadge = screen.getByTestId("actionability-badge");
    expect(actionBadge.textContent).toBe("NOT ACTIONABLE");
    expect(actionBadge.classList.contains("bg-slate-900/80")).toBe(true);

    // Reason is displayed
    const reason = screen.getByTestId("decision-reason");
    expect(reason.textContent).toContain("Stage 2 accumulation structure confirmed");
  });

  it("Section 7: Missing-Data State: Spot exists but buy-zone is unavailable -> renders UNAVAILABLE, zero manufactured corridor", () => {
    const insight = createMockInsight();
    // Simulate missing buy-zone geometry
    insight.standard.keyLevels.watchZone = "Unavailable";
    insight.standard.keyLevels.entryMin = undefined;
    insight.standard.keyLevels.entryMax = undefined;

    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div>Chart</div>}
        planSlot={<div>Plan</div>}
      />
    );

    const preview = screen.getByTestId("mobile-chart-preview");
    expect(preview).toBeDefined();

    // Spot price still renders truthfully
    const spot = within(preview).getByTestId("preview-spot-price");
    expect(spot.textContent).toBe("$28.42");

    // Watch zone fails closed to UNAVAILABLE (zero manufactured corridor)
    const zone = within(preview).getByTestId("preview-watch-zone");
    expect(zone.textContent).toBe("UNAVAILABLE");
    expect(zone.textContent).not.toContain("$");
  });

  it("Section 8: Full-Missing State: Both spot and plan geometry are unavailable -> fails closed with explicit UNAVAILABLE", () => {
    const insight = createMockInsight();
    // Simulate full-missing data
    insight.price = 0;
    insight.standard.keyLevels.currentPrice = 0;
    insight.standard.keyLevels.watchZone = "Unavailable";
    insight.standard.keyLevels.entryMin = undefined;
    insight.standard.keyLevels.entryMax = undefined;

    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div>Chart</div>}
        planSlot={<div>Plan</div>}
      />
    );

    const preview = screen.getByTestId("mobile-chart-preview");
    expect(preview).toBeDefined();

    // Both spot price and watch zone must strictly fail closed to UNAVAILABLE
    const spot = within(preview).getByTestId("preview-spot-price");
    expect(spot.textContent).toBe("UNAVAILABLE");
    expect(spot.textContent).not.toContain("$0.00");

    const zone = within(preview).getByTestId("preview-watch-zone");
    expect(zone.textContent).toBe("UNAVAILABLE");
    expect(zone.textContent).not.toContain("$");
  });

  it("Section 7 Scenario 3: Watch Zone present but spot price missing -> spot renders UNAVAILABLE, watch zone renders corridor", () => {
    const insight = createMockInsight();
    // Simulate missing spot price while zone exists
    insight.price = 0;
    insight.standard.keyLevels.currentPrice = 0;

    render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div>Chart</div>}
        planSlot={<div>Plan</div>}
      />
    );

    const preview = screen.getByTestId("mobile-chart-preview");
    expect(preview).toBeDefined();

    // Spot fails closed to UNAVAILABLE
    const spot = within(preview).getByTestId("preview-spot-price");
    expect(spot.textContent).toBe("UNAVAILABLE");
    expect(spot.textContent).not.toContain("$0.00");

    // Watch zone renders the valid corridor
    const zone = within(preview).getByTestId("preview-watch-zone");
    expect(zone.textContent).toBe("$27.50 - $28.20");
  });

  it("Section 2: Desktop Layout Parity: Right column does not contain space-y-2 on wrapper, preventing 8px top shift", () => {
    const insight = createMockInsight();
    const { container } = render(
      <StandardTerminalView
        insight={insight}
        onOpenSizer={vi.fn()}
        onOpenWhy={vi.fn()}
        chartSlot={<div>Chart</div>}
        planSlot={<div>Plan</div>}
      />
    );

    const rightCol = container.querySelector(".lg\\:col-span-7.xl\\:col-span-7");
    expect(rightCol).not.toBeNull();
    // Prohibit space-y-2 on column container to protect desktop alignment parity
    expect(rightCol?.classList.contains("space-y-2")).toBe(false);
  });
});
