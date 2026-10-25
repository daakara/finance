import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import PreFlightReadinessBar from "../PreFlightReadinessBar";

describe("PreFlightReadinessBar Component (Synthesis E V3 Inline Presentation)", () => {
  const pristineActionableProps = {
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    setupPattern: "Minervini VCP",
    isActionable: true,
    vix: 14.5,
    vixObservationTime: new Date().toISOString(),
    isStage4: false,
    optimalEntryMin: 215.0,
    optimalEntryMax: 222.0,
    catalystEvidence: {
      state: "PASS" as const,
      daysToEarnings: 18.0,
      eventDate: "2026-11-15",
      confirmationStatus: "CONFIRMED" as const,
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE" as const,
      riskTier: "NORMAL_RUNWAY",
    },
  };

  it("1. Pristine Actionable setup renders CLEARED with 5/5 PASS", () => {
    render(<PreFlightReadinessBar {...pristineActionableProps} />);

    // Regional landmark
    expect(screen.getByTestId("preflight-readiness-bar")).toBeDefined();

    // Actionability and clearance badge
    expect(screen.getByTestId("inline-actionability-badge").textContent).toBe("ACTIONABLE");
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("CLEARED");

    // All 5 checks show PASS
    for (let i = 1; i <= 5; i++) {
      expect(screen.getByTestId(`inline-chk-status-${i}`).textContent).toBe("PASS");
    }

    // Active blocker is NONE
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toBe("NONE");
  });

  it("2. AAPL Scenario: 5/5 checks PASS but non-actionable setup remains strictly BLOCKED", () => {
    render(
      <PreFlightReadinessBar
        {...pristineActionableProps}
        symbol="AAPL"
        isActionable={false}
      />
    );

    // Badges must display Awaiting Trigger & Not Cleared
    expect(screen.getByTestId("inline-actionability-badge").textContent).toBe("AWAITING TRIGGER");
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED · 5/5 PASS (AWAITING CONFIRMATION)");

    // Active blocker must explicitly cite Canonical Setup Non-Actionable
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toBe("CANONICAL SETUP NON-ACTIONABLE");
    expect(screen.getByTestId("inline-what-needs-to-change").textContent).toContain("confirm the canonical setup trigger");
  });

  it("3. NVDA Scenario: 4/5 checks PASS, Stage 4 correction strictly BLOCKED", () => {
    render(
      <PreFlightReadinessBar
        {...pristineActionableProps}
        symbol="NVDA"
        currentPrice={184.2}
        stopLoss={172.0}
        takeProfit1={215.0}
        isStage4={true}
        isActionable={true}
      />
    );

    // Clearance badge must cite Stage 4 Wait
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED — STAGE 4 WAIT");

    // Check 2 must show STAGE 4 WAIT
    expect(screen.getByTestId("inline-chk-status-2").textContent).toBe("STAGE 4 WAIT");

    // Active blocker must be STAGE 4 DISTRIBUTION STRUCTURE
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toBe("STAGE 4 DISTRIBUTION STRUCTURE");
  });

  it("4. Missing catalyst coverage fails closed to UNAVAILABLE and blocks clearance", () => {
    render(
      <PreFlightReadinessBar
        {...pristineActionableProps}
        catalystEvidence={null}
        hasImminentEarnings={null}
      />
    );

    // Check 4 must be UNAVAILABLE
    expect(screen.getByTestId("inline-chk-status-4").textContent).toBe("UNAVAILABLE");

    // Must be NOT CLEARED
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED (4/5)");
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toBe("CATALYST COVERAGE UNAVAILABLE");
  });

  it("5. Telemetry drawer toggles inline and opens optional deep modal", () => {
    const handleOpenModal = vi.fn();
    render(
      <PreFlightReadinessBar
        {...pristineActionableProps}
        onOpenModal={handleOpenModal}
      />
    );

    // Drawer is closed initially
    expect(screen.queryByTestId("preflight-details-drawer")).toBeNull();

    const toggleBtn = screen.getByTestId("inline-toggle-details-btn");
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("false");

    // Expand drawer
    fireEvent.click(toggleBtn);
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByTestId("preflight-details-drawer")).toBeDefined();

    // Deep-dive modal trigger
    const modalBtn = screen.getByTestId("inline-open-modal-btn");
    fireEvent.click(modalBtn);
    expect(handleOpenModal).toHaveBeenCalledTimes(1);
  });
});
