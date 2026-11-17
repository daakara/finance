import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import PreFlightChecklistModal from "../PreFlightChecklistModal";
import PreFlightReadinessBar from "../PreFlightReadinessBar";

vi.mock("../../lib/matomo", () => ({
  trackPreFlightOutcome: vi.fn(),
  trackTradePlanCopied: vi.fn(),
}));

describe("Pre-Flight Decision Integrity & Epistemic Separation (Issue #197)", () => {
  const baseLiveProps = {
    symbol: "AAPL",
    currentPrice: 220.0,
    stopLoss: 210.0,
    takeProfit1: 245.0,
    riskRewardRatio: 2.5,
    setupPattern: "Minervini VCP",
    isActionable: true,
    canSizeTrade: true,
    vix: 15.0,
    vixObservationTime: new Date().toISOString(),
    isStage4: false,
    optimalEntryMin: 215.0,
    optimalEntryMax: 222.0,
    catalystEvidence: {
      state: "PASS" as const,
      daysToEarnings: 15.0,
      eventDate: "2026-11-20",
      confirmationStatus: "CONFIRMED" as const,
      coverageStatus: "CONFIRMED_UPCOMING",
      freshnessStatus: "LIVE" as const,
      riskTier: "NORMAL_RUNWAY",
    },
    dataSource: "live",
    liveSource: "ALPACA_IEX",
    liveFreshnessStatus: "LIVE",
  };

  it("1. Synthetic fixture modal displays SIMULATED CLEARANCE — NOT EXECUTABLE", () => {
    render(
      <PreFlightChecklistModal
        isOpen={true}
        onClose={vi.fn()}
        {...baseLiveProps}
        dataSource="synthetic"
        isSynthetic={true}
      />
    );

    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/SIMULATED CLEARANCE — NOT EXECUTABLE/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();
  });

  it("2. Historical EOD modal displays HISTORICAL REFERENCE — NOT EXECUTABLE", () => {
    render(
      <PreFlightChecklistModal
        isOpen={true}
        onClose={vi.fn()}
        {...baseLiveProps}
        dataSource="historical"
        isHistorical={true}
      />
    );

    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/HISTORICAL REFERENCE — NOT EXECUTABLE/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();
  });

  it("3. Stale price observation blocks execution clearance in modal", () => {
    render(
      <PreFlightChecklistModal
        isOpen={true}
        onClose={vi.fn()}
        {...baseLiveProps}
        liveFreshnessStatus="STALE"
      />
    );

    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — STALE PRICE OBSERVATION/i)).toBeDefined();
  });

  it("4. Historical tag renders in PreFlightReadinessBar when historical", () => {
    render(
      <PreFlightReadinessBar
        {...baseLiveProps}
        dataSource="historical"
        isHistorical={true}
      />
    );

    expect(screen.getByTestId("inline-historical-tag").textContent).toContain("HISTORICAL");
  });

  it("5. False Clearance Case 1: Delayed quote with unverified provenance blocks live clearance across bar, modal, and report", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    const delayedProps = {
      ...baseLiveProps,
      liveFreshnessStatus: "DELAYED",
      quoteStatus: "DELAYED",
    };

    // 1. Verify PreFlightReadinessBar
    const { unmount: unmountBar } = render(<PreFlightReadinessBar {...delayedProps} />);
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED · DELAYED QUOTE (5/5)");
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toBe("DELAYED PRICE OBSERVATION");
    expect(screen.getByTestId("inline-what-needs-to-change").textContent).toContain("Wait for real-time exchange feed");
    unmountBar();

    // 2. Verify PreFlightChecklistModal
    render(<PreFlightChecklistModal isOpen={true} onClose={vi.fn()} {...delayedProps} />);
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — DELAYED PRICE OBSERVATION/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();

    // 3. Verify copied trade plan markdown agreement
    const copyBtn = screen.getByText(/Copy Trade Plan for Journal/i);
    copyBtn.click();

    expect(writeTextMock).toHaveBeenCalled();
    const copiedText = writeTextMock.mock.calls[0][0];
    expect(copiedText).toContain("RESEARCH / JOURNAL ONLY — NOT EXECUTABLE");
    expect(copiedText).toContain("Clearance Blockers: DELAYED PRICE OBSERVATION; UNVERIFIED PRICE PROVENANCE");
    expect(copiedText).toContain("⚠️ NOT CLEARED — DELAYED PRICE OBSERVATION");
  });

  it("6. False Clearance Case 2: Missing canSizeTrade blocks live clearance across bar, modal, and report", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    const missingSizingProps = {
      ...baseLiveProps,
      canSizeTrade: undefined, // Missing sizing permission
    };

    // 1. Verify PreFlightReadinessBar
    const { unmount: unmountBar } = render(<PreFlightReadinessBar {...missingSizingProps} />);
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED · 5/5 PASS (SIZING LOCKED)");
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toBe("CANONICAL SIZING PROHIBITED");
    expect(screen.getByTestId("inline-what-needs-to-change").textContent).toContain("Position sizing cannot be authorized");
    unmountBar();

    // 2. Verify PreFlightChecklistModal
    render(<PreFlightChecklistModal isOpen={true} onClose={vi.fn()} {...missingSizingProps} />);
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText(/⚠️ NOT CLEARED — CANONICAL SIZING PROHIBITED/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();

    // 3. Verify copied trade plan markdown agreement
    const copyBtn = screen.getByText(/Copy Trade Plan for Journal/i);
    copyBtn.click();

    expect(writeTextMock).toHaveBeenCalled();
    const copiedText = writeTextMock.mock.calls[0][0];
    expect(copiedText).toContain("Clearance Blockers: CANONICAL SIZING PROHIBITED");
    expect(copiedText).toContain("⚠️ NOT CLEARED — CANONICAL SIZING PROHIBITED");
  });

  it("7. Verified real-time quote with both canonical permissions and all mandatory checks clears across bar and modal", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    // 1. Verify PreFlightReadinessBar
    const { unmount: unmountBar } = render(<PreFlightReadinessBar {...baseLiveProps} />);
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("CLEARED");
    expect(screen.getByTestId("inline-active-blocker-text").textContent).toBe("NONE");
    unmountBar();

    // 2. Verify PreFlightChecklistModal
    render(<PreFlightChecklistModal isOpen={true} onClose={vi.fn()} {...baseLiveProps} />);
    expect(screen.getByText(/🟢 CLEARED TO EXECUTE/i)).toBeDefined();
    expect(screen.getByText("5/5")).toBeDefined();

    // 3. Verify copied trade plan markdown
    const copyBtn = screen.getByText(/Copy Trade Plan for Journal/i);
    copyBtn.click();

    expect(writeTextMock).toHaveBeenCalled();
    const copiedText = writeTextMock.mock.calls[0][0];
    expect(copiedText).toContain("🟢 CLEARED");
    expect(copiedText).not.toContain("NOT EXECUTABLE");
  });

  it.each([
    [258, "$258.00 (+17.27%)"],
    [215, "$215.00 (-2.27%)"],
    [undefined, "Unavailable"],
    [null, "Unavailable"],
    [Infinity, "Unavailable"],
    [0, "Unavailable"],
  ])("uses the plan Target 2 value %s in the modal and journal copy", async (takeProfit2, expected) => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    render(<PreFlightChecklistModal {...baseLiveProps} isOpen onClose={vi.fn()} takeProfit2={takeProfit2} />);

    expect(screen.getByTestId("preflight-target-2").textContent).toContain(expected);
    screen.getByRole("button", { name: /Copy Trade Plan for Journal/i }).click();
    expect(writeText.mock.calls[0]?.[0]).toContain(`**Profit Goal 2 (TP2 Runner)**: ${expected}`);
    expect(writeText.mock.calls[0]?.[0]).not.toContain("$274.40");
  });

  it("8. Audited NVDA snapshot returns 2/5 checks passed and Not cleared across bar and modal", () => {
    const auditedNvdaProps = {
      symbol: "NVDA",
      currentPrice: 140.0,
      stopLoss: 125.0,
      takeProfit1: 142.0, // Sub-threshold R:R (< 2.0:1) -> Check 1 FAILS
      riskRewardRatio: 0.13,
      isStage4: true, // Stage 4 -> Check 2 FAILS
      optimalEntryMin: 138.0,
      optimalEntryMax: 142.0,
      breakoutPivot: 145.0,
      qualityScore: 95,
      shortFloat: 2.2, // Capital -> Check 3 PASSES
      hasImminentEarnings: true, // Imminent earnings -> Check 4 FAILS
      catalystEvidence: {
        state: "WARN" as const,
        daysToEarnings: 0.5,
        eventDate: "2026-11-17",
        confirmationStatus: "CONFIRMED" as const,
        coverageStatus: "CONFIRMED_UPCOMING",
        freshnessStatus: "LIVE" as const,
        riskTier: "BINARY_HAZARD",
      },
      vix: 15.2, // Macro -> Check 5 PASSES
      vixObservationTime: new Date().toISOString(),
      macroEvidence: {
        vix: 15.2,
        observationTime: new Date().toISOString(),
        marketSession: "REGULAR",
        freshnessStatus: "LIVE",
      },
      isActionable: false,
      canSizeTrade: false,
      dataSource: "live",
      liveSource: "ALPACA_IEX",
      liveFreshnessStatus: "LIVE",
    };

    // 1. Verify PreFlightReadinessBar
    const { unmount: unmountBar } = render(<PreFlightReadinessBar {...auditedNvdaProps} />);
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED — STAGE 4 WAIT");
    expect(screen.getByTestId("inline-chk-status-1").textContent).toBe("Below minimum");
    expect(screen.getByTestId("inline-chk-status-2").textContent).toBe("STAGE 4 WAIT");
    expect(screen.getByTestId("inline-chk-status-3").textContent).toBe("PASS");
    expect(screen.getByTestId("inline-chk-status-4").textContent).toBe("HAZARD");
    expect(screen.getByTestId("inline-chk-status-5").textContent).toBe("PASS");
    unmountBar();

    // 2. Verify PreFlightChecklistModal
    render(<PreFlightChecklistModal isOpen={true} onClose={vi.fn()} {...auditedNvdaProps} />);
    expect(screen.queryByText(/🟢 CLEARED TO EXECUTE/i)).toBeNull();
    expect(screen.getByText("2/5")).toBeDefined();
    expect(screen.getByText(/⚠️ NOT CLEARED/i)).toBeDefined();
  });
});
