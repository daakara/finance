import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Standard from "../terminal/StandardTerminalView";
import Guided from "../terminal/GuidedTerminalView";
import Quant from "../terminal/AdvancedTerminalView";
import PositionSizerModal from "../PositionSizerModal";
import PreFlightReadinessBar from "../PreFlightReadinessBar";
import { generateQuantitativeInsight } from "../../lib/insightGenerator";

afterEach(() => { cleanup(); localStorage.clear(); });

describe("UX-002 canonical authority acceptance (synthetic component inputs, no backend acceptance)", () => {
  for (const [mode, View] of [["guided", Guided], ["standard", Standard], ["quant", Quant]] as const) {
    it.each(["STALE_DATA", "UNVERIFIED", "VALID_SETUP"])(`${mode} preserves a blocked %s decision and its reference levels`, (decisionState) => {
      const insight = generateQuantitativeInsight("AAPL", "Acceptance fixture", 100, 0, 75, 2, "SWING", "NOT_OWNED", "USER_DECLARED", [], "unavailable");
      insight.terminalState = { ...insight.terminalState, decisionState, isActionable: false, canSizeTrade: false,
        headlineExplanation: "Canonical evidence is incomplete", whatWouldChangeAssessment: "Verified trigger and complete clearance" };
      insight.verdictLabel = "Acceptance fixture: await verified evidence";
      insight.standard.keyLevels = { ...insight.standard.keyLevels, currentPrice: 100, entryMin: 99, entryMax: 101, stopLoss: 95, target1: 115 };
      render(<View insight={insight} />);
      expect(screen.getByTestId("actionability-badge").textContent).toContain("NOT ACTIONABLE");
      expect(screen.getByTestId("decision-verdict").textContent).toContain(insight.verdictLabel);
      expect(screen.getByTestId("cell-stop").textContent).toContain("95.00");
      expect(screen.getByTestId("cell-target").textContent).toContain("115.00");
      expect(screen.queryByText(/Size & Execute/i)).toBeNull();
    });
  }

  it("unavailable backend evidence cannot generate an actionable canonical state from an attractive score", () => {
    const insight = generateQuantitativeInsight("AAPL", "Unavailable fixture", 100, 2, 99, 2, "SWING", "NOT_OWNED", "USER_DECLARED", [], "unavailable");
    expect(insight.terminalState.isActionable).toBe(false);
    expect(insight.terminalState.canSizeTrade).not.toBe(true);
    expect(insight.setupScore).toBe(0);
  });

  it("nominally passing geometry cannot clear an explicit canonical block", () => {
    render(<PreFlightReadinessBar symbol="AAPL" currentPrice={100} stopLoss={95} takeProfit1={115}
      riskRewardRatio={3} optimalEntryMin={99} optimalEntryMax={101} isStage4={false} vix={16}
      isActionable={false} decisionState="UNVERIFIED" catalystState="PASS" />);
    expect(screen.getByTestId("inline-clearance-badge").textContent).toContain("NOT CLEARED");
  });

  it("blocked sizing modal remains an inspector and cannot write a paper record", () => {
    render(<PositionSizerModal isOpen onClose={vi.fn()} symbol="AAPL" entryPrice={100} stopLoss={95}
      takeProfit1={115} canSizeTrade={false} isActionable={false} />);
    const save = screen.getByRole("button", { name: /Sizing Disabled/ }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.click(save);
    expect(localStorage.getItem("FINANCE_USER_PORTFOLIO")).toBeNull();
    expect(screen.getByText("Shares", { exact: true }).parentElement?.textContent).toContain("0 Shares");
  });
});
