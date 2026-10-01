import React from "react";
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import EtfCostOfOwnershipCard, { getFeeTierCategory } from "../EtfCostOfOwnershipCard";

describe("EtfCostOfOwnershipCard", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("renders authoritative expense ratio correctly with nominal direct fee accounting", () => {
    render(<EtfCostOfOwnershipCard symbol="SPY" expenseRatio={0.09} />);

    // Header & Category
    expect(screen.getByText("Cost of Ownership")).toBeDefined();
    expect(screen.getByText("Very Low")).toBeDefined();

    // Expense ratio & basis points
    expect(screen.getAllByText("0.09%").length).toBeGreaterThan(0);
    expect(screen.getByText("9.0 bps")).toBeDefined();

    // Direct Cost grid: $10,000 * 0.09% = $9 annual; $9 * 10 = $90 10-yr direct fee
    expect(screen.getByText("$9")).toBeDefined();
    expect(screen.getByText("$90")).toBeDefined();

    // Insight text clearly describes direct fees without imputed compounding
    expect(
      screen.getByText(/At 0.09%, SPY incurs approximately \$9 per year in direct fund deductions/)
    ).toBeDefined();

    // Spectrum bar
    expect(screen.getByText("Fee Tier Comparison (0.00% – 1.00%)")).toBeDefined();
    expect(screen.getByText("SPY 0.09%")).toBeDefined();
  });

  it("handles case-insensitive and prefixed symbols", () => {
    render(<EtfCostOfOwnershipCard symbol="amex:qqq" expenseRatio={0.20} />);

    expect(screen.getAllByText("0.20%").length).toBeGreaterThan(0);
    expect(screen.getByText("Low")).toBeDefined();
    expect(screen.getByText("$20")).toBeDefined();
    expect(screen.getByText("$200")).toBeDefined();
  });

  it("toggles to PRO_QUANT mode and displays nominal fee accounting without arbitrary assumptions", () => {
    localStorage.setItem("ARX_VERNACULAR_MODE", "PRO_QUANT");
    render(<EtfCostOfOwnershipCard symbol="SPY" expenseRatio={0.09} />);

    // Header in PRO_QUANT
    expect(screen.getByText("Total Expense Analysis")).toBeDefined();

    // Accounting elements
    expect(screen.getByText("DIRECT FUND FEE SUMMARY")).toBeDefined();
    expect(screen.getByText("Annual Fee = P₀ × TER • 10-Yr Fee = 10 × (P₀ × TER)")).toBeDefined();
  });

  it("reacts dynamically to custom finance:vernacular-change event", () => {
    render(<EtfCostOfOwnershipCard symbol="SPY" expenseRatio={0.09} />);
    expect(screen.getByText("Cost of Ownership")).toBeDefined();

    act(() => {
      window.dispatchEvent(
        new CustomEvent("finance:vernacular-change", { detail: "PRO_QUANT" })
      );
    });

    expect(screen.getByText("Total Expense Analysis")).toBeDefined();
  });

  it("correctly categorizes fee tier thresholds", () => {
    expect(getFeeTierCategory(0.03)).toBe("Very Low");
    expect(getFeeTierCategory(0.14)).toBe("Very Low");
    expect(getFeeTierCategory(0.15)).toBe("Low");
    expect(getFeeTierCategory(0.34)).toBe("Low");
    expect(getFeeTierCategory(0.35)).toBe("Moderate");
    expect(getFeeTierCategory(0.69)).toBe("Moderate");
    expect(getFeeTierCategory(0.70)).toBe("High");
    expect(getFeeTierCategory(1.20)).toBe("High");
  });

  it("renders fee tier badges correctly across spectrum", () => {
    const { rerender } = render(<EtfCostOfOwnershipCard symbol="GLD" expenseRatio={0.40} />);
    expect(screen.getByText("Moderate")).toBeDefined();
    expect(screen.getAllByText("0.40%").length).toBeGreaterThan(0);

    rerender(<EtfCostOfOwnershipCard symbol="ARKK" expenseRatio={0.75} />);
    expect(screen.getByText("High")).toBeDefined();
    expect(screen.getAllByText("0.75%").length).toBeGreaterThan(0);
    expect(screen.getByText("$75")).toBeDefined();
  });

  it("handles zero expense ratio cleanly", () => {
    render(<EtfCostOfOwnershipCard symbol="FZROX" expenseRatio={0} />);

    expect(screen.getByText("Very Low")).toBeDefined();
    expect(screen.getAllByText("0.00%").length).toBeGreaterThan(0);
    expect(screen.getByText("0.0 bps")).toBeDefined();
    expect(screen.getAllByText("$0").length).toBe(2);
  });

  it("renders graceful unverified state when expense ratio is missing (zero synthetic data)", () => {
    render(<EtfCostOfOwnershipCard symbol="XYZUNKNOWN" />);

    expect(screen.getByText("TER: UNVERIFIED")).toBeDefined();
    expect(
      screen.getByText(/Certified Total Expense Ratio \(TER\) is not currently available/)
    ).toBeDefined();
    expect(screen.getByText(/NO_RUNTIME_TER_METADATA/)).toBeDefined();
  });

  it("renders graceful unverified state when expense ratio is null or negative", () => {
    const { rerender } = render(<EtfCostOfOwnershipCard symbol="SPY" expenseRatio={null} />);
    expect(screen.getByText("TER: UNVERIFIED")).toBeDefined();

    rerender(<EtfCostOfOwnershipCard symbol="SPY" expenseRatio={-0.05} />);
    expect(screen.getByText("TER: UNVERIFIED")).toBeDefined();
  });
});
