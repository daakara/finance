import React, { useState } from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { isETF, isStock, isUnknownAsset } from "../../lib/assetTypeUtils";
import EtfCostOfOwnershipCard from "../EtfCostOfOwnershipCard";
import OptimalEntryExitCard from "../OptimalEntryExitCard";
import AssetFactorRadar from "../AssetFactorRadar";
import TraderArchetypesCard from "../TraderArchetypesCard";
import EtfRiskProfileCard from "../EtfRiskProfileCard";

const mockExecutionPlan: any = {
  optimal_entry_min: 120,
  optimal_entry_max: 125,
  stop_loss: 110,
  take_profit_1: 140,
  take_profit_2: 155,
  risk_reward_ratio: 2.1,
  current_price: 122,
  execution_status: "IN_BUY_ZONE",
  entry_thesis: "Breakout retest",
};

const mockFactorScores: any = {
  growthScore: 85,
  qualityScore: 90,
  valuationScore: 70,
  momentumScore: 80,
  tailRiskScore: 75,
  compositeFactorScore: 80,
  verdict: "Strong Accumulation",
};

/**
 * CockpitTabRoutingHarness reflects the exact tab routing contract
 * implemented in frontend/app/page.tsx.
 */
function CockpitTabRoutingHarness({ initialSymbol = "NVDA" }: { initialSymbol?: string }) {
  const [selectedSymbol, setSelectedSymbol] = useState(initialSymbol);
  const [activeTab, setActiveTab] = useState<"EXECUTION" | "SMART_MONEY" | "FUNDAMENTALS">("EXECUTION");

  return (
    <div>
      {/* Control Strip for testing reactive symbol and tab switches */}
      <div data-testid="controls">
        <button onClick={() => setSelectedSymbol("NVDA")}>Select NVDA</button>
        <button onClick={() => setSelectedSymbol("SPY")}>Select SPY</button>
        <button onClick={() => setSelectedSymbol("UNKNOWN_XYZ")}>Select UNKNOWN</button>
        <button onClick={() => setActiveTab("EXECUTION")}>Tab Execution</button>
        <button onClick={() => setActiveTab("SMART_MONEY")}>Tab Smart Money</button>
        <button onClick={() => setActiveTab("FUNDAMENTALS")}>Tab Fundamentals</button>
      </div>

      {/* TAB 1: EXECUTION & LEVELS (Exact page.tsx contract) */}
      {activeTab === "EXECUTION" && (
        <div data-testid="tab-execution">
          {isETF(selectedSymbol) ? (
            <EtfCostOfOwnershipCard symbol={selectedSymbol} />
          ) : isStock(selectedSymbol) ? (
            <OptimalEntryExitCard symbol={selectedSymbol} executionPlan={mockExecutionPlan} />
          ) : (
            <div data-testid="unresolved-execution-card">
              <h3>🎯 {selectedSymbol} Execution Unresolved</h3>
              <p>Asset classification is unverified.</p>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: SMART MONEY (Exact page.tsx contract) */}
      {activeTab === "SMART_MONEY" && (
        <div data-testid="tab-smart-money">
          <div data-testid="congress-trades-placeholder">Congressional Trades: {selectedSymbol}</div>
          {!isETF(selectedSymbol) && (
            <TraderArchetypesCard symbol={selectedSymbol} />
          )}
        </div>
      )}

      {/* TAB 3: FUNDAMENTALS (Exact page.tsx contract) */}
      {activeTab === "FUNDAMENTALS" && (
        <div data-testid="tab-fundamentals">
          {isStock(selectedSymbol) && (
            <AssetFactorRadar symbol={selectedSymbol} factorScores={mockFactorScores} />
          )}
          {isETF(selectedSymbol) && (
            <EtfRiskProfileCard symbol={selectedSymbol} />
          )}
          <div data-testid="institutional-feeds">Feeds for {selectedSymbol}</div>
        </div>
      )}
    </div>
  );
}

describe("Cockpit ETF Routing & Behavioral Boundary (Milestone P1)", () => {
  it("renders stock Execution panel (OptimalEntryExitCard) for canonical stock", () => {
    render(<CockpitTabRoutingHarness initialSymbol="NVDA" />);

    expect(screen.getByText(/Safe Buy & Sell Plan/i)).toBeDefined();
    expect(screen.queryByText(/Cost of Ownership/i)).toBeNull();
    expect(screen.queryByTestId("unresolved-execution-card")).toBeNull();
  });

  it("renders ETF Execution panel (EtfCostOfOwnershipCard) for canonical ETF", () => {
    render(<CockpitTabRoutingHarness initialSymbol="SPY" />);

    expect(screen.getByText(/Cost of Ownership/i)).toBeDefined();
    expect(screen.queryByText(/Safe Buy & Sell Plan/i)).toBeNull();
    expect(screen.queryByTestId("unresolved-execution-card")).toBeNull();
  });

  it("renders bounded safe state for UNKNOWN asset in Execution tab", () => {
    render(<CockpitTabRoutingHarness initialSymbol="UNKNOWN_XYZ" />);

    expect(screen.getByTestId("unresolved-execution-card")).toBeDefined();
    expect(screen.getByText(/UNKNOWN_XYZ Execution Unresolved/i)).toBeDefined();
    expect(screen.queryByText(/Cost of Ownership/i)).toBeNull();
    expect(screen.queryByText(/Safe Buy & Sell Plan/i)).toBeNull();
  });

  it("suppresses corporate TraderArchetypesCard for ETFs in Smart Money tab", () => {
    render(<CockpitTabRoutingHarness initialSymbol="SPY" />);

    fireEvent.click(screen.getByText("Tab Smart Money"));

    expect(screen.getByTestId("congress-trades-placeholder")).toBeDefined();
    expect(screen.queryByText(/Warren Buffett/i)).toBeNull();
  });

  it("renders corporate TraderArchetypesCard for Stocks in Smart Money tab", () => {
    render(<CockpitTabRoutingHarness initialSymbol="NVDA" />);

    fireEvent.click(screen.getByText("Tab Smart Money"));

    expect(screen.getByTestId("congress-trades-placeholder")).toBeDefined();
    expect(screen.getByText(/Warren Buffett/i)).toBeDefined();
  });

  it("suppresses broken AssetFactorRadar and renders EtfRiskProfileCard for ETFs in Fundamentals tab", () => {
    render(<CockpitTabRoutingHarness initialSymbol="SPY" />);

    fireEvent.click(screen.getByText("Tab Fundamentals"));

    expect(screen.getByTestId("institutional-feeds")).toBeDefined();
    expect(screen.queryByText(/Business DNA & BS Detector/i)).toBeNull();
    expect(screen.getByLabelText(/ETF Risk Profile/i)).toBeDefined();
  });

  it("renders AssetFactorRadar and suppresses EtfRiskProfileCard for Stocks in Fundamentals tab", () => {
    render(<CockpitTabRoutingHarness initialSymbol="NVDA" />);

    fireEvent.click(screen.getByText("Tab Fundamentals"));

    expect(screen.getByTestId("institutional-feeds")).toBeDefined();
    expect(screen.getByText(/Business DNA & BS Detector/i)).toBeDefined();
    expect(screen.queryByText(/ETF Risk Profile/i)).toBeNull();
  });

  it("dynamically updates all tabs when switching STOCK -> ETF", () => {
    render(<CockpitTabRoutingHarness initialSymbol="NVDA" />);

    // In Execution tab initially: stock
    expect(screen.getByText(/Safe Buy & Sell Plan/i)).toBeDefined();

    // Switch to SPY (ETF)
    fireEvent.click(screen.getByText("Select SPY"));

    // Execution tab now renders ETF Cost of Ownership
    expect(screen.getByText(/Cost of Ownership/i)).toBeDefined();
    expect(screen.queryByText(/Safe Buy & Sell Plan/i)).toBeNull();

    // Switch to Fundamentals tab: broken radar is suppressed
    fireEvent.click(screen.getByText("Tab Fundamentals"));
    expect(screen.queryByText(/Business DNA & BS Detector/i)).toBeNull();

    // Switch to Smart Money tab: corporate archetypes are suppressed
    fireEvent.click(screen.getByText("Tab Smart Money"));
    expect(screen.queryByText(/Warren Buffett/i)).toBeNull();
  });

  it("dynamically restores stock content when switching ETF -> STOCK", () => {
    render(<CockpitTabRoutingHarness initialSymbol="SPY" />);

    // In Execution tab initially: ETF
    expect(screen.getByText(/Cost of Ownership/i)).toBeDefined();

    // Switch to NVDA (Stock)
    fireEvent.click(screen.getByText("Select NVDA"));

    // Execution tab restores stock ladder
    expect(screen.getByText(/Safe Buy & Sell Plan/i)).toBeDefined();
    expect(screen.queryByText(/Cost of Ownership/i)).toBeNull();

    // Switch to Fundamentals: restores factor radar
    fireEvent.click(screen.getByText("Tab Fundamentals"));
    expect(screen.getByText(/Business DNA & BS Detector/i)).toBeDefined();

    // Switch to Smart Money: restores corporate archetypes
    fireEvent.click(screen.getByText("Tab Smart Money"));
    expect(screen.getByText(/Warren Buffett/i)).toBeDefined();
  });
});
