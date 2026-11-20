import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import CatalystForecastCard from "../CatalystForecastCard";
import { CatalystForecastData } from "../../lib/api";

describe("DEF-CAT-002: CatalystForecastCard Multi-Year Forecast Empty State", () => {
  const baseStockData: CatalystForecastData = {
    symbol: "NVDA",
    company_name: "NVIDIA Corporation",
    sector: "Semiconductors & AI Hardware Infrastructure",
    primary_drug_trial: "Blackwell Architecture Ramp",
    trial_phase: "Commercial Ramp Phase",
    trial_readout_timeline: "2026-2027",
    efficacy_summary: "High throughput AI datacenter platform.",
    competitive_edge: "Full stack CUDA moat.",
    upcoming_milestones: [
      { date: "2026-11-20", event: "Earnings Release", impact: "High Positive" }
    ],
    multi_year_forecast: [],
  };

  const populatedForecast = [
    { year: 2025, revenue_billions: 128.5, net_margin_pct: 54.0, projected_eps: 2.85, implied_pe: 45.0, implied_target: 128.25 },
    { year: 2026, revenue_billions: 165.0, net_margin_pct: 55.2, projected_eps: 3.75, implied_pe: 42.0, implied_target: 157.50 },
  ];

  it("renders the clear institutional disclosure when multi_year_forecast is empty array []", () => {
    render(<CatalystForecastCard data={baseStockData} selectedSymbol="NVDA" />);

    // Must display disclosure
    expect(screen.getByText("Verified multi-year revenue and earnings projections are unavailable.")).toBeDefined();

    // Must NOT render empty table headers
    expect(screen.queryByText("Net Margin")).toBeNull();
    expect(screen.queryByText("Proj. EPS")).toBeNull();
    expect(screen.queryByText("Target (P/E)")).toBeNull();
  });

  it("renders the clear institutional disclosure when multi_year_forecast is undefined or missing", () => {
    const dataWithoutForecast = { ...baseStockData, multi_year_forecast: undefined as any };
    render(<CatalystForecastCard data={dataWithoutForecast} selectedSymbol="NVDA" />);

    expect(screen.getByText("Verified multi-year revenue and earnings projections are unavailable.")).toBeDefined();
    expect(screen.queryByText("Net Margin")).toBeNull();
    expect(screen.queryByText("Proj. EPS")).toBeNull();
  });

  it("renders the populated forecast table with headers, rows, and disclaimer when data is present", () => {
    const dataWithForecast: CatalystForecastData = {
      ...baseStockData,
      multi_year_forecast: populatedForecast,
    };
    render(<CatalystForecastCard data={dataWithForecast} selectedSymbol="NVDA" />);

    // Projections disclosure should NOT be shown
    expect(screen.queryByText("Verified multi-year revenue and earnings projections are unavailable.")).toBeNull();

    // Table headers must be present
    expect(screen.getByText("Year")).toBeDefined();
    expect(screen.getByText("Revenue")).toBeDefined();
    expect(screen.getByText("Net Margin")).toBeDefined();
    expect(screen.getByText("Proj. EPS")).toBeDefined();
    expect(screen.getByText("Target (P/E)")).toBeDefined();

    // Data rows must be rendered
    expect(screen.getByText("2025")).toBeDefined();
    expect(screen.getByText("$128.5B")).toBeDefined();
    expect(screen.getByText("54%")).toBeDefined();
    expect(screen.getByText("$2.85")).toBeDefined();
    expect(screen.getByText("$128.25")).toBeDefined();

    expect(screen.getByText("2026")).toBeDefined();
    expect(screen.getByText("$165B")).toBeDefined();
    expect(screen.getByText("$3.75")).toBeDefined();

    // Archetype disclaimer must be present
    expect(screen.getByText(/Simulation estimates future free cash flows/i)).toBeDefined();
  });

  it("renders disclosure when ETF multi_year_forecast is empty, and renders table when populated", () => {
    const etfEmptyData: CatalystForecastData = {
      symbol: "SPY",
      company_name: "SPDR S&P 500 ETF Trust",
      sector: "Index ETF",
      efficacy_summary: "Diversified index allocation.",
      competitive_edge: "Ultra-low expense ratio.",
      upcoming_milestones: [],
      multi_year_forecast: [],
    };

    const { rerender } = render(<CatalystForecastCard data={etfEmptyData} selectedSymbol="SPY" />);

    // Empty state on ETF
    expect(screen.getByText("Verified multi-year revenue and earnings projections are unavailable.")).toBeDefined();
    expect(screen.queryByText("Net Margin")).toBeNull();

    // Populated state on ETF
    rerender(
      <CatalystForecastCard
        data={{
          ...etfEmptyData,
          multi_year_forecast: populatedForecast,
        }}
        selectedSymbol="SPY"
      />
    );

    expect(screen.queryByText("Verified multi-year revenue and earnings projections are unavailable.")).toBeNull();
    expect(screen.getByText("Net Margin")).toBeDefined();
    expect(screen.getByText("$128.5B")).toBeDefined();
    expect(screen.getByText(/Macro index simulation tracks aggregated/i)).toBeDefined();
  });
});
