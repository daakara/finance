import React from "react";
import fs from "fs";
import path from "path";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  buildChartTradeMarkers,
  classifyForm4Transaction,
  groupTradeMarkersForChart,
} from "../../lib/tradeMarkers";
import { evaluatePreFlightChecks } from "../../lib/preflightEvaluator";
import PriceChart from "../PriceChart";
import PreFlightReadinessBar from "../PreFlightReadinessBar";
import PreFlightChecklistModal from "../PreFlightChecklistModal";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock("../../lib/matomo", () => ({
  trackAnalysisToSetup: vi.fn(),
  trackWorkspaceSwitch: vi.fn(),
  trackRoleSwitch: vi.fn(),
  trackSymbolSearch: vi.fn(),
  trackPreFlightOutcome: vi.fn(),
  trackTradePlanCopied: vi.fn(),
}));

const mockCandleSetData = vi.fn();
const mockLineSetData = vi.fn();
const mockSetMarkers = vi.fn();
const mockFitContent = vi.fn();
const mockApplyOptions = vi.fn();
const mockRemove = vi.fn();
const mockCreatePriceLine = vi.fn();
const mockRemovePriceLine = vi.fn();

vi.mock("lightweight-charts", () => ({
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createChart: vi.fn(() => ({
    applyOptions: mockApplyOptions,
    remove: mockRemove,
    addCandlestickSeries: vi.fn(() => ({
      setData: mockCandleSetData,
      setMarkers: mockSetMarkers,
      applyOptions: vi.fn(),
      createPriceLine: mockCreatePriceLine,
      removePriceLine: mockRemovePriceLine,
    })),
    addLineSeries: vi.fn(() => ({
      setData: mockLineSetData,
      applyOptions: vi.fn(),
    })),
    timeScale: vi.fn(() => ({
      fitContent: mockFitContent,
      subscribeVisibleLogicalRangeChange: vi.fn(),
    })),
  })),
}));

describe("Gate 1: Trade-Marker Classification & Provenance Truth", () => {
  it("classifies SEC Form 4 sales, awards, gifts, tax withholding, exercises, and unparsed filings accurately without false Insider Buy", () => {
    expect(classifyForm4Transaction({ transaction_code: "P" })).toMatchObject({
      category: "PURCHASE",
      direction: "BUY",
      label: "Open Market Purchase",
    });
    expect(classifyForm4Transaction({ transaction_code: "S" })).toMatchObject({
      category: "SALE",
      direction: "SELL",
      label: "Open Market Sale",
    });
    expect(classifyForm4Transaction({ transaction_code: "A" })).toMatchObject({
      category: "AWARD",
      direction: "NEUTRAL",
      label: "Grant / Award",
    });
    expect(classifyForm4Transaction({ transaction_code: "F" })).toMatchObject({
      category: "WITHHOLDING",
      direction: "NEUTRAL",
      label: "Tax Withholding",
    });
    expect(classifyForm4Transaction({ transaction_code: "G" })).toMatchObject({
      category: "GIFT",
      direction: "NEUTRAL",
      label: "Bona Fide Gift",
    });
    expect(classifyForm4Transaction({ transaction_code: "M" })).toMatchObject({
      category: "EXERCISE",
      direction: "NEUTRAL",
      label: "Option Exercise",
    });
    // Curated grant text masquerading as purchase must never become Open Market Purchase
    expect(
      classifyForm4Transaction({ transaction_type: "Purchase (Grant / Long Hold)" })
    ).toMatchObject({
      category: "AWARD",
      direction: "NEUTRAL",
      label: "Grant / Award",
    });
    // Unparsed filing never infers buying conviction
    expect(
      classifyForm4Transaction({ has_parsed_transaction: false })
    ).toMatchObject({
      category: "UNPARSED",
      direction: "NEUTRAL",
      label: "Form 4 Filing (Unparsed)",
    });
  });

  it("traces mixed Form 4 filings to actual source transactions and separates retained SEC evidence from curated samples", () => {
    const markers = buildChartTradeMarkers({
      subdomains: {
        sec_form_4: {
          status: "PIPELINE_HELD",
          is_live: false,
          provenance: "SEC EDGAR Form 4 Feed",
        },
      },
      secForm4Trades: [
        {
          accession_number: "0001045810-26-000101",
          filing_date: "2026-10-15",
          is_mixed: true,
          transaction_code: "MIXED",
          sec_url: "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000101/doc.txt",
          transactions: [
            {
              transaction_id: "tx-1",
              transaction_date: "2026-10-14",
              transaction_code: "M",
              reporting_owner_name: "Exec A",
              shares: 5000,
              is_effective: true,
            },
            {
              transaction_id: "tx-2",
              transaction_date: "2026-10-14",
              transaction_code: "F",
              reporting_owner_name: "Exec A",
              shares: 2000,
              is_effective: true,
            },
            {
              transaction_id: "tx-3",
              transaction_date: "2026-10-14",
              transaction_code: "S",
              reporting_owner_name: "Exec A",
              shares: 3000,
              total_value_usd: 450000,
              is_effective: true,
            },
          ],
        },
        {
          id: "sec-nvda-1",
          filing_date: "2026-08-20",
          insider_name: "Sample Exec",
          transaction_type: "Purchase (Grant / Long Hold)",
          is_curated: true,
        },
      ],
      optionsFlow: [
        // Undated options flow must NOT fabricate "2026-04-01"
        { type: "Sweep Call", premium: "$1.2M" },
      ],
    });

    expect(markers.length).toBe(4);
    expect(markers.some((m) => m.label.includes("Insider Buy"))).toBe(false);
    expect(markers.some((m) => m.date === "2026-04-01")).toBe(false);

    expect(markers[0]).toMatchObject({
      label: "Option Exercise",
      category: "EXERCISE",
      direction: "NEUTRAL",
      provenance: "RETAINED_SEC_EVIDENCE",
      provenanceLabel: "Retained SEC Form 4",
    });
    expect(markers[1]).toMatchObject({
      label: "Tax Withholding",
      category: "WITHHOLDING",
      direction: "NEUTRAL",
      provenance: "RETAINED_SEC_EVIDENCE",
    });
    expect(markers[2]).toMatchObject({
      label: "Open Market Sale",
      category: "SALE",
      direction: "SELL",
      provenance: "RETAINED_SEC_EVIDENCE",
    });
    expect(markers[3]).toMatchObject({
      label: "Grant / Award",
      category: "AWARD",
      direction: "NEUTRAL",
      provenance: "CURATED_SAMPLE",
      provenanceLabel: "Curated Sample",
    });
  });
});

describe("Gate 2: Mobile Chart Readability, Control Separation & Safe-Area CSS", () => {
  const originalInnerWidth = window.innerWidth;

  const sampleCandles = [
    { time: "2026-10-14", open: 180, high: 185, low: 178, close: 182, volume: 1000000 },
    { time: "2026-10-15", open: 182, high: 186, low: 179, close: 184, volume: 1100000 },
    { time: "2026-11-20", open: 156, high: 158, low: 152, close: 155, volume: 1500000 },
  ];

  const sampleMarkers = buildChartTradeMarkers({
    secForm4Trades: [
      {
        accession_number: "0001045810-26-000101",
        filing_date: "2026-10-14",
        transaction_code: "S",
        insider_name: "Director X",
        shares: 10000,
      },
      {
        accession_number: "0001045810-26-000102",
        filing_date: "2026-10-14",
        transaction_code: "S",
        insider_name: "Officer Y",
        shares: 5000,
      },
      {
        accession_number: "0001045810-26-000103",
        filing_date: "2026-10-14",
        transaction_code: "G",
        insider_name: "Director Z",
        shares: 1000,
      },
    ],
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    });
  });

  it("groups same-day markers cleanly and keeps mobile price lines unobstructed while exposing compact legend & details", () => {
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 375,
    });

    const handleIntervalChange = vi.fn();

    render(
      <PriceChart
        symbol="NVDA"
        candles={sampleCandles}
        userRole="LONG_TERM"
        interval="1y_hist"
        onIntervalChange={handleIntervalChange}
        tradeMarkers={sampleMarkers}
        optimalExecution={{
          optimal_entry_min: 164.41,
          optimal_entry_max: 179.17,
          stop_loss: 153.72,
          execution_stop_visible: false,
          take_profit_1: 193.45,
          take_profit_2: 209.08,
        }}
      />
    );

    // Price lines render without covering mobile right-side candles with inline text pills
    expect(mockCreatePriceLine).toHaveBeenCalled();
    const createdLines = mockCreatePriceLine.mock.calls.map((c) => c[0]);
    expect(createdLines.every((l) => l.axisLabelVisible === false)).toBe(true);

    // Canvas markers are grouped (2 sales -> 1 SELL group, 1 gift -> 1 NEUTRAL group) and omit overlapping text on mobile
    expect(mockSetMarkers).toHaveBeenCalled();
    const renderedMarkers = mockSetMarkers.mock.calls[mockSetMarkers.mock.calls.length - 1][0];
    expect(renderedMarkers.length).toBe(2);
    expect(renderedMarkers.every((m: any) => m.text === "")).toBe(true);

    // Compact legend exposes exact prices cleanly
    const legend = screen.getByTestId("chart-levels-legend");
    expect(legend.textContent).toContain("$164.41–$179.17");
    expect(legend.textContent).toContain("Invalidation:");
    expect(legend.textContent).toContain("$153.72");
    expect(legend.textContent).toContain("$193.45");
    expect(legend.textContent).toContain("$209.08");

    // Activity details drawer toggles open and shows exact non-buy labels & provenance
    const detailsBtn = screen.getByTestId("chart-activity-details-btn");
    fireEvent.click(detailsBtn);
    const panel = screen.getByTestId("chart-activity-details-panel");
    expect(within(panel).getAllByText("Open Market Sale").length).toBe(2);
    expect(within(panel).getByText("Bona Fide Gift")).toBeDefined();
    expect(within(panel).getAllByText("Retained SEC Form 4").length).toBe(3);

    // Markers can be toggled off cleanly
    const toggleMarkersBtn = screen.getByTestId("chart-markers-toggle-btn");
    fireEvent.click(toggleMarkersBtn);
    const clearedMarkers = mockSetMarkers.mock.calls[mockSetMarkers.mock.calls.length - 1][0];
    expect(clearedMarkers).toEqual([]);

    // Separated Candle resolution and Range labels + 5Y / Fit reachable
    expect(screen.getByText("Candle")).toBeDefined();
    expect(screen.getByText("Range")).toBeDefined();
    const btn5Y = screen.getByRole("button", { name: /5-Year History/i });
    fireEvent.click(btn5Y);
    expect(handleIntervalChange).toHaveBeenCalledWith("5y_hist");
    expect(screen.getByRole("button", { name: /Reset chart zoom to fit content/i })).toBeDefined();
  });

  it("preserves desktop inline level labels and concise grouped marker tags without false Insider Buy", () => {
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      configurable: true,
      value: 1280,
    });

    const grouped = groupTradeMarkersForChart(sampleMarkers, ["2026-10-14", "2026-10-15"], false);
    expect(grouped.length).toBe(2);
    expect(grouped[0].text).toBe("S ×2");
    expect(grouped[1].text).toBe("G: Bona Fide Gift");
  });

  it("uses overflow-x: clip instead of overflow-x: hidden in globals.css so sticky mobile safe-area header stays pinned", () => {
    const css = fs.readFileSync(path.join(process.cwd(), "app/globals.css"), "utf8");
    expect(css).toContain("overflow-x: clip");
    expect(css).not.toMatch(/html,\s*\n\s*body\s*\{\s*\n\s*overflow-x:\s*hidden/);
  });
});

describe("Gate 3 & Gate 4: Preflight Summary Separation & Fixed 5-Check Denominator", () => {
  const nvdaLikeStage4UnavailableInputs = {
    symbol: "NVDA",
    currentPrice: 153.66,
    stopLoss: 145.0,
    takeProfit1: 185.0,
    riskRewardRatio: 3.45,
    isStage4: true,
    optimalEntryMin: 164.41,
    optimalEntryMax: 179.17,
    isActionable: false,
    canSizeTrade: false,
    vix: 18.0,
    vixObservationTime: new Date().toISOString(),
    dataSource: "live",
    liveSource: "ALPACA_IEX",
    liveFreshnessStatus: "LIVE",
    catalystState: "UNAVAILABLE" as const,
    catalystEvidence: {
      daysToEarnings: -8.0,
      eventDate: "2026-11-17",
      confirmationStatus: "CONFIRMED" as const,
      coverageStatus: "ELAPSED_RECENT",
      freshnessStatus: "STALE",
      riskTier: "UNVERIFIED_RISK",
    },
  };

  it("represents PASS, FAIL, and UNAVAILABLE separately without shrinking denominator 5", () => {
    const res = evaluatePreFlightChecks(nvdaLikeStage4UnavailableInputs);
    // CHK-1 PASS, CHK-2 FAIL (Stage 4), CHK-3 PASS (from catalog), CHK-4 UNAVAILABLE (elapsed), CHK-5 PASS
    expect(res.totalCount).toBe(5);
    expect(res.passedCount).toBe(3);
    expect(res.failedCount).toBe(1);
    expect(res.unavailableCount).toBe(1);
    expect(res.availableCount).toBe(4);
    expect(res.passedCount + res.failedCount + res.unavailableCount).toBe(5);
    expect(res.convictionPct).toBe(60); // 3/5, never 3/4 = 75%
    expect(res.isCleared).toBe(false);
    expect(res.resolvedCatalystState).toBe("UNAVAILABLE");
  });

  it("keeps 4/5 = 80% (never 100%) when only Check #4 is unavailable and keeps clearance blocked", () => {
    const res = evaluatePreFlightChecks({
      ...nvdaLikeStage4UnavailableInputs,
      symbol: "AAPL",
      currentPrice: 220.0,
      stopLoss: 210.0,
      takeProfit1: 245.0,
      riskRewardRatio: 2.5,
      optimalEntryMin: 215.0,
      optimalEntryMax: 222.0,
      isStage4: false,
      isActionable: true,
      canSizeTrade: true,
    });

    expect(res.totalCount).toBe(5);
    expect(res.passedCount).toBe(4);
    expect(res.failedCount).toBe(0);
    expect(res.unavailableCount).toBe(1);
    expect(res.availableCount).toBe(4);
    expect(res.convictionPct).toBe(80);
    expect(res.isCleared).toBe(false);
  });

  it("renders explicit Pass/Fail/Unavailable counts and evidence availability in PreFlightReadinessBar and PreFlightChecklistModal", () => {
    render(<PreFlightReadinessBar {...nvdaLikeStage4UnavailableInputs} />);
    const countsBar = screen.getByTestId("inline-preflight-counts");
    expect(countsBar.textContent).toContain("3/5 Pass");
    expect(countsBar.textContent).toContain("1 Fail");
    expect(countsBar.textContent).toContain("1 Unavailable");
    expect(countsBar.textContent).toContain("Evidence: 4/5 available");

    render(
      <PreFlightChecklistModal
        {...nvdaLikeStage4UnavailableInputs}
        isOpen={true}
        onClose={vi.fn()}
      />
    );
    const modalBreakdown = screen.getByTestId("modal-preflight-breakdown");
    expect(modalBreakdown.textContent).toContain("3 Pass · 1 Fail · 1 Unavailable");
    expect(modalBreakdown.textContent).toContain("Evidence: 4/5 available");
    expect(
      screen.getByText(/not an imminent-earnings hazard/i)
    ).toBeDefined();
  });
});
