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

describe("Gate 1: Trade-Marker Classification, Venue Claims & Date Provenance Truth", () => {
  it("maps P=Purchase and S=Sale, adding open-market only when explicitly evidenced, and keeps awards/gifts/withholding/exercises/unknown/unparsed distinct", () => {
    expect(classifyForm4Transaction({ transaction_code: "P" })).toMatchObject({
      category: "PURCHASE",
      direction: "BUY",
      label: "Purchase",
    });
    expect(
      classifyForm4Transaction({
        transaction_code: "P",
        is_open_market_purchase_evidence: true,
      })
    ).toMatchObject({
      category: "PURCHASE",
      direction: "BUY",
      label: "Open Market Purchase",
    });
    expect(
      classifyForm4Transaction({
        transaction_code: "P",
        transaction_type: "Open Market Purchase (Code P)",
        is_open_market_purchase_evidence: false,
      })
    ).toMatchObject({
      category: "PURCHASE",
      direction: "BUY",
      label: "Purchase",
    });
    expect(classifyForm4Transaction({ transaction_code: "S" })).toMatchObject({
      category: "SALE",
      direction: "SELL",
      label: "Sale",
    });
    expect(
      classifyForm4Transaction({
        transaction_code: "S",
        transaction_type: "Open Market Sale (Code S)",
        is_open_market_purchase_evidence: false,
      })
    ).toMatchObject({
      category: "SALE",
      direction: "SELL",
      label: "Sale",
    });
    expect(
      classifyForm4Transaction({
        transaction_code: "S",
        execution_venue: "OPEN_MARKET",
      })
    ).toMatchObject({
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
      label: "Gift",
    });
    expect(classifyForm4Transaction({ transaction_code: "M" })).toMatchObject({
      category: "EXERCISE",
      direction: "NEUTRAL",
      label: "Option Exercise",
    });
    expect(classifyForm4Transaction({ transaction_code: "UNKNOWN" })).toMatchObject({
      category: "UNKNOWN",
      direction: "NEUTRAL",
      label: "Unknown Form 4 Transaction",
    });
    expect(classifyForm4Transaction({ transaction_code: "Q" })).toMatchObject({
      category: "UNKNOWN",
      direction: "NEUTRAL",
      label: "Unknown Form 4 Transaction (Code Q)",
    });
    expect(
      classifyForm4Transaction({ transaction_type: "Purchase (Grant / Long Hold)" })
    ).toMatchObject({
      category: "AWARD",
      direction: "NEUTRAL",
      label: "Grant / Award",
    });
    expect(
      classifyForm4Transaction({ has_parsed_transaction: false })
    ).toMatchObject({
      category: "UNPARSED",
      direction: "NEUTRAL",
      label: "Form 4 Filing (Unparsed)",
    });
  });

  it("uses transaction dates where available, explicitly marks filing dates when fallback is used, and separates retained SEC from curated samples", () => {
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
          accession_number: "0001045810-26-000102",
          filing_date: "2026-10-18",
          has_parsed_transaction: false,
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
        { type: "Sweep Call", premium: "$1.2M" },
      ],
    });

    expect(markers.length).toBe(5);
    expect(markers.some((m) => m.label.includes("Insider Buy"))).toBe(false);
    expect(markers.some((m) => m.date === "2026-04-01")).toBe(false);

    expect(markers[0]).toMatchObject({
      date: "2026-10-14",
      dateSource: "TRANSACTION_DATE",
      label: "Option Exercise",
      provenance: "RETAINED_SEC_EVIDENCE",
      provenanceLabel: "Retained SEC Form 4",
    });
    expect(markers[1]).toMatchObject({
      date: "2026-10-14",
      dateSource: "TRANSACTION_DATE",
      label: "Tax Withholding",
    });
    expect(markers[2]).toMatchObject({
      date: "2026-10-14",
      dateSource: "TRANSACTION_DATE",
      label: "Sale",
    });
    expect(markers[3]).toMatchObject({
      date: "2026-10-18",
      dateSource: "FILING_DATE",
      label: "Form 4 Filing (Unparsed)",
      provenance: "RETAINED_SEC_EVIDENCE",
    });
    expect(markers[4]).toMatchObject({
      date: "2026-08-20",
      dateSource: "FILING_DATE",
      label: "Grant / Award",
      provenance: "CURATED_SAMPLE",
      provenanceLabel: "Curated Sample",
    });
  });

  it("renders 'Sale' (not 'Open Market Sale') for retained NVDA accession 0001696841-26-000016 and preserves transaction date, provenance & grouped details", () => {
    const retained0001696841 = {
      form: "4",
      filing_date: "2026-11-23",
      report_date: "2026-11-23",
      acceptance_datetime: "2026-11-23T21:54:28Z",
      available_from: "2026-11-23T21:54:28Z",
      description: "Form 4 Insider Filing",
      accession_number: "0001696841-26-000016",
      sec_url: "https://www.sec.gov/Archives/edgar/data/1045810/000169684126000016/wk-form4_1795470863.xml",
      is_amendment: false,
      xml_hash: "c7ee40386d6a142d3c644acc348ce994aaf1f57a1ce8207fa40e0b9148f73077",
      is_superseded: false,
      has_parsed_transaction: true,
      transaction_count: 12,
      is_mixed: false,
      insider_name: "Teter Timothy S.",
      insider_role: "EVP, General Counsel and Sec",
      transaction_type: "Open Market Sale (Code S)",
      transaction_code: "S",
      transaction_date: "2026-11-19",
      shares_traded: 30000.0,
      total_value_usd: 4542781.48,
      price_per_share: 151.426,
      is_buy: false,
      is_sell: true,
      is_effective: true,
      effective_status: "EFFECTIVE",
      transactions: [
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:0", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 1434.0, price_per_share: 146.0298, total_value_usd: 209406.73, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:1", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 4217.0, price_per_share: 147.2413, total_value_usd: 620916.56, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:2", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 4860.0, price_per_share: 148.4804, total_value_usd: 721614.74, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:3", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 2760.0, price_per_share: 149.3466, total_value_usd: 412196.62, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:4", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 1160.0, price_per_share: 150.4106, total_value_usd: 174476.3, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:5", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 930.0, price_per_share: 151.7508, total_value_usd: 141128.24, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:6", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 4104.0, price_per_share: 152.6977, total_value_usd: 626671.36, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:7", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 2971.0, price_per_share: 153.3654, total_value_usd: 455648.6, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:8", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 1459.0, price_per_share: 154.5982, total_value_usd: 225558.77, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:9", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 2312.0, price_per_share: 155.5607, total_value_usd: 359656.34, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:10", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 2383.0, price_per_share: 156.7099, total_value_usd: 373439.69, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
        { transaction_id: "0001696841-26-000016:0001696841:NON_DERIVATIVE:11", accession_number: "0001696841-26-000016", transaction_date: "2026-11-19", transaction_code: "S", transaction_type: "Open Market Sale (Code S)", shares: 1410.0, price_per_share: 157.4947, total_value_usd: 222067.53, reporting_owner_name: "Teter Timothy S.", reporting_owner_role: "EVP, General Counsel and Sec", is_sell: true, is_open_market_purchase_evidence: false, is_effective: true, effective_status: "EFFECTIVE" },
      ],
    };

    const markers = buildChartTradeMarkers({
      subdomains: {
        sec_form_4: {
          status: "PIPELINE_HELD",
          is_live: false,
          provenance: "SEC EDGAR Form 4 Feed",
        },
      },
      secForm4Trades: [retained0001696841],
    });

    expect(markers.length).toBe(12);
    for (const m of markers) {
      expect(m.label).toBe("Sale");
      expect(m.label).not.toContain("Open Market");
      expect(m.date).toBe("2026-11-19");
      expect(m.dateSource).toBe("TRANSACTION_DATE");
      expect(m.filingDate).toBe("2026-11-23");
      expect(m.accessionNumber).toBe("0001696841-26-000016");
      expect(m.provenance).toBe("RETAINED_SEC_EVIDENCE");
      expect(m.provenanceLabel).toBe("Retained SEC Form 4");
      expect(m.actor).toBe("Teter Timothy S.");
    }

    const groupedDesktop = groupTradeMarkersForChart(markers, ["2026-11-19"], false);
    expect(groupedDesktop.length).toBe(1);
    expect(groupedDesktop[0].count).toBe(12);
    expect(groupedDesktop[0].text).toBe("S ×12");
    expect(groupedDesktop[0].items.length).toBe(12);

    render(
      <PriceChart
        symbol="NVDA"
        candles={[
          { time: "2026-11-19", open: 155, high: 158, low: 146, close: 153, volume: 1000000 },
        ]}
        userRole="LONG_TERM"
        interval="1y_hist"
        onIntervalChange={vi.fn()}
        tradeMarkers={markers}
      />
    );

    fireEvent.click(screen.getByTestId("chart-activity-details-btn"));
    const panel = screen.getByTestId("chart-activity-details-panel");
    expect(panel.textContent).not.toContain("Open Market Sale");
    expect(within(panel).getAllByText("Sale").length).toBe(12);
    expect(within(panel).getAllByText("Retained SEC Form 4").length).toBe(12);
    expect(within(panel).getAllByText("0001696841-26-000016").length).toBe(12);
    expect(within(panel).getAllByText("Tx 2026-11-19").length).toBe(12);
  });
});

describe("Gate 2: Mobile Chart Readability, Grouping Access, Control Separation & Safe-Area CSS", () => {
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
        transaction_date: "2026-10-14",
        filing_date: "2026-10-15",
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
        transaction_date: "2026-10-14",
        filing_date: "2026-10-15",
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

  it("groups markers cleanly while preserving access to every transaction, date source, and provenance in the details panel", () => {
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

    const createdLines = mockCreatePriceLine.mock.calls.map((c) => c[0]);
    expect(createdLines.every((l) => l.axisLabelVisible === false)).toBe(true);

    const renderedMarkers = mockSetMarkers.mock.calls[mockSetMarkers.mock.calls.length - 1][0];
    expect(renderedMarkers.length).toBe(2);
    expect(renderedMarkers.every((m: any) => m.text === "")).toBe(true);

    const legend = screen.getByTestId("chart-levels-legend");
    expect(legend.textContent).toContain("$164.41–$179.17");
    expect(legend.textContent).toContain("$153.72");
    expect(legend.textContent).toContain("$193.45");
    expect(legend.textContent).toContain("$209.08");

    const detailsBtn = screen.getByTestId("chart-activity-details-btn");
    fireEvent.click(detailsBtn);
    const panel = screen.getByTestId("chart-activity-details-panel");
    expect(within(panel).getAllByText("Sale").length).toBe(2);
    expect(within(panel).getByText("Gift")).toBeDefined();
    expect(within(panel).getAllByText("Retained SEC Form 4").length).toBe(3);
    expect(within(panel).getAllByText("Tx 2026-10-14").length).toBe(2);
    expect(within(panel).getByText("Filed 2026-10-14")).toBeDefined();

    const toggleMarkersBtn = screen.getByTestId("chart-markers-toggle-btn");
    fireEvent.click(toggleMarkersBtn);
    expect(mockSetMarkers.mock.calls[mockSetMarkers.mock.calls.length - 1][0]).toEqual([]);

    expect(screen.getByText("Candle")).toBeDefined();
    expect(screen.getByText("Range")).toBeDefined();
    const btn5Y = screen.getByRole("button", { name: /5-Year History/i });
    fireEvent.click(btn5Y);
    expect(handleIntervalChange).toHaveBeenCalledWith("5y_hist");
    expect(screen.getByRole("button", { name: /Reset chart zoom to fit content/i })).toBeDefined();
  });

  it("preserves grouped items and desktop labels in groupTradeMarkersForChart", () => {
    const grouped = groupTradeMarkersForChart(sampleMarkers, ["2026-10-14", "2026-10-15"], false);
    expect(grouped.length).toBe(2);
    expect(grouped[0].text).toBe("S ×2");
    expect(grouped[0].items.length).toBe(2);
    expect(grouped[1].text).toBe("G: Gift");
    expect(grouped[1].items.length).toBe(1);
  });

  it("uses overflow-x: clip instead of overflow-x: hidden in globals.css", () => {
    const css = fs.readFileSync(path.join(process.cwd(), "app/globals.css"), "utf8");
    expect(css).toContain("overflow-x: clip");
    expect(css).not.toMatch(/html,\s*\n\s*body\s*\{\s*\n\s*overflow-x:\s*hidden/);
  });
});

describe("Gate 3 & Gate 4: Preflight Accounting Invariants", () => {
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

  it("maintains PASS + FAIL + UNAVAILABLE = 5 and check.passed alignment across valid, missing-price, and Stage 4 states", () => {
    const states = [
      nvdaLikeStage4UnavailableInputs,
      { ...nvdaLikeStage4UnavailableInputs, currentPrice: 0 },
      {
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
      },
    ];

    for (const input of states) {
      const res = evaluatePreFlightChecks(input);
      expect(res.totalCount).toBe(5);
      expect(res.passedCount + res.failedCount + res.unavailableCount).toBe(5);
      expect(res.availableCount).toBe(res.passedCount + res.failedCount);
      expect(res.isCleared).toBe(false);
      for (const chk of res.checks) {
        expect(chk.passed).toBe(chk.outcomeCategory === "PASS");
      }
    }
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
  });
});
