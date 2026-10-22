import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import PriceChart, { normalizeCandleTimestamp, sanitizeCandles } from "../PriceChart";
import { createChart } from "lightweight-charts";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

// Mock matomo
vi.mock("../../lib/matomo", () => ({
  trackAnalysisToSetup: vi.fn(),
  trackWorkspaceSwitch: vi.fn(),
  trackRoleSwitch: vi.fn(),
  trackSymbolSearch: vi.fn(),
}));

const mockCandleSetData = vi.fn();
const mockLineSetData = vi.fn();
const mockSetMarkers = vi.fn();
const mockFitContent = vi.fn();
const mockApplyOptions = vi.fn();
const mockRemove = vi.fn();

vi.mock("lightweight-charts", () => ({
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createChart: vi.fn(() => ({
    applyOptions: mockApplyOptions,
    remove: mockRemove,
    addCandlestickSeries: vi.fn(() => ({
      setData: mockCandleSetData,
      setMarkers: mockSetMarkers,
      applyOptions: vi.fn(),
      createPriceLine: vi.fn(),
      removePriceLine: vi.fn(),
    })),
    addLineSeries: vi.fn(() => ({
      setData: mockLineSetData,
      applyOptions: vi.fn(),
    })),
    timeScale: vi.fn(() => ({
      fitContent: mockFitContent,
    })),
  })),
}));

describe("PriceChart Production Visibility & Self-Healing Resilience", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const validCandles = [
    { time: "2025-10-21", open: 260.92, high: 264.31, low: 260.87, close: 261.8, volume: 46695900 },
    { time: "2025-10-22", open: 261.68, high: 261.88, low: 254.49, close: 257.5, volume: 45015300 },
    { time: "2025-10-23", open: 258.98, high: 259.66, low: 257.06, close: 258.62, volume: 32754900 },
  ];

  it("renders loading synchronizing overlay when loading is true", () => {
    render(
      <PriceChart
        symbol="AAPL"
        candles={[]}
        loading={true}
        userRole="LONG_TERM"
      />
    );

    expect(screen.getByText(/SYNCHRONIZING AAPL QUANT FEED/i)).toBeTruthy();
  });

  it("does not incorrectly label previously valid candles as unavailable during a subsequent loading request", () => {
    render(
      <PriceChart
        symbol="AAPL"
        candles={validCandles}
        loading={true}
        userRole="LONG_TERM"
      />
    );

    // Shows loading synchronizing overlay
    expect(screen.getByText(/SYNCHRONIZING AAPL QUANT FEED/i)).toBeTruthy();
    // Does NOT render "Candlestick Telemetry Unavailable" overlay
    expect(screen.queryByText("Candlestick Telemetry Unavailable")).toBeNull();
  });

  it("renders explicit Candlestick Telemetry Unavailable overlay when loading is false and candles are empty", () => {
    const handleIntervalChange = vi.fn();
    render(
      <PriceChart
        symbol="AAPL"
        candles={[]}
        loading={false}
        interval="1y_hist"
        userRole="LONG_TERM"
        onIntervalChange={handleIntervalChange}
      />
    );

    expect(screen.getByText("Candlestick Telemetry Unavailable")).toBeTruthy();
    expect(screen.getByText(/Historical candle bars for AAPL/i)).toBeTruthy();

    const retryBtn = screen.getByRole("button", { name: /Retry Candle Feed/i });
    expect(retryBtn).toBeTruthy();
    fireEvent.click(retryBtn);
    expect(handleIntervalChange).toHaveBeenCalledWith("1y_hist");
  });

  it("displays 'Chart Telemetry Pending' badge instead of misleading active indicator when candles are empty", () => {
    render(
      <PriceChart
        symbol="AAPL"
        candles={[]}
        loading={false}
        userRole="LONG_TERM"
      />
    );

    expect(screen.getByText("Chart Telemetry Pending")).toBeTruthy();
    expect(screen.queryByText("🏛️ 20 EMA Active")).toBeNull();
  });

  it("displays '🏛️ 20 EMA Active' badge when at least 20 daily candles exist and are rendered", () => {
    const twentyCandles = Array.from({ length: 25 }, (_, i) => ({
      time: `2026-09-${String(i + 1).padStart(2, "0")}`,
      open: 300 + i,
      high: 305 + i,
      low: 298 + i,
      close: 302 + i,
      volume: 1000000,
    }));

    render(
      <PriceChart
        symbol="AAPL"
        candles={twentyCandles}
        loading={false}
        userRole="LONG_TERM"
      />
    );

    expect(screen.getByText("🏛️ 20 EMA Active")).toBeTruthy();
    expect(screen.queryByText("Chart Telemetry Pending")).toBeNull();
  });

  it("displays '⚡ VWAP Active' badge in DAY_TRADER mode with valid candles", () => {
    render(
      <PriceChart
        symbol="AAPL"
        candles={validCandles}
        loading={false}
        userRole="DAY_TRADER"
      />
    );

    expect(screen.getByText("⚡ VWAP Active")).toBeTruthy();
  });

  it("feeds sanitized and sorted data to candlestick series when valid candles provided", () => {
    render(
      <PriceChart
        symbol="AAPL"
        candles={validCandles}
        loading={false}
        userRole="LONG_TERM"
      />
    );

    expect(mockCandleSetData).toHaveBeenCalledTimes(1);
    const passedCandles = mockCandleSetData.mock.calls[0][0];
    expect(passedCandles.length).toBe(3);
    expect(passedCandles[0].time).toBe("2025-10-21");
    expect(passedCandles[2].time).toBe("2025-10-23");
    expect(mockFitContent).toHaveBeenCalled();
  });

  it("handles out-of-order and duplicate timestamps gracefully by sorting and deduplicating", () => {
    const unorderedCandles = [
      { time: "2025-10-23", open: 258.98, high: 259.66, low: 257.06, close: 258.62, volume: 32754900 },
      { time: "2025-10-21", open: 260.92, high: 264.31, low: 260.87, close: 261.8, volume: 46695900 },
      { time: "2025-10-21", open: 260.92, high: 264.31, low: 260.87, close: 261.85, volume: 46695900 }, // Duplicate date
      { time: "2025-10-22", open: 261.68, high: 261.88, low: 254.49, close: 257.5, volume: 45015300 },
    ];

    render(
      <PriceChart
        symbol="AAPL"
        candles={unorderedCandles}
        loading={false}
        userRole="LONG_TERM"
      />
    );

    expect(mockCandleSetData).toHaveBeenCalledTimes(1);
    const passedCandles = mockCandleSetData.mock.calls[0][0];
    expect(passedCandles.length).toBe(3); // Deduplicated 4 to 3
    expect(passedCandles[0].time).toBe("2025-10-21");
    expect(passedCandles[1].time).toBe("2025-10-22");
    expect(passedCandles[2].time).toBe("2025-10-23");
  });

  it("catches series rendering error, sanitizes user error message without stack trace, and displays retry button", () => {
    mockCandleSetData.mockImplementationOnce(() => {
      throw new Error("Internal Lightweight Charts memory fault: 0x00FF4");
    });

    render(
      <PriceChart
        symbol="AAPL"
        candles={validCandles}
        loading={false}
        userRole="LONG_TERM"
      />
    );

    expect(screen.getByText("Chart Visualization Fault")).toBeTruthy();
    // Verifies no raw stack trace or internal memory fault is leaked
    expect(screen.queryByText(/0x00FF4/)).toBeNull();
    expect(screen.getByText(/Chart series rendering encountered an unexpected issue/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Reload Visualization/i })).toBeTruthy();
  });

  it("catches chart initialization exception, cleans up partial resources, and presents safe error state", () => {
    const createChartMock = vi.mocked(createChart);
    createChartMock.mockImplementationOnce(() => {
      throw new Error("WebGLContextCreationError: canvas unavailable");
    });

    render(
      <PriceChart
        symbol="AAPL"
        candles={validCandles}
        loading={false}
        userRole="LONG_TERM"
      />
    );

    expect(screen.getByText("Chart Visualization Fault")).toBeTruthy();
    // Raw WebGL stack trace is NOT exposed
    expect(screen.queryByText(/WebGLContextCreationError/)).toBeNull();
    expect(screen.getByText(/Interactive chart canvas failed to initialize/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Reload Visualization/i })).toBeTruthy();
  });

  it("indicator status authority: never displays 'Active' when visualization error is present", () => {
    mockCandleSetData.mockImplementationOnce(() => {
      throw new Error("Render pipeline error");
    });

    const twentyCandles = Array.from({ length: 25 }, (_, i) => ({
      time: `2026-09-${String(i + 1).padStart(2, "0")}`,
      open: 300 + i,
      high: 305 + i,
      low: 298 + i,
      close: 302 + i,
      volume: 1000000,
    }));

    render(
      <PriceChart
        symbol="AAPL"
        candles={twentyCandles}
        loading={false}
        userRole="LONG_TERM"
      />
    );

    // Indicator must NOT say "20 EMA Active" when rendering failed
    expect(screen.queryByText("🏛️ 20 EMA Active")).toBeNull();
    expect(screen.getByText("Chart Telemetry Unavailable")).toBeTruthy();
  });
});

describe("Timestamp and OHLC Validation Unit Tests", () => {
  it("normalizes accepted daily timestamp formats to strict YYYY-MM-DD format", () => {
    expect(normalizeCandleTimestamp("2026-10-21", false)).toBe("2026-10-21");
    expect(normalizeCandleTimestamp("2026-10-21T14:30:00Z", false)).toBe("2026-10-21");
    expect(normalizeCandleTimestamp("2026/10/21", false)).toBe("2026-10-21");
    expect(normalizeCandleTimestamp(1761091200, false)).toBe("2025-10-22"); // Seconds
    expect(normalizeCandleTimestamp(1761091200000, false)).toBe("2025-10-22"); // Milliseconds
  });

  it("rejects invalid, empty, or non-finite timestamps", () => {
    expect(normalizeCandleTimestamp("not-a-date", false)).toBeNull();
    expect(normalizeCandleTimestamp("", false)).toBeNull();
    expect(normalizeCandleTimestamp(null, false)).toBeNull();
    expect(normalizeCandleTimestamp(undefined, false)).toBeNull();
    expect(normalizeCandleTimestamp(NaN, false)).toBeNull();
    expect(normalizeCandleTimestamp(Infinity, false)).toBeNull();
    expect(normalizeCandleTimestamp(-100, false)).toBeNull();
  });

  it("normalizes intraday timestamps to positive integer seconds", () => {
    expect(normalizeCandleTimestamp(1761091200, true)).toBe(1761091200);
    expect(normalizeCandleTimestamp(1761091200000, true)).toBe(1761091200);
    expect(normalizeCandleTimestamp("1761091200", true)).toBe(1761091200);
    expect(normalizeCandleTimestamp("2025-10-21T00:00:00Z", true)).toBe(1761004800);
  });

  it("rejects structurally invalid bars: non-positive prices, high < low, high < open/close, low > open/close, non-finite values", () => {
    const corruptBars = [
      { time: "2026-10-01", open: 0, high: 10, low: 5, close: 8, volume: 100 }, // open is 0
      { time: "2026-10-02", open: 10, high: 5, low: 8, close: 7, volume: 100 }, // high < low
      { time: "2026-10-03", open: 10, high: 9, low: 5, close: 8, volume: 100 }, // high < open
      { time: "2026-10-04", open: 10, high: 15, low: 11, close: 12, volume: 100 }, // low > open
      { time: "2026-10-05", open: 10, high: 15, low: 8, close: 16, volume: 100 }, // high < close
      { time: "2026-10-06", open: 10, high: 15, low: 9, close: 8, volume: 100 }, // low > close
      { time: "2026-10-07", open: -5, high: 15, low: 5, close: 10, volume: 100 }, // negative price
      { time: "2026-10-08", open: NaN, high: 15, low: 5, close: 10, volume: 100 }, // NaN
      { time: "2026-10-09", open: 10, high: Infinity, low: 5, close: 10, volume: 100 }, // Infinity
      { time: "2026-10-10", open: 10, high: 15, low: 5, close: 10, volume: -1 }, // Negative volume
      { time: "2026-10-11", open: 10, high: 15, low: 5, close: 10, volume: 100 }, // VALID BAR
    ];

    const sanitized = sanitizeCandles(corruptBars as any, false);
    expect(sanitized.length).toBe(1);
    expect(sanitized[0].time).toBe("2026-10-11");
    expect(sanitized[0].open).toBe(10);
    expect(sanitized[0].close).toBe(10);
  });

  it("preserves legitimate corporate-action-adjusted prices", () => {
    const adjustedBars = [
      { time: "2026-10-01", open: 260.9154, high: 262.3301, low: 259.8820, close: 261.4500, volume: 45000000 },
      { time: "2026-10-02", open: 261.5000, high: 264.2000, low: 260.1000, close: 263.8000, volume: 50000000 },
    ];

    const sanitized = sanitizeCandles(adjustedBars, false);
    expect(sanitized.length).toBe(2);
    expect(sanitized[0].open).toBe(260.9154);
    expect(sanitized[1].close).toBe(263.8000);
  });
});
