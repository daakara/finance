import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import PriceChart from "../PriceChart";

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
    
    // Check retry button exists and triggers interval callback
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

  it("displays '🏛️ 20 EMA Active' badge when at least 20 daily candles exist", () => {
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

  it("catches series rendering error and renders self-healing fault overlay with retry button", () => {
    mockCandleSetData.mockImplementationOnce(() => {
      throw new Error("Lightweight Charts assertion fault: invalid time format");
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
    expect(screen.getByText(/invalid time format/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Reload Visualization/i })).toBeTruthy();
  });
});
