import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import PriceChart, {
  normalizeCandleTimestamp,
  sanitizeCandles,
  DAY_TRADER_INTERVALS,
  SWING_INTERVALS,
  SWING_RANGES,
} from "../PriceChart";

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
const mockCreatePriceLine = vi.fn();
const mockRemovePriceLine = vi.fn();
let mockRangeCallback: ((range: any) => void) | null = null;

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
      subscribeVisibleLogicalRangeChange: vi.fn((cb) => {
        mockRangeCallback = cb;
      }),
    })),
  })),
}));

describe("PriceChart Adversarial Integrity & Chronology Suite (Directive 049)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockRangeCallback = null;
  });

  describe("1. test_chart_chronology_strictly_monotonic", () => {
    it("enforces ascending sort when input candles are out of chronological order", () => {
      const unsorted = [
        { time: "2025-10-24", open: 260, high: 265, low: 259, close: 262, volume: 1000 },
        { time: "2025-10-21", open: 255, high: 258, low: 254, close: 256, volume: 1000 },
        { time: "2025-10-22", open: 256, high: 259, low: 255, close: 258, volume: 1000 },
      ];

      const sanitized = sanitizeCandles(unsorted, false);
      expect(sanitized.map((c) => c.time)).toEqual(["2025-10-21", "2025-10-22", "2025-10-24"]);
    });

    it("strictly deduplicates duplicate timestamps, preserving the authoritative latest bar", () => {
      const withDuplicates = [
        { time: "2025-10-21", open: 255, high: 258, low: 254, close: 256, volume: 1000 },
        { time: "2025-10-22", open: 256, high: 259, low: 255, close: 257, volume: 1000 },
        { time: "2025-10-22", open: 256, high: 260, low: 255, close: 259, volume: 2000 },
      ];

      const sanitized = sanitizeCandles(withDuplicates, false);
      expect(sanitized.length).toBe(2);
      expect(sanitized[1].time).toBe("2025-10-22");
      expect(sanitized[1].close).toBe(259);
      expect(sanitized[1].volume).toBe(2000);
    });

    it("rejects non-finite, null, undefined, NaN, and infinity OHLCV values", () => {
      const corrupt = [
        { time: "2025-10-21", open: NaN, high: 258, low: 254, close: 256, volume: 1000 },
        { time: "2025-10-22", open: 256, high: Infinity, low: 255, close: 257, volume: 1000 },
        { time: "2025-10-23", open: 256, high: 258, low: -5, close: 257, volume: 1000 },
        { time: "2025-10-24", open: 256, high: 258, low: 255, close: 257, volume: -100 },
        { time: "2025-10-25", open: 256, high: 258, low: 255, close: 257, volume: 1000 },
      ];

      const sanitized = sanitizeCandles(corrupt, false);
      expect(sanitized.length).toBe(1);
      expect(sanitized[0].time).toBe("2025-10-25");
    });

    it("rejects invalid candle geometry where high < low, high < open, or low > close", () => {
      const geometricallyInvalid = [
        { time: "2025-10-21", open: 255, high: 250, low: 260, close: 256, volume: 1000 }, // high < low
        { time: "2025-10-22", open: 265, high: 260, low: 250, close: 255, volume: 1000 }, // high < open
        { time: "2025-10-23", open: 255, high: 260, low: 258, close: 250, volume: 1000 }, // low > close
        { time: "2025-10-24", open: 255, high: 260, low: 250, close: 258, volume: 1000 }, // valid
      ];

      const sanitized = sanitizeCandles(geometricallyInvalid, false);
      expect(sanitized.length).toBe(1);
      expect(sanitized[0].time).toBe("2025-10-24");
    });
  });

  describe("2. test_weekly_monthly_candle_aggregation_and_timestamps", () => {
    it("handles daily, weekly, and monthly timestamps as valid ISO date strings in UTC without drift", () => {
      const dailyTime = normalizeCandleTimestamp("2025-10-21", false);
      expect(dailyTime).toBe("2025-10-21");

      const weeklyTime = normalizeCandleTimestamp("2025-10-20T00:00:00Z", false);
      expect(weeklyTime).toBe("2025-10-20");

      const unixMs = 1729468800000;
      const normalizedUnix = normalizeCandleTimestamp(unixMs, false);
      expect(normalizedUnix).toBe("2024-10-21");
    });

    it("validates swing interval and range definitions for UI controls", () => {
      expect(SWING_INTERVALS.map((i) => i.label)).toEqual(["1D", "1W", "1M"]);
      expect(SWING_RANGES.map((r) => r.label)).toEqual(["1M", "6M", "1Y", "3Y", "5Y"]);
      expect(DAY_TRADER_INTERVALS.map((i) => i.label)).toEqual(["1m", "5m", "15m", "1h"]);
    });
  });

  describe("3. test_intraday_vwap_anchors_and_resets", () => {
    it("anchors and resets intraday VWAP on trading session boundary (day change)", () => {
      // Day 1: 2025-10-21 13:30 UTC & 14:00 UTC
      const day1_bar1 = 1761053400;
      const day1_bar2 = 1761055200;
      // Day 2: 2025-10-22 13:30 UTC & 14:00 UTC
      const day2_bar1 = 1761139800;
      const day2_bar2 = 1761141600;

      const intradayCandles = [
        { time: day1_bar1, open: 100, high: 105, low: 95, close: 100, volume: 1000 },
        { time: day1_bar2, open: 100, high: 115, low: 95, close: 110, volume: 1000 },
        { time: day2_bar1, open: 200, high: 205, low: 195, close: 200, volume: 500 },
        { time: day2_bar2, open: 200, high: 215, low: 195, close: 210, volume: 500 },
      ];

      render(
        <PriceChart
          symbol="AAPL"
          candles={intradayCandles}
          userRole="DAY_TRADER"
          interval="5m"
        />
      );

      expect(mockLineSetData).toHaveBeenCalled();
      const lastLineCalls = mockLineSetData.mock.calls;
      const vwapPoints = lastLineCalls[lastLineCalls.length - 1][0];

      expect(vwapPoints.length).toBe(4);
      expect(vwapPoints[0].value).toBe(100);
      expect(vwapPoints[1].value).toBe(103.33);
      // Session boundary reset on day change:
      expect(vwapPoints[2].value).toBe(200);
      expect(vwapPoints[3].value).toBe(203.33);
    });
  });

  describe("4. test_price_level_collision_avoidance", () => {
    const baseCandles = [
      { time: "2025-10-21", open: 100, high: 105, low: 98, close: 102, volume: 1000 },
      { time: "2025-10-22", open: 102, high: 108, low: 101, close: 106, volume: 1200 },
    ];

    it("renders a unified entry zone line when entry min and max are tight (<0.75%)", () => {
      render(
        <PriceChart
          symbol="AAPL"
          candles={baseCandles}
          userRole="LONG_TERM"
          optimalExecution={{
            optimal_entry_min: 100.0,
            optimal_entry_max: 100.5,
            stop_loss: 95.0,
            target_conservative: 110.0,
            target_stretch: 120.0,
          }}
        />
      );

      expect(mockCreatePriceLine).toHaveBeenCalled();
      const createdLines = mockCreatePriceLine.mock.calls.map((call) => call[0]);

      const titles = createdLines.map((l) => l.title);
      expect(titles).not.toContain("Entry Min");
      expect(titles).not.toContain("Entry Max");
      const unifiedLine = createdLines.find((l) => l.title.startsWith("Entry: $100.00-$100.50"));
      expect(unifiedLine).toBeDefined();
    });

    it("renders separate Entry Min and Entry Max lines when levels are wide (>=0.75%)", () => {
      render(
        <PriceChart
          symbol="AAPL"
          candles={baseCandles}
          userRole="LONG_TERM"
          optimalExecution={{
            optimal_entry_min: 100.0,
            optimal_entry_max: 103.0,
            stop_loss: 95.0,
            target_conservative: 110.0,
            target_stretch: 120.0,
          }}
        />
      );

      expect(mockCreatePriceLine).toHaveBeenCalled();
      const createdLines = mockCreatePriceLine.mock.calls.map((call) => call[0]);
      const titles = createdLines.map((l) => l.title);

      expect(titles).toContain("Entry Min");
      expect(titles).toContain("Entry Max");
    });
  });

  describe("5. test_chart_interaction_and_range_preservation", () => {
    it("preserves active pan/zoom when incoming streaming candle updates occur for the same dataset key", () => {
      const initialCandles = [
        { time: "2025-10-21", open: 100, high: 105, low: 98, close: 102, volume: 1000 },
        { time: "2025-10-22", open: 102, high: 108, low: 101, close: 106, volume: 1200 },
      ];

      const { rerender } = render(
        <PriceChart
          symbol="AAPL"
          candles={initialCandles}
          userRole="LONG_TERM"
          interval="1y_hist"
        />
      );

      expect(mockFitContent).toHaveBeenCalledTimes(1);

      if (mockRangeCallback) {
        mockRangeCallback({ from: 10, to: 20 });
      }

      rerender(
        <PriceChart
          symbol="AAPL"
          candles={[...initialCandles]}
          userRole="LONG_TERM"
          interval="1y_hist"
        />
      );

      expect(mockFitContent).toHaveBeenCalledTimes(1);

      // Fit button aria-label: "Reset chart zoom to fit content"
      const fitButton = screen.getByRole("button", { name: "Reset chart zoom to fit content" });
      fireEvent.click(fitButton);

      expect(mockFitContent).toHaveBeenCalledTimes(2);
    });
  });

  describe("6. test_rapid_interval_switching_race_conditions", () => {
    it("handles switching intervals via controls without rendering errors", () => {
      const handleIntervalChange = vi.fn();
      const initialCandles = [
        { time: "2025-10-21", open: 100, high: 105, low: 98, close: 102, volume: 1000 },
      ];

      render(
        <PriceChart
          symbol="AAPL"
          candles={initialCandles}
          userRole="DAY_TRADER"
          interval="5m"
          onIntervalChange={handleIntervalChange}
        />
      );

      // Buttons have aria-labels: "Set intraday interval to X (...)"
      const btn1m = screen.getByRole("button", { name: /Set intraday interval to 1m/i });
      fireEvent.click(btn1m);
      expect(handleIntervalChange).toHaveBeenCalledWith("1m");

      const btn15m = screen.getByRole("button", { name: /Set intraday interval to 15m/i });
      fireEvent.click(btn15m);
      expect(handleIntervalChange).toHaveBeenCalledWith("15m");

      const btn1h = screen.getByRole("button", { name: /Set intraday interval to 1h/i });
      fireEvent.click(btn1h);
      expect(handleIntervalChange).toHaveBeenCalledWith("1h");

      expect(screen.queryByText(/Interactive chart canvas failed to initialize/i)).toBeNull();
    });
  });
});
