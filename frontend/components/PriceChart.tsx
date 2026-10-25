"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, IPriceLine, LineStyle, Time } from "lightweight-charts";
import { CandleData, OptimalExecutionPlan } from "../lib/api";

export interface SanitizedCandle {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export function normalizeCandleTimestamp(timeVal: unknown, isIntraday: boolean): string | number | null {
  if (timeVal === null || timeVal === undefined) return null;

  if (isIntraday) {
    if (typeof timeVal === "number") {
      if (!Number.isFinite(timeVal) || timeVal <= 0) return null;
      const sec = timeVal > 20000000000 ? Math.floor(timeVal / 1000) : Math.floor(timeVal);
      return Number.isFinite(sec) && sec > 0 ? sec : null;
    }
    if (typeof timeVal === "string") {
      const trimmed = timeVal.trim();
      if (!trimmed) return null;
      const num = Number(trimmed);
      if (Number.isFinite(num) && num > 0) {
        const sec = num > 20000000000 ? Math.floor(num / 1000) : Math.floor(num);
        return Number.isFinite(sec) && sec > 0 ? sec : null;
      }
      const parsed = new Date(trimmed).getTime();
      if (Number.isFinite(parsed) && !isNaN(parsed) && parsed > 0) {
        return Math.floor(parsed / 1000);
      }
    }
    return null;
  }

  // Daily / Multi-day: strictly YYYY-MM-DD string
  if (typeof timeVal === "number") {
    if (!Number.isFinite(timeVal) || timeVal <= 0) return null;
    const ms = timeVal > 20000000000 ? timeVal : timeVal * 1000;
    const d = new Date(ms);
    return !isNaN(d.getTime()) ? d.toISOString().split("T")[0] : null;
  }
  if (typeof timeVal === "string") {
    const trimmed = timeVal.trim();
    if (!trimmed) return null;
    if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
      const d = new Date(trimmed + "T00:00:00Z");
      return !isNaN(d.getTime()) ? trimmed : null;
    }
    if (trimmed.includes("T")) {
      const datePart = trimmed.split("T")[0];
      if (/^\d{4}-\d{2}-\d{2}$/.test(datePart)) {
        const d = new Date(datePart + "T00:00:00Z");
        return !isNaN(d.getTime()) ? datePart : null;
      }
    }
    if (/^\d{4}\/\d{2}\/\d{2}$/.test(trimmed)) {
      const formatted = trimmed.replace(/\//g, "-");
      const d = new Date(formatted + "T00:00:00Z");
      return !isNaN(d.getTime()) ? formatted : null;
    }
    const num = Number(trimmed);
    if (Number.isFinite(num) && num > 0) {
      const ms = num > 20000000000 ? num : num * 1000;
      const d = new Date(ms);
      return !isNaN(d.getTime()) ? d.toISOString().split("T")[0] : null;
    }
    const parsed = new Date(trimmed);
    if (!isNaN(parsed.getTime())) {
      return parsed.toISOString().split("T")[0];
    }
  }
  return null;
}

export function sanitizeCandles(rawCandles: CandleData[], isIntraday: boolean): SanitizedCandle[] {
  if (!rawCandles || rawCandles.length === 0) return [];

  const validBars: SanitizedCandle[] = [];

  for (const c of rawCandles) {
    if (!c) continue;
    const time = normalizeCandleTimestamp(c.time, isIntraday);
    if (time === null) continue;

    const open = typeof c.open === "number" ? c.open : Number(c.open);
    const high = typeof c.high === "number" ? c.high : Number(c.high);
    const low = typeof c.low === "number" ? c.low : Number(c.low);
    const close = typeof c.close === "number" ? c.close : Number(c.close);
    const volume = typeof c.volume === "number" ? c.volume : Number(c.volume ?? 0);

    // Rule 1: All values must be finite numbers
    if (
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close) ||
      !Number.isFinite(volume)
    ) {
      continue;
    }

    // Rule 2: Non-positive prices rejected, non-negative volume
    if (open <= 0 || high <= 0 || low <= 0 || close <= 0 || volume < 0) {
      continue;
    }

    // Rule 3: Structural candle invariants
    // High cannot be below Low
    if (high < low) continue;
    // High cannot be below Open or Close
    if (high < open || high < close) continue;
    // Low cannot be above Open or Close
    if (low > open || low > close) continue;

    validBars.push({
      time,
      open,
      high,
      low,
      close,
      volume,
    });
  }

  // Rule 4: Stable chronological ordering
  validBars.sort((a, b) => {
    const tA = typeof a.time === "number" ? a.time : new Date(a.time + "T00:00:00Z").getTime();
    const tB = typeof b.time === "number" ? b.time : new Date(b.time + "T00:00:00Z").getTime();
    return tA - tB;
  });

  // Rule 5: Deterministic duplicate handling (preserve latest occurrence)
  const candleMap = new Map<string | number, SanitizedCandle>();
  for (const bar of validBars) {
    candleMap.set(bar.time, bar);
  }

  return Array.from(candleMap.values());
}

interface PriceChartProps {
  symbol: string;
  candles: CandleData[];
  currentPrice?: number | null;
  liveSpotPrice?: number | null;
  analysisReferencePrice?: number | null;
  optimalExecution?: OptimalExecutionPlan | null;
  marketPriceState?: {
    liveSpotPrice?: number | null;
    liveFreshness?: string;
    liveSource?: string;
    analysisReferencePrice?: number;
    analysisReferenceDate?: string;
    marketSession?: string;
  };
  priceChangePct?: number;
  interval?: string;
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onRoleChange?: (role: "DAY_TRADER" | "LONG_TERM") => void;
  onIntervalChange?: (interval: string) => void;
  smartMoneyHeadline?: string;
  catalystHeadline?: string;
  loading?: boolean;
  tradeMarkers?: Array<{
    date: string;
    label: string;
    type?: "CONGRESS" | "INSIDER" | "OPTIONS";
    amount?: string;
  }>;
  technicals?: {
    vwap?: number | null;
    rsi_14?: number;
    ema_20?: number | null;
    atr_14?: number | null;
  };
}

// ⚡ Dedicated Day Trader Timeframes (Intraday Momentum & Execution)
const DAY_TRADER_INTERVALS = [
  { label: "1m", value: "1m", desc: "1-Minute Scalp" },
  { label: "5m", value: "5m", desc: "5-Minute VWAP" },
  { label: "15m", value: "15m", desc: "15-Minute Flag" },
  { label: "1h", value: "1h", desc: "1-Hour Trend" },
];

// 🏛️ Dedicated Long-Term Macro Horizons (Expanded from 1-Month to 5-Year Deep History)
// 1D = explicit Daily Candle view (1Y window, 1d bars)
// 1W = explicit Weekly Candle view (3Y window, 1wk bars)
const LONG_TERM_INTERVALS = [
  { label: "1M", value: "1m_hist", desc: "1-Month Swing (Daily)" },
  { label: "6M", value: "6m_hist", desc: "6-Month Cyclical (Daily)" },
  { label: "1D", value: "1y_hist", desc: "Daily Candles - 1 Year" },
  { label: "1W", value: "3y_hist", desc: "Weekly Candles - 3 Years" },
  { label: "5Y", value: "5y_hist", desc: "5-Year Secular (Monthly)" },
];

export default function PriceChart({
  symbol,
  candles,
  currentPrice,
  liveSpotPrice,
  analysisReferencePrice,
  optimalExecution,
  marketPriceState,
  priceChangePct = 0,
  interval = "1y_hist",
  userRole = "LONG_TERM",
  onRoleChange,
  onIntervalChange,
  smartMoneyHeadline,
  catalystHeadline,
  loading = false,
  tradeMarkers,
  technicals,
}: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const overlayLineSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const [initEpoch, setInitEpoch] = useState<number>(0);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [chartTelemetry, setChartTelemetry] = useState<{
    candlesAvailable: boolean;
    seriesInitialized: boolean;
    candlesAssigned: boolean;
    overlayAssigned: boolean;
    candleCount: number;
  }>({
    candlesAvailable: false,
    seriesInitialized: false,
    candlesAssigned: false,
    overlayAssigned: false,
    candleCount: 0,
  });

  const activeIntervalList = userRole === "DAY_TRADER" ? DAY_TRADER_INTERVALS : LONG_TERM_INTERVALS;
  const isIntraday = userRole === "DAY_TRADER";

  // Handle button clicks directly notifying parent to trigger API load
  const handleIntervalClick = (val: string) => {
    if (onIntervalChange) {
      onIntervalChange(val);
    }
  };

  // 1. Initialize Chart Container and Series on Mount or Role Change
  useEffect(() => {
    if (!chartContainerRef.current) return;

    let chart: IChartApi | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let observer: MutationObserver | null = null;
    let tvObserver: MutationObserver | null = null;

    try {
      chartContainerRef.current.innerHTML = "";

      const width = chartContainerRef.current.clientWidth || 800;
      const height = chartContainerRef.current.clientHeight || 340;

      const isPaperTheme = typeof document !== "undefined" && document.documentElement.getAttribute("data-theme") === "paper";

      chart = createChart(chartContainerRef.current, {
        width,
        height,
        layout: {
          background: { color: isPaperTheme ? "#ffffff" : "#0b0f19" },
          textColor: isPaperTheme ? "#475569" : "#94a3b8",
        },
        grid: {
          vertLines: { color: isPaperTheme ? "#f1f5f9" : "#162032" },
          horzLines: { color: isPaperTheme ? "#f1f5f9" : "#162032" },
        },
        crosshair: {
          mode: 1,
        },
        rightPriceScale: {
          borderColor: isPaperTheme ? "#e2e8f0" : "#243044",
          autoScale: true,
        },
        timeScale: {
          borderColor: isPaperTheme ? "#e2e8f0" : "#243044",
          timeVisible: isIntraday,
          secondsVisible: false,
        },
      });

      // Candlestick Series
      const candleSeries = chart.addCandlestickSeries({
        upColor: "#10b981",
        downColor: "#f43f5e",
        borderUpColor: "#10b981",
        borderDownColor: "#f43f5e",
        wickUpColor: "#10b981",
        wickDownColor: "#f43f5e",
      });

      // Indicator Overlay Line (VWAP or 20 EMA)
      const overlayLine = chart.addLineSeries({
        color: isIntraday ? "#f59e0b" : "#38bdf8",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        title: isIntraday ? "VWAP" : "20 EMA",
        priceLineVisible: false,
      });

      candlestickSeriesRef.current = candleSeries;
      overlayLineSeriesRef.current = overlayLine;
      chartRef.current = chart;
      setRenderError(null);
      setChartTelemetry((prev) => ({ ...prev, seriesInitialized: true }));

      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
          });
        }
      };

      if (typeof ResizeObserver !== "undefined" && chartContainerRef.current) {
        resizeObserver = new ResizeObserver((entries) => {
          for (const entry of entries) {
            const w = entry.contentRect.width;
            const h = entry.contentRect.height;
            if (chartRef.current && w > 0 && h > 0) {
              chartRef.current.applyOptions({
                width: w,
                height: h,
              });
              chartRef.current.timeScale().fitContent();
            }
          }
        });
        resizeObserver.observe(chartContainerRef.current);
      }

      const handleThemeChange = () => {
        const isLight = document.documentElement.getAttribute("data-theme") === "paper";
        if (chartRef.current) {
          chartRef.current.applyOptions({
            layout: {
              background: { color: isLight ? "#ffffff" : "#0b0f19" },
              textColor: isLight ? "#475569" : "#94a3b8",
            },
            grid: {
              vertLines: { color: isLight ? "#f1f5f9" : "#162032" },
              horzLines: { color: isLight ? "#f1f5f9" : "#162032" },
            },
            rightPriceScale: {
              borderColor: isLight ? "#e2e8f0" : "#243044",
            },
            timeScale: {
              borderColor: isLight ? "#e2e8f0" : "#243044",
            },
          });
        }
      };

      window.addEventListener("resize", handleResize);
      window.addEventListener("finance:theme-change", handleThemeChange);

      observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          if (mutation.type === "attributes" && mutation.attributeName === "data-theme") {
            handleThemeChange();
          }
        }
      });
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

      const ensureAccessibleAttribution = () => {
        if (chartContainerRef.current) {
          const tvLinks = chartContainerRef.current.querySelectorAll("a");
          tvLinks.forEach((a) => {
            if (!a.getAttribute("aria-label")) {
              a.setAttribute("aria-label", "TradingView Charts");
            }
          });
        }
      };
      ensureAccessibleAttribution();
      tvObserver = new MutationObserver(ensureAccessibleAttribution);
      tvObserver.observe(chartContainerRef.current, { childList: true, subtree: true });

      return () => {
        if (resizeObserver) resizeObserver.disconnect();
        window.removeEventListener("resize", handleResize);
        window.removeEventListener("finance:theme-change", handleThemeChange);
        if (observer) observer.disconnect();
        if (tvObserver) tvObserver.disconnect();
        if (chart) {
          try {
            chart.remove();
          } catch {}
        }
        chartRef.current = null;
        candlestickSeriesRef.current = null;
        overlayLineSeriesRef.current = null;
        priceLinesRef.current = [];
        setChartTelemetry((prev) => ({
          ...prev,
          seriesInitialized: false,
          candlesAssigned: false,
          overlayAssigned: false,
        }));
      };
    } catch (initErr) {
      console.warn("Chart initialization failed:", initErr);
      if (chart) {
        try {
          chart.remove();
        } catch {}
      }
      chartRef.current = null;
      candlestickSeriesRef.current = null;
      overlayLineSeriesRef.current = null;
      priceLinesRef.current = [];
      if (chartContainerRef.current) {
        chartContainerRef.current.innerHTML = "";
      }
      setRenderError("Interactive chart canvas failed to initialize. Please reload the visualization.");
      setChartTelemetry((prev) => ({
        ...prev,
        seriesInitialized: false,
        candlesAssigned: false,
        overlayAssigned: false,
      }));
    }
  }, [isIntraday, initEpoch]);

  // 2. Feed Data to Candlestick and Line Overlay Series whenever candles or interval changes
  useEffect(() => {
    if (!candlestickSeriesRef.current || !overlayLineSeriesRef.current) {
      setChartTelemetry((prev) => ({
        ...prev,
        candlesAvailable: !!(candles && candles.length > 0),
        candlesAssigned: false,
        overlayAssigned: false,
        candleCount: 0,
      }));
      return;
    }

    if (!candles || candles.length === 0) {
      try {
        candlestickSeriesRef.current.setData([]);
        overlayLineSeriesRef.current.setData([]);
        candlestickSeriesRef.current.setMarkers([]);
      } catch {
        // Safe clear
      }
      setChartTelemetry((prev) => ({
        ...prev,
        candlesAvailable: false,
        candlesAssigned: false,
        overlayAssigned: false,
        candleCount: 0,
      }));
      return;
    }

    try {
      const uniqueCandles = sanitizeCandles(candles, isIntraday);

      if (uniqueCandles.length > 0) {
        // Set Candlestick Data
        candlestickSeriesRef.current.setData(
          uniqueCandles.map((c) => ({
            time: c.time as Time,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          }))
        );

        // Generate Overlay Line Data (VWAP for Intraday, 20 EMA for Long Term)
        if (isIntraday) {
          let cumVol = 0;
          let cumVP = 0;
          const vwapPoints = uniqueCandles.map((c) => {
            const typical = (c.high + c.low + c.close) / 3;
            cumVol += c.volume;
            cumVP += typical * c.volume;
            const val = cumVol > 0 ? cumVP / cumVol : c.close;
            return {
              time: c.time as Time,
              value: Number(val.toFixed(2)),
            };
          });
          overlayLineSeriesRef.current.setData(vwapPoints);
        } else {
          const maPoints = uniqueCandles.map((c, idx, arr) => {
            const slice = arr.slice(Math.max(0, idx - 19), idx + 1);
            const avg = slice.reduce((sum, item) => sum + item.close, 0) / slice.length;
            return {
              time: c.time as Time,
              value: Number(avg.toFixed(2)),
            };
          });
          overlayLineSeriesRef.current.setData(maPoints);
        }

        // Attach Deep Trade Timestamp Markers (Guarded)
        if (candlestickSeriesRef.current) {
          if (tradeMarkers && tradeMarkers.length > 0) {
            const markers: any[] = [];
            const candleTimes = uniqueCandles.map((c) => c.time);

            tradeMarkers.forEach((tm) => {
              if (!tm.date) return;
              const targetDateStr = tm.date.split("T")[0];

              let matchedTime = candleTimes.find((ct) => {
                if (typeof ct === "string") return ct === targetDateStr;
                return false;
              });

              if (!matchedTime && candleTimes.length > 0) {
                const targetMs = new Date(targetDateStr).getTime();
                let closest = candleTimes[0];
                let minDiff = Infinity;
                for (const ct of candleTimes) {
                  const ctMs = typeof ct === "number" ? (ct > 20000000000 ? ct : ct * 1000) : new Date(ct).getTime();
                  const diff = Math.abs(ctMs - targetMs);
                  if (diff < minDiff) {
                    minDiff = diff;
                    closest = ct;
                  }
                }
                if (minDiff <= 14 * 24 * 60 * 60 * 1000) {
                  matchedTime = closest;
                }
              }

              if (matchedTime) {
                const isCongress = tm.type === "CONGRESS";
                const isOptions = tm.type === "OPTIONS";
                markers.push({
                  time: matchedTime as Time,
                  position: isCongress ? "aboveBar" : "belowBar",
                  color: isCongress ? "#06b6d4" : isOptions ? "#f59e0b" : "#10b981",
                  shape: isCongress ? "arrowDown" : "arrowUp",
                  text: `${tm.label}${tm.amount ? ` (${tm.amount})` : ""}`,
                });
              }
            });

            markers.sort((a, b) => {
              const tA = typeof a.time === "number" ? a.time : new Date(a.time).getTime();
              const tB = typeof b.time === "number" ? b.time : new Date(b.time).getTime();
              return tA - tB;
            });

            try {
              candlestickSeriesRef.current.setMarkers(markers);
            } catch (markerErr) {
              console.warn("Failed to set trade markers:", markerErr);
              try {
                candlestickSeriesRef.current.setMarkers([]);
              } catch {}
            }
          } else {
            try {
              candlestickSeriesRef.current.setMarkers([]);
            } catch {}
          }
        }

        // Refresh chart view and auto-fit timeScale immediately
        if (chartRef.current) {
          chartRef.current.timeScale().fitContent();
          if (typeof window !== "undefined" && window.requestAnimationFrame) {
            window.requestAnimationFrame(() => {
              if (chartRef.current) {
                chartRef.current.timeScale().fitContent();
              }
            });
          }
        }
        setRenderError(null);
        setChartTelemetry({
          candlesAvailable: true,
          seriesInitialized: true,
          candlesAssigned: true,
          overlayAssigned: true,
          candleCount: uniqueCandles.length,
        });
      } else {
        candlestickSeriesRef.current.setData([]);
        overlayLineSeriesRef.current.setData([]);
        setChartTelemetry({
          candlesAvailable: false,
          seriesInitialized: true,
          candlesAssigned: false,
          overlayAssigned: false,
          candleCount: 0,
        });
      }
    } catch (err) {
      console.warn("Error rendering chart series:", err);
      setRenderError("Chart series rendering encountered an unexpected issue. Please reload the visualization.");
      setChartTelemetry((prev) => ({
        ...prev,
        candlesAssigned: false,
        overlayAssigned: false,
      }));
    }
  }, [candles, isIntraday, interval, tradeMarkers, initEpoch]);

  // 3. Execution Level Overlays (CHART-001 through CHART-010: Parity with OptimalExecutionPlan)
  useEffect(() => {
    if (!candlestickSeriesRef.current) return;

    // Clear existing price lines before attaching fresh overlays
    priceLinesRef.current.forEach((line) => {
      try {
        candlestickSeriesRef.current?.removePriceLine(line);
      } catch {
        // Safe cleanup
      }
    });
    priceLinesRef.current = [];

    if (!optimalExecution) return;

    const lines: IPriceLine[] = [];

    // CHART-002: Entry minimum parity (optimal_entry_min)
    if (typeof optimalExecution.optimal_entry_min === "number" && optimalExecution.optimal_entry_min > 0) {
      lines.push(
        candlestickSeriesRef.current.createPriceLine({
          price: optimalExecution.optimal_entry_min,
          color: "#06b6d4", // Cyan
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: "Entry Min",
        })
      );
    }

    // CHART-003: Entry maximum parity (optimal_entry_max)
    if (typeof optimalExecution.optimal_entry_max === "number" && optimalExecution.optimal_entry_max > 0) {
      lines.push(
        candlestickSeriesRef.current.createPriceLine({
          price: optimalExecution.optimal_entry_max,
          color: "#06b6d4", // Cyan
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: "Entry Max",
        })
      );
    }

    // CHART-004: Stop loss parity (stop_loss)
    if (typeof optimalExecution.stop_loss === "number" && optimalExecution.stop_loss > 0) {
      const isStopActive = optimalExecution.execution_stop_visible !== false;
      lines.push(
        candlestickSeriesRef.current.createPriceLine({
          price: optimalExecution.stop_loss,
          color: isStopActive ? "#f43f5e" : "#94a3b8",
          lineWidth: isStopActive ? 2 : 1,
          lineStyle: isStopActive ? LineStyle.Solid : LineStyle.Dotted,
          axisLabelVisible: true,
          title: isStopActive ? "Stop Loss" : "Invalidation Floor",
        })
      );
    }

    // CHART-005: Target 1 parity (take_profit_1)
    if (typeof optimalExecution.take_profit_1 === "number" && optimalExecution.take_profit_1 > 0) {
      lines.push(
        candlestickSeriesRef.current.createPriceLine({
          price: optimalExecution.take_profit_1,
          color: "#10b981", // Emerald
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: "Target 1",
        })
      );
    }

    // CHART-006: Target 2 parity (take_profit_2 when supplied)
    if (typeof optimalExecution.take_profit_2 === "number" && optimalExecution.take_profit_2 > 0) {
      lines.push(
        candlestickSeriesRef.current.createPriceLine({
          price: optimalExecution.take_profit_2,
          color: "#059669", // Dark Emerald
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: "Target 2",
        })
      );
    }

    priceLinesRef.current = lines;

    return () => {
      lines.forEach((line) => {
        try {
          candlestickSeriesRef.current?.removePriceLine(line);
        } catch {
          // Safe cleanup
        }
      });
      priceLinesRef.current = [];
    };
  }, [optimalExecution, isIntraday]);

  // Calculate dynamic period return based on the active candle dataset
  let dynamicPeriodReturn = priceChangePct;
  let dynamicTimeframeLabel = interval.replace("_hist", "").toUpperCase();

  if (candles && candles.length >= 2) {
    const firstOpen = Number(candles[0].open);
    const lastClose = Number(candles[candles.length - 1].close);
    if (firstOpen > 0 && !isNaN(firstOpen) && !isNaN(lastClose)) {
      dynamicPeriodReturn = ((lastClose - firstOpen) / firstOpen) * 100;
    }
  }

  const isPositive = dynamicPeriodReturn >= 0;

  return (
    <section aria-labelledby="chart-header-symbol" className="bg-[#111722] border border-[#243044] rounded-xl p-3.5 sm:p-5 shadow-xl flex flex-col h-full font-mono">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#1b2434]">
        <div className="space-y-1.5 w-full sm:w-auto">
                {/* Left: Symbol & Live Tabular Price */}
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <h2 id="chart-header-symbol" className="text-lg sm:text-2xl font-bold text-white tracking-tight">{symbol}</h2>
          {(() => {
            const clean = symbol.toUpperCase().replace("-USD", "");
            const isAdr = (clean.length === 5 && (clean.endsWith("Y") || clean.endsWith("F"))) || ["DHLGY", "NVO", "ASML", "TSM", "BABA", "AZN", "BP", "SHEL", "SAP", "SNY", "TM", "HMC", "VALE", "BTI", "RIO", "UL", "LVMUY", "TCEHY"].includes(clean);
            if (!isAdr) return null;
            return (
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-800/80 font-mono" title="American Depositary Receipt (Foreign Private Issuer)">
                ADR (Foreign Issuer)
              </span>
            );
          })()}
          {typeof liveSpotPrice === "number" && Number.isFinite(liveSpotPrice) && liveSpotPrice > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5">
              <span aria-label={`Live spot price: $${liveSpotPrice.toFixed(2)}`} className="text-base sm:text-xl font-bold text-emerald-400 tabular-nums">
                ${liveSpotPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span
                className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                  marketPriceState?.liveFreshness === "REALTIME"
                    ? "bg-emerald-950/80 text-emerald-300 border-emerald-800"
                    : "bg-amber-950/80 text-amber-300 border-amber-800"
                }`}
                title={`Live source: ${marketPriceState?.liveSource || "IEX"} | Freshness: ${marketPriceState?.liveFreshness || "REALTIME"}`}
              >
                {marketPriceState?.liveFreshness === "REALTIME" ? "● LIVE SPOT" : "DELAYED SPOT"}
              </span>
              {typeof (analysisReferencePrice ?? currentPrice) === "number" && (
                <span className="text-[11px] font-mono text-slate-400 border-l border-slate-700 pl-1.5 ml-0.5" title="Completed session baseline for technical indicators">
                  Analysis Ref: ${(analysisReferencePrice ?? currentPrice)!.toFixed(2)} (Prior Close)
                </span>
              )}
            </div>
          ) : typeof currentPrice === "number" && Number.isFinite(currentPrice) && currentPrice > 0 ? (
            <div className="flex items-center gap-1.5">
              <span aria-label={`Reference price: $${currentPrice.toFixed(2)}`} className="text-base sm:text-xl font-bold text-slate-100 tabular-nums">
                ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              {!isIntraday && (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#162030] text-slate-400 border border-[#243044]" title="Evaluated against last completed session close">
                  Prior Close
                </span>
              )}
            </div>
          ) : (
            <span aria-label="Current price unavailable" className="text-sm sm:text-base font-medium text-slate-400">
              — Data Unavailable
            </span>
          )}
          <span
            title={`Active ${dynamicTimeframeLabel} Horizon Return calculated from ${candles && candles.length > 0 ? (typeof candles[0].time === "number" ? new Date(candles[0].time * 1000).toLocaleDateString() : candles[0].time) : "period start"} to current price`}
            aria-label={`Timeframe ${dynamicTimeframeLabel} change: ${isPositive ? "+" : ""}${dynamicPeriodReturn.toFixed(2)} percent`}
            className={`px-2 py-0.5 rounded text-xs font-bold tabular-nums flex items-center gap-1 cursor-help ${
              isPositive
                ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/80"
                : "bg-rose-950/80 text-rose-400 border border-rose-800/80"
            }`}
          >
            <span>{isPositive ? `+${dynamicPeriodReturn.toFixed(2)}%` : `${dynamicPeriodReturn.toFixed(2)}%`}</span>
            <span className="text-[9px] opacity-90 font-semibold px-1 py-0.2 rounded bg-black/40 border border-white/10 uppercase tracking-wider">
              {dynamicTimeframeLabel}
            </span>
          </span>
          {(() => {
            if (renderError !== null) {
              return (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded border border-rose-800 bg-rose-950/80 text-rose-300 hidden sm:inline">
                  Chart Telemetry Unavailable
                </span>
              );
            }
            if (loading || !chartTelemetry.seriesInitialized || !chartTelemetry.candlesAssigned || !chartTelemetry.overlayAssigned || chartTelemetry.candleCount === 0) {
              return (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded border border-slate-700 bg-slate-900/80 text-slate-400 hidden sm:inline">
                  Chart Telemetry Pending
                </span>
              );
            }
            return (
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded border hidden sm:inline ${
                isIntraday ? "bg-amber-950/80 text-amber-300 border-amber-800" : "bg-cyan-950/80 text-cyan-300 border-cyan-800"
              }`}>
                {isIntraday ? "⚡ VWAP Active" : (chartTelemetry.candleCount >= 20 ? "🏛️ 20 EMA Active" : "🏛️ Moving Avg Active")}
              </span>
            );
          })()}
          {smartMoneyHeadline && (
              <span className="hidden xl:inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-800/80 animate-pulse">
                <span>🏛️ Smart Money:</span>
                <span className="text-white">{smartMoneyHeadline}</span>
              </span>
            )}
          </div>

          {/* 🎯 STAGE 1 DISCOVERY: Streamlined Catalyst Headline Chip */}
          {catalystHeadline && (
            <div className="flex items-center space-x-1.5 text-[11px] sm:text-xs text-amber-300 bg-amber-950/40 border border-amber-800/60 px-2.5 py-1 rounded-lg w-full">
              <span className="text-xs shrink-0">🔥</span>
              <span className="text-amber-100 font-sans leading-snug font-medium">
                {catalystHeadline}
              </span>
            </div>
          )}
        </div>

        {/* Right: Technicals Badges & Role-Adaptive Interval Group */}
        <div className="flex flex-wrap items-center justify-between sm:justify-end gap-2 w-full sm:w-auto mt-1 sm:mt-0">
          {technicals?.rsi_14 !== undefined && (
            <div aria-label={`Relative Strength Index 14: ${technicals.rsi_14.toFixed(1)}`} className="hidden sm:flex items-center space-x-1.5 bg-[#090d14] px-2.5 py-1 rounded-md border border-[#243044] text-[11px]">
              <span className="text-slate-400">RSI:</span>
              <span
                className={`font-bold tabular-nums ${
                  technicals.rsi_14 > 70
                    ? "text-rose-400"
                    : technicals.rsi_14 < 30
                    ? "text-emerald-400"
                    : "text-cyan-400"
                }`}
              >
                {technicals.rsi_14.toFixed(1)}
              </span>
            </div>
          )}

          {/* Role-Adaptive Timeframe Interval Buttons */}
          <div role="group" aria-label="Candlestick chart interval" className="flex items-center space-x-1 bg-[#090d14] p-1 rounded-lg border border-[#243044]">
            <span className="text-[10px] text-slate-500 font-bold px-1 hidden md:inline">
              {userRole === "DAY_TRADER" ? "⚡ Scalp:" : "🏛️ Horizon:"}
            </span>
            {activeIntervalList.map((item) => (
              <button
                type="button"
                key={item.value}
                onClick={() => handleIntervalClick(item.value)}
                aria-pressed={interval === item.value}
                aria-label={`Set timeframe interval to ${item.label} (${item.desc})`}
                title={item.desc}
                className={`px-2.5 sm:px-3 py-1 sm:py-1.5 min-h-[36px] sm:min-h-[30px] rounded text-xs font-bold transition-colors active:scale-[0.96] transition-transform duration-100 cursor-pointer touch-manipulation focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  interval === item.value
                    ? userRole === "DAY_TRADER"
                      ? "bg-amber-500 text-slate-950 shadow-sm font-extrabold"
                      : "bg-cyan-500 text-slate-950 shadow-sm font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart Canvas Container */}
      <div
        role="region"
        aria-label={`${symbol} interactive candlestick and trend chart`}
        className="flex-1 w-full min-h-[320px] h-[320px] sm:h-[380px] xl:h-[430px] mt-2 relative rounded-lg overflow-hidden border border-[#1b2434]"
      >
        <div ref={chartContainerRef} className="w-full h-full" />
        {/* State 1: Synchronizing / Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-[#0b0f19]/80 backdrop-blur-[2px] flex flex-col items-center justify-center space-y-3 z-10 transition-opacity">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400 animate-pulse"></span>
              <span className="w-2.5 h-2.5 rounded-full bg-purple-400 animate-ping"></span>
            </div>
            <div className="text-xs font-mono font-bold text-cyan-300 tracking-wider flex items-center gap-1.5">
              <span>⚡</span>
              <span>SYNCHRONIZING {symbol} QUANT FEED...</span>
            </div>
          </div>
        )}

        {/* State 2: Series Initialization Fault Overlay */}
        {!loading && renderError && (
          <div className="absolute inset-0 bg-[#0b0f19]/95 flex flex-col items-center justify-center p-6 text-center z-10">
            <div className="w-10 h-10 rounded-full bg-rose-950/80 border border-rose-800/80 flex items-center justify-center text-lg mb-3">
              ⚠️
            </div>
            <h3 className="text-sm font-bold text-rose-300 mb-1">
              Chart Visualization Fault
            </h3>
            <p className="text-xs text-slate-400 max-w-sm mb-4 leading-relaxed font-mono">
              {renderError}
            </p>
            <button
              type="button"
              onClick={() => {
                setRenderError(null);
                setInitEpoch((prev) => prev + 1);
                if (typeof window !== "undefined") {
                  window.dispatchEvent(new CustomEvent("finance:cache-purge"));
                }
                if (onIntervalChange) onIntervalChange(interval);
              }}
              className="px-3.5 py-1.5 rounded-lg bg-rose-950/80 hover:bg-rose-900 border border-rose-800/80 text-rose-300 text-xs font-bold transition-all active:scale-95 cursor-pointer flex items-center gap-1.5"
            >
              <span>🔄</span>
              <span>Reload Visualization</span>
            </button>
          </div>
        )}

        {/* State 3: Empty / Unavailable Candles Overlay */}
        {!loading && !renderError && (!candles || candles.length === 0) && (
          <div className="absolute inset-0 bg-[#0b0f19]/95 flex flex-col items-center justify-center p-6 text-center z-10">
            <div className="w-10 h-10 rounded-full bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-lg mb-3">
              📊
            </div>
            <h3 className="text-sm font-bold text-slate-200 mb-1">
              Candlestick Telemetry Unavailable
            </h3>
            <p className="text-xs text-slate-400 max-w-sm mb-4 leading-relaxed">
              Historical candle bars for {symbol} ({dynamicTimeframeLabel}) are not currently populated from market feeds.
            </p>
            {onIntervalChange && (
              <button
                type="button"
                onClick={() => {
                  if (typeof window !== "undefined") {
                    window.dispatchEvent(new CustomEvent("finance:cache-purge"));
                  }
                  onIntervalChange(interval);
                }}
                className="px-3.5 py-1.5 rounded-lg bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-800/80 text-cyan-300 text-xs font-bold transition-all active:scale-95 cursor-pointer flex items-center gap-1.5"
              >
                <span>🔄</span>
                <span>Retry Candle Feed</span>
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

