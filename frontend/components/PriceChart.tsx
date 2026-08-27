"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, LineStyle } from "lightweight-charts";
import { CandleData } from "../lib/api";

interface PriceChartProps {
  symbol: string;
  candles: CandleData[];
  currentPrice?: number;
  priceChangePct?: number;
  interval?: string;
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onIntervalChange?: (interval: string) => void;
  smartMoneyHeadline?: string;
  catalystHeadline?: string;
  loading?: boolean;
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
const LONG_TERM_INTERVALS = [
  { label: "1M", value: "1m_hist", desc: "1-Month Swing (Daily)" },
  { label: "6M", value: "6m_hist", desc: "6-Month Cyclical (Daily)" },
  { label: "1Y", value: "1y_hist", desc: "1-Year Macro (Daily)" },
  { label: "3Y", value: "3y_hist", desc: "3-Year Multi-Year (Weekly)" },
  { label: "5Y", value: "5y_hist", desc: "5-Year Secular (Monthly)" },
];

export default function PriceChart({
  symbol,
  candles,
  currentPrice,
  priceChangePct = 0,
  interval = "1y_hist",
  userRole = "LONG_TERM",
  onIntervalChange,
  smartMoneyHeadline,
  catalystHeadline,
  loading = false,
  technicals,
}: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const overlayLineSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

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

    chartContainerRef.current.innerHTML = "";

    const width = chartContainerRef.current.clientWidth || 800;
    const height = chartContainerRef.current.clientHeight || 340;

    const chart = createChart(chartContainerRef.current, {
      width: width,
      height: height,
      layout: {
        background: { color: "#0b0f19" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#162032" },
        horzLines: { color: "#162032" },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: "#243044",
        autoScale: true,
      },
      timeScale: {
        borderColor: "#243044",
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

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      candlestickSeriesRef.current = null;
      overlayLineSeriesRef.current = null;
    };
  }, [isIntraday]);

  // 2. Feed Data to Candlestick and Line Overlay Series whenever candles or interval changes
  useEffect(() => {
    if (!candlestickSeriesRef.current || !overlayLineSeriesRef.current || !candles || candles.length === 0) return;

    try {
      // Format timestamps according to Lightweight Charts rules
      const sanitized = candles
        .map((c) => {
          let timeVal: any = c.time;

          if (isIntraday) {
            // Numeric epoch in seconds
            if (typeof timeVal === "string") {
              if (timeVal.includes("-") || timeVal.includes("T")) {
                timeVal = Math.floor(new Date(timeVal).getTime() / 1000);
              } else {
                timeVal = Math.floor(Number(timeVal));
              }
            } else if (typeof timeVal === "number" && timeVal > 20000000000) {
              timeVal = Math.floor(timeVal / 1000);
            }
          } else {
            // YYYY-MM-DD date string
            if (typeof timeVal === "number") {
              const ms = timeVal > 20000000000 ? timeVal : timeVal * 1000;
              timeVal = new Date(ms).toISOString().split("T")[0];
            } else if (typeof timeVal === "string" && timeVal.includes("T")) {
              timeVal = timeVal.split("T")[0];
            }
          }

          return {
            time: timeVal,
            open: Number(c.open),
            high: Number(c.high),
            low: Number(c.low),
            close: Number(c.close),
            volume: Number(c.volume || 1000),
          };
        })
        .filter((c) => !isNaN(c.open) && !isNaN(c.close) && c.open > 0 && c.close > 0 && c.time)
        .sort((a, b) => {
          const tA = typeof a.time === "number" ? a.time : new Date(a.time).getTime();
          const tB = typeof b.time === "number" ? b.time : new Date(b.time).getTime();
          return tA - tB;
        });

      // Deduplicate sorted timestamps
      const uniqueCandles: any[] = [];
      const seenTimes = new Set();
      for (const item of sanitized) {
        if (!seenTimes.has(item.time)) {
          seenTimes.add(item.time);
          uniqueCandles.push(item);
        }
      }

      if (uniqueCandles.length > 0) {
        // Set Candlestick Data
        candlestickSeriesRef.current.setData(uniqueCandles.map(({ time, open, high, low, close }) => ({
          time, open, high, low, close
        })));

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
              time: c.time,
              value: Number(val.toFixed(2)),
            };
          });
          overlayLineSeriesRef.current.setData(vwapPoints);
        } else {
          const maPoints = uniqueCandles.map((c, idx, arr) => {
            const slice = arr.slice(Math.max(0, idx - 19), idx + 1);
            const avg = slice.reduce((sum, item) => sum + item.close, 0) / slice.length;
            return {
              time: c.time,
              value: Number(avg.toFixed(2)),
            };
          });
          overlayLineSeriesRef.current.setData(maPoints);
        }

        // Refresh chart view and auto-fit timeScale immediately
        if (chartRef.current) {
          chartRef.current.timeScale().fitContent();
        }
      }
    } catch (err) {
      console.warn("Error rendering chart series:", err);
    }
  }, [candles, isIntraday, interval]);

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
          <h1 id="chart-header-symbol" className="text-lg sm:text-2xl font-bold text-white tracking-tight">{symbol}</h1>
          {currentPrice && (
            <span aria-label={`Current price: $${currentPrice.toFixed(2)}`} className="text-base sm:text-xl font-bold text-slate-100 tabular-nums">
              ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
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
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded border hidden sm:inline ${
            isIntraday ? "bg-amber-950/80 text-amber-300 border-amber-800" : "bg-cyan-950/80 text-cyan-300 border-cyan-800"
          }`}>
            {isIntraday ? "⚡ VWAP Active" : "🏛️ 20 EMA Active"}
          </span>
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
        <div className="flex items-center space-x-2">
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
        className="flex-1 w-full min-h-[320px] h-[340px] sm:h-[400px] mt-2 relative rounded-lg overflow-hidden border border-[#1b2434]"
      >
        <div ref={chartContainerRef} className="w-full h-full" />
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
      </div>
    </section>
  );
}

