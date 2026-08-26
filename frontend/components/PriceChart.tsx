"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi } from "lightweight-charts";
import { CandleData } from "../lib/api";

interface PriceChartProps {
  symbol: string;
  candles: CandleData[];
  currentPrice?: number;
  priceChangePct?: number;
  interval?: string;
  onIntervalChange?: (interval: string) => void;
  technicals?: {
    vwap?: number | null;
    rsi_14?: number;
    ema_20?: number | null;
    atr_14?: number | null;
  };
}

export default function PriceChart({
  symbol,
  candles,
  currentPrice,
  priceChangePct = 0,
  interval = "1d",
  onIntervalChange,
  technicals,
}: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  const [activeInterval, setActiveInterval] = useState<string>(interval);

  // Synchronize internal active interval whenever parent prop changes
  useEffect(() => {
    setActiveInterval(interval);
  }, [interval]);

  const intervals = [
    { label: "1m", value: "1m" },
    { label: "5m", value: "5m" },
    { label: "15m", value: "15m" },
    { label: "1h", value: "1h" },
    { label: "1D", value: "1d" },
  ];

  const handleIntervalClick = (val: string) => {
    setActiveInterval(val);
    if (onIntervalChange) {
      onIntervalChange(val);
    }
  };

  const isIntraday = activeInterval !== "1d";

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Reset container DOM before instantiating new chart
    chartContainerRef.current.innerHTML = "";

    const width = chartContainerRef.current.clientWidth || 800;
    const height = chartContainerRef.current.clientHeight || 320;

    const chart = createChart(chartContainerRef.current, {
      width: width,
      height: height,
      layout: {
        background: { color: "#0b0f19" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#1e293b" },
        horzLines: { color: "#1e293b" },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: "#334155",
        autoScale: true,
      },
      timeScale: {
        borderColor: "#334155",
        timeVisible: isIntraday,
        secondsVisible: false,
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#f43f5e",
      borderUpColor: "#10b981",
      borderDownColor: "#f43f5e",
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });

    const overlaySeries = chart.addLineSeries({
      color: isIntraday ? "#f59e0b" : "#38bdf8",
      lineWidth: 2,
      title: isIntraday ? "VWAP" : "20 EMA",
    });

    seriesRef.current = candlestickSeries;
    lineSeriesRef.current = overlaySeries;
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
    };
  }, [activeInterval, isIntraday]);

  useEffect(() => {
    if (!seriesRef.current || candles.length === 0) return;

    try {
      // Map and sanitize candlestick timestamps
      const formattedData = candles
        .map((c) => {
          let timeVal: any = c.time;
          
          if (isIntraday) {
            // Intraday timestamps MUST be numeric Unix timestamp in seconds (UTCTimestamp)
            if (typeof timeVal === "string") {
              if (timeVal.includes("-") || timeVal.includes("T")) {
                timeVal = Math.floor(new Date(timeVal).getTime() / 1000);
              } else {
                timeVal = Math.floor(Number(timeVal));
              }
            } else if (typeof timeVal === "number" && timeVal > 20000000000) {
              // Milliseconds -> convert to seconds
              timeVal = Math.floor(timeVal / 1000);
            }
          } else {
            // Daily timestamps MUST be YYYY-MM-DD string
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
          };
        })
        .filter((c) => !isNaN(c.open) && !isNaN(c.close) && c.open > 0 && c.close > 0 && c.time)
        .sort((a, b) => {
          const tA = typeof a.time === "number" ? a.time : new Date(a.time).getTime();
          const tB = typeof b.time === "number" ? b.time : new Date(b.time).getTime();
          return tA - tB;
        });

      // Deduplicate timestamps strictly to prevent Lightweight Charts engine drop
      const uniqueData: any[] = [];
      const seenTimes = new Set();
      for (const item of formattedData) {
        if (!seenTimes.has(item.time)) {
          seenTimes.add(item.time);
          uniqueData.push(item);
        }
      }

      if (uniqueData.length > 0) {
        seriesRef.current.setData(uniqueData as any);

        if (lineSeriesRef.current) {
          if (isIntraday) {
            // Compute Intraday VWAP
            let cumVol = 0;
            let cumVP = 0;
            const vwapData = uniqueData.map((c, i) => {
              const vol = candles[i]?.volume || 1000;
              const typical = (c.high + c.low + c.close) / 3;
              cumVol += vol;
              cumVP += typical * vol;
              const val = cumVol > 0 ? cumVP / cumVol : c.close;
              return {
                time: c.time,
                value: Number(val.toFixed(2)),
              };
            });
            lineSeriesRef.current.setData(vwapData as any);
          } else {
            // Compute Daily 20-Day Moving Average
            const maData = uniqueData.map((c, idx, arr) => {
              const slice = arr.slice(Math.max(0, idx - 19), idx + 1);
              const avg = slice.reduce((sum, item) => sum + item.close, 0) / slice.length;
              return {
                time: c.time,
                value: Number(avg.toFixed(2)),
              };
            });
            lineSeriesRef.current.setData(maData as any);
          }
        }

        // Fit content on next tick
        setTimeout(() => {
          chartRef.current?.timeScale().fitContent();
        }, 50);
      }
    } catch (err) {
      console.warn("Error setting chart data:", err);
    }
  }, [candles, isIntraday]);

  const isPositive = priceChangePct >= 0;

  return (
    <section aria-labelledby="chart-header-symbol" className="bg-[#111722] border border-[#243044] rounded-xl p-3.5 sm:p-5 shadow-xl flex flex-col h-full font-mono">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#1b2434]">
        {/* Left: Symbol & Live Tabular Price */}
        <div className="flex items-center space-x-2.5 sm:space-x-3">
          <h1 id="chart-header-symbol" className="text-lg sm:text-2xl font-bold text-white tracking-tight">{symbol}</h1>
          {currentPrice && (
            <span aria-label={`Current price: $${currentPrice.toFixed(2)}`} className="text-base sm:text-xl font-bold text-slate-100 tabular-nums">
              ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          )}
          <span
            aria-label={`24 hour change: ${isPositive ? "+" : ""}${priceChangePct.toFixed(2)} percent`}
            className={`px-2 py-0.5 rounded text-xs font-semibold tabular-nums ${
              isPositive
                ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/80"
                : "bg-rose-950/80 text-rose-400 border border-rose-800/80"
            }`}
          >
            {isPositive ? `+${priceChangePct.toFixed(2)}%` : `${priceChangePct.toFixed(2)}%`}
          </span>
        </div>

        {/* Right: Technicals Badges & Interval Group */}
        <div className="flex items-center space-x-2">
          {technicals?.rsi_14 !== undefined && (
            <div aria-label={`Relative Strength Index 14: ${technicals.rsi_14.toFixed(1)}`} className="hidden sm:flex items-center space-x-1.5 bg-[#090d14] px-2.5 py-1 rounded-md border border-[#243044] text-[11px]">
              <span className="text-slate-400">RSI(14):</span>
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

          {/* Timeframe Interval Buttons */}
          <div role="group" aria-label="Candlestick chart interval" className="flex items-center space-x-1 bg-[#090d14] p-1 rounded-lg border border-[#243044]">
            {intervals.map((item) => (
              <button
                key={item.value}
                onClick={() => handleIntervalClick(item.value)}
                aria-pressed={activeInterval === item.value}
                aria-label={`Set timeframe interval to ${item.label}`}
                className={`px-2.5 sm:px-3 py-1 sm:py-1.5 min-h-[32px] sm:min-h-[30px] rounded text-xs font-bold transition-colors active:scale-[0.96] transition-transform duration-100 focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  activeInterval === item.value
                    ? "bg-cyan-500 text-slate-950 shadow-sm font-extrabold"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart Canvas */}
      <div
        role="region"
        aria-label={`${symbol} interactive candlestick and trend chart`}
        className="flex-1 w-full min-h-[320px] h-[340px] sm:h-[400px] mt-2 relative rounded-lg overflow-hidden border border-[#1b2434]"
      >
        <div ref={chartContainerRef} className="w-full h-full" />
      </div>
    </section>
  );
}
