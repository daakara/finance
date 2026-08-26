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
  const vwapSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  const [activeInterval, setActiveInterval] = useState<string>(interval);

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

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
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
      },
      timeScale: {
        borderColor: "#334155",
        timeVisible: activeInterval !== "1d",
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

    const vwapSeries = chart.addLineSeries({
      color: "#f59e0b",
      lineWidth: 2,
      title: "VWAP",
    });

    seriesRef.current = candlestickSeries;
    vwapSeriesRef.current = vwapSeries;
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
  }, [activeInterval]);

  useEffect(() => {
    if (!seriesRef.current || candles.length === 0) return;

    try {
      const formattedData = candles
        .map((c) => ({
          time: c.time as any,
          open: Number(c.open),
          high: Number(c.high),
          low: Number(c.low),
          close: Number(c.close),
        }))
        .filter((c) => !isNaN(c.open) && !isNaN(c.close) && c.open > 0 && c.close > 0)
        .sort((a, b) => {
          const tA = typeof a.time === "number" ? a.time : new Date(a.time).getTime();
          const tB = typeof b.time === "number" ? b.time : new Date(b.time).getTime();
          return tA - tB;
        });

      if (formattedData.length > 0) {
        seriesRef.current.setData(formattedData as any);

        if (vwapSeriesRef.current) {
          let cumVol = 0;
          let cumVP = 0;
          const vwapData = formattedData.map((c, i) => {
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
          vwapSeriesRef.current.setData(vwapData as any);
        }

        chartRef.current?.timeScale().fitContent();
      }
    } catch (err) {
      console.warn("Error setting chart data:", err);
    }
  }, [candles]);

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
                    ? "bg-cyan-500 text-slate-950 shadow-sm"
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
        aria-label={`${symbol} interactive candlestick and VWAP price chart`}
        className="flex-1 w-full min-h-[280px] sm:min-h-[320px] mt-2 relative rounded-lg overflow-hidden border border-[#1b2434]"
      >
        <div ref={chartContainerRef} className="w-full h-full" />
      </div>
    </section>
  );
}

