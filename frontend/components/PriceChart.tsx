"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, CandlestickData, LineData } from "lightweight-charts";
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

    // Create lightweight-charts instance
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
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: true,
      handleScale: true,
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    });

    // VWAP Line Overlay
    const vwapSeries = chart.addLineSeries({
      color: "#f59e0b",
      lineWidth: 2,
      title: "VWAP",
    });

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;
    vwapSeriesRef.current = vwapSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener("resize", handleResize);
    handleResize();

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !candles || candles.length === 0) return;

    // Format candle data for lightweight charts
    const formattedData: CandlestickData[] = candles
      .filter((c) => c && c.time && c.close > 0)
      .map((c) => ({
        time: c.time as any,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));

    // Ensure strictly chronological
    formattedData.sort((a, b) => {
      const timeA = typeof a.time === "number" ? a.time : new Date(a.time as string).getTime();
      const timeB = typeof b.time === "number" ? b.time : new Date(b.time as string).getTime();
      return timeA - timeB;
    });

    // Remove duplicates
    const uniqueData: CandlestickData[] = [];
    const seen = new Set();
    for (const item of formattedData) {
      if (!seen.has(item.time)) {
        seen.add(item.time);
        uniqueData.push(item);
      }
    }

    try {
      seriesRef.current.setData(uniqueData);

      // Compute simple intraday VWAP curve
      if (vwapSeriesRef.current && uniqueData.length > 5) {
        let cumVol = 0;
        let cumVP = 0;
        const vwapData: LineData[] = [];

        for (let i = 0; i < uniqueData.length; i++) {
          const c = uniqueData[i];
          const rawCandle = candles[i];
          const vol = rawCandle?.volume || 1000;
          const typ = (c.high + c.low + c.close) / 3;
          cumVol += vol;
          cumVP += typ * vol;
          vwapData.push({
            time: c.time,
            value: Number((cumVP / (cumVol || 1)).toFixed(2)),
          });
        }
        vwapSeriesRef.current.setData(vwapData);
      }

      chartRef.current?.timeScale().fitContent();
    } catch (err) {
      console.warn("Error updating chart data:", err);
    }
  }, [candles]);

  const isPositive = priceChangePct >= 0;

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 shadow-xl flex flex-col h-full">
      {/* Chart Top Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-3 mb-3">
        {/* Symbol & Price Banner */}
        <div className="flex items-center space-x-3">
          <span className="text-xl font-bold font-mono text-slate-100">{symbol}</span>
          {currentPrice && (
            <div className="flex items-baseline space-x-2">
              <span className="text-xl font-mono font-extrabold text-white">${currentPrice}</span>
              <span
                className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded ${
                  isPositive ? "bg-emerald-950 text-emerald-400 border border-emerald-800" : "bg-rose-950 text-rose-400 border border-rose-800"
                }`}
              >
                {isPositive ? "+" : ""}
                {priceChangePct}%
              </span>
            </div>
          )}
        </div>

        {/* Intraday Timeframe Pills & Technical Overlays */}
        <div className="flex items-center space-x-2">
          {technicals?.vwap && (
            <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-mono bg-amber-950/60 text-amber-300 border border-amber-800/80 px-2 py-0.5 rounded">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
              VWAP: ${technicals.vwap}
            </span>
          )}
          {technicals?.rsi_14 && (
            <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-mono bg-[#1b2434] text-cyan-300 border border-cyan-800/60 px-2 py-0.5 rounded">
              RSI: {technicals.rsi_14}
            </span>
          )}

          {/* Timeframe Buttons */}
          <div className="flex items-center bg-[#090d14] p-0.5 rounded-lg border border-[#243044]">
            {intervals.map((item) => (
              <button
                key={item.value}
                onClick={() => handleIntervalClick(item.value)}
                className={`px-2.5 py-1 text-xs font-mono font-semibold rounded-md transition-all ${
                  activeInterval === item.value
                    ? "bg-cyan-600 text-white shadow-md"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Lightweight Chart Container */}
      <div className="flex-1 w-full min-h-[360px] relative rounded-lg overflow-hidden" ref={chartContainerRef} />
    </div>
  );
}

