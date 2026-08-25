"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType } from "lightweight-charts";

interface TradingViewChartProps {
  symbol: string;
  data?: { time: string; open: number; high: number; low: number; close: number }[];
  onTimeframeChange?: (timeframe: string) => void;
}

const TIMEFRAME_OPTIONS = [
  { label: "1D", days: 1 },
  { label: "1W", days: 7 },
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "1Y", days: 365 },
  { label: "ALL", days: 1095 },
];

export default function TradingViewChart({ symbol, data, onTimeframeChange }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState("1Y");

  // Generate realistic historical daily OHLC data leading up to TODAY (August 2026)
  const generateRealtimeHistory = (days: number) => {
    const candles = [];
    const endDate = new Date(); // Current date (2026)
    let currentClose = symbol.includes("BTC") ? 64000 : symbol.includes("ETH") ? 3400 : 180;

    for (let i = days; i >= 0; i--) {
      const d = new Date(endDate);
      d.setDate(d.getDate() - i);

      // Skip weekends for equities
      if (!symbol.includes("-USD") && (d.getDay() === 0 || d.getDay() === 6)) {
        continue;
      }

      const timeStr = d.toISOString().split("T")[0];
      const volatility = currentClose * 0.018;
      const change = (Math.random() - 0.48) * volatility;
      const open = currentClose;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * (volatility * 0.5);
      const low = Math.min(open, close) - Math.random() * (volatility * 0.5);
      currentClose = close;

      candles.push({
        time: timeStr,
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
      });
    }

    return candles;
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111722" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#1b2434" },
        horzLines: { color: "#1b2434" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 420,
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#f43f5e",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });

    const targetOption = TIMEFRAME_OPTIONS.find((t) => t.label === selectedTimeframe) || TIMEFRAME_OPTIONS[4];
    const chartData = data && data.length > 0 ? data : generateRealtimeHistory(targetOption.days);
    candlestickSeries.setData(chartData);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [symbol, data, selectedTimeframe]);

  const handleTfClick = (tfLabel: string) => {
    setSelectedTimeframe(tfLabel);
    if (onTimeframeChange) {
      onTimeframeChange(tfLabel);
    }
  };

  return (
    <div className="w-full bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <span className="text-xl font-bold text-slate-100 font-mono">{symbol}</span>
          <span className="text-xs bg-[#1b2434] text-slate-300 border border-[#364866] px-2.5 py-0.5 rounded font-mono">
            Candlestick Chart
          </span>
        </div>

        {/* Timeframe Selector Pills */}
        <div className="flex items-center space-x-1.5 bg-[#090d14] p-1 rounded-lg border border-[#243044]">
          {TIMEFRAME_OPTIONS.map((tf) => (
            <button
              key={tf.label}
              onClick={() => handleTfClick(tf.label)}
              className={`px-3 py-1 rounded text-xs font-mono font-medium transition-colors ${
                selectedTimeframe === tf.label
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-md"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      <div ref={chartContainerRef} className="w-full rounded-lg overflow-hidden border border-[#1b2434]" />
    </div>
  );
}

