"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType } from "lightweight-charts";
import { CandleData } from "../lib/api";

interface TradingViewChartProps {
  symbol: string;
  data?: CandleData[];
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

    if (data && data.length > 0) {
      const targetOption = TIMEFRAME_OPTIONS.find((t) => t.label === selectedTimeframe) || TIMEFRAME_OPTIONS[4];
      const sliceCount = Math.min(data.length, targetOption.days);
      const filteredData = data.slice(-sliceCount);
      candlestickSeries.setData(filteredData);
    }

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

  const latestCandle = data && data.length > 0 ? data[data.length - 1] : null;

  return (
    <div className="w-full bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <span className="text-xl font-bold text-slate-100 font-mono">{symbol}</span>
          {latestCandle && (
            <span className="text-sm font-mono font-bold text-cyan-400">
              ${latestCandle.close.toFixed(2)}
            </span>
          )}
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

