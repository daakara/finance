"use client";

import React, { useEffect, useRef, useState } from "react";
import { createChart, ColorType, IChartApi, Time, CandlestickData } from "lightweight-charts";
import { CandleData } from "../lib/api";

interface TradingViewChartProps {
  data: CandleData[];
  symbol?: string;
}

const TIMEFRAME_OPTIONS = [
  { label: "1W", days: 7 },
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "1Y", days: 365 },
  { label: "ALL", days: 9999 },
];

export default function TradingViewChart({ data, symbol = "AAPL" }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [activeTimeframe, setActiveTimeframe] = useState("1M");

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0B0E14" },
        textColor: "#64748B",
      },
      grid: {
        vertLines: { color: "#1E293B" },
        horzLines: { color: "#1E293B" },
      },
      crosshair: {
        vertLine: { color: "#38BDF8", width: 1, style: 3 },
        horzLine: { color: "#38BDF8", width: 1, style: 3 },
      },
      timeScale: {
        borderColor: "#1E293B",
      },
      rightPriceScale: {
        borderColor: "#1E293B",
      },
      autoSize: true,
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: "#10B981",
      downColor: "#EF4444",
      borderVisible: false,
      wickUpColor: "#10B981",
      wickDownColor: "#EF4444",
    });

    if (data && data.length > 0) {
      const targetOption = TIMEFRAME_OPTIONS.find((o) => o.label === activeTimeframe) || TIMEFRAME_OPTIONS[1];
      const sliceCount = Math.min(data.length, targetOption.days);
      const filteredData: CandlestickData<Time>[] = data.slice(-sliceCount).map((c) => ({
        time: c.time as any,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));
      candlestickSeries.setData(filteredData);
    }

    chart.timeScale().fitContent();
    chartRef.current = chart;

    return () => {
      chart.remove();
    };
  }, [data, activeTimeframe]);

  return (
    <div className="flex flex-col h-full w-full bg-[#0B0E14] rounded-xl border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">{symbol} Multi-Timeframe Analysis</h2>
          <p className="text-xs text-slate-500">Live candlestick price series & indicators</p>
        </div>
        <div className="flex gap-1 bg-[#161B22] p-1 rounded-lg border border-slate-800">
          {TIMEFRAME_OPTIONS.map((tf) => (
            <button
              key={tf.label}
              onClick={() => setActiveTimeframe(tf.label)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                activeTimeframe === tf.label
                  ? "bg-[#38BDF8] text-slate-950 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>
      <div ref={chartContainerRef} className="flex-1 w-full min-h-[300px]" />
    </div>
  );
}

