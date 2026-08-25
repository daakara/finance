"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

interface TradingViewChartProps {
  symbol: string;
  data?: { time: string; open: number; high: number; low: number; close: number }[];
}

export default function TradingViewChart({ symbol, data }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);

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
      candlestickSeries.setData(data);
    } else {
      const dummyData = [
        { time: "2024-01-01", open: 150, high: 155, low: 148, close: 153 },
        { time: "2024-01-02", open: 153, high: 158, low: 151, close: 156 },
        { time: "2024-01-03", open: 156, high: 160, low: 154, close: 158 },
        { time: "2024-01-04", open: 158, high: 162, low: 156, close: 161 },
        { time: "2024-01-05", open: 161, high: 165, low: 159, close: 163 },
      ];
      candlestickSeries.setData(dummyData);
    }

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
  }, [symbol, data]);

  return (
    <div className="w-full bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center space-x-3">
          <span className="text-xl font-bold text-slate-100 font-mono">{symbol}</span>
          <span className="text-xs bg-[#1b2434] text-slate-300 border border-[#364866] px-2.5 py-0.5 rounded font-mono">
            1D Candle
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs text-emerald-400 bg-emerald-950/60 border border-emerald-800/80 px-2.5 py-0.5 rounded-full font-mono flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
            Canvas Renderer 60FPS
          </span>
        </div>
      </div>

      <div ref={chartContainerRef} className="w-full rounded-lg overflow-hidden border border-[#1b2434]" />
    </div>
  );
}

