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
        background: { type: ColorType.Solid, color: "#161b22" },
        textColor: "#c9d1d9",
      },
      grid: {
        vertLines: { color: "#21262d" },
        horzLines: { color: "#21262d" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: "#00c851",
      downColor: "#ff4444",
      borderVisible: false,
      wickUpColor: "#00c851",
      wickDownColor: "#ff4444",
    });

    if (data && data.length > 0) {
      candlestickSeries.setData(data);
    } else {
      // Mock historical data for canvas verification
      const dummyData = [
        { time: "2024-01-01", open: 150, high: 155, low: 148, close: 153 },
        { time: "2024-01-02", open: 153, high: 158, low: 151, close: 156 },
        { time: "2024-01-03", open: 156, high: 160, low: 154, close: 158 },
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
    <div className="w-full bg-[#161b22] border border-[#30363d] rounded-lg p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">60fps TradingView Chart — {symbol}</h3>
        <span className="text-xs bg-[#21262d] text-sky-400 px-3 py-1 rounded-full font-mono">
          Canvas Renderer
        </span>
      </div>
      <div ref={chartContainerRef} className="w-full h-[400px]" />
    </div>
  );
}

