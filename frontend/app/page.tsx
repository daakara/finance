"use client";

import { useState } from "react";
import TradingViewChart from "@/components/TradingViewChart";
import RiskMetricsCard from "@/components/RiskMetricsCard";

export default function DashboardPage() {
  const [symbol, setSymbol] = useState("AAPL");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Market Dashboard</h1>
          <p className="text-sm text-gray-400">
            Real-time multi-asset technical indicators, GARCH volatility & risk analytics
          </p>
        </div>

        <div className="flex space-x-2">
          {["AAPL", "MSFT", "GOOGL", "BTC-USD"].map((sym) => (
            <button
              key={sym}
              onClick={() => setSymbol(sym)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                symbol === sym ? "bg-sky-600 text-white" : "bg-[#161b22] text-gray-400 hover:bg-[#21262d]"
              }`}
            >
              {sym}
            </button>
          ))}
        </div>
      </div>

      <TradingViewChart symbol={symbol} />
      <RiskMetricsCard />
    </div>
  );
}

