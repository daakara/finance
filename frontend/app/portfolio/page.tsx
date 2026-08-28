"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import {
  PortfolioPosition,
  PortfolioSummary,
  loadPortfolioPositions,
  savePortfolioPositions,
  calculatePortfolioSummary,
  getAnonymousUserId,
  exportPortfolioToCsv,
} from "../../lib/portfolio";
import { SHARED_FACTOR_SCORES } from "../../lib/constants";
import { fetchAssetAnalytics } from "../../lib/api";
import { trackMatomoEvent } from "../../lib/matomo";

export default function PortfolioPage() {
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary>({
    totalEquity: 0,
    totalCost: 0,
    totalUnrealizedPnL: 0,
    totalUnrealizedPnLPct: 0,
    positionsCount: 0,
  });
  const [showAddModal, setShowAddModal] = useState(false);
  const [anonId, setAnonId] = useState<string>("");
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [lastSyncTime, setLastSyncTime] = useState<string>("");

  // Form State for Adding Position
  const [newSymbol, setNewSymbol] = useState("NVDA");
  const [newShares, setNewShares] = useState("10");
  const [newEntryPrice, setNewEntryPrice] = useState("200.00");
  const [newStopLoss, setNewStopLoss] = useState("185.00");
  const [newTarget, setNewTarget] = useState("240.00");

  const refreshQuotes = useCallback(async (basePositions: PortfolioPosition[]) => {
    if (basePositions.length === 0) return;
    setIsRefreshing(true);
    try {
      const updatedPromises = basePositions.map(async (pos) => {
        try {
          const res = await fetchAssetAnalytics(pos.symbol, "1mo", "1d");
          if (res && res.currentPrice && !isNaN(res.currentPrice)) {
            return {
              ...pos,
              currentPrice: res.currentPrice,
            };
          }
        } catch {
          // Fallback to SHARED_FACTOR_SCORES if API request fails
          const matched = SHARED_FACTOR_SCORES[pos.symbol.toUpperCase()];
          if (matched && matched.price) {
            return {
              ...pos,
              currentPrice: matched.price,
            };
          }
        }
        return pos;
      });

      const resolved = await Promise.all(updatedPromises);
      setPositions(resolved);
      setSummary(calculatePortfolioSummary(resolved));
      savePortfolioPositions(resolved);
      setLastSyncTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    } catch (err) {
      console.warn("Failed to refresh live portfolio quotes:", err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    setAnonId(getAnonymousUserId());
    const loaded = loadPortfolioPositions();
    setPositions(loaded);
    setSummary(calculatePortfolioSummary(loaded));
    refreshQuotes(loaded);

    const handlePurge = () => {
      refreshQuotes(loaded);
    };

    window.addEventListener("finance:cache-purge", handlePurge);
    return () => {
      window.removeEventListener("finance:cache-purge", handlePurge);
    };
  }, [refreshQuotes]);

  const handleAddPosition = (e: React.FormEvent) => {
    e.preventDefault();
    const symUpper = newSymbol.trim().toUpperCase();
    const sharesNum = parseFloat(newShares);
    const entryNum = parseFloat(newEntryPrice);
    const stopNum = newStopLoss ? parseFloat(newStopLoss) : undefined;
    const targetNum = newTarget ? parseFloat(newTarget) : undefined;

    if (!symUpper || isNaN(sharesNum) || isNaN(entryNum) || sharesNum <= 0) return;

    const matched = SHARED_FACTOR_SCORES[symUpper];
    const curPrice = matched ? matched.price : entryNum;

    const newPos: PortfolioPosition = {
      symbol: symUpper,
      name: `${symUpper} Corporation`,
      shares: sharesNum,
      entryPrice: entryNum,
      currentPrice: curPrice,
      targetPrice: targetNum,
      stopLossPrice: stopNum,
      addedAt: new Date().toISOString().split("T")[0],
      assetType: "Stock",
    };

    const updated = [newPos, ...positions.filter((p) => p.symbol !== symUpper)];
    setPositions(updated);
    setSummary(calculatePortfolioSummary(updated));
    savePortfolioPositions(updated);
    setShowAddModal(false);

    trackMatomoEvent("User Journey", "Add Portfolio Position", `${symUpper} (${sharesNum} shares)`);
  };

  const handleRemovePosition = (symbol: string) => {
    const updated = positions.filter((p) => p.symbol !== symbol);
    setPositions(updated);
    setSummary(calculatePortfolioSummary(updated));
    savePortfolioPositions(updated);

    trackMatomoEvent("User Journey", "Remove Portfolio Position", symbol);
  };

  const handleExportCsv = () => {
    exportPortfolioToCsv(positions);
    trackMatomoEvent("User Journey", "Export Portfolio CSV", `Positions count: ${positions.length}`);
  };

  const isPositive = summary.totalUnrealizedPnL >= 0;

  return (
    <div className="min-h-screen bg-[#070a10] text-slate-100 font-sans selection:bg-cyan-500 selection:text-black">
      <Navbar />

      <main className="max-w-[1450px] mx-auto p-4 sm:p-6 space-y-6 font-mono pb-28 sm:pb-8">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#243044] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-cyan-950/80 text-cyan-400 border border-cyan-800">
                🔒 ZERO-LOGIN PRIVATE STORAGE
              </span>
              <span className="text-slate-500 text-xs hidden sm:inline">• Persistent in Client Storage</span>
              {lastSyncTime && (
                <span className="text-slate-400 text-xs hidden md:inline">
                  • Synced: <span className="text-slate-300 font-bold">{lastSyncTime}</span>
                </span>
              )}
            </div>
            <h1 className="text-xl sm:text-3xl font-extrabold text-white tracking-tight mt-1">
              My Portfolio & Risk Allocations
            </h1>
            <p className="text-xs text-slate-400 font-sans mt-0.5">
              Client-side position tracker calculating real-time PnL, volatility exposure, and target risk ladders.
            </p>
          </div>

          <div className="flex items-center space-x-2 sm:space-x-3">
            <button
              onClick={() => refreshQuotes(positions)}
              disabled={isRefreshing}
              className={`px-3 py-2 bg-[#162030] hover:bg-[#1f2d44] border border-[#243044] text-slate-200 rounded-xl text-xs font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer ${
                isRefreshing ? "opacity-60 cursor-not-allowed" : ""
              }`}
            >
              <span className={isRefreshing ? "animate-spin" : ""}>🔄</span>
              <span className="hidden sm:inline">{isRefreshing ? "Syncing Quotes..." : "Refresh Quotes"}</span>
            </button>

            <button
              onClick={handleExportCsv}
              className="px-3 py-2 bg-[#162030] hover:bg-[#1f2d44] border border-[#243044] text-slate-200 rounded-xl text-xs font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer"
            >
              <span>📥</span>
              <span className="hidden sm:inline">Export CSV</span>
            </button>

            <button
              onClick={() => setShowAddModal(true)}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-bold shadow-sm flex items-center gap-1.5 transition-transform active:scale-95 cursor-pointer"
            >
              <span>➕</span>
              <span>Add Position</span>
            </button>
          </div>
        </div>

        {/* Portfolio Summary KPI Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Total Portfolio Value</span>
            <span className="text-xl sm:text-2xl font-extrabold text-white tabular-nums">
              ${summary.totalEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className="text-[10px] text-slate-500 block mt-1">{summary.positionsCount} Active Holdings</span>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Cost Basis</span>
            <span className="text-xl sm:text-2xl font-extrabold text-slate-300 tabular-nums">
              ${summary.totalCost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className="text-[10px] text-slate-500 block mt-1">Principal Capital</span>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Total Unrealized P&L</span>
            <span className={`text-xl sm:text-2xl font-extrabold tabular-nums flex items-center gap-1 ${isPositive ? "text-emerald-400" : "text-rose-400"}`}>
              <span>{isPositive ? `+$${summary.totalUnrealizedPnL.toFixed(2)}` : `-$${Math.abs(summary.totalUnrealizedPnL).toFixed(2)}`}</span>
            </span>
            <span className={`text-[10px] font-bold block mt-1 ${isPositive ? "text-emerald-500" : "text-rose-500"}`}>
              {isPositive ? `+${summary.totalUnrealizedPnLPct.toFixed(2)}%` : `${summary.totalUnrealizedPnLPct.toFixed(2)}%`} Total Return
            </span>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] shadow-xl">
            <span className="text-[11px] text-slate-400 block uppercase font-semibold">Anonymous Trader ID</span>
            <span className="text-xs font-bold text-cyan-400 truncate block mt-1">
              {anonId}
            </span>
            <span className="text-[10px] text-slate-500 block mt-1">Attributed Journey Active</span>
          </div>
        </div>

        {/* Positions Table */}
        <div className="bg-[#111722] border border-[#243044] rounded-xl shadow-xl overflow-hidden">
          <div className="p-4 border-b border-[#1b2434] flex items-center justify-between">
            <h2 className="text-sm sm:text-base font-bold text-white tracking-tight flex items-center gap-2">
              <span>📊</span>
              <span>Open Quant Holdings</span>
            </h2>
            <span className="text-xs text-slate-400">Click any symbol to open in Terminal</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#090d14] text-slate-400 border-b border-[#1b2434] uppercase text-[10px]">
                <tr>
                  <th className="py-3 px-4">Asset</th>
                  <th className="py-3 px-4">Execution Status</th>
                  <th className="py-3 px-4">Shares</th>
                  <th className="py-3 px-4">Entry Price</th>
                  <th className="py-3 px-4">Current Price</th>
                  <th className="py-3 px-4">Market Value</th>
                  <th className="py-3 px-4">Unrealized P&L</th>
                  <th className="py-3 px-4">Risk Ladder</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1b2434] font-medium tabular-nums">
                {positions.map((pos) => {
                  const mktVal = pos.shares * pos.currentPrice;
                  const cost = pos.shares * pos.entryPrice;
                  const pnl = mktVal - cost;
                  const pnlPct = cost > 0 ? (pnl / cost) * 100 : 0;
                  const posUp = pnl >= 0;

                  // Execution state alert
                  let statusBadge = null;
                  if (pos.targetPrice && pos.currentPrice >= pos.targetPrice) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold text-[10px] whitespace-nowrap animate-pulse">
                        🎯 TP1 TARGET HIT
                      </span>
                    );
                  } else if (pos.stopLossPrice && pos.currentPrice <= pos.stopLossPrice) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800 font-bold text-[10px] whitespace-nowrap">
                        🛑 STOP LOSS HIT
                      </span>
                    );
                  } else if (pos.stopLossPrice && pos.currentPrice <= pos.stopLossPrice * 1.02) {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 font-bold text-[10px] whitespace-nowrap">
                        ⚠️ NEAR STOP FLOOR
                      </span>
                    );
                  } else {
                    statusBadge = (
                      <span className="px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-400 border border-cyan-800/60 font-bold text-[10px] whitespace-nowrap">
                        🟢 ACTIVE HOLDING
                      </span>
                    );
                  }

                  return (
                    <tr key={pos.symbol} className="hover:bg-[#151e2d] transition-colors">
                      <td className="py-3 px-4">
                        <Link href={`/?symbol=${pos.symbol}`} className="font-bold text-cyan-400 hover:text-cyan-300 text-sm flex items-center gap-1.5">
                          <span>{pos.symbol}</span>
                          <span className="text-[10px] text-slate-500 font-normal">({pos.name})</span>
                        </Link>
                      </td>
                      <td className="py-3 px-4">{statusBadge}</td>
                      <td className="py-3 px-4 text-slate-200">{pos.shares}</td>
                      <td className="py-3 px-4 text-slate-300">${pos.entryPrice.toFixed(2)}</td>
                      <td className="py-3 px-4 text-white font-bold">${pos.currentPrice.toFixed(2)}</td>
                      <td className="py-3 px-4 text-slate-100 font-bold">${mktVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className="py-3 px-4">
                        <span className={`font-bold ${posUp ? "text-emerald-400" : "text-rose-400"}`}>
                          {posUp ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`} ({posUp ? `+${pnlPct.toFixed(2)}%` : `${pnlPct.toFixed(2)}%`})
                        </span>
                      </td>
                      <td className="py-3 px-4 text-[11px]">
                        <span className="text-rose-400">Stop: ${pos.stopLossPrice?.toFixed(2) || "None"}</span>
                        <span className="text-slate-600 mx-1">|</span>
                        <span className="text-emerald-400">Target: ${pos.targetPrice?.toFixed(2) || "None"}</span>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={() => handleRemovePosition(pos.symbol)}
                          className="px-2.5 py-1 text-[11px] rounded bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800/80 transition-colors"
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Add Position Modal */}
        {showAddModal && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-[#111722] border border-[#243044] rounded-2xl max-w-md w-full p-5 space-y-4 shadow-2xl">
              <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
                <h3 className="text-base font-bold text-white">Add Portfolio Holding</h3>
                <button onClick={() => setShowAddModal(false)} className="text-slate-400 hover:text-white">✕</button>
              </div>

              <form onSubmit={handleAddPosition} className="space-y-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Ticker Symbol</label>
                  <input
                    type="text"
                    value={newSymbol}
                    onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                    className="w-full bg-[#090d14] border border-[#243044] rounded-lg p-2 text-white font-bold"
                    placeholder="e.g. NVDA, AAPL, LNTH"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-slate-400 mb-1">Shares Count</label>
                    <input
                      type="number"
                      step="any"
                      value={newShares}
                      onChange={(e) => setNewShares(e.target.value)}
                      className="w-full bg-[#090d14] border border-[#243044] rounded-lg p-2 text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-slate-400 mb-1">Entry Price ($)</label>
                    <input
                      type="number"
                      step="any"
                      value={newEntryPrice}
                      onChange={(e) => setNewEntryPrice(e.target.value)}
                      className="w-full bg-[#090d14] border border-[#243044] rounded-lg p-2 text-white"
                      required
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-slate-400 mb-1">Stop Loss ($)</label>
                    <input
                      type="number"
                      step="any"
                      value={newStopLoss}
                      onChange={(e) => setNewStopLoss(e.target.value)}
                      className="w-full bg-[#090d14] border border-[#243044] rounded-lg p-2 text-rose-300"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-400 mb-1">Target Price ($)</label>
                    <input
                      type="number"
                      step="any"
                      value={newTarget}
                      onChange={(e) => setNewTarget(e.target.value)}
                      className="w-full bg-[#090d14] border border-[#243044] rounded-lg p-2 text-emerald-300"
                    />
                  </div>
                </div>

                <div className="pt-3 flex items-center justify-end space-x-2 border-t border-[#1b2434]">
                  <button
                    type="button"
                    onClick={() => setShowAddModal(false)}
                    className="px-3 py-1.5 bg-[#162030] text-slate-300 rounded-lg"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg"
                  >
                    Save Holding
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}