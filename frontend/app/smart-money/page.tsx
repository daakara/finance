"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import { fetchSmartMoneyOverview, SmartMoneyOverview } from "../../lib/api";

export default function SmartMoneyPage() {
  const [userRole, setUserRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const [data, setData] = useState<SmartMoneyOverview | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterQuery, setFilterQuery] = useState<string>("");

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setUserRole(saved);
    }
  }, []);

  const handleRoleChange = (role: "DAY_TRADER" | "LONG_TERM") => {
    setUserRole(role);
  };

  useEffect(() => {
    let isMounted = true;
    async function loadOverview() {
      setLoading(true);
      try {
        const res = await fetchSmartMoneyOverview();
        if (isMounted) setData(res);
      } catch (err) {
        console.error("Failed to load smart money overview:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }
    loadOverview();
    return () => {
      isMounted = false;
    };
  }, []);

  const isDayTrader = userRole === "DAY_TRADER";

  const congressTrades = (data?.congress_trades || []).filter((t) =>
    filterQuery ? t.ticker.includes(filterQuery.toUpperCase()) || t.politician.toLowerCase().includes(filterQuery.toLowerCase()) : true
  );

  const optionsFlow = (data?.options_flow || []).filter((f) =>
    filterQuery ? f.ticker.includes(filterQuery.toUpperCase()) || f.type.toLowerCase().includes(filterQuery.toLowerCase()) : true
  );

  return (
    <div className="min-h-screen bg-[#070a10] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Skip to Main Content Link for Accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-cyan-500 focus:text-black focus:font-bold focus:rounded-md focus:shadow-lg"
      >
        Skip to main content
      </a>

      <Navbar userRole={userRole} onRoleChange={handleRoleChange} />

      <main id="main-content" role="main" className="flex-1 max-w-[1600px] w-full mx-auto p-3 sm:p-6 space-y-5 pb-20 sm:pb-6 font-mono">
        {/* Page Header */}
        <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-6 shadow-xl flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className={`w-3 h-3 rounded-full ${isDayTrader ? "bg-amber-400" : "bg-purple-400"} animate-pulse`}></span>
              <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                <span>{isDayTrader ? "⚡ Smart Money: Institutional Options Sweeps & Dark Pool" : "🏛️ Smart Money: US Congressional STOCK Act Portfolio Tracker"}</span>
              </h1>
            </div>
            <p className="text-xs sm:text-sm text-slate-400 max-w-3xl">
              {isDayTrader
                ? "Real-time institutional call/put sweeps, block orders, dark pool cross trades, and gamma positioning across mega-caps."
                : "Tracking insider stock filings from the US House of Representatives & Senate under the STOCK Act with post-filing return attribution."}
            </p>
          </div>

          {/* Quick Stats Badges */}
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="bg-[#090d14] px-3 py-1.5 rounded-lg border border-[#243044] text-right">
              <span className="text-[10px] text-slate-500 block uppercase">30D Filings</span>
              <span className="text-base font-bold text-slate-200 tabular-nums">{data?.total_congress_filings_30d ?? 6}</span>
            </div>
            <div className="bg-[#090d14] px-3 py-1.5 rounded-lg border border-[#243044] text-right">
              <span className="text-[10px] text-slate-500 block uppercase">Political Sentiment</span>
              <span className="text-base font-bold text-emerald-400">83% Bullish</span>
            </div>
            <div className="bg-[#090d14] px-3 py-1.5 rounded-lg border border-[#243044] text-right">
              <span className="text-[10px] text-slate-500 block uppercase">Flow Vol Today</span>
              <span className="text-base font-bold text-amber-400 tabular-nums">$58.2M</span>
            </div>
          </div>
        </div>

        {/* Filter & Search Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-[#0d121c] p-3 rounded-lg border border-[#1e293b]">
          <div className="flex items-center space-x-2 w-full sm:w-auto">
            <input
              type="text"
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              placeholder="Search ticker (e.g. NVDA, NVO, AAPL)..."
              className="bg-[#070a10] border border-[#243044] rounded-md px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 w-full sm:w-64"
            />
          </div>
          <div className="text-[11px] text-slate-400">
            Showing <strong className="text-slate-200">{isDayTrader ? optionsFlow.length : congressTrades.length}</strong> smart money records
          </div>
        </div>

        {/* Table Rendering */}
        {isDayTrader ? (
          /* Day Trader Mode: Unusual Options Tape */
          <div className="bg-[#111722] border border-amber-900/40 rounded-xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-[#0b0f19] border-b border-[#1b2434] text-slate-400 text-[11px] uppercase">
                    <th className="p-3 font-semibold">Time</th>
                    <th className="p-3 font-semibold">Ticker</th>
                    <th className="p-3 font-semibold">Order Type</th>
                    <th className="p-3 font-semibold">Strike / Expiration</th>
                    <th className="p-3 font-semibold text-right">Spot Price</th>
                    <th className="p-3 font-semibold text-right">Premium ($)</th>
                    <th className="p-3 font-semibold text-right">Vol / OI</th>
                    <th className="p-3 font-semibold text-right">Sentiment</th>
                    <th className="p-3 font-semibold text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#162032]">
                  {optionsFlow.map((f, idx) => (
                    <tr key={idx} className="hover:bg-[#162030] transition-colors">
                      <td className="p-3 text-slate-400 text-[11px]">{f.time}</td>
                      <td className="p-3">
                        <span className="font-bold text-sm text-white">{f.ticker}</span>
                      </td>
                      <td className="p-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          f.type.includes("CALL")
                            ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/60"
                            : f.type.includes("PUT")
                            ? "bg-rose-950/80 text-rose-400 border border-rose-800/60"
                            : "bg-cyan-950/80 text-cyan-400 border border-cyan-800/60"
                        }`}>
                          {f.type}
                        </span>
                      </td>
                      <td className="p-3 text-slate-200">
                        <div className="font-bold">{f.strike}</div>
                        <div className="text-[10px] text-slate-400">{f.expiration}</div>
                      </td>
                      <td className="p-3 text-right font-semibold text-slate-200 tabular-nums">${f.spot_price.toFixed(2)}</td>
                      <td className="p-3 text-right font-bold text-amber-400 tabular-nums">{f.premium}</td>
                      <td className="p-3 text-right font-semibold text-slate-300 tabular-nums">{f.volume_oi_ratio}x</td>
                      <td className="p-3 text-right">
                        <span className="text-[11px] font-bold text-emerald-400">{f.sentiment}</span>
                      </td>
                      <td className="p-3 text-center">
                        <Link
                          href={`/?symbol=${f.ticker}`}
                          className="px-2.5 py-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500 hover:text-black font-bold text-[11px] transition-colors inline-block"
                        >
                          Analyze →
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          /* Long Term Mode: Congressional STOCK Act Disclosures */
          <div className="bg-[#111722] border border-[#243044] rounded-xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-[#0b0f19] border-b border-[#1b2434] text-slate-400 text-[11px] uppercase">
                    <th className="p-3 font-semibold">Politician</th>
                    <th className="p-3 font-semibold">Chamber</th>
                    <th className="p-3 font-semibold">Ticker</th>
                    <th className="p-3 font-semibold">Transaction</th>
                    <th className="p-3 font-semibold">Amount Range</th>
                    <th className="p-3 font-semibold">Filing Date</th>
                    <th className="p-3 font-semibold text-right">Return Since Filing</th>
                    <th className="p-3 font-semibold text-right">Sentiment</th>
                    <th className="p-3 font-semibold text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#162032]">
                  {congressTrades.map((t, idx) => (
                    <tr key={idx} className="hover:bg-[#162030] transition-colors">
                      <td className="p-3 font-bold text-slate-100">{t.politician}</td>
                      <td className="p-3">
                        <span className="px-1.5 py-0.5 rounded bg-[#1b2434] text-slate-400 font-semibold text-[10px]">
                          {t.chamber}
                        </span>
                      </td>
                      <td className="p-3">
                        <div className="font-bold text-sm text-cyan-400">{t.ticker}</div>
                        <div className="text-[10px] text-slate-500">{t.asset_name}</div>
                      </td>
                      <td className="p-3 font-semibold text-slate-200">{t.transaction_type}</td>
                      <td className="p-3 font-bold text-amber-300">{t.amount_range}</td>
                      <td className="p-3 text-slate-400 text-[11px]">
                        <div>{t.filing_date}</div>
                        <div className="text-[9px] text-slate-500">Lag: {t.days_to_filing} days</div>
                      </td>
                      <td className="p-3 text-right">
                        <span className={`font-bold text-sm tabular-nums ${
                          t.performance_since_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                        }`}>
                          {t.performance_since_pct >= 0 ? `+${t.performance_since_pct}%` : `${t.performance_since_pct}%`}
                        </span>
                      </td>
                      <td className="p-3 text-right">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                          t.sentiment.includes("Bullish")
                            ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/80"
                            : "bg-slate-800 text-slate-300 border-slate-700"
                        }`}>
                          {t.sentiment}
                        </span>
                      </td>
                      <td className="p-3 text-center">
                        <Link
                          href={`/?symbol=${t.ticker}`}
                          className="px-2.5 py-1 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500 hover:text-black font-bold text-[11px] transition-colors inline-block"
                        >
                          Analyze →
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
