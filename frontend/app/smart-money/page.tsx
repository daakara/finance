"use client";

import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import SmartMoneyDetailModal from "../../components/SmartMoneyDetailModal";
import {
  fetchSmartMoneyOverview,
  SmartMoneyOverview,
  CongressTradeItem,
  OptionsFlowItem,
  SecInsiderTradeItem,
} from "../../lib/api";

function SmartMoneyContent() {
  const [data, setData] = useState<SmartMoneyOverview | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [userRole, setUserRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const [activeTab, setActiveTab] = useState<"CONGRESS" | "SEC_FORM_4" | "OPTIONS_FLOW">("CONGRESS");
  const [filterQuery, setFilterQuery] = useState<string>("");
  const [activeSector, setActiveSector] = useState<string>("ALL");

  // Interactive Forensic Modal Selection
  const [selectedCongress, setSelectedCongress] = useState<CongressTradeItem | null>(null);
  const [selectedOptions, setSelectedOptions] = useState<OptionsFlowItem | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setUserRole(saved);
      if (saved === "DAY_TRADER") setActiveTab("OPTIONS_FLOW");
      else setActiveTab("CONGRESS");
    }
  }, []);

  const handleRoleChange = (role: "DAY_TRADER" | "LONG_TERM") => {
    setUserRole(role);
    if (role === "DAY_TRADER") setActiveTab("OPTIONS_FLOW");
    else setActiveTab("CONGRESS");
  };

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
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
    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  // Filter options flow
  const optionsFlow = (data?.options_flow || []).filter((item) => {
    if (filterQuery) {
      const q = filterQuery.toUpperCase();
      const match =
        item.ticker.includes(q) ||
        item.type.toUpperCase().includes(q) ||
        item.order_type.toUpperCase().includes(q) ||
        (item.sentiment && item.sentiment.toUpperCase().includes(q));
      if (!match) return false;
    }
    return true;
  });

  // Filter congressional trades
  const congressTrades = (data?.congress_trades || []).filter((item) => {
    if (filterQuery) {
      const q = filterQuery.toUpperCase();
      const match =
        item.ticker.includes(q) ||
        item.politician.toUpperCase().includes(q) ||
        item.asset_name.toUpperCase().includes(q) ||
        (item.sector && item.sector.toUpperCase().includes(q));
      if (!match) return false;
    }
    if (activeSector !== "ALL") {
      if (!item.sector || !item.sector.toUpperCase().includes(activeSector.toUpperCase())) {
        return false;
      }
    }
    return true;
  });

  // Filter SEC Form 4 insider trades
  const secInsiders: SecInsiderTradeItem[] = (data?.sec_insider_trades || []).filter((item) => {
    if (filterQuery) {
      const q = filterQuery.toUpperCase();
      const match =
        item.ticker.includes(q) ||
        item.insider_name.toUpperCase().includes(q) ||
        item.company_name.toUpperCase().includes(q) ||
        item.role.toUpperCase().includes(q);
      if (!match) return false;
    }
    return true;
  });

  // Top Actionable Radar Assets
  const actionableAssets = [
    { ticker: "NVDA", name: "NVIDIA", type: "Nancy Pelosi Calls", postTradeAlpha: "+14.8%", dailyChange: "+3.14%", badge: "Whale Alert", bg: "from-emerald-950/60 to-slate-900", border: "border-emerald-700/60" },
    { ticker: "PLTR", name: "Palantir", type: "$190 Call Squeeze", postTradeAlpha: "+12.1%", dailyChange: "+4.12%", badge: "Gamma Squeeze", bg: "from-cyan-950/60 to-slate-900", border: "border-cyan-700/60" },
    { ticker: "VRT", name: "Vertiv", type: "Liquid Cooling Surge", postTradeAlpha: "+18.2%", dailyChange: "+2.85%", badge: "High Momentum", bg: "from-purple-950/60 to-slate-900", border: "border-purple-700/60" },
    { ticker: "NVO", name: "Novo Nordisk", type: "$145 Call Block", postTradeAlpha: "+6.4%", dailyChange: "+1.85%", badge: "Policy Fit", bg: "from-blue-950/60 to-slate-900", border: "border-blue-700/60" },
    { ticker: "CRWD", name: "CrowdStrike", type: "Cybersecurity Sweep", postTradeAlpha: "+16.5%", dailyChange: "+3.40%", badge: "Rapid Filing", bg: "from-indigo-950/60 to-slate-900", border: "border-indigo-700/60" },
    { ticker: "TSM", name: "TSMC", type: "2nm CHIPS Flow", postTradeAlpha: "+9.3%", dailyChange: "+2.15%", badge: "Whale Tier", bg: "from-amber-950/60 to-slate-900", border: "border-amber-700/60" },
  ];

  return (
    <div className="min-h-screen bg-[#070a10] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-cyan-500 focus:text-black focus:font-bold focus:rounded-md focus:shadow-lg"
      >
        Skip to main content
      </a>

      <Navbar userRole={userRole} onRoleChange={handleRoleChange} />

      <main id="main-content" role="main" className="flex-1 max-w-[1750px] w-full mx-auto p-3 sm:p-6 space-y-5 pb-20 sm:pb-6 font-mono">
        {/* Page Header */}
        <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-6 shadow-xl flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></span>
              <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                <span>🏛️ Smart Money Intelligence: Regulatory-Backed Institutional Scanner</span>
              </h1>
            </div>
            <p className="text-xs sm:text-sm text-slate-400 max-w-3xl">
              Tracking Capitol Hill STOCK Act filings, SEC Form 4 corporate insider purchases, and FINRA ATS dark pool sweeps with statutory lag and return attribution.
            </p>
          </div>

          {/* Quick Stats Badges */}
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="bg-[#090d14] px-3 py-1.5 rounded-lg border border-[#243044] text-right">
              <span className="text-[10px] text-slate-500 block uppercase">30D Disclosures</span>
              <span className="text-base font-bold text-slate-200 tabular-nums">{data?.total_congress_filings_30d ?? 12}</span>
            </div>
            <div className="bg-[#090d14] px-3 py-1.5 rounded-lg border border-[#243044] text-right">
              <span className="text-[10px] text-slate-500 block uppercase">SEC Insiders</span>
              <span className="text-base font-bold text-cyan-400 tabular-nums">{data?.total_sec_insiders_30d ?? 5}</span>
            </div>
            <div className="bg-[#090d14] px-3 py-1.5 rounded-lg border border-[#243044] text-right">
              <span className="text-[10px] text-slate-500 block uppercase">Options Volume</span>
              <span className="text-base font-bold text-amber-400 tabular-nums">$112.8M</span>
            </div>
          </div>
        </div>

        {/* TOP ACTIONABLE SMART MONEY RADAR (Stocknear style instant cards) */}
        <section aria-label="Top Smart Money Discoveries" className="space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-slate-300">
            <span className="flex items-center gap-1.5">
              <span>🔥</span>
              <span>TOP ACTIONABLE SMART MONEY DISCOVERIES TODAY</span>
            </span>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">Instant Terminal Jump</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {actionableAssets.map((card, i) => (
              <Link
                key={i}
                href={`/?symbol=${card.ticker}`}
                className={`bg-gradient-to-b ${card.bg} border ${card.border} rounded-xl p-3 hover:scale-[1.02] transition-transform shadow-lg group block`}
              >
                <div className="flex items-start justify-between gap-1">
                  <div>
                    <span className="text-base font-black text-white group-hover:text-cyan-400 font-mono tracking-tight">
                      {card.ticker}
                    </span>
                    <div className="text-[10px] text-slate-400">
                      24h: <span className="text-emerald-400 font-bold">{card.dailyChange}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-black text-emerald-400 tabular-nums">{card.postTradeAlpha}</div>
                    <div className="text-[9px] text-cyan-400/90 font-semibold uppercase tracking-wider">Since Filing</div>
                  </div>
                </div>
                <div className="text-[11px] text-slate-300 font-medium truncate mt-0.5">{card.name}</div>
                <div className="text-[10px] text-slate-400 truncate mt-1">{card.type}</div>
                <div className="mt-2 pt-2 border-t border-white/10 flex items-center justify-between text-[10px]">
                  <span className="text-cyan-300 font-semibold">{card.badge}</span>
                  <span className="text-slate-400 group-hover:text-white">Analyze →</span>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* 🎛️ 3-WAY REGULATORY FEED SUB-VIEW SELECTOR */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-[#0d121c] p-2.5 rounded-xl border border-[#1e293b]">
          <div className="flex items-center space-x-1.5">
            <button
              onClick={() => setActiveTab("CONGRESS")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
                activeTab === "CONGRESS"
                  ? "bg-purple-900/80 text-purple-200 border border-purple-600 shadow-md"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              <span>🏛️</span>
              <span>Capitol Hill STOCK Act ({congressTrades.length})</span>
            </button>

            <button
              onClick={() => setActiveTab("SEC_FORM_4")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
                activeTab === "SEC_FORM_4"
                  ? "bg-cyan-900/80 text-cyan-200 border border-cyan-600 shadow-md"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              <span>🏢</span>
              <span>Corporate Insiders SEC Form 4 ({secInsiders.length})</span>
            </button>

            <button
              onClick={() => setActiveTab("OPTIONS_FLOW")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
                activeTab === "OPTIONS_FLOW"
                  ? "bg-amber-900/80 text-amber-200 border border-amber-600 shadow-md"
                  : "text-slate-400 hover:text-slate-200 hover:bg-[#162030]"
              }`}
            >
              <span>⚡</span>
              <span>Options Sweeps & FINRA Dark Pool ({optionsFlow.length})</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="text"
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              placeholder="Search ticker, politician, or CEO..."
              className="bg-[#070a10] border border-[#243044] rounded-md px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 w-48 sm:w-64"
            />
          </div>
        </div>

        {/* Dynamic Sector Filter for Congressional Tab */}
        {activeTab === "CONGRESS" && (
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
            <span className="text-[11px] text-slate-500 font-bold uppercase mr-1">Sector:</span>
            {["ALL", "Semiconductors", "Healthcare", "AI", "Defense"].map((sec) => (
              <button
                key={sec}
                onClick={() => setActiveSector(sec)}
                className={`px-2.5 py-1 rounded text-[11px] font-bold transition-colors ${
                  activeSector === sec
                    ? "bg-purple-900 text-purple-200 border border-purple-500"
                    : "bg-[#090d14] text-slate-400 hover:text-slate-200 border border-[#1e293b]"
                }`}
              >
                {sec}
              </button>
            ))}
          </div>
        )}

        {/* TABLE RENDERING BY ACTIVE REGULATORY TAB */}
        {activeTab === "OPTIONS_FLOW" ? (
          /* TAB 1: Institutional Options Sweeps & Dark Pool Blocks */
          <div className="bg-[#111722] border border-amber-900/40 rounded-xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-[#090d14] border-b border-[#243044] text-[11px] text-slate-400 uppercase tracking-wider">
                    <th className="py-3 px-4">Time</th>
                    <th className="py-3 px-4">Ticker</th>
                    <th className="py-3 px-4">Type</th>
                    <th className="py-3 px-4">Strike / Exp</th>
                    <th className="py-3 px-4 text-right">Spot Price</th>
                    <th className="py-3 px-4 text-right">Premium ($)</th>
                    <th className="py-3 px-4 text-right">Vol / OI</th>
                    <th className="py-3 px-4">Conviction Tier</th>
                    <th className="py-3 px-4">Sentiment</th>
                    <th className="py-3 px-4 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e293b]">
                  {loading ? (
                    <tr>
                      <td colSpan={10} className="py-8 text-center text-slate-500">
                        Streaming institutional options flow and dark pool tape...
                      </td>
                    </tr>
                  ) : optionsFlow.length === 0 ? (
                    <tr>
                      <td colSpan={10} className="py-8 text-center text-slate-500">
                        No options flow records found matching &quot;{filterQuery}&quot;.
                      </td>
                    </tr>
                  ) : (
                    optionsFlow.map((f, idx) => (
                      <tr
                        key={idx}
                        onClick={() => setSelectedOptions(f)}
                        className="hover:bg-[#162030] cursor-pointer transition-colors group"
                      >
                        <td className="py-3 px-4 text-slate-400 tabular-nums">{f.time}</td>
                        <td className="py-3 px-4 font-bold text-white group-hover:text-cyan-400 transition-colors">
                          <span className="bg-[#070a10] px-2 py-0.5 rounded border border-[#243044]">
                            {f.ticker}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              f.type.includes("CALL")
                                ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                                : f.type.includes("PUT")
                                ? "bg-rose-950 text-rose-400 border border-rose-800"
                                : "bg-purple-950 text-purple-400 border border-purple-800"
                            }`}
                          >
                            {f.type}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <div className="font-semibold text-slate-200">{f.strike}</div>
                          <div className="text-[10px] text-slate-500">{f.expiration}</div>
                        </td>
                        <td className="py-3 px-4 text-right text-slate-300 tabular-nums">${f.spot_price.toFixed(2)}</td>
                        <td className="py-3 px-4 text-right font-bold text-amber-400 tabular-nums">{f.premium}</td>
                        <td className="py-3 px-4 text-right font-bold text-cyan-400 tabular-nums">{f.volume_oi_ratio}x</td>
                        <td className="py-3 px-4">
                          <span className="bg-amber-950/70 text-amber-300 border border-amber-800/80 px-2 py-0.5 rounded text-[10px] font-bold">
                            {f.conviction_tier || "⚡ Unusual Sweep"}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <span className="text-[11px] text-emerald-400 font-semibold">{f.sentiment}</span>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedOptions(f);
                            }}
                            className="bg-cyan-950 hover:bg-cyan-900 text-cyan-400 border border-cyan-800 px-2.5 py-1 rounded text-[10px] font-bold transition-transform active:scale-95"
                          >
                            Inspect 🔍
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        ) : activeTab === "SEC_FORM_4" ? (
          /* TAB 2: SEC Form 4 Corporate Insider Transactions */
          <div className="bg-[#111722] border border-cyan-900/40 rounded-xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-[#090d14] border-b border-[#243044] text-[11px] text-slate-400 uppercase tracking-wider">
                    <th className="py-3 px-4">Corporate Insider & Role</th>
                    <th className="py-3 px-4">Ticker</th>
                    <th className="py-3 px-4">Company</th>
                    <th className="py-3 px-4">Type</th>
                    <th className="py-3 px-4 text-right">Shares Traded</th>
                    <th className="py-3 px-4 text-right">Total Value ($)</th>
                    <th className="py-3 px-4">Filing Date</th>
                    <th className="py-3 px-4">Ownership</th>
                    <th className="py-3 px-4">Conviction Tier</th>
                    <th className="py-3 px-4 text-center">SEC EDGAR</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e293b]">
                  {secInsiders.length === 0 ? (
                    <tr>
                      <td colSpan={10} className="py-8 text-center text-slate-500">
                        No SEC Form 4 insider transactions found matching filter.
                      </td>
                    </tr>
                  ) : (
                    secInsiders.map((s, idx) => (
                      <tr key={idx} className="hover:bg-[#162030] transition-colors group">
                        <td className="py-3 px-4">
                          <div className="font-bold text-slate-100 group-hover:text-cyan-400 transition-colors">
                            {s.insider_name}
                          </div>
                          <div className="text-[10px] text-slate-400">{s.role}</div>
                        </td>
                        <td className="py-3 px-4 font-bold text-white">
                          <span className="bg-[#070a10] px-2 py-0.5 rounded border border-[#243044]">
                            {s.ticker}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-slate-300">{s.company_name}</td>
                        <td className="py-3 px-4">
                          <span className="bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded text-[10px] font-semibold">
                            {s.transaction_type}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right text-slate-300 tabular-nums">
                          {s.shares_traded.toLocaleString()}
                        </td>
                        <td className="py-3 px-4 text-right font-bold text-emerald-400 tabular-nums">
                          {s.total_value}
                        </td>
                        <td className="py-3 px-4 text-slate-400 tabular-nums">{s.filing_date}</td>
                        <td className="py-3 px-4 text-cyan-400 font-semibold">{s.direct_ownership_pct}</td>
                        <td className="py-3 px-4">
                          <span className="bg-cyan-950 text-cyan-300 border border-cyan-800 px-2 py-0.5 rounded text-[10px] font-bold">
                            {s.conviction_tier || "👑 C-Suite Buy"}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <Link
                            href={`/?symbol=${s.ticker}`}
                            className="bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 px-2.5 py-1 rounded text-[10px] font-bold"
                          >
                            Analyze →
                          </Link>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          /* TAB 3: US Congressional STOCK Act Portfolio Disclosures */
          <div className="bg-[#111722] border border-purple-900/40 rounded-xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-[#090d14] border-b border-[#243044] text-[11px] text-slate-400 uppercase tracking-wider">
                    <th className="py-3 px-4">Politician</th>
                    <th className="py-3 px-4">Chamber</th>
                    <th className="py-3 px-4">Ticker</th>
                    <th className="py-3 px-4">Asset Name & Sector</th>
                    <th className="py-3 px-4">Type</th>
                    <th className="py-3 px-4 text-right">Amount ($)</th>
                    <th className="py-3 px-4">Filing Date</th>
                    <th className="py-3 px-4 text-right">Lag</th>
                    <th className="py-3 px-4">Conviction Badge</th>
                    <th className="py-3 px-4 text-right">Return Since</th>
                    <th className="py-3 px-4 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e293b]">
                  {loading ? (
                    <tr>
                      <td colSpan={11} className="py-8 text-center text-slate-500">
                        Synchronizing US House & Senate STOCK Act disclosures...
                      </td>
                    </tr>
                  ) : congressTrades.length === 0 ? (
                    <tr>
                      <td colSpan={11} className="py-8 text-center text-slate-500">
                        No congressional trades found matching filter criteria.
                      </td>
                    </tr>
                  ) : (
                    congressTrades.map((t, idx) => (
                      <tr
                        key={idx}
                        onClick={() => setSelectedCongress(t)}
                        className="hover:bg-[#162030] cursor-pointer transition-colors group"
                      >
                        <td className="py-3 px-4 font-bold text-slate-100 group-hover:text-purple-300 transition-colors">
                          {t.politician}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              t.chamber === "Senate"
                                ? "bg-indigo-950 text-indigo-300 border border-indigo-800"
                                : "bg-purple-950 text-purple-300 border border-purple-800"
                            }`}
                          >
                            {t.chamber}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-bold text-white">
                          <span className="bg-[#070a10] px-2 py-0.5 rounded border border-[#243044]">
                            {t.ticker}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-slate-300 max-w-[180px]">
                          <div className="font-semibold truncate">{t.asset_name}</div>
                          {t.sector && <div className="text-[10px] text-cyan-400 truncate">{t.sector}</div>}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                              t.transaction_type.includes("Purchase")
                                ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                                : "bg-rose-950 text-rose-400 border border-rose-800"
                            }`}
                          >
                            {t.transaction_type}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right font-bold text-emerald-400 tabular-nums">{t.amount_range}</td>
                        <td className="py-3 px-4 text-slate-400 tabular-nums">{t.filing_date}</td>
                        <td className="py-3 px-4 text-right text-slate-300 tabular-nums">{t.days_to_filing}d</td>
                        <td className="py-3 px-4">
                          <span className="bg-purple-950 text-purple-300 border border-purple-800 px-2 py-0.5 rounded text-[10px] font-bold">
                            {t.conviction_tier || "🏛️ STOCK Act"}
                          </span>
                        </td>
                        <td
                          className={`py-3 px-4 text-right font-bold tabular-nums ${
                            t.performance_since_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          {t.performance_since_pct >= 0 ? `+${t.performance_since_pct}%` : `${t.performance_since_pct}%`}
                        </td>
                        <td className="py-3 px-4 text-center">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedCongress(t);
                            }}
                            className="bg-purple-950 hover:bg-purple-900 text-purple-300 border border-purple-800 px-2.5 py-1 rounded text-[10px] font-bold transition-transform active:scale-95"
                          >
                            Inspect 🔍
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {/* Forensic Interactive Detail Drawer Modal */}
      <SmartMoneyDetailModal
        congressItem={selectedCongress}
        optionsItem={selectedOptions}
        onClose={() => {
          setSelectedCongress(null);
          setSelectedOptions(null);
        }}
      />
    </div>
  );
}

export default function SmartMoneyPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#070a10] text-slate-100 flex items-center justify-center font-mono">Loading Smart Money Hub...</div>}>
      <SmartMoneyContent />
    </Suspense>
  );
}
