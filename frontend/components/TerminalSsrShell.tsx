import Link from "next/link";
import { CATALOG_BASELINE_PRICES } from "../lib/masterCatalog";

export default function TerminalSsrShell() {
  return (
    <div className="min-h-screen bg-[#070a10] text-[#e2e8f0] flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Semantic Top Navigation Shell */}
      <header className="sticky top-0 z-40 bg-[#070a10]/95 backdrop-blur-md border-b border-[#1b2434] px-4 py-2.5">
        <div className="max-w-[1750px] mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2 group">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 via-cyan-600 to-blue-700 flex items-center justify-center shadow-lg shadow-cyan-500/20 font-black text-black font-mono text-sm">
                ARX
              </div>
              <div className="flex flex-col">
                <span className="font-black tracking-wider text-white text-base font-mono flex items-center gap-1.5">
                  ARX <span className="text-cyan-400 font-sans font-light text-xs tracking-normal">TERMINAL</span>
                </span>
                <span className="text-[9px] text-slate-400 tracking-widest font-mono uppercase">
                  QUANTITATIVE INTELLIGENCE & RISK
                </span>
              </div>
            </Link>

            {/* Desktop Navigation Links */}
            <nav aria-label="Main Navigation" className="hidden md:flex items-center gap-1 text-xs font-semibold">
              <Link href="/" className="px-3 py-1.5 rounded-lg bg-cyan-950/80 text-cyan-300 border border-cyan-800">
                Terminal
              </Link>
              <Link href="/screener" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Gems Screener
              </Link>
              <Link href="/smart-money" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Smart Money
              </Link>
              <Link href="/compare" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Compare
              </Link>
              <Link href="/portfolio" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Portfolio
              </Link>
              <Link href="/guide" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Field Manual
              </Link>
            </nav>
          </div>

          {/* Status Badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/80 text-[11px] font-mono text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>Live Market Engine</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Terminal Grid Shell */}
      <main className="flex-1 max-w-[1750px] w-full mx-auto p-2.5 sm:p-5 grid grid-cols-1 lg:grid-cols-4 gap-3 sm:gap-5">
        {/* Main Terminal Workspace Column */}
        <section aria-label="Market Workspace and Quantitative Analytics" className="lg:col-span-3 space-y-4 sm:space-y-5 order-1 lg:order-2 min-w-0">
          
          {/* Top 3 High-Confluence Plays Ribbon */}
          <div className="bg-[#0b0f19] border border-cyan-900/60 rounded-xl p-3.5 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
                  🎯 Top 3 High-Confluence Plays of the Week
                </span>
                <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                  LONG-TERM SIEVE
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs font-mono">
                <span className="text-slate-300">#1 NVO <strong className="text-emerald-400">${CATALOG_BASELINE_PRICES["NVO"]?.toFixed(2) || "45.12"}</strong></span>
                <span className="text-slate-300">#2 CPRX <strong className="text-emerald-400">${CATALOG_BASELINE_PRICES["CPRX"]?.toFixed(2) || "22.85"}</strong></span>
                <span className="text-slate-300">#3 ANET <strong className="text-emerald-400">${CATALOG_BASELINE_PRICES["ANET"]?.toFixed(2) || "320.00"}</strong></span>
              </div>
            </div>
          </div>

          {/* Active Asset Header Strip */}
          <div className="bg-[#0e1420] border border-[#1b2434] rounded-xl p-4 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <h1 className="text-2xl sm:text-3xl font-black text-white font-mono tracking-tight">
                  AAPL <span className="text-xl sm:text-2xl font-bold text-slate-200">${CATALOG_BASELINE_PRICES["AAPL"]?.toFixed(2) || "224.23"}</span>
                </h1>
                <span className="px-2 py-0.5 rounded text-xs font-bold font-mono bg-emerald-950 text-emerald-400 border border-emerald-800">
                  +0.45% 1Y
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-[#111726] text-cyan-400 border border-cyan-900/60">
                  20 EMA Active
                </span>
              </div>
            </div>

            {/* Catalyst Badge */}
            <div className="mt-3 p-2.5 rounded-lg bg-[#080c14] border border-[#1a2333] text-xs text-slate-300 flex items-center gap-2">
              <span>🔥</span>
              <span className="font-semibold text-slate-200">
                Secular AI Ecosystem Integration & Services Margin Expansion for Apple Inc. (Consumer Silicon Refresh Cycle)
              </span>
            </div>
          </div>

          {/* NO-BS Executive Summary Card */}
          <div className="bg-[#0b0f19] border border-[#1b2434] rounded-xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                <h2 className="text-sm font-black tracking-wider text-slate-200 uppercase font-mono">
                  NO-BS SUMMARY • AAPL SOLID ACCUMULATION SETUP
                </h2>
              </div>
              <div className="text-right font-mono">
                <span className="text-xs text-slate-400 block">SETUP SCORE</span>
                <span className="text-lg font-black text-cyan-400">75 / 100</span>
              </div>
            </div>

            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
              Disciplined trade structure with supporting macro and fundamental pillars. Favorable risk floor with asymmetric reward-to-risk above 2.0:1.
            </p>

            {/* 4 Quantitative Pillars */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-2">
              <div className="bg-[#070a12] p-3 rounded-lg border border-[#162032]">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Chart Structure</span>
                <p className="text-xs text-slate-300">Stage 2 Accumulation Breakout holding above 20-day EMA.</p>
              </div>
              <div className="bg-[#070a12] p-3 rounded-lg border border-[#162032]">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Company Health</span>
                <p className="text-xs text-slate-300">Piotroski F-Score 8/9 with robust free cash flow yield.</p>
              </div>
              <div className="bg-[#070a12] p-3 rounded-lg border border-[#162032]">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Smart Money Flow</span>
                <p className="text-xs text-slate-300">Institutional stealth accumulation with zero statutory late-filing warnings.</p>
              </div>
              <div className="bg-[#070a12] p-3 rounded-lg border border-[#162032]">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Market Tailwinds</span>
                <p className="text-xs text-slate-300">FRED 10Y-2Y yield curve positive with tight high-yield credit spreads.</p>
              </div>
            </div>
          </div>

          {/* Optimal Execution Ladder Preview */}
          <div className="bg-[#0e1420] border border-[#1b2434] rounded-xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
              <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <span>🎯</span>
                <span>AAPL Safe Buy & Sell Plan (Optimal Execution Ladder)</span>
              </h2>
              <span className="text-xs font-mono font-bold px-2.5 py-1 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                PROFIT : RISK 2.15 : 1.0
              </span>
            </div>

            <div className="space-y-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-800/60 flex justify-between items-center text-emerald-300">
                <span>🟢 PROFIT GOAL 2 (Extended Gains)</span>
                <span className="font-bold">$255.00 (+13.7%)</span>
              </div>
              <div className="p-2.5 rounded-lg bg-emerald-950/20 border border-emerald-800/40 flex justify-between items-center text-emerald-400">
                <span>🟢 PROFIT GOAL 1 (Sell Half Here / Risk-Free Runner)</span>
                <span className="font-bold">$242.00 (+7.9%)</span>
              </div>
              <div className="p-2.5 rounded-lg bg-cyan-950/40 border border-cyan-800/80 flex justify-between items-center text-cyan-300 font-bold">
                <span>⚡ CURRENT MARKET PRICE</span>
                <span>$224.23</span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#0a101d] border border-cyan-900/50 flex justify-between items-center text-slate-300">
                <span>🔵 BEST BUYING PRICE RANGE (Accumulation Area)</span>
                <span className="font-semibold">$218.50 – $225.00</span>
              </div>
              <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-800/60 flex justify-between items-center text-rose-300">
                <span>🔴 SAFETY EXIT (Cut Loss Floor)</span>
                <span className="font-bold">$212.00 (-5.4%)</span>
              </div>
            </div>
          </div>

          {/* Semantic SEO Methodology Block for Search & AI Crawlers */}
          <article className="p-5 rounded-xl bg-[#090d16] border border-[#162030] text-xs text-slate-400 space-y-3">
            <h3 className="text-sm font-bold text-slate-200">
              ARX Terminal Quantitative Architecture & Decision Engine
            </h3>
            <p className="leading-relaxed">
              ARX Terminal is an institutional equity intelligence platform engineered for active swing traders, hedge fund analysts, and quantitative capital allocators. The platform unifies <strong>Mark Minervini Volatility Contraction Pattern (VCP)</strong> breakout execution, <strong>Peter Lynch Growth At A Reasonable Price (GARP)</strong> valuation screeners, <strong>Joel Greenblatt Magic Formula</strong> scoring, <strong>Cornish-Fisher Value-at-Risk (VaR)</strong> fat-tail modeling, and <strong>Federal Reserve (FRED)</strong> macro regime detection.
            </p>
            <div className="flex flex-wrap gap-2 pt-1 font-mono text-[11px] text-cyan-400">
              <Link href="/screener" className="hover:underline">Explore 60+ Screener Gems →</Link>
              <span>•</span>
              <Link href="/smart-money" className="hover:underline">Inspect STOCK Act Disclosures →</Link>
              <span>•</span>
              <Link href="/compare" className="hover:underline">Head-to-Head Asset Comparisons →</Link>
              <span>•</span>
              <Link href="/guide" className="hover:underline">Read Quantitative Field Manual →</Link>
            </div>
          </article>
        </section>

        {/* Watchlist Sidebar Column Shell */}
        <aside aria-label="Watchlist and Real-Time Feeds" className="lg:col-span-1 h-full order-2 lg:order-1 min-w-0">
          <div className="bg-[#0b0f19] border border-[#1b2434] rounded-xl p-4 shadow-xl space-y-3">
            <div className="flex items-center justify-between border-b border-[#1b2434] pb-2.5">
              <span className="text-xs font-black tracking-wider text-slate-300 uppercase font-mono">
                CORE WATCHLIST
              </span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800 px-1.5 py-0.5 rounded">
                8 ASSETS
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              {[
                { sym: "NVDA", name: "NVIDIA Corp.", price: CATALOG_BASELINE_PRICES["NVDA"] || 128.50, change: "+2.4%" },
                { sym: "AAPL", name: "Apple Inc.", price: CATALOG_BASELINE_PRICES["AAPL"] || 224.23, change: "+0.5%" },
                { sym: "MSFT", name: "Microsoft Corp.", price: CATALOG_BASELINE_PRICES["MSFT"] || 448.35, change: "-0.2%" },
                { sym: "TSLA", name: "Tesla, Inc.", price: CATALOG_BASELINE_PRICES["TSLA"] || 215.80, change: "+3.1%" },
                { sym: "PLTR", name: "Palantir Technologies", price: CATALOG_BASELINE_PRICES["PLTR"] || 32.40, change: "+1.8%" },
                { sym: "NVO", name: "Novo Nordisk A/S", price: CATALOG_BASELINE_PRICES["NVO"] || 45.12, change: "+0.9%" },
                { sym: "SMCI", name: "Super Micro Computer", price: CATALOG_BASELINE_PRICES["SMCI"] || 36.86, change: "+1.2%" },
                { sym: "IREN", name: "Iris Energy Limited", price: CATALOG_BASELINE_PRICES["IREN"] || 36.85, change: "+4.6%" },
              ].map((item) => (
                <div key={item.sym} className="p-2 rounded-lg bg-[#070a12] border border-[#141d2c] flex items-center justify-between hover:border-cyan-800 transition-colors">
                  <div>
                    <span className="font-bold text-white block">{item.sym}</span>
                    <span className="text-[10px] text-slate-500 truncate max-w-[110px] block">{item.name}</span>
                  </div>
                  <div className="text-right">
                    <span className="font-semibold text-slate-200 block">${item.price.toFixed(2)}</span>
                    <span className="text-[10px] text-emerald-400 font-bold">{item.change}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
