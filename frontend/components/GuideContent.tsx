"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

export default function GuideContent() {
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved === "PLAIN_ENGLISH" || saved === "PRO_QUANT") {
        setVernacularMode(saved);
      }
    } catch {}

    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail === "PLAIN_ENGLISH" || custom.detail === "PRO_QUANT") {
        setVernacularMode(custom.detail);
      }
    };

    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  return (
    <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 font-mono space-y-10 sm:space-y-14 pb-24 sm:pb-16">
      {/* Hero Header */}
      <header className="border-b border-[#243044] pb-6 sm:pb-8 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-[11px] font-bold">
              {isPlain ? "⚡ PLAIN ENGLISH TRADER FIELD MANUAL" : "INSTITUTIONAL QUANTITATIVE FIELD MANUAL"}
            </span>
            <span className="text-slate-500 text-xs">• Version 2.10.0 Specification</span>
          </div>

          <div className="flex items-center gap-1.5 text-xs font-mono bg-[#090d14] px-2.5 py-1 rounded-lg border border-[#1e293b]">
            <span className="text-slate-400">Active Mode:</span>
            <span className={isPlain ? "text-emerald-400 font-bold" : "text-cyan-400 font-bold"}>
              {isPlain ? "💬 Plain English" : "🏛️ Pro Quant"}
            </span>
          </div>
        </div>

        <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight">
          {isPlain 
            ? "The No-BS Trader Field Manual: How to Use This Terminal" 
            : "Quantitative Platform Blueprint & Execution Handbook"}
        </h1>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "A crystal-clear, zero-jargon handbook for everyday traders and long-term investors. Learn exactly where to buy, how to size your bets so you never blow up, how to copy politician trades legally, and how to verify setups before risking a single dollar."
            : "A comprehensive, mathematically rigorous reference manual for professional traders, quantitative analysts, and fundamental investors. Learn the formulas, invalidation rules, statutory insider metrics, legislative alignment algorithms, Decision Intelligence risk gates, and portfolio equations powering this terminal."}
        </p>

        {/* Quick Jump Navigation */}
        <div className="bg-[#090d14] p-3 rounded-xl border border-[#1b2434] flex flex-wrap gap-2 text-[11px] pt-3">
          <span className="text-slate-500 font-bold uppercase py-0.5">Quick Jump:</span>
          <a href="#chapter-1" className="text-cyan-400 hover:underline">
            {isPlain ? "Ch 1: The 4 Workspaces" : "Ch 1: Workspaces & Mobile UX"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-chart" className="text-amber-400 hover:underline font-bold">
            {isPlain ? "Ch 2: Live Chart Guide" : "Ch 2: Dual Chart Engine"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-2" className="text-cyan-400 hover:underline">
            {isPlain ? "Ch 3: Buy Zones & Sizing" : "Ch 3: Execution & Screener Math"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-3" className="text-amber-400 hover:underline font-bold">
            {isPlain ? "Ch 4: Politician Trades" : "Ch 4: STOCK Act & Alignment"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-4" className="text-emerald-400 hover:underline">
            {isPlain ? "Ch 5: Quality & BS Detector" : "Ch 5: 5-Factor Radar"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-5" className="text-purple-400 hover:underline">
            {isPlain ? "Ch 6: Crash Risk & Safety" : "Ch 6: Risk & Self-Healing VaR"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-6" className="text-cyan-400 hover:underline font-bold">
            {isPlain ? "Ch 7: Fed Rates & Insiders" : "Ch 7: Multi-Source Synthesis"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-8" className="text-emerald-400 hover:underline font-bold">
            {isPlain ? "Ch 8: Jargon Cheat Sheet" : "Ch 8: Plain-English Buster"}
          </a>
          <span className="text-slate-600">•</span>
          <a href="#chapter-9" className="text-amber-400 hover:underline font-bold">
            {isPlain ? "Ch 9: The 4 Trade Co-Pilots" : "Ch 9: Decision Intelligence Suite"}
          </a>
        </div>
      </header>

      {/* CHAPTER 1: WORKSPACES & DUAL USER JOURNEYS */}
      <section id="chapter-1" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">🗂️</span>
          <h2 className="text-lg sm:text-xl font-bold text-cyan-400 tracking-tight">
            {isPlain 
              ? "Chapter 1: The 4 Power Workspaces & Mobile Navigation" 
              : "Chapter 1: The 4 In-Terminal Workspaces & Viewport-Adaptive Architecture"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "Old financial websites clutter your screen with 50 complicated widgets that make your head spin. We organized everything into 4 clean workspaces located right below the live chart:"
            : "Traditional financial terminals force analysts to scroll through dozens of stacked widgets, causing severe cognitive overload and losing chart context. Our terminal splits analysis into 4 modular domains anchored directly beneath the live price chart, with fully responsive viewport layout optimization:"}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
            <strong className="text-cyan-300 text-sm flex items-center gap-1.5 font-mono">
              <span>🎯</span>
              <span>{isPlain ? "Where to Buy & Where to Exit (Execution)" : "Execution & Levels (Default)"}</span>
            </strong>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain
                ? "Answers: 'Where do I enter, where is my safety stop, and when do I take profit?' Houses exact buy zones, the position sizer, and volatility bands."
                : "Answers: 'Where do I enter, where is my stop, and when do I take profit?' Houses the Minervini VCP accumulation ladder, Intraday Position Sizer, and ATR14 volatility bands."}
            </p>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
            <strong className="text-amber-300 text-sm flex items-center gap-1.5 font-mono">
              <span>🏛️</span>
              <span>{isPlain ? "What Big Money is Doing (Insiders & Congress)" : "Smart Money & Insiders"}</span>
            </strong>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain
                ? "Answers: 'What are US politicians, CEOs, and big institutional funds buying?' Shows legal House/Senate STOCK Act filings and Form 4 C-Suite insider purchases."
                : "Answers: 'What are politicians, corporate executives, and option market makers doing?' Displays US House/Senate STOCK Act filings, Legislative Alignment Index (0-100), dark pool ATS volumes, and options sweeps."}
            </p>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
            <strong className="text-emerald-300 text-sm flex items-center gap-1.5 font-mono">
              <span>📊</span>
              <span>{isPlain ? "How Healthy the Company Is (Financial DNA)" : "Factors & Macro Intelligence"}</span>
            </strong>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain
                ? "Answers: 'Is this company printing real cash or drowning in hidden debt?' Displays the 5-point quality radar, clean-accounting scores, and interest rate trends."
                : "Answers: 'How healthy is the company and what is the macro regime?' Displays 5-Factor profile radar, 9-point Piotroski F-Scores, and FRED 10Y-2Y yield curve spreads."}
            </p>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
            <strong className="text-purple-300 text-sm flex items-center gap-1.5 font-mono">
              <span>🛡️</span>
              <span>{isPlain ? "Crash Risk & Contagion (Worst-Case Test)" : "Risk & Contagion Networks"}</span>
            </strong>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain
                ? "Answers: 'If the market crashes or a major supplier collapses, how much could I lose?' Tests worst-case single-day losses and supply chain ripple risks."
                : "Answers: 'If a peer or supplier collapses, how does the shock cascade?' Displays directed supply-chain topologies, Cornish-Fisher M-VaR, and self-healing hit rate calibrations."}
            </p>
          </div>
        </div>

        {/* Mobile-Adaptive UX Callout */}
        <div className="bg-[#090d14] p-4 rounded-xl border border-[#1e293b] space-y-2 text-xs">
          <h3 className="font-bold text-slate-100 uppercase tracking-wider flex items-center gap-2">
            <span>📱 {isPlain ? "Mobile Experience & Fast Ticker Switching" : "Mobile-First Layout Reordering & Viewport Physics"}</span>
          </h3>
          <p className="text-slate-300 font-sans leading-relaxed">
            {isPlain
              ? "On mobile phones, the terminal places the chart, current price, and buy/sell levels right at the top so you don't have to scroll. The watchlist collapses into clean swipeable pills that automatically close once you pick a stock."
              : "On mobile viewports (<1024px), the terminal dynamically inverts DOM order (order-1 lg:order-2) so the active ticker hero, price, timeframe controls, and candlestick chart appear at the focal top. The Watchlist sidebar collapses into an intuitive accordion with horizontally scrollable filter pills (overflow-x-auto no-scrollbar) and auto-collapses upon selecting a ticker."}
          </p>
        </div>

        {/* 4-Tier Asset Universe & Architecture Scope */}
        <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-cyan-300 uppercase tracking-wider flex items-center gap-2">
              <span>🌐 {isPlain ? "Asset Coverage: Pre-Built High-Conviction Stocks vs. Search Any Ticker" : "4-Tier Asset Architecture: Static Edge Pre-Rendering vs. Dynamic Execution"}</span>
            </h3>
            <span className="text-[10px] bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800 font-mono">
              ARCHITECTURE SCOPE
            </span>
          </div>
          <p className="text-slate-300 font-sans leading-relaxed">
            {isPlain
              ? "Why does the terminal highlight a focused group of stocks, and can you search your own favorites? Here is how our 4-tier engine works:"
              : "To maintain sub-10ms edge delivery while providing open-universe flexibility, the platform operates a 4-tier hierarchical asset pipeline:"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <div className="flex items-center justify-between">
                <strong className="text-white font-mono text-xs">{isPlain ? "1. Master Catalog (38 Core Assets)" : "1. Master Catalog (38 Assets)"}</strong>
                <span className="text-[9px] bg-[#162030] text-amber-300 px-1.5 py-0.2 rounded font-mono">HYBRID</span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                {isPlain
                  ? "Pre-audited high-conviction companies (NVDA, LLY, PLTR, CPRX, etc.) with verified balance sheets and live spot price hydration."
                  : "Single Source of Truth for verified fundamental ratios (ROIC, Gross Margin, PEG, Piotroski) with real-time price store hydration."}
              </p>
            </div>
            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <div className="flex items-center justify-between">
                <strong className="text-white font-mono text-xs">{isPlain ? "2. Multi-Factor Screener (35 Assets)" : "2. Multi-Factor Screener (35 Assets)"}</strong>
                <span className="text-[9px] bg-emerald-950 text-emerald-300 px-1.5 py-0.2 rounded font-mono">100% DYNAMIC</span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                {isPlain
                  ? "Dynamically scans and calculates buy zones, stop losses, and multi-bagger criteria on every refresh."
                  : "Live execution engine calculating ATR corridors, Peter Lynch GARP, Greenblatt Magic Formula, and Rule Breaker criteria on the fly."}
              </p>
            </div>
            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <div className="flex items-center justify-between">
                <strong className="text-white font-mono text-xs">{isPlain ? "3. Pre-Rendered Pages (94 Static Routes)" : "3. Pre-Rendered Pages (94 Static Routes)"}</strong>
                <span className="text-[9px] bg-[#162030] text-cyan-300 px-1.5 py-0.2 rounded font-mono">HYBRID (SSG)</span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                {isPlain
                  ? "Dedicated landing pages pre-compiled for instant <10ms loading speed, social previews, and Google search indexing."
                  : "Next.js static site generation (SSG) with Schema.org JSON-LD and OpenGraph metadata delivering zero-latency edge delivery."}
              </p>
            </div>
            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <div className="flex items-center justify-between">
                <strong className="text-white font-mono text-xs">{isPlain ? "4. Universal Omnisearch (Unlimited)" : "4. Universal Omnisearch (Unlimited)"}</strong>
                <span className="text-[9px] bg-purple-950 text-purple-300 px-1.5 py-0.2 rounded font-mono">100% DYNAMIC</span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                {isPlain
                  ? "Type ANY ticker (e.g. COIN, AMD, DIS, BTC) into the search bar (press '/') to pull live charts and calculate buy zones on demand."
                  : "Live on-demand query engine generating dynamic Minervini VCP levels, ATR ladders, and trade execution plans for any US equity or crypto pair."}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CHAPTER 2: INTERACTIVE DUAL-HORIZON CANDLESTICK & INDICATOR ENGINE */}
      <section id="chapter-chart" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">📈</span>
          <h2 className="text-lg sm:text-xl font-bold text-amber-400 tracking-tight">
            {isPlain 
              ? "Chapter 2: The Live Interactive Chart (Day Trader vs. Long-Term)" 
              : "Chapter 2: Interactive Dual-Horizon Candlestick & Indicator Engine"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "The platform includes a live institutional candlestick chart that automatically adjusts its indicators and timelines based on your trading style:"
            : "The platform embeds an institutional TradingView Lightweight Charts canvas that dynamically adapts its time scale, indicators, and mathematical percentage baselines according to your selected trading persona:"}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
          <div className="bg-[#111722] p-4 rounded-xl border border-amber-800/60 space-y-2.5">
            <div className="flex items-center justify-between">
              <strong className="text-amber-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                <span>⚡</span>
                <span>{isPlain ? "Day Trader Fast Scalp Mode" : "Day Trader Scalp Sessions"}</span>
              </strong>
              <span className="text-[10px] bg-amber-950 text-amber-300 px-1.5 py-0.5 rounded border border-amber-700">INTRADAY</span>
            </div>
            <ul className="text-xs text-slate-300 font-sans space-y-1.5 list-disc pl-4">
              <li><strong>Timeframes:</strong> <code>1m</code> (fast scalps), <code>5m</code> (session trends), <code>15m</code> (multi-hour momentum).</li>
              <li><strong>Key Line:</strong> <strong>VWAP (Volume-Weighted Average Price)</strong> in amber line. The benchmark price paid by institutional algorithms today. Buying below VWAP means you got a discount; selling above it is strength.</li>
            </ul>
          </div>

          <div className="bg-[#111722] p-4 rounded-xl border border-cyan-800/60 space-y-2.5">
            <div className="flex items-center justify-between">
              <strong className="text-cyan-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                <span>🏛️</span>
                <span>{isPlain ? "Long-Term Wealth Building Mode" : "Long-Term Macro Horizons"}</span>
              </strong>
              <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-700">MACRO</span>
            </div>
            <ul className="text-xs text-slate-300 font-sans space-y-1.5 list-disc pl-4">
              <li><strong>Horizons:</strong> <code>1M</code>, <code>6M</code>, <code>1Y</code>, <code>3Y</code>, <code>5Y</code> multi-year compounding cycles.</li>
              <li><strong>Key Line:</strong> <strong>20-Day Exponential Moving Average (20 EMA)</strong> in sky blue. Acts as a dynamic safety trampoline where winning stocks find support during pullbacks.</li>
            </ul>
          </div>
        </div>

        {/* Metric Disambiguation Callout */}
        <div className="bg-[#090d14] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
          <h3 className="text-xs sm:text-sm font-bold text-slate-100 uppercase tracking-wider flex items-center gap-2">
            <span>🎯 {isPlain ? "Never Mix Up Today's Change vs. Your Total Multi-Month Return" : "Metric Disambiguation: Watchlist Row vs. Chart Header Return"}</span>
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed font-sans">
            {isPlain
              ? "To prevent confusing daily noise with big long-term trends, the terminal clearly labels two separate return numbers:"
              : "To eliminate confusion between short-term noise and long-term trends, the terminal explicitly separates two distinct percentage return calculations:"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1.5">
              <strong className="text-slate-200 block font-mono">{isPlain ? "Watchlist Badge (Today's 24-Hour Change)" : "Watchlist Sidebar Badge (24H Daily Return)"}</strong>
              <p className="text-[11px] text-slate-400 font-sans leading-snug">
                {isPlain
                  ? "Measures strictly how much the stock gained or lost today relative to yesterday's closing bell (e.g. +2.65% 24H)."
                  : "Measures purely today's daily trading change relative to yesterday's closing bell (e.g. +2.65% 24H)."}
              </p>
            </div>

            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1.5">
              <strong className="text-slate-200 block font-mono">{isPlain ? "Chart Header Badge (Total Period Return)" : "Chart Header Badge (Active Horizon Return)"}</strong>
              <p className="text-[11px] text-slate-400 font-sans leading-snug">
                {isPlain
                  ? "Measures the total profit or loss across the entire timeline selected on your chart (e.g. +28.40% 1Y or +0.80% 5M)."
                  : "Measures total cumulative trajectory across the active dataset with explicit horizon pill tag (e.g. +28.40% 1Y or +0.80% 5M)."}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CHAPTER 3: ALGORITHMIC EXECUTION, ATR STATES & SCREENER MATH */}
      <section id="chapter-2" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">🧮</span>
          <h2 className="text-lg sm:text-xl font-bold text-cyan-400 tracking-tight">
            {isPlain 
              ? "Chapter 3: Where to Buy, Sizing Your Trades & The 4 Zones" 
              : "Chapter 3: Algorithmic Execution Formulas, ATR States & Screener Math"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "Never guess where to enter or panic when price dips. The terminal calculates exact entry corridors, stop losses, and profit targets using historical volatility:"
            : "Our execution engine replaces emotional discretion with concrete mathematical boundaries based on Mark Minervini's Volatility Contraction Pattern (VCP) and Linda Raschke's 20 EMA pullback setup."}
        </p>

        {/* Sizing Math */}
        <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
          <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>📐 {isPlain ? "1. The Position Sizing Safety Formula (Never Risk >1%)" : "1. Exact Position Sizing Equation"}</span>
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed font-sans">
            {isPlain
              ? "The number one reason retail traders lose money is risking too much on a single bad trade. We calculate the exact number of shares to buy so that if you hit your stop loss, you only lose 1% of your total account:"
              : "Never risk more than your pre-defined capital threshold (1%–2% of account equity per trade). The terminal calculates exact share volume using:"}
          </p>
          <div className="bg-[#070a10] p-3 rounded-lg border border-[#162030] text-cyan-300 font-bold text-xs font-mono">
            Shares to Buy = (Total Account Capital * Risk % Budget) / (Entry Price - Stop Loss Price)
          </div>
          <p className="text-[11px] text-slate-400 leading-snug font-sans">
            <strong>Example:</strong> With a \$50,000 account risking 1% (\$500 max loss) buying NVDA at \$213.05 with a Stop Loss at \$201.35 (\$11.70 risk per share), the calculator tells you to buy exactly <strong>42 shares</strong>.
          </p>
        </div>

        {/* 4 Mathematical ATR Execution States */}
        <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
          <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>⚡ {isPlain ? "2. The 4 Clear Trade Zones (When to Buy & When to Walk Away)" : "2. The 4 Mathematical ATR Execution States"}</span>
          </h3>
          <p className="text-xs text-slate-300 font-sans leading-relaxed">
            {isPlain
              ? "Every stock in the scanner is automatically sorted into one of four distinct color-coded zones:"
              : "Every candidate in the terminal and screener is dynamically classified into one of four disjoint execution states based on price relative to ATR bands:"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-[#111722] p-3 rounded-lg border border-emerald-900/60 space-y-1">
              <strong className="text-emerald-400 font-bold block">🟢 {isPlain ? "IN BUY ZONE (Great Price to Accumulate)" : "IN_BUY_ZONE (Optimal Accumulation)"}</strong>
              <p className="text-slate-300 text-[11px] font-sans">
                {isPlain 
                  ? "Price is sitting comfortably on support with low risk. Best time to enter." 
                  : "Condition: Optimal Entry Min <= Spot <= Optimal Entry Max. Resting inside institutional volume."}
              </p>
            </div>
            <div className="bg-[#111722] p-3 rounded-lg border border-cyan-900/60 space-y-1">
              <strong className="text-cyan-400 font-bold block">🔵 {isPlain ? "ON THE MOVE (Heading to Profit Target)" : "APPROACHING_TARGET (Momentum Expansion)"}</strong>
              <p className="text-slate-300 text-[11px] font-sans">
                {isPlain 
                  ? "Trade is working nicely and climbing toward the first profit scale-out target." 
                  : "Condition: Optimal Entry Max < Spot < Target 1. Active momentum trend in progress."}
              </p>
            </div>
            <div className="bg-[#111722] p-3 rounded-lg border border-amber-900/60 space-y-1">
              <strong className="text-amber-400 font-bold block">🟡 {isPlain ? "DON'T CHASE (Wait for the Pullback)" : "WAITING_PULLBACK (Overextended / High Risk)"}</strong>
              <p className="text-slate-300 text-[11px] font-sans">
                {isPlain 
                  ? "Price has spiked too high above support. Buying here is risky; wait for a healthy dip." 
                  : "Condition: Spot > Target 1. Price is extended past ATR bands; elevated mean-reversion risk."}
              </p>
            </div>
            <div className="bg-[#111722] p-3 rounded-lg border border-rose-900/60 space-y-1">
              <strong className="text-rose-400 font-bold block">🔴 {isPlain ? "GET OUT (Trade Invalidated / Cut Loss)" : "STOPPED_OUT (Invalidation Exit)"}</strong>
              <p className="text-slate-300 text-[11px] font-sans">
                {isPlain 
                  ? "Support has broken. Cut the trade immediately to protect your capital." 
                  : "Condition: Spot < Stop Loss. Technical structure broken; strict capital preservation exit."}
              </p>
            </div>
          </div>
        </div>

        {/* Screener Quantitative Filter Engine */}
        <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
          <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>🔍 {isPlain ? "3. The 6 Screener Presets Explained" : "3. Screener Numerical Filter Thresholds (Calibrated)"}</span>
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-xs">
            <div className="bg-[#090d14] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <span className="text-[10px] text-cyan-400 font-bold block uppercase">{isPlain ? "⚡ Best Profit vs Risk" : "⚡ Asymmetric Actionability"}</span>
              <strong className="text-white font-mono text-xs">{isPlain ? "High Upside & In Buy Zone" : "R:R >= 2.0:1 & In Buy Zone"}</strong>
              <p className="text-[10px] text-slate-400 font-sans">{isPlain ? "Only shows stocks currently in the buy zone with >=2x upside." : "Restricted to setups currently within <=+2% of accumulation floor."}</p>
            </div>
            <div className="bg-[#090d14] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <span className="text-[10px] text-emerald-400 font-bold block uppercase">{isPlain ? "🧪 Cash Machine (Greenblatt)" : "🧪 Joel Greenblatt Quality"}</span>
              <strong className="text-white font-mono text-xs">ROIC &ge; 28.0%</strong>
              <p className="text-[10px] text-slate-400 font-sans">{isPlain ? "Generates massive cash returns on every dollar invested." : "Top-decile return on invested capital with high operating margins."}</p>
            </div>
            <div className="bg-[#090d14] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <span className="text-[10px] text-purple-400 font-bold block uppercase">{isPlain ? "📈 Bargain Growth (Peter Lynch)" : "📈 Peter Lynch GARP"}</span>
              <strong className="text-white font-mono text-xs">0 &lt; PEG &le; 1.05</strong>
              <p className="text-[10px] text-slate-400 font-sans">{isPlain ? "Fast revenue growth without paying an absurd valuation." : "High earnings expansion at reasonable valuation multiples."}</p>
            </div>
            <div className="bg-[#090d14] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <span className="text-[10px] text-amber-400 font-bold block uppercase">{isPlain ? "🔥 Market Leaders" : "🔥 Category Disruptors"}</span>
              <strong className="text-white font-mono text-xs">Gross Margin &ge; 65.0%</strong>
              <p className="text-[10px] text-slate-400 font-sans">{isPlain ? "Companies with huge profit margins that competitors can't touch." : "David Gardner Rule Breakers with durable pricing power moats."}</p>
            </div>
            <div className="bg-[#090d14] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <span className="text-[10px] text-cyan-400 font-bold block uppercase">{isPlain ? "🚀 Heavy Volume Surges" : "🚀 Momentum Flow"}</span>
              <strong className="text-white font-mono text-xs">RVOL &ge; 2.5x</strong>
              <p className="text-[10px] text-slate-400 font-sans">{isPlain ? "Trading volume is 2.5x higher than normal. Big money is active." : "Institutional volume surges exceeding 250% of 20-day baseline."}</p>
            </div>
            <div className="bg-[#090d14] p-3 rounded-lg border border-[#1b2434] space-y-1">
              <span className="text-[10px] text-rose-400 font-bold block uppercase">{isPlain ? "💥 Short Squeeze Alert" : "💥 Short Squeeze"}</span>
              <strong className="text-white font-mono text-xs">Short Float &ge; 6.0%</strong>
              <p className="text-[10px] text-slate-400 font-sans">{isPlain ? "Heavy short sellers trapped if the stock breaks out upward." : "Elevated borrowing rates with rapid upward squeeze pressure."}</p>
            </div>
          </div>
        </div>
      </section>

      {/* CHAPTER 4: CONGRESSIONAL STOCK ACT, LEGISLATIVE ALIGNMENT & STALENESS DECAY */}
      <section id="chapter-3" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">🏛️</span>
          <h2 className="text-lg sm:text-xl font-bold text-amber-400 tracking-tight">
            {isPlain 
              ? "Chapter 4: Tracking US Politicians & The 45-Day Delay Penalty" 
              : "Chapter 4: Congressional STOCK Act, Legislative Alignment & Staleness Decay"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "Under US Federal Law (the STOCK Act of 2012), members of Congress and the Senate must legally disclose all stock trades. We track these trades directly from official Capitol Hill disclosures:"
            : "Under Public Law 112-105 (Stop Trading on Congressional Knowledge Act of 2012), members of the US Congress and Senate are legally mandated to disclose securities transactions within 45 days. The terminal processes these disclosures through a quantitative intelligence pipeline:"}
        </p>

        {/* Legislative Alignment Score */}
        <div className="bg-[#111722] p-4 sm:p-5 rounded-xl border border-purple-800/60 space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <h3 className="text-xs sm:text-sm font-bold text-purple-300 uppercase tracking-wider flex items-center gap-2">
              <span>⚖️ {isPlain ? "1. The Politician Alignment Score (0 to 100)" : "1. Quantitative Legislative Alignment Index (0–100)"}</span>
            </h3>
            <span className="text-[10px] bg-purple-950 text-purple-300 px-2 py-0.5 rounded border border-purple-700 font-bold">ALGORITHM</span>
          </div>
          <p className="text-slate-300 font-sans leading-relaxed">
            {isPlain
              ? "Measures whether the politician sits on a committee that directly regulates or awards government contracts to the stock they bought:"
              : "Measures the empirical strength of regulatory and legislative tailwinds behind a politician's trade:"}
          </p>
          <ul className="space-y-2 text-slate-300 font-sans list-disc pl-5">
            <li>
              <strong>{isPlain ? "Committee Oversight Overlap (+16 to +32 pts):" : "Committee Jurisdiction Overlap (+16 to +32 pts):"}</strong> {isPlain ? "e.g. Armed Services members buying Defense Tech, or Energy & Commerce members buying semiconductor chips." : "Direct committee oversight matching asset sector."}
            </li>
            <li>
              <strong>{isPlain ? "Big Dollar Bet Size (+5 to +15 pts):" : "Dollar Sizing Bracket (+5 to +15 pts):"}</strong> {isPlain ? "Gives higher points to massive trades ($250k to $1M+) vs minor $1,000 purchases." : "Scales conviction according to transaction size ($50k to $1M+)."}
            </li>
            <li>
              <strong>{isPlain ? "Politician's Historical Win Rate (+4 to +10 pts):" : "3-Year Historical Track Record (+4 to +10 pts):"}</strong> {isPlain ? "Factors in the politician's multi-year track record of beating the S&P 500." : "Factors in audited multi-year politician win rates (>75%)."}
            </li>
          </ul>
        </div>

        {/* Staleness Decay and Late Filer Warnings */}
        <div className="bg-[#111722] p-4 sm:p-5 rounded-xl border border-rose-800/60 space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <h3 className="text-xs sm:text-sm font-bold text-rose-300 uppercase tracking-wider flex items-center gap-2">
              <span>⏱️ {isPlain ? "2. The Delay Penalty (Don't Buy Stale Old Trades!)" : "2. STOCK Act Filing Latency & Signal Time-Decay"}</span>
            </h3>
            <span className="text-[10px] bg-rose-950 text-rose-300 px-2 py-0.5 rounded border border-rose-700 font-bold">RISK PROTECTION</span>
          </div>
          <p className="text-slate-300 font-sans leading-relaxed">
            {isPlain
              ? "Politicians have up to 45 days to disclose their trades. If a politician bought 40 days ago, the stock has probably already moved! We penalize old disclosures so you don't chase stale trades:"
              : "If a politician disclosed a trade 60 days after execution, the price move is already priced in. To protect retail traders from buying stale news, the terminal applies an exponential time-decay penalty:"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
            <div className="bg-[#090d14] p-2.5 rounded border border-emerald-800/50">
              <span className="text-emerald-400 font-bold block">{isPlain ? "⚡ Brand New (<15 Days Lag)" : "⚡ Fresh (<15 Days Lag)"}</span>
              <p className="text-[11px] text-slate-400 font-sans">{isPlain ? "0 point penalty. Trade is fresh and actionable." : "0 pt penalty. 100% full signal conviction."}</p>
            </div>
            <div className="bg-[#090d14] p-2.5 rounded border border-cyan-800/50">
              <span className="text-cyan-400 font-bold block">{isPlain ? "⏳ Standard (16–30 Days Lag)" : "⏳ Standard (16–30 Days Lag)"}</span>
              <p className="text-[11px] text-slate-400 font-sans">{isPlain ? "-5 point penalty. Standard disclosure speed." : "-5 pt penalty. Normal statutory compliance."}</p>
            </div>
            <div className="bg-[#090d14] p-2.5 rounded border border-amber-800/50">
              <span className="text-amber-400 font-bold block">{isPlain ? "⚠️ Aging Trade (31–45 Days Lag)" : "⚠️ Aging Signal (31–45 Days Lag)"}</span>
              <p className="text-[11px] text-slate-400 font-sans">{isPlain ? "-16 point penalty. Price move may already be priced in." : "-16 pt penalty. Approaching statutory deadline."}</p>
            </div>
            <div className="bg-[#090d14] p-2.5 rounded border border-rose-800/50">
              <span className="text-rose-400 font-bold block">{isPlain ? "🛑 Late Filer Warning (>45 Days Lag)" : "🛑 Late Filer (>45 Days Lag)"}</span>
              <p className="text-[11px] text-slate-400 font-sans">{isPlain ? "-32 point penalty. High risk of buying the top of old news." : "-32 pt penalty. Explicit priced-in mean reversion warning."}</p>
            </div>
          </div>
        </div>
      </section>

      {/* CHAPTER 5: 5-FACTOR RADAR & PIOTROSKI F-SCORE */}
      <section id="chapter-4" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">📊</span>
          <h2 className="text-lg sm:text-xl font-bold text-emerald-400 tracking-tight">
            {isPlain ? "Chapter 5: The 5-Point Quality Score & Clean Accounting Check" : "Chapter 5: 5-Factor Fundamental DNA & Piotroski Score"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "We summarize hundreds of balance sheet data points into clear scores from 0 to 100:"
            : "The platform synthesizes thousands of fundamental balance sheet data points into a multi-dimensional quantitative profile:"}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
            <strong className="text-cyan-400 font-bold block">1. {isPlain ? "Revenue & Cash Growth (0-100)" : "Growth Score (0-100)"}</strong>
            <p className="text-slate-300 text-[11px] font-sans">{isPlain ? "Measures 3-year sales expansion and free cash flow generation." : "Calculated via 3-year revenue CAGR and forward EPS expansion."}</p>
          </div>
          <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
            <strong className="text-emerald-400 font-bold block">2. {isPlain ? "Business Moat & Profit Margins (0-100)" : "Quality Score (0-100)"}</strong>
            <p className="text-slate-300 text-[11px] font-sans">{isPlain ? "High Return on Capital (ROIC > 20%), fat profit margins, and low debt." : "Measures ROIC > 15%, gross profit margin moats, and low financial leverage."}</p>
          </div>
          <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
            <strong className="text-purple-400 font-bold block">3. {isPlain ? "Price & Valuation Discount (0-100)" : "Valuation Score (0-100)"}</strong>
            <p className="text-slate-300 text-[11px] font-sans">{isPlain ? "Compares stock price against expected growth (PEG ratio & fair cash value)." : "Derived from PEG ratio, EV/EBITDA multiple discounts, and DCF spreads."}</p>
          </div>
          <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
            <strong className="text-amber-400 font-bold block">4. {isPlain ? "Piotroski BS Detector (0–9)" : "Piotroski 9-Point F-Score"}</strong>
            <p className="text-slate-300 text-[11px] font-sans">{isPlain ? "Scores 8-9 mean squeaky clean accounting books. Scores <= 3 mean avoid!" : "Scores 8-9 indicate pristine balance sheet quality; scores <= 3 signal risk."}</p>
          </div>
        </div>
      </section>

      {/* CHAPTER 6: RISK & SELF-HEALING FORECAST AUDITOR */}
      <section id="chapter-5" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">🛡️</span>
          <h2 className="text-lg sm:text-xl font-bold text-purple-400 tracking-tight">
            {isPlain ? "Chapter 6: Managing Crash Risk & Worst-Case Downside" : "Chapter 6: Mathematical Invariants, Cornish-Fisher VaR & Self-Healing Engine"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "Most websites assume stock market drops follow a smooth bell curve. In real life, market crashes are violent. Our risk engine accounts for real-world crash spikes (fat tails):"
            : "Standard Gaussian Value-at-Risk assumes symmetric normal distributions, severely underestimating fat-tail crash risks in equity markets. Our quantitative engine applies a polynomial Cornish-Fisher expansion adjusted for non-normal Skewness and excess Kurtosis:"}
        </p>

        <div className="bg-[#090d14] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
          <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider">
            {isPlain ? "The Worst-Case Day Crash Formula (VaR 95% & 99%):" : "Cornish-Fisher Expansion Formula:"}
          </h3>
          <div className="bg-[#05070a] p-3 rounded-lg border border-[#162030] font-mono text-cyan-300 text-xs leading-relaxed overflow-x-auto">
            {isPlain
              ? "Max Estimated 1-Day Loss = Account Capital * (Volatility * Adjusted Shock Factor)"
              : "Z_cf = z_alpha + (z_alpha^2 - 1)*S / 6 + (z_alpha^3 - 3*z_alpha)*K / 24 - (2*z_alpha^3 - 5*z_alpha)*S^2 / 36"}
          </div>
          <p className="text-xs text-slate-300 font-sans leading-relaxed">
            {isPlain
              ? "VaR 95% means: 'On 19 out of 20 trading days, your losses won't exceed this dollar amount.' It gives you peace of mind before entering a position."
              : "Where S is sample skewness and K is excess kurtosis. This strictly guarantees that 99% VaR is more conservative than 95% VaR."}
          </p>
        </div>
      </section>

      {/* CHAPTER 7: MULTI-SOURCE SYNTHESIS, FRED MACRO & SEC EDGAR FORM 4 */}
      <section id="chapter-6" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">💎</span>
          <h2 className="text-lg sm:text-xl font-bold text-cyan-400 tracking-tight">
            {isPlain ? "Chapter 7: Fed Interest Rates, Real Insiders & Live Alerts" : "Chapter 7: Multi-Source Synthesis, FRED Macro Regimes & Cross-App Sync"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "Looking at charts alone creates false signals when the broader market is under stress. The terminal combines 4 live intelligence feeds into one unified conviction score (0 to 100):"
            : "Pure technical price action creates frequent false breakouts when market liquidity is hostile. To eliminate blindspots, the terminal continuously correlates 4 authoritative quantitative streams into a unified Composite Conviction Score (0–100):"}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
          <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
            <div className="flex items-center justify-between">
              <strong className="text-emerald-400 font-bold text-sm">{isPlain ? "🏛️ Federal Reserve FRED Data" : "🏛️ Federal Reserve FRED Macro"}</strong>
              <span className="text-[10px] bg-emerald-950 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-800">MACRO REGIME</span>
            </div>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              {isPlain
                ? "Tracks US Treasury Yield Curve spreads and corporate bond stress. Automatically warns you to reduce bet sizes when credit conditions tighten."
                : "Tracks 10Y-2Y Treasury Yield Curve spreads and Credit Spreads. Applies a dynamic 0.5x to 1.25x Macro Risk Multiplier."}
            </p>
          </div>

          <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
            <div className="flex items-center justify-between">
              <strong className="text-cyan-400 font-bold text-sm">{isPlain ? "🏢 SEC Form 4 (CEO & Director Buys)" : "🏢 SEC EDGAR Form 4 (C-Suite)"}</strong>
              <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-800">LEGAL INSIDERS</span>
            </div>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              {isPlain
                ? "Filters real open-market purchases (>= $100k) by CEOs and Directors under strict 2-day mandatory SEC disclosure rules."
                : "Filters open-market stock purchases (>= $100,000 USD) by CEOs, CFOs, and Board Directors under Section 16(a) of the Securities Exchange Act."}
            </p>
          </div>
        </div>
      </section>

      {/* CHAPTER 8: THE NO-BS PLAIN-ENGLISH JARGON BUSTER */}
      <section id="chapter-8" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">💬</span>
          <h2 className="text-lg sm:text-xl font-bold text-emerald-400 tracking-tight">
            Chapter 8: The No-BS Plain-English Jargon Buster
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          Wall Street loves complicated words because it lets them charge high management fees. Here is what all that technical jargon actually means in plain, unfiltered human English:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 text-xs">
          <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
            <div className="flex items-center justify-between">
              <strong className="text-rose-400 font-bold text-sm">Value at Risk (VaR)</strong>
              <span className="text-[10px] bg-rose-950 text-rose-300 px-1.5 py-0.5 rounded border border-rose-800">HUMAN TRANSLATION</span>
            </div>
            <p className="text-slate-200 text-xs font-semibold">The Worst-Case Crash Test</p>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              If the market has a terrible day tomorrow, how much money will you actually lose? VaR 95% means <em>&ldquo;19 out of 20 days, your losses won&rsquo;t exceed this number.&rdquo;</em>
            </p>
          </div>

          <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
            <div className="flex items-center justify-between">
              <strong className="text-cyan-400 font-bold text-sm">Piotroski F-Score (0–9)</strong>
              <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-800">HUMAN TRANSLATION</span>
            </div>
            <p className="text-slate-200 text-xs font-semibold">The BS & Accounting Truth Detector</p>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              A 9-point checklist created by a Stanford professor to see if a company is secretly drowning in debt or actually printing real cash. A score of 8 or 9 means their financial books are squeaky clean.
            </p>
          </div>

          <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
            <div className="flex items-center justify-between">
              <strong className="text-amber-400 font-bold text-sm">ROIC (Return on Capital)</strong>
              <span className="text-[10px] bg-amber-950 text-amber-300 px-1.5 py-0.5 rounded border border-amber-800">HUMAN TRANSLATION</span>
            </div>
            <p className="text-slate-200 text-xs font-semibold">Money-Making Efficiency</p>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              If you hand the CEO \$100, how many dollars do they bring back? An ROIC of 25% means they turn every \$100 of invested cash into \$25 of pure profit every single year.
            </p>
          </div>

          <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
            <div className="flex items-center justify-between">
              <strong className="text-purple-400 font-bold text-sm">STOCK Act Filing Lag</strong>
              <span className="text-[10px] bg-purple-950 text-purple-300 px-1.5 py-0.5 rounded border border-purple-800">HUMAN TRANSLATION</span>
            </div>
            <p className="text-slate-200 text-xs font-semibold">The Politician Delay Penalty</p>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              US politicians are legally allowed up to 45 days to tell the public what stocks they bought. If a politician bought 35 days ago, you are seeing old news — beware of chasing green candles late.
            </p>
          </div>
        </div>
      </section>

      {/* CHAPTER 9: THE 4 DECISION INTELLIGENCE ENGINES, DUAL VERNACULAR & EDGE ARCHITECTURE */}
      <section id="chapter-9" className="space-y-4">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <span className="text-lg sm:text-xl">🧠</span>
          <h2 className="text-lg sm:text-xl font-bold text-amber-400 tracking-tight">
            {isPlain 
              ? "Chapter 9: The 4 Trade Co-Pilots (Your Safety Check Before Buying)" 
              : "Chapter 9: The 4 Decision Intelligence Engines, Dual Vernacular & Edge Architecture"}
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
          {isPlain
            ? "Before you execute any trade, the terminal runs 4 automated co-pilot checks to keep your capital safe:"
            : "To bridge the gap between quantitative calculation and real-world execution, the terminal features a dedicated Decision Intelligence Suite that acts as an institutional co-pilot before you risk real capital:"}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
          {/* 1. Pre-Flight Checklist */}
          <div className="bg-[#111722] p-4 rounded-xl border border-cyan-800/60 space-y-2.5">
            <div className="flex items-center justify-between">
              <strong className="text-cyan-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                <span>✈️</span>
                <span>{isPlain ? "1. The 5-Point Green Light Checklist" : "1. 5-Point Pre-Flight Clearance Gate"}</span>
              </strong>
              <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-700">EXECUTION GATE</span>
            </div>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain 
                ? "A strict 5-item safety check that must give you 5/5 passes before you pull the trigger:" 
                : "A non-negotiable checklist that must achieve 5/5 passes before clearing execution:"}
            </p>
            <ul className="text-[11px] text-slate-400 font-sans space-y-1 list-disc pl-4">
              <li><strong>{isPlain ? "Trend & Support:" : "Trend & Moving Averages:"}</strong> {isPlain ? "Price is resting above key moving averages." : "Spot holding above 20 EMA and 50-day pivot."}</li>
              <li><strong>{isPlain ? "Coiled Spring Base:" : "Minervini VCP Base:"}</strong> {isPlain ? "Volatility has contracted, sellers are exhausted." : "Tight contraction volatility with defined support."}</li>
              <li><strong>{isPlain ? "Smart Money Backing:" : "Smart Money Asymmetry:"}</strong> {isPlain ? "Insiders are buying without sneaky distribution traps." : "Clean institutional backing without insider selling."}</li>
              <li><strong>{isPlain ? "No Impending Earnings Trap:" : "Binary Catalyst Buffer:"}</strong> {isPlain ? "Zero binary earnings or FDA announcements in the next 48 hours." : "Zero major hazard events within 48 hours."}</li>
              <li><strong>{isPlain ? "Calm Market Floor:" : "Macro Volatility Floor:"}</strong> {isPlain ? "Overall market fear index (VIX) is strictly below 26." : "Broad market VIX index strictly below 26.0."}</li>
            </ul>
          </div>

          {/* 2. Smart Money Divergence Radar */}
          <div className="bg-[#111722] p-4 rounded-xl border border-amber-800/60 space-y-2.5">
            <div className="flex items-center justify-between">
              <strong className="text-amber-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                <span>📡</span>
                <span>{isPlain ? "2. Smart Money vs. Retail Trap Radar" : "2. Smart Money Divergence Radar"}</span>
              </strong>
              <span className="text-[10px] bg-amber-950 text-amber-300 px-1.5 py-0.5 rounded border border-amber-700">ORDER FLOW</span>
            </div>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain
                ? "Catches when retail price and big institutional money are doing opposite things:"
                : "Detects decoupling between retail price direction and dark pool block flow:"}
            </p>
            <ul className="text-[11px] text-slate-400 font-sans space-y-1 list-disc pl-4">
              <li><strong>{isPlain ? "Stealth Accumulation (Bullish):" : "Stealth Accumulation:"}</strong> {isPlain ? "Price is flat, but big institutions and CEOs are quietly loading shares." : "Consolidation with high dark pool & C-suite buying."}</li>
              <li><strong>{isPlain ? "Distribution Trap (Bearish Warning):" : "Distribution Trap:"}</strong> {isPlain ? "Price is pumping, but executives and funds are dumping shares into the hype." : "New price highs on negative institutional flow & selling."}</li>
            </ul>
          </div>

          {/* 3. Historical Edge Scorecard */}
          <div className="bg-[#111722] p-4 rounded-xl border border-emerald-800/60 space-y-2.5">
            <div className="flex items-center justify-between">
              <strong className="text-emerald-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                <span>📊</span>
                <span>{isPlain ? "3. Track Record Edge Scorecard" : "3. Historical Edge Scorecard"}</span>
              </strong>
              <span className="text-[10px] bg-emerald-950 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-700">TRACK RECORD</span>
            </div>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain
                ? "Shows real backtested win rates (>65%) and profit factors across 5 proven trading patterns so you know the odds before risking capital."
                : "Audits empirical performance across 5 proven quantitative archetypes (Minervini VCP, Magic Formula, Peter Lynch GARP, Rule Breakers, and Turnaround Watch), reporting backtested win rates (>65%) and profit factors (>=2.0)."}
            </p>
          </div>

          {/* 4. Macro Stress Test Simulator */}
          <div className="bg-[#111722] p-4 rounded-xl border border-purple-800/60 space-y-2.5">
            <div className="flex items-center justify-between">
              <strong className="text-purple-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                <span>🌪️</span>
                <span>{isPlain ? "4. Portfolio Crash Test Simulator" : "4. Macro Stress Test Simulator"}</span>
              </strong>
              <span className="text-[10px] bg-purple-950 text-purple-300 px-1.5 py-0.5 rounded border border-purple-700">PORTFOLIO SHOCK</span>
            </div>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {isPlain
                ? "Simulates what happens to your stocks if the market tanks -5%, yields spike, or volatility explodes, and tells you exactly how much cash to hold as a safety buffer."
                : "Simulates systemic stress events (Tech Selloff -5%, Yield Surge +50bps, VIX Spike 35) across your positions using covariance Beta weighting, projecting total portfolio drawdown and recommending exact defensive cash reserves."}
            </p>
          </div>
        </div>

        {/* Dual Vernacular & Privacy Infrastructure Callout */}
        <div className="bg-[#090d14] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
          <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>⚡ {isPlain ? "One-Click Language Switcher & Speed Shield" : "Platform Infrastructure: Dual Vernacular & Edge Security"}</span>
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1.5">
              <strong className="text-emerald-400 block font-mono">{isPlain ? "Instant Language Switcher" : "Dual-Vernacular Mode Switcher"}</strong>
              <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
                {isPlain
                  ? "Toggle between Plain English (no-BS explanations) and Pro Quant (institutional math) instantly via the top navigation bar."
                  : "Toggle between Plain English Mode (clear, punchy explanations) and Pro Quant Mode (institutional Greek formulas and econometric terminology) instantly across all cards via the top navigation bar."}
              </p>
            </div>
            <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1.5">
              <strong className="text-cyan-400 block font-mono">{isPlain ? "Edge Speed & Zero-Cookie Privacy" : "Cloudflare Edge SWR & Cookieless Privacy"}</strong>
              <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
                {isPlain
                  ? "Supercharged with Cloudflare edge caching for instant <10ms loading, with 100% cookieless privacy and zero tracking cookies."
                  : "Engineered with RFC 5861 stale-while-revalidate edge caching delivering <10ms query responses, coupled with 100% cookieless, privacy-first GDPR compliance and Do Not Track (DNT) enforcement."}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer Navigation */}
      <footer className="border-t border-[#243044] pt-6 flex flex-wrap items-center justify-between gap-4">
        <Link
          href="/"
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold rounded-xl shadow-sm transition-transform active:scale-95 cursor-pointer"
        >
          ← Return to ARX Terminal
        </Link>
        <div className="text-xs text-slate-500">
          Grounded in SEC EDGAR, Capitol Hill STOCK Act &amp; Federal Reserve FRED Data
        </div>
      </footer>
    </main>
  );
}
