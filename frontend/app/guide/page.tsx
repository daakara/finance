import type { Metadata } from "next";
import Link from "next/link";
import Navbar from "../../components/Navbar";

export const metadata: Metadata = {
  title: "Quantitative Terminal Field Manual & Algorithmic Handbook",
  description:
    "Institutional manual detailing Congressional STOCK Act tracking, Mark Minervini VCP algorithmic entry points, Linda Raschke 20 EMA pullbacks, Cornish-Fisher Modified VaR, and 9-Point Piotroski scoring.",
  openGraph: {
    title: "Finance Terminal: Quantitative Field Manual & Algorithmic Handbook",
    description: "Master institutional quantitative trading, legislative STOCK Act signals, volatility invalidation ladders, and tail risk management.",
    url: "https://finance-xp8.pages.dev/guide/",
    siteName: "Finance Terminal",
    type: "article",
  },
  alternates: {
    canonical: "https://finance-xp8.pages.dev/guide/",
  },
};

export default function GuidePage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    "headline": "Quantitative Terminal Field Manual & Algorithmic Handbook",
    "description": "Comprehensive guide to Congressional STOCK Act tracking, Mark Minervini VCP algorithmic entry points, Cornish-Fisher Modified VaR risk modeling, and FRED macroeconomic regimes.",
    "author": {
      "@type": "Organization",
      "name": "Finance Terminal Quantitative Intelligence"
    },
    "publisher": {
      "@type": "Organization",
      "name": "Finance Terminal",
      "logo": {
        "@type": "ImageObject",
        "url": "https://finance-xp8.pages.dev/icons/icon-512x512.png"
      }
    },
    "datePublished": "2026-08-26",
    "dateModified": "2026-08-26"
  };

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans selection:bg-cyan-500 selection:text-black transition-colors duration-200">
      {/* Schema.org Structured Data */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <Navbar />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 font-mono space-y-10 sm:space-y-14 pb-24 sm:pb-16">
        {/* Hero Header */}
        <header className="border-b border-[#243044] pb-6 sm:pb-8 space-y-3">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-[11px] font-bold">
              INSTITUTIONAL QUANTITATIVE FIELD MANUAL
            </span>
            <span className="text-slate-500 text-xs">• Version 2.4 Specification</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight">
            Quantitative Platform Blueprint & Execution Handbook
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            A comprehensive, mathematically rigorous reference manual for professional traders, quantitative analysts, and fundamental investors. Learn the formulas, invalidation rules, statutory insider metrics, and portfolio risk equations powering this terminal.
          </p>

          {/* Quick Jump Navigation */}
          <div className="bg-[#090d14] p-3 rounded-xl border border-[#1b2434] flex flex-wrap gap-2 text-[11px] pt-3">
            <span className="text-slate-500 font-bold uppercase py-0.5">Quick Jump:</span>
            <a href="#chapter-1" className="text-cyan-400 hover:underline">Ch 1: Workspaces</a>
            <span className="text-slate-600">•</span>
            <a href="#chapter-chart" className="text-amber-400 hover:underline font-bold">Ch 2: Dual Chart Engine</a>
            <span className="text-slate-600">•</span>
            <a href="#chapter-2" className="text-cyan-400 hover:underline">Ch 3: Algorithmic Math</a>
            <span className="text-slate-600">•</span>
            <a href="#chapter-3" className="text-amber-400 hover:underline">Ch 4: STOCK Act Insiders</a>
            <span className="text-slate-600">•</span>
            <a href="#chapter-4" className="text-emerald-400 hover:underline">Ch 5: 5-Factor Radar</a>
            <span className="text-slate-600">•</span>
            <a href="#chapter-5" className="text-purple-400 hover:underline">Ch 6: Risk & Modified VaR</a>
            <span className="text-slate-600">•</span>
            <a href="#chapter-6" className="text-cyan-400 hover:underline font-bold">Ch 7: Multi-Source Synthesis & FRED Macro</a>
          </div>
        </header>

        {/* CHAPTER 1: WORKSPACES & DUAL USER JOURNEYS */}
        <section id="chapter-1" className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
            <span className="text-lg sm:text-xl">🗂️</span>
            <h2 className="text-lg sm:text-xl font-bold text-cyan-400 tracking-tight">
              Chapter 1: The 4 In-Terminal Modular Workspaces
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            Traditional terminals force analysts to scroll through hundreds of stacked widgets, causing severe cognitive overload and losing chart context. Our terminal splits analysis into 4 mobile-optimized modular domains (with dynamic viewport-responsive labels) anchored directly beneath the live price chart:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
              <strong className="text-cyan-300 text-sm flex items-center gap-1.5 font-mono">
                <span>🎯</span>
                <span>Execution & Levels (Default)</span>
              </strong>
              <p className="text-xs text-slate-300 font-sans leading-relaxed">
                Answers: <em>&ldquo;Where do I enter, where is my stop, and when do I take profit?&rdquo;</em> Houses the Minervini VCP accumulation ladder, Intraday Position Sizer, and ATR14 volatility bands.
              </p>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
              <strong className="text-amber-300 text-sm flex items-center gap-1.5 font-mono">
                <span>🏛️</span>
                <span>Smart Money & Insiders</span>
              </strong>
              <p className="text-xs text-slate-300 font-sans leading-relaxed">
                Answers: <em>&ldquo;What are politicians, corporate executives, and option market makers doing?&rdquo;</em> Displays US House/Senate STOCK Act filings, dark pool ATS volumes, and options sweeps.
              </p>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
              <strong className="text-emerald-300 text-sm flex items-center gap-1.5 font-mono">
                <span>📊</span>
                <span>Factors & Macro Intelligence</span>
              </strong>
              <p className="text-xs text-slate-300 font-sans leading-relaxed">
                Answers: <em>&ldquo;How healthy is the company and what is the macro regime?&rdquo;</em> Displays 5-Factor profile radar, 9-point Piotroski F-Scores, and FRED 10Y-2Y yield curve spreads.
              </p>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
              <strong className="text-purple-300 text-sm flex items-center gap-1.5 font-mono">
                <span>🛡️</span>
                <span>Risk & Contagion Networks</span>
              </strong>
              <p className="text-xs text-slate-300 font-sans leading-relaxed">
                Answers: <em>&ldquo;If a peer or supplier collapses, how does the shock cascade?&rdquo;</em> Displays directed supply-chain topologies, Cornish-Fisher M-VaR, and self-healing hit rate calibrations.
              </p>
            </div>
          </div>
        </section>

        {/* CHAPTER 2: INTERACTIVE DUAL-HORIZON CANDLESTICK & INDICATOR ENGINE */}
        <section id="chapter-chart" className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
            <span className="text-lg sm:text-xl">📈</span>
            <h2 className="text-lg sm:text-xl font-bold text-amber-400 tracking-tight">
              Chapter 2: Interactive Dual-Horizon Candlestick & Indicator Engine
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            The platform embeds an institutional TradingView Lightweight Charts canvas that dynamically adapts its time scale, indicators, and mathematical percentage baselines according to your selected trading persona:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
            <div className="bg-[#111722] p-4 rounded-xl border border-amber-800/60 space-y-2.5">
              <div className="flex items-center justify-between">
                <strong className="text-amber-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                  <span>⚡</span>
                  <span>Day Trader Scalp Sessions</span>
                </strong>
                <span className="text-[10px] bg-amber-950 text-amber-300 px-1.5 py-0.5 rounded border border-amber-700">INTRADAY</span>
              </div>
              <ul className="text-xs text-slate-300 font-sans space-y-1.5 list-disc pl-4">
                <li><strong>Timeframes:</strong> <code>1m</code> (45-min scalp), <code>5m</code> (3.75-hr session), <code>15m</code> (12-hr multi-session), <code>1h</code> (weekly trend).</li>
                <li><strong>Primary Overlay:</strong> <strong>Volume-Weighted Average Price (VWAP)</strong> in amber (<code>#f59e0b</code>). Institutional benchmark for intraday mean-reversion.</li>
                <li><strong>Time Scale:</strong> Microsecond UTC Unix epoch timestamps formatted for high-frequency price action.</li>
              </ul>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-cyan-800/60 space-y-2.5">
              <div className="flex items-center justify-between">
                <strong className="text-cyan-300 text-sm font-bold flex items-center gap-1.5 font-mono">
                  <span>🏛️</span>
                  <span>Long-Term Macro Horizons</span>
                </strong>
                <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-700">MACRO</span>
              </div>
              <ul className="text-xs text-slate-300 font-sans space-y-1.5 list-disc pl-4">
                <li><strong>Horizons:</strong> <code>1M</code> (22 daily bars), <code>6M</code> (130 daily bars), <code>1Y</code> (252 daily bars), <code>3Y</code> (156 weekly bars), <code>5Y</code> (60 monthly bars).</li>
                <li><strong>Primary Overlay:</strong> <strong>20 Exponential Moving Average (20 EMA)</strong> in sky blue (<code>#38bdf8</code>). Dynamic support floor for institutional pullbacks.</li>
                <li><strong>Time Scale:</strong> Calendar-accurate ISO <code>YYYY-MM-DD</code> date formatting with strict monotonic ordering.</li>
              </ul>
            </div>
          </div>

          {/* Metric Disambiguation Callout */}
          <div className="bg-[#090d14] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
            <h3 className="text-xs sm:text-sm font-bold text-slate-100 uppercase tracking-wider flex items-center gap-2">
              <span>🎯 Metric Disambiguation: Watchlist Row vs. Chart Header Return</span>
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed font-sans">
              To eliminate confusion between short-term noise and long-term trends, the terminal explicitly separates two distinct percentage return calculations:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1.5">
                <strong className="text-slate-200 block font-mono">Watchlist Sidebar Badge (24H Daily Return)</strong>
                <div className="bg-[#0b1019] p-2 rounded text-emerald-400 font-mono text-[11px]">
                  R_24h = (Current Price - Previous Close) / Previous Close
                </div>
                <p className="text-[11px] text-slate-400 font-sans leading-snug">
                  Measures purely today&apos;s daily trading change relative to yesterday&apos;s closing bell (e.g. <code>+2.65% 24H</code>).
                </p>
              </div>

              <div className="bg-[#06090f] p-3 rounded-lg border border-[#1b2434] space-y-1.5">
                <strong className="text-slate-200 block font-mono">Chart Header Badge (Active Horizon Return)</strong>
                <div className="bg-[#0b1019] p-2 rounded text-cyan-300 font-mono text-[11px]">
                  R_horizon = (Last Candle Close - First Candle Open) / First Candle Open
                </div>
                <p className="text-[11px] text-slate-400 font-sans leading-snug">
                  Measures total cumulative trajectory across the active dataset with explicit horizon pill tag (e.g. <code>+28.40% 1Y</code> or <code>+0.80% 5M</code>).
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CHAPTER 3: ALGORITHMIC EXECUTION & FORMULAS */}
        <section id="chapter-2" className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
            <span className="text-lg sm:text-xl">🧮</span>
            <h2 className="text-lg sm:text-xl font-bold text-cyan-400 tracking-tight">
              Chapter 3: Algorithmic Execution Formulas & Sizing Math
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            Our execution engine replaces emotional discretion with concrete mathematical boundaries based on Mark Minervini&apos;s Volatility Contraction Pattern (VCP) and Linda Raschke&apos;s 20 EMA pullback setup.
          </p>

          {/* Sizing Math */}
          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
            <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>📐 1. Exact Position Sizing Equation</span>
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed font-sans">
              Never risk more than your pre-defined capital threshold ($1\%–2\%$ of account equity per trade). The terminal calculates exact share volume using:
            </p>
            <div className="bg-[#070a10] p-3 rounded-lg border border-[#162030] text-cyan-300 font-bold text-xs">
              Shares to Buy = (Account Capital * Risk Budget %) / (Entry Price - Stop Loss Price)
            </div>
            <p className="text-[11px] text-slate-400 leading-snug font-sans">
              <strong>Example:</strong> With a \$50,000 portfolio risking 1% (\$500) buying NVDA at \$213.05 with a Stop Loss at \$201.35 (\$11.70 per share risk), the sizer dictates buying exactly <strong>42 shares</strong>.
            </p>
          </div>

          {/* Multi-Tier Take Profit Ladder */}
          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
            <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>🎯 2. Dual Take-Profit Ladder (TP1 & TP2)</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="bg-[#111722] p-3 rounded-lg border border-[#243044] space-y-1">
                <strong className="text-emerald-400 block font-bold">Target 1 (Scale 50% Position)</strong>
                <p className="text-slate-300 text-[11px] font-sans">
                  Formula: <code>Spot + (2.5 * ATR14)</code>. Once price touches TP1, sell 50% of the position to lock in profit and automatically raise the stop loss on the remaining 50% to breakeven.
                </p>
              </div>
              <div className="bg-[#111722] p-3 rounded-lg border border-[#243044] space-y-1">
                <strong className="text-purple-400 block font-bold">Target 2 (Extended Runner Exit)</strong>
                <p className="text-slate-300 text-[11px] font-sans">
                  Formula: <code>Spot + (4.5 * ATR14)</code>. Trail the remaining 50% runner along the 20 EMA until a daily candle closes below the moving average.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CHAPTER 4: CONGRESSIONAL STOCK ACT INTELLIGENCE */}
        <section id="chapter-3" className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
            <span className="text-lg sm:text-xl">🏛️</span>
            <h2 className="text-lg sm:text-xl font-bold text-amber-400 tracking-tight">
              Chapter 4: Congressional STOCK Act & Political Alpha
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            Under Public Law 112-105 (Stop Trading on Congressional Knowledge Act of 2012), members of the US Congress and Senate are legally required to disclose securities transactions within 45 days.
          </p>

          <div className="bg-[#111722] p-4 sm:p-5 rounded-xl border border-[#243044] space-y-3 text-xs">
            <h3 className="text-xs sm:text-sm font-bold text-slate-100 uppercase tracking-wider">
              How the Terminal Filters Actionable Political Signals:
            </h3>
            <ul className="space-y-2 text-slate-300 font-sans list-disc pl-5">
              <li>
                <strong>Filing Lag Decay:</strong> Trades disclosed near the 45-day statutory deadline are penalized in conviction scoring to prevent retail traders from buying the tail end of an expired legislative catalyst.
              </li>
              <li>
                <strong>Committee Jurisdiction Overlap:</strong> Trades are weighted 2.5x higher when a member sits on a committee with direct oversight of the traded asset (e.g. Armed Services purchasing Defense contractors, Energy & Commerce purchasing Semiconductors).
              </li>
              <li>
                <strong>Option Sweeps vs. Equity:</strong> Deep In-The-Money (ITM) Call purchases (e.g., Nancy Pelosi LEAPS) indicate leveraged institutional accumulation with defined downside floor risk.
              </li>
            </ul>
          </div>
        </section>

        {/* CHAPTER 5: 5-FACTOR RADAR & PIOTROSKI F-SCORE */}
        <section id="chapter-4" className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
            <span className="text-lg sm:text-xl">📊</span>
            <h2 className="text-lg sm:text-xl font-bold text-emerald-400 tracking-tight">
              Chapter 5: 5-Factor Fundamental DNA & Piotroski Score
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            The platform synthesizes thousands of fundamental balance sheet data points into a multi-dimensional quantitative profile:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
              <strong className="text-cyan-400 font-bold block">1. Growth Score (0-100)</strong>
              <p className="text-slate-300 text-[11px] font-sans">Calculated via 3-year revenue CAGR, forward EPS expansion rate, and free cash flow acceleration.</p>
            </div>
            <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
              <strong className="text-emerald-400 font-bold block">2. Quality Score (0-100)</strong>
              <p className="text-slate-300 text-[11px] font-sans">Measures Return on Invested Capital (ROIC &gt; 15%), gross profit margin moats, and low financial leverage.</p>
            </div>
            <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
              <strong className="text-purple-400 font-bold block">3. Valuation Score (0-100)</strong>
              <p className="text-slate-300 text-[11px] font-sans">Derived from PEG ratio, EV/EBITDA multiple discounts, and enterprise DCF fair value spreads.</p>
            </div>
            <div className="bg-[#0b1019] p-3.5 rounded-xl border border-[#1b2434] space-y-1">
              <strong className="text-amber-400 font-bold block">4. Piotroski 9-Point F-Score</strong>
              <p className="text-slate-300 text-[11px] font-sans">Scores 8-9 indicate pristine balance sheet quality; scores &le; 3 signal structural accounting insolvency.</p>
            </div>
          </div>
        </section>

        {/* CHAPTER 6: RISK & MODIFIED VAR */}
        <section id="chapter-5" className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
            <span className="text-lg sm:text-xl">🛡️</span>
            <h2 className="text-lg sm:text-xl font-bold text-purple-400 tracking-tight">
              Chapter 6: Mathematical Invariants & Cornish-Fisher Modified VaR
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            Standard Gaussian Value-at-Risk assumes symmetric normal distributions, severely underestimating fat-tail crash risks in equity markets. Our quantitative engine applies a polynomial Cornish-Fisher expansion adjusted for non-normal Skewness and excess Kurtosis:
          </p>

          <div className="bg-[#090d14] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
            <h3 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider">
              Cornish-Fisher Expansion Formula:
            </h3>
            <div className="bg-[#05070a] p-3 rounded-lg border border-[#162030] font-mono text-cyan-300 text-xs leading-relaxed overflow-x-auto">
              Z_cf = z_alpha + (z_alpha^2 - 1)*S / 6 + (z_alpha^3 - 3*z_alpha)*K / 24 - (2*z_alpha^3 - 5*z_alpha)*S^2 / 36
            </div>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              Where <code>S</code> is sample skewness (clipped to <code>[-3.0, 3.0]</code>) and <code>K</code> is excess kurtosis (clipped to <code>[-1.0, 10.0]</code>). This eliminates polynomial inversion on outlier shocks and strictly guarantees that <strong>99% VaR is more conservative than 95% VaR</strong>.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
              <strong className="text-purple-300 text-sm font-bold block">📐 Strict Execution Invariant</strong>
              <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
                The terminal guarantees: <code>Stop Loss &lt; Optimal Entry Min &le; Optimal Entry Max &le; Current Spot &lt; Target 1 &lt; Target 2</code>. Accumulation zones are strictly capped at or below spot price.
              </p>
            </div>

            <div className="bg-[#111722] p-4 rounded-xl border border-[#243044] space-y-2">
              <strong className="text-emerald-300 text-sm font-bold block">⚖️ Bounded Risk Denominators</strong>
              <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
                Sortino, Calmar, Pain, and Reward-to-Risk ratios enforce minimum denominator floors to prevent artificial ratio spikes on ultra-tight stops.
              </p>
            </div>
          </div>
        </section>

        {/* CHAPTER 7: MULTI-SOURCE SYNTHESIS, FRED MACRO & SEC EDGAR FORM 4 */}
        <section id="chapter-6" className="space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
            <span className="text-lg sm:text-xl">💎</span>
            <h2 className="text-lg sm:text-xl font-bold text-cyan-400 tracking-tight">
              Chapter 7: Multi-Source Synthesis, FRED Macro Regimes & SEC Form 4
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            Pure technical price action creates frequent false breakouts when market liquidity is hostile. To eliminate blindspots, the terminal continuously correlates 4 authoritative quantitative streams into a unified <strong>Composite Conviction Score (0–100)</strong>:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
            <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
              <div className="flex items-center justify-between">
                <strong className="text-emerald-400 font-bold text-sm">🏛️ Federal Reserve FRED Macro</strong>
                <span className="text-[10px] bg-emerald-950 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-800">MACRO REGIME</span>
              </div>
              <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
                Tracks 10Y-2Y Treasury Yield Curve spreads (<code>T10Y2Y</code>) and High-Yield Option-Adjusted Credit Spreads (<code>BAMLH0A0HYM2</code>). Applies a dynamic <strong>0.5x to 1.25x Macro Risk Multiplier</strong> to scale position budgets based on systemic credit stress.
              </p>
            </div>

            <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
              <div className="flex items-center justify-between">
                <strong className="text-cyan-400 font-bold text-sm">🏢 SEC EDGAR Form 4 (C-Suite)</strong>
                <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-800">LEGAL INSIDERS</span>
              </div>
              <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
                Filters open-market stock purchases (&ge; $100,000 USD) by CEOs, CFOs, and Board Directors under Section 16(a) of the Securities Exchange Act of 1934 (mandatory 2-day disclosure). Verified directly against official SEC EDGAR CIK databases.
              </p>
            </div>

            <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
              <div className="flex items-center justify-between">
                <strong className="text-amber-400 font-bold text-sm">🔥 1-Line Catalyst Micro-Tags</strong>
                <span className="text-[10px] bg-amber-950 text-amber-300 px-1.5 py-0.5 rounded border border-amber-800">DISCOVERY PULSE</span>
              </div>
              <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
                Embedded directly into the main chart header to answer <em>&ldquo;Why is this stock moving today?&rdquo;</em> in under 1 second, connecting earnings beats, FDA trial readouts, and AI chip demand to price momentum.
              </p>
            </div>

            <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1b2434] space-y-2">
              <div className="flex items-center justify-between">
                <strong className="text-purple-400 font-bold text-sm">💼 1-Click Execution Hand-off</strong>
                <span className="text-[10px] bg-purple-950 text-purple-300 px-1.5 py-0.5 rounded border border-purple-800">PORTFOLIO SYNC</span>
              </div>
              <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
                Directly sends mathematically calculated share quantities, volatility stop-losses, and multi-tier profit targets from the Position Sizer into persistent, anonymous local storage tracking with zero manual re-typing.
              </p>
            </div>
          </div>
        </section>

        {/* Footer Navigation */}
        <footer className="border-t border-[#243044] pt-6 flex flex-wrap items-center justify-between gap-4">
          <Link
            href="/"
            className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-xs font-bold rounded-xl shadow transition-transform active:scale-95 cursor-pointer"
          >
            ← Return to Quantitative Terminal
          </Link>
          <div className="text-xs text-slate-500">
            Grounded in SEC EDGAR, Capitol Hill STOCK Act & Federal Reserve FRED Data
          </div>
        </footer>
      </main>
    </div>
  );
}