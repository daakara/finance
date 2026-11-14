"use client";

import Link from "next/link";
import { 
  Shield, 
  TrendingUp, 
  FileText, 
  Landmark, 
  Sliders, 
  HelpCircle, 
  AlertTriangle, 
  CheckCircle2, 
  ArrowRight,
  BookOpen,
  Compass
} from "lucide-react";

export default function GuideContent() {
  return (
    <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-12 pb-24 sm:pb-16 text-slate-300 font-sans">
      {/* Header & Mission */}
      <header className="border-b border-[#243044] pb-8 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800 text-[11px] font-mono font-bold tracking-wider uppercase">
              Field Manual &amp; Quantitative Methodology
            </span>
            <span className="text-slate-500 font-mono text-xs">• Version 2.11.0</span>
          </div>
          <Link
            href="/glossary/"
            className="flex items-center gap-1.5 text-xs font-mono text-cyan-400 hover:text-cyan-300 bg-[#090d14] px-3 py-1.5 rounded-lg border border-[#1e293b] hover:border-cyan-800 transition-colors"
          >
            <Compass className="w-3.5 h-3.5 text-cyan-400" />
            <span>Open Quantitative Glossary →</span>
          </Link>
        </div>

        <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight font-mono">
          ARX Terminal Quantitative Field Manual
        </h1>
        <p className="text-sm sm:text-base text-slate-300 leading-relaxed font-sans">
          This manual provides a rigorous, transparent reference for the analytical models, capital allocation risk boundaries, and market surveillance systems powering the ARX Terminal. It explains how our execution corridors are calculated, how capital is defended through mathematical volatility bounds, and how multi-pillar evidence is synthesized into decision-ready context.
        </p>

        {/* Quick Jump Curriculum Navigation */}
        <nav aria-label="Curriculum Navigation" className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] text-xs font-mono">
          <div className="text-[11px] text-slate-400 font-bold uppercase tracking-wider mb-2">Curriculum Modules:</div>
          <div className="flex flex-wrap gap-2.5">
            <a href="#module-foundations" className="text-cyan-400 hover:underline">
              1. Foundations &amp; Capital Sizing
            </a>
            <span className="text-slate-600">•</span>
            <a href="#module-technical" className="text-amber-400 hover:underline">
              2. Technical Execution &amp; VCP
            </a>
            <span className="text-slate-600">•</span>
            <a href="#module-fundamental" className="text-emerald-400 hover:underline">
              3. Fundamental DNA &amp; Accounting
            </a>
            <span className="text-slate-600">•</span>
            <a href="#module-smart-money" className="text-purple-400 hover:underline">
              4. Smart Money &amp; STOCK Act
            </a>
            <span className="text-slate-600">•</span>
            <a href="#module-using-arx" className="text-cyan-400 hover:underline">
              5. Confluence &amp; Risk Gates
            </a>
          </div>
        </nav>
      </header>

      {/* MODULE 1: FOUNDATIONS & CAPITAL PRESERVATION */}
      <section id="module-foundations" className="space-y-6">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <Shield className="w-5 h-5 text-cyan-400 shrink-0" />
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight font-mono">
            Module 1: Foundations of Capital Preservation &amp; Sizing
          </h2>
        </div>

        <div className="space-y-4">
          <h3 className="text-base font-bold text-white font-mono flex items-center gap-2">
            <span>1.1 The 1% Non-Discretionary Risk Allocation Rule</span>
          </h3>
          <p className="text-sm leading-relaxed">
            The foundation of professional capital preservation is strictly capping the dollar risk of any single trade to a predetermined fraction of total portfolio equity—typically 1.0% to 2.0%. Sizing is never dictated by emotional conviction; it is calculated from the mathematical distance between your planned entry price and your technical stop-loss level.
          </p>

          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3 font-mono">
            <div className="text-xs uppercase text-slate-400 font-bold tracking-wider">Position Sizing Equation</div>
            <div className="p-3 bg-[#05070a] rounded-lg border border-[#162030] text-cyan-300 text-xs sm:text-sm overflow-x-auto">
              <code>Shares to Buy = (Total Portfolio Equity × Risk Budget %) / (Entry Price - Stop-Loss Invalidation Price)</code>
            </div>
            <div className="text-xs text-slate-400 space-y-1 font-sans">
              <p>
                <strong>Concrete Example:</strong> With a \$50,000 account, risking 1.0% caps maximum allowable loss on the trade at \$500. If buying an asset at \$213.00 with a confirmed stop-loss at \$201.00 (\$12.00 risk per share):
              </p>
              <p className="font-mono text-slate-300">
                \$500 / \$12.00 = 41.67 → Buy exactly <strong>41 shares</strong> (\$8,733 total position value).
              </p>
              <p>
                If the stop-loss triggers, exactly \$492 (0.98% of equity) is lost. 99% of your capital remains intact to trade again.
              </p>
            </div>
          </div>

          <h3 className="text-base font-bold text-white font-mono pt-4 flex items-center gap-2">
            <span>1.2 The 4-Tier Asset Architecture</span>
          </h3>
          <p className="text-sm leading-relaxed">
            To provide sub-second screen responsiveness while retaining unlimited search flexibility across U.S. equities, ARX Terminal organizes market surveillance into four distinct tiers:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
            <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1b2434] space-y-1">
              <strong className="text-cyan-300 block">Tier 1: Master Catalog</strong>
              <p className="text-slate-400 font-sans">Verified fundamental baseline assets with balance-sheet auditing and live price store hydration.</p>
            </div>
            <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1b2434] space-y-1">
              <strong className="text-emerald-300 block">Tier 2: Screener Candidates</strong>
              <p className="text-slate-400 font-sans">Dynamically evaluated Stage 2 candidates scored against volatility contraction, GARP, and quality moats.</p>
            </div>
            <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1b2434] space-y-1">
              <strong className="text-amber-300 block">Tier 3: Published Research Dossiers</strong>
              <p className="text-slate-400 font-sans">In-depth, point-in-time equity dossiers featuring five scenario DCFs and capital structure analysis.</p>
            </div>
            <div className="p-3 rounded-lg bg-[#0c1017] border border-[#1b2434] space-y-1">
              <strong className="text-purple-300 block">Tier 4: Universal OmniSearch</strong>
              <p className="text-slate-400 font-sans">On-demand query engine generating dynamic execution levels, ATR bands, and trade plans for any U.S. ticker.</p>
            </div>
          </div>
        </div>
      </section>

      {/* MODULE 2: TECHNICAL ANALYSIS & EXECUTION GEOMETRY */}
      <section id="module-technical" className="space-y-6">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <TrendingUp className="w-5 h-5 text-amber-400 shrink-0" />
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight font-mono">
            Module 2: Technical Execution &amp; Volatility Contraction
          </h2>
        </div>

        <div className="space-y-4">
          <h3 className="text-base font-bold text-white font-mono">
            2.1 Stan Weinstein Stage 2 &amp; Mark Minervini VCP Principles
          </h3>
          <p className="text-sm leading-relaxed">
            ARX Terminal filters the market through Stan Weinstein&apos;s Stage Analysis: only assets in a confirmed <strong>Stage 2 Advancing Uptrend</strong> are eligible for actionable consideration. In Stage 2, price is trading above an upward-sloping 50-day and 200-day moving average, with the 50-day positioned above the 200-day.
          </p>
          <p className="text-sm leading-relaxed">
            Within Stage 2, the terminal scans for Mark Minervini&apos;s <strong>Volatility Contraction Pattern (VCP)</strong>. A VCP visually reflects the progressive absorption of floating supply by institutional buyers:
          </p>

          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3">
            <div className="text-xs uppercase text-slate-400 font-mono font-bold tracking-wider">VCP Contraction Sequence Mechanics</div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
              <div className="p-3 bg-[#06090f] rounded-lg border border-[#162030] space-y-1">
                <span className="text-slate-500 font-bold">WAVE 1 (T1)</span>
                <div className="text-rose-400 font-bold">-15% to -25% Pullback</div>
                <p className="text-slate-400 text-[11px] font-sans">Initial correction off the highs. Weak hands panic and sell into institutional bids.</p>
              </div>
              <div className="p-3 bg-[#06090f] rounded-lg border border-[#162030] space-y-1">
                <span className="text-slate-500 font-bold">WAVE 2 (T2)</span>
                <div className="text-amber-400 font-bold">-7% to -12% Pullback</div>
                <p className="text-slate-400 text-[11px] font-sans">Shallower pullback. Sellers are visibly fewer; price recovers faster.</p>
              </div>
              <div className="p-3 bg-[#06090f] rounded-lg border border-[#162030] space-y-1">
                <span className="text-slate-500 font-bold">WAVE 3 (T3 - Pivot)</span>
                <div className="text-emerald-400 font-bold">-2% to -5% Pullback</div>
                <p className="text-slate-400 text-[11px] font-sans">Tight consolidation. Volume dries up below 50% of the 50-day moving average. Pivot forms.</p>
              </div>
            </div>
            <div className="text-xs text-slate-400 font-sans pt-1">
              <strong>Limitation:</strong> VCP patterns fail when the broad market index (e.g. S&amp;P 500) enters distribution or when volatility (VIX) spikes above 26.0. Always verify macro market conditions before acting on a breakout.
            </div>
          </div>

          <h3 className="text-base font-bold text-white font-mono pt-4">
            2.2 The 4 Mathematical ATR Execution States
          </h3>
          <p className="text-sm leading-relaxed">
            Every candidate evaluated by the terminal is classified into one of four mutually exclusive execution states based on price relation to its 14-day Average True Range (ATR) corridors:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
            <div className="p-3.5 rounded-xl bg-[#111722] border border-emerald-900/50 space-y-1.5">
              <div className="flex items-center justify-between">
                <strong className="text-emerald-400 font-bold">1. IN_BUY_ZONE</strong>
                <span className="text-[10px] bg-emerald-950 text-emerald-300 px-1.5 py-0.5 rounded">ACCUMULATION</span>
              </div>
              <p className="text-slate-300 font-sans text-xs">
                Price is resting within the optimal accumulation corridor (between the entry ceiling and technical support). Reward-to-risk ratio is at least 2.0:1.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-[#111722] border border-cyan-900/50 space-y-1.5">
              <div className="flex items-center justify-between">
                <strong className="text-cyan-400 font-bold">2. APPROACHING_TARGET</strong>
                <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded">MOMENTUM</span>
              </div>
              <p className="text-slate-300 font-sans text-xs">
                Price has broken out above the entry corridor and is advancing toward Target 1. Risk is no longer favorable for new entries, but existing holders can trail stops.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-[#111722] border border-amber-900/50 space-y-1.5">
              <div className="flex items-center justify-between">
                <strong className="text-amber-400 font-bold">3. WAITING_PULLBACK</strong>
                <span className="text-[10px] bg-amber-950 text-amber-300 px-1.5 py-0.5 rounded">OVEREXTENDED</span>
              </div>
              <p className="text-slate-300 font-sans text-xs">
                Price has extended more than 2% beyond the accumulation corridor. Buying here exposes the trader to elevated mean-reversion risk. Wait for a disciplined pullback.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-[#111722] border border-rose-900/50 space-y-1.5">
              <div className="flex items-center justify-between">
                <strong className="text-rose-400 font-bold">4. STOPPED_OUT</strong>
                <span className="text-[10px] bg-rose-950 text-rose-300 px-1.5 py-0.5 rounded">INVALIDATED</span>
              </div>
              <p className="text-slate-300 font-sans text-xs">
                Price has closed beneath the stop-loss invalidation floor. The technical thesis is broken. The position is exited immediately to preserve capital.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* MODULE 3: FUNDAMENTAL PROFILES & ACCOUNTING QUALITY */}
      <section id="module-fundamental" className="space-y-6">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <FileText className="w-5 h-5 text-emerald-400 shrink-0" />
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight font-mono">
            Module 3: Fundamental DNA &amp; Balance Sheet Forensics
          </h2>
        </div>

        <div className="space-y-4">
          <h3 className="text-base font-bold text-white font-mono">
            3.1 The Piotroski 9-Point F-Score Framework
          </h3>
          <p className="text-sm leading-relaxed">
            In quantitative finance, technical breakouts must be confirmed by financial health. ARX Terminal implements Stanford Professor Joseph Piotroski&apos;s 9-point fundamental accounting score across three core operational dimensions:
          </p>

          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3 text-xs font-mono">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-[#06090f] border border-[#162030] space-y-1.5">
                <strong className="text-cyan-300 block">PROFITABILITY (4 pts)</strong>
                <ul className="text-slate-400 font-sans space-y-1 list-disc pl-4 text-[11px]">
                  <li>Positive Net Income (1 pt)</li>
                  <li>Positive Operating Cash Flow (1 pt)</li>
                  <li>Return on Assets (ROA) higher than prior year (1 pt)</li>
                  <li>Cash Flow from Operations &gt; Net Income (Accruals test) (1 pt)</li>
                </ul>
              </div>
              <div className="p-3 rounded-lg bg-[#06090f] border border-[#162030] space-y-1.5">
                <strong className="text-emerald-300 block">LEVERAGE &amp; LIQUIDITY (3 pts)</strong>
                <ul className="text-slate-400 font-sans space-y-1 list-disc pl-4 text-[11px]">
                  <li>Lower long-term debt-to-equity ratio vs prior year (1 pt)</li>
                  <li>Higher Current Ratio vs prior year (1 pt)</li>
                  <li>Zero new shares issued (no equity dilution) (1 pt)</li>
                </ul>
              </div>
              <div className="p-3 rounded-lg bg-[#06090f] border border-[#162030] space-y-1.5">
                <strong className="text-purple-300 block">OPERATING EFFICIENCY (2 pts)</strong>
                <ul className="text-slate-400 font-sans space-y-1 list-disc pl-4 text-[11px]">
                  <li>Higher Gross Margin vs prior year (Pricing power moat) (1 pt)</li>
                  <li>Higher Asset Turnover Ratio vs prior year (1 pt)</li>
                </ul>
              </div>
            </div>
            <div className="pt-2 text-slate-300 font-sans text-xs">
              <strong>Interpretation:</strong> A score of <strong>8–9</strong> represents exceptional balance sheet strength and earnings quality. Scores of <strong>4–7</strong> indicate standard operating stability. Scores of <strong>0–3</strong> flag high risk of accounting manipulation or impending liquidity distress.
            </div>
          </div>

          <h3 className="text-base font-bold text-white font-mono pt-4">
            3.2 Return on Invested Capital (ROIC) vs Cash Conversion
          </h3>
          <p className="text-sm leading-relaxed">
            Revenue growth alone does not create shareholder value if a business consumes excessive capital to achieve it. ARX evaluates <strong>Return on Invested Capital (ROIC)</strong>: if a company generates an ROIC of 25%, it earns \$25 in after-tax operating profit for every \$100 of debt and equity invested. Companies with an ROIC consistently above 15% and positive free-cash-flow conversion possess durable competitive moats.
          </p>
        </div>
      </section>

      {/* MODULE 4: SMART MONEY & REGULATORY FORENSICS */}
      <section id="module-smart-money" className="space-y-6">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <Landmark className="w-5 h-5 text-purple-400 shrink-0" />
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight font-mono">
            Module 4: Smart Money Forensics &amp; Statutory Disclosures
          </h2>
        </div>

        <div className="space-y-4">
          <h3 className="text-base font-bold text-white font-mono">
            4.1 SEC Form 4 Insider Tracking
          </h3>
          <p className="text-sm leading-relaxed">
            Under Section 16(a) of the Securities Exchange Act of 1934, corporate officers, directors, and 10% principal shareholders must report transactions within two business days on SEC Form 4. ARX Terminal isolates <strong>open-market purchases</strong> by CEOs and CFOs exceeding \$100,000, filtering out routine option exercises, tax-withholding sales, and programmatic 10b5-1 selling schedules.
          </p>

          <h3 className="text-base font-bold text-white font-mono pt-4">
            4.2 Congressional STOCK Act Disclosures &amp; The 45-Day Delay Penalty
          </h3>
          <p className="text-sm leading-relaxed">
            Under the Stop Trading on Congressional Knowledge Act of 2012 (Public Law 112-105), members of the U.S. Senate and House of Representatives must disclose securities transactions within 45 days. Because 45 days is a substantial lag in modern electronic markets, raw political filings cannot be copied blindly.
          </p>

          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-3 font-mono text-xs">
            <div className="text-xs uppercase text-slate-400 font-bold tracking-wider">Staleness Time-Decay Schedule</div>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2.5">
              <div className="p-3 bg-[#06090f] rounded-lg border border-emerald-800/40 space-y-1">
                <span className="text-emerald-400 font-bold block">0 to 15 Days Lag</span>
                <span className="text-white font-bold text-xs">0 pt penalty</span>
                <p className="text-[11px] text-slate-400 font-sans">Fresh filing. Maximum predictive weight.</p>
              </div>
              <div className="p-3 bg-[#06090f] rounded-lg border border-cyan-800/40 space-y-1">
                <span className="text-cyan-400 font-bold block">16 to 30 Days Lag</span>
                <span className="text-white font-bold text-xs">-5 pt penalty</span>
                <p className="text-[11px] text-slate-400 font-sans">Normal statutory compliance window.</p>
              </div>
              <div className="p-3 bg-[#06090f] rounded-lg border border-amber-800/40 space-y-1">
                <span className="text-amber-400 font-bold block">31 to 45 Days Lag</span>
                <span className="text-white font-bold text-xs">-16 pt penalty</span>
                <p className="text-[11px] text-slate-400 font-sans">Aging signal; price may have priced in the move.</p>
              </div>
              <div className="p-3 bg-[#06090f] rounded-lg border border-rose-800/40 space-y-1">
                <span className="text-rose-400 font-bold block">&gt; 45 Days (Late)</span>
                <span className="text-white font-bold text-xs">-32 pt penalty</span>
                <p className="text-[11px] text-slate-400 font-sans">Late filing. High risk of chasing stale information.</p>
              </div>
            </div>
            <p className="text-slate-400 font-sans text-xs pt-1">
              <strong>Legislative Alignment:</strong> Filings receive positive score weighting (+16 to +32 points) when the lawmaker sits on a congressional committee with direct regulatory or budgetary oversight over the company&apos;s primary sector.
            </p>
          </div>
        </div>
      </section>

      {/* MODULE 5: USING ARX TERMINAL IN PRACTICE */}
      <section id="module-using-arx" className="space-y-6">
        <div className="flex items-center space-x-2 border-b border-[#243044] pb-2">
          <Sliders className="w-5 h-5 text-cyan-400 shrink-0" />
          <h2 className="text-lg sm:text-xl font-bold text-white tracking-tight font-mono">
            Module 5: Confluence Scoring, Risk Gates &amp; VaR
          </h2>
        </div>

        <div className="space-y-4">
          <h3 className="text-base font-bold text-white font-mono">
            5.1 Interpreting the Multi-Pillar Confluence Score (0–100)
          </h3>
          <p className="text-sm leading-relaxed">
            The Confluence Score is a composite conviction metric synthesizing four independent analytical pillars: Technical Structure, Fundamental Health, Smart Money Forensics, and Macro Credit Regimes.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
            <div className="p-3.5 rounded-xl bg-[#111722] border border-rose-900/40 space-y-1">
              <strong className="text-rose-400 block font-bold">0 to 48: Sub-Threshold</strong>
              <p className="text-slate-400 font-sans text-xs">
                Low analytical conviction or failing risk bounds. Rendered in defensive styling. Never actionable for capital commitment.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-[#111722] border border-cyan-900/40 space-y-1">
              <strong className="text-cyan-400 block font-bold">49 to 79: Analytical Focus</strong>
              <p className="text-slate-400 font-sans text-xs">
                Solid structural alignment with at least two confirming pillars. Suitable for watchlist monitoring and execution preparation.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-[#111722] border border-emerald-900/40 space-y-1">
              <strong className="text-emerald-400 block font-bold">80 to 100: High Conviction</strong>
              <p className="text-slate-400 font-sans text-xs">
                Exceptional multi-stream alignment across technical Stage 2 base, pristine F-Score, insider backing, and supportive macro liquidity.
              </p>
            </div>
          </div>

          <h3 className="text-base font-bold text-white font-mono pt-4">
            5.2 The 5-Point Pre-Flight Execution Gate
          </h3>
          <p className="text-sm leading-relaxed">
            Before any trade reaches execution clearance in the Trade Plan hub (`/setups`), it must satisfy a non-negotiable five-point risk gate:
          </p>

          <div className="bg-[#0b1019] p-4 sm:p-5 rounded-xl border border-[#1e293b] space-y-2.5 text-xs font-mono">
            <div className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-white">Gate 1: Trend Alignment</strong>
                <p className="text-slate-400 font-sans">Price must be holding above the rising 20-day EMA and 50-day moving average.</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-white">Gate 2: Volatility Contraction (VCP)</strong>
                <p className="text-slate-400 font-sans">Defined multi-wave contraction base with volume dry-up on the final consolidation.</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-white">Gate 3: Institutional Backing</strong>
                <p className="text-slate-400 font-sans">Confirmed positive smart money accumulation without heavy insider distribution.</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-white">Gate 4: Binary Catalyst Buffer</strong>
                <p className="text-slate-400 font-sans">Strict 48-hour buffer: zero impending earnings releases or major FDA announcements.</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-white">Gate 5: Macro Volatility Floor</strong>
                <p className="text-slate-400 font-sans">CBOE Volatility Index (VIX) must be strictly below 26.0. Systemic market fear blocks execution.</p>
              </div>
            </div>
          </div>

          <h3 className="text-base font-bold text-white font-mono pt-4">
            5.3 Cornish-Fisher Modified VaR (M-VaR)
          </h3>
          <p className="text-sm leading-relaxed">
            Standard Value-at-Risk assumes a Gaussian normal distribution, which severely understates tail-crash risk during market selloffs. ARX computes 95% and 99% Cornish-Fisher Modified VaR by incorporating sample skewness ($S$) and excess kurtosis ($K$).
          </p>
          <div className="bg-[#0b1019] p-4 rounded-xl border border-[#1e293b] font-mono text-xs text-cyan-300 overflow-x-auto">
            <code>z_cf = z_alpha + (z_alpha^2 - 1)*S / 6 + (z_alpha^3 - 3*z_alpha)*K / 24 - (2*z_alpha^3 - 5*z_alpha)*S^2 / 36</code>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed font-sans">
            A 95% 1-day VaR of -\$1,200 indicates that under historical statistical conditions, daily losses are expected to exceed \$1,200 on only 5% of trading days (1 day in 20). It does not represent a guaranteed maximum loss ceiling.
          </p>
        </div>
      </section>

      {/* Footer Navigation */}
      <footer className="border-t border-[#243044] pt-6 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
        <Link
          href="/"
          className="px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl shadow-sm transition-transform active:scale-95 cursor-pointer flex items-center gap-1.5"
        >
          <span>← Return to ARX Terminal</span>
        </Link>
        <Link
          href="/glossary/"
          className="px-4 py-2.5 bg-[#111722] hover:bg-[#162030] text-cyan-400 border border-[#243044] hover:border-cyan-700 font-bold rounded-xl transition-colors flex items-center gap-1.5"
        >
          <span>Browse 10 Quantitative Terms →</span>
        </Link>
      </footer>
    </main>
  );
}
