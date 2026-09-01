import Link from "next/link";

export default function CompareSsrShell() {
  return (
    <div className="min-h-screen bg-[#070a10] text-[#e2e8f0] flex flex-col font-sans">
      {/* Header Shell */}
      <header className="sticky top-0 z-40 bg-[#070a10]/95 backdrop-blur-md border-b border-[#1b2434] px-4 py-2.5">
        <div className="max-w-[1750px] mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 via-cyan-600 to-blue-700 flex items-center justify-center font-black text-black font-mono text-sm">
                ARX
              </div>
              <div className="flex flex-col">
                <span className="font-black tracking-wider text-white text-base font-mono">
                  ARX <span className="text-cyan-400 font-sans font-light text-xs tracking-normal">COMPARE</span>
                </span>
                <span className="text-[9px] text-slate-400 tracking-widest font-mono uppercase">
                  HEAD-TO-HEAD QUANTITATIVE SYNTHESIS
                </span>
              </div>
            </Link>

            <nav aria-label="Navigation" className="hidden md:flex items-center gap-1 text-xs font-semibold">
              <Link href="/" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Terminal
              </Link>
              <Link href="/screener" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Gems Screener
              </Link>
              <Link href="/smart-money" className="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition-colors">
                Smart Money
              </Link>
              <Link href="/compare" className="px-3 py-1.5 rounded-lg bg-cyan-950/80 text-cyan-300 border border-cyan-800">
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

          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/80 text-[11px] font-mono text-cyan-300">
              ⚡ Multi-Factor Comparator Active
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Shell */}
      <main className="flex-1 max-w-[1750px] w-full mx-auto p-4 sm:p-6 space-y-6">
        <div className="bg-[#0b0f19] border border-cyan-900/40 rounded-xl p-5 shadow-xl">
          <h1 className="text-xl sm:text-2xl font-black text-white font-mono tracking-tight flex items-center gap-2">
            <span>⚖️</span>
            <span>Head-to-Head Asset Comparison & Tactical Factor Allocation</span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl leading-relaxed">
            Direct multi-factor synthesis comparing Return on Invested Capital (ROIC), PEG ratio valuation, gross margins, and Cornish-Fisher downside risk.
          </p>

          {/* Popular Pairs Links */}
          <div className="pt-4 flex flex-wrap gap-2 text-xs font-mono">
            <span className="text-slate-400 font-bold uppercase py-1">Popular Duels:</span>
            <Link href="/compare/nvo-vs-lly" className="px-3 py-1 rounded-lg bg-[#0e1422] text-cyan-300 border border-cyan-800 hover:bg-cyan-950">
              NVO vs LLY (GLP-1 Duopoly)
            </Link>
            <Link href="/compare/nvda-vs-aapl" className="px-3 py-1 rounded-lg bg-[#0e1422] text-cyan-300 border border-cyan-800 hover:bg-cyan-950">
              NVDA vs AAPL (AI Compute vs Consumer Silicon)
            </Link>
            <Link href="/compare/spy-vs-qqq" className="px-3 py-1 rounded-lg bg-[#0e1422] text-cyan-300 border border-cyan-800 hover:bg-cyan-950">
              SPY vs QQQ (Macro Index vs Tech Growth)
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
