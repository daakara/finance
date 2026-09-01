import Link from "next/link";

export default function SmartMoneySsrShell() {
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
                  ARX <span className="text-purple-400 font-sans font-light text-xs tracking-normal">SMART MONEY</span>
                </span>
                <span className="text-[9px] text-slate-400 tracking-widest font-mono uppercase">
                  CONGRESSIONAL STOCK ACT & INSIDER RADAR
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
              <Link href="/smart-money" className="px-3 py-1.5 rounded-lg bg-purple-950/80 text-purple-300 border border-purple-800">
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

          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full bg-purple-950/60 border border-purple-800/80 text-[11px] font-mono text-purple-300">
              ⚡ Live STOCK Act Feed Active
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Shell */}
      <main className="flex-1 max-w-[1750px] w-full mx-auto p-4 sm:p-6 space-y-6">
        <div className="bg-[#0b0f19] border border-purple-900/40 rounded-xl p-5 shadow-xl">
          <h1 className="text-xl sm:text-2xl font-black text-white font-mono tracking-tight flex items-center gap-2">
            <span>🏛️</span>
            <span>Institutional Smart Money & Congressional STOCK Act Disclosures</span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl leading-relaxed">
            Real-time tracking of Capitol Hill transactions (Nancy Pelosi, Dan Crenshaw, Tommy Tuberville), SEC Form 4 insider whale accumulation, and legislative committee subsidy flows.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
            <div className="bg-[#070a12] p-4 rounded-lg border border-[#1a2334]">
              <span className="text-xs text-slate-400 font-mono block mb-1">MONITORED POLITICIANS</span>
              <span className="text-2xl font-black text-purple-400 font-mono">535</span>
              <span className="text-[11px] text-slate-500 block mt-0.5">House & Senate STOCK Act Filers</span>
            </div>
            <div className="bg-[#070a12] p-4 rounded-lg border border-[#1a2334]">
              <span className="text-xs text-slate-400 font-mono block mb-1">AVERAGE FILING LAG</span>
              <span className="text-2xl font-black text-cyan-400 font-mono">28 Days</span>
              <span className="text-[11px] text-slate-500 block mt-0.5">Statutory 45-day transparency window</span>
            </div>
            <div className="bg-[#070a12] p-4 rounded-lg border border-[#1a2334]">
              <span className="text-xs text-slate-400 font-mono block mb-1">ACTIVE SIGNALS</span>
              <span className="text-2xl font-black text-emerald-400 font-mono">100% Verified</span>
              <span className="text-[11px] text-slate-500 block mt-0.5">Cross-referenced with SEC EDGAR</span>
            </div>
          </div>
        </div>

        {/* Featured Politicians Links */}
        <div className="p-4 rounded-xl bg-[#090d16] border border-[#162030] text-xs font-mono flex flex-wrap items-center gap-3">
          <span className="text-slate-400 font-bold uppercase">Key Politician Hubs:</span>
          <Link href="/politician/nancy-pelosi" className="text-purple-300 hover:underline">Nancy Pelosi (NVDA/MSFT/AAPL)</Link>
          <span>•</span>
          <Link href="/politician/dan-crenshaw" className="text-purple-300 hover:underline">Dan Crenshaw (Defense/Energy)</Link>
          <span>•</span>
          <Link href="/politician/tommy-tuberville" className="text-purple-300 hover:underline">Tommy Tuberville (Agriculture/Commodities)</Link>
          <span>•</span>
          <Link href="/smart-money/late-filers" className="text-rose-400 hover:underline">Late-Filing Watchlist ⚠️</Link>
        </div>
      </main>
    </div>
  );
}
