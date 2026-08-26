"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import Navbar from "../../components/Navbar";

interface CompetitorAsset {
  symbol: string;
  name: string;
  category: string;
  
  // Long-Term Fundamental Lens
  marketCap: string;
  peRatio: string;
  pegRatio: string;
  roic: string;
  grossMargin: string;
  piotroski: number;
  keyCatalyst: string;
  trialEfficacy: string;
  primaryRisk: string;
  longTermVerdict: string;

  // Day Trader Momentum & Volatility Lens
  atr14: string;
  rvol: string;
  intradayBeta: string;
  liquidityTier: string;
  dayTraderSetup: string;
  bestTradingWindow: string;
  dayTradeVerdict: string;
}

const ASSET_DATABASE: Record<string, CompetitorAsset> = {
  NVO: {
    symbol: "NVO",
    name: "Novo Nordisk A/S",
    category: "Global Diabetes & Obesity Leader",
    marketCap: "$615 Billion",
    peRatio: "38.2x",
    pegRatio: "1.25",
    roic: "62.4%",
    grossMargin: "84.5%",
    piotroski: 9,
    keyCatalyst: "Amycretin Phase 2/3 (Oral GLP-1/Amylin Pill with 13.1% 12-wk weight loss)",
    trialEfficacy: "High convenience oral pill bypassing cold-chain supply chain bottlenecks",
    primaryRisk: "Capacity manufacturing constraints & compounded semaglutide litigation",
    longTermVerdict: "Elite ROIC & Margin Moat; Massive scaling if oral formulation succeeds",
    atr14: "$3.85",
    rvol: "1.9x",
    intradayBeta: "0.88 (Defensive / Steady)",
    liquidityTier: "High ($1.2B Daily Vol)",
    dayTraderSetup: "News-driven catalyst reactions & morning gap-fill scalps near 5m VWAP.",
    bestTradingWindow: "9:30 AM - 11:00 AM EST (Morning European Session overlap)",
    dayTradeVerdict: "Ideal for low-slippage mean reversion and pharmaceutical news breakouts",
  },
  LLY: {
    symbol: "LLY",
    name: "Eli Lilly & Company",
    category: "Global Oncology & Incretin Leader",
    marketCap: "$870 Billion",
    peRatio: "62.5x",
    pegRatio: "1.45",
    roic: "34.8%",
    grossMargin: "79.2%",
    piotroski: 8,
    keyCatalyst: "Retatrutide Phase 3 (Triple GGG Agonist with 24.2% 48-wk weight loss)",
    trialEfficacy: "Industry-leading clinical absolute weight reduction and liver fat clearance",
    primaryRisk: "Elevated valuation multiple (62x P/E) requiring flawless execution",
    longTermVerdict: "Clinical efficacy champion; Leading highest-tier weight reduction trials",
    atr14: "$18.40",
    rvol: "2.4x",
    intradayBeta: "1.35 (High-Beta Momentum)",
    liquidityTier: "Ultra-High ($3.5B Daily Vol)",
    dayTraderSetup: "Large-range trend continuation runner with wide multi-point afternoon trend expansions.",
    bestTradingWindow: "10:00 AM - 12:00 PM EST & 3:00 PM - 4:00 PM EST",
    dayTradeVerdict: "High-octane trend runner favoring momentum breakout traders",
  },
  SPY: {
    symbol: "SPY",
    name: "SPDR S&P 500 ETF Trust",
    category: "Broad Market Benchmark",
    marketCap: "$580 Billion AUM",
    peRatio: "26.4x",
    pegRatio: "1.80",
    roic: "18.5%",
    grossMargin: "N/A (Index)",
    piotroski: 8,
    keyCatalyst: "US Economic Soft Landing & Federal Reserve Interest Rate Easing Cycles",
    trialEfficacy: "Broad diversification across 500 market leaders",
    primaryRisk: "Macro recession or systemic credit spread blowouts",
    longTermVerdict: "Core wealth compounding foundation with maximum market diversification",
    atr14: "$5.60",
    rvol: "1.0x",
    intradayBeta: "1.00 (Market Benchmark)",
    liquidityTier: "Maximum World Liquidity ($35B+ Daily)",
    dayTraderSetup: "Tightest penny bid-ask spreads. Perfect for 0DTE options hedging and algorithmic mean-reversion.",
    bestTradingWindow: "All-Day Active Liquidity (9:30 AM - 4:00 PM EST)",
    dayTradeVerdict: "Zero-slippage benchmark for high-frequency scalping and options execution",
  },
  QQQ: {
    symbol: "QQQ",
    name: "Invesco QQQ Trust (Nasdaq-100)",
    category: "High-Growth Tech Benchmark",
    marketCap: "$290 Billion AUM",
    peRatio: "31.2x",
    pegRatio: "1.35",
    roic: "28.2%",
    grossMargin: "N/A (Index)",
    piotroski: 9,
    keyCatalyst: "Enterprise Generative AI Monetization & Hyperscaler Capex Expansion",
    trialEfficacy: "Concentrated secular tech dominance (Apple, Microsoft, Nvidia)",
    primaryRisk: "Multiple contraction during sudden interest rate surges",
    longTermVerdict: "High-beta growth engine capturing secular technology expansion",
    atr14: "$8.90",
    rvol: "1.4x",
    intradayBeta: "1.28 (Tech-Weighted High Beta)",
    liquidityTier: "Ultra-High ($18B+ Daily)",
    dayTraderSetup: "Strong directional momentum swings following semiconductor & cloud earnings releases.",
    bestTradingWindow: "9:30 AM - 11:30 AM EST (Tech Opening Range Volatility)",
    dayTradeVerdict: "Superior intraday range expansion for directional momentum day trading",
  },
  NVDA: {
    symbol: "NVDA",
    name: "NVIDIA Corporation",
    category: "Accelerated Computing & AI GPU Monopolist",
    marketCap: "$3.2 Trillion",
    peRatio: "44.5x",
    pegRatio: "0.95",
    roic: "85.2%",
    grossMargin: "75.4%",
    piotroski: 9,
    keyCatalyst: "Blackwell & Rubin GPU Architecture Multi-Gigawatt Data Center rollouts",
    trialEfficacy: "Unassailable CUDA software moat and NVLink cluster interconnects",
    primaryRisk: "Customer concentration (Hyperscalers) and geopolitical export bans",
    longTermVerdict: "Generative AI foundational hardware standard with extreme pricing power",
    atr14: "$6.20",
    rvol: "2.8x",
    intradayBeta: "1.85 (Ultra-High Beta)",
    liquidityTier: "Maximum World Stock Volume ($40B+ Daily)",
    dayTraderSetup: "World's leading intraday momentum vehicle. Extreme volume reactions at key round numbers.",
    bestTradingWindow: "9:30 AM - 11:30 AM & 3:00 PM - 4:00 PM EST",
    dayTradeVerdict: "Premier intraday momentum stock with pristine trend clarity and liquidity",
  },
  AAPL: {
    symbol: "AAPL",
    name: "Apple Inc.",
    category: "Consumer Hardware & Services Ecosystem",
    marketCap: "$3.4 Trillion",
    peRatio: "32.8x",
    pegRatio: "2.10",
    roic: "54.1%",
    grossMargin: "46.2%",
    piotroski: 8,
    keyCatalyst: "Apple Intelligence on-device GenAI supercycle and Services segment expansion",
    trialEfficacy: "2.2+ Billion active device installed base with unmatched consumer retention",
    primaryRisk: "Regulatory App Store scrutiny (DOJ/EU) and China hardware sales slowdown",
    longTermVerdict: "Cash machine fortress balance sheet with massive capital returns via buybacks",
    atr14: "$3.20",
    rvol: "1.2x",
    intradayBeta: "0.92 (Defensive Tech)",
    liquidityTier: "High ($8B+ Daily)",
    dayTraderSetup: "Low-volatility mean reversion. Solid liquidity with predictable price levels.",
    bestTradingWindow: "9:30 AM - 11:00 AM EST",
    dayTradeVerdict: "Steady large-cap institutional anchor with tight spreads and minimal chop",
  },
  TSLA: {
    symbol: "TSLA",
    name: "Tesla Inc.",
    category: "Autonomous AI, Robotics & EV Energy",
    marketCap: "$820 Billion",
    peRatio: "78.4x",
    pegRatio: "3.20",
    roic: "16.4%",
    grossMargin: "18.2%",
    piotroski: 7,
    keyCatalyst: "Full Self-Driving (FSD) Unsupervised Robotaxi commercial deployment & Optimus humanoid robot",
    trialEfficacy: "Billion-mile real-world vision fleet data moat and Megapack energy storage growth",
    primaryRisk: "Global EV price war compressing gross auto margins",
    longTermVerdict: "High-optionality moonshot on physical AI robotics and autonomous transport",
    atr14: "$12.80",
    rvol: "3.2x",
    intradayBeta: "2.10 (Extreme Retail Volatility)",
    liquidityTier: "Ultra-High ($15B+ Daily)",
    dayTraderSetup: "Wild intraday range expansions. Excellent candidate for momentum trend scalp continuation.",
    bestTradingWindow: "9:30 AM - 11:30 AM EST",
    dayTradeVerdict: "Elite day-trading vehicle for aggressive momentum and breakout scalping",
  },
  PLTR: {
    symbol: "PLTR",
    name: "Palantir Technologies",
    category: "Enterprise AI & Defense Ontology Platform",
    marketCap: "$110 Billion",
    peRatio: "95.0x",
    pegRatio: "2.80",
    roic: "22.5%",
    grossMargin: "81.4%",
    piotroski: 8,
    keyCatalyst: "Artificial Intelligence Platform (AIP) Bootcamp commercial customer conversion",
    trialEfficacy: "Unbreakable US Defense & Intelligence contracts expanding into commercial enterprise ontology",
    primaryRisk: "High valuation multiple leaving little room for earnings misses",
    longTermVerdict: "Pure-play enterprise GenAI operating system with accelerating operating leverage",
    atr14: "$4.10",
    rvol: "2.6x",
    intradayBeta: "1.90 (High Momentum)",
    liquidityTier: "High ($3.8B Daily)",
    dayTraderSetup: "Fast-moving retail & institutional tape. Aggressive gap-up breakouts on DoD contract wins.",
    bestTradingWindow: "9:30 AM - 11:00 AM EST",
    dayTradeVerdict: "Premier enterprise software runner with clean intraday trend momentum",
  },
  ELF: {
    symbol: "ELF",
    name: "e.l.f. Beauty Inc.",
    category: "High-Growth Small/Mid-Cap Disruptor",
    marketCap: "$8.5 Billion",
    peRatio: "34.0x",
    pegRatio: "0.84",
    roic: "28.4%",
    grossMargin: "71.2%",
    piotroski: 9,
    keyCatalyst: "International retail expansion (UK/Europe) & skincare product portfolio scaling",
    trialEfficacy: "Digitally-native viral marketing with industry-leading inventory turns",
    primaryRisk: "Consumer discretionary spending slowdown",
    longTermVerdict: "Peter Lynch GARP Compounder taking rapid market share with clean balance sheet",
    atr14: "$3.45",
    rvol: "2.4x",
    intradayBeta: "1.40 (Growth Mid-Cap)",
    liquidityTier: "Medium ($400M Daily)",
    dayTraderSetup: "Opening range breakout candidate. Fast morning volume expansions off 5m VWAP.",
    bestTradingWindow: "9:30 AM - 11:00 AM EST",
    dayTradeVerdict: "High-beta small/mid-cap runner with explosive morning momentum",
  },
  DUOL: {
    symbol: "DUOL",
    name: "Duolingo Inc.",
    category: "EdTech & GenAI Mobile Platform",
    marketCap: "$9.2 Billion",
    peRatio: "48.2x",
    pegRatio: "1.05",
    roic: "26.1%",
    grossMargin: "73.4%",
    piotroski: 9,
    keyCatalyst: "Duolingo Max GenAI monetization & enterprise English test global adoption",
    trialEfficacy: "Viral organic user acquisition and accelerating operating margins",
    primaryRisk: "Freemium user monetization saturation",
    longTermVerdict: "Disruptive Rule Breaker with sticky consumer moat and high recurring margins",
    atr14: "$6.80",
    rvol: "3.1x",
    intradayBeta: "1.65 (High Beta)",
    liquidityTier: "Medium ($350M Daily)",
    dayTraderSetup: "High-octane breakout scalps. Wide daily dollar range creates generous profit targets.",
    bestTradingWindow: "9:30 AM - 11:30 AM EST",
    dayTradeVerdict: "Excellent mid-cap momentum stock with wide intraday ranges and clean trend days",
  }
};

const SEO_CURATED_PRESETS = [
  { id: "nvo-vs-lly", label: "💊 Novo Nordisk (NVO) vs. Eli Lilly (LLY)", a: "NVO", b: "LLY" },
  { id: "spy-vs-qqq", label: "📊 S&P 500 (SPY) vs. Nasdaq-100 (QQQ)", a: "SPY", b: "QQQ" },
  { id: "nvda-vs-tsla", label: "⚡ NVIDIA (NVDA) vs. Tesla (TSLA)", a: "NVDA", b: "TSLA" },
  { id: "elf-vs-duol", label: "💎 e.l.f. Beauty (ELF) vs. Duolingo (DUOL)", a: "ELF", b: "DUOL" },
];

function CompareContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const paramA = searchParams.get("a")?.toUpperCase();
  const paramB = searchParams.get("b")?.toUpperCase();

  const [symbolA, setSymbolA] = useState<string>(paramA && ASSET_DATABASE[paramA] ? paramA : "NVO");
  const [symbolB, setSymbolB] = useState<string>(paramB && ASSET_DATABASE[paramB] ? paramB : "LLY");
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");

  // Synchronize URL parameters if changed
  useEffect(() => {
    if (paramA && ASSET_DATABASE[paramA] && paramA !== symbolA) setSymbolA(paramA);
    if (paramB && ASSET_DATABASE[paramB] && paramB !== symbolB) setSymbolB(paramB);
  }, [paramA, paramB]);

  useEffect(() => {
    const saved = localStorage.getItem("FINANCE_USER_ROLE");
    if (saved === "DAY_TRADER" || saved === "LONG_TERM") {
      setActiveRole(saved);
    }
  }, []);

  const handleRoleToggle = (role: "DAY_TRADER" | "LONG_TERM") => {
    setActiveRole(role);
    localStorage.setItem("FINANCE_USER_ROLE", role);
  };

  const handlePresetSelect = (a: string, b: string) => {
    setSymbolA(a);
    setSymbolB(b);
    router.push(`/compare?a=${a}&b=${b}`);
  };

  const handleSymbolChange = (side: "A" | "B", newSym: string) => {
    if (side === "A") {
      setSymbolA(newSym);
      router.push(`/compare?a=${newSym}&b=${symbolB}`);
    } else {
      setSymbolB(newSym);
      router.push(`/compare?a=${symbolA}&b=${newSym}`);
    }
  };

  const isDayTrader = activeRole === "DAY_TRADER";
  const assetA = ASSET_DATABASE[symbolA] || ASSET_DATABASE["NVO"];
  const assetB = ASSET_DATABASE[symbolB] || ASSET_DATABASE["LLY"];

  const allAvailableSymbols = Object.keys(ASSET_DATABASE);

  return (
    <main id="main-content" role="main" className="min-h-screen bg-[#070a11] text-slate-100 font-mono flex flex-col pb-20 sm:pb-8">
      <Navbar userRole={activeRole} onRoleChange={handleRoleToggle} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1">
        {/* Header with Dual-Horizon Lens Toggle */}
        <div className="mb-6 border-b border-[#1b2434] pb-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl">⚔️</span>
                <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                  Head-to-Head Asset & Pipeline Comparison
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
                {isDayTrader
                  ? "⚡ Day Trader Lens Active: Comparing 14-day ATR volatility ($), relative volume (RVOL), intraday beta, and opening range setups."
                  : "🏛️ Long-Term Compounder Lens Active: Comparing multi-year ROIC, gross margins, clinical trial pipelines, and fundamental valuations."}
              </p>
            </div>

            {/* Dual-Horizon Lens Switcher */}
            <div className="flex items-center space-x-2 bg-[#0d131f] p-1.5 rounded-xl border border-[#243044]">
              <span className="text-[11px] text-slate-400 font-bold px-2 hidden sm:inline">Comparison Lens:</span>
              <button
                onClick={() => handleRoleToggle("DAY_TRADER")}
                className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                  isDayTrader
                    ? "bg-amber-500 text-slate-950 shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                ⚡ Day Trader (ATR/Vol)
              </button>
              <button
                onClick={() => handleRoleToggle("LONG_TERM")}
                className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                  !isDayTrader
                    ? "bg-cyan-500 text-slate-950 shadow-md font-extrabold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                🏛️ Long-Term (ROIC/Trials)
              </button>
            </div>
          </div>
        </div>

        {/* SEO Curated Presets Bar */}
        <div className="mb-4">
          <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block mb-2">
            ⭐ Curated SEO Battleground Matchups:
          </span>
          <div className="flex flex-wrap items-center gap-2">
            {SEO_CURATED_PRESETS.map((preset) => {
              const isSelected = symbolA === preset.a && symbolB === preset.b;
              return (
                <button
                  key={preset.id}
                  onClick={() => handlePresetSelect(preset.a, preset.b)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all border active:scale-[0.96] ${
                    isSelected
                      ? isDayTrader
                        ? "bg-amber-500 text-slate-950 border-amber-400 font-extrabold shadow-lg"
                        : "bg-cyan-500 text-slate-950 border-cyan-400 font-extrabold shadow-lg"
                      : "bg-[#0d131f] text-slate-400 border-[#243044] hover:text-slate-200"
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Dynamic Asset Selectors (Compare ANY 2 assets) */}
        <div className="bg-[#0e131d] border border-[#1b2434] rounded-xl p-3.5 mb-6 flex flex-wrap items-center justify-between gap-3 shadow-lg">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-slate-300">Custom Matchup:</span>
            <select
              value={symbolA}
              onChange={(e) => handleSymbolChange("A", e.target.value)}
              className="bg-[#111722] border border-cyan-500/50 rounded-lg px-2.5 py-1 text-xs font-bold text-cyan-300 focus:outline-none focus:ring-1 focus:ring-cyan-400"
            >
              {allAvailableSymbols.map((sym) => (
                <option key={sym} value={sym}>
                  {sym} - {ASSET_DATABASE[sym].name}
                </option>
              ))}
            </select>
            <span className="text-xs font-black text-slate-500">VS</span>
            <select
              value={symbolB}
              onChange={(e) => handleSymbolChange("B", e.target.value)}
              className="bg-[#111722] border border-cyan-500/50 rounded-lg px-2.5 py-1 text-xs font-bold text-cyan-300 focus:outline-none focus:ring-1 focus:ring-cyan-400"
            >
              {allAvailableSymbols.map((sym) => (
                <option key={sym} value={sym}>
                  {sym} - {ASSET_DATABASE[sym].name}
                </option>
              ))}
            </select>
          </div>
          <span className="text-[11px] text-slate-400">
            Comparing <strong className="text-cyan-400">{symbolA}</strong> against <strong className="text-cyan-400">{symbolB}</strong>
          </span>
        </div>

        {/* Side-by-Side Comparison Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[assetA, assetB].map((asset) => (
            <div
              key={asset.symbol}
              className={`bg-[#0e131d] border rounded-2xl p-5 shadow-2xl space-y-4 flex flex-col justify-between transition-all ${
                isDayTrader ? "border-amber-900/40 hover:border-amber-500/50" : "border-[#1b2434] hover:border-cyan-500/40"
              }`}
            >
              <div>
                {/* Header */}
                <div className="flex items-start justify-between border-b border-[#1b2434] pb-4">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xl font-black text-white">{asset.symbol}</span>
                      <span className={`text-[11px] font-bold px-2 py-0.5 rounded border ${
                        isDayTrader
                          ? "bg-amber-950/80 border-amber-800 text-amber-300"
                          : "bg-cyan-950 border border-cyan-800 text-cyan-400"
                      }`}>
                        {asset.category}
                      </span>
                    </div>
                    <h2 className="text-sm text-slate-300 font-bold mt-1">{asset.name}</h2>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">PIOTROSKI</span>
                    <span className="text-base font-black text-emerald-400">{asset.piotroski}/9</span>
                  </div>
                </div>

                {/* Adaptive Metrics Table (Swaps between Day Trader and Long-Term parameters) */}
                {isDayTrader ? (
                  <div className="grid grid-cols-3 gap-2 my-4 bg-[#130f08] p-3 rounded-xl border border-amber-900/40 text-center text-xs tabular-nums">
                    <div>
                      <span className="text-[10px] text-amber-500/80 block">ATR (14D RANGE)</span>
                      <span className="font-bold text-amber-300">{asset.atr14}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-amber-500/80 block">REL VOL (RVOL)</span>
                      <span className="font-bold text-emerald-400">{asset.rvol}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-amber-500/80 block">INTRADAY BETA</span>
                      <span className="font-bold text-cyan-300">{asset.intradayBeta.split(" ")[0]}</span>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-3 gap-2 my-4 bg-[#080c14] p-3 rounded-xl border border-[#182232] text-center text-xs tabular-nums">
                    <div>
                      <span className="text-[10px] text-slate-500 block">MARKET CAP</span>
                      <span className="font-bold text-slate-200">{asset.marketCap}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block">P/E RATIO</span>
                      <span className="font-bold text-purple-300">{asset.peRatio}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-500 block">ROIC</span>
                      <span className="font-bold text-emerald-400">{asset.roic}</span>
                    </div>
                  </div>
                )}

                {/* Adaptive Narrative / Setup Highlight */}
                {isDayTrader ? (
                  <div className="space-y-3 text-xs">
                    <div className="p-3 bg-[#18120a] rounded-xl border border-amber-900/60">
                      <span className="text-[10px] text-amber-400 font-black block uppercase tracking-wider">⚡ Intraday Scalp & Execution Setup</span>
                      <p className="text-slate-200 mt-1 font-semibold text-[11px] leading-snug">{asset.dayTraderSetup}</p>
                    </div>

                    <div>
                      <span className="text-[10px] text-amber-500/80 font-black block uppercase tracking-wider">⏰ Optimal Trading Window</span>
                      <p className="text-slate-300 text-[11px] mt-0.5">{asset.bestTradingWindow}</p>
                    </div>

                    <div>
                      <span className="text-[10px] text-emerald-500 font-black block uppercase tracking-wider">💡 Day Trade Verdict</span>
                      <p className="text-slate-300 text-[11px] mt-0.5">{asset.dayTradeVerdict}</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3 text-xs">
                    <div className="p-3 bg-[#111722] rounded-xl border border-cyan-950">
                      <span className="text-[10px] text-cyan-400 font-black block uppercase tracking-wider">🔬 Primary Clinical Trial / Catalyst</span>
                      <p className="text-slate-200 mt-1 font-semibold text-[11px] leading-snug">{asset.keyCatalyst}</p>
                      <p className="text-slate-400 text-[10px] mt-1">{asset.trialEfficacy}</p>
                    </div>

                    <div>
                      <span className="text-[10px] text-slate-500 font-black block uppercase tracking-wider">⚠️ Key Operational Risk</span>
                      <p className="text-slate-400 text-[11px] mt-0.5">{asset.primaryRisk}</p>
                    </div>

                    <div>
                      <span className="text-[10px] text-emerald-500 font-black block uppercase tracking-wider">💡 Fundamental Verdict</span>
                      <p className="text-slate-300 text-[11px] mt-0.5">{asset.longTermVerdict}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Button that Preserves Active Trading Horizon into Terminal */}
              <div className="pt-4 border-t border-[#1b2434]">
                <Link
                  href={`/?symbol=${asset.symbol}`}
                  className={`w-full py-2 rounded-xl text-xs font-bold transition-all flex items-center justify-center space-x-1 active:scale-[0.98] border ${
                    isDayTrader
                      ? "bg-amber-600/20 hover:bg-amber-500 hover:text-slate-950 border-amber-500/50 text-amber-300"
                      : "bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/50 text-cyan-300"
                  }`}
                >
                  <span>{isDayTrader ? `Trade ${asset.symbol} in Terminal (5m) →` : `Analyze ${asset.symbol} in Terminal (1D) →`}</span>
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#070a11] text-slate-100 flex items-center justify-center font-mono">Loading Comparison Engine...</div>}>
      <CompareContent />
    </Suspense>
  );
}
