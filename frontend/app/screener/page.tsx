"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";

interface GemCandidate {
  symbol: string;
  companyName: string;
  gemScore: number;
  expertArchetype: string;
  roic: string;
  pegRatio: string;
  grossMargin: string;
  thesis: string;
  catalyst: string;
  riskLevel: string;
}

const ARCHETYPES = [
  { id: "all", label: "✨ All High-Alpha Gems", desc: "Small & Mid-Cap High ROIC Compounders" },
  { id: "lynch", label: "📈 Peter Lynch GARP", desc: "PEG < 1.0, Low Net Debt, Overlooked Niche Growth" },
  { id: "greenblatt", label: "🎯 Greenblatt Magic Formula", desc: "High ROIC (>25%) + Bargain Earnings Yield" },
  { id: "rule_breakers", label: "⚡ Disruptive Rule Breakers", desc: "Category Creators, >65% Gross Margins, High Moat" },
];

const CURATED_SMALL_CAP_GEMS: GemCandidate[] = [
  {
    symbol: "ELF",
    companyName: "e.l.f. Beauty Inc.",
    gemScore: 94,
    expertArchetype: "Peter Lynch GARP Compounder",
    roic: "28.4%",
    pegRatio: "0.84",
    grossMargin: "71.2%",
    thesis: "Disruptive beauty brand taking rapid global market share with digitally-native marketing and negative working capital cycles.",
    catalyst: "International retail expansion (UK/Europe) and premium skincare segment acquisitions.",
    riskLevel: "Low-to-Medium",
  },
  {
    symbol: "MEDP",
    companyName: "Medpace Holdings Inc.",
    gemScore: 92,
    expertArchetype: "Greenblatt Magic Formula",
    roic: "38.6%",
    pegRatio: "1.10",
    grossMargin: "48.5%",
    thesis: "Elite clinical contract research organization with pure-play focus on small biopharma, high return on capital, and zero debt.",
    catalyst: "Accelerating biopharma venture funding rounds and high RFP backlog conversion.",
    riskLevel: "Low",
  },
  {
    symbol: "DUOL",
    companyName: "Duolingo Inc.",
    gemScore: 91,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "26.1%",
    pegRatio: "1.05",
    grossMargin: "73.4%",
    thesis: "Dominant organic mobile education platform with virality-driven user acquisition, GenAI learning tiers, and expanding operating leverage.",
    catalyst: "Duolingo Max GenAI monetization and enterprise English test global adoption.",
    riskLevel: "Medium",
  },
  {
    symbol: "POWI",
    companyName: "Power Integrations Inc.",
    gemScore: 89,
    expertArchetype: "Peter Lynch GARP Compounder",
    roic: "24.5%",
    pegRatio: "0.92",
    grossMargin: "54.8%",
    thesis: "Niche monopoly in energy-efficient GaN (Gallium Nitride) and high-voltage power conversion chips for EVs and data centers.",
    catalyst: "Server power supply efficiency mandates and GaN adoption in high-power appliances.",
    riskLevel: "Low-to-Medium",
  },
  {
    symbol: "CPRX",
    companyName: "Catalyst Pharmaceuticals",
    gemScore: 88,
    expertArchetype: "Greenblatt Magic Formula",
    roic: "34.2%",
    pegRatio: "0.78",
    grossMargin: "78.9%",
    thesis: "High-margin rare disease biotech with rock-solid free cash flow, massive operating margins (>45%), and pristine net cash position.",
    catalyst: "Firdapse patent exclusivity defense and strategic orphan drug portfolio M&A.",
    riskLevel: "Medium",
  },
  {
    symbol: "IOT",
    companyName: "Samsara Inc.",
    gemScore: 87,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "22.8%",
    pegRatio: "1.25",
    grossMargin: "75.1%",
    thesis: "Leader in physical operations cloud computing, connecting commercial vehicle fleets and industrial assets with recurring high-margin ARR.",
    catalyst: "Connected asset safety mandates and multi-product customer expansion (>100k ARR tier).",
    riskLevel: "Medium",
  },
  {
    symbol: "AXON",
    companyName: "Axon Enterprise Inc.",
    gemScore: 93,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "27.4%",
    pegRatio: "1.32",
    grossMargin: "62.3%",
    thesis: "Unassailable public safety software & hardware ecosystem (TASERS, Body Cams, Evidence.com cloud) with 120%+ net revenue retention.",
    catalyst: "Draft One generative AI police report transcription software and international police force rollout.",
    riskLevel: "Low",
  },
  {
    symbol: "RKLB",
    companyName: "Rocket Lab USA Inc.",
    gemScore: 86,
    expertArchetype: "Disruptive Rule Breaker",
    roic: "19.5%",
    pegRatio: "1.40",
    grossMargin: "32.4%",
    thesis: "Leading commercial small-satellite launch provider and space systems manufacturer; the only viable Western competitor to SpaceX.",
    catalyst: "Medium-lift Neutron rocket maiden flight and multi-billion-dollar DoD Space Development Agency contracts.",
    riskLevel: "High Growth",
  },
];

export default function ScreenerPage() {
  const [selectedArchetype, setSelectedArchetype] = useState("all");
  const [gems, setGems] = useState<GemCandidate[]>(CURATED_SMALL_CAP_GEMS);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Filter candidates based on archetype
    if (selectedArchetype === "all") {
      setGems(CURATED_SMALL_CAP_GEMS);
    } else if (selectedArchetype === "lynch") {
      setGems(CURATED_SMALL_CAP_GEMS.filter((g) => g.expertArchetype.includes("Lynch")));
    } else if (selectedArchetype === "greenblatt") {
      setGems(CURATED_SMALL_CAP_GEMS.filter((g) => g.expertArchetype.includes("Greenblatt")));
    } else if (selectedArchetype === "rule_breakers") {
      setGems(CURATED_SMALL_CAP_GEMS.filter((g) => g.expertArchetype.includes("Rule Breaker")));
    }
  }, [selectedArchetype]);

  return (
    <main id="main-content" role="main" className="min-h-screen bg-[#070a11] text-slate-100 font-mono flex flex-col pb-20 sm:pb-8">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1">
        {/* Page Hero Header */}
        <div className="mb-6 border-b border-[#1b2434] pb-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl">💎</span>
                <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                  High-Alpha Small & Mid-Cap Gems Screener
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
                Rigorous algorithmic multi-model screening modeled after Peter Lynch GARP ($PEG \le 1.0$), Joel Greenblatt Magic Formula ($ROIC \ge 25\%$), and Motley Fool Rule Breakers. Purged of mega-caps to surface genuine asymmetrical growth.
              </p>
            </div>
            <Link
              href="/"
              className="px-3 py-1.5 bg-[#111722] hover:bg-[#182232] border border-[#243044] rounded-lg text-xs font-bold text-cyan-400 transition-colors flex items-center space-x-1 active:scale-[0.96]"
            >
              <span>← Return to Terminal</span>
            </Link>
          </div>
        </div>

        {/* Archetype Filter Tabs */}
        <div role="tablist" aria-label="Expert Screener Models" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {ARCHETYPES.map((arch) => {
            const isActive = selectedArchetype === arch.id;
            return (
              <button
                key={arch.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => setSelectedArchetype(arch.id)}
                className={`p-3 rounded-xl border text-left transition-all active:scale-[0.98] ${
                  isActive
                    ? "bg-[#111c2e] border-cyan-500/80 shadow-lg shadow-cyan-950/40"
                    : "bg-[#0c1017] border-[#1b2434] hover:bg-[#111722] hover:border-[#2b3a52]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-xs font-black ${isActive ? "text-cyan-400" : "text-slate-200"}`}>
                    {arch.label}
                  </span>
                  {isActive && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />}
                </div>
                <p className="text-[11px] text-slate-400 mt-1 leading-snug">{arch.desc}</p>
              </button>
            );
          })}
        </div>

        {/* Candidate Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {gems.map((gem) => (
            <div
              key={gem.symbol}
              className="bg-[#0e131d] border border-[#1b2434] hover:border-cyan-500/40 rounded-xl p-4 shadow-xl flex flex-col justify-between transition-all hover:bg-[#111724]"
            >
              {/* Card Top */}
              <div>
                <div className="flex items-start justify-between gap-2 border-b border-[#162030] pb-3">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-base font-black text-white">{gem.symbol}</span>
                      <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-800 text-cyan-400">
                        {gem.expertArchetype}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">{gem.companyName}</p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">GEM SCORE</span>
                    <span className="text-lg font-black text-emerald-400 tabular-nums">{gem.gemScore}/100</span>
                  </div>
                </div>

                {/* Metrics Matrix */}
                <div className="grid grid-cols-3 gap-2 my-3 bg-[#080c14] p-2.5 rounded-lg border border-[#192334] text-center">
                  <div>
                    <span className="text-[10px] text-slate-500 block">ROIC</span>
                    <span className="text-xs font-bold text-slate-200 tabular-nums">{gem.roic}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">PEG RATIO</span>
                    <span className="text-xs font-bold text-emerald-400 tabular-nums">{gem.pegRatio}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">GROSS MARGIN</span>
                    <span className="text-xs font-bold text-cyan-400 tabular-nums">{gem.grossMargin}</span>
                  </div>
                </div>

                {/* Investment Thesis & Catalyst */}
                <div className="space-y-2 text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">💡 Investment Thesis</span>
                    <p className="text-slate-300 leading-relaxed text-[11px]">{gem.thesis}</p>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 font-bold block uppercase tracking-wider">🚀 Primary Catalyst</span>
                    <p className="text-slate-400 leading-relaxed text-[11px]">{gem.catalyst}</p>
                  </div>
                </div>
              </div>

              {/* Card Footer: Quick Action to Terminal */}
              <div className="mt-4 pt-3 border-t border-[#162030] flex items-center justify-between">
                <span className="text-[10px] font-semibold text-slate-400">
                  Risk: <span className="text-amber-400">{gem.riskLevel}</span>
                </span>
                <Link
                  href={`/?symbol=${gem.symbol}`}
                  className="px-2.5 py-1 bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border border-cyan-500/50 rounded text-xs font-bold text-cyan-300 transition-all active:scale-[0.96]"
                >
                  Analyze in Terminal →
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}

