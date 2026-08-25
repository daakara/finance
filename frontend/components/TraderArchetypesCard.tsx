"use client";

import { TraderArchetypeConsensus } from "../lib/api";

interface TraderArchetypesCardProps {
  symbol: string;
  traderArchetypes?: TraderArchetypeConsensus;
}

export default function TraderArchetypesCard({ symbol, traderArchetypes }: TraderArchetypesCardProps) {
  const data = traderArchetypes || {
    consensusScore: 82,
    verdict: "Strong Smart-Money Accumulation",
    archetypes: [
      {
        name: "Warren Buffett (The Oracle)",
        archetype: "Defensive Quality & Wide Moats",
        alignmentScore: 88,
        status: "High Moat Alignment",
        thesis: "High free cash flow yield with durable competitive moat and disciplined capital allocation.",
        catalyst: "Resilient operating margins and consistent share repurchases across market cycles.",
      },
      {
        name: "Nancy Pelosi (The Capitol Whale)",
        archetype: "Legislative Catalysts & High-Conviction Tech",
        alignmentScore: 94,
        status: "Active Policy Tailwinds",
        thesis: "Strategic semiconductor, defense, and federal infrastructure policy subsidy beneficiary.",
        catalyst: "Federal digital transformation mandates and long-duration LEAPS call option flow.",
      },
      {
        name: "Stanley Druckenmiller (The Macro Sorcerer)",
        archetype: "Macro Liquidity & Trend Reflexivity",
        alignmentScore: 82,
        status: "Strong Macro Inflection",
        thesis: "Positive monetary easing backdrop with strong trend momentum above key moving averages.",
        catalyst: "Yield curve steepening and accelerating institutional accumulation.",
      },
      {
        name: "Jim Simons (The Medallion Quant)",
        archetype: "Statistical Arbitrage & Volatility Mean Reversion",
        alignmentScore: 76,
        status: "Statistically Favorable",
        thesis: "Superior Sortino ratio with bounded tail loss probability under non-normal distribution models.",
        catalyst: "Mathematical volatility compression and persistent factor momentum.",
      },
    ],
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-4">
      {/* Header with Consensus Titan Score */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-400 animate-pulse"></span>
            <h3 className="text-base font-bold text-slate-100 font-mono tracking-tight flex items-center gap-2">
              <span>???</span>
              <span>{symbol} Iconic Trader & Smart-Money Alignment</span>
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Multi-strategy evaluation across Buffett Moat, Congressional Policy Flow, Macro Liquidity, and Statistical Arbitrage models
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <div className="bg-purple-950/80 border border-purple-700/80 px-3 py-1 rounded-lg text-right font-mono">
            <span className="text-[10px] text-purple-300 block uppercase leading-none font-bold">Consensus</span>
            <span className="text-base font-bold text-purple-400">{data.consensusScore}%</span>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 font-mono">
            {data.verdict}
          </span>
        </div>
      </div>

      {/* 4 Iconic Trader Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data.archetypes.map((a) => {
          const isHigh = a.alignmentScore >= 80;
          const isMid = a.alignmentScore >= 65;

          return (
            <div
              key={a.name}
              className="bg-[#090d14] border border-[#243044] hover:border-[#364866] rounded-lg p-4 space-y-2.5 transition-all"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h4 className="text-sm font-bold font-mono text-slate-100">{a.name}</h4>
                  <span className="text-[10px] text-slate-400 font-mono block">{a.archetype}</span>
                </div>

                <div className="text-right shrink-0">
                  <span
                    className={`text-base font-bold font-mono ${
                      isHigh ? "text-emerald-400" : isMid ? "text-cyan-400" : "text-amber-400"
                    }`}
                  >
                    {a.alignmentScore}%
                  </span>
                  <span
                    className={`text-[9px] block font-mono px-1.5 py-0.2 rounded border mt-0.5 ${
                      isHigh
                        ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
                        : "bg-[#1b2434] text-slate-300 border-[#364866]"
                    }`}
                  >
                    {a.status}
                  </span>
                </div>
              </div>

              {/* Score Progress Bar */}
              <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    isHigh ? "bg-emerald-500" : isMid ? "bg-cyan-500" : "bg-amber-500"
                  }`}
                  style={{ width: `${a.alignmentScore}%` }}
                ></div>
              </div>

              {/* Thesis & Catalyst Snippet */}
              <div className="text-[11px] text-slate-300 font-mono space-y-1 pt-1 leading-relaxed">
                <p>
                  <span className="text-slate-400 font-semibold">Thesis: </span>
                  {a.thesis}
                </p>
                <div className="flex items-center gap-1.5 text-[10px] text-purple-300 bg-purple-950/40 px-2 py-1 rounded border border-purple-900/60">
                  <span className="text-purple-400 font-bold">Key Catalyst:</span>
                  <span className="truncate">{a.catalyst}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

