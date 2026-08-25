"use client";

import { TraderArchetypeConsensus } from "../lib/api";

interface TraderArchetypesCardProps {
  symbol: string;
  traderArchetypes?: TraderArchetypeConsensus;
}

export default function TraderArchetypesCard({ symbol, traderArchetypes }: TraderArchetypesCardProps) {
  const data = traderArchetypes || {
    consensusScore: 82,
    verdict: "Strong Buy / Core Accumulation",
    archetypes: [
      {
        name: "Warren Buffett (Value & Moat)",
        archetype: "High Cash Flow & Wide Moats",
        alignmentScore: 88,
        status: "High Moat Alignment",
        thesis: "High cash generation with strong pricing power, low corporate debt, and consistent share buybacks.",
        catalyst: "Durable competitive advantage and steady profit margins across economic cycles.",
      },
      {
        name: "Nancy Pelosi (Policy & Government Catalysts)",
        archetype: "Government Spending & High-Conviction Tech",
        alignmentScore: 94,
        status: "Strong Policy Support",
        thesis: "Direct beneficiary of federal technology subsidies, infrastructure spending, and government contracts.",
        catalyst: "Federal digital modernization mandates and legislative funding programs.",
      },
      {
        name: "Stanley Druckenmiller (Macro Trends)",
        archetype: "Interest Rate Trends & Market Momentum",
        alignmentScore: 82,
        status: "Positive Macro Trend",
        thesis: "The lower interest rate environment and upward price momentum favor holding this asset.",
        catalyst: "Central bank rate cuts and strong institutional buying momentum.",
      },
      {
        name: "Jim Simons (Quantitative Risk)",
        archetype: "Statistical Stability & Crash Protection",
        alignmentScore: 76,
        status: "Low Downside Risk",
        thesis: "Solid risk-adjusted returns with limited crash risk in down markets.",
        catalyst: "Low downside volatility and steady historical recovery during market pullbacks.",
      },
    ],
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-4">
      {/* Header with Consensus Score */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-400 animate-pulse"></span>
            <h3 className="text-base font-bold text-slate-100 font-mono tracking-tight flex items-center gap-2">
              <span>???</span>
              <span>{symbol} Institutional Strategy Alignment</span>
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Multi-strategy analysis combining Value/Moat, Government Policy, Macro Trend, and Downside Risk models
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

      {/* 4 Strategy Cards Grid */}
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

              {/* Progress Bar */}
              <div className="w-full bg-[#1b2434] h-1.5 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    isHigh ? "bg-emerald-500" : isMid ? "bg-cyan-500" : "bg-amber-500"
                  }`}
                  style={{ width: `${a.alignmentScore}%` }}
                ></div>
              </div>

              {/* Clear Simple Summary & Key Driver */}
              <div className="text-[11px] text-slate-300 font-mono space-y-1.5 pt-1 leading-relaxed">
                <p>
                  <span className="text-slate-400 font-semibold">Summary: </span>
                  {a.thesis}
                </p>
                <div className="flex items-center gap-1.5 text-[10px] text-purple-300 bg-purple-950/40 px-2 py-1 rounded border border-purple-900/60">
                  <span className="text-purple-400 font-bold shrink-0">Key Driver:</span>
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

