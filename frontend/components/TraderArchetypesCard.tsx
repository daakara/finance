"use client";

import { TraderArchetypeConsensus } from "../lib/api";

interface TraderArchetypesCardProps {
  symbol: string;
  traderArchetypes?: TraderArchetypeConsensus;
}

export default function TraderArchetypesCard({ symbol, traderArchetypes }: TraderArchetypesCardProps) {
  const data = traderArchetypes || {
    consensusScore: 84,
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
      {
        name: "David Gardner (Motley Fool Rule Breakers)",
        archetype: "First-Mover Disruptors & Hyper-Growth",
        alignmentScore: 92,
        status: "High-Conviction Rule Breaker",
        thesis: "Top-dog enterprise architecture with high gross margins, founder-led leadership, and rapid market expansion.",
        catalyst: "Secular migration toward modern digital compute and AI cloud workflows.",
      },
    ],
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono">
      {/* Header with Consensus Score */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-400 animate-pulse"></span>
            <h3 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <svg className="w-4 h-4 text-purple-400 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
              <span>{symbol} Institutional Strategy Alignment</span>
            </h3>
          </div>
          <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
            5 iconic investor archetypes evaluating moat, policy subsidies, macro trends, quantitative risk, and disruptive growth
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <div className="bg-purple-950/80 border border-purple-700/80 px-3 py-1 rounded-lg text-right">
            <span className="text-[9px] sm:text-[10px] text-purple-300 block uppercase leading-none font-bold">Consensus</span>
            <span className="text-sm sm:text-base font-bold text-purple-400">{data.consensusScore} / 100</span>
          </div>
          <span className="text-[11px] sm:text-xs font-semibold px-2.5 py-1 rounded-md bg-emerald-950/80 text-emerald-400 border border-emerald-800/80">
            {data.verdict}
          </span>
        </div>
      </div>

      {/* 5 Archetypes Adaptive Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3.5">
        {data.archetypes.map((item, idx) => {
          const isHigh = item.alignmentScore >= 80;
          const isMid = item.alignmentScore >= 65;

          return (
            <div
              key={idx}
              className="bg-[#090d14] border border-[#243044] hover:border-purple-500/50 rounded-xl p-3.5 space-y-2.5 transition-all flex flex-col justify-between"
            >
              <div className="space-y-2">
                {/* Archetype Title & Score */}
                <div className="flex items-start justify-between gap-2">
                  <span className="text-xs font-bold text-slate-200 leading-snug line-clamp-2">
                    {item.name}
                  </span>
                  <span
                    className={`text-sm font-bold shrink-0 ${
                      isHigh ? "text-cyan-400" : isMid ? "text-emerald-400" : "text-amber-400"
                    }`}
                  >
                    {item.alignmentScore}%
                  </span>
                </div>

                {/* Status Badge */}
                <span className="inline-block text-[10px] px-2 py-0.5 rounded bg-[#1b2434] text-purple-300 border border-purple-800/60 font-semibold">
                  {item.status}
                </span>

                {/* Plain-English Takeaway */}
                <div className="text-[11px] text-slate-300 space-y-1 leading-relaxed">
                  <p>
                    <span className="text-slate-400 font-semibold">Summary: </span>
                    {item.thesis}
                  </p>
                </div>
              </div>

              {/* Key Driver Catalyst */}
              <div className="bg-[#111722] p-2 rounded border border-[#1b2434] text-[10px] text-cyan-300">
                <span className="text-cyan-400 font-bold block mb-0.5">Key Driver:</span>
                <span className="line-clamp-3">{item.catalyst}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

