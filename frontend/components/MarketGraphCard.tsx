"use client";

import { MarketGraphData } from "../lib/api";

interface MarketGraphCardProps {
  symbol: string;
  marketGraph?: MarketGraphData;
}

export default function MarketGraphCard({ symbol, marketGraph }: MarketGraphCardProps) {
  const data = marketGraph?.topology || {
    upstream: [{ name: "Silicon & Component Suppliers", link: "Core supply chain inputs", impact: "High" }],
    downstream: [{ name: "Enterprise & Institutional Inflows", link: "Revenue and cash flow sources", impact: "High" }],
    macro: [{ name: "FRED Rate & Yield Curve", link: "Discount rate and capital cost sensitivity", impact: "High" }],
    peers: [{ name: "Sector Industry Benchmark", link: "Relative multiple contagion", impact: "Medium" }],
  };

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-4 font-mono">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <h3 className="text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <svg className="w-4 h-4 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="18" cy="5" r="3" />
                <circle cx="6" cy="12" r="3" />
                <circle cx="18" cy="19" r="3" />
                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
                <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
              </svg>
              <span>{symbol} Market Graph & Systemic Contagion Engine</span>
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Directed topology mapping supply chain dependencies, downstream customers, macro drivers, and peer contagion
          </p>
        </div>

        <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-[#1b2434] text-emerald-400 border border-emerald-800/60">
          Systemic Risk: Low-to-Moderate
        </span>
      </div>

      {/* 4 Graph Pillars Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Upstream Supply Chain */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2">
          <div className="flex items-center justify-between border-b border-[#1b2434] pb-1.5">
            <span className="text-xs font-bold text-amber-400 flex items-center gap-1.5">
              <span>⬆️</span>
              <span>Upstream Suppliers</span>
            </span>
            <span className="text-[10px] text-slate-500">Hardware & Inputs</span>
          </div>
          {data.upstream.map((item, idx) => (
            <div key={idx} className="text-xs text-slate-300 space-y-0.5">
              <div className="flex justify-between font-bold text-slate-200">
                <span>{item.name}</span>
                <span className="text-[10px] text-amber-400">{item.impact} Sensitivity</span>
              </div>
              <p className="text-[11px] text-slate-400">{item.link}</p>
            </div>
          ))}
        </div>

        {/* Downstream Customers */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2">
          <div className="flex items-center justify-between border-b border-[#1b2434] pb-1.5">
            <span className="text-xs font-bold text-emerald-400 flex items-center gap-1.5">
              <span>⬇️</span>
              <span>Downstream Customers</span>
            </span>
            <span className="text-[10px] text-slate-500">Revenue & Flow</span>
          </div>
          {data.downstream.map((item, idx) => (
            <div key={idx} className="text-xs text-slate-300 space-y-0.5">
              <div className="flex justify-between font-bold text-slate-200">
                <span>{item.name}</span>
                <span className="text-[10px] text-emerald-400">{item.impact} Sensitivity</span>
              </div>
              <p className="text-[11px] text-slate-400">{item.link}</p>
            </div>
          ))}
        </div>

        {/* Macro Sensitivity Links */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2">
          <div className="flex items-center justify-between border-b border-[#1b2434] pb-1.5">
            <span className="text-xs font-bold text-cyan-400 flex items-center gap-1.5">
              <span>🌐</span>
              <span>Macro Economic Linkages</span>
            </span>
            <span className="text-[10px] text-slate-500">FRED & Rates</span>
          </div>
          {data.macro.map((item, idx) => (
            <div key={idx} className="text-xs text-slate-300 space-y-0.5">
              <div className="flex justify-between font-bold text-slate-200">
                <span>{item.name}</span>
                <span className="text-[10px] text-cyan-400">{item.impact} Sensitivity</span>
              </div>
              <p className="text-[11px] text-slate-400">{item.link}</p>
            </div>
          ))}
        </div>

        {/* Competitor & Peer Contagion */}
        <div className="bg-[#090d14] p-3.5 rounded-lg border border-[#243044] space-y-2">
          <div className="flex items-center justify-between border-b border-[#1b2434] pb-1.5">
            <span className="text-xs font-bold text-purple-400 flex items-center gap-1.5">
              <span>🔄</span>
              <span>Peer Contagion & Beta</span>
            </span>
            <span className="text-[10px] text-slate-500">Sector Correlated</span>
          </div>
          {data.peers.map((item, idx) => (
            <div key={idx} className="text-xs text-slate-300 space-y-0.5">
              <div className="flex justify-between font-bold text-slate-200">
                <span>{item.name}</span>
                <span className="text-[10px] text-purple-400">{item.impact} Sensitivity</span>
              </div>
              <p className="text-[11px] text-slate-400">{item.link}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

