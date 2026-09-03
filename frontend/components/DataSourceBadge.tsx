"use client";

export type ProvenanceSource = "live" | "delayed" | "curated" | "fallback" | "unavailable";

interface DataSourceBadgeProps {
  source?: ProvenanceSource;
  className?: string;
  labelLive?: string;
  labelFallback?: string;
  labelCurated?: string;
  labelDelayed?: string;
  labelUnavailable?: string;
}

export default function DataSourceBadge({
  source = "fallback",
  className = "",
  labelLive = "📡 Live Market Feed",
  labelFallback = "📊 Model Estimate",
  labelCurated = "📚 Curated Dataset",
  labelDelayed = "⏱️ Regulatory Delayed",
  labelUnavailable = "⚠️ Feed Unavailable",
}: DataSourceBadgeProps) {
  let badgeColor = "bg-amber-950/70 border-amber-700/80 text-amber-300 shadow-[0_0_10px_rgba(245,158,11,0.15)]";
  let dotColor = "bg-amber-400";
  let label = labelFallback;
  let tooltip = "Deterministic quantitative model estimate based on trailing market close.";

  if (source === "live") {
    badgeColor = "bg-emerald-950/70 border-emerald-700/80 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.15)]";
    dotColor = "bg-emerald-400 animate-pulse";
    label = labelLive;
    tooltip = "Real-time exchange market feed and authentic OHLCV candle streams.";
  } else if (source === "curated") {
    badgeColor = "bg-purple-950/70 border-purple-700/80 text-purple-300 shadow-[0_0_10px_rgba(168,85,247,0.15)]";
    dotColor = "bg-purple-400";
    label = labelCurated;
    tooltip = "Verified historical research dataset curated from public statutory records.";
  } else if (source === "delayed") {
    badgeColor = "bg-cyan-950/70 border-cyan-700/80 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.15)]";
    dotColor = "bg-cyan-400";
    label = labelDelayed;
    tooltip = "Official regulatory disclosure subject to statutory reporting latency.";
  } else if (source === "unavailable") {
    badgeColor = "bg-slate-900/80 border-slate-700 text-slate-400";
    dotColor = "bg-slate-500";
    label = labelUnavailable;
    tooltip = "Live external provider is disconnected or offline.";
  }

  return (
    <div
      className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-bold border transition-colors ${badgeColor} ${className}`}
      title={tooltip}
    >
      <span className={`w-2 h-2 rounded-full ${dotColor}`} />
      <span>{label}</span>
    </div>
  );
}
