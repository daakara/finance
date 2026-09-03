"use client";

interface DataSourceBadgeProps {
  source?: "live" | "fallback";
  className?: string;
  labelLive?: string;
  labelFallback?: string;
}

export default function DataSourceBadge({
  source = "fallback",
  className = "",
  labelLive = "📡 Live Market Feed",
  labelFallback = "📊 Model Estimate",
}: DataSourceBadgeProps) {
  const isLive = source === "live";

  return (
    <div
      className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-bold border transition-colors ${
        isLive
          ? "bg-emerald-950/70 border-emerald-700/80 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.15)]"
          : "bg-amber-950/70 border-amber-700/80 text-amber-300 shadow-[0_0_10px_rgba(245,158,11,0.15)]"
      } ${className}`}
      title={
        isLive
          ? "Streaming real-time market data from exchange feeds, SEC EDGAR and OPRA sweeps."
          : "Deterministic quantitative model estimate based on trailing market close."
      }
    >
      <span
        className={`w-2 h-2 rounded-full ${
          isLive ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
        }`}
      />
      <span>{isLive ? labelLive : labelFallback}</span>
    </div>
  );
}
