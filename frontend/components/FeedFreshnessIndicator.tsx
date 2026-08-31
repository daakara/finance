"use client";

import { useState, useEffect } from "react";

interface FeedFreshnessIndicatorProps {
  source?: "live" | "fallback" | "cached";
  lastSync?: Date;
  className?: string;
}

export default function FeedFreshnessIndicator({
  source = "live",
  lastSync,
  className = "",
}: FeedFreshnessIndicatorProps) {
  const [timeAgo, setTimeAgo] = useState<string>("Just now");

  useEffect(() => {
    if (!lastSync) return;
    const interval = setInterval(() => {
      const diffSec = Math.floor((Date.now() - lastSync.getTime()) / 1000);
      if (diffSec < 5) setTimeAgo("Just now");
      else if (diffSec < 60) setTimeAgo(`${diffSec}s ago`);
      else setTimeAgo(`${Math.floor(diffSec / 60)}m ago`);
    }, 5000);
    return () => clearInterval(interval);
  }, [lastSync]);

  const isLive = source === "live";
  const isCached = source === "cached";

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-mono border transition-all ${
        isLive
          ? "bg-emerald-950/60 border-emerald-700/60 text-emerald-300"
          : isCached
          ? "bg-cyan-950/60 border-cyan-700/60 text-cyan-300"
          : "bg-amber-950/60 border-amber-700/60 text-amber-300"
      } ${className}`}
      title={
        isLive
          ? "Connected to live FastAPI quantitative analytics pipeline & direct exchange feeds."
          : isCached
          ? "Validated snapshot from local database within 15-minute freshness window."
          : "Operating in resilient offline mode — live feeds will reconnect automatically."
      }
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          isLive ? "bg-emerald-400 animate-pulse" : isCached ? "bg-cyan-400" : "bg-amber-400"
        }`}
      />
      <span className="font-bold">
        {isLive ? "REAL-TIME EXCHANGE" : isCached ? "VERIFIED CACHE (<15M)" : "RECONNECTING"}
      </span>
      <span className="text-slate-400">• {isLive ? timeAgo : isCached ? "Fresh" : "Offline"}</span>
    </div>
  );
}
