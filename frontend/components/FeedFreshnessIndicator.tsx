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

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-mono border transition-all ${
        isLive
          ? "bg-emerald-950/60 border-emerald-700/60 text-emerald-300"
          : "bg-amber-950/60 border-amber-700/60 text-amber-300"
      } ${className}`}
      title={
        isLive
          ? "Connected to live FastAPI quantitative analytics pipeline"
          : "Operating on pre-rendered resilience catalog baseline"
      }
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          isLive ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
        }`}
      />
      <span className="font-bold">
        {isLive ? "REAL-TIME FEED" : "BASELINE CATALOG"}
      </span>
      <span className="text-slate-400">• {isLive ? timeAgo : "Resilience Mode"}</span>
    </div>
  );
}
