"use client";

import React from "react";
import { CommitteeFeedItem } from "../../types/committee-intelligence";

export interface CommitteeFeedProps {
  items: CommitteeFeedItem[];
  onSelectItem?: (item: CommitteeFeedItem) => void;
  className?: string;
}

export default function CommitteeFeed({
  items,
  onSelectItem,
  className = "",
}: CommitteeFeedProps) {
  return (
    <div
      data-testid="committee-feed"
      className={`p-6 rounded-2xl bg-bg-surface border border-border-subtle shadow-xl space-y-4 font-sans ${className}`}
    >
      <div className="flex items-center justify-between pb-3 border-b border-border-subtle">
        <div className="flex items-center gap-2.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />
          <h3 className="text-header-2 text-text-primary">
            {"Committee Review & Conflict Queue"}
          </h3>
          <span className="px-2 py-0.5 text-xs font-mono font-bold rounded-full bg-bg-surface-raised text-text-secondary border border-border-subtle">
            {items.length} Active Items
          </span>
        </div>

        <span className="text-caption-mono text-text-muted text-xs hidden sm:inline">
          Priority: Critical Disagreements First
        </span>
      </div>

      <div className="space-y-3">
        {items.length === 0 ? (
          <p className="text-xs font-mono text-text-muted text-center py-6">
            Zero pending committee conflicts or escalated reviews.
          </p>
        ) : (
          items.map((item) => (
            <div
              key={item.id}
              onClick={() => onSelectItem && onSelectItem(item)}
              className="p-4 rounded-xl bg-bg-surface-raised hover:bg-bg-surface-elevated border border-border-subtle hover:border-accent-info/50 transition-all cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-3 group"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-header-2 font-black text-text-primary font-mono group-hover:text-accent-info transition-colors">
                    ${item.ticker}
                  </span>
                  <span
                    className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded border uppercase ${
                      item.severity === "CRITICAL"
                        ? "bg-rose-950 text-rose-300 border-rose-800"
                        : "bg-amber-950 text-amber-300 border-amber-800"
                    }`}
                  >
                    {item.severity} · {item.category}
                  </span>
                </div>
                <p className="text-body-ui text-text-secondary text-xs">
                  {item.headline}
                </p>
              </div>

              <div className="flex items-center gap-3 shrink-0">
                <span className="text-caption-mono text-text-muted text-xs">
                  {new Date(item.createdAt).toLocaleDateString()}
                </span>
                <span className="px-3 py-1 rounded-lg bg-accent-info/10 text-accent-info border border-accent-info/30 text-xs font-mono font-bold group-hover:bg-accent-info/20 transition-colors">
                  Review Thesis →
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
