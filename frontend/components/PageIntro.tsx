"use client";

import React from "react";
import Link from "next/link";

export interface PageIntroAction {
  label: string;
  href?: string;
  onClick?: () => void;
  icon?: React.ReactNode;
}

export interface PageIntroProps {
  hubId: "radar" | "analysis" | "setups" | "portfolio" | "journal" | "performance";
  title: string;
  purpose: string;
  badge?: string;
  symbol?: string | null;
  isDemo?: boolean;
  demoNotice?: string;
  primaryAction?: PageIntroAction;
  secondaryAction?: PageIntroAction;
  children?: React.ReactNode;
  className?: string;
}

/**
 * PageIntro: Decision-oriented header contract for core ARX Terminal hubs.
 * Answers within the first viewport:
 * 1. Where am I? (title + badge)
 * 2. What is this page for? (purpose)
 * 3. What should I do next? (primaryAction + secondaryAction)
 * 4. What does the current state mean? (contextual badges, demo notices)
 */
export default function PageIntro({
  hubId,
  title,
  purpose,
  badge,
  symbol,
  isDemo = false,
  demoNotice,
  primaryAction,
  secondaryAction,
  children,
  className = "",
}: PageIntroProps) {
  return (
    <div
      data-hub-intro={hubId}
      className={`p-4 md:p-5 rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-md flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-lg ${className}`}
    >
      <div className="space-y-1.5 max-w-3xl">
        <div className="flex flex-wrap items-center gap-2">
          {badge && (
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-cyan-400 bg-cyan-950/80 border border-cyan-800/80 px-2 py-0.5 rounded">
              {badge}
            </span>
          )}
          {symbol && (
            <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700">
              Active Context: {symbol}
            </span>
          )}
          {isDemo && (
            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-800/80">
              Demonstration Asset
            </span>
          )}
        </div>

        <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight flex items-center gap-2 font-mono">
          {title}
        </h1>

        <p className="text-xs sm:text-sm text-slate-300 font-sans leading-relaxed">
          {purpose}
        </p>

        {isDemo && demoNotice && (
          <div className="text-[11px] font-mono text-amber-300/90 pt-0.5 flex items-center gap-1.5">
            <span>ℹ️</span>
            <span>{demoNotice}</span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2.5 shrink-0">
        {children}

        {secondaryAction && (
          secondaryAction.href ? (
            <Link
              href={secondaryAction.href}
              className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-bold transition-all border border-slate-700 flex items-center gap-1.5 cursor-pointer hover:border-slate-600 active:scale-95"
            >
              {secondaryAction.icon}
              <span>{secondaryAction.label}</span>
            </Link>
          ) : (
            <button
              type="button"
              onClick={secondaryAction.onClick}
              className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-bold transition-all border border-slate-700 flex items-center gap-1.5 cursor-pointer hover:border-slate-600 active:scale-95"
            >
              {secondaryAction.icon}
              <span>{secondaryAction.label}</span>
            </button>
          )
        )}

        {primaryAction && (
          primaryAction.href ? (
            <Link
              href={primaryAction.href}
              className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold tracking-tight transition-all shadow-md shadow-cyan-950/50 flex items-center gap-1.5 cursor-pointer active:scale-95"
            >
              {primaryAction.icon}
              <span>{primaryAction.label}</span>
            </Link>
          ) : (
            <button
              type="button"
              onClick={primaryAction.onClick}
              className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold tracking-tight transition-all shadow-md shadow-cyan-950/50 flex items-center gap-1.5 cursor-pointer active:scale-95"
            >
              {primaryAction.icon}
              <span>{primaryAction.label}</span>
            </button>
          )
        )}
      </div>
    </div>
  );
}
