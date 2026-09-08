"use client";

import React, { ReactNode } from "react";
import ExecutiveIntelligenceNav from "../committee/ExecutiveIntelligenceNav";
import { LayoutContainers } from "../../lib/ui/layout";

export interface IntelligenceShellProps {
  title: string;
  subtitle?: string;
  badge?: string;
  actions?: ReactNode;
  children: ReactNode;
  containerClass?: string;
  activeNavTab?: string;
}

export default function IntelligenceShell({
  title,
  subtitle,
  badge = "PHASE 31-M11 CERTIFIED",
  actions,
  children,
  containerClass = LayoutContainers.standard,
  activeNavTab,
}: IntelligenceShellProps) {
  return (
    <div className="min-h-screen bg-[#0B1220] text-[#F8FAFC] flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-200">
      <ExecutiveIntelligenceNav activeTab={activeNavTab} badgeText={badge} />

      <main className={`flex-1 py-6 ${containerClass} w-full space-y-6`}>
        {/* Horizon Page Header */}
        <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#24324A]">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[11px] font-mono uppercase tracking-widest text-cyan-400 font-semibold">
                ARX Horizon Executive Platform
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#F8FAFC]">
              {title}
            </h1>
            {subtitle && (
              <p className="text-sm text-[#94A3B8] mt-1 font-mono">
                {subtitle}
              </p>
            )}
          </div>

          {actions && (
            <div className="flex items-center gap-3 shrink-0">
              {actions}
            </div>
          )}
        </header>

        {/* Main Viewport Content */}
        <div className="space-y-6">
          {children}
        </div>
      </main>
    </div>
  );
}
