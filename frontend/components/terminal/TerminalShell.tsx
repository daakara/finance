"use client";

/**
 * Horizon 14.1: Persistent Terminal Shell
 *
 * Enforces INV-OI115-P: Persistent Terminal Navigation.
 * Encapsulates the global Navbar, persistent sub-header, live market regime,
 * and persistent mobile dock across all 5 flagship hubs:
 * - /radar: "What deserves attention today?"
 * - /setups: "What is actionable right now?"
 * - /portfolio: "What risk am I carrying?"
 * - /journal: "Did I follow my rules?"
 * - /performance: "Is ARX actually improving my results?"
 */

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Navbar from "../Navbar";
import { fetchMacroRibbon } from "../../lib/api";

export type TerminalHubId = "radar" | "setups" | "portfolio" | "journal" | "performance";
export type TerminalHub = TerminalHubId;

export interface TerminalHubMeta {
  id: TerminalHubId;
  route: string;
  name: string;
  question: string;
  badge: string;
}

export const TERMINAL_HUBS: TerminalHubMeta[] = [
  { id: "radar", route: "/radar", name: "Radar", question: "What deserves attention today?", badge: "CONFLUENCE" },
  { id: "setups", route: "/setups", name: "Setups", question: "What is actionable right now?", badge: "EXECUTION" },
  { id: "portfolio", route: "/portfolio", name: "Portfolio", question: "What risk am I carrying?", badge: "RISK HEAT" },
  { id: "journal", route: "/journal", name: "Journal", question: "Did I follow my rules?", badge: "DISCIPLINE" },
  { id: "performance", route: "/performance", name: "Performance", question: "Is ARX actually improving my results?", badge: "PROOF OF EDGE" },
];

interface TerminalShellProps {
  activeHub: TerminalHubId;
  children: React.ReactNode;
}

export default function TerminalShell({
  activeHub,
  children,
}: TerminalShellProps) {
  const currentHub = TERMINAL_HUBS.find((h) => h.id === activeHub) || TERMINAL_HUBS[0];

  const [dynamicRegime, setDynamicRegime] = useState<string | null>(null);

  useEffect(() => {
    fetchMacroRibbon().then((data) => {
      if (data && data.regime) {
        if (data.regime === "RISK_ON") setDynamicRegime("Confirmed Uptrend");
        else if (data.regime === "DEFENSIVE") setDynamicRegime("Defensive / High Vol");
        else if (data.regime === "NEUTRAL") setDynamicRegime("Neutral / Selective");
        else if (data.regime === "UNAVAILABLE") setDynamicRegime("Regime Unavailable");
        else setDynamicRegime(data.regime);
      } else {
        setDynamicRegime("Regime Unavailable");
      }
    }).catch(() => {
      setDynamicRegime("Regime Unavailable");
    });
  }, []);

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-100 font-sans selection:bg-cyan-500 selection:text-black flex flex-col transition-colors duration-200">
      {/* 1. Global Persistent Terminal Navbar */}
      <Navbar hideMobileDock={true} />

      {/* 2. Persistent Terminal Sub-Header & Question Breadcrumb (INV-OI115-P) */}
      <section
        role="region"
        aria-label="Terminal Hub Header"
        className="w-full bg-[#0b1019]/90 border-b border-[#1b2537] border-t border-[#1e293b] px-4 md:px-8 py-2"
      >
        <div className="max-w-[1750px] mx-auto flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-cyan-400 bg-cyan-950/80 border border-cyan-800 px-2 py-0.5 rounded">
              {currentHub.badge}
            </span>
            <span className="text-xs font-mono text-slate-400">
              ARX TERMINAL &gt; <strong className="text-white uppercase">{currentHub.name}</strong>
            </span>
            <span className="text-slate-600 hidden sm:inline">•</span>
            <span className="text-xs text-slate-300 font-medium hidden sm:inline italic">
              &quot;{currentHub.question}&quot;
            </span>
          </div>

          {/* Direct Flagship Navigation Switching */}
          <nav className="hidden lg:flex items-center space-x-1">
            {TERMINAL_HUBS.map((hub) => (
              <Link
                key={hub.id}
                href={hub.route}
                className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-colors ${
                  activeHub === hub.id
                    ? "bg-cyan-950/80 text-cyan-300 border border-cyan-700/60"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#151f30]"
                }`}
              >
                {hub.name}
              </Link>
            ))}
          </nav>

          <div className="flex items-center space-x-4 text-xs font-mono">
            <div className="flex items-center space-x-1.5 text-slate-400">
              <span className={`w-2 h-2 rounded-full ${
                !dynamicRegime ? 'bg-slate-600' :
                dynamicRegime.includes('Defensive') ? 'bg-rose-400' :
                dynamicRegime.includes('Neutral') ? 'bg-amber-400' :
                dynamicRegime.includes('Unavailable') ? 'bg-slate-500' : 'bg-emerald-400'
              } animate-pulse`}></span>
              <span>REGIME: {dynamicRegime || "Market Discovery..."}</span>
            </div>
            <span className="text-slate-700 hidden sm:inline">|</span>
            <span className="hidden sm:inline text-slate-500 text-[11px]">
              <kbd className="px-1.5 py-0.5 bg-[#162030] border border-[#23334d] rounded text-cyan-400">⌘K</kbd> Palette
            </span>
          </div>
        </div>
      </section>

      {/* 3. Main Hub Content Canvas */}
      <main className="flex-1 w-full max-w-[1750px] mx-auto p-4 sm:p-6 lg:p-8 space-y-6 pb-24 lg:pb-12">
        {children}
      </main>

      {/* 4. Persistent Mobile Bottom Navigation Dock (INV-OI115-P) */}
      <aside
        role="navigation"
        aria-label="Mobile Terminal Navigation"
        data-testid="mobile-nav-dock"
        className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-[#0b1019]/95 backdrop-blur-md border-t border-[#1e293b] flex items-center justify-around px-2 py-1.5 pb-[calc(0.5rem+env(safe-area-inset-bottom,0px))] text-[10px] font-mono shadow-2xl"
      >
        <Link
          href="/radar"
          className={`flex flex-col items-center justify-center min-w-[44px] min-h-[44px] px-1.5 py-1 rounded-lg transition-colors ${
            activeHub === 'radar'
              ? 'text-cyan-400 font-bold bg-cyan-950/50 border border-cyan-800/60 shadow-inner'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <span className="text-base mb-0.5 leading-none">📡</span>
          <span className="truncate">Radar</span>
        </Link>
        <Link
          href="/setups"
          className={`flex flex-col items-center justify-center min-w-[44px] min-h-[44px] px-1.5 py-1 rounded-lg transition-colors ${
            activeHub === 'setups'
              ? 'text-cyan-400 font-bold bg-cyan-950/50 border border-cyan-800/60 shadow-inner'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <span className="text-base mb-0.5 leading-none">⚡</span>
          <span className="truncate">Setups</span>
        </Link>
        <Link
          href="/portfolio"
          className={`flex flex-col items-center justify-center min-w-[44px] min-h-[44px] px-1.5 py-1 rounded-lg transition-colors ${
            activeHub === 'portfolio'
              ? 'text-cyan-400 font-bold bg-cyan-950/50 border border-cyan-800/60 shadow-inner'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <span className="text-base mb-0.5 leading-none">💼</span>
          <span className="truncate">Portfolio</span>
        </Link>
        <Link
          href="/journal"
          className={`flex flex-col items-center justify-center min-w-[44px] min-h-[44px] px-1.5 py-1 rounded-lg transition-colors ${
            activeHub === 'journal'
              ? 'text-cyan-400 font-bold bg-cyan-950/50 border border-cyan-800/60 shadow-inner'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <span className="text-base mb-0.5 leading-none">📖</span>
          <span className="truncate">Journal</span>
        </Link>
        <Link
          href="/performance"
          className={`flex flex-col items-center justify-center min-w-[44px] min-h-[44px] px-1.5 py-1 rounded-lg transition-colors ${
            activeHub === 'performance'
              ? 'text-emerald-400 font-bold bg-emerald-950/50 border border-emerald-800/60 shadow-inner'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <span className="text-base mb-0.5 leading-none">📈</span>
          <span className="truncate">Alpha</span>
        </Link>
      </aside>
    </div>
  );
}
