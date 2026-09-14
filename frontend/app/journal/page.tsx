"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import TerminalShell from "../../components/terminal/TerminalShell";
import PageIntro from "../../components/PageIntro";
import { fetchJournalTrades, fetchUserRiskTelemetry, UserRiskTelemetry } from "../../lib/api";

export interface TradeLogEntry {
  id: string;
  ticker: string;
  date: string;
  setup: string;
  rAchieved?: number | null;
  followedRules?: boolean | null;
  pnl?: string | null;
  confidence?: number | null;
  status?: string;
  executionRole?: string | null;
  remainingShares?: number | null;
}

export default function JournalPage() {
  const [tradeLogs, setTradeLogs] = useState<TradeLogEntry[]>([]);
  const [, setTelemetry] = useState<UserRiskTelemetry | null>(null);
  const [, setIsLoading] = useState<boolean>(true);
  const [urlSymbol, setUrlSymbol] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const sym = params.get("symbol") || params.get("ticker");
      if (sym) {
        setUrlSymbol(sym.trim().toUpperCase());
      }
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    async function loadJournalData() {
      setIsLoading(true);
      try {
        const [apiTrades, apiTelemetry] = await Promise.all([
          fetchJournalTrades(100),
          fetchUserRiskTelemetry(),
        ]);

        if (!isMounted) return;

        if (apiTrades && apiTrades.length > 0) {
          const mapped: TradeLogEntry[] = apiTrades.map((t) => ({
            id: String(t.id),
            ticker: t.ticker || t.symbol || "",
            date: t.date || t.entryDate || "",
            setup: t.setup || t.setupName || "Breakout",
            rAchieved: t.rAchieved,
            followedRules: t.followedRules,
            pnl: t.pnl,
            confidence: t.confidence,
            status: t.status,
            executionRole: t.executionRole,
            remainingShares: t.remainingShares,
          }));
          setTradeLogs(mapped);
        } else if (typeof window !== "undefined") {
          const raw = localStorage.getItem("FINANCE_JOURNAL_LOGS");
          if (raw) {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
              setTradeLogs(parsed);
            }
          }
        }

        if (apiTelemetry) {
          setTelemetry(apiTelemetry);
        }
      } catch (err) {
        console.warn("Could not load journal trade logs or telemetry from API:", err);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    loadJournalData();

    return () => {
      isMounted = false;
    };
  }, []);

  const tradesLogged = tradeLogs.length;

  return (
    <TerminalShell activeHub="journal" activeSymbol={urlSymbol}>
      <div className="space-y-6 pb-20">
        <PageIntro
          hubId="journal"
          title="Trade Journal & Discipline Log"
          purpose="Execution discipline records and behavioral analytics (Planned for Post-R1)."
          badge="Deferred Scope"
          symbol={urlSymbol || null}
          primaryAction={{
            label: "Manage Portfolio →",
            href: "/portfolio",
          }}
          secondaryAction={{
            label: "Explore Setups →",
            href: "/setups",
          }}
        />

        {/* Option A Deferred Scope Card */}
        <div className="p-6 md:p-8 rounded-2xl bg-slate-900/80 border border-amber-500/30 shadow-2xl backdrop-blur space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-2.5">
              <span className="px-2.5 py-1 rounded-md text-[11px] font-mono font-black uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40">
                DEFERRED SCOPE · POST-R1
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Canonical Journey: Radar → Analysis → Trade Plan → Portfolio
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs font-mono text-emerald-400">
                Persistence Contracts Active ({tradesLogged} recorded)
              </span>
            </div>
          </div>

          <div className="space-y-3">
            <h2 className="text-lg md:text-xl font-bold text-white">
              Dedicated Journal Surface Deferred to Post-R1
            </h2>
            <p className="text-sm text-slate-300 font-sans max-w-3xl leading-relaxed">
              The dedicated Trade Journal review and calibration surface is deferred to a post-R1 release to focus Release 1 strictly on the core decision loop: <strong className="text-white">Find → Understand → Plan → Manage</strong>.
            </p>
            <p className="text-xs text-slate-400 font-sans max-w-3xl leading-relaxed">
              All backend persistence, broker fill records, exit logs, position reconciliation, and risk telemetry contracts remain active in the background. Executions recorded from the Trade Plan and exits recorded in Portfolio continue to persist in the database.
            </p>
          </div>

          {/* Core Release 1 Hub Return CTAs */}
          <div className="pt-2">
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
              Active Release 1 Destinations
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              <Link
                href="/portfolio"
                className="flex flex-col p-4 rounded-xl bg-cyan-950/40 hover:bg-cyan-900/50 border border-cyan-800 hover:border-cyan-500 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-cyan-400 font-bold mb-1">
                  <span>💼 Manage Portfolio</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Track position risk, Cornish-Fisher VaR, and record exits.
                </span>
              </Link>

              <Link
                href="/setups"
                className="flex flex-col p-4 rounded-xl bg-slate-950/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-white font-bold mb-1">
                  <span>⚡ Plan Trade</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Calculate position sizing, stops, and target brackets.
                </span>
              </Link>

              <Link
                href="/"
                className="flex flex-col p-4 rounded-xl bg-slate-950/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-white font-bold mb-1">
                  <span>🔬 Deep Analysis</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Understand asset technicals, fundamentals, and regimes.
                </span>
              </Link>

              <Link
                href="/radar"
                className="flex flex-col p-4 rounded-xl bg-slate-950/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition group cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-white font-bold mb-1">
                  <span>📡 Market Radar</span>
                  <span className="text-slate-400 group-hover:translate-x-0.5 transition-transform">→</span>
                </div>
                <span className="text-[11px] text-slate-300">
                  Scan for Stage 2 breakouts, volume dry-up, and momentum.
                </span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </TerminalShell>
  );
}
