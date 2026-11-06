"use client";

/**
 * frontend/app/poc-market-wide/page.tsx
 *
 * ARX Terminal — Directive ARX-MW-001
 * Isolated Visual POC: Market-Wide Discovery and Top-35 Radar Preview.
 *
 * Governance Invariants:
 * - Labeled: "EXPERIMENTAL MARKET-WIDE RADAR — SHADOW POC"
 * - Non-production shadow preview; strictly decoupled from production Radar.
 * - Candidates are discovery-only and non-executable (canSizeTrade: false, isActionable: false).
 */

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";

interface Candidate {
  rank: number;
  symbol: string;
  company_name: string;
  ranking_cohort: "COHORT_A_QUALIFIED" | "COHORT_B_DEVELOPING";
  strategy_classification: string;
  discovery_reason: string;
  confluence_score: number;
  current_price: number;
  risk_reward_ratio: number | null;
  relative_strength: number | null;
  setup_status: string;
  invalidation_condition: string;
  technical_conditions: {
    setup_pattern?: string;
    stage_phase?: string;
    trend_template_score?: number;
    breakout_pivot?: number;
    atr_14?: number;
    candle_count?: number;
  };
  missing_evidence: string[];
}

interface CoverageAccounting {
  source_market_population: number;
  eligible_universe: number;
  ineligible_population: number;
  eligible_data_blocked: number;
  scannable_securities: number;
  stage1_evaluated: number;
  stage1_rejected: number;
  stage2_admitted: number;
  qualified_candidates_cohort_a: number;
  developing_discoveries_cohort_b: number;
  displayed_candidates: number;
  universe_completeness_pct: number;
  processing_completeness_pct: number;
  data_coverage_status: string;
  arithmetic_closure: boolean;
}

interface SnapshotData {
  snapshot_id: string;
  universe_build_id: string;
  as_of: string;
  governance_label: string;
  coverage_accounting: CoverageAccounting;
  candidates: Candidate[];
}

const HISTORICAL_35_SYMBOLS = new Set([
  "LNTH", "CPRX", "MEDP", "TMDX", "ISRG", "VRTX", "LLY", "NVO", "DXCM", "PODD",
  "ACLS", "POWI", "ON", "MPWR", "KLAC", "LRCX", "ASML", "AVGO",
  "ELF", "DECK", "LULU", "ONON", "MNST", "ULTA",
  "VRT", "ETN", "PWR", "GEV", "FIX", "EME",
  "DUOL", "ANET", "NOW", "SNPS", "CDNS"
]);

export default function MarketWideRadarPocPage() {
  const [data, setData] = useState<SnapshotData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"ALL" | "COHORT_A" | "COHORT_B" | "INCREMENTAL">("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    fetch("/data/poc/market_wide_radar_snapshot.json")
      .then((res) => res.json())
      .then((json: SnapshotData) => {
        setData(json);
        setLoading(false);
      })
      .catch(() => {
        // Fallback to empty state
        setLoading(false);
      });
  }, []);

  const accounting = data?.coverage_accounting;
  const candidates = data?.candidates || [];

  const filteredCandidates = useMemo(() => {
    return candidates.filter((c) => {
      // Tab filter
      if (activeTab === "COHORT_A" && c.ranking_cohort !== "COHORT_A_QUALIFIED") return false;
      if (activeTab === "COHORT_B" && c.ranking_cohort !== "COHORT_B_DEVELOPING") return false;
      if (activeTab === "INCREMENTAL" && HISTORICAL_35_SYMBOLS.has(c.symbol)) return false;

      // Search filter
      if (searchQuery.trim()) {
        const q = searchQuery.trim().toUpperCase();
        return c.symbol.toUpperCase().includes(q) || c.company_name.toUpperCase().includes(q);
      }
      return true;
    });
  }, [candidates, activeTab, searchQuery]);

  const incrementalCount = useMemo(() => {
    return candidates.filter((c) => !HISTORICAL_35_SYMBOLS.has(c.symbol)).length;
  }, [candidates]);

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-200 font-sans antialiased">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Governance Notice Banner */}
        <div
          role="status"
          aria-live="polite"
          className="p-5 md:p-6 rounded-2xl border border-amber-500/40 bg-amber-950/20 text-amber-200 font-mono text-xs space-y-2 shadow-lg"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className="text-2xl">📡</span>
              <div>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-500/20 border border-amber-500/40 text-amber-300">
                  EXPERIMENTAL MARKET-WIDE RADAR — SHADOW POC
                </span>
                <h1 className="text-sm sm:text-base font-bold text-white mt-1">
                  Directive ARX-MW-001: Top-35 Full-Market Discovery Proof of Concept
                </h1>
              </div>
            </div>
            <div className="text-right text-[11px] text-amber-400/80">
              <div>Snapshot ID: {data?.snapshot_id || "LOADING..."}</div>
              <div>Build: {data?.universe_build_id || "BUILD_MW_POC"}</div>
            </div>
          </div>
          <p className="text-slate-300 text-xs font-sans leading-relaxed pt-1">
            This experimental preview scans the entire eligible US stock market universe beyond the historical 35-symbol fixture.
            The number 35 is a presentation cap, not a scan limitation. Candidates are strictly discovery opportunities and cannot be executed directly.
          </p>
        </div>

        {/* Discovery Funnel & Population Accounting Card */}
        {accounting && (
          <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider">
                Discovery Funnel & Population Accounting Closure
              </h2>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                {accounting.data_coverage_status}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-center">
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px] font-mono">Source Market</div>
                <div className="text-base sm:text-lg font-bold font-mono text-white mt-0.5">
                  {accounting.source_market_population.toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-500 font-sans">Raw Directory</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px] font-mono">Eligible Stocks</div>
                <div className="text-base sm:text-lg font-bold font-mono text-emerald-400 mt-0.5">
                  {accounting.eligible_universe.toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-500 font-sans">US Common Stock</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px] font-mono">Data-Blocked</div>
                <div className="text-base sm:text-lg font-bold font-mono text-rose-400 mt-0.5">
                  {accounting.eligible_data_blocked.toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-500 font-sans">Provider Gap</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px] font-mono">Scannable Pool</div>
                <div className="text-base sm:text-lg font-bold font-mono text-cyan-400 mt-0.5">
                  {accounting.scannable_securities}
                </div>
                <div className="text-[10px] text-slate-500 font-sans">In Local Store</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px] font-mono">Stage 1 Passed</div>
                <div className="text-base sm:text-lg font-bold font-mono text-blue-400 mt-0.5">
                  {accounting.stage2_admitted}
                </div>
                <div className="text-[10px] text-slate-500 font-sans">Survivors</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px] font-mono">Cohort A (Qual)</div>
                <div className="text-base sm:text-lg font-bold font-mono text-emerald-400 mt-0.5">
                  {accounting.qualified_candidates_cohort_a}
                </div>
                <div className="text-[10px] text-slate-500 font-sans">Stage 2 Verified</div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px] font-mono">Cohort B (Dev)</div>
                <div className="text-base sm:text-lg font-bold font-mono text-amber-400 mt-0.5">
                  {accounting.developing_discoveries_cohort_b}
                </div>
                <div className="text-[10px] text-slate-500 font-sans">Developing Bases</div>
              </div>
            </div>
          </section>
        )}

        {/* Filter Controls & Tabs */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setActiveTab("ALL")}
              className={`px-4 py-2 min-h-[44px] min-w-[44px] rounded-xl font-mono text-xs font-bold transition-colors cursor-pointer ${
                activeTab === "ALL"
                  ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950/40"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              All Top 35 ({candidates.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("COHORT_A")}
              className={`px-4 py-2 min-h-[44px] min-w-[44px] rounded-xl font-mono text-xs font-bold transition-colors cursor-pointer ${
                activeTab === "COHORT_A"
                  ? "bg-emerald-500 text-slate-950 shadow-md shadow-emerald-950/40"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              Cohort A — Qualified ({candidates.filter((c) => c.ranking_cohort === "COHORT_A_QUALIFIED").length})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("COHORT_B")}
              className={`px-4 py-2 min-h-[44px] min-w-[44px] rounded-xl font-mono text-xs font-bold transition-colors cursor-pointer ${
                activeTab === "COHORT_B"
                  ? "bg-amber-500 text-slate-950 shadow-md shadow-amber-950/40"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              Cohort B — Developing ({candidates.filter((c) => c.ranking_cohort === "COHORT_B_DEVELOPING").length})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("INCREMENTAL")}
              className={`px-4 py-2 min-h-[44px] min-w-[44px] rounded-xl font-mono text-xs font-bold transition-colors cursor-pointer ${
                activeTab === "INCREMENTAL"
                  ? "bg-indigo-500 text-white shadow-md shadow-indigo-950/40"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-300"
              }`}
            >
              ✨ Incremental Discoveries ({incrementalCount})
            </button>
          </div>

          <div className="relative">
            <input
              type="text"
              placeholder="Search symbol or name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full sm:w-64 px-4 py-2.5 min-h-[44px] rounded-xl bg-slate-900 border border-slate-700 text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-400 font-mono"
            />
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="p-16 text-center text-slate-400 font-mono text-xs animate-pulse">
            ⏳ Scanning multi-factor equity tape and quantitative confluence filters across full market...
          </div>
        )}

        {/* Candidates Grid */}
        {!loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredCandidates.map((c) => {
              const isIncremental = !HISTORICAL_35_SYMBOLS.has(c.symbol);
              const isCohortA = c.ranking_cohort === "COHORT_A_QUALIFIED";

              return (
                <article
                  key={c.symbol}
                  className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 hover:border-slate-700 transition-all flex flex-col justify-between"
                >
                  <div className="space-y-3">
                    {/* Card Header: Rank, Symbol, Badges */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span className="font-mono text-xs font-bold text-slate-500 w-6">
                          #{c.rank.toString().padStart(2, "0")}
                        </span>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-base text-white tracking-tight">
                              {c.symbol}
                            </span>
                            {isIncremental && (
                              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                                Discovery
                              </span>
                            )}
                          </div>
                          <p className="text-[11px] text-slate-400 truncate max-w-[180px] font-sans">
                            {c.company_name}
                          </p>
                        </div>
                      </div>

                      <div className="text-right">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                            isCohortA
                              ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/30"
                              : "bg-amber-500/10 text-amber-300 border border-amber-500/30"
                          }`}
                        >
                          {isCohortA ? "Cohort A" : "Cohort B"}
                        </span>
                      </div>
                    </div>

                    {/* Metrics Bar: Price, Score, R:R */}
                    <div className="grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80 text-center font-mono text-xs">
                      <div>
                        <div className="text-[10px] text-slate-500">Price</div>
                        <div className="font-bold text-white mt-0.5">${c.current_price.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500">Confluence</div>
                        <div className="font-bold text-cyan-400 mt-0.5">{c.confluence_score.toFixed(0)}/100</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500">R:R Ratio</div>
                        <div className="font-bold text-emerald-400 mt-0.5">
                          {c.risk_reward_ratio ? `${c.risk_reward_ratio.toFixed(1)}:1` : "—"}
                        </div>
                      </div>
                    </div>

                    {/* Pattern & Stage */}
                    <div className="space-y-1 font-mono text-[11px]">
                      <div className="text-slate-300">
                        <span className="text-slate-500">Pattern: </span>
                        {c.technical_conditions.setup_pattern || "Base Building"}
                      </div>
                      <div className="text-slate-400 text-[10px] font-sans leading-relaxed line-clamp-2">
                        {c.discovery_reason}
                      </div>
                    </div>

                    {/* Missing Evidence Indicators */}
                    {c.missing_evidence && c.missing_evidence.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {c.missing_evidence.map((ev, idx) => (
                          <span
                            key={idx}
                            className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-slate-800 text-slate-400"
                          >
                            ⚠️ {ev.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Card Footer: Non-Actionable Notice */}
                  <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-500">
                    <span>Non-Executable</span>
                    <Link
                      href={`/?symbol=${c.symbol}`}
                      className="text-cyan-400 hover:text-cyan-300 transition-colors font-sans"
                    >
                      Inspect in Analysis →
                    </Link>
                  </div>
                </article>
              );
            })}
          </div>
        )}

        {/* Empty Search Results */}
        {!loading && filteredCandidates.length === 0 && (
          <div className="p-16 text-center text-slate-500 font-mono text-xs">
            No candidates matching the current filter.
          </div>
        )}
      </main>
    </div>
  );
}
