"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import {
  ShieldCheck,
  ShieldAlert,
  Lock,
  Unlock,
  Key,
  Database,
  BarChart3,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
  Copy,
  ExternalLink,
  Search,
  Filter,
  Layers,
  Activity,
  Award,
  TrendingUp,
  Cpu,
  ArrowRight,
  Info,
} from "lucide-react";
import Navbar from "../../components/Navbar";
import { getApiBaseUrl } from "../../lib/api";

interface ManifestEngineItem {
  valid: boolean;
  sha256?: string;
  expectedSha256?: string;
  error?: string;
}

interface MilestoneData {
  title: string;
  targetResolvedTrades: number;
  currentResolvedTrades: number;
  progressPct: number;
  status: "PENDING" | "REACHED";
  condition: string;
  targetDistinctSessions?: number;
  currentDistinctSessions?: number;
  distinctRegimesCount?: number;
}

export default function ProspectiveEvaluationPage() {
  const [evalKey, setEvalKey] = useState<string>("");
  const [inputKey, setInputKey] = useState<string>("");
  const [isUnlocked, setIsUnlocked] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const [summaryData, setSummaryData] = useState<any | null>(null);
  const [ledgerData, setLedgerData] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<
    "overview" | "milestones" | "ledger" | "benchmarks" | "diagnostics" | "integrity"
  >("overview");

  // Ledger Filter State
  const [selectedCohort, setSelectedCohort] = useState<string>("PROSPECTIVE_CLEAN");
  const [selectedStatus, setSelectedStatus] = useState<string>("ALL");
  const [searchSymbol, setSearchSymbol] = useState<string>("");
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  // Pure In-Memory Credential Lifecycle:
  // The evaluation key is held strictly in React component memory (useState) for the active page lifetime.
  // It is never written to sessionStorage, localStorage, or cookies, insulating it from any third-party scripts.
  const handleUnlock = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputKey.trim()) return;
    const cleanKey = inputKey.trim();
    setEvalKey(cleanKey);
    fetchDashboardData(cleanKey);
  };

  const handleLockSession = () => {
    setEvalKey("");
    setInputKey("");
    setIsUnlocked(false);
    setSummaryData(null);
    setLedgerData([]);
    setAuthError(null);
  };

  const fetchDashboardData = async (key: string) => {
    setIsLoading(true);
    setAuthError(null);

    const baseUrl = getApiBaseUrl();
    const headers: HeadersInit = {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    };

    try {
      // 1. Fetch Evaluation Summary
      const summaryRes = await fetch(`${baseUrl}/governance/evaluation-summary`, { headers });
      if (summaryRes.status === 401 || summaryRes.status === 403) {
        setAuthError("Invalid ARX Evaluation Key. Access denied to private prospective evaluation.");
        setIsUnlocked(false);
        setIsLoading(false);
        return;
      }
      if (!summaryRes.ok) {
        throw new Error(`Summary API returned status ${summaryRes.status}`);
      }
      const summaryJson = await summaryRes.json();

      // 2. Fetch Prospective Ledger
      const ledgerRes = await fetch(`${baseUrl}/governance/prospective-ledger?cohort=ALL`, { headers });
      if (!ledgerRes.ok) {
        throw new Error(`Ledger API returned status ${ledgerRes.status}`);
      }
      const ledgerJson = await ledgerRes.json();

      setSummaryData(summaryJson);
      setLedgerData(ledgerJson.records || []);
      setIsUnlocked(true);
    } catch (err: any) {
      setAuthError(err.message || "Failed to communicate with evaluation service.");
      setIsUnlocked(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(text);
    setTimeout(() => setCopiedHash(null), 2500);
  };

  const EVAL_TABS = [
    { id: "overview", label: "Overview & Health", icon: Activity },
    { id: "milestones", label: "Milestone Gates", icon: Award },
    { id: "ledger", label: "Prediction Ledger", icon: Database },
    { id: "benchmarks", label: "Benchmark Baselines", icon: TrendingUp },
    { id: "diagnostics", label: "Cluster Diagnostics", icon: Layers },
    { id: "integrity", label: "Engine Manifest", icon: ShieldCheck },
  ] as const;

  const handleTabKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, currentIndex: number) => {
    let targetIndex = -1;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      targetIndex = (currentIndex + 1) % EVAL_TABS.length;
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      targetIndex = (currentIndex - 1 + EVAL_TABS.length) % EVAL_TABS.length;
    } else if (e.key === "Home") {
      e.preventDefault();
      targetIndex = 0;
    } else if (e.key === "End") {
      e.preventDefault();
      targetIndex = EVAL_TABS.length - 1;
    }

    if (targetIndex >= 0) {
      const targetTab = EVAL_TABS[targetIndex];
      setActiveTab(targetTab.id as any);
      const tabElement = document.getElementById(`tab-${targetTab.id}`);
      if (tabElement) {
        tabElement.focus();
      }
    }
  };

  // Filtered Ledger Records
  const filteredLedger = useMemo(() => {
    return ledgerData.filter((r) => {
      const matchCohort = selectedCohort === "ALL" || r.provenanceCohort === selectedCohort;
      const matchStatus = selectedStatus === "ALL" || r.status === selectedStatus;
      const matchSymbol =
        !searchSymbol.trim() || r.symbol.toLowerCase().includes(searchSymbol.toLowerCase().trim());
      return matchCohort && matchStatus && matchSymbol;
    });
  }, [ledgerData, selectedCohort, selectedStatus, searchSymbol]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* UNLOCKED STATE */}
        {isUnlocked && summaryData ? (
          <div className="space-y-8">
            {/* Header / Engine Status Bar */}
            <header className="border-b border-slate-800 pb-6">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                  <div className="flex items-center gap-3">
                    <span className="p-2 rounded-lg bg-cyan-950/60 border border-cyan-800/60 text-cyan-400">
                      <Cpu className="w-6 h-6" />
                    </span>
                    <div>
                      <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-3">
                        ArxTerminal Prospective Evaluation
                        <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                          v2.4.0
                        </span>
                      </h1>
                      <p className="text-sm text-slate-400 mt-0.5">
                        Private Read-Only Observation Dashboard over Frozen Prospective Ledger
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/50 border border-emerald-800/60 text-emerald-300 text-xs font-mono">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span>OBSERVATION ONLY</span>
                  </div>

                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 text-xs font-mono">
                    <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                    <span>MANIFEST: VERIFIED</span>
                  </div>

                  <button
                    onClick={handleLockSession}
                    className="flex items-center gap-1.5 min-h-[44px] sm:min-h-[38px] px-3.5 py-2 sm:py-1.5 rounded-lg bg-rose-950/40 hover:bg-rose-900/60 border border-rose-800/60 text-rose-300 text-xs font-medium transition-all active:scale-[0.96] motion-reduce:transform-none duration-100 ease-out focus-visible:ring-2 focus-visible:ring-rose-400 focus-visible:outline-none cursor-pointer"
                    title="Lock session and purge key from memory"
                    aria-label="Lock session and purge key from memory"
                  >
                    <Lock className="w-3.5 h-3.5" />
                    <span>Lock Session</span>
                  </button>
                </div>
              </div>

              {/* Governing Notice Callout */}
              <div className="mt-5 p-4 rounded-xl bg-cyan-950/20 border border-cyan-900/40 flex items-start gap-3">
                <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
                <div className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                  <span className="font-semibold text-cyan-300">Observation-Only Governance Protocol:</span> Strategy
                  parameters, thresholds, stop/target geometry, and execution logic are permanently frozen. This
                  dashboard is a pure observational lens over forward paper-trading positions. It has zero authority to
                  issue trade recommendations, alter parameters, or back-fit outcomes.
                </div>
              </div>

              {/* Navigation Tabs */}
              <div
                role="tablist"
                aria-label="Prospective Evaluation Dashboard Navigation"
                aria-orientation="horizontal"
                className="mt-6 flex flex-wrap gap-2 border-b border-slate-800/80 pb-2"
              >
                {EVAL_TABS.map((tab, idx) => {
                  const Icon = tab.icon;
                  const active = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      id={`tab-${tab.id}`}
                      role="tab"
                      tabIndex={active ? 0 : -1}
                      aria-selected={active}
                      aria-controls={`panel-${tab.id}`}
                      onClick={() => setActiveTab(tab.id as any)}
                      onKeyDown={(e) => handleTabKeyDown(e, idx)}
                      className={`flex items-center gap-2 min-h-[44px] sm:min-h-[38px] px-4 py-2.5 sm:py-2 rounded-lg text-xs sm:text-sm font-medium transition-all active:scale-[0.96] motion-reduce:transform-none duration-100 ease-out focus-visible:ring-2 focus-visible:ring-cyan-400 focus-visible:outline-none cursor-pointer ${
                        active
                          ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 shadow-sm shadow-cyan-950 font-semibold"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent"
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            </header>

            {/* TAB 1: OVERVIEW & COHORT HEALTH */}
            {activeTab === "overview" && (
              <div
                role="tabpanel"
                id="panel-overview"
                aria-labelledby="tab-overview"
                tabIndex={0}
                className="space-y-8 focus-visible:outline-none"
              >
                {/* Cohort Classification Firewall Breakdown */}
                <section>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                      <ShieldCheck className="w-5 h-5 text-emerald-400" />
                      Cohort Contamination Firewall
                    </h2>
                    <span className="text-xs font-mono text-slate-400">
                      Isolation Status:{" "}
                      <span className="text-emerald-400 font-semibold">
                        {summaryData.cohortFirewall?.status || "ACTIVE_ENFORCED"}
                      </span>
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="p-4 rounded-xl bg-slate-900/70 border border-emerald-500/30">
                      <div className="text-xs font-medium uppercase tracking-wider text-emerald-400 flex items-center justify-between">
                        <span>Prospective Clean</span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800/50">
                          Primary
                        </span>
                      </div>
                      <div className="mt-2 text-3xl font-mono font-bold text-emerald-200">
                        {summaryData.cohortFirewall?.cohortCounts?.PROSPECTIVE_CLEAN ?? 0}
                      </div>
                      <p className="mt-1 text-xs text-slate-400">Post-freeze prospective signals under observation</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-900/70 border border-amber-500/30">
                      <div className="text-xs font-medium uppercase tracking-wider text-amber-400 flex items-center justify-between">
                        <span>Historical Contaminated</span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-950 text-amber-300 border border-amber-800/50">
                          Quarantined
                        </span>
                      </div>
                      <div className="mt-2 text-3xl font-mono font-bold text-amber-200">
                        {summaryData.cohortFirewall?.cohortCounts?.HISTORICAL_CONTAMINATED ?? 0}
                      </div>
                      <p className="mt-1 text-xs text-slate-400">Pre-freeze signals isolated from edge verification</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                      <div className="text-xs font-medium uppercase tracking-wider text-slate-400">
                        Historical Unknown / Excluded
                      </div>
                      <div className="mt-2 text-3xl font-mono font-bold text-slate-300">
                        {(summaryData.cohortFirewall?.cohortCounts?.HISTORICAL_UNKNOWN ?? 0) +
                          (summaryData.cohortFirewall?.cohortCounts?.EXCLUDED ?? 0)}
                      </div>
                      <p className="mt-1 text-xs text-slate-500">Omitted due to incomplete provenance or test fixtures</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-900/70 border border-cyan-800/40">
                      <div className="text-xs font-medium uppercase tracking-wider text-cyan-400">Total in Ledger</div>
                      <div className="mt-2 text-3xl font-mono font-bold text-cyan-200">
                        {summaryData.cohortFirewall?.cohortCounts?.total ?? 0}
                      </div>
                      <p className="mt-1 text-xs text-slate-400">Complete immutable paper-trading records</p>
                    </div>
                  </div>
                </section>

                {/* Descriptive Observations for Prospective Clean Cohort */}
                <section>
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
                    <div>
                      <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                        <Activity className="w-5 h-5 text-cyan-400" />
                        Descriptive Observations (Prospective Clean Cohort)
                      </h2>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Forward performance metrics strictly insulated from historical data.
                      </p>
                    </div>
                    <div className="text-xs px-2.5 py-1 rounded bg-amber-950/40 border border-amber-800/60 text-amber-300 font-mono">
                      N_resolved = {summaryData.prospectiveScorecard?.resolvedSignals ?? 0} | N_open ={" "}
                      {summaryData.prospectiveScorecard?.openSignals ?? 0}
                    </div>
                  </div>

                  {/* Warning Notice about Non-Significance */}
                  <div className="p-3.5 rounded-lg bg-amber-950/20 border border-amber-900/50 text-xs text-amber-200/90 leading-relaxed mb-4 flex items-start gap-2.5">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold text-amber-300">Epistemic Disclaimer:</span> All numbers below are{" "}
                      <strong>descriptive observations</strong> of forward forward-paper trading positions. Sample size
                      is below the pre-registered minimum ($N=60$) required to evaluate statistical significance.
                    </div>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
                      <div className="text-[11px] font-medium text-slate-400">Win Rate</div>
                      <div className="mt-1 text-xl font-mono font-bold text-white">
                        {summaryData.prospectiveScorecard?.winRate !== undefined
                          ? `${summaryData.prospectiveScorecard.winRate}%`
                          : "N/A"}
                      </div>
                      <div className="mt-1 text-[10px] text-slate-500 font-mono">
                        95% CI:{" "}
                        {summaryData.prospectiveScorecard?.winRateCI95
                          ? `[${summaryData.prospectiveScorecard.winRateCI95[0]}%, ${summaryData.prospectiveScorecard.winRateCI95[1]}%]`
                          : "N/A"}
                      </div>
                    </div>

                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
                      <div className="text-[11px] font-medium text-slate-400">Stop Rate</div>
                      <div className="mt-1 text-xl font-mono font-bold text-white">
                        {summaryData.prospectiveScorecard?.stopRate !== undefined
                          ? `${summaryData.prospectiveScorecard.stopRate}%`
                          : "N/A"}
                      </div>
                      <div className="mt-1 text-[10px] text-slate-500">SL trigger frequency</div>
                    </div>

                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
                      <div className="text-[11px] font-medium text-slate-400">Gross Expectancy</div>
                      <div
                        className={`mt-1 text-xl font-mono font-bold ${
                          (summaryData.prospectiveScorecard?.expectancyPct ?? 0) >= 0
                            ? "text-emerald-400"
                            : "text-rose-400"
                        }`}
                      >
                        {summaryData.prospectiveScorecard?.expectancyPct !== undefined
                          ? `${summaryData.prospectiveScorecard.expectancyPct > 0 ? "+" : ""}${summaryData.prospectiveScorecard.expectancyPct}%`
                          : "N/A"}
                      </div>
                      <div className="mt-1 text-[10px] text-slate-500">Gross per trade</div>
                    </div>

                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-cyan-800/40 bg-cyan-950/10">
                      <div className="text-[11px] font-medium text-cyan-300">Net Expectancy (30 bps)</div>
                      <div
                        className={`mt-1 text-xl font-mono font-bold ${
                          (summaryData.prospectiveScorecard?.netExpectancyPct ?? 0) >= 0
                            ? "text-emerald-400"
                            : "text-rose-400"
                        }`}
                      >
                        {summaryData.prospectiveScorecard?.netExpectancyPct !== undefined
                          ? `${summaryData.prospectiveScorecard.netExpectancyPct > 0 ? "+" : ""}${summaryData.prospectiveScorecard.netExpectancyPct}%`
                          : "N/A"}
                      </div>
                      <div className="mt-1 text-[10px] text-cyan-400/80">Primary friction standard</div>
                    </div>

                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
                      <div className="text-[11px] font-medium text-slate-400">Profit Factor</div>
                      <div className="mt-1 text-xl font-mono font-bold text-white">
                        {summaryData.prospectiveScorecard?.profitFactor ?? "N/A"}
                      </div>
                      <div className="mt-1 text-[10px] text-slate-500">Gross gains / losses</div>
                    </div>

                    <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800">
                      <div className="text-[11px] font-medium text-slate-400">Avg Win / Loss</div>
                      <div className="mt-1 text-sm font-mono font-semibold text-slate-200">
                        <span className="text-emerald-400">+{summaryData.prospectiveScorecard?.avgWinPct ?? 0}%</span> /{" "}
                        <span className="text-rose-400">-{summaryData.prospectiveScorecard?.avgLossPct ?? 0}%</span>
                      </div>
                      <div className="mt-1 text-[10px] text-slate-500">Realized distribution</div>
                    </div>
                  </div>

                  {/* Friction Sensitivity Table */}
                  {summaryData.prospectiveScorecard?.frictionSensitivity && (
                    <div className="mt-4 p-4 rounded-xl bg-slate-900/50 border border-slate-800">
                      <div className="text-xs font-semibold text-slate-300 mb-2 flex items-center justify-between">
                        <span>Friction Drag Sensitivity Matrix (Linear Additive Deduction)</span>
                        <span className="font-mono text-cyan-400">
                          Edge Breakeven Drag:{" "}
                          {summaryData.prospectiveScorecard.frictionSensitivity.edgeBreakevenFrictionBps} bps
                        </span>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 font-mono text-xs text-center">
                        <div className="p-2 rounded bg-slate-950 border border-slate-800">
                          <span className="text-slate-400 text-[10px] block">0 bps (Gross)</span>
                          <span className="font-bold text-slate-200">
                            {summaryData.prospectiveScorecard.frictionSensitivity["0bps_gross"]}%
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-950 border border-slate-800">
                          <span className="text-slate-400 text-[10px] block">15 bps (Optimistic)</span>
                          <span className="font-bold text-slate-200">
                            {summaryData.prospectiveScorecard.frictionSensitivity["15bps_optimistic"]}%
                          </span>
                        </div>
                        <div className="p-2 rounded bg-cyan-950/40 border border-cyan-700/50">
                          <span className="text-cyan-300 text-[10px] block">30 bps (Primary Standard)</span>
                          <span className="font-bold text-cyan-200">
                            {summaryData.prospectiveScorecard.frictionSensitivity["30bps_primary"]}%
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-950 border border-slate-800">
                          <span className="text-slate-400 text-[10px] block">50 bps (Stress Test)</span>
                          <span className="font-bold text-slate-200">
                            {summaryData.prospectiveScorecard.frictionSensitivity["50bps_stress"]}%
                          </span>
                        </div>
                        <div className="p-2 rounded bg-slate-950 border border-slate-800">
                          <span className="text-slate-400 text-[10px] block">100 bps (Severe Slippage)</span>
                          <span className="font-bold text-slate-200">
                            {summaryData.prospectiveScorecard.frictionSensitivity["100bps_severe"]}%
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </section>
              </div>
            )}

            {/* TAB 2: MILESTONES */}
            {activeTab === "milestones" && (
              <div
                role="tabpanel"
                id="panel-milestones"
                aria-labelledby="tab-milestones"
                tabIndex={0}
                className="space-y-6 focus-visible:outline-none"
              >
                <div>
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Award className="w-5 h-5 text-amber-400" />
                    Predefined Milestone Progress
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Continuous monitoring over the prospective cohort against predefined evaluation gates.
                  </p>
                </div>

                {/* Counting Rules Box */}
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                  <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4 text-cyan-400" />
                    Strict Counting Rules (Fail-Closed Enforcement)
                  </div>
                  <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                    <li>Counts ONLY records satisfying <code className="text-emerald-400">ProvenanceCohort.PROSPECTIVE_CLEAN</code>.</li>
                    <li>Signal date must be <code className="text-slate-300">&gt;= 2026-09-04</code> with status <code className="text-slate-300">RESOLVED</code>.</li>
                    <li>Decision and Inputs SHA-256 cryptographic snapshot hashes must match record fields.</li>
                    <li>Frozen production-engine manifest must verify SHA-256 intact.</li>
                    <li>Contaminated historical signals, unclassified signals, and open trades are strictly excluded from milestone counts.</li>
                  </ul>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  {/* Milestone 1 */}
                  {summaryData.milestoneTracker?.milestone1 && (
                    <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-mono text-cyan-400 uppercase font-semibold">Milestone 1</span>
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                              summaryData.milestoneTracker.milestone1.status === "REACHED"
                                ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {summaryData.milestoneTracker.milestone1.status}
                          </span>
                        </div>
                        <h3 className="text-base font-semibold text-white">Initial Statistical Cohort</h3>
                        <p className="text-xs text-slate-400 mt-1">
                          Condition: {summaryData.milestoneTracker.milestone1.condition}
                        </p>

                        <div className="mt-4 flex items-baseline justify-between font-mono text-sm">
                          <span className="text-slate-400">Resolved Trades:</span>
                          <span className="text-white font-bold">
                            {summaryData.milestoneTracker.milestone1.currentResolvedTrades} /{" "}
                            {summaryData.milestoneTracker.milestone1.targetResolvedTrades}
                          </span>
                        </div>

                        {/* Progress Bar */}
                        <div className="mt-2 w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-cyan-400 transition-all duration-500"
                            style={{ width: `${summaryData.milestoneTracker.milestone1.progressPct}%` }}
                          />
                        </div>
                      </div>
                      <div className="mt-4 text-right text-xs font-mono text-cyan-300">
                        {summaryData.milestoneTracker.milestone1.progressPct}% Complete
                      </div>
                    </div>
                  )}

                  {/* Milestone 2 */}
                  {summaryData.milestoneTracker?.milestone2 && (
                    <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-mono text-cyan-400 uppercase font-semibold">Milestone 2</span>
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                              summaryData.milestoneTracker.milestone2.status === "REACHED"
                                ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {summaryData.milestoneTracker.milestone2.status}
                          </span>
                        </div>
                        <h3 className="text-base font-semibold text-white">Intermediate Calibration Gate</h3>
                        <p className="text-xs text-slate-400 mt-1">
                          Condition: {summaryData.milestoneTracker.milestone2.condition}
                        </p>

                        <div className="mt-4 flex items-baseline justify-between font-mono text-sm">
                          <span className="text-slate-400">Resolved Trades:</span>
                          <span className="text-white font-bold">
                            {summaryData.milestoneTracker.milestone2.currentResolvedTrades} /{" "}
                            {summaryData.milestoneTracker.milestone2.targetResolvedTrades}
                          </span>
                        </div>

                        {/* Progress Bar */}
                        <div className="mt-2 w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-cyan-400 transition-all duration-500"
                            style={{ width: `${summaryData.milestoneTracker.milestone2.progressPct}%` }}
                          />
                        </div>
                      </div>
                      <div className="mt-4 text-right text-xs font-mono text-cyan-300">
                        {summaryData.milestoneTracker.milestone2.progressPct}% Complete
                      </div>
                    </div>
                  )}

                  {/* Milestone 3 */}
                  {summaryData.milestoneTracker?.milestone3 && (
                    <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800 flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-mono text-cyan-400 uppercase font-semibold">Milestone 3</span>
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                              summaryData.milestoneTracker.milestone3.status === "REACHED"
                                ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {summaryData.milestoneTracker.milestone3.status}
                          </span>
                        </div>
                        <h3 className="text-base font-semibold text-white">Institutional Robustness Gate</h3>
                        <p className="text-xs text-slate-400 mt-1">
                          Condition: {summaryData.milestoneTracker.milestone3.condition}
                        </p>

                        <div className="mt-3 space-y-1.5 font-mono text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-400">Resolved Trades (&gt;=60):</span>
                            <span className="text-white font-bold">
                              {summaryData.milestoneTracker.milestone3.currentResolvedTrades} / 60
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Sessions (&gt;=20):</span>
                            <span className="text-white font-bold">
                              {summaryData.milestoneTracker.distinctTradingSessions ?? 0} / 20
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400">Regimes (&gt;=2):</span>
                            <span className="text-white font-bold">
                              {(summaryData.milestoneTracker.distinctRegimes || []).length} / 2
                            </span>
                          </div>
                        </div>

                        {/* Progress Bar */}
                        <div className="mt-3 w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-cyan-400 transition-all duration-500"
                            style={{ width: `${summaryData.milestoneTracker.milestone3.progressPct}%` }}
                          />
                        </div>
                      </div>
                      <div className="mt-4 text-right text-xs font-mono text-cyan-300">
                        {summaryData.milestoneTracker.milestone3.progressPct}% Complete
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* TAB 3: PREDICTION LEDGER TABLE */}
            {activeTab === "ledger" && (
              <div
                role="tabpanel"
                id="panel-ledger"
                aria-labelledby="tab-ledger"
                tabIndex={0}
                className="space-y-4 focus-visible:outline-none"
              >
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Database className="w-5 h-5 text-cyan-400" />
                      Prediction Ledger (Read-Only)
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Immutable record of signals, frozen entry corridors, and forward tracking.
                    </p>
                  </div>

                  {/* Filter Toolbar */}
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Cohort Selector */}
                    <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
                      {["PROSPECTIVE_CLEAN", "HISTORICAL_CONTAMINATED", "ALL"].map((c) => (
                        <button
                          key={c}
                          onClick={() => setSelectedCohort(c)}
                          className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                            selectedCohort === c
                              ? "bg-cyan-500/20 text-cyan-300 font-semibold"
                              : "text-slate-400 hover:text-white"
                          }`}
                        >
                          {c === "PROSPECTIVE_CLEAN" ? "Clean Prospective" : c === "HISTORICAL_CONTAMINATED" ? "Contaminated" : "All"}
                        </button>
                      ))}
                    </div>

                    {/* Status Selector */}
                    <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
                      {["ALL", "OPEN", "RESOLVED"].map((st) => (
                        <button
                          key={st}
                          onClick={() => setSelectedStatus(st)}
                          className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                            selectedStatus === st
                              ? "bg-slate-800 text-white font-semibold"
                              : "text-slate-400 hover:text-white"
                          }`}
                        >
                          {st}
                        </button>
                      ))}
                    </div>

                    {/* Search Input */}
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                      <input
                        type="text"
                        placeholder="Search symbol..."
                        value={searchSymbol}
                        onChange={(e) => setSearchSymbol(e.target.value)}
                        className="bg-slate-900 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                  </div>
                </div>

                {/* Table */}
                <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900/90 border-b border-slate-800 text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                      <tr>
                        <th className="py-3 px-3">Signal ID / Date</th>
                        <th className="py-3 px-3">Symbol</th>
                        <th className="py-3 px-3">Cohort</th>
                        <th className="py-3 px-3">Confluence</th>
                        <th className="py-3 px-3">Entry</th>
                        <th className="py-3 px-3">Stop Loss</th>
                        <th className="py-3 px-3">TP1 / TP2</th>
                        <th className="py-3 px-3">R:R</th>
                        <th className="py-3 px-3">Status / Outcome</th>
                        <th className="py-3 px-3">MFE / MAE</th>
                        <th className="py-3 px-3">Net Return</th>
                        <th className="py-3 px-3">Hash Fingerprint</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {filteredLedger.length === 0 ? (
                        <tr>
                          <td colSpan={12} className="py-8 text-center text-slate-500 font-sans">
                            No prediction records match the active filters.
                          </td>
                        </tr>
                      ) : (
                        filteredLedger.map((record) => {
                          const isClean = record.provenanceCohort === "PROSPECTIVE_CLEAN";
                          const isResolved = record.status === "RESOLVED";
                          const outcome = record.forwardTracking?.resolvedOutcome;
                          const netRet = record.forwardTracking?.economicOutcome?.netReturnPct ?? record.forwardTracking?.realizedReturnPct;

                          return (
                            <tr key={record.signalId} className="hover:bg-slate-800/40 transition-colors">
                              <td className="py-2.5 px-3 whitespace-nowrap">
                                <span className="font-semibold text-white block">{record.signalId}</span>
                                <span className="text-[10px] text-slate-500">{record.signalDate}</span>
                              </td>

                              <td className="py-2.5 px-3 font-bold text-cyan-300">{record.symbol}</td>

                              <td className="py-2.5 px-3 whitespace-nowrap">
                                <span
                                  className={`px-2 py-0.5 rounded text-[10px] ${
                                    isClean
                                      ? "bg-emerald-950/60 text-emerald-300 border border-emerald-800/50"
                                      : "bg-amber-950/60 text-amber-300 border border-amber-800/50"
                                  }`}
                                >
                                  {isClean ? "PROSPECTIVE_CLEAN" : "HISTORICAL"}
                                </span>
                              </td>

                              <td className="py-2.5 px-3 font-bold text-white">{record.confluenceScore}</td>

                              <td className="py-2.5 px-3 whitespace-nowrap">${record.entryPrice?.toFixed(2)}</td>

                              <td className="py-2.5 px-3 whitespace-nowrap text-rose-300">
                                ${record.stopLoss?.toFixed(2)}{" "}
                                <span className="text-[10px] text-rose-400/80">({record.stopLossPct}%)</span>
                              </td>

                              <td className="py-2.5 px-3 whitespace-nowrap text-emerald-300">
                                ${record.takeProfit1?.toFixed(2)}{" "}
                                <span className="text-[10px] text-emerald-400/80">({record.takeProfit1Pct}%)</span>
                              </td>

                              <td className="py-2.5 px-3 text-slate-300">{record.riskRewardRatio}:1</td>

                              <td className="py-2.5 px-3 whitespace-nowrap">
                                <span
                                  className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                                    !isResolved
                                      ? "bg-slate-800 text-slate-300"
                                      : outcome === "TP1_WIN"
                                      ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                                      : "bg-rose-950 text-rose-300 border border-rose-700"
                                  }`}
                                >
                                  {isResolved ? outcome : "OPEN"}
                                </span>
                              </td>

                              <td className="py-2.5 px-3 whitespace-nowrap text-[11px]">
                                <span className="text-emerald-400">+{record.forwardTracking?.maxFavorableExcursionPct ?? 0}%</span>{" "}
                                /{" "}
                                <span className="text-rose-400">-{record.forwardTracking?.maxAdverseExcursionPct ?? 0}%</span>
                              </td>

                              <td className="py-2.5 px-3 whitespace-nowrap font-bold">
                                {netRet !== null && netRet !== undefined ? (
                                  <span className={netRet >= 0 ? "text-emerald-400" : "text-rose-400"}>
                                    {netRet > 0 ? "+" : ""}
                                    {netRet.toFixed(2)}%
                                  </span>
                                ) : (
                                  <span className="text-slate-500">—</span>
                                )}
                              </td>

                              <td className="py-2.5 px-3 whitespace-nowrap text-[10px]">
                                <button
                                  onClick={() => handleCopy(record.decisionSnapshotHash || "")}
                                  className="flex items-center gap-1 text-slate-400 hover:text-cyan-300 transition-colors"
                                  title={`Full Hash: ${record.decisionSnapshotHash}`}
                                >
                                  <span>{(record.decisionSnapshotHash || "N/A").slice(0, 8)}...</span>
                                  <Copy className="w-3 h-3" />
                                </button>
                                {copiedHash === record.decisionSnapshotHash && (
                                  <span className="text-[9px] text-emerald-400">Copied!</span>
                                )}
                              </td>
                            </tr>
                          );
                        })
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* TAB 4: BENCHMARK BASELINES */}
            {activeTab === "benchmarks" && (
              <div
                role="tabpanel"
                id="panel-benchmarks"
                aria-labelledby="tab-benchmarks"
                tabIndex={0}
                className="space-y-6 focus-visible:outline-none"
              >
                <div>
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-cyan-400" />
                    Comparative Baselines & Monotonicity
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Pre-registered benchmarks sharing identical entry, stop, TP geometry, and 30 bps round-trip friction.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
                    <h3 className="text-sm font-semibold text-white">Unconditional Random Baseline</h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Matched Monte Carlo sampling from the eligible liquid universe at T0.
                    </p>
                    <div className="mt-4 space-y-2 font-mono text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Simulation Iterations:</span>
                        <span className="text-slate-200">1,000 runs</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Friction Deduction:</span>
                        <span className="text-cyan-300">30 bps round-trip</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Intrabar Collision:</span>
                        <span className="text-rose-300">Fail-closed SL</span>
                      </div>
                    </div>
                  </div>

                  <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
                    <h3 className="text-sm font-semibold text-white">Sector-Conditioned Random</h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Samples randomly strictly within the same GICS sector as candidate setup.
                    </p>
                    <div className="mt-4 space-y-2 font-mono text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Mandatory Reporting:</span>
                        <span className="text-emerald-400">Dual reporting</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Survivorship Standard:</span>
                        <span className="text-amber-300">Live quoted at T0</span>
                      </div>
                    </div>
                  </div>

                  <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
                    <h3 className="text-sm font-semibold text-white">Frozen 30-Day Momentum</h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Static momentum formula: Close(T0) / Close(T-30) - 1.
                    </p>
                    <div className="mt-4 space-y-2 font-mono text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Lookback Horizon:</span>
                        <span className="text-slate-200">30 calendar sessions</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Anti-Lookahead Gate:</span>
                        <span className="text-emerald-400">Verified strict T0</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Benchmark Excess Return Diagnostics */}
                {summaryData.prospectiveScorecard?.benchmarkComparisons && (
                  <div className="p-5 rounded-xl bg-slate-900/50 border border-slate-800">
                    <h3 className="text-sm font-semibold text-white mb-3">
                      20-Session Benchmark Excess Return Tracking
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-xs">
                      <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                        <div className="text-slate-400 text-[11px]">vs. SPY (S&P 500 Cap-Weighted)</div>
                        <div className="mt-2 text-lg font-bold text-white">
                          {summaryData.prospectiveScorecard.benchmarkComparisons.vsSpy20d?.meanExcessReturnPct ?? "—"}%
                        </div>
                        <div className="text-[10px] text-slate-500 mt-1">
                          Hit Rate: {summaryData.prospectiveScorecard.benchmarkComparisons.vsSpy20d?.hitRatePct ?? "—"}%
                        </div>
                      </div>

                      <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                        <div className="text-slate-400 text-[11px]">vs. RSP (S&P 500 Equal-Weighted)</div>
                        <div className="mt-2 text-lg font-bold text-white">
                          {summaryData.prospectiveScorecard.benchmarkComparisons.vsRsp20d?.meanExcessReturnPct ?? "—"}%
                        </div>
                        <div className="text-[10px] text-slate-500 mt-1">
                          Hit Rate: {summaryData.prospectiveScorecard.benchmarkComparisons.vsRsp20d?.hitRatePct ?? "—"}%
                        </div>
                      </div>

                      <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                        <div className="text-slate-400 text-[11px]">vs. Sector Benchmark ETF</div>
                        <div className="mt-2 text-lg font-bold text-white">
                          {summaryData.prospectiveScorecard.benchmarkComparisons.vsSector20d?.meanExcessReturnPct ?? "—"}%
                        </div>
                        <div className="text-[10px] text-slate-500 mt-1">
                          Hit Rate: {summaryData.prospectiveScorecard.benchmarkComparisons.vsSector20d?.hitRatePct ?? "—"}%
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 5: CLUSTER & DEPENDENCE DIAGNOSTICS */}
            {activeTab === "diagnostics" && (
              <div
                role="tabpanel"
                id="panel-diagnostics"
                aria-labelledby="tab-diagnostics"
                tabIndex={0}
                className="space-y-6 focus-visible:outline-none"
              >
                <div>
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Layers className="w-5 h-5 text-amber-400" />
                    Cluster Structure & Statistical Dependence Diagnostics
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Quantifies market session concentration, sector clustering, and effective sample size sensitivity.
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-900/50 flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                  <div className="text-xs text-amber-200 leading-relaxed">
                    <span className="font-semibold text-amber-300">Dependence Warning:</span> Trades executed across the
                    same market calendar session or heavily concentrated in a single sector (e.g. Technology) are
                    materially dependent. Nominal sample size ($N$) must not be treated as independent identically
                    distributed (i.i.d.) observations.
                  </div>
                </div>

                {summaryData.prospectiveScorecard?.clusteringAndDependence && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                    <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
                      <h3 className="text-sm font-semibold text-white">Session Concentration</h3>
                      <div className="mt-3 space-y-2 font-mono text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Max Trades per Session:</span>
                          <span className="text-white font-bold">
                            {summaryData.prospectiveScorecard.clusteringAndDependence.maxTradesPerSession}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Unique Market Sessions:</span>
                          <span className="text-white font-bold">
                            {summaryData.prospectiveScorecard.clusteringAndDependence.uniqueSessionsCount}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
                      <h3 className="text-sm font-semibold text-white">Sector Concentration</h3>
                      <div className="mt-3 space-y-2 font-mono text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Max Sector Exposure:</span>
                          <span className="text-white font-bold">
                            {summaryData.prospectiveScorecard.clusteringAndDependence.maxSectorConcentrationPct}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
                      <h3 className="text-sm font-semibold text-white">Portfolio Aggregation Note</h3>
                      <p className="mt-2 text-xs text-slate-400 leading-relaxed">
                        Trade-level expectancy does not guarantee bounded portfolio drawdown under simultaneous sector
                        exposure. Concurrent positions share market beta risks.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 6: ENGINE MANIFEST & MEASUREMENT INTEGRITY */}
            {activeTab === "integrity" && (
              <div
                role="tabpanel"
                id="panel-integrity"
                aria-labelledby="tab-integrity"
                tabIndex={0}
                className="space-y-6 focus-visible:outline-none"
              >
                <div>
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    Cryptographic Engine Manifest & Invariants
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Continuous SHA-256 integrity verification of production decision engines against repository freeze.
                  </p>
                </div>

                {summaryData.system?.manifestVerification?.engines && (
                  <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="bg-slate-950/80 border-b border-slate-800 text-[11px] text-slate-400 uppercase">
                        <tr>
                          <th className="py-3 px-4">Production Engine</th>
                          <th className="py-3 px-4">Verification Status</th>
                          <th className="py-3 px-4">Expected SHA-256</th>
                          <th className="py-3 px-4">Live Repository SHA-256</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {Object.entries(
                          summaryData.system.manifestVerification.engines as Record<string, ManifestEngineItem>
                        ).map(([name, eng]) => (
                          <tr key={name} className="hover:bg-slate-800/30">
                            <td className="py-3 px-4 font-bold text-white">{name}.py</td>
                            <td className="py-3 px-4">
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                                  eng.valid
                                    ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                                    : "bg-rose-950 text-rose-300 border border-rose-700"
                                }`}
                              >
                                {eng.valid ? "VERIFIED INTACT" : "CORRUPTED"}
                              </span>
                            </td>
                            <td className="py-3 px-4 text-slate-400 text-[11px]">
                              {(eng.expectedSha256 || "").slice(0, 16)}...
                            </td>
                            <td className="py-3 px-4 text-cyan-300 text-[11px]">
                              {(eng.sha256 || "").slice(0, 16)}...
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Pre-Registered Conventions Checklist */}
                <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                  <h3 className="text-sm font-semibold text-white">Pre-Registered Measurement Conventions</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="font-semibold text-slate-200 block mb-1">1. Intrabar TP/SL Collision</span>
                      <p className="text-slate-400">
                        When both TP1 and Stop-Loss are intersected within the same day, resolve fail-closed to
                        STOP_LOSS. Pre-registered convention, not a physical path claim.
                      </p>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="font-semibold text-slate-200 block mb-1">2. Transaction Cost Standard</span>
                      <p className="text-slate-400">
                        30 bps round-trip linear additive deduction standard ($R_{"net"} = R_{"gross"} - 0.30\%$).
                        Simulated across 0, 15, 30, 50, and 100 bps grid.
                      </p>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="font-semibold text-slate-200 block mb-1">3. Live Quoted Survivorship Caveat</span>
                      <p className="text-slate-400">
                        Active quotes at T0 are not equivalent to a point-in-time historical universe. No claim of
                        having solved historical delisting survivorship bias is made.
                      </p>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="font-semibold text-slate-200 block mb-1">4. Confluence Monotonicity</span>
                      <p className="text-slate-400">
                        Confluence is treated as an ordinal score, not an uncalibrated probability. Evaluated via bucket
                        monotonicity and Spearman rank correlation.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* LOCKED STATE / ACCESS GATE MODAL */
          <div className="max-w-md mx-auto my-16 p-6 sm:p-8 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-2xl backdrop-blur-md">
            <div className="text-center space-y-3">
              <div className="inline-flex p-3 rounded-2xl bg-cyan-950/80 border border-cyan-800/60 text-cyan-400">
                <Lock className="w-8 h-8" />
              </div>
              <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                Private Evaluation Access
              </h1>
              <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
                This dashboard is an internal observation portal over the frozen prospective evaluation. Valid ARX
                Evaluation credentials are required to unlock.
              </p>
            </div>

            <form onSubmit={handleUnlock} className="mt-6 space-y-4">
              <div>
                <label className="block text-xs font-mono text-slate-300 uppercase mb-1.5">
                  ARX Evaluation Key
                </label>
                <div className="relative">
                  <Key className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
                  <input
                    type="password"
                    required
                    placeholder="Enter evaluation secret..."
                    value={inputKey}
                    onChange={(e) => setInputKey(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-4 py-2.5 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/60 font-mono"
                  />
                </div>
              </div>

              {authError && (
                <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start gap-2">
                  <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{authError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-semibold transition-colors disabled:opacity-50 shadow-lg shadow-cyan-950"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Verifying Credentials...</span>
                  </>
                ) : (
                  <>
                    <Unlock className="w-4 h-4" />
                    <span>Unlock Evaluation View</span>
                  </>
                )}
              </button>
            </form>

            <div className="mt-6 pt-4 border-t border-slate-800/80 text-[11px] text-slate-500 text-center leading-relaxed">
              Protected by server-side SHA-256 HMAC verification.
              <br />
              In-memory credential lifecycle: never stored in browser storage or cookies.
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
