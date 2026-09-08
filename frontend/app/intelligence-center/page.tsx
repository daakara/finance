"use client";

import React, { useState, Suspense } from "react";
import Link from "next/link";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import IntelligenceHeader from "../../components/ui/IntelligenceHeader";
import IntelligenceMetricCard from "../../components/ui/IntelligenceMetricCard";
import SeverityBadge from "../../components/ui/SeverityBadge";
import HorizonCard from "../../components/ui/HorizonCard";
import RelatedArtifactsPanel, { RelatedArtifactLink } from "../../components/ui/RelatedArtifactsPanel";
import CertificationPanel, { CertificationGateItem } from "../../components/ui/CertificationPanel";
import IntelligenceLoadingState from "../../components/ui/IntelligenceLoadingState";
import IntelligenceEmptyState from "../../components/ui/IntelligenceEmptyState";
import IntelligenceErrorState from "../../components/ui/IntelligenceErrorState";
import IntelligenceSuccessState from "../../components/ui/IntelligenceSuccessState";
import { generateExecutiveBriefing } from "../../lib/narrative/executiveNarrativeEngine";

// Structured 3-Tier Categorized Directory (Strategic, Operational, Evidence & Audit)
const STRATEGIC_CENTERS = [
  { href: "/committee-intelligence", name: "Committee Intelligence", metric: "ODEI 86.4", status: "HEALTHY", desc: "Committee scorecards, composition, and cognitive bias metrics." },
  { href: "/learning-intelligence", name: "Learning Intelligence", metric: "+3.8/qtr", status: "HEALTHY", desc: "Institutional memory, outcome feedback, and learning velocity." },
  { href: "/risks-and-groupthink", name: "Risks & Groupthink", metric: "Low Stress (12%)", status: "HEALTHY", desc: "Suppressed dissent detection, stress tests, and correlation shocks." },
  { href: "/simulation-intelligence", name: "Futures & Simulation", metric: "4 Regimes Certified", status: "CERTIFIED", desc: "Institutional forward projections and counterfactual analysis." },
];

const OPERATIONAL_CENTERS = [
  { href: "/governance-center", name: "Governance Center", metric: "98.4% Compliant", status: "HEALTHY", desc: "Real-time policy bounds, SLA timers, and compliance audits." },
  { href: "/resilience-intelligence", name: "Resilience Intelligence", metric: "RTO 4.8s", status: "CERTIFIED", desc: "Autonomous failover, snapshot integrity, and survivability." },
  { href: "/autonomous-governance", name: "Autonomous Governance", metric: "10/10 Invariants", status: "CERTIFIED", desc: "Safety gates, automated execution controls, and override logs." },
  { href: "/action-center", name: "Action Center", metric: "5 Priority Actions", status: "WARNING", desc: "Unified triage queue: alerts, approvals, and runbooks." },
];

const EVIDENCE_CENTERS = [
  { href: "/decision-explorer", name: "Decision Explorer", metric: "24 Decisions", status: "HEALTHY", desc: "Searchable ledger of ratified institutional choices." },
  { href: "/dissent-explorer", name: "Dissent Explorer", metric: "8 Dissents", status: "HEALTHY", desc: "Formal minority opinions and challenge lineages." },
  { href: "/audit-explorer", name: "Audit Explorer", metric: "100% Cryptographic", status: "CERTIFIED", desc: "Immutable SHA-256 reconstruction and replay proof." },
  { href: "/graph-explorer", name: "Universal Graph", metric: "9 Node Types", status: "CERTIFIED", desc: "End-to-end causal relationship and lineage explorer." },
];

const SAMPLE_RELATED_ARTIFACTS: RelatedArtifactLink[] = [
  { id: "DEC-001", type: "DECISION", title: "Strategic Tech Allocation ($12M)", href: "/decision-explorer?decisionId=DEC-001", badge: "Ratified" },
  { id: "RSK-001", type: "RISK", title: "Sector Concentration Shock (Tech)", href: "/risks-and-groupthink?tab=matrix", badge: "Controlled" },
  { id: "LRN-001", type: "LEARNING", title: "Minervini Stage Filter Rule", href: "/learning-intelligence?tab=knowledge", badge: "Codified" },
  { id: "FUT-001", type: "SIMULATION", title: "Adverse Macro Rate Shock 360d", href: "/simulation-intelligence?simId=FUT-001", badge: "Certified" },
  { id: "REC-001", type: "RECOMMENDATION", title: "Deploy Contrarian Reviewer", href: "/coaching-intelligence?tab=interventions", badge: "Active" },
  { id: "RB-001", type: "RUNBOOK", title: "Transient Telemetry Failover", href: "/resilience-intelligence?tab=runbooks", badge: "Ready" },
];

const CERTIFICATION_GATES: CertificationGateItem[] = [
  { id: "M15-Gate-01", name: "Unified Executive Home Certification", status: "PASS", rule: "UH-001..UH-015 100% KPI & Action Coverage", evidence: "All 7 KPIs, trend panel, and narrative present" },
  { id: "M15-Gate-02", name: "Executive Action Center Certification", status: "PASS", rule: "UH-011-AT-001..010 Multi-Center Aggregation", evidence: "Severity-first sorting verified" },
  { id: "M15-Gate-03", name: "Universal Graph & Impact Tracing", status: "PASS", rule: "9 Node Types & Multi-Hop Lineage", evidence: "Full causal chain traversable" },
  { id: "M15-Gate-04", name: "Narrative Intelligence Layer", status: "PASS", rule: "Driver Explanation & Natural Language Briefing", evidence: "Why-improved and why-degraded drivers active" },
  { id: "M15-Gate-05", name: "ARX Horizon Design System", status: "PASS", rule: "CSS Variable Tokens & Standardized Status", evidence: "WCAG AA compliant contrast" },
];

// Historical Trend Data Mock for 30d, 90d, 1y
const TREND_DATA = {
  "30D": [
    { label: "Day 1", ohi: 86.2, risk: 14.5, learning: 3.2 },
    { label: "Day 10", ohi: 87.1, risk: 13.8, learning: 3.4 },
    { label: "Day 20", ohi: 88.0, risk: 12.9, learning: 3.6 },
    { label: "Day 30", ohi: 89.4, risk: 12.0, learning: 3.8 },
  ],
  "90D": [
    { label: "Month 1", ohi: 84.5, risk: 16.2, learning: 2.9 },
    { label: "Month 2", ohi: 86.8, risk: 14.1, learning: 3.4 },
    { label: "Month 3", ohi: 89.4, risk: 12.0, learning: 3.8 },
  ],
  "1Y": [
    { label: "Q1", ohi: 81.2, risk: 19.5, learning: 2.2 },
    { label: "Q2", ohi: 83.8, risk: 16.8, learning: 2.8 },
    { label: "Q3", ohi: 86.5, risk: 14.2, learning: 3.3 },
    { label: "Q4", ohi: 89.4, risk: 12.0, learning: 3.8 },
  ],
};

function IntelligenceCenterContent() {
  const [viewState, setViewState] = useState<"normal" | "loading" | "empty" | "error" | "success">("normal");
  const [trendPeriod, setTrendPeriod] = useState<"30D" | "90D" | "1Y">("90D");
  const briefing = generateExecutiveBriefing();

  const currentTrends = TREND_DATA[trendPeriod];

  return (
    <IntelligenceShell
      title="Executive Intelligence Center"
      subtitle="Single-Entry Executive Workspace & Unified Institutional Operating System"
      badge="PHASE 31-M15 CERTIFIED"
      activeNavTab="/intelligence-center"
      actions={
        <div className="flex items-center gap-2">
          <select
            value={viewState}
            onChange={(e) => setViewState(e.target.value as any)}
            className="px-3 py-1.5 rounded-lg bg-[#182336] text-xs font-mono text-cyan-300 border border-[#24324A] focus:outline-none focus:ring-1 focus:ring-cyan-400"
            aria-label="Toggle UI State Simulator"
          >
            <option value="normal">State: Live Operational</option>
            <option value="loading">State: Loading Skeleton</option>
            <option value="empty">State: Empty Data</option>
            <option value="error">State: Fail-Closed Error</option>
            <option value="success">State: Certified Success</option>
          </select>
          <Link
            href="/action-center"
            className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-400"
          >
            Action Center &rarr;
          </Link>
        </div>
      }
    >
      {viewState === "loading" && <IntelligenceLoadingState message="Recomputing institutional OHI telemetry across 12 centers..." />}
      {viewState === "empty" && (
        <IntelligenceEmptyState
          title="No Active Institutional Anomalies"
          description="All decision committees, risk monitors, and autonomous models are operating within nominal baseline parameters."
          actionLabel="Refresh Telemetry Stream"
          onAction={() => setViewState("normal")}
        />
      )}
      {viewState === "error" && (
        <IntelligenceErrorState
          errorCode="ERR-GOV-001"
          failureClass="Simulated Invariant Degradation"
          message="Simulated fail-close error condition: telemetry validation detected divergence exceeding tolerance boundaries."
          onRetry={() => setViewState("normal")}
          onActivateSafeMode={() => alert("Engaged L4 Safe Mode")}
        />
      )}
      {viewState === "success" && (
        <IntelligenceSuccessState
          title="All 10 M15 Modernization Gates Successfully Certified"
          message="The ARX Horizon unified executive intelligence layer meets all contrast, responsive reflow, and fail-close criteria."
          auditHash="e7c5698e9db9d3548c6eb6b81c56b2ff782e7ecb8087d0663458fe15d9321da4"
          onProceed={() => setViewState("normal")}
        />
      )}

      {viewState === "normal" && (
        <div className="space-y-6">
          {/* Executive Header Component */}
          <IntelligenceHeader
            title="ARX Horizon Executive Overview"
            subtitle="Real-time organizational health, active alerts, forward projections, and governance boundaries."
            status="HEALTHY"
            certification="M15 CERTIFIED"
            replayHash="ed3eae1b72a49482aa960c2156f959c6725df52b64195845645cd5466643211b"
            actions={
              <div className="flex items-center gap-2">
                <Link
                  href="/graph-explorer"
                  className="px-3 py-1.5 rounded-lg bg-[#182336] hover:bg-[#22324d] text-cyan-300 border border-cyan-500/30 text-xs font-mono font-semibold transition-colors"
                >
                  Universal Graph &rarr;
                </Link>
              </div>
            }
          />

          {/* Master Executive KPI Grid (UH-002, UH-003) */}
          <section aria-label="Executive Institutional KPIs">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 font-semibold">
                Executive Health Indicators (7 Bounded Metrics)
              </h2>
              <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Live Invariant Bounds Active
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7 gap-3">
              <IntelligenceMetricCard
                label="Org Health (OHI)"
                value="89.4"
                delta="+3.2 QoQ"
                deltaPositive={true}
                target=">=80.0"
                confidence="99% Conf"
                status="HEALTHY"
                targetRoute="/oos"
              />
              <IntelligenceMetricCard
                label="Dec Quality (ODEI)"
                value="86.4"
                delta="+1.8 QoQ"
                deltaPositive={true}
                target=">=80.0"
                confidence="Calibrated"
                status="HEALTHY"
                targetRoute="/committee-intelligence"
              />
              <IntelligenceMetricCard
                label="Delib Quality (CDQI)"
                value="82.1"
                delta="+0.9 QoQ"
                deltaPositive={true}
                target=">=75.0"
                confidence="Audited"
                status="HEALTHY"
                targetRoute="/committee-intelligence"
              />
              <IntelligenceMetricCard
                label="Dissent Ratio"
                value="1.42"
                delta="Balanced"
                deltaPositive={true}
                target="1.2 - 1.8"
                confidence="Optimal"
                status="HEALTHY"
                targetRoute="/dissent-explorer"
              />
              <IntelligenceMetricCard
                label="Learning Vel"
                value="+3.8/q"
                delta="+12% YoY"
                deltaPositive={true}
                target=">=2.0"
                confidence="Verified"
                status="HEALTHY"
                targetRoute="/learning-intelligence"
              />
              <IntelligenceMetricCard
                label="Forecast Risk"
                value="12.0%"
                delta="-3.4%"
                deltaPositive={true}
                target="<=20.0%"
                confidence="High Rigor"
                status="HEALTHY"
                targetRoute="/risks-and-groupthink"
              />
              <IntelligenceMetricCard
                label="Survivability"
                value="99.4%"
                delta="+0.4%"
                deltaPositive={true}
                target=">=95.0%"
                confidence="M14 Certified"
                status="CERTIFIED"
                targetRoute="/resilience-intelligence"
              />
            </div>
          </section>

          {/* Organizational Trend Panel (UH-006) */}
          <HorizonCard
            title="Organizational Performance & Risk Trends"
            subtitle="Historical Trajectory Modeling across Multi-Quarter Horizons"
            actions={
              <div className="flex items-center gap-1.5 p-1 rounded-lg bg-[#0B1220] border border-[#24324A]">
                {(['30D', '90D', '1Y'] as const).map((period) => (
                  <button
                    key={period}
                    onClick={() => setTrendPeriod(period)}
                    className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-colors ${
                      trendPeriod === period
                        ? 'bg-cyan-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-[#182336]'
                    }`}
                  >
                    {period}
                  </button>
                ))}
              </div>
            }
          >
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-3 rounded-lg bg-[#0B1220] border border-[#24324A]">
                  <div className="text-[11px] font-mono text-slate-400 uppercase">Current Period OHI</div>
                  <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
                    {currentTrends[currentTrends.length - 1].ohi.toFixed(1)}
                  </div>
                  <div className="text-[10px] font-mono text-slate-400 mt-0.5">Trend: Expansionary (+3.2)</div>
                </div>
                <div className="p-3 rounded-lg bg-[#0B1220] border border-[#24324A]">
                  <div className="text-[11px] font-mono text-slate-400 uppercase">Risk Probability</div>
                  <div className="text-xl font-bold font-mono text-cyan-400 mt-1">
                    {currentTrends[currentTrends.length - 1].risk.toFixed(1)}%
                  </div>
                  <div className="text-[10px] font-mono text-slate-400 mt-0.5">Stress Ceiling: 25.0%</div>
                </div>
                <div className="p-3 rounded-lg bg-[#0B1220] border border-[#24324A]">
                  <div className="text-[11px] font-mono text-slate-400 uppercase">Learning Velocity</div>
                  <div className="text-xl font-bold font-mono text-purple-400 mt-1">
                    +{currentTrends[currentTrends.length - 1].learning.toFixed(1)}/q
                  </div>
                  <div className="text-[10px] font-mono text-slate-400 mt-0.5">Feedback Loop: Accelerated</div>
                </div>
              </div>

              {/* Visual Trend Representation */}
              <div className="p-4 rounded-xl bg-[#0B1220] border border-[#24324A]">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-3">
                  <span>Period Trajectory ({trendPeriod})</span>
                  <span className="text-cyan-400">OHI vs Risk vs Learning</span>
                </div>
                <div className="grid grid-cols-4 gap-2 text-center font-mono">
                  {currentTrends.map((t, idx) => (
                    <div key={idx} className="p-2.5 rounded bg-[#121B2A] border border-[#24324A]/70">
                      <div className="text-[10px] text-slate-400">{t.label}</div>
                      <div className="text-sm font-bold text-emerald-400 mt-1">OHI {t.ohi}</div>
                      <div className="text-[11px] text-cyan-400">Risk {t.risk}%</div>
                      <div className="text-[10px] text-purple-400 mt-0.5">+{t.learning} LRN</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </HorizonCard>

          {/* Executive Narrative & Driver Explanation (UH-009, M15-04) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <HorizonCard
                title="Executive Narrative Synthesis"
                subtitle="Natural Language Strategic Analysis & Invariant Status"
                badge={<SeverityBadge status="HEALTHY" size="sm" />}
              >
                <div className="space-y-4">
                  <div className="p-4 rounded-xl bg-[#182336] border border-[#24324A]">
                    <h3 className="text-base font-semibold text-white mb-1.5">
                      {briefing.headline}
                    </h3>
                    <p className="text-xs md:text-sm text-[#94A3B8] leading-relaxed font-mono">
                      {briefing.executiveSummary}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg bg-[#0B1220] border border-emerald-500/30">
                      <div className="text-xs font-semibold text-emerald-400 font-mono flex items-center gap-1.5 mb-1.5">
                        <span>↑</span>
                        <span>Why Improved? (Top Positive Drivers)</span>
                      </div>
                      <ul className="text-xs font-mono text-slate-300 space-y-1">
                        <li>• Learning velocity expanded 12% via stage-rule codified patterns.</li>
                        <li>• ODEI increased +1.8 from enhanced dissent participation in COM-001.</li>
                        <li>• Autonomous failover RTO dropped to 4.8s (90% below SLA).</li>
                      </ul>
                    </div>

                    <div className="p-3 rounded-lg bg-[#0B1220] border border-amber-500/30">
                      <div className="text-xs font-semibold text-amber-400 font-mono flex items-center gap-1.5 mb-1.5">
                        <span>↓</span>
                        <span>Why Degraded / Key Concerns</span>
                      </div>
                      <ul className="text-xs font-mono text-slate-300 space-y-1">
                        <li>• Knowledge transfer collapse warning in Credit Subcommittee (COM-002).</li>
                        <li>• Liquidity minority dissent DIS-004 nearing 24h response SLA.</li>
                        <li>• Adverse macro scenario simulation projects 8% drawdown without rebalance.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </HorizonCard>
            </div>

            {/* Action Center Preview (UH-004, UH-005) */}
            <div>
              <HorizonCard
                title="Priority Actions Preview"
                subtitle="Critical Items Requiring Executive Triage"
                badge={<span className="text-[10px] font-mono text-red-400 font-semibold">2 CRITICAL</span>}
                actions={
                  <Link href="/action-center" className="text-xs font-mono text-cyan-400 hover:underline">
                    View All (5) &rarr;
                  </Link>
                }
              >
                <div className="space-y-2.5">
                  <div className="p-2.5 rounded-lg bg-[#182336] border border-red-500/40 space-y-1">
                    <div className="flex items-center justify-between">
                      <SeverityBadge level="CRITICAL" size="sm" />
                      <span className="text-[10px] font-mono text-red-400">Due: 15m</span>
                    </div>
                    <div className="text-xs font-semibold text-white">
                      Replay Drift Remediation & Snapshot Verification
                    </div>
                    <div className="text-[11px] font-mono text-slate-400">
                      Owner: Audit & Risk Board | Action: Launch Runbook M9-RB-02
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-[#182336] border border-orange-500/40 space-y-1">
                    <div className="flex items-center justify-between">
                      <SeverityBadge level="HIGH" size="sm" />
                      <span className="text-[10px] font-mono text-orange-400">Due: 1h 45m</span>
                    </div>
                    <div className="text-xs font-semibold text-white">
                      Capital Rebalancing Plan Execution Authorization
                    </div>
                    <div className="text-[11px] font-mono text-slate-400">
                      Owner: Investment Committee | Action: Approve Allocation
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-[#182336] border border-amber-500/40 space-y-1">
                    <div className="flex items-center justify-between">
                      <SeverityBadge level="MEDIUM" size="sm" />
                      <span className="text-[10px] font-mono text-amber-400">Due: 6h</span>
                    </div>
                    <div className="text-xs font-semibold text-white">
                      Knowledge Transfer Collapse Coaching Plan
                    </div>
                    <div className="text-[11px] font-mono text-slate-400">
                      Owner: Coaching Board | Action: Deploy Devil&apos;s Advocate
                    </div>
                  </div>

                  <Link
                    href="/action-center"
                    className="block text-center py-2 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 text-cyan-300 font-mono text-xs font-semibold transition-colors"
                  >
                    Open Unified Action Center Queue &rarr;
                  </Link>
                </div>
              </HorizonCard>
            </div>
          </div>

          {/* Categorized Directory of All 12 Centers (UH-007) */}
          <section aria-label="Intelligence Centers Directory">
            <h2 className="text-xs font-mono uppercase tracking-widest text-slate-400 font-semibold mb-3">
              Institutional Intelligence Centers (Categorized Directory)
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Strategic Intelligence */}
              <div className="p-4 rounded-xl bg-[#121B2A] border border-[#24324A] space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-[#24324A]">
                  <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider">
                    Strategic Intelligence
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">4 Modules</span>
                </div>
                <div className="space-y-2">
                  {STRATEGIC_CENTERS.map((c) => (
                    <Link
                      key={c.href}
                      href={c.href}
                      className="p-2.5 rounded-lg bg-[#0B1220] border border-[#24324A] hover:border-cyan-500/40 hover:bg-[#162236] transition-all block group focus:outline-none focus:ring-2 focus:ring-cyan-400"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-200 group-hover:text-cyan-300">
                          {c.name}
                        </span>
                        <SeverityBadge status={c.status} size="sm" />
                      </div>
                      <div className="text-[11px] font-mono text-cyan-400 mt-1">{c.metric}</div>
                      <div className="text-[10px] text-slate-400 font-mono mt-0.5 truncate">{c.desc}</div>
                    </Link>
                  ))}
                </div>
              </div>

              {/* Operational Intelligence */}
              <div className="p-4 rounded-xl bg-[#121B2A] border border-[#24324A] space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-[#24324A]">
                  <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider">
                    Operational Intelligence
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">4 Modules</span>
                </div>
                <div className="space-y-2">
                  {OPERATIONAL_CENTERS.map((c) => (
                    <Link
                      key={c.href}
                      href={c.href}
                      className="p-2.5 rounded-lg bg-[#0B1220] border border-[#24324A] hover:border-emerald-500/40 hover:bg-[#162236] transition-all block group focus:outline-none focus:ring-2 focus:ring-emerald-400"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-200 group-hover:text-emerald-300">
                          {c.name}
                        </span>
                        <SeverityBadge status={c.status} size="sm" />
                      </div>
                      <div className="text-[11px] font-mono text-emerald-400 mt-1">{c.metric}</div>
                      <div className="text-[10px] text-slate-400 font-mono mt-0.5 truncate">{c.desc}</div>
                    </Link>
                  ))}
                </div>
              </div>

              {/* Evidence & Audit */}
              <div className="p-4 rounded-xl bg-[#121B2A] border border-[#24324A] space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-[#24324A]">
                  <span className="text-xs font-mono font-bold text-purple-400 uppercase tracking-wider">
                    Evidence & Audit
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">4 Modules</span>
                </div>
                <div className="space-y-2">
                  {EVIDENCE_CENTERS.map((c) => (
                    <Link
                      key={c.href}
                      href={c.href}
                      className="p-2.5 rounded-lg bg-[#0B1220] border border-[#24324A] hover:border-purple-500/40 hover:bg-[#162236] transition-all block group focus:outline-none focus:ring-2 focus:ring-purple-400"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-200 group-hover:text-purple-300">
                          {c.name}
                        </span>
                        <SeverityBadge status={c.status} size="sm" />
                      </div>
                      <div className="text-[11px] font-mono text-purple-400 mt-1">{c.metric}</div>
                      <div className="text-[10px] text-slate-400 font-mono mt-0.5 truncate">{c.desc}</div>
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Related Artifacts Panel (UH-008) */}
          <RelatedArtifactsPanel artifacts={SAMPLE_RELATED_ARTIFACTS} />

          {/* Autonomous Governance & M15 Certification Panel */}
          <CertificationPanel
            title="ARX Horizon Experience Certification Gates (M15)"
            subtitle="Automated Verification of Accessibility, Responsive Layouts, and Fail-Close Safety"
            overallStatus="CERTIFIED"
            auditHash="ed3eae1b72a49482aa960c2156f959c6725df52b64195845645cd5466643211b"
            timestamp="2026-09-08 23:59:00 UTC"
            gates={CERTIFICATION_GATES}
          />
        </div>
      )}
    </IntelligenceShell>
  );
}

export default function IntelligenceCenterPage() {
  return (
    <Suspense fallback={<IntelligenceLoadingState />}>
      <IntelligenceCenterContent />
    </Suspense>
  );
}
