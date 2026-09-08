"use client";

import React, { useState, Suspense } from "react";
import Link from "next/link";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import HorizonCard from "../../components/ui/HorizonCard";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import IntelligenceLoadingState from "../../components/ui/IntelligenceLoadingState";
import IntelligenceEmptyState from "../../components/ui/IntelligenceEmptyState";
import IntelligenceErrorState from "../../components/ui/IntelligenceErrorState";
import IntelligenceSuccessState from "../../components/ui/IntelligenceSuccessState";
import { generateExecutiveBriefing, CANONICAL_TELEMETRY_BASELINE } from "../../lib/narrative/executiveNarrativeEngine";

const CENTERS = [
  { href: "/committee-intelligence", name: "Committee Intelligence", status: "HEALTHY", metric: "ODEI 86.4" },
  { href: "/decision-explorer", name: "Decision Explorer", status: "HEALTHY", metric: "24 Decisions" },
  { href: "/dissent-explorer", name: "Dissent Explorer", status: "HEALTHY", metric: "8 Dissents" },
  { href: "/committee-network", name: "Committee Network", status: "HEALTHY", metric: "18 Edges" },
  { href: "/audit-explorer", name: "Audit Explorer", status: "HEALTHY", metric: "100% Verified" },
  { href: "/learning-intelligence", name: "Learning Intelligence", status: "HEALTHY", metric: "+3.8/qtr" },
  { href: "/risks-and-groupthink", name: "Risks & Groupthink", status: "HEALTHY", metric: "Low Stress" },
  { href: "/coaching-intelligence", name: "Coaching Intelligence", status: "HEALTHY", metric: "5 Active" },
  { href: "/oos", name: "Operating System", status: "HEALTHY", metric: "OHI 84.2" },
  { href: "/optimization-intelligence", name: "Optimization Intelligence", status: "HEALTHY", metric: "+2.4% Target" },
  { href: "/resilience-intelligence", name: "Resilience Intelligence", status: "CERTIFIED", metric: "RTO 4.8s" },
  { href: "/autonomous-governance", name: "Autonomous Governance", status: "CERTIFIED", metric: "10/10 Invariants" },
];

function IntelligenceCenterContent() {
  const [viewState, setViewState] = useState<"normal" | "loading" | "empty" | "error" | "success">("normal");
  const briefing = generateExecutiveBriefing();

  return (
    <IntelligenceShell
      title="Executive Intelligence Center"
      subtitle="Unified Institutional Command & Strategic Decision Hub"
      badge="PHASE 31-M11 CERTIFIED"
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
            className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold transition-colors shadow-sm"
          >
            Action Center &rarr;
          </Link>
        </div>
      }
    >
      {/* 4-State Demonstrations */}
      {viewState === "loading" && <IntelligenceLoadingState message="Recomputing multi-center OHI telemetry..." />}
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
          title="All 10 M11 UX Gates Successfully Certified"
          message="The ARX Horizon unified executive intelligence layer meets all contrast, responsive reflow, and fail-close criteria."
          auditHash="0x7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
          onProceed={() => setViewState("normal")}
        />
      )}

      {viewState === "normal" && (
        <>
          {/* Executive Narrative Briefing */}
          <HorizonCard
            title="Institutional Strategic Briefing"
            subtitle="Automated Natural-Language Cross-Center Synthesis"
            badge={
              <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] font-mono font-semibold uppercase">
                {briefing.overallStatus}
              </span>
            }
          >
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-[#182336] border border-[#24324A]">
                <h3 className="text-base font-semibold text-white mb-2">
                  {briefing.headline}
                </h3>
                <p className="text-sm text-[#94A3B8] leading-relaxed">
                  {briefing.executiveSummary}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h4 className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
                    Key Institutional Findings:
                  </h4>
                  <ul className="space-y-1.5 text-xs font-mono text-slate-300">
                    {briefing.keyFindings.map((finding, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-cyan-400 mt-0.5">&bull;</span>
                        <span>{finding}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="space-y-2">
                  <h4 className="text-xs font-mono font-semibold text-amber-400 uppercase tracking-wider">
                    Immediate Priority Recommendations:
                  </h4>
                  <div className="space-y-2">
                    {briefing.priorityActions.map((action) => (
                      <div
                        key={action.id}
                        className="p-2.5 rounded bg-[#152033] border border-[#24324A] flex items-center justify-between text-xs"
                      >
                        <div>
                          <div className="font-semibold text-slate-200">{action.title}</div>
                          <div className="text-[11px] font-mono text-slate-400">{action.ownerCommittee}</div>
                        </div>
                        <Link
                          href="/action-center"
                          className="px-2 py-1 rounded bg-[#1e2e4a] hover:bg-cyan-900/50 text-cyan-300 text-[10px] font-mono font-semibold border border-cyan-500/30"
                        >
                          Triage &rarr;
                        </Link>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </HorizonCard>

          {/* Master KPI Grid */}
          <div>
            <h3 className="text-xs font-mono uppercase tracking-widest text-slate-400 font-semibold mb-3">
              Executive Institutional Metrics (Real-Time Invariant Bounds)
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
              <HorizonMetricCard
                label="Org Health (OHI)"
                value="84.2"
                delta="+2.1"
                deltaPositive={true}
                target=">=80.0"
                confidence="99% Conf"
                severity="PASS"
                subtext="Institutional Benchmark"
              />
              <HorizonMetricCard
                label="Risk Stress"
                value="12.0%"
                delta="-3.4%"
                deltaPositive={true}
                target="<=25.0%"
                confidence="High Rigor"
                severity="PASS"
                subtext="Portfolio Stress Prob"
              />
              <HorizonMetricCard
                label="Learning Vel"
                value="+3.8/q"
                delta="+0.9"
                deltaPositive={true}
                target=">=2.0"
                confidence="Verified"
                severity="PASS"
                subtext="Quarterly Expansion"
              />
              <HorizonMetricCard
                label="Governance"
                value="98.4%"
                delta="Nominal"
                deltaPositive={true}
                target=">=90.0%"
                confidence="Audited"
                severity="PASS"
                subtext="Policy Bound Compliance"
              />
              <HorizonMetricCard
                label="Failover RTO"
                value="4.8s"
                delta="Optimal"
                deltaPositive={true}
                target="<30.0s"
                confidence="Certified"
                severity="PASS"
                subtext="L1 Transient Recovery"
              />
              <HorizonMetricCard
                label="Safety Invariants"
                value="10/10"
                delta="0 Drift"
                deltaPositive={true}
                target="10/10 PASS"
                confidence="M10 Certified"
                severity="PASS"
                subtext="Fail-Close Active"
              />
            </div>
          </div>

          {/* 11 Intelligence Centers Fast Access Grid */}
          <HorizonCard
            title="Institutional Intelligence Centers (11 Modules)"
            subtitle="Unified Access Matrix to Platform Analytic & Operational Capabilities"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {CENTERS.map((c) => (
                <Link
                  key={c.href}
                  href={c.href}
                  className="p-3 rounded-xl bg-[#182336] border border-[#24324A] hover:border-cyan-500/50 hover:bg-[#1f2f47] transition-all flex flex-col justify-between group"
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="text-xs font-semibold text-slate-200 group-hover:text-cyan-300">
                      {c.name}
                    </span>
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span>{c.metric}</span>
                    <span className="text-cyan-400 group-hover:translate-x-0.5 transition-transform">&rarr;</span>
                  </div>
                </Link>
              ))}
            </div>
          </HorizonCard>

          {/* Cross-Center Insights Matrix */}
          <HorizonCard
            title="Cross-Center Institutional Signals"
            subtitle="Multi-Vector Correlation and Anomaly Detection"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {briefing.insights.map((ins, idx) => (
                <div key={idx} className="p-3.5 rounded-xl bg-[#182336] border border-[#24324A] space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-cyan-400">{ins.center}</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#121B2A] border border-[#24324A] text-emerald-400 font-semibold">
                      {ins.signal}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300">
                    {ins.interpretation}
                  </p>
                </div>
              ))}
            </div>
          </HorizonCard>
        </>
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
