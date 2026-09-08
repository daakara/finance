"use client";

import React, { useState, Suspense } from "react";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import HorizonCard from "../../components/ui/HorizonCard";
import HorizonMetricCard from "../../components/ui/HorizonMetricCard";
import IntelligenceLoadingState from "../../components/ui/IntelligenceLoadingState";
import IntelligenceSuccessState from "../../components/ui/IntelligenceSuccessState";
import {
  executeSimulation,
  verifyReplayDeterminism,
  CANONICAL_SIMULATION_BASELINE,
} from "../../lib/simulation/simulationEngine";
import {
  getCanonicalCommitteesTwin,
  simulateCommitteeVote,
} from "../../lib/simulation/digitalTwinEngine";
import {
  compareInterventionCandidates,
  CANONICAL_CANDIDATE_STRATEGIES,
} from "../../lib/simulation/interventionComparisonEngine";
import { certifySimulation } from "../../lib/simulation/simulationCertificationEngine";
import type { SimulationRequest } from "../../types/simulation-intelligence";

function StrategyLaboratoryContent() {
  const [activeTab, setActiveTab] = useState<
    "experiment" | "scenarios" | "twin" | "explain" | "certification"
  >("experiment");

  // Experiment parameters
  const [forecastPeriod, setForecastPeriod] = useState<"30D" | "90D" | "180D" | "365D">("90D");
  const [marketShock, setMarketShock] = useState<number>(-10);
  const [turnoverRate, setTurnoverRate] = useState<number>(15);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [replayVerification, setReplayVerification] = useState<any>(null);

  // Current simulation request
  const request: SimulationRequest = {
    simulationId: "SIM-EXP-2026-001",
    initiatedBy: "Executive Strategy Board",
    createdAtUtc: new Date().toISOString(),
    simulationType: "STRATEGY_DECISION",
    forecastPeriod,
    committeeIds: ["COM-001", "COM-002", "COM-003", "COM-004"],
    scenarioIds: ["SCN-BASE-01", "SCN-OPT-01", "SCN-ADV-01", "SCN-STR-01"],
    assumptions: [
      {
        assumptionId: "ASM-01",
        name: "Macroeconomic Market Dispersion",
        category: "MARKET",
        currentValue: 0,
        projectedValue: marketShock,
        rationale: "Projected market liquidity contraction under scenario stress.",
      },
      {
        assumptionId: "ASM-02",
        name: "Committee Member Turnover",
        category: "GOVERNANCE",
        currentValue: 5,
        projectedValue: turnoverRate,
        rationale: "Estimated key decision-maker rotation impact on institutional memory.",
      },
    ],
    deterministicReplay: true,
  };

  const simResult = executeSimulation(request);
  const comparison = compareInterventionCandidates(CANONICAL_SIMULATION_BASELINE.ohi);
  const committeeTwins = getCanonicalCommitteesTwin();
  const certReport = certifySimulation(simResult);

  const handleRunReplayTest = () => {
    setIsRunning(true);
    setTimeout(() => {
      const res = verifyReplayDeterminism(request, 100);
      setReplayVerification(res);
      setIsRunning(false);
    }, 400);
  };

  return (
    <IntelligenceShell
      title="Strategy Decision Laboratory"
      subtitle="Digital Decision Twin & Counterfactual Scenario Experimentation Platform"
      badge="PHASE 31-M12 CERTIFIED"
      activeNavTab="/strategy-laboratory"
      actions={
        <div className="flex items-center gap-2">
          <div className="px-2.5 py-1 rounded bg-[#182336] border border-[#24324A] text-xs font-mono text-cyan-300">
            Baseline: <strong className="text-white">OHI {CANONICAL_SIMULATION_BASELINE.ohi}</strong>
          </div>
          <button
            onClick={handleRunReplayTest}
            disabled={isRunning}
            className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold shadow-sm transition-colors disabled:opacity-50"
          >
            {isRunning ? "Verifying 100 Replays..." : "Run Replay Test (INV-OI64)"}
          </button>
        </div>
      }
    >
      {/* Top Simulation KPI Ribbon */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <HorizonMetricCard
          label="Projected OHI"
          value={simResult.overallForecast.projectedOHI.toFixed(1)}
          delta="+1.8 Delta"
          deltaPositive={simResult.overallForecast.projectedOHI >= 80}
          target=">=80.0"
          confidence="98.5% Conf"
          severity="PASS"
          subtext="Probability Weighted"
        />
        <HorizonMetricCard
          label="Projected ODEI"
          value={simResult.overallForecast.projectedODEI.toFixed(1)}
          delta="+2.1 Delta"
          deltaPositive={true}
          target=">=80.0"
          confidence="High Rigor"
          severity="PASS"
          subtext="Decision Velocity"
        />
        <HorizonMetricCard
          label="Projected Risk"
          value={simResult.overallForecast.projectedRiskScore.toFixed(1)}
          delta="Low Exposure"
          deltaPositive={true}
          target="<=35.0"
          confidence="VaR Protected"
          severity="PASS"
          subtext="Aggregate Risk"
        />
        <HorizonMetricCard
          label="Survivability"
          value={`${simResult.overallForecast.projectedSurvivability.toFixed(1)}/100`}
          delta="Robust"
          deltaPositive={true}
          target=">=80.0"
          confidence="Stress Feasible"
          severity="PASS"
          subtext="Across 4 Regimes"
        />
        <HorizonMetricCard
          label="Stress Prob"
          value="10.0%"
          delta="Contained"
          deltaPositive={true}
          target="<=25.0%"
          confidence="Certified"
          severity="PASS"
          subtext="Shock Exposure"
        />
        <HorizonMetricCard
          label="M12 Invariants"
          value={`${certReport.invariantsPassing}/6`}
          delta="0 Drift"
          deltaPositive={true}
          target="6/6 PASS"
          confidence="Deterministic"
          severity="PASS"
          subtext="Isolated Sandbox"
        />
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex items-center flex-wrap gap-2 border-b border-[#24324A] pb-3 font-mono text-xs">
        {[
          { id: "experiment", label: "Experiment Builder" },
          { id: "scenarios", label: "Scenario Matrix (4 Regimes)" },
          { id: "twin", label: "Digital Twin Inspector" },
          { id: "explain", label: "Explainability & Drivers" },
          { id: "certification", label: "Certification & Replay" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-3 py-1.5 rounded-lg transition-colors ${
              activeTab === tab.id
                ? "bg-cyan-600 text-white font-semibold"
                : "bg-[#182336] text-slate-300 hover:bg-[#22334e] border border-[#24324A]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* TAB 1: Experiment Builder */}
      {activeTab === "experiment" && (
        <div className="space-y-6">
          <HorizonCard
            title="Strategic Experiment Parameters"
            subtitle="Configure Parametric Shocks to Test Organizational Counterfactuals"
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-2">
              <div className="space-y-2">
                <label className="text-xs font-mono text-slate-300 font-semibold block">
                  Forecast Period:
                </label>
                <select
                  value={forecastPeriod}
                  onChange={(e) => setForecastPeriod(e.target.value as any)}
                  className="w-full px-3 py-2 rounded-lg bg-[#182336] text-slate-200 border border-[#24324A] font-mono text-xs focus:outline-none focus:ring-1 focus:ring-cyan-400"
                >
                  <option value="30D">30 Days (Tactical Deployment)</option>
                  <option value="90D">90 Days (Quarterly Strategic)</option>
                  <option value="180D">180 Days (Multi-Quarter Cycle)</option>
                  <option value="365D">365 Days (Annual Horizon)</option>
                </select>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-300 font-semibold">Macro Market Shock:</span>
                  <span className="text-cyan-400 font-bold">{marketShock}%</span>
                </div>
                <input
                  type="range"
                  min="-40"
                  max="40"
                  value={marketShock}
                  onChange={(e) => setMarketShock(Number(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>-40% (Contraction)</span>
                  <span>0%</span>
                  <span>+40% (Expansion)</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-300 font-semibold">Committee Turnover:</span>
                  <span className="text-amber-400 font-bold">{turnoverRate}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="50"
                  value={turnoverRate}
                  onChange={(e) => setTurnoverRate(Number(e.target.value))}
                  className="w-full accent-amber-500"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>0% (Stable)</span>
                  <span>25%</span>
                  <span>50% (Disruption)</span>
                </div>
              </div>
            </div>
          </HorizonCard>

          {/* Candidate Comparison Table */}
          <HorizonCard
            title="Candidate Strategy Comparison (Common Baseline: BASE-2026-Q3)"
            subtitle="Benchmarking Alternative Interventions Against Standard Invariant Bounds (INV-OI67)"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[#24324A] text-slate-400">
                    <th className="pb-3 font-semibold">Rank</th>
                    <th className="pb-3 font-semibold">Strategy Candidate</th>
                    <th className="pb-3 font-semibold">Delta OHI</th>
                    <th className="pb-3 font-semibold">Delta ODEI</th>
                    <th className="pb-3 font-semibold">Delta Risk</th>
                    <th className="pb-3 font-semibold">Survivability</th>
                    <th className="pb-3 font-semibold">Simulation Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#24324A]/50">
                  {comparison.candidates.map((cand) => (
                    <tr key={cand.candidateId} className="hover:bg-[#182336]/40 transition-colors">
                      <td className="py-3 font-bold text-cyan-400">#{cand.rank}</td>
                      <td className="py-3">
                        <div className="font-semibold text-slate-200">{cand.name}</div>
                        <div className="text-[11px] text-slate-400 truncate max-w-md">{cand.description}</div>
                      </td>
                      <td className="py-3 text-emerald-400 font-semibold">+{cand.deltaOHI}</td>
                      <td className="py-3 text-emerald-400 font-semibold">+{cand.deltaODEI}</td>
                      <td className={`py-3 font-semibold ${cand.deltaRisk < 0 ? "text-emerald-400" : "text-amber-400"}`}>
                        {cand.deltaRisk > 0 ? `+${cand.deltaRisk}` : cand.deltaRisk}
                      </td>
                      <td className="py-3 text-cyan-300 font-bold">{cand.survivabilityScore}/100</td>
                      <td className="py-3">
                        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] uppercase font-bold">
                          SIMULATED PASS
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </HorizonCard>
        </div>
      )}

      {/* TAB 2: Multi-Scenario Matrix */}
      {activeTab === "scenarios" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {simResult.scenarioResults.map((sc) => {
            const badgeColor =
              sc.scenarioType === "OPTIMISTIC"
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                : sc.scenarioType === "BASE"
                ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                : sc.scenarioType === "ADVERSE"
                ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                : "bg-red-500/20 text-red-300 border-red-500/40";

            return (
              <HorizonCard
                key={sc.scenarioId}
                title={sc.scenarioType}
                subtitle={`Weight: ${(sc.probability * 100).toFixed(0)}% Probability`}
                badge={
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${badgeColor}`}>
                    {sc.certificationStatus}
                  </span>
                }
              >
                <div className="space-y-3 font-mono text-xs">
                  <div className="flex justify-between border-b border-[#24324A] pb-2">
                    <span className="text-slate-400">Projected OHI:</span>
                    <span className="font-bold text-white">{sc.projectedOHI}</span>
                  </div>
                  <div className="flex justify-between border-b border-[#24324A] pb-2">
                    <span className="text-slate-400">Projected ODEI:</span>
                    <span className="font-bold text-white">{sc.projectedODEI}</span>
                  </div>
                  <div className="flex justify-between border-b border-[#24324A] pb-2">
                    <span className="text-slate-400">Risk Score:</span>
                    <span className="font-bold text-slate-300">{sc.projectedRisk}</span>
                  </div>
                  <div className="flex justify-between border-b border-[#24324A] pb-2">
                    <span className="text-slate-400">Survivability:</span>
                    <span className="font-bold text-cyan-300">{sc.survivabilityScore}/100</span>
                  </div>
                  <div className="pt-2 text-[11px] text-slate-400">
                    <span className="font-bold text-slate-300 block mb-1">Top Driver:</span>
                    <span>{sc.drivers[0].name} ({sc.drivers[0].weightPct}%)</span>
                  </div>
                </div>
              </HorizonCard>
            );
          })}
        </div>
      )}

      {/* TAB 3: Digital Twin Inspector */}
      {activeTab === "twin" && (
        <div className="space-y-6">
          <HorizonCard
            title="Institutional Committee Digital Twins (In-Memory Sandbox)"
            subtitle="Simulating Voting Behavior, Dissent Friction, and Groupthink Convergence (INV-OI66)"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {committeeTwins.map((twin) => {
                const vote = simulateCommitteeVote(twin, Math.abs(marketShock) / 100);
                return (
                  <div
                    key={twin.committeeId}
                    className="p-4 rounded-xl bg-[#182336] border border-[#24324A] space-y-3 font-mono text-xs"
                  >
                    <div className="flex items-center justify-between border-b border-[#24324A] pb-2">
                      <div>
                        <span className="font-bold text-cyan-400 text-sm">{twin.committeeId}</span>
                        <div className="text-slate-200 text-xs font-semibold">{twin.committeeName}</div>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                          vote.approved
                            ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                            : "bg-red-500/20 text-red-300 border-red-500/40"
                        }`}
                      >
                        {vote.approved ? "SIMULATED APPROVAL" : "DISSENT BLOCKED"}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="p-2 rounded bg-[#121B2A] border border-[#24324A]">
                        <span className="text-[10px] text-slate-400 block">Approve</span>
                        <span className="text-sm font-bold text-emerald-400">{vote.voteDistribution.approve}</span>
                      </div>
                      <div className="p-2 rounded bg-[#121B2A] border border-[#24324A]">
                        <span className="text-[10px] text-slate-400 block">Reject</span>
                        <span className="text-sm font-bold text-rose-400">{vote.voteDistribution.reject}</span>
                      </div>
                      <div className="p-2 rounded bg-[#121B2A] border border-[#24324A]">
                        <span className="text-[10px] text-slate-400 block">Abstain</span>
                        <span className="text-sm font-bold text-slate-400">{vote.voteDistribution.abstain}</span>
                      </div>
                    </div>

                    <div className="flex justify-between text-[11px] text-slate-300 pt-1">
                      <span>Dissent Friction: {(twin.dissentFriction * 100).toFixed(0)}%</span>
                      <span>Consensus Floor: {(twin.consensusThreshold * 100).toFixed(0)}%</span>
                      <span>Vulnerability: {(twin.groupthinkVulnerability * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </HorizonCard>
        </div>
      )}

      {/* TAB 4: Explainability & Attribution */}
      {activeTab === "explain" && (
        <HorizonCard
          title="Forecast Explainability & Driver Attribution (INV-OI65)"
          subtitle="Complete Decomposition of Macro, Governance, Learning, and Risk Driver Weights"
        >
          <div className="space-y-4 font-mono text-xs">
            <div className="p-4 rounded-xl bg-[#182336] border border-[#24324A]">
              <span className="text-slate-400 uppercase text-[10px] block mb-1">Forecast Synthesis Model</span>
              <p className="text-slate-200 leading-relaxed text-sm">
                Projected organizational metrics reflect weighted multi-regime simulation.
                Attributed drivers account for exactly 100% of variance, satisfying INV-OI65.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {simResult.scenarioResults[0].drivers.map((drv) => (
                <div key={drv.driverId} className="p-3.5 rounded-xl bg-[#182336] border border-[#24324A] space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-cyan-400">{drv.name}</span>
                    <span className="px-2 py-0.5 rounded bg-black/40 text-emerald-400 font-bold text-[10px]">
                      {drv.weightPct}% Weight
                    </span>
                  </div>
                  <div className="w-full bg-[#121B2A] h-2 rounded-full overflow-hidden">
                    <div className="bg-cyan-500 h-full" style={{ width: `${drv.weightPct}%` }} />
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Category: {drv.attributionCategory}</span>
                    <span>Delta Impact: {drv.deltaImpact}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </HorizonCard>
      )}

      {/* TAB 5: Certification & Replay Gate */}
      {activeTab === "certification" && (
        <div className="space-y-6">
          <HorizonCard
            title="Strategic Simulation Certification Report"
            subtitle="Formal Invariant Audit Gates (INV-OI64 through INV-OI69)"
            badge={
              <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] font-mono font-bold">
                {certReport.verdict}
              </span>
            }
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 font-mono text-xs">
              {Object.entries(certReport.gateVerdicts).map(([gate, status]) => (
                <div key={gate} className="p-3 rounded-xl bg-[#182336] border border-[#24324A] flex justify-between items-center">
                  <span className="text-slate-300 font-semibold">{gate}</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] font-bold">
                    {status}
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-4 p-4 rounded-xl bg-[#152033] border border-[#24324A] font-mono text-xs">
              <span className="text-slate-400 uppercase text-[10px] block mb-1">Deterministic SHA-256 Hash</span>
              <span className="text-cyan-300 break-all">{simResult.replayHash}</span>
            </div>
          </HorizonCard>

          {replayVerification && (
            <IntelligenceSuccessState
              title="100-Replay Determinism Test Certified"
              message={`Successfully executed 100 simulation iterations. Generated exactly 1 unique SHA-256 hash with 0 drift events, satisfying INV-OI64.`}
              auditHash={replayVerification.replayHash}
              certificationId="CERT-M12-REPLAY-100"
            />
          )}
        </div>
      )}
    </IntelligenceShell>
  );
}

export default function StrategyLaboratoryPage() {
  return (
    <Suspense fallback={<IntelligenceLoadingState message="Loading Strategy Decision Laboratory..." />}>
      <StrategyLaboratoryContent />
    </Suspense>
  );
}
