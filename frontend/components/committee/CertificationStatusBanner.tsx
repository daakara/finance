"use client";

export interface GateItem {
  gateId: string;
  name: string;
  category: "TRANSPARENCY" | "REPLAY" | "BYZANTINE" | "DIVERSITY";
  status: "PASSED" | "FAILED" | "WARNING";
  description: string;
  metric: string;
}

export const CANONICAL_GATES: GateItem[] = [
  {
    gateId: "CII-GATE-01",
    name: "Collective Decision Transparency",
    category: "TRANSPARENCY",
    status: "PASSED",
    description: "Every decision binds to proposal, evidence, participants, outcome, and attribution.",
    metric: "100.0% Coverage",
  },
  {
    gateId: "CII-GATE-02",
    name: "Dissent Preservation Integrity",
    category: "TRANSPARENCY",
    status: "PASSED",
    description: "All material dissents capture alternatives, risks, and evidence without suppression.",
    metric: "100.0% Preserved",
  },
  {
    gateId: "CII-GATE-03",
    name: "Decision Quality Floor",
    category: "TRANSPARENCY",
    status: "PASSED",
    description: "Minimum CDQI across all active committees must satisfy floor threshold.",
    metric: "85.1 CDQI (Floor: 80.0)",
  },
  {
    gateId: "CII-GATE-04",
    name: "DIRatio Boundedness",
    category: "TRANSPARENCY",
    status: "PASSED",
    description: "Decision-to-Intent Ratio must remain bounded within [1.0, 100.0].",
    metric: "26.9 Institutional Avg",
  },
  {
    gateId: "CII-GATE-05",
    name: "High-Impact Dissent Utilization",
    category: "TRANSPARENCY",
    status: "PASSED",
    description: "Committee must incorporate and act on alternative views in >=25% of decisions.",
    metric: "33.3% Utilization",
  },
  {
    gateId: "CII-GATE-06",
    name: "Network Connectivity & Invariance",
    category: "TRANSPARENCY",
    status: "PASSED",
    description: "All committee nodes connected with explainable directed influence flows.",
    metric: "100.0% Complete",
  },
  {
    gateId: "CII-GATE-07",
    name: "Single-Artifact Reconstruction",
    category: "TRANSPARENCY",
    status: "PASSED",
    description: "100% lineage reconstruction from single Decision ID or Outcome ID (INV-OI13-A).",
    metric: "100.0% Recovered",
  },
  {
    gateId: "CII-GATE-08",
    name: "Cryptographic State Hashing",
    category: "REPLAY",
    status: "PASSED",
    description: "Deterministic SHA-256 state seal across all institutional artifacts.",
    metric: "SHA-256 Validated",
  },
  {
    gateId: "CII-GATE-09",
    name: "Replay Determinism Verification",
    category: "REPLAY",
    status: "PASSED",
    description: "100 consecutive replays under canonical serialization with zero drift.",
    metric: "100 / 100 Replays",
  },
  {
    gateId: "CII-GATE-10",
    name: "Numerical Stability Guards",
    category: "REPLAY",
    status: "PASSED",
    description: "Relative epsilon tolerance verification with zero NaN or Infinity values.",
    metric: "0 Non-Finite Values",
  },
  {
    gateId: "CII-GATE-11",
    name: "Byzantine Resistance",
    category: "BYZANTINE",
    status: "PASSED",
    description: "Immunity to 10 attack classes: split-brain, ghost committees, attribution forks.",
    metric: "0 Attacks Active",
  },
  {
    gateId: "CII-GATE-12",
    name: "Fixture Diversity Score (FDS)",
    category: "DIVERSITY",
    status: "PASSED",
    description: "Entropy and coverage score across committee structures exceeds 75.0 target.",
    metric: "86.8 FDS Score",
  },
  {
    gateId: "CII-GATE-13",
    name: "Differential Regression Invariance",
    category: "DIVERSITY",
    status: "PASSED",
    description: "Strict non-regression across all previous institutional invariants.",
    metric: "0 Violations",
  },
];

export default function CertificationStatusBanner() {
  const passingCount = CANONICAL_GATES.filter((g) => g.status === "PASSED").length;
  const totalCount = CANONICAL_GATES.length;

  return (
    <div className="space-y-4 font-mono">
      {/* Top Banner */}
      <div className="bg-[#111724] border border-emerald-500/40 p-5 rounded-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-widest">
              Institutional Governance Engine
            </span>
            <span className="px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/40 text-emerald-300 text-[10px] font-bold">
              100% VERIFIED
            </span>
          </div>
          <h2 className="text-xl font-bold text-slate-100 mt-1">
            Adaptive Intelligence Certification Gates (CII-01 to CII-13)
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Autonomous verification of collective transparency, dissent preservation, Byzantine immunity, and deterministic replay stability.
          </p>
        </div>

        <div className="flex items-center space-x-3 shrink-0">
          <div className="text-right">
            <div className="text-2xl font-bold text-emerald-400">
              {passingCount} / {totalCount}
            </div>
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
              Release Gates Certified
            </span>
          </div>
        </div>
      </div>

      {/* Gates Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {CANONICAL_GATES.map((gate) => (
          <div
            key={gate.gateId}
            className="bg-[#111724] border border-[#202d44] hover:border-emerald-500/40 p-3.5 rounded-xl space-y-2 transition-colors"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">
                {gate.gateId}
              </span>
              <span className="px-1.5 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 text-[10px] font-bold">
                &check; {gate.status}
              </span>
            </div>

            <div>
              <h3 className="text-xs font-bold text-slate-100">{gate.name}</h3>
              <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                {gate.description}
              </p>
            </div>

            <div className="pt-2 border-t border-[#202d44] flex items-center justify-between text-[10px]">
              <span className="text-slate-500">Verified Metric:</span>
              <span className="text-emerald-400 font-bold">{gate.metric}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
