"use client";

import React, { useState, Suspense } from "react";
import Link from "next/link";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import IntelligenceHeader from "../../components/ui/IntelligenceHeader";
import HorizonCard from "../../components/ui/HorizonCard";
import SeverityBadge from "../../components/ui/SeverityBadge";
import RelatedArtifactsPanel from "../../components/ui/RelatedArtifactsPanel";

export interface GraphNode {
  id: string;
  label: string;
  type: "COMMITTEE" | "DECISION" | "OUTCOME" | "LEARNING" | "RISK" | "RECOMMENDATION" | "INCIDENT" | "SCENARIO" | "RUNBOOK";
  status: "OPTIMAL" | "CERTIFIED" | "ACTIVE" | "RESOLVED" | "WARNING" | "CRITICAL";
  x: number;
  y: number;
  description: string;
  targetRoute: string;
  upstreamIds: string[];
  downstreamIds: string[];
}

export interface GraphEdge {
  from: string;
  to: string;
  label: string;
}

const CANONICAL_NODES: GraphNode[] = [
  {
    id: "COM-001",
    label: "Investment Committee",
    type: "COMMITTEE",
    status: "OPTIMAL",
    x: 80,
    y: 180,
    description: "Primary capital allocation committee (ODEI 86.4, CDQI 82.1).",
    targetRoute: "/committee-intelligence?committeeId=COM-001",
    upstreamIds: [],
    downstreamIds: ["DEC-001"],
  },
  {
    id: "DEC-001",
    label: "Strategic Tech Allocation",
    type: "DECISION",
    status: "CERTIFIED",
    x: 260,
    y: 180,
    description: "Approved $12M equity tranche in automated algorithmic infrastructure.",
    targetRoute: "/decision-explorer?decisionId=DEC-001",
    upstreamIds: ["COM-001"],
    downstreamIds: ["OUT-001"],
  },
  {
    id: "OUT-001",
    label: "+$420k Alpha Realized",
    type: "OUTCOME",
    status: "OPTIMAL",
    x: 440,
    y: 180,
    description: "Excess return verified at +4.8% against S&P 500 benchmark.",
    targetRoute: "/audit-explorer?queryId=OUT-001",
    upstreamIds: ["DEC-001"],
    downstreamIds: ["LRN-001"],
  },
  {
    id: "LRN-001",
    label: "Minervini Stage Filter",
    type: "LEARNING",
    status: "CERTIFIED",
    x: 620,
    y: 180,
    description: "Systematized 200-day MA slope rule for future tech allocation tranches.",
    targetRoute: "/learning-intelligence?tab=knowledge",
    upstreamIds: ["OUT-001"],
    downstreamIds: ["RSK-001"],
  },
  {
    id: "RSK-001",
    label: "Sector Concentration Shock",
    type: "RISK",
    status: "RESOLVED",
    x: 800,
    y: 180,
    description: "Residual correlation hedge verified below 15% stress floor.",
    targetRoute: "/risks-and-groupthink?tab=matrix",
    upstreamIds: ["LRN-001"],
    downstreamIds: ["REC-001"],
  },
  {
    id: "REC-001",
    label: "Autonomous Rebalance #14",
    type: "RECOMMENDATION",
    status: "ACTIVE",
    x: 980,
    y: 180,
    description: "Algorithmic adjustment pending human executive authorization.",
    targetRoute: "/coaching-intelligence?tab=interventions",
    upstreamIds: ["RSK-001"],
    downstreamIds: ["INC-001"],
  },
  {
    id: "INC-001",
    label: "Concentration Drift Event",
    type: "INCIDENT",
    status: "RESOLVED",
    x: 1160,
    y: 180,
    description: "Transient sector exposure surge resolved within 4.8s failover SLA.",
    targetRoute: "/resilience-intelligence?tab=incidents",
    upstreamIds: ["REC-001"],
    downstreamIds: ["SCN-001"],
  },
  {
    id: "SCN-001",
    label: "Stressed Rates 360d Scenario",
    type: "SCENARIO",
    status: "CERTIFIED",
    x: 1340,
    y: 180,
    description: "Multi-regime forward projection under adverse liquidity conditions.",
    targetRoute: "/simulation-intelligence?simId=FUT-001",
    upstreamIds: ["INC-001"],
    downstreamIds: ["RB-001"],
  },
  {
    id: "RB-001",
    label: "Runbook M9-RB-02",
    type: "RUNBOOK",
    status: "CERTIFIED",
    x: 1520,
    y: 180,
    description: "Deterministic replay lock protocol and snapshot chain verification.",
    targetRoute: "/action-center",
    upstreamIds: ["SCN-001"],
    downstreamIds: [],
  },
];

const CANONICAL_EDGES: GraphEdge[] = [
  { from: "COM-001", to: "DEC-001", label: "Ratifies" },
  { from: "DEC-001", to: "OUT-001", label: "Yields" },
  { from: "OUT-001", to: "LRN-001", label: "Codifies" },
  { from: "LRN-001", to: "RSK-001", label: "Mitigates" },
  { from: "RSK-001", to: "REC-001", label: "Drives" },
  { from: "REC-001", to: "INC-001", label: "Triggers" },
  { from: "INC-001", to: "SCN-001", label: "Simulates" },
  { from: "SCN-001", to: "RB-001", label: "Executes" },
];

function GraphExplorerContent() {
  const [selectedNode, setSelectedNode] = useState<GraphNode>(CANONICAL_NODES[1]);
  const [filterType, setFilterType] = useState<string>("ALL");

  const filteredNodes = CANONICAL_NODES.filter(
    (n) => filterType === "ALL" || n.type === filterType
  );

  return (
    <IntelligenceShell
      title="Universal Relationship & Graph Explorer"
      subtitle="Complete Lineage Traversal across 9 Institutional Node Types"
      badge="PHASE 31-M15 CERTIFIED"
      activeNavTab="/graph-explorer"
    >
      <div className="space-y-6">
        <IntelligenceHeader
          title="Institutional Causal Lineage Graph"
          subtitle="Committee -> Decision -> Outcome -> Learning -> Risk -> Recommendation -> Incident -> Scenario -> Runbook"
          status="CERTIFIED"
          certification="M15 CERTIFIED"
        />

        {/* Filter Controls */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs font-mono">
          <span className="text-slate-400 shrink-0">Filter Nodes:</span>
          {['ALL', 'COMMITTEE', 'DECISION', 'OUTCOME', 'LEARNING', 'RISK', 'RECOMMENDATION', 'INCIDENT', 'SCENARIO', 'RUNBOOK'].map((t) => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={`px-2.5 py-1 rounded transition-colors shrink-0 ${
                filterType === t
                  ? 'bg-cyan-600 text-white font-semibold'
                  : 'bg-[#121B2A] text-slate-400 hover:text-slate-200 border border-[#24324A]'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Interactive Lineage Canvas */}
        <HorizonCard
          title="Universal Causal Graph (9 Institutional Nodes)"
          subtitle="Click any node in the lineage chain to inspect upstream inputs, downstream blast radius, and related artifacts"
        >
          <div className="overflow-x-auto py-6 bg-[#0B1220] rounded-xl border border-[#24324A] p-4">
            <svg className="w-[1650px] h-[340px]" viewBox="0 0 1650 340">
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M 0 1 L 8 5 L 0 9 z" fill="#06B6D4" />
                </marker>
              </defs>

              {/* Render Edges */}
              {CANONICAL_EDGES.map((e, idx) => {
                const fromN = CANONICAL_NODES.find((n) => n.id === e.from);
                const toN = CANONICAL_NODES.find((n) => n.id === e.to);
                if (!fromN || !toN) return null;
                const midX = (fromN.x + toN.x) / 2;
                return (
                  <g key={idx}>
                    <line
                      x1={fromN.x + 55}
                      y1={fromN.y}
                      x2={toN.x - 55}
                      y2={toN.y}
                      stroke="#06B6D4"
                      strokeWidth="2"
                      strokeDasharray="4 2"
                      markerEnd="url(#arrow)"
                    />
                    <text
                      x={midX}
                      y={fromN.y - 12}
                      fill="#94A3B8"
                      fontSize="10"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {e.label}
                    </text>
                  </g>
                );
              })}

              {/* Render Nodes */}
              {filteredNodes.map((node) => {
                const isSelected = selectedNode.id === node.id;
                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    onClick={() => setSelectedNode(node)}
                    className="cursor-pointer transition-transform hover:scale-105"
                  >
                    <circle
                      r="46"
                      fill={isSelected ? "#1E2A45" : "#121B2A"}
                      stroke={isSelected ? "#38BDF8" : "#24324A"}
                      strokeWidth={isSelected ? "3" : "1.5"}
                    />
                    <text
                      y="-12"
                      fill="#38BDF8"
                      fontSize="10"
                      fontWeight="bold"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {node.id}
                    </text>
                    <text
                      y="6"
                      fill="#F8FAFC"
                      fontSize="9"
                      fontFamily="sans-serif"
                      fontWeight="600"
                      textAnchor="middle"
                    >
                      {node.type}
                    </text>
                    <text
                      y="22"
                      fill="#10B981"
                      fontSize="8"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {node.status}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </HorizonCard>

        {/* Selected Node Inspector */}
        {selectedNode && (
          <HorizonCard
            title={`Node Inspector: ${selectedNode.id} (${selectedNode.type})`}
            subtitle="Upstream Inputs, Downstream Causal Consequences & Route Resolution"
            badge={<SeverityBadge status={selectedNode.status} size="sm" />}
            actions={
              <Link
                href={selectedNode.targetRoute}
                className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold transition-colors"
              >
                Open Center View &rarr;
              </Link>
            }
          >
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div className="p-3 rounded-xl bg-[#182336] border border-[#24324A]">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Node Identifier</span>
                  <div className="text-sm font-semibold text-white mt-0.5">{selectedNode.label}</div>
                  <div className="text-[10px] font-mono text-cyan-400 mt-0.5">ID: {selectedNode.id}</div>
                </div>
                <div className="p-3 rounded-xl bg-[#182336] border border-[#24324A]">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Upstream Causal Predecessors</span>
                  <div className="text-xs font-mono text-slate-200 mt-1">
                    {selectedNode.upstreamIds.length > 0 ? selectedNode.upstreamIds.join(', ') : 'Root Source Node'}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-[#182336] border border-[#24324A]">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Downstream Blast Radius</span>
                  <div className="text-xs font-mono text-slate-200 mt-1">
                    {selectedNode.downstreamIds.length > 0 ? selectedNode.downstreamIds.join(', ') : 'Terminal Execution Node'}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-[#182336] border border-[#24324A]">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Lineage Hop Depth</span>
                  <div className="text-xs font-mono text-emerald-400 mt-1">
                    Hop {CANONICAL_NODES.findIndex((n) => n.id === selectedNode.id) + 1} of {CANONICAL_NODES.length}
                  </div>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-[#182336]/60 border border-[#24324A]">
                <span className="text-xs font-mono text-slate-400 uppercase tracking-wider block mb-1">
                  Institutional Context & Rationale:
                </span>
                <p className="text-xs text-slate-300 leading-relaxed font-mono">
                  {selectedNode.description}
                </p>
              </div>

              {/* Related Artifacts Component */}
              <RelatedArtifactsPanel
                title={`Related Lineage Connections for ${selectedNode.id}`}
                artifacts={[
                  { id: selectedNode.id, type: selectedNode.type as any, title: selectedNode.label, href: selectedNode.targetRoute, badge: selectedNode.status },
                  ...(selectedNode.upstreamIds.map((uId) => ({
                    id: uId,
                    type: "COMMITTEE" as const,
                    title: `Predecessor: ${uId}`,
                    href: `/decision-explorer?id=${uId}`,
                    badge: "Upstream",
                  }))),
                  ...(selectedNode.downstreamIds.map((dId) => ({
                    id: dId,
                    type: "OUTCOME" as const,
                    title: `Successor: ${dId}`,
                    href: `/audit-explorer?id=${dId}`,
                    badge: "Downstream",
                  }))),
                ]}
              />
            </div>
          </HorizonCard>
        )}
      </div>
    </IntelligenceShell>
  );
}

export default function GraphExplorerPage() {
  return (
    <Suspense fallback={<div className="p-8 font-mono text-cyan-400">Loading Universal Graph Explorer...</div>}>
      <GraphExplorerContent />
    </Suspense>
  );
}
