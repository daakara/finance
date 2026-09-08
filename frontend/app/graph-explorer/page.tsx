"use client";

import React, { useState, Suspense } from "react";
import Link from "next/link";
import IntelligenceShell from "../../components/ui/IntelligenceShell";
import HorizonCard from "../../components/ui/HorizonCard";

interface GraphNode {
  id: string;
  label: string;
  type: "COMMITTEE" | "DECISION" | "OUTCOME" | "LEARNING" | "RISK" | "RECOMMENDATION";
  status: "OPTIMAL" | "CERTIFIED" | "ACTIVE" | "RESOLVED";
  x: number;
  y: number;
  description: string;
  targetRoute: string;
}

interface GraphEdge {
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
    x: 100,
    y: 180,
    description: "Primary capital allocation committee (ODEI 86.4, CDQI 82.1).",
    targetRoute: "/committee-intelligence?committeeId=COM-001",
  },
  {
    id: "DEC-001",
    label: "Strategic Tech Allocation",
    type: "DECISION",
    status: "CERTIFIED",
    x: 340,
    y: 180,
    description: "Approved \$12M equity tranche in automated infrastructure.",
    targetRoute: "/decision-explorer?decisionId=DEC-001",
  },
  {
    id: "OUT-001",
    label: "+\$420k Alpha Realized",
    type: "OUTCOME",
    status: "OPTIMAL",
    x: 580,
    y: 180,
    description: "Excess return verified at +4.8% against benchmark.",
    targetRoute: "/audit-explorer?queryId=OUT-001",
  },
  {
    id: "LRN-001",
    label: "Minervini Stage Filter Rule",
    type: "LEARNING",
    status: "CERTIFIED",
    x: 820,
    y: 180,
    description: "Systematized 200-day MA slope rule for future tech tranches.",
    targetRoute: "/learning-intelligence?tab=knowledge",
  },
  {
    id: "RSK-001",
    label: "Sector Concentration Risk",
    type: "RISK",
    status: "RESOLVED",
    x: 1060,
    y: 180,
    description: "Residual correlation hedge verified below 15% stress floor.",
    targetRoute: "/risks-and-groupthink?tab=matrix",
  },
  {
    id: "REC-001",
    label: "Autonomous Rebalance #14",
    type: "RECOMMENDATION",
    status: "ACTIVE",
    x: 1300,
    y: 180,
    description: "Algorithmic adjustment pending human executive sign-off.",
    targetRoute: "/coaching-intelligence?tab=interventions",
  },
];

const CANONICAL_EDGES: GraphEdge[] = [
  { from: "COM-001", to: "DEC-001", label: "Ratifies" },
  { from: "DEC-001", to: "OUT-001", label: "Yields" },
  { from: "OUT-001", to: "LRN-001", label: "Codifies" },
  { from: "LRN-001", to: "RSK-001", label: "Mitigates" },
  { from: "RSK-001", to: "REC-001", label: "Drives" },
];

function GraphExplorerContent() {
  const [selectedNode, setSelectedNode] = useState<GraphNode>(CANONICAL_NODES[1]);

  return (
    <IntelligenceShell
      title="Universal Relationship & Graph Explorer"
      subtitle="Complete Lineage Traversal: Committee -> Decision -> Outcome -> Learning -> Risk -> Recommendation"
      badge="PHASE 31-M11 CERTIFIED"
      activeNavTab="/graph-explorer"
    >
      {/* Interactive Lineage Canvas */}
      <HorizonCard
        title="Institutional Causal Lineage (Zero External Dependencies)"
        subtitle="Click any node in the lineage chain to inspect upstream/downstream causal links"
      >
        <div className="overflow-x-auto py-6 bg-[#0B1220] rounded-xl border border-[#24324A] p-4">
          <svg className="w-[1450px] h-[340px]" viewBox="0 0 1450 340">
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M 0 1 L 8 5 L 0 9 z" fill="#06B6D4" />
              </marker>
            </defs>

            {/* Render Edges */}
            {CANONICAL_EDGES.map((e, idx) => {
              const fromN = CANONICAL_NODES.find((n) => n.id === e.from)!;
              const toN = CANONICAL_NODES.find((n) => n.id === e.to)!;
              const midX = (fromN.x + toN.x) / 2;
              return (
                <g key={idx}>
                  <line
                    x1={fromN.x + 60}
                    y1={fromN.y}
                    x2={toN.x - 60}
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
                    fontSize="11"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    {e.label}
                  </text>
                </g>
              );
            })}

            {/* Render Nodes */}
            {CANONICAL_NODES.map((node) => {
              const isSelected = selectedNode.id === node.id;
              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x}, ${node.y})`}
                  onClick={() => setSelectedNode(node)}
                  className="cursor-pointer transition-transform hover:scale-105"
                >
                  <circle
                    r="48"
                    fill={isSelected ? "#1E2A45" : "#121B2A"}
                    stroke={isSelected ? "#38BDF8" : "#24324A"}
                    strokeWidth={isSelected ? "3" : "1.5"}
                  />
                  <text
                    y="-12"
                    fill="#38BDF8"
                    fontSize="11"
                    fontWeight="bold"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    {node.id}
                  </text>
                  <text
                    y="6"
                    fill="#F8FAFC"
                    fontSize="10"
                    fontFamily="sans-serif"
                    fontWeight="600"
                    textAnchor="middle"
                  >
                    {node.type}
                  </text>
                  <text
                    y="22"
                    fill="#10B981"
                    fontSize="9"
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
          subtitle="Causal Context, Metadata, and Primary Route Resolution"
          badge={
            <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 text-[10px] font-mono font-semibold">
              {selectedNode.status}
            </span>
          }
          actions={
            <Link
              href={selectedNode.targetRoute}
              className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold transition-colors"
            >
              Open Center View &rarr;
            </Link>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3.5 rounded-xl bg-[#182336] border border-[#24324A]">
              <span className="text-[10px] font-mono uppercase text-slate-400">Node Title</span>
              <div className="text-sm font-semibold text-white mt-1">{selectedNode.label}</div>
            </div>
            <div className="p-3.5 rounded-xl bg-[#182336] border border-[#24324A]">
              <span className="text-[10px] font-mono uppercase text-slate-400">Canonical Path</span>
              <div className="text-xs font-mono text-cyan-300 mt-1 truncate">{selectedNode.targetRoute}</div>
            </div>
            <div className="p-3.5 rounded-xl bg-[#182336] border border-[#24324A]">
              <span className="text-[10px] font-mono uppercase text-slate-400">Multi-Hop Position</span>
              <div className="text-xs font-mono text-emerald-400 mt-1">
                Hop {CANONICAL_NODES.findIndex((n) => n.id === selectedNode.id) + 1} of {CANONICAL_NODES.length}
              </div>
            </div>
          </div>
          <div className="mt-4 p-4 rounded-xl bg-[#182336]/60 border border-[#24324A]">
            <span className="text-xs font-mono text-slate-400 uppercase tracking-wider block mb-1">
              Institutional Context & Rationale:
            </span>
            <p className="text-xs text-slate-300 leading-relaxed font-mono">
              {selectedNode.description}
            </p>
          </div>
        </HorizonCard>
      )}
    </IntelligenceShell>
  );
}

export default function GraphExplorerPage() {
  return (
    <Suspense fallback={<div className="p-8 font-mono text-cyan-400">Loading Graph Explorer...</div>}>
      <GraphExplorerContent />
    </Suspense>
  );
}
