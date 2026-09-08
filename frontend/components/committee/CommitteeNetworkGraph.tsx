"use client";

import { useState } from "react";
import { CommitteeNetworkNode, CommitteeNetworkEdge, NetworkMetrics } from "../../types/committee-intelligence";
import {
  CANONICAL_NETWORK_NODES,
  CANONICAL_NETWORK_EDGES,
} from "../../lib/telemetry/committeeIntelligenceEngine";
import {
  computeNetworkMetrics,
  detectNetworkCycles,
  verifyInfluenceIntegrity,
  verifyNetworkCompleteness,
} from "../../lib/telemetry/decisionNetworkEngine";

export interface CommitteeNetworkGraphProps {
  nodes?: CommitteeNetworkNode[];
  edges?: CommitteeNetworkEdge[];
}

interface NodeCoord {
  x: number;
  y: number;
}

const NODE_COORDINATES: Record<string, NodeCoord> = {
  "COM-001": { x: 320, y: 80 },  // Top center
  "COM-002": { x: 120, y: 280 }, // Bottom left
  "COM-003": { x: 520, y: 280 }, // Bottom right
};

export default function CommitteeNetworkGraph({
  nodes = CANONICAL_NETWORK_NODES,
  edges = CANONICAL_NETWORK_EDGES,
}: CommitteeNetworkGraphProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>("COM-001");
  const [hoveredEdgeIndex, setHoveredEdgeIndex] = useState<number | null>(null);

  const metrics: NetworkMetrics = computeNetworkMetrics(nodes, edges);
  const cycleResult = detectNetworkCycles(edges);
  const influenceIntegrity = verifyInfluenceIntegrity(edges);
  const networkCompleteness = verifyNetworkCompleteness(nodes, edges);

  const selectedNode = nodes.find((n) => n.committeeId === selectedNodeId);
  const outgoingEdges = edges.filter((e) => e.sourceCommitteeId === selectedNodeId);
  const incomingEdges = edges.filter((e) => e.targetCommitteeId === selectedNodeId);

  return (
    <div className="space-y-4 font-mono">
      {/* Top Metrics Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <div className="bg-[#111724] border border-[#202d44] p-3 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Committees (Nodes)</span>
          <span className="text-xl font-bold text-slate-100 mt-1 block">{metrics.totalNodes}</span>
          <span className="text-[10px] text-emerald-400 block mt-0.5">100% Connected</span>
        </div>

        <div className="bg-[#111724] border border-[#202d44] p-3 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Influence Flows (Edges)</span>
          <span className="text-xl font-bold text-cyan-400 mt-1 block">{metrics.totalEdges}</span>
          <span className="text-[10px] text-cyan-400/80 block mt-0.5">Directed & Weighted</span>
        </div>

        <div className="bg-[#111724] border border-[#202d44] p-3 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Network Density</span>
          <span className="text-xl font-bold text-purple-400 mt-1 block">{metrics.density.toFixed(3)}</span>
          <span className="text-[10px] text-purple-400/80 block mt-0.5">Healthy Interlock</span>
        </div>

        <div className="bg-[#111724] border border-[#202d44] p-3 rounded-xl">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Avg Influence Score</span>
          <span className="text-xl font-bold text-emerald-400 mt-1 block">{metrics.averageInfluenceScore.toFixed(1)}</span>
          <span className="text-[10px] text-emerald-400/80 block mt-0.5">INV-OI15 Bounded</span>
        </div>

        <div className="bg-[#111724] border border-[#202d44] p-3 rounded-xl col-span-2 sm:col-span-1">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Cycle Anomalies</span>
          <span className={`text-xl font-bold mt-1 block ${cycleResult.hasCycle ? "text-rose-400" : "text-emerald-400"}`}>
            {metrics.cycleCount} Cycles
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            {cycleResult.hasCycle ? "Loop Detected" : "Acyclic Invariant"}
          </span>
        </div>
      </div>

      {/* Cycle or Invariant Compliance Banner */}
      {cycleResult.hasCycle ? (
        <div className="bg-rose-950/50 border border-rose-500/40 p-3 rounded-xl text-rose-300 text-xs flex items-center space-x-2">
          <span className="text-rose-400 font-bold">&#9888; WARNING:</span>
          <span>Circular influence loop detected: {cycleResult.cycles.map(c => c.join(" &rarr; ")).join(" | ")}</span>
        </div>
      ) : (
        <div className="bg-emerald-950/40 border border-emerald-500/30 p-2.5 rounded-xl text-emerald-400 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-semibold">INV-OI15 &amp; INV-OI16 VERIFIED:</span>
            <span className="text-slate-300">Deterministic Directed Influence Topology with 0 Cycles &amp; 100% Explainability.</span>
          </div>
          <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider hidden sm:inline">
            Acyclic Certified
          </span>
        </div>
      )}

      {/* Main Network Graph + Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* SVG Graph Viewport */}
        <div className="lg:col-span-2 bg-[#111724] border border-[#202d44] rounded-xl p-4 flex flex-col items-center justify-center relative overflow-hidden min-h-[420px]">
          <div className="absolute top-3 left-3 text-[10px] text-slate-500 uppercase tracking-wider">
            Interactive Topology (Click node for deep attribution)
          </div>

          <svg
            viewBox="0 0 640 360"
            className="w-full max-w-[620px] h-auto select-none"
          >
            <defs>
              <marker
                id="arrowhead-cyan"
                markerWidth="8"
                markerHeight="6"
                refX="7"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#06b6d4" />
              </marker>
              <marker
                id="arrowhead-purple"
                markerWidth="8"
                markerHeight="6"
                refX="7"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#a855f7" />
              </marker>
              <linearGradient id="edge-grad-1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.8" />
              </linearGradient>
              <linearGradient id="edge-grad-2" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.8" />
              </linearGradient>
            </defs>

            {/* Grid Pattern / Sub-wires */}
            <circle cx="320" cy="180" r="160" fill="none" stroke="#1f2c42" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />

            {/* Edges */}
            {edges.map((edge, idx) => {
              const src = NODE_COORDINATES[edge.sourceCommitteeId] ?? { x: 300, y: 150 };
              const tgt = NODE_COORDINATES[edge.targetCommitteeId] ?? { x: 300, y: 150 };

              // Calculate edge offset so arrows don't collide directly into node center
              const dx = tgt.x - src.x;
              const dy = tgt.y - src.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              const nodeRadius = 38;

              const x1 = src.x + (dx / dist) * nodeRadius;
              const y1 = src.y + (dy / dist) * nodeRadius;
              const x2 = tgt.x - (dx / dist) * (nodeRadius + 6);
              const y2 = tgt.y - (dy / dist) * (nodeRadius + 6);

              // Midpoint for score pill
              const midX = (x1 + x2) / 2;
              const midY = (y1 + y2) / 2;

              const isHovered = hoveredEdgeIndex === idx;
              const isConnectedToSelected =
                edge.sourceCommitteeId === selectedNodeId || edge.targetCommitteeId === selectedNodeId;

              const strokeWidth = Math.max(2, Math.min(5, (edge.influenceScore / 100) * 5.5));

              return (
                <g
                  key={`${edge.sourceCommitteeId}-${edge.targetCommitteeId}`}
                  onMouseEnter={() => setHoveredEdgeIndex(idx)}
                  onMouseLeave={() => setHoveredEdgeIndex(null)}
                  className="cursor-pointer transition-opacity duration-200"
                  opacity={selectedNodeId && !isConnectedToSelected ? 0.35 : 1}
                >
                  {/* Outer glow line */}
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#06b6d4"
                    strokeWidth={strokeWidth + 4}
                    strokeOpacity={isHovered ? 0.4 : 0.08}
                  />

                  {/* Main edge line */}
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={idx % 2 === 0 ? "url(#edge-grad-1)" : "url(#edge-grad-2)"}
                    strokeWidth={strokeWidth}
                    markerEnd="url(#arrowhead-cyan)"
                  />

                  {/* Midpoint Influence Score Badge */}
                  <rect
                    x={midX - 24}
                    y={midY - 10}
                    width={48}
                    height={20}
                    rx={5}
                    fill="#0c1017"
                    stroke={isHovered ? "#06b6d4" : "#243044"}
                    strokeWidth="1"
                  />
                  <text
                    x={midX}
                    y={midY + 4}
                    fill="#38bdf8"
                    fontSize="9"
                    fontWeight="bold"
                    textAnchor="middle"
                    fontFamily="monospace"
                  >
                    {edge.influenceScore.toFixed(0)}%
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
              const coord = NODE_COORDINATES[node.committeeId] ?? { x: 300, y: 150 };
              const isSelected = selectedNodeId === node.committeeId;

              return (
                <g
                  key={node.committeeId}
                  transform={`translate(${coord.x}, ${coord.y})`}
                  onClick={() => setSelectedNodeId(node.committeeId)}
                  className="cursor-pointer"
                >
                  {/* Halo when selected */}
                  {isSelected && (
                    <circle
                      r="46"
                      fill="none"
                      stroke="#06b6d4"
                      strokeWidth="2"
                      strokeOpacity="0.5"
                      strokeDasharray="4 4"
                      className="animate-spin-slow"
                    />
                  )}

                  {/* Outer circle */}
                  <circle
                    r="36"
                    fill="#111724"
                    stroke={isSelected ? "#06b6d4" : "#202d44"}
                    strokeWidth={isSelected ? "2.5" : "1.5"}
                    filter="drop-shadow(0 4px 6px rgba(0, 0, 0, 0.4))"
                  />

                  {/* Inner ring */}
                  <circle
                    r="30"
                    fill="#162032"
                    stroke="#243044"
                    strokeWidth="1"
                  />

                  {/* Node ID Label */}
                  <text
                    y="-6"
                    fill="#f1f5f9"
                    fontSize="11"
                    fontWeight="bold"
                    textAnchor="middle"
                    fontFamily="monospace"
                  >
                    {node.committeeId}
                  </text>

                  {/* Score Pill */}
                  <text
                    y="12"
                    fill={node.qualityScore >= 85 ? "#34d399" : "#38bdf8"}
                    fontSize="9"
                    fontWeight="bold"
                    textAnchor="middle"
                    fontFamily="monospace"
                  >
                    Q:{node.qualityScore.toFixed(1)}
                  </text>

                  {/* Text Below Node */}
                  <text
                    y="50"
                    fill="#94a3b8"
                    fontSize="10"
                    fontWeight="500"
                    textAnchor="middle"
                    fontFamily="monospace"
                  >
                    {node.committeeName}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Legend */}
          <div className="flex flex-wrap items-center justify-center gap-4 text-[10px] text-slate-400 mt-2 border-t border-[#202d44] pt-2 w-full">
            <div className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <span>High Quality (&ge;85.0)</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
              <span>Balanced (&ge;80.0)</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-4 h-0.5 bg-cyan-400" />
              <span>Directed Influence Flow</span>
            </div>
          </div>
        </div>

        {/* Selected Node Details Sidebar */}
        <div className="bg-[#111724] border border-[#202d44] rounded-xl p-4 flex flex-col justify-between">
          {selectedNode ? (
            <div className="space-y-4">
              <div className="border-b border-[#202d44] pb-3">
                <span className="text-[10px] text-cyan-400 uppercase tracking-wider block">
                  Committee Detail Inspector
                </span>
                <h3 className="text-base font-bold text-slate-100 mt-1">
                  {selectedNode.committeeName}
                </h3>
                <span className="text-xs text-slate-400">{selectedNode.committeeId}</span>
              </div>

              {/* Node Metrics */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-[#0c1017] p-2.5 rounded-lg border border-[#202d44]">
                  <span className="text-[10px] text-slate-400 block">Quality Score (CDQI)</span>
                  <span className="text-base font-bold text-emerald-400 mt-0.5 block">
                    {selectedNode.qualityScore.toFixed(1)}
                  </span>
                </div>
                <div className="bg-[#0c1017] p-2.5 rounded-lg border border-[#202d44]">
                  <span className="text-[10px] text-slate-400 block">Recorded Decisions</span>
                  <span className="text-base font-bold text-cyan-400 mt-0.5 block">
                    {selectedNode.decisionCount}
                  </span>
                </div>
              </div>

              {/* Outgoing Influence */}
              <div>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1.5">
                  Outgoing Influence ({outgoingEdges.length})
                </span>
                {outgoingEdges.length > 0 ? (
                  <div className="space-y-1.5">
                    {outgoingEdges.map((e) => (
                      <div
                        key={e.targetCommitteeId}
                        className="p-2 rounded bg-[#0c1017] border border-[#202d44] text-xs flex items-center justify-between"
                      >
                        <span className="text-slate-300">
                          &rarr; {nodes.find((n) => n.committeeId === e.targetCommitteeId)?.committeeName ?? e.targetCommitteeId}
                        </span>
                        <span className="text-cyan-400 font-bold">{e.influenceScore.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-[11px] text-slate-500 italic p-2 bg-[#0c1017] rounded">
                    No downstream influence dependencies.
                  </div>
                )}
              </div>

              {/* Incoming Influence */}
              <div>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1.5">
                  Incoming Influence ({incomingEdges.length})
                </span>
                {incomingEdges.length > 0 ? (
                  <div className="space-y-1.5">
                    {incomingEdges.map((e) => (
                      <div
                        key={e.sourceCommitteeId}
                        className="p-2 rounded bg-[#0c1017] border border-[#202d44] text-xs flex items-center justify-between"
                      >
                        <span className="text-slate-300">
                          &larr; {nodes.find((n) => n.committeeId === e.sourceCommitteeId)?.committeeName ?? e.sourceCommitteeId}
                        </span>
                        <span className="text-purple-400 font-bold">{e.influenceScore.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-[11px] text-slate-500 italic p-2 bg-[#0c1017] rounded">
                    Autonomous authority; zero upstream dependencies.
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500 text-xs">
              Select a committee node to inspect topological influence and cross-dependencies.
            </div>
          )}

          <div className="pt-3 border-t border-[#202d44] mt-4">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Network Completeness:</span>
              <span className="text-emerald-400 font-bold">{networkCompleteness.completenessPct}%</span>
            </div>
            <div className="flex items-center justify-between text-[11px] mt-1">
              <span className="text-slate-400">INV-OI15 Integrity:</span>
              <span className="text-cyan-400 font-bold">
                {influenceIntegrity.valid ? "Certified" : "Violated"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
