'use client';

/**
 * Phase 29: Knowledge Graph Explorer — Institutional Memory
 *
 * OI-103: Search and explore the institutional knowledge graph.
 * Sections:
 * 1. Graph Stats Banner (node counts, relationship coverage)
 * 2. Learning Pattern Library (Success / Failure / Emerging)
 * 3. Knowledge Node Explorer (filterable list)
 */

import React, { useState } from 'react';
import { buildKnowledgeGraph, CANONICAL_LEARNING_PATTERNS } from '@/lib/telemetry/knowledgeGraphEngine';
import type { KnowledgeNodeType, LearningPatternType } from '@/types/organizational-intelligence';

const GRAPH = buildKnowledgeGraph();

function NodeTypeTag({ type }: { type: KnowledgeNodeType }) {
  const colors: Record<KnowledgeNodeType, string> = {
    DECISION: 'bg-blue-900 text-blue-300',
    PREDICTION: 'bg-purple-900 text-purple-300',
    OUTCOME: 'bg-green-900 text-green-300',
    LEARNING: 'bg-amber-900 text-amber-300',
    PLAYBOOK: 'bg-teal-900 text-teal-300',
    GOVERNANCE: 'bg-rose-900 text-rose-300',
  };
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${colors[type]}`}>{type}</span>
  );
}

function PatternTypeBadge({ type }: { type: LearningPatternType }) {
  const map: Record<LearningPatternType, { label: string; color: string }> = {
    SUCCESS: { label: 'SUCCESS', color: 'bg-green-900 text-green-300' },
    FAILURE: { label: 'FAILURE', color: 'bg-rose-900 text-rose-300' },
    EMERGING: { label: 'EMERGING', color: 'bg-amber-900 text-amber-300' },
  };
  const { label, color } = map[type];
  return <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${color}`}>{label}</span>;
}

function GraphStatsBanner() {
  const g = GRAPH;
  const stats = [
    { label: 'Decisions', value: g.decisionNodes, color: 'text-blue-400' },
    { label: 'Predictions', value: g.predictionNodes, color: 'text-purple-400' },
    { label: 'Outcomes', value: g.outcomeNodes, color: 'text-green-400' },
    { label: 'Learnings', value: g.learningNodes, color: 'text-amber-400' },
    { label: 'Playbooks', value: g.playbookNodes, color: 'text-teal-400' },
    { label: 'Governance', value: g.governanceNodes, color: 'text-rose-400' },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6" data-testid="knowledge-graph-stats" role="region" aria-label="Institutional Knowledge Graph Statistics">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-gray-400 text-sm uppercase tracking-widest">Institutional Knowledge Graph</p>
          <p className="text-white font-bold text-xl mt-0.5">
            {g.totalNodes} Nodes · {g.totalEdges} Edges · {g.relationshipCoverage}% Coverage
          </p>
        </div>
        <div className="text-right">
          <span className="text-xs font-bold px-3 py-1 rounded-full bg-green-900 text-green-400">
            ✅ Fully Connected
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
        {stats.map(s => (
          <div key={s.label} className="bg-gray-800 rounded-lg p-3 text-center">
            <p className={`text-2xl font-black ${s.color}`}>{s.value}</p>
            <p className="text-gray-400 text-xs mt-0.5">{s.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function LearningPatternLibrary() {
  const patterns = CANONICAL_LEARNING_PATTERNS;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6" data-testid="learning-pattern-library" role="region" aria-label="Institutional Learning Patterns">
      <h3 className="text-white font-bold text-lg mb-4">Learning Patterns</h3>
      <div className="space-y-4">
        {patterns.map(p => (
          <div key={p.patternId} className="bg-gray-800 rounded-lg p-4" data-testid={`pattern-${p.patternId}`}>
            <div className="flex items-start justify-between gap-4 mb-2">
              <div className="flex items-center gap-2">
                <PatternTypeBadge type={p.patternType} />
                <p className="text-white font-semibold text-sm">{p.title}</p>
              </div>
              <div className="text-right shrink-0">
                <p className="text-gray-400 text-xs">Confidence</p>
                <p className="text-white font-bold text-sm">{p.confidence}%</p>
              </div>
            </div>
            <p className="text-gray-300 text-sm mb-2">{p.description}</p>
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>{p.occurrences} occurrences · {p.affectedTeams.join(', ')}</span>
              <span className="text-green-400 font-semibold">{p.economicImpact}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function KnowledgeNodeExplorer() {
  const [filter, setFilter] = useState<KnowledgeNodeType | 'ALL'>('ALL');
  const nodeTypes: Array<KnowledgeNodeType | 'ALL'> = ['ALL', 'DECISION', 'PREDICTION', 'OUTCOME', 'LEARNING', 'PLAYBOOK', 'GOVERNANCE'];
  const filtered = filter === 'ALL' ? GRAPH.nodes : GRAPH.nodes.filter(n => n.type === filter);

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6" data-testid="knowledge-node-explorer" role="region" aria-label="Knowledge Node Explorer">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h3 className="text-white font-bold text-lg">Institutional Memory Explorer</h3>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter knowledge nodes by type">
          {nodeTypes.map(t => (
            <button
              key={t}
              onClick={() => setFilter(t)}
              className={`text-xs px-3 py-1.5 rounded-full font-semibold transition-colors min-h-[44px] min-w-[44px] ${filter === t ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
              aria-pressed={filter === t}
            >
              {t}
            </button>
          ))}
        </div>
      </div>
      <div className="space-y-2 max-h-80 overflow-y-auto">
        {filtered.map(node => (
          <div key={node.nodeId} className="flex items-center gap-3 bg-gray-800 rounded-lg px-4 py-2.5" data-testid={`node-${node.nodeId}`}>
            <NodeTypeTag type={node.type} />
            <div className="flex-1 min-w-0">
              <p className="text-white text-sm font-medium truncate">{node.title}</p>
              <p className="text-gray-400 text-xs">{node.teamId} · {node.createdAt}</p>
            </div>
            <div className="text-right shrink-0">
              <p className="text-gray-400 text-xs">{node.confidence}% CI</p>
              <p className="text-gray-500 text-xs">{node.linkedNodes.length} links</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function KnowledgeGraphExplorer() {
  return (
    <div role="main" aria-label="Institutional Knowledge Graph Explorer">
      <GraphStatsBanner />
      <LearningPatternLibrary />
      <KnowledgeNodeExplorer />
    </div>
  );
}

