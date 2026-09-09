/**
 * Horizon 2: Directed Dependency Graph Engine
 *
 * Models causal organizational relationships as a Directed Acyclic Graph (DAG).
 * Provides cycle detection, topological pathing, transitive dependency discovery,
 * and impact propagation mathematics.
 */

import type {
  DependencyNode,
  DependencyEdge,
  DependencyGraph,
  DependencyDefinition,
  ImpactPath,
} from '../../types/simulation-digital-twin';

export const CANONICAL_DEPENDENCY_DEFINITIONS: DependencyDefinition[] = [
  {
    sourceMetric: 'TRAINING_BUDGET',
    targetMetric: 'LEARNING_VELOCITY',
    impactWeight: 0.8,
    relationship: 'INFLUENCES',
    confidencePct: 95,
    description: 'Direct investment in training curriculums expands institutional learning velocity',
  },
  {
    sourceMetric: 'LEARNING_VELOCITY',
    targetMetric: 'TRANSFER_RATE',
    impactWeight: 0.6,
    relationship: 'INFLUENCES',
    confidencePct: 92,
    description: 'Higher velocity of insight acquisition accelerates knowledge transfer between committees',
  },
  {
    sourceMetric: 'TRANSFER_RATE',
    targetMetric: 'DECISION_QUALITY',
    impactWeight: 0.7,
    relationship: 'INFLUENCES',
    confidencePct: 90,
    description: 'Systemic cross-functional transfer elevates executive decision quality and prevents blind spots',
  },
  {
    sourceMetric: 'DECISION_QUALITY',
    targetMetric: 'OHI',
    impactWeight: 0.85,
    relationship: 'INFLUENCES',
    confidencePct: 98,
    description: 'Rigorous decision quality directly lifts aggregate Organizational Health Index (OHI)',
  },
  {
    sourceMetric: 'GOVERNANCE_ADHERENCE',
    targetMetric: 'DECISION_QUALITY',
    impactWeight: 0.5,
    relationship: 'INFLUENCES',
    confidencePct: 94,
    description: 'Committee charter adherence and fail-closed policies enforce quality discipline',
  },
  {
    sourceMetric: 'COACHING_FREQUENCY',
    targetMetric: 'LEARNING_VELOCITY',
    impactWeight: 0.45,
    relationship: 'INFLUENCES',
    confidencePct: 88,
    description: 'Targeted coaching interventions amplify skill synthesis rate',
  },
  {
    sourceMetric: 'DISSENT_INTEGRATION',
    targetMetric: 'RISK_SCORE',
    impactWeight: -0.6,
    relationship: 'MITIGATES',
    confidencePct: 91,
    description: 'Rigorous integration of contrarian dissents dampens hidden operational & governance risk',
  },
  {
    sourceMetric: 'RISK_SCORE',
    targetMetric: 'OHI',
    impactWeight: -0.5,
    relationship: 'INFLUENCES',
    confidencePct: 95,
    description: 'Elevated aggregate risk score drags down organizational survivability and health',
  },
  {
    sourceMetric: 'RESILIENCE_INVESTMENT',
    targetMetric: 'RESILIENCE_RTO',
    impactWeight: -0.7,
    relationship: 'MITIGATES',
    confidencePct: 93,
    description: 'Investment in multi-level fallback runbooks compresses Recovery Time Objective (RTO)',
  },
  {
    sourceMetric: 'RESILIENCE_RTO',
    targetMetric: 'OHI',
    impactWeight: -0.3,
    relationship: 'INFLUENCES',
    confidencePct: 89,
    description: 'Prolonged recovery times reduce strategic confidence and organizational health',
  },
];

export const CANONICAL_NODES_CATALOG: DependencyNode[] = [
  { id: 'TRAINING_BUDGET', name: 'Training & Development Budget', type: 'BUDGET', state: { value: 1.0, baseline: 1.0, unitScale: 1000000 } },
  { id: 'LEARNING_VELOCITY', name: 'Organizational Learning Velocity', type: 'LEARNING', state: { value: 82.0, baseline: 82.0 } },
  { id: 'TRANSFER_RATE', name: 'Cross-Functional Transfer Rate', type: 'LEARNING', state: { value: 74.0, baseline: 74.0 } },
  { id: 'DECISION_QUALITY', name: 'Executive Decision Quality (CDQI)', type: 'DECISION', state: { value: 78.5, baseline: 78.5 } },
  { id: 'GOVERNANCE_ADHERENCE', name: 'Charter Governance Adherence', type: 'COMMITTEE', state: { value: 92.0, baseline: 92.0 } },
  { id: 'COACHING_FREQUENCY', name: 'Executive Coaching Sessions', type: 'RECOMMENDATION', state: { value: 24.0, baseline: 24.0 } },
  { id: 'DISSENT_INTEGRATION', name: 'Dissent Integration Ratio (DIR)', type: 'DECISION', state: { value: 0.65, baseline: 0.65 } },
  { id: 'RISK_SCORE', name: 'Aggregate Risk Score', type: 'RISK', state: { value: 28.5, baseline: 28.5 } },
  { id: 'RESILIENCE_INVESTMENT', name: 'Resilience Engineering Budget', type: 'BUDGET', state: { value: 0.5, baseline: 0.5, unitScale: 1000000 } },
  { id: 'RESILIENCE_RTO', name: 'Recovery Time Objective (RTO Minutes)', type: 'RISK', state: { value: 15.0, baseline: 15.0 } },
  { id: 'OHI', name: 'Organizational Health Index (OHI)', type: 'DECISION', state: { value: 84.2, baseline: 84.2 } },
];

export function getCanonicalDependencies(): DependencyDefinition[] {
  return [...CANONICAL_DEPENDENCY_DEFINITIONS];
}

export function buildDependencyGraph(): DependencyGraph {
  const edges: DependencyEdge[] = CANONICAL_DEPENDENCY_DEFINITIONS.map(def => ({
    sourceId: def.sourceMetric,
    targetId: def.targetMetric,
    dependencyWeight: def.impactWeight,
    relationship: def.relationship || 'INFLUENCES',
    confidencePct: def.confidencePct || 90,
  }));

  return {
    nodes: CANONICAL_NODES_CATALOG.map(n => ({ ...n, state: { ...n.state } })),
    edges,
    lastCalibratedUtc: '2026-09-09T08:00:00Z',
  };
}

/**
 * Validates that the graph is a Directed Acyclic Graph (DAG) using DFS.
 * Detects any causality loops (A -> B -> C -> A).
 */
export function validateAcyclicGraph(edges: DependencyEdge[]): { isDag: boolean; cycle?: string[] } {
  const adj = new Map<string, string[]>();
  for (const edge of edges) {
    if (!adj.has(edge.sourceId)) adj.set(edge.sourceId, []);
    adj.get(edge.sourceId)!.push(edge.targetId);
  }

  const visited = new Set<string>();
  const visiting = new Set<string>();
  const path: string[] = [];

  function dfs(node: string): boolean {
    visiting.add(node);
    path.push(node);

    const neighbors = adj.get(node) || [];
    for (const neighbor of neighbors) {
      if (visiting.has(neighbor)) {
        path.push(neighbor);
        return false; // Cycle detected
      }
      if (!visited.has(neighbor)) {
        if (!dfs(neighbor)) return false;
      }
    }

    visiting.delete(node);
    visited.add(node);
    path.pop();
    return true;
  }

  for (const edge of edges) {
    if (!visited.has(edge.sourceId)) {
      if (!dfs(edge.sourceId)) {
        return { isDag: false, cycle: path };
      }
    }
  }

  return { isDag: true };
}

/**
 * Finds all direct and indirect causal paths from source to target metric.
 */
export function findImpactPaths(sourceMetric: string, targetMetric: string): ImpactPath[] {
  const edges = CANONICAL_DEPENDENCY_DEFINITIONS;
  const adj = new Map<string, Array<{ target: string; weight: number }>>();

  for (const edge of edges) {
    if (!adj.has(edge.sourceMetric)) adj.set(edge.sourceMetric, []);
    adj.get(edge.sourceMetric)!.push({ target: edge.targetMetric, weight: edge.impactWeight });
  }

  const results: ImpactPath[] = [];

  function dfs(current: string, currentPath: string[], steps: Array<{ source: string; target: string; weight: number }>, cumulativeWeight: number) {
    if (current === targetMetric) {
      results.push({
        path: [...currentPath],
        totalWeight: cumulativeWeight,
        steps: [...steps],
      });
      return;
    }

    const nextEdges = adj.get(current) || [];
    for (const next of nextEdges) {
      if (!currentPath.includes(next.target)) {
        currentPath.push(next.target);
        steps.push({ source: current, target: next.target, weight: next.weight });
        dfs(next.target, currentPath, steps, cumulativeWeight * next.weight);
        steps.pop();
        currentPath.pop();
      }
    }
  }

  dfs(sourceMetric, [sourceMetric], [], 1.0);
  return results;
}

/**
 * Returns immediate upstream dependencies (parents) of a given metric.
 */
export function getUpstreamDependencies(metricId: string): string[] {
  return CANONICAL_DEPENDENCY_DEFINITIONS
    .filter(d => d.targetMetric === metricId)
    .map(d => d.sourceMetric);
}

/**
 * Returns immediate downstream dependencies (children) of a given metric.
 */
export function getDownstreamDependencies(metricId: string): string[] {
  return CANONICAL_DEPENDENCY_DEFINITIONS
    .filter(d => d.sourceMetric === metricId)
    .map(d => d.targetMetric);
}

/**
 * Computes the transitive closure of all upstream dependencies.
 */
export function getAllUpstreamDependencies(metricId: string): string[] {
  const result = new Set<string>();
  const queue = [metricId];

  while (queue.length > 0) {
    const current = queue.shift()!;
    const parents = getUpstreamDependencies(current);
    for (const parent of parents) {
      if (!result.has(parent)) {
        result.add(parent);
        queue.push(parent);
      }
    }
  }

  return Array.from(result);
}

/**
 * Propagates an input percentage delta through causal paths to estimate final target impact.
 */
export function calculateCumulativeImpact(
  sourceMetric: string,
  targetMetric: string,
  initialDelta: number
): number {
  const paths = findImpactPaths(sourceMetric, targetMetric);
  if (paths.length === 0) return 0;

  // Aggregate weighted paths
  let netMultiplier = 0;
  for (const p of paths) {
    netMultiplier += p.totalWeight;
  }

  return initialDelta * netMultiplier;
}
