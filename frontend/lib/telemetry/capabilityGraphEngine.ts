/**
 * Phase 30: Capability Graph & Dependency Engine (CI-100 & CI-200)
 *
 * Implements:
 * - Capability Dependency Graph
 * - Capability Impact Index (CII)
 * - Capability Impact Efficiency (CIE)
 * - INV-CI1: Value Attribution Integrity (Sum(Attributed) <= Actual Realized)
 * - Capability Retirement & Sunset Detection
 */

import type {
  CapabilityNode,
  CapabilityDependencyEdge,
  CapabilityDependencyGraph,
  ValueAttributionIntegrityResult,
  CapabilityImpactMetrics,
} from '@/types/capability-intelligence';

export const CANONICAL_CAPABILITY_NODES: CapabilityNode[] = [
  {
    id: 'institutional-flow-filter',
    name: 'Institutional Flow Filter',
    category: 'ANALYTICS',
    status: 'CORE',
    metrics: {
      behaviorImpactScore: 94.0,
      outcomeImpactScore: 92.0,
      valueImpactScore: 96.0,
      adoptionImpactScore: 82.0,
      cii: 92.4,
    },
    economics: {
      valueGeneratedDollars: 1100000,
      operationalCostDollars: 250000,
      cie: 4.4, // $1.1M / $250K = 4.40x
      capitalPreservedDollars: 1100000,
      excessReturnContributionPct: 1.4,
      confidencePct: 96.0,
      sampleSize: 1847,
    },
    dependencies: [],
    recommendedAction: 'INVEST_MORE',
    actionRationale: 'Primary value engine with 4.40x CIE. Deepen institutional flow footprint.',
  },
  {
    id: 'ai-mentor-engine',
    name: 'AI Mentor Engine',
    category: 'COACHING',
    status: 'CORE',
    metrics: {
      behaviorImpactScore: 91.0,
      outcomeImpactScore: 88.0,
      valueImpactScore: 89.0,
      adoptionImpactScore: 74.0,
      cii: 87.2,
    },
    economics: {
      valueGeneratedDollars: 850000,
      operationalCostDollars: 200000,
      cie: 4.25, // $850K / $200K = 4.25x
      capitalPreservedDollars: 850000,
      excessReturnContributionPct: 1.1,
      confidencePct: 95.0,
      sampleSize: 1620,
    },
    dependencies: ['playbook-engine'],
    recommendedAction: 'INVEST_MORE',
    actionRationale: 'Exceptional 4.25x CIE. Expand to personalized behavioral cohort coaching.',
  },
  {
    id: 'playbook-engine',
    name: 'Playbook Engine',
    category: 'EXECUTION',
    status: 'CORE',
    metrics: {
      behaviorImpactScore: 85.0,
      outcomeImpactScore: 83.0,
      valueImpactScore: 81.0,
      adoptionImpactScore: 69.0,
      cii: 81.2,
    },
    economics: {
      valueGeneratedDollars: 450000,
      operationalCostDollars: 120000,
      cie: 3.75, // $450K / $120K = 3.75x
      capitalPreservedDollars: 450000,
      excessReturnContributionPct: 0.6,
      confidencePct: 92.0,
      sampleSize: 1512,
    },
    dependencies: [],
    recommendedAction: 'MAINTAIN',
    actionRationale: 'Solid 3.75x CIE core capability. Continue standard maintenance.',
  },
  {
    id: 'committee-governance',
    name: 'Committee Governance Gate',
    category: 'GOVERNANCE',
    status: 'PROTECTED',
    metrics: {
      behaviorImpactScore: 82.0,
      outcomeImpactScore: 78.0,
      valueImpactScore: 75.0,
      adoptionImpactScore: 91.0,
      cii: 80.8,
    },
    economics: {
      valueGeneratedDollars: 290000,
      operationalCostDollars: 100000,
      cie: 2.9, // $290K / $100K = 2.90x
      capitalPreservedDollars: 290000,
      excessReturnContributionPct: 0.4,
      confidencePct: 91.0,
      sampleSize: 1994,
    },
    dependencies: [],
    recommendedAction: 'MAINTAIN',
    actionRationale: 'Mandatory risk gate. Protected under INV-OI11.',
  },
  {
    id: 'decision-simulator',
    name: 'What-If Decision Simulator',
    category: 'SIMULATION',
    status: 'PILOT',
    metrics: {
      behaviorImpactScore: 76.0,
      outcomeImpactScore: 72.0,
      valueImpactScore: 70.0,
      adoptionImpactScore: 58.0,
      cii: 71.0,
    },
    economics: {
      valueGeneratedDollars: 210000,
      operationalCostDollars: 80000,
      cie: 2.63, // $210K / $80K = 2.63x
      capitalPreservedDollars: 210000,
      excessReturnContributionPct: 0.3,
      confidencePct: 88.0,
      sampleSize: 840,
    },
    dependencies: ['committee-governance'],
    recommendedAction: 'OPTIMIZE',
    actionRationale: 'High user interest. Optimize UI friction to drive adoption.',
  },
  {
    id: 'decision-journal',
    name: 'Decision Journal',
    category: 'EXECUTION',
    status: 'RETIREMENT_REVIEW',
    metrics: {
      behaviorImpactScore: 58.0,
      outcomeImpactScore: 54.0,
      valueImpactScore: 50.0,
      adoptionImpactScore: 48.0,
      cii: 53.7,
    },
    economics: {
      valueGeneratedDollars: 140000,
      operationalCostDollars: 90000,
      cie: 1.56, // $140K / $90K = 1.56x
      capitalPreservedDollars: 140000,
      excessReturnContributionPct: 0.2,
      confidencePct: 83.0,
      sampleSize: 1053,
    },
    dependencies: [],
    recommendedAction: 'REDESIGN',
    actionRationale: 'Low 1.56x CIE and sub-50% adoption. Schedule for overhaul or sunset.',
  },
];

export const CANONICAL_DEPENDENCY_EDGES: CapabilityDependencyEdge[] = [
  {
    sourceCapabilityId: 'playbook-engine',
    targetCapabilityId: 'ai-mentor-engine',
    dependencyType: 'PREREQUISITE',
    criticality: 'CRITICAL',
  },
  {
    sourceCapabilityId: 'committee-governance',
    targetCapabilityId: 'decision-simulator',
    dependencyType: 'ENHANCER',
    criticality: 'CRITICAL',
  },
  {
    sourceCapabilityId: 'institutional-flow-filter',
    targetCapabilityId: 'playbook-engine',
    dependencyType: 'FEEDBACK_LOOP',
    criticality: 'OPTIONAL',
  },
];

export function computeCapabilityCII(metrics: Omit<CapabilityImpactMetrics, 'cii'>): number {
  const raw =
    0.35 * metrics.behaviorImpactScore +
    0.30 * metrics.outcomeImpactScore +
    0.20 * metrics.valueImpactScore +
    0.15 * metrics.adoptionImpactScore;
  return Math.round(raw * 10) / 10;
}

export function computeCapabilityCIE(valueDollars: number, costDollars: number): number {
  if (costDollars <= 0) return 0;
  return Math.round((valueDollars / costDollars) * 100) / 100;
}

export function buildCapabilityDependencyGraph(): CapabilityDependencyGraph {
  const nodes = CANONICAL_CAPABILITY_NODES;
  const edges = CANONICAL_DEPENDENCY_EDGES;

  const sortedByCii = [...nodes].sort((a, b) => b.metrics.cii - a.metrics.cii);
  const highValueThreshold = Math.ceil(nodes.length * 0.2); // Top 20%
  const highValueCount = Math.max(1, highValueThreshold);

  const underperformingCount = nodes.filter(
    n => n.economics.cie < 2.0 || n.status === 'RETIREMENT_REVIEW'
  ).length;

  const retirementCandidatesCount = nodes.filter(
    n => n.status === 'RETIREMENT_REVIEW' || n.status === 'SUNSET'
  ).length;

  return {
    nodes,
    edges,
    totalCapabilities: nodes.length,
    highValueCount,
    underperformingCount,
    retirementCandidatesCount,
  };
}

/**
 * INV-CI1: Value Attribution Integrity
 * Rule: Sum of Attributed Value across all capabilities <= Actual Realized Value
 */
export function verifyValueAttributionIntegrity(
  capabilities: CapabilityNode[] = CANONICAL_CAPABILITY_NODES,
  actualRealizedValueDollars = 3040000 // Total realized value across institution
): ValueAttributionIntegrityResult {
  const totalAttributed = capabilities.reduce((sum, c) => sum + c.economics.valueGeneratedDollars, 0);
  const discrepancyDollars = totalAttributed - actualRealizedValueDollars;
  const inflationRatio = actualRealizedValueDollars > 0 ? totalAttributed / actualRealizedValueDollars : 1.0;
  const isSatisfied = totalAttributed <= actualRealizedValueDollars;

  return {
    isSatisfied,
    totalAttributedValueDollars: totalAttributed,
    actualRealizedValueDollars,
    discrepancyDollars: isSatisfied ? 0 : discrepancyDollars,
    inflationRatio: Math.round(inflationRatio * 1000) / 1000,
    details: isSatisfied
      ? `INV-CI1 SATISFIED: Total attributed value ($${totalAttributed.toLocaleString()}) <= Realized value ($${actualRealizedValueDollars.toLocaleString()}). Inflation ratio: ${inflationRatio.toFixed(3)}.`
      : `INV-CI1 BREACH: Total attributed value ($${totalAttributed.toLocaleString()}) exceeds actual realized value ($${actualRealizedValueDollars.toLocaleString()}) by $${discrepancyDollars.toLocaleString()}.`,
  };
}
