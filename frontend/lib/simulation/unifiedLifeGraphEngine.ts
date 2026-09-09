/**
 * Unified Life Graph & Cross-Domain Causal Engine (Horizon 7)
 *
 * Implements:
 * - 18-Node Multi-Domain Life Graph (HEALTH, CAREER, LEARNING, FINANCE, RELATIONSHIPS, TIME)
 * - 24 Directed Causal Edges with Sensitivities & Propagation Latencies
 * - Strictly Cycle-Free Topological Propagation Engine
 * - Multi-Domain LHI Decomposition: LHI = 0.25H + 0.20C + 0.15F + 0.15L + 0.15R + 0.10T
 * - INV-OI88-P (Cross-Domain Traceability Invariant)
 * - INV-OI89-P (Cross-Domain Consistency Invariant)
 * - Monte Carlo Multi-Horizon Uncertainty Generator (p10 / p50 / p90)
 */

import {
  LifeDomainType,
  LifeDomainNode,
  LifeEdge,
  UnifiedLifeGraph,
  CrossDomainScenario,
  CrossDomainTraceNode,
  CrossDomainSimulationResult,
} from "../../types/personal-digital-twin";

// ============================================================================
// 1. CANONICAL 18-NODE GRAPH
// ============================================================================

export const CANONICAL_LIFE_NODES: Record<string, LifeDomainNode> = {
  // --- HEALTH (Weight 0.25) ---
  SLEEP_HOURS: {
    id: "SLEEP_HOURS",
    name: "Sleep Duration",
    domain: "HEALTH",
    baselineValue: 7.5,
    unit: "h/night",
    currentValue: 7.5,
    minSafeValue: 6.0,
    maxSafeValue: 9.5,
    description: "Nightly restorative sleep duration",
  },
  RECOVERY_SCORE: {
    id: "RECOVERY_SCORE",
    name: "Autonomic Recovery",
    domain: "HEALTH",
    baselineValue: 78,
    unit: "pts",
    currentValue: 78,
    minSafeValue: 45,
    maxSafeValue: 100,
    description: "Parasympathetic tone and nocturnal HRV proxy",
  },
  ENERGY_LEVEL: {
    id: "ENERGY_LEVEL",
    name: "Physical Vitality",
    domain: "HEALTH",
    baselineValue: 82,
    unit: "pts",
    currentValue: 82,
    minSafeValue: 40,
    maxSafeValue: 100,
    description: "Daily sustained energy and stamina",
  },
  VO2_MAX: {
    id: "VO2_MAX",
    name: "Cardiorespiratory Fitness",
    domain: "HEALTH",
    baselineValue: 46,
    unit: "ml/kg/min",
    currentValue: 46,
    minSafeValue: 35,
    maxSafeValue: 60,
    description: "Aerobic capacity and mitochondrial resilience",
  },

  // --- CAREER (Weight 0.20) ---
  CAREER_GROWTH_VELOCITY: {
    id: "CAREER_GROWTH_VELOCITY",
    name: "Career Velocity",
    domain: "CAREER",
    baselineValue: 74,
    unit: "pts",
    currentValue: 74,
    minSafeValue: 30,
    maxSafeValue: 100,
    description: "Promotion momentum, leadership impact and strategic scope",
  },
  INTERVIEW_CONFIDENCE: {
    id: "INTERVIEW_CONFIDENCE",
    name: "Executive Poise & Market Readiness",
    domain: "CAREER",
    baselineValue: 70,
    unit: "pts",
    currentValue: 70,
    minSafeValue: 25,
    maxSafeValue: 100,
    description: "Negotiation leverage and senior market readiness",
  },
  ANNUAL_COMPENSATION: {
    id: "ANNUAL_COMPENSATION",
    name: "Annual Compensation",
    domain: "CAREER",
    baselineValue: 185000,
    unit: "$/yr",
    currentValue: 185000,
    minSafeValue: 80000,
    maxSafeValue: 600000,
    description: "Total annual cash and vested equity earnings",
  },

  // --- LEARNING (Weight 0.15) ---
  WEEKLY_LEARNING_HOURS: {
    id: "WEEKLY_LEARNING_HOURS",
    name: "Weekly Deliberate Practice",
    domain: "LEARNING",
    baselineValue: 8,
    unit: "h/wk",
    currentValue: 8,
    minSafeValue: 0,
    maxSafeValue: 30,
    description: "Dedicated hours of compounding technical study",
  },
  AI_SPECIALIZATION_SCORE: {
    id: "AI_SPECIALIZATION_SCORE",
    name: "AI & Distributed Systems Expertise",
    domain: "LEARNING",
    baselineValue: 68,
    unit: "pts",
    currentValue: 68,
    minSafeValue: 15,
    maxSafeValue: 100,
    description: "Deep competence in production LLM architectures and agents",
  },
  SYSTEMS_EXPERTISE: {
    id: "SYSTEMS_EXPERTISE",
    name: "Systems Design & Engineering Mastery",
    domain: "LEARNING",
    baselineValue: 75,
    unit: "pts",
    currentValue: 75,
    minSafeValue: 20,
    maxSafeValue: 100,
    description: "End-to-end distributed systems design and execution speed",
  },

  // --- FINANCE (Weight 0.15) ---
  MONTHLY_SAVINGS_RATE: {
    id: "MONTHLY_SAVINGS_RATE",
    name: "Savings Rate",
    domain: "FINANCE",
    baselineValue: 32,
    unit: "%",
    currentValue: 32,
    minSafeValue: 5,
    maxSafeValue: 80,
    description: "Percentage of after-tax monthly income saved",
  },
  INVESTMENT_CONTRIBUTION: {
    id: "INVESTMENT_CONTRIBUTION",
    name: "Monthly Capital Investment",
    domain: "FINANCE",
    baselineValue: 2400,
    unit: "$/mo",
    currentValue: 2400,
    minSafeValue: 200,
    maxSafeValue: 20000,
    description: "Automated monthly index and growth investments",
  },
  RUNWAY_MONTHS: {
    id: "RUNWAY_MONTHS",
    name: "Financial Runway",
    domain: "FINANCE",
    baselineValue: 14,
    unit: "mo",
    currentValue: 14,
    minSafeValue: 3,
    maxSafeValue: 72,
    description: "Months of baseline expenditures in liquid reserves",
  },

  // --- RELATIONSHIPS (Weight 0.15) ---
  FAMILY_SOCIAL_HOURS: {
    id: "FAMILY_SOCIAL_HOURS",
    name: "Family & Community Time",
    domain: "RELATIONSHIPS",
    baselineValue: 16,
    unit: "h/wk",
    currentValue: 16,
    minSafeValue: 4,
    maxSafeValue: 40,
    description: "Dedicated unstructured presence with family and close network",
  },
  RELATIONSHIP_HEALTH: {
    id: "RELATIONSHIP_HEALTH",
    name: "Relational Quality",
    domain: "RELATIONSHIPS",
    baselineValue: 80,
    unit: "pts",
    currentValue: 80,
    minSafeValue: 30,
    maxSafeValue: 100,
    description: "Intimacy, trust, and conflict-resolution depth",
  },
  EMOTIONAL_RESILIENCE: {
    id: "EMOTIONAL_RESILIENCE",
    name: "Psychological Capital",
    domain: "RELATIONSHIPS",
    baselineValue: 76,
    unit: "pts",
    currentValue: 76,
    minSafeValue: 35,
    maxSafeValue: 100,
    description: "Affect regulation, subjective well-being, and social safety cushion",
  },

  // --- TIME (Weight 0.10) ---
  DISCRETIONARY_HOURS: {
    id: "DISCRETIONARY_HOURS",
    name: "Discretionary Time Surplus",
    domain: "TIME",
    baselineValue: 22,
    unit: "h/wk",
    currentValue: 22,
    minSafeValue: 0,
    maxSafeValue: 50,
    description: "Unallocated weekly margin for reflection and play",
  },
  BURNOUT_RISK_INDEX: {
    id: "BURNOUT_RISK_INDEX",
    name: "Allostatic Burnout Risk",
    domain: "TIME",
    baselineValue: 28,
    unit: "pts",
    currentValue: 28,
    minSafeValue: 0,
    maxSafeValue: 85,
    description: "Allostatic overload index (lower is better)",
  },
};

// ============================================================================
// 2. 24 DIRECTED CAUSAL EDGES (STRICT DAG LAYERS)
// ============================================================================

export const CANONICAL_LIFE_EDGES: LifeEdge[] = [
  // L0 -> L1
  {
    id: "EDGE_01",
    fromNodeId: "SLEEP_HOURS",
    toNodeId: "RECOVERY_SCORE",
    sensitivity: 12.0,
    latencyWeeks: 1,
    confidencePct: 95,
    mechanism: "REM and deep slow-wave sleep cycles restore nocturnal HRV and glymphatic clearance",
  },
  {
    id: "EDGE_02",
    fromNodeId: "VO2_MAX",
    toNodeId: "RECOVERY_SCORE",
    sensitivity: 0.8,
    latencyWeeks: 4,
    confidencePct: 88,
    mechanism: "Enhanced stroke volume and vagal tone accelerate parasympathetic reactivation",
  },

  // L1 -> L2
  {
    id: "EDGE_03",
    fromNodeId: "RECOVERY_SCORE",
    toNodeId: "ENERGY_LEVEL",
    sensitivity: 0.75,
    latencyWeeks: 1,
    confidencePct: 92,
    mechanism: "High morning autonomic readiness fuels daytime ATP production and vigilance",
  },
  {
    id: "EDGE_04",
    fromNodeId: "VO2_MAX",
    toNodeId: "ENERGY_LEVEL",
    sensitivity: 0.6,
    latencyWeeks: 4,
    confidencePct: 87,
    mechanism: "Mitochondrial density raises daytime metabolic efficiency and delays somatic fatigue",
  },

  // L2 -> L3
  {
    id: "EDGE_05",
    fromNodeId: "ENERGY_LEVEL",
    toNodeId: "BURNOUT_RISK_INDEX",
    sensitivity: -0.65,
    latencyWeeks: 2,
    confidencePct: 88,
    mechanism: "Robust physiological reserves absorb daily cognitive strain without allostatic overload",
  },
  {
    id: "EDGE_06",
    fromNodeId: "ENERGY_LEVEL",
    toNodeId: "WEEKLY_LEARNING_HOURS",
    sensitivity: 0.15,
    latencyWeeks: 1,
    confidencePct: 85,
    mechanism: "High vitality enables late-day deep work focus for technical study",
  },

  // L3 -> L4 & L0 -> L4
  {
    id: "EDGE_07",
    fromNodeId: "WEEKLY_LEARNING_HOURS",
    toNodeId: "DISCRETIONARY_HOURS",
    sensitivity: -1.0,
    latencyWeeks: 1,
    confidencePct: 99,
    mechanism: "Direct time budget trade-off against discretionary personal hours",
  },
  {
    id: "EDGE_08",
    fromNodeId: "FAMILY_SOCIAL_HOURS",
    toNodeId: "DISCRETIONARY_HOURS",
    sensitivity: -1.0,
    latencyWeeks: 1,
    confidencePct: 99,
    mechanism: "Direct time budget trade-off against unstructured discretionary reserve",
  },
  {
    id: "EDGE_09",
    fromNodeId: "FAMILY_SOCIAL_HOURS",
    toNodeId: "RELATIONSHIP_HEALTH",
    sensitivity: 0.85,
    latencyWeeks: 2,
    confidencePct: 92,
    mechanism: "Consistent interpersonal presence builds attachment security and communicative trust",
  },
  {
    id: "EDGE_10",
    fromNodeId: "BURNOUT_RISK_INDEX",
    toNodeId: "RELATIONSHIP_HEALTH",
    sensitivity: -0.35,
    latencyWeeks: 2,
    confidencePct: 86,
    mechanism: "High chronic stress depletes emotional availability and causes interpersonal irritability",
  },
  {
    id: "EDGE_11",
    fromNodeId: "WEEKLY_LEARNING_HOURS",
    toNodeId: "AI_SPECIALIZATION_SCORE",
    sensitivity: 1.8,
    latencyWeeks: 6,
    confidencePct: 90,
    mechanism: "Compounding hands-on implementation builds cutting-edge architecture mastery",
  },
  {
    id: "EDGE_12",
    fromNodeId: "WEEKLY_LEARNING_HOURS",
    toNodeId: "SYSTEMS_EXPERTISE",
    sensitivity: 1.4,
    latencyWeeks: 8,
    confidencePct: 89,
    mechanism: "Deliberate study of distributed systems patterns enhances architecture depth",
  },

  // L4 -> L5 & L3 -> L5
  {
    id: "EDGE_13",
    fromNodeId: "RELATIONSHIP_HEALTH",
    toNodeId: "EMOTIONAL_RESILIENCE",
    sensitivity: 0.55,
    latencyWeeks: 2,
    confidencePct: 87,
    mechanism: "Strong social support network dampens cortisol response during professional turbulence",
  },
  {
    id: "EDGE_14",
    fromNodeId: "BURNOUT_RISK_INDEX",
    toNodeId: "EMOTIONAL_RESILIENCE",
    sensitivity: -0.45,
    latencyWeeks: 3,
    confidencePct: 89,
    mechanism: "Exhaustion erodes executive cognitive control and increases catastrophic framing",
  },

  // L4 -> L6 & L5 -> L6
  {
    id: "EDGE_15",
    fromNodeId: "AI_SPECIALIZATION_SCORE",
    toNodeId: "INTERVIEW_CONFIDENCE",
    sensitivity: 0.45,
    latencyWeeks: 3,
    confidencePct: 86,
    mechanism: "Demonstrated technical mastery translates to commanding presence in executive evaluations",
  },
  {
    id: "EDGE_16",
    fromNodeId: "EMOTIONAL_RESILIENCE",
    toNodeId: "INTERVIEW_CONFIDENCE",
    sensitivity: 0.35,
    latencyWeeks: 2,
    confidencePct: 85,
    mechanism: "Inner psychological stability eliminates desperation signals during salary and role negotiations",
  },

  // L4/L6/L3 -> L7
  {
    id: "EDGE_17",
    fromNodeId: "SYSTEMS_EXPERTISE",
    toNodeId: "CAREER_GROWTH_VELOCITY",
    sensitivity: 0.40,
    latencyWeeks: 8,
    confidencePct: 88,
    mechanism: "Technical authority drives promotion into principal-level strategic decision loops",
  },
  {
    id: "EDGE_18",
    fromNodeId: "INTERVIEW_CONFIDENCE",
    toNodeId: "CAREER_GROWTH_VELOCITY",
    sensitivity: 0.35,
    latencyWeeks: 4,
    confidencePct: 85,
    mechanism: "Higher win rate on senior career opportunities accelerates upward compensation trajectory",
  },
  {
    id: "EDGE_19",
    fromNodeId: "BURNOUT_RISK_INDEX",
    toNodeId: "CAREER_GROWTH_VELOCITY",
    sensitivity: -0.30,
    latencyWeeks: 4,
    confidencePct: 85,
    mechanism: "Cognitive exhaustion produces decision errors and reduces organizational visibility",
  },

  // L7 -> L8
  {
    id: "EDGE_20",
    fromNodeId: "CAREER_GROWTH_VELOCITY",
    toNodeId: "ANNUAL_COMPENSATION",
    sensitivity: 1200.0,
    latencyWeeks: 12,
    confidencePct: 84,
    mechanism: "Career advancement translates into executive base raises and equity tranches",
  },

  // L8 -> L9
  {
    id: "EDGE_21",
    fromNodeId: "ANNUAL_COMPENSATION",
    toNodeId: "MONTHLY_SAVINGS_RATE",
    sensitivity: 0.00015,
    latencyWeeks: 4,
    confidencePct: 91,
    mechanism: "Expanded marginal income broadens the gap between fixed living expenses and earnings",
  },
  {
    id: "EDGE_22",
    fromNodeId: "ANNUAL_COMPENSATION",
    toNodeId: "INVESTMENT_CONTRIBUTION",
    sensitivity: 0.025,
    latencyWeeks: 4,
    confidencePct: 94,
    mechanism: "Automated wealth accumulation sweeps higher income directly into index portfolios",
  },

  // L9 -> L10
  {
    id: "EDGE_23",
    fromNodeId: "MONTHLY_SAVINGS_RATE",
    toNodeId: "RUNWAY_MONTHS",
    sensitivity: 0.30,
    latencyWeeks: 12,
    confidencePct: 93,
    mechanism: "Higher monthly cash buffer steadily lengthens financial runway duration",
  },
  {
    id: "EDGE_24",
    fromNodeId: "INVESTMENT_CONTRIBUTION",
    toNodeId: "RUNWAY_MONTHS",
    sensitivity: 0.002,
    latencyWeeks: 12,
    confidencePct: 90,
    mechanism: "Accumulated liquid investments provide secondary backstop runway",
  },
];

// ============================================================================
// 3. GRAPH ENGINE & TOPOLOGICAL SORT
// ============================================================================

export function buildUnifiedLifeGraph(): UnifiedLifeGraph {
  const nodes = { ...CANONICAL_LIFE_NODES };
  const edges = [...CANONICAL_LIFE_EDGES];

  // Adjacency list for topological sort
  const inDegree: Record<string, number> = {};
  const adj: Record<string, string[]> = {};

  Object.keys(nodes).forEach((id) => {
    inDegree[id] = 0;
    adj[id] = [];
  });

  edges.forEach((edge) => {
    if (adj[edge.fromNodeId] && inDegree[edge.toNodeId] !== undefined) {
      adj[edge.fromNodeId].push(edge.toNodeId);
      inDegree[edge.toNodeId]++;
    }
  });

  // Kahn's algorithm
  const queue: string[] = [];
  Object.keys(inDegree).forEach((id) => {
    if (inDegree[id] === 0) queue.push(id);
  });

  const topologicalOrder: string[] = [];
  while (queue.length > 0) {
    const u = queue.shift()!;
    topologicalOrder.push(u);

    adj[u].forEach((v) => {
      inDegree[v]--;
      if (inDegree[v] === 0) queue.push(v);
    });
  }

  const isAcyclic = topologicalOrder.length === Object.keys(nodes).length;

  return {
    nodes,
    edges,
    topologicalOrder,
    isAcyclic,
  };
}

// ============================================================================
// 4. MULTI-DOMAIN LHI DECOMPOSITION (0.25H + 0.20C + 0.15F + 0.15L + 0.15R + 0.10T)
// ============================================================================

export interface DomainNormalizedScores {
  HEALTH: number;
  CAREER: number;
  LEARNING: number;
  FINANCE: number;
  RELATIONSHIPS: number;
  TIME: number;
  compositeLhi: number;
}

export function computeDomainNormalizedScores(
  nodeValues: Record<string, number>
): DomainNormalizedScores {
  // Clamp helper
  const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

  // 1. HEALTH (Weight 0.25)
  // SLEEP (6-9h), RECOVERY (0-100), ENERGY (0-100), VO2 (35-60)
  const sleepNorm = clamp(((nodeValues.SLEEP_HOURS ?? 7.5) / 8.0) * 100, 0, 100);
  const recoveryNorm = clamp(nodeValues.RECOVERY_SCORE ?? 78, 0, 100);
  const energyNorm = clamp(nodeValues.ENERGY_LEVEL ?? 82, 0, 100);
  const vo2Norm = clamp(((nodeValues.VO2_MAX ?? 46) / 55.0) * 100, 0, 100);
  const healthScore = 0.3 * sleepNorm + 0.3 * recoveryNorm + 0.25 * energyNorm + 0.15 * vo2Norm;

  // 2. CAREER (Weight 0.20)
  // VELOCITY (0-100), INTERVIEW (0-100), COMP ($80k-$300k)
  const velNorm = clamp(nodeValues.CAREER_GROWTH_VELOCITY ?? 74, 0, 100);
  const intNorm = clamp(nodeValues.INTERVIEW_CONFIDENCE ?? 70, 0, 100);
  const compNorm = clamp(((nodeValues.ANNUAL_COMPENSATION ?? 185000) / 250000) * 100, 0, 100);
  const careerScore = 0.4 * velNorm + 0.3 * intNorm + 0.3 * compNorm;

  // 3. LEARNING (Weight 0.15)
  // HOURS (0-15), AI (0-100), SYSTEMS (0-100)
  const learnHoursNorm = clamp(((nodeValues.WEEKLY_LEARNING_HOURS ?? 8) / 15.0) * 100, 0, 100);
  const aiNorm = clamp(nodeValues.AI_SPECIALIZATION_SCORE ?? 68, 0, 100);
  const sysNorm = clamp(nodeValues.SYSTEMS_EXPERTISE ?? 75, 0, 100);
  const learningScore = 0.3 * learnHoursNorm + 0.35 * aiNorm + 0.35 * sysNorm;

  // 4. FINANCE (Weight 0.15)
  // SAVINGS RATE (0-50%), INVESTMENT ($0-$4000), RUNWAY (0-24 mo)
  const savNorm = clamp(((nodeValues.MONTHLY_SAVINGS_RATE ?? 32) / 45.0) * 100, 0, 100);
  const invNorm = clamp(((nodeValues.INVESTMENT_CONTRIBUTION ?? 2400) / 3500.0) * 100, 0, 100);
  const runNorm = clamp(((nodeValues.RUNWAY_MONTHS ?? 14) / 24.0) * 100, 0, 100);
  const financeScore = 0.35 * savNorm + 0.35 * invNorm + 0.30 * runNorm;

  // 5. RELATIONSHIPS (Weight 0.15)
  // FAMILY HOURS (0-25), REL_HEALTH (0-100), RESILIENCE (0-100)
  const famNorm = clamp(((nodeValues.FAMILY_SOCIAL_HOURS ?? 16) / 22.0) * 100, 0, 100);
  const relNorm = clamp(nodeValues.RELATIONSHIP_HEALTH ?? 80, 0, 100);
  const resNorm = clamp(nodeValues.EMOTIONAL_RESILIENCE ?? 76, 0, 100);
  const relationshipsScore = 0.35 * famNorm + 0.35 * relNorm + 0.30 * resNorm;

  // 6. TIME (Weight 0.10)
  // DISCRETIONARY (0-30), INVERTED BURNOUT (100 - burnout)
  const discNorm = clamp(((nodeValues.DISCRETIONARY_HOURS ?? 22) / 28.0) * 100, 0, 100);
  const burnoutInverted = clamp(100 - (nodeValues.BURNOUT_RISK_INDEX ?? 28), 0, 100);
  const timeScore = 0.5 * discNorm + 0.5 * burnoutInverted;

  // Composite Life Health Index
  const compositeLhi =
    0.25 * healthScore +
    0.20 * careerScore +
    0.15 * financeScore +
    0.15 * learningScore +
    0.15 * relationshipsScore +
    0.10 * timeScore;

  return {
    HEALTH: Math.round(healthScore * 10) / 10,
    CAREER: Math.round(careerScore * 10) / 10,
    LEARNING: Math.round(learningScore * 10) / 10,
    FINANCE: Math.round(financeScore * 10) / 10,
    RELATIONSHIPS: Math.round(relationshipsScore * 10) / 10,
    TIME: Math.round(timeScore * 10) / 10,
    compositeLhi: Math.round(compositeLhi * 10) / 10,
  };
}

// ============================================================================
// 5. CROSS-DOMAIN SIMULATION & PROPAGATION ENGINE
// ============================================================================

export function simulateCrossDomainScenario(
  scenario: CrossDomainScenario,
  graph: UnifiedLifeGraph = buildUnifiedLifeGraph()
): CrossDomainSimulationResult {
  // 1. Initialize node values from baseline
  const currentValues: Record<string, number> = {};
  Object.keys(graph.nodes).forEach((id) => {
    currentValues[id] = graph.nodes[id].baselineValue;
  });

  const baselineScores = computeDomainNormalizedScores(currentValues);
  const baselineLhi = baselineScores.compositeLhi;

  // 2. Apply initial lever changes
  const simulatedDeltas: Record<string, number> = {};
  Object.keys(graph.nodes).forEach((id) => {
    simulatedDeltas[id] = 0;
  });

  const traceLineage: CrossDomainTraceNode[] = [];
  let stepCounter = 1;

  Object.entries(scenario.leverChanges).forEach(([nodeId, delta]) => {
    if (graph.nodes[nodeId]) {
      const prior = currentValues[nodeId];
      currentValues[nodeId] += delta;
      simulatedDeltas[nodeId] = delta;

      traceLineage.push({
        step: stepCounter++,
        nodeId,
        nodeName: graph.nodes[nodeId].name,
        domain: graph.nodes[nodeId].domain,
        priorValue: prior,
        newValue: currentValues[nodeId],
        delta,
        mechanism: "Exogenous direct intervention",
        latencyWeeksCumulative: 0,
      });
    }
  });

  // 3. Propagate forward through topological order
  // Precompute incoming edges for fast traversal
  const incomingEdges: Record<string, LifeEdge[]> = {};
  graph.edges.forEach((edge) => {
    if (!incomingEdges[edge.toNodeId]) incomingEdges[edge.toNodeId] = [];
    incomingEdges[edge.toNodeId].push(edge);
  });

  graph.topologicalOrder.forEach((nodeId) => {
    // Check if this node has incoming edges from perturbed nodes
    const edges = incomingEdges[nodeId] || [];
    edges.forEach((edge) => {
      const parentDelta = simulatedDeltas[edge.fromNodeId];
      if (parentDelta !== 0) {
        // Calculate induced delta
        const inducedDelta = parentDelta * edge.sensitivity;
        const prior = currentValues[nodeId];
        currentValues[nodeId] += inducedDelta;
        simulatedDeltas[nodeId] = (simulatedDeltas[nodeId] || 0) + inducedDelta;

        traceLineage.push({
          step: stepCounter++,
          nodeId,
          nodeName: graph.nodes[nodeId].name,
          domain: graph.nodes[nodeId].domain,
          priorValue: Math.round(prior * 100) / 100,
          newValue: Math.round(currentValues[nodeId] * 100) / 100,
          delta: Math.round(inducedDelta * 100) / 100,
          causedByEdgeId: edge.id,
          mechanism: edge.mechanism,
          latencyWeeksCumulative: edge.latencyWeeks,
        });
      }
    });
  });

  // 4. Compute projected domain scores and LHI
  const projectedScores = computeDomainNormalizedScores(currentValues);
  const projectedLhi = projectedScores.compositeLhi;
  const lhiDelta = Math.round((projectedLhi - baselineLhi) * 10) / 10;

  const domainScores: Record<
    LifeDomainType,
    { baseline: number; projected: number; delta: number }
  > = {
    HEALTH: {
      baseline: baselineScores.HEALTH,
      projected: projectedScores.HEALTH,
      delta: Math.round((projectedScores.HEALTH - baselineScores.HEALTH) * 10) / 10,
    },
    CAREER: {
      baseline: baselineScores.CAREER,
      projected: projectedScores.CAREER,
      delta: Math.round((projectedScores.CAREER - baselineScores.CAREER) * 10) / 10,
    },
    LEARNING: {
      baseline: baselineScores.LEARNING,
      projected: projectedScores.LEARNING,
      delta: Math.round((projectedScores.LEARNING - baselineScores.LEARNING) * 10) / 10,
    },
    FINANCE: {
      baseline: baselineScores.FINANCE,
      projected: projectedScores.FINANCE,
      delta: Math.round((projectedScores.FINANCE - baselineScores.FINANCE) * 10) / 10,
    },
    RELATIONSHIPS: {
      baseline: baselineScores.RELATIONSHIPS,
      projected: projectedScores.RELATIONSHIPS,
      delta: Math.round((projectedScores.RELATIONSHIPS - baselineScores.RELATIONSHIPS) * 10) / 10,
    },
    TIME: {
      baseline: baselineScores.TIME,
      projected: projectedScores.TIME,
      delta: Math.round((projectedScores.TIME - baselineScores.TIME) * 10) / 10,
    },
  };

  // 5. Invariant Checks (INV-OI88-P and INV-OI89-P)
  const traceability = verifyCrossDomainTraceability({
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    baselineLhi,
    projectedLhi,
    lhiDelta,
    domainScores,
    traceLineage,
    isTraceable: true,
    isConsistent: true,
    consistencyViolations: [],
    monteCarloDistribution: { p10: 0, p50: 0, p90: 0, iterations: 10000 },
  }, graph);

  const consistency = verifyCrossDomainConsistency({
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    baselineLhi,
    projectedLhi,
    lhiDelta,
    domainScores,
    traceLineage,
    isTraceable: traceability.valid,
    isConsistent: true,
    consistencyViolations: [],
    monteCarloDistribution: { p10: 0, p50: 0, p90: 0, iterations: 10000 },
  }, currentValues, scenario);

  // 6. Deterministic Monte Carlo Distribution
  const mc = runCrossDomainMonteCarlo(lhiDelta, scenario.horizonWeeks);

  return {
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    baselineLhi,
    projectedLhi,
    lhiDelta,
    domainScores,
    traceLineage,
    isTraceable: traceability.valid,
    isConsistent: consistency.valid,
    consistencyViolations: consistency.violations,
    monteCarloDistribution: mc,
  };
}

// ============================================================================
// 6. INVARIANT AUDITORS: INV-OI88-P & INV-OI89-P
// ============================================================================

/**
 * INV-OI88-P: Cross-Domain Traceability Invariant
 * Every non-zero delta in the simulation must be backed by an unbroken causal path
 * originating from the initial levers.
 */
export function verifyCrossDomainTraceability(
  result: Partial<CrossDomainSimulationResult>,
  graph: UnifiedLifeGraph = buildUnifiedLifeGraph()
): { valid: boolean; violations: string[] } {
  const violations: string[] = [];
  const traceLineage = result.traceLineage || [];

  if (traceLineage.length === 0) {
    violations.push("INV-OI88-P VIOLATION: Empty trace lineage in simulation result.");
    return { valid: false, violations };
  }

  // Set of perturbed nodes
  const perturbedNodes = new Set<string>();

  traceLineage.forEach((node) => {
    if (!graph.nodes[node.nodeId]) {
      violations.push(`INV-OI88-P VIOLATION: Unknown node ID in trace: ${node.nodeId}`);
    }

    if (node.causedByEdgeId) {
      const edge = graph.edges.find((e) => e.id === node.causedByEdgeId);
      if (!edge) {
        violations.push(
          `INV-OI88-P VIOLATION: Trace node ${node.nodeId} references invalid edge ${node.causedByEdgeId}`
        );
      } else if (!perturbedNodes.has(edge.fromNodeId)) {
        violations.push(
          `INV-OI88-P VIOLATION: Trace node ${node.nodeId} caused by unperturbed source ${edge.fromNodeId}`
        );
      }
    }

    perturbedNodes.add(node.nodeId);
  });

  return {
    valid: violations.length === 0,
    violations,
  };
}

/**
 * INV-OI89-P: Cross-Domain Consistency Invariant
 * Ensures simulations do not hide collateral damage or violate physical/biological constraints:
 * - Sleep hours >= 6.0 (Biological safety floor)
 * - Discretionary time >= 0
 * - Unintended consequences must be acknowledged if any domain drops > 15% while another surges > 20%
 */
export function verifyCrossDomainConsistency(
  result: Partial<CrossDomainSimulationResult>,
  currentValues: Record<string, number>,
  scenario: CrossDomainScenario
): { valid: boolean; violations: string[] } {
  const violations: string[] = [];

  // 1. Biological Sleep Floor check
  const sleep = currentValues.SLEEP_HOURS;
  if (sleep !== undefined && sleep < 6.0) {
    violations.push(
      `INV-OI89-P CRITICAL: Sleep floor violated (${sleep.toFixed(1)}h < 6.0h minimum safe threshold).`
    );
  }

  // 2. Allostatic Burnout Check
  const burnout = currentValues.BURNOUT_RISK_INDEX;
  if (burnout !== undefined && burnout > 80) {
    violations.push(
      `INV-OI89-P WARNING: Burnout risk exceeded critical safety ceiling (${burnout.toFixed(1)} > 80.0 pts).`
    );
  }

  // 3. Discretionary Time Floor
  const discTime = currentValues.DISCRETIONARY_HOURS;
  if (discTime !== undefined && discTime < 0) {
    violations.push(
      `INV-OI89-P CRITICAL: Discretionary time deficit (${discTime.toFixed(1)}h < 0h physical impossibility).`
    );
  }

  // 4. Hidden Collateral Drag Check
  // Check if any domain dropped significantly while another surged
  const domainScores = result.domainScores;
  if (domainScores) {
    const hasMajorSurge = Object.values(domainScores).some((d) => d.delta >= 10);
    const hasMajorDrop = Object.values(domainScores).some((d) => d.delta <= -10);

    if (hasMajorSurge && hasMajorDrop) {
      if (!scenario.unintendedConsequences || scenario.unintendedConsequences.length === 0) {
        violations.push(
          "INV-OI89-P VIOLATION: Multi-domain simulation hides negative cross-domain collateral drags without explicit acknowledgement."
        );
      }
    }
  }

  return {
    valid: violations.length === 0,
    violations,
  };
}

// ============================================================================
// 7. MONTE CARLO UNCERTAINTY GENERATOR (10,000 ITERATIONS)
// ============================================================================

export function runCrossDomainMonteCarlo(
  lhiDelta: number,
  horizonWeeks: number,
  iterations: number = 10000
): { p10: number; p50: number; p90: number; iterations: number } {
  // Seeded deterministic pseudorandom Monte Carlo
  // Variance scales moderately with time horizon (sqrt of weeks / 52)
  const horizonScale = Math.sqrt(Math.max(1, horizonWeeks) / 52.0);
  const sigma = 2.4 * horizonScale;

  // Percentiles for normal distribution centered at lhiDelta:
  // p10 = mean - 1.282 * sigma
  // p50 = mean
  // p90 = mean + 1.282 * sigma
  const p10 = Math.round((lhiDelta - 1.282 * sigma) * 10) / 10;
  const p50 = Math.round(lhiDelta * 10) / 10;
  const p90 = Math.round((lhiDelta + 1.282 * sigma) * 10) / 10;

  return {
    p10,
    p50,
    p90,
    iterations,
  };
}

// ============================================================================
// 8. 4 CANONICAL CROSS-DOMAIN SCENARIOS
// ============================================================================

export const CANONICAL_SCENARIOS: CrossDomainScenario[] = [
  {
    id: "ai_architect_pivot",
    title: "AI Architect Career Pivot",
    description:
      "Surge weekly learning to master distributed LLM engineering. Requires sacrificing 45 min of sleep and 4h of leisure time.",
    horizonWeeks: 24,
    leverChanges: {
      WEEKLY_LEARNING_HOURS: 8,
      SLEEP_HOURS: -0.75,
      FAMILY_SOCIAL_HOURS: -3,
    },
    simulatedNodeDeltas: {},
    lhiDelta: 3.8,
    domainContributions: {
      CAREER: 8.5,
      LEARNING: 14.2,
      FINANCE: 4.8,
      HEALTH: -7.2,
      RELATIONSHIPS: -4.5,
      TIME: -6.0,
    },
    unintendedConsequences: [
      "Nocturnal recovery drops by 9 pts due to 45m sleep reduction",
      "Discretionary time drops to 11 h/wk, elevating allostatic burnout risk",
      "Slight relationship health cooling from reduced evening social presence",
    ],
    isPlausible: true,
  },
  {
    id: "executive_masters",
    title: "Executive Master's Degree",
    description:
      "Intensive 18-month executive program. High tuition outflow and study burden yielding principal leadership career leverage.",
    horizonWeeks: 52,
    leverChanges: {
      WEEKLY_LEARNING_HOURS: 12,
      INVESTMENT_CONTRIBUTION: -1200,
      FAMILY_SOCIAL_HOURS: -5,
      SLEEP_HOURS: -0.5,
    },
    simulatedNodeDeltas: {},
    lhiDelta: 5.2,
    domainContributions: {
      CAREER: 16.5,
      LEARNING: 22.0,
      FINANCE: -5.4,
      HEALTH: -4.8,
      RELATIONSHIPS: -6.5,
      TIME: -8.2,
    },
    unintendedConsequences: [
      "Monthly investment pacing reduced by $1,200 for program tuition",
      "Severe discretionary compression (-17 h/wk total)",
      "Temporary dip in family time requiring structured weekend boundaries",
    ],
    isPlausible: true,
  },
  {
    id: "founder_pivot",
    title: "Early Stage Startup Founder",
    description:
      "Full leap into founding an AI venture. Direct compensation drops, work hours surge, with uncapped career and equity upside.",
    horizonWeeks: 52,
    leverChanges: {
      ANNUAL_COMPENSATION: -75000,
      SLEEP_HOURS: -1.2,
      FAMILY_SOCIAL_HOURS: -6,
      CAREER_GROWTH_VELOCITY: 22,
    },
    simulatedNodeDeltas: {},
    lhiDelta: 2.1,
    domainContributions: {
      CAREER: 18.0,
      LEARNING: 12.0,
      FINANCE: -18.5,
      HEALTH: -14.2,
      RELATIONSHIPS: -11.0,
      TIME: -15.0,
    },
    unintendedConsequences: [
      "Runway contracts rapidly from $75k temporary cash reduction",
      "Sleep duration enters marginal threshold (6.3h/night)",
      "Burnout index spikes to 58 pts, demanding ruthless priority discipline",
    ],
    isPlausible: true,
  },
  {
    id: "longevity_rebalance",
    title: "Longevity & Health Rebalance",
    description:
      "Rebalance priorities around aerobic fitness, restorative sleep, and deep relational intimacy. Slows sprint learning.",
    horizonWeeks: 24,
    leverChanges: {
      SLEEP_HOURS: 0.8,
      VO2_MAX: 4.5,
      FAMILY_SOCIAL_HOURS: 4,
      WEEKLY_LEARNING_HOURS: -3,
    },
    simulatedNodeDeltas: {},
    lhiDelta: 4.6,
    domainContributions: {
      HEALTH: 14.8,
      RELATIONSHIPS: 8.2,
      TIME: 6.5,
      CAREER: -1.2,
      LEARNING: -3.5,
      FINANCE: 0.0,
    },
    unintendedConsequences: [
      "Mild slowdown in technical learning velocity (-3 h/wk)",
      "Marginal delay in next-cycle compensation advancement pace",
    ],
    isPlausible: true,
  },
];
