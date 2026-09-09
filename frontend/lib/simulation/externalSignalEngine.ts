/**
 * Horizon 4: External Signal Normalization & Causal Graph Integration Engine (M17)
 * 
 * Enforces Invariants:
 * - INV-OI72: External Signal Integrity (Timestamped, sourced, range [-100, 100], confidence in [0, 100])
 * - INV-OI74: Signal-to-Outcome Traceability (Signals injected as upstream causal root nodes)
 */

import {
  ExternalSignal,
  NormalizedSignal,
  SignalCategory,
  TraceNode,
  TraceEdge,
  TraceRecord,
  INV_OI72,
  INV_OI74,
} from '../../types/simulation-digital-twin';

// -------------------------------------------------------------
// NORMALIZATION FORMULAS & DICTIONARIES
// -------------------------------------------------------------

export const REGULATORY_SEVERITY_MAP: Record<string, number> = {
  NONE: 0,
  LOW: -20,
  MEDIUM: -50,
  HIGH: -80,
  CRITICAL: -100,
};

export const MARKET_OUTLOOK_MAP: Record<string, number> = {
  VERY_NEGATIVE: -100,
  NEGATIVE: -50,
  NEUTRAL: 0,
  POSITIVE: 50,
  VERY_POSITIVE: 100,
};

/**
 * Normalizes raw external signal values into a standard impact score in range [-100, +100].
 */
export function normalizeRawSignal(
  category: SignalCategory,
  rawValue: number | string
): number {
  if (category === 'REGULATORY') {
    const key = String(rawValue).toUpperCase();
    return REGULATORY_SEVERITY_MAP[key] ?? -50;
  }

  if (category === 'MARKET') {
    const key = String(rawValue).toUpperCase();
    return MARKET_OUTLOOK_MAP[key] ?? 0;
  }

  if (category === 'ECONOMIC') {
    // Treat positive inflation or high interest rates as headwind (0% -> 0, 15% -> -100)
    const num = typeof rawValue === 'number' ? rawValue : parseFloat(rawValue) || 0;
    const clamped = Math.max(0, Math.min(15, num));
    return Number(((clamped / 15) * -100).toFixed(1));
  }

  if (category === 'WORKFORCE') {
    // Treat attrition rate (0% -> 0, 30% -> -100)
    const num = typeof rawValue === 'number' ? rawValue : parseFloat(rawValue) || 0;
    const clamped = Math.max(0, Math.min(30, num));
    return Number(((clamped / 30) * -100).toFixed(1));
  }

  return 0;
}

/**
 * Calculates effective impact scaled by statistical confidence:
 * effectiveImpact = normalizedImpact * (confidencePct / 100)
 */
export function calculateEffectiveImpact(
  normalizedImpact: number,
  confidencePct: number
): number {
  const conf = Math.max(0, Math.min(100, confidencePct));
  return Number((normalizedImpact * (conf / 100)).toFixed(1));
}

// -------------------------------------------------------------
// CANONICAL EXTERNAL SIGNALS CATALOG
// -------------------------------------------------------------

export const CANONICAL_EXTERNAL_SIGNALS: ExternalSignal[] = [
  {
    signalId: 'INF-001',
    source: 'Bureau of Labor Statistics (BLS) CPI Report',
    category: 'ECONOMIC',
    rawValue: 6.8,
    normalizedImpact: -45.3,
    confidencePct: 94.0,
    observedAtUtc: '2026-09-01T08:00:00Z',
    description: 'Headline CPI inflation elevated at 6.8%, inducing corporate spending caution and capital budget strain.',
  },
  {
    signalId: 'REG-001',
    source: 'SEC / Regulatory Compliance Bulletin 2026-Q3',
    category: 'REGULATORY',
    rawValue: 'HIGH',
    normalizedImpact: -80.0,
    confidencePct: 96.5,
    observedAtUtc: '2026-09-02T10:30:00Z',
    description: 'Mandatory autonomous AI oversight and strict audit trails imposed across quantitative modeling units.',
  },
  {
    signalId: 'WRK-001',
    source: 'Internal HR People Analytics & Market Attrition Index',
    category: 'WORKFORCE',
    rawValue: 18.2,
    normalizedImpact: -60.7,
    confidencePct: 88.0,
    observedAtUtc: '2026-09-03T14:15:00Z',
    description: 'Specialized quant analyst attrition at 18.2% annual rate, dampening cross-desk knowledge transfer.',
  },
  {
    signalId: 'MKT-001',
    source: 'Institutional Liquidity & Volatility Composite (VIX/MOVE)',
    category: 'MARKET',
    rawValue: 'NEGATIVE',
    normalizedImpact: -50.0,
    confidencePct: 91.0,
    observedAtUtc: '2026-09-04T12:00:00Z',
    description: 'Elevated market dispersion and compressed spreads requiring heightened risk margins.',
  },
];

export const CANONICAL_SIGNAL_TARGET_MAPPING: Record<string, string> = {
  'INF-001': 'TRAINING_BUDGET',
  'REG-001': 'GOVERNANCE_ADHERENCE',
  'WRK-001': 'TRANSFER_RATE',
  'MKT-001': 'RISK_SCORE',
};

/**
 * Returns all normalized external signals with computed effective impacts.
 */
export function getNormalizedExternalSignals(): NormalizedSignal[] {
  return CANONICAL_EXTERNAL_SIGNALS.map((sig) => {
    const normalizedImpact = normalizeRawSignal(sig.category, sig.rawValue);
    const effectiveImpact = calculateEffectiveImpact(normalizedImpact, sig.confidencePct);
    const affectsMetricId = CANONICAL_SIGNAL_TARGET_MAPPING[sig.signalId] || 'UNKNOWN_TARGET';

    return {
      signalId: sig.signalId,
      category: sig.category,
      normalizedImpact,
      effectiveImpact,
      confidencePct: sig.confidencePct,
      timestampUtc: sig.observedAtUtc,
      source: sig.source,
      affectsMetricId,
    };
  });
}

/**
 * Validates external signal integrity per Invariant INV-OI72.
 */
export function verifyExternalSignalIntegrity(signals: ExternalSignal[]): {
  valid: boolean;
  violations: string[];
} {
  const violations: string[] = [];

  for (const s of signals) {
    if (!s.signalId || !s.source || !s.observedAtUtc) {
      violations.push(`SIGNAL_MISSING_METADATA: Signal ${s.signalId || 'UNKNOWN'} lacks source or timestamp`);
    }
    if (s.normalizedImpact < -100 || s.normalizedImpact > 100) {
      violations.push(`SIGNAL_OUT_OF_BOUNDS: Signal ${s.signalId} normalized impact ${s.normalizedImpact} outside [-100, 100]`);
    }
    if (s.confidencePct < 0 || s.confidencePct > 100) {
      violations.push(`SIGNAL_CONFIDENCE_INVALID: Signal ${s.signalId} confidence ${s.confidencePct}% outside [0, 100]`);
    }
  }

  return {
    valid: violations.length === 0,
    violations,
  };
}

/**
 * Injects external signals into the simulation graph as upstream root nodes,
 * strictly fulfilling Invariant INV-OI74 (Signal-to-Outcome Traceability).
 */
export function injectSignalsIntoGraph(
  existingNodes: TraceNode[],
  existingEdges: TraceEdge[],
  simulationId: string
): { nodes: TraceNode[]; edges: TraceEdge[]; injectedRecords: TraceRecord[] } {
  const normalizedSignals = getNormalizedExternalSignals();
  const newNodes = [...existingNodes];
  const newEdges = [...existingEdges];
  const injectedRecords: TraceRecord[] = [];

  for (const sig of normalizedSignals) {
    const signalNodeId = `TN-SIG-${sig.signalId}`;
    
    // Add external signal node
    newNodes.unshift({
      nodeId: signalNodeId,
      metricId: sig.signalId,
      metricName: `External Signal: ${sig.signalId} (${sig.category})`,
      beforeValue: 0,
      afterValue: sig.effectiveImpact,
      delta: sig.effectiveImpact,
      simulationId,
    });

    const targetMetricId = sig.affectsMetricId;
    const targetNode = newNodes.find((n) => n.metricId === targetMetricId);

    if (targetNode) {
      const edgeId = `TE-${sig.signalId}->${targetMetricId}`;
      newEdges.unshift({
        edgeId,
        sourceNodeId: signalNodeId,
        targetNodeId: targetNode.nodeId,
        contributionPct: Math.abs(sig.effectiveImpact),
        confidencePct: sig.confidencePct,
        sensitivityScore: Number((Math.abs(sig.effectiveImpact) / 100).toFixed(2)),
      });

      injectedRecords.push({
        traceId: `TR-SIG-${sig.signalId}`,
        simulationId,
        sourceMetric: sig.signalId,
        targetMetric: targetMetricId,
        contributionPct: Math.abs(sig.effectiveImpact),
        valueBefore: 0,
        valueAfter: sig.effectiveImpact,
        weightUsed: Number((sig.confidencePct / 100).toFixed(2)),
        createdAtUtc: new Date().toISOString(),
      });
    }
  }

  return { nodes: newNodes, edges: newEdges, injectedRecords };
}
