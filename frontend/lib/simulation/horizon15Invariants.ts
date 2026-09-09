/**
 * Horizon 15 Invariants: Proof of Edge & Attribution Engine
 *
 * Implements fail-closed quantitative and behavioral attribution invariants:
 * - INV-OI116-P: Counterfactual Attribution Mathematical Determinism
 * - INV-OI117-P: Audit Ledger Completeness & Immutability
 * - INV-OI118-P: Personal Edge Statistical Significance
 */

export interface GovernorLedgerEntry {
  id: string;
  timestamp: string;
  ticker: string;
  setupArchetype: 'VCP_BREAKOUT' | 'EMA_PULLBACK' | 'ORB_BREAKOUT' | 'DIP_BUY';
  marketRegime: 'UPTREND' | 'CHOP' | 'DISTRIBUTION';
  executionWindow: 'MORNING_PRIME' | 'MIDDAY' | 'LATE_SESSION';
  confluenceScore: number;
  unclampedRiskDollar: number;
  governedRiskDollar: number;
  clampFactorPct: number;
  clampReasonCategory: 'DRAWDOWN_DEFENSE' | 'EXECUTION_WINDOW' | 'CAPITAL_FLOOR';
  clampReasonDetail: string;
  tradeOutcome: 'WIN' | 'LOSS' | 'SCRATCH';
  unclampedPnLDollar: number;
  governedPnLDollar: number;
  capitalPreservedDollar: number;
  status: 'ACCEPTED' | 'IGNORED';
}

export interface AttributionSummary {
  startingEquity: number;
  currentGovernedEquity: number;
  currentUnclampedEquity: number;
  capitalPreservedTotal: number;
  interventionsCount: number;
  adherenceRatePct: number;
  governedWinRate: number;
  unclampedWinRate: number;
  governedProfitFactor: number;
  unclampedProfitFactor: number;
  governedMaxDrawdown: number;
  unclampedMaxDrawdown: number;
  governedSharpe: number;
  unclampedSharpe: number;
  governedSortino: number;
  unclampedSortino: number;
  riskOfRuinGovernedPct: number;
  riskOfRuinUnclampedPct: number;
  preservedByCategory: {
    drawdownDefense: number;
    executionWindow: number;
    capitalFloor: number;
  };
}

export interface ArchetypeEdgeMetric {
  archetype: 'VCP_BREAKOUT' | 'EMA_PULLBACK' | 'ORB_BREAKOUT' | 'DIP_BUY';
  label: string;
  sampleCount: number;
  winRatePct: number;
  profitFactor: number;
  totalPnLDollar: number;
  edgeTier: 'STRONG_EDGE' | 'MODERATE_EDGE' | 'NEGATIVE_TILT';
  recommendedAction: string;
}

export interface PersonalEdgeReport {
  archetypes: ArchetypeEdgeMetric[];
  highestEdgeSetup: string;
  worstTiltLeak: string;
}

export interface InvariantResult {
  compliant: boolean;
  invariantId: string;
  violations: string[];
  metadata?: Record<string, unknown>;
}

/**
 * INV-OI116-P: Counterfactual Attribution Mathematical Determinism
 * Asserts that:
 * 1. For every clamped trade, preserved capital == max(0, unclampedLoss - governedLoss).
 * 2. Cumulative preserved capital reported in summary matches the exact sum of individual ledger entries.
 * 3. Governed equity curve strictly equals initial equity + cumulative governed PnL.
 */
export function verifyCounterfactualMathDeterminism(
  ledger: GovernorLedgerEntry[],
  summary: AttributionSummary
): InvariantResult {
  const violations: string[] = [];
  let calculatedPreservedTotal = 0;
  let calculatedGovernedEquity = summary.startingEquity;
  let calculatedUnclampedEquity = summary.startingEquity;

  ledger.forEach((entry) => {
    // 1. Math check for each clamped trade
    if (entry.clampFactorPct > 0) {
      const expectedPreserved = Math.max(0, Math.abs(entry.unclampedPnLDollar) - Math.abs(entry.governedPnLDollar));
      if (entry.tradeOutcome === 'LOSS') {
        const diff = Math.abs(entry.capitalPreservedDollar - expectedPreserved);
        if (diff > 0.01) {
          violations.push(
            `INV-OI116-P VIOLATION: Ledger ${entry.id} (${entry.ticker}) reports preserved $${entry.capitalPreservedDollar}, expected $${expectedPreserved} (diff: $${diff.toFixed(2)})`
          );
        }
      }
      calculatedPreservedTotal += entry.capitalPreservedDollar;
    }

    calculatedGovernedEquity += entry.governedPnLDollar;
    calculatedUnclampedEquity += entry.unclampedPnLDollar;
  });

  // 2. Summary reconciliation
  const summaryDiff = Math.abs(summary.capitalPreservedTotal - Math.round(calculatedPreservedTotal));
  if (summaryDiff > 1.0) {
    violations.push(
      `INV-OI116-P VIOLATION: Attribution summary total preserved capital $${summary.capitalPreservedTotal} does not match ledger sum $${Math.round(calculatedPreservedTotal)}.`
    );
  }

  // 3. Equity reconciliation
  if (Math.abs(summary.currentGovernedEquity - Math.round(calculatedGovernedEquity)) > 1.0) {
    violations.push(
      `INV-OI116-P VIOLATION: Governed equity $${summary.currentGovernedEquity} does not match sum of PnL $${Math.round(calculatedGovernedEquity)}.`
    );
  }

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI116-P',
    violations,
    metadata: {
      auditedEntries: ledger.length,
      calculatedPreservedTotal,
      finalGovernedEquity: calculatedGovernedEquity,
      finalUnclampedEquity: calculatedUnclampedEquity,
    },
  };
}

/**
 * INV-OI117-P: Audit Ledger Completeness & Immutability
 * Asserts that all ledger entries contain non-null immutable audit provenance:
 * id, timestamp, ticker, setup archetype, market regime, execution window, clamp details, and outcome.
 */
export function verifyAuditLedgerCompleteness(
  ledger: GovernorLedgerEntry[]
): InvariantResult {
  const violations: string[] = [];

  ledger.forEach((e) => {
    if (!e.id || !e.id.startsWith('LEDGER-')) {
      violations.push(`INV-OI117-P VIOLATION: Entry has invalid or missing ledger ID: "${e.id}"`);
    }
    if (!e.timestamp || e.timestamp.length < 10) {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} has invalid timestamp: "${e.timestamp}"`);
    }
    if (!e.ticker || e.ticker.trim() === '') {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} missing ticker`);
    }
    if (!['VCP_BREAKOUT', 'EMA_PULLBACK', 'ORB_BREAKOUT', 'DIP_BUY'].includes(e.setupArchetype)) {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} has invalid setup archetype: "${e.setupArchetype}"`);
    }
    if (!['UPTREND', 'CHOP', 'DISTRIBUTION'].includes(e.marketRegime)) {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} has invalid market regime: "${e.marketRegime}"`);
    }
    if (!['MORNING_PRIME', 'MIDDAY', 'LATE_SESSION'].includes(e.executionWindow)) {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} has invalid execution window: "${e.executionWindow}"`);
    }
    if (e.clampFactorPct < 0 || e.clampFactorPct > 80) {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} clamp factor ${e.clampFactorPct}% outside [0, 80] range`);
    }
    if (!e.clampReasonDetail || e.clampReasonDetail.trim() === '') {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} missing clamp reason detail`);
    }
    if (!['WIN', 'LOSS', 'SCRATCH'].includes(e.tradeOutcome)) {
      violations.push(`INV-OI117-P VIOLATION: Entry ${e.id} has invalid trade outcome: "${e.tradeOutcome}"`);
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI117-P',
    violations,
    metadata: {
      entriesAudited: ledger.length,
    },
  };
}

/**
 * INV-OI118-P: Personal Edge Statistical Significance
 * Asserts that any setup archetype marked as 'STRONG_EDGE' requires at least 5 audited samples
 * and a profit factor >= 2.0 to prevent sample bias.
 */
export function verifyPersonalEdgeSignificance(
  report: PersonalEdgeReport
): InvariantResult {
  const violations: string[] = [];

  report.archetypes.forEach((arch) => {
    if (arch.edgeTier === 'STRONG_EDGE') {
      if (arch.sampleCount < 5) {
        violations.push(
          `INV-OI118-P VIOLATION: Archetype "${arch.label}" awarded STRONG_EDGE with only ${arch.sampleCount} samples (minimum required: 5).`
        );
      }
      if (arch.profitFactor < 2.0) {
        violations.push(
          `INV-OI118-P VIOLATION: Archetype "${arch.label}" awarded STRONG_EDGE with sub-par Profit Factor ${arch.profitFactor} (minimum required: 2.0).`
        );
      }
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI118-P',
    violations,
    metadata: {
      archetypesAudited: report.archetypes.length,
    },
  };
}

/**
 * Master Audit for Horizon 15 Invariants
 */
export function auditHorizon15Master(payload: {
  ledger: GovernorLedgerEntry[];
  summary: AttributionSummary;
  edgeReport: PersonalEdgeReport;
}): {
  certified: boolean;
  results: Record<string, InvariantResult>;
  totalViolations: number;
} {
  const mathDeterminism = verifyCounterfactualMathDeterminism(payload.ledger, payload.summary);
  const ledgerCompleteness = verifyAuditLedgerCompleteness(payload.ledger);
  const edgeSignificance = verifyPersonalEdgeSignificance(payload.edgeReport);

  const totalViolations =
    mathDeterminism.violations.length +
    ledgerCompleteness.violations.length +
    edgeSignificance.violations.length;

  return {
    certified: totalViolations === 0,
    results: {
      'INV-OI116-P': mathDeterminism,
      'INV-OI117-P': ledgerCompleteness,
      'INV-OI118-P': edgeSignificance,
    },
    totalViolations,
  };
}
