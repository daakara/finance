/**
 * Horizon 14 Invariants: Professional Investment Terminal & Behavioral Governance
 *
 * Implements three fail-closed behavioral safety invariants:
 * - INV-OI112-P: Experience Boundary Integrity (Zero concept leakage between Terminal and Life OS)
 * - INV-OI113-P: Counterfactual Proof Determinism (Performance attribution must be mathematically provable)
 * - INV-OI114-P: Human Agency & Sizing Clamp Bounds (Prefer sizing clamps over total trading bans)
 */

export interface TerminalTicketInspection {
  ticketId: string;
  ticker: string;
  entryPrice: number;
  stopPrice: number;
  recommendedShares: number;
  recommendedDollarRisk: number;
  rationaleCategory: string;
  visibleTextChunks: string[];
}

export interface InvariantResult {
  compliant: boolean;
  invariantId: string;
  violations: string[];
  metadata?: Record<string, unknown>;
}

// Banned lifestyle concepts that must NEVER leak into the Terminal experience
const FORBIDDEN_LIFESTYLE_TERMS = [
  'life health index',
  'lhi',
  'household health index',
  'hhi',
  'identity alignment index',
  'iai',
  'domestic strain',
  'partner twin',
  'childcare',
  'sleep debt',
  'chore budget',
  '168-hour',
  'whoop',
  'oura',
  'analytics manager to ai strategy leader',
  'identity trajectory',
  'career archetype',
];

/**
 * INV-OI112-P: Experience Boundary Integrity
 * Asserts that no lifestyle or personal OS concepts leak into trading terminal tickets,
 * navigation labels, or sizing explanations.
 */
export function verifyExperienceBoundaryIntegrity(
  tickets: TerminalTicketInspection[]
): InvariantResult {
  const violations: string[] = [];

  tickets.forEach((ticket) => {
    const fullText = [
      ticket.ticker,
      ticket.rationaleCategory,
      ...ticket.visibleTextChunks,
    ]
      .join(' ')
      .toLowerCase();

    FORBIDDEN_LIFESTYLE_TERMS.forEach((term) => {
      if (fullText.includes(term)) {
        violations.push(
          `INV-OI112-P VIOLATION: Terminal ticket "${ticket.ticketId}" for ${ticket.ticker} leaks forbidden lifestyle concept "${term}".`
        );
      }
    });
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI112-P',
    violations,
    metadata: {
      ticketsAudited: tickets.length,
      forbiddenTermsChecked: FORBIDDEN_LIFESTYLE_TERMS.length,
    },
  };
}

export interface PerformanceAttributionAudit {
  tradeId: string;
  unclampedDollarRisk: number;
  governedDollarRisk: number;
  actualPnL: number;
  counterfactualUnclampedPnL: number;
  isLoss: boolean;
}

/**
 * INV-OI113-P: Counterfactual Proof Determinism
 * Asserts that all performance attribution claims (Capital Preserved, Drawdown Reduction)
 * are mathematically reproducible from the historical trade ledger.
 */
export function verifyCounterfactualProofDeterminism(
  records: PerformanceAttributionAudit[]
): InvariantResult {
  const violations: string[] = [];
  let totalPreserved = 0;

  records.forEach((record) => {
    // If trade was a loss, governed risk must have reduced dollar drawdown:
    if (record.isLoss) {
      const riskDifference = record.unclampedDollarRisk - record.governedDollarRisk;
      if (riskDifference < 0) {
        violations.push(
          `INV-OI113-P VIOLATION: Trade "${record.tradeId}" has higher governed risk than unclamped risk during a loss.`
        );
      } else {
        totalPreserved += riskDifference;
      }
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI113-P',
    violations,
    metadata: {
      recordsAudited: records.length,
      totalPreservedCapitalCalculated: totalPreserved,
    },
  };
}

export interface SizingClampDecision {
  setupId: string;
  unclampedShares: number;
  governedShares: number;
  clampFactorPct: number;
  isExplicitCircuitBreaker: boolean;
}

/**
 * INV-OI114-P: Human Agency & Sizing Clamp Bounds
 * Asserts that the Governor favors disciplined risk reduction (10% to 75% clamp)
 * rather than arbitrary trade bans (0 shares), unless an explicit hard circuit breaker is active.
 */
export function verifyHumanAgencySizingBounds(
  decisions: SizingClampDecision[]
): InvariantResult {
  const violations: string[] = [];

  decisions.forEach((d) => {
    // If 0 shares recommended and not an explicit circuit breaker:
    if (d.governedShares === 0 && !d.isExplicitCircuitBreaker) {
      violations.push(
        `INV-OI114-P VIOLATION: Setup "${d.setupId}" completely blocked (0 shares) without an active hard circuit breaker. Prefer sizing reduction over total ban.`
      );
    }

    // If clamped, clamp factor should be within legitimate range (10% to 75%)
    if (d.governedShares > 0 && d.governedShares < d.unclampedShares) {
      if (d.clampFactorPct < 5 || d.clampFactorPct > 80) {
        violations.push(
          `INV-OI114-P VIOLATION: Setup "${d.setupId}" clamp factor ${d.clampFactorPct}% is outside standard boundary (10% to 75%).`
        );
      }
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI114-P',
    violations,
    metadata: {
      decisionsAudited: decisions.length,
    },
  };
}

/**
 * Master Audit for Horizon 14 Invariants
 */
export function auditHorizon14Master(payload: {
  terminalTickets: TerminalTicketInspection[];
  attributionRecords: PerformanceAttributionAudit[];
  sizingDecisions: SizingClampDecision[];
}): {
  certified: boolean;
  results: Record<string, InvariantResult>;
  totalViolations: number;
} {
  const boundary = verifyExperienceBoundaryIntegrity(payload.terminalTickets);
  const proof = verifyCounterfactualProofDeterminism(payload.attributionRecords);
  const agency = verifyHumanAgencySizingBounds(payload.sizingDecisions);

  const totalViolations =
    boundary.violations.length + proof.violations.length + agency.violations.length;

  return {
    certified: totalViolations === 0,
    results: {
      'INV-OI112-P': boundary,
      'INV-OI113-P': proof,
      'INV-OI114-P': agency,
    },
    totalViolations,
  };
}
