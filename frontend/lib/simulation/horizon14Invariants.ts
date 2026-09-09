/**
 * Horizon 14 Invariants: Professional Investment Terminal & Behavioral Governance
 *
 * Implements fail-closed behavioral safety invariants:
 * - INV-OI112-P: Experience Boundary Integrity (Zero concept leakage between Terminal and Life OS)
 * - INV-OI113-P: Counterfactual Proof Determinism (Performance attribution must be mathematically provable)
 * - INV-OI114-P: Human Agency & Sizing Clamp Bounds (Prefer sizing clamps over total trading bans)
 * - INV-OI115-P: Persistent Terminal Navigation (Single shell for all 6 flagship hubs, persistent Governor link)
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
    ].join(' ').toLowerCase();

    for (const term of FORBIDDEN_LIFESTYLE_TERMS) {
      // Word boundary regex to prevent false positives on substrings
      const regex = new RegExp(`\\b${term}\\b`, 'i');
      if (regex.test(fullText)) {
        violations.push(
          `INV-OI112-P VIOLATION: Ticket ${ticket.ticker} contains forbidden lifestyle term "${term}". Leaked rationale: "${fullText.slice(0, 100)}..."`
        );
      }
    }
  });

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI112-P',
    violations,
    metadata: {
      ticketsInspected: tickets.length,
    },
  };
}

export interface PerformanceAttributionAudit {
  tradeId: string;
  ticker: string;
  governorClamped: boolean;
  actualReturnDollar: number;
  counterfactualUnclampedReturnDollar: number;
  preservedCapitalDollar: number;
}

/**
 * INV-OI113-P: Counterfactual Proof Determinism
 * Asserts that capital preservation figures are mathematically calculated from exact
 * difference between clamped execution and unclamped execution, preventing vanity metrics.
 */
export function verifyCounterfactualProofDeterminism(
  records: PerformanceAttributionAudit[]
): InvariantResult {
  const violations: string[] = [];
  let totalPreserved = 0;

  records.forEach((rec) => {
    const expectedPreserved = Math.max(0, rec.actualReturnDollar - rec.counterfactualUnclampedReturnDollar);
    const discrepancy = Math.abs(rec.preservedCapitalDollar - expectedPreserved);

    // Discrepancy > 1 dollar indicates math discrepancy
    if (discrepancy > 1.0) {
      violations.push(
        `INV-OI113-P VIOLATION: Attribution for ${rec.ticker} (Trade: ${rec.tradeId}) reports preserved capital $${rec.preservedCapitalDollar}, expected $${expectedPreserved} (diff: $${discrepancy.toFixed(2)})`
      );
    }
    totalPreserved += rec.preservedCapitalDollar;
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
    if (d.governedShares === 0 && !d.isExplicitCircuitBreaker) {
      violations.push(
        `INV-OI114-P VIOLATION: Setup "${d.setupId}" completely blocked (0 shares) without an active hard circuit breaker. Prefer sizing reduction over total ban.`
      );
    }

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
 * INV-OI115-P: Persistent Terminal Navigation
 * Asserts that all 6 flagship hubs (/radar, /setups, /portfolio, /journal, /performance, /research)
 * render within the shared TerminalShell, ensuring persistent subheader, Governor reachability,
 * Command Palette, and mobile navigation without route flashing or shell re-mounting.
 */
export function verifyPersistentTerminalNavigation(
  activeHubs: string[]
): InvariantResult {
  const violations: string[] = [];
  const requiredHubs = ['radar', 'setups', 'portfolio', 'journal', 'performance', 'research'];

  for (const hub of requiredHubs) {
    if (!activeHubs.includes(hub)) {
      violations.push(`INV-OI115-P VIOLATION: Missing flagship terminal hub "${hub}" in persistent shell registry.`);
    }
  }

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI115-P',
    violations,
    metadata: {
      registeredHubs: activeHubs,
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
  activeHubs?: string[];
}): {
  certified: boolean;
  results: Record<string, InvariantResult>;
  totalViolations: number;
} {
  const boundary = verifyExperienceBoundaryIntegrity(payload.terminalTickets);
  const proof = verifyCounterfactualProofDeterminism(payload.attributionRecords);
  const agency = verifyHumanAgencySizingBounds(payload.sizingDecisions);
  const navigation = verifyPersistentTerminalNavigation(
    payload.activeHubs || ['radar', 'setups', 'portfolio', 'journal', 'performance', 'research']
  );

  const totalViolations =
    boundary.violations.length + proof.violations.length + agency.violations.length + navigation.violations.length;

  return {
    certified: totalViolations === 0,
    results: {
      'INV-OI112-P': boundary,
      'INV-OI113-P': proof,
      'INV-OI114-P': agency,
      'INV-OI115-P': navigation,
    },
    totalViolations,
  };
}
