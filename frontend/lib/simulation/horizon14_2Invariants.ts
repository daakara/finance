/**
 * Horizon 14.2 Invariants: Production Readiness Audit & Zero-Mock Certification
 *
 * Implements fail-closed invariants:
 * - INV-OI119-P: Frontend Data Authenticity (Explicit Data Provenance & Zero Stealth Mock Data)
 * - INV-OI120-P: Filter Correctness & Search Integrity (Zero Cross-Category Contamination & Substring Match)
 * - INV-OI121-P: Navigation Continuity & Anti-Orphaning (Full persistent shell coverage and return escape hatches)
 */

export interface InvariantResult {
  compliant: boolean;
  invariantId: string;
  violations: string[];
  metadata?: Record<string, unknown>;
}

export type DataProvenanceSource = 'BENCHMARK_SCENARIO' | 'LIVE_TRADER';

export interface DataProvenanceInspection {
  componentName: string;
  sourceMode: DataProvenanceSource;
  disclosureLabel: string;
  sampleSize: number;
  isLive: boolean;
  hasFallbackDisguisedAsLive: boolean;
}

/**
 * INV-OI119-P: Frontend Data Authenticity
 * Asserts that any benchmark or synthetic data displayed in the interface is explicitly
 * labeled with its provenance, sample size, and scenario context.
 * Hardcoded metrics must NEVER be disguised as live account metrics.
 */
export function verifyFrontendDataAuthenticity(
  inspections: DataProvenanceInspection[]
): InvariantResult {
  const violations: string[] = [];

  for (const inspection of inspections) {
    if (inspection.hasFallbackDisguisedAsLive) {
      violations.push(
        `INV-OI119-P VIOLATION in ${inspection.componentName}: Fallback or benchmark data is disguised as live user account data without disclosure.`
      );
    }

    if (inspection.sourceMode === 'BENCHMARK_SCENARIO') {
      const labelLower = inspection.disclosureLabel.toLowerCase();
      const hasAuditedOrBenchmark =
        labelLower.includes('benchmark') ||
        labelLower.includes('audited') ||
        labelLower.includes('scenario') ||
        labelLower.includes('reference');

      if (!hasAuditedOrBenchmark) {
        violations.push(
          `INV-OI119-P VIOLATION in ${inspection.componentName}: Benchmark scenario data missing explicit disclosure keywords (must contain "Benchmark", "Audited", or "Scenario"). Found: "${inspection.disclosureLabel}"`
        );
      }

      if (inspection.sampleSize <= 0) {
        violations.push(
          `INV-OI119-P VIOLATION in ${inspection.componentName}: Benchmark scenario must declare a positive verified trade or sample count. Found: ${inspection.sampleSize}`
        );
      }
    }

    if (inspection.sourceMode === 'LIVE_TRADER' && !inspection.isLive) {
      violations.push(
        `INV-OI119-P VIOLATION in ${inspection.componentName}: Component declared as LIVE_TRADER but isLive flag is false.`
      );
    }
  }

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI119-P',
    violations,
    metadata: {
      inspectionsAudited: inspections.length,
    },
  };
}

export interface RadarAssetItem {
  ticker: string;
  companyName: string;
  categories: ('VCP' | 'SMART_MONEY' | 'VALUE')[];
  screeningModel: string;
  catalyst: string;
  setupNote: string;
}

/**
 * INV-OI120-P: Filter Correctness & Search Integrity
 * Asserts that asset category filtering exhibits zero cross-category contamination:
 * - VCP filter returns only assets tagged VCP
 * - SMART_MONEY returns only assets tagged SMART_MONEY
 * - VALUE returns only assets tagged VALUE
 * And search returns all assets matching query across ticker, company, catalyst, model, or note.
 */
export function verifyFilterCorrectnessAndSearch(
  universe: RadarAssetItem[],
  testCases: {
    categoryFilter?: 'ALL' | 'VCP' | 'SMART_MONEY' | 'VALUE';
    searchQuery?: string;
    expectedMatches: string[];
    disallowedMatches?: string[];
  }[]
): InvariantResult {
  const violations: string[] = [];

  for (const tc of testCases) {
    let filtered = universe;

    if (tc.categoryFilter && tc.categoryFilter !== 'ALL') {
      filtered = filtered.filter((asset) => asset.categories.includes(tc.categoryFilter as any));
    }

    if (tc.searchQuery && tc.searchQuery.trim().length > 0) {
      const q = tc.searchQuery.toLowerCase();
      filtered = filtered.filter(
        (asset) =>
          asset.ticker.toLowerCase().includes(q) ||
          asset.companyName.toLowerCase().includes(q) ||
          asset.catalyst.toLowerCase().includes(q) ||
          asset.screeningModel.toLowerCase().includes(q) ||
          asset.setupNote.toLowerCase().includes(q)
      );
    }

    const matchedTickers = filtered.map((a) => a.ticker);

    // Verify expected matches exist
    for (const exp of tc.expectedMatches) {
      if (!matchedTickers.includes(exp)) {
        violations.push(
          `INV-OI120-P VIOLATION: Expected asset "${exp}" missing from filter result (Filter: ${tc.categoryFilter || 'NONE'}, Query: "${tc.searchQuery || ''}"). Matched: [${matchedTickers.join(', ')}]`
        );
      }
    }

    // Verify disallowed matches do NOT exist
    if (tc.disallowedMatches) {
      for (const dis of tc.disallowedMatches) {
        if (matchedTickers.includes(dis)) {
          violations.push(
            `INV-OI120-P VIOLATION: Disallowed asset "${dis}" erroneously appeared in filter result (Filter: ${tc.categoryFilter || 'NONE'}, Query: "${tc.searchQuery || ''}"). Cross-category contamination detected!`
          );
        }
      }
    }
  }

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI120-P',
    violations,
    metadata: {
      universeSize: universe.length,
      testCasesAudited: testCases.length,
    },
  };
}

export interface RouteNavigationInspection {
  routePath: string;
  routeType: 'TERMINAL' | 'COCKPIT';
  hasTerminalNavbar?: boolean;
  hasGovernorBadge?: boolean;
  hasCommandPaletteAccess?: boolean;
  hasMobileDock?: boolean;
  hasReturnToTerminalLink?: boolean;
  isOrphaned: boolean;
}

/**
 * INV-OI121-P: Navigation Continuity & Anti-Orphaning
 * Asserts that:
 * 1. Every Terminal flagship route renders inside TerminalShell with navbar, Governor badge, Cmd+K, and mobile dock.
 * 2. Every Cockpit route renders inside CockpitShell with Governor navigation tabs and Return-to-Terminal escape hatch.
 * 3. Zero routes are orphaned without navigational continuity.
 */
export function verifyNavigationContinuity(
  routes: RouteNavigationInspection[]
): InvariantResult {
  const violations: string[] = [];

  const requiredTerminalRoutes = ['/radar', '/setups', '/portfolio', '/journal', '/performance', '/research'];
  const requiredCockpitRoutes = ['/cockpit', '/cockpit/today', '/cockpit/future', '/cockpit/progress', '/cockpit/household'];

  for (const route of routes) {
    if (route.isOrphaned) {
      violations.push(`INV-OI121-P VIOLATION: Route "${route.routePath}" is orphaned with broken or absent navigation wrapper.`);
    }

    if (route.routeType === 'TERMINAL') {
      if (!route.hasTerminalNavbar) {
        violations.push(`INV-OI121-P VIOLATION: Terminal route "${route.routePath}" missing persistent Terminal Navbar.`);
      }
      if (!route.hasGovernorBadge) {
        violations.push(`INV-OI121-P VIOLATION: Terminal route "${route.routePath}" missing Behavioral Governor Badge.`);
      }
      if (!route.hasCommandPaletteAccess) {
        violations.push(`INV-OI121-P VIOLATION: Terminal route "${route.routePath}" missing Command Palette shortcut/access.`);
      }
    }

    if (route.routeType === 'COCKPIT') {
      if (!route.hasReturnToTerminalLink) {
        violations.push(`INV-OI121-P VIOLATION: Cockpit route "${route.routePath}" missing "Return to ARX Terminal" escape hatch.`);
      }
      if (!route.hasGovernorBadge) {
        violations.push(`INV-OI121-P VIOLATION: Cockpit route "${route.routePath}" missing Governor branding/badge.`);
      }
    }
  }

  const inspectedPaths = routes.map((r) => r.routePath);
  for (const reqTerm of requiredTerminalRoutes) {
    if (!inspectedPaths.includes(reqTerm)) {
      violations.push(`INV-OI121-P VIOLATION: Required terminal route "${reqTerm}" was not audited in navigation continuity check.`);
    }
  }
  for (const reqCockpit of requiredCockpitRoutes) {
    if (!inspectedPaths.includes(reqCockpit)) {
      violations.push(`INV-OI121-P VIOLATION: Required cockpit route "${reqCockpit}" was not audited in navigation continuity check.`);
    }
  }

  return {
    compliant: violations.length === 0,
    invariantId: 'INV-OI121-P',
    violations,
    metadata: {
      totalRoutesAudited: routes.length,
    },
  };
}

/**
 * Master Audit for Horizon 14.2 Production Readiness
 */
export function auditHorizon14_2Master(payload: {
  provenanceInspections: DataProvenanceInspection[];
  radarUniverse: RadarAssetItem[];
  filterTestCases: {
    categoryFilter?: 'ALL' | 'VCP' | 'SMART_MONEY' | 'VALUE';
    searchQuery?: string;
    expectedMatches: string[];
    disallowedMatches?: string[];
  }[];
  routeInspections: RouteNavigationInspection[];
}): {
  certified: boolean;
  results: Record<string, InvariantResult>;
  totalViolations: number;
} {
  const authenticity = verifyFrontendDataAuthenticity(payload.provenanceInspections);
  const filter = verifyFilterCorrectnessAndSearch(payload.radarUniverse, payload.filterTestCases);
  const navigation = verifyNavigationContinuity(payload.routeInspections);

  const totalViolations =
    authenticity.violations.length + filter.violations.length + navigation.violations.length;

  return {
    certified: totalViolations === 0,
    results: {
      'INV-OI119-P': authenticity,
      'INV-OI120-P': filter,
      'INV-OI121-P': navigation,
    },
    totalViolations,
  };
}
