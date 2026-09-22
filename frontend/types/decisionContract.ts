/**
 * ARX Universal Decision Contract
 *
 * Defines the single authoritative contract for decision state,
 * trigger status, actionability, and execution levels across all surfaces.
 */

export const ACTIONABLE_EXECUTION_STATUSES = [
  'IN_BUY_ZONE',
  'READY_TO_BUY',
] as const;

export type ActionableExecutionStatus = typeof ACTIONABLE_EXECUTION_STATUSES[number];

export const NON_ACTIONABLE_EXECUTION_STATUSES = [
  'WAITING_PULLBACK',
  'IN_BUY_ZONE_AWAITING_TRIGGER',
  'APPROACHING_TARGET',
  'STOPPED_OUT',
  'INSUFFICIENT_HISTORY',
  'UNVERIFIED_ASSET',
  'STALE_MARKET_DATA',
  'UNKNOWN',
] as const;

export const ALL_EXECUTION_STATUSES = [
  ...ACTIONABLE_EXECUTION_STATUSES,
  ...NON_ACTIONABLE_EXECUTION_STATUSES,
] as const;

export type ExecutionStatus = typeof ALL_EXECUTION_STATUSES[number] | string;

export function isStatusActionable(status: string | null | undefined): boolean {
  if (!status) return false;
  return (ACTIONABLE_EXECUTION_STATUSES as readonly string[]).includes(status);
}

export type TimeHorizon = 'INTRADAY' | 'SWING' | 'POSITION' | 'LONG_TERM';

export type UserRole = 'DAY_TRADER' | 'SWING_TRADER' | 'LONG_TERM';

export type DataCompleteness = 'FULL' | 'PARTIAL' | 'INSUFFICIENT' | 'DEGRADED';

export enum DecisionState {
  UNVERIFIED = 'UNVERIFIED',
  INSUFFICIENT_DATA = 'INSUFFICIENT_DATA',
  STALE_DATA = 'STALE_DATA',
  EVIDENCE_INCOMPLETE = 'EVIDENCE_INCOMPLETE',
  VALID_SETUP = 'VALID_SETUP',
  ACTIONABLE_SETUP = 'ACTIONABLE_SETUP',
}

export function isDecisionActionable(
  decisionState: DecisionState | string | null | undefined,
  executionStatus: string | null | undefined
): boolean {
  if (!isStatusActionable(executionStatus)) return false;
  // Strict fail-closed: must be explicitly ACTIONABLE_SETUP. Missing or undefined decisionState is rejected.
  return decisionState === DecisionState.ACTIONABLE_SETUP || decisionState === 'ACTIONABLE_SETUP';
}

export interface ExecutionLevels {
  entryMin?: number;
  entryMax?: number;
  stopLoss?: number;
  stopLossPct?: number;
  target1?: number;
  target1Pct?: number;
  target2?: number;
  target2Pct?: number;
  riskRewardRatio?: number;
}

export interface DecisionVerdict {
  symbol: string;
  horizon: TimeHorizon;
  userRole: UserRole;
  isActionable: boolean;
  canSizeTrade: boolean;
  decisionState: DecisionState | string;
  executionStatus: ExecutionStatus;
  verdictLabel: string;
  disqualificationReason: string | null;
  confluenceScore: number;
  observationDate: string;
  levels: ExecutionLevels;
  dataCompleteness: DataCompleteness;
}

// ---------------------------------------------------------------------------
// Phase 1: Canonical Decision Context & Evidence Contracts
// ---------------------------------------------------------------------------

export type EvidenceQualityState =
  | 'AUTHORITATIVE'
  | 'PROVISIONAL'
  | 'FALLBACK'
  | 'UNAVAILABLE'
  | 'STALE';

export type EvidenceDomain =
  | 'MARKET_DATA'
  | 'FUNDAMENTALS'
  | 'MACRO'
  | 'LIQUIDITY'
  | 'ORDER_FLOW'
  | 'DERIVATIVES';

export interface MarketEvidenceContract {
  candlesCount: number;
  lastClose: number;
  vwap?: number | null;
  atr14?: number | null;
  provider: string;
  asOf: string | number;
}

export interface FundamentalEvidenceContract {
  peRatio?: number | null;
  marketCap?: number | null;
  sector?: string | null;
  source: string;
  filingDate?: string | null;
}

export interface MacroEvidenceContract {
  regimeLabel?: string | null;
  yieldSpread10Y2Y?: number | null;
  inflationRate?: number | null;
  fredObservationDate?: string | null;
  source: string;
}

export interface LiquidityEvidenceContract {
  spreadBps?: number | null;
  avgVolume30D?: number | null;
  liquidityGatePassed: boolean;
  source: string;
}

export interface ARXEvidenceItem<T = any> {
  domain: EvidenceDomain;
  quality: EvidenceQualityState;
  source: string;
  observedAt: string | number;
  payload: T;
  isStale: boolean;
  stalenessReason?: string | null;
}

export interface ARXDecisionContext {
  decisionId: string;
  symbol: string;
  horizon: TimeHorizon;
  userRole: UserRole;
  timestamp: string;
  marketEvidence: ARXEvidenceItem<MarketEvidenceContract>;
  fundamentalEvidence: ARXEvidenceItem<FundamentalEvidenceContract>;
  macroEvidence: ARXEvidenceItem<MacroEvidenceContract>;
  liquidityEvidence: ARXEvidenceItem<LiquidityEvidenceContract>;
  evidenceCompleteness: DataCompleteness;
  isDegraded: boolean;
}

export type ARXDecisionAuthority = 'BACKEND_CANONICAL' | 'DISPLAY_ONLY_MARKET_DATA';

export interface ARXDecision {
  context: ARXDecisionContext;
  verdict: DecisionVerdict;
  confluenceScore: number;
  modelTrace: {
    modelName: string;
    version: string;
    generatedAt: string;
    passedGates: string[];
    failedGates: string[];
  };
  authority: ARXDecisionAuthority;
}

/**
 * Creates a degraded, display-only decision context when analytical backend is unreachable.
 * Guaranteed to fail-closed under Phase 0/1 governance.
 */
export function createDegradedDecisionContext(
  symbol: string,
  horizon: TimeHorizon = 'SWING',
  userRole: UserRole = 'LONG_TERM',
  marketData?: Partial<MarketEvidenceContract>
): ARXDecisionContext {
  const now = new Date().toISOString();
  return {
    decisionId: `degraded-${symbol.toUpperCase()}-${Date.now()}`,
    symbol: symbol.toUpperCase(),
    horizon,
    userRole,
    timestamp: now,
    marketEvidence: {
      domain: 'MARKET_DATA',
      quality: marketData?.lastClose ? 'FALLBACK' : 'UNAVAILABLE',
      source: marketData?.provider || 'yahoo_finance_direct',
      observedAt: marketData?.asOf || now,
      payload: {
        candlesCount: marketData?.candlesCount ?? 0,
        lastClose: marketData?.lastClose ?? 0,
        vwap: marketData?.vwap ?? null,
        atr14: marketData?.atr14 ?? null,
        provider: marketData?.provider || 'yahoo_finance_direct',
        asOf: marketData?.asOf || now,
      },
      isStale: false,
    },
    fundamentalEvidence: {
      domain: 'FUNDAMENTALS',
      quality: 'UNAVAILABLE',
      source: 'none',
      observedAt: now,
      payload: {
        peRatio: null,
        marketCap: null,
        sector: null,
        source: 'none',
        filingDate: null,
      },
      isStale: true,
      stalenessReason: 'Backend analytics engine unreachable',
    },
    macroEvidence: {
      domain: 'MACRO',
      quality: 'UNAVAILABLE',
      source: 'none',
      observedAt: now,
      payload: {
        regimeLabel: null,
        yieldSpread10Y2Y: null,
        inflationRate: null,
        fredObservationDate: null,
        source: 'none',
      },
      isStale: true,
      stalenessReason: 'Backend analytics engine unreachable',
    },
    liquidityEvidence: {
      domain: 'LIQUIDITY',
      quality: 'UNAVAILABLE',
      source: 'none',
      observedAt: now,
      payload: {
        spreadBps: null,
        avgVolume30D: null,
        liquidityGatePassed: false,
        source: 'none',
      },
      isStale: true,
      stalenessReason: 'Backend analytics engine unreachable',
    },
    evidenceCompleteness: 'DEGRADED',
    isDegraded: true,
  };
}


