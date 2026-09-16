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
  if (!decisionState) return true; // Backward compatibility if backend omitted decisionState
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

