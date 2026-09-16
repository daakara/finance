/**
 * ARX Universal Decision Contract
 *
 * Defines the single authoritative contract for decision state,
 * trigger status, actionability, and execution levels across all surfaces.
 */

export type ExecutionStatus =
  | 'IN_BUY_ZONE'
  | 'IN_BUY_ZONE_AWAITING_TRIGGER'
  | 'WAITING_PULLBACK'
  | 'APPROACHING_TARGET'
  | 'INSUFFICIENT_HISTORY'
  | 'STALE_MARKET_DATA'
  | 'UNKNOWN';

export type TimeHorizon = 'INTRADAY' | 'SWING';

export type DataCompleteness = 'FULL' | 'PARTIAL' | 'INSUFFICIENT';

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
  isActionable: boolean;
  canSizeTrade: boolean;
  executionStatus: ExecutionStatus;
  verdictLabel: string;
  disqualificationReason: string | null;
  confluenceScore: number;
  observationDate: string;
  levels: ExecutionLevels;
  dataCompleteness: DataCompleteness;
}
