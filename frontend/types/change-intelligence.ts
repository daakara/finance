/**
 * ARX Terminal vNext - Change Intelligence & Materiality Engine Contracts
 * Source: docs/sprints/SPRINT_3_CHANGE_INTELLIGENCE_EXECUTION_PACKAGE.md
 */

import { DomainConfidence, ExecutionState, LiquidityTier, MarketRegime } from "./workstation";

export type MaterialitySeverity = "NONE" | "INFO" | "MATERIAL" | "CRITICAL";

export type MaterialityCategory =
  | "SCORE"
  | "FLOW"
  | "EXECUTION"
  | "REGIME"
  | "LIQUIDITY"
  | "VALIDATION"
  | "STRUCTURE";

export interface TickerSnapshot {
  ticker: string;
  snapshotId: string;
  timestamp: string; // ISO 8601
  modelVer: string;
  setupScore: number; // 0 - 100
  domainConfidence: DomainConfidence;
  executionState: ExecutionState;
  marketRegime: MarketRegime;
  liquidityTier: LiquidityTier;
  spotPrice: number;
  entryZone: { low: number; high: number };
  stopLossFloor: number;
  takeProfit1: number;
  takeProfit2: number;
  flowZScore: number;
  validationTier: "THIN" | "DEVELOPING" | "ESTABLISHED";
  topDriverIds: string[];
}

export interface DeltaItem {
  field: string;
  category: MaterialityCategory;
  previousValue: string | number;
  currentValue: string | number;
  deltaDisplay: string;
  severity: MaterialitySeverity;
  reason: string;
}

export interface AttentionSignal {
  id: string;
  ticker: string;
  severity: "CRITICAL" | "MATERIAL";
  category: MaterialityCategory;
  headline: string;
  rationale: string;
  timestamp: string;
}

export interface DeltaReport {
  ticker: string;
  baselineSnapshotId: string;
  baselineTimestamp: string;
  latestSnapshotId: string;
  latestTimestamp: string;
  daysSinceBaseline: number;
  items: DeltaItem[];
  maxSeverity: MaterialitySeverity;
  isMaterial: boolean;
  headline: string;
  attentionSignal?: AttentionSignal;
}

export interface RollingSnapshotEntry {
  timestamp: string;
  setupScore: number;
  executionState: ExecutionState;
  spotPrice: number;
}

export interface TickerSnapshotRecord {
  ticker: string;
  baselineSnapshot: TickerSnapshot;
  latestSnapshot: TickerSnapshot;
  deltaSummary?: DeltaReport;
  acknowledgedAt: string | null;
  history?: RollingSnapshotEntry[]; // Max 30 entries
}
