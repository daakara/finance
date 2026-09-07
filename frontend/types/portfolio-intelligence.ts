/**
 * ARX Terminal vNext - Portfolio Intelligence & Attention Aggregation Contracts
 * Source: docs/sprints/SPRINT_4_PORTFOLIO_INTELLIGENCE_PACKAGE.md
 */

import { MaterialityCategory, MaterialitySeverity } from "./change-intelligence";
import { ExecutionState } from "./workstation";

export type DataQualityStatus = "VALID" | "DEGRADED" | "UNTRUSTED" | "REJECTED";

export interface ReliabilityMetadata {
  ingestionId: string;
  sequenceNumber: number;
  retryCount: number;
  sampled: boolean;
  qualityStatus: DataQualityStatus;
}

export interface DataQualityResult {
  status: DataQualityStatus;
  passedChecks: string[];
  failedChecks: string[];
  qualityScore: number;
  timestamp: string;
  rejectionReason?: string;
}

export type PortfolioAttentionCategory =
  | "EXECUTION"
  | "REGIME"
  | "FLOW"
  | "VALIDATION"
  | "SETUP";

export interface PortfolioAttentionEntry {
  ticker: string;
  severity: "INFO" | "MATERIAL" | "CRITICAL";
  category: PortfolioAttentionCategory;
  headline: string;
  generatedAt: string;
  sourceDeltaId: string;
  quality: "TRUSTED" | "DEGRADED";
  itemCount: number;
  executionState?: ExecutionState;
  previousState?: ExecutionState;
  spotPrice?: number;
  scoreChange?: number;
}

export interface PortfolioAttentionFeed {
  criticalItems: PortfolioAttentionEntry[];
  materialItems: PortfolioAttentionEntry[];
  summaryCount: number;
  totalAttentionCount: number;
  generatedAt: string;
  quarantinedCount?: number;
}

export interface MorningBriefingSummary {
  headline: string;
  criticalCount: number;
  materialCount: number;
  infoCount: number;
  totalAssetsReviewed: number;
  primaryRiskRegime: string;
  generatedAt: string;
  topActionTickers: string[];
}
