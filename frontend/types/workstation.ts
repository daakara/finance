/**
 * ARX Terminal vNext - Workstation & Command Strip Type Definitions
 * Source of Truth: docs/api/API_CONTRACTS_VNEXT.md & docs/sprints/SPRINT_1_ENGINEERING_EXECUTION_PACKAGE.md
 */

import { ExperienceMode } from "./insight";

export type DomainConfidence = "HIGH" | "MODERATE" | "LIMITED";

export type ExecutionState =
  | "IN_BUY_ZONE"
  | "APPROACHING_TARGET"
  | "WAITING_PULLBACK"
  | "STOPPED_OUT"
  | "NEUTRAL";

export type LiquidityTier = "HIGH" | "MODERATE" | "RISK" | "UNKNOWN";

export type MarketRegime = "RISK_ON" | "NEUTRAL" | "DEFENSIVE";

export type MarketSession = "OPEN" | "CLOSED" | "PRE" | "POST";

export interface TickerIdentity {
  name: string;
  exchange: string;
  sector: string;
  industry?: string;
  marketCap?: number;
}

export interface TickerMarketData {
  spotPrice: number;
  change: number;
  changePct: number;
  volume20D?: number;
  marketSession?: MarketSession;
  settlementPinned?: boolean;
}

export interface Stage1Orientation {
  setupScore: number; // 0 - 100
  domainConfidence: DomainConfidence;
  executionState: ExecutionState;
  liquidityTier: LiquidityTier;
  amihudScore?: number;
  liquiditySummary?: string;
}

export interface EntryZone {
  low: number;
  high: number;
}

export interface VolatilityBands {
  atr: number;
  upperBand: number;
  lowerBand: number;
}

export interface Stage2Geometry {
  entryZone: EntryZone;
  stopLossFloor: number;
  takeProfit1: number;
  takeProfit2: number;
  riskRewardRatio: number;
  maxAdvShareLimit: number;
  volatility?: VolatilityBands;
}

export type ConvictionDimensionType =
  | "HEALTH"
  | "FLOW"
  | "REGIME"
  | "STRUCTURE"
  | "VALIDATION";

export type ConvictionStatus = "FAVORABLE" | "CAUTION" | "UNFAVORABLE" | "NEUTRAL";

export interface ConvictionItem {
  dimension: ConvictionDimensionType;
  status: ConvictionStatus;
  label: string;
  value: string | number;
  summary: string;
  reasons?: string[];
  provenanceSource: string;
  methodologyNote?: string;
  metrics?: { label: string; value: string | number; status?: ConvictionStatus }[];
}

export interface FactorAttribution {
  factorId: string;
  name: string;
  category: "TECHNICAL" | "FUNDAMENTAL" | "MACRO" | "FLOW" | "SENTIMENT" | "VALIDATION" | "STRUCTURE";
  rawSignal: number | string;
  weight: number; // percentage, e.g. 25
  contribution: number; // net point score impact, e.g. +18
  direction: "BULLISH" | "BEARISH" | "NEUTRAL";
  provenance: string;
}

export interface Stage4Driver {
  id: string;
  category: string;
  direction: "BULLISH" | "BEARISH" | "NEUTRAL";
  headline: string;
  detail: string;
  contributionPoints?: number;
}

export interface Stage4Explanation {
  confluenceScore: number;
  headline?: string;
  drivers: Stage4Driver[];
  factors?: FactorAttribution[];
  modelVer?: string;
  decisionHash?: string;
}

export interface WorkstationPayload {
  ticker: string;
  identity: TickerIdentity;
  marketData: TickerMarketData;
  stage1_orientation: Stage1Orientation;
  stage2_geometry: Stage2Geometry;
  stage3_conviction: ConvictionItem[];
  stage4_explanation: Stage4Explanation;
}

export interface TickerCommandStripProps {
  ticker: string;
  companyName: string;
  spotPrice: number;
  priceChangePct: number;
  priceChange?: number;
  setupScore: number; // 0 - 100
  domainConfidence: DomainConfidence;
  executionState: ExecutionState;
  liquidityTier: LiquidityTier;
  marketRegime: MarketRegime;
  isSettlementPinned?: boolean;
  marketSession?: MarketSession;
  sector?: string;
  exchange?: string;
  amihudScore?: number;
  mode?: ExperienceMode;
  className?: string;
}
