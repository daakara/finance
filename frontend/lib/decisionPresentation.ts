/**
 * ARX Universal Shared Presentation Authority (Synthesis E Wave 4).
 *
 * Single, unified formatting authority for presentation badges, labels,
 * and visual styling across Radar, Terminal Cockpit, and Execution Cards.
 *
 * STRICT INVARIANTS:
 * 1. CONSUMER ONLY: Pure formatting resolver. Never re-evaluates quantitative models,
 *    never infers decisions independently, and never overrides canonical backend authority.
 * 2. QA-ESC-011 PRESERVED: Actionability domain is strictly ACTIONABLE vs NOT ACTIONABLE.
 *    Non-actionable states (AVOID, HOLD, UNVERIFIED, WAITING_PULLBACK) NEVER collapse into
 *    "WAIT FOR TRIGGER".
 * 3. ZERO HEADLINE/BADGE DUPLICATION: Headline verdict and presentation badge are decoupled.
 * 4. FAIL-CLOSED FALLBACK: Missing decision state defaults to "Setup Evaluation Pending" (Neutral),
 *    NEVER "Wait for Trigger".
 */

import { DecisionReadinessResult } from "./decisionReadiness";

export interface PresentationBadgeStyle {
  bg: string;
  text: string;
  border: string;
  icon: string;
}

export interface FormattedPresentationState {
  /** Canonical display badge text (e.g., "[ACTIONABLE SETUP]", "[AWAITING PULLBACK]") */
  badgeLabel: string;
  badgeStyle: PresentationBadgeStyle;
  /** Primary analytical verdict headline */
  headlineLabel: string;
  /** Binary actionability status badge (QA-ESC-011: ACTIONABLE vs NOT ACTIONABLE) */
  actionabilityLabel: "ACTIONABLE" | "NOT ACTIONABLE";
  actionabilityStyle: {
    bg: string;
    text: string;
    border: string;
  };
  /** Short diagnostic explanation of current state */
  summaryExplanation: string;
}

export interface PresentationResolverParams {
  decisionState?: string | null;
  decisionStateLabel?: string | null;
  executionStatus?: string | null;
  isActionable?: boolean | null;
  disqualificationReason?: string | null;
  readinessResult?: DecisionReadinessResult | null;
}

export function resolvePresentationState(params: PresentationResolverParams): FormattedPresentationState {
  const {
    decisionState,
    decisionStateLabel,
    executionStatus,
    isActionable = false,
    disqualificationReason,
    readinessResult,
  } = params;

  const actionable = Boolean(isActionable);
  const statusUpper = (executionStatus || "").toUpperCase().trim();
  const stateUpper = (decisionState || "").toUpperCase().trim();

  // 1. Binary Actionability Styling (Strict QA-ESC-011 Contract)
  const actionabilityLabel: "ACTIONABLE" | "NOT ACTIONABLE" = actionable
    ? "ACTIONABLE"
    : "NOT ACTIONABLE";

  const actionabilityStyle = actionable
    ? {
        bg: "bg-emerald-950/40",
        text: "text-emerald-400",
        border: "border-emerald-500/50",
      }
    : {
        bg: "bg-slate-900/60",
        text: "text-slate-400",
        border: "border-slate-700/60",
      };

  // 2. Headline Resolution (Fail-Closed to "Setup Evaluation Pending")
  let headlineLabel =
    decisionStateLabel ||
    (stateUpper === "ACTIONABLE_SETUP"
      ? "Actionable Setup — Buy Zone Confirmed"
      : stateUpper === "VALID_SETUP"
      ? "Valid Setup — Awaiting Trigger"
      : stateUpper === "EVIDENCE_INCOMPLETE"
      ? "Evidence Incomplete"
      : stateUpper === "INSUFFICIENT_DATA"
      ? "Insufficient History"
      : stateUpper === "STALE_DATA"
      ? "Stale Market Data"
      : stateUpper === "UNVERIFIED"
      ? "Unverified Asset"
      : "Setup Evaluation Pending");

  // 3. Presentation Badge & Styling Resolution
  let badgeLabel = "SETUP EVALUATION PENDING";
  let badgeStyle: PresentationBadgeStyle = {
    bg: "bg-slate-900/60",
    text: "text-slate-400",
    border: "border-slate-700/50",
    icon: "—",
  };
  let summaryExplanation =
    disqualificationReason ||
    readinessResult?.nextRequiredCondition ||
    "Setup evaluation is pending verification.";

  if (actionable || stateUpper === "ACTIONABLE_SETUP") {
    badgeLabel = "ACTIONABLE SETUP";
    badgeStyle = {
      bg: "bg-emerald-950/40",
      text: "text-emerald-400",
      border: "border-emerald-500/50",
      icon: "✓",
    };
    summaryExplanation = "All execution criteria satisfied. In accumulation corridor with verified trigger and favorable asymmetry.";
  } else if (stateUpper === "EVIDENCE_INCOMPLETE") {
    badgeLabel = "EVIDENCE INCOMPLETE";
    badgeStyle = {
      bg: "bg-slate-900/60",
      text: "text-slate-400",
      border: "border-slate-700/50",
      icon: "📋",
    };
    summaryExplanation = disqualificationReason || "Statutory regulatory filing incomplete or pending verification.";
  } else if (stateUpper === "INSUFFICIENT_DATA" || statusUpper === "INSUFFICIENT_HISTORY") {
    badgeLabel = "INSUFFICIENT HISTORY";
    badgeStyle = {
      bg: "bg-slate-900/60",
      text: "text-slate-400",
      border: "border-slate-700/50",
      icon: "⏳",
    };
    summaryExplanation = disqualificationReason || "Minimum 50 daily trading sessions required for trend validation.";
  } else if (stateUpper === "STALE_DATA" || statusUpper === "STALE_MARKET_DATA") {
    badgeLabel = "STALE TAPE";
    badgeStyle = {
      bg: "bg-amber-950/30",
      text: "text-amber-400",
      border: "border-amber-700/40",
      icon: "⚠️",
    };
    summaryExplanation = "Market data tape is older than freshness limits. Live trading triggers suspended.";
  } else if (stateUpper === "UNVERIFIED" || statusUpper === "UNVERIFIED_ASSET") {
    badgeLabel = "UNVERIFIED ASSET";
    badgeStyle = {
      bg: "bg-slate-900/60",
      text: "text-slate-400",
      border: "border-slate-700/50",
      icon: "🛑",
    };
    summaryExplanation = "Security listing identity unverified. Capital allocation prohibited.";
  } else if (statusUpper === "APPROACHING_TARGET") {
    badgeLabel = "APPROACHING TARGET";
    badgeStyle = {
      bg: "bg-purple-950/30",
      text: "text-purple-400",
      border: "border-purple-700/40",
      icon: "🎯",
    };
    summaryExplanation = "Price is nearing prospective profit targets. Review ratchet rule.";
  } else if (statusUpper === "STOPPED_OUT") {
    badgeLabel = "STOPPED OUT";
    badgeStyle = {
      bg: "bg-rose-950/30",
      text: "text-rose-400",
      border: "border-rose-700/40",
      icon: "🛑",
    };
    summaryExplanation = "Price breached invalidation floor. Capital protection active.";
  } else if (
    readinessResult?.activeBlockingGate === "GATE_1_LOCATION" &&
    readinessResult.gates[0].explanation.toLowerCase().includes("stage 4")
  ) {
    badgeLabel = "STAGE 4 DEFENSE";
    badgeStyle = {
      bg: "bg-rose-950/30",
      text: "text-rose-400",
      border: "border-rose-700/40",
      icon: "🛡️",
    };
    summaryExplanation = "Asset in Stage 4 distribution below 50-day SMA. Awaiting structural base formation.";
  } else if (
    statusUpper === "WAITING_PULLBACK" ||
    (readinessResult?.activeBlockingGate === "GATE_1_LOCATION" &&
      readinessResult.gates[0].explanation.toLowerCase().includes("extended"))
  ) {
    badgeLabel = "AWAITING PULLBACK";
    badgeStyle = {
      bg: "bg-amber-950/30",
      text: "text-amber-400",
      border: "border-amber-700/40",
      icon: "⏳",
    };
    summaryExplanation = readinessResult?.gates[0].explanation || "Price extended above accumulation corridor. Awaiting pullback.";
  } else if (
    statusUpper === "IN_BUY_ZONE_AWAITING_TRIGGER" ||
    readinessResult?.activeBlockingGate === "GATE_2_TRIGGER"
  ) {
    badgeLabel = "AWAITING TRIGGER";
    badgeStyle = {
      bg: "bg-cyan-950/30",
      text: "text-cyan-400",
      border: "border-cyan-700/40",
      icon: "⚡",
    };
    summaryExplanation = readinessResult?.gates[1].explanation || "Price inside accumulation corridor. Awaiting volume breakout confirmation.";
  } else if (readinessResult?.activeBlockingGate === "GATE_3_RISK_CLEARANCE") {
    badgeLabel = "RISK FLOOR UNMET";
    badgeStyle = {
      bg: "bg-amber-950/30",
      text: "text-amber-400",
      border: "border-amber-700/40",
      icon: "⚖️",
    };
    summaryExplanation = readinessResult?.gates[2].explanation || "Prospective Risk/Reward below 2:1 or macro volatility elevated.";
  } else if (stateUpper === "VALID_SETUP") {
    badgeLabel = "VALID SETUP";
    badgeStyle = {
      bg: "bg-cyan-950/30",
      text: "text-cyan-400",
      border: "border-cyan-700/40",
      icon: "⏳",
    };
    summaryExplanation = "Valid structural setup present; awaiting execution prerequisites.";
  }

  return {
    badgeLabel,
    badgeStyle,
    headlineLabel,
    actionabilityLabel,
    actionabilityStyle,
    summaryExplanation,
  };
}
