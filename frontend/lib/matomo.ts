"use client";

/**
 * Matomo Privacy-First Analytics Engine for Single Page Applications (Next.js App Router).
 * Automatically tracks page views, user journey path transitions, and high-value quant interactions.
 */

declare global {
  interface Window {
    _paq?: Array<any>;
  }
}

/**
 * High-Value Conversion Funnel Goals in Matomo.
 * Mapped to Matomo Goal IDs for conversion rate and ROI tracking.
 */
export const MATOMO_GOALS = {
  ONBOARDING_COMPLETED: 1,      // User finished 4-slide institutional orientation
  PREFLIGHT_CLEARED: 2,         // User validated 5/5 trade sanity checks
  TRADE_PLAN_COPIED: 3,         // User copied Markdown execution plan for trade journal
  PORTFOLIO_POSITION_ADDED: 4,  // User logged trade into private browser storage
  PRICE_ALERT_CREATED: 5,       // User configured breakout pivot or pullback alert
  MACRO_STRESS_SIMULATED: 6,    // User ran beta-weighted market crash stress test
  STOCK_COMPARISON_RUN: 7,      // User executed head-to-head competitor factor analysis
  SCREENER_FILTER_APPLIED: 8,   // User filtered Peter Lynch/Magic Formula/GARP gems
} as const;

export type MatomoGoalId = typeof MATOMO_GOALS[keyof typeof MATOMO_GOALS];

/**
 * Custom Matomo Event Tracking Helper
 */
export function trackMatomoEvent(
  category: "User Journey" | "Terminal Interaction" | "Smart Money" | "Screener" | "Risk Engine" | "Decision Intelligence",
  action: string,
  name?: string,
  value?: number
) {
  if (typeof window !== "undefined") {
    window._paq = window._paq || [];
    window._paq.push(["trackEvent", category, action, name, value]);
    if (process.env.NODE_ENV === "development") {
      console.log(`[Matomo Event] Category: ${category} | Action: ${action} | Name: ${name || "-"}`);
    }
  }
}

/**
 * Matomo Goal Conversion Tracker
 */
export function trackMatomoGoal(goalId: MatomoGoalId | number, customRevenue?: number) {
  if (typeof window !== "undefined") {
    window._paq = window._paq || [];
    window._paq.push(["trackGoal", goalId, customRevenue]);
    if (process.env.NODE_ENV === "development") {
      console.log(`[Matomo Goal] Goal ID: ${goalId}${customRevenue ? ` | Revenue: $${customRevenue}` : ""}`);
    }
  }
}

/**
 * Track Onboarding Tour Completion (Goal 1)
 */
export function trackOnboardingCompleted(slideTitle?: string) {
  trackMatomoEvent("User Journey", "Complete Onboarding", slideTitle || "Orientation Tour");
  trackMatomoGoal(MATOMO_GOALS.ONBOARDING_COMPLETED);
}

/**
 * Track In-Terminal Workspace Domain Switches
 */
export function trackWorkspaceSwitch(workspace: string, symbol: string) {
  trackMatomoEvent("Terminal Interaction", "Switch Workspace Tab", `${symbol} -> ${workspace}`);
}

/**
 * Track Trading Horizon Role Changes (Day Trader vs Long Term Investor)
 */
export function trackRoleSwitch(role: "DAY_TRADER" | "LONG_TERM") {
  trackMatomoEvent("User Journey", "Change User Role", role);
}

/**
 * Track Asset Searches & Symbol Changes
 */
export function trackSymbolSearch(symbol: string, source: "OmniSearch" | "Watchlist" | "Compare" | "Screener" | "Chip") {
  trackMatomoEvent("Terminal Interaction", "Select Symbol", `${symbol} (via ${source})`);
}

/**
 * Track Grounded Provenance & Statutory Link Audits
 */
export function trackProvenanceInspection(symbol: string, source: string) {
  trackMatomoEvent("Smart Money", "Inspect Source Provenance", `${symbol} - ${source}`);
}

/**
 * Track Pre-Flight Trade Clearance Gate Outcomes (Goal 2 on Cleared)
 */
export function trackPreFlightOutcome(symbol: string, passedCount: number, isCleared: boolean) {
  trackMatomoEvent(
    "Decision Intelligence",
    isCleared ? "Pre-Flight Clearance Passed" : "Pre-Flight Clearance Conditional",
    `${symbol} (${passedCount}/5 Checks)`,
    passedCount
  );
  if (isCleared) {
    trackMatomoGoal(MATOMO_GOALS.PREFLIGHT_CLEARED);
  }
}

/**
 * Track Trade Plan Export / Clipboard Copy (Goal 3)
 */
export function trackTradePlanCopied(symbol: string, setupPattern: string) {
  trackMatomoEvent("Decision Intelligence", "Copy Trade Plan for Journal", `${symbol} (${setupPattern})`);
  trackMatomoGoal(MATOMO_GOALS.TRADE_PLAN_COPIED);
}

/**
 * Track Position Sizing Calculations & Portfolio Position Adds (Goal 4)
 */
export function trackPositionSizer(symbol: string, riskPct: number, shares: number) {
  trackMatomoEvent("Risk Engine", "Calculate Position Size", `${symbol} @ ${riskPct}% risk (${shares} shares)`, shares);
}

export function trackPortfolioPositionAdded(symbol: string, positionValue?: number) {
  trackMatomoEvent("User Journey", "Add Portfolio Position", symbol, positionValue ? Math.round(positionValue) : undefined);
  trackMatomoGoal(MATOMO_GOALS.PORTFOLIO_POSITION_ADDED, positionValue ? Math.round(positionValue) : undefined);
}

/**
 * Track Price Alerts & Breakout Pivot Triggers (Goal 5)
 */
export function trackAlertSet(symbol: string, targetPrice: number, isStage4: boolean) {
  trackMatomoEvent(
    "Decision Intelligence",
    isStage4 ? "Set Stage 4 Breakout Pivot Alert" : "Set Pullback Buy Zone Alert",
    `${symbol} @ $${targetPrice.toFixed(2)}`
  );
  trackMatomoGoal(MATOMO_GOALS.PRICE_ALERT_CREATED);
}

/**
 * Track Macro Stress Test Simulations (Goal 6)
 */
export function trackMacroShockSimulation(scenarioName: string, impactPct: number) {
  trackMatomoEvent("Risk Engine", "Run Macro Stress Shock", `${scenarioName} (${impactPct.toFixed(2)}% loss)`, Math.round(Math.abs(impactPct)));
  trackMatomoGoal(MATOMO_GOALS.MACRO_STRESS_SIMULATED, Math.round(Math.abs(impactPct)));
}

/**
 * Track Head-to-Head Asset Comparisons (Goal 7)
 */
export function trackComparisonRun(symbolA: string, symbolB: string) {
  trackMatomoEvent("Terminal Interaction", "Compare Assets", `${symbolA} vs ${symbolB}`);
  trackMatomoGoal(MATOMO_GOALS.STOCK_COMPARISON_RUN);
}

/**
 * Track Watchlist Favorite Starring / Unstarring
 */
export function trackFavoriteToggle(symbol: string, isFavorited: boolean) {
  trackMatomoEvent("User Journey", isFavorited ? "Star Favorite Asset" : "Unstar Favorite Asset", symbol);
}

/**
 * Track Dual-Vernacular Mode Switches (Plain English vs Pro Quant)
 */
export function trackVernacularSwitch(mode: "PLAIN_ENGLISH" | "PRO_QUANT") {
  trackMatomoEvent("User Journey", "Switch Vernacular Mode", mode);
}

/**
 * Track Screener Filter & Preset Selections (Goal 8)
 */
export function trackScreenerSelection(presetName: string, resultsCount: number) {
  trackMatomoEvent("Screener", "Apply Screener Preset", `${presetName} (${resultsCount} gems)`, resultsCount);
  trackMatomoGoal(MATOMO_GOALS.SCREENER_FILTER_APPLIED, resultsCount);
}

/**
 * One-Click Analytics Opt-Out & Forget for GDPR Article 21 Compliance
 */
export function toggleMatomoOptOut(optOut: boolean) {
  if (typeof window !== "undefined") {
    window._paq = window._paq || [];
    if (optOut) {
      window._paq.push(["optUserOut"]);
    } else {
      window._paq.push(["forgetUserOptOut"]);
    }
  }
}

/**
 * Check if the user is currently opted out of analytics
 */
export function isMatomoUserOptedOut(): boolean {
  if (typeof window !== "undefined" && window._paq) {
    let isOptedOut = false;
    window._paq.push([function(this: any) {
      isOptedOut = this.isUserOptedOut();
    }]);
    return isOptedOut;
  }
  return false;
}