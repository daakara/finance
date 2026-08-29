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
 * Track Pre-Flight Trade Clearance Gate Outcomes
 */
export function trackPreFlightOutcome(symbol: string, passedCount: number, isCleared: boolean) {
  trackMatomoEvent(
    "Decision Intelligence",
    isCleared ? "Pre-Flight Clearance Passed" : "Pre-Flight Clearance Conditional",
    `${symbol} (${passedCount}/5 Checks)`,
    passedCount
  );
}

/**
 * Track Trade Plan Export / Clipboard Copy
 */
export function trackTradePlanCopied(symbol: string, setupPattern: string) {
  trackMatomoEvent("Decision Intelligence", "Copy Trade Plan for Journal", `${symbol} (${setupPattern})`);
}

/**
 * Track Position Sizing Calculations
 */
export function trackPositionSizer(symbol: string, riskPct: number, shares: number) {
  trackMatomoEvent("Risk Engine", "Calculate Position Size", `${symbol} @ ${riskPct}% risk (${shares} shares)`, shares);
}

/**
 * Track Price Alerts & Breakout Pivot Triggers
 */
export function trackAlertSet(symbol: string, targetPrice: number, isStage4: boolean) {
  trackMatomoEvent(
    "Decision Intelligence",
    isStage4 ? "Set Stage 4 Breakout Pivot Alert" : "Set Pullback Buy Zone Alert",
    `${symbol} @ $${targetPrice.toFixed(2)}`
  );
}

/**
 * Track Macro Stress Test Simulations
 */
export function trackMacroShockSimulation(scenarioName: string, impactPct: number) {
  trackMatomoEvent("Risk Engine", "Run Macro Stress Shock", `${scenarioName} (${impactPct.toFixed(2)}% loss)`, Math.round(Math.abs(impactPct)));
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
 * Track Screener Filter & Preset Selections
 */
export function trackScreenerSelection(presetName: string, resultsCount: number) {
  trackMatomoEvent("Screener", "Apply Screener Preset", `${presetName} (${resultsCount} gems)`, resultsCount);
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