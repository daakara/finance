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
  category: "User Journey" | "Terminal Interaction" | "Smart Money" | "Screener" | "Risk Engine",
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
export function trackSymbolSearch(symbol: string, source: "OmniSearch" | "Watchlist" | "Compare") {
  trackMatomoEvent("Terminal Interaction", "Select Symbol", `${symbol} (via ${source})`);
}

/**
 * Track Grounded Provenance & Statutory Link Audits
 */
export function trackProvenanceInspection(symbol: string, source: string) {
  trackMatomoEvent("Smart Money", "Inspect Source Provenance", `${symbol} - ${source}`);
}