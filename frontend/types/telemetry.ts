/**
 * ARX Terminal vNext - Telemetry Type Definitions
 * Source: docs/api/API_CONTRACTS_VNEXT.md & Sprint 1 W1.8 Specification
 */

import { ExperienceMode } from "./insight";

export type TelemetryEventCategory =
  | "ORIENTATION"
  | "DECISION"
  | "NAVIGATION"
  | "SYSTEM"
  | "PERFORMANCE";

export type TelemetryEventName =
  | "workspace_loaded"
  | "experience_mode_loaded"
  | "experience_mode_changed"
  | "navigation_category_clicked"
  | "market_ribbon_viewed"
  | "market_regime_viewed"
  | "watchlist_drawer_opened"
  | "watchlist_drawer_closed"
  | "command_strip_viewed"
  | "chart_viewed"
  | "execution_corridor_viewed"
  | "position_sizer_opened"
  | "ttc_started"
  | "ttc_completed"
  | "ttfmi_completed"
  | "conviction_matrix_viewed"
  | "conviction_popover_opened"
  | "conviction_popover_closed"
  | "why_arx_card_viewed"
  | "confluence_trace_opened"
  | "confluence_trace_closed"
  | "accordion_section_toggled"
  | "institutional_tooltip_viewed"
  | "returning_user_detected"
  | "delta_generated"
  | "delta_banner_displayed"
  | "delta_banner_expanded"
  | "delta_acknowledged"
  | "attention_item_opened"
  | "portfolio_feed_loaded"
  | "portfolio_item_opened"
  | "portfolio_item_acknowledged"
  | "morning_brief_viewed"
  | "critical_item_reviewed"
  | "telemetry_burst_detected"
  | "due_diligence_viewed"
  | "due_diligence_exported"
  | "thesis_confirmed"
  | "reread_session_started"
  | "material_change_presented"
  | "reread_session_completed"
  | "materiality_decision"
  | "data_quality_trusted"
  | "data_quality_degraded"
  | "data_quality_stale"
  | "data_quality_invalid"
  | "stale_snapshot_suppressed"
  | "duplicate_delta_suppressed"
  | "noise_change_suppressed";

export interface AnalyticsEvent {
  eventId?: string;
  eventName: TelemetryEventName | string;
  sessionId?: string;
  userIdHash?: string;
  timestamp?: string;
  ticker?: string;
  experienceMode?: ExperienceMode;
  viewport?: "DESKTOP" | "TABLET" | "MOBILE";
  version?: string;
  metadata?: Record<string, unknown>;
  reliability?: import("./portfolio-intelligence").ReliabilityMetadata;
}

export interface TelemetryEnvelope {
  sessionId: string;
  timestamp: string; // ISO 8601
  sessionElapsedMs: number;
  workspaceMode: ExperienceMode;
  activeWorkspace: string;
  ticker?: string;
  eventCategory: TelemetryEventCategory;
  eventName: TelemetryEventName;
  payload?: Record<string, unknown>;
}

export interface WatchlistOpenedPayload {
  source: "keyboard" | "mouse" | "touch";
}

export interface WatchlistClosedPayload {
  durationMs: number;
}
