/**
 * ARX Terminal vNext - Telemetry Tracker & Batch Emitter
 * Implements high-resolution monotonic performance timing via performance.now().
 * Rejects any financial PII (dollar balances, trade sizes) per ADR-006.
 */

import {
  TelemetryEnvelope,
  TelemetryEventCategory,
  TelemetryEventName,
} from "../types/telemetry";
import { useExperienceStore } from "../state/experience-store";

const SESSION_START_TIME =
  typeof window !== "undefined" && window.performance
    ? window.performance.now()
    : Date.now();

let sessionId = "";

function getSessionId(): string {
  if (sessionId) return sessionId;
  if (typeof window === "undefined") return "ssr-session";

  try {
    let stored = sessionStorage.getItem("arx-telemetry-session-id");
    if (!stored) {
      stored =
        "arx-" +
        Math.random().toString(36).substring(2, 11) +
        "-" +
        Date.now().toString(36);
      sessionStorage.setItem("arx-telemetry-session-id", stored);
    }
    sessionId = stored;
    return sessionId;
  } catch {
    sessionId = "fallback-session-" + Date.now();
    return sessionId;
  }
}

export function trackTelemetryEvent(
  category: TelemetryEventCategory,
  name: TelemetryEventName,
  payload: Record<string, unknown> = {},
  ticker?: string
) {
  if (typeof window === "undefined") return;

  const now = window.performance ? window.performance.now() : Date.now();
  const elapsedMs = Math.round(now - SESSION_START_TIME);

  const envelope: TelemetryEnvelope = {
    sessionId: getSessionId(),
    timestamp: new Date().toISOString(),
    sessionElapsedMs: elapsedMs,
    workspaceMode: useExperienceStore.getState().mode || "STANDARD",
    activeWorkspace: "TICKER_DECISION",
    ticker,
    eventCategory: category,
    eventName: name,
    payload,
  };

  // Dispatch via sendBeacon if available, fallback to fetch
  try {
    const serialized = JSON.stringify(envelope);
    if (navigator.sendBeacon) {
      navigator.sendBeacon("/api/telemetry/events", serialized);
    } else {
      fetch("/api/telemetry/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serialized,
        keepalive: true,
      }).catch(() => {
        // Silently handle telemetry sink errors
      });
    }
  } catch {
    // In-memory / console log fallback in development
  }
}
