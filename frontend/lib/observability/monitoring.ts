/**
 * ARX Provider-Independent Frontend Monitoring Wrapper (Subwave 1B).
 *
 * Enforces:
 * 1. Provider independence: application code calls this module, never Sentry directly.
 * 2. Static export compatibility: safe client-side initialization.
 * 3. No-provider mode: fails open gracefully when no monitoring DSN is configured.
 * 4. Automatic binding of ARX correlation and release context.
 */

import {
  FrontendMonitoringAdapter,
  SentryFrontendAdapter,
} from "./sentryAdapter";
import { CorrelationOperation } from "./correlation";

let activeAdapter: FrontendMonitoringAdapter | null = null;

export function getMonitoringAdapter(): FrontendMonitoringAdapter | null {
  return activeAdapter;
}

export function setMonitoringAdapter(adapter: FrontendMonitoringAdapter | null): void {
  activeAdapter = adapter;
}

export function isMonitoringEnabled(): boolean {
  return Boolean(activeAdapter && activeAdapter.isEnabled());
}

/**
 * Initializes frontend exception monitoring.
 * Uses SentryFrontendAdapter by default if DSN is configured.
 */
export function initFrontendMonitoring(config?: {
  dsn?: string;
  environment?: string;
  release?: string;
  adapter?: FrontendMonitoringAdapter;
}): boolean {
  const targetAdapter = config?.adapter || new SentryFrontendAdapter();
  const enabled = targetAdapter.init(config);
  activeAdapter = targetAdapter;
  return enabled;
}

/**
 * Captures a frontend exception.
 * Supports passing an active CorrelationOperation or custom context dictionary.
 * Fail-open: errors during capture never bubble up or halt application execution.
 */
export function captureException(
  error: unknown,
  contextOrOperation?: CorrelationOperation | Record<string, unknown>
): string | null {
  if (!activeAdapter || !activeAdapter.isEnabled()) {
    return null;
  }

  try {
    let context: Record<string, unknown> | undefined;

    if (
      contextOrOperation &&
      typeof contextOrOperation === "object" &&
      "correlationId" in contextOrOperation
    ) {
      context = {
        correlation_id: (contextOrOperation as CorrelationOperation).correlationId,
      };
    } else if (contextOrOperation && typeof contextOrOperation === "object") {
      context = contextOrOperation as Record<string, unknown>;
    }

    return activeAdapter.captureException(error, context);
  } catch {
    return null;
  }
}

/**
 * Captures an informational or warning message across the active monitoring adapter.
 */
export function captureMessage(
  message: string,
  level: "info" | "warning" | "error" = "info",
  context?: Record<string, unknown>
): string | null {
  if (!activeAdapter || !activeAdapter.isEnabled()) {
    return null;
  }

  try {
    return activeAdapter.captureMessage(message, level, context);
  } catch {
    return null;
  }
}
