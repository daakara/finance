/**
 * Sentry Frontend Monitoring Adapter for ARX Observability (Subwave 1B).
 *
 * Enforces:
 * 1. Client-side browser compatibility (zero Node-specific SSR runtime assumptions).
 * 2. Static export compatibility (Next.js output: 'export').
 * 3. Safe initialization: fails open to no-op when NEXT_PUBLIC_SENTRY_DSN is absent.
 * 4. Strict privacy & data minimization:
 *    - sendDefaultPii = false
 *    - Deep recursive redaction of auth headers, cookies, API keys, and portfolio balances.
 *    - Query parameter scrubbing.
 * 5. ARX correlation binding: binds request_id, correlation_id, and release SHA.
 * 6. Telemetry failure safety: telemetry errors never crash frontend rendering.
 */

import * as Sentry from "@sentry/browser";
import { isValidUUIDv4 } from "./correlation";

export const SENSITIVE_FIELDS = new Set([
  "authorization",
  "cookie",
  "set-cookie",
  "x-api-key",
  "api_key",
  "apikey",
  "token",
  "access_token",
  "refresh_token",
  "secret",
  "password",
  "portfolio",
  "portfolio_value",
  "balance",
  "holdings",
  "cash",
  "account_id",
]);

const SENSITIVE_SUBSTRINGS = [
  "auth",
  "secret",
  "token",
  "password",
  "cookie",
  "portfolio",
  "balance",
  "api_key",
  "apikey",
  "private_key",
  "secret_key",
];

/**
 * Sanitizes URLs and query strings to remove sensitive parameters.
 */
export function sanitizeUrlOrQuery(urlStr: string): string {
  if (!urlStr || typeof urlStr !== "string") return "";
  try {
    const isAbsolute = urlStr.startsWith("http://") || urlStr.startsWith("https://");
    const dummyBase = "https://arxterminal.internal";
    const parsed = new URL(urlStr, isAbsolute ? undefined : dummyBase);

    const params = new URLSearchParams(parsed.search);
    let mutated = false;
    for (const [k] of Array.from(params.entries())) {
      const kLower = k.toLowerCase();
      if (SENSITIVE_FIELDS.has(kLower) || SENSITIVE_SUBSTRINGS.some((s) => kLower.includes(s))) {
        params.set(k, "[REDACTED]");
        mutated = true;
      }
    }

    if (mutated) {
      parsed.search = params.toString();
    }

    if (isAbsolute) {
      return parsed.toString();
    }
    if (urlStr.startsWith("/")) {
      return parsed.pathname + parsed.search;
    }
    return parsed.search;
  } catch {
    return "[REDACTED_URL]";
  }
}

/**
 * Recursively scrubs sensitive keys and values from nested objects, arrays, and primitives.
 */
export function recursiveSanitize<T>(data: T, maxDepth: number = 10): T {
  if (maxDepth <= 0 || data === null || data === undefined) {
    return data;
  }

  if (typeof data === "string") {
    if (data.includes("?") || data.includes("=")) {
      return sanitizeUrlOrQuery(data) as unknown as T;
    }
    return data;
  }

  if (Array.isArray(data)) {
    return data.map((item) => recursiveSanitize(item, maxDepth - 1)) as unknown as T;
  }

  if (typeof data === "object") {
    const cleaned: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(data as Record<string, unknown>)) {
      const kLower = k.toLowerCase();
      if (SENSITIVE_FIELDS.has(kLower) || SENSITIVE_SUBSTRINGS.some((s) => kLower.includes(s))) {
        cleaned[k] = "[REDACTED]";
      } else {
        cleaned[k] = recursiveSanitize(v, maxDepth - 1);
      }
    }
    return cleaned as T;
  }

  return data;
}

export interface FrontendMonitoringAdapter {
  readonly providerName: string;
  init(config?: { dsn?: string; environment?: string; release?: string }): boolean;
  captureException(error: unknown, context?: Record<string, unknown>): string | null;
  captureMessage(message: string, level?: "info" | "warning" | "error", context?: Record<string, unknown>): string | null;
  isEnabled(): boolean;
}

export class SentryFrontendAdapter implements FrontendMonitoringAdapter {
  readonly providerName = "sentry";
  private _enabled: boolean = false;

  isEnabled(): boolean {
    return this._enabled;
  }

  init(config?: { dsn?: string; environment?: string; release?: string; isTest?: boolean }): boolean {
    // Fail-safe: only initialize in browser context (or explicit test mode)
    if (typeof window === "undefined" && !config?.isTest) {
      this._enabled = false;
      return false;
    }

    const rawDsn = config?.dsn || process.env.NEXT_PUBLIC_SENTRY_DSN || "";
    const dsn = rawDsn.trim();

    if (!dsn) {
      this._enabled = false;
      return false;
    }

    try {
      const release = config?.release || process.env.NEXT_PUBLIC_ARX_RELEASE || "";
      const environment =
        config?.environment || process.env.NEXT_PUBLIC_ENVIRONMENT || process.env.ENVIRONMENT || "production";

      Sentry.init({
        dsn,
        environment,
        release: release || undefined,
        sendDefaultPii: false,
        tracesSampleRate: 0.0, // Error monitoring only; APM traces quarantined
        beforeSend: (event: Sentry.SentryEvent) => {
          try {
            // 1. Sanitize request
            if (event.request) {
              if (event.request.headers) {
                event.request.headers = recursiveSanitize(event.request.headers);
              }
              if (event.request.url) {
                event.request.url = sanitizeUrlOrQuery(event.request.url);
              }
              if (event.request.query_string) {
                event.request.query_string = sanitizeUrlOrQuery(String(event.request.query_string));
              }
              if (event.request.data) {
                event.request.data = "[BODY_REDACTED]";
              }
            }

            // 2. Sanitize user (zero PII)
            if (event.user) {
              event.user = { ip_address: "[REDACTED]" };
            }

            // 3. Sanitize extra & tags
            if (event.extra) {
              event.extra = recursiveSanitize(event.extra);
            }
            if (event.tags) {
              event.tags = recursiveSanitize(event.tags);
            }

            // 4. Sanitize breadcrumbs
            if (event.breadcrumbs) {
              for (const b of event.breadcrumbs) {
                if (b.data) {
                  b.data = recursiveSanitize(b.data);
                }
                if (b.message) {
                  b.message = sanitizeUrlOrQuery(b.message);
                }
              }
            }

            return event;
          } catch {
            return null; // Fail-safe: drop event rather than leak
          }
        },
        beforeBreadcrumb: (breadcrumb: Sentry.Breadcrumb) => {
          try {
            if (breadcrumb.data) {
              breadcrumb.data = recursiveSanitize(breadcrumb.data);
            }
            if (breadcrumb.message) {
              breadcrumb.message = sanitizeUrlOrQuery(breadcrumb.message);
            }
            return breadcrumb;
          } catch {
            return null;
          }
        },
      });

      this._enabled = true;
      return true;
    } catch {
      this._enabled = false;
      return false;
    }
  }

  captureException(error: unknown, context?: Record<string, unknown>): string | null {
    if (!this._enabled) {
      return null;
    }

    try {
      let eventId: string | null = null;
      Sentry.withScope((scope: Sentry.Scope) => {
        scope.setTag("service", "arx-frontend");
        scope.setTag(
          "environment",
          process.env.NEXT_PUBLIC_ENVIRONMENT || process.env.ENVIRONMENT || "production"
        );

        const clientRelease = process.env.NEXT_PUBLIC_ARX_RELEASE;
        if (clientRelease) {
          scope.setTag("frontend_release_sha", clientRelease);
        }

        if (context) {
          const reqId = context.request_id || context.requestId;
          if (typeof reqId === "string" && isValidUUIDv4(reqId)) {
            scope.setTag("request_id", reqId);
          }

          const corrId = context.correlation_id || context.correlationId;
          if (typeof corrId === "string" && isValidUUIDv4(corrId)) {
            scope.setTag("correlation_id", corrId);
          }

          if (typeof context.route === "string") {
            scope.setTag("route", sanitizeUrlOrQuery(context.route));
          }

          const sanitized = recursiveSanitize(context);
          for (const [k, v] of Object.entries(sanitized)) {
            if (!["request_id", "requestId", "correlation_id", "correlationId"].includes(k)) {
              scope.setExtra(k, v);
            }
          }
        }

        eventId = Sentry.captureException(error);
      });
      return eventId;
    } catch {
      return null;
    }
  }

  captureMessage(
    message: string,
    level: "info" | "warning" | "error" = "info",
    context?: Record<string, unknown>
  ): string | null {
    if (!this._enabled) {
      return null;
    }

    try {
      let eventId: string | null = null;
      Sentry.withScope((scope: Sentry.Scope) => {
        scope.setTag("service", "arx-frontend");
        if (context) {
          const sanitized = recursiveSanitize(context);
          for (const [k, v] of Object.entries(sanitized)) {
            scope.setExtra(k, v);
          }
        }
        eventId = Sentry.captureMessage(sanitizeUrlOrQuery(message), level);
      });
      return eventId;
    } catch {
      return null;
    }
  }
}
