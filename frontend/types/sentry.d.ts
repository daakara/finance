// Ambient type declaration for @sentry/browser in ARX Terminal frontend.
declare module "@sentry/browser" {
  export interface Scope {
    setTag(key: string, value: string): this;
    setExtra(key: string, value: unknown): this;
    setUser(user: { ip_address?: string; [key: string]: unknown } | null): this;
  }

  export interface Breadcrumb {
    message?: string;
    data?: Record<string, unknown>;
    category?: string;
    level?: string;
    type?: string;
    timestamp?: number;
  }

  export interface SentryEvent {
    request?: {
      url?: string;
      query_string?: string | null;
      headers?: Record<string, unknown>;
      data?: unknown;
    };
    user?: {
      ip_address?: string;
      [key: string]: unknown;
    };
    extra?: Record<string, unknown>;
    tags?: Record<string, string>;
    breadcrumbs?: Breadcrumb[];
    message?: string;
    [key: string]: unknown;
  }

  export interface BrowserOptions {
    dsn?: string;
    environment?: string;
    release?: string;
    sendDefaultPii?: boolean;
    tracesSampleRate?: number;
    beforeSend?: (event: SentryEvent, hint?: unknown) => SentryEvent | null;
    beforeBreadcrumb?: (breadcrumb: Breadcrumb, hint?: unknown) => Breadcrumb | null;
    [key: string]: unknown;
  }

  export function init(options?: BrowserOptions): void;
  export function captureException(error: unknown, captureContext?: unknown): string;
  export function captureMessage(message: string, captureContext?: unknown): string;
  export function withScope(callback: (scope: Scope) => void): void;
}
