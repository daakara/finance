/**
 * ARX Terminal Prototype Sandbox — Network Isolation Guard
 *
 * Enforces fail-closed network isolation within the /prototype research environment.
 * Blocks all outbound HTTP/WebSocket requests outside the approved local fixture allowlist.
 */

const ALLOWED_PATH_PREFIXES = [
  "/fixtures/scenarios/",
  "/prototype",
  "/_next/",
  "/favicon",
  "/icon",
  "/api/research/custody/",
];

export function installPrototypeNetworkGuard(): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }

  // 1. Guard window.fetch
  const originalFetch = window.fetch;
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const urlString =
      typeof input === "string"
        ? input
        : input instanceof URL
        ? input.toString()
        : input instanceof Request
        ? input.url
        : "";

    // Parse relative or absolute URL
    try {
      const parsedUrl = new URL(urlString, window.location.origin);
      const pathname = parsedUrl.pathname;
      const isAllowedOrigin = parsedUrl.origin === window.location.origin;

      if (!isAllowedOrigin) {
        const errorMsg = `[SECURITY_VIOLATION] Prototype sandbox blocked outbound external network request: ${urlString}`;
        console.error(errorMsg);
        throw new Error(errorMsg);
      }

      const isAllowedPath = ALLOWED_PATH_PREFIXES.some((prefix) => pathname.startsWith(prefix));
      if (!isAllowedPath) {
        // Return fail-closed 404 response for unauthorized local routes (e.g. Next.js link prefetching)
        // to prevent unhandled promise rejections that force browser full-page reload
        return new Response(JSON.stringify({ error: "BLOCKED_BY_PROTOTYPE_SANDBOX", path: pathname }), {
          status: 404,
          statusText: "Blocked by Prototype Sandbox",
          headers: { "Content-Type": "application/json" },
        });
      }
    } catch (e: any) {
      if (e.message && e.message.startsWith("[SECURITY_VIOLATION]")) {
        throw e;
      }
      throw new Error(`[SECURITY_VIOLATION] Invalid or blocked request URL: ${urlString}`);
    }

    return originalFetch(input, init);
  };

  // 2. Guard XMLHttpRequest
  const OriginalXHR = window.XMLHttpRequest;
  class GuardedXHR extends OriginalXHR {
    open(method: string, url: string | URL, ...rest: any[]) {
      const urlString = url.toString();
      try {
        const parsedUrl = new URL(urlString, window.location.origin);
        const isAllowedOrigin = parsedUrl.origin === window.location.origin;
        const isAllowedPath = ALLOWED_PATH_PREFIXES.some((prefix) => parsedUrl.pathname.startsWith(prefix));

        if (!isAllowedOrigin || !isAllowedPath) {
          throw new Error(`[SECURITY_VIOLATION] Prototype sandbox blocked XHR request: ${urlString}`);
        }
      } catch (err) {
        console.error(err);
        throw err;
      }
      return (super.open as any)(method, url, ...rest);
    }
  }
  (window as any).XMLHttpRequest = GuardedXHR;

  // 3. Guard WebSocket
  const OriginalWebSocket = window.WebSocket;
  const BlockedWebSocket = function (url: string | URL, protocols?: string | string[]) {
    const errorMsg = `[SECURITY_VIOLATION] Prototype sandbox strictly prohibits WebSocket connections: ${url}`;
    console.error(errorMsg);
    throw new Error(errorMsg);
  } as unknown as typeof WebSocket;
  (window as any).WebSocket = BlockedWebSocket;

  // 4. Guard navigator.sendBeacon
  const originalSendBeacon = typeof navigator !== "undefined" && navigator.sendBeacon ? navigator.sendBeacon.bind(navigator) : null;
  if (originalSendBeacon) {
    navigator.sendBeacon = (url: string | URL, data?: BodyInit | null): boolean => {
      const urlString = url.toString();
      try {
        const parsedUrl = new URL(urlString, window.location.origin);
        const isAllowedOrigin = parsedUrl.origin === window.location.origin;
        const isAllowedPath = ALLOWED_PATH_PREFIXES.some((prefix) => parsedUrl.pathname.startsWith(prefix));
        if (!isAllowedOrigin || !isAllowedPath) {
          console.warn(`[SECURITY_VIOLATION] Prototype sandbox blocked beacon: ${urlString}`);
          return false;
        }
      } catch {
        return false;
      }
      return originalSendBeacon(url, data);
    };
  }

  // Teardown / cleanup function
  return () => {
    window.fetch = originalFetch;
    window.XMLHttpRequest = OriginalXHR;
    window.WebSocket = OriginalWebSocket;
    if (originalSendBeacon) {
      navigator.sendBeacon = originalSendBeacon;
    }
  };
}
