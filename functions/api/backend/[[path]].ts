// Cloudflare Pages Function: /api/backend/[[path]]
// Enables true cross-origin Edge Proxying on Cloudflare Pages to the Railway API origin.

const ALLOWED_ORIGINS = new Set([
  "https://www.arxterminal.com",
  "https://arxterminal.com",
  "https://finance-xp8.pages.dev",
  "http://localhost:3000",
  "http://localhost:3005",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:3005",
  "http://localhost:8000",
]);

const ALLOWED_METHODS = "GET, POST, PUT, DELETE, OPTIONS";
const ALLOWED_HEADERS = "Content-Type, X-API-Key, Authorization, Accept, Origin, User-Agent, X-User-Id, X-Profile-Id, Cache-Control, Pragma";

function resolveAllowedOrigin(originHeader: string | null): string | null {
  if (!originHeader) return null;
  if (ALLOWED_ORIGINS.has(originHeader)) return originHeader;
  if (/^https:\/\/(www\.)?arxterminal\.com$/.test(originHeader)) return originHeader;
  if (/^https:\/\/[a-z0-9-]+\.finance-xp8\.pages\.dev$/.test(originHeader)) return originHeader;
  if (/^http:\/\/localhost:\d+$/.test(originHeader)) return originHeader;
  if (/^http:\/\/127\.0\.0\.1:\d+$/.test(originHeader)) return originHeader;
  return null;
}

export const onRequest = async (context: any): Promise<Response> => {
  const origin = context.request.headers.get("Origin");
  const allowedOrigin = resolveAllowedOrigin(origin);

  if (context.request.method === "OPTIONS") {
    if (origin && !allowedOrigin) {
      return new Response(JSON.stringify({ error: "Forbidden origin" }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      });
    }

    const headers: Record<string, string> = {
      "Access-Control-Allow-Methods": ALLOWED_METHODS,
      "Access-Control-Allow-Headers": ALLOWED_HEADERS,
      "Access-Control-Max-Age": "86400",
      "Vary": "Origin",
    };
    if (allowedOrigin) {
      headers["Access-Control-Allow-Origin"] = allowedOrigin;
      headers["Access-Control-Allow-Credentials"] = "true";
    }

    return new Response(null, {
      status: 204,
      headers,
    });
  }

  const pathParam = context.params.path;
  const path = Array.isArray(pathParam) ? pathParam.join("/") : (pathParam || "");
  const requestUrl = new URL(context.request.url);
  const targetUrl = `https://web-production-e370b.up.railway.app/api/v1/${path}${requestUrl.search}`;

  const forwardHeaders = new Headers(context.request.headers);
  forwardHeaders.set("Host", "web-production-e370b.up.railway.app");

  try {
    const apiResponse = await fetch(targetUrl, {
      method: context.request.method,
      headers: forwardHeaders,
      body: context.request.body,
    });

    const responseHeaders = new Headers(apiResponse.headers);
    responseHeaders.set("Vary", "Origin");
    if (allowedOrigin) {
      responseHeaders.set("Access-Control-Allow-Origin", allowedOrigin);
      responseHeaders.set("Access-Control-Allow-Credentials", "true");
      responseHeaders.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
      responseHeaders.set("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization, Accept, Origin, User-Agent, X-User-Id, X-Profile-Id, Cache-Control, Pragma");
    }

    return new Response(apiResponse.body, {
      status: apiResponse.status,
      statusText: apiResponse.statusText,
      headers: responseHeaders,
    });
  } catch (err: any) {
    const errorHeaders: Record<string, string> = {
      "Content-Type": "application/json",
      "Vary": "Origin",
    };
    if (allowedOrigin) {
      errorHeaders["Access-Control-Allow-Origin"] = allowedOrigin;
      errorHeaders["Access-Control-Allow-Credentials"] = "true";
      errorHeaders["Access-Control-Allow-Methods"] = ALLOWED_METHODS;
      errorHeaders["Access-Control-Allow-Headers"] = ALLOWED_HEADERS;
    }

    return new Response(JSON.stringify({ error: "Backend origin gateway error" }), {
      status: 502,
      headers: errorHeaders,
    });
  }
};
