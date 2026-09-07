// Cloudflare Pages Function: /api/backend/[[path]]
// Enables true cross-origin Edge Proxying on Cloudflare Pages to the Railway API origin.

export const onRequest = async (context: any): Promise<Response> => {
  if (context.request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-API-Key, Authorization, Accept, Origin, User-Agent",
        "Access-Control-Max-Age": "86400",
      },
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
    responseHeaders.set("Access-Control-Allow-Origin", "*");
    responseHeaders.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    responseHeaders.set("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization, Accept, Origin, User-Agent");

    // Strict Governance Security Invariant: Never allow Cloudflare edge to cache private evaluation endpoints
    if (path.startsWith("governance") || path.includes("/governance")) {
      responseHeaders.set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
      responseHeaders.set("CDN-Cache-Control", "no-store");
      responseHeaders.set("Cloudflare-CDN-Cache-Control", "no-store");
      responseHeaders.set("Vary", "Authorization, X-Evaluation-Key");
      responseHeaders.set("X-Robots-Tag", "noindex, nofollow, noarchive, nosnippet");
    }

    return new Response(apiResponse.body, {
      status: apiResponse.status,
      statusText: apiResponse.statusText,
      headers: responseHeaders,
    });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: "Backend origin gateway error", detail: String(err) }), {
      status: 502,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  }
};
