import { normalizeDiscoveryResults } from "../../../lib/assetDiscovery";

type Context = { request: Request };

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: {
    "Content-Type": "application/json", "X-Content-Type-Options": "nosniff",
    "Cache-Control": status === 200 ? "public, max-age=60, s-maxage=300" : "no-store",
  } });
}

/** Fixed provider destination; no client URLs, credentials, analytics or DB calls. */
export async function onRequestGet({ request }: Context): Promise<Response> {
  const query = new URL(request.url).searchParams.get("q")?.trim() || "";
  if (query.length < 2 || query.length > 80 || /[\x00-\x1f\x7f]/.test(query)) {
    return json({ status: "INVALID_QUERY", results: [] }, 400);
  }
  const upstream = new URL("https://query1.finance.yahoo.com/v1/finance/search");
  upstream.search = new URLSearchParams({ q: query, quotesCount: "8", newsCount: "0", enableFuzzyQuery: "false" }).toString();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  try {
    const response = await fetch(upstream, { redirect: "error", signal: controller.signal,
      headers: { Accept: "application/json" } });
    if (!response.ok) return json({ status: "UNAVAILABLE", results: [] }, 503);
    const payload = await response.json();
    return json({ status: "AVAILABLE", results: normalizeDiscoveryResults(payload),
      source: "YAHOO_FINANCE_SEARCH", identityVerified: false });
  } catch {
    return json({ status: "UNAVAILABLE", results: [] }, 503);
  } finally {
    clearTimeout(timer);
  }
}
