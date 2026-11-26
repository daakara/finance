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
  const upstream = new URL("https://web-production-e370b.up.railway.app/api/v1/market/assets/search");
  upstream.search = new URLSearchParams({ q: query }).toString();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6500);
  try {
    // Workers supports manual/follow, not the Fetch standard's error mode.
    const response = await fetch(upstream, { redirect: "manual", signal: controller.signal,
      headers: { Accept: "application/json" } });
    if (!response.ok) return json({ status: "UNAVAILABLE", results: [],
      reason: "PROVIDER_HTTP_ERROR", upstreamStatus: response.status }, 503);
    const payload = await response.json();
    if (payload.status !== "AVAILABLE" || payload.source !== "ALPACA_ASSET_DIRECTORY" || !Array.isArray(payload.results)) {
      throw new Error("Invalid directory response");
    }
    const results = normalizeDiscoveryResults({ quotes: payload.results.map((r: any) => ({
      symbol: r.symbol, longname: r.name, quoteType: r.type, exchDisp: r.exchange,
    })) });
    return json({ status: "AVAILABLE", results, sourceObservedAt: payload.sourceObservedAt,
      scope: "ACTIVE_US_EQUITIES", source: "ALPACA_ASSET_DIRECTORY", identityVerified: false });
  } catch {
    return json({ status: "UNAVAILABLE", results: [],
      reason: controller.signal.aborted ? "PROVIDER_TIMEOUT" : "PROVIDER_RESPONSE_ERROR" }, 503);
  } finally {
    clearTimeout(timer);
  }
}
