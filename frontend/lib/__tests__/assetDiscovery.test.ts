import { afterEach, describe, expect, it, vi } from "vitest";
import { normalizeDiscoveryResults, discoverAssets, resolveDiscoveredSymbol } from "../assetDiscovery";
import { onRequestGet } from "../../functions/api/assets/search";

afterEach(() => vi.unstubAllGlobals());
const quote = { symbol: "NEWCO", longname: "Newly Listed Company", quoteType: "EQUITY", exchDisp: "NASDAQ" };

describe("provider-backed discovery without analytics prerequisites", () => {
  it("discovers an unregistered listing without price/history/scoring fields", () => {
    expect(normalizeDiscoveryResults({ quotes: [quote] })).toEqual([{
      symbol: "NEWCO", name: quote.longname, type: "EQUITY", exchange: "NASDAQ",
      source: "ALPACA_ASSET_DIRECTORY", identityVerified: false,
    }]);
  });
  it("rejects malformed symbols/types, deduplicates and bounds results", () => {
    const list = [quote, quote, { ...quote, symbol: "https://bad" }, { ...quote, quoteType: "OPTION" },
      ...Array.from({ length: 20 }, (_, i) => ({ ...quote, symbol: `NEW${i}` }))];
    expect(normalizeDiscoveryResults({ quotes: list })).toHaveLength(8);
    expect(normalizeDiscoveryResults({ quotes: [null, {}, { ...quote, longname: "" }] })).toEqual([]);
    expect(() => normalizeDiscoveryResults({})).toThrow();
  });
  it("does not turn empty provider results into an error", () => {
    expect(normalizeDiscoveryResults({ quotes: [] })).toEqual([]);
  });
  it("resolves a company name among related funds, but never guesses ambiguous names", () => {
    const matches = normalizeDiscoveryResults({ quotes: [
      { ...quote, symbol: "ANT", longname: "Anthropic, PBC" },
      { ...quote, symbol: "AIL", longname: "GraniteShares 2x Long Anthropic Daily ETF" },
    ] });
    expect(resolveDiscoveredSymbol("Anthropic", matches)).toBe("ANT");
    expect(resolveDiscoveredSymbol("AIL", matches)).toBe("AIL");
    expect(resolveDiscoveredSymbol("Anthropic", [...matches, { ...matches[0], symbol: "OTHER" }])).toBeNull();
  });
  it("uses one fixed upstream request and never forwards client credentials", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "AVAILABLE", source: "ALPACA_ASSET_DIRECTORY", results: [{ symbol: quote.symbol, name: quote.longname, type: quote.quoteType, exchange: quote.exchDisp }] })));
    vi.stubGlobal("fetch", fetcher);
    const response = await onRequestGet({ request: new Request("https://arxterminal.com/api/assets/search?q=Newly%20Listed", {
      headers: { Authorization: "secret", Cookie: "private" },
    }) });
    expect(response.status).toBe(200);
    expect(fetcher).toHaveBeenCalledTimes(1);
    const [url, options] = fetcher.mock.calls[0];
    expect(url.origin).toBe("https://web-production-e370b.up.railway.app");
    expect(url.searchParams.get("q")).toBe("Newly Listed");
    expect(url.pathname).toBe("/api/v1/market/assets/search");
    expect(options.redirect).toBe("manual");
    expect(options.headers).toEqual({ Accept: "application/json" });
    expect((await response.json()).results[0].identityVerified).toBe(false);
  });
  it.each(["", "A", "x".repeat(81), "bad\u0000query"])("blocks invalid query before fetching %s", async (q) => {
    const fetcher = vi.fn(); vi.stubGlobal("fetch", fetcher);
    expect((await onRequestGet({ request: new Request(`https://arxterminal.com/api/assets/search?q=${encodeURIComponent(q)}`) })).status).toBe(400);
    expect(fetcher).not.toHaveBeenCalled();
  });
  it.each(["http", "network", "malformed"])("fails gracefully on provider %s failure", async (kind) => {
    const fetcher = kind === "network" ? vi.fn().mockRejectedValue(new Error("secret internal failure")) :
      vi.fn().mockResolvedValue(new Response(JSON.stringify(kind === "malformed" ? {} : { quotes: [] }), { status: kind === "http" ? 429 : 200 }));
    vi.stubGlobal("fetch", fetcher);
    const response = await onRequestGet({ request: new Request("https://arxterminal.com/api/assets/search?q=NEWCO") });
    expect(response.status).toBe(503);
    expect(await response.json()).toMatchObject({ status: "UNAVAILABLE", results: [],
      reason: kind === "http" ? "PROVIDER_HTTP_ERROR" : "PROVIDER_RESPONSE_ERROR" });
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it("rejects a redirect without following or retrying", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(null, { status: 302, headers: { Location: "https://elsewhere.example" } }));
    vi.stubGlobal("fetch", fetcher);
    const response = await onRequestGet({ request: new Request("https://arxterminal.com/api/assets/search?q=NEWCO") });
    expect(response.status).toBe(503);
    expect(await response.json()).toMatchObject({ reason: "PROVIDER_HTTP_ERROR", upstreamStatus: 302 });
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher.mock.calls[0][1].redirect).toBe("manual");
  });
  it("browser requests the same-origin discovery route, not analytics", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "AVAILABLE", results: [{
      symbol: "NEWCO", name: quote.longname, type: "EQUITY", exchange: "NASDAQ",
    }] })));
    vi.stubGlobal("fetch", fetcher);
    expect((await discoverAssets("Newly Listed"))[0].symbol).toBe("NEWCO");
    expect(fetcher.mock.calls[0][0]).toBe("/api/assets/search?q=Newly%20Listed");
  });
});
