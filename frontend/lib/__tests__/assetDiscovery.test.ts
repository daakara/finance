import { afterEach, describe, expect, it, vi } from "vitest";
import { normalizeDiscoveryResults, discoverAssets } from "../assetDiscovery";
import { onRequestGet } from "../../functions/api/assets/search";

afterEach(() => vi.unstubAllGlobals());
const quote = { symbol: "NEWCO", longname: "Newly Listed Company", quoteType: "EQUITY", exchDisp: "NASDAQ" };

describe("provider-backed discovery without analytics prerequisites", () => {
  it("discovers an unregistered listing without price/history/scoring fields", () => {
    expect(normalizeDiscoveryResults({ quotes: [quote] })).toEqual([{
      symbol: "NEWCO", name: quote.longname, type: "EQUITY", exchange: "NASDAQ",
      source: "YAHOO_FINANCE_SEARCH", identityVerified: false,
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
  it("uses one fixed upstream request and never forwards client credentials", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ quotes: [quote] })));
    vi.stubGlobal("fetch", fetcher);
    const response = await onRequestGet({ request: new Request("https://arxterminal.com/api/assets/search?q=Newly%20Listed", {
      headers: { Authorization: "secret", Cookie: "private" },
    }) });
    expect(response.status).toBe(200);
    expect(fetcher).toHaveBeenCalledTimes(1);
    const [url, options] = fetcher.mock.calls[0];
    expect(url.origin).toBe("https://query1.finance.yahoo.com");
    expect(url.searchParams.get("q")).toBe("Newly Listed");
    expect(url.searchParams.get("newsCount")).toBe("0");
    expect(options.redirect).toBe("error");
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
    expect(await response.json()).toEqual({ status: "UNAVAILABLE", results: [] });
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    expect(fetcher).toHaveBeenCalledTimes(1);
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
