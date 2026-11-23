/** Discovery matches are provider suggestions, never scoring or identity authority. */
export interface DiscoveredAsset {
  symbol: string;
  name: string;
  exchange: string;
  type: string;
  source: "YAHOO_FINANCE_SEARCH";
  identityVerified: false;
}

export const DISCOVERY_SYMBOL = /^[A-Z0-9^][A-Z0-9.^=_-]{0,15}$/;
const TYPES = new Set(["EQUITY", "ETF", "CRYPTOCURRENCY", "INDEX", "CURRENCY", "FUTURE", "MUTUALFUND"]);

export function normalizeDiscoveryResults(payload: unknown): DiscoveredAsset[] {
  const quotes = (payload as { quotes?: unknown })?.quotes;
  if (!Array.isArray(quotes)) throw new Error("Invalid discovery response");
  const seen = new Set<string>();
  const matches: DiscoveredAsset[] = [];
  for (const quote of quotes) {
    if (!quote || typeof quote !== "object") continue;
    const { symbol, shortname, longname, quoteType, exchDisp, exchange } = quote;
    if (typeof symbol !== "string" || !DISCOVERY_SYMBOL.test(symbol) || seen.has(symbol) || !TYPES.has(quoteType)) continue;
    const name = typeof longname === "string" ? longname : shortname;
    if (typeof name !== "string" || !name.trim()) continue;
    seen.add(symbol);
    matches.push({ symbol, name: name.slice(0, 160), type: quoteType,
      exchange: typeof exchDisp === "string" ? exchDisp.slice(0, 60) : typeof exchange === "string" ? exchange.slice(0, 60) : "Not supplied",
      source: "YAHOO_FINANCE_SEARCH", identityVerified: false });
    if (matches.length === 8) break;
  }
  return matches;
}

export async function discoverAssets(query: string, signal?: AbortSignal): Promise<DiscoveredAsset[]> {
  const response = await fetch(`/api/assets/search?q=${encodeURIComponent(query.trim())}`, { signal });
  if (!response.ok) throw new Error("Asset discovery is temporarily unavailable");
  const payload = await response.json();
  if (payload.status !== "AVAILABLE" || !Array.isArray(payload.results)) throw new Error("Invalid discovery response");
  // Re-normalize at the browser boundary rather than trusting arbitrary endpoint fields.
  return normalizeDiscoveryResults({ quotes: payload.results.map((r: DiscoveredAsset) => ({
    symbol: r.symbol, longname: r.name, quoteType: r.type, exchDisp: r.exchange,
  })) });
}
