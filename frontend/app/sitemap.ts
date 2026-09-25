import type { MetadataRoute } from "next";
import { getAllMasterTickers } from "../lib/masterCatalog";
import { SHARED_WATCHLIST_ITEMS } from "../lib/constants";
import { COMPETITOR_CATALOG } from "../lib/competitorCatalog";
import { GLOSSARY_CATALOG } from "../lib/glossaryCatalog";
import {
  STRATEGY_DATABASE,
  POLITICIAN_DATABASE,
  COMMITTEE_DATABASE,
  COMPARISON_PAIRS,
} from "../lib/seoCatalogs";

const BASE_URL = "https://www.arxterminal.com";

/**
 * Authoritative Next.js App Router Sitemap Generator
 *
 * Implements strict Indexable Route Authority policy:
 * - Emits only public, canonical, indexable routes with canonical trailing slash.
 * - Excludes private/internal surfaces (Executive OS, workbench, cockpit, me, research).
 * - Excludes client redirect stubs (/committee, /politician, /strategy).
 * - Excludes noindex routes (/performance, /journal).
 * - Excludes error routes (/404).
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const routes: MetadataRoute.Sitemap = [];
  const seen = new Set<string>();

  const addUrl = (
    path: string,
    priority: number = 0.7,
    changeFrequency:
      | "always"
      | "hourly"
      | "daily"
      | "weekly"
      | "monthly"
      | "yearly"
      | "never" = "daily",
    lastModified: Date = new Date("2026-09-24T00:00:00.000Z")
  ) => {
    // Canonical format: Ensure clean trailing slash
    const normalizedPath = path === "/" ? "/" : `/${path.replace(/^\/|\/$/g, "")}/`;
    const fullUrl = `${BASE_URL}${normalizedPath}`;
    if (!seen.has(fullUrl)) {
      seen.add(fullUrl);
      routes.push({
        url: fullUrl,
        lastModified,
        changeFrequency,
        priority,
      });
    }
  };

  // 1. Root & Core Public Hubs (11 URLs)
  addUrl("/", 1.0, "daily");
  addUrl("/radar", 0.9, "daily");
  addUrl("/setups", 0.9, "daily");
  addUrl("/smart-money", 0.9, "daily");
  addUrl("/smart-money/late-filers", 0.8, "daily");
  addUrl("/screener", 0.9, "daily");
  addUrl("/portfolio", 0.8, "daily");
  addUrl("/guide", 0.7, "weekly");
  addUrl("/compare", 0.8, "daily");
  addUrl("/glossary", 0.8, "weekly");
  addUrl("/vs", 0.8, "weekly");

  // 2. Programmatic Stock Detail Hubs (49 tickers)
  const masterTickers = getAllMasterTickers().map((t) => t.toLowerCase());
  const watchlistTickers = SHARED_WATCHLIST_ITEMS.map((item) => item.symbol.toLowerCase());
  const uniqueTickers = Array.from(new Set([...masterTickers, ...watchlistTickers])).sort();
  for (const ticker of uniqueTickers) {
    addUrl(`/stock/${ticker}`, 0.8, "daily");
  }

  // 3. Programmatic Strategy Screener Hubs (5 strategies)
  for (const strategy of STRATEGY_DATABASE) {
    addUrl(`/strategy/${strategy.slug.toLowerCase()}`, 0.8, "weekly");
  }

  // 4. Programmatic Congressional Committee Hubs (5 committees)
  for (const committee of COMMITTEE_DATABASE) {
    addUrl(`/committee/${committee.slug.toLowerCase()}`, 0.7, "weekly");
  }

  // 5. Programmatic Politician Portfolios (8 politicians)
  for (const politician of POLITICIAN_DATABASE) {
    addUrl(`/politician/${politician.slug.toLowerCase()}`, 0.7, "weekly");
  }

  // 6. Programmatic Quantitative Glossary Terms (10 terms)
  for (const term of GLOSSARY_CATALOG) {
    addUrl(`/glossary/${term.slug.toLowerCase()}`, 0.6, "monthly");
  }

  // 7. Programmatic Competitor Versus Pages (4 competitors)
  for (const comp of COMPETITOR_CATALOG) {
    addUrl(`/vs/${comp.slug.toLowerCase()}`, 0.7, "weekly");
  }

  // 8. Programmatic Head-to-Head Comparison Pairs (7 pairs)
  for (const pair of COMPARISON_PAIRS) {
    addUrl(`/compare/${pair.pair.toLowerCase()}`, 0.7, "weekly");
  }

  return routes;
}
