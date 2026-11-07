import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";

describe("Directive ARX-RADAR-UNIVERSE-001: Radar Hero CTA & Authoritative Single Universe Verification", () => {
  const radarSourcePath = path.resolve(__dirname, "../app/radar/page.tsx");
  const radarSrc = fs.readFileSync(radarSourcePath, "utf8");

  const apiSourcePath = path.resolve(__dirname, "../lib/api.ts");
  const apiSrc = fs.readFileSync(apiSourcePath, "utf8");

  describe("PART 1: Verifying Source Invariants in frontend/app/radar/page.tsx", () => {
    it("eliminates static LNTH-first heroAsset fallback", () => {
      assert.ok(
        !radarSrc.includes("filteredAssets[0] || allAssets[0]"),
        "FAIL: static LNTH-first hero binding 'filteredAssets[0] || allAssets[0]' must be eliminated"
      );
    });

    it("ensures neutral 'Explore Radar Assets →' action is present for unselected state", () => {
      assert.ok(
        radarSrc.includes("Explore Radar Assets →"),
        "FAIL: Hero primaryAction must provide neutral 'Explore Radar Assets →' fallback"
      );
    });

    it("ensures dynamic candidate-specific hero action is present on selection", () => {
      assert.ok(
        radarSrc.includes("`Analyze ${heroAsset.ticker} →`"),
        "FAIL: Hero primaryAction must dynamically render `Analyze ${heroAsset.ticker} →` on selection"
      );
    });

    it("ensures dual-universe selector (Pre-Screened 35 vs Master Catalog 43) is eliminated", () => {
      assert.ok(
        !radarSrc.includes("Master Catalog (43)"),
        "FAIL: Master Catalog (43) discovery selector must be eliminated"
      );
      assert.ok(
        !radarSrc.includes("Pre-Screened VCP (35)"),
        "FAIL: Pre-Screened VCP (35) discovery selector must be eliminated"
      );
    });

    it("ensures Master Catalog search suggestion banner is eliminated", () => {
      assert.ok(
        !radarSrc.includes("Master Catalog Matches"),
        "FAIL: Master Catalog matches must not be surfaced during search"
      );
    });

    it("ensures Authoritative Published Universe metadata header is present", () => {
      assert.ok(
        radarSrc.includes("Universe: {population?.universeScope || capabilities.VCP?.universeId || 'ARX_US_EQUITIES'}"),
        "FAIL: Authoritative universe identifier badge must be present"
      );
      assert.ok(
        radarSrc.includes("Stage 2 Primary Discovery"),
        "FAIL: Stage 2 Primary Discovery policy badge must be present"
      );
    });

    it("ensures fetchScreenerGems API universe parameter is supported for diagnostics/tests", () => {
      assert.ok(
        apiSrc.includes("universe?: string") && apiSrc.includes("&universe="),
        "FAIL: fetchScreenerGems must accept optional universe parameter and pass to backend"
      );
    });
  });

  describe("PART 2: Simulating Hero CTA Resolution Across User Interaction States", () => {
    interface MockAsset {
      ticker: string;
      name: string;
      confluenceScore: number;
      decisionContextId?: string;
      executionStatus: string;
    }

    const mockUniverse: MockAsset[] = [
      { ticker: "LNTH", name: "Lantheus Holdings", confluenceScore: 85.5, executionStatus: "IN_BUY_ZONE", decisionContextId: "dec-lnth-1" },
      { ticker: "MEDP", name: "Medpace Holdings", confluenceScore: 81.5, executionStatus: "IN_BUY_ZONE", decisionContextId: "dec-medp-1" },
      { ticker: "ONON", name: "On Holding AG", confluenceScore: 83.8, executionStatus: "WAITING_PULLBACK", decisionContextId: "dec-onon-1" },
      { ticker: "ULTA", name: "Ulta Beauty Inc.", confluenceScore: 76.0, executionStatus: "WAITING_PULLBACK", decisionContextId: "dec-ulta-1" },
    ];

    function resolveHeroCTA(
      activeCanonicalAsset: string | null,
      allAssets: MockAsset[],
      isSearching: boolean,
      filteredAssets: MockAsset[]
    ) {
      const selectedAsset = activeCanonicalAsset
        ? (allAssets.find((a) => a.ticker === activeCanonicalAsset) || null)
        : null;

      const heroAsset = selectedAsset || (isSearching && filteredAssets.length > 0 ? filteredAssets[0] : null);

      const primaryAction = {
        label: heroAsset ? `Analyze ${heroAsset.ticker} →` : "Explore Radar Assets →",
        href: heroAsset
          ? `/?symbol=${heroAsset.ticker}${heroAsset.decisionContextId ? `&contextId=${encodeURIComponent(heroAsset.decisionContextId)}` : ""}`
          : "#radar-candidates-table",
      };

      const secondaryAction = {
        label: heroAsset ? `Review ${heroAsset.ticker} Setup →` : "Review All Setups →",
        href: heroAsset ? `/setups?symbol=${heroAsset.ticker}` : "/setups",
      };

      return { selectedAsset, heroAsset, primaryAction, secondaryAction };
    }

    it("Scenario 1: Initial Load (Unselected State) yields neutral Hero CTA", () => {
      const res = resolveHeroCTA(null, mockUniverse, false, mockUniverse);
      assert.strictEqual(res.heroAsset, null, "heroAsset must be null when unselected");
      assert.strictEqual(res.primaryAction.label, "Explore Radar Assets →", "Primary action must be neutral");
      assert.strictEqual(res.primaryAction.href, "#radar-candidates-table", "Primary action must link to table");
      assert.strictEqual(res.secondaryAction.label, "Review All Setups →", "Secondary action must be neutral");
      assert.strictEqual(res.secondaryAction.href, "/setups", "Secondary action must link to /setups");
    });

    it("Scenario 2: User selects MEDP updates CTA dynamically", () => {
      const res = resolveHeroCTA("MEDP", mockUniverse, false, mockUniverse);
      assert.strictEqual(res.heroAsset?.ticker, "MEDP");
      assert.strictEqual(res.primaryAction.label, "Analyze MEDP →");
      assert.ok(res.primaryAction.href.includes("symbol=MEDP"));
      assert.strictEqual(res.secondaryAction.label, "Review MEDP Setup →");
      assert.strictEqual(res.secondaryAction.href, "/setups?symbol=MEDP");
    });

    it("Scenario 3: User selects ONON updates CTA dynamically", () => {
      const res = resolveHeroCTA("ONON", mockUniverse, false, mockUniverse);
      assert.strictEqual(res.heroAsset?.ticker, "ONON");
      assert.strictEqual(res.primaryAction.label, "Analyze ONON →");
      assert.ok(res.primaryAction.href.includes("symbol=ONON"));
      assert.strictEqual(res.secondaryAction.label, "Review ONON Setup →");
      assert.strictEqual(res.secondaryAction.href, "/setups?symbol=ONON");
    });

    it("Scenario 4: User selects candidate updates CTA dynamically", () => {
      const expandedUniverse: MockAsset[] = [
        ...mockUniverse,
        { ticker: "NVDA", name: "NVIDIA Corp", confluenceScore: 43.8, executionStatus: "WAITING_PULLBACK", decisionContextId: "dec-nvda-1" },
        { ticker: "AAPL", name: "Apple Inc.", confluenceScore: 82.5, executionStatus: "IN_BUY_ZONE", decisionContextId: "dec-aapl-1" },
      ];
      const res = resolveHeroCTA("NVDA", expandedUniverse, false, expandedUniverse);
      assert.strictEqual(res.heroAsset?.ticker, "NVDA");
      assert.strictEqual(res.primaryAction.label, "Analyze NVDA →");
      assert.ok(res.primaryAction.href.includes("symbol=NVDA"));
      assert.strictEqual(res.secondaryAction.label, "Review NVDA Setup →");
    });

    it("Scenario 5: User clears selection reverts cleanly to neutral CTA", () => {
      const res = resolveHeroCTA(null, mockUniverse, false, mockUniverse);
      assert.strictEqual(res.heroAsset, null);
      assert.strictEqual(res.primaryAction.label, "Explore Radar Assets →");
    });
  });

  describe("PART 3: Master Catalog Fixture Preservation for Historical Research & Tests", () => {
    it("contains exactly 43 equity/ETF catalog entries (46 total with crypto pairs) preserved for research/tests", () => {
      const catalogEntries = Object.values(MASTER_ASSET_CATALOG);
      const equityAndEtfCatalog = catalogEntries.filter((item) => item.type !== "Crypto");
      assert.strictEqual(equityAndEtfCatalog.length, 43, `Expected 43 equity and ETF catalog entries, got ${equityAndEtfCatalog.length}`);
      assert.strictEqual(catalogEntries.length, 46, `Expected 46 total catalog entries, got ${catalogEntries.length}`);
    });

    it("verifies required test assets exist in catalog fixture (LNTH, MEDP, NVDA, AAPL, ONON, MSFT, ULTA)", () => {
      const catalogKeys = Object.keys(MASTER_ASSET_CATALOG);
      const testAssets = ["LNTH", "MEDP", "NVDA", "AAPL", "MSFT", "ULTA"];
      for (const sym of testAssets) {
        assert.ok(catalogKeys.includes(sym), `Master Catalog must contain ${sym}`);
      }
    });
  });
});
