import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

describe("ARX Sprint P2: Radar Investor Attention Contract Suite (#230 / ARX-P2-PR249-FINALIZE-010)", () => {
  const radarPath = path.resolve(__dirname, "../app/radar/page.tsx");
  const radarSrc = fs.readFileSync(radarPath, "utf8");

  describe("1. Fallback Elimination & Honest Zero in Qualified Opportunities", () => {
    it("eliminates automatic fallback from QUALIFIED_OPPORTUNITIES to ALL_RESEARCH when totalActive === 0", () => {
      // Must not fall back to displaying all assets when totalActive === 0
      assert.strictEqual(
        radarSrc.includes("experienceMode === 'ALL_RESEARCH' || ownershipCounts.totalActive === 0"),
        false,
        "FAIL: Automatic fallback to all research when totalActive === 0 must be eliminated"
      );
      assert.ok(
        radarSrc.includes(
          "const matchesExperience =\n          experienceMode === 'ALL_RESEARCH' || ownershipFilter === 'MY_HOLDINGS'\n            ? true\n            : isCandidateEligibleOpportunity(asset);"
        ),
        "FAIL: matchesExperience must strictly enforce candidate eligibility in QUALIFIED_OPPORTUNITIES"
      );
    });

    it("presents honest zero-opportunity state and 1-click switch to All Research", () => {
      assert.ok(
        radarSrc.includes("No Active Qualified Opportunities (0 Candidates in Buy Zone)"),
        "FAIL: Empty state must clearly state 0 Candidates in Buy Zone"
      );
      assert.ok(
        radarSrc.includes("Switch to 'All Research / Catalog' to review reference setups."),
        "FAIL: Empty state must provide guidance to switch to All Research"
      );
      assert.ok(
        radarSrc.includes("📚 View All Research Assets ({allAssets.length}) →"),
        "FAIL: Must offer 1-click button to view all research assets when totalActive === 0"
      );
    });
  });

  describe("2. Analytical Stop Invalidation vs Portfolio Exit Semantics", () => {
    it("formats STOPPED_OUT as technical setup invalidation rather than portfolio liquidation", () => {
      assert.ok(
        radarSrc.includes("case 'STOPPED_OUT':\n      return 'SETUP INVALIDATED (STOP)';"),
        "FAIL: STOPPED_OUT must format as SETUP INVALIDATED (STOP) to prevent confusing model invalidation with trade exit"
      );
    });

    it("provides explicit tooltip distinguishing setup invalidation from portfolio position exit", () => {
      assert.ok(
        radarSrc.includes(
          "Analytical setup invalidated (price below technical stop). This indicates model invalidation, not a portfolio position exit."
        ),
        "FAIL: Status badge must include explanatory tooltip for STOPPED_OUT setups"
      );
    });

    it("explicitly excludes STOPPED_OUT setups from candidate opportunity qualification", () => {
      assert.ok(
        radarSrc.includes("if (asset.executionStatus === 'STOPPED_OUT') return false;"),
        "FAIL: isCandidateEligibleOpportunity must explicitly reject STOPPED_OUT setups"
      );
    });
  });

  describe("3. Multi-Strategy Discovery & Non-Prescriptive Discovery Integrity", () => {
    it("preserves research access across multi-strategy categories in ALL_RESEARCH", () => {
      // In ALL_RESEARCH mode, matchesExperience is true regardless of executionStatus
      assert.ok(
        radarSrc.includes("experienceMode === 'ALL_RESEARCH' || ownershipFilter === 'MY_HOLDINGS'\n            ? true"),
        "FAIL: ALL_RESEARCH mode must allow discovery across all catalog assets"
      );
    });

    it("preserves discovery-only contract (isActionable remains false)", () => {
      assert.ok(
        radarSrc.includes("isActionable: false, // Invariant: Radar is a discovery scanner"),
        "FAIL: Radar must never mark candidates as actionable trade executions"
      );
    });
  });

  describe("4. Portfolio Holdings & Opportunity Separation Contract", () => {
    it("excludes non-qualified held positions from QUALIFIED_OPPORTUNITIES when not in My Holdings filter", () => {
      // In QUALIFIED_OPPORTUNITIES with default ownershipFilter === 'ALL', non-qualified held setups do not leak
      assert.ok(
        radarSrc.includes(
          "const matchesExperience =\n          experienceMode === 'ALL_RESEARCH' || ownershipFilter === 'MY_HOLDINGS'\n            ? true\n            : isCandidateEligibleOpportunity(asset);"
        ),
        "FAIL: Non-qualified held positions must not leak into Qualified Opportunities list"
      );
    });

    it("permits access to all held positions (including invalidated setups) when My Holdings is explicitly selected", () => {
      assert.ok(
        radarSrc.includes("ownershipFilter === 'MY_HOLDINGS'"),
        "FAIL: Must support explicit My Holdings view"
      );
    });
  });

  describe("5. Behavioral Simulation: 6 Contract Scenarios", () => {
    interface MockAsset {
      ticker: string;
      executionStatus: string;
      screeningEligible?: boolean;
      marketStage?: string;
      isBullishEligible?: boolean;
    }

    const isCandidateEligibleOpportunity = (asset: MockAsset): boolean => {
      if (asset.executionStatus === 'DELISTED_SECURITY') return false;
      if (asset.executionStatus === 'UNVERIFIED_ASSET') return false;
      if (asset.executionStatus === 'UNKNOWN') return false;
      if (asset.executionStatus === 'STOPPED_OUT') return false;
      if (asset.marketStage && asset.marketStage !== 'STAGE_2') return false;
      if (asset.isBullishEligible === false) return false;
      return Boolean(asset.screeningEligible);
    };

    const filterMock = (
      assets: MockAsset[],
      experienceMode: 'QUALIFIED_OPPORTUNITIES' | 'ALL_RESEARCH',
      ownershipFilter: 'ALL' | 'NEW_OPPORTUNITIES' | 'MY_HOLDINGS',
      getOwnershipState: (ticker: string) => 'HELD' | 'NOT_HELD' | 'UNKNOWN'
    ) => {
      return assets.filter((asset) => {
        const ownState = getOwnershipState(asset.ticker);
        const matchesOwnership =
          ownershipFilter === 'ALL'
            ? true
            : (ownershipFilter === 'NEW_OPPORTUNITIES' || (ownershipFilter as any) === 'NOT_HELD')
            ? ownState === 'NOT_HELD' && isCandidateEligibleOpportunity(asset)
            : ownershipFilter === 'MY_HOLDINGS' || (ownershipFilter as any) === 'HELD'
            ? ownState === 'HELD'
            : true;

        const matchesExperience =
          experienceMode === 'ALL_RESEARCH' || ownershipFilter === 'MY_HOLDINGS'
            ? true
            : isCandidateEligibleOpportunity(asset);

        return matchesOwnership && matchesExperience;
      });
    };

    const sampleAssets: MockAsset[] = [
      { ticker: "NVDA", executionStatus: "IN_BUY_ZONE", screeningEligible: true, marketStage: "STAGE_2", isBullishEligible: true },
      { ticker: "AAPL", executionStatus: "WAITING_PULLBACK", screeningEligible: false, marketStage: "STAGE_2", isBullishEligible: true },
      { ticker: "MSFT", executionStatus: "STOPPED_OUT", screeningEligible: false, marketStage: "STAGE_4", isBullishEligible: false },
      { ticker: "CPRX", executionStatus: "DELISTED_SECURITY", screeningEligible: false, marketStage: "STAGE_UNRESOLVED", isBullishEligible: false },
      { ticker: "TSLA", executionStatus: "APPROACHING_TARGET", screeningEligible: false, marketStage: "STAGE_2", isBullishEligible: true },
    ];

    it("1. Held and qualified: appears in Qualified Opportunities AND in My Holdings", () => {
      const getOwnState = (ticker: string) => ticker === "NVDA" ? "HELD" : "NOT_HELD";
      // In default Qualified Opportunities (ownershipFilter = 'ALL')
      const inQualified = filterMock(sampleAssets, 'QUALIFIED_OPPORTUNITIES', 'ALL', getOwnState);
      assert.ok(inQualified.some(a => a.ticker === "NVDA"), "Held qualified asset must appear in Qualified Opportunities");

      // In My Holdings (ownershipFilter = 'MY_HOLDINGS')
      const inHoldings = filterMock(sampleAssets, 'QUALIFIED_OPPORTUNITIES', 'MY_HOLDINGS', getOwnState);
      assert.ok(inHoldings.some(a => a.ticker === "NVDA"), "Held qualified asset must appear in My Holdings");
    });

    it("2. Held and invalidated: does NOT appear in Qualified Opportunities, but DOES appear in My Holdings and All Research", () => {
      // MSFT is STOPPED_OUT and HELD
      const getOwnState = (ticker: string) => ticker === "MSFT" ? "HELD" : "NOT_HELD";

      // Must NOT appear in default Qualified Opportunities
      const inQualified = filterMock(sampleAssets, 'QUALIFIED_OPPORTUNITIES', 'ALL', getOwnState);
      assert.strictEqual(
        inQualified.some(a => a.ticker === "MSFT"),
        false,
        "Held stopped-out asset must NOT appear in Qualified Opportunities"
      );

      // DOES appear in My Holdings
      const inHoldings = filterMock(sampleAssets, 'QUALIFIED_OPPORTUNITIES', 'MY_HOLDINGS', getOwnState);
      assert.ok(inHoldings.some(a => a.ticker === "MSFT"), "Held stopped-out asset MUST appear in My Holdings");

      // DOES appear in All Research
      const inResearch = filterMock(sampleAssets, 'ALL_RESEARCH', 'ALL', getOwnState);
      assert.ok(inResearch.some(a => a.ticker === "MSFT"), "Held stopped-out asset MUST appear in All Research");
    });

    it("3. Unheld and qualified: appears in Qualified Opportunities, does NOT appear in My Holdings", () => {
      const getOwnState = () => "NOT_HELD";
      const inQualified = filterMock(sampleAssets, 'QUALIFIED_OPPORTUNITIES', 'ALL', getOwnState);
      assert.ok(inQualified.some(a => a.ticker === "NVDA"), "Unheld qualified asset must appear in Qualified Opportunities");

      const inHoldings = filterMock(sampleAssets, 'QUALIFIED_OPPORTUNITIES', 'MY_HOLDINGS', getOwnState);
      assert.strictEqual(inHoldings.length, 0, "Unheld asset must not appear in My Holdings");
    });

    it("4. Zero qualified candidates: renders 0 candidates in Qualified Opportunities even if some are held", () => {
      // Suppose only AAPL (pullback), MSFT (stopped out, held), CPRX (delisted), TSLA (target) exist
      const inactivePool = sampleAssets.slice(1);
      const getOwnState = (ticker: string) => ticker === "MSFT" ? "HELD" : "NOT_HELD";

      const inQualified = filterMock(inactivePool, 'QUALIFIED_OPPORTUNITIES', 'ALL', getOwnState);
      assert.strictEqual(
        inQualified.length,
        0,
        "Qualified Opportunities must display exactly 0 when no candidate is qualified, even if non-qualified assets are held"
      );
    });

    it("5. Explicit My Holdings selection: displays all held securities in portfolio context", () => {
      const getOwnState = (ticker: string) => ticker === "MSFT" || ticker === "AAPL" ? "HELD" : "NOT_HELD";
      const inHoldings = filterMock(sampleAssets, 'QUALIFIED_OPPORTUNITIES', 'MY_HOLDINGS', getOwnState);
      assert.strictEqual(inHoldings.length, 2, "My Holdings must show both held positions");
      assert.ok(inHoldings.some(a => a.ticker === "MSFT"));
      assert.ok(inHoldings.some(a => a.ticker === "AAPL"));
    });

    it("6. All Research accessibility: displays all research candidates regardless of qualification or execution status", () => {
      const getOwnState = () => "NOT_HELD";
      const inResearch = filterMock(sampleAssets, 'ALL_RESEARCH', 'ALL', getOwnState);
      assert.strictEqual(inResearch.length, sampleAssets.length, "All Research must display all candidates");
    });
  });
});
