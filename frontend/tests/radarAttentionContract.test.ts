import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

describe("ARX Sprint P2: Radar Investor Attention Contract Suite (#230 / ARX-P2-ATTENTION-CONTRACT-009)", () => {
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
          "const matchesExperience =\n          experienceMode === 'ALL_RESEARCH'\n            ? true\n            : (ownState === 'HELD' ? true : isCandidateEligibleOpportunity(asset));"
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
        radarSrc.includes("experienceMode === 'ALL_RESEARCH'\n            ? true"),
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

  describe("4. Portfolio Holdings Access Preservation", () => {
    it("preserves access to HELD portfolio positions in QUALIFIED_OPPORTUNITIES even if not in buy zone", () => {
      assert.ok(
        radarSrc.includes("ownState === 'HELD' ? true : isCandidateEligibleOpportunity(asset)"),
        "FAIL: HELD positions must remain visible to the investor"
      );
    });
  });

  describe("5. Behavioral Simulation: Eligibility & Experience Mode Filtering", () => {
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
      getOwnershipState: (ticker: string) => 'HELD' | 'NOT_HELD' | 'UNKNOWN'
    ) => {
      return assets.filter((asset) => {
        const ownState = getOwnershipState(asset.ticker);
        const matchesExperience =
          experienceMode === 'ALL_RESEARCH'
            ? true
            : (ownState === 'HELD' ? true : isCandidateEligibleOpportunity(asset));
        return matchesExperience;
      });
    };

    const sampleAssets: MockAsset[] = [
      { ticker: "NVDA", executionStatus: "IN_BUY_ZONE", screeningEligible: true, marketStage: "STAGE_2", isBullishEligible: true },
      { ticker: "AAPL", executionStatus: "WAITING_PULLBACK", screeningEligible: false, marketStage: "STAGE_2", isBullishEligible: true },
      { ticker: "MSFT", executionStatus: "STOPPED_OUT", screeningEligible: false, marketStage: "STAGE_4", isBullishEligible: false },
      { ticker: "CPRX", executionStatus: "DELISTED_SECURITY", screeningEligible: false, marketStage: "STAGE_UNRESOLVED", isBullishEligible: false },
      { ticker: "TSLA", executionStatus: "APPROACHING_TARGET", screeningEligible: false, marketStage: "STAGE_2", isBullishEligible: true },
    ];

    it("correctly identifies active eligible opportunities", () => {
      assert.strictEqual(isCandidateEligibleOpportunity(sampleAssets[0]), true); // NVDA
      assert.strictEqual(isCandidateEligibleOpportunity(sampleAssets[1]), false); // AAPL (WAITING_PULLBACK)
      assert.strictEqual(isCandidateEligibleOpportunity(sampleAssets[2]), false); // MSFT (STOPPED_OUT)
      assert.strictEqual(isCandidateEligibleOpportunity(sampleAssets[3]), false); // CPRX (DELISTED_SECURITY)
      assert.strictEqual(isCandidateEligibleOpportunity(sampleAssets[4]), false); // TSLA (APPROACHING_TARGET)
    });

    it("renders authentic zero candidates when totalActive === 0 and no held assets", () => {
      // Suppose only AAPL, MSFT, CPRX, TSLA exist (0 in buy zone)
      const inactivePool = sampleAssets.slice(1);
      const ownershipMap: Record<string, 'HELD' | 'NOT_HELD'> = {
        AAPL: 'NOT_HELD',
        MSFT: 'NOT_HELD',
        CPRX: 'NOT_HELD',
        TSLA: 'NOT_HELD',
      };
      const getOwnState = (ticker: string) => ownershipMap[ticker] || 'NOT_HELD';

      const qualified = filterMock(inactivePool, 'QUALIFIED_OPPORTUNITIES', getOwnState);
      assert.strictEqual(qualified.length, 0, "Must be honest zero candidates");

      const research = filterMock(inactivePool, 'ALL_RESEARCH', getOwnState);
      assert.strictEqual(research.length, 4, "ALL_RESEARCH preserves all 4 research candidates");
    });

    it("keeps portfolio holdings visible in QUALIFIED_OPPORTUNITIES even if not in buy zone", () => {
      const inactivePool = sampleAssets.slice(1);
      const ownershipMap: Record<string, 'HELD' | 'NOT_HELD'> = {
        AAPL: 'HELD', // Held position awaiting pullback
        MSFT: 'NOT_HELD',
        CPRX: 'NOT_HELD',
        TSLA: 'NOT_HELD',
      };
      const getOwnState = (ticker: string) => ownershipMap[ticker] || 'NOT_HELD';

      const qualified = filterMock(inactivePool, 'QUALIFIED_OPPORTUNITIES', getOwnState);
      assert.strictEqual(qualified.length, 1, "Held position remains visible");
      assert.strictEqual(qualified[0].ticker, "AAPL");
    });
  });
});
