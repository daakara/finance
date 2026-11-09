import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import {
  normalizeRadarCapabilities,
  type RadarCapabilityDetail,
  type CandidatePopulationModel,
} from "../lib/api";

describe("ARX Sprint P2: Radar Discovery Quality & Evidence Verification Suite (#230)", () => {
  const radarPath = path.resolve(__dirname, "../app/radar/page.tsx");
  const radarSrc = fs.readFileSync(radarPath, "utf8");

  describe("1. Type & Capability Contract Extensions (frontend/lib/api.ts)", () => {
    it("preserves displayedStage2IntersectionCount in normalizeRadarCapabilities", () => {
      const rawPayload = {
        VCP: {
          universeScreening: "AVAILABLE",
          matchedCount: 0,
          populationDenominator: 500,
          universeBuildId: "UB-ARX-EXPANSION-500-20261108",
          publicationDigest: "sha256-verified-digest",
          publicationIntegrityVerified: true,
          dataAsOf: "2026-11-06",
        },
        SMART_MONEY: {
          universeScreening: "AVAILABLE",
          matchedCount: 10,
          populationDenominator: 10,
          displayedStage2IntersectionCount: 5,
          universeBuildId: "UB-SM-PILOT-10-20261025",
          snapshotId: "SNAP_SM_1792939989_48f30db6",
          publicationDigest: "sha256-sm-digest",
          publicationIntegrityVerified: true,
          dataAsOf: "2026-10-25",
        },
      };

      const normalized = normalizeRadarCapabilities(rawPayload);
      assert.strictEqual(
        normalized.SMART_MONEY?.displayedStage2IntersectionCount,
        5,
        "displayedStage2IntersectionCount must be preserved as 5"
      );
      assert.strictEqual(
        normalized.SMART_MONEY?.populationDenominator,
        10,
        "populationDenominator must be 10"
      );
      assert.strictEqual(
        normalized.VCP?.displayedStage2IntersectionCount,
        null,
        "missing displayedStage2IntersectionCount must default cleanly to null"
      );
    });

    it("verifies CandidatePopulationModel supports extended population telemetry", () => {
      const model: CandidatePopulationModel = {
        referenceUniverseCount: 500,
        activeScreeningDenominator: 500,
        currentOpportunityCount: 132,
        delistedRecordCount: 0,
        unverifiedRecordCount: 0,
        stage2OpportunityCount: 194,
        stage1RecordCount: 12,
        stage4RecordCount: 45,
        stageUnresolvedRecordCount: 8,
        universeScope: "US_EXPANSION_500",
        universeBuildId: "UB-ARX-EXPANSION-500-20261108",
        marketDataAsOf: "2026-11-06",
      };

      assert.strictEqual(model.stage2OpportunityCount, 194);
      assert.strictEqual(model.universeScope, "US_EXPANSION_500");
      assert.strictEqual(model.universeBuildId, "UB-ARX-EXPANSION-500-20261108");
    });
  });

  describe("2. Value / GARP Dynamic Count Authority (F-01)", () => {
    it("eliminates static hardcoded '16 of 16 Value / GARP Matches'", () => {
      assert.strictEqual(
        radarSrc.includes("{filteredAssets.length} of 16 Value / GARP Matches"),
        false,
        "FAIL: Static hardcoded '16 of 16 Value / GARP Matches' must be eliminated"
      );
    });

    it("eliminates static hardcoded 'REFERENCE CATALOG (16 MATCHES / 35 EVALUATED)'", () => {
      assert.strictEqual(
        radarSrc.includes("REFERENCE CATALOG (16 MATCHES / 35 EVALUATED)"),
        false,
        "FAIL: Static hardcoded 'REFERENCE CATALOG (16 MATCHES / 35 EVALUATED)' must be eliminated"
      );
    });

    it("dynamically resolves Value / GARP matches and total evaluated population", () => {
      assert.ok(
        radarSrc.includes("const authValueGarp = typeof capabilities.VALUE_GARP?.matchedCount === 'number'"),
        "FAIL: Must dynamically resolve authValueGarp from capabilities or filtered category assets"
      );
      assert.ok(
        radarSrc.includes("const totalEvaluated = capabilities.VALUE_GARP?.populationDenominator"),
        "FAIL: Must dynamically resolve totalEvaluated from populationDenominator or candidate count"
      );
    });
  });

  describe("3. Smart Money Dedicated Scope Banner & Pilot Clarification (F-02)", () => {
    it("renders dedicated data-testid='smart-money-scope-banner'", () => {
      assert.ok(
        radarSrc.includes('data-testid="smart-money-scope-banner"'),
        "FAIL: Smart Money scope banner with data-testid='smart-money-scope-banner' must be present"
      );
    });

    it("discloses historical 10-issuer pilot and active HOLD_SEC_001 hold", () => {
      assert.ok(
        radarSrc.includes("PILOT POPULATION (HOLD_SEC_001 ACTIVE)"),
        "FAIL: Smart Money banner must disclose 'PILOT POPULATION (HOLD_SEC_001 ACTIVE)'"
      );
      assert.ok(
        radarSrc.includes("HOLD_SEC_001"),
        "FAIL: Smart Money banner must reference HOLD_SEC_001"
      );
      assert.ok(
        radarSrc.includes("Stage 2 Intersections"),
        "FAIL: Smart Money banner must clarify Stage 2 Intersections"
      );
    });

    it("exposes snapshotId and dataAsOf in Smart Money banner", () => {
      assert.ok(
        radarSrc.includes("capabilities.SMART_MONEY?.snapshotId"),
        "FAIL: Smart Money banner must surface snapshotId when present"
      );
      assert.ok(
        radarSrc.includes("capabilities.SMART_MONEY?.dataAsOf"),
        "FAIL: Smart Money banner must surface dataAsOf when present"
      );
    });
  });

  describe("4. VCP Authentic Zero & Stage 2 Clarification in Empty State (F-03)", () => {
    it("explicitly distinguishes Minervini Stage 2 Advancing from confirmed VCP geometry", () => {
      assert.ok(
        radarSrc.includes("Minervini VCP — 0 Confirmed Contraction Setups (Stage 2 Qualification ≠ Confirmed VCP Geometry)"),
        "FAIL: Empty state title must explicitly state Stage 2 Qualification ≠ Confirmed VCP Geometry"
      );
    });

    it("explains the authentic zero across evaluated equities and points to All Confluences", () => {
      assert.ok(
        radarSrc.includes("candidates meet Minervini Stage 2 Advancing criteria (accessible under \"All Confluences\")"),
        "FAIL: Empty state body must explain candidates meet Stage 2 Advancing accessible under All Confluences"
      );
      assert.ok(
        radarSrc.includes("This is an authentic analytical zero, not a scanner pipeline failure."),
        "FAIL: Empty state body must confirm authentic analytical zero, not a scanner pipeline failure"
      );
    });

    it("provides truthful empty state for Smart Money zero signals when AVAILABLE", () => {
      assert.ok(
        radarSrc.includes("Smart Money — 0 Qualifying Institutional Flow Signals"),
        "FAIL: Smart Money empty state must display explicit 0 Qualifying Signals when AVAILABLE"
      );
    });
  });

  describe("5. Category Tab Semantics & Tooltips", () => {
    it("VCP tab tooltip clarifies 0 contraction matches across evaluated universe", () => {
      assert.ok(
        radarSrc.includes("Minervini VCP: 0 contraction matches across"),
        "FAIL: VCP tab tooltip must clarify 0 contraction matches across evaluated population"
      );
      assert.ok(
        radarSrc.includes("Stage 2 qualification does not imply confirmed VCP geometry"),
        "FAIL: VCP tab tooltip must clarify Stage 2 qualification does not imply VCP geometry"
      );
    });

    it("SMART_MONEY tab tooltip clarifies historical 10-issuer pilot and HOLD_SEC_001", () => {
      assert.ok(
        radarSrc.includes("Stage 2 matches from historical 10-issuer pilot"),
        "FAIL: SMART_MONEY tab tooltip must clarify Stage 2 matches from historical 10-issuer pilot"
      );
      assert.ok(
        radarSrc.includes("HOLD_SEC_001"),
        "FAIL: SMART_MONEY tab tooltip must reference HOLD_SEC_001"
      );
    });
  });
});
