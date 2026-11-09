import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

describe("ARX Sprint P2: Radar Coverage Integrity & Category Count Consistency Suite (#230)", () => {
  const radarPath = path.resolve(__dirname, "../app/radar/page.tsx");
  const radarSrc = fs.readFileSync(radarPath, "utf8");

  describe("1. Denominator Integrity & Elimination of Invented Fallbacks", () => {
    it("eliminates invented fallback to 35 in Value / GARP population denominator", () => {
      // Must not fall back to magic number 35 when populationDenominator is null
      assert.strictEqual(
        radarSrc.includes(": 35)"),
        false,
        "FAIL: Fallback to magic number 35 must be eliminated from radar/page.tsx"
      );
      assert.strictEqual(
        radarSrc.includes("|| 35"),
        false,
        "FAIL: Fallback to magic number 35 must be eliminated from radar/page.tsx"
      );
      assert.strictEqual(
        radarSrc.includes("?? 35"),
        false,
        "FAIL: Fallback to magic number 35 must be eliminated from radar/page.tsx"
      );
    });

    it("eliminates invented fallback constants in VCP empty state description", () => {
      assert.strictEqual(
        radarSrc.includes("capabilities.VCP?.populationDenominator ?? 500"),
        false,
        "FAIL: Hardcoded fallback to 500 in VCP empty state must be eliminated"
      );
      assert.strictEqual(
        radarSrc.includes("allAssets.length > 0 ? allAssets.length : 194"),
        false,
        "FAIL: Hardcoded fallback to 194 in VCP empty state must be eliminated"
      );
      assert.strictEqual(
        radarSrc.includes('capabilities.VCP?.universeBuildId ?? "UB-ARX-EXPANSION-500-20261108"'),
        false,
        "FAIL: Hardcoded fallback build ID must be eliminated"
      );
      assert.strictEqual(
        radarSrc.includes('capabilities.VCP?.dataAsOf ?? "2026-11-06"'),
        false,
        "FAIL: Hardcoded fallback dataAsOf must be eliminated"
      );
    });
  });

  describe("2. Verified Completeness & 100% Claim Gating", () => {
    it("gates '100% of Defined Reference Catalog Evaluated' on verified complete status", () => {
      assert.ok(
        radarSrc.includes("const isVerifiedComplete = typeof totalEvaluated === 'number' && totalEvaluated > 0 && !fetchError && capabilities.VALUE_GARP?.status === 'AVAILABLE'"),
        "FAIL: 100% evaluated claim must be gated on verified complete status"
      );
      assert.ok(
        radarSrc.includes("Catalog Coverage Pending"),
        "FAIL: Must provide 'Catalog Coverage Pending' fallback when coverage is unverified"
      );
      assert.ok(
        radarSrc.includes("COVERAGE PENDING"),
        "FAIL: Reference catalog badge must support 'COVERAGE PENDING'"
      );
    });
  });

  describe("3. Tab Badge & Count Consistency (VCP & Smart Money)", () => {
    it("synchronizes VCP tab badge with authoritative matchedCount when AVAILABLE", () => {
      // VCP badge must use authoritative matchedCount when available, not raw client-filtered length
      assert.ok(
        radarSrc.includes("typeof capabilities.VCP?.matchedCount === 'number' && !capabilities.VCP?.isRollback\n              ? capabilities.VCP.matchedCount\n              : allAssets.filter((a) => a.categories.includes('VCP')).length"),
        "FAIL: VCP badge must synchronize with authoritative matchedCount"
      );
    });

    it("synchronizes SMART_MONEY tab badge with authoritative displayedStage2IntersectionCount / matchedCount", () => {
      assert.ok(
        radarSrc.includes("typeof capabilities.SMART_MONEY?.displayedStage2IntersectionCount === 'number'\n              ? capabilities.SMART_MONEY.displayedStage2IntersectionCount"),
        "FAIL: Smart Money badge must prioritize authoritative displayedStage2IntersectionCount"
      );
    });

    it("preserves 'Universe Scanner Pending' badge fallback when scanner is not available", () => {
      assert.ok(
        radarSrc.includes(": 'Universe Scanner Pending'"),
        "FAIL: Badge must fall back to 'Universe Scanner Pending' when not available"
      );
    });
  });

  describe("4. Authentic Zero Semantics & Stage 2 Distinction", () => {
    it("preserves authentic zero distinction in VCP empty state", () => {
      assert.ok(
        radarSrc.includes("Minervini VCP — 0 Confirmed Contraction Setups (Stage 2 Qualification ≠ Confirmed VCP Geometry)"),
        "FAIL: Must distinguish Stage 2 qualification from confirmed VCP geometry"
      );
      assert.ok(
        radarSrc.includes("This is an authentic analytical zero, not a scanner pipeline failure."),
        "FAIL: Must confirm authentic analytical zero"
      );
    });

    it("preserves authentic zero distinction in Smart Money empty state", () => {
      assert.ok(
        radarSrc.includes("Smart Money — 0 Qualifying Institutional Flow Signals"),
        "FAIL: Must display explicit zero signals when AVAILABLE"
      );
    });
  });
});
