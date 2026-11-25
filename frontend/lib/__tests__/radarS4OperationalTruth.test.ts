import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchRadarSnapshot,
  normalizeRadarCapabilities,
  normalizeRadarEnvelopeState,
} from "../api";
import { resolveRadarEmptyStateCopy } from "../radarPresentation";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("ARX-S4-RADAR-001 frontend operational truth", () => {
  it("preserves reason and recoveryCondition across pending, stale, rollback, and error capabilities", () => {
    const caps = normalizeRadarCapabilities({
      VCP: {
        status: "PIPELINE_PENDING",
        universeScreening: "PIPELINE_PENDING",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: null,
        isRollback: true,
        rollbackReason: "CORRUPT_PUBLISH",
        universeBuildId: "UB-ROLLBACK-01",
      },
      SMART_MONEY: {
        status: "ERROR",
        universeScreening: "ERROR",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: 5,
        reason: "STORAGE_ERROR",
        recoveryCondition: "Restore Smart Money Postgres connection.",
      },
    });

    expect(caps.VCP.status).toBe("PIPELINE_PENDING");
    expect(caps.VCP.matchedCount).toBeNull();
    expect(caps.VCP.reason).toBe("ROLLED_BACK_PUBLICATION_POINTER");
    expect(caps.VCP.recoveryCondition).toContain("rollback");

    expect(caps.SMART_MONEY.status).toBe("ERROR");
    expect(caps.SMART_MONEY.matchedCount).toBeNull();
    expect(caps.SMART_MONEY.reason).toBe("STORAGE_ERROR");
    expect(caps.SMART_MONEY.recoveryCondition).toBe("Restore Smart Money Postgres connection.");
  });

  it("fetchRadarSnapshot does not invent 500 denominator, 0 matches, COMPLETE, or LIVE when pending/error/stale", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          scanner_id: "MINERVINI_VCP",
          status: "PIPELINE_PENDING",
          reason: "NO_ACTIVE_SNAPSHOT",
          recovery_condition: "Commit an authoritative verified VCP snapshot.",
          matched_count: null,
          snapshot: { universe_size: 35, matched_count: null },
          coverage: { coverage_status: "PENDING", scanned_successfully_count: 0 },
          freshness: { status: "UNKNOWN" },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const pendingSnap = await fetchRadarSnapshot();
    expect(pendingSnap).not.toBeNull();
    expect(pendingSnap?.status).toBe("PIPELINE_PENDING");
    expect(pendingSnap?.matchedCount).toBeNull();
    expect(pendingSnap?.populationDenominator).toBeNull();
    expect(pendingSnap?.coverageStatus).toBe("PENDING");
    expect(pendingSnap?.freshnessStatus).toBe("UNKNOWN");
    expect(pendingSnap?.reason).toBe("NO_ACTIVE_SNAPSHOT");
    expect(pendingSnap?.recoveryCondition).toBe("Commit an authoritative verified VCP snapshot.");

    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          scanner_id: "MINERVINI_VCP",
          status: "STALE",
          snapshot_id: "vcp-stale-1",
          matched_count: 4,
          data_as_of: "2026-10-15",
          universe: { eligible_universe_count: 250, universe_build_id: "UB-250", membership_hash: "abc" },
          coverage: { coverage_status: "COMPLETE", scanned_successfully_count: 250 },
          freshness: { status: "STALE" },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    const staleSnap = await fetchRadarSnapshot();
    expect(staleSnap?.status).toBe("PIPELINE_PENDING");
    expect(staleSnap?.matchedCount).toBeNull();
    expect(staleSnap?.freshnessStatus).toBe("STALE");
    expect(staleSnap?.reason).toBe("STALE_SNAPSHOT");
  });

  it("normalizeRadarEnvelopeState accepts PIPELINE_PENDING filtered scans without misclassifying as ERROR", () => {
    const pendingRes = normalizeRadarEnvelopeState({
      status: "SUCCESS",
      scanState: "COMPLETED",
      datasetState: "PIPELINE_PENDING",
    });
    expect(pendingRes.normalizedStatus).toBe("SUCCESS");
    expect(pendingRes.datasetState).toBe("PIPELINE_PENDING");
    expect(pendingRes.normalizedError).toBeNull();

    const errRes = normalizeRadarEnvelopeState({
      status: "ERROR",
      scanState: "FAILED",
      datasetState: "SCAN_FAILED",
      error: "VCP snapshot store failure",
    });
    expect(errRes.normalizedStatus).toBe("ERROR");
    expect(errRes.datasetState).toBe("SCAN_FAILED");
  });

  it("resolveRadarEmptyStateCopy aligns desktop and mobile empty states across error, pending, rollback, stale, and genuine zero", () => {
    const smErr = resolveRadarEmptyStateCopy({
      activeFilter: "SMART_MONEY",
      searchQuery: "",
      isTickerQuery: false,
      cleanQ: "",
      ownershipFilter: "ALL",
      isPortfolioVerified: true,
      allAssetsCount: 194,
      referenceUniverseCount: 500,
      capabilities: {
        SMART_MONEY: {
          status: "ERROR",
          reason: "STORAGE_ERROR",
          recoveryCondition: "Restore Postgres connectivity.",
        },
      },
      vcpStatus: "AVAILABLE",
      vcpCount: 0,
      smStatus: "ERROR",
      smCount: null,
    });
    expect(smErr.title).toContain("Error");
    expect(smErr.description).not.toContain("zero matches evaluated");
    expect(smErr.reasonCode).toBe("STORAGE_ERROR");
    expect(smErr.recoveryCondition).toBe("Restore Postgres connectivity.");

    const vcp250Zero = resolveRadarEmptyStateCopy({
      activeFilter: "VCP",
      searchQuery: "",
      isTickerQuery: false,
      cleanQ: "",
      ownershipFilter: "ALL",
      isPortfolioVerified: true,
      allAssetsCount: 100,
      referenceUniverseCount: 250,
      capabilities: {
        VCP: { status: "AVAILABLE", matchedCount: 0, populationDenominator: 250 },
      },
      vcpStatus: "AVAILABLE",
      vcpCount: 0,
      smStatus: "PIPELINE_PENDING",
      smCount: null,
    });
    expect(vcp250Zero.title).toBe("Minervini VCP \u2014 Zero Matching Contraction Setups (250 Evaluated)");
    expect(vcp250Zero.description).toContain("250 published equities");
  });

  it("page.tsx removes stale/invented fallbacks and keeps mobile card formatting aligned", () => {
    const pageSrc = fs.readFileSync(path.resolve(__dirname, "../../app/radar/page.tsx"), "utf-8");
    expect(pageSrc).not.toContain("Quotes Live");
    expect(pageSrc).not.toContain("November 3, 2026");
    expect(pageSrc).not.toContain("?? 200");
    expect(pageSrc).toContain("formatPriceObservationDate(asset.priceDate, asset.priceBasis)");
    expect(pageSrc).toContain('data-testid="radar-desktop-empty-state"');
    expect(pageSrc).toContain('data-testid="radar-mobile-empty-state"');
  });
});
