import { describe, expect, it } from "vitest";
import { normalizeRadarCapabilities } from "../lib/api";

describe("Radar stale VCP capability truth", () => {
  const staleVcp = {
    status: "AVAILABLE" as const,
    universeScreening: "AVAILABLE" as const,
    singleAssetAnalysis: "AVAILABLE" as const,
    matchedCount: 3,
    freshnessStatus: "STALE",
    snapshotId: "historical-vcp-snapshot",
    dataAsOf: "2026-10-31",
    universeBuildId: "UB-ARX-EXPANSION-250-20261107",
    populationDenominator: 250,
  };

  it("fails closed when stale VCP evidence carries a historical match count", () => {
    const cap = normalizeRadarCapabilities({ VCP: staleVcp }).VCP;
    expect(cap.status).toBe("PIPELINE_PENDING");
    expect(cap.universeScreening).toBe("PIPELINE_PENDING");
    expect(cap.matchedCount).toBeNull();
    expect(cap.freshnessStatus).toBe("STALE");
    expect(cap.snapshotId).toBe(staleVcp.snapshotId);
    expect(cap.universeBuildId).toBe(staleVcp.universeBuildId);
    expect(cap.lifecycleState).toBe("STALE_SNAPSHOT");
  });

  it("still recognizes a fresh, authentic VCP match count", () => {
    const cap = normalizeRadarCapabilities({ VCP: { ...staleVcp, freshnessStatus: "LIVE" } }).VCP;
    expect(cap.status).toBe("AVAILABLE");
    expect(cap.universeScreening).toBe("AVAILABLE");
    expect(cap.matchedCount).toBe(3);
  });

  it("does not mistake a pending scan for a real zero", () => {
    const cap = normalizeRadarCapabilities({ VCP: {
      ...staleVcp,
      status: "PIPELINE_PENDING",
      universeScreening: "PIPELINE_PENDING",
      matchedCount: null,
    } }).VCP;
    expect(cap.status).toBe("PIPELINE_PENDING");
    expect(cap.matchedCount).toBeNull();
  });
});
