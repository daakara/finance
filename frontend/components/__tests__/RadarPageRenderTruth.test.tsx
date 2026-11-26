import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/radar",
}));

vi.mock("../../lib/DataSourceContext", () => ({
  useDataSource: () => ({
    source: "LIVE",
    isLive: true,
    toggleSource: vi.fn(),
  }),
}));

vi.mock("../../lib/matomo", () => ({
  trackRadarAssetClick: vi.fn(),
  trackRadarFilterChange: vi.fn(),
}));

vi.mock("../../hooks/usePortfolioContext", () => ({
  usePortfolioContext: () => ({
    holdings: [],
    isPortfolioVerified: true,
    getOwnershipState: () => "NOT_HELD",
    getHolding: () => undefined,
  }),
}));

vi.mock("../../components/terminal/TerminalShell", () => ({
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-terminal-shell">{children}</div>
  ),
}));

vi.mock("../../components/PageIntro", () => ({
  default: ({ title }: { title: string }) => <h1>{title}</h1>,
}));

vi.mock("../../components/synthesis/SynthesisMobileDock", () => ({
  default: () => <div data-testid="mock-mobile-dock" />,
}));

import RadarPage from "../../app/radar/page";

const originalFetch = globalThis.fetch;

function buildSampleCandidate(symbol: string, overrides: Record<string, any> = {}) {
  return {
    ticker: symbol,
    symbol,
    companyName: `${symbol} Corp`,
    currentPrice: 120.5,
    priceBasis: "DAILY_CLOSE",
    priceDate: "2026-11-23",
    gemScore: 82,
    expertArchetype: "Peter Lynch GARP Compounder",
    categories: ["VALUE_GARP"],
    categoryEvidence: { VALUE_GARP: "CRITERIA_MATCHED" },
    executionStatus: "IN_BUY_ZONE",
    screeningStatus: "SCREENING_ZONE",
    screeningGeometry: "WITHIN_TOLERANCE",
    screeningEligible: true,
    confluenceScore: 76.4,
    confluenceRating: "Constructive Setup",
    stageClassification: "STAGE_2",
    decisionState: "VALID_SETUP",
    isActionable: false,
    canSizeTrade: false,
    allowedActions: ["RESEARCH_PROFILE"],
    ...overrides,
  };
}

function mockRadarFetches({
  snapshotPayload,
  runPayload,
  runStatus = 200,
}: {
  snapshotPayload: any;
  runPayload: any;
  runStatus?: number;
}) {
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/screener/vcp/snapshot")) {
      return new Response(JSON.stringify(snapshotPayload), {
        status: snapshotPayload?.__httpStatus ?? 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (url.includes("/screener/run")) {
      return new Response(JSON.stringify(runPayload), {
        status: runStatus,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("{}", { status: 404 });
  }) as any;
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("ARX-S4-RADAR-001 UI Desktop & Mobile State Rendering", () => {
  it("keeps ERROR distinct from PIPELINE_PENDING across tabs, VCP/SM scope banners, and desktop/mobile views", async () => {
    mockRadarFetches({
      snapshotPayload: {
        scanner_id: "MINERVINI_VCP",
        status: "ERROR",
        reason: "STORAGE_ERROR",
        recovery_condition: "Restore VCP SQLite snapshot store connectivity.",
      },
      runPayload: {
        status: "SUCCESS",
        scanState: "COMPLETED",
        datasetState: "CANDIDATES_AVAILABLE",
        totalCandidates: 500,
        gemsFound: 1,
        stage2OpportunityCount: 1,
        marketDataAsOf: "2026-11-23",
        population: {
          referenceUniverseCount: 500,
          activeScreeningCount: 500,
          currentOpportunityCount: 1,
          stage2OpportunityCount: 1,
          delistedRecordCount: 0,
          unverifiedRecordCount: 0,
          marketDataAsOf: "2026-11-23",
          displayedDataAsOf: "2026-11-23",
          displayedDataOldestAsOf: "2026-11-23",
          displayedDataMixedDates: false,
        },
        capabilities: {
          VALUE_GARP: {
            status: "AVAILABLE",
            universeScreening: "AVAILABLE",
            singleAssetAnalysis: "AVAILABLE",
            catalogType: "CURATED_REFERENCE_CATALOG",
            countAuthority: "DYNAMIC_CATALOG",
            catalogAssetCount: 43,
            matchedCount: 1,
          },
          VCP: {
            status: "ERROR",
            universeScreening: "ERROR",
            singleAssetAnalysis: "AVAILABLE",
            matchedCount: null,
            reason: "STORAGE_ERROR",
            recoveryCondition: "Restore VCP SQLite snapshot store connectivity.",
          },
          SMART_MONEY: {
            status: "ERROR",
            universeScreening: "ERROR",
            singleAssetAnalysis: "AVAILABLE",
            matchedCount: null,
            reason: "POSTGRES_ERROR",
            recoveryCondition: "Restore Smart Money Postgres connection.",
          },
        },
        candidates: [buildSampleCandidate("LNTH")],
      },
    });

    render(<RadarPage />);

    const vcpTab = await screen.findByRole("tab", { name: /Minervini VCP/i });
    const smTab = screen.getByRole("tab", { name: /Smart Money/i });
    expect(vcpTab.textContent).toContain("Error");
    expect(vcpTab.textContent).not.toContain("Universe Scanner Pending");
    expect(smTab.textContent).toContain("Error");
    expect(smTab.textContent).not.toContain("Universe Scanner Pending");

    fireEvent.click(vcpTab);
    await waitFor(() => {
      expect(screen.getByText(/Scanner Error \(STORAGE_ERROR\)/i)).toBeDefined();
    });
    expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("Scanner Snapshot Error");
    expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("STORAGE_ERROR");
    expect(screen.getByTestId("radar-mobile-empty-state").textContent).toContain("Scanner Snapshot Error");
    expect(screen.getByTestId("radar-mobile-empty-state").textContent).toContain("Restore VCP SQLite snapshot store connectivity.");

    fireEvent.click(smTab);
    await waitFor(() => {
      expect(screen.getByTestId("smart-money-scope-banner").textContent).toContain("Scanner Error (POSTGRES_ERROR)");
    });
    expect(screen.getByTestId("smart-money-scope-banner").textContent).not.toContain("Historical 10-Issuer Pilot");
    expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("Smart Money — Scanner Snapshot Error");
    expect(screen.getByTestId("radar-mobile-empty-state").textContent).toContain("Smart Money — Scanner Snapshot Error");
  });

  it("renders STALE, ROLLBACK, MISSING capability, and GENUINE_ZERO distinctly with null unavailable vs numeric zero", async () => {
    // 1. Stale VCP + Missing SMART_MONEY capability (omitted -> must fail closed to PIPELINE_PENDING, never 0)
    mockRadarFetches({
      snapshotPayload: {
        scanner_id: "MINERVINI_VCP",
        status: "STALE",
        matched_count: 4,
        data_as_of: "2026-10-15",
        universe: { eligible_universe_count: 250, universe_build_id: "UB-250", membership_hash: "h250" },
        coverage: { coverage_status: "COMPLETE", scanned_successfully_count: 250 },
        freshness: { status: "STALE" },
      },
      runPayload: {
        status: "SUCCESS",
        scanState: "COMPLETED",
        datasetState: "CANDIDATES_AVAILABLE",
        totalCandidates: 250,
        gemsFound: 1,
        stage2OpportunityCount: 1,
        population: {
          referenceUniverseCount: 250,
          activeScreeningCount: 250,
          currentOpportunityCount: 1,
          stage2OpportunityCount: 1,
          delistedRecordCount: 0,
          unverifiedRecordCount: 0,
          marketDataAsOf: "2026-11-23",
          displayedDataAsOf: "2026-11-23",
          displayedDataOldestAsOf: "2026-11-23",
          displayedDataMixedDates: false,
        },
        capabilities: {
          VALUE_GARP: {
            status: "AVAILABLE",
            universeScreening: "AVAILABLE",
            singleAssetAnalysis: "AVAILABLE",
            catalogType: "CURATED_REFERENCE_CATALOG",
            countAuthority: "DYNAMIC_CATALOG",
            matchedCount: 1,
          },
          VCP: {
            status: "AVAILABLE",
            universeScreening: "AVAILABLE",
            singleAssetAnalysis: "AVAILABLE",
            matchedCount: 4,
            freshnessStatus: "STALE",
            populationDenominator: 250,
            universeBuildId: "UB-250",
            publicationDigest: "h250",
            dataAsOf: "2026-10-15",
          },
        },
        candidates: [buildSampleCandidate("AAPL", { categories: ["VALUE_GARP"] })],
      },
    });

    const { unmount } = render(<RadarPage />);
    const vcpTab = await screen.findByRole("tab", { name: /Minervini VCP/i });
    const smTab = screen.getByRole("tab", { name: /Smart Money/i });
    expect(vcpTab.textContent).toContain("Universe Scanner Pending");
    expect(vcpTab.textContent).not.toContain("4");
    expect(smTab.textContent).toContain("Universe Scanner Pending");
    expect(smTab.textContent).not.toContain("0");

    fireEvent.click(vcpTab);
    await waitFor(() => {
      expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("Historical Snapshot Stale");
    });
    expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("STALE_SNAPSHOT");
    expect(screen.getByTestId("radar-mobile-empty-state").textContent).toContain("Historical Snapshot Stale");

    fireEvent.click(smTab);
    await waitFor(() => {
      expect(screen.getByTestId("smart-money-scope-banner").textContent).toContain("Universe Scanner Pending");
    });
    expect(screen.getByTestId("smart-money-scope-banner").textContent).not.toContain("Historical 10-Issuer Pilot");
    unmount();

    // 2. Rollback VCP + Genuine Zero VCP (separate render checks)
    mockRadarFetches({
      snapshotPayload: {
        scanner_id: "MINERVINI_VCP",
        status: "PIPELINE_PENDING",
        is_rollback: true,
        rollback_reason: "INTEGRITY_FAIL",
        universe: { eligible_universe_count: 500, universe_build_id: "UB-RB", membership_hash: "hrb" },
      },
      runPayload: {
        status: "SUCCESS",
        scanState: "COMPLETED",
        datasetState: "CANDIDATES_AVAILABLE",
        totalCandidates: 500,
        gemsFound: 1,
        stage2OpportunityCount: 1,
        population: {
          referenceUniverseCount: 500,
          activeScreeningCount: 500,
          currentOpportunityCount: 1,
          stage2OpportunityCount: 1,
          delistedRecordCount: 0,
          unverifiedRecordCount: 0,
          displayedDataAsOf: "2026-11-23",
        },
        capabilities: {
          VALUE_GARP: {
            status: "AVAILABLE",
            universeScreening: "AVAILABLE",
            singleAssetAnalysis: "AVAILABLE",
            catalogType: "CURATED_REFERENCE_CATALOG",
            countAuthority: "DYNAMIC_CATALOG",
            matchedCount: 1,
          },
          VCP: {
            status: "AVAILABLE",
            universeScreening: "AVAILABLE",
            singleAssetAnalysis: "AVAILABLE",
            matchedCount: 0,
            isRollback: true,
            rollbackReason: "INTEGRITY_FAIL",
            universeBuildId: "UB-RB",
            populationDenominator: 500,
          },
        },
        candidates: [buildSampleCandidate("NVDA", { categories: ["VALUE_GARP"] })],
      },
    });

    const rbRender = render(<RadarPage />);
    fireEvent.click(await screen.findByRole("tab", { name: /Minervini VCP/i }));
    await waitFor(() => {
      expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("Publication Rolled Back");
    });
    expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("ROLLED_BACK_PUBLICATION_POINTER");
    expect(screen.getByTestId("radar-mobile-empty-state").textContent).toContain("Publication Rolled Back");
    rbRender.unmount();

    // 3. Genuine zero VCP matches keeps count 0
    mockRadarFetches({
      snapshotPayload: {
        scanner_id: "MINERVINI_VCP",
        status: "AVAILABLE",
        snapshot_id: "vcp-zero-ok",
        matched_count: 0,
        data_as_of: "2026-11-23",
        universe: { eligible_universe_count: 500, universe_build_id: "UB-500", membership_hash: "h500" },
        coverage: { coverage_status: "COMPLETE", scanned_successfully_count: 500, stage_2_opportunity_count: 1 },
        freshness: { status: "LIVE", data_as_of: "2026-11-23" },
        publicationIntegrity: "VERIFIED",
        publicationIntegrityVerified: true,
      },
      runPayload: {
        status: "SUCCESS",
        scanState: "COMPLETED",
        datasetState: "CANDIDATES_AVAILABLE",
        totalCandidates: 500,
        gemsFound: 1,
        stage2OpportunityCount: 1,
        population: {
          referenceUniverseCount: 500,
          activeScreeningCount: 500,
          currentOpportunityCount: 1,
          stage2OpportunityCount: 1,
          delistedRecordCount: 0,
          unverifiedRecordCount: 0,
          displayedDataAsOf: "2026-11-23",
        },
        capabilities: {
          VALUE_GARP: {
            status: "AVAILABLE",
            universeScreening: "AVAILABLE",
            singleAssetAnalysis: "AVAILABLE",
            catalogType: "CURATED_REFERENCE_CATALOG",
            countAuthority: "DYNAMIC_CATALOG",
            matchedCount: 1,
          },
          VCP: {
            status: "AVAILABLE",
            universeScreening: "AVAILABLE",
            singleAssetAnalysis: "AVAILABLE",
            matchedCount: 0,
            freshnessStatus: "LIVE",
            populationDenominator: 500,
            universeBuildId: "UB-500",
            publicationDigest: "h500",
            dataAsOf: "2026-11-23",
            publicationIntegrity: "VERIFIED",
            publicationIntegrityVerified: true,
          },
        },
        candidates: [buildSampleCandidate("ONON", { categories: ["VALUE_GARP"] })],
      },
    });

    render(<RadarPage />);
    const zeroVcpTab = await screen.findByRole("tab", { name: /Minervini VCP/i });
    expect(zeroVcpTab.textContent).toContain("0");
    fireEvent.click(zeroVcpTab);
    await waitFor(() => {
      expect(screen.getByTestId("radar-desktop-empty-state").textContent).toContain("Zero Matching Contraction Setups (500 Evaluated)");
    });
    expect(screen.getByTestId("radar-mobile-empty-state").textContent).toContain("Zero Matching Contraction Setups (500 Evaluated)");
  });

  it("renders tape HTTP 500 error without fabricating 0 VALUE_GARP matches in tab tooltip or scope banner", async () => {
    mockRadarFetches({
      snapshotPayload: {
        scanner_id: "MINERVINI_VCP",
        status: "PIPELINE_PENDING",
        reason: "NO_ACTIVE_SNAPSHOT",
        recovery_condition: "Commit an authoritative verified VCP snapshot.",
      },
      runPayload: { error: "Upstream unavailable" },
      runStatus: 500,
    });

    render(<RadarPage />);
    await waitFor(() => {
      expect(screen.getByText(/Radar Tape Connection Error/i)).toBeDefined();
    });

    const vgTab = screen.getByRole("tab", { name: /Value \/ GARP/i });
    expect(vgTab.getAttribute("title") || "").not.toContain("0 Value / GARP matches");
    expect(vgTab.getAttribute("title") || "").toContain("unavailable");
  });
});
