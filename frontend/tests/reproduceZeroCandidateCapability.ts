import assert from "node:assert";
import { fetchScreenerGems, ScreenerResponse, RadarCapabilities } from "../lib/api";

interface TestCase {
  id: number;
  name: string;
  mockResponseBody: any;
  mockStatus?: number;
}

const testCases: TestCase[] = [
  {
    id: 1,
    name: "Zero candidates with valid operational scanner capability and matchedCount: 0",
    mockResponseBody: {
      candidates: [],
      capabilities: {
        VCP: {
          status: "AVAILABLE",
          universeScreening: "AVAILABLE",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: 0,
          populationDenominator: 150,
          universeBuildId: "build-s209-150-prod",
          publicationDigest: "sha256-verified-digest",
          isRollback: false,
        },
        VALUE_GARP: {
          status: "AVAILABLE",
          universeScreening: "AVAILABLE",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: 0,
        },
        SMART_MONEY: {
          status: "PIPELINE_PENDING",
          universeScreening: "PIPELINE_PENDING",
          singleAssetAnalysis: "AVAILABLE",
        },
      },
    },
  },
  {
    id: 2,
    name: "Zero candidates with PIPELINE_PENDING",
    mockResponseBody: {
      candidates: [],
      capabilities: {
        VCP: {
          status: "PIPELINE_PENDING",
          universeScreening: "PIPELINE_PENDING",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: null,
          populationDenominator: 150,
          universeBuildId: "build-s209-150-prod",
          publicationDigest: "sha256-verified-digest",
          isRollback: false,
        },
      },
    },
  },
  {
    id: 3,
    name: "Zero candidates with missing capabilities",
    mockResponseBody: {
      candidates: [],
    },
  },
  {
    id: 4,
    name: "Zero candidates with a rolled-back publication pointer",
    mockResponseBody: {
      candidates: [],
      capabilities: {
        VCP: {
          status: "PIPELINE_PENDING",
          universeScreening: "PIPELINE_PENDING",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: null,
          populationDenominator: 150,
          universeBuildId: "build-s209-150-rollback",
          publicationDigest: "sha256-rollback-digest",
          isRollback: true,
          rollbackReason: "Reverted due to anomaly",
        },
      },
    },
  },
  {
    id: 5,
    name: "Zero candidates with valid count but missing denominator provenance",
    mockResponseBody: {
      candidates: [],
      capabilities: {
        VCP: {
          status: "AVAILABLE",
          universeScreening: "AVAILABLE",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: 0,
          populationDenominator: 150,
          universeBuildId: null,
          publicationDigest: null,
          isRollback: false,
        },
      },
    },
  },
  {
    id: 6,
    name: "Nonzero candidates with valid operational capability",
    mockResponseBody: {
      candidates: [
        {
          ticker: "NVDA",
          companyName: "NVIDIA Corporation",
          categories: ["VCP"],
          confluenceScore: 92,
          gemScore: 88,
          peg_ratio: 1.4,
          executionStatus: "IN_BUY_ZONE",
        },
      ],
      capabilities: {
        VCP: {
          status: "AVAILABLE",
          universeScreening: "AVAILABLE",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: 1,
          populationDenominator: 150,
          universeBuildId: "build-s209-150-prod",
          publicationDigest: "sha256-verified-digest",
          isRollback: false,
        },
      },
    },
  },
];

async function runReproduction() {
  console.log("=== PHASE 1 REPRODUCTION: fetchScreenerGems() Output Audit ===\n");
  const results: Record<number, { name: string; output: ScreenerResponse; hasCapabilities: boolean }> = {};

  for (const tc of testCases) {
    // Mock globalThis.fetch
    globalThis.fetch = async () => {
      return {
        ok: tc.mockStatus ? tc.mockStatus === 200 : true,
        status: tc.mockStatus || 200,
        json: async () => tc.mockResponseBody,
      } as any;
    };

    const actual = await fetchScreenerGems("all");
    const hasCapabilities = actual.capabilities !== undefined;
    results[tc.id] = {
      name: tc.name,
      output: actual,
      hasCapabilities,
    };

    console.log(`--- Case ${tc.id}: ${tc.name} ---`);
    console.log("Input Fixture:", JSON.stringify(tc.mockResponseBody, null, 2));
    console.log("fetchScreenerGems() Output:", JSON.stringify(actual, null, 2));
    console.log(`Capabilities preserved in output? ${hasCapabilities ? "YES" : "NO (DEFECT CONFIRMED)"}`);
    console.log("\n");
  }

  return results;
}

runReproduction().catch((err) => {
  console.error("Reproduction failed:", err);
  process.exit(1);
});
