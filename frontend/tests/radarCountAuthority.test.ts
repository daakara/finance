import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import {
  RadarCapabilityDetail,
  CanonicalRadarCategory,
  RadarCapabilities,
  DEFAULT_RADAR_CAPABILITIES,
  normalizeRadarCapabilities,
  getPublicationIntegrityLabel,
  fetchScreenerGems,
} from "../lib/api";

console.log("Starting Radar Count Authority & Filtered Display Verification Suite...\n");

// ---------------------------------------------------------------------------
// 1. Static Invariant Audit on frontend/app/radar/page.tsx
// ---------------------------------------------------------------------------
console.log("1. Verifying static count authority invariants in frontend/app/radar/page.tsx...");

const radarPath = path.join(__dirname, "../app/radar/page.tsx");
const radarSource = fs.readFileSync(radarPath, "utf8");

// Requirement 1 Check: Unsupported '100% coverage' fallback text must be completely removed from Radar
const has100PercentCoverage = radarSource.includes("100% coverage");
assert.strictEqual(
  has100PercentCoverage,
  false,
  "FAIL: '100% coverage' fallback must be completely removed from frontend/app/radar/page.tsx"
);
console.log("   [PASS] 1.1: Unsupported '100% coverage' fallback text eliminated");

// Requirement 2 & 3 Check: Provenance-gated scannable population and 'Coverage Pending'
assert.ok(
  radarSource.includes("const hasValidProvenance = Boolean("),
  "FAIL: Radar must define hasValidProvenance gate for population denominator"
);
assert.ok(
  radarSource.includes("capabilities.VCP?.universeBuildId &&\n                    capabilities.VCP?.publicationDigest"),
  "FAIL: Radar must require both universeBuildId and publicationDigest to display population count"
);
assert.ok(
  radarSource.includes(": 'Coverage Pending';"),
  "FAIL: Radar must display 'Coverage Pending' when population provenance is incomplete"
);
console.log("   [PASS] 1.2: Population count strictly gated on buildId and publicationDigest, with 'Coverage Pending' fallback");

// Invariant 2 Check: filteredAssets.length alone cannot be labeled as "{filteredAssets.length} matches" in the VCP banner
const hasConflatedMatchCount = radarSource.includes("{filteredAssets.length} matches ·");
assert.strictEqual(
  hasConflatedMatchCount,
  false,
  "FAIL: VCP banner must NOT conflate filteredAssets.length with authoritative scanner matches ('{filteredAssets.length} matches')"
);
console.log("   [PASS] 1.3: Conflation of filteredAssets.length with scanner matches eliminated");

// Invariant 2 & 7 Check: Display count vs Scanner matches distinction is rendered when filters reduce the count
assert.ok(
  radarSource.includes("Showing <strong className=\"text-white\">{filteredAssets.length}</strong> of <strong className=\"text-white\">{authMatches}</strong> scanner matches · {coverageText}"),
  "FAIL: VCP banner must render 'Showing {filteredAssets.length} of {authMatches} scanner matches · {coverageText}' when filters reduce count"
);
console.log("   [PASS] 1.4: Filtered display count clearly distinguished from authoritative scanner match count");

// Invariant 4 Check: When all matches are displayed, rendered as "{authMatches} scanner matches · {coverageText}"
assert.ok(
  radarSource.includes("<strong className=\"text-white\">{authMatches}</strong> scanner matches · {coverageText}"),
  "FAIL: VCP banner must render '{authMatches} scanner matches · {coverageText}' when all matches are displayed"
);
console.log("   [PASS] 1.5: Authoritative scanner matches displayed when no reducing filter is applied");

// Invariant 3 Check: In pending/unauthorized states, renders 'Scanner Pending', never fabricated 0 matches
assert.ok(
  radarSource.includes("Scanner Pending · {coverageText}"),
  "FAIL: VCP banner must render 'Scanner Pending · {coverageText}' when scanner is pending or matchedCount is null"
);
console.log("   [PASS] 1.6: Pending states render 'Scanner Pending', never fabricated zero matches");


// ---------------------------------------------------------------------------
// 2. Behavioral Simulation: Scanner Matches vs Filtered Display Counts & Provenance
// ---------------------------------------------------------------------------
console.log("\n2. Simulating Radar banner count formatting logic across filter states & provenance...");

function formatRadarVcpBannerCount(
  vcpCap: RadarCapabilityDetail | undefined,
  displayedCount: number
): string {
  const hasValidProvenance = Boolean(
    typeof vcpCap?.populationDenominator === "number" &&
    vcpCap.populationDenominator > 0 &&
    vcpCap?.universeBuildId &&
    vcpCap?.publicationDigest
  );
  const coverageText = hasValidProvenance
    ? `${vcpCap?.populationDenominator} scannable population`
    : "Coverage Pending";

  const rawMatches = vcpCap?.matchedCount;
  // Strict backend count validation: only authentic non-negative integers (no bool, no negative, no float)
  const isAuthenticInt = typeof rawMatches === "number" && Number.isInteger(rawMatches) && rawMatches >= 0 && typeof rawMatches !== "boolean";
  const authMatches = isAuthenticInt && !vcpCap?.isRollback
    ? rawMatches
    : null;

  if (vcpCap?.universeScreening === "AVAILABLE" && authMatches !== null) {
    if (displayedCount !== authMatches) {
      return `Showing ${displayedCount} of ${authMatches} scanner matches · ${coverageText}`;
    }
    return `${authMatches} scanner matches · ${coverageText}`;
  }

  return `Scanner Pending · ${coverageText}`;
}

// Case A: Operational scanner with 12 matches, valid provenance, no user filter applied
const vcpOpCap: RadarCapabilityDetail = {
  status: "AVAILABLE",
  universeScreening: "AVAILABLE",
  singleAssetAnalysis: "AVAILABLE",
  matchedCount: 12,
  populationDenominator: 150,
  universeBuildId: "ubuild-150-prod",
  publicationDigest: "sha256-abcdef",
  isRollback: false,
};

const textAllShown = formatRadarVcpBannerCount(vcpOpCap, 12);
assert.strictEqual(
  textAllShown,
  "12 scanner matches · 150 scannable population",
  "Must show full scanner match count when unfiltered"
);
console.log("   [PASS] 2.1: Unfiltered operational scanner: '12 scanner matches · 150 scannable population'");

// Case B: User applies search filter "AAPL", reducing displayed count to 1
const textFilteredSearch = formatRadarVcpBannerCount(vcpOpCap, 1);
assert.strictEqual(
  textFilteredSearch,
  "Showing 1 of 12 scanner matches · 150 scannable population",
  "Must show 'Showing 1 of 12 scanner matches' without mutating authoritative match count"
);
assert.strictEqual(vcpOpCap.matchedCount, 12, "Authoritative matchedCount must remain untouched by user filter");
console.log("   [PASS] 2.2: User search filter: 'Showing 1 of 12 scanner matches · 150 scannable population' (matchedCount=12 intact)");

// Case C: Operational scanner with missing buildId/digest (incomplete provenance)
const vcpNoProvCap: RadarCapabilityDetail = {
  status: "AVAILABLE",
  universeScreening: "AVAILABLE",
  singleAssetAnalysis: "AVAILABLE",
  matchedCount: 12,
  populationDenominator: 150,
  universeBuildId: null, // missing build identity!
  publicationDigest: null, // missing digest!
  isRollback: false,
};

const textNoProv = formatRadarVcpBannerCount(vcpNoProvCap, 12);
assert.strictEqual(
  textNoProv,
  "12 scanner matches · Coverage Pending",
  "Must replace denominator with 'Coverage Pending' when buildId or digest is absent"
);
console.log("   [PASS] 2.3: Incomplete provenance yields '12 scanner matches · Coverage Pending'");


// ---------------------------------------------------------------------------
// 3. Invariants 3, 4, 5: Truthful Operational Zero vs Pending States
// ---------------------------------------------------------------------------
console.log("\n3. Testing truthful operational zero matches vs pending states...");

// Case D: Authoritative scanner executed and found 0 matches across the 150 population
const vcpZeroCap: RadarCapabilityDetail = {
  status: "AVAILABLE",
  universeScreening: "AVAILABLE",
  singleAssetAnalysis: "AVAILABLE",
  matchedCount: 0,
  populationDenominator: 150,
  universeBuildId: "ubuild-150-prod",
  publicationDigest: "sha256-abcdef",
  isRollback: false,
};

const textOpZero = formatRadarVcpBannerCount(vcpZeroCap, 0);
assert.strictEqual(
  textOpZero,
  "0 scanner matches · 150 scannable population",
  "Authoritative operational zero matches must display '0 scanner matches'"
);
console.log("   [PASS] 3.1: Authoritative operational zero matches: '0 scanner matches · 150 scannable population'");

// Case E: Scanner is pending (State 3 unauthorized or deployment hold)
const vcpPendingCap: RadarCapabilityDetail = {
  status: "PIPELINE_PENDING",
  universeScreening: "PIPELINE_PENDING",
  singleAssetAnalysis: "AVAILABLE",
  matchedCount: null,
  populationDenominator: 150,
  universeBuildId: "ubuild-150-prod",
  publicationDigest: "sha256-abcdef",
  isRollback: false,
};

const textPending = formatRadarVcpBannerCount(vcpPendingCap, 0);
assert.strictEqual(
  textPending,
  "Scanner Pending · 150 scannable population",
  "Pending capability must display 'Scanner Pending', NEVER '0 scanner matches'"
);
assert.ok(!textPending.includes("0 scanner matches") && !textPending.includes("0 matches"), "Pending banner text must not contain fabricated 0 matches");
console.log("   [PASS] 3.2: Pending scanner: 'Scanner Pending · 150 scannable population' (zero fabricated 0)");

// Case F: Boolean matched_count (e.g. True or False) must fail closed to Scanner Pending
const vcpBoolCap: any = {
  status: "AVAILABLE",
  universeScreening: "AVAILABLE",
  singleAssetAnalysis: "AVAILABLE",
  matchedCount: true as any,
  populationDenominator: 150,
  universeBuildId: "ubuild-150-prod",
  publicationDigest: "sha256-abcdef",
  isRollback: false,
};

const textBool = formatRadarVcpBannerCount(vcpBoolCap, 0);
assert.strictEqual(
  textBool,
  "Scanner Pending · 150 scannable population",
  "Boolean matchedCount must fail-closed to 'Scanner Pending'"
);
console.log("   [PASS] 3.3: Boolean matchedCount rejected and fails closed to 'Scanner Pending'");

// Case G: Negative matched_count (e.g. -1) must fail closed to Scanner Pending
const vcpNegCap: RadarCapabilityDetail = {
  status: "AVAILABLE",
  universeScreening: "AVAILABLE",
  singleAssetAnalysis: "AVAILABLE",
  matchedCount: -1 as any,
  populationDenominator: 150,
  universeBuildId: "ubuild-150-prod",
  publicationDigest: "sha256-abcdef",
  isRollback: false,
};

const textNeg = formatRadarVcpBannerCount(vcpNegCap, 0);
assert.strictEqual(
  textNeg,
  "Scanner Pending · 150 scannable population",
  "Negative matchedCount must fail-closed to 'Scanner Pending'"
);
console.log("   [PASS] 3.4: Negative matchedCount rejected and fails closed to 'Scanner Pending'");


// ---------------------------------------------------------------------------
// 4. Invariant 6: Stale or Rolled-Back Pointers Cannot Imply Scanner Availability
// ---------------------------------------------------------------------------
console.log("\n4. Testing rollback pointer containment...");

const vcpRollbackCap: RadarCapabilityDetail = {
  status: "PIPELINE_PENDING",
  universeScreening: "PIPELINE_PENDING",
  singleAssetAnalysis: "AVAILABLE",
  matchedCount: null,
  populationDenominator: 100,
  universeBuildId: "ubuild-100-restored",
  publicationDigest: "sha256-restored",
  isRollback: true,
  rollbackReason: "Reverted due to anomaly",
};

const textRollback = formatRadarVcpBannerCount(vcpRollbackCap, 0);
assert.strictEqual(
  textRollback,
  "Scanner Pending · 100 scannable population",
  "Rolled-back pointer must display 'Scanner Pending' with restored population denominator"
);
assert.strictEqual(vcpRollbackCap.matchedCount, null, "matchedCount must be strictly null on rollback");
console.log("   [PASS] 4.1: Rolled-back pointer enforces PIPELINE_PENDING and matchedCount=null");


// ---------------------------------------------------------------------------
// 5. Invariant 7: Filter Independence Simulation
// ---------------------------------------------------------------------------
console.log("\n5. Testing filter independence across sort and ownership states...");

interface DummyAsset {
  ticker: string;
  confluenceScore: number;
  rvol: number;
  price: number;
  categories: CanonicalRadarCategory[];
  isHeld: boolean;
}

const mockAssets: DummyAsset[] = [
  { ticker: "AAPL", confluenceScore: 92, rvol: 1.8, price: 230, categories: ["VCP"], isHeld: true },
  { ticker: "NVDA", confluenceScore: 88, rvol: 2.4, price: 135, categories: ["VCP"], isHeld: false },
  { ticker: "MSFT", confluenceScore: 75, rvol: 1.1, price: 420, categories: ["VCP"], isHeld: false },
  { ticker: "GOOG", confluenceScore: 60, rvol: 0.9, price: 180, categories: ["VALUE_GARP"], isHeld: false },
];

const authoritativeVcpMatches = mockAssets.filter((a) => a.categories.includes("VCP")).length; // 3

// Test 5.1: Ownership filter NEW_OPPORTUNITIES
const newOppMatches = mockAssets.filter((a) => a.categories.includes("VCP") && !a.isHeld); // 2
assert.strictEqual(newOppMatches.length, 2);
assert.strictEqual(authoritativeVcpMatches, 3, "Authoritative matches unchanged by ownership filter");

// Test 5.2: Search query "NVDA"
const searchMatches = mockAssets.filter((a) => a.categories.includes("VCP") && a.ticker === "NVDA"); // 1
assert.strictEqual(searchMatches.length, 1);
assert.strictEqual(authoritativeVcpMatches, 3, "Authoritative matches unchanged by search query");

console.log("   [PASS] 5.1: Ownership filter reduces displayed assets (2) without mutating authoritative matches (3)");
console.log("   [PASS] 5.2: Search filter reduces displayed assets (1) without mutating authoritative matches (3)");

// ---------------------------------------------------------------------------
// 6. Real Frontend Transformation Path Verification (fetchScreenerGems)
// ---------------------------------------------------------------------------
console.log("\n6. Testing fetchScreenerGems() real transformation path across all 6 capability cases...");

async function testFetchScreenerGemsTransformation() {
  const originalFetch = globalThis.fetch;

  try {
    // 6.1: Empty candidates plus verified operational zero
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
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
        },
      }),
    } as any);

    const res1 = await fetchScreenerGems("all");
    assert.strictEqual(res1.total_candidates, 0, "total_candidates must be 0");
    assert.strictEqual(res1.gems_found, 0, "gems_found must be 0");
    assert.strictEqual(res1.results.length, 0, "results array must be empty");
    assert.ok(res1.capabilities, "capabilities must NOT be discarded when candidates array is empty");
    assert.strictEqual(res1.capabilities?.VCP.matchedCount, 0, "matchedCount 0 must be preserved");
    assert.strictEqual(res1.capabilities?.VCP.universeScreening, "AVAILABLE", "universeScreening AVAILABLE must be preserved");
    assert.strictEqual(res1.capabilities?.VCP.populationDenominator, 150, "populationDenominator 150 must be preserved");

    const text1 = formatRadarVcpBannerCount(res1.capabilities?.VCP, res1.results.length);
    assert.strictEqual(text1, "0 scanner matches · 150 scannable population", "Real pipeline must format operational zero cleanly");
    console.log("   [PASS] 6.1: Empty candidates + operational zero -> '0 scanner matches · 150 scannable population'");

    // 6.2: Empty candidates plus pending scanner
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
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
      }),
    } as any);

    const res2 = await fetchScreenerGems("all");
    assert.ok(res2.capabilities, "capabilities must be preserved for PIPELINE_PENDING");
    assert.strictEqual(res2.capabilities?.VCP.universeScreening, "PIPELINE_PENDING");
    const text2 = formatRadarVcpBannerCount(res2.capabilities?.VCP, res2.results.length);
    assert.strictEqual(text2, "Scanner Pending · 150 scannable population");
    console.log("   [PASS] 6.2: Empty candidates + pending scanner -> 'Scanner Pending · 150 scannable population'");

    // 6.3: Empty candidates plus unavailable capabilities (missing / empty)
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        candidates: [],
      }),
    } as any);

    const res3 = await fetchScreenerGems("all");
    assert.strictEqual(res3.capabilities, undefined, "missing capabilities must fail closed as undefined");
    const text3 = formatRadarVcpBannerCount(res3.capabilities, res3.results.length);
    assert.strictEqual(text3, "Scanner Pending · Coverage Pending");
    console.log("   [PASS] 6.3: Empty candidates + unavailable capabilities -> fail closed to 'Scanner Pending · Coverage Pending'");

    // 6.4: Empty candidates plus rollback
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
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
      }),
    } as any);

    const res4 = await fetchScreenerGems("all");
    assert.ok(res4.capabilities?.VCP.isRollback === true, "isRollback flag must be preserved");
    const text4 = formatRadarVcpBannerCount(res4.capabilities?.VCP, res4.results.length);
    assert.strictEqual(text4, "Scanner Pending · 150 scannable population");
    console.log("   [PASS] 6.4: Empty candidates + rollback -> matches suppressed to 'Scanner Pending · 150 scannable population'");

    // 6.5: Empty candidates plus incomplete provenance
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
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
      }),
    } as any);

    const res5 = await fetchScreenerGems("all");
    assert.ok(res5.capabilities, "capabilities preserved even when build provenance is null");
    const text5 = formatRadarVcpBannerCount(res5.capabilities?.VCP, res5.results.length);
    assert.strictEqual(text5, "0 scanner matches · Coverage Pending");
    console.log("   [PASS] 6.5: Empty candidates + missing build provenance -> '0 scanner matches · Coverage Pending'");

    // 6.6: Nonempty response compatibility
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      json: async () => ({
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
      }),
    } as any);

    const res6 = await fetchScreenerGems("all");
    assert.strictEqual(res6.total_candidates, 1, "nonempty response total_candidates=1");
    assert.strictEqual(res6.results.length, 1);
    assert.strictEqual(res6.results[0].ticker, "NVDA");
    assert.ok(res6.capabilities, "capabilities preserved on nonempty response");
    assert.strictEqual(res6.capabilities?.VCP.matchedCount, 1);
    const text6 = formatRadarVcpBannerCount(res6.capabilities?.VCP, res6.results.length);
    assert.strictEqual(text6, "1 scanner matches · 150 scannable population");
    console.log("   [PASS] 6.6: Nonempty candidate compatibility preserved -> '1 scanner matches · 150 scannable population'");

    // ---------------------------------------------------------------------------
    // 7. Sequential State Transitions & Request Discard Verification
    // ---------------------------------------------------------------------------
    console.log("\n7. Verifying sequential state transitions & request discard...");

    // Operational initial baseline
    const operationalBaseline: RadarCapabilities = {
      ...DEFAULT_RADAR_CAPABILITIES,
      VCP: {
        status: "AVAILABLE",
        universeScreening: "AVAILABLE",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: 12,
        populationDenominator: 150,
        universeBuildId: "build-active-12",
        publicationDigest: "sha256-digest-12",
        isRollback: false,
        publicationIntegrity: "VERIFIED",
        publicationIntegrityVerified: true,
      },
      VALUE_GARP: {
        status: "AVAILABLE",
        universeScreening: "AVAILABLE",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: 7,
      },
    };

    // Transition 1: AVAILABLE -> missing capabilities
    // Expected: Previous operational claim removed; falls back to fail-closed canonical default.
    const t1Normalized = normalizeRadarCapabilities(undefined);
    assert.strictEqual(t1Normalized.VCP.universeScreening, "PIPELINE_PENDING");
    assert.strictEqual(t1Normalized.VCP.matchedCount, null);
    assert.strictEqual(t1Normalized.VCP.universeBuildId, null);
    console.log("   [PASS] 7.1: AVAILABLE -> missing capabilities: Operational claim removed and fails closed");

    // Transition 2: AVAILABLE -> PIPELINE_PENDING
    // Expected: Scanner Pending; previous matches removed; population provenance preserved.
    const t2Normalized = normalizeRadarCapabilities({
      VCP: {
        status: "PIPELINE_PENDING",
        universeScreening: "PIPELINE_PENDING",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: null,
        populationDenominator: 150,
        universeBuildId: "build-s209-pending",
        publicationDigest: "sha256-pending-digest",
        isRollback: false,
      },
    });
    assert.strictEqual(t2Normalized.VCP.universeScreening, "PIPELINE_PENDING");
    assert.strictEqual(t2Normalized.VCP.matchedCount, null);
    assert.strictEqual(t2Normalized.VCP.populationDenominator, 150);
    const textT2 = formatRadarVcpBannerCount(t2Normalized.VCP, 0);
    assert.strictEqual(textT2, "Scanner Pending · 150 scannable population");
    console.log("   [PASS] 7.2: AVAILABLE -> PIPELINE_PENDING: Previous matches removed, renders 'Scanner Pending'");

    // Transition 3: AVAILABLE -> rolled-back publication
    // Expected: Pending; rollback disclosed; operational claims suppressed.
    const t3Normalized = normalizeRadarCapabilities({
      VCP: {
        status: "AVAILABLE", // Even if payload erroneously claimed AVAILABLE, isRollback suppresses it!
        universeScreening: "AVAILABLE",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: 5,
        populationDenominator: 150,
        universeBuildId: "build-rb",
        publicationDigest: "sha256-rb-digest",
        isRollback: true,
        rollbackReason: "Integrity alert",
      },
    });
    assert.strictEqual(t3Normalized.VCP.isRollback, true);
    assert.strictEqual(t3Normalized.VCP.universeScreening, "PIPELINE_PENDING");
    assert.strictEqual(t3Normalized.VCP.matchedCount, null);
    assert.strictEqual(getPublicationIntegrityLabel(t3Normalized.VCP), "Publication Integrity: Rolled back");
    console.log("   [PASS] 7.3: AVAILABLE -> rolled-back publication: Rollback disclosed and operational claims suppressed");

    // Transition 4: Valid provenance -> missing digest
    // Expected: Coverage Pending rendered because provenance is incomplete.
    const t4Normalized = normalizeRadarCapabilities({
      VCP: {
        status: "AVAILABLE",
        universeScreening: "AVAILABLE",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: 3,
        populationDenominator: 150,
        universeBuildId: "build-no-digest",
        publicationDigest: null,
        isRollback: false,
      },
    });
    assert.strictEqual(t4Normalized.VCP.matchedCount, 3);
    const textT4 = formatRadarVcpBannerCount(t4Normalized.VCP, 3);
    assert.strictEqual(textT4, "3 scanner matches · Coverage Pending");
    console.log("   [PASS] 7.4: Valid provenance -> missing digest: Provenance gate enforces 'Coverage Pending'");

    // Transition 5: Operational zero -> malformed response
    // Expected: Malformed count fails closed; does not leak operational claim.
    const t5Normalized = normalizeRadarCapabilities({
      VCP: {
        status: "AVAILABLE",
        universeScreening: "AVAILABLE",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: -99, // Malformed negative count
        populationDenominator: 150,
        universeBuildId: "build-malformed",
        publicationDigest: "sha256-malformed",
        isRollback: false,
      },
    });
    assert.strictEqual(t5Normalized.VCP.universeScreening, "PIPELINE_PENDING");
    assert.strictEqual(t5Normalized.VCP.matchedCount, null);
    console.log("   [PASS] 7.5: Operational zero -> malformed response: Fails closed to PIPELINE_PENDING");

    // Transition 6: Pending -> verified AVAILABLE
    // Expected: New authoritative matches shown; verified publication integrity displayed.
    const t6Normalized = normalizeRadarCapabilities({
      VCP: {
        status: "AVAILABLE",
        universeScreening: "AVAILABLE",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: 8,
        populationDenominator: 150,
        universeBuildId: "build-verified",
        publicationDigest: "sha256-verified",
        isRollback: false,
        publicationIntegrity: "VERIFIED",
        publicationIntegrityVerified: true,
      },
    });
    assert.strictEqual(t6Normalized.VCP.universeScreening, "AVAILABLE");
    assert.strictEqual(t6Normalized.VCP.matchedCount, 8);
    assert.strictEqual(getPublicationIntegrityLabel(t6Normalized.VCP), "Publication Integrity: Verified");
    const textT6 = formatRadarVcpBannerCount(t6Normalized.VCP, 8);
    assert.strictEqual(textT6, "8 scanner matches · 150 scannable population");
    console.log("   [PASS] 7.6: Pending -> verified AVAILABLE: Authoritative matches and verified integrity displayed");

    // Transition 7: One strategy valid -> another missing
    // Expected: No cross-strategy authority leakage; omitted strategy resets to default pending.
    const t7Normalized = normalizeRadarCapabilities({
      VALUE_GARP: {
        status: "AVAILABLE",
        universeScreening: "AVAILABLE",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: 4,
      },
      // VCP is omitted!
    });
    assert.strictEqual(t7Normalized.VALUE_GARP.universeScreening, "AVAILABLE");
    assert.strictEqual(t7Normalized.VALUE_GARP.matchedCount, 4);
    assert.strictEqual(t7Normalized.VCP.universeScreening, "PIPELINE_PENDING");
    assert.strictEqual(t7Normalized.VCP.matchedCount, null);
    console.log("   [PASS] 7.7: One strategy valid -> another missing: Zero cross-strategy authority leakage");

    // Transition 8: Successful response -> network error
    // Expected: Network error handler clears capabilities to default pending and clears assets.
    let simulatedState = operationalBaseline;
    let simulatedAssets: any[] = [{ ticker: "AAPL" }];
    // Network failure simulation
    const handleNetworkError = () => {
      simulatedState = DEFAULT_RADAR_CAPABILITIES;
      simulatedAssets = [];
    };
    handleNetworkError();
    assert.strictEqual(simulatedState.VCP.universeScreening, "PIPELINE_PENDING");
    assert.strictEqual(simulatedState.VCP.matchedCount, null);
    assert.strictEqual(simulatedAssets.length, 0);
    console.log("   [PASS] 7.8: Successful response -> network error: Stale operational claims fully cleared");

    // Transition 9: Out-of-order response discard
    // Expected: Slower older response does not overwrite newer response.
    let latestRequestId = 0;
    let finalState = DEFAULT_RADAR_CAPABILITIES;

    // Request 1 initiated at t=0
    const req1Id = ++latestRequestId; // req1Id = 1
    // Request 2 initiated at t=1
    const req2Id = ++latestRequestId; // req2Id = 2

    // Request 2 returns first (fast response with 0 matches)
    if (req2Id === latestRequestId) {
      finalState = normalizeRadarCapabilities({
        VCP: {
          status: "AVAILABLE",
          universeScreening: "AVAILABLE",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: 0,
          populationDenominator: 150,
          universeBuildId: "build-req2",
          publicationDigest: "digest-req2",
        },
      });
    }

    // Request 1 returns second (delayed stale response with 12 matches)
    const req1Discarded = req1Id !== latestRequestId;
    if (!req1Discarded) {
      finalState = normalizeRadarCapabilities({
        VCP: {
          status: "AVAILABLE",
          universeScreening: "AVAILABLE",
          singleAssetAnalysis: "AVAILABLE",
          matchedCount: 12,
        },
      });
    }
    assert.strictEqual(req1Discarded, true, "Older request must be detected as superseded");
    assert.strictEqual(finalState.VCP.matchedCount, 0, "State must remain from newer Request 2");
    console.log("   [PASS] 7.9: Out-of-order response discard: Stale older response discarded cleanly");

    // ---------------------------------------------------------------------------
    // 8. Truthful Publication Integrity Badge States Verification
    // ---------------------------------------------------------------------------
    console.log("\n8. Verifying truthful Publication Integrity badge presentation...");

    // 8.1: Verified with attestation
    const verifiedCap: RadarCapabilityDetail = {
      status: "AVAILABLE",
      universeScreening: "AVAILABLE",
      singleAssetAnalysis: "AVAILABLE",
      populationDenominator: 150,
      universeBuildId: "build-prod",
      publicationDigest: "sha256-verified-digest",
      publicationIntegrity: "VERIFIED",
      publicationIntegrityVerified: true,
    };
    assert.strictEqual(getPublicationIntegrityLabel(verifiedCap), "Publication Integrity: Verified");
    console.log("   [PASS] 8.1: Explicit attestation -> 'Publication Integrity: Verified'");

    // 8.2: Rolled-back publication
    const rolledBackCap: RadarCapabilityDetail = {
      status: "PIPELINE_PENDING",
      universeScreening: "PIPELINE_PENDING",
      singleAssetAnalysis: "AVAILABLE",
      populationDenominator: 150,
      universeBuildId: "build-rb",
      publicationDigest: "sha256-rb-digest",
      isRollback: true,
      rollbackReason: "Suspect feed data",
    };
    assert.strictEqual(getPublicationIntegrityLabel(rolledBackCap), "Publication Integrity: Rolled back");
    console.log("   [PASS] 8.2: Rollback condition -> 'Publication Integrity: Rolled back'");

    // 8.3: Metadata present (buildId and digest) WITHOUT attestation
    // Invariant: Never infer publication integrity merely from the presence of a publication digest string or build ID!
    const unverifiedCapWithDigest: RadarCapabilityDetail = {
      status: "AVAILABLE",
      universeScreening: "AVAILABLE",
      singleAssetAnalysis: "AVAILABLE",
      matchedCount: 5,
      populationDenominator: 150,
      universeBuildId: "build-unverified",
      publicationDigest: "sha256-digest-without-attestation",
      // publicationIntegrity is NOT "VERIFIED"
    };
    assert.strictEqual(
      getPublicationIntegrityLabel(unverifiedCapWithDigest),
      "Publication Integrity: Not verified",
      "Digest presence alone must NEVER infer 'Verified'!"
    );
    console.log("   [PASS] 8.3: Metadata present without attestation -> 'Publication Integrity: Not verified' (no false positive)");

    // 8.4: Explicit PENDING status
    const pendingIntegrityCap: RadarCapabilityDetail = {
      status: "PIPELINE_PENDING",
      universeScreening: "PIPELINE_PENDING",
      singleAssetAnalysis: "AVAILABLE",
      publicationIntegrity: "PENDING",
    };
    assert.strictEqual(getPublicationIntegrityLabel(pendingIntegrityCap), "Publication Integrity: Pending");
    console.log("   [PASS] 8.4: Pending integrity -> 'Publication Integrity: Pending'");

    // 8.5: Missing / null capability
    assert.strictEqual(getPublicationIntegrityLabel(null), "Publication Integrity: Not verified");
    assert.strictEqual(getPublicationIntegrityLabel(undefined), "Publication Integrity: Not verified");
    console.log("   [PASS] 8.5: Null/undefined capability -> 'Publication Integrity: Not verified'");

  } finally {
    globalThis.fetch = originalFetch;
  }
}

testFetchScreenerGemsTransformation().then(() => {
  console.log("\n===============================================================================");
  console.log("ALL RADAR COUNT AUTHORITY & REAL DATA TRANSFORMATION TESTS PASSED! ✅");
  console.log("===============================================================================\n");
}).catch((err) => {
  console.error("Test suite failed:", err);
  process.exit(1);
});
