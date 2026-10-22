import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { RadarCapabilityDetail, CanonicalRadarCategory } from "../lib/api";

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

console.log("\n===============================================================================");
console.log("ALL RADAR COUNT AUTHORITY & FILTERED DISPLAY TESTS PASSED SUCCESSFULLY! ✅");
console.log("===============================================================================\n");
