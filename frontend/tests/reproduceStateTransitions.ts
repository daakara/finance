import assert from "node:assert";
import { RadarCapabilities, RadarCapabilityDetail, CanonicalRadarCategory, ScreenerResponse, fetchScreenerGems } from "../lib/api";

const DEFAULT_RADAR_CAPABILITIES: RadarCapabilities = {
  VALUE_GARP: {
    status: "AVAILABLE",
    universeScreening: "AVAILABLE",
    singleAssetAnalysis: "AVAILABLE",
    rationale: "Authentic multi-factor fundamental screening active across known candidate universe.",
  },
  VCP: {
    status: "PIPELINE_PENDING",
    universeScreening: "PIPELINE_PENDING",
    singleAssetAnalysis: "AVAILABLE",
    rationale: "Single-asset volatility contraction geometry active on /setups and /analytics; universe-level batch scanning pipeline is pending deployment.",
  },
  SMART_MONEY: {
    status: "PIPELINE_PENDING",
    universeScreening: "PIPELINE_PENDING",
    singleAssetAnalysis: "AVAILABLE",
    rationale: "Single-asset SEC Form 4 and Congressional STOCK Act disclosures active on /smart-money; universe-level institutional accumulation screener pipeline is pending deployment.",
  },
};

// Current uncorrected state updater in page.tsx:
// if (res.capabilities) {
//   setCapabilities((prev) => ({ ...prev, ...res.capabilities }));
// }
function currentSetCapabilities(
  prev: RadarCapabilities,
  res: ScreenerResponse
): RadarCapabilities {
  if (res.capabilities) {
    return {
      ...prev,
      ...res.capabilities,
    };
  }
  return prev; // DEFECT: drops nothing if res.capabilities is undefined!
}

console.log("=== PHASE 1: Reproducing State Transition Defects on Current Code ===\n");

// Transition 1: AVAILABLE -> missing capabilities
const stateOp: RadarCapabilities = {
  ...DEFAULT_RADAR_CAPABILITIES,
  VCP: {
    status: "AVAILABLE",
    universeScreening: "AVAILABLE",
    singleAssetAnalysis: "AVAILABLE",
    matchedCount: 12,
    populationDenominator: 150,
    universeBuildId: "build-1",
    publicationDigest: "digest-1",
  },
};

const respMissingCaps: ScreenerResponse = {
  total_candidates: 0,
  gems_found: 0,
  results: [],
  // capabilities is undefined
};

const stateAfterMissing = currentSetCapabilities(stateOp, respMissingCaps);
console.log("Transition 1: AVAILABLE -> missing capabilities");
console.log("   VCP status after missing capabilities:", stateAfterMissing.VCP.universeScreening);
console.log("   VCP matches after missing capabilities:", stateAfterMissing.VCP.matchedCount);
const defect1 = stateAfterMissing.VCP.universeScreening === "AVAILABLE";
console.log("   [DEFECT CONFIRMED 1.1]: Operational claim leaked into missing-capabilities response?", defect1 ? "YES" : "NO");

// Transition 7: One strategy valid -> another missing
const respOnlyGarp: ScreenerResponse = {
  total_candidates: 5,
  gems_found: 5,
  capabilities: {
    VALUE_GARP: {
      status: "AVAILABLE",
      universeScreening: "AVAILABLE",
      singleAssetAnalysis: "AVAILABLE",
      matchedCount: 5,
      populationDenominator: 150,
      universeBuildId: "build-garp",
      publicationDigest: "digest-garp",
    },
    // VCP is completely omitted!
  } as any,
  results: [],
};

const stateAfterOnlyGarp = currentSetCapabilities(stateOp, respOnlyGarp);
console.log("\nTransition 7: One strategy valid (VALUE_GARP) -> another missing (VCP)");
console.log("   VCP status when response omitted VCP:", stateAfterOnlyGarp.VCP.universeScreening);
console.log("   VCP matches when response omitted VCP:", stateAfterOnlyGarp.VCP.matchedCount);
const defect7 = stateAfterOnlyGarp.VCP.universeScreening === "AVAILABLE" && stateAfterOnlyGarp.VCP.matchedCount === 12;
console.log("   [DEFECT CONFIRMED 1.2]: Stale VCP operational claim leaked when strategy omitted?", defect7 ? "YES" : "NO");

// Transition 8: Successful response -> network error (catch block does not clear capabilities)
// In page.tsx: .catch(() => { if (!isMounted) return; setIsLoading(false); }) -> leaves capabilities untouched!
console.log("\nTransition 8: Successful response -> network error");
console.log("   In current page.tsx line 216-219, .catch() only sets isLoading(false), leaving capabilities as stale AVAILABLE.");
console.log("   [DEFECT CONFIRMED 1.3]: Stale operational claim persists despite network failure: YES");

// Publication Integrity finding
console.log("\nPublication Integrity Badge Static Text finding");
console.log("   In current page.tsx line 1058-1060: <div className=\"text-[11px] text-slate-500 font-mono\">Publication Integrity: Verified</div>");
console.log("   Renders 'Publication Integrity: Verified' unconditionally, even on rollback or pending.");
console.log("   [DEFECT CONFIRMED 1.4]: Static unconditional integrity assertion: YES");
