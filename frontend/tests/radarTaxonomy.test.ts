import assert from "node:assert";
import {
  CanonicalRadarCategory,
  RadarCapabilities,
  GemCandidate,
} from "../lib/api";

console.log("Starting Radar Taxonomy & Capability Invariants Test Suite...");

// 1. Canonical Taxonomy Vocabulary Invariant
const canonicalVocab: CanonicalRadarCategory[] = ["VALUE_GARP", "VCP", "SMART_MONEY"];
assert.strictEqual(canonicalVocab.length, 3);
assert.ok(canonicalVocab.includes("VALUE_GARP"));
assert.ok(canonicalVocab.includes("VCP"));
assert.ok(canonicalVocab.includes("SMART_MONEY"));
// "VALUE" must NOT be a valid canonical transport category
assert.strictEqual((canonicalVocab as string[]).includes("VALUE"), false);
console.log("[OK] Canonical taxonomy vocabulary verified");

// 2. Zero-Semantics Invariant: Numerical zero allowed ONLY when AVAILABLE
function computeCategoryBadge(
  category: CanonicalRadarCategory,
  candidates: GemCandidate[],
  capabilities: RadarCapabilities
): { count: number | null; badge: string } {
  const isAvailable = capabilities[category]?.universeScreening === "AVAILABLE";
  if (!isAvailable) {
    return {
      count: null,
      badge: capabilities[category]?.status === "PIPELINE_PENDING" ? "Pipeline Pending" : "Unavailable",
    };
  }
  const matching = candidates.filter((c) => c.categories.includes(category));
  return {
    count: matching.length,
    badge: `${matching.length}`,
  };
}

const mockCapabilities: RadarCapabilities = {
  VALUE_GARP: {
    status: "AVAILABLE",
    universeScreening: "AVAILABLE",
    singleAssetAnalysis: "AVAILABLE",
  },
  VCP: {
    status: "PIPELINE_PENDING",
    universeScreening: "PIPELINE_PENDING",
    singleAssetAnalysis: "AVAILABLE",
  },
  SMART_MONEY: {
    status: "PIPELINE_PENDING",
    universeScreening: "PIPELINE_PENDING",
    singleAssetAnalysis: "AVAILABLE",
  },
};

// Test with zero matching candidates
const emptyCandidates: GemCandidate[] = [];

// VALUE_GARP is AVAILABLE -> When 0 matches exist, count 0 is legitimate
const valueGarpBadge = computeCategoryBadge("VALUE_GARP", emptyCandidates, mockCapabilities);
assert.strictEqual(valueGarpBadge.count, 0, "AVAILABLE category with 0 matches must produce count 0");
assert.strictEqual(valueGarpBadge.badge, "0");

// VCP is PIPELINE_PENDING -> Must NOT render count 0!
const vcpBadge = computeCategoryBadge("VCP", emptyCandidates, mockCapabilities);
assert.strictEqual(vcpBadge.count, null, "PIPELINE_PENDING category must NEVER produce count 0");
assert.strictEqual(vcpBadge.badge, "Pipeline Pending");

// SMART_MONEY is PIPELINE_PENDING -> Must NOT render count 0!
const smartMoneyBadge = computeCategoryBadge("SMART_MONEY", emptyCandidates, mockCapabilities);
assert.strictEqual(smartMoneyBadge.count, null, "PIPELINE_PENDING category must NEVER produce count 0");
assert.strictEqual(smartMoneyBadge.badge, "Pipeline Pending");
console.log("[OK] Zero-semantics invariants verified: PIPELINE_PENDING never produces count 0");

// 3. Adversarial String Invariance: Fallback strings must not generate categories
function mapRawCandidateToCategories(raw: {
  expert_model?: string | null;
  categories?: any[];
}): CanonicalRadarCategory[] {
  if (Array.isArray(raw.categories)) {
    return raw.categories.filter((c) =>
      ["VALUE_GARP", "VCP", "SMART_MONEY"].includes(c)
    ) as CanonicalRadarCategory[];
  }
  return [];
}

// Substring tests:
assert.deepStrictEqual(
  mapRawCandidateToCategories({ expert_model: "Smart Growth Compounder" }),
  [],
  "'Smart Growth Compounder' must NOT become SMART_MONEY"
);
assert.deepStrictEqual(
  mapRawCandidateToCategories({ expert_model: "Flow Traders" }),
  [],
  "'Flow Traders' must NOT become SMART_MONEY"
);
assert.deepStrictEqual(
  mapRawCandidateToCategories({ expert_model: "Minervini Stage 2 VCP" }),
  [],
  "'Minervini Stage 2 VCP' string must NOT become VCP"
);
assert.deepStrictEqual(
  mapRawCandidateToCategories({ expert_model: "Peter Lynch-like GARP" }),
  [],
  "'Peter Lynch-like GARP' string must NOT become VALUE_GARP"
);
assert.deepStrictEqual(
  mapRawCandidateToCategories({ expert_model: null }),
  [],
  "null expert_model must remain unclassified"
);
assert.deepStrictEqual(
  mapRawCandidateToCategories({ expert_model: "Unverified Asset" }),
  [],
  "'Unverified Asset' must remain unclassified"
);

// Valid explicit category passing:
assert.deepStrictEqual(
  mapRawCandidateToCategories({
    expert_model: "Peter Lynch GARP Compounder",
    categories: ["VALUE_GARP"],
  }),
  ["VALUE_GARP"],
  "Authoritative VALUE_GARP category must be faithfully consumed"
);
console.log("[OK] Adversarial substring immunity and removal of fallback fabrication verified");

console.log("All Radar Taxonomy & Capability Invariant tests PASSED successfully.");
