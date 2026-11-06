/**
 * Frontend Targeted Tests for Directive NVDA-STRATEGY-ALIGNMENT-002:
 * Targeted Presentation, Provenance & Assessment-Semantics Correction.
 *
 * Verifies:
 * 1. Missing API data never displays hard-coded fictional scores (84, 88, 94, 82, 76, 92).
 * 2. Individual alignment indices are displayed as `/ 100`, not percentages or probabilities.
 * 3. Coverage description is truthful: `Coverage: X / Y Modeled` (not `Measured`).
 * 4. Named-investor comparisons explicitly labeled as hypothetical strategy archetypes.
 * 5. Static congressional policy and disruptor priors identified as modeled assumptions.
 * 6. Existing supporting-model accordion preserves touch target min-height 44px and responsive structure.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("\n===============================================================================");
console.log("   ARX TERMINAL — STRATEGY ALIGNMENT SEMANTICS TEST SUITE (002)                ");
console.log("===============================================================================\n");

const CARD_PATH = path.resolve(__dirname, "../components/TraderArchetypesCard.tsx");
const PAGE_PATH = path.resolve(__dirname, "../app/page.tsx");

assert(fs.existsSync(CARD_PATH), `Card file not found at ${CARD_PATH}`);
assert(fs.existsSync(PAGE_PATH), `Page file not found at ${PAGE_PATH}`);

const cardSrc = fs.readFileSync(CARD_PATH, "utf8");
const pageSrc = fs.readFileSync(PAGE_PATH, "utf8");

// 1. Elimination of hard-coded fictional example fallback scores
console.log("1. Verifying elimination of hard-coded fictional scores on missing data...");
const forbiddenFictionalScores = [
  "consensusScore: 84",
  "alignmentScore: 88",
  "alignmentScore: 94",
  "alignmentScore: 82",
  "alignmentScore: 92",
  "Strong Accumulation Candidate"
];

for (const forbidden of forbiddenFictionalScores) {
  assert(
    !cardSrc.includes(forbidden),
    `FAIL: Found hard-coded fictional fallback score/verdict '${forbidden}' in TraderArchetypesCard.tsx`
  );
}

assert(
  cardSrc.includes("if (!traderArchetypes") || cardSrc.includes("!traderArchetypes.archetypes"),
  "FAIL: TraderArchetypesCard must check for missing/empty traderArchetypes and branch to explicit unavailable state."
);
assert(
  cardSrc.includes("Strategy archetype telemetry is currently unavailable for"),
  "FAIL: TraderArchetypesCard must display truthful unavailable state description when data is missing."
);
console.log("   [PASS] Hard-coded example fallback scores completely eliminated; truthful unavailable state enforced.");

// 2. Individual alignment scores formatted as / 100, not %
console.log("2. Verifying alignment scores formatted as / 100 (not percentage)...");
assert(
  cardSrc.includes("`${item.alignmentScore} / 100`"),
  "FAIL: Individual archetype alignment score must be formatted as `${item.alignmentScore} / 100`."
);
assert(
  !cardSrc.includes("`${item.alignmentScore}%`"),
  "FAIL: Individual archetype alignment score must NOT be formatted as percentage `${item.alignmentScore}%`."
);
console.log("   [PASS] Individual alignment scores displayed strictly as / 100, eliminating probability confusion.");

// 3. Truthful coverage description (Modeled, not Measured)
console.log("3. Verifying coverage description uses 'Modeled' instead of 'Measured'...");
assert(
  cardSrc.includes("Modeled"),
  "FAIL: TraderArchetypesCard coverage badge must use 'Modeled' descriptor."
);
assert(
  !cardSrc.includes("Measured"),
  "FAIL: TraderArchetypesCard coverage badge must NOT use 'Measured' descriptor for heuristic models."
);
console.log("   [PASS] Coverage badge truthfully states 'Modeled' rather than 'Measured'.");

// 4. Named-investor comparisons explicitly disclaim actual holdings
console.log("4. Verifying hypothetical strategy archetype labeling and holdings disclaimer...");
assert(
  cardSrc.includes("Strategy Alignment Model (Not Actual Holdings)"),
  "FAIL: Card must contain 'Strategy Alignment Model (Not Actual Holdings)' badge."
);
assert(
  cardSrc.includes("Hypothetical Archetypes"),
  "FAIL: Card title must indicate 'Hypothetical Archetypes'."
);
assert(
  cardSrc.includes("Model-derived similarity to selected investment strategies. These scores do not indicate current trades, positions, endorsements, or intentions of the named investors."),
  "FAIL: Card must contain comprehensive subtitle disclaimer."
);
console.log("   [PASS] Hypothetical strategy archetype identity and holdings disclaimer verified.");

// 5. Static priors identified as modeled assumptions
console.log("5. Verifying static domain priors identified as modeled assumptions...");
assert(
  cardSrc.includes("Modeled Prior (Assumption)"),
  "FAIL: Static prior badge must be labeled 'Modeled Prior (Assumption)'."
);
assert(
  cardSrc.includes("Modeled qualitative assumption; not verified statutory disclosure or government action"),
  "FAIL: Prior tooltip must explicitly clarify modeled qualitative assumption."
);
console.log("   [PASS] Static congressional and disruptor priors identified as modeled assumptions.");

// 6. Accordion preservation, touch targets, and mobile structure
console.log("6. Verifying accordion preservation and touch target compliance...");
assert(
  pageSrc.includes("min-h-[44px]"),
  "FAIL: Accordion summary touch target must preserve min-h-[44px] for mobile accessibility."
);
assert(
  pageSrc.includes("<details") && pageSrc.includes("<summary"),
  "FAIL: Collapsible <details> accordion structure must be preserved."
);
assert(
  pageSrc.includes("Hypothetical strategy alignment"),
  "FAIL: Page summary title must clarify hypothetical strategy alignment."
);
assert(
  cardSrc.includes("grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5"),
  "FAIL: 5-column adaptive responsive grid must be preserved for mobile/desktop parity."
);
console.log("   [PASS] Accordion structure, 44px touch target, and 5-column responsive grid verified.");

console.log("\n===============================================================================");
console.log("   ALL NVDA-STRATEGY-ALIGNMENT-002 TEST INVARIANTS VERIFIED (6/6 AREAS PASSED)  ");
console.log("===============================================================================\n");
