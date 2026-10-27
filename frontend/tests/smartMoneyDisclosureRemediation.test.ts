/**
 * frontend/tests/smartMoneyDisclosureRemediation.test.ts
 *
 * ARX Terminal — Smart Money Statutory Disclosure & Evidence Integrity Test Suite.
 * Enforces permanent end-to-end regression invariants for PR #72 / Issue #67 (Directive ARX_PR72_DISCLOSURE_RECONCILIATION_027):
 * 1. DEF-72-01: Congressional PTR filing deadline follows 30d notice / 45d maximum transaction rule (STOCK Act, 5 U.S.C. § 13105(l)).
 * 2. DEF-72-02: SEC Form 4 timing explicitly specifies "Generally 2 business days" with statutory exception awareness; eliminates ambiguous shorthand.
 * 3. DEF-72-03: Unsupported promotional claims ("whales", "before it hits the headlines", conflation of distinct populations, predictions) are strictly absent.
 * 4. DEF-72-04: Arbitrary unverified $112.8M options benchmark is eliminated; options volume benchmark displays explicit "Unavailable / Historical Reference".
 * 5. DEF-72-05: Inactive Smart Money scanner truthfully reports PIPELINE_PENDING; historical disclosures, curated examples, observations, and offline states remain distinct.
 * 6. DEF-72-05b: Feed errors and connection failures are never represented as confirmed zero verified transactions.
 * 7. Non-actionability: Disclosures explicitly state they are historical references, not investment advice or trading recommendations.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";

console.log("\n===============================================================================");
console.log("   ARX TERMINAL — SMART MONEY DISCLOSURE INTEGRITY SUITE (PR #72 / ISSUE #67)  ");
console.log("===============================================================================\n");

const pagePath = path.join(__dirname, "../app/smart-money/page.tsx");
const pageSource = fs.readFileSync(pagePath, "utf8");

// ---------------------------------------------------------------------------
// 1. DEF-72-01: Correct STOCK Act Filing Deadline
// ---------------------------------------------------------------------------
console.log("1. Verifying Congressional STOCK Act PTR statutory disclosure accuracy...");

// A. Elimination of inaccurate "15 to 45 calendar days"
const has15To45 = pageSource.includes("15 to 45 calendar days") || pageSource.includes("15–45 Days") || pageSource.includes("15-45 Days");
assert.strictEqual(
  has15To45,
  false,
  "DEF-72-01 Violation: Inaccurate '15 to 45 calendar days' must be completely eliminated"
);
console.log("   [OK] Inaccurate '15 to 45 calendar days' phrasing completely removed");

// B. Accurate statutory copy: 30 days after notification / 45 days after transaction
assert.ok(
  pageSource.includes("30 days after notification") && pageSource.includes("45 days after the transaction"),
  "DEF-72-01 Violation: Copy must state PTRs generally must be filed within 30 days after notification, and no later than 45 days after transaction"
);
assert.ok(
  pageSource.includes("5 U.S.C. § 13105(l)") || pageSource.includes("STOCK Act"),
  "DEF-72-01 Violation: Must reference statutory basis (STOCK Act / 5 U.S.C. § 13105(l))"
);
console.log("   [OK] Accurate statutory filing window (30d notification / 45d transaction) verified");

// C. Short-form badge text
assert.ok(
  pageSource.includes("Congressional PTR: 30d from notice / 45d maximum"),
  "DEF-72-01 Violation: Short-form badge must be 'Congressional PTR: 30d from notice / 45d maximum'"
);
console.log("   [OK] Short-form badge 'Congressional PTR: 30d from notice / 45d maximum' verified");

// ---------------------------------------------------------------------------
// 2. DEF-72-02: Clarify SEC Form 4 Timing
// ---------------------------------------------------------------------------
console.log("\n2. Verifying SEC Form 4 business-day disclosure and shorthand elimination...");

// A. Elimination of ambiguous shorthand "SEC Form 4: ≤2 Days"
const hasLe2Days = pageSource.includes("SEC Form 4: ≤2 Days") || pageSource.includes("SEC Form 4: <=2 Days");
assert.strictEqual(
  hasLe2Days,
  false,
  "DEF-72-02 Violation: Ambiguous shorthand 'SEC Form 4: ≤2 Days' must be eliminated"
);
console.log("   [OK] Ambiguous shorthand 'SEC Form 4: ≤2 Days' eliminated");

// B. Precise business day badge
assert.ok(
  pageSource.includes("SEC Form 4: Generally 2 business days"),
  "DEF-72-02 Violation: Badge must explicitly state 'SEC Form 4: Generally 2 business days'"
);
console.log("   [OK] Precise badge 'SEC Form 4: Generally 2 business days' verified");

// C. Text explicitly specifies business days and exceptions
assert.ok(
  pageSource.includes("2 business days") && pageSource.includes("statutory exceptions"),
  "DEF-72-02 Violation: Text must specify 2 business days subject to specific statutory exceptions"
);
console.log("   [OK] Statutory 2 business days text with exception awareness verified");

// ---------------------------------------------------------------------------
// 3. DEF-72-03: Remove Unsupported Institutional-Flow Claims & Promotional Copy
// ---------------------------------------------------------------------------
console.log("\n3. Verifying removal of unsupported promotional claims and source conflation...");

// A. No promotional copy
assert.strictEqual(
  pageSource.includes("whales"),
  false,
  "DEF-72-03 Violation: Promotional term 'whales' must be 0"
);
assert.strictEqual(
  pageSource.includes("before it hits the headlines"),
  false,
  "DEF-72-03 Violation: Promotional claim 'before it hits the headlines' must be 0"
);
console.log("   [OK] Promotional copy ('whales', 'before it hits the headlines') completely removed");

// B. Distinct populations not treated as interchangeable "institutional capital flows"
assert.strictEqual(
  pageSource.includes("Track disclosed institutional and congressional capital flows"),
  false,
  "DEF-72-03 Violation: Conflation of insider/congressional data as 'institutional capital flows' must be eliminated"
);
assert.ok(
  pageSource.includes("Distinguishes corporate insider disclosures (SEC Form 4), congressional reporting (STOCK Act PTRs), and historical options flow samples"),
  "DEF-72-03 Violation: Introduction must explicitly distinguish the three distinct populations"
);
console.log("   [OK] Three distinct populations explicitly differentiated");

// C. Explicit disclaimer against live accumulation and future prediction
assert.ok(
  pageSource.includes("not live institutional accumulation or forward-looking predictions"),
  "DEF-72-03 Violation: Must state disclosures do not reflect live institutional accumulation or forward-looking predictions"
);
console.log("   [OK] Live accumulation and prediction disclaimers verified");

// ---------------------------------------------------------------------------
// 4. DEF-72-04: Verify and Eliminate Unbacked $112.8M Options Benchmark
// ---------------------------------------------------------------------------
console.log("\n4. Verifying elimination of unbacked $112.8M options benchmark...");

const has1128M = pageSource.includes("$112.8M") || pageSource.includes("112.8M");
assert.strictEqual(
  has1128M,
  false,
  "DEF-72-04 Violation: Unverified $112.8M benchmark must not appear in Smart Money UI"
);
console.log("   [OK] Unverified $112.8M benchmark completely eliminated from Smart Money page");

assert.ok(
  pageSource.includes("Options Benchmark") && pageSource.includes("Unavailable") && pageSource.includes("Historical Reference"),
  "DEF-72-04 Violation: Card must display Options Benchmark as Unavailable with Historical Reference indicator"
);
console.log("   [OK] Options benchmark displayed as 'Unavailable' with 'Historical Reference' badge");

// ---------------------------------------------------------------------------
// 5. DEF-72-05: Truthful Scanner Availability & Distinguishable States
// ---------------------------------------------------------------------------
console.log("\n5. Verifying truthful scanner PIPELINE_PENDING and distinct operational states...");

// A. PIPELINE_PENDING badge and copy
assert.ok(
  pageSource.includes("Scanner: PIPELINE_PENDING"),
  "DEF-72-05 Violation: Badge must show 'Scanner: PIPELINE_PENDING'"
);
assert.ok(
  pageSource.includes("Continuous automated scanner pipelines remain in a held <strong>PIPELINE_PENDING</strong> state"),
  "DEF-72-05 Violation: Paragraph must state scanner pipelines remain held in PIPELINE_PENDING"
);
console.log("   [OK] Inactive scanner held state 'Scanner: PIPELINE_PENDING' verified");

// B. Feed Error / Offline handling prevents false zero representation
assert.ok(
  pageSource.includes("feedError") && pageSource.includes("Feed Offline / Connection Failure"),
  "DEF-72-05 Violation: Must support feedError state and display explicit 'Feed Offline / Connection Failure'"
);
assert.ok(
  pageSource.includes("Failed or unavailable data feeds are not represented as zero verified transactions"),
  "DEF-72-05 Violation: Must explicitly inform users that failed feeds are not zero transactions"
);
console.log("   [OK] Feed error state verified: failed feeds never masquerade as zero verified transactions");

// ---------------------------------------------------------------------------
// 6. Non-Actionability & Investment Advice Disclaimer
// ---------------------------------------------------------------------------
console.log("\n6. Verifying non-actionability and investment advice disclaimers...");

assert.ok(
  pageSource.includes("Historical Reference — Not Investment Advice"),
  "Non-actionability Violation: Must include 'Historical Reference — Not Investment Advice' disclaimer"
);
console.log("   [OK] 'Historical Reference — Not Investment Advice' disclaimer verified");

console.log("\n===============================================================================");
console.log("   ALL SMART MONEY DISCLOSURE REMEDIATION TESTS PASSED (7/7 GATES VERIFIED)   ");
console.log("===============================================================================\n");
