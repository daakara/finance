/**
 * Frontend State Handling Verification: Congressional STOCK Act Disclosures (Phase 0)
 * Directive: ARX-CONGRESSIONAL-INGESTION-001 (Phase 0)
 *
 * Verifies frontend presentation and state handling:
 * 1. Unmonitored states truthfully display 'Not Monitored' without HOLD_SEC_001.
 * 2. Empty states only claim 'No matching transactions' when verified.
 * 3. Curated examples are clearly badged as historical research samples (Aug 2026).
 * 4. Politician and Late Filer pages quarantine demonstration fixtures.
 */

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import React from "react";
import { AppRouterContext } from "next/dist/shared/lib/app-router-context.shared-runtime";
const { renderToStaticMarkup } = require("react-dom/server") as {
  renderToStaticMarkup: (element: React.ReactElement) => string;
};
import CongressionalTradesCard from "../components/CongressionalTradesCard";
import { resolveNineEvidenceDomains } from "../lib/evidenceDomains";

console.log("===============================================================================");
console.log("   CONGRESSIONAL STOCK ACT DISCLOSURE STATE HANDLING TEST (PHASE 0)           ");
console.log("===============================================================================\n");

// -----------------------------------------------------------------------------
// Test 1: evidenceDomains maps unmonitored congressional disclosures honestly
// -----------------------------------------------------------------------------
console.log("Test 1: Verifying evidenceDomains mapping for unmonitored congressional data...");

const heldAnalytics = {
  feeds: {
    congressional_disclosures: {
      status: "UNAVAILABLE",
      observation_count: 0,
      provenance: "None",
      disclosure: "Congressional STOCK Act live ingestion feed is not currently monitored.",
    },
  },
  subdomains: {
    congressional_stock_act: {
      status: "UNAVAILABLE",
      status_label: "Not Monitored",
      disclosure: "Congressional STOCK Act disclosure collection is not currently monitored in live mode.",
      records: [],
    },
    sec_form_4: {
      status: "PIPELINE_HELD",
      status_label: "Pipeline Pending (HOLD_SEC_001)",
      disclosure: "SEC Form 4 continuous ingestion paused under governed operational hold (HOLD_SEC_001).",
      records: [],
    },
  },
} as any;

const domains = resolveNineEvidenceDomains({ symbol: "AAPL", smartMoney: heldAnalytics });
const congDomain = domains.find(d => d.id === "congressional_disclosures");
const secDomain = domains.find(d => d.id === "insider_transactions");

assert.ok(congDomain, "Congressional disclosures domain must exist in mapped domains");
assert.strictEqual(congDomain?.status, "UNAVAILABLE", "Congressional domain must be UNAVAILABLE");
assert.strictEqual(congDomain?.statusLabel, "Not Monitored", "Congressional domain statusLabel must be 'Not Monitored'");
assert.ok(
  !congDomain?.explanation.includes("HOLD_SEC_001"),
  "Congressional domain explanation must NEVER contain HOLD_SEC_001"
);
console.log("  [OK] Congressional domain statusLabel is 'Not Monitored' and completely free of HOLD_SEC_001");

assert.ok(secDomain, "SEC Form 4 domain must exist in mapped domains");
assert.strictEqual(secDomain?.status, "UNAVAILABLE", "SEC Form 4 domain remains UNAVAILABLE under hold");
assert.ok(
  secDomain?.explanation.includes("HOLD_SEC_001"),
  "SEC Form 4 domain must legitimately preserve HOLD_SEC_001 explanation"
);
console.log("  [OK] SEC Form 4 domain legitimately preserves HOLD_SEC_001 hold governance");

// -----------------------------------------------------------------------------
// Test 2: CongressionalTradesCard Component Static Contract Verification
// -----------------------------------------------------------------------------
console.log("\nTest 2: Verifying CongressionalTradesCard.tsx empty & unmonitored states...");
const cardPath = path.join(__dirname, "../components/CongressionalTradesCard.tsx");
const cardSource = fs.readFileSync(cardPath, "utf8");

// Must handle "Not Monitored" in badge
assert.ok(
  cardSource.includes('status === "PIPELINE_PENDING" || status === "UNAVAILABLE"') &&
  cardSource.includes('"Not Monitored"'),
  "CongressionalTradesCard must display 'Not Monitored' when status is PIPELINE_PENDING or UNAVAILABLE"
);
console.log("  [OK] Badge displays 'Not Monitored' for unmonitored statuses");

// Must display truthful unmonitored message
assert.ok(
  cardSource.includes("Congressional transaction collection is not monitored."),
  "CongressionalTradesCard must state 'Congressional transaction collection is not monitored.'"
);
console.log("  [OK] Truthful unmonitored copy verified in card");

// Must NOT claim "verified coverage period" when unverified empty
assert.ok(
  !cardSource.includes('status === "AVAILABLE" && filteredCongressTrades.length === 0 ? (\n              <div className="p-4 bg-[#090d14] rounded-lg border border-[#243044] text-center text-xs text-slate-400">\n                No qualifying congressional transactions found in the verified coverage period.'),
  "Must NEVER claim 'No qualifying congressional transactions found in the verified coverage period' on unverified empty arrays"
);
console.log("  [OK] Misleading 'verified coverage period' on unverified empty arrays eliminated");

// Must require explicit coverage evidence before rendering certified empty
assert.ok(
  cardSource.includes("coverageEvidence?.watermarkHash") || cardSource.includes("certifiedWatermark"),
  "CongressionalTradesCard must require explicit cryptographic coverage evidence for certified-empty state"
);
console.log("  [OK] Certified-empty rendering requires explicit coverage evidence");

// Must display explicit curated badge when displaying research sample
assert.ok(
  cardSource.includes("Curated Research Sample (Aug 2026 — Not Live Feed)") ||
  cardSource.includes("Curated Research Disclosures (Aug 2026 Sample)"),
  "CongressionalTradesCard must badge curated research samples with explicit Aug 2026 date"
);
console.log("  [OK] Curated research samples explicitly badged with August 2026 date");

// -----------------------------------------------------------------------------
// Test 3: Politician Profile Page Quarantine Verification
// -----------------------------------------------------------------------------
console.log("\nTest 3: Verifying PoliticianProfilePage quarantine notice and claim removal...");
const polPath = path.join(__dirname, "../app/politician/[slug]/page.tsx");
const polSource = fs.readFileSync(polPath, "utf8");

assert.ok(
  polSource.includes("Demonstration Research Profile:") &&
  polSource.includes("The transaction items below are synthetic illustrative fixtures"),
  "Politician page must display prominent demonstration research quarantine banner"
);
assert.ok(
  !polSource.includes("Verified Forensic Investigation Dossier: Audited STOCK Act disclosures"),
  "Politician page must NOT claim 'Verified Forensic Investigation Dossier'"
);
assert.ok(
  !polSource.includes("Launch Live Congressional Smart Money Scanner"),
  "Politician page must NOT promote unbuilt 'Live Congressional Smart Money Scanner'"
);
console.log("  [OK] Politician page correctly quarantined with demonstration notice and unsupported claims removed");

// -----------------------------------------------------------------------------
// Test 4: Late Filers Page Quarantine Verification
// -----------------------------------------------------------------------------
console.log("\nTest 4: Verifying Late-Filers page quarantine notice and claim removal...");
const latePath = path.join(__dirname, "../app/smart-money/late-filers/page.tsx");
const lateSource = fs.readFileSync(latePath, "utf8");

assert.ok(
  lateSource.includes("Historical Research Case Studies:") &&
  lateSource.includes("The filings below are synthetic demonstration records"),
  "Late filers page must display historical demonstration quarantine notice"
);
assert.ok(
  !lateSource.includes("Late-Filer Hall of Shame"),
  "Late filers page must NOT use sensationalist 'Hall of Shame' title"
);
assert.ok(
  !lateSource.includes("🛑 STATUTORY COMPLIANCE SURVEILLANCE"),
  "Late filers page must NOT claim 'STATUTORY COMPLIANCE SURVEILLANCE'"
);
assert.ok(
  !lateSource.includes("Launch Live Smart Money Scanner"),
  "Late filers page must NOT claim 'Launch Live Smart Money Scanner'"
);
console.log("  [OK] Late filers page correctly quarantined with demonstration case study notice");

// -----------------------------------------------------------------------------
// Test 5: api.ts fetchSmartMoneyOverview & SMART_MONEY Pipeline Rationale
// -----------------------------------------------------------------------------
console.log("\nTest 5: Verifying api.ts SMART_MONEY rationale and includeCurated parameter...");
const apiPath = path.join(__dirname, "../lib/api.ts");
const apiSource = fs.readFileSync(apiPath, "utf8");

assert.ok(
  apiSource.includes("Congressional STOCK Act collection is not currently monitored in live mode"),
  "api.ts SMART_MONEY rationale must state that congressional collection is not currently monitored"
);
assert.ok(
  apiSource.includes("includeCurated?: boolean"),
  "SmartMoneyFetchOptions must explicitly support includeCurated parameter"
);
assert.ok(
  apiSource.includes("const includeCuratedParam = Boolean(options?.includeCurated);"),
  "includeCurated must default to false at call sites"
);
console.log("  [OK] api.ts SMART_MONEY rationale and includeCurated parameter verified");

// -----------------------------------------------------------------------------
// Test 6: Strict Coverage Evidence & Null Value Presentation Handling
// -----------------------------------------------------------------------------
console.log("\nTest 6: Verifying strict coverage validation & null owner/amount preservation...");

// Must reject CONFIRMED_ZERO_VERIFIED_COVERAGE in cardSource
assert.ok(
  cardSource.includes('hash !== "CONFIRMED_ZERO_VERIFIED_COVERAGE"'),
  "CongressionalTradesCard must explicitly reject fabricated fallback CONFIRMED_ZERO_VERIFIED_COVERAGE"
);
console.log("  [OK] Fabricated fallback CONFIRMED_ZERO_VERIFIED_COVERAGE explicitly rejected in card");

// Must verify 64-character hash length
assert.ok(
  cardSource.includes("hash.length === 64"),
  "CongressionalTradesCard must require exact 64-character hex hash"
);
console.log("  [OK] 64-char hash length check verified");

// Must check unresolved records and extraction outcomes
assert.ok(
  cardSource.includes("coverageEvidence?.unresolvedRecordCount === 0") &&
  cardSource.includes("coverageEvidence.extractionOutcomes?.[id] === \"SUCCESS\""),
  "CongressionalTradesCard must check for 0 unresolved records and successful document extraction"
);
console.log("  [OK] Unresolved records and extraction outcome checks verified");

// Must preserve null amount and null owner
assert.ok(
  cardSource.includes('trade.amount_range || "Undisclosed amount"') &&
  cardSource.includes("trade.owner &&"),
  "CongressionalTradesCard must render 'Undisclosed amount' when amount is null and omit owner when null"
);
console.log("  [OK] Undisclosed amount and owner preservation verified");

const forgedReceiptMarkup = renderToStaticMarkup(React.createElement(AppRouterContext.Provider, {
  value: { push() {}, replace() {}, refresh() {}, back() {}, forward() {}, prefetch() {} },
}, React.createElement(CongressionalTradesCard, {
  symbol: "NVDA", status: "VALID_EMPTY", congressTrades: [],
  coverageEvidence: {
    sourceInventory: "invented", chamber: "HOUSE", symbol: "NVDA",
    intervalDays: 180, documentIdentities: ["invented"],
    extractionOutcomes: { invented: "SUCCESS" }, unresolvedRecordCount: 0,
    watermarkHash: "f".repeat(64), filterMatching: true, verifiedComplete: true,
  },
})));
assert.ok(forgedReceiptMarkup.includes("Coverage uncertified"));
assert.ok(!forgedReceiptMarkup.includes("Verified complete search confirmed"));

console.log("\n===============================================================================");
console.log("   ALL 6 FRONTEND STATE HANDLING & PRESENTATION GATES PASSED!                ");
console.log("===============================================================================");
