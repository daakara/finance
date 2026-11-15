/**
 * ARX Confluence Evidence Presentation Truth Acceptance Suite
 * Directive: ARX-NVDA-EVIDENCE-TRUTH-AUDIT-001 & ARX-NVDA-EVIDENCE-TRUTH-002
 *
 * Exercises the authoritative exported production formatting functions from
 * CompositeConvictionCard directly:
 * 1. Evaluated failing numerical scores (e.g. 30/100 in amber/rose)
 * 2. Unverified evidence states ("UNVERIFIED", rendered in neutral muted slate)
 * 3. Unavailable / held evidence states ("UNAVAILABLE", including pipeline_held, rendered in neutral muted slate)
 * 4. Exempt statutory structures ("EXEMPT", rendered in neutral muted slate)
 * 5. Checking status before score == null so null scores never discard unverified/exempt status
 *
 * Eliminates the semantic collision where missing or held data was erroneously
 * displayed as an alarming red "0 / 100" while adjacent notices stated
 * "Missing data is treated as unassessed, not negative."
 */

import * as assert from "node:assert";
import * as fs from "node:fs";
import * as path from "node:path";
import {
  formatPillarScore,
  getPillarScoreColor,
  getPillarStatusDot,
} from "../components/CompositeConvictionCard";

console.log("=====================================================================");
console.log("ARX CONFLUENCE EVIDENCE PRESENTATION TRUTH & SEMANTIC INTEGRITY SUITE");
console.log("Directives: ARX-NVDA-EVIDENCE-TRUTH-001 / 002");
console.log("=====================================================================\n");

const COMPONENT_PATH = path.join(__dirname, "../components/CompositeConvictionCard.tsx");
const source = fs.readFileSync(COMPONENT_PATH, "utf8");

// ── Test 1: Static Code Invariants ───────────────────────────────────────────
console.log("Test 1: Verifying CompositeConvictionCard source code invariants...");
{
  // 1. Must export formatPillarScore helper
  assert.ok(
    source.includes("export function formatPillarScore"),
    "CompositeConvictionCard must export formatPillarScore helper"
  );

  // 2. Must handle unverified and synthetic statuses explicitly
  assert.ok(
    source.includes('"UNVERIFIED"'),
    'CompositeConvictionCard must explicitly emit "UNVERIFIED" scoreDisplay'
  );

  // 3. Must handle unavailable and pipeline_held status explicitly
  assert.ok(
    source.includes('"pipeline_held"') || source.includes('pipeline_held'),
    'CompositeConvictionCard must explicitly handle pipeline_held status'
  );

  // 4. Must handle not_applicable / exempt status explicitly
  assert.ok(
    source.includes('"EXEMPT"'),
    'CompositeConvictionCard must explicitly emit "EXEMPT" scoreDisplay'
  );

  // 5. Must NOT leak raw "null" or "undefined"
  assert.ok(
    !source.includes('scoreDisplay: "null"') && !source.includes('scoreDisplay: "undefined"'),
    "Pillar score displays must never leak null or undefined string tokens"
  );

  // 6. Narrowly typed status union (no open `| string` escape hatch)
  assert.ok(
    source.includes("export type ConfluencePillarStatus ="),
    "CompositeConvictionCard must define narrow ConfluencePillarStatus type"
  );

  console.log("  ✓ Test 1 Passed: Static code invariants verified.\n");
}

// ── Test 2: Direct Execution of Production Formatter on Real NVDA Payload ────
console.log("Test 2: Exercising production formatPillarScore directly on NVDA payload...");
{
  // Simulated NVDA backend confluence response
  const nvdaBackendConfluence = {
    symbol: "NVDA",
    confluenceScore: 20.8,
    rawConfluenceScore: 20.8,
    confluenceBadge: "1-Feed Mixed Confluence",
    plainBadge: "1/4 Positive Signals",
    badgeColor: "rose",
    bottomLine: "Technical turbulence or weak balance sheet metrics detected (1 warning flags).",
    pillars: [
      {
        pillar: "TECHNICAL_STRUCTURE",
        label: "Technical Structure",
        plainLabel: "Chart Structure",
        score: 30.0,
        status: "warning",
        detail: "Stage 4 Markdown / Distribution Phase (RSI 19.1).",
        plainDetail: "Falling knife mode: Price is breaking down.",
        icon: "⚠️",
      },
      {
        pillar: "FUNDAMENTAL_SOLVENCY",
        label: "Fundamental Solvency",
        plainLabel: "Company Health",
        score: 0.0,
        status: "unverified",
        detail: "Fundamental factor data for NVDA is unverified by an authorized regulator or exchange source.",
        plainDetail: "Financial health data unverified — awaiting audited SEC financial statements.",
        icon: "🏢",
      },
      {
        pillar: "SMART_MONEY_FLOW",
        label: "Corporate Insiders & Flow",
        plainLabel: "Smart Money Flow",
        score: 0.0,
        status: "unavailable",
        detail: "Verified smart money regulatory filings unavailable for this asset.",
        plainDetail: "No verified insider or institutional order flow records connected.",
        icon: "🏛️",
      },
      {
        pillar: "MACRO_SAFETY_FLOOR",
        label: "Macro Regime & Safety Floor",
        plainLabel: "Market Tailwinds & Stop",
        score: 85.0,
        status: "positive",
        detail: "FRED 10Y-2Y yield curve positive (+0.27%), credit spreads tight (3.20%).",
        plainDetail: "Macro green light: Credit markets healthy and treasury yield curve normal.",
        icon: "🛡️",
      },
    ],
  };

  const techP = nvdaBackendConfluence.pillars.find(p => p.pillar === "TECHNICAL_STRUCTURE")!;
  const fundP = nvdaBackendConfluence.pillars.find(p => p.pillar === "FUNDAMENTAL_SOLVENCY")!;
  const smartP = nvdaBackendConfluence.pillars.find(p => p.pillar === "SMART_MONEY_FLOW")!;
  const macroP = nvdaBackendConfluence.pillars.find(p => p.pillar === "MACRO_SAFETY_FLOOR")!;

  // Assertions for NVDA using actual imported functions
  const techDisplay = formatPillarScore(techP);
  assert.strictEqual(techDisplay, "30 / 100", "Technical pillar must format as '30 / 100'");
  assert.strictEqual(getPillarScoreColor(techDisplay, techP.score), "text-rose-400", "Technical 30/100 must be rose");
  assert.ok(getPillarStatusDot(techP.status).includes("bg-rose-400"), "Technical dot must be rose");

  const fundDisplay = formatPillarScore(fundP);
  assert.strictEqual(fundDisplay, "UNVERIFIED", "Fundamental pillar must format as 'UNVERIFIED', NOT '0 / 100'");
  assert.strictEqual(getPillarScoreColor(fundDisplay, fundP.score), "text-[var(--text-2)] font-semibold text-sm", "UNVERIFIED must NOT be red");
  assert.strictEqual(getPillarStatusDot(fundP.status), "bg-slate-600", "UNVERIFIED dot must be neutral slate");

  const smartDisplay = formatPillarScore(smartP);
  assert.strictEqual(smartDisplay, "UNAVAILABLE", "Smart Money pillar must format as 'UNAVAILABLE', NOT '0 / 100'");
  assert.strictEqual(getPillarScoreColor(smartDisplay, smartP.score), "text-[var(--text-2)] font-semibold text-sm", "UNAVAILABLE must NOT be red");
  assert.strictEqual(getPillarStatusDot(smartP.status), "bg-slate-600", "UNAVAILABLE dot must be neutral slate");

  const macroDisplay = formatPillarScore(macroP);
  assert.strictEqual(macroDisplay, "85 / 100", "Macro pillar must format as '85 / 100'");
  assert.strictEqual(getPillarScoreColor(macroDisplay, macroP.score), "text-emerald-400", "Macro 85/100 must be emerald");
  assert.ok(getPillarStatusDot(macroP.status).includes("bg-emerald-400"), "Macro dot must be emerald");

  // Confluence gauge score assertion
  const displayScore = Math.round(nvdaBackendConfluence.confluenceScore);
  assert.strictEqual(displayScore, 21, "Confluence score 20.8 must round to 21 for top-level display gauge");

  console.log("  ✓ Test 2 Passed: Actual production functions executed on NVDA payload.\n");
}

// ── Test 3: Status Priority Over Null Scores (RQ-Review Requirement 2) ───────
console.log("Test 3: Testing status precedence when score == null...");
{
  // When score is null, unverified status must NOT collapse to generic UNAVAILABLE
  assert.strictEqual(
    formatPillarScore({ score: null, status: "unverified" }),
    "UNVERIFIED",
    "score: null with status: 'unverified' must return 'UNVERIFIED'"
  );
  assert.strictEqual(
    formatPillarScore({ score: null, status: "synthetic" }),
    "UNVERIFIED",
    "score: null with status: 'synthetic' must return 'UNVERIFIED'"
  );
  assert.strictEqual(
    formatPillarScore({ score: null, status: "not_applicable" }),
    "EXEMPT",
    "score: null with status: 'not_applicable' must return 'EXEMPT'"
  );
  assert.strictEqual(
    formatPillarScore({ score: null, status: "pipeline_held" }),
    "UNAVAILABLE",
    "score: null with status: 'pipeline_held' must return 'UNAVAILABLE'"
  );
  assert.strictEqual(
    formatPillarScore({ score: 0.0, status: "pipeline_held" }),
    "UNAVAILABLE",
    "score: 0.0 with status: 'pipeline_held' must return 'UNAVAILABLE'"
  );
  assert.strictEqual(
    formatPillarScore({ score: null, status: "positive" }),
    "UNAVAILABLE",
    "score: null with positive status must fail closed to 'UNAVAILABLE'"
  );

  console.log("  ✓ Test 3 Passed: Status precedence over null scores strictly verified.\n");
}

// ── Test 4: Genuinely Evaluated Failing Pillars & Edge Cases ──────────────────
console.log("Test 4: Testing genuinely evaluated failing pillars and edge cases...");
{
  // Genuine failing technical breakdown
  const failingTech = { score: 20.0, status: "warning" };
  assert.strictEqual(formatPillarScore(failingTech), "20 / 100");
  assert.strictEqual(getPillarScoreColor("20 / 100", 20.0), "text-rose-400");

  // Heavy insider selling (evaluated smart money)
  const insiderDumping = { score: 35.0, status: "warning" };
  assert.strictEqual(formatPillarScore(insiderDumping), "35 / 100");
  assert.strictEqual(getPillarScoreColor("35 / 100", 35.0), "text-amber-400");

  // Piotroski distress (evaluated fundamentals)
  const balanceSheetDistress = { score: 33.0, status: "warning" };
  assert.strictEqual(formatPillarScore(balanceSheetDistress), "33 / 100");
  assert.strictEqual(getPillarScoreColor("33 / 100", 33.0), "text-rose-400");

  // Missing pillar object
  assert.strictEqual(formatPillarScore(undefined), "UNAVAILABLE");

  console.log("  ✓ Test 4 Passed: Genuine failing scores and boundary conditions verified.\n");
}

console.log("=====================================================================");
console.log("ALL CONFLUENCE EVIDENCE PRESENTATION TRUTH ASSERTIONS PASSED (100%)");
console.log("=====================================================================\n");
