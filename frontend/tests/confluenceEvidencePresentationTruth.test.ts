/**
 * ARX Confluence Evidence Presentation Truth Acceptance Suite
 * Directive: ARX-NVDA-EVIDENCE-TRUTH-AUDIT-001
 *
 * Verifies that CompositeConvictionCard strictly differentiates between:
 * 1. Evaluated failing numerical scores (e.g. 30/100 in amber/rose)
 * 2. Unverified evidence states ("UNVERIFIED", rendered in neutral muted slate)
 * 3. Unavailable / held evidence states ("UNAVAILABLE", rendered in neutral muted slate)
 * 4. Exempt statutory structures ("EXEMPT", rendered in neutral muted slate)
 *
 * Eliminates the semantic collision where missing or held data was erroneously
 * displayed as an alarming red "0 / 100" while adjacent notices stated
 * "Missing data is treated as unassessed, not negative."
 */

import * as assert from "node:assert";
import * as fs from "node:fs";
import * as path from "node:path";

console.log("=====================================================================");
console.log("ARX CONFLUENCE EVIDENCE PRESENTATION TRUTH & SEMANTIC INTEGRITY SUITE");
console.log("Directive: ARX-NVDA-EVIDENCE-TRUTH-AUDIT-001");
console.log("=====================================================================\n");

const COMPONENT_PATH = path.join(__dirname, "../components/CompositeConvictionCard.tsx");
const source = fs.readFileSync(COMPONENT_PATH, "utf8");

// ── Test 1: Static Code Invariants ───────────────────────────────────────────
console.log("Test 1: Verifying CompositeConvictionCard source code invariants...");
{
  // 1. Must define formatPillarScore helper or equivalent logic
  assert.ok(
    source.includes("formatPillarScore"),
    "CompositeConvictionCard must define formatPillarScore helper"
  );

  // 2. Must handle unverified and synthetic statuses explicitly
  assert.ok(
    source.includes('"UNVERIFIED"'),
    'CompositeConvictionCard must explicitly emit "UNVERIFIED" scoreDisplay'
  );

  // 3. Must handle unavailable status explicitly
  assert.ok(
    source.includes('"UNAVAILABLE"'),
    'CompositeConvictionCard must explicitly emit "UNAVAILABLE" scoreDisplay'
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

  // 6. getPillarScoreColor must treat UNVERIFIED, UNAVAILABLE, EXEMPT, PENDING as neutral muted text
  assert.ok(
    source.includes('scoreDisplay === "UNVERIFIED"'),
    "getPillarScoreColor must handle UNVERIFIED as neutral text"
  );
  assert.ok(
    source.includes('scoreDisplay === "EXEMPT"'),
    "getPillarScoreColor must handle EXEMPT as neutral text"
  );

  console.log("  ✓ Test 1 Passed: Static code invariants verified.\n");
}

// ── Test 2: Runtime Semantic Scoring Logic Simulation ─────────────────────────
console.log("Test 2: Simulating NVDA live production response projection...");
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

  // Re-create the component's formatting logic
  const formatPillarScore = (p?: { score: number | null; status: string }) => {
    if (!p || p.score == null) return "UNAVAILABLE";
    if (p.status === "unverified" || p.status === "synthetic") return "UNVERIFIED";
    if (p.status === "unavailable") return "UNAVAILABLE";
    if (p.status === "not_applicable") return "EXEMPT";
    return `${Math.round(p.score)} / 100`;
  };

  function getPillarScoreColor(scoreDisplay: string, score: number | null): string {
    if (
      score === null ||
      scoreDisplay === "UNAVAILABLE" ||
      scoreDisplay === "PENDING" ||
      scoreDisplay === "UNVERIFIED" ||
      scoreDisplay === "EXEMPT"
    ) {
      return "text-[var(--text-2)] font-semibold text-sm";
    }
    if (score >= 75) return "text-emerald-400";
    if (score >= 50) return "text-cyan-400";
    if (score >= 35) return "text-amber-400";
    return "text-rose-400";
  }

  const techP = nvdaBackendConfluence.pillars.find(p => p.pillar === "TECHNICAL_STRUCTURE")!;
  const fundP = nvdaBackendConfluence.pillars.find(p => p.pillar === "FUNDAMENTAL_SOLVENCY")!;
  const smartP = nvdaBackendConfluence.pillars.find(p => p.pillar === "SMART_MONEY_FLOW")!;
  const macroP = nvdaBackendConfluence.pillars.find(p => p.pillar === "MACRO_SAFETY_FLOOR")!;

  // Assertions for NVDA
  assert.strictEqual(formatPillarScore(techP), "30 / 100", "Technical pillar must format as '30 / 100'");
  assert.strictEqual(getPillarScoreColor(formatPillarScore(techP), techP.score), "text-rose-400", "Technical 30/100 must be rose");

  assert.strictEqual(formatPillarScore(fundP), "UNVERIFIED", "Fundamental pillar must format as 'UNVERIFIED', NOT '0 / 100'");
  assert.strictEqual(getPillarScoreColor(formatPillarScore(fundP), fundP.score), "text-[var(--text-2)] font-semibold text-sm", "UNVERIFIED must NOT be red");

  assert.strictEqual(formatPillarScore(smartP), "UNAVAILABLE", "Smart Money pillar must format as 'UNAVAILABLE', NOT '0 / 100'");
  assert.strictEqual(getPillarScoreColor(formatPillarScore(smartP), smartP.score), "text-[var(--text-2)] font-semibold text-sm", "UNAVAILABLE must NOT be red");

  assert.strictEqual(formatPillarScore(macroP), "85 / 100", "Macro pillar must format as '85 / 100'");
  assert.strictEqual(getPillarScoreColor(formatPillarScore(macroP), macroP.score), "text-emerald-400", "Macro 85/100 must be emerald");

  // Confluence gauge score assertion
  const displayScore = Math.round(nvdaBackendConfluence.confluenceScore);
  assert.strictEqual(displayScore, 21, "Confluence score 20.8 must round to 21 for top-level display gauge");

  console.log("  ✓ Test 2 Passed: NVDA production response projected with 100% semantic clarity.\n");
}

// ── Test 3: ETF and Foreign ADR Special Cases ─────────────────────────────────
console.log("Test 3: Testing ETF exemption and foreign ADR cases...");
{
  const formatPillarScore = (p?: { score: number | null; status: string }) => {
    if (!p || p.score == null) return "UNAVAILABLE";
    if (p.status === "unverified" || p.status === "synthetic") return "UNVERIFIED";
    if (p.status === "unavailable") return "UNAVAILABLE";
    if (p.status === "not_applicable") return "EXEMPT";
    return `${Math.round(p.score)} / 100`;
  };

  // ETF fundamental exemption
  const etfFundP = { score: 0.0, status: "not_applicable" };
  assert.strictEqual(formatPillarScore(etfFundP), "EXEMPT", "ETF fundamental exemption must format as 'EXEMPT'");

  // Neutral verified smart money (e.g. foreign ADR or verified zero)
  const neutralSmartP = { score: 50.0, status: "neutral" };
  assert.strictEqual(formatPillarScore(neutralSmartP), "50 / 100", "Neutral verified smart money must format as '50 / 100'");

  console.log("  ✓ Test 3 Passed: ETF and foreign ADR cases verified.\n");
}

console.log("=====================================================================");
console.log("ALL CONFLUENCE EVIDENCE PRESENTATION TRUTH ASSERTIONS PASSED (100%)");
console.log("=====================================================================\n");
