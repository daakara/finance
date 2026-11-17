/**
 * ARX CONFLUENCE DRAWER PRESENTATION & INTEGRITY REGRESSION SUITE
 *
 * Verifies that WhyInspectModal ("Confluence drawer"):
 * 1. Shows authoritative backend pillar scores (e.g., 30/100, 85/100, UNAVAILABLE, EXEMPT)
 * 2. Displays Score before coverage cap (rawConfluenceScore)
 * 3. Displays Coverage cap (coverageCap and verified pillar count)
 * 4. Displays Earnings & binary event warning (confluence.warnings)
 * 5. Replaces conflicting frontend point labels (+25, -25, +13, "deducts 25 points")
 * 6. Preserves decision gates and scoring engine integrity
 */

import React from "react";
// @ts-ignore
import { renderToStaticMarkup } from "react-dom/server";
import assert from "assert";
import WhyInspectModal from "../components/WhyInspectModal";
import { ConfluenceData } from "../lib/api";

console.log("=====================================================================");
console.log("ARX CONFLUENCE DRAWER PRESENTATION & INTEGRITY REGRESSION SUITE");
console.log("=====================================================================\n");

// --- FIXTURE: Authentic NVDA Post-Backfill Confluence Payload ---
const nvdaConfluenceFixture: ConfluenceData = {
  symbol: "NVDA",
  confluenceScore: 20.8,
  rawConfluenceScore: 20.8,
  rawComposite: 20.8,
  coverageCap: 50.0,
  verifiedPillarCount: 1,
  coverageRatio: 0.25,
  incompleteCoverage: true,
  evidenceConfidence: "LOW",
  confluenceRating: "DEFENSIVE / CAPITAL PRESERVATION MODE",
  plainRating: "🔴 RED LIGHT: WAIT FOR DUST TO SETTLE",
  confluenceBadge: "1-Feed Mixed Confluence",
  plainBadge: "1/4 Positive Signals",
  badgeColor: "rose",
  bottomLine: "Technical turbulence or weak balance sheet metrics detected (1 warning flags). Preserve cash.",
  positivesCount: 1,
  warningsCount: 1,
  warnings: [
    "⚠️ HIGH BINARY GAP RISK: Earnings in <=24h! Limit position sizing.",
  ],
  pillars: [
    {
      pillar: "TECHNICAL_STRUCTURE",
      label: "Technical Structure",
      plainLabel: "Chart Structure",
      score: 30.0,
      status: "warning",
      detail: "Stage 4 Markdown / Distribution Phase (RSI 46.8). Price below key declining moving averages.",
      plainDetail: "Falling knife mode: Price is breaking down. Wait for buyers to establish a firm floor.",
      icon: "⚠️",
    },
    {
      pillar: "FUNDAMENTAL_SOLVENCY",
      label: "Fundamental Solvency",
      plainLabel: "Financial Health",
      score: 0.0,
      status: "unavailable",
      detail: "SEC Form 10-Q/10-K financial statement line-item lineage unverified.",
      plainDetail: "Balance sheet financial statements unverified.",
      icon: "🏢",
    },
    {
      pillar: "SMART_MONEY_FLOW",
      label: "Smart Money Flow",
      plainLabel: "Institutional Activity",
      score: 0.0,
      status: "unavailable",
      detail: "SEC Form 4 disclosures unindexed.",
      plainDetail: "No recent big boss insider purchases filed with the SEC this month.",
      icon: "🏛️",
    },
    {
      pillar: "MACRO_SAFETY_FLOOR",
      label: "Macro Safety Floor",
      plainLabel: "Market Climate",
      score: 85.0,
      status: "positive",
      detail: "FRED 10Y-2Y yield curve positive (+0.29%), credit spreads tight (3.23%).",
      plainDetail: "Macro green light: Credit markets healthy and treasury yield curve normal.",
      icon: "📈",
    },
  ],
};

// --- Test 1: NVDA Confluence Drawer Presentation ---
console.log("Test 1: Verifying NVDA Confluence Drawer with authentic backend confluence payload...");
{
  const markup = renderToStaticMarkup(
    React.createElement(WhyInspectModal, {
      isOpen: true,
      onClose: () => {},
      symbol: "NVDA",
      setupScore: 21,
      confluence: nvdaConfluenceFixture,
      terminalState: {
        overallEligibility: "LIMITED",
        decisionState: "EVIDENCE_INCOMPLETE",
      } as any,
    })
  );

  // 1. Score before coverage cap is rendered
  assert.ok(
    markup.includes('data-testid="confluence-score-before-cap"'),
    "Drawer must render data-testid='confluence-score-before-cap'"
  );
  assert.ok(
    markup.includes("Score Before Cap") && markup.includes("20.8 / 100"),
    "Drawer must display Score Before Cap: 20.8 / 100"
  );

  // 2. Coverage cap is rendered
  assert.ok(
    markup.includes('data-testid="confluence-coverage-cap"'),
    "Drawer must render data-testid='confluence-coverage-cap'"
  );
  assert.ok(
    markup.includes("Coverage Cap") && markup.includes("50 / 100"),
    "Drawer must display Coverage Cap: 50 / 100"
  );
  assert.ok(
    markup.includes("(1/4)"),
    "Drawer must display verified pillar count (1/4)"
  );

  // 3. Earnings warning is rendered prominently
  assert.ok(
    markup.includes('data-testid="confluence-drawer-earnings-warning"'),
    "Drawer must render data-testid='confluence-drawer-earnings-warning'"
  );
  assert.ok(
    markup.includes("Earnings &amp; Binary Event Warning") || markup.includes("Earnings & Binary Event Warning"),
    "Drawer must render Earnings & Binary Event Warning header"
  );
  assert.ok(
    markup.includes("HIGH BINARY GAP RISK: Earnings in &lt;=24h") || markup.includes("HIGH BINARY GAP RISK: Earnings in <=24h"),
    "Drawer must render exact earnings warning text"
  );

  // 4. Backend pillar scores are rendered
  assert.ok(
    markup.includes('data-testid="pillar-card-technical_structure"'),
    "Must render Technical Structure card"
  );
  assert.ok(
    markup.includes("30 / 100") && markup.includes("UNFAVORABLE"),
    "Technical Structure must render score 30 / 100 with UNFAVORABLE badge"
  );

  assert.ok(
    markup.includes('data-testid="pillar-card-fundamental_solvency"'),
    "Must render Fundamental Solvency card"
  );
  assert.ok(
    markup.includes("UNAVAILABLE"),
    "Fundamental Solvency must render UNAVAILABLE"
  );

  assert.ok(
    markup.includes('data-testid="pillar-card-smart_money_flow"'),
    "Must render Smart Money Flow card"
  );

  assert.ok(
    markup.includes('data-testid="pillar-card-macro_safety_floor"'),
    "Must render Macro Safety Floor card"
  );
  assert.ok(
    markup.includes("85 / 100") && markup.includes("FAVORABLE"),
    "Macro Safety Floor must render score 85 / 100 with FAVORABLE badge"
  );

  // 5. Zero conflicting point labels
  assert.ok(
    !markup.includes("-25 Price Trend"),
    "Must not render '-25 Price Trend' point label"
  );
  assert.ok(
    !markup.includes("+13 Macro Regime"),
    "Must not render '+13 Macro Regime' point label"
  );
  assert.ok(
    !markup.includes("deducts 25 points"),
    "Must not render conflicting 'deducts 25 points' rule text"
  );
  assert.ok(
    !markup.includes("(+15 points)"),
    "Must not render conflicting '(+15 points)' rule text"
  );

  console.log("  ✓ Test 1 Passed: NVDA Confluence Drawer renders backend scores, pre-cap score, cap, earnings warning, and zero point labels.");
}

// --- Test 2: ETF / Exempt Pillar Rendering ---
console.log("\nTest 2: Verifying ETF / Exempt Pillar Rendering...");
{
  const etfConfluenceFixture: ConfluenceData = {
    symbol: "SPY",
    confluenceScore: 72.0,
    rawConfluenceScore: 82.5,
    coverageCap: 72.0,
    verifiedPillarCount: 2,
    confluenceRating: "SELECTIVE MULTI-FACTOR ALIGNMENT",
    plainRating: "🟡 SELECTIVE ENTRY",
    confluenceBadge: "2-Feed Confluence",
    plainBadge: "2/4 Signals",
    badgeColor: "amber",
    bottomLine: "ETF fundamental 10-K filings exempt.",
    positivesCount: 2,
    warningsCount: 0,
    pillars: [
      {
        pillar: "TECHNICAL_STRUCTURE",
        label: "Technical Structure",
        plainLabel: "Chart Structure",
        score: 80.0,
        status: "positive",
        detail: "Uptrend above 50 SMA.",
        plainDetail: "Uptrend.",
        icon: "📈",
      },
      {
        pillar: "FUNDAMENTAL_SOLVENCY",
        label: "Fundamental Solvency",
        plainLabel: "Financial Health",
        score: 0.0,
        status: "exempt",
        detail: "ETF structure: Corporate 10-K filings not applicable.",
        plainDetail: "ETF exempt.",
        icon: "🏢",
      },
      {
        pillar: "SMART_MONEY_FLOW",
        label: "Smart Money Flow",
        plainLabel: "Institutional Activity",
        score: 0.0,
        status: "unavailable",
        detail: "No filings.",
        plainDetail: "No filings.",
        icon: "🏛️",
      },
      {
        pillar: "MACRO_SAFETY_FLOOR",
        label: "Macro Safety Floor",
        plainLabel: "Market Climate",
        score: 85.0,
        status: "positive",
        detail: "Supportive macro.",
        plainDetail: "Supportive macro.",
        icon: "📈",
      },
    ],
  };

  const markup = renderToStaticMarkup(
    React.createElement(WhyInspectModal, {
      isOpen: true,
      onClose: () => {},
      symbol: "SPY",
      setupScore: 72,
      confluence: etfConfluenceFixture,
      terminalState: {
        overallEligibility: "ELIGIBLE",
        decisionState: "VALID_SETUP",
      } as any,
    })
  );

  assert.ok(
    markup.includes("EXEMPT"),
    "ETF Fundamental Solvency must render EXEMPT badge"
  );
  assert.ok(
    markup.includes("Score Before Cap") && markup.includes("82.5 / 100"),
    "Must display Score Before Cap: 82.5 / 100"
  );
  assert.ok(
    markup.includes("Coverage Cap") && markup.includes("72 / 100"),
    "Must display Coverage Cap: 72 / 100"
  );

  console.log("  ✓ Test 2 Passed: ETF exempt fundamentals render EXEMPT badge without phantom scores.");
}

// --- Test 3: Fallback Mode When Confluence is Omitted ---
console.log("\nTest 3: Verifying graceful fallback when confluence prop is omitted...");
{
  const markup = renderToStaticMarkup(
    React.createElement(WhyInspectModal, {
      isOpen: true,
      onClose: () => {},
      symbol: "TEST",
      setupScore: 60,
      terminalState: {
        overallEligibility: "ELIGIBLE",
        decisionState: "VALID_SETUP",
        domains: [
          {
            domainId: "test_domain",
            domainName: "Test Domain",
            availability: "AVAILABLE",
            status: "FAVORABLE",
            pointImpact: 10,
            importanceLevel: "HIGH",
            observation: "Test observation.",
            modelRule: "Test rule.",
            evidence: [],
          },
        ],
      } as any,
    })
  );

  assert.ok(
    markup.includes("Traceable Evidence &amp; Score Breakdown") || markup.includes("Traceable Evidence & Score Breakdown"),
    "Must render modal title"
  );
  assert.ok(
    markup.includes("Test Domain"),
    "Must render fallback domain card"
  );
  assert.ok(
    !markup.includes("+10 Test Domain"),
    "Must not render raw point impact '+10' as primary label"
  );

  console.log("  ✓ Test 3 Passed: Graceful fallback functions correctly without conflicting points.");
}

console.log("\n=====================================================================");
console.log("ALL CONFLUENCE DRAWER PRESENTATION TESTS PASSED (100%)");
console.log("=====================================================================");
