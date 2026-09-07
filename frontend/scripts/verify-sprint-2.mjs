import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

console.log("========================================================================");
console.log("  ARX Terminal vNext: Sprint 2 Verification Suite                       ");
console.log("  (Layered Intelligence, Progressive Disclosure & Research Efficiency)  ");
console.log("========================================================================\n");

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${err.message}`);
    failed++;
  }
}

const resolveComp = (rel) => {
  const p1 = path.resolve(rel);
  if (fs.existsSync(p1)) return fs.readFileSync(p1, "utf-8");
  const p2 = path.resolve("frontend", rel);
  if (fs.existsSync(p2)) return fs.readFileSync(p2, "utf-8");
  throw new Error(`File not found: ${rel}`);
};

const tooltipSrc = resolveComp("components/tooltips/InstitutionalTooltip.tsx");
const popoverSrc = resolveComp("components/conviction/ConvictionPillDetailPopover.tsx");
const matrixSrc = resolveComp("components/conviction/ConvictionMatrix.tsx");
const whyCardSrc = resolveComp("components/explanation/WhyARXCard.tsx");
const traceModalSrc = resolveComp("components/explanation/ConfluenceTraceModal.tsx");
const sectionSrc = resolveComp("components/research/AccordionSection.tsx");
const stackSrc = resolveComp("components/research/ResearchAccordionStack.tsx");
const canvasSrc = resolveComp("components/workstation/WorkstationCanvas.tsx");
const telemetrySrc = resolveComp("types/telemetry.ts");
const workstationTypesSrc = resolveComp("types/workstation.ts");

// Test Suite 1: Institutional Tooltip Rule
test("InstitutionalTooltip.tsx adheres to Progressive Disclosure Rule (explains methodology, uses role='tooltip')", () => {
  assert(tooltipSrc.includes("role=\"tooltip\""), "Must render role='tooltip'");
  assert(tooltipSrc.includes("aria-describedby"), "Must bind aria-describedby");
  assert(tooltipSrc.includes("Methodology"), "Must identify as methodology tooltip");
  assert(tooltipSrc.includes("formula"), "Must support formula disclosure");
  assert(tooltipSrc.includes("institutional_tooltip_viewed"), "Must track telemetry");
});

// Test Suite 2: Stage 3 Conviction Matrix & Popover
test("ConvictionPillDetailPopover.tsx renders causal drivers and enforces Anti-Cyan palette", () => {
  assert(popoverSrc.includes("role=\"dialog\""), "Popover must have role='dialog'");
  assert(popoverSrc.includes("aria-expanded"), "Must track aria-expanded");
  assert(popoverSrc.includes("Key Causal Drivers"), "Must surface plain-English causal drivers");
  assert(popoverSrc.includes("bg-emerald"), "Must use Emerald for favorable status");
  assert(popoverSrc.includes("bg-amber"), "Must use Amber for caution status");
  assert(popoverSrc.includes("bg-rose"), "Must use Rose for unfavorable status");
  assert(popoverSrc.includes("conviction_popover_opened"), "Must track opened event");
  assert(popoverSrc.includes("conviction_popover_closed"), "Must track closed duration event");
});

test("ConvictionMatrix.tsx coordinates the 5 core conviction pillars with Stage 3 labeling", () => {
  assert(matrixSrc.includes("Stage 3"), "Must be tagged Stage 3");
  assert(matrixSrc.includes("conviction_matrix_viewed"), "Must emit conviction_matrix_viewed");
  assert(matrixSrc.includes("Institutional Conviction Matrix"), "Must display institutional title");
  assert(matrixSrc.includes("ConvictionPillDetailPopover"), "Must render detail popovers");
});

// Test Suite 3: Stage 4 Why ARX Thinks This & Confluence Trace Modal
test("WhyARXCard.tsx enforces top-3 deterministic drivers and server-authoritative explanations", () => {
  assert(whyCardSrc.includes("Why ARX Thinks This"), "Must render approved title");
  assert(whyCardSrc.includes("Stage 4"), "Must be tagged Stage 4");
  assert(whyCardSrc.includes("drivers.slice(0, 3)"), "Must strictly limit to top 3 drivers");
  assert(whyCardSrc.includes("why_arx_card_viewed"), "Must emit why_arx_card_viewed");
  assert(whyCardSrc.includes("View Full Confluence Trace"), "Must offer CTA to trace modal");
});

test("ConfluenceTraceModal.tsx renders Level 3 factor attribution table with audit hashes", () => {
  assert(traceModalSrc.includes("role=\"dialog\""), "Must have role='dialog'");
  assert(traceModalSrc.includes("aria-modal=\"true\""), "Must have aria-modal='true'");
  assert(traceModalSrc.includes("Factor Name & Category"), "Must display factor attribution table");
  assert(traceModalSrc.includes("Net Contribution"), "Must show net contribution");
  assert(traceModalSrc.includes("Audit Provenance Hash"), "Must disclose decision provenance hash");
  assert(traceModalSrc.includes("confluence_trace_opened"), "Must emit confluence_trace_opened");
});

// Test Suite 4: Stage 5 Progressive Research Accordion Stack
test("ResearchAccordionStack.tsx implements WAI-ARIA collapsible sections with telemetry", () => {
  assert(sectionSrc.includes("aria-expanded"), "Section must bind aria-expanded");
  assert(sectionSrc.includes("aria-controls"), "Section must bind aria-controls");
  assert(sectionSrc.includes("role=\"region\""), "Section content must have role='region'");
  assert(stackSrc.includes("Stage 5"), "Stack must be tagged Stage 5");
  assert(stackSrc.includes("accordion_section_toggled"), "Stack must emit accordion_section_toggled");
});

// Test Suite 5: WorkstationCanvas & Lazy Hydration Architecture
test("WorkstationCanvas.tsx integrates Stage 1 through Stage 5 with next/dynamic lazy loading", () => {
  assert(canvasSrc.includes("TickerCommandStrip"), "Must render Stage 1 Command Strip");
  assert(canvasSrc.includes("WorkstationGrid"), "Must render Stage 2 65/35 Grid");
  assert(canvasSrc.includes("ConvictionMatrix"), "Must render Stage 3 Conviction Matrix");
  assert(canvasSrc.includes("WhyARXCard"), "Must render Stage 4 Why ARX Card");
  assert(canvasSrc.includes("ResearchAccordionStack"), "Must render Stage 5 Research Accordion");
  assert(canvasSrc.includes("dynamic("), "Stage 5 must be dynamically imported for lazy hydration");
});

// Test Suite 6: Types & Telemetry Envelopes
test("types/telemetry.ts and types/workstation.ts define all Sprint 2 contracts", () => {
  assert(telemetrySrc.includes("conviction_matrix_viewed"), "Must include conviction_matrix_viewed");
  assert(telemetrySrc.includes("why_arx_card_viewed"), "Must include why_arx_card_viewed");
  assert(telemetrySrc.includes("confluence_trace_opened"), "Must include confluence_trace_opened");
  assert(telemetrySrc.includes("accordion_section_toggled"), "Must include accordion_section_toggled");
  assert(telemetrySrc.includes("institutional_tooltip_viewed"), "Must include institutional_tooltip_viewed");
  assert(workstationTypesSrc.includes("FactorAttribution"), "Must export FactorAttribution");
  assert(workstationTypesSrc.includes("reasons?: string[]"), "ConvictionItem must support reasons");
});

console.log("\n========================================================================");
console.log(`  VERIFICATION RESULTS: ${passed} PASSED, ${failed} FAILED`);
console.log("========================================================================\n");

if (failed > 0) {
  process.exit(1);
}
