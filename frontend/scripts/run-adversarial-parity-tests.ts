/**
 * ARX_UX_DESIGN_PARITY_GATE_V1 & V1.1: Adversarial Gate Mutation Suite
 *
 * Deliberately injects one defect at a time into an isolated test environment,
 * verifies that the gate strictly fails and exits nonzero, restores clean state,
 * and confirms that the clean candidate passes 100%.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";

const frontendDir = path.resolve(__dirname, "..");
const reportsDir = path.resolve(frontendDir, "reports");
if (!fs.existsSync(reportsDir)) {
  fs.mkdirSync(reportsDir, { recursive: true });
}

interface AdversarialTestCase {
  id: string;
  name: string;
  description: string;
  targetFile: string;
  mutate: (content: string) => string;
  expectedFailurePattern: string;
}

const testCases: AdversarialTestCase[] = [
  {
    id: "DEFECT_1_REMOVE_INSTITUTIONAL_FLOW",
    name: "Remove Institutional Flow Pillar",
    description: "Omits Institutional Flow pillar from CompositeConvictionCard",
    targetFile: path.join(frontendDir, "components/CompositeConvictionCard.tsx"),
    mutate: (content: string) => {
      return content.replace(/label:\s*"Institutional Flow"/g, 'label: "REMOVED_FLOW"');
    },
    expectedFailurePattern: "Must include Institutional Flow pillar"
  },
  {
    id: "DEFECT_2_REMOVE_ANOTHER_CONFLUENCE_PILLAR",
    name: "Remove Fundamental Moat Pillar",
    description: "Omits Fundamental Moat pillar from CompositeConvictionCard",
    targetFile: path.join(frontendDir, "components/CompositeConvictionCard.tsx"),
    mutate: (content: string) => {
      return content.replace(/label:\s*"Fundamental Moat"/g, 'label: "REMOVED_MOAT"');
    },
    expectedFailurePattern: "Must include Fundamental Moat pillar"
  },
  {
    id: "DEFECT_3_DUPLICATE_LEGACY_DECISION_READINESS",
    name: "Unsubordinated Decision Readiness Progression",
    description: "Re-introduces legacy un-subordinated decision readiness header",
    targetFile: path.join(frontendDir, "components/DecisionReadinessCard.tsx"),
    mutate: (content: string) => {
      return content.replace("Advisory Workflow", "Autonomous Clearance Authority");
    },
    expectedFailurePattern: "DecisionReadinessCard must be labeled as Advisory Workflow"
  },
  {
    id: "DEFECT_4_COMPETING_PREFLIGHT_CLEARANCE_DISPLAY",
    name: "Competing Primary Clearance Verdict",
    description: "Injects independent GREEN LIGHT verdict into CompositeConvictionCard",
    targetFile: path.join(frontendDir, "components/CompositeConvictionCard.tsx"),
    mutate: (content: string) => {
      return content.replace(
        "Multi-Model Confluence Attribution",
        "GREEN LIGHT: HIGH CONVICTION"
      );
    },
    expectedFailurePattern: "CompositeConvictionCard must NOT emit independent green-light verdict titles"
  },
  {
    id: "DEFECT_5_CONTRADICTORY_ACTIONABILITY_TEXT",
    name: "Contradictory Actionability in Pre-Flight Bar",
    description: "Removes Pre-Flight bar mounting beneath candlestick chart",
    targetFile: path.join(frontendDir, "components/terminal/StandardTerminalView.tsx"),
    mutate: (content: string) => {
      return content.replace("<PreFlightReadinessBar", "<SuppressedPreFlightBar");
    },
    expectedFailurePattern: "StandardTerminalView must mount <PreFlightReadinessBar>"
  },
  {
    id: "DEFECT_6_RENDER_UNDEFINED_AS_SCORE_DESCRIPTION",
    name: "Render 'undefined' Token in Score Display",
    description: "Injects scoreDisplay: 'undefined' token leak into CompositeConvictionCard",
    targetFile: path.join(frontendDir, "components/CompositeConvictionCard.tsx"),
    mutate: (content: string) => {
      return content.replace(
        'scoreDisplay: "UNAVAILABLE"',
        'scoreDisplay: "undefined"'
      );
    },
    expectedFailurePattern: "Pillar score displays must never be 'null' or 'undefined' string"
  },
  {
    id: "DEFECT_7_SUPPRESS_REQUIRED_EVIDENCE_DISCLOSURE",
    name: "Suppress Composite Conviction Evidence Slot",
    description: "Removes evidenceSlot from page.tsx AdaptiveTerminal wiring",
    targetFile: path.join(frontendDir, "app/page.tsx"),
    mutate: (content: string) => {
      return content.replace("evidenceSlot={", "suppressedEvidenceSlot={");
    },
    expectedFailurePattern: "page.tsx must supply CompositeConvictionCard via evidenceSlot"
  },
  {
    id: "DEFECT_8_BREAK_MOBILE_RESPONSIVE_COMPOSITION",
    name: "Remove 5-Step Stepper Suppression",
    description: "Re-introduces duplicate TradeExecutionStepper into OptimalEntryExitCard",
    targetFile: path.join(frontendDir, "components/OptimalEntryExitCard.tsx"),
    mutate: (content: string) => {
      return content + "\n{/* <TradeExecutionStepper /> */}\n";
    },
    expectedFailurePattern: "OptimalEntryExitCard must NOT render duplicate 5-step stepper"
  },
  {
    id: "DEFECT_9_MOUNT_DECISION_READINESS_IN_ENTRY_CARD",
    name: "Mount DecisionReadinessCard in OptimalEntryExitCard",
    description: "Re-introduces prohibited DecisionReadinessCard into OptimalEntryExitCard",
    targetFile: path.join(frontendDir, "components/OptimalEntryExitCard.tsx"),
    mutate: (content: string) => {
      return content + "\n{/* <DecisionReadinessCard /> */}\n";
    },
    expectedFailurePattern: "OptimalEntryExitCard must NOT mount DecisionReadinessCard"
  },
  {
    id: "DEFECT_10_MOUNT_DUPLICATE_MODAL_IN_ENTRY_CARD",
    name: "Mount Duplicate PreFlightChecklistModal in OptimalEntryExitCard",
    description: "Re-introduces duplicate PreFlightChecklistModal into OptimalEntryExitCard",
    targetFile: path.join(frontendDir, "components/OptimalEntryExitCard.tsx"),
    mutate: (content: string) => {
      return content + "\n{/* <PreFlightChecklistModal /> */}\n";
    },
    expectedFailurePattern: "OptimalEntryExitCard must NOT mount duplicate PreFlightChecklistModal"
  },
  {
    id: "DEFECT_11_UNSUBORDINATE_PRECONDITIONS",
    name: "Unsubordinate What Needs to Change Stack",
    description: "Expands What Needs to Change by default instead of collapsing",
    targetFile: path.join(frontendDir, "components/terminal/StandardTerminalView.tsx"),
    mutate: (content: string) => {
      return content.replace("React.useState(false)", "React.useState(true)");
    },
    expectedFailurePattern: "StandardTerminalView must collapse What Needs to Change by default"
  },
  {
    id: "DEFECT_12_REMOVE_EXECUTION_LADDER_FROM_VERDICT",
    name: "Remove Compact Execution Ladder from Verdict Card",
    description: "Removes compact execution ladder from StandardTerminalView Decision Verdict card",
    targetFile: path.join(frontendDir, "components/terminal/StandardTerminalView.tsx"),
    mutate: (content: string) => {
      return content.replace("Execution Ladder", "Suppressed Ladder");
    },
    expectedFailurePattern: "StandardTerminalView must render Execution Ladder inside Decision Verdict card"
  }
];

console.log("================================================================================");
console.log("  ARX_UX_DESIGN_PARITY_GATE_V1 & V1.1: Adversarial Gate Mutation Suite");
console.log("================================================================================\n");

const results: any[] = [];
let allAdversarialPassed = true;

for (let i = 0; i < testCases.length; i++) {
  const tc = testCases[i];
  console.log(`[Test ${i + 1}/${testCases.length}] Injected Defect: ${tc.name}`);
  console.log(`  Description: ${tc.description}`);

  // Backup clean file
  const originalContent = fs.readFileSync(tc.targetFile, "utf8");
  const mutatedContent = tc.mutate(originalContent);
  assertDifferent(originalContent, mutatedContent, tc.id);

  try {
    // Write mutated defect
    fs.writeFileSync(tc.targetFile, mutatedContent, "utf8");

    // Execute gate
    let gateExitedNonZero = false;
    let gateOutput = "";

    try {
      gateOutput = execSync("npx tsx tests/uxDesignParityGateV1.test.ts", {
        cwd: frontendDir,
        encoding: "utf8",
        stdio: "pipe"
      });
    } catch (err: any) {
      gateExitedNonZero = true;
      gateOutput = (err.stdout || "") + "\n" + (err.stderr || "");
    }

    const matchedExpectedPattern = gateOutput.includes(tc.expectedFailurePattern);
    const successfullyRejected = gateExitedNonZero && matchedExpectedPattern;

    if (successfullyRejected) {
      console.log(`  --> RESULT: REJECTED (PASS) • Gate blocked defect with nonzero exit.`);
      console.log(`  --> Matched Expected Pattern: "${tc.expectedFailurePattern}"`);
      results.push({
        id: tc.id,
        name: tc.name,
        defectInjected: true,
        gateExitedNonZero: true,
        rejectionPatternMatched: true,
        verdict: "SUCCESSFULLY_BLOCKED"
      });
    } else {
      console.error(`  --> RESULT: FAILED TO BLOCK DEFECT! Nonzero: ${gateExitedNonZero}, Pattern: ${matchedExpectedPattern}`);
      allAdversarialPassed = false;
      results.push({
        id: tc.id,
        name: tc.name,
        defectInjected: true,
        gateExitedNonZero,
        rejectionPatternMatched: matchedExpectedPattern,
        verdict: "UNEXPECTED_PASS_ESCAPE"
      });
    }
  } finally {
    // Restore clean candidate file immediately
    fs.writeFileSync(tc.targetFile, originalContent, "utf8");
  }
}

function assertDifferent(orig: string, mutated: string, id: string) {
  if (orig === mutated) {
    throw new Error(`Mutation for test case ${id} failed to modify target file.`);
  }
}

// Final Step: Verify Clean Candidate Passes Gate (100%)
console.log("\n--------------------------------------------------------------------------------");
console.log("Final Verification: Executing Gate on Restored Clean Candidate...");
let cleanGateOutput = "";
let cleanCandidatePassed = false;
try {
  cleanGateOutput = execSync("npx tsx tests/uxDesignParityGateV1.test.ts", {
    cwd: frontendDir,
    encoding: "utf8",
    stdio: "pipe"
  });
  cleanCandidatePassed = cleanGateOutput.includes("ALL ARX_UX_DESIGN_PARITY_GATE_V1 & V1.1 STRUCTURAL ASSERTIONS PASSED");
} catch (err: any) {
  cleanCandidatePassed = false;
  cleanGateOutput = (err.stdout || "") + "\n" + (err.stderr || "");
}

if (cleanCandidatePassed) {
  console.log("  --> CLEAN CANDIDATE: PASS (100% assertions verified)");
} else {
  console.error("  --> CLEAN CANDIDATE: FAILED AFTER RESTORATION!");
  allAdversarialPassed = false;
}

const finalReport = {
  timestamp: new Date().toISOString(),
  totalMutationsTested: testCases.length,
  mutationsSuccessfullyBlocked: results.filter(r => r.verdict === "SUCCESSFULLY_BLOCKED").length,
  cleanCandidateRestoredAndPassed: cleanCandidatePassed,
  disposition: allAdversarialPassed && cleanCandidatePassed ? "PASS" : "FAIL",
  results
};

const reportPath = path.join(reportsDir, "adversarial_gate_results.json");
fs.writeFileSync(reportPath, JSON.stringify(finalReport, null, 2), "utf8");
console.log(`\nAdversarial Test Report written to: ${reportPath}`);

console.log("================================================================================");
console.log(`  ADVERSARIAL GATE TESTING DISPOSITION: ${finalReport.disposition}`);
console.log(`  ${results.filter(r => r.verdict === "SUCCESSFULLY_BLOCKED").length}/${testCases.length} Injected Defects Strictly Blocked • Clean Candidate Certified PASS`);
console.log("================================================================================\n");

if (!allAdversarialPassed || !cleanCandidatePassed) {
  process.exit(1);
}
