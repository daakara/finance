import puppeteer from "puppeteer";
import { computeAccessibleName } from "./accessible-name-evaluator.mjs";

async function runControls() {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const page = await browser.newPage();
  await page.setContent(`<!DOCTYPE html>
<html>
<head><title>Accessible Name Controls</title></head>
<body>
  <!-- NEGATIVE CONTROLS -->
  <!-- Neg 1: icon-only button with aria-hidden SVG -->
  <button id="neg-1"><svg aria-hidden="true"><path d="M0 0h10v10H0z"/></svg></button>

  <!-- Neg 2: input with placeholder only -->
  <input id="neg-2" placeholder="Search assets..." />

  <!-- Neg 3: aria-labelledby pointing to missing ID -->
  <button id="neg-3" aria-labelledby="non-existent-header-id"></button>

  <!-- Neg 4: aria-labelledby pointing to empty element -->
  <span id="empty-title"></span>
  <button id="neg-4" aria-labelledby="empty-title"></button>

  <!-- Neg 5: decorative image inside unlabeled link -->
  <a href="/test" id="neg-5"><img src="/icon.png" alt="" /></a>

  <!-- POSITIVE CONTROLS -->
  <!-- Pos 1: aria-label -->
  <button id="pos-1" aria-label="Close settings dialog"><svg aria-hidden="true"></svg></button>

  <!-- Pos 2: valid aria-labelledby -->
  <span id="pos-2-label">Execute Trade Plan</span>
  <button id="pos-2" aria-labelledby="pos-2-label"></button>

  <!-- Pos 3: visible button text -->
  <button id="pos-3"><span>Copy Trade Plan</span></button>

  <!-- Pos 4: proper <label for=""> -->
  <label for="pos-4">Account Allocation Capital</label>
  <input id="pos-4" type="text" />

  <!-- Pos 5: image link with meaningful alt -->
  <a href="/home" id="pos-5"><img src="/logo.png" alt="ARX Terminal Home" /></a>
</body>
</html>`);

  // Expose function to browser context
  await page.evaluate(`
    ${computeAccessibleName.toString()}
    window.computeAccessibleName = computeAccessibleName;
  `);

  const results = await page.evaluate(() => {
    const checks = [];

    // Negative controls (MUST return null or empty)
    for (let i = 1; i <= 5; i++) {
      const el = document.getElementById(`neg-${i}`);
      const name = window.computeAccessibleName(el);
      checks.push({
        id: `neg-${i}`,
        type: "NEGATIVE",
        computedName: name,
        pass: name === null || name === "",
      });
    }

    // Positive controls (MUST return expected non-empty string)
    for (let i = 1; i <= 5; i++) {
      const el = document.getElementById(`pos-${i}`);
      const name = window.computeAccessibleName(el);
      checks.push({
        id: `pos-${i}`,
        type: "POSITIVE",
        computedName: name,
        pass: typeof name === "string" && name.trim().length > 0,
      });
    }

    return checks;
  });

  console.log("================================================================================");
  console.log("         ACCESSIBLE-NAME HARNESS NEGATIVE & POSITIVE CONTROLS REPORT           ");
  console.log("================================================================================");
  
  let failed = 0;
  for (const c of results) {
    if (c.pass) {
      console.log(`  ✔ [PASS] ${c.type} CONTROL ${c.id}: correctly evaluated -> "${c.computedName || "null"}"`);
    } else {
      console.log(`  ❌ [FAIL] ${c.type} CONTROL ${c.id}: FAILED -> "${c.computedName}"`);
      failed++;
    }
  }

  await browser.close();

  if (failed > 0) {
    console.error(`\nFAILED: ${failed} controls failed to evaluate correctly.`);
    process.exit(1);
  } else {
    console.log("\nALL 5 NEGATIVE & 5 POSITIVE CONTROLS PASSED FAIL-CLOSED.");
    process.exit(0);
  }
}

runControls().catch((err) => {
  console.error("FATAL ERROR:", err);
  process.exit(1);
});
