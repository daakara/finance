import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

console.log("Starting Smart Money Semantics & Firewall Verification Suite...");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");
const SMART_MONEY_PAGE = path.join(ROOT_DIR, "app", "smart-money", "page.tsx");
const RADAR_PAGE = path.join(ROOT_DIR, "app", "radar", "page.tsx");
const CONGRESS_CARD = path.join(ROOT_DIR, "components", "CongressionalTradesCard.tsx");
const TERMINAL_PAGE = path.join(ROOT_DIR, "app", "page.tsx");

const smartMoneySrc = fs.readFileSync(SMART_MONEY_PAGE, "utf-8");
const radarSrc = fs.readFileSync(RADAR_PAGE, "utf-8");
const congressCardSrc = fs.readFileSync(CONGRESS_CARD, "utf-8");
const terminalSrc = fs.readFileSync(TERMINAL_PAGE, "utf-8");

// 1. Verify Subtitle Overclaiming Removal (Rule: Never claim 'before it hits the headlines')
console.log("1. Verifying subtitle copy quality on /smart-money...");
assert.strictEqual(
  smartMoneySrc.includes("before it hits the headlines"),
  false,
  "VIOLATION: Overclaiming copy 'before it hits the headlines' must not be present"
);
assert.ok(
  smartMoneySrc.includes("Inspect verified public STOCK Act disclosures"),
  "Professional forensic copy must be present on /smart-money"
);
console.log("   [PASS] Overclaiming copy eliminated; professional forensic copy verified");

// 2. Verify Single-Asset Status Binding
console.log("2. Verifying Single-Asset Status Propagation in Terminal...");
assert.ok(
  terminalSrc.includes("status={data?.smartMoney?.status}"),
  "Terminal page must pass smartMoney.status to CongressionalTradesCard"
);
assert.ok(
  congressCardSrc.includes("status?: string"),
  "CongressionalTradesCardProps must accept status prop"
);
assert.ok(
  congressCardSrc.includes('status === "PIPELINE_PENDING"'),
  "CongressionalTradesCard must branch on PIPELINE_PENDING status"
);
console.log("   [PASS] CongressionalTradesCard correctly distinguishes PIPELINE_PENDING from 0 trades");

// 3. Verify Radar Search Counter False-Zero Prevention
console.log("3. Verifying Radar Search Toolbar False-Zero Suppression...");
assert.ok(
  radarSrc.includes("activeFilter === 'SMART_MONEY' && categoryMeta.SMART_MONEY.status === 'PIPELINE_PENDING'"),
  "Radar toolbar must check for SMART_MONEY PIPELINE_PENDING"
);
assert.ok(
  radarSrc.includes("Scanner Pending · ${allAssets.length} candidates"),
  "Radar toolbar must render 'Scanner Pending' instead of '0 of 35'"
);
console.log("   [PASS] Radar toolbar suppresses false-zero counter during PIPELINE_PENDING");

console.log("\n===============================================================================");
console.log("ALL SMART MONEY SEMANTIC TESTS PASSED SUCCESSFULLY!");
console.log("===============================================================================");
