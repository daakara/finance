/**
 * Issue #67: unsupported static divergence examples cannot appear as
 * current Smart Money investment signals. This is a source-level guard;
 * a future evidence-backed implementation needs separate runtime tests.
 */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const component = fs.readFileSync(
  path.join(__dirname, "../components/SmartMoneyDivergenceRadar.tsx"),
  "utf8"
);
const page = fs.readFileSync(
  path.join(__dirname, "../app/smart-money/page.tsx"),
  "utf8"
);

assert.match(component, /data-testid="smart-money-divergence-unavailable"/);
assert.match(component, /Evidence unavailable/);
assert.match(component, /No verified divergence signals available/);
assert.match(component, /not a current market signal/);
assert.match(component, /Missing evidence does not mean zero transactions/);

// The widget must not render legacy unsourced six-ticker scores or price tiles.
for (const legacy of [
  "DIVERGENCE_DATASET",
  "insiderSentimentScore",
  "retailHypeScore",
  "divergenceType",
  "price: 31.20",
  "price: 23.40",
  "QUIET ACCUMULATION",
  "HYPE TRAP WARNING",
  "Dan Crenshaw",
  "Piotroski corporate insider accumulation",
  "href={\`/?symbol=\${asset.symbol}\`}",
]) {
  assert.ok(!component.includes(legacy), `Unsupported active divergence element: ${legacy}`);
}

// Source-independent fallback must remain visible regardless of feed status.
assert.match(page, /<SmartMoneyDivergenceRadar \/>/);
assert.ok(!component.includes("fetch("), "No ad-hoc data feed added");
assert.ok(!component.includes("useEffect"), "No scan-on-read or async side effect added");

console.log("[PASS] Smart Money unsupported divergence suppression (#67)");
