import assert from "node:assert";
import {
  resolveAssetType,
  resolveExecutionEligibility,
  resolveCapabilities,
  isStock,
  isETF,
  isCrypto,
  isUnknownAsset,
  hasEtfSectorData,
} from "../lib/assetTypeUtils";

console.log("Starting ARX Canonical Security Master Frontend Routing Test Suite...\n");

// ---------------------------------------------------------------------------
// 1. Backward Compatibility & Non-Authoritative Fallback (Shell Bootstrapping)
// ---------------------------------------------------------------------------
console.log("1. Testing presentation catalog fallback for synchronous bootstrapping...");

assert.strictEqual(resolveAssetType("NVDA"), "Stock");
assert.strictEqual(resolveAssetType("AAPL"), "Stock");
assert.strictEqual(resolveAssetType("SPY"), "ETF");
assert.strictEqual(resolveAssetType("QQQ"), "ETF");
assert.strictEqual(resolveAssetType("BTC-USD"), "Crypto");

// Strict fail-closed for uncatalogued tickers in absence of server context
assert.strictEqual(resolveAssetType("PLSE"), "UNKNOWN");
assert.strictEqual(resolveAssetType("RANDOM_UNLISTED_999"), "UNKNOWN");
assert.strictEqual(resolveAssetType(""), "UNKNOWN");
assert.strictEqual(resolveAssetType(null), "UNKNOWN");
console.log("   [OK] Synchronous fallback preserves fail-closed boundary");

// ---------------------------------------------------------------------------
// 2. Authoritative Server-Owned Execution Eligibility
// ---------------------------------------------------------------------------
console.log("2. Testing resolveExecutionEligibility with server CanonicalInstrument envelope...");

const stockPayload = {
  canonicalInstrument: {
    symbol: "PLSE",
    provider_symbol: "PLSE",
    asset_class: "EQUITY",
    security_type: "COMMON_STOCK",
    primary_exchange: "NASDAQ",
    listing_status: "ACTIVE",
    classification_status: "VERIFIED",
    execution_eligibility: "STOCK_EXECUTION",
  },
};

const etfPayload = {
  canonicalInstrument: {
    symbol: "SPY",
    provider_symbol: "SPY",
    asset_class: "ETF",
    security_type: "ETF",
    primary_exchange: "ARCA",
    listing_status: "ACTIVE",
    classification_status: "VERIFIED",
    execution_eligibility: "ETF_EXECUTION",
  },
};

const cryptoPayload = {
  canonicalInstrument: {
    symbol: "BTC-USD",
    provider_symbol: "BTC-USD",
    asset_class: "CRYPTO",
    security_type: "CRYPTO",
    primary_exchange: "CRYPTO",
    listing_status: "ACTIVE",
    classification_status: "VERIFIED",
    execution_eligibility: "CRYPTO_EXECUTION",
  },
};

const failClosedPayload = {
  canonicalInstrument: {
    symbol: "TSM",
    provider_symbol: "TSM",
    asset_class: "EQUITY",
    security_type: "ADR",
    primary_exchange: "NYSE",
    listing_status: "ACTIVE",
    classification_status: "VERIFIED",
    execution_eligibility: "FAIL_CLOSED",
  },
};

assert.strictEqual(resolveExecutionEligibility(stockPayload), "STOCK_EXECUTION");
assert.strictEqual(resolveExecutionEligibility(etfPayload), "ETF_EXECUTION");
assert.strictEqual(resolveExecutionEligibility(cryptoPayload), "CRYPTO_EXECUTION");
assert.strictEqual(resolveExecutionEligibility(failClosedPayload), "FAIL_CLOSED");
assert.strictEqual(resolveExecutionEligibility(null), "FAIL_CLOSED");
assert.strictEqual(resolveExecutionEligibility({}), "FAIL_CLOSED");
console.log("   [OK] resolveExecutionEligibility strictly enforces server contract");

// ---------------------------------------------------------------------------
// 3. PLSE Acceptance Case (Organic Resolution, No Hardcoding)
// ---------------------------------------------------------------------------
console.log("3. Testing PLSE positive acceptance case...");

// PLSE is absent from masterCatalog, but server certifies it as Common Stock
assert.strictEqual(isStock("PLSE", stockPayload), true, "PLSE must be recognized as Stock when server provides STOCK_EXECUTION");
assert.strictEqual(isETF("PLSE", stockPayload), false, "PLSE must not be routed to ETF");
assert.strictEqual(isCrypto("PLSE", stockPayload), false, "PLSE must not be routed to Crypto");
assert.strictEqual(isUnknownAsset("PLSE", stockPayload), false, "PLSE must not be unresolved when server authorizes execution");
console.log("   [OK] PLSE successfully routes to Stock Execution without symbol-specific exceptions");

// ---------------------------------------------------------------------------
// 4. Negative Routing: Unsupported, Derivative, Inactive, Conflicted
// ---------------------------------------------------------------------------
console.log("4. Testing negative routing for unauthorized subtypes & fail-closed states...");

const negativeCases = [
  { name: "ADR (TSM)", subtype: "ADR" },
  { name: "REIT (O)", subtype: "REIT" },
  { name: "Preferred (BAC.PR.L)", subtype: "PREFERRED" },
  { name: "Warrant (LUNR.WS)", subtype: "WARRANT" },
  { name: "Unit (AAC.U)", subtype: "UNIT" },
  { name: "Right (BMA.RT)", subtype: "RIGHT" },
  { name: "UNKNOWN", subtype: "UNKNOWN" },
];

for (const tc of negativeCases) {
  const payload = {
    canonicalInstrument: {
      symbol: tc.name,
      security_type: tc.subtype,
      execution_eligibility: "FAIL_CLOSED",
      classification_status: "VERIFIED",
    },
  };
  assert.strictEqual(isStock(tc.name, payload), false, `${tc.name} must never reach stock execution`);
  assert.strictEqual(isETF(tc.name, payload), false, `${tc.name} must never reach ETF execution`);
  assert.strictEqual(isCrypto(tc.name, payload), false, `${tc.name} must never reach Crypto execution`);
  assert.strictEqual(isUnknownAsset(tc.name, payload), true, `${tc.name} must resolve to fail-closed UNKNOWN`);
}

// Inactive common stock must fail closed
const inactiveCommonStockPayload = {
  canonicalInstrument: {
    symbol: "INACTIVE_CO",
    security_type: "COMMON_STOCK",
    listing_status: "INACTIVE",
    execution_eligibility: "FAIL_CLOSED",
    classification_status: "VERIFIED",
  },
};
assert.strictEqual(isStock("INACTIVE_CO", inactiveCommonStockPayload), false, "Inactive common stock must fail closed");

// Conflicted classification must fail closed
const conflictedPayload = {
  canonicalInstrument: {
    symbol: "CONFLICTED_SYM",
    security_type: "COMMON_STOCK",
    listing_status: "ACTIVE",
    execution_eligibility: "FAIL_CLOSED",
    classification_status: "CONFLICTED",
  },
};
assert.strictEqual(isStock("CONFLICTED_SYM", conflictedPayload), false, "Conflicted instruments must fail closed");
console.log("   [OK] All negative routing cases strictly fail closed");

// ---------------------------------------------------------------------------
// 5. Generic Non-ETF Non-Crypto Fallback Prohibition
// ---------------------------------------------------------------------------
console.log("5. Testing generic fallback prohibition...");

const unverifiedPayload = {
  canonicalInstrument: {
    symbol: "UNVERIFIED_XYZ",
    security_type: "UNKNOWN",
    execution_eligibility: "FAIL_CLOSED",
    classification_status: "UNVERIFIED",
  },
};

assert.strictEqual(isStock("UNVERIFIED_XYZ", unverifiedPayload), false, "Generic fallback to Stock is strictly prohibited");
assert.strictEqual(isETF("UNVERIFIED_XYZ", unverifiedPayload), false);
assert.strictEqual(isCrypto("UNVERIFIED_XYZ", unverifiedPayload), false);
console.log("   [OK] Generic fallback to Stock is strictly prohibited and fails closed");

// ---------------------------------------------------------------------------
// 6. Comprehensive resolveCapabilities Tests
// ---------------------------------------------------------------------------
console.log("6. Testing resolveCapabilities across security types, data states, and modalities...");

// 6.1 Verified common stock (e.g. PLSE) with complete trade plan data
const validTradePlan = {
  current_price: 10.50,
  optimal_entry_min: 10.20,
  optimal_entry_max: 10.60,
  stop_loss: 9.75,
  take_profit_1: 12.00,
  risk_reward_ratio: 2.0,
};

const plseCaps = resolveCapabilities(stockPayload, validTradePlan);
assert.strictEqual(plseCaps.canRenderStockExecution, true, "PLSE must be authorized for stock execution");
assert.strictEqual(plseCaps.canRenderStockDetails, true, "PLSE must be authorized for stock detail blueprint");
assert.strictEqual(plseCaps.canOpenPreflight, true, "PLSE with trade plan must open preflight checklist");
assert.strictEqual(plseCaps.canOpenPositionSizer, true, "PLSE with trade plan must open position sizer");
assert.strictEqual(plseCaps.canUseETFWorkflow, false, "PLSE must not route to ETF workflow");
assert.strictEqual(plseCaps.canUseCryptoWorkflow, false, "PLSE must not route to Crypto workflow");
assert.strictEqual(plseCaps.disqualificationReason, undefined, "PLSE must not have disqualification reason");
console.log("   [OK] Verified common stock with trade plan passes all capabilities");

// 6.2 Verified common stock without trade plan (e.g. data still loading or uncalculated)
const plseNoPlanCaps = resolveCapabilities(stockPayload, null);
assert.strictEqual(plseNoPlanCaps.canRenderStockExecution, true, "Stock execution card rendered");
assert.strictEqual(plseNoPlanCaps.canRenderStockDetails, true, "Stock details rendered");
assert.strictEqual(plseNoPlanCaps.canOpenPreflight, false, "Preflight disabled without valid price/stop levels");
assert.strictEqual(plseNoPlanCaps.canOpenPositionSizer, false, "Sizer disabled without valid price/stop levels");
console.log("   [OK] Verified common stock without trade plan safely disables sizing/preflight modals");

// 6.3 Verified ETF (e.g. SPY)
const spyCaps = resolveCapabilities(etfPayload, validTradePlan);
assert.strictEqual(spyCaps.canUseETFWorkflow, true, "ETF must activate ETF workflow");
assert.strictEqual(spyCaps.canRenderStockExecution, false, "ETF must not activate stock execution");
assert.strictEqual(spyCaps.canRenderStockDetails, false, "ETF must not activate stock details");
assert.strictEqual(spyCaps.canOpenPreflight, false, "ETF must not activate stock preflight checklist");
console.log("   [OK] ETF capabilities strictly segregated to ETF workflow");

// 6.4 ADR / REIT / Unsupported fail-closed
const adrCaps = resolveCapabilities(failClosedPayload, validTradePlan);
assert.strictEqual(adrCaps.canRenderStockExecution, false, "ADR must not render stock execution");
assert.strictEqual(adrCaps.canRenderStockDetails, false, "ADR must not render stock details");
assert.strictEqual(adrCaps.canUseETFWorkflow, false, "ADR must not route to ETF");
assert.ok(adrCaps.disqualificationReason?.includes("ADR"), "Disqualification reason must mention ADR");
console.log("   [OK] Specialized equity subtype (ADR) fails closed with descriptive reason");

// 6.5 Null / Missing context fails closed
const nullCaps = resolveCapabilities(null, null);
assert.strictEqual(nullCaps.canRenderStockExecution, false);
assert.strictEqual(nullCaps.canRenderStockDetails, false);
assert.strictEqual(nullCaps.canOpenPreflight, false);
assert.ok(nullCaps.disqualificationReason !== undefined, "Null context must have disqualification explanation");
console.log("   [OK] Null context fails closed safely");

console.log("\nALL ARX CANONICAL SECURITY MASTER FRONTEND ROUTING TESTS PASSED SUCCESSFULLY!\n");
