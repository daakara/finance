import { describe, it, expect } from "vitest";
import {
  resolveAssetType,
  isETF,
  isStock,
  isCrypto,
  isUnknownAsset,
  hasEtfSectorData,
} from "../../lib/assetTypeUtils";

describe("assetTypeUtils (Canonical Asset Type Discrimination)", () => {
  it("resolves canonical stocks authoritatively from masterCatalog and watchlist", () => {
    expect(resolveAssetType("NVDA")).toBe("Stock");
    expect(resolveAssetType("AAPL")).toBe("Stock");
    expect(resolveAssetType("MSFT")).toBe("Stock");

    expect(isStock("NVDA")).toBe(true);
    expect(isETF("NVDA")).toBe(false);
    expect(isCrypto("NVDA")).toBe(false);
    expect(isUnknownAsset("NVDA")).toBe(false);
  });

  it("resolves canonical ETFs authoritatively from masterCatalog and watchlist", () => {
    expect(resolveAssetType("SPY")).toBe("ETF");
    expect(resolveAssetType("QQQ")).toBe("ETF");
    expect(resolveAssetType("SMH")).toBe("ETF");
    expect(resolveAssetType("XLK")).toBe("ETF");
    expect(resolveAssetType("IWM")).toBe("ETF");
    expect(resolveAssetType("GLD")).toBe("ETF");
    expect(resolveAssetType("TLT")).toBe("ETF");
    expect(resolveAssetType("XLE")).toBe("ETF");

    expect(isETF("SPY")).toBe(true);
    expect(isStock("SPY")).toBe(false);
    expect(isCrypto("SPY")).toBe(false);
    expect(isUnknownAsset("SPY")).toBe(false);
  });

  it("resolves canonical Crypto pairs ending with -USD", () => {
    expect(resolveAssetType("BTC-USD")).toBe("Crypto");
    expect(resolveAssetType("ETH-USD")).toBe("Crypto");
    expect(resolveAssetType("SOL-USD")).toBe("Crypto");

    expect(isCrypto("BTC-USD")).toBe(true);
    expect(isStock("BTC-USD")).toBe(false);
    expect(isETF("BTC-USD")).toBe(false);
    expect(isUnknownAsset("BTC-USD")).toBe(false);
  });

  it("handles case normalization deterministically", () => {
    expect(resolveAssetType("spy")).toBe("ETF");
    expect(resolveAssetType("  nvda  ")).toBe("Stock");
    expect(resolveAssetType("btc-usd")).toBe("Crypto");

    expect(isETF("spy")).toBe(true);
    expect(isStock("aapl")).toBe(true);
  });

  it("strictly fails closed as UNKNOWN for unknown / uncataloged symbols (zero silent Stock fallback)", () => {
    const unknownSymbols = [
      "TOTALLY_UNKNOWN_TICKER",
      "UNLISTED_CORP_999",
      "RANDOM_XYZ",
      "ETF_WITHOUT_CATALOG_ENTRY",
    ];

    for (const sym of unknownSymbols) {
      expect(resolveAssetType(sym)).toBe("UNKNOWN");
      expect(isStock(sym)).toBe(false);
      expect(isETF(sym)).toBe(false);
      expect(isCrypto(sym)).toBe(false);
      expect(isUnknownAsset(sym)).toBe(true);
    }
  });

  it("handles empty or null/undefined inputs safely as UNKNOWN", () => {
    expect(resolveAssetType("")).toBe("UNKNOWN");
    expect(resolveAssetType("   ")).toBe("UNKNOWN");
    expect(resolveAssetType(null)).toBe("UNKNOWN");
    expect(resolveAssetType(undefined)).toBe("UNKNOWN");

    expect(isStock(null)).toBe(false);
    expect(isETF(null)).toBe(false);
    expect(isCrypto(null)).toBe(false);
    expect(isUnknownAsset(null)).toBe(true);
  });

  it("verifies canonical sector weight availability correctly", () => {
    expect(hasEtfSectorData("SPY")).toBe(true);
    expect(hasEtfSectorData("QQQ")).toBe(true);
    expect(hasEtfSectorData("NVDA")).toBe(false);
    expect(hasEtfSectorData("UNKNOWN_FUND")).toBe(false);
  });
});
