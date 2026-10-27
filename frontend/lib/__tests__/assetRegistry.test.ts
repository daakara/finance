import { describe, it, expect } from "vitest";
import {
  normalizeAssetSymbol,
  isCanonicalSecurityIdentifier,
  resolveCanonicalSecurity,
} from "../assetRegistry";

describe("normalizeAssetSymbol (INV-RADAR-PORTFOLIO-07)", () => {
  it("normalizes case and trims whitespace", () => {
    expect(normalizeAssetSymbol("  nvda  ")).toBe("NVDA");
    expect(normalizeAssetSymbol("aapl")).toBe("AAPL");
    expect(normalizeAssetSymbol("TsLa")).toBe("TSLA");
  });

  it("normalizes US dual-class share dot notation to hyphen", () => {
    expect(normalizeAssetSymbol("BRK.B")).toBe("BRK-B");
    expect(normalizeAssetSymbol("brk.b")).toBe("BRK-B");
    expect(normalizeAssetSymbol("BF.A")).toBe("BF-A");
    expect(normalizeAssetSymbol("BF.B")).toBe("BF-B");
  });

  it("preserves international exchange dot notation suffixes", () => {
    expect(normalizeAssetSymbol("SHEL.L")).toBe("SHEL.L");
    expect(normalizeAssetSymbol("shel.l")).toBe("SHEL.L");
    expect(normalizeAssetSymbol("SAP.DE")).toBe("SAP.DE");
  });

  it("handles standard hyphenated share classes", () => {
    expect(normalizeAssetSymbol("BRK-B")).toBe("BRK-B");
    expect(normalizeAssetSymbol("BF-B")).toBe("BF-B");
  });

  it("rejects invalid, empty, or unparseable input symbols (fails closed)", () => {
    expect(normalizeAssetSymbol("")).toBe(null);
    expect(normalizeAssetSymbol("   ")).toBe(null);
    expect(normalizeAssetSymbol(null)).toBe(null);
    expect(normalizeAssetSymbol(undefined)).toBe(null);
    expect(normalizeAssetSymbol("INVALID$$$TICKER")).toBe(null);
    expect(normalizeAssetSymbol("TOOLONGTICKEREXCEEDINGLIMITS")).toBe(null);
  });
});

describe("isCanonicalSecurityIdentifier & resolveCanonicalSecurity (MOB-03)", () => {
  const catalog = [{ ticker: "NVDA" }, { ticker: "TSLA" }, { ticker: "CPRX" }, { ticker: "BRK.B" }];

  it("accepts valid canonical security identifiers", () => {
    expect(isCanonicalSecurityIdentifier("NVDA", catalog)).toBe(true);
    expect(isCanonicalSecurityIdentifier("tsla", catalog)).toBe(true);
    expect(isCanonicalSecurityIdentifier("  AAPL  ", catalog)).toBe(true);
    expect(isCanonicalSecurityIdentifier("BRK.B", catalog)).toBe(true);
    expect(resolveCanonicalSecurity("NVDA", catalog)).toBe("NVDA");
    expect(resolveCanonicalSecurity(" tsla ", catalog)).toBe("TSLA");
    expect(resolveCanonicalSecurity("BRK.B", catalog)).toBe("BRK.B");
  });

  it("rejects non-ticker search keywords and sectors (MOB-03)", () => {
    expect(isCanonicalSecurityIdentifier("ENERGY", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("energy", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("TECH", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("VALUE", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("GROWTH", catalog)).toBe(false);
    expect(resolveCanonicalSecurity("ENERGY", catalog)).toBe(null);
    expect(resolveCanonicalSecurity("tech", catalog)).toBe(null);
  });

  it("rejects company names without explicit ticker resolution", () => {
    expect(isCanonicalSecurityIdentifier("Apple Inc.", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("Tesla, Inc.", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("NVIDIA Corporation", catalog)).toBe(false);
    expect(resolveCanonicalSecurity("Apple Inc.", catalog)).toBe(null);
  });

  it("rejects invalid, empty, and whitespace-only queries", () => {
    expect(isCanonicalSecurityIdentifier("", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("   ", catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier(null, catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier(undefined, catalog)).toBe(false);
    expect(isCanonicalSecurityIdentifier("INVALID!@#", catalog)).toBe(false);
    expect(resolveCanonicalSecurity(null, catalog)).toBe(null);
    expect(resolveCanonicalSecurity("   ", catalog)).toBe(null);
  });
});
