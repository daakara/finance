import { describe, it, expect } from "vitest";
import { normalizeAssetSymbol } from "../assetRegistry";

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
