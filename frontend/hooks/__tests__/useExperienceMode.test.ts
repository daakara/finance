import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  isValidMode,
  normalizeMode,
  STORAGE_KEY,
} from "../useExperienceMode";

describe("useExperienceMode helper logic", () => {
  it("validates allowed experience modes", () => {
    expect(isValidMode("guided")).toBe(true);
    expect(isValidMode("standard")).toBe(true);
    expect(isValidMode("quant")).toBe(true);
    expect(isValidMode("GUIDED")).toBe(true);
    expect(isValidMode("STANDARD")).toBe(true);
    expect(isValidMode("QUANT")).toBe(true);

    expect(isValidMode("foobar")).toBe(false);
    expect(isValidMode("banana")).toBe(false);
    expect(isValidMode(null)).toBe(false);
    expect(isValidMode("")).toBe(false);
  });

  it("normalizes mode strings and aliases", () => {
    expect(normalizeMode("guided")).toBe("guided");
    expect(normalizeMode("STANDARD")).toBe("standard");
    expect(normalizeMode("quant")).toBe("quant");
    expect(normalizeMode("advanced")).toBe("quant"); // Backward-compatible alias
    expect(normalizeMode("ADVANCED")).toBe("quant");
    expect(normalizeMode("invalid")).toBe(null);
    expect(normalizeMode(null)).toBe(null);
  });
});

describe("ADR-003 precedence rules", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("prioritizes URL mode over localStorage (ADR-003 Precedence #1)", () => {
    localStorage.setItem(STORAGE_KEY, "quant");
    const urlParam = "guided";
    const resolvedMode = normalizeMode(urlParam) || normalizeMode(localStorage.getItem(STORAGE_KEY)) || "standard";
    expect(resolvedMode).toBe("guided");
  });

  it("restores localStorage mode when URL parameter is absent (ADR-003 Precedence #2)", () => {
    localStorage.setItem(STORAGE_KEY, "quant");
    const urlParam = null;
    const resolvedMode = normalizeMode(urlParam) || normalizeMode(localStorage.getItem(STORAGE_KEY)) || "standard";
    expect(resolvedMode).toBe("quant");
  });

  it("ignores corrupt localStorage and falls back to STANDARD (ADR-003 Precedence #3)", () => {
    localStorage.setItem(STORAGE_KEY, "banana_12345");
    const urlParam = null;
    const resolvedMode = normalizeMode(urlParam) || normalizeMode(localStorage.getItem(STORAGE_KEY)) || "standard";
    expect(resolvedMode).toBe("standard");
  });

  it("falls back to STANDARD for invalid URL parameter", () => {
    const rawUrl = "invalid_mode";
    const normalized = normalizeMode(rawUrl);
    const resolvedMode = normalized || "standard";
    expect(resolvedMode).toBe("standard");
  });
});
