// @vitest-environment jsdom
import { describe, it, expect, afterEach } from "vitest";
import { isCloudflarePagesHost, getApiBaseUrl } from "../api";

describe("isCloudflarePagesHost", () => {
  it("identifies canonical production domains", () => {
    expect(isCloudflarePagesHost("arxterminal.com")).toBe(true);
    expect(isCloudflarePagesHost("www.arxterminal.com")).toBe(true);
  });

  it("identifies official Cloudflare Pages root project and preview deployments", () => {
    expect(isCloudflarePagesHost("finance-xp8.pages.dev")).toBe(true);
    expect(isCloudflarePagesHost("7b970fd8.finance-xp8.pages.dev")).toBe(true);
    expect(isCloudflarePagesHost("fix-production-candle-chart.finance-xp8.pages.dev")).toBe(true);
    expect(isCloudflarePagesHost("d4881bd6.finance-xp8.pages.dev")).toBe(true);
  });

  it("rejects non-target Cloudflare Pages domains and arbitrary hostnames (fails closed)", () => {
    expect(isCloudflarePagesHost("other.pages.dev")).toBe(false);
    expect(isCloudflarePagesHost("fakefinance-xp8.pages.dev")).toBe(false);
    expect(isCloudflarePagesHost("finance-xp8.pages.dev.attacker.com")).toBe(false);
    expect(isCloudflarePagesHost("pages.dev")).toBe(false);
    expect(isCloudflarePagesHost("attacker.com")).toBe(false);
    expect(isCloudflarePagesHost("localhost")).toBe(false);
    expect(isCloudflarePagesHost("127.0.0.1")).toBe(false);
    expect(isCloudflarePagesHost("")).toBe(false);
    expect(isCloudflarePagesHost(null as unknown as string)).toBe(false);
    expect(isCloudflarePagesHost(undefined as unknown as string)).toBe(false);
  });
});

describe("getApiBaseUrl browser proxy resolution", () => {
  const originalLocation = window.location;

  afterEach(() => {
    Object.defineProperty(window, "location", {
      writable: true,
      value: originalLocation,
    });
  });

  it("returns same-origin proxy /api/backend on canonical production domain", () => {
    Object.defineProperty(window, "location", {
      writable: true,
      value: { hostname: "arxterminal.com" },
    });
    expect(getApiBaseUrl()).toBe("/api/backend");
  });

  it("returns same-origin proxy /api/backend on Cloudflare Pages preview deployments", () => {
    Object.defineProperty(window, "location", {
      writable: true,
      value: { hostname: "7b970fd8.finance-xp8.pages.dev" },
    });
    expect(getApiBaseUrl()).toBe("/api/backend");

    Object.defineProperty(window, "location", {
      writable: true,
      value: { hostname: "finance-xp8.pages.dev" },
    });
    expect(getApiBaseUrl()).toBe("/api/backend");
  });

  it("returns raw backend API URL when running outside Cloudflare Pages", () => {
    Object.defineProperty(window, "location", {
      writable: true,
      value: { hostname: "localhost" },
    });
    const url = getApiBaseUrl();
    expect(url).not.toBe("/api/backend");
    expect(url.endsWith("/api/v1")).toBe(true);
  });
});
