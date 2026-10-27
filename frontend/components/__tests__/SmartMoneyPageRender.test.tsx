import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import SmartMoneyPage from "../../app/smart-money/page";

describe("SmartMoneyPage React Render Integration", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("distinguishes genuine zero count from unavailable count when API returns authoritative empty arrays", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: [],
        sec_insider_trades: [],
        options_flow: [],
      }),
    } as any);

    render(<SmartMoneyPage />);

    // Wait for data load
    await waitFor(() => {
      // Disclosures and SEC Insiders quick stats badges should render '0' (not 'Unavailable')
      const zeros = screen.getAllByText("0");
      expect(zeros.length).toBeGreaterThanOrEqual(2);
    });

    // Confirms authentic zero disclosures in active filter is displayed, not offline failure banner
    expect(screen.getByText(/Zero filings in active/i)).toBeDefined();
    expect(screen.queryByText(/⚠️ Feed Offline \/ Connection Failure/i)).toBeNull();
  });

  it("renders 'Unavailable' and offline banner on HTTP 500 error", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => ({ error: "Internal Server Error" }),
    } as any);

    render(<SmartMoneyPage />);

    await waitFor(() => {
      // Offline failure banner must be present
      expect(screen.getByText(/⚠️ Feed Offline \/ Connection Failure/i)).toBeDefined();
    });

    // Disclosures and SEC Insiders must display 'Unavailable'
    const unavailables = screen.getAllByText("Unavailable");
    expect(unavailables.length).toBeGreaterThanOrEqual(2);
  });

  it("handles partial availability: renders authentic count for present feeds and Unavailable for omitted feeds", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: [
          {
            id: "cong-1",
            politician: "Dan Crenshaw (R-TX)",
            chamber: "House",
            ticker: "PLTR",
            asset_name: "Palantir Technologies",
            sector: "Defense & Enterprise AI",
            transaction_type: "Purchase (Common Stock)",
            amount_range: "$50,000 - $100,000",
            filing_date: new Date().toISOString().slice(0, 10),
            transaction_date: new Date().toISOString().slice(0, 10),
            days_to_filing: 2,
            performance_since_pct: 12.1,
            sentiment: "Strong Bullish",
            signal_strength: 95,
          },
        ],
        // sec_insider_trades is omitted
        options_flow: [],
      }),
    } as any);

    render(<SmartMoneyPage />);

    await waitFor(() => {
      // Congress disclosures should display 1
      expect(screen.getByText("1")).toBeDefined();
    });

    // SEC Form 4 tab button must display Unavailable, NOT 0
    expect(screen.getByText(/Corporate Insiders SEC Form 4 \(Unavailable\)/i)).toBeDefined();
    // Congress tab button must display (1)
    expect(screen.getByText(/Capitol Hill STOCK Act \(1\)/i)).toBeDefined();
  });

  it("handles non-array corrupt feed: fails shape validation, triggers catch and renders 'Unavailable'", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "LIVE",
        congress_trades: "malformed_string_payload",
        sec_insider_trades: [],
      }),
    } as any);

    render(<SmartMoneyPage />);

    await waitFor(() => {
      expect(screen.getByText(/⚠️ Feed Offline \/ Connection Failure/i)).toBeDefined();
    });

    const unavailables = screen.getAllByText("Unavailable");
    expect(unavailables.length).toBeGreaterThanOrEqual(2);
  });
});
