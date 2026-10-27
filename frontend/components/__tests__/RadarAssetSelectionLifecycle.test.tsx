import React, { useState, useMemo } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import SynthesisMobileDock from "../synthesis/SynthesisMobileDock";
import {
  isCanonicalSecurityIdentifier,
  resolveCanonicalSecurity,
} from "../../lib/assetRegistry";

vi.mock("next/navigation", () => ({
  usePathname: () => "/radar",
  useSearchParams: () => new URLSearchParams(),
}));

// Test harness implementing the exact Radar selection & activeCanonicalAsset lifecycle
function RadarHarness({
  initialUniverse = [
    { ticker: "NVDA", name: "NVIDIA Corp" },
    { ticker: "TSLA", name: "Tesla Inc" },
    { ticker: "XOM", name: "Exxon Mobil" },
  ],
  initialTicker = "NVDA",
}: {
  initialUniverse?: { ticker: string; name: string }[];
  initialTicker?: string | null;
}) {
  const [allAssets, setAllAssets] = useState(initialUniverse);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAssetTicker, setSelectedAssetTicker] = useState<string | null>(initialTicker);

  const cleanQ = searchQuery.trim().toUpperCase();

  const filteredAssets = useMemo(() => {
    if (!cleanQ) return allAssets;
    return allAssets.filter(
      (a) =>
        a.ticker.toUpperCase().includes(cleanQ) ||
        a.name.toUpperCase().includes(cleanQ)
    );
  }, [allAssets, cleanQ]);

  const activeCanonicalAsset = useMemo(() => {
    // 1. Active ticker query supersedes prior selection
    if (cleanQ && isCanonicalSecurityIdentifier(cleanQ, allAssets)) {
      return resolveCanonicalSecurity(cleanQ, allAssets);
    }

    // 2. Explicit candidate row selection is authoritative when valid and visible in current filtered set
    if (selectedAssetTicker && isCanonicalSecurityIdentifier(selectedAssetTicker, allAssets)) {
      const isVisibleInFilter = filteredAssets.length === 0 ? true : filteredAssets.some((a) => a.ticker === selectedAssetTicker);
      if (isVisibleInFilter) {
        return resolveCanonicalSecurity(selectedAssetTicker, allAssets);
      }
    }

    return null;
  }, [selectedAssetTicker, cleanQ, filteredAssets, allAssets]);

  return (
    <div>
      <div data-testid="active-symbol">{activeCanonicalAsset || "NONE"}</div>
      <input
        data-testid="search-input"
        aria-label="Search candidates"
        value={searchQuery}
        onChange={(e) => {
          const nextQ = e.target.value;
          setSearchQuery(nextQ);
          if (selectedAssetTicker && nextQ.trim().toUpperCase() !== selectedAssetTicker) {
            setSelectedAssetTicker(null);
          }
        }}
      />
      <button
        data-testid="clear-search-btn"
        onClick={() => {
          setSearchQuery("");
          setSelectedAssetTicker(null);
        }}
      >
        Clear
      </button>

      <ul>
        {filteredAssets.map((asset) => (
          <li key={asset.ticker}>
            <button
              data-testid={`row-${asset.ticker}`}
              onClick={() => setSelectedAssetTicker(asset.ticker)}
            >
              {asset.ticker}
            </button>
          </li>
        ))}
      </ul>

      <SynthesisMobileDock activeSymbol={activeCanonicalAsset} />
    </div>
  );
}

describe("Radar Asset Selection Lifecycle & React Integration Suite", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("Transition 1: NVDA selected → typing 'ENERGY' invalidates selection and reverts dock to neutral", () => {
    render(<RadarHarness initialTicker="NVDA" />);

    // Initial state: NVDA selected
    const analysisLink = screen.getByRole("link", { name: /Analysis/i });
    const planLink = screen.getByRole("link", { name: /Plan/i });
    expect(analysisLink.getAttribute("href")).toBe("/?symbol=NVDA");
    expect(planLink.getAttribute("href")).toBe("/setups?ticker=NVDA");
    expect(screen.getByTestId("active-symbol").textContent).toBe("NVDA");

    // User types "ENERGY"
    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "ENERGY" } });

    // Active symbol must become NONE, and dock links must revert to neutral
    expect(screen.getByTestId("active-symbol").textContent).toBe("NONE");
    expect(analysisLink.getAttribute("href")).toBe("/");
    expect(planLink.getAttribute("href")).toBe("/setups");
  });

  it("Transition 2: NVDA selected → typing unrelated company name invalidates NVDA", () => {
    render(<RadarHarness initialTicker="NVDA" />);

    const analysisLink = screen.getByRole("link", { name: /Analysis/i });
    expect(analysisLink.getAttribute("href")).toBe("/?symbol=NVDA");

    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "Apple Inc" } });

    expect(screen.getByTestId("active-symbol").textContent).toBe("NONE");
    expect(analysisLink.getAttribute("href")).toBe("/");
  });

  it("Transition 3: NVDA selected → typing 'TSLA' without clicking row supersedes NVDA with TSLA", () => {
    render(<RadarHarness initialTicker="NVDA" />);

    const analysisLink = screen.getByRole("link", { name: /Analysis/i });
    const planLink = screen.getByRole("link", { name: /Plan/i });
    expect(analysisLink.getAttribute("href")).toBe("/?symbol=NVDA");

    const input = screen.getByTestId("search-input");
    fireEvent.change(input, { target: { value: "TSLA" } });

    expect(screen.getByTestId("active-symbol").textContent).toBe("TSLA");
    expect(analysisLink.getAttribute("href")).toBe("/?symbol=TSLA");
    expect(planLink.getAttribute("href")).toBe("/setups?ticker=TSLA");
  });

  it("Transition 4: Explicit candidate click sets new authoritative selection", () => {
    render(<RadarHarness initialTicker="NVDA" />);

    const tslaRowBtn = screen.getByTestId("row-TSLA");
    fireEvent.click(tslaRowBtn);

    expect(screen.getByTestId("active-symbol").textContent).toBe("TSLA");
    const analysisLink = screen.getByRole("link", { name: /Analysis/i });
    expect(analysisLink.getAttribute("href")).toBe("/?symbol=TSLA");
  });

  it("Transition 5: Clear button resets both query and selected asset", () => {
    render(<RadarHarness initialTicker="NVDA" />);

    const clearBtn = screen.getByTestId("clear-search-btn");
    fireEvent.click(clearBtn);

    expect(screen.getByTestId("active-symbol").textContent).toBe("NONE");
    const analysisLink = screen.getByRole("link", { name: /Analysis/i });
    const planLink = screen.getByRole("link", { name: /Plan/i });
    expect(analysisLink.getAttribute("href")).toBe("/");
    expect(planLink.getAttribute("href")).toBe("/setups");
  });
});
