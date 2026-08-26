"use client";

/**
 * Robust Zero-Login Portfolio Engine.
 * Stored persistently in browser LocalStorage with automatic PnL calculations,
 * position sizing metrics, and anonymous user journey attribution.
 */

export interface PortfolioPosition {
  symbol: string;
  name: string;
  shares: number;
  entryPrice: number;
  currentPrice: number;
  targetPrice?: number;
  stopLossPrice?: number;
  addedAt: string;
  assetType: "Stock" | "ETF" | "Crypto";
}

export interface PortfolioSummary {
  totalEquity: number;
  totalCost: number;
  totalUnrealizedPnL: number;
  totalUnrealizedPnLPct: number;
  positionsCount: number;
}

const STORAGE_KEY = "FINANCE_USER_PORTFOLIO";

export function getAnonymousUserId(): string {
  if (typeof window === "undefined") return "trader_anon";
  try {
    let id = localStorage.getItem("FINANCE_ANON_USER_ID");
    if (!id) {
      id = "trader_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now().toString(36);
      localStorage.setItem("FINANCE_ANON_USER_ID", id);
    }
    return id;
  } catch {
    return "trader_fallback";
  }
}

export function loadPortfolioPositions(): PortfolioPosition[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      return JSON.parse(raw);
    }
    // High-quality starter portfolio
    const defaultPositions: PortfolioPosition[] = [
      {
        symbol: "NVDA",
        name: "NVIDIA Corp.",
        shares: 25,
        entryPrice: 185.50,
        currentPrice: 213.05,
        targetPrice: 245.00,
        stopLossPrice: 172.00,
        addedAt: "2026-08-01",
        assetType: "Stock",
      },
      {
        symbol: "AAPL",
        name: "Apple Inc.",
        shares: 15,
        entryPrice: 295.00,
        currentPrice: 309.90,
        targetPrice: 340.00,
        stopLossPrice: 280.00,
        addedAt: "2026-08-10",
        assetType: "Stock",
      },
      {
        symbol: "LNTH",
        name: "Lantheus Holdings",
        shares: 40,
        entryPrice: 68.20,
        currentPrice: 74.80,
        targetPrice: 88.00,
        stopLossPrice: 63.50,
        addedAt: "2026-08-15",
        assetType: "Stock",
      },
    ];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(defaultPositions));
    return defaultPositions;
  } catch (err) {
    console.warn("Could not load portfolio from storage:", err);
    return [];
  }
}

export function savePortfolioPositions(positions: PortfolioPosition[]): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(positions));
  } catch (err) {
    console.error("Failed to save portfolio positions:", err);
  }
}

export function calculatePortfolioSummary(positions: PortfolioPosition[]): PortfolioSummary {
  let totalEquity = 0;
  let totalCost = 0;

  positions.forEach((pos) => {
    const cost = pos.shares * pos.entryPrice;
    const equity = pos.shares * pos.currentPrice;
    totalCost += cost;
    totalEquity += equity;
  });

  const totalUnrealizedPnL = totalEquity - totalCost;
  const totalUnrealizedPnLPct = totalCost > 0 ? (totalUnrealizedPnL / totalCost) * 100 : 0;

  return {
    totalEquity,
    totalCost,
    totalUnrealizedPnL,
    totalUnrealizedPnLPct,
    positionsCount: positions.length,
  };
}