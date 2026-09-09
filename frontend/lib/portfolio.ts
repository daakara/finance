"use client";

/**
 * Robust Zero-Login Portfolio Engine.
 * Stored persistently in browser LocalStorage with automatic PnL calculations,
 * position sizing metrics, and anonymous user journey attribution.
 */

import { trackPortfolioPositionAdded } from "./matomo";

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
    // Clean session policy: zero synthetic starter holdings
    return [];
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

/**
 * Syncs portfolio holdings from authoritative backend API (SQLite store).
 * Automatically preserves and migrates existing local holdings.
 */
export async function syncPortfolioFromApi(): Promise<PortfolioPosition[]> {
  if (typeof window === "undefined") return [];
  const localPositions = loadPortfolioPositions();

  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-e370b.up.railway.app/api/v1";
    const anonId = getAnonymousUserId();
    const res = await fetch(`${baseUrl}/portfolio`, {
      headers: {
        "Content-Type": "application/json",
        "X-User-Id": anonId,
      },
      signal: AbortSignal.timeout(4000),
    });

    if (res.ok) {
      const apiHoldings = await res.json();
      if (Array.isArray(apiHoldings)) {
        if (apiHoldings.length === 0 && localPositions.length > 0) {
          // Transparent Migration: Local holdings exist but API store is empty, migrate them
          fetch(`${baseUrl}/portfolio/migrate`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-User-Id": anonId,
            },
            body: JSON.stringify({ holdings: localPositions }),
          }).catch((err) => console.warn("Background migration skipped:", err));
          return localPositions;
        }

        // Save fresh authoritative holdings from API to local cache
        savePortfolioPositions(apiHoldings);
        return apiHoldings;
      }
    }
  } catch (err) {
    // Backend temporarily unreachable, return cached local positions
  }
  return localPositions;
}

export async function persistHoldingToApi(holding: PortfolioPosition): Promise<void> {
  if (typeof window === "undefined") return;
  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-e370b.up.railway.app/api/v1";
    const anonId = getAnonymousUserId();
    await fetch(`${baseUrl}/portfolio`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-Id": anonId,
      },
      body: JSON.stringify(holding),
    });
  } catch (err) {
    console.warn("Could not persist holding to backend API:", err);
  }
}

export async function removeHoldingFromApi(symbol: string): Promise<void> {
  if (typeof window === "undefined") return;
  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-e370b.up.railway.app/api/v1";
    const anonId = getAnonymousUserId();
    await fetch(`${baseUrl}/portfolio/${encodeURIComponent(symbol)}`, {
      method: "DELETE",
      headers: {
        "X-User-Id": anonId,
      },
    });
  } catch (err) {
    console.warn("Could not delete holding from backend API:", err);
  }
}

export function addPortfolioPosition(pos: {
  symbol: string;
  name: string;
  shares?: number;
  entryPrice: number;
  currentPrice: number;
  targetPrice?: number;
  stopLossPrice?: number;
  assetType?: "Stock" | "ETF" | "Crypto";
}): { success: boolean; isDuplicate: boolean; message: string } {
  if (typeof window === "undefined") return { success: false, isDuplicate: false, message: "Window undefined" };
  try {
    const existing = loadPortfolioPositions();
    const symUpper = (pos.symbol || "").toUpperCase().trim();
    const existingIdx = existing.findIndex((p) => p.symbol.toUpperCase() === symUpper);

    if (existingIdx >= 0) {
      return {
        success: false,
        isDuplicate: true,
        message: `${symUpper} is already in your Paper Portfolio`,
      };
    }

    if (!pos.entryPrice || pos.entryPrice <= 0) {
      return {
        success: false,
        isDuplicate: false,
        message: `Cannot add ${symUpper} to portfolio without a verified market price.`,
      };
    }

    const calculatedShares = (pos.shares && pos.shares > 0)
      ? pos.shares
      : Math.max(1, Math.round(2500 / pos.entryPrice));

    const newPos: PortfolioPosition = {
      symbol: symUpper,
      name: pos.name || symUpper,
      shares: calculatedShares,
      entryPrice: pos.entryPrice,
      currentPrice: (pos.currentPrice && pos.currentPrice > 0) ? pos.currentPrice : pos.entryPrice,
      targetPrice: pos.targetPrice,
      stopLossPrice: pos.stopLossPrice,
      addedAt: new Date().toISOString().split("T")[0],
      assetType: pos.assetType || (symUpper.includes("-USD") || ["BTC", "ETH", "SOL"].includes(symUpper) ? "Crypto" : "Stock"),
    };

    savePortfolioPositions([newPos, ...existing]);
    persistHoldingToApi(newPos).catch(() => {});
    window.dispatchEvent(new CustomEvent("finance:portfolio-updated"));
    trackPortfolioPositionAdded(symUpper, newPos.shares * newPos.entryPrice);
    return {
      success: true,
      isDuplicate: false,
      message: `Added ${symUpper} to Paper Portfolio!`,
    };
  } catch (err) {
    console.error("Failed to add portfolio position:", err);
    return { success: false, isDuplicate: false, message: "Failed to save position" };
  }
}

export function updatePortfolioPosition(pos: {
  symbol: string;
  shares: number;
  entryPrice?: number;
  currentPrice?: number;
  targetPrice?: number;
  stopLossPrice?: number;
  name?: string;
}): { success: boolean; message: string } {
  if (typeof window === "undefined") return { success: false, message: "Window undefined" };
  try {
    const existing = loadPortfolioPositions();
    const symUpper = (pos.symbol || "").toUpperCase().trim();
    const idx = existing.findIndex((p) => p.symbol.toUpperCase() === symUpper);
    if (idx < 0) {
      return { success: false, message: `${symUpper} position not found in portfolio` };
    }
    const current = existing[idx];
    const updatedPos: PortfolioPosition = {
      ...current,
      shares: pos.shares,
      entryPrice: pos.entryPrice !== undefined ? pos.entryPrice : current.entryPrice,
      currentPrice: pos.currentPrice !== undefined ? pos.currentPrice : current.currentPrice,
      targetPrice: pos.targetPrice !== undefined ? pos.targetPrice : current.targetPrice,
      stopLossPrice: pos.stopLossPrice !== undefined ? pos.stopLossPrice : current.stopLossPrice,
      name: pos.name || current.name,
    };
    const updatedList = [...existing];
    updatedList[idx] = updatedPos;
    savePortfolioPositions(updatedList);
    persistHoldingToApi(updatedPos).catch(() => {});
    window.dispatchEvent(new CustomEvent("finance:portfolio-updated"));
    return { success: true, message: `Updated ${symUpper} holding (${pos.shares} shares)!` };
  } catch (err) {
    console.error("Failed to update portfolio position:", err);
    return { success: false, message: "Failed to update position" };
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

export function exportPortfolioToCsv(positions: PortfolioPosition[]): void {
  if (typeof window === "undefined" || positions.length === 0) return;
  const headers = [
    "Symbol",
    "Name",
    "Asset Type",
    "Shares",
    "Entry Price ($)",
    "Current Price ($)",
    "Target Price ($)",
    "Stop Loss ($)",
    "Cost Basis ($)",
    "Market Value ($)",
    "Unrealized P&L ($)",
    "Unrealized P&L (%)",
    "Added Date",
  ];

  const rows = positions.map((pos) => {
    const cost = pos.shares * pos.entryPrice;
    const value = pos.shares * pos.currentPrice;
    const pnl = value - cost;
    const pnlPct = cost > 0 ? (pnl / cost) * 100 : 0;
    return [
      pos.symbol,
      `"${pos.name.replace(/"/g, '""')}"`,
      pos.assetType,
      pos.shares,
      pos.entryPrice.toFixed(2),
      pos.currentPrice.toFixed(2),
      pos.targetPrice ? pos.targetPrice.toFixed(2) : "N/A",
      pos.stopLossPrice ? pos.stopLossPrice.toFixed(2) : "N/A",
      cost.toFixed(2),
      value.toFixed(2),
      pnl.toFixed(2),
      `${pnlPct.toFixed(2)}%`,
      pos.addedAt || "N/A",
    ].join(",");
  });

  const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent([headers.join(","), ...rows].join("\n"));
  const link = document.createElement("a");
  link.setAttribute("href", csvContent);
  link.setAttribute("download", `finance_terminal_portfolio_${new Date().toISOString().split("T")[0]}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}