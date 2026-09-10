"use client";

/**
 * Robust Zero-Login Portfolio Engine.
 * Stored persistently in browser LocalStorage with automatic PnL calculations,
 * position sizing metrics, and anonymous user journey attribution.
 *
 * Epistemic Invariants:
 * - Truthful persistence: Writes await API confirmation and propagate errors.
 * - Honest pricing: Null/unverified prices are preserved; never substituted with entry price.
 * - Explicit quantities: Invalid/missing shares are rejected; no manufactured $2,500 allocations.
 * - Active edit isolation: Background sync never clobbers active UI edits.
 */

import { trackPortfolioPositionAdded } from "./matomo";

export interface PortfolioPosition {
  symbol: string;
  name: string;
  shares: number;
  entryPrice: number;
  currentPrice: number | null;
  targetPrice?: number;
  stopLossPrice?: number;
  addedAt: string;
  assetType: "Stock" | "ETF" | "Crypto";
}

export interface PortfolioSummary {
  totalEquity: number | null;
  totalCost: number;
  totalUnrealizedPnL: number | null;
  totalUnrealizedPnLPct: number | null;
  positionsCount: number;
  isComplete: boolean;
  unpricedCount: number;
}

const STORAGE_KEY = "FINANCE_USER_PORTFOLIO";

let activeEditCount = 0;

export function beginActivePortfolioEdit(): void {
  activeEditCount += 1;
}

export function endActivePortfolioEdit(): void {
  activeEditCount = Math.max(0, activeEditCount - 1);
}

export function isPortfolioEditActive(): boolean {
  return activeEditCount > 0;
}

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
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed.map((p) => ({
          ...p,
          currentPrice: (p.currentPrice !== undefined && p.currentPrice !== null && !isNaN(Number(p.currentPrice)) && Number(p.currentPrice) > 0)
            ? Number(p.currentPrice)
            : null,
          shares: Number(p.shares),
          entryPrice: Number(p.entryPrice),
        }));
      }
    }
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

let activeReadSequence = 0;
let lastConfirmedWriteTimestamp = 0;

export function notifyPortfolioWriteConfirmed(): void {
  lastConfirmedWriteTimestamp = Date.now();
  activeReadSequence += 1;
}

export async function migrateLocalHoldingsToApi(holdings: PortfolioPosition[]): Promise<{
  success: boolean;
  migratedCount: number;
  totalSubmitted: number;
  failedCount: number;
  status?: string;
  error?: string;
}> {
  if (typeof window === "undefined" || holdings.length === 0) {
    return { success: true, migratedCount: 0, totalSubmitted: 0, failedCount: 0, status: "no_op" };
  }
  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-e370b.up.railway.app/api/v1";
    const anonId = getAnonymousUserId();
    const res = await fetch(`${baseUrl}/portfolio/migrate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-Id": anonId,
      },
      body: JSON.stringify({ holdings }),
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return {
        success: false,
        migratedCount: 0,
        totalSubmitted: holdings.length,
        failedCount: holdings.length,
        error: `Migration failed (${res.status}): ${errText || res.statusText}`,
      };
    }
    const data = await res.json();
    const migratedCount = Number(data.migratedCount) || 0;
    const totalSubmitted = Number(data.totalSubmitted) || holdings.length;
    const failedCount = Number(data.failedCount) || (totalSubmitted - migratedCount);

    if (totalSubmitted > 0 && migratedCount === 0) {
      return {
        success: false,
        migratedCount: 0,
        totalSubmitted,
        failedCount,
        status: "failed",
        error: "Zero holdings could be persisted by backend",
      };
    }

    return {
      success: true,
      migratedCount,
      totalSubmitted,
      failedCount,
      status: data.status || "migrated",
    };
  } catch (err: any) {
    return {
      success: false,
      migratedCount: 0,
      totalSubmitted: holdings.length,
      failedCount: holdings.length,
      error: err?.message || "Network error migrating holdings",
    };
  }
}

/**
 * Syncs portfolio holdings from authoritative backend API (SQLite store).
 * Automatically preserves and migrates existing local holdings.
 * Will NOT clobber state if an active edit session is in progress in the UI.
 * Prevents late reads from replacing state after a newer confirmed write.
 */
export async function syncPortfolioFromApi(): Promise<PortfolioPosition[]> {
  if (typeof window === "undefined") return [];
  const localPositions = loadPortfolioPositions();

  if (isPortfolioEditActive()) {
    return localPositions;
  }

  const readSeq = ++activeReadSequence;
  const readStartTime = Date.now();

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
        // Prevent stale read from overwriting state if an edit occurred while request was in-flight
        if (readSeq !== activeReadSequence || lastConfirmedWriteTimestamp > readStartTime || isPortfolioEditActive()) {
          return loadPortfolioPositions();
        }

        if (apiHoldings.length === 0 && localPositions.length > 0) {
          // Transparent Migration: Await migration and validate response
          const migrationResult = await migrateLocalHoldingsToApi(localPositions);
          if (!migrationResult.success || migrationResult.migratedCount === 0) {
            console.warn("Portfolio migration failed:", migrationResult.error);
            // Preserve original local records; do not overwrite with empty list
            return loadPortfolioPositions();
          }
          notifyPortfolioWriteConfirmed();
          return loadPortfolioPositions();
        }

        const normalized: PortfolioPosition[] = apiHoldings.map((h: any) => ({
          symbol: (h.symbol || "").toUpperCase(),
          name: h.name || h.symbol,
          shares: Number(h.shares),
          entryPrice: Number(h.entryPrice || h.entry_price),
          currentPrice: (h.currentPrice !== undefined && h.currentPrice !== null && !isNaN(Number(h.currentPrice)) && Number(h.currentPrice) > 0)
            ? Number(h.currentPrice)
            : (h.current_price !== undefined && h.current_price !== null && !isNaN(Number(h.current_price)) && Number(h.current_price) > 0)
            ? Number(h.current_price)
            : null,
          targetPrice: h.targetPrice ?? h.target_price,
          stopLossPrice: h.stopLossPrice ?? h.stop_loss,
          addedAt: h.addedAt || h.added_at || new Date().toISOString().split("T")[0],
          assetType: h.assetType || h.asset_type || "Stock",
        }));

        // Final currency re-check before saving to cache or returning
        if (readSeq !== activeReadSequence || lastConfirmedWriteTimestamp > readStartTime || isPortfolioEditActive()) {
          return loadPortfolioPositions();
        }

        savePortfolioPositions(normalized);
        return normalized;
      }
    }
  } catch (err) {
    // Backend temporarily unreachable or error, return cached local positions
    console.warn("Could not sync portfolio from API:", err);
  }
  return loadPortfolioPositions();
}

export async function persistHoldingToApi(holding: PortfolioPosition): Promise<{ success: boolean; error?: string }> {
  if (typeof window === "undefined") return { success: false, error: "Window undefined" };
  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-e370b.up.railway.app/api/v1";
    const anonId = getAnonymousUserId();
    const res = await fetch(`${baseUrl}/portfolio`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-Id": anonId,
      },
      body: JSON.stringify({
        symbol: holding.symbol,
        name: holding.name,
        shares: holding.shares,
        entryPrice: holding.entryPrice,
        currentPrice: holding.currentPrice,
        targetPrice: holding.targetPrice,
        stopLossPrice: holding.stopLossPrice,
        assetType: holding.assetType,
      }),
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return { success: false, error: `API error (${res.status}): ${errText || res.statusText}` };
    }
    return { success: true };
  } catch (err: any) {
    console.warn("Could not persist holding to backend API:", err);
    return { success: false, error: err?.message || "Network error persisting to API" };
  }
}

export async function removeHoldingFromApi(symbol: string): Promise<{ success: boolean; error?: string }> {
  if (typeof window === "undefined") return { success: false, error: "Window undefined" };
  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://web-production-e370b.up.railway.app/api/v1";
    const anonId = getAnonymousUserId();
    const res = await fetch(`${baseUrl}/portfolio/${encodeURIComponent(symbol)}`, {
      method: "DELETE",
      headers: {
        "X-User-Id": anonId,
      },
    });
    if (!res.ok && res.status !== 404) {
      const errText = await res.text().catch(() => "");
      return { success: false, error: `API error (${res.status}): ${errText || res.statusText}` };
    }
    return { success: true };
  } catch (err: any) {
    console.warn("Could not delete holding from backend API:", err);
    return { success: false, error: err?.message || "Network error deleting from API" };
  }
}

export async function addPortfolioPosition(pos: {
  symbol: string;
  name?: string;
  shares?: number;
  entryPrice: number;
  currentPrice?: number | null;
  targetPrice?: number;
  stopLossPrice?: number;
  assetType?: "Stock" | "ETF" | "Crypto";
}): Promise<{ success: boolean; isDuplicate: boolean; message: string }> {
  if (typeof window === "undefined") return { success: false, isDuplicate: false, message: "Window undefined" };
  try {
    const symUpper = (pos.symbol || "").toUpperCase().trim();
    if (!symUpper) {
      return { success: false, isDuplicate: false, message: "Valid asset symbol is required." };
    }

    if (pos.shares === undefined || pos.shares === null || isNaN(pos.shares) || pos.shares <= 0) {
      return {
        success: false,
        isDuplicate: false,
        message: "Explicit positive share quantity is required (e.g. 10, 0.25). Default position sizing is prohibited.",
      };
    }

    if (!pos.entryPrice || isNaN(pos.entryPrice) || pos.entryPrice <= 0) {
      return {
        success: false,
        isDuplicate: false,
        message: `Cannot add ${symUpper} without an authentic positive entry price.`,
      };
    }

    const existing = loadPortfolioPositions();
    const existingIdx = existing.findIndex((p) => p.symbol.toUpperCase() === symUpper);
    if (existingIdx >= 0) {
      return {
        success: false,
        isDuplicate: true,
        message: `${symUpper} is already in your Paper Portfolio`,
      };
    }

    const authenticCurrentPrice = (pos.currentPrice !== undefined && pos.currentPrice !== null && !isNaN(pos.currentPrice) && pos.currentPrice > 0)
      ? pos.currentPrice
      : null;

    const newPos: PortfolioPosition = {
      symbol: symUpper,
      name: pos.name || symUpper,
      shares: pos.shares,
      entryPrice: pos.entryPrice,
      currentPrice: authenticCurrentPrice,
      targetPrice: pos.targetPrice,
      stopLossPrice: pos.stopLossPrice,
      addedAt: new Date().toISOString().split("T")[0],
      assetType: pos.assetType || (symUpper.includes("-USD") || ["BTC", "ETH", "SOL"].includes(symUpper) ? "Crypto" : "Stock"),
    };

    const persistRes = await persistHoldingToApi(newPos);
    if (!persistRes.success) {
      return {
        success: false,
        isDuplicate: false,
        message: `Failed to persist position to database: ${persistRes.error || "Server error"}.`,
      };
    }

    savePortfolioPositions([newPos, ...existing]);
    notifyPortfolioWriteConfirmed();
    window.dispatchEvent(new CustomEvent("finance:portfolio-updated"));
    trackPortfolioPositionAdded(symUpper, newPos.shares * newPos.entryPrice);

    return {
      success: true,
      isDuplicate: false,
      message: `Added ${newPos.shares} shares of ${symUpper} to Paper Portfolio!`,
    };
  } catch (err: any) {
    console.error("Failed to add portfolio position:", err);
    return { success: false, isDuplicate: false, message: `Failed to save position: ${err?.message || "Unknown error"}` };
  }
}

export async function updatePortfolioPosition(pos: {
  symbol: string;
  shares: number;
  entryPrice?: number;
  currentPrice?: number | null;
  targetPrice?: number;
  stopLossPrice?: number;
  name?: string;
}): Promise<{ success: boolean; message: string }> {
  if (typeof window === "undefined") return { success: false, message: "Window undefined" };
  try {
    const symUpper = (pos.symbol || "").toUpperCase().trim();
    if (!symUpper) {
      return { success: false, message: "Valid asset symbol is required." };
    }

    if (pos.shares === undefined || pos.shares === null || isNaN(pos.shares) || pos.shares <= 0) {
      return {
        success: false,
        message: "Explicit positive share quantity is required (e.g. 10, 0.25). Zero or negative quantities are prohibited.",
      };
    }

    const existing = loadPortfolioPositions();
    const idx = existing.findIndex((p) => p.symbol.toUpperCase() === symUpper);
    if (idx < 0) {
      return { success: false, message: `${symUpper} position not found in portfolio` };
    }
    const current = existing[idx];

    let resolvedCurrentPrice: number | null = current.currentPrice;
    if (pos.currentPrice !== undefined) {
      resolvedCurrentPrice = (pos.currentPrice !== null && !isNaN(pos.currentPrice) && pos.currentPrice > 0)
        ? pos.currentPrice
        : null;
    }

    const updatedPos: PortfolioPosition = {
      ...current,
      shares: pos.shares,
      entryPrice: pos.entryPrice !== undefined && !isNaN(pos.entryPrice) && pos.entryPrice > 0 ? pos.entryPrice : current.entryPrice,
      currentPrice: resolvedCurrentPrice,
      targetPrice: pos.targetPrice !== undefined ? pos.targetPrice : current.targetPrice,
      stopLossPrice: pos.stopLossPrice !== undefined ? pos.stopLossPrice : current.stopLossPrice,
      name: pos.name || current.name,
    };

    const persistRes = await persistHoldingToApi(updatedPos);
    if (!persistRes.success) {
      return { success: false, message: `Failed to persist update to database: ${persistRes.error || "Server error"}.` };
    }

    const updatedList = [...existing];
    updatedList[idx] = updatedPos;
    savePortfolioPositions(updatedList);
    notifyPortfolioWriteConfirmed();
    window.dispatchEvent(new CustomEvent("finance:portfolio-updated"));
    return { success: true, message: `Updated ${symUpper} holding (${pos.shares} shares)!` };
  } catch (err: any) {
    console.error("Failed to update portfolio position:", err);
    return { success: false, message: `Failed to update position: ${err?.message || "Unknown error"}` };
  }
}

export async function removePortfolioPosition(symbol: string): Promise<{ success: boolean; message: string }> {
  if (typeof window === "undefined") return { success: false, message: "Window undefined" };
  try {
    const symUpper = (symbol || "").toUpperCase().trim();
    const existing = loadPortfolioPositions();
    const idx = existing.findIndex((p) => p.symbol.toUpperCase() === symUpper);
    if (idx < 0) {
      return { success: false, message: `${symUpper} holding not found in portfolio.` };
    }

    const deleteRes = await removeHoldingFromApi(symUpper);
    if (!deleteRes.success) {
      return { success: false, message: `Failed to delete from database: ${deleteRes.error || "Server error"}.` };
    }

    const updated = existing.filter((p) => p.symbol.toUpperCase() !== symUpper);
    savePortfolioPositions(updated);
    notifyPortfolioWriteConfirmed();
    window.dispatchEvent(new CustomEvent("finance:portfolio-updated"));
    return { success: true, message: `Removed ${symUpper} from portfolio.` };
  } catch (err: any) {
    console.error("Failed to remove portfolio position:", err);
    return { success: false, message: `Failed to remove position: ${err?.message || "Unknown error"}` };
  }
}

export function calculatePortfolioSummary(positions: PortfolioPosition[]): PortfolioSummary {
  let totalCost = 0;
  let totalEquity: number | null = 0;
  let unpricedCount = 0;

  positions.forEach((pos) => {
    const cost = pos.shares * pos.entryPrice;
    totalCost += cost;

    if (pos.currentPrice === null || pos.currentPrice === undefined || isNaN(pos.currentPrice) || pos.currentPrice <= 0) {
      unpricedCount += 1;
      totalEquity = null;
    } else if (totalEquity !== null) {
      totalEquity += pos.shares * pos.currentPrice;
    }
  });

  const isComplete = unpricedCount === 0;
  const totalUnrealizedPnL = isComplete && totalEquity !== null ? totalEquity - totalCost : null;
  const totalUnrealizedPnLPct = isComplete && totalUnrealizedPnL !== null && totalCost > 0
    ? (totalUnrealizedPnL / totalCost) * 100
    : null;

  return {
    totalEquity,
    totalCost,
    totalUnrealizedPnL,
    totalUnrealizedPnLPct,
    positionsCount: positions.length,
    isComplete,
    unpricedCount,
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
    const hasPrice = pos.currentPrice !== null && pos.currentPrice > 0;
    const value = hasPrice ? pos.shares * pos.currentPrice! : null;
    const pnl = value !== null ? value - cost : null;
    const pnlPct = pnl !== null && cost > 0 ? (pnl / cost) * 100 : null;

    return [
      pos.symbol,
      `"${pos.name.replace(/"/g, '""')}"`,
      pos.assetType,
      pos.shares,
      pos.entryPrice.toFixed(2),
      hasPrice ? pos.currentPrice!.toFixed(2) : "UNPRICED",
      pos.targetPrice ? pos.targetPrice.toFixed(2) : "N/A",
      pos.stopLossPrice ? pos.stopLossPrice.toFixed(2) : "N/A",
      cost.toFixed(2),
      value !== null ? value.toFixed(2) : "N/A",
      pnl !== null ? pnl.toFixed(2) : "N/A",
      pnlPct !== null ? `${pnlPct.toFixed(2)}%` : "N/A",
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
