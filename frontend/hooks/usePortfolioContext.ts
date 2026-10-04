"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import {
  fetchAuthoritativePortfolio,
  PortfolioPosition,
} from "../lib/portfolio";
import { normalizeAssetSymbol } from "../lib/assetRegistry";

export type OwnershipState = "HELD" | "NOT_HELD" | "UNKNOWN";
export type OwnershipFilter = "ALL" | "NEW_OPPORTUNITIES" | "MY_HOLDINGS";

export interface PortfolioContextValue {
  isVerified: boolean;
  isLoading: boolean;
  isDegraded: boolean;
  error: string | null;
  getOwnershipState: (symbol: string | null | undefined) => OwnershipState;
  getHolding: (symbol: string | null | undefined) => PortfolioPosition | undefined;
  holdingsCount: number;
  refresh: () => Promise<void>;
}

/**
 * Hook providing authoritative portfolio context for Radar and Terminal presentation.
 *
 * Invariants Enforced:
 * - INV-RADAR-PORTFOLIO-01: Pure consumer; never alters Radar domain inputs, scores, or ranking.
 * - INV-RADAR-PORTFOLIO-02: Factual ownership state only; never produces trade advice.
 * - INV-RADAR-PORTFOLIO-05: Portfolio failure fails closed gracefully; leaves caller intact.
 * - INV-RADAR-PORTFOLIO-06: Authoritative state derives exclusively from server persistence.
 * - INV-RADAR-PORTFOLIO-07: Ambiguous or unparseable symbols fail closed to UNKNOWN.
 */
export function usePortfolioContext(): PortfolioContextValue {
  const [isVerified, setIsVerified] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isDegraded, setIsDegraded] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [holdingsMap, setHoldingsMap] = useState<Map<string, PortfolioPosition>>(() => new Map());

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetchAuthoritativePortfolio();
      if (res.isVerified && Array.isArray(res.positions)) {
        const nextMap = new Map<string, PortfolioPosition>();
        for (const pos of res.positions) {
          const norm = normalizeAssetSymbol(pos.symbol);
          if (norm && pos.shares > 0) {
            nextMap.set(norm, pos);
          }
        }
        setHoldingsMap(nextMap);
        setIsVerified(true);
        setIsDegraded(false);
        setError(null);
      } else {
        // Authoritative verification failed or unreachable
        setIsVerified(false);
        setIsDegraded(true);
        setError(res.error || "Authoritative portfolio connection unavailable");
        setHoldingsMap(new Map());
      }
    } catch (err: any) {
      setIsVerified(false);
      setIsDegraded(true);
      setError(err?.message || "Failed to load authoritative portfolio");
      setHoldingsMap(new Map());
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    refresh().then(() => {
      if (!isMounted) return;
    });

    const handlePortfolioUpdated = () => {
      if (!isMounted) return;
      refresh();
    };

    if (typeof window !== "undefined") {
      window.addEventListener("finance:portfolio-updated", handlePortfolioUpdated);
    }

    return () => {
      isMounted = false;
      if (typeof window !== "undefined") {
        window.removeEventListener("finance:portfolio-updated", handlePortfolioUpdated);
      }
    };
  }, [refresh]);

  const getOwnershipState = useCallback(
    (symbol: string | null | undefined): OwnershipState => {
      const norm = normalizeAssetSymbol(symbol);
      // Invariant INV-RADAR-PORTFOLIO-07: Unresolved or ambiguous identity cannot produce HELD
      if (!norm) {
        return "UNKNOWN";
      }

      // Invariant INV-RADAR-PORTFOLIO-06: State requires verified authoritative portfolio
      if (!isVerified) {
        return "UNKNOWN";
      }

      const holding = holdingsMap.get(norm);
      if (holding && holding.shares > 0) {
        return "HELD";
      }

      // NOT_HELD is strictly valid only when portfolio is verified and ticker is absent
      return "NOT_HELD";
    },
    [isVerified, holdingsMap]
  );

  const getHolding = useCallback(
    (symbol: string | null | undefined): PortfolioPosition | undefined => {
      const norm = normalizeAssetSymbol(symbol);
      if (!norm || !isVerified) return undefined;
      return holdingsMap.get(norm);
    },
    [isVerified, holdingsMap]
  );

  const holdingsCount = useMemo(() => {
    return isVerified ? holdingsMap.size : 0;
  }, [isVerified, holdingsMap]);

  return {
    isVerified,
    isLoading,
    isDegraded,
    error,
    getOwnershipState,
    getHolding,
    holdingsCount,
    refresh,
  };
}
