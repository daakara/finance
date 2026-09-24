"use client";

import React, { createContext, useContext, useState, useMemo, useCallback } from "react";

export type DataProviderType =
  | "Alpaca IEX"
  | "SEC EDGAR"
  | "FRED Macro"
  | "FINRA ATS"
  | "Curated Research Archive"
  | "Delayed Fallback"
  | "Unknown";

export type FreshnessType =
  | "IEX Real-Time"
  | "Regulatory Delayed (Up to 45d)"
  | "Curated Research (Aug 2026)"
  | "EOD Close"
  | "Static Fallback"
  | "Unknown";

export interface DataSourceState {
  provider: DataProviderType;
  freshness: FreshnessType;
  isRealtime: boolean;
  latencyLabel: string;
  sourceAttribution: string;
  lastSynced: string | null;
}

export const DEFAULT_DATA_SOURCE_STATE: DataSourceState = {
  provider: "Unknown",
  freshness: "Unknown",
  isRealtime: false,
  latencyLabel: "Awaiting Provider Stream",
  sourceAttribution: "Unverified / Awaiting Provider Stream",
  lastSynced: null,
};

interface DataSourceContextValue extends DataSourceState {
  setDataSourceInfo: (update: Partial<DataSourceState>) => void;
  setFromHeaderOrResponse: (sourceHeader?: string | null, isCurated?: boolean) => void;
}

const DataSourceContext = createContext<DataSourceContextValue | null>(null);

export function DataSourceProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<DataSourceState>(DEFAULT_DATA_SOURCE_STATE);

  const setDataSourceInfo = useCallback((update: Partial<DataSourceState>) => {
    setState((prev) => ({
      ...prev,
      ...update,
    }));
  }, []);

  const setFromHeaderOrResponse = useCallback((sourceHeader?: string | null, isCurated?: boolean) => {
    if (isCurated) {
      setState({
        provider: "Curated Research Archive",
        freshness: "Curated Research (Aug 2026)",
        isRealtime: false,
        latencyLabel: "Curated verification ledger",
        sourceAttribution: "ARX Quantitative Research Archive",
        lastSynced: new Date().toLocaleTimeString(),
      });
      return;
    }

    const header = (sourceHeader || "").toLowerCase();
    if (header.includes("alpaca") || header.includes("iex")) {
      setState({
        provider: "Alpaca IEX",
        freshness: "IEX Real-Time",
        isRealtime: true,
        latencyLabel: "IEX Real-Time Tape",
        sourceAttribution: "Alpaca Market Data API",
        lastSynced: new Date().toLocaleTimeString(),
      });
    } else if (header.includes("sec") || header.includes("edgar")) {
      setState({
        provider: "SEC EDGAR",
        freshness: "Regulatory Delayed (Up to 45d)",
        isRealtime: false,
        latencyLabel: "Official SEC EDGAR Filings",
        sourceAttribution: "SEC Public Disclosures",
        lastSynced: new Date().toLocaleTimeString(),
      });
    } else if (header.includes("fred")) {
      setState({
        provider: "FRED Macro",
        freshness: "EOD Close",
        isRealtime: false,
        latencyLabel: "Federal Reserve Economic Data",
        sourceAttribution: "Federal Reserve Bank of St. Louis",
        lastSynced: new Date().toLocaleTimeString(),
      });
    }
  }, []);

  const value = useMemo(
    () => ({
      ...state,
      setDataSourceInfo,
      setFromHeaderOrResponse,
    }),
    [state, setDataSourceInfo, setFromHeaderOrResponse]
  );

  return <DataSourceContext.Provider value={value}>{children}</DataSourceContext.Provider>;
}

export function useDataSource(): DataSourceContextValue {
  const ctx = useContext(DataSourceContext);
  if (!ctx) {
    return {
      ...DEFAULT_DATA_SOURCE_STATE,
      setDataSourceInfo: () => {},
      setFromHeaderOrResponse: () => {},
    };
  }
  return ctx;
}
