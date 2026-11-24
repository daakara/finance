import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import RiskMetricsCard from "../RiskMetricsCard";
import AssetFactorRadar from "../AssetFactorRadar";
import DayTraderPositionSizer from "../DayTraderPositionSizer";
import { AnalyticsResponse, AssetFactorScores } from "../../lib/api";

beforeEach(() => {
  if (typeof localStorage !== "undefined") {
    localStorage.clear();
  }
});

afterEach(() => cleanup());

describe("RiskMetricsCard & AssetFactorRadar Behavioral Coverage", () => {
  it("RiskMetricsCard displays Unavailable and warning without fabricated numbers when metrics are null", () => {
    const unavailableData: AnalyticsResponse = {
      symbol: "TEST",
      currentPrice: 100,
      analytics: {
        advanced_metrics: {
          Modified_VaR_95: null,
          Modified_VaR_99: null,
          Modified_CVaR_95: null,
          Sortino_Ratio: null,
          Calmar_Ratio: null,
        },
        cornish_fisher: {
          status: "UNAVAILABLE",
          reason: "INSUFFICIENT_OBSERVATIONS",
          actual_observations: 18,
          required_observations: 30,
          message: "Cornish-Fisher risk estimates require at least 30 valid closed daily return observations (found 18).",
        },
      },
    };

    render(<RiskMetricsCard analyticsData={unavailableData} userRole="LONG_TERM" />);

    // Must show Unavailable text
    const unavailableElements = screen.getAllByText("Unavailable");
    expect(unavailableElements.length).toBeGreaterThanOrEqual(2);

    // Must show warning banner citing sample requirement
    const warningBanners = screen.getAllByText(/Cornish-Fisher risk estimates require at least 30 valid closed daily return observations/i);
    expect(warningBanners.length).toBeGreaterThanOrEqual(1);

    // Must NOT contain fabricated baseline numbers
    expect(screen.queryByText(/-3.42%/)).toBeNull();
    expect(screen.queryByText(/-5.18%/)).toBeNull();
    expect(screen.queryByText(/0.00%/)).toBeNull();
  });

  it("RiskMetricsCard clears previous valid metrics when switching to an insufficient-history asset", () => {
    const validData: AnalyticsResponse = {
      symbol: "NVDA",
      currentPrice: 130,
      analytics: {
        advanced_metrics: {
          Modified_VaR_95: -4.58,
          Modified_VaR_99: -7.11,
          Sortino_Ratio: 1.85,
        },
        cornish_fisher: {
          status: "AVAILABLE",
          actual_observations: 200,
          required_observations: 30,
        },
      },
    };

    const insufficientData: AnalyticsResponse = {
      symbol: "NEWCO",
      currentPrice: 50,
      analytics: {
        advanced_metrics: {
          Modified_VaR_95: null,
          Modified_VaR_99: null,
          Sortino_Ratio: null,
        },
        cornish_fisher: {
          status: "UNAVAILABLE",
          reason: "INSUFFICIENT_OBSERVATIONS",
          actual_observations: 10,
          required_observations: 30,
        },
      },
    };

    // First render with valid asset
    const { rerender } = render(<RiskMetricsCard analyticsData={validData} userRole="LONG_TERM" />);
    expect(screen.getByText("-4.58%")).toBeDefined();
    expect(screen.getByText("-7.11%")).toBeDefined();

    // Rerender with insufficient asset
    rerender(<RiskMetricsCard analyticsData={insufficientData} userRole="LONG_TERM" />);

    // Old valid metrics must be completely cleared
    expect(screen.queryByText("-4.58%")).toBeNull();
    expect(screen.queryByText("-7.11%")).toBeNull();

    // Unavailable text must be rendered
    const unavails = screen.getAllByText("Unavailable");
    expect(unavails.length).toBeGreaterThanOrEqual(2);
  });

  it("AssetFactorRadar displays N/A for Tail Risk and clears on asset switch", () => {
    const validScores: AssetFactorScores = {
      growthScore: 85,
      qualityScore: 90,
      valuationScore: 65,
      momentumScore: 78,
      tailRiskScore: 82,
      compositeFactorScore: 80,
      verdict: "High Quality Profile",
    };

    const insufficientScores: AssetFactorScores = {
      growthScore: 60,
      qualityScore: 55,
      valuationScore: null,
      momentumScore: 50,
      tailRiskScore: null, // Cornish-Fisher unavailable
      compositeFactorScore: 55,
      verdict: "Speculative Profile",
    };

    const { rerender } = render(<AssetFactorRadar symbol="NVDA" factorScores={validScores} />);
    expect(screen.getByText("82")).toBeDefined();

    // Switch to asset with insufficient history
    rerender(<AssetFactorRadar symbol="NEWCO" factorScores={insufficientScores} />);

    // Previous score of 82 must be cleared, showing N/A
    expect(screen.queryByText("82")).toBeNull();
    const naElements = screen.getAllByText("N/A");
    expect(naElements.length).toBeGreaterThanOrEqual(1);
  });
});

describe("DayTraderPositionSizer Behavioral & Null-Safety Coverage", () => {
  it("uses Cornish-Fisher Modified VaR 95% when valid and labels as mVaR Vol", () => {
    const data: AnalyticsResponse = {
      symbol: "NVDA",
      currentPrice: 100,
      technicals: {
        vwap: 99.5,
        rsi_14: 55.0,
        atr_14: 3.0,
      },
      analytics: {
        advanced_metrics: {
          Modified_VaR_95: -4.0, // 4.0% mVaR
        },
      },
    };

    render(<DayTraderPositionSizer symbol="NVDA" data={data} />);

    // Stop distance pct = Math.max(0.8, Math.min(6.0, 4.0 * 0.65)) = 2.60%
    // Stop distance dollar = 100 * (2.60 / 100) = $2.60
    expect(screen.getByText("$2.60")).toBeDefined();
    expect(screen.getByText("2.60% mVaR Vol")).toBeDefined();
  });

  it("permits ATR-14 fallback when Modified_VaR_95 is null, accurately labeled as ATR Vol", () => {
    const data: AnalyticsResponse = {
      symbol: "TEST",
      currentPrice: 100,
      technicals: {
        vwap: 100.2,
        rsi_14: 50.0,
        atr_14: 4.0, // ATR-14 is 4.0 -> (4.0 / 100) * 100 = 4.0% volatility
      },
      analytics: {
        advanced_metrics: {
          Modified_VaR_95: null, // Suppressed due to Cornish-Fisher gate
        },
        cornish_fisher: {
          status: "UNAVAILABLE",
          reason: "INSUFFICIENT_OBSERVATIONS",
          actual_observations: 20,
          required_observations: 30,
        },
      },
    };

    render(<DayTraderPositionSizer symbol="TEST" data={data} />);

    // Stop distance pct = Math.max(0.8, Math.min(6.0, 4.0 * 0.65)) = 2.60%
    expect(screen.getByText("2.60% ATR Vol")).toBeDefined();
    expect(screen.queryByText(/mVaR Vol/)).toBeNull();
  });

  it("disables sizing with Volatility Telemetry Unavailable when BOTH mVaR and ATR-14 are null or missing", () => {
    const data: AnalyticsResponse = {
      symbol: "GAPCO",
      currentPrice: 100,
      technicals: {
        vwap: 100.0,
        rsi_14: 50.0,
        atr_14: null as any, // Missing ATR
      },
      analytics: {
        advanced_metrics: {
          Modified_VaR_95: null,
        },
      },
    };

    render(<DayTraderPositionSizer symbol="GAPCO" data={data} />);

    expect(screen.getByText("Volatility Telemetry Unavailable:")).toBeDefined();
    expect(
      screen.getByText(/Neither Cornish-Fisher Modified VaR nor valid ATR-14 volatility telemetry could be established/i)
    ).toBeDefined();

    // Must NOT invent 2.5% or recommend shares
    expect(screen.queryByText(/Recommended Size/i)).toBeNull();
    expect(screen.queryByText(/2.50%/)).toBeNull();
  });

  it("disables sizing when ATR-14 is zero or non-positive without mVaR", () => {
    const data: AnalyticsResponse = {
      symbol: "ZEROATR",
      currentPrice: 50,
      technicals: {
        vwap: 50.0,
        rsi_14: 45.0,
        atr_14: 0, // Non-positive ATR
      },
      analytics: {
        advanced_metrics: {
          Modified_VaR_95: null,
        },
      },
    };

    render(<DayTraderPositionSizer symbol="ZEROATR" data={data} />);

    expect(screen.getByText("Volatility Telemetry Unavailable:")).toBeDefined();
    expect(screen.queryByText(/Recommended Size/i)).toBeNull();
  });

  it("handles non-positive price and missing technicals safely without NaN or Infinity", () => {
    const dataZeroPrice: AnalyticsResponse = {
      symbol: "ZEROPRICE",
      currentPrice: 0,
      technicals: {
        vwap: 0,
        atr_14: 2.0,
      },
    };

    render(<DayTraderPositionSizer symbol="ZEROPRICE" data={dataZeroPrice} />);
    expect(screen.getByText("Intraday Technicals Unavailable:")).toBeDefined();
  });
});
