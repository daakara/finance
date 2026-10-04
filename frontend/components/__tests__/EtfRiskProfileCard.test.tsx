import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import EtfRiskProfileCard from "../EtfRiskProfileCard";
import { EtfRiskProfileData } from "../../lib/api";

const mockEtfProfile: EtfRiskProfileData = {
  symbol: "SPY",
  as_of: "2026-10-02T12:00:00Z",
  history_start: "2025-10-02",
  history_end: "2026-10-02",
  period: "1y",
  observation_count: 252,
  quality: {
    state: "ESTABLISHED",
    warnings: [],
  },
  drawdown: {
    maximum: 0.245,
    maximum_pct: 24.5,
    current: 0.021,
    current_pct: 2.1,
    peak_date: "2026-02-15",
    trough_date: "2026-03-20",
    recovery_date: "2026-06-10",
    recovery_days: 58,
    recovery_state: "RECOVERED",
  },
  risk_adjusted_returns: {
    sharpe: 1.15,
    sortino: 1.48,
    calmar: 0.85,
  },
  value_at_risk: {
    var_95: {
      confidence: 0.95,
      method: "MODIFIED_CORNISH_FISHER",
      daily_var: 0.0185,
      daily_var_pct: 1.85,
      unit: "RETURN_FRACTION",
      sign_convention: "POSITIVE_LOSS",
      z_gaussian: -1.6449,
      z_cornish_fisher: -1.712,
      skewness: -0.28,
      excess_kurtosis: 2.15,
      observations: 252,
    },
    var_99: {
      confidence: 0.99,
      method: "MODIFIED_CORNISH_FISHER",
      daily_var: 0.0315,
      daily_var_pct: 3.15,
      unit: "RETURN_FRACTION",
      sign_convention: "POSITIVE_LOSS",
      z_gaussian: -2.3263,
      z_cornish_fisher: -2.485,
      skewness: -0.28,
      excess_kurtosis: 2.15,
      observations: 252,
    },
    horizon: "1d",
    method: "MODIFIED_CORNISH_FISHER",
    unit: "RETURN_FRACTION",
  },
  volatility: {
    realized_annualized_pct: 13.8,
    regime: "MODERATE",
    history_200d: [12.5, 13.1, 14.2, 13.8],
  },
  max_drawdown_pct: 24.5,
  max_drawdown_date: "2026-03-20",
  recovery_days: 58,
  current_drawdown_pct: 2.1,
  sharpe_ratio: 1.15,
  sortino_ratio: 1.48,
  calmar_ratio: 0.85,
  var_95_daily_pct: 1.85,
  var_99_daily_pct: 3.15,
  annualized_volatility_pct: 13.8,
  volatility_regime: "MODERATE",
  vol_history_200d: [12.5, 13.1, 14.2, 13.8],
  sectors: [
    { sector: "Information Technology", weightPct: 31.4 },
    { sector: "Financials", weightPct: 13.2 },
    { sector: "Healthcare", weightPct: 11.8 },
  ],
  source_provenance: "ARX_ETF_ANALYZER_V1",
};

describe("EtfRiskProfileCard", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("renders with initial data cleanly", () => {
    render(<EtfRiskProfileCard symbol="SPY" initialData={mockEtfProfile} />);

    expect(screen.getByText(/ETF Risk Profile & Downside Volatility Model/i)).toBeDefined();
    expect(screen.getByText(/\(SPY\)/i)).toBeDefined();
    expect(screen.getByText(/ESTABLISHED/i)).toBeDefined();

    // Drawdown
    expect(screen.getByText(/-24.50%/i)).toBeDefined();
    expect(screen.getByText(/Trough: 2026-03-20/i)).toBeDefined();
    expect(screen.getByText(/58d/i)).toBeDefined();

    // Ratios
    expect(screen.getByText("1.15")).toBeDefined(); // Sharpe
    expect(screen.getByText("1.48")).toBeDefined(); // Sortino
    expect(screen.getByText("0.85")).toBeDefined(); // Calmar

    // VaR
    expect(screen.getByText(/-1.85%/i)).toBeDefined();
    expect(screen.getByText(/-3.15%/i)).toBeDefined();

    // Volatility regime
    expect(screen.getByText(/Moderate Volatility Regime/i)).toBeDefined();

    // Sectors
    expect(screen.getByText("Information Technology")).toBeDefined();
    expect(screen.getByText("31.4%")).toBeDefined();
  });

  it("toggles between Plain English and Pro Quant mode", () => {
    render(<EtfRiskProfileCard symbol="SPY" initialData={mockEtfProfile} />);

    // Starts in Plain English mode
    expect(screen.getByText(/Plain English/i)).toBeDefined();
    expect(screen.getByText(/In the observed lookback, SPY's most severe drop was 24.5%/i)).toBeDefined();

    // Click toggle button
    const toggleBtn = screen.getByRole("button", { name: /Toggle between Plain English and Pro Quant modes/i });
    fireEvent.click(toggleBtn);

    // Transitions to Pro Quant mode
    expect(screen.getByText(/Pro Quant/i)).toBeDefined();
    expect(screen.getByText(/Time-to-recovery \(TTR\): 58 sessions/i)).toBeDefined();
    expect(screen.getByText(/CF-Modified VaR uses 3rd\/4th standardized sample moments/i)).toBeDefined();

    // Verify localStorage persistence
    expect(localStorage.getItem("ARX_VERNACULAR_MODE")).toBe("PRO_QUANT");
  });

  it("renders fail-closed error fallback when data is null and fetch fails", async () => {
    render(<EtfRiskProfileCard symbol="UNKNOWN_ETF" initialData={null} />);

    await waitFor(() => {
      expect(screen.getByText(/RISK DATA UNAVAILABLE/i)).toBeDefined();
    }, { timeout: 3000 });
    expect(screen.getByText(/Zero synthetic numbers are imputed/i)).toBeDefined();
  });

  it("handles unrecovered drawdowns correctly without fabricating days", () => {
    const unrecoveredProfile: EtfRiskProfileData = {
      ...mockEtfProfile,
      drawdown: {
        ...mockEtfProfile.drawdown,
        recovery_date: null,
        recovery_days: null,
        recovery_state: "UNRECOVERED",
      },
      recovery_days: null,
    };

    render(<EtfRiskProfileCard symbol="SPY" initialData={unrecoveredProfile} />);

    expect(screen.getByText(/Active Drawdown \(Unrecovered\)/i)).toBeDefined();
    expect(screen.getByText(/Still below peak ATH/i)).toBeDefined();
  });

  it("conforms to accessibility standards with ARIA attributes", () => {
    render(<EtfRiskProfileCard symbol="SPY" initialData={mockEtfProfile} />);

    const card = screen.getByRole("region", { name: /ETF Risk Profile & Downside Volatility Model/i });
    expect(card).toBeDefined();

    const toggleBtn = screen.getByRole("button", { name: /Toggle between Plain English and Pro Quant modes/i });
    expect(toggleBtn.getAttribute("aria-pressed")).toBe("false");
  });
});
