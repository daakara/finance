"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { AlertCircle, Shield, TrendingDown, Activity, Sparkles, BarChart2, Layers } from "lucide-react";
import { fetchEtfProfile, EtfRiskProfileData } from "../lib/api";
import { emitEtfInteractionEvent } from "../lib/telemetry/etfIntentClient";

export interface EtfRiskProfileCardProps {
  symbol: string;
  initialData?: EtfRiskProfileData | null;
  className?: string;
}

export type VernacularMode = "PLAIN_ENGLISH" | "PRO_QUANT";

export default function EtfRiskProfileCard({
  symbol,
  initialData,
  className = "",
}: EtfRiskProfileCardProps) {
  const [data, setData] = useState<EtfRiskProfileData | null>(initialData || null);
  const [loading, setLoading] = useState<boolean>(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [vernacularMode, setVernacularMode] = useState<VernacularMode>("PLAIN_ENGLISH");

  const cleanSymbol = useMemo(() => {
    return (symbol || "").toUpperCase().replace(/.*:/, "").replace("-USD", "").trim();
  }, [symbol]);

  // Synchronize vernacular mode with global ARX Horizon system
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as VernacularMode | null;
      if (saved === "PLAIN_ENGLISH" || saved === "PRO_QUANT") {
        setVernacularMode(saved);
      }
    }
    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<VernacularMode>;
      if (custom.detail) {
        setVernacularMode(custom.detail);
      }
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const toggleVernacular = useCallback(() => {
    const nextMode: VernacularMode = vernacularMode === "PLAIN_ENGLISH" ? "PRO_QUANT" : "PLAIN_ENGLISH";
    setVernacularMode(nextMode);
    if (typeof window !== "undefined") {
      try {
        localStorage.setItem("ARX_VERNACULAR_MODE", nextMode);
        window.dispatchEvent(new CustomEvent("finance:vernacular-change", { detail: nextMode }));
      } catch {
        // Safe fallback in restricted environments
      }
    }
    // Telemetry: P2 interaction event
    emitEtfInteractionEvent("ETF_VERNACULAR_TOGGLE", cleanSymbol, { new_mode: nextMode });
  }, [vernacularMode, cleanSymbol]);

  // Fetch ETF profile data
  useEffect(() => {
    let isMounted = true;
    if (!cleanSymbol) return;

    if (initialData && initialData.symbol === cleanSymbol) {
      setData(initialData);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    fetchEtfProfile(cleanSymbol, "1y")
      .then((profile) => {
        if (!isMounted) return;
        if (!profile) {
          setError(`Institutional risk profile data is currently unavailable for ${cleanSymbol}.`);
          setData(null);
        } else {
          setData(profile);
          // Telemetry: P2 interaction view event
          emitEtfInteractionEvent("ETF_RISK_PROFILE_VIEW", cleanSymbol, {
            observation_count: profile.observation_count,
            quality_state: profile.quality?.state,
          });
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err.message || "Failed to load ETF risk profile.");
        setData(null);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [cleanSymbol, initialData]);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  // Volatility regime color mapping
  const regimeBadge = useMemo(() => {
    const reg = data?.volatility?.regime || data?.volatility_regime || "UNKNOWN";
    switch (reg) {
      case "LOW":
        return {
          label: "Low Volatility Regime (<12%)",
          classes: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          dot: "bg-emerald-400",
        };
      case "MODERATE":
        return {
          label: "Moderate Volatility Regime (12–22%)",
          classes: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
          dot: "bg-cyan-400",
        };
      case "HIGH":
        return {
          label: "High Volatility Regime (>22%)",
          classes: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          dot: "bg-rose-400",
        };
      default:
        return {
          label: "Vol Regime Unclassified",
          classes: "bg-slate-800 text-slate-400 border-slate-700",
          dot: "bg-slate-500",
        };
    }
  }, [data]);

  // SVG sparkline path calculation
  const sparklinePath = useMemo(() => {
    const points = data?.vol_history_200d || data?.volatility?.history_200d || [];
    if (!points || points.length < 2) return "";
    const minVal = Math.min(...points);
    const maxVal = Math.max(...points);
    const range = maxVal - minVal || 1.0;
    const width = 200;
    const height = 36;
    const pad = 3;

    return points
      .map((val, idx) => {
        const x = (idx / (points.length - 1)) * width;
        const y = height - pad - ((val - minVal) / range) * (height - 2 * pad);
        return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  }, [data]);

  // Loading Skeleton State
  if (loading) {
    return (
      <section
        aria-label="Loading ETF Risk Profile"
        className={`bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl font-sans space-y-4 animate-pulse ${className}`}
      >
        <div className="flex justify-between items-center border-b border-[#243044]/60 pb-3">
          <div className="flex items-center space-x-2">
            <div className="w-5 h-5 bg-slate-700 rounded" />
            <div className="h-4 w-48 bg-slate-700 rounded" />
          </div>
          <div className="h-6 w-24 bg-slate-800 rounded" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-20 bg-[#090d14] rounded-lg border border-[#1b2434]" />
          ))}
        </div>
        <div className="h-24 bg-[#090d14] rounded-lg border border-[#1b2434]" />
      </section>
    );
  }

  // Missing or Error Fallback State (Fail-Closed)
  if (error || !data) {
    return (
      <section
        aria-label="ETF Risk Profile Error"
        className={`bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl font-sans space-y-3 ${className}`}
      >
        <div className="flex items-center justify-between border-b border-[#243044]/60 pb-3">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-bold text-slate-200">
              ETF Risk Profile • {cleanSymbol}
            </h2>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
            RISK DATA UNAVAILABLE
          </span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          {error || `Certified price history is insufficient to compute Cornish-Fisher Value-at-Risk or drawdown analytics for ${cleanSymbol}.`}
        </p>
        <div className="text-[11px] font-mono text-slate-500 bg-[#090d14] p-2.5 rounded border border-[#1b2434]">
          Zero synthetic numbers are imputed under ARX quantitative governance.
        </div>
      </section>
    );
  }

  const drawdown = data.drawdown || {};
  const maxDdPct = data.max_drawdown_pct ?? drawdown.maximum_pct ?? null;
  const currentDdPct = data.current_drawdown_pct ?? drawdown.current_pct ?? null;
  const recoveryDays = data.recovery_days ?? drawdown.recovery_days ?? null;
  const recoveryState = drawdown.recovery_state || (recoveryDays !== null ? "RECOVERED" : "UNRECOVERED");

  const var95 = data.value_at_risk?.var_95;
  const var99 = data.value_at_risk?.var_99;
  const var95Pct = data.var_95_daily_pct ?? var95?.daily_var_pct ?? null;
  const var99Pct = data.var_99_daily_pct ?? var99?.daily_var_pct ?? null;

  const sharpe = data.sharpe_ratio ?? data.risk_adjusted_returns?.sharpe ?? null;
  const sortino = data.sortino_ratio ?? data.risk_adjusted_returns?.sortino ?? null;
  const calmar = data.calmar_ratio ?? data.risk_adjusted_returns?.calmar ?? null;
  const annVol = data.annualized_volatility_pct ?? data.volatility?.realized_annualized_pct ?? null;

  const topSectors = (data.sectors || []).slice(0, 5);

  return (
    <section
      aria-labelledby="etf-risk-title"
      className={`bg-[#111722] border border-[#243044] rounded-xl p-4 sm:p-5 shadow-xl font-sans space-y-4 ${className}`}
    >
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#243044]/60 pb-3">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <h2 id="etf-risk-title" className="text-sm sm:text-base font-bold text-white tracking-tight flex items-center gap-2">
              ETF Risk Profile & Downside Volatility Model
              <span className="text-xs font-mono text-cyan-400 font-semibold">({cleanSymbol})</span>
            </h2>
            <div className="flex flex-wrap items-center gap-2 mt-0.5 text-[11px] text-slate-400">
              <span>Institutional Risk Engine</span>
              <span>•</span>
              <span className="font-mono text-[10px] text-slate-500">
                {data.observation_count} Sessions ({data.history_start || "1Y"} → {data.history_end || "Current"})
              </span>
            </div>
          </div>
        </div>

        {/* Action Controls: Dual Vernacular Mode Toggle */}
        <div className="flex items-center space-x-2">
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded border inline-flex items-center gap-1 ${
            data.quality?.state === "ESTABLISHED"
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              : "bg-amber-500/10 text-amber-400 border-amber-500/30"
          }`}>
            <span className="w-1.5 h-1.5 rounded-full bg-current" />
            {data.quality?.state || "ESTABLISHED"}
          </span>

          <button
            type="button"
            onClick={toggleVernacular}
            aria-pressed={!isPlain}
            aria-label="Toggle between Plain English and Pro Quant modes"
            className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-mono font-medium bg-[#162030] hover:bg-[#1d2a40] text-slate-200 border border-[#2e3e58] transition-colors focus:outline-none focus:ring-1 focus:ring-cyan-500"
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>{isPlain ? "Plain English" : "Pro Quant"}</span>
          </button>
        </div>
      </div>

      {/* Primary Drawdown & Recovery Section */}
      <div className="bg-[#0b1019] border border-[#1b2434] rounded-xl p-3.5 space-y-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5 font-mono">
            <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
            <span>Historical Drawdown & Recovery Trajectory</span>
          </span>
          <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
            recoveryState === "RECOVERED"
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              : "bg-amber-500/10 text-amber-400 border-amber-500/30"
          }`}>
            {recoveryState === "RECOVERED"
              ? `Recovered in ${recoveryDays ?? "—"} Trading Days`
              : "Active Drawdown (Unrecovered)"}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
          <div className="bg-[#111722] p-3 rounded-lg border border-[#1e293b]">
            <div className="text-[11px] text-slate-400 font-medium">Max Drawdown</div>
            <div className="text-lg font-bold font-mono text-rose-400 tabular-nums">
              {maxDdPct !== null ? `-${maxDdPct.toFixed(2)}%` : "—"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono mt-0.5">
              Trough: {drawdown.trough_date || "—"}
            </div>
          </div>

          <div className="bg-[#111722] p-3 rounded-lg border border-[#1e293b]">
            <div className="text-[11px] text-slate-400 font-medium">Current Drawdown</div>
            <div className={`text-lg font-bold font-mono tabular-nums ${
              currentDdPct !== null && currentDdPct > 5.0 ? "text-amber-400" : "text-slate-200"
            }`}>
              {currentDdPct !== null ? `-${currentDdPct.toFixed(2)}%` : "—"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono mt-0.5">
              From Running All-Time Peak
            </div>
          </div>

          <div className="bg-[#111722] p-3 rounded-lg border border-[#1e293b]">
            <div className="text-[11px] text-slate-400 font-medium">Recovery Duration</div>
            <div className="text-lg font-bold font-mono text-cyan-400 tabular-nums">
              {recoveryDays !== null ? `${recoveryDays}d` : "Unrecovered"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono mt-0.5">
              {recoveryState === "RECOVERED" ? `Full Recovery: ${drawdown.recovery_date || "—"}` : "Still below peak ATH"}
            </div>
          </div>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed font-sans pt-1">
          {isPlain
            ? `In the observed lookback, ${cleanSymbol}'s most severe drop was ${
                maxDdPct !== null ? `${maxDdPct.toFixed(1)}%` : "N/A"
              }. ${
                recoveryState === "RECOVERED" && recoveryDays !== null
                  ? `It fully recovered all losses in ${recoveryDays} trading days.`
                  : "The fund has not yet recovered to its prior peak price."
              }`
            : `Peak: ${drawdown.peak_date || "—"} → Trough: ${
                drawdown.trough_date || "—"
              } (DD: -${maxDdPct?.toFixed(2) ?? "—"}%). Time-to-recovery (TTR): ${
                recoveryDays !== null ? `${recoveryDays} sessions` : "UNRECOVERED"
              }. Current depth: -${currentDdPct?.toFixed(2) ?? "—"}%.`}
        </p>
      </div>

      {/* Grid: Value at Risk + Risk-Adjusted Return Ratios */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Value at Risk (Modified Cornish-Fisher) */}
        <div className="bg-[#0b1019] border border-[#1b2434] rounded-xl p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5 font-mono">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span>Value at Risk (Cornish-Fisher)</span>
            </span>
            <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/40 border border-cyan-800/60 px-2 py-0.5 rounded">
              Modified 1-Day Horizon
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            <div className="bg-[#111722] p-2.5 rounded-lg border border-[#1e293b]">
              <div className="text-[10px] font-mono text-slate-400">95% Daily VaR (1-in-20)</div>
              <div className="text-base font-bold font-mono text-rose-400 tabular-nums mt-0.5">
                {var95Pct !== null ? `-${var95Pct.toFixed(2)}%` : "—"}
              </div>
              <div className="text-[9px] text-slate-500 font-mono">
                Gaussian z: {var95?.z_gaussian ?? "—"}
              </div>
            </div>

            <div className="bg-[#111722] p-2.5 rounded-lg border border-[#1e293b]">
              <div className="text-[10px] font-mono text-slate-400">99% Daily VaR (1-in-100)</div>
              <div className="text-base font-bold font-mono text-rose-400 tabular-nums mt-0.5">
                {var99Pct !== null ? `-${var99Pct.toFixed(2)}%` : "—"}
              </div>
              <div className="text-[9px] text-slate-500 font-mono">
                Fat-tail adjusted z: {var99?.z_cornish_fisher ?? "—"}
              </div>
            </div>
          </div>

          <div className="p-2.5 rounded-lg bg-[#111722]/80 border border-[#1b2434] text-xs text-slate-300 leading-relaxed font-sans">
            {isPlain ? (
              <span>
                Under normal market liquidity, you should expect daily losses to stay under{" "}
                <strong className="text-rose-400 font-mono">
                  {var95Pct !== null ? `${var95Pct.toFixed(1)}%` : "the model estimate"}
                </strong>{" "}
                on 19 out of 20 trading days (95% confidence).
              </span>
            ) : (
              <span className="font-mono text-[11px] text-slate-300">
                CF-Modified VaR uses 3rd/4th standardized sample moments (Skewness:{" "}
                <span className="text-cyan-400">{var95?.skewness ?? "—"}</span>, Kurtosis:{" "}
                <span className="text-cyan-400">{var95?.excess_kurtosis ?? "—"}</span>) to model fat-tail extreme loss risks without Gaussian underestimation.
              </span>
            )}
          </div>
        </div>

        {/* Right: Risk-Adjusted Ratios & Volatility Regime */}
        <div className="bg-[#0b1019] border border-[#1b2434] rounded-xl p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5 font-mono">
              <BarChart2 className="w-3.5 h-3.5 text-indigo-400" />
              <span>Risk-Adjusted Return Ratios</span>
            </span>
            <div className={`text-[10px] font-mono px-2 py-0.5 rounded border inline-flex items-center gap-1 ${regimeBadge.classes}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${regimeBadge.dot}`} />
              <span>{regimeBadge.label}</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <div className="bg-[#111722] p-2.5 rounded-lg border border-[#1e293b] text-center">
              <div className="text-[10px] font-mono text-slate-400">Sharpe (Rf=2%)</div>
              <div className={`text-base font-bold font-mono tabular-nums mt-0.5 ${
                sharpe !== null && sharpe >= 1.0 ? "text-emerald-400" : sharpe !== null && sharpe < 0 ? "text-rose-400" : "text-slate-200"
              }`}>
                {sharpe !== null ? sharpe.toFixed(2) : "—"}
              </div>
              <div className="text-[9px] text-slate-500 font-mono mt-0.5">Total Vol Base</div>
            </div>

            <div className="bg-[#111722] p-2.5 rounded-lg border border-[#1e293b] text-center">
              <div className="text-[10px] font-mono text-slate-400">Sortino</div>
              <div className={`text-base font-bold font-mono tabular-nums mt-0.5 ${
                sortino !== null && sortino >= 1.2 ? "text-emerald-400" : "text-slate-200"
              }`}>
                {sortino !== null ? sortino.toFixed(2) : "—"}
              </div>
              <div className="text-[9px] text-slate-500 font-mono mt-0.5">Downside Vol</div>
            </div>

            <div className="bg-[#111722] p-2.5 rounded-lg border border-[#1e293b] text-center">
              <div className="text-[10px] font-mono text-slate-400">Calmar</div>
              <div className="text-base font-bold font-mono text-slate-200 tabular-nums mt-0.5">
                {calmar !== null ? calmar.toFixed(2) : "—"}
              </div>
              <div className="text-[9px] text-slate-500 font-mono mt-0.5">Max DD Return</div>
            </div>
          </div>

          {/* Sparkline & Realized Volatility */}
          <div className="p-2.5 rounded-lg bg-[#111722] border border-[#1e293b] flex items-center justify-between gap-3">
            <div>
              <div className="text-[10px] font-mono text-slate-400">Realized 1Y Volatility</div>
              <div className="text-sm font-bold font-mono text-white tabular-nums">
                {annVol !== null ? `${annVol.toFixed(1)}% Ann.` : "—"}
              </div>
            </div>

            {sparklinePath ? (
              <div className="flex flex-col items-end">
                <svg
                  width="130"
                  height="28"
                  viewBox="0 0 200 36"
                  className="overflow-visible"
                  aria-label="200-day rolling volatility sparkline"
                >
                  <path
                    d={sparklinePath}
                    fill="none"
                    stroke="#06b6d4"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <span className="text-[8px] font-mono text-slate-500">200-Day Rolling Vol Trend</span>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* Dynamic Sector Weight Summary (if available) */}
      {topSectors.length > 0 && (
        <div className="bg-[#0b1019] border border-[#1b2434] rounded-xl p-3.5 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5 font-mono">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Top Sector Weight Allocations (Dynamic Mandate)</span>
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              Source: {data.sectors.length} Sectors Sourced
            </span>
          </div>

          <div className="space-y-2 pt-1">
            {topSectors.map((s, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 font-medium">{s.sector}</span>
                  <span className="text-cyan-400 font-bold font-mono">{s.weightPct.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-[#141d2c] rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-cyan-500 to-indigo-600 h-1.5 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, Math.max(0, s.weightPct))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
