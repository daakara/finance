"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Navbar from "../../components/Navbar";
import PositionSizerModal from "../../components/PositionSizerModal";
import AlertTriggerModal from "../../components/AlertTriggerModal";
import DataSourceBadge from "../../components/DataSourceBadge";
import { API_BASE_URL, SpotPriceRegistry } from "../../lib/api";
import { getPersistedMarketSnapshot } from "../../lib/marketDatabase";
import { SHARED_FACTOR_SCORES } from "../../lib/constants";
import { MASTER_ASSET_CATALOG } from "../../lib/masterCatalog";
import { trackScreenerSelection, trackMatomoEvent } from "../../lib/matomo";
import {
  getCanonicalAssetName,
  getCanonicalAssetMoat,
  getCanonicalAssetRisk,
  getCanonicalAssetCatalyst,
} from "../../lib/assetRegistry";
import { addPortfolioPosition } from "../../lib/portfolio";

interface GemCandidate {
  symbol: string;
  companyName: string;
  currentPrice?: number;
  gemScore: number;
  expertArchetype: string;
  // Long-Term Fundamental Lens
  roic: string;
  pegRatio: string;
  grossMargin: string;
  thesis: string;
  // Day Trader Momentum & Scalp Lens
  atr14: string;
  rvol: string;
  shortFloat: string;
  dayTraderSetup: string;
  catalyst: string;
  riskLevel: string;
  // Execution Scanner Levels
  executionStatus?: "IN_BUY_ZONE" | "APPROACHING_TARGET" | "WAITING_PULLBACK" | "STOPPED_OUT";
  statusLabel?: string;
  statusColor?: string;
  optimalEntryMin?: number;
  optimalEntryMax?: number;
  stopLoss?: number;
  stopLossPct?: number;
  takeProfit1?: number;
  takeProfit1Pct?: number;
  takeProfit2?: number;
  takeProfit2Pct?: number;
  riskRewardRatio?: number;
  setupPattern?: string;
  entryThesis?: string;
  // Confluence Conviction Score
  confluenceScore?: number;
  confluenceRating?: string;
  confluenceBadgeColor?: string;
  confluenceReasons?: string[];
  confluenceWarnings?: string[];
}

const LONG_TERM_FILTER_TABS = [
  { id: "all", label: "🏛️ All Quality Compounders", desc: "Small & Mid-Cap High-Conviction Setups", badge: "Universe" },
  { id: "high_confluence", label: "⭐ High Confluence", desc: "Multi-Factor Technical, Smart Money & Catalyst Alignment (≥ 80%)", badge: "High Conviction" },
  { id: "in_buy_zone", label: "🎯 In Buy Zone", desc: "Price within optimal 50-day SMA support accumulation range", badge: "Actionable" },
  { id: "approaching_target", label: "🚀 Near TP Target", desc: "Price approaching Take-Profit 1 or 2 ladders", badge: "Profit Taking" },
  { id: "high_rr", label: "⚡ High R:R", desc: "Asymmetric risk-reward setups with tight stop losses (≥ 2.0:1)", badge: "Asymmetric" },
  { id: "lynch", label: "📈 Peter Lynch GARP", desc: "PEG < 1.0, Low Net Debt, Overlooked Compounders", badge: "GARP" },
  { id: "greenblatt", label: "🧪 Magic Formula", desc: "High ROIC (>25%) + Bargain Earnings Yield", badge: "Value" },
  { id: "rule_breakers", label: "🔥 Rule Breakers", desc: "Category Creators, >65% Gross Margins, High Moat", badge: "Disruptive" },
];

const DAY_TRADER_FILTER_TABS = [
  { id: "all", label: "⚡ All High-Beta Scalps", desc: "High Liquidity, High-ATR Intraday Leaders", badge: "Day Trade" },
  { id: "high_confluence", label: "⭐ High Confluence", desc: "High-Momentum Technical, Flow & VWAP Alignment", badge: "High Conviction" },
  { id: "in_buy_zone", label: "🎯 VWAP Pullback", desc: "Bid defense on 20 EMA / 5m VWAP anchor", badge: "Long Entry" },
  { id: "approaching_target", label: "🚀 ORB Breakout", desc: "Session high opening range breakout expansion", badge: "Momentum" },
  { id: "high_rr", label: "⚡ High R:R Scalps", desc: "Tight -1.5% stop with >2.0 R:R scalp target", badge: "Tight Risk" },
  { id: "high_rvol", label: "🔥 High RVOL (>2.5x)", desc: "Institutional volume surge & elevated liquidity", badge: "Flow" },
  { id: "squeeze", label: "💥 Short Squeeze", desc: "High Short Float & rapid momentum squeeze candidates", badge: "Squeeze" },
];

// Built-in 60-Asset Dual-Horizon Catalog Generator with diverse execution states
function generateBuiltinGems(role: "DAY_TRADER" | "LONG_TERM", customQuery?: string): GemCandidate[] {
  const isDayTrader = role === "DAY_TRADER";

  let tickers = isDayTrader
    ? ["NVDA", "TSLA", "PLTR", "ARM", "SMCI", "AMD", "META", "AAPL", "MSFT", "AMZN", "CRWD", "PANW", "NET", "DDOG", "MDB", "COIN", "MARA", "MSTR", "HOOD", "DUOL", "CELH", "IONQ", "RKLB", "APP"]
    : ["LNTH", "CPRX", "MEDP", "TMDX", "ISRG", "VRTX", "LLY", "NVO", "DXCM", "PODD", "ACLS", "POWI", "ON", "MPWR", "KLAC", "LRCX", "ASML", "AVGO", "ELF", "DECK", "LULU", "ONON", "MNST", "ULTA", "VRT", "ETN", "PWR", "GEV", "FIX", "EME", "ANET", "NOW", "SNPS", "CDNS"];

  if (customQuery && customQuery.trim()) {
    const parsed = customQuery.replace(/,/g, " ").split(/\s+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (parsed.length > 0) tickers = parsed;
  }

  const BASE_PRICES: Record<string, { price: number; name: string; roic: number; peg: number; margin: number; rvol: number; short: number }> = {
    LNTH: { price: 100.78, name: "Lantheus Holdings", roic: 32.4, peg: 0.82, margin: 68.5, rvol: 2.4, short: 4.8 },
    CPRX: { price: 23.40, name: "Catalyst Pharmaceuticals", roic: 28.5, peg: 0.74, margin: 76.2, rvol: 2.8, short: 5.2 },
    MEDP: { price: 342.10, name: "Medpace Holdings", roic: 36.8, peg: 0.92, margin: 71.0, rvol: 1.8, short: 3.5 },
    TMDX: { price: 92.60, name: "TransMedics Group", roic: 24.2, peg: 1.15, margin: 78.4, rvol: 3.2, short: 8.4 },
    ISRG: { price: 446.50, name: "Intuitive Surgical", roic: 22.5, peg: 1.45, margin: 67.2, rvol: 1.7, short: 2.1 },
    VRTX: { price: 482.10, name: "Vertex Pharmaceuticals", roic: 26.0, peg: 0.95, margin: 82.5, rvol: 1.9, short: 2.4 },
    LLY: { price: 924.50, name: "Eli Lilly and Company", roic: 38.2, peg: 1.35, margin: 79.5, rvol: 2.1, short: 1.8 },
    NVO: { price: 136.40, name: "Novo Nordisk A/S", roic: 44.0, peg: 0.88, margin: 84.0, rvol: 2.3, short: 1.5 },
    DXCM: { price: 89.29, name: "DexCom Inc.", roic: 18.5, peg: 1.20, margin: 64.0, rvol: 2.9, short: 7.2 },
    PODD: { price: 143.41, name: "Insulet Corporation", roic: 21.0, peg: 1.10, margin: 68.0, rvol: 2.2, short: 6.4 },
    ACLS: { price: 84.20, name: "Axcelis Technologies", roic: 27.5, peg: 0.78, margin: 54.0, rvol: 2.6, short: 5.8 },
    POWI: { price: 68.50, name: "Power Integrations", roic: 22.0, peg: 0.85, margin: 56.5, rvol: 1.8, short: 3.2 },
    ON: { price: 72.40, name: "ON Semiconductor", roic: 24.5, peg: 0.89, margin: 52.0, rvol: 2.2, short: 4.1 },
    MPWR: { price: 812.30, name: "Monolithic Power Systems", roic: 31.0, peg: 1.25, margin: 55.4, rvol: 2.0, short: 3.8 },
    KLAC: { price: 734.50, name: "KLA Corporation", roic: 35.0, peg: 0.98, margin: 61.2, rvol: 1.9, short: 2.5 },
    LRCX: { price: 792.10, name: "Lam Research", roic: 33.5, peg: 0.91, margin: 58.0, rvol: 2.1, short: 2.8 },
    ASML: { price: 824.60, name: "ASML Holding", roic: 42.0, peg: 1.12, margin: 52.5, rvol: 2.4, short: 1.9 },
    AVGO: { price: 158.40, name: "Broadcom Inc.", roic: 29.5, peg: 1.05, margin: 64.8, rvol: 2.5, short: 2.2 },
    ELF: { price: 118.40, name: "e.l.f. Beauty", roic: 26.5, peg: 0.84, margin: 71.0, rvol: 3.1, short: 7.8 },
    DECK: { price: 86.33, name: "Deckers Outdoor", roic: 34.2, peg: 0.96, margin: 56.0, rvol: 2.0, short: 3.4 },
    LULU: { price: 264.50, name: "Lululemon Athletica", roic: 31.0, peg: 0.88, margin: 58.5, rvol: 2.5, short: 6.2 },
    ONON: { price: 44.20, name: "On Holding AG", roic: 23.5, peg: 1.05, margin: 60.2, rvol: 3.4, short: 8.1 },
    MNST: { price: 46.70, name: "Monster Beverage", roic: 28.0, peg: 1.15, margin: 54.0, rvol: 1.6, short: 2.0 },
    ULTA: { price: 368.40, name: "Ulta Beauty", roic: 35.5, peg: 0.79, margin: 52.8, rvol: 2.2, short: 4.5 },
    VRT: { price: 88.40, name: "Vertiv Holdings", roic: 25.0, peg: 0.89, margin: 54.2, rvol: 3.5, short: 4.6 },
    ETN: { price: 312.50, name: "Eaton Corporation", roic: 21.5, peg: 1.20, margin: 53.0, rvol: 1.7, short: 2.2 },
    PWR: { price: 268.10, name: "Quanta Services", roic: 19.8, peg: 1.30, margin: 51.5, rvol: 1.8, short: 2.5 },
    GEV: { price: 224.60, name: "GE Vernova", roic: 22.0, peg: 1.10, margin: 54.0, rvol: 2.6, short: 3.8 },
    FIX: { price: 346.20, name: "Comfort Systems USA", roic: 30.5, peg: 0.86, margin: 52.0, rvol: 2.3, short: 3.1 },
    EME: { price: 382.40, name: "EMCOR Group", roic: 28.0, peg: 0.92, margin: 51.8, rvol: 2.0, short: 2.9 },
    ANET: { price: 324.50, name: "Arista Networks", roic: 32.0, peg: 0.94, margin: 65.4, rvol: 2.2, short: 3.0 },
    NOW: { price: 785.40, name: "ServiceNow Inc.", roic: 24.0, peg: 1.35, margin: 78.5, rvol: 1.8, short: 2.1 },
    SNPS: { price: 464.89, name: "Synopsys Inc.", roic: 21.0, peg: 1.40, margin: 80.0, rvol: 1.6, short: 1.8 },
    CDNS: { price: 254.20, name: "Cadence Design Systems", roic: 22.5, peg: 1.38, margin: 88.0, rvol: 1.7, short: 2.0 },
    DUOL: { price: 284.50, name: "Duolingo Inc.", roic: 26.0, peg: 1.10, margin: 73.5, rvol: 3.6, short: 8.5 },
    NVDA: { price: 128.50, name: "NVIDIA Corp.", roic: 48.0, peg: 0.92, margin: 75.0, rvol: 2.8, short: 2.2 },
    TSLA: { price: 218.40, name: "Tesla Inc.", roic: 18.0, peg: 1.60, margin: 54.5, rvol: 3.1, short: 6.8 },
    PLTR: { price: 31.20, name: "Palantir Technologies", roic: 23.0, peg: 1.25, margin: 81.0, rvol: 3.8, short: 7.2 },
    ARM: { price: 134.80, name: "Arm Holdings", roic: 25.5, peg: 1.45, margin: 95.0, rvol: 3.0, short: 6.5 },
    SMCI: { price: 43.60, name: "Super Micro Computer", roic: 22.4, peg: 0.72, margin: 52.0, rvol: 4.2, short: 14.8 },
    AMD: { price: 146.20, name: "Advanced Micro Devices", roic: 19.5, peg: 1.18, margin: 52.5, rvol: 2.4, short: 3.2 },
    META: { price: 512.40, name: "Meta Platforms", roic: 32.0, peg: 0.88, margin: 81.5, rvol: 2.1, short: 1.6 },
    AAPL: { price: 319.64, name: "Apple Inc.", roic: 45.0, peg: 1.30, margin: 56.0, rvol: 1.7, short: 1.4 },
    MSFT: { price: 418.20, name: "Microsoft Corp.", roic: 36.0, peg: 1.22, margin: 69.5, rvol: 1.8, short: 1.2 },
    AMZN: { price: 178.60, name: "Amazon.com Inc.", roic: 22.0, peg: 1.15, margin: 58.0, rvol: 2.0, short: 1.5 },
    CRWD: { price: 272.50, name: "CrowdStrike Holdings", roic: 24.5, peg: 1.20, margin: 76.0, rvol: 3.2, short: 6.4 },
    PANW: { price: 348.10, name: "Palo Alto Networks", roic: 21.0, peg: 1.30, margin: 74.0, rvol: 2.3, short: 4.2 },
    NET: { price: 82.40, name: "Cloudflare Inc.", roic: 19.0, peg: 1.40, margin: 77.0, rvol: 2.9, short: 7.5 },
    DDOG: { price: 114.20, name: "Datadog Inc.", roic: 22.0, peg: 1.25, margin: 81.0, rvol: 2.7, short: 5.8 },
    MDB: { price: 288.60, name: "MongoDB Inc.", roic: 18.0, peg: 1.50, margin: 75.0, rvol: 2.8, short: 8.2 },
    COIN: { price: 212.30, name: "Coinbase Global", roic: 27.0, peg: 0.95, margin: 85.0, rvol: 3.9, short: 11.2 },
    MARA: { price: 16.80, name: "MARA Holdings", roic: 19.0, peg: 0.85, margin: 54.0, rvol: 4.1, short: 16.4 },
    MSTR: { price: 134.20, name: "MicroStrategy Inc.", roic: 22.0, peg: 1.10, margin: 72.0, rvol: 3.7, short: 13.5 },
    HOOD: { price: 21.60, name: "Robinhood Markets", roic: 21.5, peg: 0.90, margin: 78.0, rvol: 3.1, short: 7.9 },
    CELH: { price: 38.40, name: "Celsius Holdings", roic: 21.0, peg: 0.92, margin: 52.0, rvol: 3.3, short: 9.2 },
    IONQ: { price: 9.20, name: "IonQ Inc.", roic: 18.0, peg: 1.60, margin: 62.0, rvol: 3.6, short: 12.4 },
    RKLB: { price: 7.10, name: "Rocket Lab USA", roic: 19.5, peg: 1.40, margin: 55.0, rvol: 3.4, short: 8.8 },
    APP: { price: 86.40, name: "AppLovin Corp.", roic: 34.0, peg: 0.82, margin: 69.0, rvol: 3.5, short: 6.8 },
  };

  return tickers.map((sym, idx) => {
    const canonicalName = getCanonicalAssetName(sym);
    const canonicalMoat = getCanonicalAssetMoat(sym);
    const canonicalCatalyst = getCanonicalAssetCatalyst(sym);

    const base = BASE_PRICES[sym] || {
      price: 100.0,
      name: canonicalName,
      roic: 24.0,
      peg: 0.95,
      margin: 62.0,
      rvol: 2.2,
      short: 5.0,
    };

    const price = base.price;
    const h = (idx * 17 + sym.charCodeAt(0) * 31) % 100;

    let executionStatus: "IN_BUY_ZONE" | "APPROACHING_TARGET" | "WAITING_PULLBACK" | "STOPPED_OUT";
    let statusLabel: string;
    let statusColor: string;

    const isStage4Candidate = sym === "DECK" || sym === "PODD" || sym === "MNST" || sym === "ULTA" || sym === "LULU";

    if (isStage4Candidate) {
      executionStatus = "WAITING_PULLBACK";
      statusLabel = sym === "ULTA" || sym === "LULU" 
        ? "⚠️ Stage 4 Turnaround Watch" 
        : "⏳ Awaiting Base Formation";
      statusColor = sym === "ULTA" || sym === "LULU" ? "amber" : "cyan";
    } else if (h < 40) {
      executionStatus = "IN_BUY_ZONE";
      statusLabel = isDayTrader ? "🎯 Active VWAP Bounce" : "🎯 Active Buy Zone";
      statusColor = "emerald";
    } else if (h < 70) {
      executionStatus = "APPROACHING_TARGET";
      statusLabel = isDayTrader ? "🚀 Session ORB Breakout" : "🚀 Near TP Target";
      statusColor = "amber";
    } else if (h < 95) {
      executionStatus = "WAITING_PULLBACK";
      statusLabel = "⏳ Pullback Pending";
      statusColor = "cyan";
    } else {
      executionStatus = "STOPPED_OUT";
      statusLabel = "🛑 Invalidation Alert";
      statusColor = "rose";
    }

    const optimalEntryMin = Number((price * (isDayTrader ? 0.992 : 0.965)).toFixed(2));
    const optimalEntryMax = Number((price * (isDayTrader ? 1.004 : 1.015)).toFixed(2));
    const stopLoss = Number((price * (isDayTrader ? 0.985 : 0.945)).toFixed(2));
    const stopLossPct = -5.5;
    const takeProfit1 = Number((price * 1.085).toFixed(2));
    const takeProfit1Pct = 8.5;
    const takeProfit2 = Number((price * 1.155).toFixed(2));
    const takeProfit2Pct = 15.5;
    const riskRewardRatio = Number(((takeProfit1 - price) / Math.max(0.01, price - stopLoss)).toFixed(2));

    const rawConfluence = Math.min(96, Math.max(72, 75 + (h % 22)));
    const confluenceScore = isStage4Candidate ? Math.min(68, rawConfluence) : rawConfluence;
    const confluenceRating = isStage4Candidate ? "⚠️ TURNAROUND WATCH" : (confluenceScore >= 85 ? "⭐ HIGH CONFLUENCE" : "MODERATE CONFLUENCE");
    const confluenceBadgeColor = isStage4Candidate ? "amber" : (confluenceScore >= 85 ? "emerald" : "cyan");

    let archetype = "Peter Lynch & Greenblatt GARP";
    if (isDayTrader) {
      archetype = base.short > 8.0 ? "Short Squeeze High-Beta Scalp" : "High RVOL Trend Momentum Leader";
    } else {
      if (sym === "ULTA" || sym === "LULU") {
        archetype = "Deep Value & Capital Return (Decelerating Comp Watch)";
      } else if (base.peg <= 1.0) {
        archetype = "Peter Lynch GARP Compounder";
      } else if (base.roic >= 25.0) {
        archetype = "Joel Greenblatt Magic Formula";
      } else if (base.margin >= 65.0) {
        archetype = "David Gardner Rule Breaker";
      }
    }

    const confluenceWarnings = (sym === "ULTA" || sym === "LULU") 
      ? ["Negative 1Y/3Y momentum trend", "Prestige beauty comp deceleration", "Trading below 200-day EMA"]
      : (isStage4Candidate ? ["Awaiting Stage 1 base completion"] : []);

    return {
      symbol: sym,
      companyName: base.name,
      currentPrice: price,
      gemScore: isStage4Candidate ? Math.min(74, 76 + (h % 10)) : 82 + (h % 16),
      expertArchetype: archetype,
      roic: `${base.roic}%`,
      pegRatio: `${base.peg}`,
      grossMargin: `${base.margin}%`,
      atr14: `$${(price * (isDayTrader ? 0.032 : 0.024)).toFixed(2)}`,
      rvol: `${base.rvol}x`,
      shortFloat: `${base.short}%`,
      dayTraderSetup: isDayTrader
        ? "Intraday momentum trend-following above 5m VWAP anchor with defined ATR risk."
        : (isStage4Candidate ? "Stage 4 consolidation — awaiting base formation and comp stabilization." : "Stage 2 accumulation breakout above 50-day pivot."),
      thesis: canonicalMoat || `${base.name} demonstrates ${base.roic}% ROIC with ${base.margin}% gross margins.`,
      catalyst: canonicalCatalyst?.trial || canonicalCatalyst?.thesis || "Upcoming product cycle expansion and institutional accumulation.",
      riskLevel: (sym === "ULTA" || sym === "LULU") ? "High Turnaround Risk" : (isDayTrader ? "High Volatility (Intraday)" : "Low-to-Medium Risk"),
      executionStatus,
      statusLabel,
      statusColor,
      optimalEntryMin,
      optimalEntryMax,
      stopLoss,
      stopLossPct,
      takeProfit1,
      takeProfit1Pct,
      takeProfit2,
      takeProfit2Pct,
      riskRewardRatio: Math.max(1.85, riskRewardRatio),
      setupPattern: isStage4Candidate ? "Stage 4 Mean-Reversion Base" : "Minervini Volatility Contraction Pattern (VCP 3-Stage)",
      entryThesis: isStage4Candidate ? "Awaiting Stage 1 base completion before new entry." : "Stage 2 accumulation breakout above 50-day pivot.",
      confluenceScore,
      confluenceRating,
      confluenceBadgeColor,
      confluenceReasons: isStage4Candidate ? ["Compressed valuation multiple", "High historical ROIC"] : ["Above 20 EMA / 50 SMA support", "Institutional accumulation surge"],
      confluenceWarnings,
    };
  });
}

export default function ScreenerPage() {
  const [selectedFilter, setSelectedFilter] = useState("all");
  const [activeRole, setActiveRole] = useState<"DAY_TRADER" | "LONG_TERM">("LONG_TERM");
  const [gems, setGems] = useState<GemCandidate[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [dataSource, setDataSource] = useState<"live" | "fallback">("live");
  const [sizerGem, setSizerGem] = useState<GemCandidate | null>(null);
  const [alertGem, setAlertGem] = useState<GemCandidate | null>(null);
  const [customTickerInput, setCustomTickerInput] = useState<string>("");
  const [activeCustomQuery, setActiveCustomQuery] = useState<string>("");
  const [copyToast, setCopyToast] = useState<boolean>(false);
  const [loggedGemSymbol, setLoggedGemSymbol] = useState<string | null>(null);
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  useEffect(() => {
    const savedRole = localStorage.getItem("FINANCE_USER_ROLE");
    if (savedRole === "DAY_TRADER" || savedRole === "LONG_TERM") {
      setActiveRole(savedRole);
    }
    const savedFilter = localStorage.getItem("FINANCE_SCREENER_TAB");
    if (savedFilter) {
      setSelectedFilter(savedFilter);
    }
    const savedQuery = localStorage.getItem("FINANCE_SCREENER_QUERY");
    if (savedQuery) {
      setActiveCustomQuery(savedQuery);
      setCustomTickerInput(savedQuery);
    }
    const savedV = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
    if (savedV) {
      setVernacularMode(savedV);
    }

    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const handleRoleToggle = (role: "DAY_TRADER" | "LONG_TERM") => {
    setActiveRole(role);
    setSelectedFilter("all");
    localStorage.setItem("FINANCE_USER_ROLE", role);
    localStorage.setItem("FINANCE_SCREENER_TAB", "all");
  };

  const handleSelectFilter = (tabId: string) => {
    setSelectedFilter(tabId);
    localStorage.setItem("FINANCE_SCREENER_TAB", tabId);
  };

  const executeScreenerFetch = async (role: "DAY_TRADER" | "LONG_TERM", customQuery?: string) => {
    setLoading(true);
    try {
      let url = `${API_BASE_URL}/screener/run?filter_type=all&user_role=${role}`;
      if (customQuery && customQuery.trim()) {
        url += `&custom_tickers=${encodeURIComponent(customQuery.trim())}`;
      }
      const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.candidates) && data.candidates.length > 0) {
          const liveGems: GemCandidate[] = data.candidates.map((c: any) => {
            const masterMeta = MASTER_ASSET_CATALOG[c.symbol as keyof typeof MASTER_ASSET_CATALOG];
            const sym = c.symbol;
            let archetype = c.expertArchetype;
            if (!archetype || archetype === "Peter Lynch & Greenblatt GARP") {
              if (role === "DAY_TRADER") {
                archetype = (masterMeta?.shortFloat || 5) > 8 ? "Short Squeeze High-Beta Scalp" : "High RVOL Trend Momentum Leader";
              } else if (sym === "ULTA" || sym === "LULU") {
                archetype = "Deep Value & Capital Return (Decelerating Comp Watch)";
              } else if ((masterMeta?.peg || 1.1) <= 1.05) {
                archetype = "Peter Lynch GARP Compounder";
              } else if ((masterMeta?.roic || 20) >= 28.0) {
                archetype = "Joel Greenblatt Magic Formula";
              } else if ((masterMeta?.grossMargin || 50) >= 65.0) {
                archetype = "David Gardner Rule Breaker";
              } else {
                archetype = "Quality Compounder";
              }
            }

            return {
              symbol: c.symbol,
              companyName: c.companyName || masterMeta?.name || c.symbol,
              currentPrice: c.currentPrice || masterMeta?.price || 100.0,
              gemScore: c.gemScore || 88,
              expertArchetype: archetype,
              roic: c.roic || (masterMeta ? `${masterMeta.roic}%` : "28.5%"),
              pegRatio: c.pegRatio || (masterMeta ? `${masterMeta.peg}` : "0.85"),
              grossMargin: c.grossMargin || (masterMeta ? `${masterMeta.grossMargin}%` : "65.0%"),
              thesis: c.thesis || masterMeta?.thesis || "High return on capital with strong free cash flows.",
              atr14: c.atr14 || `$${((c.currentPrice || 100) * 0.025).toFixed(2)}`,
              rvol: c.rvol || `${masterMeta?.rvol || 2.1}x`,
              shortFloat: c.shortFloat || `${masterMeta?.shortFloat || 6.8}%`,
              dayTraderSetup: c.dayTraderSetup || "Intraday momentum trend-following above 5m VWAP with clear risk-defined support.",
              catalyst: c.catalyst || masterMeta?.upcomingCatalyst || "Upcoming product cycle expansion and institutional accumulation.",
              riskLevel: role === "DAY_TRADER" ? "High Volatility (Intraday)" : (c.riskLevel || "Low-to-Medium Risk"),
              executionStatus: c.executionStatus || "IN_BUY_ZONE",
              statusLabel: c.statusLabel || "🎯 Active Buy Zone",
              statusColor: c.statusColor || "emerald",
              optimalEntryMin: c.optimalEntryMin || Number(((c.currentPrice || 100) * 0.975).toFixed(2)),
              optimalEntryMax: c.optimalEntryMax || Number(((c.currentPrice || 100) * 0.995).toFixed(2)),
              stopLoss: c.stopLoss || Number(((c.currentPrice || 100) * 0.955).toFixed(2)),
              stopLossPct: c.stopLossPct || -4.5,
              takeProfit1: c.takeProfit1 || Number(((c.currentPrice || 100) * 1.045).toFixed(2)),
              takeProfit1Pct: c.takeProfit1Pct || 4.5,
              takeProfit2: c.takeProfit2 || Number(((c.currentPrice || 100) * 1.095).toFixed(2)),
              takeProfit2Pct: c.takeProfit2Pct || 9.5,
              riskRewardRatio: c.riskRewardRatio || 2.85,
              setupPattern: c.setupPattern || "Minervini Volatility Contraction Pattern (VCP 3-Stage)",
              entryThesis: c.entryThesis || "Stage 2 accumulation breakout above 50-day pivot.",
              confluenceScore: c.confluenceScore || 85,
              confluenceRating: c.confluenceRating || "⭐ HIGH CONFLUENCE",
              confluenceBadgeColor: c.confluenceBadgeColor || "emerald",
              confluenceReasons: c.confluenceReasons || [],
              confluenceWarnings: c.confluenceWarnings || [],
            };
          });
          setGems(liveGems);
          setDataSource("live");
          return;
        }
      }
      // Fallback if API response is not OK or empty
      const fallbackList = generateBuiltinGems(role, customQuery);
      setGems(fallbackList);
      setDataSource("fallback");
    } catch (err) {
      console.warn("Live screener fetch offline/timed out, using quantitative model catalog:", err);
      const fallbackList = generateBuiltinGems(role, customQuery);
      setGems(fallbackList);
      setDataSource("fallback");
    } finally {
      setLoading(false);
    }
  };

  // Fetch Live Screener Data when activeRole changes
  useEffect(() => {
    executeScreenerFetch(activeRole, activeCustomQuery);

    const handlePurge = () => {
      executeScreenerFetch(activeRole, activeCustomQuery);
    };

    const handleRoleEvent = (e: Event) => {
      const custom = e as CustomEvent<"DAY_TRADER" | "LONG_TERM">;
      if (custom.detail === "DAY_TRADER" || custom.detail === "LONG_TERM") {
        setActiveRole(custom.detail);
        setSelectedFilter("all");
      }
    };

    window.addEventListener("finance:cache-purge", handlePurge);
    window.addEventListener("finance:role-change", handleRoleEvent);
    return () => {
      window.removeEventListener("finance:cache-purge", handlePurge);
      window.removeEventListener("finance:role-change", handleRoleEvent);
    };
  }, [activeRole, activeCustomQuery]);

  // Background Live Quote Synchronization to guarantee 100% price parity with Terminal
  useEffect(() => {
    if (gems.length === 0) return;
    let isMounted = true;

    const syncLiveQuotes = () => {
      const quoteMap = new Map<string, number>();

      for (const gem of gems) {
        const upper = gem.symbol.toUpperCase();
        const reg = SpotPriceRegistry.get(upper);
        const snap = getPersistedMarketSnapshot(upper);
        const effectivePrice = (reg?.price && reg.price > 0)
          ? reg.price
          : (snap?.currentPrice && snap.currentPrice > 0)
          ? snap.currentPrice
          : MASTER_ASSET_CATALOG[upper]?.price;

        if (effectivePrice && Math.abs(effectivePrice - (gem.currentPrice || 0)) > 0.05) {
          quoteMap.set(gem.symbol, effectivePrice);
        }
      }

      if (isMounted && quoteMap.size > 0) {
        setGems((prev) =>
          prev.map((g) => {
            const livePrice = quoteMap.get(g.symbol);
            if (!livePrice) return g;
            const optimalEntryMin = Number((livePrice * 0.975).toFixed(2));
            const optimalEntryMax = Number((livePrice * 1.018).toFixed(2));
            const stopLoss = Number((livePrice * 0.945).toFixed(2));
            const takeProfit1 = Number((livePrice * 1.085).toFixed(2));
            const takeProfit2 = Number((livePrice * 1.155).toFixed(2));
            return {
              ...g,
              currentPrice: livePrice,
              optimalEntryMin,
              optimalEntryMax,
              stopLoss,
              takeProfit1,
              takeProfit2,
            };
          })
        );
      }
    };

    syncLiveQuotes();

    return () => {
      isMounted = false;
    };
  }, [gems.length, activeRole]);

  const handleCustomSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (customTickerInput.trim()) {
      setActiveCustomQuery(customTickerInput.trim());
      setSelectedFilter("all");
    }
  };

  const handleScanSavedWatchlist = () => {
    try {
      const savedPortfolio = localStorage.getItem("FINANCE_USER_PORTFOLIO") || localStorage.getItem("FINANCE_PORTFOLIO_V1");
      let tickers = "NVDA, AAPL, MSFT, TSLA, AMZN, PLTR, AMD";
      if (savedPortfolio) {
        const parsed = JSON.parse(savedPortfolio);
        if (Array.isArray(parsed) && parsed.length > 0) {
          tickers = parsed.map((p: any) => p.symbol || p.ticker).filter(Boolean).join(", ");
        }
      }
      setCustomTickerInput(tickers);
      setActiveCustomQuery(tickers);
      setSelectedFilter("all");
    } catch (err) {
      console.warn("Error scanning watchlist:", err);
    }
  };

  const handleClearCustomQuery = () => {
    setCustomTickerInput("");
    setActiveCustomQuery("");
    setSelectedFilter("all");
  };

  const handleCopyTickers = () => {
    try {
      const symbols = displayGems.map((g) => g.symbol).join(", ");
      navigator.clipboard.writeText(symbols);
      setCopyToast(true);
      trackMatomoEvent("Screener", "Copy Screener Tickers", `Count: ${displayGems.length}`);
      setTimeout(() => setCopyToast(false), 2500);
    } catch (err) {
      console.warn("Could not copy tickers:", err);
    }
  };

  const handleExportScreenerCsv = () => {
    if (typeof window === "undefined" || displayGems.length === 0) return;
    trackMatomoEvent("Screener", "Export Screener CSV", `Filter: ${selectedFilter} (${displayGems.length} rows)`);
    const headers = [
      "Symbol",
      "Company Name",
      "Current Price ($)",
      "Execution Status",
      "Optimal Entry Min ($)",
      "Optimal Entry Max ($)",
      "Stop Loss ($)",
      "Take Profit 1 ($)",
      "Take Profit 2 ($)",
      "Risk-Reward Ratio",
      "Confluence Score",
      "Confluence Rating",
      "Setup Pattern",
      "Archetype",
      "Catalyst",
      "Risk Level",
    ];

    const rows = displayGems.map((g) =>
      [
        g.symbol,
        `"${(g.companyName || g.symbol).replace(/"/g, '""')}"`,
        g.currentPrice ? g.currentPrice.toFixed(2) : "N/A",
        g.executionStatus || "N/A",
        g.optimalEntryMin ? g.optimalEntryMin.toFixed(2) : "N/A",
        g.optimalEntryMax ? g.optimalEntryMax.toFixed(2) : "N/A",
        g.stopLoss ? g.stopLoss.toFixed(2) : "N/A",
        g.takeProfit1 ? g.takeProfit1.toFixed(2) : "N/A",
        g.takeProfit2 ? g.takeProfit2.toFixed(2) : "N/A",
        g.riskRewardRatio ? g.riskRewardRatio.toFixed(2) : "N/A",
        g.confluenceScore !== undefined ? g.confluenceScore : "N/A",
        g.confluenceRating || "N/A",
        `"${(g.setupPattern || "").replace(/"/g, '""')}"`,
        `"${(g.expertArchetype || "").replace(/"/g, '""')}"`,
        `"${(g.catalyst || "").replace(/"/g, '""')}"`,
        g.riskLevel || "N/A",
      ].join(",")
    );

    const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent([headers.join(","), ...rows].join("\n"));
    const link = document.createElement("a");
    link.setAttribute("href", csvContent);
    link.setAttribute("download", `finance_screener_${selectedFilter}_${new Date().toISOString().split("T")[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const isDayTrader = activeRole === "DAY_TRADER";
  const isPlain = vernacularMode === "PLAIN_ENGLISH";

  const plainLongTermTabs = [
    { id: "all", label: "🏛️ All Quality Stocks", desc: "Solid, Profitable Long-Term Businesses", badge: "Universe" },
    { id: "high_confluence", label: "⭐ Top Consensus Picks", desc: "Top Picks Supported by Multiple Financial Models (≥ 80%)", badge: "Top Ranked" },
    { id: "in_buy_zone", label: "🎯 Great Price to Buy", desc: "Currently in an optimal buying price range", badge: "Actionable" },
    { id: "approaching_target", label: "🚀 Near Profit Goal", desc: "Stock is reaching its calculated profit targets", badge: "Take Gains" },
    { id: "high_rr", label: "⚡ Low Risk / High Reward", desc: "High upside potential with tight downside safety (≥ 2.0:1)", badge: "Asymmetric" },
    { id: "lynch", label: "📈 Bargain Growth (Peter Lynch)", desc: "Fast-growing companies at reasonable valuations", badge: "Value" },
    { id: "greenblatt", label: "🧪 High Return on Capital (Joel Greenblatt)", desc: "Generates high cash returns per dollar invested", badge: "Quality" },
    { id: "rule_breakers", label: "🔥 Category Disruptors", desc: "Industry leaders with wide competitive advantages and high margins", badge: "High Growth" },
  ];

  const plainDayTraderTabs = [
    { id: "all", label: "⚡ All Active Movers", desc: "Fast-Moving Intraday Leaders", badge: "Active" },
    { id: "high_confluence", label: "⭐ High Conviction Setup", desc: "Strong Technical Momentum & Heavy Buying Flow", badge: "Top Flow" },
    { id: "in_buy_zone", label: "🎯 Healthy Dip to Buy", desc: "Pullback holding support on moving averages", badge: "Dip Buy" },
    { id: "approaching_target", label: "🚀 Breakout in Progress", desc: "Pushing into new session highs with momentum", badge: "Breakout" },
    { id: "high_rr", label: "⚡ Best Profit vs Risk", desc: "Tight safety stop with strong scalp targets", badge: "Tight Risk" },
    { id: "high_rvol", label: "🔥 Heavy Trading Volume", desc: "Volume surge: institutional trading active", badge: "Surge" },
    { id: "squeeze", label: "💥 Short Squeeze Candidates", desc: "High short interest with rapid upward pressure", badge: "Squeeze" },
  ];

  const activeTabs = isPlain
    ? (isDayTrader ? plainDayTraderTabs : plainLongTermTabs)
    : (isDayTrader ? DAY_TRADER_FILTER_TABS : LONG_TERM_FILTER_TABS);

  const parseNum = (val: any): number => {
    if (val === null || val === undefined) return 0;
    if (typeof val === "number") return isNaN(val) ? 0 : val;
    if (typeof val === "string") {
      const cleaned = val.replace(/[^0-9.-]/g, "");
      const parsed = parseFloat(cleaned);
      return isNaN(parsed) ? 0 : parsed;
    }
    return 0;
  };

  const hasArchetype = (gem: GemCandidate, keyword: string): boolean => {
    if (!gem || !gem.expertArchetype || typeof gem.expertArchetype !== "string") return false;
    return gem.expertArchetype.toLowerCase().includes(keyword.toLowerCase());
  };

  // Instant Client-Side Filter with 0ms Latency and Robust Type Parsing
  const isMatchFilter = (gem: GemCandidate, filterId: string): boolean => {
    if (!filterId || filterId === "all") return true;
    if (filterId === "high_confluence") return (gem.confluenceScore || 0) >= 80;
    if (filterId === "in_buy_zone" || filterId === "vwap_pullback") return gem.executionStatus === "IN_BUY_ZONE";
    if (filterId === "approaching_target" || filterId === "orb_breakout") return gem.executionStatus === "APPROACHING_TARGET";
    if (filterId === "high_rr") {
      // Asymmetric plan geometry (>= 2.0:1) AND Actionable buy zone proximity (In buy zone or spot <= optimalEntryMax * 1.02)
      const isActionable = gem.executionStatus === "IN_BUY_ZONE" || (gem.currentPrice || 0) <= (gem.optimalEntryMax || gem.currentPrice || 0) * 1.02;
      return (gem.riskRewardRatio || 0) >= 2.0 && isActionable;
    }
    if (filterId === "high_rvol") return parseNum(gem.rvol) >= 2.5;
    if (filterId === "squeeze") return parseNum(gem.shortFloat) >= 6.0;
    if (filterId === "lynch") {
      const peg = parseNum(gem.pegRatio);
      return (peg > 0 && peg <= 1.05) || hasArchetype(gem, "Lynch") || hasArchetype(gem, "GARP");
    }
    if (filterId === "greenblatt") {
      return parseNum(gem.roic) >= 28.0 || hasArchetype(gem, "Greenblatt") || hasArchetype(gem, "Magic");
    }
    if (filterId === "rule_breakers") {
      return parseNum(gem.grossMargin) >= 65.0 || hasArchetype(gem, "Rule Breakers") || hasArchetype(gem, "Disruptive");
    }
    return true;
  };

  const displayGems = gems.filter((gem) => isMatchFilter(gem, selectedFilter));
  const getTabCount = (tabId: string) => gems.filter((gem) => isMatchFilter(gem, tabId)).length;

  return (
    <main id="main-content" role="main" className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-mono flex flex-col pb-28 sm:pb-8 transition-colors duration-200">
      <Navbar userRole={activeRole} onRoleChange={handleRoleToggle} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1">
        {/* Page Hero Header with Dual-Horizon View Mode Indicator */}
        <div className="mb-5 border-b border-[#1b2434] pb-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-2xl">💎</span>
                <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                  Market Scanner: Unfair Advantage Stock Finder
                </h1>
              </div>
              <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-3xl">
                Filter high-probability trade setups by clear buy zones, calculated profit targets, insider backing, and mathematical edge.
              </p>
            </div>

            {/* Lens Switcher Pill */}
            <div className="flex items-center gap-2">
              <DataSourceBadge source={dataSource} />
              <div role="radiogroup" aria-label="Execution Lens" className="flex items-center bg-[#070a10] p-1 rounded-xl border border-[#243044]">
                <span className="text-[11px] text-slate-400 font-bold px-2 hidden sm:inline">Execution Lens:</span>
                <button
                  onClick={() => handleRoleToggle("DAY_TRADER")}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                    isDayTrader
                      ? "bg-amber-500 text-slate-950 shadow-md font-extrabold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  ⚡ Day Trader (Scalps/Intraday)
                </button>
                <button
                  onClick={() => handleRoleToggle("LONG_TERM")}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all active:scale-[0.96] ${
                    !isDayTrader
                      ? "bg-cyan-500 text-slate-950 shadow-md font-extrabold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  🏛️ Long-Term (Compounders)
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Custom Multi-Ticker Search Bar & Watchlist Scanner */}
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3 bg-[#0d121c] p-3 rounded-2xl border border-[#1e293b]">
          <form onSubmit={handleCustomSearch} className="flex-1 flex items-center gap-2 min-w-[280px]">
            <div className="relative flex-1">
              <span className="absolute left-3 top-2.5 text-xs text-slate-500">🔍</span>
              <input
                type="text"
                value={customTickerInput}
                onChange={(e) => setCustomTickerInput(e.target.value)}
                placeholder="Search specific tickers: e.g. NVDA, AAPL, PLTR, MSFT, LLY..."
                className="w-full bg-[#070a11] border border-[#243044] rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono transition-colors"
              />
            </div>
            <button
              type="submit"
              className={`px-4 py-2 rounded-xl text-xs font-black transition-all active:scale-[0.96] flex items-center gap-1.5 shadow ${
                isDayTrader ? "bg-amber-500 hover:bg-amber-400 text-slate-950" : "bg-cyan-500 hover:bg-cyan-400 text-slate-950"
              }`}
            >
              <span>⚡</span>
              <span>Scan</span>
            </button>
          </form>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={handleScanSavedWatchlist}
              className="px-3 py-2 rounded-xl text-xs font-bold bg-[#141b29] hover:bg-[#1c2638] border border-[#223149] text-slate-200 transition-all active:scale-[0.96] flex items-center gap-1.5 shadow"
            >
              <span>💼</span>
              <span>Scan My Portfolio</span>
            </button>

            <button
              type="button"
              onClick={handleCopyTickers}
              className="px-3 py-2 rounded-xl text-xs font-bold bg-[#141b29] hover:bg-[#1c2638] border border-[#223149] text-cyan-300 hover:text-white transition-all active:scale-[0.96] flex items-center gap-1.5 shadow"
              title="Copy filtered tickers to clipboard for Thinkorswim, TradingView, or IBKR"
            >
              <span>📋</span>
              <span>{copyToast ? "✅ Tickers Copied!" : `Copy ${displayGems.length} Tickers`}</span>
            </button>

            <button
              type="button"
              onClick={handleExportScreenerCsv}
              className="px-3 py-2 rounded-xl text-xs font-bold bg-[#141b29] hover:bg-[#1c2638] border border-[#223149] text-slate-200 hover:text-white transition-all active:scale-[0.96] flex items-center gap-1.5 shadow"
              title="Download filtered screener candidates as CSV spreadsheet"
            >
              <span>📥</span>
              <span>Export CSV</span>
            </button>

            {activeCustomQuery && (
              <button
                type="button"
                onClick={handleClearCustomQuery}
                className="px-3 py-2 rounded-xl text-xs font-bold bg-[#1a2333] hover:bg-[#223046] border border-[#2b3c58] text-slate-300 transition-all active:scale-[0.96] flex items-center gap-1.5 shadow"
              >
                <span>✕</span>
                <span>Reset (60-Asset Catalog)</span>
              </button>
            )}
          </div>
        </div>

        {/* Execution & Archetype Filter Clusters */}
        <div className="space-y-4 mb-6">
          {/* Cluster 1: Actionable Execution Status */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider">
                🎯 Cluster 1: Execution Levels & Asymmetry
              </span>
              <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                (Price action, buy zones, and calculated profit milestones)
              </span>
            </div>
            <div role="tablist" aria-label="Execution Filter Tabs" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-2.5">
              {activeTabs.slice(0, isDayTrader ? 5 : 5).map((tab) => {
                const isActive = selectedFilter === tab.id;
                const count = getTabCount(tab.id);
                return (
                  <button
                    key={tab.id}
                    role="tab"
                    aria-selected={isActive}
                    onClick={() => handleSelectFilter(tab.id)}
                    className={`p-3 rounded-xl border text-left transition-all active:scale-[0.98] flex flex-col justify-between ${
                      isActive
                        ? isDayTrader
                          ? "bg-[#21190c] border-amber-500 shadow-md shadow-amber-950/40"
                          : "bg-[#111c2e] border-cyan-500 shadow-md shadow-cyan-950/40"
                        : "bg-[#0c1017] border-[#1b2434] hover:bg-[#111722] hover:border-[#2b3a52]"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <span className={`text-xs font-black truncate ${isActive ? (isDayTrader ? "text-amber-400" : "text-cyan-400") : "text-slate-200"}`}>
                          {tab.label}
                        </span>
                        <span className={`text-[10px] font-black px-1.5 py-0.2 rounded shrink-0 font-mono tabular-nums ${
                          isActive ? (isDayTrader ? "bg-amber-400 text-slate-950" : "bg-cyan-400 text-slate-950") : "bg-[#1b2639] text-slate-300"
                        }`}>
                          {count}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-1 leading-relaxed font-sans line-clamp-2">{tab.desc}</p>
                    </div>
                    <span className="text-[9px] px-2 py-0.5 rounded font-bold uppercase tracking-wider bg-[#1e293b] text-slate-300 self-start mt-2">
                      {tab.badge}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Cluster 2: Multi-Factor Quant Archetypes */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider">
                🏛️ Cluster 2: Multi-Factor Expert Archetypes
              </span>
              <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                (Balance sheet quality, Peter Lynch GARP, and Joel Greenblatt ROIC)
              </span>
            </div>
            <div role="tablist" aria-label="Quant Archetype Filter Tabs" className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {activeTabs.slice(isDayTrader ? 5 : 5).map((tab) => {
                const isActive = selectedFilter === tab.id;
                const count = getTabCount(tab.id);
                return (
                  <button
                    key={tab.id}
                    role="tab"
                    aria-selected={isActive}
                    onClick={() => handleSelectFilter(tab.id)}
                    className={`p-3 rounded-xl border text-left transition-all active:scale-[0.98] flex flex-col justify-between ${
                      isActive
                        ? isDayTrader
                          ? "bg-[#21190c] border-amber-500 shadow-md shadow-amber-950/40"
                          : "bg-[#111c2e] border-cyan-500 shadow-md shadow-cyan-950/40"
                        : "bg-[#0c1017] border-[#1b2434] hover:bg-[#111722] hover:border-[#2b3a52]"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <span className={`text-xs font-black truncate ${isActive ? (isDayTrader ? "text-amber-400" : "text-cyan-400") : "text-slate-200"}`}>
                          {tab.label}
                        </span>
                        <span className={`text-[10px] font-black px-1.5 py-0.2 rounded shrink-0 font-mono tabular-nums ${
                          isActive ? (isDayTrader ? "bg-amber-400 text-slate-950" : "bg-cyan-400 text-slate-950") : "bg-[#1b2639] text-slate-300"
                        }`}>
                          {count}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-1 leading-relaxed font-sans line-clamp-2">{tab.desc}</p>
                    </div>
                    <span className="text-[9px] px-2 py-0.5 rounded font-bold uppercase tracking-wider bg-[#1e293b] text-slate-300 self-start mt-2">
                      {tab.badge}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Loading Skeleton */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((idx) => (
              <div key={idx} className="bg-[#0e131d] border border-[#1b2434] rounded-xl p-5 animate-pulse h-72 flex flex-col justify-between">
                <div className="space-y-3">
                  <div className="h-4 bg-slate-800 rounded w-1/3"></div>
                  <div className="h-3 bg-slate-800 rounded w-2/3"></div>
                  <div className="h-14 bg-slate-900 rounded"></div>
                </div>
                <div className="h-8 bg-slate-800 rounded"></div>
              </div>
            ))}
          </div>
        )}

        {/* Candidate Cards Grid */}
        {!loading && displayGems.length === 0 && (
          <div className="p-8 text-center bg-[#0e131d] border border-[#1b2434] rounded-2xl max-w-xl mx-auto my-6">
            <span className="text-4xl block mb-3">🔍</span>
            <h3 className="text-base font-bold text-white">
              {activeCustomQuery ? `No matches found for "${activeCustomQuery}"` : `No active candidates in "${selectedFilter}"`}
            </h3>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              {activeCustomQuery
                ? "Try scanning common US market leaders or click a suggested sector basket below:"
                : "Try selecting 'All Setups' or switching between Day Trader and Long Term lenses."}
            </p>

            {activeCustomQuery && (
              <div className="flex flex-wrap items-center justify-center gap-2 mt-4">
                {["NVDA, TSLA, PLTR", "AAPL, MSFT, AMD", "LNTH, CPRX, ISRG", "COIN, MARA, MSTR"].map((basket) => (
                  <button
                    key={basket}
                    type="button"
                    onClick={() => {
                      setCustomTickerInput(basket);
                      setActiveCustomQuery(basket);
                    }}
                    className="px-2.5 py-1 bg-[#141b29] hover:bg-[#1f293d] border border-[#233249] text-cyan-300 rounded-lg text-[11px] font-bold transition-all"
                  >
                    + {basket}
                  </button>
                ))}
              </div>
            )}

            <div className="flex items-center justify-center gap-3 mt-5">
              <button
                type="button"
                onClick={() => setSelectedFilter("all")}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-bold transition-all shadow"
              >
                View All Setups
              </button>
              {activeCustomQuery && (
                <button
                  type="button"
                  onClick={handleClearCustomQuery}
                  className="px-4 py-2 bg-[#1b2434] hover:bg-[#263349] text-slate-200 rounded-xl text-xs font-bold transition-all"
                >
                  Reset to 60-Asset Catalog
                </button>
              )}
            </div>
          </div>
        )}

        {!loading && displayGems.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {displayGems.map((gem) => {
              const statusBg =
                gem.executionStatus === "IN_BUY_ZONE"
                  ? "bg-emerald-950/80 border-emerald-500/80 text-emerald-300"
                  : gem.executionStatus === "APPROACHING_TARGET"
                  ? "bg-amber-950/80 border-amber-500/80 text-amber-300"
                  : gem.executionStatus === "STOPPED_OUT"
                  ? "bg-rose-950/80 border-rose-500/80 text-rose-300"
                  : "bg-cyan-950/80 border-cyan-500/80 text-cyan-300";

              return (
                <div
                  key={gem.symbol}
                  className={`bg-[#0e131d] border rounded-xl p-4 shadow-xl flex flex-col justify-between transition-all hover:bg-[#111724] ${
                    gem.executionStatus === "IN_BUY_ZONE" ? "border-emerald-900/60 hover:border-emerald-500/50" : "border-[#1b2434] hover:border-cyan-500/40"
                  }`}
                >
                  {/* Card Top Header */}
                  <div>
                    <div className="flex items-start justify-between gap-2 border-b border-[#162030] pb-3">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-lg font-black text-white">{gem.symbol}</span>
                          <span className="text-xs font-bold text-slate-300 tabular-nums">
                            ${gem.currentPrice?.toFixed(2)}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">{gem.companyName}</p>
                      </div>
                      <div className="text-right space-y-1">
                        <button
                          type="button"
                          onClick={() => {
                            if (gem.executionStatus === "IN_BUY_ZONE") handleSelectFilter("in_buy_zone");
                            else if (gem.executionStatus === "APPROACHING_TARGET") handleSelectFilter("approaching_target");
                          }}
                          className={`text-[11px] font-bold px-2 py-0.5 rounded border inline-block cursor-pointer transition-transform hover:scale-105 ${statusBg}`}
                          title="Click to filter screener to this execution status"
                        >
                          {gem.statusLabel}
                        </button>
                        {gem.confluenceScore && (
                          <div className="text-[10px] text-cyan-400 font-bold">
                            ⭐ {gem.confluenceScore}% Confluence
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Optimal Trade Execution Level Ladder */}
                    <div
                      onClick={() => setSizerGem(gem)}
                      className="my-3 bg-[#080c14] p-3 rounded-lg border border-[#192334] space-y-2 cursor-pointer hover:border-cyan-500/50 transition-colors group/ladder"
                      title="Click to calculate exact position sizing for this optimal buy zone"
                    >
                      <div className="flex items-center justify-between text-[11px] pb-1 border-b border-[#141b28]">
                        <span className="text-slate-400 font-bold group-hover/ladder:text-cyan-300 transition-colors">
                          🎯 Optimal Buy Zone <span className="text-[9px] text-slate-500 font-normal">(Click to Size)</span>
                        </span>
                        <span className="text-emerald-400 font-black tabular-nums">
                          ${gem.optimalEntryMin?.toFixed(2)} – ${gem.optimalEntryMax?.toFixed(2)}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
                        <div className="bg-[#110d0f] p-1.5 rounded border border-rose-950">
                          <span className="text-rose-400 block font-semibold">🛑 Stop Loss</span>
                          <span className="text-rose-200 font-bold tabular-nums">
                            ${gem.stopLoss?.toFixed(2)} ({gem.stopLossPct}%)
                          </span>
                        </div>
                        <div className="bg-[#0b1414] p-1.5 rounded border border-emerald-950">
                          <span className="text-emerald-400 block font-semibold">🎯 Target 1</span>
                          <span className="text-emerald-200 font-bold tabular-nums">
                            ${gem.takeProfit1?.toFixed(2)} (+{gem.takeProfit1Pct}%)
                          </span>
                        </div>
                        <div className="bg-[#14120a] p-1.5 rounded border border-amber-950">
                          <span className="text-amber-400 block font-semibold">⚖️ Risk:Reward</span>
                          <span className="text-amber-200 font-black tabular-nums">
                            {gem.riskRewardRatio}:1 R:R
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Fundamental / Technical Dual-Horizon Thesis */}
                    <div className="space-y-1.5 text-xs">
                      <div>
                        <span className={`text-[10px] font-bold uppercase tracking-wider block ${isDayTrader ? "text-amber-400" : "text-cyan-400"}`}>
                          📐 {isDayTrader ? "Day Trade Scalp Setup:" : "Setup Pattern:"}
                        </span>
                        <p className="text-slate-300 leading-relaxed text-[11px]">
                          {isDayTrader ? gem.dayTraderSetup : gem.setupPattern}
                        </p>
                      </div>

                      {isDayTrader ? (
                        <div className="bg-[#131109] p-2 rounded border border-amber-950/60 flex items-center justify-between text-[10px]">
                          <div>
                            <span className="text-slate-500 block">ATR 14:</span>
                            <span className="text-amber-300 font-bold">{gem.atr14}</span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">RVOL:</span>
                            <span className="text-amber-300 font-bold">{gem.rvol}</span>
                          </div>
                          <div>
                            <span className="text-slate-500 block">Short Float:</span>
                            <span className="text-rose-400 font-bold">{gem.shortFloat}</span>
                          </div>
                        </div>
                      ) : (
                        <div>
                          <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider block">
                            🚀 Catalyst & Growth:
                          </span>
                          <p className="text-slate-400 leading-relaxed text-[11px]">{gem.catalyst}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Card Footer: Action linking to Terminal, Position Sizer and Alerts */}
                  <div className="mt-4 pt-3 border-t border-[#162030] flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => setSizerGem(gem)}
                        className="px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all active:scale-[0.96] border bg-[#111726] hover:bg-slate-800 border-[#223149] text-slate-300 flex items-center gap-1 shadow"
                      >
                        <span>⚖️</span>
                        <span>Size</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => setAlertGem(gem)}
                        className="px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all active:scale-[0.96] border bg-[#18140c] hover:bg-amber-950 border-amber-900/60 text-amber-300 flex items-center gap-1 shadow"
                      >
                        <span>🔔</span>
                        <span>Alert</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          const res = addPortfolioPosition({
                            symbol: gem.symbol,
                            name: gem.companyName,
                            shares: Math.max(1, Math.round(2500 / (gem.currentPrice || 100))),
                            entryPrice: gem.currentPrice || 100,
                            currentPrice: gem.currentPrice || 100,
                            targetPrice: gem.takeProfit1,
                            stopLossPrice: gem.stopLoss,
                          });
                          setLoggedGemSymbol(`${gem.symbol}: ${res.isDuplicate ? "Already In Portfolio" : "Logged!"}`);
                          setTimeout(() => setLoggedGemSymbol(null), 3000);
                        }}
                        className="px-2.5 py-1.5 rounded-lg text-xs font-bold transition-all active:scale-[0.96] border bg-indigo-600/20 hover:bg-indigo-500 hover:text-slate-950 border-indigo-500/40 text-indigo-300 flex items-center gap-1 shadow"
                        title="Log directly to Paper Portfolio"
                      >
                        <span>💼</span>
                        <span>{loggedGemSymbol && loggedGemSymbol.startsWith(gem.symbol) ? loggedGemSymbol.split(":")[1] : "Log"}</span>
                      </button>
                    </div>

                    <Link
                      href={`/?symbol=${gem.symbol}`}
                      className="px-3 py-1.5 rounded-lg text-xs font-bold transition-all active:scale-[0.96] border bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/50 text-cyan-300 flex items-center gap-1 shadow"
                    >
                      <span>Analyze in Terminal</span>
                      <span>→</span>
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Interactive Position Sizer Modal */}
      {sizerGem && (
        <PositionSizerModal
          isOpen={!!sizerGem}
          onClose={() => setSizerGem(null)}
          symbol={sizerGem.symbol}
          entryPrice={sizerGem.currentPrice || 100}
          stopLoss={sizerGem.stopLoss || (sizerGem.currentPrice || 100) * 0.95}
          takeProfit1={sizerGem.takeProfit1 || (sizerGem.currentPrice || 100) * 1.05}
          riskRewardRatio={sizerGem.riskRewardRatio || 2.5}
        />
      )}

      {/* Interactive Alert Trigger Modal */}
      {alertGem && (
        <AlertTriggerModal
          isOpen={!!alertGem}
          onClose={() => setAlertGem(null)}
          symbol={alertGem.symbol}
          currentPrice={alertGem.currentPrice || 100}
          optimalEntryMin={alertGem.optimalEntryMin || (alertGem.currentPrice || 100) * 0.97}
          optimalEntryMax={alertGem.optimalEntryMax || (alertGem.currentPrice || 100)}
          stopLoss={alertGem.stopLoss || (alertGem.currentPrice || 100) * 0.95}
          takeProfit1={alertGem.takeProfit1 || (alertGem.currentPrice || 100) * 1.05}
        />
      )}
    </main>
  );
}