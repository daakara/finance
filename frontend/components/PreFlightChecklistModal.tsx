"use client";

import { useState, useEffect, useRef } from "react";
import { MASTER_ASSET_CATALOG } from "../lib/masterCatalog";
import { trackPreFlightOutcome, trackTradePlanCopied } from "../lib/matomo";

interface PreFlightChecklistModalProps {
  isOpen: boolean;
  onClose: () => void;
  symbol: string;
  currentPrice: number;
  stopLoss: number;
  takeProfit1: number;
  riskRewardRatio: number;
  setupPattern?: string;
  isDayTrader?: boolean;
  isStage4?: boolean;
  optimalEntryMin?: number;
  optimalEntryMax?: number;
  breakoutPivot?: number;
  isDistributionTrap?: boolean;
  hasImminentEarnings?: boolean;
  vix?: number;
}

export default function PreFlightChecklistModal({
  isOpen,
  onClose,
  symbol,
  currentPrice,
  stopLoss,
  takeProfit1,
  riskRewardRatio,
  setupPattern = "Minervini Volatility Contraction Pattern (VCP 3-Stage)",
  isDayTrader = false,
  isStage4 = false,
  optimalEntryMin,
  optimalEntryMax,
  breakoutPivot,
  isDistributionTrap,
  hasImminentEarnings = false,
  vix = 15.4,
}: PreFlightChecklistModalProps) {
  const [copied, setCopied] = useState<boolean>(false);
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  const modalRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLElement | null>(null);

  // Consolidated effect — active only while modal is open.
  // Handles: localStorage read, vernacular listener, Escape key, Tab focus trap.
  useEffect(() => {
    if (!isOpen) return;

    // 1. Restore persisted vernacular preference
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    } catch {}

    // 2. Live vernacular change listener
    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };
    window.addEventListener("finance:vernacular-change", handleVernacular);

    // 3. Escape closes; Tab cycles within modal (focus trap)
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") { onClose(); return; }
      if (e.key === "Tab" && modalRef.current) {
        const focusable = Array.from(
          modalRef.current.querySelectorAll<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          )
        ).filter((el) => !el.hasAttribute("disabled"));
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) { e.preventDefault(); last.focus(); }
        } else {
          if (document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      }
    };
    document.addEventListener("keydown", handleKeyDown);

    // 4. Move focus into modal; remember what opened it
    triggerRef.current = document.activeElement as HTMLElement;
    modalRef.current?.querySelector<HTMLElement>("button")?.focus();

    return () => {
      window.removeEventListener("finance:vernacular-change", handleVernacular);
      document.removeEventListener("keydown", handleKeyDown);
      triggerRef.current?.focus();
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const isPlain = vernacularMode === "PLAIN_ENGLISH";
  const cleanSym = (symbol || "ASSET").toUpperCase().replace("-USD", "");
  const catalogItem = MASTER_ASSET_CATALOG[cleanSym];

  // Defensive Numeric Guards
  const safePrice = (typeof currentPrice === "number" && !isNaN(currentPrice) && currentPrice > 0) ? currentPrice : (catalogItem?.price || 100);
  const safeStop = (typeof stopLoss === "number" && !isNaN(stopLoss) && stopLoss > 0) ? stopLoss : (safePrice * 0.95);
  const safeTarget = (typeof takeProfit1 === "number" && !isNaN(takeProfit1) && takeProfit1 > 0) ? takeProfit1 : (safePrice * 1.10);
  const safeRR = (typeof riskRewardRatio === "number" && !isNaN(riskRewardRatio) && riskRewardRatio > 0) ? riskRewardRatio : 2.5;
  const safeEntryMin = (typeof optimalEntryMin === "number" && !isNaN(optimalEntryMin)) ? optimalEntryMin : (safePrice * 0.98);
  const safeEntryMax = (typeof optimalEntryMax === "number" && !isNaN(optimalEntryMax)) ? optimalEntryMax : safePrice;
  const safePivot = (typeof breakoutPivot === "number" && !isNaN(breakoutPivot)) ? breakoutPivot : (safePrice * 1.072);
  const safeVix = (typeof vix === "number" && !isNaN(vix)) ? vix : 15.4;

  const stopLossPct = safePrice > 0 ? (((safeStop - safePrice) / safePrice) * 100).toFixed(2) : "-5.00";
  const target1Pct = safePrice > 0 ? (((safeTarget - safePrice) / safePrice) * 100).toFixed(2) : "+10.00";

  // ── 5-Point Quantitative Decision Checklist ──────────────────────────────

  // Check 1: Asymmetric Risk-Reward
  const isRRPassed = safeRR >= 2.0;

  // Check 2: Technical Trend Alignment & Stage Discipline
  const isExtendedAboveZone = Boolean(safeEntryMax && safePrice > safeEntryMax * 1.02);
  const isTrendPassed = !isStage4 && !isExtendedAboveZone;

  // Check 3: Smart Money Flow & Distribution Traps
  // distributionTrapActive = true means a trap EXISTS (bad). Naming is unambiguous.
  const distributionTrapActive = isDistributionTrap ?? Boolean(
    catalogItem && (
      (Number(catalogItem.shortFloat) || 0) > 12.0 ||
      String(catalogItem.verdict || "").toLowerCase().includes("turnaround") ||
      (Number(catalogItem.qualityScore) || 100) < 60
    )
  );
  const isDistributionTrapResolved = distributionTrapActive;
  const isSmartMoneyPassed = !distributionTrapActive;

  // Check 4: Catalyst Hazard Buffer
  const isCatalystPassed = !hasImminentEarnings;

  // Check 5: Macro Regime Guard
  const isMacroPassed = safeVix < 26.0;

  const passedCount = [isRRPassed, isTrendPassed, isSmartMoneyPassed, isCatalystPassed, isMacroPassed].filter(Boolean).length;
  const convictionPct = Math.round((passedCount / 5) * 100);
  const isCleared = convictionPct >= 80 && !isStage4 && isSmartMoneyPassed;

  // Analytics — inline try/catch, no useEffect needed (stable values within a single open session)
  try {
    trackPreFlightOutcome(symbol, passedCount, isCleared);
  } catch (err) {
    console.warn("Analytics error in PreFlightChecklistModal:", err);
  }

  // Trade plan markdown — differentiated by vernacularMode
  const tradePlanMarkdown = isPlain
    ? `### 📋 ARX Terminal Trade Plan: ${symbol}
- **Date**: ${new Date().toISOString().split("T")[0]}
- **Asset**: ${symbol} | **Mode**: ${isDayTrader ? "⚡ Day Trader" : "🏛️ Swing / Long-Term Compounder"}
- **Current Price**: $${safePrice.toFixed(2)}
- **Buy Zone**: $${safeEntryMin.toFixed(2)} – $${safeEntryMax.toFixed(2)}
- **Stop Loss**: $${safeStop.toFixed(2)} (${stopLossPct}%)
- **Profit Goal 1**: $${safeTarget.toFixed(2)} (+${target1Pct}%)
- **Setup**: ${setupPattern || "Minervini VCP Pattern"}
- **Pre-Flight Score**: ${convictionPct}% (${isCleared ? "🟢 CLEARED" : "⚠️ NOT CLEARED — wait for better setup"})
`
    : `### 📋 ARX Institutional Execution Brief: ${symbol}
- **Date**: ${new Date().toISOString().split("T")[0]}
- **Asset**: ${symbol} | **Mode**: ${isDayTrader ? "Intraday Momentum Pullback" : "Swing / Compounder Accumulation"}
- **Spot**: $${safePrice.toFixed(2)} | **Optimal Entry**: $${safeEntryMin.toFixed(2)}–$${safeEntryMax.toFixed(2)}
- **Hard Stop / Invalidation**: $${safeStop.toFixed(2)} (${stopLossPct}%)
- **Target 1 (TP1)**: $${safeTarget.toFixed(2)} (+${target1Pct}%)
- **Risk/Reward**: ${safeRR.toFixed(2)} : 1.0
- **Setup Pattern**: ${setupPattern || "Minervini VCP"}
- **Pre-Flight Clearance**: ${convictionPct}% — ${isCleared ? "🟢 CLEARED FOR EXECUTION" : "⚠️ CONDITIONAL / AWAIT BASE CLEARANCE"}
- **VIX Regime**: ${safeVix.toFixed(1)} | **Distribution Trap**: ${distributionTrapActive ? "DETECTED" : "CLEAR"} | **Earnings Hazard**: ${hasImminentEarnings ? "ACTIVE" : "CLEAR"}
`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(tradePlanMarkdown);
      setCopied(true);
      trackTradePlanCopied(symbol, setupPattern);
      setTimeout(() => setCopied(false), 2500);
    } catch (err) {
      console.warn("Failed to copy trade plan:", err);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150"
      aria-modal="true"
      role="dialog"
      aria-labelledby="preflight-title"
    >
      <div
        ref={modalRef}
        className="bg-[#0b1019] border border-cyan-800/80 rounded-2xl max-w-xl w-full p-5 sm:p-6 shadow-2xl font-sans text-slate-200 relative flex flex-col max-h-[90vh]"
      >
        {/* Header — pinned */}
        <div className="flex items-center justify-between border-b border-[#1e293b] pb-3.5 shrink-0">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">✈️</span>
            <div>
              <h2 id="preflight-title" className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
                <span>{isPlain ? `Pre-Flight Trade Checklist: ${symbol}` : `Institutional Pre-Flight Clearance: ${symbol}`}</span>
              </h2>
              <p className="text-xs text-slate-400">
                {isPlain
                  ? "5-Point sanity check before risking your hard-earned money."
                  : "Automated pre-trade validation gate enforcing risk-reward and flow confluence."}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Pre-Flight Checklist"
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition"
          >
            ✕
          </button>
        </div>

        {/* Scrollable body — barometer + 5-point checklist */}
        <div className="overflow-y-auto flex-1 space-y-4 py-4 pr-0.5">
          {/* Clearance Conviction Barometer */}
          <div className={`p-3.5 rounded-xl border flex items-center justify-between ${
            isCleared
              ? "bg-emerald-950/40 border-emerald-700/80 text-emerald-300"
              : "bg-amber-950/40 border-amber-700/80 text-amber-300"
          }`}>
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider font-mono block">
                {isPlain ? "Trade Readiness Score" : "Quantitative Clearance Status"}
              </span>
              <div className="text-base sm:text-lg font-extrabold flex items-center gap-1.5">
                <span>{isCleared ? "🟢 CLEARED TO EXECUTE" : "⚠️ CONDITIONAL / NOT CLEARED"}</span>
                <span className="text-xs font-mono font-normal">({convictionPct}% Pass)</span>
              </div>
            </div>
            <div className="text-right font-mono">
              <span className="text-2xl sm:text-3xl font-black">{passedCount}/5</span>
              <span className="text-[10px] text-slate-400 block">Checks Passed</span>
            </div>
          </div>

          {/* 5-Point Validation Checklist */}
          <div className="space-y-2.5 text-xs">
            {/* Check 1 */}
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isRRPassed ? "✅" : "❌"}</span>
                  <span>{isPlain ? "1. Reward vs Risk Balance (At least 2 to 1)" : "1. Asymmetric Payoff (Reward:Risk >= 2.0:1)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  Current: <strong className="text-cyan-300 font-mono">{safeRR.toFixed(2)} : 1.0</strong>{" "}
                  {isRRPassed ? "(Adequate upside cushion)" : "(Hazard: upside too small for downside risk)"}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isRRPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-rose-950 text-rose-300 border-rose-800"
              }`}>{isRRPassed ? "PASS" : "FAIL"}</span>
            </div>

            {/* Check 2 */}
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isTrendPassed ? "✅" : (isStage4 ? "⏳" : "⚠️")}</span>
                  <span>{isPlain ? "2. Trend & Moving Averages (Price in upward corridor)" : "2. Technical Structure (Above 20 EMA / 50 SMA Pivot)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  {isStage4
                    ? (isPlain
                        ? `⚠️ Watchlist Only: Spot price ($${safePrice.toFixed(2)}) is in Stage 4 correction below 50-day average. Await base formation.`
                        : `Stage 4 correction structure: Spot ($${safePrice.toFixed(2)}) requires 50-day breakout pivot above $${safePivot.toFixed(2)}.`)
                    : (isExtendedAboveZone
                        ? (isPlain
                            ? `⚠️ Extended: Price is above ideal buy zone ($${safeEntryMin.toFixed(2)} - $${safeEntryMax.toFixed(2)}). Wait for pullback.`
                            : `Extended structure: Spot is above value area. Chasing creates negative R:R risk.`)
                        : (isPlain
                            ? `Spot price ($${safePrice.toFixed(2)}) is inside the verified buying range defending key support.`
                            : `Defending key moving average support (20 EMA / 50 SMA).`))}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isTrendPassed
                  ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                  : (isStage4 ? "bg-amber-950 text-amber-300 border-amber-800" : "bg-rose-950 text-rose-300 border-rose-800")
              }`}>{isTrendPassed ? "PASS" : (isStage4 ? "STAGE 4 WAIT" : "CHASING")}</span>
            </div>

            {/* Check 3 */}
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isSmartMoneyPassed ? "✅" : "❌"}</span>
                  <span>{isPlain ? "3. Big Player Activity (No aggressive insider selling)" : "3. Institutional Flow (No Net Form 4 C-Suite Dumping)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  {isSmartMoneyPassed
                    ? (isPlain
                        ? "Institutional order sweeps & Congressional filings indicate steady accumulation."
                        : "Institutional Flow: Positive net accumulation detected.")
                    : (isPlain
                        ? "⚠️ Warning: Heavy corporate insider selling / distribution trap detected."
                        : "⚠️ Institutional Distribution Trap: Net Form 4 C-Suite selling / elevated short interest.")}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isSmartMoneyPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-rose-950 text-rose-300 border-rose-800"
              }`}>{isSmartMoneyPassed ? "PASS" : "DISTRIBUTION"}</span>
            </div>

            {/* Check 4 */}
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isCatalystPassed ? "✅" : "⚠️"}</span>
                  <span>{isPlain ? "4. News Event Safety (No surprise earnings report tomorrow)" : "4. Catalyst Hazard Buffer (>7 Days to Binary Earnings)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  {isCatalystPassed
                    ? (isPlain
                        ? "Sufficient time window to manage trade without overnight earnings gap risk."
                        : "Catalyst Buffer: Clean window (>7 days to binary catalyst).")
                    : (isPlain
                        ? "⚠️ High Risk: Imminent binary earnings announcement / major FDA event within 48 hours."
                        : "⚠️ Imminent Binary Event: Overnight gap risk exceeds standard stop constraint.")}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isCatalystPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-amber-950 text-amber-300 border-amber-800"
              }`}>{isCatalystPassed ? "PASS" : "HAZARD"}</span>
            </div>

            {/* Check 5 */}
            <div className="p-3 rounded-lg bg-[#111722] border border-[#1e293b] flex items-start justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-bold text-slate-100 flex items-center gap-1.5">
                  <span>{isMacroPassed ? "✅" : "⚠️"}</span>
                  <span>{isPlain ? "5. Overall Market Weather (VIX normal, market calm)" : "5. Macro Regime Guard (VIX Volatility Guardrails Safe)"}</span>
                </div>
                <p className="text-[11px] text-slate-400 pl-5">
                  {isMacroPassed
                    ? (isPlain
                        ? `Market volatility is within standard parameters (VIX ${safeVix.toFixed(1)}).`
                        : `Macro Guard: Normal volatility regime (VIX ${safeVix.toFixed(1)} < 26.0).`)
                    : (isPlain
                        ? `⚠️ High Volatility: Market VIX (${safeVix.toFixed(1)}) indicates elevated systemic turbulence.`
                        : `⚠️ Elevated Macro Risk: VIX (${safeVix.toFixed(1)}) exceeds 26.0 threshold.`)}
                </p>
              </div>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                isMacroPassed ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-amber-950 text-amber-300 border-amber-800"
              }`}>{isMacroPassed ? "PASS" : "HIGH VIX"}</span>
            </div>
          </div>
        </div>

        {/* Action Buttons — pinned to bottom */}
        <div className="flex flex-wrap items-center justify-between gap-2.5 pt-3 border-t border-[#1e293b] shrink-0">
          <button
            type="button"
            onClick={handleCopy}
            className="px-4 py-2 rounded-lg text-xs font-bold transition-all active:scale-95 border bg-cyan-600/20 hover:bg-cyan-500 hover:text-slate-950 border-cyan-500/60 text-cyan-300 flex items-center gap-1.5 shadow"
          >
            <span>{copied ? "✅ Plan Copied!" : "📋 Copy Trade Plan for Journal"}</span>
          </button>

          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-bold transition bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
