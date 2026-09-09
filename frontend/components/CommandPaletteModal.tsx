"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { MASTER_ASSET_CATALOG, MasterAssetEntry } from "../lib/masterCatalog";
import { SpotPriceRegistry } from "../lib/api";
import { getPersistedMarketSnapshot } from "../lib/marketDatabase";
import MiniSparkline from "./MiniSparkline";

interface CommandItem {
  id: string;
  category: "ASSET" | "POLITICIAN" | "ACTION" | "NAVIGATION" | "DECISION" | "SIGNAL" | "FORECAST" | "SCENARIO" | "JOURNAL" | "WORKBENCH" | "HUB";
  title: string;
  subtitle: string;
  badge?: string;
  icon: string;
  price?: number;
  changePct?: number;
  action: () => void;
}

interface CommandPaletteModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSymbol?: (sym: string) => void;
}

export default function CommandPaletteModal({
  isOpen,
  onClose,
  onSelectSymbol,
}: CommandPaletteModalProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Build the complete searchable command registry
  const allCommands: CommandItem[] = useMemo(() => {
    const items: CommandItem[] = [];

    // 1. Assets from Master Catalog
    Object.values(MASTER_ASSET_CATALOG).forEach((asset) => {
      const reg = SpotPriceRegistry.get(asset.symbol);
      const snap = getPersistedMarketSnapshot(asset.symbol);
      const effectivePrice = (reg?.price && reg.price > 0)
        ? reg.price
        : (snap?.currentPrice && snap.currentPrice > 0)
        ? snap.currentPrice
        : undefined;
      const effectiveChange = (reg?.changePct !== undefined)
        ? reg.changePct
        : (snap?.priceChangePct24h !== undefined)
        ? snap.priceChangePct24h
        : undefined;

      items.push({
        id: `asset-${asset.symbol}`,
        category: "ASSET",
        title: asset.symbol,
        subtitle: `${asset.name} • ${asset.sector} (${asset.category})`,
        badge: `${asset.piotroski}/9 Piotroski`,
        icon: "📈",
        price: effectivePrice,
        changePct: effectiveChange,
        action: () => {
          if (onSelectSymbol) {
            onSelectSymbol(asset.symbol);
          } else {
            router.push(`/?symbol=${asset.symbol}`);
          }
          onClose();
        },
      });
    });

    // 2. Congressional & Committee Hubs
    const politicians = [
      { slug: "nancy-pelosi", name: "Nancy Pelosi", chamber: "House", desc: "LEAPS Call Strategy & Tech Flow" },
      { slug: "dan-crenshaw", name: "Dan Crenshaw", chamber: "House", desc: "Energy & Commerce Committee Trades" },
      { slug: "tommy-tuberville", name: "Tommy Tuberville", chamber: "Senate", desc: "Armed Services & Ag Flow" },
      { slug: "ro-khanna", name: "Ro Khanna", chamber: "House", desc: "Silicon Valley Tech Committee Overlap" },
      { slug: "mitch-mcconnell", name: "Mitch McConnell", chamber: "Senate", desc: "Defense & Infrastructure Appropriations" },
    ];

    politicians.forEach((pol) => {
      items.push({
        id: `pol-${pol.slug}`,
        category: "POLITICIAN",
        title: pol.name,
        subtitle: `${pol.chamber} • ${pol.desc}`,
        badge: "STOCK Act",
        icon: "🏛️",
        action: () => {
          router.push(`/politician/${pol.slug}`);
          onClose();
        },
      });
    });

    // 3. Quick Actions
    items.push({
      id: "action-toggle-vernacular",
      category: "ACTION",
      title: "Toggle Vernacular Mode (Plain English ⚡ ⇄ Pro Quant 🏛️)",
      subtitle: "Switch explanations between approachable terms and hedge fund metrics",
      badge: "Instant",
      icon: "⚡",
      action: () => {
        const current = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
        const next = current === "PRO_QUANT" ? "PLAIN_ENGLISH" : "PRO_QUANT";
        localStorage.setItem("ARX_VERNACULAR_MODE", next);
        window.dispatchEvent(new CustomEvent("finance:vernacular-change", { detail: next }));
        onClose();
      },
    });

    items.push({
      id: "action-toggle-theme",
      category: "ACTION",
      title: "Toggle Theme (Obsidian Dark 🌑 ⇄ Paper Light 🌓)",
      subtitle: "Switch between command-center dark and financial journal light theme",
      badge: "Theme",
      icon: "🌓",
      action: () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        const nextTheme = currentTheme === "paper" ? "dark" : "paper";
        if (nextTheme === "paper") {
          document.documentElement.setAttribute("data-theme", "paper");
          localStorage.setItem("theme", "paper");
        } else {
          document.documentElement.removeAttribute("data-theme");
          localStorage.setItem("theme", "dark");
        }
        window.dispatchEvent(new CustomEvent("finance:theme-change", { detail: nextTheme }));
        onClose();
      },
    });

    items.push({
      id: "action-toggle-density",
      category: "ACTION",
      title: "Toggle Data Density Mode (Compact ⚡ ⇄ Comfortable 🖥️)",
      subtitle: "Switch between ultra-dense command center layout and spacious cards",
      badge: "Density",
      icon: "🎚️",
      action: () => {
        const currentDensity = localStorage.getItem("ARX_DENSITY_MODE") || "COMFORTABLE";
        const nextDensity = currentDensity === "COMPACT" ? "COMFORTABLE" : "COMPACT";
        localStorage.setItem("ARX_DENSITY_MODE", nextDensity);
        window.dispatchEvent(new CustomEvent("finance:density-change", { detail: nextDensity }));
        onClose();
      },
    });

    // 4. Navigation Hubs
    items.push({
      id: "nav-screener",
      category: "NAVIGATION",
      title: "Market Scanner & Gem Screener",
      subtitle: "Filter 60+ assets by Minervini VCP, Magic Formula, and Peter Lynch GARP",
      badge: "Scanner",
      icon: "💎",
      action: () => {
        router.push("/screener");
        onClose();
      },
    });

    items.push({
      id: "nav-smart-money",
      category: "NAVIGATION",
      title: "Smart Money & Dark Pool Radar",
      subtitle: "Track Congressional STOCK Act trades, SEC Form 4 sweeps, and options flow",
      badge: "Radar",
      icon: "📡",
      action: () => {
        router.push("/smart-money");
        onClose();
      },
    });

    items.push({
      id: "nav-portfolio",
      category: "NAVIGATION",
      title: "Paper Portfolio & Macro Simulator",
      subtitle: "Track simulated positions and test portfolio drawdowns under macro shocks",
      badge: "Portfolio",
      icon: "💼",
      action: () => {
        router.push("/portfolio");
        onClose();
      },
    });

    items.push({
      id: "nav-compare",
      category: "NAVIGATION",
      title: "Multi-Asset Head-to-Head Compare",
      subtitle: "Benchmark financial metrics, ROIC spreads, and beta correlations",
      badge: "Benchmark",
      icon: "📊",
      action: () => {
        router.push("/compare");
        onClose();
      },
    });

    // 5. Core Human Hubs
    items.push({
      id: "hub-today",
      category: "HUB",
      title: "Today & Execution (/today)",
      subtitle: "Triad Index (LHI 84 · HHI 89 · IAI 61), Next Best Action, and Circadian Window",
      badge: "Core Hub",
      icon: "⚡",
      action: () => {
        router.push("/today");
        onClose();
      },
    });

    items.push({
      id: "hub-future",
      category: "HUB",
      title: "Future & Scenarios (/future)",
      subtitle: "Runway Shield (14.2 Mo), 3-Year Wealth Compounding ($1.24M), and Future Paths",
      badge: "Core Hub",
      icon: "🔮",
      action: () => {
        router.push("/future");
        onClose();
      },
    });

    items.push({
      id: "hub-progress",
      category: "HUB",
      title: "Progress & Calibration (/progress)",
      subtitle: "Identity Twin, Behavioral Drift (Public Influence 68d), and Brier Calibration",
      badge: "Core Hub",
      icon: "🎯",
      action: () => {
        router.push("/progress");
        onClose();
      },
    });

    items.push({
      id: "hub-household",
      category: "HUB",
      title: "Household & Relational (/household)",
      subtitle: "Household Health Index (HHI 89), Partner Alignment (86%), and Shared Resources",
      badge: "Core Hub",
      icon: "🏡",
      action: () => {
        router.push("/household");
        onClose();
      },
    });

    // 6. Specialist Workbenches
    items.push({
      id: "wb-life-graph",
      category: "WORKBENCH",
      title: "Life Graph Workbench",
      subtitle: "Causal node topology, multi-domain ripple propagation, and systemic friction",
      badge: "Workbench",
      icon: "🕸️",
      action: () => {
        router.push("/workbench/life-graph");
        onClose();
      },
    });

    items.push({
      id: "wb-signals",
      category: "WORKBENCH",
      title: "Personal Signals Workbench",
      subtitle: "High-frequency biometrics, telemetry streams, and circadian conviction signals",
      badge: "Workbench",
      icon: "📡",
      action: () => {
        router.push("/workbench/signals");
        onClose();
      },
    });

    items.push({
      id: "wb-allocator",
      category: "WORKBENCH",
      title: "168-Hour Allocator Workbench",
      subtitle: "Time, energy, and capital envelope budgeting with calendar collision resolution",
      badge: "Workbench",
      icon: "⏳",
      action: () => {
        router.push("/workbench/allocator");
        onClose();
      },
    });

    items.push({
      id: "wb-journal",
      category: "WORKBENCH",
      title: "Decision Journal Workbench",
      subtitle: "Probabilistic prediction auditing, Brier calibration (0.18), and post-mortems",
      badge: "Workbench",
      icon: "📓",
      action: () => {
        router.push("/workbench/journal");
        onClose();
      },
    });

    items.push({
      id: "wb-simulation",
      category: "WORKBENCH",
      title: "Simulation & Trajectories Workbench",
      subtitle: "Multi-year Monte Carlo trajectories, macroeconomic stress-testing, and future states",
      badge: "Workbench",
      icon: "🎲",
      action: () => {
        router.push("/workbench/simulation");
        onClose();
      },
    });

    // 7. Decisions
    items.push({
      id: "decision-nba-01",
      category: "DECISION",
      title: "Decision: Deep Work — AI Systems Architecture RFC",
      subtitle: "45m high-cognitive block during peak morning chronotype window (+24 identity pts)",
      badge: "Decision",
      icon: "🧠",
      action: () => {
        router.push("/today");
        onClose();
      },
    });

    items.push({
      id: "decision-zone2-run",
      category: "DECISION",
      title: "Decision: Zone 2 Aerobic Recovery Run (30m)",
      subtitle: "Prevents cardiovascular fatigue and maintains autonomic nervous system HRV baseline",
      badge: "Decision",
      icon: "🏃",
      action: () => {
        router.push("/today");
        onClose();
      },
    });

    // 8. Signals
    items.push({
      id: "signal-hrv-optimal",
      category: "SIGNAL",
      title: "Signal: Autonomic Recovery Optimal (Sleep 84)",
      subtitle: "Telemetry stream confirms prime cognitive bandwidth window 09:00 - 12:30",
      badge: "Signal",
      icon: "💓",
      action: () => {
        router.push("/workbench/signals");
        onClose();
      },
    });

    items.push({
      id: "signal-telemetry-freshness",
      category: "SIGNAL",
      title: "Signal: Realtime Telemetry Connected (24 Streams)",
      subtitle: "Confidence 91% • High conviction ratio 0.82 across biometrics and market data",
      badge: "Signal",
      icon: "📶",
      action: () => {
        router.push("/workbench/signals");
        onClose();
      },
    });

    // 9. Forecasts
    items.push({
      id: "forecast-net-worth",
      category: "FORECAST",
      title: "Forecast: 3-Year Net Liquid Wealth ($840k → $1.24M)",
      subtitle: "88% confidence based on systematic equity allocation and executive compensation growth",
      badge: "Forecast",
      icon: "📈",
      action: () => {
        router.push("/future");
        onClose();
      },
    });

    items.push({
      id: "forecast-exec-scope",
      category: "FORECAST",
      title: "Forecast: Executive AI Leadership Trajectory (VP Scope)",
      subtitle: "82% confidence based on published enterprise RFCs and architecture board leadership",
      badge: "Forecast",
      icon: "🔭",
      action: () => {
        router.push("/future");
        onClose();
      },
    });

    // 10. Scenarios
    items.push({
      id: "scenario-pivot",
      category: "SCENARIO",
      title: "Scenario: Systematic AI Strategy Pivot (74% Prob)",
      subtitle: "Recommended pathway: expected net worth $1.24M, identity fulfillment 92%",
      badge: "Scenario",
      icon: "🛤️",
      action: () => {
        router.push("/future");
        onClose();
      },
    });

    items.push({
      id: "scenario-status-quo",
      category: "SCENARIO",
      title: "Scenario: Status Quo Analytics Management (18% Prob)",
      subtitle: "Low near-term friction, but compounds career obsolescence ($1.02M net worth)",
      badge: "Scenario",
      icon: "⚠️",
      action: () => {
        router.push("/future");
        onClose();
      },
    });

    // 11. Journal Entries
    items.push({
      id: "journal-calibration",
      category: "JOURNAL",
      title: "Journal: Brier Score Calibration (0.18 Score)",
      subtitle: "Audited across 42 decisions • Probabilities strictly match actual outcomes",
      badge: "Journal",
      icon: "📝",
      action: () => {
        router.push("/workbench/journal");
        onClose();
      },
    });

    items.push({
      id: "journal-remedy-drift",
      category: "JOURNAL",
      title: "Journal: Identity Drift Remedy — Public Influence",
      subtitle: "15m draft industry case note on autonomous decision engines",
      badge: "Journal",
      icon: "✍️",
      action: () => {
        router.push("/progress");
        onClose();
      },
    });

    return items;
  }, [router, onClose, onSelectSymbol]);

  // Filter commands by query
  const filteredCommands = useMemo(() => {
    if (!query.trim()) return allCommands;
    const q = query.toLowerCase().trim();
    return allCommands.filter((cmd) => {
      return (
        cmd.title.toLowerCase().includes(q) ||
        cmd.subtitle.toLowerCase().includes(q) ||
        (cmd.badge && cmd.badge.toLowerCase().includes(q))
      );
    });
  }, [allCommands, query]);

  // Reset selected index if results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredCommands.length]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % (filteredCommands.length || 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % (filteredCommands.length || 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          filteredCommands[selectedIndex].action();
        }
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    },
    [filteredCommands, selectedIndex, onClose]
  );

  // Scroll active item into view
  useEffect(() => {
    if (listRef.current) {
      const activeEl = listRef.current.children[selectedIndex] as HTMLElement;
      if (activeEl) {
        activeEl.scrollIntoView({ block: "nearest" });
      }
    }
  }, [selectedIndex]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Command Palette"
      className="fixed inset-0 z-[1200] flex items-start justify-center pt-16 sm:pt-24 px-4 bg-slate-950/80 backdrop-blur-md animate-fadeIn"
      onClick={onClose}
    >
      <div
        className="bg-[#0b1019] border border-cyan-500/40 rounded-2xl w-full max-w-2xl shadow-[0_0_50px_rgba(6,182,212,0.18)] overflow-hidden flex flex-col max-h-[80vh] animate-scaleUp"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Search Input Bar */}
        <div className="flex items-center px-4 py-3.5 border-b border-[#1b2434] bg-[#070a10]">
          <span className="text-cyan-400 text-lg mr-3">⚡</span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search tickers (NVDA, PLTR), politicians (Pelosi), or quick actions..."
            className="w-full bg-transparent text-sm sm:text-base text-white placeholder-slate-500 font-mono outline-none border-none"
            aria-label="Command search query"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="text-xs font-mono text-slate-400 hover:text-white px-2 py-1"
            >
              Clear
            </button>
          )}
          <span className="hidden sm:inline-block text-[10px] font-mono text-slate-500 border border-[#243044] px-1.5 py-0.5 rounded bg-[#0f172a]">
            ESC
          </span>
        </div>

        {/* Results List */}
        <div ref={listRef} className="overflow-y-auto p-2 space-y-1 divide-y divide-[#151c2a] flex-1">
          {filteredCommands.length === 0 ? (
            <div className="p-8 text-center text-slate-400 font-mono text-xs">
              <span className="text-2xl block mb-2">🔍</span>
              No matching assets, politicians, or commands found for &quot;{query}&quot;
            </div>
          ) : (
            filteredCommands.map((cmd, idx) => {
              const isSelected = idx === selectedIndex;

              return (
                <div
                  key={cmd.id}
                  onClick={() => cmd.action()}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`flex items-center justify-between p-3 rounded-xl cursor-pointer transition-colors ${
                    isSelected
                      ? "bg-[#142033] border border-cyan-500/50 shadow-inner"
                      : "hover:bg-[#0f1724] border border-transparent"
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="text-xl shrink-0">{cmd.icon}</span>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`font-bold font-mono text-sm truncate ${isSelected ? "text-cyan-300" : "text-white"}`}>
                          {cmd.title}
                        </span>
                        {cmd.badge && (
                          <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#162030] text-slate-400 border border-[#243044]">
                            {cmd.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 truncate mt-0.5">{cmd.subtitle}</p>
                    </div>
                  </div>

                  {/* Asset price & sparkline if applicable */}
                  {cmd.price !== undefined && (
                    <div className="flex items-center gap-3 shrink-0 text-right font-mono text-xs">
                      <MiniSparkline basePrice={cmd.price} changePct={cmd.changePct || 0} width={48} height={18} />
                      <div>
                        <div className="font-bold text-white tabular-nums">${cmd.price.toFixed(2)}</div>
                        {cmd.changePct !== undefined && (
                          <div className={`text-[10px] font-bold tabular-nums ${cmd.changePct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                            {cmd.changePct >= 0 ? "+" : ""}{cmd.changePct}%
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {cmd.category === "ACTION" && (
                    <span className="text-xs text-cyan-400 font-mono font-bold shrink-0">Run ↵</span>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer Shortcut Helper */}
        <div className="px-4 py-2 bg-[#070a10] border-t border-[#1b2434] flex items-center justify-between text-[11px] font-mono text-slate-500">
          <div className="flex items-center gap-3">
            <span>↑↓ Navigate</span>
            <span>↵ Select</span>
            <span>ESC Close</span>
          </div>
          <span className="hidden sm:inline">ARX Terminal Omnisearch</span>
        </div>
      </div>
    </div>
  );
}
