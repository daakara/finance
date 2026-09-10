"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { MASTER_ASSET_CATALOG, MasterAssetEntry } from "../lib/masterCatalog";
import { SpotPriceRegistry, fetchTacticalSetups } from "../lib/api";
import { getPersistedMarketSnapshot } from "../lib/marketDatabase";
import { TradeSetupSpec } from "../lib/simulation/governorSizingEngine";
import MiniSparkline from "./MiniSparkline";

interface CommandItem {
  id: string;
  category: "HUB" | "ACTION" | "TICKET" | "ASSET" | "POLITICIAN" | "GOVERNOR" | "NAVIGATION";
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
  const [tacticalSetups, setTacticalSetups] = useState<TradeSetupSpec[]>([]);
  const [setupsLoading, setSetupsLoading] = useState(false);
  const [setupsError, setSetupsError] = useState<string | null>(null);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Fetch dynamic setups whenever modal opens
  useEffect(() => {
    if (!isOpen) return;
    let isMounted = true;
    setSetupsLoading(true);
    setSetupsError(null);
    fetchTacticalSetups()
      .then((data) => {
        if (!isMounted) return;
        setTacticalSetups(data || []);
        setSetupsLoading(false);
      })
      .catch((err) => {
        if (!isMounted) return;
        console.warn("Failed to load dynamic setups for CommandPalette:", err);
        setSetupsError(err?.message || "Failed to load setups");
        setSetupsLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [isOpen]);

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

    // 1. Flagship Terminal Hubs (INV-OI115-P)
    items.push({
      id: "hub-radar",
      category: "HUB",
      title: "Radar Confluence Screener",
      subtitle: "Multi-factor VCP, Smart Money, and GARP screen",
      badge: "Hub · Level 0",
      icon: "📡",
      action: () => {
        router.push("/radar");
        onClose();
      },
    });

    items.push({
      id: "hub-setups",
      category: "HUB",
      title: "Tactical Setups & Execution Ticket",
      subtitle: "Asymmetric trade ladder & Governor sizing",
      badge: "Hub · Level 0",
      icon: "⚡",
      action: () => {
        router.push("/setups");
        onClose();
      },
    });

    items.push({
      id: "hub-portfolio",
      category: "HUB",
      title: "Portfolio Risk Heat Map & Stop Loss Floors",
      subtitle: "Exposure concentration & exit alerts",
      badge: "Hub · Level 0",
      icon: "💼",
      action: () => {
        router.push("/portfolio");
        onClose();
      },
    });

    items.push({
      id: "hub-journal",
      category: "HUB",
      title: "Execution Discipline & Brier Calibration Journal",
      subtitle: "Rule adherence & anti-tilt monitor",
      badge: "Hub · Level 0",
      icon: "📖",
      action: () => {
        router.push("/journal");
        onClose();
      },
    });

    items.push({
      id: "hub-performance",
      category: "HUB",
      title: "Attribution Proof & Capital Preserved Engine",
      subtitle: "+$5,225+ counterfactual ROI",
      badge: "Hub · Level 0",
      icon: "📈",
      action: () => {
        router.push("/performance");
        onClose();
      },
    });

    items.push({
      id: "hub-research",
      category: "HUB",
      title: "Terminal & Research Intelligence",
      subtitle: "Unified market overview, multi-factor analysis & catalyst dossiers",
      badge: "Terminal",
      icon: "🖥️",
      action: () => {
        router.push("/");
        onClose();
      },
    });

    items.push({
      id: "hub-cockpit",
      category: "GOVERNOR",
      title: "Behavioral Governor & Risk Guardrails",
      subtitle: "Deep risk telemetry, sizing clamps & regime controls in Setups",
      badge: "Governor",
      icon: "🛡️",
      action: () => {
        router.push("/setups");
        onClose();
      },
    });

    items.push({
      id: "gov-risk-status",
      category: "GOVERNOR",
      title: "Behavioral Governor: View Active Sizing Clamps & Constraints",
      subtitle: "Loss streak mitigation, capital floor defense, and risk telemetry in Setups",
      badge: "-25% Clamp",
      icon: "🛡️",
      action: () => {
        router.push("/setups");
        onClose();
      },
    });

    // 2. Primary Terminal Actions & Utilities
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
      id: "action-toggle-horizon",
      category: "ACTION",
      title: "Toggle Horizon (Day Trader ⚡ ⇄ Long-Term Investor 🏛️)",
      subtitle: "Switch between intraday momentum setups and secular compounding",
      badge: "Horizon",
      icon: "⚡",
      action: () => {
        const current = localStorage.getItem("FINANCE_USER_ROLE") as "DAY_TRADER" | "LONG_TERM" | null;
        const next = current === "DAY_TRADER" ? "LONG_TERM" : "DAY_TRADER";
        try { localStorage.setItem("FINANCE_USER_ROLE", next); } catch {}
        window.dispatchEvent(new CustomEvent("finance:role-change", { detail: next }));
        onClose();
      },
    });

    items.push({
      id: "action-purge-cache",
      category: "ACTION",
      title: "Purge Cache & Re-sync Live Quotes",
      subtitle: "Clear local cache and force live WebSocket/API re-sync",
      badge: "Cache",
      icon: "🔄",
      action: () => {
        try {
          localStorage.removeItem("FINANCE_MARKET_SNAPSHOTS_V1");
          sessionStorage.clear();
          window.dispatchEvent(new CustomEvent("finance:cache-purge"));
        } catch (err) {
          console.warn("Failed to purge client cache:", err);
        }
        onClose();
      },
    });

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
        try { localStorage.setItem("ARX_VERNACULAR_MODE", next); } catch {}
        window.dispatchEvent(new CustomEvent("finance:vernacular-change", { detail: next }));
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
        try { localStorage.setItem("ARX_DENSITY_MODE", nextDensity); } catch {}
        window.dispatchEvent(new CustomEvent("finance:density-change", { detail: nextDensity }));
        onClose();
      },
    });

    // 3. Tactical Execution Tickets (Dynamic API-Backed Setups)
    if (tacticalSetups && tacticalSetups.length > 0) {
      tacticalSetups.forEach((setup) => {
        const score = typeof setup.confluenceScore === "number" ? Math.round(setup.confluenceScore) : null;
        const pattern = setup.setupName || "Breakout Setup";
        const status = setup.isActionable ? "Ready to Buy" : "Criteria Pending";
        items.push({
          id: `ticket-${setup.ticker.toLowerCase()}`,
          category: "TICKET",
          title: `${setup.ticker} — Tactical Execution Ticket`,
          subtitle: `${score !== null ? `Confluence ${score} · ` : ""}${pattern} · ${status}`,
          badge: score !== null ? `Score ${score}` : undefined,
          icon: "🎯",
          action: () => {
            router.push(`/setups?ticker=${encodeURIComponent(setup.ticker)}`);
            onClose();
          },
        });
      });
    } else if (setupsLoading) {
      items.push({
        id: "ticket-loading",
        category: "TICKET",
        title: "Scanning Live Tactical Setups...",
        subtitle: "Querying exchange tape and multi-factor confluence engine",
        badge: "Loading",
        icon: "⏳",
        action: () => {
          router.push("/setups");
          onClose();
        },
      });
    } else if (setupsError) {
      items.push({
        id: "ticket-error",
        category: "TICKET",
        title: "Tactical Setups Tape Unavailable",
        subtitle: `${setupsError} · Click to open Setups hub directly`,
        badge: "Offline",
        icon: "⚠️",
        action: () => {
          router.push("/setups");
          onClose();
        },
      });
    } else {
      items.push({
        id: "ticket-empty",
        category: "TICKET",
        title: "No Active Qualifying Setups on Tape",
        subtitle: "No assets currently meet Stage 2 VCP breakout criteria · Open Setups hub to scan all",
        badge: "0 Active",
        icon: "⚡",
        action: () => {
          router.push("/setups");
          onClose();
        },
      });
    }

    // 4. Assets from Master Catalog
    Object.values(MASTER_ASSET_CATALOG).forEach((asset) => {
      const reg = SpotPriceRegistry.get(asset.symbol);
      const snap = getPersistedMarketSnapshot(asset.symbol);
      const effectivePrice = (reg?.price && reg.price > 0)
        ? reg.price
        : (snap?.currentPrice && snap.currentPrice > 0)
        ? snap.currentPrice
        : undefined;

      items.push({
        id: `asset-${asset.symbol.toLowerCase()}`,
        category: "ASSET",
        title: `${asset.symbol} — ${asset.name}`,
        subtitle: `${asset.type} • ${asset.sector || asset.category || "Asset"}`,
        badge: asset.type,
        icon: asset.type === "Crypto" ? "🪙" : "📊",
        price: effectivePrice,
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

    // 5. Congressional & Committee Hubs
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

    return items;
  }, [router, onClose, onSelectSymbol, tacticalSetups, setupsLoading, setupsError]);

  // Filter commands by query with dynamic ticker navigation support
  const filteredCommands = useMemo(() => {
    const q = query.trim().toUpperCase();
    const isTickerQuery = /^[A-Z]{1,5}$/.test(q);

    let base = allCommands;
    if (query.trim()) {
      const qLower = query.toLowerCase().trim();
      base = allCommands.filter((cmd) => {
        return (
          cmd.title.toLowerCase().includes(qLower) ||
          cmd.subtitle.toLowerCase().includes(qLower) ||
          (cmd.badge && cmd.badge.toLowerCase().includes(qLower))
        );
      });
    }

    // If query looks like a ticker symbol, prepend explicit direct routes honoring the ticker parameter
    if (isTickerQuery && !base.some((b) => b.id === `ticket-${q.toLowerCase()}`)) {
      const dynamicTickerCommands: CommandItem[] = [
        {
          id: `dynamic-setup-${q.toLowerCase()}`,
          category: "TICKET",
          title: `${q} — Tactical Execution Ticket`,
          subtitle: `Open tactical trade ladder and behavioral sizing for ${q}`,
          badge: "Setups",
          icon: "⚡",
          action: () => {
            router.push(`/setups?ticker=${encodeURIComponent(q)}`);
            onClose();
          },
        },
        {
          id: `dynamic-research-${q.toLowerCase()}`,
          category: "ASSET",
          title: `${q} — Research & Multi-Factor Analysis`,
          subtitle: `Open fundamentals, SEC filings, and factor model for ${q}`,
          badge: "Terminal",
          icon: "🔬",
          action: () => {
            router.push(`/?symbol=${encodeURIComponent(q)}`);
            onClose();
          },
        },
      ];
      return [...dynamicTickerCommands, ...base];
    }

    return base;
  }, [allCommands, query, router, onClose]);

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
            placeholder="Jump to hub (/radar, /setups), search tickers (NVDA, GOOGL), actions..."
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
                  {cmd.category === "HUB" && (
                    <span className="text-xs text-cyan-400 font-mono font-bold shrink-0">Jump ↵</span>
                  )}
                  {cmd.category === "TICKET" && (
                    <span className="text-xs text-emerald-400 font-mono font-bold shrink-0">Order ↵</span>
                  )}
                  {cmd.category === "GOVERNOR" && (
                    <span className="text-xs text-amber-400 font-mono font-bold shrink-0">View ↵</span>
                  )}
                  {cmd.category === "POLITICIAN" && (
                    <span className="text-xs text-slate-400 font-mono font-bold shrink-0">Track ↵</span>
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
