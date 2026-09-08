"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  resolveEntityQuery,
  getSearchTelemetryLog,
} from "../../lib/telemetry/entityResolverEngine";

export interface ExecutiveGlobalSearchProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export default function ExecutiveGlobalSearch({
  isOpen: controlledIsOpen,
  onClose,
}: ExecutiveGlobalSearchProps) {
  const router = useRouter();
  const [internalOpen, setInternalOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const isModalOpen = controlledIsOpen !== undefined ? controlledIsOpen : internalOpen;

  // Global keyboard shortcut: Cmd+K / Ctrl+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setInternalOpen((prev) => !prev);
      } else if (e.key === "Escape" && isModalOpen) {
        setInternalOpen(false);
        onClose?.();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isModalOpen, onClose]);

  // Focus input when opened
  useEffect(() => {
    if (isModalOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      setQuery("");
    }
  }, [isModalOpen]);

  const resolution = useMemo(() => {
    return resolveEntityQuery(query);
  }, [query]);

  const recentSearches = useMemo(() => {
    return getSearchTelemetryLog().slice(0, 4);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, isModalOpen]);

  const handleNavigate = (route: string) => {
    setInternalOpen(false);
    onClose?.();
    router.push(route);
  };

  const QUICK_PREFIXES = ["DEC-", "OUT-", "DIS-", "COM-", "PROP-", "LRN-", "INC-", "RSK-", "GT-"];
  const SAMPLE_ENTITIES = [
    { id: "DEC-001", label: "Flow Regime Allocation", type: "DECISION" },
    { id: "OUT-001", label: "+$145k Realized Return", type: "OUTCOME" },
    { id: "DIS-001", label: "Liquidity Macro Dissent", type: "DISSENT" },
    { id: "COM-001", label: "Investment Committee", type: "COMMITTEE" },
    { id: "PROP-001", label: "Momentum Scale Objective", type: "PROPOSAL" },
    { id: "LRN-001", label: "Institutional Flow Filter", type: "LEARNING" },
    { id: "INC-201", label: "Learning Breakdown Incident", type: "INCIDENT" },
    { id: "RSK-001", label: "Suppressed Dissent Exposure", type: "RISK" },
    { id: "GT-COM-001-01", label: "Excessive Unanimity Signal", type: "GROUPTHINK" },
  ];

  if (!isModalOpen) {
    return (
      <button
        type="button"
        onClick={() => setInternalOpen(true)}
        className="flex items-center space-x-2 px-3 py-1.5 bg-[#121824] hover:bg-[#1c273a] border border-[#202d44] hover:border-cyan-500/50 rounded-lg text-xs font-mono text-slate-400 hover:text-slate-200 transition-all group"
        title="Search any decision, outcome, dissent, or committee (Ctrl+K / Cmd+K)"
      >
        <svg className="w-3.5 h-3.5 text-slate-400 group-hover:text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <span className="hidden sm:inline">Search artifacts (DEC, OUT, DIS)...</span>
        <span className="sm:hidden">Search...</span>
        <kbd className="px-1.5 py-0.5 rounded bg-[#1c273a] border border-[#2d3f5e] text-[10px] text-slate-400">
          ⌘K
        </kbd>
      </button>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/70 backdrop-blur-sm font-mono animate-in fade-in duration-150">
      <div
        className="bg-[#0e131d] border border-cyan-500/30 w-full max-w-2xl rounded-2xl shadow-2xl shadow-cyan-950/60 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div className="flex items-center border-b border-[#202d44] px-4 py-3 bg-[#131b29]">
          <svg className="w-4 h-4 text-cyan-400 mr-3 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && resolution.found && resolution.canonicalRoute) {
                handleNavigate(resolution.canonicalRoute);
              }
            }}
            placeholder="Type artifact ID (e.g. DEC-001, OUT-001, DIS-001, COM-001)..."
            className="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
          />
          <button
            type="button"
            onClick={() => {
              setInternalOpen(false);
              onClose?.();
            }}
            className="px-1.5 py-0.5 rounded bg-[#1c273a] text-slate-400 hover:text-slate-200 text-xs ml-2"
          >
            ESC
          </button>
        </div>

        {/* Quick Prefix Badges */}
        <div className="flex items-center flex-wrap gap-1.5 px-4 py-2 bg-[#0c1017] border-b border-[#202d44] text-xs">
          <span className="text-[10px] text-slate-500 mr-1">Prefixes:</span>
          {QUICK_PREFIXES.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setQuery(p)}
              className="px-2 py-0.5 rounded bg-[#162032] hover:bg-[#1f2c42] border border-[#202d44] text-[10px] text-cyan-400 transition-colors"
            >
              {p}
            </button>
          ))}
        </div>

        {/* Results / Suggestions Area */}
        <div className="p-4 max-h-80 overflow-y-auto space-y-3 text-xs">
          {/* Exact Match Found */}
          {resolution.found && resolution.canonicalRoute && (
            <div
              onClick={() => handleNavigate(resolution.canonicalRoute!)}
              className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-500/40 hover:bg-cyan-900/40 transition-colors cursor-pointer flex items-center justify-between"
            >
              <div>
                <div className="flex items-center space-x-2">
                  <span className="px-2 py-0.5 rounded bg-cyan-500 text-slate-950 font-bold text-[10px]">
                    {resolution.entityType}
                  </span>
                  <span className="font-bold text-slate-100 text-sm">{resolution.entityId}</span>
                </div>
                <p className="text-xs text-slate-300 mt-1">{resolution.title}</p>
              </div>
              <span className="text-cyan-400 text-xs font-bold flex items-center space-x-1">
                <span>Navigate</span>
                <span>&rarr;</span>
              </span>
            </div>
          )}

          {/* Error / Not Found Message */}
          {query.trim().length > 0 && !resolution.found && resolution.error && (
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-300">
              <div className="flex items-center space-x-1.5 font-bold mb-1">
                <span>&#9888;</span>
                <span>{resolution.error}</span>
              </div>
              {resolution.suggestions.length > 0 && (
                <div className="mt-2">
                  <span className="text-[10px] text-slate-400 block mb-1">Did you mean:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {resolution.suggestions.map((sug) => (
                      <button
                        key={sug}
                        type="button"
                        onClick={() => setQuery(sug)}
                        className="px-2 py-0.5 rounded bg-[#1c273a] hover:bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-xs"
                      >
                        {sug}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Default State: Sample Entities */}
          {query.trim().length === 0 && (
            <div className="space-y-2">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block">
                Direct Artifact Access
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {SAMPLE_ENTITIES.map((ent) => (
                  <button
                    key={ent.id}
                    type="button"
                    onClick={() => setQuery(ent.id)}
                    className="p-2.5 rounded-lg bg-[#121824] hover:bg-[#1a2334] border border-[#202d44] text-left transition-colors flex items-center justify-between"
                  >
                    <div>
                      <div className="flex items-center space-x-1.5">
                        <span className="px-1.5 py-0.2 rounded bg-[#1c273a] text-cyan-400 font-bold text-[9px]">
                          {ent.type}
                        </span>
                        <span className="font-bold text-slate-200">{ent.id}</span>
                      </div>
                      <span className="text-[10px] text-slate-400 mt-0.5 block truncate">
                        {ent.label}
                      </span>
                    </div>
                    <span className="text-slate-500 text-xs">&rarr;</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Recent Search Telemetry */}
          {recentSearches.length > 0 && (
            <div className="pt-2 border-t border-[#202d44]/60">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1.5">
                Recent Queries
              </span>
              <div className="flex flex-wrap gap-1.5">
                {recentSearches.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setQuery(item.query)}
                    className="px-2 py-0.5 rounded bg-[#111724] border border-[#202d44] text-[10px] text-slate-400 hover:text-slate-200"
                  >
                    {item.query} ({item.latencyMs}ms)
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="px-4 py-2 bg-[#0c1017] border-t border-[#202d44] text-[10px] text-slate-500 flex items-center justify-between">
          <span>Supported: Decision, Outcome, Dissent, Committee, Proposal</span>
          <span>1-Click Reachable (M2-Gate-01)</span>
        </div>
      </div>
    </div>
  );
}
