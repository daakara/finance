"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ExecutiveGlobalSearch from "./ExecutiveGlobalSearch";

export interface ExecutiveIntelligenceNavProps {
  activeTab?: string;
  badgeText?: string;
}

const NAV_LINKS = [
  {
    href: "/committee-intelligence",
    label: "Scorecards",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
        <path d="M6 6h10M6 10h10M6 14h6" />
      </svg>
    ),
  },
  {
    href: "/decision-explorer",
    label: "Decisions",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <polyline points="12 6 12 12 16 14" />
      </svg>
    ),
  },
  {
    href: "/dissent-explorer",
    label: "Dissents",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    ),
  },
  {
    href: "/committee-network",
    label: "Network",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="6" cy="6" r="3" />
        <circle cx="18" cy="6" r="3" />
        <circle cx="18" cy="18" r="3" />
        <circle cx="6" cy="18" r="3" />
        <line x1="8.5" y1="7.5" x2="15.5" y2="16.5" />
        <line x1="6" y1="9" x2="6" y2="15" />
      </svg>
    ),
  },
  {
    href: "/audit-explorer",
    label: "Audit Explorer",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 21l-4.35-4.35M19 11a8 8 0 1 1-16 0 8 8 0 0 1 16 0z" />
      </svg>
    ),
  },
  {
    href: "/learning-intelligence",
    label: "Learning",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
        <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
      </svg>
    ),
  },
  {
    href: "/risks-and-groupthink",
    label: "Risks & Groupthink",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  },
  {
    href: "/coaching-intelligence",
    label: "Coaching",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
      </svg>
    ),
  },
  {
    href: "/oos",
    label: "Operating System",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
      </svg>
    ),
  },
  {
    href: "/optimization-intelligence",
    label: "Optimization",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
  },
  {
    href: "/resilience-intelligence",
    label: "Resilience",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="m9 12 2 2 4-4" />
      </svg>
    ),
  },
  {
    href: "/governance-center",
    label: "Governance Center",
    icon: (
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      </svg>
    ),
  },
];

export default function ExecutiveIntelligenceNav({ badgeText = "13/13 RESILIENCE GATES CERTIFIED" }: ExecutiveIntelligenceNavProps) {
  const pathname = usePathname();

  return (
    <header className="border-b border-[#243044] bg-[#0c1017]/90 backdrop-blur sticky top-14 z-40">
      <div className="max-w-[1750px] mx-auto px-4 sm:px-6 py-2 flex flex-col md:flex-row items-start md:items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 text-[10px] font-mono tracking-wider font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            <span>PHASE 31-M8</span>
          </div>
          <h1 className="text-sm font-mono font-bold text-slate-100 tracking-tight">
            EXECUTIVE DECISION INTELLIGENCE
          </h1>
          <span className="text-[11px] text-slate-400 font-mono hidden lg:inline">
            | Organizational Decision OS
          </span>
        </div>

        {/* Links */}
        <nav aria-label="Executive Intelligence Sub-navigation" className="flex items-center flex-wrap gap-1 font-mono text-xs">
          {NAV_LINKS.map((item) => {
            const isActive = pathname === item.href || (item.href !== "/committee-intelligence" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-1 px-2.5 py-1 rounded transition-colors focus-visible:ring-1 focus-visible:ring-cyan-400 focus-visible:outline-none ${
                  isActive
                    ? "bg-[#1f2c42] text-cyan-400 font-semibold border border-cyan-500/30 shadow-sm shadow-cyan-950/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-[#162032]"
                }`}
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Global Search Command Palette */}
        <div className="flex items-center">
          <ExecutiveGlobalSearch />
        </div>

        {/* Badges */}
        <div className="hidden xl:flex items-center space-x-2 shrink-0 font-mono text-[10px]">
          <div className="px-2 py-0.5 rounded bg-[#162032] border border-[#243044] text-slate-300">
            ODEI Floor: <span className="text-emerald-400 font-semibold">&ge;80.0</span>
          </div>
          <div className="px-2 py-0.5 rounded bg-[#162032] border border-[#243044] text-slate-300">
            Replay: <span className="text-cyan-400 font-semibold">100/100 (0 Drift)</span>
          </div>
          <div className="px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-500/40 text-emerald-400 font-semibold">
            {badgeText}
          </div>
        </div>
      </div>
    </header>
  );
}
