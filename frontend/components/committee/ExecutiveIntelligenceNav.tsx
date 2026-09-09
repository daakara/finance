"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ExecutiveGlobalSearch from "./ExecutiveGlobalSearch";

export interface ExecutiveIntelligenceNavProps {
  activeTab?: string;
  badgeText?: string;
}

// 4 Executive Nav Groups
const WORKSPACE_LINKS = [
  { href: "/executive-workspace", label: "Workspace" },
  { href: "/decision-inbox", label: "Inbox" },
  { href: "/intelligence-center", label: "Overview" },
  { href: "/action-center", label: "Actions" },
  { href: "/adoption-center", label: "Adoption" },
  { href: "/graph-explorer", label: "Graph" },
];

const STRATEGIC_LINKS = [
  { href: "/committee-intelligence", label: "Scorecards" },
  { href: "/learning-intelligence", label: "Learning" },
  { href: "/risks-and-groupthink", label: "Risks" },
  { href: "/simulation-intelligence", label: "Simulation" },
  { href: "/strategy-laboratory", label: "Strategy Lab" },
];

const OPERATIONAL_LINKS = [
  { href: "/governance-center", label: "Governance" },
  { href: "/resilience-intelligence", label: "Resilience" },
  { href: "/autonomous-governance", label: "Autonomy" },
  { href: "/oos", label: "OS" },
  { href: "/optimization-intelligence", label: "Optimization" },
];

const EVIDENCE_LINKS = [
  { href: "/decision-explorer", label: "Decisions" },
  { href: "/dissent-explorer", label: "Dissents" },
  { href: "/audit-explorer", label: "Audit" },
  { href: "/committee-network", label: "Network" },
  { href: "/release-dashboard", label: "Release" },
];

export default function ExecutiveIntelligenceNav({
  activeTab,
  badgeText = "PHASE 31-M16 CERTIFIED",
}: ExecutiveIntelligenceNavProps) {
  const pathname = usePathname();
  const currentPath = activeTab || pathname || "/intelligence-center";

  const isLinkActive = (href: string) => {
    if (href === "/intelligence-center") {
      return currentPath === "/intelligence-center" || currentPath === "/intelligence";
    }
    if (href === "/executive-workspace") {
      return currentPath === "/executive-workspace" || currentPath === "/workspace";
    }
    return currentPath === href || currentPath.startsWith(href + "/");
  };

  return (
    <header className="border-b border-[#24324A] bg-[#0B1220]/95 backdrop-blur-md sticky top-0 z-40">
      {/* Top Bar: Brand, Badge, Search, Executive Actions */}
      <div className="max-w-[1600px] mx-auto px-4 py-2.5 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link href="/intelligence-center" className="flex items-center gap-2 group">
            <div className="w-7 h-7 rounded-lg bg-cyan-600/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-mono font-bold text-xs group-hover:bg-cyan-600/30 transition-colors">
              ARX
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-white tracking-wider font-mono">
                HORIZON OS
              </span>
              <span className="text-[9px] font-mono text-slate-400 -mt-0.5">
                Executive Intelligence System
              </span>
            </div>
          </Link>

          <span className="px-2 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-300 text-[10px] font-mono font-semibold">
            {badgeText}
          </span>
        </div>

        {/* Global Command Center Search (Ctrl+K) */}
        <div className="flex items-center gap-3">
          <ExecutiveGlobalSearch />
        </div>
      </div>

      {/* Categorized Navigation Rail */}
      <nav aria-label="Executive Intelligence Workflows" className="max-w-[1600px] mx-auto px-4 overflow-x-auto">
        <div className="flex items-center gap-4 py-1 text-xs font-mono min-w-max border-t border-[#182336]">
          {/* Workspaces Group */}
          <div className="flex items-center gap-1 pr-3 border-r border-[#182336]">
            <span className="text-[9px] uppercase tracking-wider text-slate-500 mr-1">Workspaces:</span>
            {WORKSPACE_LINKS.map((link) => {
              const active = isLinkActive(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    active
                      ? 'bg-cyan-600 text-white font-semibold shadow-sm'
                      : 'text-slate-300 hover:text-cyan-300 hover:bg-[#121B2A]'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Strategic Group */}
          <div className="flex items-center gap-1 pr-3 border-r border-[#182336]">
            <span className="text-[9px] uppercase tracking-wider text-slate-500 mr-1">Strategic:</span>
            {STRATEGIC_LINKS.map((link) => {
              const active = isLinkActive(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-2 py-1 rounded-md transition-colors ${
                    active
                      ? 'bg-blue-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#121B2A]'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Operational Group */}
          <div className="flex items-center gap-1 pr-3 border-r border-[#182336]">
            <span className="text-[9px] uppercase tracking-wider text-slate-500 mr-1">Operational:</span>
            {OPERATIONAL_LINKS.map((link) => {
              const active = isLinkActive(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-2 py-1 rounded-md transition-colors ${
                    active
                      ? 'bg-emerald-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#121B2A]'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Evidence Group */}
          <div className="flex items-center gap-1">
            <span className="text-[9px] uppercase tracking-wider text-slate-500 mr-1">Evidence:</span>
            {EVIDENCE_LINKS.map((link) => {
              const active = isLinkActive(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-2 py-1 rounded-md transition-colors ${
                    active
                      ? 'bg-purple-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#121B2A]'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>
    </header>
  );
}
