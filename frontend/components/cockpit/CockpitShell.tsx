"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface CockpitShellProps {
  children: React.ReactNode;
  activeHub?: 'overview' | 'today' | 'future' | 'progress' | 'household';
}

export default function CockpitShell({ children, activeHub }: CockpitShellProps) {
  const pathname = usePathname();

  const hubs = [
    { id: 'overview', href: '/cockpit', label: 'Engine Overview' },
    { id: 'today', href: '/cockpit/today', label: 'Today & Execution' },
    { id: 'future', href: '/cockpit/future', label: 'Future & Scenarios' },
    { id: 'progress', href: '/cockpit/progress', label: 'Progress & Drift' },
    { id: 'household', href: '/cockpit/household', label: 'Household Alignment' },
  ];

  return (
    <div className="min-h-screen bg-[#070b12] text-gray-100 selection:bg-purple-500 selection:text-white">
      {/* Top Escape Hatch & Cockpit Sub-Header */}
      <nav
        role="navigation"
        aria-label="Behavioral Governor Navigation"
        className="w-full bg-[#0b1019] border-b border-gray-800 px-4 md:px-8 py-2.5"
      >
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
          <div className="flex items-center space-x-4">
            <Link
              href="/radar"
              className="inline-flex items-center space-x-1.5 text-xs font-mono text-cyan-400 hover:text-cyan-300 font-bold transition-colors"
            >
              <span>← Return to ARX Terminal</span>
            </Link>
            <span className="text-gray-700 hidden sm:inline">|</span>
            <span className="text-xs font-mono text-purple-400 font-semibold hidden sm:inline">
              🛡️ Behavioral Governor Engine
            </span>
          </div>

          <div className="flex items-center space-x-1 overflow-x-auto text-xs font-mono">
            {hubs.map((h) => {
              const isActive = activeHub === h.id || pathname === h.href;
              return (
                <Link
                  key={h.id}
                  href={h.href}
                  className={`px-2.5 py-1 rounded text-xs transition-colors shrink-0 ${
                    isActive
                      ? 'bg-purple-950/80 text-purple-300 border border-purple-800 font-bold'
                      : 'text-gray-400 hover:text-white hover:bg-gray-900'
                  }`}
                >
                  {h.label}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Cockpit Canvas */}
      <div className="max-w-7xl mx-auto p-4 md:p-8 pb-20 lg:pb-8 space-y-8">
        {children}
      </div>

      {/* Mobile Navigation Dock */}
      <nav
        data-testid="mobile-nav-dock"
        aria-label="Mobile Cockpit Navigation Dock"
        className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-[#070b12]/95 backdrop-blur-md border-t border-slate-800 px-2 py-1 flex items-center justify-around"
      >
        <Link
          href="/radar"
          className="flex flex-col items-center justify-center min-w-[48px] min-h-[44px] text-[10px] font-mono text-cyan-400 hover:text-cyan-200"
        >
          <span className="text-sm">⚡</span>
          <span>Terminal</span>
        </Link>
        {hubs.map((h) => {
          const isActive = activeHub === h.id || pathname === h.href;
          return (
            <Link
              key={h.id}
              href={h.href}
              className={`flex flex-col items-center justify-center min-w-[48px] min-h-[44px] text-[10px] font-mono transition-colors ${
                isActive
                  ? 'text-purple-400 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <span>{h.id === 'overview' ? '📊' : h.id === 'today' ? '🎯' : h.id === 'future' ? '🔮' : h.id === 'progress' ? '📈' : '👥'}</span>
              <span>{h.label.split(' ')[0]}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
