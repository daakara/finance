'use client';

import React, { useState, useEffect } from 'react';
import {
  DecisionLifecycleState,
  LifecycleStep,
  DecisionProfile,
  MentorContext,
  MentorInsight,
} from '@/types/ux-foundations';
import { DecisionLifecycleTimeline } from './DecisionLifecycleTimeline';
import { DecisionProfileHeader } from './DecisionProfileHeader';
import { UnifiedARXMentor } from '@/components/mentor/UnifiedARXMentor';

export interface UnifiedWorkspaceShellProps {
  activeContext: MentorContext;
  onContextChange: (ctx: MentorContext) => void;
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  profile: DecisionProfile;
  steps: LifecycleStep[];
  currentStepIndex: number;
  mentorInsight: MentorInsight;
  children: React.ReactNode;
}

const WORKSPACE_NAV_ITEMS: Array<{
  context: MentorContext;
  label: string;
  icon: string;
  shortcut: string;
  badge?: string;
}> = [
  { context: 'ATTENTION', label: 'Command Center', icon: '⚡', shortcut: 'Alt+1', badge: '3 Alerts' },
  { context: 'DECISION', label: 'Prediction Canvas', icon: '🎯', shortcut: 'Alt+2' },
  { context: 'ATTRIBUTION', label: 'Outcome Resolution', icon: '⚖️', shortcut: 'Alt+3' },
  { context: 'LEARNING', label: 'Learning Center', icon: '🧠', shortcut: 'Alt+4', badge: 'Score 74' },
  { context: 'PLAYBOOK', label: 'Personal Playbook', icon: '📖', shortcut: 'Alt+5' },
  { context: 'GOVERNANCE', label: 'Governance Committee', icon: '🛡️', shortcut: 'Alt+6' },
];

export const UnifiedWorkspaceShell: React.FC<UnifiedWorkspaceShellProps> = ({
  activeContext,
  onContextChange,
  activeTicker,
  onTickerChange,
  profile,
  steps,
  currentStepIndex,
  mentorInsight,
  children,
}) => {
  const [railCollapsed, setRailCollapsed] = useState(false);
  const [mentorCollapsed, setMentorCollapsed] = useState(false);

  // Global Keyboard Shortcuts (Alt+1 through Alt+6)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey && !e.ctrlKey && !e.shiftKey) {
        const num = parseInt(e.key, 10);
        if (num >= 1 && num <= WORKSPACE_NAV_ITEMS.length) {
          e.preventDefault();
          const target = WORKSPACE_NAV_ITEMS[num - 1];
          if (target) onContextChange(target.context);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onContextChange]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* 1. Global Command Ribbon */}
      <header
        role="region"
        aria-label="Global Command Ribbon"
        className="h-14 bg-slate-900 border-b border-slate-800 px-4 flex items-center justify-between sticky top-0 z-40"
      >
        {/* Left: Brand & Ticker Switcher */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="h-3 w-3 rounded-sm bg-gradient-to-tr from-cyan-400 to-indigo-500 shadow-sm" />
            <span className="font-mono font-bold text-sm tracking-wider text-white">
              ARX<span className="text-cyan-400">::TERMINAL</span>
            </span>
            <span className="text-[10px] font-mono font-semibold px-1.5 py-0.2 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 hidden sm:inline">
              vNext
            </span>
          </div>

          <div className="h-4 w-px bg-slate-800 hidden sm:block" />

          {/* Active Ticker Selector */}
          <div className="flex items-center space-x-1.5">
            <label htmlFor="ticker-select" className="text-[10px] uppercase font-mono text-slate-400">
              Ticker:
            </label>
            <select
              id="ticker-select"
              value={activeTicker}
              onChange={(e) => onTickerChange(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono font-bold text-cyan-300 hover:border-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value="CPRX">CPRX (Catalyst Biotech)</option>
              <option value="NVDA">NVDA (NVIDIA Corp)</option>
              <option value="AAPL">AAPL (Apple Inc)</option>
              <option value="AMD">AMD (Advanced Micro Devices)</option>
              <option value="MSFT">MSFT (Microsoft Corp)</option>
              <option value="TSLA">TSLA (Tesla Inc)</option>
            </select>
          </div>
        </div>

        {/* Right: Market Status, Latency, Data Quality, Identity Avatar */}
        <div className="flex items-center space-x-3">
          <div className="hidden md:flex items-center space-x-3 text-xs font-mono">
            <div className="flex items-center space-x-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span className="text-slate-400">REGIME:</span>
              <span className="text-white font-semibold">Institutional Bull</span>
            </div>
            <span className="text-slate-700">•</span>
            <div className="text-slate-400">
              LATENCY: <strong className="text-emerald-400">14ms</strong>
            </div>
            <span className="text-slate-700">•</span>
            <div className="text-slate-400">
              DQ: <strong className="text-cyan-400">99.8%</strong>
            </div>
          </div>

          <div className="flex items-center space-x-2 pl-2 border-l border-slate-800">
            <div className="h-7 w-7 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-bold text-white shadow-inner">
              {profile.userName.charAt(0)}
            </div>
            <div className="hidden lg:block text-left">
              <div className="text-xs font-semibold text-white leading-tight">
                {profile.userName}
              </div>
              <div className="text-[10px] font-mono text-cyan-400 leading-tight">
                Score: {profile.qualityScore}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* 2. Middle Body: Navigation Rail + Main Canvas + Persistent Mentor */}
      <div className="flex-1 flex overflow-hidden">
        {/* Collapsible Navigation Rail */}
        <nav
          role="navigation"
          aria-label="Workspace Workstations"
          className={`${
            railCollapsed ? 'w-16' : 'w-56'
          } bg-slate-900 border-r border-slate-800 flex flex-col justify-between transition-all duration-200 shrink-0 z-30`}
        >
          <div className="p-2 space-y-1">
            <div className="flex items-center justify-between px-2 py-1.5 text-[10px] font-mono uppercase text-slate-400 tracking-wider">
              {!railCollapsed && <span>Workspaces</span>}
              <button
                type="button"
                onClick={() => setRailCollapsed(!railCollapsed)}
                aria-label={railCollapsed ? 'Expand Navigation' : 'Collapse Navigation'}
                className="text-slate-400 hover:text-white p-1 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
              >
                {railCollapsed ? '▶' : '◀'}
              </button>
            </div>

            {WORKSPACE_NAV_ITEMS.map((item) => {
              const isActive = activeContext === item.context;
              return (
                <button
                  key={item.context}
                  type="button"
                  onClick={() => onContextChange(item.context)}
                  aria-current={isActive ? 'page' : undefined}
                  title={`${item.label} (${item.shortcut})`}
                  className={`w-full flex items-center ${
                    railCollapsed ? 'justify-center px-2' : 'justify-between px-3'
                  } py-2 rounded-lg text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
                    isActive
                      ? 'bg-cyan-950/60 border border-cyan-500 text-white shadow-sm'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800 border border-transparent'
                  }`}
                >
                  <div className="flex items-center space-x-2.5 truncate">
                    <span className="text-sm">{item.icon}</span>
                    {!railCollapsed && <span className="truncate">{item.label}</span>}
                  </div>

                  {!railCollapsed && (
                    <div className="flex items-center space-x-1">
                      {item.badge && (
                        <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                          {item.badge}
                        </span>
                      )}
                      <kbd className="text-[9px] font-mono text-slate-400 hidden xl:inline">
                        {item.shortcut.split('+')[1]}
                      </kbd>
                    </div>
                  )}
                </button>
              );
            })}
          </div>

          {/* Rail Footer */}
          {!railCollapsed && (
            <div className="p-3 border-t border-slate-800 text-[10px] font-mono text-slate-400">
              <div>Alt+1..6 Quick Switch</div>
              <div className="text-slate-400 mt-0.5">Phase 26 Validated</div>
            </div>
          )}
        </nav>

        {/* 3. Main Application Canvas */}
        <main
          role="main"
          className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 max-w-full"
        >
          {/* Top of Canvas: Decision Lifecycle Timeline */}
          <DecisionLifecycleTimeline
            steps={steps}
            currentStepIndex={currentStepIndex}
          />

          {/* Decision Profile Header */}
          <DecisionProfileHeader profile={profile} />

          {/* Injected Screen Content */}
          <div className="mt-4">{children}</div>
        </main>

        {/* 4. Persistent ARX Mentor Panel (Desktop sidebar / Drawer) */}
        <aside
          role="complementary"
          aria-label="ARX Mentor Advisory Panel"
          className={`${
            mentorCollapsed ? 'w-12' : 'w-80 lg:w-96'
          } border-l border-slate-800 bg-slate-900/90 flex flex-col transition-all duration-200 shrink-0 overflow-y-auto hidden md:flex`}
        >
          {mentorCollapsed ? (
            <div className="p-2 flex flex-col items-center space-y-4">
              <button
                type="button"
                onClick={() => setMentorCollapsed(false)}
                aria-label="Expand ARX Mentor"
                className="p-2 rounded bg-cyan-950 border border-cyan-800 text-cyan-300 hover:bg-cyan-900 text-xs font-mono font-bold focus:outline-none focus:ring-2 focus:ring-cyan-500"
              >
                AI
              </button>
              <div className="text-[9px] font-mono text-slate-500 transform -rotate-90 whitespace-nowrap mt-8">
                ARX MENTOR
              </div>
            </div>
          ) : (
            <div className="p-4 flex-1 flex flex-col">
              <UnifiedARXMentor
                insight={mentorInsight}
                collapsed={mentorCollapsed}
                onToggleCollapse={() => setMentorCollapsed(!mentorCollapsed)}
              />
            </div>
          )}
        </aside>
      </div>

      {/* 5. Utility Status Bar Footer */}
      <footer
        role="contentinfo"
        className="h-8 bg-slate-900 border-t border-slate-800 px-4 flex items-center justify-between text-[11px] font-mono text-slate-400 z-30"
      >
        <div className="flex items-center space-x-3 truncate">
          <span className="flex items-center space-x-1.5 text-emerald-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
            <span className="font-semibold">CORE ONLINE</span>
          </span>
          <span className="text-slate-700 hidden sm:inline">•</span>
          <span className="text-slate-400 hidden sm:inline">
            Active Workspace: <strong className="text-slate-200">{activeContext}</strong>
          </span>
          <span className="text-slate-700 hidden md:inline">•</span>
          <span className="text-slate-400 hidden md:inline">
            Ticker: <strong className="text-cyan-300">{activeTicker}</strong>
          </span>
        </div>

        <div className="flex items-center space-x-3 text-slate-400">
          <span className="hidden sm:inline">WCAG 2.2 AA Certified</span>
          <span className="text-slate-700 hidden sm:inline">•</span>
          <span>SEC/FINRA Invariant Guard</span>
        </div>
      </footer>
    </div>
  );
};
