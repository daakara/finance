# Component Consolidation Plan
## ARX Terminal: UI Component Pruning, Consolidation & Institutional Primitives Specification (Horizon 14.3)

**Document ID**: `COMP-PLAN-ARX-H14.3-M1`  
**Classification**: Component Library Architecture & Design System Consolidation  
**Governing PRD**: `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`  
**Target Systems**: `frontend/components/`, `frontend/app/`  
**Date**: 2026-09-09T18:22:00+02:00  

---

## 1. Executive Summary & Component Pruning Thesis

Over multiple iterations, the ARX Terminal frontend accumulated redundant UI components, duplicated layout patterns, and conflicting component variants. The proliferation of generic 4-card KPI grids (`grid-cols-4`) and hardcoded arbitrary classes diluted the visual hierarchy and inflated bundle overhead.

This consolidation plan establishes:
1. **The Deprecation Ledger**: Formal deprecation of redundant cards, duplicated mobile docks, duplicate copy buttons, and generic 4-box loading skeletons.
2. **The 4 Core Institutional Primitives**: Formal architectural and TypeScript specifications for four standardized, high-performance primitives:
   - `DecisionHero`: Asymmetric Level 0 focal container.
   - `DataLedgerTable`: Compact, dense financial ledger table with monospace numbers and sorting.
   - `SemanticBadge`: Strict 5-color semantic telemetry badge.
   - `AsymmetricSkeleton`: Layout-matched loading skeletons guaranteeing zero cumulative layout shift (CLS = 0).
3. **Migration & Adoption Mapping**: Detailed migration roadmap for all 6 flagship hubs.

---

## 2. Component Deprecation & Pruning Ledger

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FORMAL COMPONENT DEPRECATION LEDGER                             │
├───────────────────────┬───────────────────────────────┬────────────────────────────────┤
│ Deprecated Component  │ Current Code Location         │ Consolidated Replacement       │
├───────────────────────┼───────────────────────────────┼────────────────────────────────┤
│ Duplicate Mobile Dock │ Navbar.tsx:393-473            │ TerminalShell.tsx:122-162 Dock │
│ Generic 4-Card Farm   │ portfolio/page.tsx:298-340    │ DecisionHero (Risk Variant)    │
│ Generic 4-Card Farm   │ journal/page.tsx:27-48        │ DecisionHero (Discipline Var)  │
│ 4-Box Parameter Grid  │ setups/page.tsx:114-142       │ DecisionHero (Ticket Variant)  │
│ Duplicate Copy Button │ setups/page.tsx:287-292       │ Unified Authorize Order CTA    │
│ Generic 4-Card Pulse  │ IntelligenceLoadingState.tsx  │ AsymmetricSkeleton Primitive   │
│ 6 Mini Milestone Cards│ performance/page.tsx:252-261  │ Counterfactual Equity Curve    │
│ Retail Wallet Presets │ portfolio/page.tsx:342-438    │ Direct Scaling Input ($10k+)   │
└───────────────────────┴───────────────────────────────┴────────────────────────────────┘
```

### 2.1 Deprecation Item 1: Duplicate Mobile Navigation Dock
- **Source Location**: `frontend/components/Navbar.tsx:393-473`
- **Defect**: Renders a redundant bottom bar on mobile viewports while `TerminalShell.tsx` renders its own bottom bar.
- **Resolution**: Permanently delete lines 393–473 in `Navbar.tsx`. Standardize on `TerminalShell.tsx:122-162` as the single authoritative mobile navigation bar with `z-50` and safe-area insets.

### 2.2 Deprecation Item 2: Generic 4-Card Farm Rows
- **Source Locations**:
  - `frontend/app/portfolio/page.tsx:298-340` (`grid grid-cols-2 lg:grid-cols-4`)
  - `frontend/app/journal/page.tsx:27-48` (`grid grid-cols-1 md:grid-cols-4`)
- **Defect**: Equal-weight card rows create visual noise without establishing a Level 0 decision focal point.
- **Resolution**: Replace with tailored instances of `DecisionHero`.

### 2.3 Deprecation Item 3: Duplicate Action Buttons on `/setups`
- **Source Location**: `frontend/app/setups/page.tsx:287-292`
- **Defect**: "Copy Broker Order String" button duplicates the clipboard handler of the "Authorize Order" button directly above it.
- **Resolution**: Delete lines 287–292. Retain a single primary `button` with reactive copied state feedback.

### 2.4 Deprecation Item 4: Generic 4-Card Pulse Skeleton
- **Source Location**: `frontend/components/ui/IntelligenceLoadingState.tsx:24-32`
- **Defect**: Hardcodes `cardCount = 4` in a 4-box grid, reinforcing the card-farm anti-pattern during hydration.
- **Resolution**: Deprecate the 4-card loop in favor of `AsymmetricSkeleton`.

---

## 3. Standardized Institutional Primitives Specification

### 3.1 Primitive 1: `DecisionHero`
- **Role**: Establishes the dominant Level 0 Decision focal point (>40% visual prominence), answering the screen's singular question.
- **Visual Design**: Asymmetric container (70% primary metric focal display + 30% utility context rail), subtle slate border (`border-slate-800`), deep obsidian background (`bg-slate-900/40`), and zero decorative gradients.

```tsx
// frontend/components/terminal/primitives/DecisionHero.tsx
import React from 'react';

export interface DecisionHeroMetric {
  label: string;
  value: string | number;
  subValue?: string;
  sentiment?: 'positive' | 'negative' | 'neutral' | 'caution' | 'focus';
}

export interface DecisionHeroProps {
  headline: string;
  badge?: {
    text: string;
    variant: 'emerald' | 'rose' | 'amber' | 'cyan' | 'purple' | 'slate';
  };
  primaryMetric: {
    label: string;
    value: string;
    delta?: string;
    sentiment?: 'positive' | 'negative' | 'neutral' | 'caution';
  };
  secondaryMetrics?: DecisionHeroMetric[];
  action?: {
    label: string;
    onClick: () => void;
    icon?: React.ReactNode;
    variant?: 'primary' | 'secondary';
  };
  children?: React.ReactNode;
  className?: string;
}

export const DecisionHero: React.FC<DecisionHeroProps> = ({
  headline,
  badge,
  primaryMetric,
  secondaryMetrics = [],
  action,
  children,
  className = '',
}) => {
  return (
    <div
      data-testid="decision-hero"
      className={`relative overflow-hidden rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 sm:p-6 shadow-xl backdrop-blur-sm ${className}`}
    >
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
        {/* Left 70%: Primary Decision Focal Point */}
        <div className="space-y-3 max-w-2xl">
          <div className="flex items-center gap-2.5">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              {headline}
            </span>
            {badge && (
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold border ${
                  badge.variant === 'emerald'
                    ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60'
                    : badge.variant === 'rose'
                    ? 'bg-rose-950/60 text-rose-300 border-rose-800/60'
                    : badge.variant === 'amber'
                    ? 'bg-amber-950/60 text-amber-300 border-amber-800/60'
                    : badge.variant === 'cyan'
                    ? 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60'
                    : badge.variant === 'purple'
                    ? 'bg-purple-950/60 text-purple-300 border-purple-800/60'
                    : 'bg-slate-800/60 text-slate-300 border-slate-700/60'
                }`}
              >
                {badge.text}
              </span>
            )}
          </div>

          <div>
            <div className="text-xs text-slate-400 mb-1">{primaryMetric.label}</div>
            <div className="flex items-baseline gap-3">
              <span className="text-3xl sm:text-4xl font-mono font-bold tracking-tight text-white">
                {primaryMetric.value}
              </span>
              {primaryMetric.delta && (
                <span
                  className={`text-sm font-mono font-semibold ${
                    primaryMetric.sentiment === 'positive'
                      ? 'text-emerald-400'
                      : primaryMetric.sentiment === 'negative'
                      ? 'text-rose-400'
                      : 'text-slate-400'
                  }`}
                >
                  {primaryMetric.delta}
                </span>
              )}
            </div>
          </div>

          {children}
        </div>

        {/* Right 30%: Utility Metrics Rail & Primary Action */}
        <div className="flex flex-col sm:flex-row lg:flex-col items-start lg:items-end justify-between gap-4 border-t lg:border-t-0 lg:border-l border-slate-800/80 pt-4 lg:pt-0 lg:pl-6">
          {secondaryMetrics.length > 0 && (
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 w-full">
              {secondaryMetrics.map((m, idx) => (
                <div key={idx} className="space-y-0.5">
                  <div className="text-[11px] text-slate-400">{m.label}</div>
                  <div className="text-sm font-mono font-bold text-slate-200">{m.value}</div>
                </div>
              ))}
            </div>
          )}

          {action && (
            <button
              onClick={action.onClick}
              className={`w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg text-xs font-mono font-bold transition-all shadow-md ${
                action.variant === 'secondary'
                  ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-950/40'
              }`}
            >
              {action.icon}
              <span>{action.label}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
```

---

### 3.2 Primitive 2: `DataLedgerTable`
- **Role**: Houses dense Level 2 empirical proof and historical audit records with institutional typographic density.
- **Visual Design**: Strict row height ($\le 36\text{px}$), monospace tabular figures, right-aligned numbers, sans-serif column headers, subtle hover state (`hover:bg-slate-800/30`), and zero horizontal overflow.

```tsx
// frontend/components/terminal/primitives/DataLedgerTable.tsx
import React from 'react';

export interface ColumnDef<T> {
  key: string;
  header: string;
  align?: 'left' | 'right' | 'center';
  width?: string;
  render: (item: T, index: number) => React.ReactNode;
}

export interface DataLedgerTableProps<T> {
  data: T[];
  columns: ColumnDef<T>[];
  keyExtractor: (item: T, index: number) => string;
  emptyMessage?: string;
  className?: string;
}

export function DataLedgerTable<T>({
  data,
  columns,
  keyExtractor,
  emptyMessage = 'No records in ledger.',
  className = '',
}: DataLedgerTableProps<T>) {
  return (
    <div className={`overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-900/40 ${className}`}>
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-slate-800 bg-slate-950/60 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`py-2.5 px-3 font-sans ${
                  col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                }`}
                style={col.width ? { width: col.width } : undefined}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/50 text-xs">
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="py-8 text-center text-slate-500 font-sans">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((item, idx) => (
              <tr
                key={keyExtractor(item, idx)}
                className="hover:bg-slate-800/30 transition-colors group"
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`py-2 px-3 ${
                      col.align === 'right'
                        ? 'text-right font-mono tabular-nums'
                        : col.align === 'center'
                        ? 'text-center'
                        : 'text-left font-sans'
                    }`}
                  >
                    {col.render(item, idx)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
```

---

### 3.3 Primitive 3: `SemanticBadge`
- **Role**: Standardized telemetry indicator enforcing the 5-color semantic discipline.
- **Visual Design**: Subtle semi-transparent container with border accent, no glowing rainbow gradients.

```tsx
// frontend/components/terminal/primitives/SemanticBadge.tsx
import React from 'react';

export type SemanticTone = 'emerald' | 'rose' | 'amber' | 'cyan' | 'purple' | 'slate';

export interface SemanticBadgeProps {
  children: React.ReactNode;
  tone: SemanticTone;
  size?: 'xs' | 'sm';
  dot?: boolean;
  className?: string;
}

export const SemanticBadge: React.FC<SemanticBadgeProps> = ({
  children,
  tone,
  size = 'xs',
  dot = false,
  className = '',
}) => {
  const toneClasses: Record<SemanticTone, string> = {
    emerald: 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60',
    rose: 'bg-rose-950/60 text-rose-300 border-rose-800/60',
    amber: 'bg-amber-950/60 text-amber-300 border-amber-800/60',
    cyan: 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60',
    purple: 'bg-purple-950/60 text-purple-300 border-purple-800/60',
    slate: 'bg-slate-800/60 text-slate-300 border-slate-700/60',
  };

  const dotClasses: Record<SemanticTone, string> = {
    emerald: 'bg-emerald-400',
    rose: 'bg-rose-400',
    amber: 'bg-amber-400',
    cyan: 'bg-cyan-400',
    purple: 'bg-purple-400',
    slate: 'bg-slate-400',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border font-mono font-semibold uppercase tracking-wider ${
        size === 'xs' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-1 text-xs'
      } ${toneClasses[tone]} ${className}`}
    >
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${dotClasses[tone]}`} />}
      <span>{children}</span>
    </span>
  );
};
```

---

### 3.4 Primitive 4: `AsymmetricSkeleton`
- **Role**: Layout-matched loading state guaranteeing Cumulative Layout Shift (CLS) = 0.00.
- **Visual Design**: Mirrored layout geometry matching `DecisionHero` and `DataLedgerTable` during client hydration.

```tsx
// frontend/components/terminal/primitives/AsymmetricSkeleton.tsx
import React from 'react';

export interface AsymmetricSkeletonProps {
  variant: 'hero' | 'ledger' | 'split';
  className?: string;
}

export const AsymmetricSkeleton: React.FC<AsymmetricSkeletonProps> = ({
  variant,
  className = '',
}) => {
  if (variant === 'hero') {
    return (
      <div
        data-testid="asymmetric-hero-skeleton"
        className={`rounded-xl border border-slate-800/80 bg-slate-900/40 p-6 animate-pulse ${className}`}
      >
        <div className="flex flex-col lg:flex-row justify-between gap-6">
          <div className="space-y-4 max-w-lg w-full">
            <div className="h-3 w-32 bg-slate-800 rounded" />
            <div className="h-8 w-64 bg-slate-800 rounded" />
            <div className="h-4 w-48 bg-slate-800 rounded" />
          </div>
          <div className="grid grid-cols-2 gap-4 w-full lg:w-72 pt-4 lg:pt-0">
            <div className="h-12 bg-slate-800/60 rounded" />
            <div className="h-12 bg-slate-800/60 rounded" />
            <div className="h-10 col-span-2 bg-slate-800/80 rounded" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid="asymmetric-ledger-skeleton"
      className={`rounded-xl border border-slate-800/80 bg-slate-900/40 p-4 space-y-3 animate-pulse ${className}`}
    >
      <div className="h-4 w-40 bg-slate-800 rounded" />
      <div className="space-y-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-8 bg-slate-800/40 rounded w-full" />
        ))}
      </div>
    </div>
  );
};
```

---

## 4. Adoption & Migration Roadmap across Flagship Hubs

| Hub | Deprecated Element | Consolidated Primitive | Target Delivery |
|-----|--------------------|------------------------|-----------------|
| `/portfolio` | Lines 298–340 (`grid-cols-4` 4 cards) | `DecisionHero` (Capital at Risk variant) | Milestone M3 |
| `/portfolio` | Lines 455–538 (Custom table) | `DataLedgerTable` (Holdings ledger) | Milestone M3 |
| `/journal` | Lines 27–48 (`grid-cols-4` 4 cards) | `DecisionHero` (Discipline variant) | Milestone M3 |
| `/journal` | Lines 51–93 (Hardcoded table) | `DataLedgerTable` (Audit ledger) | Milestone M3 |
| `/setups` | Lines 114–142 (4 parameter boxes) | `DecisionHero` (Order Ticket variant) | Milestone M3 |
| `/radar` | Lines 455–527 (40-card grid) | `DecisionHero` (#1 Leader) + Confluence Stream | Milestone M3 |
| `/performance`| Lines 126–148 (4 inner boxes) | `DecisionHero` (Capital Preserved variant) | Milestone M3 |
| Global Shell | `Navbar.tsx:393-473` | Deprecated; `TerminalShell.tsx` Mobile Dock | Milestone M2 |
| Global Loading| `IntelligenceLoadingState.tsx` | `AsymmetricSkeleton` (`hero` / `ledger`) | Milestone M3 |
