import React, { ReactNode } from 'react';

export interface WorkstationGridProps {
  /**
   * Primary visual anchor: Interactive TradingView chart canvas
   * Desktop: Occupies 8 of 12 columns (66.6% / 65% width)
   * Tablet & Mobile: Renders full width at top of vertical stack
   */
  chart: ReactNode;

  /**
   * Actionable decision ladder: Optimal Entry/Exit Corridor
   * Desktop: Occupies 4 of 12 columns (33.3% / 35% width)
   * Tablet & Mobile: Renders full width directly below chart canvas
   */
  execution: ReactNode;

  /**
   * Optional custom container CSS classes
   */
  className?: string;

  /**
   * Optional custom minimum height for desktop layout (defaults to 620px to enforce CLS < 0.05)
   */
  minHeightDesktop?: string;
}

/**
 * WorkstationGrid: Institutional 65/35 Viewport Anchoring System
 * Governed by ADR-001 (docs/architecture/adrs/ADR-001-65-35-viewport-anchoring.md)
 * 
 * Breakpoint Behavior:
 * - Desktop (>= 1024px / lg): 12-column CSS grid. Left 8 cols (65%), Right 4 cols (35%).
 * - Tablet (768px - 1023px / md): Single column vertical stack. Chart first, Execution second.
 * - Mobile (< 768px): Single column vertical stack. Chart fixed 420px height, Execution full width below.
 */
export const WorkstationGrid: React.FC<WorkstationGridProps> = ({
  chart,
  execution,
  className = '',
  minHeightDesktop = 'lg:min-h-[620px]',
}) => {
  return (
    <section
      data-testid="workstation-grid"
      aria-label="Market Understanding & Execution Geometry Workspace"
      className={`w-full grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6 ${minHeightDesktop} ${className}`}
    >
      {/* 65% Primary Visual Anchor (TradingView Chart) */}
      <div
        data-testid="price-chart-workspace"
        className="col-span-1 lg:col-span-8 flex flex-col w-full min-h-[420px] lg:min-h-full bg-bg-surface border border-border-subtle rounded-xl overflow-hidden shadow-sm"
      >
        {chart}
      </div>

      {/* 35% Actionable Execution Corridor */}
      <div
        data-testid="execution-corridor"
        className="col-span-1 lg:col-span-4 flex flex-col w-full min-h-[380px] lg:min-h-full bg-bg-surface border border-border-subtle rounded-xl p-4 lg:p-5 shadow-sm justify-between"
      >
        {execution}
      </div>
    </section>
  );
};

export default WorkstationGrid;
