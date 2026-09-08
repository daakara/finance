/**
 * ARX Horizon Layout & Viewport Adaptability Utilities (Phase 31-M11)
 *
 * Implements:
 * - Semantic breakpoints (xs, executive, analyst, intelligence, command, wallboard)
 * - Layout mode detection
 * - Container max widths and responsive column systems
 */

export const HorizonBreakpoints = {
  xs: 480,
  executive: 768,
  analyst: 1024,
  intelligence: 1280,
  command: 1536,
  wallboard: 1920,
} as const;

export type HorizonBreakpointKey = keyof typeof HorizonBreakpoints;

export type HorizonLayoutMode =
  | 'mobile-compact' // < 480px
  | 'mobile'         // 480px - 767px
  | 'executive'      // 768px - 1023px
  | 'analyst'        // 1024px - 1279px
  | 'intelligence'   // 1280px - 1535px
  | 'command'        // 1536px - 1919px
  | 'wallboard';     // >= 1920px

export function getLayoutMode(viewportWidth: number): HorizonLayoutMode {
  if (viewportWidth < HorizonBreakpoints.xs) return 'mobile-compact';
  if (viewportWidth < HorizonBreakpoints.executive) return 'mobile';
  if (viewportWidth < HorizonBreakpoints.analyst) return 'executive';
  if (viewportWidth < HorizonBreakpoints.intelligence) return 'analyst';
  if (viewportWidth < HorizonBreakpoints.command) return 'intelligence';
  if (viewportWidth < HorizonBreakpoints.wallboard) return 'command';
  return 'wallboard';
}

export const LayoutContainers = {
  standard: 'max-w-[1750px] mx-auto px-4 sm:px-6 lg:px-8',
  compact: 'max-w-5xl mx-auto px-4 sm:px-6',
  full: 'w-full px-4 sm:px-6 lg:px-8',
} as const;

export const HorizonGrids = {
  kpiRow: 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4',
  twoColumn: 'grid grid-cols-1 lg:grid-cols-2 gap-6',
  threeColumn: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6',
  dashboardMain: 'grid grid-cols-1 lg:grid-cols-12 gap-6',
} as const;
