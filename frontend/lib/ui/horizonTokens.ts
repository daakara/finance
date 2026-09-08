/**
 * ARX Horizon Design System - Foundation Tokens (Phase 31-M11)
 *
 * Implements:
 * - HorizonColors: High-contrast executive color palette
 * - HorizonTypography: Typographic hierarchy with calibrated mono data styles
 * - HorizonSpacing & HorizonRadius
 * - SeverityColors & statusBadge mapping
 * - WCAG AA Contrast Verification (>= 4.5:1 text, >= 3.0:1 graphic)
 */

export const HorizonColors = {
  bg: '#0B1220',
  panel: '#121B2A',
  elevated: '#182336',
  border: '#24324A',
  pass: '#10B981',
  warning: '#F59E0B',
  high: '#EA580C',
  critical: '#DC2626',
  certified: '#2563EB',
  info: '#64748B',
  text: '#F8FAFC',
  muted: '#94A3B8',
} as const;

export type HorizonColorKey = keyof typeof HorizonColors;

export const HorizonSpacing = {
  xs: '4px',
  sm: '8px',
  md: '12px',
  lg: '16px',
  xl: '24px',
  '2xl': '32px',
  '3xl': '48px',
  '4xl': '64px',
} as const;

export const HorizonRadius = {
  none: '0px',
  sm: '4px',
  md: '8px',
  lg: '12px',
  intelligence: '16px',
  full: '9999px',
} as const;

export const HorizonTypography = {
  display: {
    fontSize: '2.25rem',
    lineHeight: '2.5rem',
    fontWeight: '700',
    letterSpacing: '-0.02em',
  },
  header1: {
    fontSize: '1.5rem',
    lineHeight: '2rem',
    fontWeight: '600',
    letterSpacing: '-0.01em',
  },
  header2: {
    fontSize: '1.125rem',
    lineHeight: '1.75rem',
    fontWeight: '600',
    letterSpacing: '-0.005em',
  },
  body: {
    fontSize: '0.875rem',
    lineHeight: '1.25rem',
    fontWeight: '400',
  },
  monoLg: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    fontSize: '1.5rem',
    lineHeight: '2rem',
    fontWeight: '600',
  },
  monoSm: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    fontSize: '0.875rem',
    lineHeight: '1.25rem',
    fontWeight: '500',
  },
  monoCaption: {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    fontSize: '0.75rem',
    lineHeight: '1rem',
    fontWeight: '400',
  },
} as const;

export type SeverityLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'PASS' | 'INFO';

export interface SeverityColorConfig {
  bg: string;
  border: string;
  text: string;
  badge: string;
  dot: string;
  label: string;
}

export const SeverityColors: Record<SeverityLevel, SeverityColorConfig> = {
  CRITICAL: {
    bg: 'bg-red-950/50',
    border: 'border-red-500/40',
    text: 'text-red-400',
    badge: 'bg-red-500/20 text-red-300 border-red-500/40',
    dot: 'bg-red-500',
    label: 'Critical',
  },
  HIGH: {
    bg: 'bg-orange-950/50',
    border: 'border-orange-500/40',
    text: 'text-orange-400',
    badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40',
    dot: 'bg-orange-500',
    label: 'High',
  },
  MEDIUM: {
    bg: 'bg-amber-950/50',
    border: 'border-amber-500/40',
    text: 'text-amber-400',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    dot: 'bg-amber-400',
    label: 'Medium',
  },
  LOW: {
    bg: 'bg-slate-900/50',
    border: 'border-slate-700/50',
    text: 'text-slate-400',
    badge: 'bg-slate-800/80 text-slate-300 border-slate-700/60',
    dot: 'bg-slate-400',
    label: 'Low',
  },
  PASS: {
    bg: 'bg-emerald-950/50',
    border: 'border-emerald-500/40',
    text: 'text-emerald-400',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    dot: 'bg-emerald-400',
    label: 'Pass',
  },
  INFO: {
    bg: 'bg-cyan-950/40',
    border: 'border-cyan-500/30',
    text: 'text-cyan-400',
    badge: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
    dot: 'bg-cyan-400',
    label: 'Info',
  },
};

/**
 * Returns Tailwind class names for a given status string.
 */
export function statusBadge(status: string): string {
  const norm = (status || '').toUpperCase();
  if (norm.includes('CRITICAL') || norm.includes('FAIL') || norm.includes('BREACH')) {
    return SeverityColors.CRITICAL.badge;
  }
  if (norm.includes('HIGH') || norm.includes('ALERT') || norm.includes('LOCK')) {
    return SeverityColors.HIGH.badge;
  }
  if (norm.includes('WARN') || norm.includes('MEDIUM') || norm.includes('DEGRAD')) {
    return SeverityColors.MEDIUM.badge;
  }
  if (norm.includes('CERTIFIED') || norm.includes('PASS') || norm.includes('HEALTHY') || norm.includes('RESOLVED')) {
    return SeverityColors.PASS.badge;
  }
  return SeverityColors.LOW.badge;
}

/**
 * Calculate relative luminance and WCAG contrast ratio between two hex colors.
 */
export function getRelativeLuminance(hex: string): number {
  const cleanHex = hex.replace('#', '');
  const r = parseInt(cleanHex.substring(0, 2), 16) / 255;
  const g = parseInt(cleanHex.substring(2, 4), 16) / 255;
  const b = parseInt(cleanHex.substring(4, 6), 16) / 255;

  const toLinear = (c: number) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const rLin = toLinear(r);
  const gLin = toLinear(g);
  const bLin = toLinear(b);

  return 0.2126 * rLin + 0.7152 * gLin + 0.0722 * bLin;
}

export function getContrastRatio(hex1: string, hex2: string): number {
  const lum1 = getRelativeLuminance(hex1);
  const lum2 = getRelativeLuminance(hex2);
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  return (brightest + 0.05) / (darkest + 0.05);
}

export function isWcagCompliant(fgHex: string, bgHex: string, isLargeText = false): boolean {
  const ratio = getContrastRatio(fgHex, bgHex);
  return isLargeText ? ratio >= 3.0 : ratio >= 4.5;
}
