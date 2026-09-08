/**
 * ARX Horizon Design System - Foundation Tokens & Types
 * Phase 31-M15: Executive Modernization Program
 */

export const BREAKPOINTS = {
  xs: 0,
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

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

export const intelligenceColors = {
  governance: '#2563EB',
  learning: '#7C3AED',
  risk: '#EA580C',
  coaching: '#059669',
  resilience: '#0891B2',
  autonomous: '#DC2626',
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export const radius = {
  sm: 6,
  md: 12,
  lg: 16,
  xl: 24,
  pill: 9999,
} as const;

export const typography = {
  display: 40,
  h1: 32,
  h2: 24,
  h3: 20,
  body: 16,
  small: 14,
  caption: 12,
} as const;

/**
 * Standardized 6-state Status Model across all intelligence centers
 */
export type HorizonStatus =
  | 'CERTIFIED'
  | 'HEALTHY'
  | 'WARNING'
  | 'HIGH_RISK'
  | 'CRITICAL'
  | 'FAILED';

export type SeverityLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'PASS' | 'INFO';

export interface SeverityColorConfig {
  bg: string;
  border: string;
  text: string;
  badge: string;
  dot: string;
  label: string;
  hex: string;
}

export const SeverityColors: Record<SeverityLevel, SeverityColorConfig> = {
  CRITICAL: {
    bg: 'bg-red-950/50',
    border: 'border-red-500/40',
    text: 'text-red-400',
    badge: 'bg-red-500/20 text-red-300 border-red-500/40',
    dot: 'bg-red-500',
    label: 'Critical',
    hex: '#DC2626',
  },
  HIGH: {
    bg: 'bg-orange-950/50',
    border: 'border-orange-500/40',
    text: 'text-orange-400',
    badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40',
    dot: 'bg-orange-500',
    label: 'High Risk',
    hex: '#EA580C',
  },
  MEDIUM: {
    bg: 'bg-amber-950/50',
    border: 'border-amber-500/40',
    text: 'text-amber-400',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    dot: 'bg-amber-400',
    label: 'Warning',
    hex: '#F59E0B',
  },
  LOW: {
    bg: 'bg-slate-900/50',
    border: 'border-slate-700/50',
    text: 'text-slate-400',
    badge: 'bg-slate-800/80 text-slate-300 border-slate-700/60',
    dot: 'bg-slate-400',
    label: 'Low',
    hex: '#2563EB',
  },
  PASS: {
    bg: 'bg-emerald-950/50',
    border: 'border-emerald-500/40',
    text: 'text-emerald-400',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    dot: 'bg-emerald-400',
    label: 'Healthy',
    hex: '#10B981',
  },
  INFO: {
    bg: 'bg-cyan-950/40',
    border: 'border-cyan-500/30',
    text: 'text-cyan-400',
    badge: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
    dot: 'bg-cyan-400',
    label: 'Info',
    hex: '#94A3B8',
  },
};

export const HorizonStatusConfig: Record<HorizonStatus, { label: string; badge: string; color: string; hex: string }> = {
  CERTIFIED: {
    label: 'Certified',
    badge: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
    color: 'text-blue-400',
    hex: '#2563EB',
  },
  HEALTHY: {
    label: 'Healthy',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    color: 'text-emerald-400',
    hex: '#10B981',
  },
  WARNING: {
    label: 'Warning',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    color: 'text-amber-400',
    hex: '#F59E0B',
  },
  HIGH_RISK: {
    label: 'High Risk',
    badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40',
    color: 'text-orange-400',
    hex: '#EA580C',
  },
  CRITICAL: {
    label: 'Critical',
    badge: 'bg-red-500/20 text-red-300 border-red-500/40',
    color: 'text-red-400',
    hex: '#DC2626',
  },
  FAILED: {
    label: 'Failed',
    badge: 'bg-rose-950/80 text-rose-300 border-rose-600/50',
    color: 'text-rose-400',
    hex: '#E11D48',
  },
};

export function normalizeHorizonStatus(status: string): HorizonStatus {
  const norm = (status || '').toUpperCase();
  if (norm.includes('FAIL') || norm.includes('BREACH')) return 'FAILED';
  if (norm.includes('CRIT')) return 'CRITICAL';
  if (norm.includes('HIGH') || norm.includes('ALERT') || norm.includes('LOCK')) return 'HIGH_RISK';
  if (norm.includes('WARN') || norm.includes('MED') || norm.includes('DEGRAD')) return 'WARNING';
  if (norm.includes('CERTIFIED') || norm.includes('AUDIT') || norm.includes('VERIF')) return 'CERTIFIED';
  return 'HEALTHY';
}

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
