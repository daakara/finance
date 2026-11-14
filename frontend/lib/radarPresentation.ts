import { ExecutionStatus } from '../types/decisionContract';

export function formatRadarExecutionStatus(status: ExecutionStatus): string {
  switch (status) {
    case 'DELISTED_SECURITY':
      return 'DELISTED / INACTIVE';
    case 'IN_BUY_ZONE':
      return 'NEAR SCREENING ZONE';
    case 'READY_TO_BUY':
      return 'NEAR SCREENING ZONE';
    case 'IN_BUY_ZONE_AWAITING_TRIGGER':
      return 'AWAITING TRIGGER';
    case 'APPROACHING_TARGET':
      return 'APPROACHING TARGET';
    case 'WAITING_PULLBACK':
      return 'PULLBACK PENDING';
    case 'STOPPED_OUT':
      return 'STOPPED OUT';
    case 'INSUFFICIENT_HISTORY':
      return 'INSUFFICIENT HISTORY';
    case 'UNVERIFIED_ASSET':
      return 'UNVERIFIED ASSET';
    case 'STALE_MARKET_DATA':
      return 'STALE MARKET DATA';
    default:
      return 'UNKNOWN';
  }
}

export type RadarScoreTier = 'HIGH_CONVICTION' | 'CONSTRUCTIVE' | 'SELECTIVE' | 'DEFENSIVE' | 'UNAVAILABLE';

export interface RadarScorePresentation {
  tier: RadarScoreTier;
  colorClass: string;
  badgeClass: string;
  label: string;
  tooltip: string;
}

export function resolveRadarScorePresentation(
  score: number | null,
  isDelisted: boolean,
  backendBadgeColor?: string | null
): RadarScorePresentation {
  if (isDelisted) {
    return {
      tier: 'DEFENSIVE',
      colorClass: 'text-rose-400',
      badgeClass: 'text-rose-400 bg-rose-950/60 border border-rose-800/60',
      label: score !== null ? String(score) : '—',
      tooltip: 'Delisted Security · Trading Suspended',
    };
  }
  if (score === null || !Number.isFinite(score) || score < 0) {
    return {
      tier: 'UNAVAILABLE',
      colorClass: 'text-[var(--text-muted)]',
      badgeClass: 'text-[var(--text-muted)] font-normal font-mono border border-[var(--border-subtle)] bg-[var(--surface-2)]',
      label: '—',
      tooltip: 'Confluence Score Unavailable · Insufficient Data',
    };
  }

  // Exact canonical boundary alignment with ConfluenceEngine:
  // 80–100: High conviction (Emerald)
  // 65–79:  Constructive (Cyan)
  // 49–64:  Selective (Amber)
  // 0–48:   Defensive / capital preservation (Rose)
  // Ensures scores like 45/100 cannot be mistaken for an actionable buy recommendation.
  if (backendBadgeColor === 'emerald' || (score >= 80 && backendBadgeColor !== 'amber')) {
    return {
      tier: 'HIGH_CONVICTION',
      colorClass: 'text-emerald-400',
      badgeClass: 'text-emerald-400 bg-emerald-950/60 border border-emerald-800/60',
      label: String(score),
      tooltip: `High Confluence Ranking (${score}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  if (backendBadgeColor === 'cyan' || (score >= 65 && score < 80)) {
    return {
      tier: 'CONSTRUCTIVE',
      colorClass: 'text-cyan-400',
      badgeClass: 'text-cyan-400 bg-cyan-950/60 border border-cyan-800/60',
      label: String(score),
      tooltip: `Constructive Confluence Ranking (${score}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  if (backendBadgeColor === 'amber' || (score >= 49 && score < 65)) {
    return {
      tier: 'SELECTIVE',
      colorClass: 'text-amber-400',
      badgeClass: 'text-amber-400 bg-amber-950/60 border border-amber-800/60',
      label: String(score),
      tooltip: `Selective Confluence Ranking (${score}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  return {
    tier: 'DEFENSIVE',
    colorClass: 'text-rose-400',
    badgeClass: 'text-rose-400 bg-rose-950/60 border border-rose-800/60',
    label: String(score),
    tooltip: `Defensive / Sub-Threshold Ranking (${score}/100) · Relative discovery ranking — not an actionable buy recommendation`,
  };
}
