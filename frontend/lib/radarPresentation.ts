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

  const displayScore = Math.round(score);

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
      label: String(displayScore),
      tooltip: `High Confluence Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  if (backendBadgeColor === 'cyan' || (score >= 65 && score < 80)) {
    return {
      tier: 'CONSTRUCTIVE',
      colorClass: 'text-cyan-400',
      badgeClass: 'text-cyan-400 bg-cyan-950/60 border border-cyan-800/60',
      label: String(displayScore),
      tooltip: `Constructive Confluence Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  if (backendBadgeColor === 'amber' || (score >= 49 && score < 65)) {
    return {
      tier: 'SELECTIVE',
      colorClass: 'text-amber-400',
      badgeClass: 'text-amber-400 bg-amber-950/60 border border-amber-800/60',
      label: String(displayScore),
      tooltip: `Selective Confluence Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
    };
  }
  return {
    tier: 'DEFENSIVE',
    colorClass: 'text-rose-400',
    badgeClass: 'text-rose-400 bg-rose-950/60 border border-rose-800/60',
    label: String(displayScore),
    tooltip: `Defensive / Sub-Threshold Ranking (${displayScore}/100) · Relative discovery ranking — not an actionable buy recommendation`,
  };
}

export interface RadarCandidateInput {
  ticker: string;
  name?: string | null;
  sector?: string | null;
  vcpStage?: string | null;
  volumeDryUpPct?: number | null;
  rvol?: string | null;
  confluenceScore?: number | null;
  categories?: string[];
  executionStatus?: ExecutionStatus | string;
  opportunityExplanation?: string | null;
  catalyst?: string | null;
  riskRewardRatio?: number | null;
  verifiedPillarCount?: number | null;
  smartMoneyEvidenceState?: string | null;
}

export interface RadarLearnLink {
  label: string;
  href: string;
  tooltip: string;
  category: 'TECHNICAL' | 'FUNDAMENTAL' | 'SMART_MONEY' | 'RISK_GOVERNANCE';
}

export interface RadarCandidateRationale {
  headline: string;
  summary: string;
  technicalContext: string;
  fullNarrative: string;
  learnLinks: RadarLearnLink[];
  coverageText: string;
}

export function resolveRadarCandidateRationale(candidate: RadarCandidateInput): RadarCandidateRationale {
  const ticker = candidate.ticker || 'UNKNOWN';
  const execStatus = String(candidate.executionStatus || 'UNKNOWN');
  const sector = candidate.sector?.trim() || null;
  const sectorText = sector ? ` within the ${sector} sector` : '';
  const categories = Array.isArray(candidate.categories) ? candidate.categories : [];
  const vcpStage = candidate.vcpStage?.trim() || null;
  const stageDisplay = vcpStage || 'Pattern Unassigned';

  const coverageText =
    typeof candidate.verifiedPillarCount === 'number'
      ? `${candidate.verifiedPillarCount}/4 Pillars Verified`
      : 'Coverage: Not Checked';

  // 1. Delisted / Ceased Trading
  if (execStatus === 'DELISTED_SECURITY') {
    const summary = 'Trading suspended for delisted security.';
    const technicalContext = 'Historical price and volume metrics preserved for point-in-time audit integrity. Active execution prohibited.';
    return {
      headline: 'Trading Suspended',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Audit Provenance',
          href: '/guide#module-foundations',
          tooltip: 'Learn how ARX preserves immutable historical baselines and corporate action lineage',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  // 2. Unverified Market Data / Feed Missing
  if (execStatus === 'UNVERIFIED_ASSET') {
    const summary = `Primary market data feed and trade tape unverified for ${ticker}.`;
    const technicalContext = 'Execution levels and position sizing suppressed until authoritative exchange feed clearance.';
    return {
      headline: 'Market Feed Unverified',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Data Governance',
          href: '/guide#module-foundations',
          tooltip: 'Learn about primary market data verification requirements',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  // 3. Insufficient History Depth
  if (execStatus === 'INSUFFICIENT_HISTORY') {
    const summary = `Historical bar depth insufficient for full econometric modeling on ${ticker}.`;
    const technicalContext = 'Confluence ranking and volatility contraction patterns require at least 250 daily trading sessions. Discovery signals suppressed.';
    return {
      headline: 'Insufficient History Depth',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Data Governance',
          href: '/guide#module-foundations',
          tooltip: 'Learn about minimum historical session requirements for quantitative models',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  // 4. Stale Market Data / Session Drift
  if (execStatus === 'STALE_MARKET_DATA') {
    const summary = `Market data timestamps for ${ticker} precede the authoritative completed session cutoff.`;
    const technicalContext = 'Signals suppressed pending exchange refresh to prevent stale-session decision bias.';
    return {
      headline: 'Stale Market Data',
      summary,
      technicalContext,
      fullNarrative: `${summary} ${technicalContext}`,
      learnLinks: [
        {
          label: 'Market Freshness',
          href: '/guide#module-foundations',
          tooltip: 'Learn about market data recency and session cutoff verification',
          category: 'RISK_GOVERNANCE',
        },
      ],
      coverageText,
    };
  }

  const hasVcp = categories.includes('VCP');
  const hasGarp = categories.includes('VALUE_GARP');
  const hasSm = categories.includes('SMART_MONEY');

  const dryUpText =
    typeof candidate.volumeDryUpPct === 'number' && candidate.volumeDryUpPct > 0
      ? ` with ${candidate.volumeDryUpPct}% volume dry-up`
      : '';
  const rvolText =
    candidate.rvol && candidate.rvol !== 'N/A' && candidate.rvol !== '—'
      ? ` on ${candidate.rvol} relative volume`
      : '';

  let headline: string;
  let summary: string;
  let technicalContext: string;

  const explicitExplanation =
    candidate.opportunityExplanation &&
    candidate.opportunityExplanation.trim().length > 0 &&
    candidate.opportunityExplanation !== 'Catalyst disclosures unavailable.'
      ? candidate.opportunityExplanation.trim()
      : null;

  if (explicitExplanation) {
    headline = `${ticker} · Discovery Candidate`;
    summary = explicitExplanation;
    technicalContext = 'Sourced directly from quantitative screening evaluation. Single-asset trigger clearance required.';
  } else if (hasVcp && hasGarp) {
    headline = `${ticker} · Dual Technical & Fundamental Confluence`;
    summary = `${ticker} combines an advancing Stage 2 trend structure (${stageDisplay}${dryUpText}) with disciplined capital efficiency${sectorText}.`;
    technicalContext = 'Structural consolidation indicates institutional supply absorption. Execution requires single-asset DecisionTrace clearance; discovery ranking is not an automatic buy signal.';
  } else if (hasVcp && hasSm) {
    headline = `${ticker} · Technical Contraction with Smart Money Classification`;
    summary = `${ticker} demonstrates progressive volatility contraction (${stageDisplay}${dryUpText}); classified under Smart Money screener snapshot (Form 4 filing transactions unverified)${rvolText}${sectorText}.`;
    technicalContext = 'Narrowing contraction waves reflect volatility compression. Smart Money classification does not confirm institutional accumulation without independent transaction filing verification.';
  } else if (hasVcp) {
    headline = `${ticker} · Minervini Stage 2 Accumulation Structure`;
    summary = `${ticker} consolidates in an advancing Stage 2 trend structure (${stageDisplay}${dryUpText})${sectorText}.`;
    technicalContext = 'Successive contraction waves absorb overhead supply prior to resistance clearance. Execution trigger requires single-asset DecisionTrace.';
  } else if (hasGarp) {
    headline = `${ticker} · Fundamental Valuation & Cash Flow Discipline`;
    summary = `${ticker} qualifies under the Value GARP screener, demonstrating operating margin stability and return on invested capital${sectorText ? ` within ${sector}` : ''}.`;
    technicalContext = 'Resilient balance sheet metrics provide structural valuation support amidst market regime shifts.';
  } else if (hasSm) {
    headline = `${ticker} · Smart Money Observation`;
    summary = `${ticker} classified under Smart Money screener snapshot; Form 4 filing transactions unverified${rvolText}${sectorText}.`;
    technicalContext = 'Prospective cohort classification based on screener criteria. Categorization alone does not confirm institutional accumulation; filing disclosure timeliness subject to statutory reporting windows.';
  } else {
    headline = `${ticker} · Multi-Factor Discovery Candidate`;
    summary = `${ticker} identified through multi-pillar screening${sectorText ? ` in ${sector}` : ''} (${stageDisplay}).`;
    technicalContext = 'Consolidating in structural discovery universe; single-asset validation required for execution parameters.';
  }

  // Compile Contextual Learn Hub & Glossary Deep-Links
  const learnLinks: RadarLearnLink[] = [];

  // Always include Confluence ranking reference
  learnLinks.push({
    label: 'Confluence Ranking',
    href: '/glossary#arx-model',
    tooltip: 'ARX Econometric Model and Confluence Score calibration (relative ranking != buy recommendation)',
    category: 'RISK_GOVERNANCE',
  });

  const stageLower = vcpStage ? vcpStage.toLowerCase() : '';
  if (hasVcp || stageLower.includes('vcp') || stageLower.includes('stage 2') || stageLower.includes('breakout')) {
    learnLinks.push({
      label: 'VCP Mechanics',
      href: '/guide#module-technical',
      tooltip: 'Field Manual: Minervini Stage 2 Trend Template and contraction wave geometry',
      category: 'TECHNICAL',
    });
    learnLinks.push({
      label: 'Glossary: VCP',
      href: '/glossary#minervini-vcp',
      tooltip: 'Quantitative Dictionary: Mark Minervini Volatility Contraction Pattern formula',
      category: 'TECHNICAL',
    });
  }

  if (hasGarp) {
    learnLinks.push({
      label: 'Fundamental Module',
      href: '/guide#module-fundamental',
      tooltip: 'Field Manual: Piotroski F-Score and fundamental quality screening',
      category: 'FUNDAMENTAL',
    });
    learnLinks.push({
      label: 'Piotroski F-Score',
      href: '/glossary#piotroski-f-score',
      tooltip: 'Quantitative Dictionary: 9-point fundamental health framework',
      category: 'FUNDAMENTAL',
    });
  }

  if (hasSm) {
    learnLinks.push({
      label: 'Smart Money Forensics',
      href: '/guide#module-smart-money',
      tooltip: 'Field Manual: SEC Form 4 forensics and insider order flow tracking',
      category: 'SMART_MONEY',
    });
    learnLinks.push({
      label: 'STOCK Act Decays',
      href: '/glossary#stock-act',
      tooltip: 'Quantitative Dictionary: Statutory 45-day reporting decay curves',
      category: 'SMART_MONEY',
    });
  }

  if (candidate.riskRewardRatio && candidate.riskRewardRatio >= 2) {
    learnLinks.push({
      label: 'M-VaR Risk Fencing',
      href: '/glossary#cornish-fisher-var',
      tooltip: 'Cornish-Fisher Modified VaR and risk floor calibration',
      category: 'RISK_GOVERNANCE',
    });
  }

  return {
    headline,
    summary,
    technicalContext,
    fullNarrative: `${summary} ${technicalContext}`,
    learnLinks,
    coverageText,
  };
}

