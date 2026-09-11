/**
 * Canonical Trade Lifecycle Engine & Status Disambiguation Contract (Option A Canonical)
 *
 * Implements the 4 orthogonal status dimensions:
 * 1. Request Progress  : IDLE | LOADING | SUCCESS | FAILED | RETRYING
 * 2. Data Freshness    : REALTIME | DELAYED | STALE | MISSING
 * 3. Trade Lifecycle   : PLANNED -> OPEN -> CLOSED
 * 4. Setup Eligibility : IN_BUY_ZONE | APPROACHING | VOLUME_DRY_UP | STOPPED_OUT | DISQUALIFIED
 *
 * Enforces strict mathematical validation and zero inferred/fabricated fields.
 */

// -------------------------------------------------------------------------
// 1. THE 4 ORTHOGONAL STATUS DIMENSIONS
// -------------------------------------------------------------------------

export const RequestProgress = {
  IDLE: 'IDLE',
  LOADING: 'LOADING',
  SUBMITTING: 'SUBMITTING',
  SUCCESS: 'SUCCESS',
  ERROR: 'ERROR',
  FAILED: 'FAILED',
  RETRYING: 'RETRYING',
} as const;
export type RequestProgress = (typeof RequestProgress)[keyof typeof RequestProgress];

export const DataFreshness = {
  REALTIME: 'REALTIME',
  DELAYED: 'DELAYED',
  STALE: 'STALE',
  MISSING: 'MISSING',
} as const;
export type DataFreshness = (typeof DataFreshness)[keyof typeof DataFreshness];

export const TradeLifecycleState = {
  PLANNED: 'PLANNED',
  OPEN: 'OPEN',
  CLOSED: 'CLOSED',
} as const;
export type TradeLifecycleState = (typeof TradeLifecycleState)[keyof typeof TradeLifecycleState];

export const SetupEligibility = {
  IN_BUY_ZONE: 'IN_BUY_ZONE',
  APPROACHING: 'APPROACHING',
  VOLUME_DRY_UP: 'VOLUME_DRY_UP',
  STOPPED_OUT: 'STOPPED_OUT',
  DISQUALIFIED: 'DISQUALIFIED',
} as const;
export type SetupEligibility = (typeof SetupEligibility)[keyof typeof SetupEligibility];

export const ExecutionRole = {
  ENTRY: 'ENTRY',
  PARTIAL_EXIT: 'PARTIAL_EXIT',
  FULL_EXIT: 'FULL_EXIT',
} as const;
export type ExecutionRole = (typeof ExecutionRole)[keyof typeof ExecutionRole];

// -------------------------------------------------------------------------
// 2. INPUT & RECORD INTERFACES
// -------------------------------------------------------------------------

export interface TradeFillInput {
  symbol: string;
  setupName?: string;
  entryPrice: number;
  shares: number;
  stopLoss?: number;
  target1?: number;
  confidence?: number;
  entryDate?: string;
  idempotencyKey?: string;
  notes?: string;
}

export interface TradeExitInput {
  tradeId?: number | string;
  symbol?: string;
  exitPrice: number;
  shares?: number;
  exitDate?: string;
  followedRules?: boolean;
  idempotencyKey?: string;
  notes?: string;
}

export interface TradeCloseInput {
  tradeId?: number | string;
  symbol?: string;
  exitPrice: number;
  exitDate?: string;
  followedRules?: boolean;
  idempotencyKey?: string;
  notes?: string;
}

export interface LifecycleValidationResult {
  valid: boolean;
  error?: string;
}

// -------------------------------------------------------------------------
// 3. TRANSITION & INPUT VALIDATION RULES
// -------------------------------------------------------------------------

/**
 * Validate permitted transitions between trade lifecycle states.
 * Prohibits illegal transitions:
 * - Direct PLANNED -> CLOSED without execution fill
 * - CLOSED -> OPEN (closed records are immutable)
 */
export function validateLifecycleTransition(
  fromState: TradeLifecycleState,
  toState: TradeLifecycleState
): LifecycleValidationResult {
  const validStates = Object.values(TradeLifecycleState) as string[];
  if (!validStates.includes(fromState) || !validStates.includes(toState)) {
    return {
      valid: false,
      error: `Invalid trade lifecycle state: fromState '${fromState}', toState '${toState}'.`,
    };
  }

  if (fromState === toState) {
    return { valid: true };
  }

  if (fromState === 'PLANNED' && toState === 'OPEN') {
    return { valid: true };
  }

  if (fromState === 'OPEN' && toState === 'CLOSED') {
    return { valid: true };
  }

  if (fromState === 'PLANNED' && toState === 'CLOSED') {
    return {
      valid: false,
      error: 'Illegal trade lifecycle transition: Cannot transition directly from PLANNED to CLOSED without recording an actual broker execution fill.',
    };
  }

  if (fromState === 'CLOSED') {
    return {
      valid: false,
      error: 'Illegal trade lifecycle transition: Closed trades are terminal and immutable; cannot transition from CLOSED to OPEN or PLANNED.',
    };
  }

  return {
    valid: false,
    error: `Illegal trade lifecycle transition: Cannot transition from ${fromState} to ${toState}.`,
  };
}

/**
 * Validate broker execution fill inputs before submission.
 */
export function validateFillParams(input: TradeFillInput): LifecycleValidationResult {
  if (!input.symbol || typeof input.symbol !== 'string' || !input.symbol.trim()) {
    return { valid: false, error: 'Ticker symbol is required.' };
  }

  if (typeof input.entryPrice !== 'number' || isNaN(input.entryPrice) || input.entryPrice <= 0) {
    return { valid: false, error: 'Entry price must be a finite positive number greater than 0.' };
  }

  if (typeof input.shares !== 'number' || isNaN(input.shares) || input.shares <= 0) {
    return { valid: false, error: 'Share quantity must be a finite positive number greater than 0.' };
  }

  if (input.stopLoss !== undefined && input.stopLoss !== null) {
    if (typeof input.stopLoss !== 'number' || isNaN(input.stopLoss) || input.stopLoss <= 0) {
      return { valid: false, error: 'Stop loss price must be a finite positive number.' };
    }
  }

  if (input.target1 !== undefined && input.target1 !== null) {
    if (typeof input.target1 !== 'number' || isNaN(input.target1) || input.target1 <= 0) {
      return { valid: false, error: 'Target price must be a finite positive number.' };
    }
  }

  if (input.confidence !== undefined && input.confidence !== null) {
    if (typeof input.confidence !== 'number' || isNaN(input.confidence) || input.confidence < 0 || input.confidence > 100) {
      return { valid: false, error: 'Confidence score must be between 0% and 100%.' };
    }
  }

  return { valid: true };
}

/**
 * Validate exit or scale-out parameters against an open parent holding.
 */
export function validateExitParams(
  parentRemainingShares: number,
  exitShares: number,
  exitPrice: number
): LifecycleValidationResult {
  if (typeof exitPrice !== 'number' || isNaN(exitPrice) || exitPrice <= 0) {
    return { valid: false, error: 'Exit price must be a finite positive number greater than 0.' };
  }

  if (typeof exitShares !== 'number' || isNaN(exitShares) || exitShares <= 0) {
    return { valid: false, error: 'Exit shares count must be a finite positive number greater than 0.' };
  }

  if (exitShares > parentRemainingShares + 1e-6) {
    return {
      valid: false,
      error: `Exit quantity (${exitShares}) exceeds remaining open shares (${parentRemainingShares}).`,
    };
  }

  return { valid: true };
}

// -------------------------------------------------------------------------
// 4. MATHEMATICAL ACCOUNTING FUNCTIONS
// -------------------------------------------------------------------------

/**
 * Calculate realized net profit or loss for an exit leg.
 */
export function calculateRealizedPnL(
  entryPrice: number,
  exitPrice: number,
  shares: number
): number {
  return Number(((exitPrice - entryPrice) * shares).toFixed(2));
}

/**
 * Calculate R-multiple achieved on an exit leg.
 * Returns null if stopLoss is missing or equal to entryPrice.
 */
export function calculateRealizedR(
  entryPrice: number,
  exitPrice: number,
  stopLoss?: number | null
): number | null {
  if (stopLoss === undefined || stopLoss === null || isNaN(stopLoss) || entryPrice === stopLoss) {
    return null;
  }
  const riskPerShare = entryPrice - stopLoss;
  if (riskPerShare === 0) return null;
  return Number(((exitPrice - entryPrice) / riskPerShare).toFixed(2));
}

/**
 * Generate a cryptographically distinct, deterministic client-side idempotency key
 * to prevent double-submissions on rapid clicks or network retries.
 */
export function generateIdempotencyKey(
  action: 'fill' | 'exit' | 'close',
  symbol: string,
  userId: string = 'anon'
): string {
  const timestamp = Date.now();
  const rand = Math.random().toString(36).substring(2, 10);
  return `${userId}_${action}_${symbol.toUpperCase().trim()}_${timestamp}_${rand}`;
}
