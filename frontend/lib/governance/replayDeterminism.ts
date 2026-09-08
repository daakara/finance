/**
 * Phase 31-M1.1: Deterministic Replay & Numerical Stability Engine
 *
 * Implements:
 * - Cycle-Safe Recursive Deep Equality with Relative Floating-Point Tolerance
 * - Cycle-Safe Canonical Serialization with $ref tracking
 * - Deterministic SHA-256 Replay Hash
 * - Order-Independence Verification
 * - 100x Replay Verification Harness (0 Drift)
 * - Numerical Stability Guards (rejection of NaN & Infinity)
 */

import { sha256 } from './sha256';
import {
  ReplayComparisonResult,
  ReplayDeterminismResult,
  DeepComparisonResult,
  DeepComparisonMismatch,
} from '../../types/committee-intelligence';

export const DEFAULT_EPSILON = 1e-9;

export function nearlyEqual(a: number, b: number, epsilon = DEFAULT_EPSILON): boolean {
  if (Number.isNaN(a) || Number.isNaN(b)) return false;
  return Math.abs(a - b) <= epsilon;
}

export function nearlyEqualRelative(a: number, b: number, epsilon = DEFAULT_EPSILON): boolean {
  if (Number.isNaN(a) || Number.isNaN(b)) return false;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  const scale = Math.max(1.0, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= epsilon * scale;
}

export function validateFiniteNumber(value: number, fieldName: string): void {
  if (Number.isNaN(value)) {
    throw new Error(`NAN_DETECTED:${fieldName}`);
  }
  if (!Number.isFinite(value)) {
    throw new Error(`INFINITE_VALUE:${fieldName}`);
  }
}

export function validateReplayNumbers(result: Record<string, unknown>): void {
  for (const [key, val] of Object.entries(result)) {
    if (typeof val === 'number') {
      validateFiniteNumber(val, key);
    } else if (val && typeof val === 'object' && !Array.isArray(val)) {
      validateReplayNumbers(val as Record<string, unknown>);
    }
  }
}

interface ComparisonState {
  visitedLeft: WeakMap<object, string>;
  visitedRight: WeakMap<object, string>;
}

export function deepEqualWithTolerance(
  left: unknown,
  right: unknown,
  epsilon = DEFAULT_EPSILON
): DeepComparisonResult {
  const mismatches: DeepComparisonMismatch[] = [];
  const state: ComparisonState = {
    visitedLeft: new WeakMap(),
    visitedRight: new WeakMap(),
  };

  compareRecursive(left, right, '$', epsilon, state, mismatches);

  return {
    equal: mismatches.length === 0,
    mismatches,
  };
}

function compareRecursive(
  left: unknown,
  right: unknown,
  path: string,
  epsilon: number,
  state: ComparisonState,
  mismatches: DeepComparisonMismatch[]
): void {
  if (typeof left === 'number' && typeof right === 'number') {
    if (Number.isNaN(left) || Number.isNaN(right)) {
      mismatches.push({ path, expected: left, actual: right, reason: 'NAN_DETECTED' });
      return;
    }
    if (!nearlyEqualRelative(left, right, epsilon)) {
      mismatches.push({ path, expected: left, actual: right, reason: 'VALUE_MISMATCH' });
    }
    return;
  }

  if (typeof left !== typeof right) {
    mismatches.push({ path, expected: typeof left, actual: typeof right, reason: 'TYPE_MISMATCH' });
    return;
  }

  if (left === null || right === null) {
    if (left !== right) {
      mismatches.push({ path, expected: left, actual: right, reason: 'VALUE_MISMATCH' });
    }
    return;
  }

  if (typeof left === 'object' && typeof right === 'object') {
    const existingLeft = state.visitedLeft.get(left as object);
    const existingRight = state.visitedRight.get(right as object);

    if (existingLeft !== undefined || existingRight !== undefined) {
      if (existingLeft !== existingRight) {
        mismatches.push({ path, expected: existingLeft, actual: existingRight, reason: 'CYCLE_MISMATCH' });
      }
      return;
    }

    state.visitedLeft.set(left as object, path);
    state.visitedRight.set(right as object, path);

    if (Array.isArray(left) && Array.isArray(right)) {
      if (left.length !== right.length) {
        mismatches.push({ path, expected: left.length, actual: right.length, reason: 'ARRAY_LENGTH_MISMATCH' });
        return;
      }
      for (let i = 0; i < left.length; i++) {
        compareRecursive(left[i], right[i], `${path}[${i}]`, epsilon, state, mismatches);
      }
      return;
    }

    const leftObj = left as Record<string, unknown>;
    const rightObj = right as Record<string, unknown>;
    const allKeys = Array.from(new Set([...Object.keys(leftObj), ...Object.keys(rightObj)])).sort();

    for (const key of allKeys) {
      if (!(key in leftObj)) {
        mismatches.push({ path: `${path}.${key}`, expected: undefined, actual: rightObj[key], reason: 'EXTRA_PROPERTY' });
      } else if (!(key in rightObj)) {
        mismatches.push({ path: `${path}.${key}`, expected: leftObj[key], actual: undefined, reason: 'MISSING_PROPERTY' });
      } else {
        compareRecursive(leftObj[key], rightObj[key], `${path}.${key}`, epsilon, state, mismatches);
      }
    }
    return;
  }

  if (left !== right) {
    mismatches.push({ path, expected: left, actual: right, reason: 'VALUE_MISMATCH' });
  }
}

interface CanonicalState {
  seen: WeakMap<object, string>;
}

export function canonicalize(value: unknown, path = '$', state?: CanonicalState): unknown {
  state = state ?? { seen: new WeakMap() };

  if (Array.isArray(value)) {
    return value.map((item, idx) => canonicalize(item, `${path}[${idx}]`, state));
  }

  if (value && typeof value === 'object') {
    const existing = state.seen.get(value as object);
    if (existing) {
      return { $ref: existing };
    }
    state.seen.set(value as object, path);

    const obj = value as Record<string, unknown>;
    const sortedKeys = Object.keys(obj).sort();
    const result: Record<string, unknown> = {};

    for (const key of sortedKeys) {
      result[key] = canonicalize(obj[key], `${path}.${key}`, state);
    }
    return result;
  }

  return value;
}

export function canonicalSerialize(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

export function createReplayHash(value: unknown): string {
  const serialized = canonicalSerialize(value);
  return sha256(serialized);
}

export function compareReplayResults(expected: unknown, actual: unknown): ReplayComparisonResult {
  const comparison = deepEqualWithTolerance(expected, actual);
  const expectedHash = createReplayHash(expected);
  const actualHash = createReplayHash(actual);

  return {
    matchesExpected: comparison.equal,
    deterministic: comparison.equal,
    expectedHash,
    actualHash,
    mismatchedFields: comparison.mismatches.map(m => m.path),
    mismatchDetails: comparison.mismatches,
  };
}

export function verifyReplayDeterminism<T>(execute: () => T, iterations = 100): ReplayDeterminismResult {
  const hashes = new Set<string>();
  const failures: string[] = [];
  let canonicalHash = '';

  for (let i = 0; i < iterations; i++) {
    const result = execute();
    const hash = createReplayHash(result);
    if (i === 0) canonicalHash = hash;
    hashes.add(hash);
    if (hash !== canonicalHash) {
      failures.push(`Replay variance at iteration ${i + 1}: expected ${canonicalHash}, got ${hash}`);
    }
  }

  return {
    deterministic: hashes.size === 1,
    iterations,
    uniqueHashes: hashes.size,
    canonicalHash,
    failures,
  };
}

export function verifyOrderIndependence<T, R>(
  baselineInput: T,
  shuffledInput: T,
  execute: (input: T) => R
): ReplayComparisonResult {
  const baseline = execute(baselineInput);
  const shuffled = execute(shuffledInput);
  return compareReplayResults(baseline, shuffled);
}

export function evaluateReplayGate(replay: ReplayDeterminismResult): boolean {
  return replay.deterministic && replay.uniqueHashes === 1 && replay.failures.length === 0;
}
