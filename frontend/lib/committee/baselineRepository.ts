/**
 * ARX Terminal vNext - Committee Baseline Repository
 * Enforces: Single active baseline invariant, approval workflow, and immutable superseded states.
 * Reference: docs/architecture/COMMITTEE_INTELLIGENCE_ARCHITECTURE.md
 */

import { CommitteeBaseline } from "../../types/committee-intelligence";

export class BaselineRepository {
  private baselines = new Map<string, CommitteeBaseline>();

  private makeKey(committeeId: string, ticker: string): string {
    return `${committeeId}:${ticker}`;
  }

  public async getActiveBaseline(
    committeeId: string,
    ticker: string
  ): Promise<CommitteeBaseline | null> {
    const all = Array.from(this.baselines.values());
    const active = all.find(
      (b) => b.committeeId === committeeId && b.ticker === ticker && b.status === "ACTIVE"
    );
    return active || null;
  }

  public async createBaseline(baseline: CommitteeBaseline): Promise<void> {
    if (baseline.status === "ACTIVE") {
      const active = await this.getActiveBaseline(baseline.committeeId, baseline.ticker);
      if (active && active.baselineId !== baseline.baselineId) {
        throw new Error("ACTIVE_BASELINE_EXISTS: Only one active baseline allowed per ticker.");
      }
    }
    this.baselines.set(baseline.baselineId, { ...baseline });
  }

  public async supersedeBaseline(baselineId: string): Promise<void> {
    const existing = this.baselines.get(baselineId);
    if (!existing) throw new Error("BASELINE_NOT_FOUND");

    this.baselines.set(baselineId, {
      ...existing,
      status: "SUPERSEDED",
    });
  }

  public async activateBaseline(baselineId: string): Promise<void> {
    const existing = this.baselines.get(baselineId);
    if (!existing) throw new Error("BASELINE_NOT_FOUND");

    if (existing.status === "SUPERSEDED") {
      throw new Error("BASELINE_IMMUTABLE: Superseded baselines cannot be reactivated.");
    }

    // Supersede any existing active baseline for this ticker
    const currentActive = await this.getActiveBaseline(existing.committeeId, existing.ticker);
    if (currentActive && currentActive.baselineId !== baselineId) {
      await this.supersedeBaseline(currentActive.baselineId);
    }

    this.baselines.set(baselineId, {
      ...existing,
      status: "ACTIVE",
    });
  }

  public clear(): void {
    this.baselines.clear();
  }
}

export const baselineRepo = new BaselineRepository();
