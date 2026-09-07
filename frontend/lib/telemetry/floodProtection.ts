/**
 * ARX Terminal vNext - Telemetry Flood Protection Engine
 * Implements deduplication windows, session rate-limiting, burst detection, and tooltip throttling.
 * Reference: docs/architecture/DATA_QUALITY_AND_EDGE_CASE_GOVERNANCE.md
 */

import { AnalyticsEvent } from "../../types/telemetry";

const DEDUPE_RULES: Record<string, number> = {
  watchlist_drawer_opened: 10000,
  delta_banner_viewed: 30000,
  institutional_tooltip_viewed: 60000,
  portfolio_feed_loaded: 30000,
  morning_brief_viewed: 30000,
};

const MAX_EVENTS_PER_MINUTE = 200;
const BURST_THRESHOLD = 100;
const BURST_WINDOW_MS = 10000;
const MAX_QUEUE_SIZE = 500;

class FloodProtectionEngine {
  private dedupeCache = new Map<string, number>();
  private tooltipSessionMetrics = new Set<string>();
  private eventTimestamps: number[] = [];
  private burstDetected = false;
  private queue: AnalyticsEvent[] = [];

  public reset(): void {
    this.dedupeCache.clear();
    this.tooltipSessionMetrics.clear();
    this.eventTimestamps = [];
    this.burstDetected = false;
    this.queue = [];
  }

  /**
   * Checks whether the event is an immediate duplicate within configured window
   */
  public isDuplicate(event: AnalyticsEvent): boolean {
    const windowMs = DEDUPE_RULES[event.eventName];
    if (!windowMs) return false;

    const key = `${event.eventName}:${event.sessionId || "default"}:${event.ticker || "global"}`;
    const now = Date.now();
    const lastSeen = this.dedupeCache.get(key);

    if (lastSeen && now - lastSeen < windowMs) {
      return true;
    }

    this.dedupeCache.set(key, now);
    return false;
  }

  /**
   * Tooltip Throttling: Only records the first view per metric per session
   */
  public isTooltipThrottled(metricName: string): boolean {
    if (this.tooltipSessionMetrics.has(metricName)) {
      return true;
    }
    this.tooltipSessionMetrics.add(metricName);
    return false;
  }

  /**
   * Burst Detection: Checks for runaway loops (>100 events in 10s)
   */
  public recordAndCheckBurst(): { action: "ALLOW" | "SAMPLE" | "DROP"; reason?: string } {
    const now = Date.now();
    this.eventTimestamps.push(now);

    // Prune events older than 60s
    this.eventTimestamps = this.eventTimestamps.filter((t) => now - t <= 60000);

    // Check burst in last 10 seconds
    const countInLast10s = this.eventTimestamps.filter((t) => now - t <= BURST_WINDOW_MS).length;
    if (countInLast10s > BURST_THRESHOLD) {
      this.burstDetected = true;
      return { action: "DROP", reason: "Burst threshold exceeded (>100 in 10s)" };
    }

    // Check rate limit per minute
    if (this.eventTimestamps.length > MAX_EVENTS_PER_MINUTE) {
      return { action: "SAMPLE", reason: "Rate limit per minute exceeded (>200/min)" };
    }

    return { action: "ALLOW" };
  }

  public isBursting(): boolean {
    return this.burstDetected;
  }

  /**
   * Bounded queue management
   */
  public enqueue(event: AnalyticsEvent): boolean {
    if (this.queue.length >= MAX_QUEUE_SIZE) {
      // Drop oldest non-critical event
      this.queue.shift();
    }
    this.queue.push(event);
    return true;
  }

  public getQueueLength(): number {
    return this.queue.length;
  }
}

export const floodProtection = new FloodProtectionEngine();
