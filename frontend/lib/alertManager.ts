"use client";

/**
 * Real-Time Browser Notification & Execution Trigger Alert Manager.
 * Handles desktop push notifications, Web Audio frequency synthesis,
 * and persistent threshold tracking for entry/exit levels.
 */

export interface ExecutionAlertRule {
  symbol: string;
  notifyOnBuyZone: boolean;
  notifyOnStopLossWarning: boolean;
  notifyOnTakeProfit1: boolean;
  optimalEntryMin: number;
  optimalEntryMax: number;
  stopLoss: number;
  takeProfit1: number;
  isStage4?: boolean;
  breakoutPivotPrice?: number;
  createdAt: number;
}

const STORAGE_KEY = "FINANCE_EXECUTION_ALERTS_V1";
const COOLDOWN_KEY = "FINANCE_ALERTS_COOLDOWN_V1";
const COOLDOWN_MS = 15 * 60 * 1000; // 15 minutes between duplicate alerts

export class AlertManager {
  /**
   * Request native browser notification permission.
   */
  static async requestPermission(): Promise<NotificationPermission> {
    if (typeof window === "undefined" || !("Notification" in window)) {
      return "denied";
    }
    if (Notification.permission === "granted") {
      return "granted";
    }
    return await Notification.requestPermission();
  }

  /**
   * Synthesize an institutional chime using Web Audio API (no audio files required).
   */
  static playAlertSound(type: "BUY" | "WARNING" | "SUCCESS" = "BUY"): void {
    if (typeof window === "undefined") return;
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.connect(gain);
      gain.connect(ctx.destination);

      const now = ctx.currentTime;
      if (type === "BUY") {
        // High ascending two-tone chime
        osc.type = "sine";
        osc.frequency.setValueAtTime(587.33, now); // D5
        osc.frequency.exponentialRampToValueAtTime(880.0, now + 0.15); // A5
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.4);
        osc.start(now);
        osc.stop(now + 0.4);
      } else if (type === "WARNING") {
        // Low descending warning buzz
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(440.0, now);
        osc.frequency.linearRampToValueAtTime(220.0, now + 0.25);
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
        osc.start(now);
        osc.stop(now + 0.35);
      } else {
        // Tri-tone success fanfare
        osc.type = "triangle";
        osc.frequency.setValueAtTime(523.25, now); // C5
        osc.frequency.setValueAtTime(659.25, now + 0.1); // E5
        osc.frequency.setValueAtTime(783.99, now + 0.2); // G5
        gain.gain.setValueAtTime(0.25, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
        osc.start(now);
        osc.stop(now + 0.5);
      }
    } catch (err) {
      console.warn("Web Audio alert sound not available:", err);
    }
  }

  /**
   * Retrieve all saved alert rules from localStorage.
   */
  static getAlertRules(): Record<string, ExecutionAlertRule> {
    if (typeof window === "undefined") return {};
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  }

  /**
   * Save or update an alert rule for a symbol.
   */
  static saveAlertRule(rule: ExecutionAlertRule): void {
    if (typeof window === "undefined") return;
    try {
      const rules = this.getAlertRules();
      rules[rule.symbol.toUpperCase()] = rule;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(rules));
    } catch (err) {
      console.warn("Failed to persist alert rule:", err);
    }
  }

  /**
   * Remove an alert rule.
   */
  static removeAlertRule(symbol: string): void {
    if (typeof window === "undefined") return;
    try {
      const rules = this.getAlertRules();
      delete rules[symbol.toUpperCase()];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(rules));
    } catch {}
  }

  /**
   * Evaluate incoming price update against active alert rules and fire notification if triggered.
   */
  static checkAndFireAlerts(
    symbol: string,
    currentPrice: number,
    levels?: {
      optimalEntryMin?: number;
      optimalEntryMax?: number;
      stopLoss?: number;
      takeProfit1?: number;
    }
  ): void {
    if (typeof window === "undefined" || currentPrice <= 0) return;
    const sym = symbol.toUpperCase();
    const rules = this.getAlertRules();
    const rule = rules[sym];

    if (!rule) return;

    const entryMin = levels?.optimalEntryMin ?? rule.optimalEntryMin;
    const entryMax = levels?.optimalEntryMax ?? rule.optimalEntryMax;
    const stopLoss = levels?.stopLoss ?? rule.stopLoss;
    const tp1 = levels?.takeProfit1 ?? rule.takeProfit1;

    const now = Date.now();
    let cooldowns: Record<string, number> = {};
    try {
      const raw = localStorage.getItem(COOLDOWN_KEY);
      cooldowns = raw ? JSON.parse(raw) : {};
    } catch {}

    const shouldNotify = (key: string) => {
      const last = cooldowns[key] || 0;
      return now - last > COOLDOWN_MS;
    };

    const recordCooldown = (key: string) => {
      cooldowns[key] = now;
      try {
        localStorage.setItem(COOLDOWN_KEY, JSON.stringify(cooldowns));
      } catch {}
    };

    // 1. Check Buy Zone Entry OR Stage 4 Breakout Pivot Reclaim
    if (rule.notifyOnBuyZone) {
      if (rule.isStage4 && rule.breakoutPivotPrice && rule.breakoutPivotPrice > 0) {
        if (currentPrice >= rule.breakoutPivotPrice * 0.998) {
          const key = `${sym}_STAGE4_BREAKOUT`;
          if (shouldNotify(key)) {
            recordCooldown(key);
            this.playAlertSound("BUY");
            this.sendDesktopNotification(
              `⏳ ${sym} 50-Day SMA Breakout Confirmed!`,
              `Current Spot: $${currentPrice.toFixed(2)} has reclaimed the 50-day moving average breakout pivot ($${rule.breakoutPivotPrice.toFixed(2)})!`
            );
          }
        }
      } else if (entryMax > 0 && entryMin > 0) {
        if (currentPrice <= entryMax * 1.01 && currentPrice >= entryMin * 0.99) {
          const key = `${sym}_BUY_ZONE`;
          if (shouldNotify(key)) {
            recordCooldown(key);
            this.playAlertSound("BUY");
            this.sendDesktopNotification(
              `🎯 ${sym} in Optimal Buy Zone!`,
              `Current Spot: $${currentPrice.toFixed(2)} is inside the accumulation target zone ($${entryMin.toFixed(2)} - $${entryMax.toFixed(2)}).`
            );
          }
        }
      }
    }

    // 2. Check Stop Loss Warning (within 1.0% of stop loss)
    if (rule.notifyOnStopLossWarning && stopLoss > 0) {
      if (currentPrice <= stopLoss * 1.015 && currentPrice >= stopLoss * 0.98) {
        const key = `${sym}_STOP_LOSS`;
        if (shouldNotify(key)) {
          recordCooldown(key);
          this.playAlertSound("WARNING");
          this.sendDesktopNotification(
            `🛑 ${sym} Stop-Loss Warning!`,
            `Current Spot: $${currentPrice.toFixed(2)} is within 1% of statutory invalidation floor ($${stopLoss.toFixed(2)}).`
          );
        }
      }
    }

    // 3. Check Take Profit 1 Target
    if (rule.notifyOnTakeProfit1 && tp1 > 0) {
      if (currentPrice >= tp1 * 0.99) {
        const key = `${sym}_TAKE_PROFIT`;
        if (shouldNotify(key)) {
          recordCooldown(key);
          this.playAlertSound("SUCCESS");
          this.sendDesktopNotification(
            `🚀 ${sym} Approaching Target 1!`,
            `Current Spot: $${currentPrice.toFixed(2)} has reached the Take-Profit ladder target ($${tp1.toFixed(2)}).`
          );
        }
      }
    }
  }

  /**
   * Helper to dispatch native browser notification.
   */
  private static sendDesktopNotification(title: string, body: string): void {
    if (typeof window === "undefined" || !("Notification" in window)) return;
    if (Notification.permission === "granted") {
      try {
        new Notification(title, {
          body,
          icon: "/favicon.ico",
        });
      } catch (err) {
        console.warn("Failed to create desktop notification:", err);
      }
    }
  }
}
