import { ModelRegistry } from "./modelRegistry";

export interface RollbackEvaluation {
  action: "NONE" | "SOFT_ROLLBACK" | "HARD_ROLLBACK";
  reason?: string;
  triggerTimestamp: string;
}

export class RollbackController {
  private static instance: RollbackController;
  private isSoftRollbackActive = false;

  private constructor() {}

  public static getInstance(): RollbackController {
    if (!RollbackController.instance) {
      RollbackController.instance = new RollbackController();
    }
    return RollbackController.instance;
  }

  public reset(): void {
    this.isSoftRollbackActive = false;
  }

  /**
   * Checks rollback criteria:
   * Immediate/Hard: ECE > 15%, Precision < 50%, False Positive Rate > 20%
   * Soft: DTI < 90%, DPR < 75%
   */
  public evaluateHealth(params: {
    dti: number;
    dpr: number;
    ece: number;
    precision: number;
    falsePositiveRate: number;
  }): RollbackEvaluation {
    const now = new Date().toISOString();

    if (params.ece > 0.15) {
      return {
        action: "HARD_ROLLBACK",
        reason: `Calibration Error (ECE: ${(params.ece * 100).toFixed(1)}%) exceeded critical safety limit of 15%`,
        triggerTimestamp: now,
      };
    }

    if (params.precision < 0.50) {
      return {
        action: "HARD_ROLLBACK",
        reason: `Prediction Precision (${(params.precision * 100).toFixed(1)}%) dropped below 50% threshold`,
        triggerTimestamp: now,
      };
    }

    if (params.falsePositiveRate > 0.20) {
      return {
        action: "HARD_ROLLBACK",
        reason: `Critical False Positive Rate (${(params.falsePositiveRate * 100).toFixed(1)}%) breached 20% ceiling`,
        triggerTimestamp: now,
      };
    }

    if (params.dti < 90 || params.dpr < 75) {
      return {
        action: "SOFT_ROLLBACK",
        reason: `Delta Trust Index (${params.dti}%) or DPR (${params.dpr}%) degraded below acceptable operating bands`,
        triggerTimestamp: now,
      };
    }

    return {
      action: "NONE",
      triggerTimestamp: now,
    };
  }

  public executeRollback(type: "SOFT" | "HARD"): { success: boolean; state: string; rolledBackTo?: string } {
    if (type === "SOFT") {
      this.isSoftRollbackActive = true;
      return { success: true, state: "SOFT_ROLLBACK_ENABLED" };
    }

    // Hard Rollback: Deactivate active model in registry and restore previous
    const registry = ModelRegistry.getInstance();
    const result = registry.rollbackModel();
    this.isSoftRollbackActive = false;

    return {
      success: true,
      state: "HARD_ROLLBACK_COMPLETED",
      rolledBackTo: result.rolledBackTo,
    };
  }

  public isPredictionSurfaceVisible(): boolean {
    return !this.isSoftRollbackActive;
  }
}
