import {
  CalibrationReport,
  ModelMetadata,
  ModelStatus,
} from "../../types/predictive-intelligence";

export class ModelRegistry {
  private static instance: ModelRegistry;
  private models: Map<string, ModelMetadata> = new Map();
  private versionIndex: Map<string, string> = new Map(); // modelName:version -> modelId
  private activeModelId: string | null = null;
  private previousActiveModelId: string | null = null;

  private constructor() {
    this.seedDefaultModel();
  }

  public static getInstance(): ModelRegistry {
    if (!ModelRegistry.instance) {
      ModelRegistry.instance = new ModelRegistry();
    }
    return ModelRegistry.instance;
  }

  public reset(): void {
    this.models.clear();
    this.versionIndex.clear();
    this.activeModelId = null;
    this.previousActiveModelId = null;
    this.seedDefaultModel();
  }

  private seedDefaultModel(): void {
    const defaultModel: ModelMetadata = {
      modelId: "mdl-v1.0.0",
      modelName: "arx-attention-predictor",
      version: "v1.0.0",
      checksum: "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
      registeredAt: "2026-09-01T00:00:00Z",
      activatedAt: "2026-09-01T00:00:00Z",
      status: ModelStatus.ACTIVE,
      calibration: {
        ece: 0.034,
        brierScore: 0.12,
        samples: 1482,
        buckets: [],
        status: "PASS",
      },
    };
    this.models.set(defaultModel.modelId, defaultModel);
    this.versionIndex.set(`${defaultModel.modelName}:${defaultModel.version}`, defaultModel.modelId);
    this.activeModelId = defaultModel.modelId;
  }

  public registerModel(params: {
    modelName: string;
    version: string;
    checksum: string;
    calibration?: CalibrationReport;
  }): ModelMetadata {
    const key = `${params.modelName}:${params.version}`;
    if (this.versionIndex.has(key)) {
      throw new Error("VERSION_ALREADY_EXISTS");
    }

    const modelId = `mdl-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`;
    const model: ModelMetadata = {
      modelId,
      modelName: params.modelName,
      version: params.version,
      checksum: params.checksum,
      registeredAt: new Date().toISOString(),
      status: ModelStatus.REGISTERED,
      calibration: params.calibration,
    };

    this.models.set(modelId, model);
    this.versionIndex.set(key, modelId);
    return model;
  }

  public promoteModel(modelId: string): ModelMetadata {
    const model = this.models.get(modelId);
    if (!model) {
      throw new Error("MODEL_NOT_FOUND");
    }

    // Safety Gate: Must have valid calibration meeting ECE <= 0.05 and Brier <= 0.15
    if (
      !model.calibration ||
      model.calibration.status !== "PASS" ||
      model.calibration.ece > 0.05 ||
      model.calibration.brierScore > 0.15
    ) {
      throw new Error("VALIDATION_REQUIRED");
    }

    // Deactivate previous active model
    if (this.activeModelId && this.activeModelId !== modelId) {
      const currentActive = this.models.get(this.activeModelId);
      if (currentActive) {
        currentActive.status = ModelStatus.DEPRECATED;
      }
      this.previousActiveModelId = this.activeModelId;
    }

    model.status = ModelStatus.ACTIVE;
    model.activatedAt = new Date().toISOString();
    this.activeModelId = modelId;
    return model;
  }

  public rollbackModel(targetModelId?: string): { rolledBackTo: string; status: "ROLLED_BACK" } {
    if (!this.activeModelId) {
      throw new Error("NO_ACTIVE_MODEL_TO_ROLLBACK");
    }

    const currentActive = this.models.get(this.activeModelId);
    if (currentActive) {
      currentActive.status = ModelStatus.ROLLED_BACK;
    }

    let restoreId = targetModelId;
    if (!restoreId && this.previousActiveModelId) {
      restoreId = this.previousActiveModelId;
    }

    if (restoreId && this.models.has(restoreId)) {
      const restored = this.models.get(restoreId)!;
      restored.status = ModelStatus.ACTIVE;
      this.activeModelId = restoreId;
      return { rolledBackTo: restored.version, status: "ROLLED_BACK" };
    }

    // Fallback to default v1.0.0
    const defaultModel = this.models.get("mdl-v1.0.0");
    if (defaultModel) {
      defaultModel.status = ModelStatus.ACTIVE;
      this.activeModelId = defaultModel.modelId;
      return { rolledBackTo: defaultModel.version, status: "ROLLED_BACK" };
    }

    throw new Error("ROLLBACK_TARGET_NOT_FOUND");
  }

  public getActiveModel(): ModelMetadata | null {
    if (!this.activeModelId) return null;
    return this.models.get(this.activeModelId) || null;
  }

  public getModel(modelId: string): ModelMetadata | null {
    return this.models.get(modelId) || null;
  }

  public getCalibrationReport(modelId: string): CalibrationReport {
    const model = this.models.get(modelId);
    if (!model) throw new Error("MODEL_NOT_FOUND");
    if (!model.calibration) {
      throw new Error("NO_CALIBRATION_AVAILABLE");
    }
    return model.calibration;
  }
}
