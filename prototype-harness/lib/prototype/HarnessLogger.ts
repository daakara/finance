/**
 * ARX Terminal Prototype Sandbox — Interaction Logger & Epistemic Telemetry
 *
 * Captures monotonic duration, granular UI events, and compiles session records.
 * Generates SHA-256 session manifest digests over exported JSON Lines data.
 */

export interface InteractionEvent {
  event_seq: number;
  event_type:
    | "TASK_RENDERED"
    | "TAB_CHANGED"
    | "INPUT_CHANGED"
    | "ACCORDION_TOGGLED"
    | "ORDER_ACTION_CLICKED"
    | "TASK_SUBMITTED"
    | "TIMEOUT_TRIGGERED"
    | "TASK_ABANDONED";
  monotonic_ms: number;
  wall_clock_iso: string;
  details: Record<string, any>;
}

export interface TaskSessionRecord {
  scenario_id: string;
  scenario_version: string;
  participant_id: string;
  participant_archetype: string;
  experiment_block: "BLOCK_1_TRADE_PLANNING" | "BLOCK_2_EVIDENCE_DEPTH";
  architecture_variant: "CONCEPT_A" | "CONCEPT_B" | "CONCEPT_C";
  sequence_position: number;
  authoritative_expected_state: string;
  participant_stated_state: string;
  participant_proposed_action: string;
  participant_explanation: string;
  risk_identification_result: "DETECTED" | "PARTIALLY_DETECTED" | "MISSED" | "INCORRECT";
  data_freshness_interpretation: "CORRECT_LIVE" | "CORRECT_STALE" | "MISIDENTIFIED";
  confidence_rating: number; // 1 to 5
  response_elapsed_ms: number; // Monotonic duration
  completion_status: "SUCCESS" | "ERROR" | "TIMEOUT" | "ABANDONED";
  canonical_cm_codes: string[];
  supplemental_sm_codes: string[];
  observer_1_classification?: string;
  observer_2_classification?: string;
  adjudicated_classification?: string;
  events: InteractionEvent[];
}

export class HarnessLogger {
  private sessionRecords: TaskSessionRecord[] = [];
  private currentEvents: InteractionEvent[] = [];
  private taskStartMonotonic: number = 0;
  private eventCounter: number = 0;
  private currentTrialIdx: number = 0;
  private durableBuffer: any = null;

  setDurableBuffer(buffer: any) {
    this.durableBuffer = buffer;
  }

  setCurrentTrialIdx(idx: number) {
    this.currentTrialIdx = idx;
  }

  restoreSessionRecords(records: TaskSessionRecord[]) {
    this.sessionRecords = [...records];
  }

  startTask() {
    this.currentEvents = [];
    this.eventCounter = 0;
    this.taskStartMonotonic = typeof performance !== "undefined" ? performance.now() : Date.now();
    this.logEvent("TASK_RENDERED", { task_initialized: true });
  }

  logEvent(eventType: InteractionEvent["event_type"], details: Record<string, any> = {}) {
    this.eventCounter += 1;
    const nowMono = typeof performance !== "undefined" ? performance.now() : Date.now();
    const event: InteractionEvent = {
      event_seq: this.eventCounter,
      event_type: eventType,
      monotonic_ms: Math.round((nowMono - this.taskStartMonotonic) * 100) / 100,
      wall_clock_iso: new Date().toISOString(),
      details,
    };
    this.currentEvents.push(event);

    if (this.durableBuffer && typeof this.durableBuffer.storeEvent === "function") {
      this.durableBuffer.storeEvent(event, this.currentTrialIdx).catch((err: any) => {
        console.error("Durable buffer storeEvent failed:", err);
      });
    }
  }

  completeTask(recordData: Omit<TaskSessionRecord, "response_elapsed_ms" | "events">): TaskSessionRecord {
    const endMono = typeof performance !== "undefined" ? performance.now() : Date.now();
    const elapsedMs = Math.round((endMono - this.taskStartMonotonic) * 100) / 100;

    const fullRecord: TaskSessionRecord = {
      ...recordData,
      response_elapsed_ms: elapsedMs,
      events: [...this.currentEvents],
    };

    this.sessionRecords.push(fullRecord);

    if (this.durableBuffer && typeof this.durableBuffer.storeTrialRecord === "function") {
      this.durableBuffer.storeTrialRecord(fullRecord).catch((err: any) => {
        console.error("Durable buffer storeTrialRecord failed:", err);
      });
    }

    return fullRecord;
  }

  getSessionRecords(): TaskSessionRecord[] {
    return [...this.sessionRecords];
  }

  async exportSessionBundle(participantId: string): Promise<{
    jsonlContent: string;
    sha256Digest: string;
    recordCount: number;
  }> {
    const lines = this.sessionRecords.map((r) => JSON.stringify(r));
    const jsonlContent = lines.join("\n") + "\n";

    // Compute SHA-256 via browser crypto.subtle
    const enc = new TextEncoder();
    const data = enc.encode(jsonlContent);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const sha256Digest = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");

    return {
      jsonlContent,
      sha256Digest,
      recordCount: this.sessionRecords.length,
    };
  }
}
