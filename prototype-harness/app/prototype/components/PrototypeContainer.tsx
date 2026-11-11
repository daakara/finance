"use client";

import React, { useState, useEffect, useRef } from "react";
import { ScenarioFixture, loadScenarioFixture, SCENARIO_FILE_MAP } from "@/lib/prototype/scenarioLoader";
import { HarnessLogger, TaskSessionRecord } from "@/lib/prototype/HarnessLogger";
import { downloadSessionFiles } from "@/lib/prototype/sessionExporter";
import { DurableBuffer } from "@/lib/prototype/durableBuffer";
import { PersistentBanner } from "./PersistentBanner";
import { TaskPrompter } from "./TaskPrompter";
import { VariantRunner, VariantType } from "./VariantRunner";
import { NASAWorkloadModal, TLXScores } from "./NASAWorkloadModal";

interface TaskPlan {
  taskId: string;
  taskTitle: string;
  taskInstructions: string;
  maxDurationSec: number;
  block: "BLOCK_1_TRADE_PLANNING" | "BLOCK_2_EVIDENCE_DEPTH";
  variant: VariantType;
  scenarioId: string;
  authoritativeExpectedState: string;
}

// Master schedules from ARX_UX_EXPERIMENT_ASSIGNMENT_V4.md (Sanitized for Blinding Integrity)
const PARTICIPANT_SCHEDULES: Record<string, TaskPlan[]> = {
  P01: [
    // Block 1: Concept A (PLSE) -> Concept B (ALAB)
    { taskId: "T01", taskTitle: "Price & Market Data Verification", taskInstructions: "Evaluate spot price, market data status, and quote freshness for PLSE.", maxDurationSec: 180, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_A", scenarioId: "SCN-NORM-PLSE-SWING", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T04", taskTitle: "Position Sizing & Invalidation", taskInstructions: "Identify invalidation level and determine position sizing for $1,000 risk budget.", maxDurationSec: 180, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_A", scenarioId: "SCN-NORM-PLSE-SWING", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T05", taskTitle: "Technical Pattern Evaluation", taskInstructions: "Evaluate consolidation structure, volume pattern, and entry trigger for ALAB.", maxDurationSec: 120, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_B", scenarioId: "SCN-NORM-ALAB-MOM", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T06", taskTitle: "System Health & Telemetry Verification", taskInstructions: "Review asset against data telemetry, pipeline health, and potential execution constraints.", maxDurationSec: 150, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_B", scenarioId: "SCN-NORM-ALAB-MOM", authoritativeExpectedState: "ACTIONABLE" },
    // Block 2: Concept B (FIX) -> Concept C (DUOL)
    { taskId: "T02", taskTitle: "Recommendation & Rationale Review", taskInstructions: "Examine verdict presentation, sizing parameters, and invalidation rules for FIX.", maxDurationSec: 180, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_B", scenarioId: "SCN-NORM-FIX-VCP", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T03", taskTitle: "Evidence & Research Evaluation", taskInstructions: "Conduct fundamental due diligence and evaluate research memo, core thesis, and counter-arguments for DUOL.", maxDurationSec: 240, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_C", scenarioId: "SCN-NORM-DUOL-FUND", authoritativeExpectedState: "EVIDENCE_INCOMPLETE" },
    { taskId: "T07", taskTitle: "Cross-Model Evidence Reconciliation", taskInstructions: "Review technical, fundamental, and data feed indications to determine overall order readiness.", maxDurationSec: 240, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_C", scenarioId: "SCN-ADVR-STALE-TAPE", authoritativeExpectedState: "UNVERIFIED" },
  ],
  P02: [
    // Block 1: Concept B (PLSE) -> Concept A (ALAB)
    { taskId: "T01", taskTitle: "Price & Market Data Verification", taskInstructions: "Evaluate spot price, market data status, and quote freshness for PLSE.", maxDurationSec: 180, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_B", scenarioId: "SCN-NORM-PLSE-SWING", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T04", taskTitle: "Position Sizing & Invalidation", taskInstructions: "Identify invalidation level and determine position sizing for $1,000 risk budget.", maxDurationSec: 180, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_B", scenarioId: "SCN-NORM-PLSE-SWING", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T05", taskTitle: "Technical Pattern Evaluation", taskInstructions: "Evaluate consolidation structure, volume pattern, and entry trigger for ALAB.", maxDurationSec: 120, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_A", scenarioId: "SCN-NORM-ALAB-MOM", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T06", taskTitle: "System Health & Telemetry Verification", taskInstructions: "Review asset against data telemetry, pipeline health, and potential execution constraints.", maxDurationSec: 150, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_A", scenarioId: "SCN-NORM-ALAB-MOM", authoritativeExpectedState: "ACTIONABLE" },
    // Block 2: Concept C (FIX) -> Concept B (DUOL)
    { taskId: "T02", taskTitle: "Recommendation & Rationale Review", taskInstructions: "Examine verdict presentation, sizing parameters, and invalidation rules for FIX.", maxDurationSec: 180, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_C", scenarioId: "SCN-NORM-FIX-VCP", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "T03", taskTitle: "Evidence & Research Evaluation", taskInstructions: "Conduct fundamental due diligence and evaluate research memo, core thesis, and counter-arguments for DUOL.", maxDurationSec: 240, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_B", scenarioId: "SCN-NORM-DUOL-FUND", authoritativeExpectedState: "EVIDENCE_INCOMPLETE" },
    { taskId: "T07", taskTitle: "Cross-Model Evidence Reconciliation", taskInstructions: "Review technical, fundamental, and data feed indications to determine overall order readiness.", maxDurationSec: 240, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_B", scenarioId: "SCN-ADVR-CONTRADICT", authoritativeExpectedState: "UNVERIFIED" },
  ],
  "P00-SYNTHETIC": [
    // Technical Rehearsal: 9 trials across all 9 scenarios
    { taskId: "SYN-01", taskTitle: "PLSE Position Staging", taskInstructions: "Review trade staging and market parameters for PLSE.", maxDurationSec: 60, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_A", scenarioId: "SCN-NORM-PLSE-SWING", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "SYN-02", taskTitle: "FIX Volatility & Sizing Review", taskInstructions: "Examine position sizing calculations and risk parameters for FIX.", maxDurationSec: 60, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_B", scenarioId: "SCN-NORM-FIX-VCP", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "SYN-03", taskTitle: "ALAB Breakout Evaluation", taskInstructions: "Evaluate technical corridor and breakout setup levels for ALAB.", maxDurationSec: 60, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_A", scenarioId: "SCN-NORM-ALAB-MOM", authoritativeExpectedState: "ACTIONABLE" },
    { taskId: "SYN-04", taskTitle: "DUOL Fundamental Research Dossier", taskInstructions: "Review the fundamental underwriting memo and research points for DUOL.", maxDurationSec: 60, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_C", scenarioId: "SCN-NORM-DUOL-FUND", authoritativeExpectedState: "EVIDENCE_INCOMPLETE" },
    { taskId: "SYN-05", taskTitle: "Market Data Freshness Audit", taskInstructions: "Evaluate market data status, quote timestamp, and order readiness.", maxDurationSec: 60, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_B", scenarioId: "SCN-ADVR-STALE-TAPE", authoritativeExpectedState: "UNVERIFIED" },
    { taskId: "SYN-06", taskTitle: "Corporate Action Evidence Review", taskInstructions: "Evaluate company disclosures and corporate action evidence for consistency.", maxDurationSec: 60, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_C", scenarioId: "SCN-ADVR-CONTRADICT", authoritativeExpectedState: "UNVERIFIED" },
    { taskId: "SYN-07", taskTitle: "Technical Anchor Verification", taskInstructions: "Inspect technical base pattern and verify structural pivot anchors.", maxDurationSec: 60, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_B", scenarioId: "SCN-ADVR-PHANTOM", authoritativeExpectedState: "UNVERIFIED" },
    { taskId: "SYN-08", taskTitle: "Screening Status & Execution Gate", taskInstructions: "Review screening status and evaluate whether criteria for execution are met.", maxDurationSec: 60, block: "BLOCK_1_TRADE_PLANNING", variant: "CONCEPT_B", scenarioId: "SCN-ADVR-SCREEN-ONLY", authoritativeExpectedState: "UNVERIFIED" },
    { taskId: "SYN-09", taskTitle: "Pipeline Telemetry & Feed Audit", taskInstructions: "Inspect system telemetry and batch pipeline indicators.", maxDurationSec: 60, block: "BLOCK_2_EVIDENCE_DEPTH", variant: "CONCEPT_C", scenarioId: "SCN-ADVR-PIPE-FAIL", authoritativeExpectedState: "UNVERIFIED" },
  ],
};

export function PrototypeContainer() {
  const [selectedParticipant, setSelectedParticipant] = useState<string>("P00-SYNTHETIC");
  const [sessionActive, setSessionActive] = useState<boolean>(false);
  const [currentTrialIdx, setCurrentTrialIdx] = useState<number>(0);
  const [currentFixture, setCurrentFixture] = useState<ScenarioFixture | null>(null);
  const [isLoadingFixture, setIsLoadingFixture] = useState<boolean>(false);
  const [fixtureError, setFixtureError] = useState<string | null>(null);

  // Cognitive reset break state
  const [isOnBreak, setIsOnBreak] = useState<boolean>(false);
  const [breakRemainingSec, setBreakRemainingSec] = useState<number>(480); // 8 minutes = 480 seconds

  // NASA-TLX state
  const [showTLXModal, setShowTLXModal] = useState<boolean>(false);

  // Research Session & Custody Lifecycle State
  const [sessionLifecycleState, setSessionLifecycleState] = useState<
    | "SESSION_IDLE"
    | "SESSION_ACTIVE"
    | "CUSTODY_TRANSFER_PENDING"
    | "CUSTODY_RECEIPT_VERIFIED"
    | "SESSION_SEALED"
    | "SESSION_HOLD_EVIDENCE_INTEGRITY"
    | "SESSION_WITHDRAWN"
  >("SESSION_IDLE");
  const [sessionCompleted, setSessionCompleted] = useState<boolean>(false);
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [custodyReceipt, setCustodyReceipt] = useState<any | null>(null);
  const [custodyError, setCustodyError] = useState<string | null>(null);
  const [bootstrapPasskey, setBootstrapPasskey] = useState<string>("");
  const [isEnrolled, setIsEnrolled] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string>(() => "SESS-" + Math.random().toString(36).substring(2, 10).toUpperCase());
  const [sessionExportData, setSessionExportData] = useState<{
    jsonlContent: string;
    sha256Digest: string;
    recordCount: number;
  } | null>(null);

  const loggerRef = useRef<HarnessLogger>(new HarnessLogger());
  const durableBufferRef = useRef<DurableBuffer>(new DurableBuffer());

  const currentPlanList: TaskPlan[] = PARTICIPANT_SCHEDULES[selectedParticipant] || PARTICIPANT_SCHEDULES["P00-SYNTHETIC"];
  const currentTask: TaskPlan | undefined = currentPlanList[currentTrialIdx];

  // Initialize DurableBuffer & check for existing unsealed journal on mount
  useEffect(() => {
    loggerRef.current.setDurableBuffer(durableBufferRef.current);
    durableBufferRef.current
      .initBuffer(sessionId, selectedParticipant)
      .then((res: any) => {
        if (res.sessionToken) {
          setIsEnrolled(true);
        }
        if (res.recovered) {
          if (res.sessionId) setSessionId(res.sessionId);
          if (res.participantId) setSelectedParticipant(res.participantId);
          durableBufferRef.current.recoverActiveJournal().then(({ trialRecords }) => {
            if (trialRecords && trialRecords.length > 0) {
              loggerRef.current.restoreSessionRecords(trialRecords);
              setCurrentTrialIdx(trialRecords.length);
              setSessionLifecycleState("SESSION_ACTIVE");
              setSessionActive(true);
              console.log(`[DURABILITY] Recovered ${trialRecords.length} completed trials from IndexedDB journal.`);
            }
          });
        }
      })
      .catch((err) => {
        console.warn("[DURABILITY] Buffer init warning:", err);
      });
  }, []);

  // Load fixture whenever currentTask changes during an active session
  useEffect(() => {
    if (!sessionActive || !currentTask || isOnBreak || showTLXModal || sessionCompleted) return;

    let isMounted = true;
    setIsLoadingFixture(true);
    setFixtureError(null);

    loadScenarioFixture(currentTask.scenarioId)
      .then((fixture) => {
        if (!isMounted) return;
        const safeFixture: ScenarioFixture = {
          ...fixture,
          technical_context: fixture.technical_context || ({} as any),
        };
        setCurrentFixture(safeFixture);
        setIsLoadingFixture(false);
        loggerRef.current.startTask();
      })
      .catch((err) => {
        if (!isMounted) return;
        setFixtureError(err.message);
        setIsLoadingFixture(false);
      });

    return () => {
      isMounted = false;
    };
  }, [sessionActive, currentTrialIdx, currentTask, isOnBreak, showTLXModal, sessionCompleted]);

  // Handle break countdown timer
  useEffect(() => {
    if (!isOnBreak) return;
    const interval = setInterval(() => {
      setBreakRemainingSec((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setIsOnBreak(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isOnBreak]);

  const handleStartSession = async () => {
    loggerRef.current = new HarnessLogger();
    loggerRef.current.setDurableBuffer(durableBufferRef.current);
    loggerRef.current.setCurrentTrialIdx(0);
    setCurrentTrialIdx(0);
    setSessionCompleted(false);
    setIsOnBreak(false);
    setShowTLXModal(false);
    setSessionExportData(null);
    setCustodyReceipt(null);
    setCustodyError(null);

    // If bootstrap token entered and not yet enrolled, perform handshake
    if (bootstrapPasskey && !isEnrolled) {
      try {
        const token = await durableBufferRef.current.performHandshake(
          bootstrapPasskey,
          sessionId,
          selectedParticipant
        );
        durableBufferRef.current.setSessionToken(token);
        setIsEnrolled(true);
      } catch (err: any) {
        setCustodyError("Bootstrap enrollment failed: " + (err.message || String(err)));
        setSessionLifecycleState("SESSION_HOLD_EVIDENCE_INTEGRITY");
        return;
      }
    }

    setSessionLifecycleState("SESSION_ACTIVE");
    setSessionActive(true);
  };

  const handleInteraction = (eventType: string, details: any) => {
    if (!sessionActive) return;
    // Map arbitrary event strings to logger types
    loggerRef.current.logEvent(
      eventType === "TAB_CHANGED" ? "TAB_CHANGED" :
      eventType === "INPUT_CHANGED" ? "INPUT_CHANGED" :
      eventType === "SECTION_TOGGLED" || eventType === "MEMO_ITEM_TOGGLED" ? "ACCORDION_TOGGLED" :
      "INPUT_CHANGED",
      details
    );
  };

  const handleOrderAction = () => {
    if (!currentFixture) return;
    loggerRef.current.logEvent("ORDER_ACTION_CLICKED", {
      ticker: currentFixture.ticker,
      spot: currentFixture.market_data.spot_price,
    });

    // Offline string clipboard copy (Zero broker routing)
    const orderString = `[ARX OFFLINE SPEC] TICKER=${currentFixture.ticker} ACTION=BUY_LIMIT PRICE=${currentFixture.market_data.spot_price.toFixed(2)} STOP=${currentFixture.technical_context.structural_invalidation_floor || 0} PROVENANCE=${currentFixture.market_data_provenance}`;

    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(orderString).catch(() => {});
    }

    setCopyFeedback("✓ Order parameters copied to clipboard");
    setTimeout(() => {
      setCopyFeedback(null);
    }, 4000);
  };

  const advanceToNextTrial = () => {
    const nextIdx = currentTrialIdx + 1;
    loggerRef.current.setCurrentTrialIdx(nextIdx);
    if (nextIdx >= currentPlanList.length) {
      // Reached end of session
      finishSession();
    } else {
      const prevBlock = currentPlanList[currentTrialIdx].block;
      const nextBlock = currentPlanList[nextIdx].block;

      // If crossing from Block 1 to Block 2, trigger 8-minute cognitive reset break
      if (prevBlock === "BLOCK_1_TRADE_PLANNING" && nextBlock === "BLOCK_2_EVIDENCE_DEPTH") {
        setShowTLXModal(true);
      } else {
        setCurrentTrialIdx(nextIdx);
      }
    }
  };

  const handleTLXSubmit = (scores: TLXScores) => {
    loggerRef.current.logEvent("TASK_SUBMITTED", { tlx_scores: scores });
    setShowTLXModal(false);

    // Begin 8-minute reset break
    setIsOnBreak(true);
    setBreakRemainingSec(480);
  };

  const handleEndBreak = () => {
    setIsOnBreak(false);
    setCurrentTrialIdx((prev) => prev + 1);
  };

  const handleTaskSubmit = (result: {
    participantStatedState: string;
    participantProposedAction: string;
    participantExplanation: string;
    confidenceRating: number;
  }) => {
    if (!currentTask || !currentFixture) return;

    loggerRef.current.logEvent("TASK_SUBMITTED", result);

    loggerRef.current.completeTask({
      scenario_id: currentFixture.fixture_id,
      scenario_version: currentFixture.version,
      participant_id: selectedParticipant,
      participant_archetype: "Quantitative Trader / Researcher",
      experiment_block: currentTask.block,
      architecture_variant: currentTask.variant,
      sequence_position: currentTrialIdx + 1,
      authoritative_expected_state: currentTask.authoritativeExpectedState,
      participant_stated_state: result.participantStatedState,
      participant_proposed_action: result.participantProposedAction,
      participant_explanation: result.participantExplanation,
      risk_identification_result: result.participantStatedState === currentTask.authoritativeExpectedState ? "DETECTED" : "MISSED",
      data_freshness_interpretation: currentFixture.market_data.is_stale ? "CORRECT_STALE" : "CORRECT_LIVE",
      confidence_rating: result.confidenceRating,
      completion_status: "SUCCESS",
      canonical_cm_codes: currentFixture.canonical_cm_codes || [],
      supplemental_sm_codes: currentFixture.supplemental_sm_codes || [],
    });

    advanceToNextTrial();
  };

  const handleTimeout = () => {
    if (!currentTask || !currentFixture) return;

    loggerRef.current.logEvent("TIMEOUT_TRIGGERED", { trialIdx: currentTrialIdx });

    loggerRef.current.completeTask({
      scenario_id: currentFixture.fixture_id,
      scenario_version: currentFixture.version,
      participant_id: selectedParticipant,
      participant_archetype: "Quantitative Trader / Researcher",
      experiment_block: currentTask.block,
      architecture_variant: currentTask.variant,
      sequence_position: currentTrialIdx + 1,
      authoritative_expected_state: currentTask.authoritativeExpectedState,
      participant_stated_state: "TIMEOUT_EXPIRED",
      participant_proposed_action: "NONE",
      participant_explanation: "Time ceiling reached without submission.",
      risk_identification_result: "MISSED",
      data_freshness_interpretation: "MISIDENTIFIED",
      confidence_rating: 1,
      completion_status: "TIMEOUT",
      canonical_cm_codes: currentFixture.canonical_cm_codes || [],
      supplemental_sm_codes: currentFixture.supplemental_sm_codes || [],
    });

    advanceToNextTrial();
  };

  const finishSession = async () => {
    setSessionActive(false);
    setSessionCompleted(true);
    setSessionLifecycleState("CUSTODY_TRANSFER_PENDING");
    setCustodyError(null);

    const bundle = await loggerRef.current.exportSessionBundle(selectedParticipant);
    setSessionExportData(bundle);

    await transferEvidenceToCustody(bundle);
  };

  const transferEvidenceToCustody = async (bundle: { jsonlContent: string; sha256Digest: string }) => {
    try {
      let token = durableBufferRef.current.getSessionToken();
      if (!token && bootstrapPasskey) {
        token = await durableBufferRef.current.performHandshake(bootstrapPasskey, sessionId, selectedParticipant);
        durableBufferRef.current.setSessionToken(token);
        setIsEnrolled(true);
      }

      if (!token) {
        setCustodyError("Moderator session authorization token required for custody transfer.");
        setSessionLifecycleState("SESSION_HOLD_EVIDENCE_INTEGRITY");
        return;
      }

      // Step 1: Automated Push to Intake Endpoint
      const res = await fetch("/api/research/custody/session_ingest", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-jsonlines",
          Authorization: `Bearer ${token}`,
        },
        body: bundle.jsonlContent,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || errData.details || `Intake rejected (HTTP ${res.status})`);
      }

      const receipt = await res.json();

      // Step 2: Cryptographic Server Receipt Verification
      const verifyRes = await fetch("/api/research/custody/verify_receipt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(receipt),
      });

      if (!verifyRes.ok) {
        throw new Error("Server-side custody receipt verification failed.");
      }

      const verifyData = await verifyRes.json();
      if (!verifyData.verified) {
        throw new Error("Custody receipt HMAC signature failed verification against ledger.");
      }

      setCustodyReceipt(receipt);
      setSessionLifecycleState("SESSION_SEALED");

      // Step 3: Verified Sealed Acceptance -> Zero-Fill Purge Unsealed Staging Buffer
      await durableBufferRef.current.purgeActiveJournal(true);
    } catch (err: any) {
      console.error("Custody transfer error:", err);
      setCustodyError(err.message || "Failed to seal evidence into local custody vault.");
      setSessionLifecycleState("SESSION_HOLD_EVIDENCE_INTEGRITY");
    }
  };

  const handleWithdraw = async () => {
    const confirmed = typeof window !== "undefined" && window.confirm(
      "Are you sure you want to withdraw? All local interaction telemetry will be immediately zero-filled and wiped."
    );
    if (!confirmed) return;

    try {
      const token = durableBufferRef.current.getSessionToken();
      if (token) {
        await fetch("/api/research/custody/withdraw", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            session_id: sessionId,
            participant_id: selectedParticipant,
            reason: "PARTICIPANT_INITIATED_WITHDRAWAL",
          }),
        }).catch(() => {});
      }
    } finally {
      await durableBufferRef.current.purgeActiveJournal(true);
      setSessionActive(false);
      setSessionCompleted(true);
      setSessionLifecycleState("SESSION_WITHDRAWN");
    }
  };

  const retryCustodyTransfer = async () => {
    if (!sessionExportData) {
      const bundle = await loggerRef.current.exportSessionBundle(selectedParticipant);
      setSessionExportData(bundle);
      await transferEvidenceToCustody(bundle);
    } else {
      await transferEvidenceToCustody(sessionExportData);
    }
  };

  const handleDownloadFiles = () => {
    if (!sessionExportData) return;
    downloadSessionFiles(selectedParticipant, sessionExportData.jsonlContent, sessionExportData.sha256Digest);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Simulation Watermark Banner */}
      <PersistentBanner />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6 flex flex-col gap-6">
        {/* Navigation & Controls Bar */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="text-sm font-mono font-bold text-emerald-400">
              ARX RESEARCH SANDBOX
            </span>
            <span className="text-xs text-slate-500 font-mono">v4.0.0-phase2</span>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <label htmlFor="participantSelect" className="text-slate-400 font-mono">PARTICIPANT:</label>
              <select
                id="participantSelect"
                disabled={sessionActive}
                value={selectedParticipant}
                onChange={(e) => setSelectedParticipant(e.target.value)}
                className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-slate-200 font-mono text-xs disabled:opacity-50"
              >
                <option value="P00-SYNTHETIC">P00-SYNTHETIC (Automated Rehearsal)</option>
                <option value="P01">P01 (Quant Swing - Forward)</option>
                <option value="P02">P02 (Quant Swing - Inverted)</option>
                <option value="P03">P03 (Fundamental - Forward)</option>
                <option value="P04">P04 (Fundamental - Inverted)</option>
                <option value="P05">P05 (Quant Swing - Forward)</option>
                <option value="P06">P06 (Risk Manager - Inverted)</option>
              </select>
            </div>

            {!sessionActive && !sessionCompleted && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <label htmlFor="bootstrapPasskey" className="text-slate-400 font-mono">PASSKEY:</label>
                  <input
                    id="bootstrapPasskey"
                    type="password"
                    placeholder="Bootstrap passkey"
                    value={bootstrapPasskey}
                    onChange={(e) => setBootstrapPasskey(e.target.value)}
                    className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs w-32"
                  />
                </div>
                <button
                  type="button"
                  onClick={handleStartSession}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold px-4 py-1.5 rounded transition-colors shadow"
                >
                  Start Assigned Protocol
                </button>
              </div>
            )}

            {sessionActive && (
              <div className="flex items-center gap-4 font-mono">
                <span className="text-slate-400">
                  TRIAL: <strong className="text-slate-200">{currentTrialIdx + 1} / {currentPlanList.length}</strong>
                </span>
                <span className="text-slate-400">
                  BLOCK: <strong className="text-sky-400">{currentTask?.block.replace("_", " ")}</strong>
                </span>
                <span className="text-slate-400">
                  VARIANT: <strong className="text-amber-400">{currentTask?.variant}</strong>
                </span>
                <button
                  type="button"
                  onClick={handleWithdraw}
                  className="bg-rose-950/70 hover:bg-rose-900 border border-rose-800 text-rose-300 text-[11px] px-2.5 py-1 rounded transition-colors"
                >
                  Withdraw
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Cognitive Reset Break View */}
        {isOnBreak && (
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-8 flex flex-col items-center justify-center text-center gap-6 my-10 shadow-2xl">
            <div className="w-16 h-16 rounded-full bg-sky-950/80 border border-sky-700 flex items-center justify-center text-2xl text-sky-400">
              ☕
            </div>
            <div>
              <span className="text-xs font-mono font-bold bg-sky-950 text-sky-300 border border-sky-800 px-3 py-1 rounded">
                MANDATORY COGNITIVE RESET BREAK
              </span>
              <h2 className="text-2xl font-bold text-slate-100 mt-3">
                8-Minute Cognitive Decontamination Period
              </h2>
              <p className="text-xs text-slate-400 max-w-lg mt-2 leading-relaxed">
                To prevent carryover anchoring and mental fatigue between Block 1 and Block 2, participants must rest before starting the second evaluation block.
              </p>
            </div>

            <div className="text-5xl font-mono font-bold text-sky-400 tracking-wider">
              {Math.floor(breakRemainingSec / 60)}:{(breakRemainingSec % 60).toString().padStart(2, "0")}
            </div>

            <button
              type="button"
              onClick={handleEndBreak}
              className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-xs px-4 py-2 rounded transition-colors"
            >
              Skip Break Countdown (Rehearsal Only)
            </button>
          </div>
        )}

        {/* NASA-TLX Workload Rating Modal */}
        {showTLXModal && (
          <NASAWorkloadModal
            isOpen={showTLXModal}
            blockName="BLOCK 1: TRADE PLANNING"
            variantId={currentTask ? currentTask.variant : "CONCEPT_A"}
            onSubmit={handleTLXSubmit}
          />
        )}

        {/* Active Session View */}
        {sessionActive && !isOnBreak && !showTLXModal && currentTask && (
          <div className="flex flex-col gap-6">
            {/* Task Prompter Component */}
            <TaskPrompter
              taskId={currentTask.taskId}
              taskTitle={currentTask.taskTitle}
              taskInstructions={currentTask.taskInstructions}
              maxDurationSec={currentTask.maxDurationSec}
              onTaskSubmit={handleTaskSubmit}
              onTimeout={handleTimeout}
            />

            {/* Fixture Error Card */}
            {fixtureError && (
              <div className="bg-rose-950/80 border border-rose-800 p-4 rounded text-rose-200 font-mono text-xs">
                [FIXTURE ERROR]: {fixtureError}
              </div>
            )}

            {/* Fixture Loading Spinner */}
            {isLoadingFixture && (
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-12 text-center text-slate-400 font-mono text-xs">
                Loading deterministic fixture ({currentTask.scenarioId})...
              </div>
            )}

            {/* Active Variant Component View */}
            {!isLoadingFixture && currentFixture && (
              <VariantRunner
                variant={currentTask.variant}
                fixture={currentFixture}
                onInteraction={handleInteraction}
                onOrderAction={handleOrderAction}
                copyFeedback={copyFeedback}
              />
            )}
          </div>
        )}

        {/* Completed Session View */}
        {(sessionCompleted || sessionLifecycleState === "SESSION_HOLD_EVIDENCE_INTEGRITY") && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 flex flex-col items-center justify-center text-center gap-6 shadow-2xl">
            {sessionLifecycleState === "SESSION_SEALED" && (
              <>
                <div className="w-16 h-16 rounded-full bg-emerald-950 border border-emerald-700 flex items-center justify-center text-2xl text-emerald-400">
                  ✓
                </div>
                <div>
                  <span className="text-xs font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-800 px-3 py-1 rounded">
                    SEALED CUSTODY VERIFIED • RECEIPT: {custodyReceipt?.receipt_id || "ISSUED"}
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 mt-3">
                    Session Sealed: {selectedParticipant}
                  </h2>
                  <p className="text-xs text-slate-400 max-w-lg mt-2">
                    Evidence successfully ingested into local restricted custody vault. Cryptographic receipt signature verified against ledger.
                  </p>
                </div>

                <div className="bg-slate-950 border border-slate-800 rounded p-4 max-w-xl w-full text-left font-mono text-xs text-slate-300 flex flex-col gap-2">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Receipt ID:</span>
                    <span className="text-emerald-400 font-bold">{custodyReceipt?.receipt_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Ledger Digest:</span>
                    <span className="text-slate-300 truncate max-w-[320px]">{custodyReceipt?.ledger_ref}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Server HMAC Signature:</span>
                    <span className="text-emerald-400 truncate max-w-[320px]">{custodyReceipt?.server_receipt_signature}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Payload SHA-256:</span>
                    <span className="text-slate-300 truncate max-w-[320px]">{custodyReceipt?.source_payload_sha256}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Accepted At (UTC):</span>
                    <span className="text-slate-400">{custodyReceipt?.accepted_at_utc}</span>
                  </div>
                </div>
              </>
            )}

            {sessionLifecycleState === "SESSION_HOLD_EVIDENCE_INTEGRITY" && (
              <>
                <div className="w-16 h-16 rounded-full bg-amber-950 border border-amber-600 flex items-center justify-center text-2xl text-amber-400">
                  ⚠
                </div>
                <div>
                  <span className="text-xs font-mono font-bold bg-amber-950 text-amber-300 border border-amber-700 px-3 py-1 rounded">
                    EVIDENCE CUSTODY HOLD • UNSEALED BUFFER PRESERVED
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 mt-3">
                    Custody Intake Pending / Blocked
                  </h2>
                  <p className="text-xs text-amber-300/80 max-w-lg mt-2">
                    {custodyError || "Custody intake rejected or server unreachable. All unsealed interaction records remain durably stored in browser IndexedDB."}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={retryCustodyTransfer}
                    className="bg-amber-600 hover:bg-amber-500 text-slate-950 font-mono font-bold text-xs px-5 py-2.5 rounded shadow transition-colors"
                  >
                    Retry Custody Transfer
                  </button>
                </div>
              </>
            )}

            {sessionLifecycleState === "CUSTODY_TRANSFER_PENDING" && (
              <div className="py-8 flex flex-col items-center gap-3">
                <div className="text-sky-400 font-mono text-sm">
                  Transferring evidence to local custody vault and verifying receipt signature...
                </div>
              </div>
            )}

            {sessionLifecycleState === "SESSION_WITHDRAWN" && (
              <>
                <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-2xl text-slate-400">
                  Ø
                </div>
                <div>
                  <span className="text-xs font-mono font-bold bg-slate-800 text-slate-300 border border-slate-700 px-3 py-1 rounded">
                    PARTICIPANT WITHDRAWAL CONFIRMED
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 mt-3">
                    Participation Withdrawn
                  </h2>
                  <p className="text-xs text-slate-400 max-w-lg mt-2">
                    All local interaction telemetry has been securely zero-filled and wiped from browser IndexedDB. Withdrawal tombstone recorded in ledger.
                  </p>
                </div>
              </>
            )}

            <div className="flex items-center gap-4 mt-2">
              <button
                type="button"
                onClick={handleDownloadFiles}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono text-xs px-5 py-2 rounded transition-colors"
              >
                Manual Backup (.jsonl)
              </button>
              <button
                type="button"
                onClick={() => {
                  setSessionCompleted(false);
                  setSessionLifecycleState("SESSION_IDLE");
                }}
                className="bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-400 font-mono text-xs px-4 py-2 rounded transition-colors"
              >
                Reset
              </button>
            </div>
          </div>
        )}

        {/* Initial Idle Instructions */}
        {!sessionActive && !sessionCompleted && (
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-lg p-6 text-xs text-slate-400 flex flex-col gap-3 font-mono">
            <span className="font-bold text-slate-300">Phase 2 Prototype Sandbox Information:</span>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li>Strict network isolation active: External fetch, XHR, and WebSocket APIs are intercepted and blocked.</li>
              <li>Only approved deterministic fixtures from <code className="text-slate-200">/fixtures/scenarios/</code> are permissible.</li>
              <li>Order dispatchers execute non-mutating clipboard copies. Zero live broker routing.</li>
              <li>To perform automated technical rehearsal, select <code className="text-emerald-400">P00-SYNTHETIC</code> and click Start Assigned Protocol.</li>
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}
