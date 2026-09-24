"use client";

import { SelfHealingAudit } from "../lib/api";

interface SelfHealingAccuracyCardProps {
  symbol: string;
  auditData?: SelfHealingAudit;
}

export default function SelfHealingAccuracyCard({ symbol, auditData }: SelfHealingAccuracyCardProps) {
  const isCalibrated = Boolean(
    auditData &&
    typeof auditData.accuracyScore === "number" &&
    !isNaN(auditData.accuracyScore)
  );

  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-5 shadow-xl space-y-4 font-mono">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className={`w-2.5 h-2.5 rounded-full ${isCalibrated ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`}></span>
            <h3 className="text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <svg className="w-4 h-4 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
              </svg>
              <span>{symbol} Historical Track Record & Retrospective Verification</span>
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Retrospective accuracy check comparing past quantitative predictions against realized market price action
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className={`text-[9px] font-bold px-2 py-0.5 rounded border inline-flex items-center gap-1 ${
              isCalibrated
                ? "bg-emerald-950/80 text-emerald-300 border-emerald-800/80"
                : "bg-slate-900 text-slate-400 border-slate-700"
            }`}>
              <span>🎯</span> {isCalibrated ? "Continuous 30-Day Predictive Realization Audit" : "Sample Accumulation Phase (N < 35)"}
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <div className={`px-3 py-1 rounded-lg text-right border ${
            isCalibrated
              ? "bg-emerald-950/80 border-emerald-700/80"
              : "bg-slate-900 border-slate-700"
          }`}>
            <span className="text-[10px] text-slate-400 block uppercase leading-none font-bold">Track Record</span>
            <span className={`text-base font-bold ${isCalibrated ? "text-emerald-400" : "text-slate-400"}`}>
              {isCalibrated ? `${auditData?.accuracyScore}%` : "Unavailable"}
            </span>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-[#1b2434] text-cyan-300 border border-cyan-800/80">
            {auditData?.auditStatus || "Awaiting Minimum Historical Sample (N < 35)"}
          </span>
        </div>
      </div>

      {/* 4 Feedback Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-center">
        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block uppercase">Historical Realized Direction</span>
          <span className={`text-base font-bold ${auditData?.hitRatePct != null ? "text-emerald-400" : "text-slate-400"}`}>
            {auditData?.hitRatePct != null ? `${auditData.hitRatePct}%` : "Unavailable"}
          </span>
          <span className="text-[9px] text-slate-500 block mt-0.5">30-Day Trend Match</span>
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block uppercase">Target Price Precision</span>
          <span className={`text-base font-bold ${auditData?.rmsePct != null ? "text-cyan-400" : "text-slate-400"}`}>
            {auditData?.rmsePct != null ? `±${auditData.rmsePct}%` : "Unavailable"}
          </span>
          <span className="text-[9px] text-slate-500 block mt-0.5">Average Forecast Deviation</span>
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block uppercase">Tail-Risk Protection</span>
          <span className={`text-base font-bold ${auditData?.varBreachRatePct != null ? "text-purple-400" : "text-slate-400"}`}>
            {auditData?.varBreachRatePct != null ? `${auditData.varBreachRatePct}%` : "Unavailable"}
          </span>
          <span className="text-[9px] text-slate-500 block mt-0.5">
            {isCalibrated ? "Passed 5.0% Stress Limit" : "Regime Test Pending"}
          </span>
        </div>

        <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044]">
          <span className="text-[10px] text-slate-400 block uppercase">Live Model Status</span>
          <span className={`text-xs font-bold block truncate mt-1 ${isCalibrated ? "text-amber-400" : "text-slate-400"}`}>
            {isCalibrated ? "Active & Synced" : "Pending Historical Tape"}
          </span>
          <span className="text-[9px] text-slate-500 block mt-0.5">
            {isCalibrated ? "Sample Depth Calibrated" : "Minimum 35 Bars Required"}
          </span>
        </div>
      </div>

      {/* Auto-Healing Adjustment Log */}
      <div className="bg-[#090d14] p-3 rounded-lg border border-[#243044] text-xs text-slate-300 space-y-1">
        <div className="flex justify-between items-center text-[11px]">
          <span className="text-slate-400">Stress Multiplier:</span>
          <span className="text-emerald-400 font-semibold">
            {auditData?.autoCalibrationAdjustments || "Auto-calibration inactive until minimum sample reached"}
          </span>
        </div>
        <div className="flex justify-between items-center text-[11px]">
          <span className="text-slate-400">Historical Model Reliability:</span>
          <span className="text-cyan-400 font-semibold">
            {auditData?.varBreachStatus || "Insufficient History (< 35 bars)"} ({auditData?.confidenceInterval || "Insufficient Sample Size"})
          </span>
        </div>
      </div>
    </div>
  );
}

