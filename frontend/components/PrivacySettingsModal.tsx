"use client";

import { useState, useEffect } from "react";
import { toggleMatomoOptOut, isMatomoUserOptedOut } from "../lib/matomo";

interface PrivacySettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function PrivacySettingsModal({
  isOpen,
  onClose,
}: PrivacySettingsModalProps) {
  const [optedOut, setOptedOut] = useState<boolean>(false);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setOptedOut(isMatomoUserOptedOut());
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleToggle = () => {
    const nextState = !optedOut;
    setOptedOut(nextState);
    toggleMatomoOptOut(nextState);
    setToast(nextState ? "🚫 Analytics disabled (You are opted out)" : "✅ Anonymous analytics enabled");
    setTimeout(() => setToast(null), 3000);
  };

  return (
    <div className="fixed inset-0 z-[1200] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-[#0b1019] border border-cyan-800/80 rounded-2xl max-w-lg w-full p-5 sm:p-6 shadow-2xl space-y-4 font-sans text-slate-200 relative">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1e293b] pb-3.5">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">🛡️</span>
            <div>
              <h2 className="text-base sm:text-lg font-bold text-white tracking-tight">
                Privacy & Data Telemetry Settings
              </h2>
              <p className="text-xs text-slate-400">
                GDPR, ePrivacy & CNIL Cookieless Exemption Compliance
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Privacy Settings"
            className="text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800 transition"
          >
            ✕
          </button>
        </div>

        {/* Core Principles */}
        <div className="space-y-2 text-xs text-slate-300 bg-[#111722] p-3.5 rounded-xl border border-[#1e293b]">
          <div className="flex items-start gap-2">
            <span className="text-emerald-400 font-bold">✓</span>
            <div>
              <strong className="text-white">100% Cookieless by Design:</strong> No tracking cookies (<code className="text-[10px] text-cyan-300">_pk_id</code>, <code className="text-[10px] text-cyan-300">_pk_ses</code>) are ever stored on your device.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-emerald-400 font-bold">✓</span>
            <div>
              <strong className="text-white">Anonymized IP Addressing:</strong> IP addresses are masked prior to storage.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-emerald-400 font-bold">✓</span>
            <div>
              <strong className="text-white">Zero PII / No Cross-Site Tracking:</strong> We never collect email, names, or real identities.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-emerald-400 font-bold">✓</span>
            <div>
              <strong className="text-white">Do Not Track (DNT) Honored:</strong> Global Privacy Control (GPC) and DNT headers automatically disable telemetry.
            </div>
          </div>
        </div>

        {/* Opt-Out Toggle */}
        <div className="p-4 rounded-xl border border-[#1e293b] bg-[#0d131f] flex items-center justify-between gap-4">
          <div>
            <div className="text-sm font-bold text-white">
              Anonymous Performance Telemetry
            </div>
            <div className="text-xs text-slate-400 mt-0.5">
              {optedOut 
                ? "You are currently opted out. No analytics events are being tracked." 
                : "Helps us audit screener load times, chart reliability, and macro models."}
            </div>
          </div>
          <button
            type="button"
            onClick={handleToggle}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition shrink-0 border ${
              optedOut
                ? "bg-rose-950/80 border-rose-700 text-rose-300 hover:bg-rose-900"
                : "bg-emerald-950/80 border-emerald-700 text-emerald-300 hover:bg-emerald-900"
            }`}
          >
            {optedOut ? "OPTED OUT ❌" : "ACTIVE (ANONYMOUS) ✅"}
          </button>
        </div>

        {toast && (
          <div className="p-2 text-center text-xs font-mono font-bold bg-cyan-950/80 border border-cyan-800 text-cyan-300 rounded-lg animate-in fade-in">
            {toast}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-[#1e293b] text-xs">
          <span className="text-slate-500 font-mono text-[10px]">
            Server: Self-Hosted Matomo (data.fpldna.com)
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold transition"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
