"use client";

import React from "react";

export interface IntelligenceEmptyStateProps {
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
}

export default function IntelligenceEmptyState({
  title,
  description,
  actionLabel,
  onAction,
  icon,
}: IntelligenceEmptyStateProps) {
  return (
    <div className="rounded-2xl border border-dashed border-[#24324A] bg-[#121B2A]/60 p-10 text-center flex flex-col items-center justify-center max-w-2xl mx-auto my-8">
      <div className="w-12 h-12 rounded-full bg-[#182336] border border-[#24324A] flex items-center justify-center text-cyan-400 mb-4">
        {icon || (
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        )}
      </div>

      <h3 className="text-base font-semibold text-[#F8FAFC] mb-2">{title}</h3>
      <p className="text-sm text-[#94A3B8] font-mono leading-relaxed mb-6 max-w-md">
        {description}
      </p>

      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold shadow-lg shadow-cyan-950/40 transition-colors focus-visible:ring-2 focus-visible:ring-cyan-400"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
