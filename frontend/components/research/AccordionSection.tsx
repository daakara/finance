"use client";

import React from "react";

export interface AccordionSectionProps {
  id: string;
  title: string;
  badge?: string;
  count?: number;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
  className?: string;
}

export default function AccordionSection({
  id,
  title,
  badge,
  count,
  isOpen,
  onToggle,
  children,
  className = "",
}: AccordionSectionProps) {
  const contentId = `accordion-content-${id}`;
  const headerId = `accordion-header-${id}`;

  return (
    <div className={`border border-border-subtle rounded-xl bg-bg-surface overflow-hidden ${className}`}>
      <button
        type="button"
        id={headerId}
        aria-expanded={isOpen}
        aria-controls={contentId}
        onClick={onToggle}
        className="w-full p-4 flex items-center justify-between gap-3 bg-bg-surface hover:bg-bg-surface-raised transition-colors cursor-pointer text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-info"
      >
        <div className="flex items-center gap-2.5">
          <span className="text-caption-mono font-bold text-text-primary text-sm sm:text-base">
            {title}
          </span>
          {count !== undefined && (
            <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded-full bg-bg-surface-raised border border-border-subtle text-text-secondary">
              {count}
            </span>
          )}
          {badge && (
            <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-accent-positive/20 text-accent-positive border border-accent-positive/30">
              {badge}
            </span>
          )}
        </div>

        <span className="text-caption-mono font-bold text-text-muted text-xs">
          {isOpen ? "▲ Collapse" : "▼ Expand"}
        </span>
      </button>

      {isOpen && (
        <div
          id={contentId}
          role="region"
          aria-labelledby={headerId}
          className="p-4 sm:p-5 border-t border-border-subtle bg-bg-app animate-fadeIn"
        >
          {children}
        </div>
      )}
    </div>
  );
}
