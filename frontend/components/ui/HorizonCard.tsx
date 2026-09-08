"use client";

import React, { ReactNode } from "react";

export interface HorizonCardProps {
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  badge?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  elevated?: boolean;
}

export function HorizonCard({
  title,
  subtitle,
  icon,
  badge,
  actions,
  children,
  className = "",
  elevated = false,
}: HorizonCardProps) {
  const bgClass = elevated ? "bg-[#182336]" : "bg-[#121B2A]";

  return (
    <section
      className={`rounded-2xl border border-[#24324A] ${bgClass} p-5 shadow-lg shadow-black/20 transition-all duration-150 hover:border-[#334769] ${className}`}
    >
      {(title || icon || badge || actions) && (
        <header className="flex items-center justify-between gap-3 mb-4 pb-3 border-b border-[#24324A]/60">
          <div className="flex items-center gap-2.5">
            {icon && <span className="text-cyan-400 shrink-0">{icon}</span>}
            <div>
              {title && (
                <h2 className="text-base font-semibold tracking-tight text-[#F8FAFC]">
                  {title}
                </h2>
              )}
              {subtitle && (
                <p className="text-xs text-[#94A3B8] font-mono mt-0.5">
                  {subtitle}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {badge}
            {actions}
          </div>
        </header>
      )}

      <div>{children}</div>
    </section>
  );
}

// Alias for convenience
export const IntelligenceCard = HorizonCard;
export default HorizonCard;
