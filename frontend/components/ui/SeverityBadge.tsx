"use client";

import React from "react";
import { SeverityLevel, SeverityColors, HorizonStatus, HorizonStatusConfig, normalizeHorizonStatus } from "../../lib/ui/horizonTokens";

export interface SeverityBadgeProps {
  level?: SeverityLevel;
  status?: HorizonStatus | string;
  size?: "sm" | "md" | "lg";
  className?: string;
  showIcon?: boolean;
}

export default function SeverityBadge({
  level,
  status,
  size = "md",
  className = "",
  showIcon = true,
}: SeverityBadgeProps) {
  let badgeClasses = "";
  let labelText = "";
  let dotColor = "";

  if (level && SeverityColors[level]) {
    const config = SeverityColors[level];
    badgeClasses = config.badge;
    labelText = config.label;
    dotColor = config.dot;
  } else if (status) {
    const norm = normalizeHorizonStatus(status);
    const config = HorizonStatusConfig[norm];
    badgeClasses = config.badge;
    labelText = config.label;
    dotColor = norm === 'HEALTHY' ? 'bg-emerald-400' :
               norm === 'CERTIFIED' ? 'bg-blue-400' :
               norm === 'WARNING' ? 'bg-amber-400' :
               norm === 'HIGH_RISK' ? 'bg-orange-400' :
               norm === 'CRITICAL' ? 'bg-red-500' : 'bg-rose-500';
  } else {
    badgeClasses = SeverityColors.INFO.badge;
    labelText = "Info";
    dotColor = SeverityColors.INFO.dot;
  }

  const sizeClasses = {
    sm: "px-1.5 py-0.5 text-[10px]",
    md: "px-2 py-0.5 text-xs",
    lg: "px-2.5 py-1 text-sm font-semibold",
  }[size];

  return (
    <span
      role="status"
      aria-label={`Status: ${labelText}`}
      className={`inline-flex items-center gap-1.5 rounded-full border font-mono font-medium transition-colors ${badgeClasses} ${sizeClasses} ${className}`}
    >
      {showIcon && (
        <span
          aria-hidden="true"
          className={`w-1.5 h-1.5 rounded-full ${dotColor} flex-shrink-0 animate-pulse`}
        />
      )}
      <span>{labelText}</span>
    </span>
  );
}
