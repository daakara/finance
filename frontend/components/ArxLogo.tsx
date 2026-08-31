"use client";

import React from "react";

interface ArxLogoProps {
  size?: "xs" | "sm" | "md" | "lg" | "xl" | number;
  showWordmark?: boolean;
  showSubtitle?: boolean;
  className?: string;
  variant?: "badge" | "plain" | "lockup";
}

export default function ArxLogo({
  size = "md",
  showWordmark = false,
  showSubtitle = false,
  className = "",
  variant = "badge",
}: ArxLogoProps) {
  let px = 28;
  if (typeof size === "number") {
    px = size;
  } else {
    switch (size) {
      case "xs":
        px = 18;
        break;
      case "sm":
        px = 22;
        break;
      case "md":
        px = 28;
        break;
      case "lg":
        px = 36;
        break;
      case "xl":
        px = 48;
        break;
    }
  }

  // Bespoke Quantitative Apex Citadel Logo Mark (Geometric A-R-X Delta)
  const IconSvg = (
    <svg
      width={px}
      height={px}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="shrink-0 transition-transform duration-300 group-hover:scale-105"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="arx-cyan-grad" x1="4" y1="4" x2="44" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="50%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#0284c7" />
        </linearGradient>
        <linearGradient id="arx-apex-grad" x1="24" y1="6" x2="24" y2="38" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#67e8f9" />
          <stop offset="60%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#10b981" />
        </linearGradient>
        <linearGradient id="arx-flank" x1="8" y1="20" x2="40" y2="42" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#34d399" />
          <stop offset="100%" stopColor="#059669" />
        </linearGradient>
        <filter id="arx-glow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#06b6d4" floodOpacity="0.4" />
        </filter>
      </defs>

      {/* Main Apex Citadel Triangle */}
      <path
        d="M24 5L42 37H34L24 19L14 37H6L24 5Z"
        fill="url(#arx-cyan-grad)"
        filter="url(#arx-glow)"
        opacity="0.95"
      />

      {/* Negative Space Cutout */}
      <path
        d="M24 16L32 31H26.5L24 26L21.5 31H16L24 16Z"
        fill="#070a10"
        opacity="0.92"
      />

      {/* Central Signal Pulse Diamond */}
      <polygon
        points="24,20 27,27 24,31 21,27"
        fill="url(#arx-apex-grad)"
      />

      {/* Foundation Chevron Winglets */}
      <path
        d="M13 39L24 44L35 39L32 36L24 39.5L16 36L13 39Z"
        fill="url(#arx-flank)"
        opacity="0.95"
      />
    </svg>
  );

  if (variant === "badge") {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="relative rounded-xl bg-[#0c1424] border border-[#1b2a40] p-1.5 flex items-center justify-center shadow-[0_0_14px_rgba(6,182,212,0.2)] hover:border-cyan-500/60 transition-all">
          {IconSvg}
        </div>

        {showWordmark && (
          <div className="min-w-0 flex flex-col justify-center">
            <span className="font-bold tracking-tight text-white font-mono text-sm sm:text-base leading-none">
              ARX <span className="text-cyan-400 font-black">TERMINAL</span>
            </span>
            {showSubtitle && (
              <span className="text-[9px] text-slate-400 font-mono tracking-wider uppercase mt-0.5">
                No-BS Market Intel
              </span>
            )}
          </div>
        )}
      </div>
    );
  }

  if (variant === "plain") {
    return IconSvg;
  }

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {IconSvg}
      {showWordmark && (
        <div className="min-w-0 flex flex-col justify-center">
          <span className="font-bold tracking-tight text-white font-mono text-sm sm:text-base leading-none">
            ARX <span className="text-cyan-400 font-black">TERMINAL</span>
          </span>
          {showSubtitle && (
            <span className="text-[9px] text-slate-400 font-mono tracking-wider uppercase mt-0.5">
              No-BS Market Intel
            </span>
          )}
        </div>
      )}
    </div>
  );
}
