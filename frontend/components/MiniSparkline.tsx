"use client";

import React, { useMemo } from "react";

interface MiniSparklineProps {
  data?: number[];
  basePrice?: number;
  changePct?: number;
  width?: number;
  height?: number;
  isPositive?: boolean;
  className?: string;
}

export default function MiniSparkline({
  data,
  basePrice = 100,
  changePct = 0,
  width = 64,
  height = 22,
  isPositive,
  className = "",
}: MiniSparklineProps) {
  const positive = isPositive !== undefined ? isPositive : changePct >= 0;

  // Generate deterministic points if data series not explicitly passed
  const points = useMemo(() => {
    if (data && data.length >= 2) return data;

    // Generate smooth 8-point deterministic curve matching changePct trend
    const pts: number[] = [];
    const count = 8;
    const start = positive ? basePrice * (1 - Math.abs(changePct) * 0.01) : basePrice * (1 + Math.abs(changePct) * 0.01);
    const end = basePrice;
    const diff = end - start;

    for (let i = 0; i < count; i++) {
      const progress = i / (count - 1);
      // Add slight organic volatility oscillation
      const oscillation = Math.sin(progress * Math.PI * 2) * (Math.abs(diff) * 0.25);
      pts.push(start + diff * progress + (i === 0 || i === count - 1 ? 0 : oscillation));
    }
    return pts;
  }, [data, basePrice, changePct, positive]);

  const { pathD, fillD } = useMemo(() => {
    if (points.length < 2) return { pathD: "", fillD: "" };

    const min = Math.min(...points);
    const max = Math.max(...points);
    const range = max - min || 1;
    const padding = 2;
    const usableHeight = height - padding * 2;

    const coords = points.map((val, idx) => {
      const x = (idx / (points.length - 1)) * width;
      const y = height - padding - ((val - min) / range) * usableHeight;
      return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
    });

    const path = coords.reduce((acc, pt, idx) => {
      return idx === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
    }, "");

    const fill = `${path} L ${width} ${height} L 0 ${height} Z`;

    return { pathD: path, fillD: fill };
  }, [points, width, height]);

  const strokeColor = positive ? "#10b981" : "#f43f5e";
  const gradId = `spark-grad-${positive ? "pos" : "neg"}-${Math.round(basePrice * 10)}`;

  return (
    <div className={`inline-block overflow-hidden shrink-0 ${className}`}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={strokeColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={strokeColor} stopOpacity="0.0" />
          </linearGradient>
        </defs>
        {fillD && <path d={fillD} fill={`url(#${gradId})`} />}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke={strokeColor}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
      </svg>
    </div>
  );
}
