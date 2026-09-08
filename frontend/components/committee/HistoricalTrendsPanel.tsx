"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  TrendTimeframe,
  TrendMetricType,
} from "../../types/navigation-intelligence";
import {
  computeHistoricalTrendSeries,
} from "../../lib/telemetry/historicalTrendsEngine";
import { CANONICAL_COMMITTEES } from "../../lib/telemetry/committeeIntelligenceEngine";

export interface HistoricalTrendsPanelProps {
  initialCommitteeId?: string;
  initialMetric?: TrendMetricType;
  initialTimeframe?: TrendTimeframe;
}

export default function HistoricalTrendsPanel({
  initialCommitteeId = "COM-001",
  initialMetric = "ODEI",
  initialTimeframe = "90D",
}: HistoricalTrendsPanelProps) {
  const [selectedCommitteeId, setSelectedCommitteeId] = useState<string>(initialCommitteeId);
  const [selectedMetric, setSelectedMetric] = useState<TrendMetricType>(initialMetric);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TrendTimeframe>(initialTimeframe);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [isCompareMode, setIsCompareMode] = useState<boolean>(false);

  // Compute active series
  const series = useMemo(() => {
    return computeHistoricalTrendSeries(selectedCommitteeId, selectedMetric, selectedTimeframe);
  }, [selectedCommitteeId, selectedMetric, selectedTimeframe]);

  // Comparison series for all committees when in compare mode
  const comparisonSeries = useMemo(() => {
    if (!isCompareMode) return [];
    return CANONICAL_COMMITTEES.map((c) =>
      computeHistoricalTrendSeries(c.committeeId, selectedMetric, selectedTimeframe)
    );
  }, [isCompareMode, selectedMetric, selectedTimeframe]);

  const TIMEFRAMES: TrendTimeframe[] = ["30D", "90D", "180D", "365D"];
  const METRICS: { id: TrendMetricType; label: string }[] = [
    { id: "ODEI", label: "ODEI" },
    { id: "CDQI", label: "CDQI" },
    { id: "DIRATIO", label: "DIRatio" },
    { id: "DISSENT_UTIL", label: "Dissent Utilization" },
    { id: "LEARNING_VELOCITY", label: "Learning Velocity" },
  ];

  // SVG Chart Geometry
  const width = 800;
  const height = 240;
  const padding = { top: 20, right: 30, bottom: 30, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Min / Max calculation
  const allPoints = isCompareMode
    ? comparisonSeries.flatMap((s) => s.points)
    : series.points;

  const rawMin = Math.min(...allPoints.map((p) => p.value));
  const rawMax = Math.max(...allPoints.map((p) => p.value));
  const floorThreshold = series.floorThreshold ?? 0;
  const yMin = Math.max(0, Math.min(rawMin, floorThreshold) - 5);
  const yMax = Math.max(rawMax, floorThreshold) + 5;

  const getX = (idx: number, total: number) => {
    return padding.left + (idx / Math.max(1, total - 1)) * chartWidth;
  };

  const getY = (val: number) => {
    return padding.top + chartHeight - ((val - yMin) / Math.max(1, yMax - yMin)) * chartHeight;
  };

  // Build SVG path
  const buildSvgPath = (points: typeof series.points) => {
    return points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${getX(i, points.length).toFixed(1)} ${getY(p.value).toFixed(1)}`)
      .join(" ");
  };

  const activeHoverPoint = hoveredIndex !== null ? series.points[hoveredIndex] : series.points[series.points.length - 1];

  return (
    <div className="bg-[#111724] border border-[#202d44] p-5 rounded-xl font-mono space-y-4">
      {/* Header & Controls Bar */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-3 border-b border-[#202d44] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Historical Decision Intelligence Trends (HT-01 to HT-06)
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Deterministic time-series analytics, quality floor alerts, and 90-day rolling averages.
          </p>
        </div>

        {/* Committee & Compare Toggle */}
        <div className="flex items-center flex-wrap gap-2">
          <div className="flex bg-[#0c1017] p-0.5 rounded-lg border border-[#243044] text-xs">
            {CANONICAL_COMMITTEES.map((com) => (
              <button
                key={com.committeeId}
                type="button"
                onClick={() => {
                  setSelectedCommitteeId(com.committeeId);
                  setIsCompareMode(false);
                }}
                className={`px-2.5 py-1 rounded transition-colors ${
                  selectedCommitteeId === com.committeeId && !isCompareMode
                    ? "bg-[#1f2c42] text-cyan-300 font-bold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {com.committeeId}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setIsCompareMode(!isCompareMode)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-bold transition-colors ${
              isCompareMode
                ? "bg-purple-950/70 border-purple-500/50 text-purple-300"
                : "bg-[#162032] border-[#243044] text-slate-400 hover:text-slate-200"
            }`}
          >
            Compare All Committees
          </button>
        </div>
      </div>

      {/* Metrics & Timeframe Switchers */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        {/* Metric Selector */}
        <div className="flex flex-wrap gap-1.5 text-xs">
          {METRICS.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setSelectedMetric(m.id)}
              className={`px-2.5 py-1 rounded-lg border transition-colors ${
                selectedMetric === m.id
                  ? "bg-cyan-950/80 border-cyan-500/50 text-cyan-300 font-bold"
                  : "bg-[#0c1017] border-[#243044] text-slate-400 hover:text-slate-200"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Timeframe Selector (30D, 90D, 180D, 365D) */}
        <div className="flex bg-[#0c1017] p-0.5 rounded-lg border border-[#243044] text-xs">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              type="button"
              onClick={() => setSelectedTimeframe(tf)}
              className={`px-2.5 py-1 rounded transition-colors ${
                selectedTimeframe === tf
                  ? "bg-[#1f2c42] text-cyan-300 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metric Snapshot Ribbon */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-[#0c1017] p-3 rounded-xl border border-[#202d44]">
        <div>
          <span className="text-[10px] text-slate-500 uppercase block">Current Score</span>
          <span className="text-xl font-bold text-slate-100 mt-0.5 block">
            {series.currentValue.toFixed(1)}
          </span>
          <span className="text-[10px] text-slate-400">{series.committeeName}</span>
        </div>

        <div>
          <span className="text-[10px] text-slate-500 uppercase block">{selectedTimeframe} Change</span>
          <div className="flex items-baseline space-x-1 mt-0.5">
            <span
              className={`text-xl font-bold ${
                series.deltaAbsolute >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {series.deltaAbsolute >= 0 ? `+${series.deltaAbsolute.toFixed(1)}` : series.deltaAbsolute.toFixed(1)}
            </span>
            <span className="text-[10px] text-slate-400">({series.deltaPct}%)</span>
          </div>
          <span className="text-[10px] text-slate-400">Direction: {series.trendDirection}</span>
        </div>

        <div>
          <span className="text-[10px] text-slate-500 uppercase block">90-Day Rolling Avg</span>
          <span className="text-xl font-bold text-cyan-400 mt-0.5 block">
            {series.rollingAverage90d ? series.rollingAverage90d.toFixed(1) : "N/A"}
          </span>
          <span className="text-[10px] text-slate-400">Smoothed Trajectory</span>
        </div>

        <div>
          <span className="text-[10px] text-slate-500 uppercase block">Institutional Floor</span>
          <span className="text-xl font-bold text-emerald-400 mt-0.5 block">
            {series.floorThreshold ? `>= ${series.floorThreshold.toFixed(1)}` : "None"}
          </span>
          <span className="text-[10px] text-slate-400">
            {series.currentValue >= (series.floorThreshold ?? 0) ? "✓ Compliant" : "⚠ Breached"}
          </span>
        </div>
      </div>

      {/* Deterioration Alert Banner (HT-04) */}
      {series.hasDeteriorationWarning && (
        <div className="p-3 rounded-xl bg-rose-950/60 border border-rose-500/50 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-rose-400 font-bold">&#9888; DETERIORATION ALERT (HT-04):</span>
            <span>
              {series.committeeName} {series.metric} declined by {Math.abs(series.deltaAbsolute).toFixed(1)} points over {selectedTimeframe} (exceeds -5.0 pt threshold).
            </span>
          </div>
          <span className="px-2 py-0.5 rounded bg-rose-900/60 border border-rose-400 text-rose-200 text-[10px] font-bold">
            Escalate to NOC
          </span>
        </div>
      )}

      {/* Interactive SVG Chart */}
      <div className="relative bg-[#0c1017] border border-[#202d44] rounded-xl p-3 overflow-hidden">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto select-none"
        >
          {/* Grid lines */}
          <line
            x1={padding.left}
            y1={padding.top}
            x2={width - padding.right}
            y2={padding.top}
            stroke="#1c273a"
            strokeDasharray="3 3"
          />
          <line
            x1={padding.left}
            y1={padding.top + chartHeight / 2}
            x2={width - padding.right}
            y2={padding.top + chartHeight / 2}
            stroke="#1c273a"
            strokeDasharray="3 3"
          />
          <line
            x1={padding.left}
            y1={padding.top + chartHeight}
            x2={width - padding.right}
            y2={padding.top + chartHeight}
            stroke="#1c273a"
          />

          {/* Institutional Floor Line */}
          {series.floorThreshold != null && (
            <g>
              <line
                x1={padding.left}
                y1={getY(series.floorThreshold)}
                x2={width - padding.right}
                y2={getY(series.floorThreshold)}
                stroke="#f59e0b"
                strokeWidth="1.5"
                strokeDasharray="4 4"
              />
              <text
                x={width - padding.right - 5}
                y={getY(series.floorThreshold) - 5}
                fill="#f59e0b"
                fontSize="9"
                textAnchor="end"
                fontFamily="monospace"
              >
                Floor ({series.floorThreshold.toFixed(1)})
              </text>
            </g>
          )}

          {/* Comparison polylines if in compare mode */}
          {isCompareMode &&
            comparisonSeries.map((comp, idx) => {
              const strokeColors = ["#06b6d4", "#a855f7", "#10b981"];
              return (
                <path
                  key={comp.committeeId}
                  d={buildSvgPath(comp.points)}
                  fill="none"
                  stroke={strokeColors[idx % strokeColors.length]}
                  strokeWidth="2"
                />
              );
            })}

          {/* Main polyline */}
          {!isCompareMode && (
            <path
              d={buildSvgPath(series.points)}
              fill="none"
              stroke="#06b6d4"
              strokeWidth="2.5"
            />
          )}

          {/* Interactive Data Points */}
          {!isCompareMode &&
            series.points.map((p, idx) => {
              const cx = getX(idx, series.points.length);
              const cy = getY(p.value);
              const isHovered = hoveredIndex === idx;

              return (
                <circle
                  key={idx}
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 5 : 2.5}
                  fill={p.isFloorBreach ? "#f43f5e" : isHovered ? "#38bdf8" : "#06b6d4"}
                  stroke="#0c1017"
                  strokeWidth="1.5"
                  onMouseEnter={() => setHoveredIndex(idx)}
                  className="cursor-pointer transition-all"
                />
              );
            })}

          {/* Axis Labels */}
          <text
            x={padding.left - 8}
            y={padding.top + 5}
            fill="#64748b"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
          >
            {yMax.toFixed(0)}
          </text>
          <text
            x={padding.left - 8}
            y={padding.top + chartHeight}
            fill="#64748b"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
          >
            {yMin.toFixed(0)}
          </text>

          {/* Dates on X Axis */}
          <text
            x={padding.left}
            y={height - 10}
            fill="#64748b"
            fontSize="9"
            fontFamily="monospace"
          >
            {new Date(series.points[0].timestampUtc).toLocaleDateString()}
          </text>
          <text
            x={width - padding.right}
            y={height - 10}
            fill="#64748b"
            fontSize="9"
            textAnchor="end"
            fontFamily="monospace"
          >
            {new Date(series.points[series.points.length - 1].timestampUtc).toLocaleDateString()}
          </text>
        </svg>

        {/* Selected Data Point Inspector (HT-05) */}
        {activeHoverPoint && (
          <div className="mt-2 pt-2 border-t border-[#202d44] flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs gap-2">
            <div className="flex items-center space-x-2">
              <span className="text-slate-500">
                {new Date(activeHoverPoint.timestampUtc).toLocaleDateString()}:
              </span>
              <span className="font-bold text-cyan-300">
                {activeHoverPoint.value.toFixed(1)}
              </span>
              {activeHoverPoint.isFloorBreach && (
                <span className="px-1.5 py-0.2 rounded bg-rose-950 text-rose-300 text-[10px] font-bold">
                  Floor Breach
                </span>
              )}
            </div>

            {/* Drill-down links to underlying decisions (HT-05) */}
            {activeHoverPoint.underlyingArtifactIds && activeHoverPoint.underlyingArtifactIds.length > 0 && (
              <div className="flex items-center space-x-1.5 text-[10px]">
                <span className="text-slate-500">Underlying Decisions (HT-05):</span>
                {activeHoverPoint.underlyingArtifactIds.map((decId) => (
                  <Link
                    key={decId}
                    href={`/decision-explorer?decisionId=${decId}`}
                    className="px-1.5 py-0.5 rounded bg-[#162032] hover:bg-cyan-950/60 border border-[#243044] text-cyan-400 font-bold transition-colors"
                  >
                    {decId} &rarr;
                  </Link>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
