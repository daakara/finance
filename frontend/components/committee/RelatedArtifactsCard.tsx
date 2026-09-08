"use client";

import { useState } from "react";
import Link from "next/link";
import { buildRelatedArtifacts } from "../../lib/telemetry/entityResolverEngine";

export interface RelatedArtifactsCardProps {
  entityId: string;
  title?: string;
  compact?: boolean;
}

export default function RelatedArtifactsCard({
  entityId,
  title,
  compact = false,
}: RelatedArtifactsCardProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const summary = buildRelatedArtifacts(entityId);

  const handleCopy = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(id);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    }
  };

  const getRelationshipColor = (rel: string) => {
    switch (rel) {
      case "PARENT_COMMITTEE":
        return "text-cyan-400 border-cyan-500/30 bg-cyan-950/40";
      case "REALIZED_OUTCOME":
        return "text-emerald-400 border-emerald-500/30 bg-emerald-950/40";
      case "PRESERVED_DISSENT":
        return "text-purple-400 border-purple-500/30 bg-purple-950/40";
      case "ORIGINAL_PROPOSAL":
        return "text-amber-400 border-amber-500/30 bg-amber-950/40";
      case "CRYPTOGRAPHIC_SNAPSHOT":
        return "text-pink-400 border-pink-500/30 bg-pink-950/40";
      default:
        return "text-slate-400 border-slate-700 bg-slate-800/40";
    }
  };

  return (
    <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl font-mono space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#202d44] pb-2.5">
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <h3 className="text-xs font-bold text-slate-100 uppercase tracking-wider">
            {title ?? `Connected Lineage Artifacts for ${entityId}`}
          </h3>
          <span className="text-[10px] text-slate-400">({summary.totalConnectedArtifacts} items)</span>
        </div>

        <div className="flex items-center space-x-2">
          <Link
            href={`/audit-explorer?queryId=${entityId}`}
            className="px-2 py-0.5 rounded bg-[#1c273a] hover:bg-cyan-950/60 border border-cyan-500/30 text-[10px] text-cyan-300 font-bold transition-colors"
          >
            Open Audit Reconstruction &rarr;
          </Link>
          <span className="px-1.5 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/30 text-[10px] text-emerald-400 font-bold">
            1-Click Linked
          </span>
        </div>
      </div>

      {/* Artifacts Grid */}
      {summary.items.length > 0 ? (
        <div className={`grid ${compact ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"} gap-2.5`}>
          {summary.items.map((item) => (
            <div
              key={`${item.entityId}-${item.relationship}`}
              className="p-3 rounded-lg bg-[#0c1017] border border-[#202d44] hover:border-cyan-500/40 transition-colors flex flex-col justify-between space-y-2"
            >
              <div>
                <div className="flex items-center justify-between gap-1 mb-1">
                  <div className="flex items-center space-x-1.5">
                    <span className="px-1.5 py-0.2 rounded bg-[#162032] text-cyan-400 text-[9px] font-bold">
                      {item.entityType}
                    </span>
                    <span className="font-bold text-slate-200 text-xs">{item.entityId}</span>
                  </div>

                  <span className={`px-1.5 py-0.2 rounded border text-[9px] font-semibold ${getRelationshipColor(item.relationship)}`}>
                    {item.relationship.replace(/_/g, " ")}
                  </span>
                </div>

                <p className="text-xs text-slate-300 font-medium line-clamp-1">{item.title}</p>
                {item.subtitle && (
                  <p className="text-[10px] text-slate-500 mt-0.5 truncate">{item.subtitle}</p>
                )}
              </div>

              {/* Action Buttons: Copy ID, Audit View, Open Entity */}
              <div className="flex items-center justify-between pt-2 border-t border-[#202d44]/60 text-[10px]">
                <button
                  type="button"
                  onClick={(e) => handleCopy(item.entityId, e)}
                  className="text-slate-400 hover:text-slate-200 flex items-center space-x-1 transition-colors"
                >
                  <span>{copiedId === item.entityId ? "✓ Copied" : "Copy ID"}</span>
                </button>

                <div className="flex items-center space-x-1.5">
                  <Link
                    href={`/audit-explorer?queryId=${item.entityId}`}
                    className="text-purple-400 hover:text-purple-300 transition-colors"
                    title="Inspect Cryptographic Audit Reconstruction"
                  >
                    Audit
                  </Link>
                  <span className="text-slate-600">&bull;</span>
                  <Link
                    href={item.canonicalRoute}
                    className="text-cyan-400 hover:text-cyan-300 font-bold transition-colors"
                  >
                    View &rarr;
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="py-6 text-center text-slate-500 text-xs">
          No secondary artifacts linked to {entityId}.
        </div>
      )}
    </div>
  );
}
