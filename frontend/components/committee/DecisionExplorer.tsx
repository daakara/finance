"use client";

import { useState, useMemo, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import DecisionTimeline from "./DecisionTimeline";
import RelatedArtifactsCard from "./RelatedArtifactsCard";
import {
  CommitteeDecision,
} from "../../types/committee-intelligence";
import {
  buildDecisionTimeline,
} from "../../lib/telemetry/decisionNetworkEngine";
import {
  CANONICAL_PROPOSALS,
  CANONICAL_EVIDENCE_STORE,
  CANONICAL_OUTCOMES,
  CANONICAL_ATTRIBUTIONS,
} from "../../lib/telemetry/auditReconstructionEngine";

export interface DecisionExplorerProps {
  decisions: CommitteeDecision[];
  initialDecisionId?: string;
}

export default function DecisionExplorer({ decisions, initialDecisionId }: DecisionExplorerProps) {
  const searchParams = useSearchParams();
  const router = useRouter();

  const urlDecisionId = searchParams.get("decisionId");
  const urlCommitteeId = searchParams.get("committeeId");

  const [selectedId, setSelectedId] = useState<string>(
    urlDecisionId || initialDecisionId || (decisions.length > 0 ? decisions[0].decisionId : "DEC-001")
  );
  const [filterCommittee, setFilterCommittee] = useState<string>(urlCommitteeId || "ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  useEffect(() => {
    if (urlDecisionId) {
      setSelectedId(urlDecisionId);
    }
  }, [urlDecisionId]);

  useEffect(() => {
    if (urlCommitteeId) {
      setFilterCommittee(urlCommitteeId);
    }
  }, [urlCommitteeId]);

  const filteredDecisions = useMemo(() => {
    return decisions.filter((d) => {
      if (filterCommittee !== "ALL" && d.committeeId !== filterCommittee) {
        return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          d.decisionId.toLowerCase().includes(q) ||
          d.title.toLowerCase().includes(q) ||
          d.proposalId.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [decisions, filterCommittee, searchQuery]);

  const selectedDecision = useMemo(() => {
    return decisions.find((d) => d.decisionId === selectedId) || decisions[0];
  }, [decisions, selectedId]);

  const timelineSteps = useMemo(() => {
    return selectedDecision ? buildDecisionTimeline(selectedDecision.decisionId) : [];
  }, [selectedDecision]);

  const handleSelectDecision = (id: string) => {
    setSelectedId(id);
    router.replace(`/decision-explorer?decisionId=${id}${filterCommittee !== "ALL" ? `&committeeId=${filterCommittee}` : ""}`, { scroll: false });
  };

  const outcome = selectedDecision?.outcomeId ? CANONICAL_OUTCOMES[selectedDecision.outcomeId] : undefined;
  const attribution = selectedDecision?.outcomeId ? CANONICAL_ATTRIBUTIONS[selectedDecision.outcomeId] : undefined;
  const proposal = selectedDecision ? CANONICAL_PROPOSALS[selectedDecision.proposalId] : undefined;

  return (
    <div className="space-y-6 font-mono">
      {/* Header Search and Filters */}
      <div className="bg-[#101622] border border-[#1f2c42] p-4 rounded-xl flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3 text-xs">
        <div className="relative flex-1 max-w-lg">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Decision ID (DEC-001), Title, or Proposal..."
            className="w-full bg-[#0b0f17] border border-[#23334d] rounded-lg px-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-400 text-xs"
          />
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-slate-400 text-[11px]">Committee:</span>
          <select
            value={filterCommittee}
            onChange={(e) => setFilterCommittee(e.target.value)}
            className="bg-[#0b0f17] border border-[#23334d] text-slate-200 rounded-lg px-3 py-1.5 focus:outline-none focus:border-cyan-400 text-xs"
          >
            <option value="ALL">All Committees (3)</option>
            <option value="COM-001">Investment Committee (COM-001)</option>
            <option value="COM-002">Governance Committee (COM-002)</option>
            <option value="COM-003">Risk & Capital Committee (COM-003)</option>
          </select>
        </div>
      </div>

      {/* Main 2-Column Explorer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Decision Selector List */}
        <div className="lg:col-span-5 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span>Filtered Decisions ({filteredDecisions.length})</span>
            <span>INV-OI13 Certified</span>
          </div>

          <div className="space-y-2 max-h-[750px] overflow-y-auto pr-1">
            {filteredDecisions.map((d) => {
              const isSelected = d.decisionId === selectedDecision?.decisionId;
              return (
                <button
                  key={d.decisionId}
                  onClick={() => handleSelectDecision(d.decisionId)}
                  className={`w-full text-left p-3.5 rounded-xl border transition-all ${
                    isSelected
                      ? "bg-[#182335] border-cyan-500/50 shadow-md shadow-cyan-950/40"
                      : "bg-[#101622] border-[#1d293d] hover:border-[#2f4263]"
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] mb-1">
                    <span className="text-cyan-400 font-bold">{d.decisionId}</span>
                    <span className="text-slate-400 text-[10px]">{d.committeeId}</span>
                  </div>
                  <h4 className="text-xs font-semibold text-slate-100 line-clamp-1 mb-1.5">
                    {d.title}
                  </h4>
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span>Quality: <strong className="text-slate-200">{d.decisionQuality}/100</strong></span>
                    <span className="px-1.5 py-0.5 rounded bg-[#0e141f] text-emerald-400 border border-emerald-500/30">
                      {d.status}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Deep Lineage & Timeline Details */}
        {selectedDecision ? (
          <div className="lg:col-span-7 space-y-4">
            {/* Decision Hero Card */}
            <div className="bg-[#121927] border border-[#202d44] p-5 rounded-xl shadow-lg">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-3 border-b border-[#1d293d]">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="text-base font-bold text-slate-100">
                      {selectedDecision.decisionId}
                    </span>
                    <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 text-[10px] font-bold">
                      {selectedDecision.status}
                    </span>
                  </div>
                  <h3 className="text-sm font-semibold text-cyan-300 mt-1">
                    {selectedDecision.title}
                  </h3>
                </div>
                <div className="text-right text-xs">
                  <span className="text-slate-400 block text-[10px]">Quality Score</span>
                  <span className="text-lg font-bold text-slate-100">
                    {selectedDecision.decisionQuality != null ? selectedDecision.decisionQuality.toFixed(1) : "N/A"} / 100
                  </span>
                </div>
              </div>

              {/* Proposal Business Objective */}
              {proposal && (
                <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1b263b] mb-4 text-xs">
                  <span className="text-[10px] uppercase text-cyan-400 font-bold block mb-1">
                    Proposal Objective ({proposal.proposalId})
                  </span>
                  <p className="text-slate-300 leading-relaxed">
                    {proposal.businessObjective}
                  </p>
                </div>
              )}

              {/* Supporting Evidence Grid */}
              <div className="mb-4">
                <span className="text-[10px] uppercase text-slate-400 font-bold block mb-2">
                  Supporting Evidence ({selectedDecision.evidenceIds.length} Linked)
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  {selectedDecision.evidenceIds.map((id) => {
                    const ev = CANONICAL_EVIDENCE_STORE[id];
                    return (
                      <div
                        key={id}
                        className="bg-[#0b1019] p-2.5 rounded-lg border border-[#1b263b] flex flex-col justify-between"
                      >
                        <div className="flex justify-between items-center text-[10px] mb-1">
                          <span className="text-cyan-400 font-bold">{id}</span>
                          <span className="text-emerald-400 font-semibold">
                            {ev?.confidencePct ?? 95.0}% Conf
                          </span>
                        </div>
                        <span className="text-slate-300 text-[11px] truncate">
                          {ev?.sourceReference ?? id}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Outcome & Attribution Row */}
              {outcome && attribution && (
                <div className="bg-[#0b1019] p-3 rounded-lg border border-[#1b263b] mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[10px] uppercase text-slate-400 font-bold">
                      Outcome Realization & Attribution ({outcome.outcomeId})
                    </span>
                    <span className="text-emerald-400 text-xs font-bold">
                      +${(outcome.realizedValueDollars / 1000).toFixed(0)}k Value
                    </span>
                  </div>

                  {/* Attribution Breakdown Bar */}
                  <div className="w-full bg-[#1b263b] h-2 rounded-full overflow-hidden flex my-2">
                    <div
                      className="bg-cyan-400 h-full"
                      style={{ width: `${attribution.individualContributionPct}%` }}
                      title={`Individual: ${attribution.individualContributionPct}%`}
                    />
                    <div
                      className="bg-purple-400 h-full"
                      style={{ width: `${attribution.teamContributionPct}%` }}
                      title={`Team: ${attribution.teamContributionPct}%`}
                    />
                    <div
                      className="bg-emerald-400 h-full"
                      style={{ width: `${attribution.committeeContributionPct}%` }}
                      title={`Committee: ${attribution.committeeContributionPct}%`}
                    />
                    <div
                      className="bg-amber-400 h-full"
                      style={{ width: `${attribution.systemContributionPct}%` }}
                      title={`System: ${attribution.systemContributionPct}%`}
                    />
                  </div>

                  <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1.5">
                    <span>Ind: {attribution.individualContributionPct}%</span>
                    <span>Team: {attribution.teamContributionPct}%</span>
                    <span>Comm: {attribution.committeeContributionPct}%</span>
                    <span>Sys: {attribution.systemContributionPct}%</span>
                    <span className="text-emerald-400 font-bold">
                      Sum: {attribution.totalContributionPct}%
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Decision Audit Timeline Component */}
            <DecisionTimeline
              steps={timelineSteps}
              decisionId={selectedDecision.decisionId}
            />

            {/* Universal Connected Lineage Artifacts (M2-Gate-02) */}
            <RelatedArtifactsCard
              entityId={selectedDecision.decisionId}
              title={`Lineage Network & Related Artifacts (${selectedDecision.decisionId})`}
            />
          </div>
        ) : (
          <div className="lg:col-span-7 bg-[#101622] border border-[#1f2c42] p-8 text-center rounded-xl text-slate-400 text-xs">
            Select a decision from the list to explore its complete audit timeline.
          </div>
        )}
      </div>
    </div>
  );
}
