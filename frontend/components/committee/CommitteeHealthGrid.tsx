"use client";

import { useState, useMemo } from "react";
import CommitteeScorecard from "./CommitteeScorecard";
import { CommitteeHealth } from "../../types/committee-intelligence";

export interface CommitteeHealthGridProps {
  committees: (CommitteeHealth & { committeeId: string; name: string; odei: number })[];
}

export type SortField = "ODEI" | "CDQI" | "DIRATIO";
export type FilterTier = "ALL" | "CERTIFIED" | "HIGH_RISK";

export default function CommitteeHealthGrid({ committees }: CommitteeHealthGridProps) {
  const [sortBy, setSortBy] = useState<SortField>("ODEI");
  const [filterBy, setFilterBy] = useState<FilterTier>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  const filteredAndSorted = useMemo(() => {
    let list = [...committees];

    // Search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        c => c.name.toLowerCase().includes(q) || c.committeeId.toLowerCase().includes(q)
      );
    }

    // Filter
    if (filterBy === "CERTIFIED") {
      list = list.filter(c => c.odei >= 80.0 && c.cdqi >= 80.0);
    } else if (filterBy === "HIGH_RISK") {
      list = list.filter(c => c.odei < 80.0 || c.cdqi < 80.0);
    }

    // Sort
    list.sort((a, b) => {
      if (sortBy === "ODEI") return b.odei - a.odei;
      if (sortBy === "CDQI") return b.cdqi - a.cdqi;
      if (sortBy === "DIRATIO") return b.committeeDIRatio - a.committeeDIRatio;
      return 0;
    });

    return list;
  }, [committees, sortBy, filterBy, searchQuery]);

  return (
    <div className="space-y-4">
      {/* Controls Strip */}
      <div className="bg-[#101622] border border-[#1f2c42] p-3 rounded-xl flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3 font-mono text-xs">
        {/* Search Input */}
        <div className="relative flex-1 max-w-md">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search committees by name or ID..."
            className="w-full bg-[#0b0f17] border border-[#23334d] rounded-lg px-3 py-1.5 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-400 text-xs"
          />
        </div>

        {/* Filters & Sorting */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1 bg-[#0b0f17] p-1 rounded-lg border border-[#1e2a3f]">
            <span className="text-slate-400 px-1 text-[11px]">Filter:</span>
            {(["ALL", "CERTIFIED", "HIGH_RISK"] as FilterTier[]).map((tier) => (
              <button
                key={tier}
                onClick={() => setFilterBy(tier)}
                className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                  filterBy === tier
                    ? "bg-[#1f2d45] text-cyan-400 font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tier === "ALL" ? "All (3)" : tier === "CERTIFIED" ? "Certified" : "High Risk"}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-1 bg-[#0b0f17] p-1 rounded-lg border border-[#1e2a3f]">
            <span className="text-slate-400 px-1 text-[11px]">Sort By:</span>
            {(["ODEI", "CDQI", "DIRATIO"] as SortField[]).map((field) => (
              <button
                key={field}
                onClick={() => setSortBy(field)}
                className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                  sortBy === field
                    ? "bg-[#1f2d45] text-cyan-400 font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {field}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grid */}
      {filteredAndSorted.length === 0 ? (
        <div className="bg-[#101622] border border-[#1f2c42] p-8 text-center rounded-xl text-slate-400 font-mono text-xs">
          No committees match current filter criteria.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filteredAndSorted.map((c, idx) => (
            <CommitteeScorecard key={c.committeeId} committee={c} rank={idx + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
