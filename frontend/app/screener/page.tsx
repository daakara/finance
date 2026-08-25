"use client";

import { useState } from "react";
import { GemCandidate, runHiddenGemsScreener } from "../../lib/api";

export default function ScreenerPage() {
  const [loading, setLoading] = useState(false);
  const [tickerInput, setTickerInput] = useState("ENPH, SEDG, RUN, FSLR, PLTR, SNOW, CRWD");
  const [results, setResults] = useState<GemCandidate[]>([]);
  const [filterQuery, setFilterQuery] = useState("");
  const [sortField, setSortField] = useState<"ticker" | "composite_score" | "risk_rating">("composite_score");
  const [sortAsc, setSortAsc] = useState(false);

  const handleRunScreener = async () => {
    setLoading(true);
    try {
      const candidateTickers = tickerInput
        .split(",")
        .map((t) => t.trim().toUpperCase())
        .filter((t) => t.length > 0);

      const response = await runHiddenGemsScreener(candidateTickers);
      setResults(response.results);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (field: "ticker" | "composite_score" | "risk_rating") => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const filteredResults = results
    .filter((r) => r.ticker.toLowerCase().includes(filterQuery.toLowerCase()))
    .sort((a, b) => {
      let compA = a[sortField];
      let compB = b[sortField];
      if (typeof compA === "string") {
        return sortAsc
          ? (compA as string).localeCompare(compB as string)
          : (compB as string).localeCompare(compA as string);
      }
      return sortAsc ? (compA as number) - (compB as number) : (compB as number) - (compA as number);
    });

  const exportCSV = () => {
    const csvContent =
      "data:text/csv;charset=utf-8," +
      ["Ticker,Composite Score,Risk Rating,Investment Thesis"]
        .concat(results.map((r) => `"${r.ticker}",${r.composite_score},"${r.risk_rating}","${r.investment_thesis}"`))
        .join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "hidden_gems_screener_export.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <div className="bg-[#111722] border border-[#243044] rounded-xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 tracking-tight">
              Hidden Gems Screener Engine
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Multi-factor quantitative screening evaluating momentum, volatility regimes & fundamental thesis
            </p>
          </div>

          <div className="flex items-center space-x-2">
            {results.length > 0 && (
              <button
                onClick={exportCSV}
                className="bg-[#1b2434] hover:bg-[#162030] text-slate-200 border border-[#364866] font-mono text-xs px-3.5 py-2.5 rounded-lg transition-colors focus-ring"
              >
                ?? Export CSV
              </button>
            )}

            <button
              onClick={handleRunScreener}
              disabled={loading}
              className="bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-slate-950 font-bold px-5 py-2.5 rounded-lg text-sm transition-all focus-ring disabled:opacity-50 shadow-lg shadow-emerald-950/40 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></span>
                  <span>Screening...</span>
                </>
              ) : (
                <span>Execute Screener</span>
              )}
            </button>
          </div>
        </div>

        {/* Ticker Input Bar */}
        <div className="space-y-1.5 pt-2 border-t border-[#1b2434]">
          <label className="text-xs font-medium text-slate-300 block">Candidate Tickers Universe (comma-separated):</label>
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            placeholder="ENPH, SEDG, RUN..."
            className="w-full bg-[#090d14] border border-[#243044] focus:border-emerald-500 text-slate-100 text-sm rounded-lg px-3.5 py-2 font-mono focus-ring"
          />
        </div>
      </div>

      {/* Interactive Data Table View */}
      {results.length > 0 ? (
        <div className="bg-[#111722] border border-[#243044] rounded-xl overflow-hidden shadow-xl space-y-4">
          <div className="p-4 border-b border-[#243044] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <span className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
              Screened Asset Universe ({filteredResults.length} Assets)
            </span>
            <input
              type="text"
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              placeholder="Filter results by ticker..."
              className="bg-[#090d14] border border-[#243044] text-xs text-slate-100 rounded-md px-3 py-1.5 font-mono focus-ring w-full sm:w-60"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-200">
              <thead className="bg-[#090d14] border-b border-[#243044] text-xs uppercase font-mono text-slate-400">
                <tr>
                  <th onClick={() => handleSort("ticker")} className="px-6 py-3.5 cursor-pointer hover:text-cyan-400 select-none">
                    Ticker {sortField === "ticker" ? (sortAsc ? "?" : "?") : ""}
                  </th>
                  <th onClick={() => handleSort("composite_score")} className="px-6 py-3.5 cursor-pointer hover:text-cyan-400 select-none">
                    Quant Score {sortField === "composite_score" ? (sortAsc ? "?" : "?") : ""}
                  </th>
                  <th onClick={() => handleSort("risk_rating")} className="px-6 py-3.5 cursor-pointer hover:text-cyan-400 select-none">
                    Risk Assessment {sortField === "risk_rating" ? (sortAsc ? "?" : "?") : ""}
                  </th>
                  <th className="px-6 py-3.5">Investment Thesis</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1b2434]">
                {filteredResults.map((gem) => (
                  <tr key={gem.ticker} className="hover:bg-[#162030] transition-colors">
                    <td className="px-6 py-4 font-bold font-mono text-cyan-400 text-base">
                      {gem.ticker}
                    </td>
                    <td className="px-6 py-4 font-mono">
                      <span className="bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 px-2.5 py-1 rounded-md text-xs font-bold">
                        {gem.composite_score.toFixed(1)} / 100
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-xs font-medium bg-[#1b2434] text-slate-200 border border-[#364866] px-2.5 py-1 rounded-md">
                        {gem.risk_rating}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-xs text-slate-300 max-w-md leading-relaxed">
                      {gem.investment_thesis}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="bg-[#111722] border border-[#243044] rounded-xl p-12 text-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-[#1b2434] text-emerald-400 flex items-center justify-center mx-auto text-xl font-mono">
            ??
          </div>
          <h3 className="text-base font-semibold text-slate-200">No Active Screening Results</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Click &quot;Execute Screener&quot; above to run multi-factor quantitative screening across candidate assets.
          </p>
        </div>
      )}
    </div>
  );
}

