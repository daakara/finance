"use client";

import { useEffect, useState } from "react";
import { SpotPriceRegistry, fetchAssetAnalytics, AnalyticsResponse } from "../lib/api";
import { getPersistedMarketSnapshot } from "../lib/marketDatabase";

interface ComparePairMatrixProps {
  symA: string;
  symB: string;
  nameA: string;
  nameB: string;
}

export default function ComparePairMatrix({ symA, symB, nameA, nameB }: ComparePairMatrixProps) {
  const [dataA, setDataA] = useState<AnalyticsResponse | null>(null);
  const [dataB, setDataB] = useState<AnalyticsResponse | null>(null);
  const [priceA, setPriceA] = useState<number | null>(() => {
    const reg = SpotPriceRegistry.get(symA);
    const snap = getPersistedMarketSnapshot(symA);
    return reg?.price && reg.price > 0 ? reg.price : snap?.currentPrice && snap.currentPrice > 0 ? snap.currentPrice : null;
  });
  const [priceB, setPriceB] = useState<number | null>(() => {
    const reg = SpotPriceRegistry.get(symB);
    const snap = getPersistedMarketSnapshot(symB);
    return reg?.price && reg.price > 0 ? reg.price : snap?.currentPrice && snap.currentPrice > 0 ? snap.currentPrice : null;
  });

  useEffect(() => {
    let isMounted = true;
    Promise.all([
      fetchAssetAnalytics(symA, "1mo", "1d").catch(() => null),
      fetchAssetAnalytics(symB, "1mo", "1d").catch(() => null),
    ]).then(([resA, resB]) => {
      if (!isMounted) return;
      if (resA) {
        setDataA(resA);
        if (resA.currentPrice !== null && resA.currentPrice > 0) setPriceA(resA.currentPrice);
      }
      if (resB) {
        setDataB(resB);
        if (resB.currentPrice !== null && resB.currentPrice > 0) setPriceB(resB.currentPrice);
      }
    });
    return () => {
      isMounted = false;
    };
  }, [symA, symB]);

  const scoresA = dataA?.factorScores || dataA?.dnaScores;
  const scoresB = dataB?.factorScores || dataB?.dnaScores;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs text-left border-collapse">
        <thead>
          <tr className="border-b border-[#1e293b] text-slate-400">
            <th className="py-2.5 px-3">Metric / Factor</th>
            <th className="py-2.5 px-3 text-cyan-400 font-bold">{symA} ({nameA})</th>
            <th className="py-2.5 px-3 text-amber-400 font-bold">{symB} ({nameB})</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#162030] text-slate-300">
          <tr>
            <td className="py-2.5 px-3 font-semibold text-slate-400">Current Spot Price</td>
            <td className="py-2.5 px-3 font-mono text-white font-bold">{priceA !== null ? `$${priceA.toFixed(2)}` : "Live Tape Required"}</td>
            <td className="py-2.5 px-3 font-mono text-white font-bold">{priceB !== null ? `$${priceB.toFixed(2)}` : "Live Tape Required"}</td>
          </tr>
          <tr>
            <td className="py-2.5 px-3 font-semibold text-slate-400">Composite Factor Score</td>
            <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">{scoresA?.compositeFactorScore !== undefined ? `${scoresA.compositeFactorScore} / 100` : "Awaiting SEC Filings"}</td>
            <td className="py-2.5 px-3 font-mono text-emerald-400 font-bold">{scoresB?.compositeFactorScore !== undefined ? `${scoresB.compositeFactorScore} / 100` : "Awaiting SEC Filings"}</td>
          </tr>
          <tr>
            <td className="py-2.5 px-3 font-semibold text-slate-400">Piotroski 9-Point F-Score</td>
            <td className="py-2.5 px-3 font-mono text-cyan-300 font-bold">{scoresA?.piotroskiFScore !== undefined ? `${scoresA.piotroskiFScore} / 9` : "N/A"}</td>
            <td className="py-2.5 px-3 font-mono text-amber-300 font-bold">{scoresB?.piotroskiFScore !== undefined ? `${scoresB.piotroskiFScore} / 9` : "N/A"}</td>
          </tr>
          <tr>
            <td className="py-2.5 px-3 font-semibold text-slate-400">Growth Score</td>
            <td className="py-2.5 px-3 font-mono">{scoresA?.growthScore !== undefined ? `${scoresA.growthScore} / 100` : "N/A"}</td>
            <td className="py-2.5 px-3 font-mono">{scoresB?.growthScore !== undefined ? `${scoresB.growthScore} / 100` : "N/A"}</td>
          </tr>
          <tr>
            <td className="py-2.5 px-3 font-semibold text-slate-400">Quality Score</td>
            <td className="py-2.5 px-3 font-mono">{scoresA?.qualityScore !== undefined ? `${scoresA.qualityScore} / 100` : "N/A"}</td>
            <td className="py-2.5 px-3 font-mono">{scoresB?.qualityScore !== undefined ? `${scoresB.qualityScore} / 100` : "N/A"}</td>
          </tr>
          <tr>
            <td className="py-2.5 px-3 font-semibold text-slate-400">Valuation Score</td>
            <td className="py-2.5 px-3 font-mono">{scoresA?.valuationScore !== undefined ? `${scoresA.valuationScore} / 100` : "N/A"}</td>
            <td className="py-2.5 px-3 font-mono">{scoresB?.valuationScore !== undefined ? `${scoresB.valuationScore} / 100` : "N/A"}</td>
          </tr>
          <tr>
            <td className="py-2.5 px-3 font-semibold text-slate-400">Institutional Verdict</td>
            <td className="py-2.5 px-3 font-sans text-emerald-400">{scoresA?.verdict || "Unverified Security — Filings Required"}</td>
            <td className="py-2.5 px-3 font-sans text-amber-400">{scoresB?.verdict || "Unverified Security — Filings Required"}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
