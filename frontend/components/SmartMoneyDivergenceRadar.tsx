"use client";

/**
 * Smart Money versus retail sentiment presentation.
 *
 * No canonical, per-record, point-in-time divergence evidence is currently
 * available to this component. Do not reconstruct apparent trading signals
 * from the former hard-coded six-symbol research examples.
 *
 * Future signal cards require independently sourced and dated observations
 * with explicit lifecycle, feed availability, and scoring provenance.
 */
export default function SmartMoneyDivergenceRadar() {
  return (
    <section
      aria-label="Smart Money vs Public Hype research"
      data-testid="smart-money-divergence-unavailable"
      className="rounded-2xl border border-[#243044] bg-[#0e1420] p-4 sm:p-6 text-slate-200 shadow-xl"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h2 className="text-base sm:text-lg font-bold text-white">
            Smart Money vs. Public Hype
          </h2>
          <p className="text-xs text-slate-400">
            Comparison research · not a current market signal
          </p>
        </div>
        <span className="inline-flex max-w-full items-center rounded-md border border-amber-800/70 bg-amber-950/50 px-2.5 py-1 text-[11px] font-semibold text-amber-300">
          Evidence unavailable
        </span>
      </div>

      <div
        role="status"
        className="mt-4 rounded-xl border border-[#304052] bg-[#090d14] p-4 space-y-2"
      >
        <p className="text-sm font-semibold text-slate-100">
          No verified divergence signals available
        </p>
        <p className="text-xs leading-relaxed text-slate-400">
          The previous example scores, prices and buying/selling narratives
          were static research illustrations without source-linked, dated
          evidence. They are withheld from investor-facing signals.
        </p>
        <p className="text-xs leading-relaxed text-slate-400">
          Congressional STOCK Act disclosures, SEC Form 4 insider activity,
          institutional 13F holdings, and public sentiment are separate data
          sources. A comparison needs verified observations, dates, security
          validity and an approved scoring method for each source.
        </p>
        <p className="text-xs font-medium text-amber-300">
          Missing evidence does not mean zero transactions, accumulation,
          distribution or a trading recommendation.
        </p>
      </div>
    </section>
  );
}
