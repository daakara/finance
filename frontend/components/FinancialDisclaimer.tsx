import React from "react";
import Link from "next/link";

interface FinancialDisclaimerProps {
  variant?: "footer" | "compact" | "card";
  className?: string;
}

export default function FinancialDisclaimer({ variant = "footer", className = "" }: FinancialDisclaimerProps) {
  if (variant === "compact") {
    return (
      <div className={`p-2.5 bg-[#06090f] border border-[#172235] rounded-xl text-[10px] text-slate-400 font-sans leading-relaxed ${className}`}>
        <strong className="text-slate-300 font-bold">Important Notice: </strong>
        ARX is an automated quantitative research and analytical tool, not a registered investment adviser or broker-dealer. Content is for informational and educational purposes only and does not constitute personalized financial or investment advice. All investments involve risk of loss.
      </div>
    );
  }

  return (
    <footer className={`bg-[#060a12] border-t border-[#182335] py-8 px-4 text-[11px] text-slate-400 font-sans ${className}`}>
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Footer Navigation Hub */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 pb-6 border-b border-[#141d2c] text-xs">
          <div className="space-y-2.5">
            <h4 className="font-mono text-white text-[11px] uppercase tracking-wider font-bold">
              Core Platform
            </h4>
            <ul className="space-y-1.5 text-slate-400 font-mono text-[11px]">
              <li><Link href="/" className="hover:text-cyan-400 transition-colors">Live Terminal</Link></li>
              <li><Link href="/screener/" className="hover:text-cyan-400 transition-colors">Multi-Factor Screener</Link></li>
              <li><Link href="/compare/" className="hover:text-cyan-400 transition-colors">Asset Comparison</Link></li>
              <li><Link href="/portfolio/" className="hover:text-cyan-400 transition-colors">Risk Allocator</Link></li>
            </ul>
          </div>

          <div className="space-y-2.5">
            <h4 className="font-mono text-white text-[11px] uppercase tracking-wider font-bold">
              Smart Money
            </h4>
            <ul className="space-y-1.5 text-slate-400 font-mono text-[11px]">
              <li><Link href="/smart-money/" className="hover:text-cyan-400 transition-colors">Congressional STOCK Act Radar</Link></li>
              <li><Link href="/smart-money/late-filers/" className="hover:text-cyan-400 transition-colors">Late-Filer Hall of Shame</Link></li>
              <li><Link href="/politician/nancy-pelosi/" className="hover:text-cyan-400 transition-colors">Nancy Pelosi Portfolio</Link></li>
              <li><Link href="/committee/armed-services/" className="hover:text-cyan-400 transition-colors">Committee Jurisdictions</Link></li>
            </ul>
          </div>

          <div className="space-y-2.5">
            <h4 className="font-mono text-white text-[11px] uppercase tracking-wider font-bold">
              Quant Glossary
            </h4>
            <ul className="space-y-1.5 text-slate-400 font-mono text-[11px]">
              <li><Link href="/glossary/arx-model/" className="hover:text-cyan-400 transition-colors">ARX Econometric Model</Link></li>
              <li><Link href="/glossary/minervini-vcp/" className="hover:text-cyan-400 transition-colors">Minervini VCP Pattern</Link></li>
              <li><Link href="/glossary/cornish-fisher-var/" className="hover:text-cyan-400 transition-colors">Cornish-Fisher VaR</Link></li>
              <li><Link href="/glossary/" className="text-cyan-400 hover:underline">All Quantitative Terms →</Link></li>
            </ul>
          </div>

          <div className="space-y-2.5">
            <h4 className="font-mono text-white text-[11px] uppercase tracking-wider font-bold">
              Platform Benchmarks
            </h4>
            <ul className="space-y-1.5 text-slate-400 font-mono text-[11px]">
              <li><Link href="/vs/quiver-quantitative/" className="hover:text-cyan-400 transition-colors">ARX vs Quiver Quant</Link></li>
              <li><Link href="/vs/unusual-whales/" className="hover:text-cyan-400 transition-colors">ARX vs Unusual Whales</Link></li>
              <li><Link href="/vs/koyfin/" className="hover:text-cyan-400 transition-colors">ARX vs Koyfin</Link></li>
              <li><Link href="/vs/bloomberg-terminal/" className="hover:text-cyan-400 transition-colors">ARX vs Bloomberg</Link></li>
              <li><Link href="/vs/" className="text-cyan-400 hover:underline">All Comparisons →</Link></li>
            </ul>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#141d2c] pb-3">
          <div className="flex items-center gap-2">
            <span className="font-black text-white text-xs tracking-wider">ARX TERMINAL</span>
            <span className="text-[10px] font-mono text-cyan-400">Quantitative Decision Engine</span>
          </div>
          <span className="text-[10px] font-mono text-slate-400">
            Ruleset Version: 2026.09-v1 • Epistemic Invariant Gated
          </span>
        </div>
        <div className="text-[10px] leading-relaxed space-y-1.5 text-slate-400">
          <p>
            <strong className="text-slate-300">Regulatory & Non-Fiduciary Safe Harbor:</strong> ARX Terminal and its parent platform provide automated mathematical, statistical, and quantitative analytics based on public SEC disclosures and market feeds. ARX is NOT a registered investment adviser, broker-dealer, financial analyst, or commodity trading advisor under the U.S. Securities and Exchange Commission (SEC), FINRA, or any state or international securities regulatory authority.
          </p>
          <p>
            <strong className="text-slate-300">No Personalized Financial Advice:</strong> Nothing published on this platform constitutes an offer, solicitation, or recommendation to buy, hold, or sell any security, derivative, or financial instrument, nor does it constitute personalized financial, tax, legal, or investment advice. Terminology such as &ldquo;ACQUIRE&rdquo;, &ldquo;WATCH&rdquo;, &ldquo;BUY ZONE&rdquo;, and &ldquo;STOP LOSS&rdquo; represents automated quantitative model states and must not be construed as individualized trading instructions.
          </p>
          <p>
            <strong className="text-slate-300">Risk Disclosure:</strong> Securities investments, equities, and derivative transactions carry substantial risk of loss, including the possible loss of principal. Past quantitative edge, historical backtest returns, Piotroski scores, and factor ratings do not guarantee future performance. Investors must conduct independent due diligence and consult a licensed fiduciary financial advisor before executing any trade.
          </p>
        </div>
      </div>
    </footer>
  );
}
