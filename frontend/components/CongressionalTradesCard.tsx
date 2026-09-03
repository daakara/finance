"use client";

import { useState, useEffect } from "react";
import { CongressTradeItem, OptionsFlowItem } from "../lib/api";
import SmartMoneyDetailModal from "./SmartMoneyDetailModal";

interface CongressionalTradesCardProps {
  symbol?: string;
  congressTrades?: CongressTradeItem[];
  optionsFlow?: OptionsFlowItem[];
  userRole?: "DAY_TRADER" | "LONG_TERM";
  onSelectSymbol?: (symbol: string) => void;
}

export default function CongressionalTradesCard({
  symbol = "NVDA",
  congressTrades = [],
  optionsFlow = [],
  userRole = "LONG_TERM",
  onSelectSymbol,
}: CongressionalTradesCardProps) {
  const isDayTrader = userRole === "DAY_TRADER";
  const [timeframe, setTimeframe] = useState<"7D" | "30D" | "90D" | "1Y" | "ALL">("30D");
  const [selectedCongress, setSelectedCongress] = useState<CongressTradeItem | null>(null);
  const [selectedOptions, setSelectedOptions] = useState<OptionsFlowItem | null>(null);
  const [vernacularMode, setVernacularMode] = useState<"PLAIN_ENGLISH" | "PRO_QUANT">("PLAIN_ENGLISH");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("ARX_VERNACULAR_MODE") as "PLAIN_ENGLISH" | "PRO_QUANT" | null;
      if (saved) setVernacularMode(saved);
    } catch {}

    const handleVernacular = (e: Event) => {
      const custom = e as CustomEvent<"PLAIN_ENGLISH" | "PRO_QUANT">;
      if (custom.detail) setVernacularMode(custom.detail);
    };

    window.addEventListener("finance:vernacular-change", handleVernacular);
    return () => window.removeEventListener("finance:vernacular-change", handleVernacular);
  }, []);

  const isPlain = vernacularMode === "PLAIN_ENGLISH";
  const cleanSym = symbol.toUpperCase().replace("-USD", "");
  const isCrypto = ["BTC", "ETH", "SOL"].includes(cleanSym) || symbol.toUpperCase().includes("BTC") || symbol.toUpperCase().includes("ETH") || symbol.toUpperCase().includes("SOL");

  const filteredCongressTrades = congressTrades.filter((t) => {
    if (timeframe === "ALL") return true;
    const dStr = t.filing_date || t.transaction_date;
    if (!dStr) return true;
    const tDate = new Date(dStr);
    if (isNaN(tDate.getTime())) return true;
    const diffDays = Math.max(0, (Date.now() - tDate.getTime()) / (1000 * 60 * 60 * 24));
    if (timeframe === "7D") return diffDays <= 7;
    if (timeframe === "30D") return diffDays <= 30;
    if (timeframe === "90D") return diffDays <= 90;
    if (timeframe === "1Y") return diffDays <= 365;
    return true;
  });

  // Specialized Crypto Institutional & Regulatory Intelligence View
  if (isCrypto) {
    const cryptoMetadata = {
      BTC: {
        name: "Bitcoin",
        category: "Decentralized Digital Commodity",
        regStatus: "CFTC Non-Security Commodity (SEC Approved Spot ETFs)",
        etfs: [
          { ticker: "IBIT", name: "BlackRock iShares Bitcoin Trust", fee: "0.25%" },
          { ticker: "FBTC", name: "Fidelity Wise Origin Bitcoin Fund", fee: "0.25%" },
          { ticker: "ARKB", name: "ARK 21Shares Bitcoin ETF", fee: "0.21%" },
        ],
        bills: [
          { name: "BITCOIN Act of 2024 (S.4912)", desc: "US Strategic Bitcoin Reserve Bill (Sen. Cynthia Lummis)" },
          { name: "FIT21 Act (H.R. 4763)", desc: "Commodity regulatory clarity under CFTC jurisdiction (Passed House)" },
        ],
        proxies: [
          { symbol: "MSTR", name: "MicroStrategy", reason: "Largest corporate Bitcoin treasury (>226k BTC)" },
          { symbol: "COIN", name: "Coinbase Global", reason: "Primary custodian for 8 of 11 US spot ETFs" },
        ],
      },
      ETH: {
        name: "Ethereum",
        category: "Layer-1 Decentralized Protocol",
        regStatus: "CFTC Commodity / Non-Security (SEC Approved Spot ETFs)",
        etfs: [
          { ticker: "ETHA", name: "iShares Ethereum Trust", fee: "0.25%" },
          { ticker: "FETH", name: "Fidelity Ethereum Fund", fee: "0.25%" },
          { ticker: "ETHE", name: "Grayscale Ethereum Trust", fee: "2.50%" },
        ],
        bills: [
          { name: "FIT21 Act (H.R. 4763)", desc: "Smart contract network decentralization thresholds" },
          { name: "SAB 121 Custody Relief", desc: "Bank custody accounting rule reform (Bipartisan CRA)" },
        ],
        proxies: [
          { symbol: "COIN", name: "Coinbase Global", reason: "Primary staking & institutional custody partner" },
          { symbol: "NVDA", name: "NVIDIA Corp", reason: "Hardware backbone for decentralized compute" },
        ],
      },
      SOL: {
        name: "Solana",
        category: "High-Throughput PoS Smart Contract Network",
        regStatus: "L1 Digital Commodity (Pending Spot ETF S-1 Filings)",
        etfs: [
          { ticker: "GSOL", name: "Grayscale Solana Trust", fee: "2.50%" },
          { ticker: "VSOL (Pending)", name: "VanEck Solana Trust S-1", fee: "TBD" },
          { ticker: "21SOL (Pending)", name: "21Shares Solana ETF S-1", fee: "TBD" },
        ],
        bills: [
          { name: "FIT21 Act (H.R. 4763)", desc: "De-facto decentralization metrics for Layer-1 networks" },
        ],
        proxies: [
          { symbol: "COIN", name: "Coinbase Global", reason: "Institutional staking and liquidity gateway" },
        ],
      },
    }[cleanSym as "BTC" | "ETH" | "SOL"] || {
      name: symbol,
      category: "Digital Asset Protocol",
      regStatus: "Cryptographic Commodity / Protocol",
      etfs: [{ ticker: "IBIT", name: "BlackRock Spot Bitcoin ETF", fee: "0.25%" }],
      bills: [{ name: "FIT21 Act", desc: "Digital Asset Regulatory Framework" }],
      proxies: [{ symbol: "COIN", name: "Coinbase Global", reason: "Crypto Market Leader" }],
    };

    return (
      <div className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono transition-colors ${
        isDayTrader ? "border-amber-900/40" : "border-[#243044]"
      }`}>
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
              <h3 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
                <span>
                  {isPlain
                    ? `🏛️ ${cryptoMetadata.name} Institutional & Regulatory Money Flow`
                    : `🏛️ ${cryptoMetadata.name} Institutional Asset Classification & Regulatory Structure`}
                </span>
              </h3>
            </div>
            <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5 font-sans">
              {isPlain
                ? "Decentralized crypto protocols have no corporate CEO filing SEC Form 4. Here is how Wall Street, Spot ETFs, and US Congress actually move billions into this asset."
                : "Decentralized digital commodities have no corporate executives filing SEC Form 4 insider disclosures. Institutional capital allocation routes through regulated Spot ETFs and public corporate proxies."}
            </p>
          </div>

          <span className="text-[10px] px-2.5 py-1 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-bold uppercase">
            {cryptoMetadata.category}
          </span>
        </div>

        {/* Regulatory & Congressional Landscape */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-2">
            <div className="flex items-center justify-between">
              <strong className="text-emerald-400 font-bold flex items-center gap-1.5">
                <span>⚖️</span>
                <span>{isPlain ? "US Legal & Regulatory Status" : "Regulatory Classification (SEC / CFTC)"}</span>
              </strong>
              <span className="text-[9px] bg-emerald-950 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-800 font-bold">
                COMMODITY
              </span>
            </div>
            <p className="text-slate-300 text-[11px] font-sans leading-relaxed">
              {cryptoMetadata.regStatus}
            </p>
          </div>

          <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-2">
            <strong className="text-purple-300 font-bold flex items-center gap-1.5">
              <span>🏛️</span>
              <span>{isPlain ? "Active US Congressional Bills" : "Legislative Jurisdictions & Bills"}</span>
            </strong>
            <ul className="text-[11px] text-slate-300 font-sans space-y-1">
              {cryptoMetadata.bills.map((b, idx) => (
                <li key={idx} className="flex flex-col">
                  <span className="text-cyan-300 font-semibold">{b.name}</span>
                  <span className="text-slate-400 text-[10px]">{b.desc}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Spot ETF Vehicles */}
        <div className="bg-[#090d14] p-3.5 rounded-xl border border-[#1b2434] space-y-2.5">
          <div className="flex items-center justify-between">
            <strong className="text-amber-300 text-xs font-bold flex items-center gap-1.5">
              <span>💼</span>
              <span>{isPlain ? "Wall Street Spot ETF Vehicles (Institutional Accumulation)" : "Institutional Spot ETP Inflow & Custody Vehicles"}</span>
            </strong>
            <span className="text-[10px] text-slate-400 font-mono">Regulated Under SEC 1933 Act</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
            {cryptoMetadata.etfs.map((etf) => (
              <div key={etf.ticker} className="bg-[#05080e] p-2.5 rounded-lg border border-[#162030] space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-cyan-400 font-bold font-mono">{etf.ticker}</span>
                  <span className="text-[10px] text-slate-400">Expense: {etf.fee}</span>
                </div>
                <p className="text-[10px] text-slate-300 font-sans leading-tight">{etf.name}</p>
              </div>
            ))}
          </div>
        </div>

        {/* 1-Click Corporate Treasury & Insider Proxies */}
        <div className="bg-[#0b1019] p-4 rounded-xl border border-cyan-900/40 space-y-3">
          <div className="flex items-center justify-between">
            <strong className="text-slate-100 text-xs font-bold flex items-center gap-1.5">
              <span>🔍</span>
              <span>
                {isPlain
                  ? "Track Corporate Insiders & Congress on Public Crypto Stocks:"
                  : "Track Executive Form 4 Insiders & STOCK Act Disclosures on Public Crypto Proxies:"}
              </span>
            </strong>
            <span className="text-[10px] bg-cyan-950 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-800 font-mono">
              1-CLICK INSPECTION
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {cryptoMetadata.proxies.map((proxy) => (
              <div
                key={proxy.symbol}
                onClick={() => onSelectSymbol && onSelectSymbol(proxy.symbol)}
                className="flex items-center justify-between p-3 rounded-lg bg-[#06090f] hover:bg-[#162030] border border-[#1b2434] hover:border-cyan-500/50 transition-all cursor-pointer group"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white group-hover:text-cyan-300">{proxy.symbol}</span>
                    <span className="text-[10px] text-slate-400 font-sans">{proxy.name}</span>
                  </div>
                  <p className="text-[10px] text-slate-400 font-sans mt-0.5">{proxy.reason}</p>
                </div>
                <button className="px-2 py-1 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 rounded text-[10px] font-bold shrink-0">
                  Inspect Insiders →
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className={`bg-[#111722] border rounded-xl p-4 sm:p-5 shadow-xl space-y-4 font-mono transition-colors ${
        isDayTrader ? "border-amber-900/40" : "border-[#243044]"
      }`}>
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2434] pb-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className={`w-2.5 h-2.5 rounded-full ${
                isDayTrader ? "bg-amber-400" : "bg-purple-400"
              } animate-pulse`}></span>
              <h3 className="text-sm sm:text-base font-bold text-slate-100 tracking-tight flex items-center gap-2">
                <span>
                  {isDayTrader
                    ? (isPlain ? "⚡ Institutional Options Sweeps & Case Studies" : "⚡ Institutional Options Sweeps & Flow Analysis")
                    : (isPlain ? "🏛️ Follow The Money (Capitol Hill Insider Trades)" : "🏛️ Capitol Hill & Institutional Order Flow Radar")}
                </span>
              </h3>
            </div>
            <p className="text-[11px] sm:text-xs text-slate-400 mt-0.5">
              {isDayTrader
                ? (isPlain ? `Curated institutional call/put sweep setups and positioning on ${symbol}.` : `Curated options order flow & volume-to-open-interest analysis for ${symbol}`)
                : (isPlain ? `See which members of US Congress bought or sold ${symbol} and their filing delays.` : `STOCK Act Title I Article 105 disclosures & congressional committee alignment for ${symbol}`)}
            </p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-[9px] font-bold px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-800/80 inline-flex items-center gap-1">
                <span>🏛️</span> {isDayTrader ? (isPlain ? "Institutional Options Flow Intelligence" : "Curated Options Flow & Gamma Exposure Analysis") : (isPlain ? "STOCK Act Legal Disclosures" : "STOCK Act Statutory Filing Disclosures (Public Law 112-105)")}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {!isDayTrader && (
              <div role="radiogroup" aria-label="Filing timeframe" className="flex items-center bg-[#090d14] p-0.5 rounded-lg border border-[#243044]">
                {[
                  { id: "7D", label: "7D" },
                  { id: "30D", label: "30D" },
                  { id: "90D", label: "90D" },
                  { id: "1Y", label: "1Y" },
                  { id: "ALL", label: "All" },
                ].map((tf) => (
                  <button
                    key={tf.id}
                    role="radio"
                    aria-checked={timeframe === tf.id}
                    onClick={() => setTimeframe(tf.id as any)}
                    className={`px-2 py-0.5 text-[10px] font-bold rounded transition-colors ${
                      timeframe === tf.id
                        ? "bg-purple-600 text-white font-black shadow-sm"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {tf.label}
                  </button>
                ))}
              </div>
            )}

            <span className={`text-[11px] px-2.5 py-1 rounded-md font-semibold border ${
              isDayTrader
                ? "text-amber-400 bg-amber-950/60 border-amber-800/80"
                : "text-purple-400 bg-purple-950/60 border-purple-800/80"
            }`}>
              {isDayTrader ? `${optionsFlow.length} Sweeps` : `${filteredCongressTrades.length} Trades`}
            </span>
          </div>
        </div>

        {/* Content Table / Card list */}
        {isDayTrader ? (
          /* Day Trader: Options Flow Table */
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse font-sans">
              <thead>
                <tr className="border-b border-[#1b2434] text-slate-400 font-mono text-[11px]">
                  <th className="pb-2 font-semibold">Time</th>
                  <th className="pb-2 font-semibold">Type</th>
                  <th className="pb-2 font-semibold">Strike / Exp</th>
                  <th className="pb-2 font-semibold text-right">Premium</th>
                  <th className="pb-2 font-semibold text-right">Vol / OI</th>
                  <th className="pb-2 font-semibold text-right">Sentiment</th>
                  <th className="pb-2 font-semibold text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#162030]">
                {optionsFlow.length > 0 ? (
                  optionsFlow.map((flow, i) => (
                    <tr
                      key={i}
                      onClick={() => setSelectedOptions(flow)}
                      className="hover:bg-[#162030] cursor-pointer transition-colors group"
                    >
                      <td className="py-2.5 font-mono text-slate-400 text-[11px]">{flow.time}</td>
                      <td className="py-2.5">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          flow.type.includes("CALL")
                            ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/60"
                            : flow.type.includes("PUT")
                            ? "bg-rose-950/80 text-rose-400 border border-rose-800/60"
                            : "bg-cyan-950/80 text-cyan-400 border border-cyan-800/60"
                        }`}>
                          {flow.type}
                        </span>
                      </td>
                      <td className="py-2.5 text-slate-200">
                        <div className="font-semibold">{flow.strike}</div>
                        <div className="text-[10px] text-slate-400">{flow.expiration}</div>
                      </td>
                      <td className="py-2.5 font-bold text-amber-400 text-right tabular-nums">{flow.premium}</td>
                      <td className="py-2.5 font-semibold text-slate-300 text-right tabular-nums">{flow.volume_oi_ratio}x</td>
                      <td className="py-2.5 text-right">
                        <span className="text-[11px] font-bold text-emerald-400">{flow.sentiment}</span>
                      </td>
                      <td className="py-2.5 text-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedOptions(flow);
                          }}
                          className="bg-cyan-950 hover:bg-cyan-900 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded text-[10px] font-bold"
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-slate-500 text-xs">
                      No unusual intraday sweeps detected for {symbol} in current session window.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        ) : (
          /* Long Term Investor: Congressional Trades Disclosure List */
          <div className="space-y-3">
            {filteredCongressTrades.length > 0 ? (
              filteredCongressTrades.map((trade, i) => (
                <div
                  key={i}
                  onClick={() => setSelectedCongress(trade)}
                  className="flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-lg bg-[#090d14] hover:bg-[#162030] border border-[#1e293b] hover:border-purple-500/40 transition-all cursor-pointer gap-3 group"
                >
                  <div className="space-y-1.5 min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-bold text-white text-sm group-hover:text-purple-300">{trade.politician}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1b2434] text-slate-400 font-semibold">{trade.chamber}</span>
                      {trade.legislative_alignment_score !== undefined && (
                        <span className={`text-[10px] px-2 py-0.5 rounded font-bold border ${
                          trade.legislative_alignment_score >= 80
                            ? "bg-purple-950/80 text-purple-300 border-purple-700/80"
                            : trade.legislative_alignment_score >= 60
                            ? "bg-cyan-950/80 text-cyan-300 border-cyan-800/80"
                            : "bg-[#162030] text-slate-400 border-[#243044]"
                        }`}>
                          ⚖️ Alignment: {trade.legislative_alignment_score}/100
                        </span>
                      )}
                      {trade.staleness_badge && (
                        <span className={`text-[10px] px-2 py-0.5 rounded font-bold border ${
                          trade.staleness_status === "LATE_FILER"
                            ? "bg-rose-950/80 text-rose-300 border-rose-800/80"
                            : trade.staleness_status === "AGING"
                            ? "bg-amber-950/80 text-amber-300 border-amber-800/80"
                            : "bg-emerald-950/80 text-emerald-300 border-emerald-800/80"
                        }`}>
                          {trade.staleness_badge}
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="font-semibold text-cyan-400">{trade.transaction_type}</span>
                      <span>•</span>
                      <span>{trade.amount_range}</span>
                      <span>•</span>
                      <span>Filed: {trade.filing_date} ({trade.days_to_filing}d lag)</span>
                    </div>
                    {trade.staleness_warning && (
                      <div className="text-[10px] text-rose-400/90 font-sans font-medium flex items-center gap-1 bg-rose-950/30 border border-rose-900/40 px-2 py-0.5 rounded">
                        <span>⚠️</span>
                        <span>{trade.staleness_warning}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center space-x-3 text-right shrink-0">
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Return Since Filing</span>
                      <span className={`text-sm sm:text-base font-bold tabular-nums ${
                        trade.performance_since_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                      }`}>
                        {trade.performance_since_pct >= 0 ? `+${trade.performance_since_pct}%` : `${trade.performance_since_pct}%`}
                      </span>
                    </div>
                    <span className={`px-2.5 py-1 rounded text-[11px] font-bold border ${
                      trade.sentiment.includes("Bullish")
                        ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/80"
                        : "bg-slate-800 text-slate-300 border-slate-700"
                    }`}>
                      {trade.sentiment}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedCongress(trade);
                      }}
                      className="bg-purple-950 hover:bg-purple-900 text-purple-300 border border-purple-800 px-2 py-1 rounded text-[10px] font-bold"
                    >
                      Inspect
                    </button>
                  </div>
                </div>
              ))
            ) : (
              <div className="p-4 bg-[#090d14] rounded-lg border border-[#243044] text-center text-xs text-slate-400">
                No recent Capitol Hill transactions reported for {symbol} within statutory filing windows.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Forensic Interactive Detail Drawer Modal */}
      <SmartMoneyDetailModal
        congressItem={selectedCongress}
        optionsItem={selectedOptions}
        onClose={() => {
          setSelectedCongress(null);
          setSelectedOptions(null);
        }}
        onSelectSymbol={onSelectSymbol}
      />
    </>
  );
}