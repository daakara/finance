"use client";

interface WatchlistSidebarProps {
  activeSymbol: string;
  onSelectSymbol: (symbol: string) => void;
}

const WATCHLIST_ITEMS = [
  { symbol: "AAPL", name: "Apple Inc.", price: "$224.30", change: "+1.25%", isUp: true },
  { symbol: "MSFT", name: "Microsoft", price: "$448.90", change: "-0.45%", isUp: false },
  { symbol: "GOOGL", name: "Alphabet", price: "$176.50", change: "+0.82%", isUp: true },
  { symbol: "NVDA", name: "NVIDIA", price: "$128.40", change: "+3.14%", isUp: true },
  { symbol: "TSLA", name: "Tesla", price: "$210.20", change: "-2.10%", isUp: false },
  { symbol: "BTC-USD", name: "Bitcoin", price: "$64,250", change: "+4.50%", isUp: true },
];

export default function WatchlistSidebar({ activeSymbol, onSelectSymbol }: WatchlistSidebarProps) {
  return (
    <div className="bg-[#111722] border border-[#243044] rounded-xl p-4 shadow-xl space-y-3 h-full">
      <div className="flex items-center justify-between border-b border-[#1b2434] pb-3">
        <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
          Watchlist & Signals
        </span>
        <span className="text-[10px] text-cyan-400 bg-cyan-950/60 border border-cyan-800/80 px-2 py-0.5 rounded font-mono">
          Live Stream
        </span>
      </div>

      <div className="space-y-1.5 overflow-y-auto max-h-[420px]">
        {WATCHLIST_ITEMS.map((item) => (
          <button
            key={item.symbol}
            onClick={() => onSelectSymbol(item.symbol)}
            className={`w-full text-left p-2.5 rounded-lg transition-all flex items-center justify-between border ${
              activeSymbol === item.symbol
                ? "bg-[#1b2434] border-cyan-500/60 text-slate-100 shadow-md"
                : "bg-[#090d14] border-[#243044] text-slate-300 hover:border-[#364866]"
            }`}
          >
            <div>
              <span className="font-bold font-mono text-sm block leading-tight text-slate-100">
                {item.symbol}
              </span>
              <span className="text-[10px] text-slate-400">{item.name}</span>
            </div>

            <div className="text-right font-mono">
              <span className="text-xs font-semibold block text-slate-200">{item.price}</span>
              <span className={`text-[10px] font-bold ${item.isUp ? "text-emerald-400" : "text-rose-400"}`}>
                {item.change}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

