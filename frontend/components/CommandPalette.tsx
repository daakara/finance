"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSymbol: (symbol: string) => void;
}

const ALL_TICKERS = [
  { ticker: "AAPL", name: "Apple Inc.", type: "Equity", sector: "Technology" },
  { ticker: "MSFT", name: "Microsoft Corporation", type: "Equity", sector: "Technology" },
  { ticker: "GOOGL", name: "Alphabet Inc.", type: "Equity", sector: "Communication" },
  { ticker: "NVDA", name: "NVIDIA Corporation", type: "Equity", sector: "Semiconductors" },
  { ticker: "TSLA", name: "Tesla, Inc.", type: "Equity", sector: "Consumer Cyclical" },
  { ticker: "BTC-USD", name: "Bitcoin USD", type: "Crypto", sector: "Digital Currency" },
  { ticker: "ETH-USD", name: "Ethereum USD", type: "Crypto", sector: "Digital Currency" },
  { ticker: "ENPH", name: "Enphase Energy", type: "Equity", sector: "Clean Tech" },
  { ticker: "PLTR", name: "Palantir Technologies", type: "Equity", sector: "Software" },
  { ticker: "CRWD", name: "CrowdStrike Holdings", type: "Equity", sector: "Cybersecurity" },
];

export default function CommandPalette({ isOpen, onClose, onSelectSymbol }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const router = useRouter();

  const filteredItems = ALL_TICKERS.filter(
    (item) =>
      item.ticker.toLowerCase().includes(query.toLowerCase()) ||
      item.name.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev < filteredItems.length - 1 ? prev + 1 : 0));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : filteredItems.length - 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (filteredItems[selectedIndex]) {
          onSelectSymbol(filteredItems[selectedIndex].ticker);
          onClose();
        } else if (query.trim()) {
          onSelectSymbol(query.trim().toUpperCase());
          onClose();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, filteredItems, selectedIndex, query, onClose, onSelectSymbol]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 bg-slate-950/80 backdrop-blur-sm p-4">
      <div
        className="w-full max-w-xl bg-[#111722] border border-[#364866] rounded-xl shadow-2xl overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center px-4 border-b border-[#243044]">
          <span className="text-slate-400 font-mono text-sm mr-2">?</span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type ticker or search command..."
            className="w-full bg-transparent py-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none font-mono"
            autoFocus
          />
          <kbd className="hidden sm:inline-block text-[10px] text-slate-400 bg-[#1b2434] border border-[#243044] px-1.5 py-0.5 rounded font-mono">
            ESC
          </kbd>
        </div>

        <div className="max-h-80 overflow-y-auto p-2 space-y-1">
          <div className="px-3 py-1.5 text-[10px] font-mono text-slate-400 uppercase tracking-wider">
            Quick Asset Jump & Command Options
          </div>

          {filteredItems.length > 0 ? (
            filteredItems.map((item, idx) => (
              <button
                key={item.ticker}
                onClick={() => {
                  onSelectSymbol(item.ticker);
                  onClose();
                }}
                className={`w-full text-left px-3.5 py-2.5 rounded-lg text-sm flex items-center justify-between transition-colors ${
                  idx === selectedIndex
                    ? "bg-cyan-950/60 text-cyan-400 border border-cyan-800/80"
                    : "text-slate-300 hover:bg-[#162030]"
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className="font-bold font-mono text-slate-100">{item.ticker}</span>
                  <span className="text-xs text-slate-400">{item.name}</span>
                </div>

                <div className="flex items-center space-x-2">
                  <span className="text-[10px] font-mono text-slate-400 bg-[#1b2434] px-2 py-0.5 rounded border border-[#243044]">
                    {item.sector}
                  </span>
                  <span className="text-xs text-cyan-400">?</span>
                </div>
              </button>
            ))
          ) : (
            <button
              onClick={() => {
                onSelectSymbol(query.trim().toUpperCase());
                onClose();
              }}
              className="w-full text-left px-3.5 py-3 text-sm text-cyan-400 hover:bg-[#162030] font-mono"
            >
              Load &quot;{query.toUpperCase()}&quot; on terminal
            </button>
          )}

          <div className="border-t border-[#1b2434] pt-2 mt-2 px-2 flex items-center justify-between">
            <button
              onClick={() => {
                router.push("/screener");
                onClose();
              }}
              className="text-xs text-emerald-400 hover:underline font-mono"
            >
              ? Open Hidden Gems Screener Page
            </button>
            <span className="text-[10px] text-slate-500 font-mono">Use ? ? to navigate</span>
          </div>
        </div>
      </div>
    </div>
  );
}

