"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Compass, Activity, ShieldCheck, Briefcase, Search } from "lucide-react";

interface SynthesisMobileDockProps {
  activeSymbol?: string | null;
}

export default function SynthesisMobileDock({ activeSymbol }: SynthesisMobileDockProps) {
  const pathname = usePathname();
  const cleanSymbol =
    activeSymbol && typeof activeSymbol === "string" && activeSymbol.trim()
      ? activeSymbol.trim().toUpperCase()
      : null;
  const symbolParam = cleanSymbol ? `?symbol=${encodeURIComponent(cleanSymbol)}` : "";

  const handleOpenSearch = () => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("arx:open-omni-search"));
    }
  };

  return (
    <nav
      className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-[var(--surface-1)] border-t border-[var(--border-subtle)] backdrop-blur-lg px-2 py-1 shadow-2xl"
      aria-label="Mobile Navigation Dock"
      data-synthesis-mobile-dock="true"
    >
      <div className="grid grid-cols-5 gap-1 items-center max-w-md mx-auto">
        <Link
          href="/radar"
          className={`flex flex-col items-center justify-center p-1.5 rounded-xl font-mono text-[10px] min-h-[48px] transition-colors ${
            pathname === "/radar"
              ? "text-cyan-400 font-bold bg-[var(--surface-active)]"
              : "text-[var(--text-muted)] hover:text-white"
          }`}
        >
          <Compass className="w-5 h-5 mb-0.5" />
          <span>Radar</span>
        </Link>

        <Link
          href={symbolParam ? `/${symbolParam}` : "/"}
          className={`flex flex-col items-center justify-center p-1.5 rounded-xl font-mono text-[10px] min-h-[48px] transition-colors ${
            pathname === "/" || pathname === ""
              ? "text-cyan-400 font-bold bg-[var(--surface-active)]"
              : "text-[var(--text-muted)] hover:text-white"
          }`}
        >
          <Activity className="w-5 h-5 mb-0.5" />
          <span>Analysis</span>
        </Link>

        {/* Plan button: scroll to execution section on Analysis page; navigate to /setups on other pages */}
        {pathname === "/" || pathname === "" ? (
          <button
            type="button"
            onClick={() => {
              const el = document.getElementById("optimal-entry-exit-card");
              if (el) el.scrollIntoView({ behavior: "smooth" });
            }}
            className="flex flex-col items-center justify-center p-1.5 rounded-xl font-mono text-[10px] min-h-[48px] text-[var(--text-muted)] hover:text-white transition-colors cursor-pointer"
          >
            <ShieldCheck className="w-5 h-5 mb-0.5" />
            <span>Plan</span>
          </button>
        ) : (
          <Link
            href={cleanSymbol ? `/setups?ticker=${encodeURIComponent(cleanSymbol)}` : "/setups"}
            className={`flex flex-col items-center justify-center p-1.5 rounded-xl font-mono text-[10px] min-h-[48px] transition-colors ${
              pathname === "/setups"
                ? "text-cyan-400 font-bold bg-[var(--surface-active)]"
                : "text-[var(--text-muted)] hover:text-white"
            }`}
          >
            <ShieldCheck className="w-5 h-5 mb-0.5" />
            <span>Plan</span>
          </Link>
        )}

        <Link
          href="/portfolio"
          className={`flex flex-col items-center justify-center p-1.5 rounded-xl font-mono text-[10px] min-h-[48px] transition-colors ${
            pathname === "/portfolio"
              ? "text-cyan-400 font-bold bg-[var(--surface-active)]"
              : "text-[var(--text-muted)] hover:text-white"
          }`}
        >
          <Briefcase className="w-5 h-5 mb-0.5" />
          <span>Portfolio</span>
        </Link>

        <button
          type="button"
          onClick={handleOpenSearch}
          className="flex flex-col items-center justify-center p-1.5 rounded-xl font-mono text-[10px] min-h-[48px] text-[var(--text-muted)] hover:text-white transition-colors cursor-pointer"
        >
          <Search className="w-5 h-5 mb-0.5" />
          <span>Search</span>
        </button>
      </div>
    </nav>
  );
}
