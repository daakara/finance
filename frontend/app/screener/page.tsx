"use client";
import { useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";

function ScreenerRedirect() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const q = searchParams.get("q") || searchParams.get("ticker") || searchParams.get("symbol");
    if (q && q.trim()) {
      router.replace(`/radar?q=${encodeURIComponent(q.trim().toUpperCase())}`);
    } else {
      router.replace("/radar");
    }
  }, [router, searchParams]);

  return (
    <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center text-slate-400 font-mono text-sm">
      <p>Redirecting to Radar…</p>
    </div>
  );
}

export default function RedirectPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#0a0e17]" />}>
      <ScreenerRedirect />
    </Suspense>
  );
}
