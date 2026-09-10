"use client";
import { useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";

function ResearchRedirect() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const symbol = searchParams.get("symbol") || searchParams.get("ticker");
    if (symbol && symbol.trim()) {
      router.replace(`/?symbol=${encodeURIComponent(symbol.trim().toUpperCase())}`);
    } else {
      router.replace("/");
    }
  }, [router, searchParams]);

  return (
    <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center text-slate-400 font-mono text-sm">
      <p>Redirecting to Terminal…</p>
    </div>
  );
}

export default function RedirectPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#0a0e17]" />}>
      <ResearchRedirect />
    </Suspense>
  );
}
