"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function RedirectPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/radar");
  }, [router]);
  return (
    <div className="min-h-screen bg-[#0a0e17] flex items-center justify-center text-slate-400 font-mono text-sm">
      <p>Redirecting…</p>
    </div>
  );
}
