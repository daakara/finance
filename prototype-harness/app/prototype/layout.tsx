"use client";

import React, { useEffect } from "react";
import { installPrototypeNetworkGuard } from "@/lib/prototype/networkGuard";

export default function PrototypeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  useEffect(() => {
    // Install fail-closed network isolation guard on prototype route mount
    const cleanup = installPrototypeNetworkGuard();
    return () => {
      cleanup();
    };
  }, []);

  return (
    <div className="prototype-sandbox min-h-screen max-w-[100vw] overflow-x-hidden bg-slate-950 text-slate-100 antialiased">
      {children}
    </div>
  );
}
