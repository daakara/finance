"use client";

import { useState, useEffect } from "react";

export default function OfflineStatusBanner() {
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsOffline(!navigator.onLine);

      const handleOnline = () => setIsOffline(false);
      const handleOffline = () => setIsOffline(true);

      window.addEventListener("online", handleOnline);
      window.addEventListener("offline", handleOffline);

      return () => {
        window.removeEventListener("online", handleOnline);
        window.removeEventListener("offline", handleOffline);
      };
    }
  }, []);

  if (!isOffline) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="bg-amber-950/90 border-b border-amber-500/60 text-amber-200 px-4 py-1.5 text-xs font-mono flex items-center justify-between shadow-lg sticky top-0 z-[60] backdrop-blur-sm"
    >
      <div className="flex items-center space-x-2">
        <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
        <span className="font-bold">⚡ Offline PWA Mode:</span>
        <span className="text-amber-300/90 hidden sm:inline">
          Operating from local service worker cache. Real-time market streaming paused.
        </span>
        <span className="text-amber-300/90 sm:hidden">
          Using cached PWA storage.
        </span>
      </div>
      <span className="text-[10px] bg-amber-900/80 px-2 py-0.5 rounded border border-amber-700 font-bold uppercase tracking-wider">
        PWA Cached
      </span>
    </div>
  );
}