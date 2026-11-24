"use client";
import { useEffect, useRef, useState } from "react";
import { discoverAssets, type DiscoveredAsset } from "../lib/assetDiscovery";

export function useAssetDiscovery(query: string, enabled: boolean) {
  const [results, setResults] = useState<DiscoveredAsset[]>([]);
  const [status, setStatus] = useState<"IDLE" | "LOADING" | "AVAILABLE" | "UNAVAILABLE">("IDLE");
  const cache = useRef(new Map<string, { at: number; results: DiscoveredAsset[] }>());
  useEffect(() => {
    let active = true;
    const key = query.trim().toUpperCase();
    setResults([]);
    if (!enabled || key.length < 2 || key.length > 80) { setStatus("IDLE"); return; }
    const cached = cache.current.get(key);
    if (cached && Date.now() - cached.at < 60_000) { setResults(cached.results); setStatus("AVAILABLE"); return; }
    setStatus("LOADING");
    const controller = new AbortController();
    const timeout = setTimeout(() => {
      discoverAssets(query, controller.signal).then((matches) => {
        if (!active) return;
        if (cache.current.size >= 100) cache.current.delete(cache.current.keys().next().value!);
        cache.current.set(key, { at: Date.now(), results: matches });
        setResults(matches); setStatus("AVAILABLE");
      }).catch(() => { if (active) setStatus("UNAVAILABLE"); });
    }, 300);
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, [query, enabled]);
  return { results, status };
}
