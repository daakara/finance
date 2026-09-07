"use client";

import { useEffect } from "react";
import { useUIStore } from "../../state/ui-store";

export default function WatchlistDrawerHotkeys() {
  const watchlistOpen = useUIStore((state) => state.watchlistOpen);
  const toggleWatchlist = useUIStore((state) => state.toggleWatchlist);
  const closeWatchlist = useUIStore((state) => state.closeWatchlist);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement;
      const isInputFocused =
        activeEl?.tagName === "INPUT" ||
        activeEl?.tagName === "TEXTAREA" ||
        (activeEl as HTMLElement)?.isContentEditable;

      // Handle Esc key when drawer is open
      if (e.key === "Escape" && watchlistOpen) {
        e.preventDefault();
        closeWatchlist();
        return;
      }

      // Handle '[' toggle or 'Ctrl+B' / 'Meta+B'
      const isBracketKey = e.key === "[" && !isInputFocused && !e.ctrlKey && !e.metaKey && !e.altKey;
      const isCtrlB = (e.ctrlKey || e.metaKey) && (e.key === "b" || e.key === "B");

      if (isBracketKey || isCtrlB) {
        e.preventDefault();
        toggleWatchlist("keyboard");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [watchlistOpen, toggleWatchlist, closeWatchlist]);

  return null;
}
