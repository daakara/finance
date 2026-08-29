"use client";

import { useEffect, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";

declare global {
  interface Window {
    _paq?: any[];
    _mtm?: any[];
  }
}

export default function MatomoTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const previousUrlRef = useRef<string>("");

  // SPA Route Change & User Journey Path Tracking for Matomo Tag Manager & Analytics (Cookieless CNIL Exemption)
  useEffect(() => {
    if (typeof window === "undefined") return;

    window._paq = window._paq || [];
    window._mtm = window._mtm || [];
    const fullUrl = window.location.pathname + window.location.search;

    if (previousUrlRef.current && previousUrlRef.current !== fullUrl) {
      window._paq.push(["setReferrerUrl", previousUrlRef.current]);
      window._paq.push(["setCustomUrl", fullUrl]);
      window._paq.push(["setDocumentTitle", document.title]);
      window._paq.push(["deleteCustomVariables", "page"]);
      window._paq.push(["trackPageView"]);
      window._paq.push(["enableLinkTracking"]);

      // Trigger Matomo Tag Manager event on SPA page transitions
      window._mtm.push({
        event: "mtm.PageView",
        customUrl: fullUrl,
        pageTitle: document.title,
      });
    } else if (!previousUrlRef.current) {
      // First page view on initial load
      window._paq.push(["trackPageView"]);
    }

    previousUrlRef.current = fullUrl;
  }, [pathname, searchParams]);

  return null;
}