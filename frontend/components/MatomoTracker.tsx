"use client";

import { useEffect, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";

export default function MatomoTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const previousUrlRef = useRef<string>("");

  const matomoUrl = process.env.NEXT_PUBLIC_MATOMO_URL || "https://analytics.example.com";
  const matomoSiteId = process.env.NEXT_PUBLIC_MATOMO_SITE_ID || "1";
  const sanitizedMatomoUrl = matomoUrl.replace(/\/+$/, "") + "/";

  // 1. Initial Setup on Client Mount
  useEffect(() => {
    if (typeof window === "undefined") return;

    window._paq = window._paq || [];
    window._paq.push(["setTrackerUrl", sanitizedMatomoUrl + "matomo.php"]);
    window._paq.push(["setSiteId", matomoSiteId]);
    window._paq.push(["enableLinkTracking"]);
    window._paq.push(["enableHeartBeatTimer", 15]);

    // Inject remote matomo.js loader
    if (!document.getElementById("matomo-js-script")) {
      const script = document.createElement("script");
      script.id = "matomo-js-script";
      script.type = "text/javascript";
      script.async = true;
      script.src = sanitizedMatomoUrl + "matomo.js";
      document.head.appendChild(script);
    }
  }, [sanitizedMatomoUrl, matomoSiteId]);

  // 2. SPA Route Change & User Journey Path Tracking
  useEffect(() => {
    if (typeof window === "undefined") return;

    window._paq = window._paq || [];
    const fullUrl = window.location.pathname + window.location.search;

    if (previousUrlRef.current && previousUrlRef.current !== fullUrl) {
      window._paq.push(["setReferrerUrl", previousUrlRef.current]);
      window._paq.push(["setCustomUrl", fullUrl]);
      window._paq.push(["setDocumentTitle", document.title]);
      window._paq.push(["deleteCustomVariables", "page"]);
      window._paq.push(["trackPageView"]);
      window._paq.push(["enableLinkTracking"]);
    } else if (!previousUrlRef.current) {
      // First page view on initial load
      window._paq.push(["trackPageView"]);
    }

    previousUrlRef.current = fullUrl;
  }, [pathname, searchParams]);

  return null;
}