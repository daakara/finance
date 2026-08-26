"use client";

import { useEffect, useRef } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import Script from "next/script";

export default function MatomoTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const previousUrlRef = useRef<string>("");

  const matomoUrl = process.env.NEXT_PUBLIC_MATOMO_URL || "https://analytics.example.com";
  const matomoSiteId = process.env.NEXT_PUBLIC_MATOMO_SITE_ID || "1";

  // SPA Route Change & User Journey Path Tracking
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
    }

    previousUrlRef.current = fullUrl;
  }, [pathname, searchParams]);

  const sanitizedMatomoUrl = matomoUrl.replace(/\/+$/, "") + "/";

  return (
    <>
      <Script id="matomo-base-init" strategy="afterInteractive">
        {`
          var _paq = window._paq = window._paq || [];
          _paq.push(['trackPageView']);
          _paq.push(['enableLinkTracking']);
          _paq.push(['enableHeartBeatTimer', 15]);
          (function() {
            var u="${sanitizedMatomoUrl}";
            _paq.push(['setTrackerUrl', u+'matomo.php']);
            _paq.push(['setSiteId', '${matomoSiteId}']);
            var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
            g.async=true; g.src=u+'matomo.js'; s.parentNode.insertBefore(g,s);
          })();
        `}
      </Script>
    </>
  );
}