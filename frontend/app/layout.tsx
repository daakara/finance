import "./globals.css";
import type { Metadata } from "next";
import { Suspense } from "react";
import ServiceWorkerRegister from "../components/ServiceWorkerRegister";
import OfflineStatusBanner from "../components/OfflineStatusBanner";
import MatomoTracker from "../components/MatomoTracker";

export const metadata: Metadata = {
  metadataBase: new URL("https://www.arxterminal.com"),
  title: {
    default: "ARX Terminal | Quantitative Intelligence, STOCK Act & Risk Platform",
    template: "%s | ARX Terminal",
  },
  description:
    "Institutional-grade quantitative terminal featuring real-time Congressional STOCK Act disclosures, Mark Minervini VCP algorithmic entry points, and Cornish-Fisher downside risk modeling.",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "ARX Terminal",
  },
  icons: {
    icon: "/icons/favicon-32x32.png",
    apple: "/icons/apple-touch-icon.png",
  },
  openGraph: {
    title: "ARX Terminal | Quantitative Intelligence & Congressional STOCK Act Platform",
    description:
      "Master institutional trading with real-time STOCK Act disclosures, Minervini VCP setups, 20 EMA pullbacks, and Cornish-Fisher VaR risk modeling.",
    url: "https://www.arxterminal.com",
    siteName: "ARX Terminal",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "ARX Terminal Institutional Analytics & Congressional STOCK Act Scanner",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "ARX Terminal | Quantitative Intelligence & Congressional STOCK Act Scanner",
    description:
      "Institutional market analytics, Nancy Pelosi STOCK Act disclosures, and algorithmic risk ladders.",
    images: ["/og-image.png"],
    creator: "@ARXTerminal",
  },
};

export const viewport = {
  themeColor: "#06b6d4",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const websiteJsonLd = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "ARX Terminal",
    "url": "https://www.arxterminal.com/",
    "potentialAction": {
      "@type": "SearchAction",
      "target": "https://www.arxterminal.com/?symbol={search_term_string}",
      "query-input": "required name=search_term_string"
    }
  };

  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="theme-color" content="#06b6d4" />
        {/* Canonical Apex & pages.dev to WWW & HTTPS Domain Enforcement */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                if (typeof window !== 'undefined') {
                  var host = window.location.hostname;
                  if (host === 'arxterminal.com' || host.endsWith('.pages.dev')) {
                    window.location.replace('https://www.arxterminal.com' + window.location.pathname + window.location.search + window.location.hash);
                  }
                }
              })();
            `,
          }}
        />
        {/* Matomo Tag Manager & Privacy-First Analytics Engine */}
        <script
          type="text/javascript"
          dangerouslySetInnerHTML={{
            __html: `
              var _paq = window._paq = window._paq || [];
              _paq.push(['disableCookies']);
              _paq.push(['setDoNotTrack', true]);
              _paq.push(['enableLinkTracking']);
              _paq.push(['setTrackerUrl', 'https://data.fpldna.com/matomo/matomo.php']);
              _paq.push(['setSiteId', '3']);
              var _mtm = window._mtm = window._mtm || [];
              _mtm.push({'mtm.startTime': (new Date().getTime()), 'event': 'mtm.Start'});
              (function() {
                var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
                g.async=true; g.src='https://data.fpldna.com/matomo/js/container_tK4RnlSN.js'; s.parentNode.insertBefore(g,s);
              })();
            `,
          }}
        />
        <noscript>
          <p>
            <img
              referrerPolicy="no-referrer-when-downgrade"
              src="https://data.fpldna.com/matomo/matomo.php?idsite=3&rec=1"
              style={{ border: 0 }}
              alt=""
            />
          </p>
        </noscript>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd).replace(/</g, "\\u003c") }}
        />
      </head>
      <body className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] antialiased transition-colors duration-200">
        <OfflineStatusBanner />
        <ServiceWorkerRegister />
        <Suspense fallback={null}>
          <MatomoTracker />
        </Suspense>
        {children}
      </body>
    </html>
  );
}