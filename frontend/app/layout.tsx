import "./globals.css";
import type { Metadata } from "next";
import { Suspense } from "react";
import ServiceWorkerRegister from "../components/ServiceWorkerRegister";
import OfflineStatusBanner from "../components/OfflineStatusBanner";
import MatomoTracker from "../components/MatomoTracker";
import OnboardingModal from "../components/OnboardingModal";

export const metadata: Metadata = {
  metadataBase: new URL("https://finance-xp8.pages.dev"),
  title: {
    default: "Finance Terminal | Quantitative Intelligence, STOCK Act & Risk Platform",
    template: "%s | Finance Terminal",
  },
  description:
    "Institutional-grade quantitative terminal featuring real-time Congressional STOCK Act disclosures, Mark Minervini VCP algorithmic entry points, and Cornish-Fisher downside risk modeling.",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "FinanceHQ",
  },
  icons: {
    icon: "/icons/favicon-32x32.png",
    apple: "/icons/apple-touch-icon.png",
  },
  openGraph: {
    title: "Finance Terminal | Quantitative Intelligence & Congressional STOCK Act Platform",
    description:
      "Master institutional trading with real-time STOCK Act disclosures, Minervini VCP setups, 20 EMA pullbacks, and Cornish-Fisher VaR risk modeling.",
    url: "https://finance-xp8.pages.dev",
    siteName: "Finance Terminal",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Finance Terminal Institutional Analytics & Congressional STOCK Act Scanner",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Finance Terminal | Quantitative Intelligence & Congressional STOCK Act Scanner",
    description:
      "Institutional market analytics, Nancy Pelosi STOCK Act disclosures, and algorithmic risk ladders.",
    images: ["/og-image.png"],
    creator: "@FinanceTerminal",
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
    "name": "Finance Terminal",
    "url": "https://finance-xp8.pages.dev/",
    "potentialAction": {
      "@type": "SearchAction",
      "target": "https://finance-xp8.pages.dev/?symbol={search_term_string}",
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
        <link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd) }}
        />
      </head>
      <body className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] antialiased transition-colors duration-200">
        <OfflineStatusBanner />
        <ServiceWorkerRegister />
        <Suspense fallback={null}>
          <MatomoTracker />
        </Suspense>
        {children}
        <OnboardingModal />
      </body>
    </html>
  );
}