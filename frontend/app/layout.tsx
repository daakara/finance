import "./globals.css";
import { Suspense } from "react";
import ServiceWorkerRegister from "../components/ServiceWorkerRegister";
import OfflineStatusBanner from "../components/OfflineStatusBanner";
import MatomoTracker from "../components/MatomoTracker";
import OnboardingModal from "../components/OnboardingModal";

export const metadata = {
  title: "Financial Market Analysis Platform",
  description: "Enterprise multi-asset analytics, quantitative risk & forecasting platform",
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
};

export const viewport = {
  themeColor: "#06b6d4",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="theme-color" content="#06b6d4" />
        <link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />
      </head>
      <body className="min-h-screen bg-[#070a10] text-[#c9d1d9] antialiased">
        <OfflineStatusBanner />
        <OnboardingModal />
        <ServiceWorkerRegister />
        <Suspense fallback={null}>
          <MatomoTracker />
        </Suspense>
        {children}
      </body>
    </html>
  );
}