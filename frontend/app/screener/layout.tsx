import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "High-Alpha Hidden Gems Screener | Small & Mid-Cap Quant Stock Screener",
  description: "Screen high-alpha small and mid-cap equities with asymmetric risk-reward profiles, Minervini VCP setups, institutional accumulation, and catalyst tracking.",
  openGraph: {
    title: "High-Alpha Hidden Gems Screener | Finance Terminal",
    description: "Screen high-alpha small and mid-cap equities with asymmetric risk-reward profiles and VCP setups.",
    url: "https://finance-xp8.pages.dev/screener/",
    siteName: "Finance Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://finance-xp8.pages.dev/screener/",
  },
};

export default function ScreenerLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "Finance Terminal Gems Screener",
    "url": "https://finance-xp8.pages.dev/screener/",
    "applicationCategory": "FinanceApplication",
    "operatingSystem": "All",
    "description": "Algorithmic small and mid-cap equity screener filtering for volatility contraction patterns and institutional accumulation.",
    "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      {children}
    </>
  );
}