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
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "Finance Terminal Gems Screener",
      "url": "https://finance-xp8.pages.dev/screener/",
      "applicationCategory": "FinanceApplication",
      "operatingSystem": "All",
      "description": "Algorithmic small and mid-cap equity screener filtering for volatility contraction patterns, 4 ATR execution states, and institutional accumulation.",
      "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
    },
    {
      "@context": "https://schema.org",
      "@type": "Dataset",
      "name": "Quantitative High-Alpha Equity Screener Dataset",
      "description": "Daily updated universe of asymmetric small and mid-cap equities with quantitative factor models and ATR invalidation levels.",
      "url": "https://finance-xp8.pages.dev/screener/",
      "license": "https://creativecommons.org/licenses/by/4.0/",
      "creator": {
        "@type": "Organization",
        "name": "Finance Terminal"
      }
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "Finance Terminal",
          "item": "https://finance-xp8.pages.dev/"
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Quantitative Stock Screener",
          "item": "https://finance-xp8.pages.dev/screener/"
        }
      ]
    }
  ];

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      {children}
    </>
  );
}