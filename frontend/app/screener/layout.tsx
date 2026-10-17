import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Opportunity Radar & Factor Screener | ARX Terminal",
  description: "Screen US equities across Mark Minervini VCP setups, value/GARP factors, volume dry-up contraction, and risk-reward asymmetry.",
  robots: {
    index: false,
    follow: true,
  },
  alternates: {
    canonical: "https://www.arxterminal.com/radar/",
  },
};

export default function ScreenerLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "ARX Terminal Gems Screener",
      "url": "https://www.arxterminal.com/screener/",
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
      "url": "https://www.arxterminal.com/screener/",
      "license": "https://creativecommons.org/licenses/by/4.0/",
      "creator": {
        "@type": "Organization",
        "name": "ARX Terminal"
      }
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "ARX Terminal",
          "item": "https://www.arxterminal.com/"
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Quantitative Stock Screener",
          "item": "https://www.arxterminal.com/screener/"
        }
      ]
    }
  ];

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }} />
      {children}
    </>
  );
}