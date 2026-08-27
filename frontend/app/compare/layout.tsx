import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Head-to-Head Asset & Pipeline Comparison Matrix | Finance Terminal",
  description: "Compare global stocks, ETFs, and cryptocurrencies side-by-side across fundamental valuations, volatility, Sharpe ratios, and beta sensitivity.",
  openGraph: {
    title: "Asset & Pipeline Comparison Matrix | Finance Terminal",
    description: "Multi-asset quantitative comparison matrix evaluating valuation, beta, and Sharpe ratios.",
    url: "https://finance-xp8.pages.dev/compare/",
    siteName: "Finance Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://finance-xp8.pages.dev/compare/",
  },
};

export default function CompareLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "Finance Terminal Comparison Matrix",
      "url": "https://finance-xp8.pages.dev/compare/",
      "applicationCategory": "FinanceApplication",
      "operatingSystem": "All",
      "description": "Multi-asset head-to-head comparison tool evaluating fundamental ratios, technical momentum, and volatility exposure.",
      "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
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
          "name": "Asset Comparison Matrix",
          "item": "https://finance-xp8.pages.dev/compare/"
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