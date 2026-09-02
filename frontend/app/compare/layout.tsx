import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Head-to-Head Asset & Pipeline Comparison Matrix | ARX Terminal",
  description: "Compare global stocks, ETFs, and cryptocurrencies side-by-side across fundamental valuations, volatility, Sharpe ratios, and beta sensitivity.",
  openGraph: {
    title: "Asset & Pipeline Comparison Matrix | ARX Terminal",
    description: "Multi-asset quantitative comparison matrix evaluating valuation, beta, and Sharpe ratios.",
    url: "https://www.arxterminal.com/compare/",
    siteName: "ARX Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://www.arxterminal.com/compare/",
  },
};

export default function CompareLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "ARX Terminal Comparison Matrix",
      "url": "https://www.arxterminal.com/compare/",
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
          "name": "ARX Terminal",
          "item": "https://www.arxterminal.com/"
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Asset Comparison Matrix",
          "item": "https://www.arxterminal.com/compare/"
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