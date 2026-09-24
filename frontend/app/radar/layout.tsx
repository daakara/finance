import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Market Opportunity Radar & Factor Screener",
  description: "Screen US equities across Mark Minervini VCP setups, value/GARP factors, volume dry-up contraction, and risk-reward asymmetry.",
  openGraph: {
    title: "Market Opportunity Radar & Factor Screener | ARX Terminal",
    description: "Algorithmic screener for VCP patterns, volume dry-up stages, and high-confluence setups.",
    url: "https://www.arxterminal.com/radar/",
    siteName: "ARX Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://www.arxterminal.com/radar/",
  },
};

export default function RadarLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "ARX Terminal Opportunity Radar",
      "url": "https://www.arxterminal.com/radar/",
      "applicationCategory": "FinanceApplication",
      "operatingSystem": "All",
      "description": "Algorithmic equity radar filtering for Volatility Contraction Patterns (VCP), volume dry-ups, and asymmetric factor setups.",
      "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" },
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "ARX Terminal",
          "item": "https://www.arxterminal.com/",
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Opportunity Radar",
          "item": "https://www.arxterminal.com/radar/",
        },
      ],
    },
  ];

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }}
      />
      {children}
    </>
  );
}
