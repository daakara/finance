import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tactical Trade Setups & Execution Corridors",
  description: "Actionable swing trade execution blueprints with pre-calculated entry corridors, 4 ATR invalidation stops, behavioral sizing, and target ladders.",
  openGraph: {
    title: "Tactical Trade Setups & Execution Corridors | ARX Terminal",
    description: "Institutional trade decision tickets with mathematical invalidation boundaries and execution state machines.",
    url: "https://www.arxterminal.com/setups/",
    siteName: "ARX Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://www.arxterminal.com/setups/",
  },
};

export default function SetupsLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "ARX Terminal Tactical Setups",
      "url": "https://www.arxterminal.com/setups/",
      "applicationCategory": "FinanceApplication",
      "operatingSystem": "All",
      "description": "Decision intelligence workstation generating pre-trade execution corridors, R-multiples, and position sizing governors.",
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
          "name": "Tactical Setups",
          "item": "https://www.arxterminal.com/setups/",
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
