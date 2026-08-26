import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "My Portfolio & Risk Allocations | Zero-Login Private Asset Tracker",
  description:
    "Zero-login private portfolio tracker calculating real-time profit & loss (P&L), position sizing, downside Value-at-Risk (VaR), and target risk ladders across global equities and crypto.",
  openGraph: {
    title: "My Portfolio & Risk Allocations | Finance Terminal",
    description: "Track your equity holdings, cost basis, unrealized P&L, and downside risk with zero-login private client storage.",
    url: "https://finance-xp8.pages.dev/portfolio/",
    siteName: "Finance Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://finance-xp8.pages.dev/portfolio/",
  },
};

export default function PortfolioLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "Finance Terminal Portfolio Tracker",
    "url": "https://finance-xp8.pages.dev/portfolio/",
    "applicationCategory": "FinanceApplication",
    "operatingSystem": "All",
    "description": "Zero-login private portfolio and risk allocation engine calculating real-time profit and loss and volatility exposure.",
    "offers": {
      "@type": "Offer",
      "price": "0",
      "priceCurrency": "USD"
    }
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {children}
    </>
  );
}