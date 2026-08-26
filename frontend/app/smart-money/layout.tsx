import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Congressional STOCK Act & Smart Money Insider Scanner | Finance Terminal",
  description: "Track US House & Senate legislative stock disclosures (STOCK Act PL 112-105), SEC Form 4 insider transactions, and unusual options flow sweeps in real time.",
  openGraph: {
    title: "Congressional STOCK Act & Smart Money Scanner | Finance Terminal",
    description: "Track US Congress stock trades and institutional smart money flow in real time.",
    url: "https://finance-xp8.pages.dev/smart-money/",
    siteName: "Finance Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://finance-xp8.pages.dev/smart-money/",
  },
};

export default function SmartMoneyLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "Finance Terminal Smart Money Scanner",
    "url": "https://finance-xp8.pages.dev/smart-money/",
    "applicationCategory": "FinanceApplication",
    "operatingSystem": "All",
    "description": "Real-time tracker for US Congressional STOCK Act disclosures and unusual options market flow.",
    "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      {children}
    </>
  );
}