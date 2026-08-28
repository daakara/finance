import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Congressional STOCK Act & Smart Money Insider Scanner | Finance Terminal",
  description: "Track US House & Senate legislative stock disclosures (STOCK Act PL 112-105), SEC Form 4 insider transactions, and unusual options flow sweeps in real time.",
  openGraph: {
    title: "Congressional STOCK Act & Smart Money Scanner | Finance Terminal",
    description: "Track US Congress stock trades and institutional smart money flow in real time.",
    url: "https://www.arxterminal.com/smart-money/",
    siteName: "Finance Terminal",
    type: "website",
  },
  alternates: {
    canonical: "https://www.arxterminal.com/smart-money/",
  },
};

export default function SmartMoneyLayout({ children }: { children: React.ReactNode }) {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "Finance Terminal Smart Money Scanner",
      "url": "https://www.arxterminal.com/smart-money/",
      "applicationCategory": "FinanceApplication",
      "operatingSystem": "All",
      "description": "Real-time tracker for US Congressional STOCK Act disclosures, Legislative Alignment Index (0-100), and unusual options market flow.",
      "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
    },
    {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "What is a Congressional STOCK Act disclosure?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Under the Stop Trading on Congressional Knowledge Act of 2012 (Public Law 112-105), members of the US Congress and Senate are legally required to publicly disclose all stock, bond, and options transactions over $1,000 within 45 days."
          }
        },
        {
          "@type": "Question",
          "name": "How does the Legislative Alignment Score work?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The Legislative Alignment Score (0-100) measures how closely a politician's trade correlates with their committee oversight (e.g. Armed Services purchasing Defense tech) combined with trade sizing brackets and historical trading alpha."
          }
        },
        {
          "@type": "Question",
          "name": "What is the STOCK Act Late Filer penalty?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Disclosures filed beyond the 45-day statutory deadline are penalized with a -32 point decay in signal strength to warn investors of mean-reversion risk on stale news."
          }
        }
      ]
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "Finance Terminal",
          "item": "https://www.arxterminal.com/"
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Smart Money & Congressional Trades",
          "item": "https://www.arxterminal.com/smart-money/"
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