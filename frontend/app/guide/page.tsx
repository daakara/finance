import type { Metadata } from "next";
import Navbar from "../../components/Navbar";
import GuideContent from "../../components/GuideContent";
import AuthorEeatBadge from "../../components/AuthorEeatBadge";

export const metadata: Metadata = {
  title: "Field Manual & Quantitative Methodology | ARX Terminal",
  description:
    "Authoritative reference guide to ARX Terminal's execution corridors, capital allocation risk gates, Volatility Contraction Pattern (VCP) detection, Cornish-Fisher VaR tail-risk models, and statutory insider forensics.",
  openGraph: {
    title: "ARX Terminal Field Manual & Quantitative Methodology",
    description: "Learn the quantitative models, capital preservation rules, execution corridors, and risk boundaries powering ARX Terminal.",
    url: "https://www.arxterminal.com/guide/",
    siteName: "ARX Terminal",
    type: "article",
  },
  alternates: {
    canonical: "https://www.arxterminal.com/guide/",
  },
};

export default function GuidePage() {
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "TechArticle",
      "headline": "ARX Terminal Field Manual & Quantitative Methodology",
      "description": "Comprehensive reference guide to ARX Terminal's execution corridors, capital allocation risk gates, Volatility Contraction Pattern (VCP) detection, Cornish-Fisher VaR tail-risk models, and statutory insider forensics.",
      "author": {
        "@type": "Organization",
        "name": "ARX Terminal Quantitative Intelligence"
      },
      "publisher": {
        "@type": "Organization",
        "name": "ARX Terminal",
        "logo": {
          "@type": "ImageObject",
          "url": "https://www.arxterminal.com/icons/icon-512x512.png"
        }
      },
      "datePublished": "2026-08-26",
      "dateModified": "2026-11-14"
    },
    {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "How does the ARX 1% risk allocation rule calculate share sizing?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The position sizer divides 1% of total portfolio equity by the dollar distance between the entry price and the technical stop-loss floor. If a $50,000 account risks 1% ($500) buying a stock at $213 with a $201 stop ($12 risk), exactly 41 shares are purchased."
          }
        },
        {
          "@type": "Question",
          "name": "What are the 5 gates in the Pre-Flight Clearance Checklist?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The Pre-Flight Gate evaluates: 1) Trend & Moving Average alignment above the 20 EMA, 2) Mark Minervini Volatility Contraction Pattern (VCP) base geometry, 3) Institutional Smart Money accumulation, 4) Binary Catalyst Buffer (zero earnings/FDA hazard within 48h), and 5) Macro Volatility Floor (CBOE VIX strictly below 26.0)."
          }
        },
        {
          "@type": "Question",
          "name": "What are the 4 Mathematical ATR Execution States?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The 4 ATR Execution States are: 1) IN_BUY_ZONE (Optimal Accumulation within support), 2) APPROACHING_TARGET (Advancing momentum toward Target 1), 3) WAITING_PULLBACK (Overextended past 2% ceiling; elevated mean-reversion risk), and 4) STOPPED_OUT (Closed below invalidation floor; strict capital defense exit)."
          }
        },
        {
          "@type": "Question",
          "name": "How does Cornish-Fisher Modified Value-at-Risk (M-VaR) differ from standard VaR?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Standard Gaussian VaR assumes a normal distribution, severely underestimating fat-tail crash risks. Cornish-Fisher M-VaR applies a polynomial expansion adjusting for empirical skewness and excess kurtosis to capture true downside tail risk."
          }
        },
        {
          "@type": "Question",
          "name": "How does the Congressional STOCK Act staleness decay schedule operate?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Lawmakers have up to 45 days to disclose securities trades under Public Law 112-105. Filings within 15 days incur 0 penalty points, 16-30 days incur a 5-point deduction, 31-45 days incur a 16-point deduction, and filings past the 45-day statutory window incur a 32-point deduction."
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
          "name": "ARX Terminal",
          "item": "https://www.arxterminal.com/"
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Field Manual & Quantitative Methodology",
          "item": "https://www.arxterminal.com/guide/"
        }
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-[var(--bg-app)] text-[var(--text-main)] font-sans selection:bg-cyan-500 selection:text-black transition-colors duration-200">
      {/* Schema.org Structured Data */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }}
      />

      <Navbar />

      <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-6 -mb-4">
        <AuthorEeatBadge
          topic="Institutional Decision Intelligence & Algorithmic Execution"
          lastUpdated="November 2026"
        />
      </div>

      <GuideContent />
    </div>
  );
}
