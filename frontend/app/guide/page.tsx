import type { Metadata } from "next";
import Navbar from "../../components/Navbar";
import GuideContent from "../../components/GuideContent";

export const metadata: Metadata = {
  title: "Quantitative Terminal Field Manual & Algorithmic Handbook",
  description:
    "Institutional manual detailing Decision Intelligence Suites (Pre-Flight Gate, Smart Money Radar, Edge Scorecard, Macro Stress Testing), Congressional STOCK Act tracking, Legislative Alignment Index (0-100), Mark Minervini VCP entry points, Cornish-Fisher Modified VaR, and Self-Healing Forecast Audits.",
  openGraph: {
    title: "ARX Terminal: Quantitative Field Manual & Algorithmic Handbook",
    description: "Master institutional quantitative trading, 5-point Pre-Flight execution gates, Smart Money order flow forensics, macro stress testing, staleness decay penalties, and tail risk management.",
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
      "headline": "Quantitative Terminal Field Manual & Algorithmic Handbook",
      "description": "Comprehensive guide to Decision Intelligence Suites, Congressional STOCK Act tracking, Legislative Alignment Index, Mark Minervini VCP algorithmic entry points, Cornish-Fisher Modified VaR risk modeling, and FRED macroeconomic regimes.",
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
      "dateModified": "2026-08-29"
    },
    {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "How many assets does the terminal track, and how does the 4-tier pipeline work?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The terminal operates a 4-tier pipeline: Tier 1 is the Master Catalog (38 high-conviction fundamental baselines), Tier 2 is the Multi-Factor Screener (35 dynamic assets), Tier 3 is Pre-Rendered Static Pages (94 pre-compiled edge routes for <10ms loading), and Tier 4 is Universal Omnisearch (unlimited real-time on-demand queries for any US equity or crypto asset)."
          }
        },
        {
          "@type": "Question",
          "name": "What is the 5-point Pre-Flight Clearance Gate?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The Pre-Flight Checklist is a strict quantitative risk gate evaluating: 1) Trend & Moving Averages / Key Support (20 EMA & 50 EMA alignment), 2) Minervini Volatility Contraction Pattern (VCP base structure), 3) Institutional Smart Money & Short Squeeze Ratio, 4) Binary Catalyst Proximity (48h earnings/FDA hazard buffer), and 5) Macro Volatility Guard (VIX < 26.0)."
          }
        },
        {
          "@type": "Question",
          "name": "How does the Smart Money Divergence Radar identify stealth accumulation vs distribution traps?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The radar compares price trend direction against institutional order flow (dark pool ATS block prints, C-suite Form 4 buys, and STOCK Act disclosures). Price consolidation during high institutional accumulation signals a high-conviction breakout setup, whereas price spikes during net insider distribution flag dangerous distribution traps."
          }
        },
        {
          "@type": "Question",
          "name": "What is the Macro Stress Test Simulator?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The Macro Stress Test simulates hypothetical systemic shocks (e.g. QQQ Tech Selloff -5%, Treasury Yield Surge +50bps, VIX Spike to 35) across your portfolio using covariance Beta weighting, projecting total portfolio drawdown and recommending exact defensive cash reserves."
          }
        },
        {
          "@type": "Question",
          "name": "What is the Congressional STOCK Act filing deadline?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Under Public Law 112-105 (Stop Trading on Congressional Knowledge Act of 2012), members of the US Congress and Senate are legally required to disclose securities transactions within 45 days of execution or 30 days of notification."
          }
        },
        {
          "@type": "Question",
          "name": "How is the Legislative Alignment Index calculated?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The Legislative Alignment Index (0–100) quantifies the correlation between a politician's trade and their legislative influence by evaluating committee jurisdiction overlap (+16 to +32 pts), transaction sizing tiers ($50k to $1M+), and audited multi-year politician win rates."
          }
        },
        {
          "@type": "Question",
          "name": "What are the 4 Mathematical ATR Execution States?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "The 4 ATR Execution States are: 1) IN_BUY_ZONE (Optimal Accumulation), 2) APPROACHING_TARGET (Momentum Expansion), 3) WAITING_PULLBACK (Overextended / Chasing Risk), and 4) STOPPED_OUT (Invalidation Exit)."
          }
        },
        {
          "@type": "Question",
          "name": "How does Cornish-Fisher Modified Value-at-Risk (M-VaR) differ from standard VaR?",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "Standard Gaussian VaR assumes a normal distribution, underestimating fat-tail crash risks. Cornish-Fisher M-VaR uses a polynomial expansion adjusting for sample skewness and excess kurtosis to provide accurate downside risk boundaries."
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
          "name": "Field Manual & Algorithmic Handbook",
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

      <GuideContent />
    </div>
  );
}