import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Stock Intelligence Directory",
  description:
    "Browse quantitative stock analytics, Minervini VCP execution blueprints, and Congressional trading disclosures across tracked equities.",
  alternates: {
    canonical: "https://www.arxterminal.com/stock/",
  },
};

export default function StockLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
