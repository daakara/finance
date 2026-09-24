import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Trading Performance & Equity Curve",
  description: "Personal account metrics, Sharpe ratio, win rate, and drawdown analysis.",
  robots: {
    index: false,
    follow: false,
  },
};

export default function PerformanceLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
