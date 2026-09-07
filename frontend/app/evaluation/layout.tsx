import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ArxTerminal — Private Prospective Evaluation Dashboard",
  description: "Private, read-only observation dashboard for the frozen ArxTerminal v2.4.0 prospective prediction evaluation.",
  robots: {
    index: false,
    follow: false,
    nocache: true,
    googleBot: {
      index: false,
      follow: false,
      noimageindex: true,
    },
  },
};

export default function EvaluationLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-cyan-500/20 selection:text-cyan-300">{children}</div>;
}
