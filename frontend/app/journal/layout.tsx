import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Trading Journal",
  description: "Personal execution logs, R-multiples, and rule compliance tracker.",
  robots: {
    index: false,
    follow: false,
  },
};

export default function JournalLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
