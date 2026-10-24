import "./globals.css";
import React from "react";

export const metadata = {
  title: "ARX Terminal — Phase 2 Prototype Sandbox",
  description: "Nonproduction isolated evaluation harness",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
