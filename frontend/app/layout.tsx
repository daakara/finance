import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata = {
  title: "Financial Market Analysis Platform",
  description: "Enterprise multi-asset analytics & forecasting platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0d1117] text-[#c9d1d9]">
        <Navbar />
        <main className="max-w-7xl mx-auto p-6">{children}</main>
      </body>
    </html>
  );
}

