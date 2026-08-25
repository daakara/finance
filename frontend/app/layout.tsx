import "./globals.css";

export const metadata = {
  title: "Financial Market Analysis Platform",
  description: "Enterprise multi-asset analytics & forecasting platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#070a10] text-[#c9d1d9] antialiased">
        {children}
      </body>
    </html>
  );
}

