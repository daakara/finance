"use client";

import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="bg-[#161b22] border-b border-[#30363d] px-6 py-4 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <span className="text-2xl">??</span>
        <span className="text-xl font-bold text-white tracking-wide">
          Financial Market Analytics
        </span>
      </div>

      <div className="flex space-x-6">
        <Link href="/" className="text-gray-300 hover:text-white font-medium transition-colors">
          Dashboard
        </Link>
        <Link href="/screener" className="text-gray-300 hover:text-white font-medium transition-colors">
          ?? Hidden Gems Screener
        </Link>
      </div>
    </nav>
  );
}

