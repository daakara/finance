/** @type {import("tailwindcss").Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
    "./frontend/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./frontend/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./frontend/lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Legacy colors preserved for backward compatibility
        background: "#0d1117",
        card: "#161b22",
        border: "#30363d",
        bullish: "#00c851",
        bearish: "#ff4444",
        accent: "#38bdf8",

        // vNext Institutional Surface Tokens
        "bg-app": "var(--bg-app, #090d16)",
        "bg-surface": "var(--bg-surface, #0f1422)",
        "bg-surface-raised": "var(--bg-surface-raised, #161f33)",
        "bg-surface-elevated": "var(--bg-surface-elevated, #1e2a45)",

        // vNext Semantic Decision Tokens (Anti-Cyan Calibration)
        "accent-positive": "var(--accent-positive, #10b981)", // Emerald 500: Favorable / In Buy Zone
        "accent-warning": "var(--accent-warning, #f59e0b)",   // Amber 500: Caution / Waiting Pullback
        "accent-risk": "var(--accent-risk, #f43f5e)",         // Rose 500: Invalidation / Stop Loss / Danger
        "accent-info": "var(--accent-info, #06b6d4)",         // Cyan 500: STRICTLY System Info / Selection
        "accent-neutral": "var(--accent-neutral, #64748b)",   // Slate 500: Inactive / Unchanged

        // vNext Typography Color Tokens
        "text-primary": "var(--text-primary, #f8fafc)",       // Pure White (Slate 50) for critical prices & scores
        "text-secondary": "var(--text-secondary, #94a3b8)",   // Slate 400 for sub-labels & descriptions
        "text-faint": "var(--text-faint, #64748b)",           // Slate 500 for disclaimers & metadata

        // vNext Border Tokens
        "border-subtle": "var(--border-subtle, rgba(51, 65, 85, 0.8))", // slate-700/80
        "border-strong": "var(--border-strong, #334155)",                // slate-700
      },
      fontSize: {
        "display-1": ["2.25rem", { lineHeight: "2.5rem", letterSpacing: "-0.02em", fontWeight: "700" }], // 36px
        "header-1": ["1.5rem", { lineHeight: "2rem", letterSpacing: "-0.01em", fontWeight: "600" }],     // 24px
        "header-2": ["1.125rem", { lineHeight: "1.75rem", letterSpacing: "-0.005em", fontWeight: "600" }], // 18px
        "body-ui": ["0.875rem", { lineHeight: "1.25rem", fontWeight: "400" }],                             // 14px
        "data-mono-lg": ["1.5rem", { lineHeight: "2rem", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace", fontWeight: "600" }], // 24px mono
        "data-mono-sm": ["0.875rem", { lineHeight: "1.25rem", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace", fontWeight: "500" }], // 14px mono
        "caption-mono": ["0.75rem", { lineHeight: "1rem", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace", fontWeight: "400" }], // 12px mono
      },
      spacing: {
        "px-4": "4px",
        "px-8": "8px",
        "px-12": "12px",
        "px-16": "16px",
        "px-24": "24px",
        "px-32": "32px",
        "px-48": "48px",
        "px-64": "64px",
      },
    },
  },
  plugins: [],
};

