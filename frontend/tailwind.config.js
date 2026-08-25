/** @type {import("tailwindcss").Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0d1117",
        card: "#161b22",
        border: "#30363d",
        bullish: "#00c851",
        bearish: "#ff4444",
        accent: "#38bdf8",
      },
    },
  },
  plugins: [],
};

