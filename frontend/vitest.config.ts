import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    environmentMatchGlobs: [
      ["components/**", "jsdom"],
      ["hooks/**", "jsdom"],
    ],
    setupFiles: ["./vitest.setup.ts"],
    include: [
      "components/**/__tests__/**/*.test.{ts,tsx}",
      "hooks/**/__tests__/**/*.test.{ts,tsx}",
      "state/**/__tests__/**/*.test.{ts,tsx}",
    ],
  },
});
