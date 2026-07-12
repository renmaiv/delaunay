/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
    fs: { allow: [".."] },
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
