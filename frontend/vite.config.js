// Vite’s config helper provides typed/validated config shape.
import { defineConfig } from "vite";
// Official React plugin: JSX transform, Fast Refresh, etc.
import react from "@vitejs/plugin-react";

// Export the Vite configuration object consumed by `vite dev` / `vite build`.
export default defineConfig({
  // Register build-time plugins (React support).
  plugins: [react()],
  // Development server tuning.
  server: {
    // Bind dev server to port 5173 (Vite default) for predictable README URLs.
    port: 5173,
    // Proxy selected API paths to the FastAPI backend to avoid CORS during local dev.
    proxy: {
      // Forward PDF upload endpoint to local Python server.
      "/upload": "http://127.0.0.1:8000",
      // Forward Q&A endpoint likewise.
      "/query": "http://127.0.0.1:8000",
      // Optional health probe passthrough for debugging.
      "/health": "http://127.0.0.1:8000",
    },
  },
});
