import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the React app runs on :5173 and proxies API calls to the
// FastAPI server on :8000. In production FastAPI serves `frontend/dist`.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/model-info": "http://127.0.0.1:8000",
      "/metrics": "http://127.0.0.1:8000",
      "/predict": "http://127.0.0.1:8000",
      "/predict-ctg": "http://127.0.0.1:8000",
      "/demo-cases": "http://127.0.0.1:8000",
      "/demo": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000"
    }
  },
  build: {
    outDir: "dist",
    sourcemap: false
  }
});
