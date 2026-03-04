import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
	// Load env file based on `mode` in the current working directory.
	const env = loadEnv(mode, process.cwd(), "");
	const apiTarget = env.VITE_API_BASE_URL || "http://localhost:8071";

	return {
		plugins: [react()],
		server: {
			proxy: {
				"/api": {
					target: apiTarget,
					changeOrigin: true,
				},
			},
		},
		resolve: {
			alias: {
				"@": path.resolve(__dirname, "./src"),
			},
			dedupe: ["three"],
		},
	};
});
