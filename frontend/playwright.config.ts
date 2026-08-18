import { defineConfig } from '@playwright/test';

const hasE2EBaseUrl = Boolean(process.env.E2E_BASE_URL);

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: hasE2EBaseUrl
    ? undefined
    : {
        // `pnpm run dev -- --port 4173` forwards a literal `--` to vite, which
        // then ignores every following flag and starts on the default 5173 —
        // so the server never appeared on the awaited port and this suite could
        // not run at all. Invoke vite directly instead, and use strictPort so a
        // busy port fails loudly rather than silently relocating.
        command: 'pnpm exec vite --host 127.0.0.1 --port 4173 --strictPort',
        port: 4173,
        reuseExistingServer: true,
      },
});
