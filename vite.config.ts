import { defineConfig } from 'vitest/config';

// `base: './'` keeps asset URLs relative so the build works under any
// GitHub Pages sub-path without configuration.
export default defineConfig({
  root: 'frontend',
  base: './',
  build: { target: 'es2022', outDir: 'dist', emptyOutDir: true },
  test: { root: '.', include: ['pipeline/**/*.test.ts', 'frontend/src/**/*.test.ts'] },
});
