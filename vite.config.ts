import { loadEnv } from 'vite';
import { defineConfig, type Plugin } from 'vitest/config';
import { readSiteConfigFile, resolveSiteConfig } from './site-config.ts';
import type { SiteConfig } from './frontend/src/types.ts';

const VIRTUAL_ID = 'virtual:site-config';
const RESOLVED_ID = `\0${VIRTUAL_ID}`;

const escapeAttribute = (value: string): string =>
  value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const readSiteConfig = (mode: string): SiteConfig => {
  const root = process.cwd();
  return resolveSiteConfig(readSiteConfigFile(root), loadEnv(mode, root, ''));
};

const assetPath = (value: string): string =>
  /^(?:[a-z]+:)?\/\//i.test(value) || value.startsWith('/') ? value : `./${value.replace(/^\.?\//, '')}`;

/** Injects the site's name, icon, and metadata into index.html and the runtime bundle. */
const siteConfigPlugin = (site: SiteConfig): Plugin => ({
  name: 'mynebula-site-config',
  resolveId: (id) => (id === VIRTUAL_ID ? RESOLVED_ID : null),
  load: (id) => (id === RESOLVED_ID ? `export default ${JSON.stringify(site)};` : null),
  transformIndexHtml(html) {
    const publicUrl = site.url.trim().replace(/\/+$/, '');
    const absolute = (path: string): string => {
      if (!publicUrl) return assetPath(path);
      try {
        return new URL(assetPath(path).replace(/^\.\//, ''), `${publicUrl}/`).href;
      } catch {
        return assetPath(path);
      }
    };
    const canonical = publicUrl
      ? `<link rel="canonical" href="${escapeAttribute(publicUrl)}/" />\n    <meta property="og:url" content="${escapeAttribute(publicUrl)}/" />`
      : '';
    const values: Record<string, string> = {
      SITE_NAME: escapeAttribute(site.name),
      SITE_DESCRIPTION: escapeAttribute(site.description),
      SITE_TAGLINE: escapeAttribute(site.tagline),
      SITE_ICON: escapeAttribute(assetPath(site.icon)),
      SITE_APPLE_ICON: escapeAttribute(assetPath(site.appleIcon)),
      SITE_OG_IMAGE: escapeAttribute(absolute(site.ogImage)),
      SITE_THEME_COLOR: escapeAttribute(site.themeColor),
      SITE_LANG: site.defaultLanguage === 'auto' ? 'en' : site.defaultLanguage,
      SITE_CANONICAL: canonical,
    };
    return html.replace(/%SITE_([A-Z_]+)%/g, (match, key: string) => values[`SITE_${key}`] ?? match);
  },
});

// `base: './'` keeps asset URLs relative so the build works under any
// GitHub Pages sub-path without configuration.
export default defineConfig(({ mode }) => ({
  root: 'frontend',
  base: './',
  plugins: [siteConfigPlugin(readSiteConfig(mode))],
  build: { target: 'es2022', outDir: 'dist', emptyOutDir: true },
  test: { root: '.', include: ['pipeline/**/*.test.ts', 'frontend/src/**/*.test.ts'] },
}));
