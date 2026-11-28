import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { SiteConfig } from './frontend/src/types.ts';

export type SiteConfigFile = Partial<Record<keyof SiteConfig, unknown>>;

const ENV_KEY: Record<keyof SiteConfig, string> = {
  name: 'SITE_NAME',
  tagline: 'SITE_TAGLINE',
  description: 'SITE_DESCRIPTION',
  icon: 'SITE_ICON',
  appleIcon: 'SITE_APPLE_ICON',
  ogImage: 'SITE_OG_IMAGE',
  themeColor: 'SITE_THEME_COLOR',
  defaultLanguage: 'SITE_DEFAULT_LANGUAGE',
  url: 'SITE_URL',
};

const pick = (key: keyof SiteConfig, file: SiteConfigFile, env: Record<string, string | undefined>, fallback: string): string => {
  const fromEnv = env[ENV_KEY[key]]?.trim();
  if (fromEnv) return fromEnv;
  const fromFile = typeof file[key] === 'string' ? String(file[key]).trim() : '';
  return fromFile || fallback;
};

/** File values first, then `SITE_*` env overrides, then built-in defaults. */
export const resolveSiteConfig = (file: SiteConfigFile, env: Record<string, string | undefined>): SiteConfig => {
  const name = pick('name', file, env, 'MyNebula');
  const tagline = pick('tagline', file, env, 'Your GitHub stars as a living semantic graph');
  const language = pick('defaultLanguage', file, env, 'auto').toLowerCase();
  return {
    name,
    tagline,
    description: pick('description', file, env, tagline),
    icon: pick('icon', file, env, 'mark.svg'),
    appleIcon: pick('appleIcon', file, env, 'apple-touch-icon.png'),
    ogImage: pick('ogImage', file, env, 'logo.png'),
    themeColor: pick('themeColor', file, env, '#fafafa'),
    defaultLanguage: language.startsWith('zh') ? 'zh' : language.startsWith('en') ? 'en' : 'auto',
    url: pick('url', file, env, ''),
  };
};

/** A missing or malformed file is a valid fork state: defaults take over. */
export const readSiteConfigFile = (root: string): SiteConfigFile => {
  try {
    return JSON.parse(readFileSync(resolve(root, 'site.config.json'), 'utf8')) as SiteConfigFile;
  } catch {
    return {};
  }
};
