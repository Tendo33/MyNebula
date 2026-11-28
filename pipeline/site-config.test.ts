import { describe, expect, it } from 'vitest';
import { resolveSiteConfig } from '../site-config.ts';

describe('resolveSiteConfig', () => {
  it('builds a complete config from empty input', () => {
    expect(resolveSiteConfig({}, {})).toEqual({
      name: 'MyNebula',
      tagline: 'Your GitHub stars as a living semantic graph',
      description: 'Your GitHub stars as a living semantic graph',
      icon: 'mark.svg',
      appleIcon: 'apple-touch-icon.png',
      ogImage: 'logo.png',
      themeColor: '#fafafa',
      defaultLanguage: 'auto',
      url: '',
    });
  });

  it('uses the fork file values', () => {
    const site = resolveSiteConfig(
      { name: 'Stars by Sam', icon: 'sam.svg', defaultLanguage: 'zh', url: 'https://sam.example/' },
      {},
    );
    expect(site.name).toBe('Stars by Sam');
    expect(site.icon).toBe('sam.svg');
    expect(site.description).toBe('Your GitHub stars as a living semantic graph');
    expect(site.defaultLanguage).toBe('zh');
    expect(site.url).toBe('https://sam.example/');
  });

  it('lets SITE_* variables override the file and ignores blank ones', () => {
    const site = resolveSiteConfig(
      { name: 'From file', icon: 'file.svg' },
      { SITE_NAME: '  From env  ', SITE_ICON: '   ', SITE_THEME_COLOR: '#000000' },
    );
    expect(site.name).toBe('From env');
    expect(site.icon).toBe('file.svg');
    expect(site.themeColor).toBe('#000000');
  });

  it('normalises known languages and keeps unknown ones on auto', () => {
    expect(resolveSiteConfig({}, { SITE_DEFAULT_LANGUAGE: 'zh-CN' }).defaultLanguage).toBe('zh');
    expect(resolveSiteConfig({}, { SITE_DEFAULT_LANGUAGE: 'en-US' }).defaultLanguage).toBe('en');
    expect(resolveSiteConfig({}, { SITE_DEFAULT_LANGUAGE: 'fr' }).defaultLanguage).toBe('auto');
  });
});
