/// <reference types="vite/client" />

declare module 'virtual:site-config' {
  import type { SiteConfig } from './types';
  const config: SiteConfig;
  export default config;
}
