import site from 'virtual:site-config';

export { site };

/** Resolve a configured icon or image path against the site base (works under any Pages sub-path). */
export const siteAsset = (path: string): string =>
  /^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith('/') ? path : `${import.meta.env.BASE_URL}${path.replace(/^\.?\//, '')}`;
