/**
 * Hash router (`#/graph?node=1`). Static hosts such as GitHub Pages cannot
 * rewrite deep links to index.html, so history routing would 404 on refresh.
 */
export interface Route {
  path: string;
  params: URLSearchParams;
}

const listeners = new Set<(route: Route) => void>();
const locationListeners = new Set<(route: Route) => void>();

const emitLocation = (route: Route) => locationListeners.forEach((fn) => fn(route));

/** Fires for hash changes and silent replaceState, so chrome can follow the URL. */
export const onLocation = (fn: (route: Route) => void): (() => void) => {
  locationListeners.add(fn);
  return () => locationListeners.delete(fn);
};

export const currentRoute = (): Route => {
  const hash = location.hash.replace(/^#/, '') || '/';
  const [path, query = ''] = hash.split('?');
  return { path: path || '/', params: new URLSearchParams(query) };
};

export const href = (path: string, params?: URLSearchParams | Record<string, string>): string => {
  const query = new URLSearchParams(params).toString();
  return `#${path}${query ? `?${query}` : ''}`;
};

export const navigate = (path: string, params?: URLSearchParams | Record<string, string>, replace = false): void => {
  const target = href(path, params);
  if (target === location.hash || (target === '#/' && !location.hash)) return;
  if (replace) {
    // Silent for pages: used to mirror page state into the URL, not to navigate.
    history.replaceState(null, '', target);
    emitLocation(currentRoute());
  } else {
    location.hash = target;
  }
};

export const onRoute = (fn: (route: Route) => void): (() => void) => {
  listeners.add(fn);
  return () => listeners.delete(fn);
};

window.addEventListener('hashchange', () => {
  const route = currentRoute();
  listeners.forEach((fn) => fn(route));
  emitLocation(route);
});
