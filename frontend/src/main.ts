import './styles.css';

import { html, onAction, qs } from './html';
import { onLangChange, t } from './i18n';
import { icon } from './icons';
import { currentRoute, href, onRoute, type Route } from './router';
import { store } from './store';
import { initTheme } from './theme';
import { initPalette, openPalette } from './palette';
import { mountGraph } from './pages/graph';
import { mountData } from './pages/data';
import { mountSettings } from './pages/settings';

type Mount = (root: HTMLElement, route: Route) => () => void;

const PAGES: { key: string; paths: string[]; label: string; mount: Mount }[] = [
  // `/graph` stays as an alias so old links and palette navigation keep working.
  { key: 'graph', paths: ['/', '/graph'], label: 'nav.graph', mount: mountGraph },
  { key: 'data', paths: ['/data'], label: 'nav.data', mount: mountData },
  { key: 'settings', paths: ['/settings'], label: 'nav.settings', mount: mountSettings },
];

const mountNotFound: Mount = (root) => {
  root.innerHTML = html`<div class="center-state">
    <h1 class="empty-title">404</h1>
    <p class="muted">${t('errors.not_found')}</p>
    <a class="btn btn-primary" href="#/">${t('nav.graph')}</a>
  </div>`.value;
  return () => {};
};

const app = document.getElementById('app')!;
const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
let unmountPage: (() => void) | null = null;
let mountedKey: string | null = null;

const pageFor = (path: string) => PAGES.find((p) => p.paths.includes(path));

const renderShell = (active: string | undefined) => {
  app.innerHTML = html`
    <a class="skip-link" href="#main-content" data-act="skip">${t('common.skip_to_content')}</a>
    <header class="topbar">
      <a class="wordmark" href="#/" aria-label="MyNebula">MyNebula</a>
      <nav class="tabs-nav" aria-label="${t('nav.label')}">
        ${PAGES.map(
          (p) => html`<a href="${href(p.paths[0])}" ${p.key === active ? html`aria-current="page"` : ''}>${t(p.label)}</a>`,
        )}
      </nav>
      <button class="search-hint" data-act="palette" aria-label="${t('common.command_palette')}">
        ${icon('search')}<span class="hide-sm">${t('common.search_command')}</span>
        <kbd class="hide-sm">${isMac ? '⌘K' : 'Ctrl K'}</kbd>
      </button>
    </header>
    <main id="main-content" class="page" tabindex="-1"></main>`.value;
};

const mountPage = (route: Route, force = false) => {
  const page = pageFor(route.path);
  const key = page?.key ?? '404';
  if (!force && key === mountedKey) return;
  unmountPage?.();
  renderShell(page?.key);
  mountedKey = key;
  unmountPage = (page?.mount ?? mountNotFound)(qs(app, '#main-content'), route);
};

initTheme();
initPalette();
onAction(app, 'click', {
  palette: () => openPalette(),
  skip: (_, event) => {
    event.preventDefault();
    qs(app, '#main-content').focus();
  },
});
mountPage(currentRoute());
onRoute((route) => mountPage(route));
onLangChange(() => mountPage(currentRoute(), true));
void store.load();
