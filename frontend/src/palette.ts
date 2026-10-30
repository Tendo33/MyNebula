import { avatarUrl, html, onAction, type Raw } from './html';
import { t } from './i18n';
import { icon } from './icons';
import { navigate } from './router';
import {
  buildLanguageFacets,
  buildTagFacets,
  matchesClusterSearch,
  matchesFacetSearch,
  matchesRepoSearch,
  normalizeSearchQuery,
  parseStarsThreshold,
} from './search';
import { store } from './store';
import type { ClusterInfo, GraphNode } from './types';

type FilterType = 'all' | 'repos' | 'clusters' | 'languages' | 'tags';
type Result =
  | { type: 'repo'; id: number; title: string; subtitle?: string | null; meta: string; data: GraphNode }
  | { type: 'cluster'; id: number; title: string; subtitle?: string | null; meta: string; data: ClusterInfo }
  | { type: 'language'; id: string; title: string; subtitle: string; meta: string }
  | { type: 'tag'; id: string; title: string; subtitle: string; meta: string };

const TYPE_LABELS: Record<Result['type'], string> = {
  repo: 'search.type_repo',
  cluster: 'search.type_cluster',
  language: 'search.type_language',
  tag: 'search.type_tag',
};
const RECENT_KEY = 'nebula_recent_searches';
const MAX_RECENT = 5;
const MAX_RESULTS = 20;
const STAR_RANGES = [1000, 10000, 50000];

const loadRecent = (): string[] => {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]');
  } catch {
    return [];
  }
};

const saveRecent = (items: string[]) => {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(items));
  } catch {
    // History stays in memory for this session.
  }
};

export const buildResults = (nodes: GraphNode[], clusters: ClusterInfo[], query: string, filter: FilterType): Result[] => {
  const normalized = normalizeSearchQuery(query);
  if (!normalized) return [];
  const facetsOff = parseStarsThreshold(normalized) !== null;
  const results: Result[] = [];
  if (filter === 'all' || filter === 'repos') {
    nodes
      .filter((node) => matchesRepoSearch(node, normalized))
      .slice(0, MAX_RESULTS)
      .forEach((node) =>
        results.push({
          type: 'repo',
          id: node.id,
          title: node.full_name,
          subtitle: node.ai_summary || node.description,
          meta: `${node.stargazers_count.toLocaleString()}${node.language ? ` · ${node.language}` : ''}`,
          data: node,
        }),
      );
  }
  if (facetsOff) return results;
  if (filter === 'all' || filter === 'clusters') {
    clusters
      .filter((cluster) => matchesClusterSearch(cluster, normalized))
      .slice(0, 5)
      .forEach((cluster) =>
        results.push({
          type: 'cluster',
          id: cluster.id,
          title: cluster.name || t('search.cluster_fallback_name', { id: cluster.id }),
          subtitle: cluster.description,
          meta: t('search.repo_count', { count: cluster.repo_count }),
          data: cluster,
        }),
      );
  }
  if (filter === 'all' || filter === 'languages') {
    buildLanguageFacets(nodes)
      .filter(([lang]) => matchesFacetSearch(lang, normalized))
      .slice(0, 5)
      .forEach(([lang, count]) =>
        results.push({
          type: 'language',
          id: lang,
          title: lang,
          subtitle: t('search.filter_by_language', { language: lang }),
          meta: t('search.repo_count', { count }),
        }),
      );
  }
  if (filter === 'all' || filter === 'tags') {
    buildTagFacets(nodes)
      .filter(([tag]) => matchesFacetSearch(tag, normalized))
      .slice(0, 5)
      .forEach(([tag, count]) =>
        results.push({
          type: 'tag',
          id: tag,
          title: tag,
          subtitle: t('search.filter_by_tag'),
          meta: t('search.repo_count', { count }),
        }),
      );
  }
  return results;
};

const resultIcon = (result: Result): Raw => {
  if (result.type === 'repo') {
    const url = avatarUrl(result.data.owner_avatar_url, 48);
    return url
      ? html`<img class="avatar avatar-sm" src="${url}" alt="" loading="lazy" decoding="async" />`
      : html`<span class="avatar avatar-sm avatar-fallback">${result.data.owner.charAt(0).toUpperCase()}</span>`;
  }
  if (result.type === 'cluster') return html`<span class="dot dot-xl" style="background:${result.data.color}"></span>`;
  return html`<span class="accent">${icon(result.type === 'language' ? 'code' : 'tag')}</span>`;
};

let open: (() => void) | null = null;

export const openPalette = (): void => open?.();

/** Cmd/Ctrl+K command palette over repos, clusters, languages, and tags. */
export const initPalette = (): void => {
  const host = document.createElement('div');
  document.body.appendChild(host);
  let isOpen = false;
  let query = '';
  let filter: FilterType = 'all';
  let selected = 0;
  let results: Result[] = [];
  let recent = loadRecent();
  let previousFocus: HTMLElement | null = null;

  const addRecent = (value: string) => {
    if (!value.trim()) return;
    recent = [value, ...recent.filter((item) => item !== value)].slice(0, MAX_RECENT);
    saveRecent(recent);
  };

  const close = () => {
    isOpen = false;
    host.innerHTML = '';
    previousFocus?.focus();
  };

  const goSearch = (value: string, facet: 'search' | 'language' = 'search') => {
    addRecent(query || value);
    close();
    navigate('/', facet === 'language' ? { language: value } : { q: value });
  };

  const choose = (result: Result) => {
    addRecent(query);
    close();
    if (result.type === 'repo') navigate('/', { node: String(result.data.id) });
    else if (result.type === 'cluster') navigate('/', { cluster: String(result.data.id) });
    else if (result.type === 'language') navigate('/', { language: result.title });
    else navigate('/', { q: result.title });
  };

  const renderBody = () => {
    const body = host.querySelector<HTMLElement>('[data-region="body"]');
    const footer = host.querySelector<HTMLElement>('[data-region="count"]');
    if (!body || !footer) return;
    results = buildResults(store.nodes, store.data?.clusters ?? [], query, filter);
    selected = Math.min(selected, Math.max(results.length - 1, 0));
    footer.textContent = results.length ? `${results.length} ${t('search.results')}` : '';
    body.setAttribute('role', results.length ? 'listbox' : 'presentation');

    if (!query.trim()) {
      const nodes = store.nodes;
      body.innerHTML = html`<div class="palette-empty">
        ${recent.length
          ? html`<div>
              <div class="palette-section-head"><span class="eyebrow">${t('search.recent')}</span>
                <button class="link-btn xsmall" data-act="clear-recent">${t('common.clear')}</button></div>
              <div class="chips">${recent.map((item) => html`<button class="chip" data-act="fill" data-value="${item}">${icon('clock')}${item}</button>`)}</div>
            </div>`
          : ''}
        <div>
          <span class="eyebrow">${t('search.quickFilters')}</span>
          <div class="chips">${buildLanguageFacets(nodes, 5).map(
            ([lang, count]) => html`<button class="chip chip-accent" data-act="fill" data-value="${lang}">${icon('code')}${lang}<span>(${count})</span></button>`,
          )}</div>
          <div class="chips">${buildTagFacets(nodes, 6).map(
            ([tag, count]) => html`<button class="chip chip-accent" data-act="fill" data-value="${tag}">${icon('tag')}${tag}<span>(${count})</span></button>`,
          )}</div>
          <div class="chips">${STAR_RANGES.map(
            (min) => html`<button class="chip chip-accent" data-act="stars" data-value="${min}">${icon('star')}${min / 1000}k+</button>`,
          )}</div>
        </div>
        <p class="muted xsmall palette-tip">${t('search.tip')}</p>
      </div>`.value;
      return;
    }
    body.innerHTML = results.length
      ? html`${results.map(
          (result, index) => html`<button class="palette-item" role="option" id="palette-option-${index}" data-act="choose"
            data-index="${index}" aria-selected="${index === selected}">
            ${resultIcon(result)}
            <span class="palette-item-body">
              <span class="palette-item-title"><span class="truncate strong">${result.title}</span><span class="type-tag">${t(TYPE_LABELS[result.type])}</span></span>
              ${result.subtitle ? html`<span class="muted small truncate">${result.subtitle}</span>` : ''}
            </span>
            <span class="stars muted xsmall nowrap">${result.type === 'repo' ? icon('star') : ''}${result.meta}</span>
            <span class="palette-arrow">${icon('arrowRight')}</span>
          </button>`,
        )}`.value
      : html`<div class="pad"><p class="small">${t('search.noResults')} "${query}"</p><p class="muted xsmall">${t('search.tryDifferent')}</p></div>`.value;
    body.setAttribute('aria-activedescendant', `palette-option-${selected}`);
    body.querySelector(`[data-index="${selected}"]`)?.scrollIntoView({ block: 'nearest' });
  };

  const render = () => {
    const filters: [FilterType, string][] = [
      ['all', t('search.all')],
      ['repos', t('search.repos')],
      ['clusters', t('search.clusters')],
      ['languages', t('search.languages')],
      ['tags', t('search.tags')],
    ];
    host.innerHTML = html`<div class="palette-backdrop" data-act="close"></div>
      <div class="palette" role="dialog" aria-modal="true" aria-label="${t('search.placeholder')}">
        <div class="palette-input">
          ${icon('search')}
          <input type="text" data-input placeholder="${t('search.placeholder')}" aria-label="${t('search.placeholder')}" />
          <kbd class="hide-sm">⌘K</kbd>
          <button class="btn btn-ghost btn-icon btn-sm" data-act="close" aria-label="${t('common.close')}">${icon('x')}</button>
        </div>
        <div class="palette-filters toggle-group">${filters.map(
          ([key, label]) => html`<button class="toggle-item" data-act="filter" data-value="${key}" aria-pressed="${filter === key}">${label}</button>`,
        )}</div>
        <div class="palette-body" data-region="body"></div>
        <div class="palette-footer xsmall muted">
          <span><kbd>↑↓</kbd>${t('search.navigate')}</span>
          <span><kbd>↵</kbd>${t('search.select')}</span>
          <span><kbd>esc</kbd>${t('search.close')}</span>
          <span class="grow"></span><span data-region="count"></span>
        </div>
      </div>`.value;
    const input = host.querySelector<HTMLInputElement>('[data-input]')!;
    input.value = query;
    renderBody();
    input.focus();
  };

  open = () => {
    if (isOpen) return;
    isOpen = true;
    previousFocus = document.activeElement as HTMLElement | null;
    query = '';
    selected = 0;
    render();
  };

  onAction(host, 'click', {
    close,
    filter: (button) => {
      filter = button.dataset.value as FilterType;
      host.querySelectorAll('[data-act="filter"]').forEach((el) => el.setAttribute('aria-pressed', String(el === button)));
      renderBody();
    },
    fill: (button) => {
      query = button.dataset.value!;
      host.querySelector<HTMLInputElement>('[data-input]')!.value = query;
      selected = 0;
      renderBody();
    },
    stars: (button) => goSearch(`stars:>${button.dataset.value}`),
    'clear-recent': () => {
      recent = [];
      saveRecent(recent);
      renderBody();
    },
    choose: (button) => choose(results[Number(button.dataset.index)]),
  });

  host.addEventListener('input', (event) => {
    query = (event.target as HTMLInputElement).value;
    selected = 0;
    renderBody();
  });

  window.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      if (isOpen) close();
      else open?.();
      return;
    }
    if (!isOpen) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      selected = Math.max(0, Math.min(results.length - 1, selected + (event.key === 'ArrowDown' ? 1 : -1)));
      renderBody();
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (results[selected]) choose(results[selected]);
      else if (query.trim()) goSearch(query.trim());
    } else if (event.key === 'Escape') {
      event.preventDefault();
      close();
    } else if (event.key === 'Tab') {
      // Keep focus inside the dialog.
      const focusable = [...host.querySelectorAll<HTMLElement>('input, button')];
      const index = focusable.indexOf(document.activeElement as HTMLElement);
      if (event.shiftKey && index <= 0) {
        event.preventDefault();
        focusable[focusable.length - 1]?.focus();
      } else if (!event.shiftKey && index === focusable.length - 1) {
        event.preventDefault();
        focusable[0]?.focus();
      }
    }
  });
};
