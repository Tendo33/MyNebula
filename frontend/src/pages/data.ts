import { avatarUrl, html, onAction, qs, type Raw } from '../html';
import { t } from '../i18n';
import { icon } from '../icons';
import { currentRoute, href, navigate, onRoute } from '../router';
import { filterRepos, nextSort, sortRepos, type DataQuery, type SortConfig, type SortField } from './dataQuery';
import { store } from '../store';
import type { ClusterInfo, GraphNode } from '../types';
import { errorState, formatDate, loadingState } from './common';

const PAGE_SIZES = [25, 50, 100];
const SORT_OPTIONS: [SortField, string][] = [
  ['starred_at', 'data.starred_date'],
  ['name', 'data.repository'],
  ['stargazers_count', 'data.stars'],
  ['language', 'data.language'],
  ['cluster', 'data.cluster'],
  ['summary', 'data.summary'],
  ['last_commit_time', 'data.last_commit'],
];

const parseClusters = (params: URLSearchParams): Set<number> =>
  new Set(
    (params.get('clusters') ?? params.get('cluster') ?? '')
      .split(',')
      .map((v) => Number.parseInt(v, 10))
      .filter(Number.isFinite),
  );

const clusterTag = (cluster: ClusterInfo | undefined): Raw =>
  cluster
    ? html`<button class="cluster-tag" data-act="cluster" data-id="${cluster.id}" title="${cluster.name}">
        <span class="dot" style="--dot:${cluster.color}"></span><span class="truncate">${cluster.name}</span>
      </button>`
    : html`<span class="dim">—</span>`;

const repoAvatar = (repo: GraphNode, size: number) => {
  const url = avatarUrl(repo.owner_avatar_url, size * 2);
  return url
    ? html`<img class="avatar" src="${url}" alt="" loading="lazy" decoding="async" width="${size}" height="${size}" />`
    : html`<span class="avatar avatar-fallback" style="width:${size}px;height:${size}px">${repo.owner.charAt(0).toUpperCase()}</span>`;
};

export const mountData = (root: HTMLElement): (() => void) => {
  let sort: SortConfig = { field: 'starred_at', direction: 'desc' };
  let page = 1;
  let pageSize = 25;
  const route = currentRoute();
  const q: DataQuery = {
    query: route.params.get('q') ?? '',
    clusters: parseClusters(route.params),
    month: route.params.get('month'),
    topic: route.params.get('topic'),
  };

  const el = (selector: string) => qs<HTMLElement>(root, selector);

  const syncUrl = () => {
    const params = new URLSearchParams();
    if (q.query.trim()) params.set('q', q.query.trim());
    const clusters = [...q.clusters].sort((a, b) => a - b);
    if (clusters.length === 1) params.set('cluster', String(clusters[0]));
    else if (clusters.length > 1) params.set('clusters', clusters.join(','));
    if (q.month) params.set('month', q.month);
    if (q.topic) params.set('topic', q.topic);
    navigate('/data', params, true);
  };

  const activeFilters = () => Boolean(q.clusters.size || q.query.trim() || q.month || q.topic);

  const sortHeader = (label: string, field: SortField, align = '') => {
    const active = sort.field === field;
    return html`<th scope="col" class="${align}" aria-sort="${active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}">
      <button class="sort-btn ${active ? 'on' : ''}" data-act="sort" data-field="${field}">
        ${label}${icon(active && sort.direction === 'asc' ? 'chevronUp' : 'chevronDown', 'icon sort-icon')}
      </button>
    </th>`;
  };

  const renderFrame = () => {
    if (store.loading) return void (root.innerHTML = loadingState().value);
    if (store.error) return void (root.innerHTML = errorState().value);
    root.innerHTML = html`<div class="data-page">
      <div class="data-toolbar">
        <label class="search-input data-search">
          ${icon('search')}
          <input type="search" data-search placeholder="${t('data.search_placeholder')}" aria-label="${t('data.search_placeholder')}" />
        </label>
        <div class="mobile-sort">
          <select class="select" data-sort-field aria-label="${t('data.sort')}">
            ${SORT_OPTIONS.map(([value, label]) => html`<option value="${value}">${t(label)}</option>`)}
          </select>
          <button class="btn btn-outline btn-icon" data-act="sort-dir" aria-label="${t('data.sort_direction')}" data-region="sort-dir"></button>
        </div>
        <span class="data-count muted tabular" data-region="count"></span>
      </div>
      <div class="data-chips" data-region="chips"></div>
      <div class="data-body" data-region="table"></div>
      <div class="data-pager" data-region="pager"></div>
    </div>`.value;
    (el('[data-search]') as HTMLInputElement).value = q.query;
    renderResults();
  };

  const renderResults = () => {
    if (!root.querySelector('[data-region="table"]')) return;
    const all = store.nodes;
    const clusters = store.data?.clusters ?? [];
    const clusterMap = new Map(clusters.map((c) => [c.id, c]));
    const filtered = sortRepos(filterRepos(all, q), sort);
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    page = Math.min(page, totalPages);
    const offset = (page - 1) * pageSize;
    const rows = filtered.slice(offset, offset + pageSize);

    (el('[data-sort-field]') as HTMLSelectElement).value = sort.field;
    el('[data-region="sort-dir"]').innerHTML = icon(sort.direction === 'asc' ? 'chevronUp' : 'chevronDown').value;
    el('[data-region="count"]').innerHTML = html`${all.length ? t('data.shown_of_total', { shown: filtered.length, total: all.length }) : ''}
      ${activeFilters() ? html`<button class="link-btn" data-act="clear">${t('common.clear_filters')}</button>` : ''}`.value;

    const sortedClusters = [...clusters].sort((a, b) => b.repo_count - a.repo_count);
    el('[data-region="chips"]').innerHTML = html`
      ${q.month
        ? html`<span class="chip chip-on">${icon('calendar')}${q.month}
            <button class="chip-x" data-act="clear-month" aria-label="${t('common.clear_filter')}">${icon('x')}</button></span>`
        : ''}
      ${q.topic
        ? html`<span class="chip chip-on" title="${q.topic}">${icon('tag')}${q.topic}
            <button class="chip-x" data-act="clear-topic" aria-label="${t('common.clear_filter')}">${icon('x')}</button></span>`
        : ''}
      ${sortedClusters.map(
        (cluster) => html`<button class="chip" data-act="cluster" data-id="${cluster.id}" aria-pressed="${q.clusters.has(cluster.id)}" title="${cluster.name}">
          <span class="dot" style="--dot:${cluster.color}"></span><span class="truncate">${cluster.name}</span><span class="chip-count">${cluster.repo_count}</span>
        </button>`,
      )}`.value;

    const empty = html`<div class="center-state">
      <h2 class="empty-title">${activeFilters() ? t('data.no_results') : t('data.no_data')}</h2>
      ${activeFilters() ? '' : html`<p class="muted">${t('data.empty_hint')}</p>`}
      ${activeFilters()
        ? html`<button class="btn btn-outline" data-act="clear">${t('common.clear_filters')}</button>`
        : html`<a class="btn btn-primary" href="#/settings">${t('common.sync_now')}</a>`}
    </div>`;

    const nodeLink = (repo: GraphNode) => href('/', { node: String(repo.id) });
    el('[data-region="table"]').innerHTML = html`
      <div class="table-scroll desktop-only">
        <table class="table">
          <thead><tr>
            ${sortHeader(t('data.repository'), 'name')}
            ${sortHeader(t('data.summary'), 'summary')}
            ${sortHeader(t('data.language'), 'language')}
            ${sortHeader(t('data.stars'), 'stargazers_count', 'num')}
            ${sortHeader(t('data.cluster'), 'cluster')}
            ${sortHeader(t('data.starred_date'), 'starred_at', 'num')}
            ${sortHeader(t('data.last_commit'), 'last_commit_time', 'num')}
          </tr></thead>
          <tbody>
            ${rows.map(
              (repo) => html`<tr>
                <td class="col-name"><a class="repo-link" href="${nodeLink(repo)}">${repoAvatar(repo, 20)}<span class="truncate">${repo.full_name}</span></a></td>
                <td class="col-summary"><p class="clamp-2 muted" title="${repo.ai_summary || repo.description || ''}">${repo.ai_summary || repo.description || '—'}</p></td>
                <td class="muted">${repo.language || '—'}</td>
                <td class="num tabular">${repo.stargazers_count.toLocaleString()}</td>
                <td class="col-cluster">${clusterTag(repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined)}</td>
                <td class="num muted tabular">${formatDate(repo.starred_at)}</td>
                <td class="num muted tabular">${formatDate(repo.last_commit_time)}</td>
              </tr>`,
            )}
          </tbody>
        </table>
        ${rows.length ? '' : empty}
      </div>
      <ul class="repo-list mobile-only">
        ${rows.map(
          (repo) => html`<li>
            <a class="repo-link" href="${nodeLink(repo)}">${repoAvatar(repo, 20)}<span class="truncate">${repo.full_name}</span></a>
            <p class="clamp-2 muted small">${repo.ai_summary || repo.description || t('data.no_summary')}</p>
            <div class="repo-meta small muted">
              <span class="stars tabular">${icon('star')}${repo.stargazers_count.toLocaleString()}</span>
              ${repo.language ? html`<span>${repo.language}</span>` : ''}
              ${clusterTag(repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined)}
              <span class="tabular">${formatDate(repo.starred_at)}</span>
            </div>
          </li>`,
        )}
        ${rows.length ? '' : html`<li>${empty}</li>`}
      </ul>`.value;

    el('[data-region="pager"]').innerHTML = filtered.length
      ? html`<label class="muted small pager-size">${t('data.rows_per_page')}
            <select class="select select-sm" data-page-size>${PAGE_SIZES.map(
              (size) => html`<option value="${size}" ${size === pageSize ? 'selected' : ''}>${size}</option>`,
            )}</select>
          </label>
          <span class="muted small tabular">${t('data.showing', { start: offset + 1, end: offset + rows.length, total: filtered.length })}</span>
          <span class="pager-buttons">
            <button class="btn btn-ghost btn-icon btn-sm" data-act="prev" ${page === 1 ? 'disabled' : ''} aria-label="${t('data.previous_page')}">${icon('chevronLeft')}</button>
            <button class="btn btn-ghost btn-icon btn-sm" data-act="next" ${page === totalPages ? 'disabled' : ''} aria-label="${t('data.next_page')}">${icon('chevronRight')}</button>
          </span>`.value
      : '';
  };

  const update = (resetPage = true) => {
    if (resetPage) page = 1;
    syncUrl();
    renderResults();
  };

  renderFrame();
  const cleanups = [
    store.subscribe(['data'], renderFrame),
    onRoute((next) => {
      if (next.path !== '/data') return;
      q.query = next.params.get('q') ?? '';
      q.clusters = parseClusters(next.params);
      q.month = next.params.get('month');
      q.topic = next.params.get('topic');
      const search = root.querySelector<HTMLInputElement>('[data-search]');
      if (search) search.value = q.query;
      page = 1;
      renderResults();
    }),
    onAction(root, 'click', {
      retry: () => void store.load(),
      sort: (button) => {
        sort = nextSort(sort, button.dataset.field as SortField);
        update();
      },
      'sort-dir': () => {
        sort = { ...sort, direction: sort.direction === 'asc' ? 'desc' : 'asc' };
        update();
      },
      cluster: (button) => {
        const id = Number(button.dataset.id);
        q.clusters = new Set(q.clusters);
        if (q.clusters.has(id)) q.clusters.delete(id);
        else q.clusters.add(id);
        update();
      },
      'clear-month': () => {
        q.month = null;
        update();
      },
      'clear-topic': () => {
        q.topic = null;
        update();
      },
      clear: () => {
        Object.assign(q, { query: '', clusters: new Set(), month: null, topic: null });
        (el('[data-search]') as HTMLInputElement).value = '';
        update();
      },
      prev: () => {
        page = Math.max(1, page - 1);
        renderResults();
      },
      next: () => {
        page += 1;
        renderResults();
      },
    }),
  ];

  let searchTimer = 0;
  const onInput = (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (!target.matches('[data-search]')) return;
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => {
      q.query = target.value;
      update();
    }, 150);
  };
  const onChange = (event: Event) => {
    const target = event.target as HTMLSelectElement;
    if (target.matches('[data-page-size]')) {
      pageSize = Number(target.value);
      update();
    } else if (target.matches('[data-sort-field]')) {
      sort = { ...sort, field: target.value as SortField };
      update();
    }
  };
  root.addEventListener('input', onInput);
  root.addEventListener('change', onChange);

  return () => {
    window.clearTimeout(searchTimer);
    cleanups.forEach((fn) => fn());
    root.removeEventListener('input', onInput);
    root.removeEventListener('change', onChange);
  };
};
