import { renderDetails, focusRelatedItem, focusRelatedTab, moveRelatedItem, moveRelatedTab, type DetailsState } from '../graph/details';
import { avatarUrl, html, onAction, qs, type Raw } from '../html';
import { t } from '../i18n';
import { icon } from '../icons';
import { currentRoute, navigate, onRoute } from '../router';
import {
  DATA_PAGE_SIZES,
  clusterFacetCounts,
  filterRepos,
  nextSort,
  readLanguages,
  readMonth,
  pageContaining,
  readPage,
  readPageSize,
  readSort,
  sortRepos,
  starMonth,
  writeLanguages,
  writeSort,
  type DataQuery,
  type SortField,
} from './dataQuery';
import { store } from '../store';
import type { ClusterInfo, GraphNode } from '../types';
import { errorState, formatDate, loadingState } from './common';

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

const clusterTag = (cluster: ClusterInfo | undefined, selected = false): Raw =>
  cluster
    ? html`<button class="cluster-tag" data-act="cluster" data-id="${cluster.id}" aria-pressed="${selected}" title="${t('search.filter_by_cluster', { cluster: cluster.name })}">
        <span class="dot" style="--dot:${cluster.color}"></span><span class="truncate">${cluster.name}</span>
      </button>`
    : html`<span class="dim">—</span>`;

const languageButton = (language: string | null | undefined, selected: boolean): Raw =>
  language
    ? html`<button class="cluster-tag" data-act="language" data-id="${language}" aria-pressed="${selected}" title="${t('search.filter_by_language', { language })}">${language}</button>`
    : html`<span class="dim">—</span>`;

const monthButton = (starredAt: string | null | undefined, selectedMonth: string | null): Raw => {
  const month = starMonth(starredAt);
  return month
    ? html`<button class="cluster-tag tabular" data-act="month" data-id="${month}" aria-pressed="${selectedMonth === month}" title="${t('search.filter_by_month', { month })}">${formatDate(starredAt)}</button>`
    : html`<span class="dim">—</span>`;
};

const parseNode = (params: URLSearchParams): number | null => {
  const id = Number.parseInt(params.get('node') ?? '', 10);
  return Number.isFinite(id) ? id : null;
};

const repoAvatar = (repo: GraphNode, size: number) => {
  const url = avatarUrl(repo.owner_avatar_url, size * 2);
  return url
    ? html`<img class="avatar" src="${url}" alt="" loading="lazy" decoding="async" width="${size}" height="${size}" />`
    : html`<span class="avatar avatar-fallback" style="width:${size}px;height:${size}px">${repo.owner.charAt(0).toUpperCase()}</span>`;
};

export const mountData = (root: HTMLElement): (() => void) => {
  const route = currentRoute();
  let sort = readSort(route.params);
  let page = readPage(route.params);
  let pageSize = readPageSize(route.params);
  const q: DataQuery = {
    query: route.params.get('q') ?? '',
    clusters: parseClusters(route.params),
    languages: new Set(readLanguages(route.params)),
    month: readMonth(route.params),
    topic: route.params.get('topic'),
  };
  // The open details panel; kept in the URL so a shared Data link reopens it.
  let selectedId = parseNode(route.params);
  const details: DetailsState = { relatedTab: 'similar', graphLink: true };

  const el = (selector: string) => qs<HTMLElement>(root, selector);

  const syncUrl = () => {
    const params = new URLSearchParams();
    if (q.query.trim()) params.set('q', q.query.trim());
    const clusters = [...q.clusters].sort((a, b) => a - b);
    if (clusters.length === 1) params.set('cluster', String(clusters[0]));
    else if (clusters.length > 1) params.set('clusters', clusters.join(','));
    if (q.month) params.set('month', q.month);
    if (q.topic) params.set('topic', q.topic);
    writeLanguages(params, q.languages);
    if (selectedId != null) params.set('node', String(selectedId));
    writeSort(params, sort);
    if (page > 1) params.set('page', String(page));
    if (pageSize !== DATA_PAGE_SIZES[0]) params.set('size', String(pageSize));
    navigate('/data', params, true);
  };

  const activeFilters = () => Boolean(q.clusters.size || q.languages.size || q.query.trim() || q.month || q.topic);

  const sortHeader = (label: string, field: SortField, cls = '') => {
    const active = sort.field === field;
    return html`<th scope="col" class="${cls}" aria-sort="${active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}">
      <button class="sort-btn ${active ? 'on' : ''}" data-act="sort" data-field="${field}">
        ${label}${icon(active && sort.direction === 'asc' ? 'chevronUp' : 'chevronDown', 'icon sort-icon')}
      </button>
    </th>`;
  };

  const renderFrame = () => {
    if (store.loading) return void (root.innerHTML = loadingState().value);
    if (store.error) return void (root.innerHTML = errorState(store.error).value);
    root.innerHTML = html`<div class="data-page">
      <div class="data-main">
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
      </div>
      <section class="data-details" id="data-details" data-region="details" aria-label="${t('common.overview')}"></section>
    </div>`.value;
    (el('[data-search]') as HTMLInputElement).value = q.query;
    renderResults();
    renderDetailsRegion();
  };

  const renderDetailsRegion = () => {
    const region = root.querySelector<HTMLElement>('[data-region="details"]');
    if (!region) return;
    const node = selectedId != null ? store.nodeById.get(selectedId) : undefined;
    root.querySelector('.data-page')?.classList.toggle('has-details', Boolean(node));
    region.innerHTML = node ? renderDetails(node, { ...details, topic: q.topic, languages: q.languages, clusters: q.clusters }).value : '';
  };

  const visibleRow = (id: number): HTMLElement | null =>
    [...root.querySelectorAll<HTMLElement>('[data-row]')].find((row) => Number(row.dataset.row) === id && row.offsetParent !== null) ??
    null;

  /** Move to the page that contains this repository, then scroll that row into view. */
  const select = (id: number | null) => {
    const fromRelated =
      document.activeElement instanceof HTMLElement &&
      document.activeElement.classList.contains('related-item') &&
      root.contains(document.activeElement);
    if (id !== selectedId) details.relatedTab = 'similar';
    selectedId = id;
    const nextPage =
      id == null ? null : pageContaining(sortRepos(filterRepos(store.nodes, q), sort).map((node) => node.id), id, pageSize);
    if (nextPage != null && nextPage !== page) {
      page = nextPage;
      syncUrl();
      renderResults(false);
    } else {
      root.querySelectorAll<HTMLElement>('[data-row]').forEach((row) => row.classList.toggle('is-selected', Number(row.dataset.row) === id));
      syncUrl();
    }
    if (id != null) visibleRow(id)?.scrollIntoView({ block: 'nearest' });
    renderDetailsRegion();
    if (fromRelated && id != null) focusRelatedItem(root);
  };

  /** The control that was focused before a results rebuild, so paging and sorting do not drop the keyboard. */
  const controlRegion = (el: HTMLElement): string => {
    if (el.closest('[data-region="details"]')) return 'details';
    if (el.closest('[data-region="chips"]')) return 'chips';
    if (el.closest('[data-region="table"]')) return 'table';
    return '';
  };

  const focusedControl = (): { act: string; field: string; id: string; region: string; row: string } | null => {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement) || !root.contains(active) || active.closest('[data-region="details"]')) return null;
    if (active.matches('[data-page-size]')) return { act: 'page-size', field: '', id: '', region: '', row: '' };
    const act = active.dataset.act;
    if (!act) return null;
    return {
      act,
      field: active.dataset.field ?? '',
      id: active.dataset.id ?? '',
      region: controlRegion(active),
      row: active.closest<HTMLElement>('[data-row]')?.dataset.row ?? '',
    };
  };

  const restoreControl = (key: { act: string; field: string; id: string; region: string; row: string } | null) => {
    if (!key) return;
    if (key.act === 'page-size') {
      root.querySelector<HTMLElement>('[data-page-size]')?.focus({ preventScroll: true });
      return;
    }
    const candidates = [...root.querySelectorAll<HTMLElement>('[data-act]')].filter(
      (el) =>
        el.dataset.act === key.act &&
        (el.dataset.field ?? '') === key.field &&
        (el.dataset.id ?? '') === key.id &&
        !el.closest('[data-region="details"]') &&
        el.offsetParent !== null &&
        !el.hasAttribute('disabled'),
    );
    const match =
      candidates.find((el) => key.row && el.closest<HTMLElement>('[data-row]')?.dataset.row === key.row) ??
      candidates.find((el) => key.region && controlRegion(el) === key.region) ??
      candidates[0];
    if (match) {
      match.focus({ preventScroll: true });
      if (match.classList.contains('repo-link')) match.scrollIntoView({ block: 'nearest' });
      return;
    }
    if (key.act === 'clear-language' || key.act === 'clear-month' || key.act === 'clear-topic') {
      const fallback =
        root.querySelector<HTMLElement>('[data-act="clear-language"], [data-act="clear-month"], [data-act="clear-topic"]') ??
        root.querySelector<HTMLElement>('[data-act="clear"]') ??
        root.querySelector<HTMLElement>('input[type="search"], input');
      if (fallback && fallback.offsetParent !== null) fallback.focus({ preventScroll: true });
      return;
    }
    if (key.act !== 'next' && key.act !== 'prev') return;
    const fallback = root.querySelector<HTMLElement>(`[data-region="pager"] [data-act="${key.act === 'next' ? 'prev' : 'next'}"]`);
    if (fallback && !fallback.hasAttribute('disabled')) fallback.focus({ preventScroll: true });
  };

  const renderResults = (rebuildChips = true) => {
    if (!root.querySelector('[data-region="table"]')) return;
    const restore = focusedControl();
    const all = store.nodes;
    const clusters = store.data?.clusters ?? [];
    const clusterMap = new Map(clusters.map((c) => [c.id, c]));
    const filtered = sortRepos(filterRepos(all, q), sort);
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    const clamped = Math.min(page, totalPages);
    if (clamped !== page) {
      page = clamped;
      syncUrl();
    }
    const offset = (page - 1) * pageSize;
    const rows = filtered.slice(offset, offset + pageSize);

    (el('[data-sort-field]') as HTMLSelectElement).value = sort.field;
    el('[data-region="sort-dir"]').innerHTML = icon(sort.direction === 'asc' ? 'chevronUp' : 'chevronDown').value;
    el('[data-region="count"]').innerHTML = html`${all.length ? t('data.shown_of_total', { shown: filtered.length, total: all.length }) : ''}
      ${activeFilters() ? html`<button class="link-btn" data-act="clear">${t('common.clear_filters')}</button>` : ''}`.value;

    const clusterCounts = clusterFacetCounts(all, q);
    const sortedClusters = clusters
      .map((cluster) => ({ cluster, count: clusterCounts.get(cluster.id) ?? 0 }))
      .filter((item) => item.count > 0 || q.clusters.has(item.cluster.id))
      .sort((a, b) => b.count - a.count || a.cluster.name.localeCompare(b.cluster.name));
    if (rebuildChips) el('[data-region="chips"]').innerHTML = html`
      ${q.month
        ? html`<span class="chip chip-on">${icon('calendar')}${q.month}
            <button class="chip-x" data-act="clear-month" aria-label="${t('common.clear_filter')}">${icon('x')}</button></span>`
        : ''}
      ${q.topic
        ? html`<span class="chip chip-on" title="${q.topic}">${icon('tag')}${q.topic}
            <button class="chip-x" data-act="clear-topic" aria-label="${t('common.clear_filter')}">${icon('x')}</button></span>`
        : ''}
      ${[...q.languages].sort().map(
        (language) => html`<span class="chip chip-on">${icon('code')}${language}
            <button class="chip-x" data-act="clear-language" data-id="${language}" aria-label="${t('common.clear_filter')}">${icon('x')}</button></span>`,
      )}
      ${sortedClusters.map(
        ({ cluster, count }) => html`<button class="chip" data-act="cluster" data-id="${cluster.id}" aria-pressed="${q.clusters.has(cluster.id)}" title="${cluster.name}">
          <span class="dot" style="--dot:${cluster.color}"></span><span class="truncate">${cluster.name}</span><span class="chip-count">${count}</span>
        </button>`,
      )}`.value;

    const empty = html`<div class="center-state">
      <h2 class="empty-title">${activeFilters() ? t('data.no_results') : t('data.no_data')}</h2>
      ${activeFilters() ? '' : html`<p class="muted">${t('data.empty_hint')}</p>`}
      ${activeFilters()
        ? html`<button class="btn btn-outline" data-act="clear">${t('common.clear_filters')}</button>`
        : html`<a class="btn btn-primary" href="#/settings">${t('common.sync_now')}</a>`}
    </div>`;

    const repoButton = (repo: GraphNode) =>
      html`<button class="repo-link" data-act="select" data-id="${repo.id}" aria-controls="data-details">${repoAvatar(repo, 20)}<span class="truncate">${repo.full_name}</span></button>`;
    el('[data-region="table"]').innerHTML = html`
      <div class="table-scroll desktop-only">
        <table class="table">
          <thead><tr>
            ${sortHeader(t('data.repository'), 'name', 'col-name')}
            ${sortHeader(t('data.summary'), 'summary')}
            ${sortHeader(t('data.language'), 'language', 'col-language col-opt')}
            ${sortHeader(t('data.stars'), 'stargazers_count', 'num col-stars')}
            ${sortHeader(t('data.cluster'), 'cluster', 'col-cluster')}
            ${sortHeader(t('data.starred_date'), 'starred_at', 'num col-date')}
            ${sortHeader(t('data.last_commit'), 'last_commit_time', 'num col-date col-opt')}
          </tr></thead>
          <tbody>
            ${rows.map(
              (repo) => html`<tr data-row="${repo.id}" class="${repo.id === selectedId ? 'is-selected' : ''}">
                <td>${repoButton(repo)}</td>
                <td><p class="clamp-2 muted" title="${repo.ai_summary || repo.description || ''}">${repo.ai_summary || repo.description || '—'}</p></td>
                <td class="muted truncate col-opt">${languageButton(repo.language, q.languages.has(repo.language ?? ''))}</td>
                <td class="num tabular">${repo.stargazers_count.toLocaleString()}</td>
                <td>${clusterTag(repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined, repo.cluster_id != null && q.clusters.has(repo.cluster_id))}</td>
                <td class="num muted tabular">${monthButton(repo.starred_at, q.month)}</td>
                <td class="num muted tabular col-opt">${formatDate(repo.last_commit_time)}</td>
              </tr>`,
            )}
          </tbody>
        </table>
        ${rows.length ? '' : empty}
      </div>
      <ul class="repo-list mobile-only">
        ${rows.map(
          (repo) => html`<li data-row="${repo.id}" class="${repo.id === selectedId ? 'is-selected' : ''}">
            ${repoButton(repo)}
            <p class="clamp-2 muted small">${repo.ai_summary || repo.description || t('data.no_summary')}</p>
            <div class="repo-meta small muted">
              <span class="stars tabular">${icon('star')}${repo.stargazers_count.toLocaleString()}</span>
              ${languageButton(repo.language, q.languages.has(repo.language ?? ''))}
              ${clusterTag(repo.cluster_id != null ? clusterMap.get(repo.cluster_id) : undefined, repo.cluster_id != null && q.clusters.has(repo.cluster_id))}
              ${monthButton(repo.starred_at, q.month)}
            </div>
          </li>`,
        )}
        ${rows.length ? '' : html`<li>${empty}</li>`}
      </ul>`.value;

    el('[data-region="pager"]').innerHTML = filtered.length
      ? html`<label class="muted small pager-size">${t('data.rows_per_page')}
            <select class="select select-sm" data-page-size>${DATA_PAGE_SIZES.map(
              (size) => html`<option value="${size}" ${size === pageSize ? 'selected' : ''}>${size}</option>`,
            )}</select>
          </label>
          <span class="muted small tabular">${t('data.showing', { start: offset + 1, end: offset + rows.length, total: filtered.length })}</span>
          <span class="pager-buttons">
            <button class="btn btn-ghost btn-icon btn-sm" data-act="prev" ${page === 1 ? 'disabled' : ''} aria-label="${t('data.previous_page')}">${icon('chevronLeft')}</button>
            <button class="btn btn-ghost btn-icon btn-sm" data-act="next" ${page === totalPages ? 'disabled' : ''} aria-label="${t('data.next_page')}">${icon('chevronRight')}</button>
          </span>`.value
      : '';
    restoreControl(restore);
  };

  const refreshDetails = (focus?: { act: string; id: string }) => {
    if (!root.querySelector('.data-page.has-details')) return;
    renderDetailsRegion();
    if (!focus) return;
    [...root.querySelectorAll<HTMLElement>('[data-region="details"] [data-act]')]
      .find((el) => el.dataset.act === focus.act && (el.dataset.id ?? '') === focus.id)
      ?.focus({ preventScroll: true });
  };

  const update = (resetPage = true, rebuildChips = true) => {
    if (resetPage) page = 1;
    syncUrl();
    renderResults(rebuildChips);
  };

  renderFrame();
  syncUrl();
  const cleanups = [
    store.subscribe(['data'], renderFrame),
    onRoute((next) => {
      if (next.path !== '/data') return;
      q.query = next.params.get('q') ?? '';
      q.clusters = parseClusters(next.params);
      q.month = readMonth(next.params);
      q.topic = next.params.get('topic');
      q.languages = new Set(readLanguages(next.params));
      const search = root.querySelector<HTMLInputElement>('[data-search]');
      if (search) search.value = q.query;
      selectedId = parseNode(next.params);
      sort = readSort(next.params);
      page = readPage(next.params);
      pageSize = readPageSize(next.params);
      renderResults();
      renderDetailsRegion();
      syncUrl();
    }),
    onAction(root, 'click', {
      retry: () => void store.load(),
      select: (button) => select(Number(button.dataset.id)),
      'close-details': () => select(null),
      'related-tab': (button) => {
        details.relatedTab = button.dataset.tab as DetailsState['relatedTab'];
        renderDetailsRegion();
        focusRelatedTab(root, details.relatedTab);
      },
      sort: (button) => {
        sort = nextSort(sort, button.dataset.field as SortField);
        update(true, false);
      },
      'sort-dir': () => {
        sort = { ...sort, direction: sort.direction === 'asc' ? 'desc' : 'asc' };
        update(true, false);
      },
      cluster: (button) => {
        const id = Number(button.dataset.id);
        if (!Number.isFinite(id)) return;
        const fromDetails = Boolean(button.closest('[data-region="details"]'));
        q.clusters = new Set(q.clusters);
        if (q.clusters.has(id)) q.clusters.delete(id);
        else q.clusters.add(id);
        update();
        refreshDetails(fromDetails ? { act: 'cluster', id: String(id) } : undefined);
      },
      month: (button) => {
        const month = button.dataset.id ?? '';
        if (!month) return;
        const fromDetails = Boolean(button.closest('[data-region="details"]'));
        q.month = q.month === month ? null : month;
        update();
        refreshDetails(fromDetails ? { act: 'month', id: month } : undefined);
      },
      topic: (button) => {
        const topic = button.dataset.id ?? '';
        if (!topic) return;
        q.topic = q.topic?.trim().toLowerCase() === topic.toLowerCase() ? null : topic;
        update();
        refreshDetails({ act: 'topic', id: topic });
      },
      'clear-month': () => {
        q.month = null;
        update();
        refreshDetails();
      },
      'clear-topic': () => {
        q.topic = null;
        update();
        refreshDetails();
      },
      language: (button) => {
        const language = button.dataset.id ?? '';
        if (!language) return;
        const fromDetails = Boolean(button.closest('[data-region="details"]'));
        q.languages = new Set(q.languages);
        if (q.languages.has(language)) q.languages.delete(language);
        else q.languages.add(language);
        update();
        refreshDetails(fromDetails ? { act: 'language', id: language } : undefined);
      },
      'clear-language': (button) => {
        q.languages = new Set(q.languages);
        q.languages.delete(button.dataset.id ?? '');
        update();
        refreshDetails();
      },
      clear: () => {
        Object.assign(q, { query: '', clusters: new Set(), languages: new Set(), month: null, topic: null });
        (el('[data-search]') as HTMLInputElement).value = '';
        update();
        refreshDetails();
      },
      prev: () => {
        page = Math.max(1, page - 1);
        syncUrl();
        renderResults(false);
      },
      next: () => {
        page += 1;
        syncUrl();
        renderResults(false);
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
      update(true, false);
    } else if (target.matches('[data-sort-field]')) {
      sort = { ...sort, field: target.value as SortField };
      update(true, false);
    }
  };
  /** Arrows move between the repository names on the current page. Enter still opens one. */
  const moveRowFocus = (event: KeyboardEvent): boolean => {
    const key = event.key;
    if (key !== 'ArrowDown' && key !== 'ArrowUp' && key !== 'Home' && key !== 'End') return false;
    if (event.altKey || event.metaKey || event.ctrlKey) return false;
    const active = document.activeElement;
    if (!(active instanceof HTMLElement) || !active.classList.contains('repo-link') || !root.contains(active)) return false;
    const items = [...root.querySelectorAll<HTMLButtonElement>('.repo-link')].filter((row) => row.offsetParent !== null);
    const index = items.indexOf(active as HTMLButtonElement);
    if (index < 0) return false;
    const next =
      key === 'Home' ? 0
      : key === 'End' ? items.length - 1
      : key === 'ArrowDown' ? Math.min(items.length - 1, index + 1)
      : Math.max(0, index - 1);
    event.preventDefault();
    if (next !== index) {
      items[next].focus({ preventScroll: true });
      items[next].scrollIntoView({ block: 'nearest' });
    }
    return true;
  };

  const onKey = (event: KeyboardEvent) => {
    const tab = moveRelatedTab(event);
    if (tab && selectedId != null) {
      details.relatedTab = tab;
      renderDetailsRegion();
      focusRelatedTab(root, tab);
      return;
    }
    if (moveRelatedItem(event, root)) return;
    if (moveRowFocus(event)) return;
    if (event.key !== 'Escape' || selectedId == null || document.querySelector('.palette')) return;
    select(null);
  };
  root.addEventListener('input', onInput);
  document.addEventListener('keydown', onKey);
  root.addEventListener('change', onChange);

  return () => {
    window.clearTimeout(searchTimer);
    cleanups.forEach((fn) => fn());
    root.removeEventListener('input', onInput);
    root.removeEventListener('change', onChange);
    document.removeEventListener('keydown', onKey);
  };
};
