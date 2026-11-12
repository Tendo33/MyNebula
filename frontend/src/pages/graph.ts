import { GraphCanvas } from '../graph/canvas';
import { renderDetails, renderHoverCard, focusRelatedItem, focusRelatedTab, moveRelatedItem, moveRelatedTab, type DetailsState } from '../graph/details';
import { monthsFromTimeRange, readGraphLink, sharedViewParams, timeRangeFromMonths, writeGraphLink } from '../graph/route';
import { attachTimelineBrush, renderTimeline } from '../graph/timeline';
import { facetNodes, hasActiveFilters, timelineFacets } from '../filters';
import { html, onAction, qs } from '../html';
import { t } from '../i18n';
import { icon } from '../icons';
import { currentRoute, navigate, onRoute, type Route } from '../router';
import { store } from '../store';
import { UNCATEGORIZED_LIST_ID } from '../types';
import { emptyState, errorState, loadingState } from './common';

/** Graph state <-> URL: node, cluster(s), lists, language, q, and a month window. */
const applyRoute = (route: Route) => {
  const data = store.data;
  if (!data) return;
  const link = readGraphLink(route.params);
  const knownLists = new Set(data.star_lists.map((list) => list.id));
  const knownLanguages = new Set(data.nodes.flatMap((node) => (node.language ? [node.language] : [])));
  store.setFilters({
    clusters: new Set(link.clusters.filter((id) => data.clusters.some((cluster) => cluster.id === id))),
    lists: new Set(link.lists.filter((id) => knownLists.has(id))),
    languages: new Set(link.languages.filter((language) => knownLanguages.has(language))),
    query: link.query,
    timeRange: timeRangeFromMonths(data.timeline, link.from, link.to),
  });
  store.select(link.nodeId != null && store.nodeById.has(link.nodeId) ? link.nodeId : null);
};

const writeRoute = () => {
  const { filters } = store;
  const months = monthsFromTimeRange(store.data?.timeline ?? [], filters.timeRange);
  navigate(
    '/',
    writeGraphLink({
      nodeId: store.selectedId,
      clusters: [...filters.clusters],
      lists: [...filters.lists],
      languages: [...filters.languages],
      query: filters.query,
      from: months.from,
      to: months.to,
    }),
    true,
  );
};

export const mountGraph = (root: HTMLElement): (() => void) => {
  let canvas: GraphCanvas | null = null;
  let hoverId: number | null = null;
  let previewRange: [number, number] | null = null;
  let showList = window.innerWidth < 1024 || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Below 900px the rail overlays the canvas, so it starts closed there.
  let filtersOpen = window.innerWidth >= 900;
  const details: DetailsState = { relatedTab: 'similar' };
  const cleanups: (() => void)[] = [];

  const el = (selector: string) => qs<HTMLElement>(root, selector);

  const renderFrame = () => {
    canvas?.destroy();
    canvas = null;
    if (store.loading) return void (root.innerHTML = loadingState().value);
    if (store.error) return void (root.innerHTML = errorState(store.error).value);
    if (!store.data || store.data.nodes.length === 0) {
      root.innerHTML = emptyState('network', t('graph.empty_title'), t('graph.empty_hint')).value;
      return;
    }
    root.innerHTML = html`<div class="graph-page ${filtersOpen ? 'filters-open' : ''}">
      <aside class="graph-filters" id="graph-filters" aria-label="${t('common.filter')}">
        <div class="rail-head">
          <label class="search-input">
            ${icon('search')}
            <input type="search" data-search placeholder="${t('graph.search_placeholder')}" aria-label="${t('graph.search_placeholder')}" />
          </label>
          <div class="rail-summary" data-region="summary"></div>
        </div>
        <div class="rail-body" data-region="panels"></div>
      </aside>
      <div class="graph-stage">
        <button class="btn btn-ghost btn-icon stage-filters-toggle" data-act="toggle-filters"
          aria-controls="graph-filters" aria-expanded="${filtersOpen}" aria-label="${t('graph.toggle_filters')}">${icon('panel')}</button>
        <div class="graph-host" data-region="canvas"></div>
        <div data-region="hover"></div>
        <div data-region="list"></div>
      </div>
      <section class="graph-details" data-region="details" aria-label="${t('common.overview')}"></section>
    </div>`.value;

    canvas = new GraphCanvas(el('[data-region="canvas"]'), {
      onSelect: (id) => store.select(id),
      onHover: (id) => {
        hoverId = id;
        renderHover();
      },
    });
    canvas.canvas.setAttribute('role', 'img');
    canvas.canvas.setAttribute('aria-label', t('graph.canvas_label'));
    canvas.setData(store.data.nodes, store.data.edges, store.data.clusters);
    applyRoute(currentRoute());
    const search = el('[data-search]') as HTMLInputElement;
    search.value = store.filters.query;
    renderAll();
    if (store.selectedId != null) canvas.focusNode(store.selectedId, 800, { deferDuringBurst: true });
  };

  const renderSummary = () => {
    const count = store.visibleNodes.length;
    el('[data-region="summary"]').innerHTML = html`
      <span class="muted tabular">${t('graph.showing_repos', { count })}</span>
      ${hasActiveFilters(store.filters) ? html`<button class="link-btn" data-act="clear-all">${t('common.clear_all_filters')}</button>` : ''}`.value;
  };

  /** One flat rail section: a small heading with an inline reset, then rows. */
  const section = (title: string, selected: number, clearAct: string, rows: unknown) =>
    html`<section class="rail-section">
      <div class="rail-section-head">
        <h2>${title}</h2>
        ${selected ? html`<button class="link-btn" data-act="${clearAct}">${t('common.clear')}</button>` : ''}
      </div>
      <div class="rail-rows">${rows}</div>
    </section>`;

  const displayTimeline = () => {
    const data = store.data!;
    const facets = timelineFacets(data.nodes, data.timeline, store.filters);
    return data.timeline.map((point, index) => ({
      ...point,
      count: facets[index]?.count ?? 0,
      top_languages: facets[index]?.topLanguages ?? [],
    }));
  };

  const renderPanels = () => {
    const data = store.data!;
    const { filters } = store;
    const listCounts = new Map<string, number>();
    for (const node of facetNodes(data.nodes, data.timeline, filters, 'lists')) {
      for (const id of node.star_list_ids) listCounts.set(id, (listCounts.get(id) ?? 0) + 1);
    }
    const lists = data.star_lists
      .map((list) => ({ list, count: listCounts.get(list.id) ?? 0 }))
      .filter((item) => item.count > 0 || filters.lists.has(item.list.id))
      .sort((a, b) => b.count - a.count || a.list.name.localeCompare(b.list.name));
    const counts = new Map<number, number>();
    for (const node of facetNodes(data.nodes, data.timeline, filters, 'clusters')) {
      if (node.cluster_id == null) continue;
      counts.set(node.cluster_id, (counts.get(node.cluster_id) ?? 0) + 1);
    }
    for (const id of filters.clusters) if (!counts.has(id)) counts.set(id, 0);
    const clusters = data.clusters
      .map((cluster) => ({ cluster, count: counts.get(cluster.id) ?? 0 }))
      .filter((item) => item.count > 0 || filters.clusters.has(item.cluster.id))
      .sort((a, b) => b.count - a.count);
    const languageCounts = new Map<string, number>();
    for (const node of facetNodes(data.nodes, data.timeline, filters, 'languages')) {
      if (!node.language) continue;
      languageCounts.set(node.language, (languageCounts.get(node.language) ?? 0) + 1);
    }
    for (const language of filters.languages) if (!languageCounts.has(language)) languageCounts.set(language, 0);
    const languages = [...languageCounts].filter(([language, count]) => count > 0 || filters.languages.has(language)).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));

    el('[data-region="panels"]').innerHTML = html`
      ${clusters.length
        ? section(
            t('graph.clusters'),
            filters.clusters.size,
            'clear-clusters',
            clusters.map(
              ({ cluster, count }) => html`<button class="facet" data-act="toggle-cluster" data-id="${cluster.id}"
                aria-pressed="${filters.clusters.has(cluster.id)}" title="${cluster.keywords.join(' · ')}">
                <span class="dot" style="--dot:${cluster.color}"></span>
                <span class="truncate">${cluster.name}</span>
                <span class="facet-count">${count}</span>
              </button>`,
            ),
          )
        : ''}
      ${lists.length
        ? section(
            t('graph.my_lists'),
            filters.lists.size,
            'clear-lists',
            lists.map(
              ({ list, count }) => html`<button class="facet" data-act="toggle-list" data-id="${list.id}"
                aria-pressed="${filters.lists.has(list.id)}" title="${list.description ?? ''}">
                ${icon('folder')}
                <span class="truncate">${list.id === UNCATEGORIZED_LIST_ID ? t('graph.uncategorized') : list.name}</span>
                <span class="facet-count">${count}</span>
              </button>`,
            ),
          )
        : ''}
      ${languages.length
        ? section(
            t('search.languages'),
            filters.languages.size,
            'clear-languages',
            languages.map(
              ([language, count]) => html`<button class="facet" data-act="toggle-language" data-id="${language}"
                aria-pressed="${filters.languages.has(language)}">
                ${icon('code')}
                <span class="truncate">${language}</span>
                <span class="facet-count">${count}</span>
              </button>`,
            ),
          )
        : ''}
      <div data-region="timeline">${renderTimeline(displayTimeline(), previewRange ?? filters.timeRange)}</div>`.value;
  };

  const renderTimelineOnly = () => {
    el('[data-region="timeline"]').innerHTML = renderTimeline(displayTimeline(), previewRange ?? store.filters.timeRange).value;
  };

  /** Ids last written to the list, so a filter change that keeps the same rows skips the rebuild. */
  let listKey = '';

  const renderList = () => {
    const region = el('[data-region="list"]');
    const nodes = store.visibleNodes;
    const key = showList ? nodes.map((node) => node.id).join(',') : 'closed';
    if (key === listKey && region.firstElementChild) return;
    listKey = key;
    const stars = (count: number) => (count >= 1000 ? `${(count / 1000).toFixed(1)}k` : count.toLocaleString());
    region.innerHTML = showList
      ? html`<div class="node-list" id="graph-node-list" aria-label="${t('graph.repository_list')}">
          <div class="node-list-head">
            <h2>${t('graph.repository_list')} <span class="muted tabular">${nodes.length}</span></h2>
            <button class="btn btn-ghost btn-icon btn-sm" data-act="hide-list" aria-label="${t('common.close')}">${icon('x')}</button>
          </div>
          <ul>
            ${nodes.length
              ? nodes.map(
                  (node) => html`<li><button class="node-list-item" data-act="select" data-id="${node.id}"
                    ${store.selectedId === node.id ? html`aria-current="true"` : ''}>
                    <span class="node-list-title">
                      <span class="truncate strong small">${node.full_name}</span>
                      ${node.language ? html`<span class="dim xsmall">${node.language}</span>` : ''}
                      <span class="dim xsmall tabular node-list-stars">${stars(node.stargazers_count)}</span>
                    </span>
                    <span class="truncate muted xsmall">${node.description || node.ai_summary || t('common.no_description')}</span>
                  </button></li>`,
                )
              : html`<li class="node-list-empty muted small">${t('graph.list_empty')}</li>`}
          </ul>
        </div>`.value
      : html`<button class="btn btn-outline btn-sm list-toggle" data-act="show-list" aria-controls="graph-node-list" aria-expanded="false">
          ${icon('list')}${t('graph.browse_as_list')}
        </button>`.value;
  };

  /** Update the current row without rebuilding the list, so scroll and focus stay put. */
  const syncListSelection = (scroll: boolean) => {
    const current = store.selectedId;
    let target: HTMLElement | null = null;
    for (const button of el('[data-region="list"]').querySelectorAll<HTMLButtonElement>('.node-list-item')) {
      const on = Number(button.dataset.id) === current;
      if (on) {
        button.setAttribute('aria-current', 'true');
        target = button;
      } else button.removeAttribute('aria-current');
    }
    if (scroll && target) target.scrollIntoView({ block: 'nearest' });
  };

  const renderDetailsRegion = () => {
    const region = el('[data-region="details"]');
    const node = store.selected;
    root.querySelector('.graph-page')?.classList.toggle('has-details', Boolean(node));
    region.innerHTML = node ? renderDetails(node, { ...details, topic: null, languages: store.filters.languages, lists: store.filters.lists, clusters: store.filters.clusters }).value : '';
  };

  const renderHover = () => {
    const node = hoverId == null ? null : store.nodeById.get(hoverId);
    el('[data-region="hover"]').innerHTML = node ? renderHoverCard(node).value : '';
  };

  const updateCanvas = () => {
    canvas?.setView({
      visible: new Set(store.visibleNodes.map((n) => n.id)),
      selectedId: store.selectedId,
      neighbors: store.selectedId == null ? new Set() : (store.adjacency.get(store.selectedId) ?? new Set()),
      showTrajectories: store.settings.showTrajectories,
      hqRendering: store.settings.hqRendering,
    });
  };

  const renderAll = () => {
    renderSummary();
    renderPanels();
    renderList();
    renderDetailsRegion();
    updateCanvas();
  };

  renderFrame();

  type FocusKey = { kind: 'act'; act: string; id: string } | { kind: 'month'; index: string };

  /** The rail or list control that was focused before a filter rebuild. */
  const focusedGraphControl = (): FocusKey | null => {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement) || !root.contains(active) || active.matches('[data-search]')) return null;
    const bar = active.closest<HTMLElement>('.timeline-bar');
    if (bar?.dataset.index) return { kind: 'month', index: bar.dataset.index };
    const act = active.dataset.act;
    if (!act || active.closest('[data-region="details"]')) return null;
    return { kind: 'act', act, id: active.dataset.id ?? '' };
  };

  const restoreGraphControl = (key: FocusKey | null) => {
    if (!key) return;
    const match =
      key.kind === 'month'
        ? [...root.querySelectorAll<HTMLElement>('.timeline-bar')].find((bar) => bar.dataset.index === key.index)
        : [...root.querySelectorAll<HTMLElement>('[data-act]')].find(
            (button) =>
              button.dataset.act === key.act &&
              (button.dataset.id ?? '') === key.id &&
              !button.closest('[data-region="details"]') &&
              button.offsetParent !== null,
          );
    const target =
      match ??
      (key.kind === 'act' && key.act === 'select'
        ? (root.querySelector<HTMLElement>('.node-list-item') ?? root.querySelector<HTMLElement>('[data-act="hide-list"]'))
        : null);
    if (!target) return;
    target.focus({ preventScroll: true });
    if (target.classList.contains('node-list-item') || target.classList.contains('timeline-bar')) {
      target.scrollIntoView({ block: 'nearest' });
    }
  };

  let filtersBefore = store.filters;
  cleanups.push(
    store.subscribe(['data'], renderFrame),
    store.subscribe(['filters'], () => {
      if (!canvas) return;
      const next = store.filters;
      const queryOnly =
        next.clusters === filtersBefore.clusters &&
        next.lists === filtersBefore.lists &&
        next.languages === filtersBefore.languages &&
        next.timeRange === filtersBefore.timeRange;
      const languagesChanged = next.languages !== filtersBefore.languages;
      const listsChanged = next.lists !== filtersBefore.lists;
      const clustersChanged = next.clusters !== filtersBefore.clusters;
      filtersBefore = next;
      const restore = focusedGraphControl();
      const active = document.activeElement;
      const detailsControl =
        active instanceof HTMLElement &&
        (active.dataset.act === 'language' || active.dataset.act === 'list' || active.dataset.act === 'cluster') &&
        active.closest('[data-region="details"]')
          ? { act: active.dataset.act, id: active.dataset.id ?? '' }
          : null;
      const search = el('[data-search]') as HTMLInputElement;
      if (document.activeElement !== search && search.value !== next.query) search.value = next.query;
      renderSummary();
      if (!queryOnly) renderPanels();
      if (languagesChanged || listsChanged || clustersChanged) {
        renderDetailsRegion();
        if (detailsControl) {
          [...root.querySelectorAll<HTMLElement>('[data-region="details"] [data-act]')]
            .find((button) => button.dataset.act === detailsControl.act && (button.dataset.id ?? '') === detailsControl.id)
            ?.focus({ preventScroll: true });
        }
      }
      renderList();
      syncListSelection(next.query === '' && store.selectedId != null);
      updateCanvas();
      writeRoute();
      restoreGraphControl(restore);
    }),
    store.subscribe(['selection'], () => {
      if (!canvas) return;
      const fromRelated =
        document.activeElement instanceof HTMLElement &&
        document.activeElement.classList.contains('related-item') &&
        root.contains(document.activeElement);
      syncListSelection(true);
      renderDetailsRegion();
      if (fromRelated && store.selectedId != null) focusRelatedItem(root);
      if (store.selectedId != null) canvas.focusNode(store.selectedId);
      updateCanvas();
      writeRoute();
    }),
    store.subscribe(['settings'], () => {
      if (!canvas) return;
      updateCanvas();
      renderDetailsRegion();
    }),
    onRoute((route) => {
      if (route.path !== '/' && route.path !== '/graph') return;
      previewRange = null;
      applyRoute(route);
    }),
  );

  cleanups.push(
    onAction(root, 'click', {
      retry: () => void store.load(),
      'toggle-filters': (button) => {
        filtersOpen = !filtersOpen;
        root.querySelector('.graph-page')?.classList.toggle('filters-open', filtersOpen);
        button.setAttribute('aria-expanded', String(filtersOpen));
      },
      'clear-all': () => {
        store.clearFilters();
        store.select(null);
      },
      'toggle-list': (button) => store.toggleIn('lists', button.dataset.id ?? ''),
      list: (button) => {
        const id = button.dataset.id ?? '';
        if (!id) return;
        store.toggleIn('lists', id);
      },
      'toggle-cluster': (button) => store.toggleIn('clusters', Number(button.dataset.id)),
      cluster: (button) => {
        const id = Number(button.dataset.id);
        if (!Number.isFinite(id)) return;
        store.toggleIn('clusters', id);
      },
      'clear-lists': () => store.setFilters({ lists: new Set() }),
      'clear-clusters': () => store.setFilters({ clusters: new Set() }),
      'clear-languages': () => store.setFilters({ languages: new Set() }),
      'toggle-language': (button) => store.toggleIn('languages', button.dataset.id ?? ''),
      language: (button) => {
        const language = button.dataset.id ?? '';
        if (!language) return;
        store.toggleIn('languages', language);
      },
      'timeline-clear': () => {
        previewRange = null;
        store.setFilters({ timeRange: null });
      },
      'show-list': () => {
        showList = true;
        renderList();
        syncListSelection(true);
        (root.querySelector<HTMLElement>('.node-list-item') ?? root.querySelector<HTMLElement>('[data-act="hide-list"]'))?.focus();
      },
      'hide-list': () => {
        showList = false;
        renderList();
        root.querySelector<HTMLElement>('[data-act="show-list"]')?.focus();
      },
      select: (button) => store.select(Number(button.dataset.id)),
      topic: (button) => {
        const topic = button.dataset.id ?? '';
        if (!topic) return;
        const params = sharedViewParams(currentRoute().params);
        params.set('topic', topic);
        navigate('/data', params);
      },
      'close-details': () => store.select(null),
      'related-tab': (button) => {
        details.relatedTab = button.dataset.tab as DetailsState['relatedTab'];
        renderDetailsRegion();
        focusRelatedTab(root, details.relatedTab);
      },
    }),
  );

  /** Arrows move through the open repository list. Enter on the focused row still selects it. */
  const moveListFocus = (event: KeyboardEvent): boolean => {
    const key = event.key;
    if (key !== 'ArrowDown' && key !== 'ArrowUp' && key !== 'Home' && key !== 'End') return false;
    if (event.altKey || event.metaKey || event.ctrlKey) return false;
    const list = root.querySelector('#graph-node-list');
    const active = document.activeElement;
    if (!(list instanceof HTMLElement) || !(active instanceof HTMLElement) || !list.contains(active)) return false;
    if (active.closest('input, select, textarea')) return false;
    const items = [...list.querySelectorAll<HTMLButtonElement>('.node-list-item')];
    if (!items.length) return false;
    const index = items.indexOf(active as HTMLButtonElement);
    const next =
      key === 'Home' ? 0
      : key === 'End' ? items.length - 1
      : key === 'ArrowDown' ? (index < 0 ? 0 : Math.min(items.length - 1, index + 1))
      : index < 0 ? items.length - 1
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
    if (tab) {
      details.relatedTab = tab;
      renderDetailsRegion();
      focusRelatedTab(root, tab);
      return;
    }
    if (moveRelatedItem(event, root)) return;
    if (moveListFocus(event)) return;
    if (event.key !== 'Escape' || store.selectedId == null || document.querySelector('.palette')) return;
    if ((event.target as HTMLElement).closest('input, select, textarea')) return;
    store.select(null);
  };
  window.addEventListener('keydown', onKey);
  cleanups.push(() => window.removeEventListener('keydown', onKey));

  let searchTimer = 0;
  const onInput = (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (!target.matches('[data-search]')) return;
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => store.setFilters({ query: target.value }), 150);
  };
  root.addEventListener('input', onInput);
  cleanups.push(() => root.removeEventListener('input', onInput));

  cleanups.push(
    attachTimelineBrush(
      root,
      () => store.data?.timeline ?? [],
      () => store.filters.timeRange,
      (range) => {
        previewRange = range;
        renderTimelineOnly();
      },
      (range) => {
        previewRange = null;
        store.setFilters({ timeRange: range });
      },
    ),
  );

  return () => {
    window.clearTimeout(searchTimer);
    cleanups.forEach((fn) => fn());
    canvas?.destroy();
  };
};
