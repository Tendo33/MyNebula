import { GraphCanvas } from '../graph/canvas';
import { renderDetails, renderHoverCard, type DetailsState } from '../graph/details';
import { attachTimelineBrush, renderTimeline } from '../graph/timeline';
import { hasActiveFilters } from '../filters';
import { html, onAction, qs } from '../html';
import { t } from '../i18n';
import { icon } from '../icons';
import { currentRoute, navigate, onRoute, type Route } from '../router';
import { store } from '../store';
import { emptyState, errorState, loadingState } from './common';

const LIST_LIMIT = 50;

const parseIds = (params: URLSearchParams): number[] =>
  (params.get('clusters') ?? params.get('cluster') ?? '')
    .split(',')
    .map((value) => Number.parseInt(value, 10))
    .filter(Number.isFinite);

/** Graph state <-> URL: node, cluster(s), language, q (with legacy `tag`). */
const applyRoute = (route: Route) => {
  const data = store.data;
  if (!data) return;
  const { params } = route;
  const clusters = parseIds(params).filter((id) => data.clusters.some((c) => c.id === id));
  const language = params.get('language');
  store.setFilters({
    clusters: new Set(clusters),
    languages: new Set(language ? [language] : []),
    query: params.get('q') ?? params.get('tag') ?? '',
  });
  const nodeId = Number.parseInt(params.get('node') ?? '', 10);
  store.select(store.nodeById.has(nodeId) ? nodeId : null);
};

const writeRoute = () => {
  const params = new URLSearchParams();
  const { filters } = store;
  if (store.selectedId != null) params.set('node', String(store.selectedId));
  const clusters = [...filters.clusters].sort((a, b) => a - b);
  if (clusters.length === 1) params.set('cluster', String(clusters[0]));
  else if (clusters.length > 1) params.set('clusters', clusters.join(','));
  if (filters.languages.size === 1) params.set('language', [...filters.languages][0]);
  if (filters.query.trim()) params.set('q', filters.query.trim());
  navigate('/', params, true);
};

export const mountGraph = (root: HTMLElement): (() => void) => {
  let canvas: GraphCanvas | null = null;
  let hoverId: number | null = null;
  let previewRange: [number, number] | null = null;
  let showList = window.innerWidth < 1024 || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let filtersOpen = window.innerWidth >= 768;
  const details: DetailsState = { relatedTab: 'similar' };
  const cleanups: (() => void)[] = [];

  const el = (selector: string) => qs<HTMLElement>(root, selector);

  const renderFrame = () => {
    canvas?.destroy();
    canvas = null;
    if (store.loading) return void (root.innerHTML = loadingState().value);
    if (store.error) return void (root.innerHTML = errorState().value);
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

  const renderPanels = () => {
    const data = store.data!;
    const { filters } = store;
    const lists = [...data.star_lists].sort((a, b) => b.repo_count - a.repo_count);
    const counts = new Map<number, number>();
    data.nodes.forEach((n) => n.cluster_id != null && counts.set(n.cluster_id, (counts.get(n.cluster_id) ?? 0) + 1));
    const clusters = data.clusters
      .map((cluster) => ({ cluster, count: counts.get(cluster.id) ?? 0 }))
      .filter((c) => c.count > 0)
      .sort((a, b) => b.count - a.count);

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
              (list) => html`<button class="facet" data-act="toggle-list" data-id="${list.id}"
                aria-pressed="${filters.lists.has(list.id)}" title="${list.description ?? ''}">
                ${icon('folder')}
                <span class="truncate">${list.id === -1 ? t('graph.uncategorized') : list.name}</span>
                <span class="facet-count">${list.repo_count}</span>
              </button>`,
            ),
          )
        : ''}
      <div data-region="timeline">${renderTimeline(data.timeline, previewRange ?? filters.timeRange)}</div>`.value;
  };

  const renderTimelineOnly = () => {
    el('[data-region="timeline"]').innerHTML = renderTimeline(store.data!.timeline, previewRange ?? store.filters.timeRange).value;
  };

  const renderList = () => {
    const region = el('[data-region="list"]');
    const nodes = store.visibleNodes;
    region.innerHTML = showList
      ? html`<div class="node-list" id="graph-node-list" aria-label="${t('graph.repository_list')}">
          <div class="node-list-head">
            <h2>${t('graph.repository_list')} <span class="muted tabular">${nodes.length}</span></h2>
            <button class="btn btn-ghost btn-icon btn-sm" data-act="hide-list" aria-label="${t('common.close')}">${icon('x')}</button>
          </div>
          <ul>
            ${nodes.slice(0, LIST_LIMIT).map(
              (node) => html`<li><button class="node-list-item" data-act="select" data-id="${node.id}"
                ${store.selectedId === node.id ? html`aria-current="true"` : ''}>
                <span class="truncate strong small">${node.full_name}</span>
                <span class="truncate muted xsmall">${node.description || node.ai_summary || t('common.no_description')}</span>
              </button></li>`,
            )}
            ${nodes.length > LIST_LIMIT ? html`<li class="muted xsmall pad">${t('graph.repository_list_limited')}</li>` : ''}
          </ul>
        </div>`.value
      : html`<button class="btn btn-outline btn-sm list-toggle" data-act="show-list" aria-controls="graph-node-list" aria-expanded="false">
          ${icon('list')}${t('graph.browse_as_list')}
        </button>`.value;
  };

  const renderDetailsRegion = () => {
    const region = el('[data-region="details"]');
    const node = store.selected;
    root.querySelector('.graph-page')?.classList.toggle('has-details', Boolean(node));
    region.innerHTML = node ? renderDetails(node, details).value : '';
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

  cleanups.push(
    store.subscribe(['data'], renderFrame),
    store.subscribe(['filters'], () => {
      if (!canvas) return;
      const search = el('[data-search]') as HTMLInputElement;
      if (document.activeElement !== search && search.value !== store.filters.query) search.value = store.filters.query;
      renderSummary();
      renderPanels();
      renderList();
      updateCanvas();
      writeRoute();
    }),
    store.subscribe(['selection'], () => {
      if (!canvas) return;
      renderList();
      renderDetailsRegion();
      if (store.selectedId != null) canvas.focusNode(store.selectedId);
      updateCanvas();
      writeRoute();
    }),
    store.subscribe(['settings'], () => {
      if (!canvas) return;
      updateCanvas();
      renderDetailsRegion();
    }),
    onRoute((route) => (route.path === '/' || route.path === '/graph') && applyRoute(route)),
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
      'toggle-list': (button) => store.toggleIn('lists', Number(button.dataset.id)),
      'toggle-cluster': (button) => store.toggleIn('clusters', Number(button.dataset.id)),
      'clear-lists': () => store.setFilters({ lists: new Set() }),
      'clear-clusters': () => store.setFilters({ clusters: new Set() }),
      'timeline-clear': () => store.setFilters({ timeRange: null }),
      'show-list': () => {
        showList = true;
        renderList();
      },
      'hide-list': () => {
        showList = false;
        renderList();
      },
      select: (button) => store.select(Number(button.dataset.id)),
      'close-details': () => store.select(null),
      'related-tab': (button) => {
        details.relatedTab = button.dataset.tab as DetailsState['relatedTab'];
        renderDetailsRegion();
      },
    }),
  );

  const onKey = (event: KeyboardEvent) => {
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
