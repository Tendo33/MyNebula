import type { GraphNode, GraphPayload } from './types';
import { buildAdjacency, emptyFilters, filterNodes, type Filters } from './filters';
import { buildRepoSearchText } from './search';

export interface Settings {
  showTrajectories: boolean;
  hqRendering: boolean;
}

export type Topic = 'data' | 'filters' | 'selection' | 'settings';

const SETTINGS_KEY = 'nebula_graph_settings';
const DEFAULT_SETTINGS: Settings = {
  showTrajectories: true,
  hqRendering: true,
};

const loadSettings = (): Settings => {
  try {
    const stored = localStorage.getItem(SETTINGS_KEY);
    return stored ? { ...DEFAULT_SETTINGS, ...JSON.parse(stored) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
};

const dataUrl = (file: string) => `${import.meta.env.BASE_URL}data/${file}`;

/**
 * App state: the loaded snapshot, filters, selection, and display settings.
 *
 * Subscribers register for topics so a selection change does not re-render the
 * Data table, and a filter change does not rebuild graph layout.
 */
class Store {
  data: GraphPayload | null = null;
  loading = true;
  error: string | null = null;
  selectedId: number | null = null;
  filters: Filters = emptyFilters();
  settings: Settings = loadSettings();

  nodeById = new Map<number, GraphNode>();
  adjacency = new Map<number, Set<number>>();
  private searchText = new Map<number, string>();
  private visibleCache: { filters: Filters; data: GraphPayload; nodes: GraphNode[] } | null = null;
  private listeners = new Map<Topic, Set<() => void>>();

  subscribe(topics: Topic[], fn: () => void): () => void {
    topics.forEach((topic) => {
      if (!this.listeners.has(topic)) this.listeners.set(topic, new Set());
      this.listeners.get(topic)!.add(fn);
    });
    return () => topics.forEach((topic) => this.listeners.get(topic)?.delete(fn));
  }

  private emit(topic: Topic): void {
    this.listeners.get(topic)?.forEach((fn) => fn());
  }

  async load(): Promise<void> {
    this.loading = true;
    this.error = null;
    this.emit('data');
    try {
      const response = await fetch(dataUrl('graph.json'), { cache: 'no-cache' });
      // No export yet. Hosts with an SPA fallback answer 200 with index.html.
      if (response.status === 404 || !response.headers.get('content-type')?.includes('json')) {
        this.setData(null);
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.setData((await response.json()) as GraphPayload);
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
      this.loading = false;
      this.emit('data');
    }
  }

  private setData(data: GraphPayload | null): void {
    this.data = data;
    this.loading = false;
    this.nodeById = new Map(data?.nodes.map((node) => [node.id, node]) ?? []);
    this.adjacency = buildAdjacency(data?.edges ?? []);
    this.searchText = new Map(data?.nodes.map((node) => [node.id, buildRepoSearchText(node)]) ?? []);
    this.visibleCache = null;
    this.emit('data');
  }

  get nodes(): GraphNode[] {
    return this.data?.nodes ?? [];
  }

  get visibleNodes(): GraphNode[] {
    if (!this.data) return [];
    const cache = this.visibleCache;
    if (cache && cache.filters === this.filters && cache.data === this.data) return cache.nodes;
    const nodes = filterNodes(this.data.nodes, this.data.timeline, this.filters, this.searchText);
    this.visibleCache = { filters: this.filters, data: this.data, nodes };
    return nodes;
  }

  get selected(): GraphNode | null {
    return this.selectedId == null ? null : (this.nodeById.get(this.selectedId) ?? null);
  }

  select(id: number | null): void {
    if (id === this.selectedId) return;
    this.selectedId = id;
    this.emit('selection');
  }

  /** Filters are replaced, never mutated, so identity doubles as a change marker. */
  setFilters(patch: Partial<Filters>): void {
    this.filters = { ...this.filters, ...patch };
    this.emit('filters');
  }

  toggleIn<K extends 'clusters' | 'lists' | 'languages'>(key: K, value: Filters[K] extends Set<infer V> ? V : never): void {
    const next = new Set(this.filters[key] as Set<unknown>);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    this.setFilters({ [key]: next } as Partial<Filters>);
  }

  clearFilters(): void {
    this.filters = emptyFilters();
    this.emit('filters');
  }

  updateSettings(patch: Partial<Settings>): void {
    this.settings = { ...this.settings, ...patch };
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings));
    } catch {
      // Settings still apply for this session.
    }
    this.emit('settings');
  }
}

export const store = new Store();
