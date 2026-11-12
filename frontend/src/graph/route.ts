/**
 * Graph filters that survive a shared link.
 *
 * List ids are the GitHub list ids exported by the pipeline. The timeline is
 * stored as months (`YYYY-MM`), not indexes, because a later sync can insert
 * a month and shift every index.
 */

export interface GraphLink {
  nodeId: number | null;
  clusters: number[];
  lists: string[];
  languages: string[];
  query: string;
  from: string | null;
  to: string | null;
}

const MONTH = /^\d{4}-\d{2}$/;

const month = (value: string | null): string | null => (value && MONTH.test(value) ? value : null);

const ids = (params: URLSearchParams, plural: string, singular: string): string[] =>
  (params.get(plural) ?? params.get(singular) ?? '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

export const readGraphLink = (params: URLSearchParams): GraphLink => {
  const nodeId = Number.parseInt(params.get('node') ?? '', 10);
  return {
    nodeId: Number.isFinite(nodeId) ? nodeId : null,
    clusters: ids(params, 'clusters', 'cluster')
      .map((value) => Number.parseInt(value, 10))
      .filter(Number.isFinite),
    lists: ids(params, 'lists', 'list'),
    languages: ids(params, 'languages', 'language'),
    query: params.get('q') ?? params.get('tag') ?? '',
    from: month(params.get('from')),
    to: month(params.get('to')),
  };
};

/**
 * Filters both Graph and Data understand. List membership and the timeline
 * stay on the graph; month, topic, sort, and paging stay on the data page.
 */
const SHARED_VIEW_KEYS = ['q', 'node', 'cluster', 'clusters', 'language', 'languages'] as const;

export const sharedViewParams = (params: URLSearchParams): URLSearchParams => {
  const next = new URLSearchParams();
  for (const key of SHARED_VIEW_KEYS) {
    const value = key === 'q' ? params.get('q') || params.get('tag') : params.get(key);
    if (value) next.set(key, value);
  }
  return next;
};

export const writeGraphLink = (link: GraphLink): URLSearchParams => {
  const params = new URLSearchParams();
  if (link.nodeId != null) params.set('node', String(link.nodeId));
  const clusters = [...new Set(link.clusters)].sort((a, b) => a - b);
  if (clusters.length === 1) params.set('cluster', String(clusters[0]));
  else if (clusters.length > 1) params.set('clusters', clusters.join(','));
  const lists = [...new Set(link.lists)].sort();
  if (lists.length) params.set('lists', lists.join(','));
  const languages = [...new Set(link.languages)].sort();
  if (languages.length === 1) params.set('language', languages[0]);
  else if (languages.length > 1) params.set('languages', languages.join(','));
  if (link.query.trim()) params.set('q', link.query.trim());
  if (link.from) params.set('from', link.from);
  if (link.to) params.set('to', link.to);
  return params;
};

/** Inclusive month window as timeline indexes. A window covering every point is no filter. */
export const timeRangeFromMonths = (
  timeline: { date: string }[],
  from: string | null,
  to: string | null,
): [number, number] | null => {
  if (!from && !to) return null;
  const start = from ?? to!;
  const end = to ?? from!;
  const [lo, hi] = start <= end ? [start, end] : [end, start];
  let first = -1;
  let last = -1;
  timeline.forEach((point, index) => {
    if (point.date >= lo && point.date <= hi) {
      if (first < 0) first = index;
      last = index;
    }
  });
  if (first < 0 || (first === 0 && last === timeline.length - 1)) return null;
  return [first, last];
};

export const monthsFromTimeRange = (
  timeline: { date: string }[],
  range: [number, number] | null,
): { from: string | null; to: string | null } => {
  if (!range) return { from: null, to: null };
  return {
    from: timeline[range[0]]?.date ?? null,
    to: timeline[range[1]]?.date ?? null,
  };
};
