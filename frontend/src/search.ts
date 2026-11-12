import { TAG_SYNONYMS } from '../../pipeline/synonyms.ts';
import type { ClusterInfo, GraphNode } from './types';

/**
 * Shared search rules. The graph filter, Data page, and command palette all
 * match through these so the three can never disagree about a query.
 */
const STARS_QUERY_PATTERN = /^stars:\s*>\s*(\d+)$/i;

export const normalizeSearchQuery = (query: string | null | undefined): string =>
  (query ?? '').trim().toLowerCase();

export const parseStarsThreshold = (query: string | null | undefined): number | null => {
  const match = normalizeSearchQuery(query).match(STARS_QUERY_PATTERN);
  return match ? Number.parseInt(match[1], 10) : null;
};

export const buildRepoSearchText = (repo: GraphNode): string => {
  const tags = [...(repo.ai_tags ?? []), ...(repo.topics ?? [])];
  return [
    repo.name,
    repo.full_name,
    repo.description,
    repo.ai_summary,
    repo.language,
    ...tags,
    ...tags.map((tag) => normalizeTag(tag)),
  ]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join('\n')
    .toLowerCase();
};

export const matchesRepoSearch = (
  repo: GraphNode,
  query: string | null | undefined,
  searchableText?: string,
): boolean => {
  const normalized = normalizeSearchQuery(query);
  if (!normalized) return true;
  const stars = parseStarsThreshold(normalized);
  if (stars !== null) return repo.stargazers_count > stars;
  return (searchableText ?? buildRepoSearchText(repo)).includes(normalized);
};

export const matchesClusterSearch = (
  cluster: Pick<ClusterInfo, 'name' | 'description' | 'keywords'>,
  query: string,
): boolean => {
  const normalized = normalizeSearchQuery(query);
  if (!normalized) return true;
  return [cluster.name, cluster.description, ...(cluster.keywords ?? [])]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .some((value) => value.toLowerCase().includes(normalized));
};

export const matchesFacetSearch = (value: string | null | undefined, query: string): boolean => {
  const normalized = normalizeSearchQuery(query);
  return !normalized || (value ?? '').toLowerCase().includes(normalized);
};

const ranked = (counts: Map<string, number>, limit: number): [string, number][] =>
  [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit);

export const buildLanguageFacets = (nodes: GraphNode[], limit = 10): [string, number][] => {
  const counts = new Map<string, number>();
  nodes.forEach((node) => {
    if (node.language) counts.set(node.language, (counts.get(node.language) ?? 0) + 1);
  });
  return ranked(counts, limit);
};

/** Topics first, then AI tags, capped at four. Only GitHub topics can drive the exact topic filter. */
export const detailTopicLabels = (
  topics: string[] = [],
  tags: string[] = [],
): { label: string; filterable: boolean }[] => {
  const topicKeys = new Set(topics.map((value) => value.trim().toLowerCase()).filter(Boolean));
  const seen = new Set<string>();
  const labels: { label: string; filterable: boolean }[] = [];
  for (const raw of [...topics, ...tags]) {
    const label = raw.trim();
    const key = label.toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    labels.push({ label, filterable: topicKeys.has(key) });
    if (labels.length === 4) break;
  }
  return labels;
};

/** AI tags and GitHub topics pooled into one "tag" facet. */
export const buildTagFacets = (nodes: GraphNode[], limit = 20): [string, number][] => {
  const counts = new Map<string, number>();
  nodes.forEach((node) => {
    [...(node.ai_tags ?? []), ...(node.topics ?? [])].forEach((tag) =>
      counts.set(tag, (counts.get(tag) ?? 0) + 1),
    );
  });
  return ranked(counts, limit);
};

/** A palette choice stays on Data. Settings and the graph both open it on the graph. */
export const paletteTarget = (path: string): '/' | '/data' => (path === '/data' ? '/data' : '/');

/** Same cleaning and synonym map the pipeline uses when it scores tag overlap. */
export const normalizeTag = (tag: string): string => {
  const cleaned = tag
    .trim()
    .toLowerCase()
    .replace(/_/g, '-')
    .replace(/[^a-z0-9\-\u4e00-\u9fff]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned ? (TAG_SYNONYMS[cleaned] ?? cleaned) : '';
};
