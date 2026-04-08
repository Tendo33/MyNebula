import type { ClusterInfo, GraphNode } from '../types';

export interface RepoSearchCandidate {
  name?: string | null;
  full_name?: string | null;
  description?: string | null;
  ai_summary?: string | null;
  language?: string | null;
  ai_tags?: string[] | null;
  topics?: string[] | null;
  stargazers_count?: number | null;
}

const STARS_QUERY_PATTERN = /^stars:\s*>\s*(\d+)$/i;

export const normalizeSearchQuery = (query: string | null | undefined): string =>
  (query ?? '').trim().toLowerCase();

export const parseStarsThreshold = (query: string | null | undefined): number | null => {
  const normalized = normalizeSearchQuery(query);
  const match = normalized.match(STARS_QUERY_PATTERN);
  if (!match) {
    return null;
  }
  return Number.parseInt(match[1], 10);
};

export const buildRepoSearchText = (repo: RepoSearchCandidate): string =>
  [
    repo.name,
    repo.full_name,
    repo.description,
    repo.ai_summary,
    repo.language,
    ...(repo.ai_tags ?? []),
    ...(repo.topics ?? []),
  ]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join('\n')
    .toLowerCase();

export const matchesRepoSearch = (
  repo: RepoSearchCandidate,
  query: string | null | undefined,
  searchableText?: string
): boolean => {
  const normalized = normalizeSearchQuery(query);
  if (!normalized) {
    return true;
  }

  const starsThreshold = parseStarsThreshold(normalized);
  if (starsThreshold !== null) {
    return Number(repo.stargazers_count ?? 0) > starsThreshold;
  }

  return (searchableText ?? buildRepoSearchText(repo)).includes(normalized);
};

export const matchesClusterSearch = (
  cluster: Pick<ClusterInfo, 'name' | 'description' | 'keywords'>,
  query: string | null | undefined
): boolean => {
  const normalized = normalizeSearchQuery(query);
  if (!normalized) {
    return true;
  }

  return [
    cluster.name,
    cluster.description,
    ...(cluster.keywords ?? []),
  ]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .some((value) => value.toLowerCase().includes(normalized));
};

export const matchesFacetSearch = (
  value: string | null | undefined,
  query: string | null | undefined
): boolean => {
  const normalized = normalizeSearchQuery(query);
  if (!normalized) {
    return true;
  }
  return (value ?? '').toLowerCase().includes(normalized);
};

export const asRepoSearchCandidate = (node: GraphNode): RepoSearchCandidate => ({
  name: node.name,
  full_name: node.full_name,
  description: node.description,
  ai_summary: node.ai_summary,
  language: node.language,
  ai_tags: node.ai_tags,
  topics: node.topics,
  stargazers_count: node.stargazers_count,
});
