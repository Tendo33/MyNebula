import { useCallback, useEffect, useMemo, useState } from 'react';
import { Code, Tag } from 'lucide-react';
import type { TFunction } from 'i18next';

import type { GraphData, GraphNode } from '../../../types';
import { searchRepos } from '../../../api/repos';
import {
  asRepoSearchCandidate,
  matchesClusterSearch,
  matchesFacetSearch,
  matchesRepoSearch,
  normalizeSearchQuery,
  parseStarsThreshold,
} from '../../../utils/search';
import { logClientWarn } from '../../../utils/debug';
import { buildLanguageFacets, buildTagFacets } from '../commandPaletteFacets';
import {
  MAX_RESULTS,
  type FilterType,
  type RepoSearchResult,
  type SearchResult,
} from '../commandPaletteTypes';

/**
 * Result assembly for the command palette.
 *
 * Local matching uses the shared `utils/search` rules, which the Data page also
 * consumes — the two must never fork. Semantic search is a *fallback*: it only
 * runs when the local pass finds nothing, and never for a `stars:>N` query,
 * which is a numeric filter rather than a semantic one.
 */
export const useCommandPaletteResults = ({
  isOpen,
  rawData,
  query,
  activeFilter,
  t,
}: {
  isOpen: boolean;
  rawData: GraphData | null;
  query: string;
  activeFilter: FilterType;
  t: TFunction;
}) => {
  const [remoteRepoResults, setRemoteRepoResults] = useState<RepoSearchResult[]>([]);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const [remoteError, setRemoteError] = useState<string | null>(null);

  const languages = useMemo(() => buildLanguageFacets(rawData?.nodes), [rawData]);
  const allTags = useMemo(() => buildTagFacets(rawData?.nodes), [rawData]);

  const normalizedQuery = useMemo(() => normalizeSearchQuery(query), [query]);
  const starsThreshold = useMemo(() => parseStarsThreshold(normalizedQuery), [normalizedQuery]);

  const buildRepoResult = useCallback(
    (node: GraphNode, source: 'local' | 'remote'): RepoSearchResult => ({
      type: 'repo',
      id: node.id,
      title: node.full_name,
      subtitle: node.ai_summary || node.description,
      icon: node.owner_avatar_url ? (
        <img
          src={node.owner_avatar_url}
          alt=""
          className="w-6 h-6 rounded"
          loading="lazy"
          decoding="async"
          width={24}
          height={24}
        />
      ) : (
        <div className="w-6 h-6 rounded bg-border-light flex items-center justify-center text-xs dark:bg-dark-border">
          {node.owner?.charAt(0).toUpperCase()}
        </div>
      ),
      meta: `${source === 'remote' ? 'Semantic · ' : ''}⭐ ${node.stargazers_count.toLocaleString()}${
        node.language ? ` · ${node.language}` : ''
      }`,
      data: node,
      source,
    }),
    []
  );

  const localRepoResults = useMemo((): RepoSearchResult[] => {
    if (!rawData || !(activeFilter === 'all' || activeFilter === 'repos') || !normalizedQuery) {
      return [];
    }
    return rawData.nodes
      .filter((node) => matchesRepoSearch(asRepoSearchCandidate(node), normalizedQuery))
      .slice(0, MAX_RESULTS)
      .map((node) => buildRepoResult(node, 'local'));
  }, [activeFilter, buildRepoResult, normalizedQuery, rawData]);

  useEffect(() => {
    if (
      !isOpen ||
      !rawData ||
      !(activeFilter === 'all' || activeFilter === 'repos') ||
      !normalizedQuery ||
      starsThreshold !== null ||
      localRepoResults.length > 0
    ) {
      setRemoteRepoResults([]);
      setRemoteLoading(false);
      setRemoteError(null);
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    setRemoteLoading(true);
    setRemoteError(null);

    const timer = window.setTimeout(async () => {
      try {
        const response = await searchRepos(
          { query: query.trim(), limit: MAX_RESULTS },
          { signal: controller.signal }
        );
        if (cancelled) return;

        const mapped = response.map((item) => {
          const existingNode = rawData.nodes.find((node) => node.id === item.repo.id);
          const repoNode: GraphNode = existingNode ?? {
            id: item.repo.id,
            github_id: item.repo.github_repo_id,
            full_name: item.repo.full_name,
            name: item.repo.name,
            description: item.repo.description,
            language: item.repo.language,
            html_url: item.repo.html_url,
            owner: item.repo.owner,
            owner_avatar_url: undefined,
            x: 0,
            y: 0,
            z: 0,
            cluster_id: item.repo.cluster_id,
            color: '#6B7280',
            size: 1,
            star_list_id: null,
            stargazers_count: item.repo.stargazers_count,
            ai_summary: item.repo.ai_summary,
            topics: item.repo.topics,
          };
          return buildRepoResult(repoNode, 'remote');
        });

        setRemoteRepoResults(mapped);
      } catch (error) {
        if (cancelled) return;
        logClientWarn('Remote semantic repo search failed', error);
        setRemoteError(t('search.remoteFailed', 'Semantic search is temporarily unavailable'));
        setRemoteRepoResults([]);
      } finally {
        if (!cancelled) {
          setRemoteLoading(false);
        }
      }
    }, 250);

    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [
    activeFilter,
    buildRepoResult,
    isOpen,
    localRepoResults.length,
    normalizedQuery,
    query,
    rawData,
    starsThreshold,
    t,
  ]);

  const results = useMemo((): SearchResult[] => {
    if (!rawData) return [];

    const searchResults: SearchResult[] = [];
    const seenRepoIds = new Set<number>();
    if (activeFilter === 'all' || activeFilter === 'repos') {
      for (const repoResult of [...localRepoResults, ...remoteRepoResults]) {
        if (seenRepoIds.has(repoResult.data.id)) continue;
        seenRepoIds.add(repoResult.data.id);
        searchResults.push(repoResult);
      }
    }

    // Search clusters
    if (activeFilter === 'all' || activeFilter === 'clusters') {
      const matchedClusters = rawData.clusters
        .filter((cluster) => {
          if (!normalizedQuery || starsThreshold !== null) return false;
          return matchesClusterSearch(cluster, normalizedQuery);
        })
        .slice(0, 5)
        .map((cluster) => ({
          type: 'cluster' as const,
          id: cluster.id,
          title: cluster.name || `Cluster ${cluster.id}`,
          subtitle: cluster.description,
          icon: <div className="w-6 h-6 rounded-full" style={{ backgroundColor: cluster.color }} />,
          meta: `${cluster.repo_count} repos`,
          data: cluster,
        }));
      searchResults.push(...matchedClusters);
    }

    // Search languages
    if (activeFilter === 'all' || activeFilter === 'languages') {
      const matchedLanguages = languages
        .filter(([lang]) => {
          if (!normalizedQuery || starsThreshold !== null) return false;
          return matchesFacetSearch(lang, normalizedQuery);
        })
        .slice(0, 5)
        .map(([lang, count]) => ({
          type: 'language' as const,
          id: lang,
          title: lang,
          subtitle: `Filter by ${lang} repositories`,
          icon: <Code className="w-5 h-5 text-action-primary" />,
          meta: `${count} repos`,
          data: { language: lang },
        }));
      searchResults.push(...matchedLanguages);
    }

    // Search tags
    if (activeFilter === 'all' || activeFilter === 'tags') {
      const matchedTags = allTags
        .filter(([tag]) => {
          if (!normalizedQuery || starsThreshold !== null) return false;
          return matchesFacetSearch(tag, normalizedQuery);
        })
        .slice(0, 5)
        .map(([tag, count]) => ({
          type: 'tag' as const,
          id: tag,
          title: tag,
          subtitle: `Filter by tag`,
          icon: <Tag className="w-5 h-5 text-action-primary" />,
          meta: `${count} repos`,
          data: { tag },
        }));
      searchResults.push(...matchedTags);
    }

    return searchResults;
  }, [
    activeFilter,
    allTags,
    languages,
    localRepoResults,
    normalizedQuery,
    rawData,
    remoteRepoResults,
    starsThreshold,
  ]);

  const quickFilters = useMemo(() => {
    if (!rawData) return { languages: [], tags: [], starRanges: [] };

    return {
      languages: languages.slice(0, 5),
      tags: allTags.slice(0, 6),
      starRanges: [
        { label: '⭐ 1k+', min: 1000 },
        { label: '⭐ 10k+', min: 10000 },
        { label: '⭐ 50k+', min: 50000 },
      ],
    };
  }, [rawData, languages, allTags]);

  return { results, quickFilters, remoteLoading, remoteError };
};
