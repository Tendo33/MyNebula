import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { GraphNode } from '../../types';
import { X, Code, ExternalLink, Tag, Link2 } from 'lucide-react';
import { useGraph } from '../../contexts/GraphContext';
import { getRelatedRepos } from '../../api/repos';
import { Button, buttonVariants } from '../ui/button';
import { cn } from '@/lib/utils';
import { ToggleGroup, ToggleGroupItem } from '../ui/toggle-group';
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar';
import { Badge } from '../ui/badge';
import { Empty, EmptyDescription, EmptyHeader } from '../ui/empty';
import { ScrollArea } from '../ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';

interface RepoDetailsPanelProps {
  node: GraphNode;
  onClose: () => void;
}

interface RelatedRepoItemProps {
  repo: GraphNode;
  onClick: () => void;
  matchReason?: string;
  score?: number;
}

type RelatedTab = 'similar' | 'sameTags' | 'sameLang';
type RelatedGraphNode = GraphNode & { _score: number; _matchReason: string };
type CachedRelatedItem = { repo: GraphNode; score: number; matchReason: string };
type LocalRelatedRepo = GraphNode | RelatedGraphNode;

const relatedRepoToGraphNode = (repo: {
  id: number;
  github_repo_id: number;
  full_name: string;
  name: string;
  owner: string;
  description?: string;
  language?: string;
  html_url: string;
  stargazers_count: number;
  ai_summary?: string;
  topics: string[];
  cluster_id?: number | null;
  coord_x?: number | null;
  coord_y?: number | null;
  coord_z?: number | null;
}): GraphNode => ({
  id: repo.id,
  github_id: repo.github_repo_id,
  full_name: repo.full_name,
  name: repo.name,
  description: repo.description,
  language: repo.language,
  html_url: repo.html_url,
  owner: repo.owner,
  x: repo.coord_x ?? 0,
  y: repo.coord_y ?? 0,
  z: repo.coord_z ?? 0,
  cluster_id: repo.cluster_id ?? null,
  color: '#8f8f8f',
  size: 1,
  star_list_id: null,
  stargazers_count: repo.stargazers_count,
  ai_summary: repo.ai_summary,
  topics: repo.topics,
});

const summarySentence = (value: string | undefined, fallback: string): string => {
  const source = (value ?? '').replace(/\s+/g, ' ').trim();
  if (!source) return fallback;
  const match = source.match(/^(.+?[.。！？!?])(?:\s|$)/);
  const sentence = (match?.[1] ?? source).trim();
  if (sentence.length <= 180) return sentence;
  return `${sentence.slice(0, 177).trimEnd()}…`;
};

const factTopics = (topics: string[] | undefined, tags: string[] | undefined): string[] => {
  const seen = new Set<string>();
  const labels: string[] = [];
  for (const raw of [...(topics ?? []), ...(tags ?? [])]) {
    const label = raw.trim();
    const key = label.toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    labels.push(label);
    if (labels.length === 4) break;
  }
  return labels;
};

const normalizeTag = (tag: string): string => {
  return tag
    .trim()
    .toLowerCase()
    .replace(/_/g, '-')
    .replace(/[^a-z0-9\-\u4e00-\u9fff]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
};

/** Related repository item component */
const RelatedRepoItem: React.FC<RelatedRepoItemProps> = ({ repo, onClick, matchReason, score }) => {
  const { t } = useTranslation();
  const stars = repo.stargazers_count >= 1000
    ? `${(repo.stargazers_count / 1000).toFixed(1)}k`
    : String(repo.stargazers_count);
  return (
    <button
      onClick={onClick}
      className="flex w-full items-start gap-3 rounded-md px-2 py-2 text-left transition-colors duration-200 hover:bg-bg-hover motion-reduce:transition-none dark:hover:bg-dark-bg-sidebar/70"
    >
      <Avatar size="sm" className="mt-0.5">
        {repo.owner_avatar_url ? <AvatarImage src={repo.owner_avatar_url} alt="" /> : null}
        <AvatarFallback>{repo.owner?.charAt(0).toUpperCase()}</AvatarFallback>
      </Avatar>

      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-baseline gap-2">
          <span className="truncate text-sm font-medium text-text-main">{repo.name}</span>
          {repo.language && (
            <span className="shrink-0 text-xs text-text-dim">{repo.language}</span>
          )}
        </div>
        <p className="mt-0.5 line-clamp-1 text-xs text-text-muted">
          {repo.description || repo.ai_summary || t('data.no_description')}
        </p>
        {(matchReason || score != null) && (
          <p className="mt-0.5 truncate text-xs text-text-dim">
            {matchReason}
            {matchReason && score != null ? ' · ' : ''}
            {score != null ? `${(score * 100).toFixed(1)}%` : ''}
          </p>
        )}
      </div>

      <span className="shrink-0 font-mono text-xs text-text-dim">{stars}</span>
    </button>
  );
};

export const RepoDetailsPanel: React.FC<RepoDetailsPanelProps> = ({ node, onClose }) => {
  const { t } = useTranslation();
  const { rawData, settings, setSelectedNode } = useGraph();
  const [activeTab, setActiveTab] = useState<RelatedTab>('similar');
  const [remoteSimilar, setRemoteSimilar] = useState<RelatedGraphNode[]>([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState<string | null>(null);
  const similarCacheRef = useRef<Map<string, CachedRelatedItem[]>>(new Map());
  const similarInFlightRef = useRef<Map<string, Promise<CachedRelatedItem[]>>>(new Map());
  const graphCacheVersion = rawData?.request_id ?? rawData?.version ?? 'no-graph-data';

  useEffect(() => {
    similarCacheRef.current.clear();
    similarInFlightRef.current.clear();
  }, [graphCacheVersion]);

  useEffect(() => {
    let disposed = false;
    if (!rawData) {
      setRemoteSimilar([]);
      setSimilarLoading(false);
      setSimilarError(null);
      return;
    }

    if (activeTab !== 'similar') {
      setSimilarLoading(false);
      setSimilarError(null);
      return;
    }

    const cacheKey = `${graphCacheVersion}:${node.id}:${settings.relatedMinSemantic.toFixed(3)}`;
    const byId = new Map(rawData.nodes.map((n) => [n.id, n]));
    const mapCachedItems = (items: CachedRelatedItem[]): RelatedGraphNode[] => {
      const mapped: RelatedGraphNode[] = [];
      for (const item of items) {
        const inGraph = byId.get(item.repo.id);
        const repo = inGraph ?? item.repo;
        mapped.push({
          ...repo,
          _score: item.score,
          _matchReason: item.matchReason,
        });
      }
      return mapped;
    };

    const cached = similarCacheRef.current.get(cacheKey);
    if (cached) {
      setRemoteSimilar(mapCachedItems(cached));
      setSimilarLoading(false);
      setSimilarError(null);
      return;
    }

    const load = async () => {
      try {
        setSimilarLoading(true);
        setSimilarError(null);
        setRemoteSimilar([]);

        let request = similarInFlightRef.current.get(cacheKey);
        if (!request) {
          request = getRelatedRepos(node.id, {
            limit: 20,
            min_score: 0.4,
            min_semantic: settings.relatedMinSemantic,
          }).then((items) => {
            const compact = items.map((item) => ({
              repo: relatedRepoToGraphNode(item.repo),
              score: item.score,
              matchReason: item.reasons.join(' · '),
            }));
            similarCacheRef.current.set(cacheKey, compact);
            return compact;
          }).finally(() => {
            similarInFlightRef.current.delete(cacheKey);
          });
          similarInFlightRef.current.set(cacheKey, request);
        }

        const items = await request;
        if (disposed) return;
        setRemoteSimilar(mapCachedItems(items));
      } catch (err) {
        if (!disposed) {
          setSimilarError(err instanceof Error ? err.message : 'Failed to load related repos');
          setRemoteSimilar([]);
        }
      } finally {
        if (!disposed) {
          setSimilarLoading(false);
        }
      }
    };

    load();
    return () => {
      disposed = true;
    };
  }, [activeTab, graphCacheVersion, node.id, rawData, settings.relatedMinSemantic]);


  // Get related repos by different dimensions
  const relatedReposByDimension = useMemo(() => {
    if (!rawData) return { similar: [], sameTags: [], sameLang: [] };

    // 1. Semantically similar (from backend related ranking API)
    const similar = remoteSimilar;

    // 2. Same tags (ai_tags or topics overlap)
    const nodeTags = new Set(
      [...(node.ai_tags || []), ...(node.topics || [])]
        .map(normalizeTag)
        .filter(Boolean)
    );
    const sameTags = nodeTags.size > 0
      ? rawData.nodes
          .filter(n => {
            if (n.id === node.id) return false;
            const nTags = new Set(
              [...(n.ai_tags || []), ...(n.topics || [])]
                .map(normalizeTag)
                .filter(Boolean)
            );
            const overlap = [...nodeTags].filter(t => nTags.has(t));
            const overlapRatio = overlap.length / Math.max(Math.min(nodeTags.size, nTags.size), 1);
            return overlap.length >= 1 && overlapRatio >= 0.25;
          })
          .map(n => {
            const nTags = new Set(
              [...(n.ai_tags || []), ...(n.topics || [])]
                .map(normalizeTag)
                .filter(Boolean)
            );
            const overlap = [...nodeTags].filter(t => nTags.has(t));
            return { ...n, _matchReason: overlap.slice(0, 2).join(', ') };
          })
          .sort((a, b) => b.stargazers_count - a.stargazers_count)
          .slice(0, 10)
      : [];


    // 4. Same language
    const sameLang = node.language
      ? rawData.nodes
          .filter(n => n.id !== node.id && n.language === node.language)
          .sort((a, b) => b.stargazers_count - a.stargazers_count)
          .slice(0, 10)
      : [];

    return { similar, sameTags, sameLang };
  }, [rawData, node, remoteSimilar]);

  // Get current tab's repos
  const currentRelatedRepos = relatedReposByDimension[activeTab] || [];

  // Handle clicking on a related repo
  const handleRelatedRepoClick = (repo: GraphNode) => {
    setSelectedNode(repo);
  };



  const tabCounts = {
    similar: relatedReposByDimension.similar.length,
    sameTags: relatedReposByDimension.sameTags.length,
    sameLang: relatedReposByDimension.sameLang.length,
  };

  const summary = summarySentence(
    node.ai_summary || node.description,
    t('repoDetails.no_description')
  );
  const topics = factTopics(node.topics, node.ai_tags);
  const starCount = node.stargazers_count?.toLocaleString() ?? '0';
  const factLine = [
    `${starCount} ${t('repoDetails.stars')}`,
    node.language,
    ...topics,
    node.star_list_name,
  ].filter(Boolean).join(' · ');
  const ownerInitial = node.owner?.charAt(0).toUpperCase() || '?';
  const fullSummary = (node.ai_summary || node.description || '').trim();
  const relatedEmptyMessage =
    activeTab === 'similar'
      ? similarError
        ? t('repoDetails.similarLoadFailed')
        : t('repoDetails.noSimilar')
      : activeTab === 'sameTags'
        ? t('repoDetails.noSameTags')
        : t('repoDetails.noSameLang');

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="flex items-start gap-3 border-b px-4 py-4">
        <Avatar>
          {node.owner_avatar_url ? <AvatarImage src={node.owner_avatar_url} alt="" /> : null}
          <AvatarFallback>{ownerInitial}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <a
            href={node.html_url}
            target="_blank"
            rel="noopener noreferrer"
            className="block hover:underline"
          >
            <h2 className="truncate font-heading text-xl font-semibold tracking-tight" title={node.full_name}>
              {node.name}
            </h2>
          </a>
          <p className="mt-1 text-sm text-muted-foreground">{summary}</p>
          <p className="mt-1 truncate text-sm text-muted-foreground">{factLine}</p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          onClick={onClose}
          aria-label={t('repoDetails.close')}
        >
          <X />
        </Button>
      </div>

      <div className="flex flex-wrap gap-2 border-b px-4 py-3">
        <a
          href={node.html_url}
          target="_blank"
          rel="noopener noreferrer"
          className={cn(buttonVariants({ variant: 'outline', size: 'sm' }))}
        >
          <ExternalLink data-icon="inline-start" />
          GitHub
        </a>
        <a
          href={`https://deepwiki.com/${node.full_name}`}
          target="_blank"
          rel="noopener noreferrer"
          title="View on DeepWiki"
          className={cn(buttonVariants({ variant: 'outline', size: 'sm' }))}
        >
          <img
            src="https://deepwiki.com/favicon.ico"
            alt=""
            className="size-3.5 rounded-sm"
            loading="lazy"
            decoding="async"
            width={14}
            height={14}
          />
          DeepWiki
        </a>
        <a
          href={`https://zread.ai/${node.full_name}`}
          target="_blank"
          rel="noopener noreferrer"
          title="View on zRead"
          className={cn(buttonVariants({ variant: 'outline', size: 'sm' }))}
        >
          <img
            src="https://zread.ai/favicon.ico"
            alt=""
            className="size-3.5 rounded-sm"
            loading="lazy"
            decoding="async"
            width={14}
            height={14}
          />
          zRead
        </a>
      </div>

      <Tabs defaultValue="overview" className="min-h-0 flex-1">
        <div className="px-4 pt-3">
          <TabsList>
            <TabsTrigger value="overview">{t('common.overview')}</TabsTrigger>
            <TabsTrigger value="related">{t('repoDetails.relatedRepos')}</TabsTrigger>
          </TabsList>
        </div>
        <ScrollArea className="min-h-0 flex-1">
          <TabsContent value="overview" className="flex flex-col gap-3 px-4 py-3">
            <p className="text-sm text-muted-foreground">
              {fullSummary || t('repoDetails.no_ai_summary')}
            </p>
            {topics.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {topics.map((topic) => (
                  <Badge key={topic} variant="secondary">{topic}</Badge>
                ))}
              </div>
            ) : null}
          </TabsContent>
          <TabsContent value="related" className="flex flex-col gap-3 px-4 py-3">
            <ToggleGroup
              value={[activeTab]}
              onValueChange={(next) => {
                const value = next[0];
                if (value) setActiveTab(value as RelatedTab);
              }}
              className="w-full"
            >
              <ToggleGroupItem value="similar" size="sm" className="flex-1">
                <Link2 data-icon="inline-start" />
                {t('repoDetails.similar')}
                {tabCounts.similar > 0 ? ` (${tabCounts.similar})` : ''}
              </ToggleGroupItem>
              <ToggleGroupItem value="sameTags" size="sm" className="flex-1">
                <Tag data-icon="inline-start" />
                {t('repoDetails.sameTags')}
                {tabCounts.sameTags > 0 ? ` (${tabCounts.sameTags})` : ''}
              </ToggleGroupItem>
              <ToggleGroupItem value="sameLang" size="sm" className="flex-1">
                <Code data-icon="inline-start" />
                {t('repoDetails.sameLang')}
                {tabCounts.sameLang > 0 ? ` (${tabCounts.sameLang})` : ''}
              </ToggleGroupItem>
            </ToggleGroup>

            {currentRelatedRepos.length > 0 ? (
              <div className="flex flex-col">
                {currentRelatedRepos.map((repo: LocalRelatedRepo) => (
                  <RelatedRepoItem
                    key={repo.id}
                    repo={repo}
                    onClick={() => handleRelatedRepoClick(repo)}
                    matchReason={'_matchReason' in repo ? repo._matchReason : undefined}
                    score={'_score' in repo ? repo._score : undefined}
                  />
                ))}
              </div>
            ) : activeTab === 'similar' && similarLoading ? (
              <p className="px-1 py-6 text-sm text-muted-foreground">{t('common.loading')}</p>
            ) : (
              <Empty className="border-0">
                <EmptyHeader>
                  <EmptyDescription>{relatedEmptyMessage}</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
};
