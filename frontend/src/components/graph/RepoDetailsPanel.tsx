import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { GraphNode } from '../../types';
import { X, Star, Code, ExternalLink, Sparkles, Tag, FolderHeart, Link2, ChevronRight } from 'lucide-react';
import { useGraph } from '../../contexts/GraphContext';
import { clsx } from 'clsx';
import { getRelatedRepos } from '../../api/repos';

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
type CachedRelatedItem = { repoId: number; score: number; matchReason: string };
type LocalRelatedRepo = GraphNode | RelatedGraphNode;

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
  return (
    <button
      onClick={onClick}
      className="w-full flex items-start gap-3 p-3 rounded-xl border border-transparent hover:bg-bg-hover hover:border-border-light/70 hover:shadow-sm transition-all text-left group dark:hover:bg-dark-bg-sidebar/70"
    >
      {/* Avatar */}
            {repo.owner_avatar_url ? (
        <img
          src={repo.owner_avatar_url}
          alt={repo.owner}
          className="w-9 h-9 rounded-lg border border-border-light flex-shrink-0"
          loading="lazy"
          decoding="async"
          width={36}
          height={36}
        />
      ) : (
        <div className="w-9 h-9 rounded-lg bg-border-light flex items-center justify-center flex-shrink-0 dark:bg-dark-border">
          <span className="text-text-dim text-xs font-medium dark:text-dark-text-main/60">
            {repo.owner?.charAt(0).toUpperCase()}
          </span>
        </div>
      )}

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-semibold text-text-main truncate">{repo.name}</span>
          {repo.language && (
            <span className="text-[10px] px-1.5 py-0.5 bg-bg-hover rounded-full text-text-muted flex-shrink-0 dark:bg-dark-bg-sidebar dark:text-dark-text-main/70">
              {repo.language}
            </span>
          )}
        </div>
        <p className="text-xs text-text-muted line-clamp-1 mt-0.5">
          {repo.description || repo.ai_summary || t('data.no_description')}
        </p>
        {(matchReason || score != null) && (
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {matchReason && (
              <span className="max-w-full text-[10px] px-1.5 py-0.5 rounded-full bg-action-primary/10 text-action-primary line-clamp-1">
                {matchReason}
              </span>
            )}
            {score != null && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-sky-50 text-sky-700">
                {(score * 100).toFixed(1)}%
              </span>
            )}
          </div>
        )}
      </div>

      {/* Stars + Arrow */}
      <div className="flex flex-col items-end justify-between h-full gap-2 flex-shrink-0 pl-1">
        <div className="flex items-center gap-1 text-xs text-action-primary bg-action-primary/10 px-1.5 py-0.5 rounded-full">
          <Star className="w-3 h-3 text-action-primary" fill="currentColor" />
          <span>{repo.stargazers_count >= 1000 ? `${(repo.stargazers_count / 1000).toFixed(1)}k` : repo.stargazers_count}</span>
        </div>
        <ChevronRight className="w-4 h-4 text-text-dim opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
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

    const cacheKey = `${node.id}:${settings.relatedMinSemantic.toFixed(3)}`;
    const byId = new Map(rawData.nodes.map((n) => [n.id, n]));
    const mapCachedItems = (items: CachedRelatedItem[]): RelatedGraphNode[] => {
      const mapped: RelatedGraphNode[] = [];
      for (const item of items) {
        const inGraph = byId.get(item.repoId);
        if (!inGraph) continue;
        mapped.push({
          ...inGraph,
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
              repoId: item.repo.id,
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
  }, [activeTab, node.id, rawData, settings.relatedMinSemantic]);


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

  return (
    <div className="absolute inset-y-0 right-0 w-full max-w-full sm:static sm:w-[25rem] h-full bg-bg-main/95 backdrop-blur-sm border-l border-border-light shadow-xl overflow-hidden animate-in fade-in slide-in-from-right-4 duration-300 flex flex-col flex-shrink-0 z-20 dark:bg-dark-bg-main/95 dark:border-dark-border">
      {/* Header with Avatar */}
      <div className="relative p-5 border-b border-border-light bg-gradient-to-b from-bg-sidebar to-bg-main dark:from-dark-bg-sidebar dark:to-dark-bg-main dark:border-dark-border">
        <div className="flex items-start gap-3 pr-8">
            {/* Owner Avatar */}
            {node.owner_avatar_url ? (
              <img
                src={node.owner_avatar_url}
                alt={node.owner}
                className="w-10 h-10 rounded-lg border border-border-light flex-shrink-0"
                loading="lazy"
                decoding="async"
                width={40}
                height={40}
              />
            ) : (
              <div className="w-10 h-10 rounded-lg bg-border-light flex items-center justify-center flex-shrink-0 dark:bg-dark-border">
                <span className="text-text-dim text-sm font-medium dark:text-dark-text-main/60">
                  {node.owner?.charAt(0).toUpperCase()}
                </span>
              </div>
            )}

            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                    <a href={node.html_url} target="_blank" rel="noopener noreferrer" className="hover:underline decoration-text-muted underline-offset-4">
                        <h2 className="text-base font-semibold text-text-main line-clamp-1 leading-snug dark:text-dark-text-main" title={node.full_name}>
                        {node.name}
                        </h2>
                    </a>
                </div>
                <p className="text-xs text-text-dim dark:text-dark-text-main/60">{node.owner}</p>
                <p className="text-sm text-text-muted leading-relaxed mt-1 dark:text-dark-text-main/70">
                {node.description || t('repoDetails.no_description')}
                </p>
            </div>
        </div>
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-md text-text-dim hover:bg-bg-hover hover:text-text-main transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Content */}
      <div className="p-5 flex flex-col gap-4 flex-1 min-h-0 overflow-hidden">

        <div className="flex flex-col gap-4 flex-[3] min-h-0 overflow-y-auto overscroll-contain pr-1 pb-2">
          {/* Quick Actions */}
          <div className="flex items-center gap-2">
            <a
              href={node.html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-text-main hover:bg-text-main/90 text-bg-main text-sm font-medium rounded-lg transition-all shadow-sm hover:shadow-md"
            >
              <ExternalLink className="w-4 h-4" />
              <span>GitHub</span>
            </a>

            {/* Deep Wiki */}
            <a
              href={`https://deepwiki.com/${node.full_name}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 px-3 py-2 bg-bg-hover hover:bg-bg-sidebar text-text-main text-sm font-medium rounded-lg transition-all shadow-sm hover:shadow-md border border-border-light dark:bg-dark-bg-sidebar/70 dark:hover:bg-dark-bg-sidebar dark:text-dark-text-main dark:border-dark-border"
              title="View on DeepWiki"
            >
              <img
                src="https://deepwiki.com/favicon.ico"
                alt="DeepWiki"
                className="w-4 h-4 rounded-sm bg-bg-main"
                loading="lazy"
                decoding="async"
                width={16}
                height={16}
              />
              <span>DeepWiki</span>
            </a>

            {/* zRead */}
            <a
              href={`https://zread.ai/${node.full_name}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 px-3 py-2 bg-bg-hover hover:bg-bg-sidebar text-text-main text-sm font-medium rounded-lg transition-all shadow-sm hover:shadow-md border border-border-light dark:bg-dark-bg-sidebar/70 dark:hover:bg-dark-bg-sidebar dark:text-dark-text-main dark:border-dark-border"
              title="View on zRead"
            >
              <img
                src="https://zread.ai/favicon.ico"
                alt="zRead"
                className="w-4 h-4 rounded-sm bg-bg-main"
                loading="lazy"
                decoding="async"
                width={16}
                height={16}
              />
              <span>zRead</span>
            </a>
          </div>

          {/* User's Star List Badge */}
          {node.star_list_name && (
            <div className="flex items-center gap-2">
              <FolderHeart className="w-4 h-4 text-action-primary" />
              <span className="text-xs font-medium text-action-primary bg-action-primary/10 px-2 py-1 rounded-full">
                {node.star_list_name}
              </span>
            </div>
          )}

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center gap-3 p-3 rounded-lg bg-bg-main border border-border-light shadow-sm dark:bg-dark-bg-main dark:border-dark-border">
                  <div className="p-1.5 bg-action-primary/10 rounded text-action-primary">
                      <Star className="w-4 h-4" fill="currentColor" />
                  </div>
                  <div>
                      <span className="block text-lg font-semibold text-text-main leading-none dark:text-dark-text-main">{node.stargazers_count?.toLocaleString() ?? 0}</span>
                      <span className="text-xs text-text-muted capitalize dark:text-dark-text-main/70">{t('repoDetails.stars')}</span>
                  </div>
              </div>
              <div className="flex items-center gap-3 p-3 rounded-lg bg-bg-main border border-border-light shadow-sm dark:bg-dark-bg-main dark:border-dark-border">
                   <div className="p-1.5 bg-action-primary/10 rounded text-action-primary">
                      <Code className="w-4 h-4" />
                  </div>
                  <div>
                      <span className="block text-lg font-semibold text-text-main leading-none truncate max-w-[100px] dark:text-dark-text-main" title={node.language || 'Unknown'}>{node.language || 'N/A'}</span>
                      <span className="text-xs text-text-muted capitalize dark:text-dark-text-main/70">{t('repoDetails.language')}</span>
                  </div>
              </div>
          </div>

          {/* AI Tags */}
          {node.ai_tags && node.ai_tags.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-action-primary uppercase tracking-wider">
                  <Tag className="w-3.5 h-3.5" />
                  <span>{t('repoDetails.aiTags', 'AI Tags')}</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {node.ai_tags.map((tag, idx) => (
                  <span
                    key={idx}
                    className="text-xs px-2 py-1 bg-action-primary/10 text-action-primary rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* GitHub Topics */}
          {node.topics && node.topics.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider dark:text-dark-text-main/70">
                  {t('repoDetails.topics', 'Topics')}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {node.topics.slice(0, 8).map((topic, idx) => (
                  <span
                    key={idx}
                    className="text-xs px-2 py-1 bg-bg-hover text-text-muted rounded-full dark:bg-dark-bg-sidebar dark:text-dark-text-main/70"
                  >
                    {topic}
                  </span>
                ))}
                {node.topics.length > 8 && (
                  <span className="text-xs text-text-dim dark:text-dark-text-main/60">+{node.topics.length - 8}</span>
                )}
              </div>
            </div>
          )}

          {/* AI Summary */}
          <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-action-primary uppercase tracking-wider">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{t('repoDetails.aiInsight', 'AI Insight')}</span>
              </div>
              <div className="bg-bg-sidebar p-3 rounded-lg border border-border-light/50 dark:bg-dark-bg-sidebar dark:border-dark-border">
                   <p className="text-sm text-text-main leading-relaxed line-clamp-5 sm:line-clamp-6 dark:text-dark-text-main">
                      {node.ai_summary || t('repoDetails.no_ai_summary')}
                  </p>
              </div>
          </div>
        </div>

        {/* Related Repositories with Tabs */}
        <div className="flex flex-col flex-[2] min-h-[200px] gap-2 pt-3 border-t border-border-light/60 dark:border-dark-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-action-primary uppercase tracking-wider">
              <Link2 className="w-3.5 h-3.5" />
              <span>{t('repoDetails.relatedRepos', 'Related Repositories')}</span>
            </div>
          </div>

          {/* Dimension Tabs */}
          <div className="flex items-center gap-1 p-1.5 bg-bg-sidebar/80 rounded-xl border border-border-light/60 dark:bg-dark-bg-sidebar/80 dark:border-dark-border">
            <button
              onClick={() => setActiveTab('similar')}
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg transition-all',
                activeTab === 'similar'
                  ? 'bg-bg-main shadow-sm text-text-main font-medium ring-1 ring-border-light/70 dark:bg-dark-bg-main dark:text-dark-text-main dark:ring-dark-border'
                  : 'text-text-muted hover:text-text-main dark:text-dark-text-main/70 dark:hover:text-dark-text-main'
              )}
            >
              <Link2 className="w-3 h-3" />
              <span>{t('repoDetails.similar', 'Similar')}</span>
              {tabCounts.similar > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-bg-sidebar text-text-dim dark:bg-dark-bg-sidebar dark:text-dark-text-main/60">({tabCounts.similar})</span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('sameTags')}
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg transition-all',
                activeTab === 'sameTags'
                  ? 'bg-bg-main shadow-sm text-text-main font-medium ring-1 ring-border-light/70 dark:bg-dark-bg-main dark:text-dark-text-main dark:ring-dark-border'
                  : 'text-text-muted hover:text-text-main dark:text-dark-text-main/70 dark:hover:text-dark-text-main'
              )}
            >
              <Tag className="w-3 h-3" />
              <span>{t('repoDetails.sameTags', 'Tags')}</span>
              {tabCounts.sameTags > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-bg-sidebar text-text-dim">({tabCounts.sameTags})</span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('sameLang')}
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg transition-all',
                activeTab === 'sameLang'
                  ? 'bg-bg-main shadow-sm text-text-main font-medium ring-1 ring-border-light/70 dark:bg-dark-bg-main dark:text-dark-text-main dark:ring-dark-border'
                  : 'text-text-muted hover:text-text-main dark:text-dark-text-main/70 dark:hover:text-dark-text-main'
              )}
            >
              <Code className="w-3 h-3" />
              <span>{t('repoDetails.sameLang', 'Lang')}</span>
              {tabCounts.sameLang > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-bg-sidebar text-text-dim dark:bg-dark-bg-sidebar dark:text-dark-text-main/60">({tabCounts.sameLang})</span>
              )}
            </button>
          </div>

          {/* Related Repos List — independently scrollable, sized to remaining space */}
          {currentRelatedRepos.length > 0 ? (
            <div className="bg-bg-sidebar/60 rounded-xl border border-border-light/60 divide-y divide-border-light/40 p-1.5 flex-1 min-h-0 overflow-y-auto overscroll-contain dark:bg-dark-bg-sidebar/60 dark:border-dark-border dark:divide-dark-border/60">
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
          ) : (
            <div className="px-4 py-6 text-center text-sm text-text-muted bg-bg-sidebar/60 rounded-xl border border-border-light/60 flex items-center justify-center flex-1 min-h-0 dark:text-dark-text-main/70 dark:bg-dark-bg-sidebar/60 dark:border-dark-border">
              {activeTab === 'similar' && (
                similarLoading
                  ? t('common.loading', 'Loading...')
                  : (
                      similarError
                        ? t('repoDetails.similarLoadFailed', 'Failed to load similar repos')
                        : t('repoDetails.noSimilar', 'No similar repos found')
                    )
              )}
              {activeTab === 'sameTags' && t('repoDetails.noSameTags', 'No repos with same tags')}
              {activeTab === 'sameLang' && t('repoDetails.noSameLang', 'No repos with same language')}
            </div>
          )}
        </div>


      </div>
    </div>
  );
};
