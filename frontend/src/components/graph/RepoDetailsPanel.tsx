import React, { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { GraphNode } from '../../types';
import { X, Star, Code, ExternalLink, Sparkles, Tag, FolderHeart, Link2, ChevronRight, User, Package, GitFork, Copy, Check } from 'lucide-react';
import { useGraph, useNodeNeighbors } from '../../contexts/GraphContext';
import { clsx } from 'clsx';

interface RepoDetailsPanelProps {
  node: GraphNode;
  onClose: () => void;
}

interface RelatedRepoItemProps {
  repo: GraphNode;
  onClick: () => void;
  matchReason?: string;
}

type RelatedTab = 'similar' | 'sameTags' | 'sameOwner' | 'sameLang';

/** Related repository item component */
const RelatedRepoItem: React.FC<RelatedRepoItemProps> = ({ repo, onClick, matchReason }) => {
  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 p-2.5 rounded-lg hover:bg-bg-hover transition-colors text-left group"
    >
      {/* Avatar */}
      {repo.owner_avatar_url ? (
        <img
          src={repo.owner_avatar_url}
          alt={repo.owner}
          className="w-8 h-8 rounded-md border border-border-light flex-shrink-0"
        />
      ) : (
        <div className="w-8 h-8 rounded-md bg-gray-200 flex items-center justify-center flex-shrink-0">
          <span className="text-gray-500 text-xs font-medium">
            {repo.owner?.charAt(0).toUpperCase()}
          </span>
        </div>
      )}

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-text-main truncate">{repo.name}</span>
          {repo.language && (
            <span className="text-[10px] px-1.5 py-0.5 bg-gray-100 rounded text-gray-600 flex-shrink-0">
              {repo.language}
            </span>
          )}
        </div>
        <p className="text-xs text-text-muted truncate mt-0.5">
          {repo.description || repo.ai_summary || 'No description'}
        </p>
        {matchReason && (
          <span className="text-[10px] text-purple-600 mt-0.5">{matchReason}</span>
        )}
      </div>

      {/* Stars + Arrow */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <div className="flex items-center gap-1 text-xs text-text-dim">
          <Star className="w-3 h-3 text-orange-400" fill="currentColor" />
          <span>{repo.stargazers_count >= 1000 ? `${(repo.stargazers_count / 1000).toFixed(1)}k` : repo.stargazers_count}</span>
        </div>
        <ChevronRight className="w-4 h-4 text-text-dim opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
    </button>
  );
};

export const RepoDetailsPanel: React.FC<RepoDetailsPanelProps> = ({ node, onClose }) => {
  const { t } = useTranslation();
  const { rawData, setSelectedNode } = useGraph();
  const neighborIds = useNodeNeighbors(node.id);
  const [activeTab, setActiveTab] = useState<RelatedTab>('similar');
  const [copied, setCopied] = useState(false);

  // Get related repos by different dimensions
  const relatedReposByDimension = useMemo(() => {
    if (!rawData) return { similar: [], sameTags: [], sameOwner: [], sameLang: [] };

    // 1. Semantically similar (neighbors in graph)
    const similar = rawData.nodes
      .filter(n => neighborIds.has(n.id))
      .sort((a, b) => b.stargazers_count - a.stargazers_count)
      .slice(0, 10);

    // 2. Same tags (ai_tags or topics overlap)
    const nodeTags = new Set([...(node.ai_tags || []), ...(node.topics || [])]);
    const sameTags = nodeTags.size > 0
      ? rawData.nodes
          .filter(n => {
            if (n.id === node.id) return false;
            const nTags = new Set([...(n.ai_tags || []), ...(n.topics || [])]);
            const overlap = [...nodeTags].filter(t => nTags.has(t));
            return overlap.length >= 2; // At least 2 common tags
          })
          .map(n => {
            const nTags = new Set([...(n.ai_tags || []), ...(n.topics || [])]);
            const overlap = [...nodeTags].filter(t => nTags.has(t));
            return { ...n, _matchReason: overlap.slice(0, 2).join(', ') };
          })
          .sort((a, b) => b.stargazers_count - a.stargazers_count)
          .slice(0, 10)
      : [];

    // 3. Same owner
    const sameOwner = rawData.nodes
      .filter(n => n.id !== node.id && n.owner === node.owner)
      .sort((a, b) => b.stargazers_count - a.stargazers_count)
      .slice(0, 10);

    // 4. Same language
    const sameLang = node.language
      ? rawData.nodes
          .filter(n => n.id !== node.id && n.language === node.language)
          .sort((a, b) => b.stargazers_count - a.stargazers_count)
          .slice(0, 10)
      : [];

    return { similar, sameTags, sameOwner, sameLang };
  }, [rawData, neighborIds, node]);

  // Get current tab's repos
  const currentRelatedRepos = relatedReposByDimension[activeTab] || [];

  // Handle clicking on a related repo
  const handleRelatedRepoClick = (repo: GraphNode) => {
    setSelectedNode(repo);
  };

  // Copy git clone command
  const handleCopyClone = () => {
    const cloneUrl = `git clone https://github.com/${node.full_name}.git`;
    navigator.clipboard.writeText(cloneUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  // Tab counts
  const tabCounts = {
    similar: relatedReposByDimension.similar.length,
    sameTags: relatedReposByDimension.sameTags.length,
    sameOwner: relatedReposByDimension.sameOwner.length,
    sameLang: relatedReposByDimension.sameLang.length,
  };

  return (
    <div className="absolute top-6 right-6 z-30 w-96 bg-white rounded-lg border border-border-light shadow-xl overflow-hidden animate-in fade-in slide-in-from-right-4 duration-300">
      {/* Header with Avatar */}
      <div className="relative p-5 border-b border-border-light bg-bg-sidebar">
        <div className="flex items-start gap-3 pr-8">
            {/* Owner Avatar */}
            {node.owner_avatar_url ? (
              <img
                src={node.owner_avatar_url}
                alt={node.owner}
                className="w-10 h-10 rounded-lg border border-border-light flex-shrink-0"
              />
            ) : (
              <div className="w-10 h-10 rounded-lg bg-gray-200 flex items-center justify-center flex-shrink-0">
                <span className="text-gray-500 text-sm font-medium">
                  {node.owner?.charAt(0).toUpperCase()}
                </span>
              </div>
            )}

            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                    <a href={node.html_url} target="_blank" rel="noopener noreferrer" className="hover:underline decoration-text-muted underline-offset-4">
                        <h2 className="text-base font-semibold text-text-main line-clamp-1 leading-snug" title={node.full_name}>
                        {node.name}
                        </h2>
                    </a>
                </div>
                <p className="text-xs text-text-dim">{node.owner}</p>
                <p className="text-sm text-text-muted line-clamp-2 leading-relaxed mt-1">
                {node.description || 'No description available'}
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
      <div className="p-5 space-y-5">

        {/* User's Star List Badge */}
        {node.star_list_name && (
          <div className="flex items-center gap-2">
            <FolderHeart className="w-4 h-4 text-pink-500" />
            <span className="text-xs font-medium text-pink-600 bg-pink-50 px-2 py-1 rounded-full">
              {node.star_list_name}
            </span>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-3 p-3 rounded-md bg-white border border-border-light shadow-sm">
                <div className="p-1.5 bg-orange-50 rounded text-orange-500">
                    <Star className="w-4 h-4" fill="currentColor" />
                </div>
                <div>
                    <span className="block text-lg font-semibold text-text-main leading-none">{node.stargazers_count?.toLocaleString() ?? 0}</span>
                    <span className="text-xs text-text-muted capitalize">{t('repoDetails.stars')}</span>
                </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-md bg-white border border-border-light shadow-sm">
                 <div className="p-1.5 bg-blue-50 rounded text-blue-500">
                    <Code className="w-4 h-4" />
                </div>
                <div>
                    <span className="block text-lg font-semibold text-text-main leading-none truncate max-w-[100px]" title={node.language || 'Unknown'}>{node.language || 'N/A'}</span>
                    <span className="text-xs text-text-muted capitalize">{t('repoDetails.language')}</span>
                </div>
            </div>
        </div>

        {/* AI Tags */}
        {node.ai_tags && node.ai_tags.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-purple-600 uppercase tracking-wider">
                <Tag className="w-3.5 h-3.5" />
                <span>AI Tags</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {node.ai_tags.map((tag, idx) => (
                <span
                  key={idx}
                  className="text-xs px-2 py-1 bg-purple-50 text-purple-700 rounded-full"
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
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">
                Topics
            </div>
            <div className="flex flex-wrap gap-1.5">
              {node.topics.slice(0, 8).map((topic, idx) => (
                <span
                  key={idx}
                  className="text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded-full"
                >
                  {topic}
                </span>
              ))}
              {node.topics.length > 8 && (
                <span className="text-xs text-text-dim">+{node.topics.length - 8}</span>
              )}
            </div>
          </div>
        )}

        {/* AI Summary */}
        <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-action-primary uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5" />
                <span>AI Insight</span>
            </div>
            <div className="bg-bg-sidebar p-3 rounded-md border border-border-light/50">
                 <p className="text-sm text-text-main leading-relaxed">
                    {node.ai_summary || "No AI summary available for this repository yet. Sync data to generate insights."}
                </p>
            </div>
        </div>

        {/* Related Repositories with Tabs */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-teal-600 uppercase tracking-wider">
              <Link2 className="w-3.5 h-3.5" />
              <span>{t('repoDetails.relatedRepos', 'Related Repositories')}</span>
            </div>
          </div>

          {/* Dimension Tabs */}
          <div className="flex items-center gap-1 p-1 bg-bg-sidebar rounded-lg">
            <button
              onClick={() => setActiveTab('similar')}
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md transition-colors',
                activeTab === 'similar'
                  ? 'bg-white shadow-sm text-text-main font-medium'
                  : 'text-text-muted hover:text-text-main'
              )}
            >
              <Link2 className="w-3 h-3" />
              <span>{t('repoDetails.similar', 'Similar')}</span>
              {tabCounts.similar > 0 && (
                <span className="text-[10px] text-text-dim">({tabCounts.similar})</span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('sameTags')}
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md transition-colors',
                activeTab === 'sameTags'
                  ? 'bg-white shadow-sm text-text-main font-medium'
                  : 'text-text-muted hover:text-text-main'
              )}
            >
              <Tag className="w-3 h-3" />
              <span>{t('repoDetails.sameTags', 'Tags')}</span>
              {tabCounts.sameTags > 0 && (
                <span className="text-[10px] text-text-dim">({tabCounts.sameTags})</span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('sameOwner')}
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md transition-colors',
                activeTab === 'sameOwner'
                  ? 'bg-white shadow-sm text-text-main font-medium'
                  : 'text-text-muted hover:text-text-main'
              )}
            >
              <User className="w-3 h-3" />
              <span>{t('repoDetails.sameOwner', 'Author')}</span>
              {tabCounts.sameOwner > 0 && (
                <span className="text-[10px] text-text-dim">({tabCounts.sameOwner})</span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('sameLang')}
              className={clsx(
                'flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md transition-colors',
                activeTab === 'sameLang'
                  ? 'bg-white shadow-sm text-text-main font-medium'
                  : 'text-text-muted hover:text-text-main'
              )}
            >
              <Code className="w-3 h-3" />
              <span>{t('repoDetails.sameLang', 'Lang')}</span>
              {tabCounts.sameLang > 0 && (
                <span className="text-[10px] text-text-dim">({tabCounts.sameLang})</span>
              )}
            </button>
          </div>

          {/* Related Repos List */}
          {currentRelatedRepos.length > 0 ? (
            <div className="bg-bg-sidebar rounded-lg border border-border-light/50 divide-y divide-border-light/50 max-h-56 overflow-y-auto">
              {currentRelatedRepos.map((repo: any) => (
                <RelatedRepoItem
                  key={repo.id}
                  repo={repo}
                  onClick={() => handleRelatedRepoClick(repo)}
                  matchReason={repo._matchReason}
                />
              ))}
            </div>
          ) : (
            <div className="py-6 text-center text-sm text-text-muted bg-bg-sidebar rounded-lg border border-border-light/50">
              {activeTab === 'similar' && t('repoDetails.noSimilar', 'No similar repos found')}
              {activeTab === 'sameTags' && t('repoDetails.noSameTags', 'No repos with same tags')}
              {activeTab === 'sameOwner' && t('repoDetails.noSameOwner', 'No other repos from this author')}
              {activeTab === 'sameLang' && t('repoDetails.noSameLang', 'No repos with same language')}
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="space-y-2">
          {/* Quick Actions */}
          <div className="flex items-center gap-2">
            <a
              href={node.html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-text-main hover:bg-black text-white text-sm font-medium rounded-md transition-all shadow hover:shadow-md"
            >
              <ExternalLink className="w-4 h-4" />
              <span>GitHub</span>
            </a>
            <a
              href={`${node.html_url}/stargazers`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-1.5 px-3 py-2 border border-border-light hover:bg-bg-hover text-text-main text-sm rounded-md transition-colors"
            >
              <Star className="w-4 h-4 text-orange-500" />
            </a>
            <a
              href={`${node.html_url}/fork`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-1.5 px-3 py-2 border border-border-light hover:bg-bg-hover text-text-main text-sm rounded-md transition-colors"
            >
              <GitFork className="w-4 h-4" />
            </a>
            <button
              onClick={handleCopyClone}
              className="flex items-center justify-center gap-1.5 px-3 py-2 border border-border-light hover:bg-bg-hover text-text-main text-sm rounded-md transition-colors"
              title="Copy git clone command"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-500" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* Clone command preview */}
          <div className="flex items-center gap-2 px-3 py-2 bg-bg-sidebar rounded-md text-xs text-text-muted font-mono overflow-hidden">
            <span className="truncate">git clone https://github.com/{node.full_name}.git</span>
          </div>
        </div>
      </div>
    </div>
  );
};
