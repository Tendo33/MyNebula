import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import {
  Search,
  X,
  Star,
  Code,
  Tag,
  Clock,
  ArrowRight,
  TrendingUp,
  Command,
} from 'lucide-react';
import { useGraph } from '../../contexts/GraphContext';
import { GraphNode, ClusterInfo } from '../../types';

// ============================================================================
// Types
// ============================================================================

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectNode?: (node: GraphNode) => void;
  onSelectCluster?: (cluster: ClusterInfo) => void;
}

type FilterType = 'all' | 'repos' | 'clusters' | 'languages' | 'tags';

interface SearchResult {
  type: 'repo' | 'cluster' | 'language' | 'tag';
  id: string | number;
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  meta?: string;
  data: any;
}

// ============================================================================
// Constants
// ============================================================================

const RECENT_SEARCHES_KEY = 'nebula_recent_searches';
const MAX_RECENT_SEARCHES = 5;
const MAX_RESULTS = 20;

// ============================================================================
// Hooks
// ============================================================================

const useRecentSearches = () => {
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(RECENT_SEARCHES_KEY);
      if (stored) {
        setRecentSearches(JSON.parse(stored));
      }
    } catch (e) {
      console.warn('Failed to load recent searches');
    }
  }, []);

  const addRecentSearch = useCallback((query: string) => {
    if (!query.trim()) return;

    setRecentSearches(prev => {
      const filtered = prev.filter(s => s !== query);
      const updated = [query, ...filtered].slice(0, MAX_RECENT_SEARCHES);
      try {
        localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(updated));
      } catch (e) {
        console.warn('Failed to save recent searches');
      }
      return updated;
    });
  }, []);

  const clearRecentSearches = useCallback(() => {
    setRecentSearches([]);
    try {
      localStorage.removeItem(RECENT_SEARCHES_KEY);
    } catch (e) {
      console.warn('Failed to clear recent searches');
    }
  }, []);

  return { recentSearches, addRecentSearch, clearRecentSearches };
};

// ============================================================================
// Component
// ============================================================================

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onSelectNode,
  onSelectCluster,
}) => {
  const { t } = useTranslation();
  const { rawData, setSelectedNode, setSearchQuery } = useGraph();
  const { recentSearches, addRecentSearch, clearRecentSearches } = useRecentSearches();

  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterType>('all');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Get unique languages
  const languages = useMemo(() => {
    if (!rawData) return [];
    const langCounts: Record<string, number> = {};
    rawData.nodes.forEach(node => {
      if (node.language) {
        langCounts[node.language] = (langCounts[node.language] || 0) + 1;
      }
    });
    return Object.entries(langCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
  }, [rawData]);

  // Get unique tags
  const allTags = useMemo(() => {
    if (!rawData) return [];
    const tagCounts: Record<string, number> = {};
    rawData.nodes.forEach(node => {
      (node.ai_tags || []).forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
      (node.topics || []).forEach(topic => {
        tagCounts[topic] = (tagCounts[topic] || 0) + 1;
      });
    });
    return Object.entries(tagCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);
  }, [rawData]);

  // Search results
  const results = useMemo((): SearchResult[] => {
    if (!rawData) return [];

    const searchResults: SearchResult[] = [];
    const lowerQuery = query.toLowerCase().trim();

    // Search repos
    if (activeFilter === 'all' || activeFilter === 'repos') {
      const matchedRepos = rawData.nodes
        .filter(node => {
          if (!lowerQuery) return false;
          return (
            node.name.toLowerCase().includes(lowerQuery) ||
            node.full_name.toLowerCase().includes(lowerQuery) ||
            node.description?.toLowerCase().includes(lowerQuery) ||
            node.ai_summary?.toLowerCase().includes(lowerQuery) ||
            node.ai_tags?.some(tag => tag.toLowerCase().includes(lowerQuery)) ||
            node.topics?.some(topic => topic.toLowerCase().includes(lowerQuery))
          );
        })
        .slice(0, MAX_RESULTS)
        .map(node => ({
          type: 'repo' as const,
          id: node.id,
          title: node.name,
          subtitle: node.description || node.ai_summary,
          icon: node.owner_avatar_url ? (
            <img src={node.owner_avatar_url} alt="" className="w-6 h-6 rounded" />
          ) : (
            <div className="w-6 h-6 rounded bg-gray-200 flex items-center justify-center text-xs">
              {node.owner?.charAt(0).toUpperCase()}
            </div>
          ),
          meta: `⭐ ${node.stargazers_count.toLocaleString()}${node.language ? ` · ${node.language}` : ''}`,
          data: node,
        }));
      searchResults.push(...matchedRepos);
    }

    // Search clusters
    if (activeFilter === 'all' || activeFilter === 'clusters') {
      const matchedClusters = rawData.clusters
        .filter(cluster => {
          if (!lowerQuery) return false;
          return (
            cluster.name?.toLowerCase().includes(lowerQuery) ||
            cluster.description?.toLowerCase().includes(lowerQuery) ||
            cluster.keywords?.some(k => k.toLowerCase().includes(lowerQuery))
          );
        })
        .slice(0, 5)
        .map(cluster => ({
          type: 'cluster' as const,
          id: cluster.id,
          title: cluster.name || `Cluster ${cluster.id}`,
          subtitle: cluster.description,
          icon: (
            <div
              className="w-6 h-6 rounded-full"
              style={{ backgroundColor: cluster.color }}
            />
          ),
          meta: `${cluster.repo_count} repos`,
          data: cluster,
        }));
      searchResults.push(...matchedClusters);
    }

    // Search languages
    if (activeFilter === 'all' || activeFilter === 'languages') {
      const matchedLanguages = languages
        .filter(([lang]) => {
          if (!lowerQuery) return false;
          return lang.toLowerCase().includes(lowerQuery);
        })
        .slice(0, 5)
        .map(([lang, count]) => ({
          type: 'language' as const,
          id: lang,
          title: lang,
          subtitle: `Filter by ${lang} repositories`,
          icon: <Code className="w-5 h-5 text-blue-500" />,
          meta: `${count} repos`,
          data: { language: lang },
        }));
      searchResults.push(...matchedLanguages);
    }

    // Search tags
    if (activeFilter === 'all' || activeFilter === 'tags') {
      const matchedTags = allTags
        .filter(([tag]) => {
          if (!lowerQuery) return false;
          return tag.toLowerCase().includes(lowerQuery);
        })
        .slice(0, 5)
        .map(([tag, count]) => ({
          type: 'tag' as const,
          id: tag,
          title: tag,
          subtitle: `Filter by tag`,
          icon: <Tag className="w-5 h-5 text-purple-500" />,
          meta: `${count} repos`,
          data: { tag },
        }));
      searchResults.push(...matchedTags);
    }

    return searchResults;
  }, [rawData, query, activeFilter, languages, allTags]);

  // Quick filters for empty state
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

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, results.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (results[selectedIndex]) {
            handleSelectResult(results[selectedIndex]);
          } else if (query.trim()) {
            const normalizedQuery = query.trim();
            addRecentSearch(normalizedQuery);
            setSearchQuery(normalizedQuery);
            onClose();
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, results, selectedIndex, query, addRecentSearch, setSearchQuery, onClose]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current) {
      const selectedItem = listRef.current.querySelector(`[data-index="${selectedIndex}"]`);
      selectedItem?.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex]);

  // Handle result selection
  const handleSelectResult = useCallback((result: SearchResult) => {
    addRecentSearch(query);

    switch (result.type) {
      case 'repo':
        setSelectedNode(result.data);
        onSelectNode?.(result.data);
        break;
      case 'cluster':
        onSelectCluster?.(result.data);
        break;
      case 'language':
        setSearchQuery(result.data.language);
        break;
      case 'tag':
        setSearchQuery(result.data.tag);
        break;
    }

    onClose();
  }, [query, addRecentSearch, setSelectedNode, onSelectNode, onSelectCluster, setSearchQuery, onClose]);

  // Handle quick filter click
  const handleQuickFilter = useCallback((filterQuery: string) => {
    setQuery(filterQuery);
  }, []);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="relative w-full max-w-2xl bg-white rounded-xl shadow-2xl border border-border-light overflow-hidden animate-in fade-in slide-in-from-top-4 duration-200">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border-light">
          <Search className="w-5 h-5 text-text-muted flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder={t('search.placeholder', 'Search repos, clusters, languages, tags...')}
            className="flex-1 text-base outline-none placeholder:text-text-dim"
          />
          <div className="flex items-center gap-2">
            <kbd className="hidden sm:flex items-center gap-1 px-2 py-1 text-xs text-text-dim bg-bg-sidebar rounded border border-border-light">
              <Command className="w-3 h-3" />K
            </kbd>
            <button
              onClick={onClose}
              className="p-1 hover:bg-bg-hover rounded transition-colors"
            >
              <X className="w-5 h-5 text-text-muted" />
            </button>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-1 px-4 py-2 border-b border-border-light bg-bg-sidebar/50">
          {[
            { key: 'all', label: t('search.all', 'All') },
            { key: 'repos', label: t('search.repos', 'Repos') },
            { key: 'clusters', label: t('search.clusters', 'Clusters') },
            { key: 'languages', label: t('search.languages', 'Languages') },
            { key: 'tags', label: t('search.tags', 'Tags') },
          ].map(filter => (
            <button
              key={filter.key}
              onClick={() => setActiveFilter(filter.key as FilterType)}
              className={clsx(
                'px-3 py-1.5 text-xs font-medium rounded-md transition-colors',
                activeFilter === filter.key
                  ? 'bg-white text-text-main shadow-sm'
                  : 'text-text-muted hover:text-text-main hover:bg-bg-hover'
              )}
            >
              {filter.label}
            </button>
          ))}
        </div>

        {/* Results / Empty State */}
        <div ref={listRef} className="max-h-[50vh] overflow-y-auto">
          {query.trim() === '' ? (
            // Empty state with quick filters
            <div className="p-4 space-y-4">
              {/* Recent Searches */}
              {recentSearches.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
                      {t('search.recent', 'Recent Searches')}
                    </span>
                    <button
                      onClick={clearRecentSearches}
                      className="text-xs text-text-dim hover:text-text-main"
                    >
                      {t('common.clear', 'Clear')}
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {recentSearches.map((search, idx) => (
                      <button
                        key={idx}
                        onClick={() => setQuery(search)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-bg-sidebar hover:bg-bg-hover rounded-full transition-colors"
                      >
                        <Clock className="w-3 h-3 text-text-dim" />
                        {search}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Quick Filters */}
              <div>
                <span className="text-xs font-medium text-text-muted uppercase tracking-wider">
                  {t('search.quickFilters', 'Quick Filters')}
                </span>
                <div className="mt-2 space-y-3">
                  {/* Languages */}
                  <div className="flex flex-wrap gap-2">
                    {quickFilters.languages.map(([lang, count]) => (
                      <button
                        key={lang}
                        onClick={() => handleQuickFilter(lang)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-blue-50 text-blue-700 hover:bg-blue-100 rounded-full transition-colors"
                      >
                        <Code className="w-3 h-3" />
                        {lang}
                        <span className="text-blue-500 text-xs">({count})</span>
                      </button>
                    ))}
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-2">
                    {quickFilters.tags.map(([tag, count]) => (
                      <button
                        key={tag}
                        onClick={() => handleQuickFilter(tag)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-purple-50 text-purple-700 hover:bg-purple-100 rounded-full transition-colors"
                      >
                        <Tag className="w-3 h-3" />
                        {tag}
                        <span className="text-purple-500 text-xs">({count})</span>
                      </button>
                    ))}
                  </div>

                  {/* Star ranges */}
                  <div className="flex flex-wrap gap-2">
                    {quickFilters.starRanges.map(range => (
                      <button
                        key={range.min}
                        onClick={() => {
                          const starQuery = `stars:>${range.min}`;
                          addRecentSearch(starQuery);
                          setSearchQuery(starQuery);
                          onClose();
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-orange-50 text-orange-700 hover:bg-orange-100 rounded-full transition-colors"
                      >
                        <Star className="w-3 h-3" />
                        {range.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Tips */}
              <div className="pt-2 border-t border-border-light">
                <div className="flex items-center gap-2 text-xs text-text-dim">
                  <TrendingUp className="w-3 h-3" />
                  <span>{t('search.tip', 'Tip: Type to search, use arrow keys to navigate, Enter to select')}</span>
                </div>
              </div>
            </div>
          ) : results.length > 0 ? (
            // Search results
            <div className="py-2">
              {results.map((result, idx) => (
                <button
                  key={`${result.type}-${result.id}`}
                  data-index={idx}
                  onClick={() => handleSelectResult(result)}
                  className={clsx(
                    'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors',
                    idx === selectedIndex
                      ? 'bg-action-primary/10'
                      : 'hover:bg-bg-hover'
                  )}
                >
                  {/* Icon */}
                  <div className="flex-shrink-0">
                    {result.icon}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-text-main truncate">
                        {result.title}
                      </span>
                      <span className={clsx(
                        'text-[10px] px-1.5 py-0.5 rounded uppercase',
                        result.type === 'repo' && 'bg-gray-100 text-gray-600',
                        result.type === 'cluster' && 'bg-teal-100 text-teal-700',
                        result.type === 'language' && 'bg-blue-100 text-blue-700',
                        result.type === 'tag' && 'bg-purple-100 text-purple-700',
                      )}>
                        {result.type}
                      </span>
                    </div>
                    {result.subtitle && (
                      <p className="text-sm text-text-muted truncate mt-0.5">
                        {result.subtitle}
                      </p>
                    )}
                  </div>

                  {/* Meta */}
                  {result.meta && (
                    <span className="text-xs text-text-dim flex-shrink-0">
                      {result.meta}
                    </span>
                  )}

                  {/* Arrow */}
                  <ArrowRight className={clsx(
                    'w-4 h-4 flex-shrink-0 transition-opacity',
                    idx === selectedIndex ? 'opacity-100 text-action-primary' : 'opacity-0'
                  )} />
                </button>
              ))}
            </div>
          ) : (
            // No results
            <div className="py-12 text-center">
              <Search className="w-12 h-12 text-text-dim mx-auto mb-3" />
              <p className="text-text-muted">
                {t('search.noResults', 'No results found for')} "{query}"
              </p>
              <p className="text-sm text-text-dim mt-1">
                {t('search.tryDifferent', 'Try a different search term')}
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border-light bg-bg-sidebar/50 text-xs text-text-dim">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-white rounded border border-border-light">↑↓</kbd>
              {t('search.navigate', 'Navigate')}
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-white rounded border border-border-light">↵</kbd>
              {t('search.select', 'Select')}
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-white rounded border border-border-light">esc</kbd>
              {t('search.close', 'Close')}
            </span>
          </div>
          {results.length > 0 && (
            <span>{results.length} {t('search.results', 'results')}</span>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
};

export default CommandPalette;
