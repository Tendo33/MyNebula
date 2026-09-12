import React, { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import {
  Clock,
  Code,
  Command,
  Search,
  Star,
  Tag,
  TrendingUp,
  X,
} from 'lucide-react';

import { useGraph } from '../../contexts/GraphContext';
import { Button } from './button';
import { Kbd } from './kbd';
import { ToggleGroup, ToggleGroupItem } from './toggle-group';
import { CommandPaletteResultList } from './CommandPaletteResultList';
import type {
  CommandPaletteProps,
  FilterType,
  SearchResult,
} from './commandPaletteTypes';
import { useRecentSearches } from './hooks/useRecentSearches';
import { useDialogFocusTrap } from './hooks/useDialogFocusTrap';
import { useCommandPaletteResults } from './hooks/useCommandPaletteResults';

/**
 * Command palette.
 *
 * Result assembly lives in `useCommandPaletteResults`, facet counting in
 * `commandPaletteFacets`, history in `useRecentSearches`, and focus management
 * in `useDialogFocusTrap`. This component owns keyboard navigation and markup.
 */
export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onSelectNode,
  onSelectCluster,
  onSelectSearch,
}) => {
  const { t } = useTranslation();
  const { rawData, setSelectedNode, setSearchQuery } = useGraph();
  const { recentSearches, addRecentSearch, clearRecentSearches } = useRecentSearches();

  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<FilterType>('all');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const dialogRef = useDialogFocusTrap(isOpen);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const { results, quickFilters, remoteLoading, remoteError } = useCommandPaletteResults({
    isOpen,
    rawData,
    query,
    activeFilter,
    t,
  });

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
        onSelectSearch?.(result.data.language, 'language');
        break;
      case 'tag':
        setSearchQuery(result.data.tag);
        onSelectSearch?.(result.data.tag, 'tag');
        break;
    }

    onClose();
  }, [query, addRecentSearch, setSelectedNode, onSelectNode, onSelectCluster, onSelectSearch, setSearchQuery, onClose]);

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
            onSelectSearch?.(normalizedQuery, 'search');
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
  }, [isOpen, results, selectedIndex, query, addRecentSearch, onSelectSearch, setSearchQuery, onClose, handleSelectResult]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current) {
      const selectedItem = listRef.current.querySelector(`[data-index="${selectedIndex}"]`);
      selectedItem?.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex]);

  // Handle quick filter click
  const handleQuickFilter = useCallback((filterQuery: string) => {
    setQuery(filterQuery);
  }, []);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
      />

      {/* Palette */}
      <div
        ref={dialogRef}
        className="relative w-full max-w-2xl bg-bg-main rounded-xl shadow-2xl border border-border-light overflow-hidden animate-in fade-in slide-in-from-top-4 duration-200 dark:bg-dark-bg-main dark:border-dark-border"
        role="dialog"
        aria-modal="true"
        aria-labelledby="command-palette-title"
      >
        <h2 id="command-palette-title" className="sr-only">
          {t('search.command_palette', 'Command Palette')}
        </h2>
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
            className="flex-1 text-base outline-none placeholder:text-text-muted"
            aria-label={t('search.placeholder', 'Search repos, clusters, languages, tags...')}
          />
          <div className="flex items-center gap-2">
            <Kbd className="hidden sm:inline-flex">
              <Command />K
            </Kbd>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              onClick={onClose}
              aria-label={t('common.close', 'Close')}
            >
              <X />
            </Button>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-1 border-b border-border-light bg-bg-sidebar/50 px-4 py-2">
          <ToggleGroup
            value={[activeFilter]}
            onValueChange={(next) => {
              const value = next[0];
              if (value) setActiveFilter(value as FilterType);
            }}
            className="flex flex-wrap"
          >
            {([
              { key: 'all', label: t('search.all', 'All') },
              { key: 'repos', label: t('search.repos', 'Repos') },
              { key: 'clusters', label: t('search.clusters', 'Clusters') },
              { key: 'languages', label: t('search.languages', 'Languages') },
              { key: 'tags', label: t('search.tags', 'Tags') },
            ] as const).map((filter) => (
              <ToggleGroupItem key={filter.key} value={filter.key} size="sm">
                {filter.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>

        {/* Results / Empty State */}
        <div
          ref={listRef}
          className="max-h-[50vh] overflow-y-auto"
          role={results.length > 0 ? 'listbox' : undefined}
          aria-activedescendant={results.length > 0 ? `command-palette-option-${selectedIndex}` : undefined}
        >
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
                      className="text-xs text-text-muted hover:text-text-main"
                    >
                      {t('common.clear', 'Clear')}
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {recentSearches.map((search) => (
                      <button
                        key={search}
                        onClick={() => setQuery(search)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-bg-sidebar hover:bg-bg-hover rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
                      >
                        <Clock className="w-3 h-3 text-text-muted" />
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
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-action-primary/10 text-action-primary hover:bg-action-primary/15 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
                      >
                        <Code className="w-3 h-3" />
                        {lang}
                        <span className="text-action-primary/80 text-xs">({count})</span>
                      </button>
                    ))}
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-2">
                    {quickFilters.tags.map(([tag, count]) => (
                      <button
                        key={tag}
                        onClick={() => handleQuickFilter(tag)}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-action-primary/10 text-action-primary hover:bg-action-primary/15 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
                      >
                        <Tag className="w-3 h-3" />
                        {tag}
                        <span className="text-action-primary/80 text-xs">({count})</span>
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
                          onSelectSearch?.(starQuery, 'search');
                          onClose();
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-action-primary/10 text-action-primary hover:bg-action-primary/15 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
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
                <div className="flex items-center gap-2 text-xs text-text-muted">
                  <TrendingUp className="w-3 h-3" />
                  <span>{t('search.tip', 'Tip: Type to search, use arrow keys to navigate, Enter to select')}</span>
                </div>
              </div>
            </div>
          ) : remoteLoading && results.length === 0 ? (
            <div className="px-4 py-8 text-sm text-muted-foreground">
              {t('common.loading', 'Loading...')}
            </div>
          ) : results.length > 0 ? (
            <CommandPaletteResultList
              results={results}
              selectedIndex={selectedIndex}
              onSelect={handleSelectResult}
            />
          ) : (
            // No results
            <div className="px-4 py-8">
              <p className="text-sm text-text-main">
                {t('search.noResults', 'No results found for')} "{query}"
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {remoteError ?? t('search.tryDifferent', 'Try a different search term')}
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border-light bg-bg-sidebar/50 text-xs text-text-muted">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <Kbd>↑↓</Kbd>
              {t('search.navigate', 'Navigate')}
            </span>
            <span className="flex items-center gap-1">
              <Kbd>↵</Kbd>
              {t('search.select', 'Select')}
            </span>
            <span className="flex items-center gap-1">
              <Kbd>esc</Kbd>
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
