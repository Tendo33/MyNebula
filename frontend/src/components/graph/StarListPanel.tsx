import React, { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { FolderHeart, Check, X, ChevronDown, ChevronUp } from 'lucide-react';
import { useGraph } from '../../contexts/GraphContext';
import { StarListInfo } from '../../types';

// ============================================================================
// Types
// ============================================================================

interface StarListPanelProps {
  /** Collapsed state */
  collapsed?: boolean;
  /** Toggle collapsed callback */
  onToggleCollapsed?: () => void;
  /** Additional class names */
  className?: string;
}

interface StarListItemProps {
  list: StarListInfo;
  isSelected: boolean;
  onToggle: () => void;
}

// ============================================================================
// StarListItem Component
// ============================================================================

const StarListItem: React.FC<StarListItemProps> = ({
  list,
  isSelected,
  onToggle,
}) => {
  return (
    <button
      onClick={onToggle}
      className={clsx(
        'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left border',
        'hover:bg-bg-hover hover:border-border-light/70 group border-transparent dark:hover:bg-dark-bg-sidebar/70',
        isSelected && 'bg-action-primary/10 ring-1 ring-action-primary/20 border-border-light shadow-sm dark:bg-dark-bg-main dark:border-dark-border'
      )}
    >
      {/* Icon */}
      <FolderHeart className={clsx(
        'w-4 h-4 flex-shrink-0',
        isSelected ? 'text-action-primary' : 'text-text-dim group-hover:text-action-primary'
      )} />

      {/* List info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={clsx(
            'text-sm truncate',
            isSelected ? 'font-medium text-text-main dark:text-dark-text-main' : 'text-text-muted dark:text-dark-text-main/70'
          )}>
            {list.name}
          </span>
          <span className="text-xs text-text-dim font-mono tabular-nums dark:text-dark-text-main/60">
            {list.repo_count}
          </span>
        </div>

        {/* Description */}
        {list.description && (
          <p className="text-[11px] text-text-dim truncate mt-0.5 dark:text-dark-text-main/60">
            {list.description}
          </p>
        )}
      </div>

      {/* Selection indicator */}
      <div className={clsx(
        'w-5 h-5 rounded flex items-center justify-center flex-shrink-0 transition-colors',
        isSelected
          ? 'bg-text-main text-bg-main'
          : 'bg-bg-hover text-transparent group-hover:bg-border-light dark:bg-dark-bg-sidebar dark:group-hover:bg-dark-border'
      )}>
        <Check className="w-3 h-3" />
      </div>
    </button>
  );
};

// ============================================================================
// StarListPanel Component
// ============================================================================

const StarListPanel: React.FC<StarListPanelProps> = ({
  collapsed = false,
  onToggleCollapsed,
  className,
}) => {
  const { t } = useTranslation();
  const {
    rawData,
    filteredData,
    filters,
    toggleStarList,
    clearStarListFilter,
  } = useGraph();

  // Get star lists with counts
  const starLists = useMemo(() => {
    if (!rawData || !rawData.star_lists) return [];

    // Sort by repo count descending
    return [...rawData.star_lists].sort((a, b) => b.repo_count - a.repo_count);
  }, [rawData]);

  const hasSelectedLists = filters.selectedStarLists?.size > 0;
  const selectedCount = filters.selectedStarLists?.size || 0;
  const totalLists = starLists.length;

  // Don't render if no star lists
  if (!rawData || starLists.length === 0) {
    return null;
  }

  return (
    <div className={clsx(
      'bg-bg-main/90 border border-border-light rounded-xl shadow-sm overflow-hidden backdrop-blur-sm',
      'transition-all duration-300',
      'dark:bg-dark-bg-main/90 dark:border-dark-border',
      className
    )}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-light dark:border-dark-border">
        <button
          type="button"
          onClick={onToggleCollapsed}
          className="flex items-center justify-between gap-2 flex-1 text-left hover:bg-bg-hover/50 rounded-md px-2 -ml-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:hover:bg-dark-bg-sidebar/70"
          aria-expanded={!collapsed}
          aria-controls="starlist-panel-list"
        >
          <div className="flex items-center gap-2">
            <FolderHeart className="w-4 h-4 text-action-primary" />
            <span className="text-sm font-medium text-text-main dark:text-dark-text-main">
              {t('graph.my_lists')}
            </span>
            <span className="text-xs text-text-dim dark:text-dark-text-main/60">
              ({hasSelectedLists ? `${selectedCount}/${totalLists}` : totalLists})
            </span>
          </div>
          {onToggleCollapsed && (
            collapsed ? (
              <ChevronDown className="w-4 h-4 text-text-dim dark:text-dark-text-main/60" />
            ) : (
              <ChevronUp className="w-4 h-4 text-text-dim dark:text-dark-text-main/60" />
            )
          )}
        </button>

        {hasSelectedLists && (
          <button
            onClick={() => clearStarListFilter?.()}
            className="p-1 rounded hover:bg-bg-hover text-text-dim hover:text-text-main transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:text-dark-text-main/60 dark:hover:text-dark-text-main dark:hover:bg-dark-bg-sidebar/70"
            title={t('common.clear_filter')}
            aria-label={t('common.clear_filter')}
            type="button"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Star list items */}
      {!collapsed && (
        <div id="starlist-panel-list" className="max-h-60 overflow-y-auto p-2 space-y-1">
          {starLists.map((list) => (
            <StarListItem
              key={list.id}
              list={list}
              isSelected={filters.selectedStarLists?.has(list.id) || false}
              onToggle={() => toggleStarList?.(list.id)}
            />
          ))}
        </div>
      )}

      {/* Footer with filter info */}
      {!collapsed && hasSelectedLists && (
        <div className="px-4 py-2 border-t border-border-light bg-bg-sidebar/50 dark:border-dark-border dark:bg-dark-bg-sidebar/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted dark:text-dark-text-main/70">
              {t('graph.showing_repos', { count: filteredData?.total_nodes || 0 })}
            </span>
            <button
              onClick={clearStarListFilter}
              className="text-xs text-action-primary hover:text-action-hover transition-colors"
            >
              {t('common.show_all')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default StarListPanel;
