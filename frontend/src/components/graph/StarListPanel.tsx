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
        'hover:bg-white hover:border-border-light/70 group border-transparent',
        isSelected && 'bg-pink-50 ring-1 ring-pink-200 border-pink-200/70 shadow-sm'
      )}
    >
      {/* Icon */}
      <FolderHeart className={clsx(
        'w-4 h-4 flex-shrink-0',
        isSelected ? 'text-pink-500' : 'text-text-dim group-hover:text-pink-400'
      )} />

      {/* List info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={clsx(
            'text-sm truncate',
            isSelected ? 'font-medium text-pink-700' : 'text-text-muted'
          )}>
            {list.name}
          </span>
          <span className="text-xs text-text-dim font-mono tabular-nums">
            {list.repo_count}
          </span>
        </div>

        {/* Description */}
        {list.description && (
          <p className="text-[11px] text-text-dim truncate mt-0.5">
            {list.description}
          </p>
        )}
      </div>

      {/* Selection indicator */}
      <div className={clsx(
        'w-5 h-5 rounded flex items-center justify-center flex-shrink-0 transition-colors',
        isSelected
          ? 'bg-pink-500 text-white'
          : 'bg-gray-100 text-transparent group-hover:bg-gray-200'
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
      'bg-white/90 border border-border-light rounded-xl shadow-sm overflow-hidden backdrop-blur-sm',
      'transition-all duration-300',
      className
    )}>
      {/* Header */}
      <div
        className={clsx(
          'flex items-center justify-between px-4 py-3 border-b border-border-light',
          'cursor-pointer hover:bg-bg-hover/50 transition-colors'
        )}
        onClick={onToggleCollapsed}
      >
        <div className="flex items-center gap-2">
          <FolderHeart className="w-4 h-4 text-pink-500" />
          <span className="text-sm font-medium text-text-main">
            {t('graph.my_lists')}
          </span>
          <span className="text-xs text-text-dim">
            ({hasSelectedLists ? `${selectedCount}/${totalLists}` : totalLists})
          </span>
        </div>

        <div className="flex items-center gap-2">
          {hasSelectedLists && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                clearStarListFilter?.();
              }}
              className="p-1 rounded hover:bg-bg-hover text-text-dim hover:text-text-main transition-colors"
              title={t('common.clear_filter')}
            >
              <X className="w-4 h-4" />
            </button>
          )}
          {onToggleCollapsed && (
            collapsed ? (
              <ChevronDown className="w-4 h-4 text-text-dim" />
            ) : (
              <ChevronUp className="w-4 h-4 text-text-dim" />
            )
          )}
        </div>
      </div>

      {/* Star list items */}
      {!collapsed && (
        <div className="max-h-60 overflow-y-auto p-2 space-y-1">
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
        <div className="px-4 py-2 border-t border-border-light bg-pink-50/50">
          <div className="flex items-center justify-between">
            <span className="text-xs text-pink-600">
              {t('graph.showing_repos', { count: filteredData?.total_nodes || 0 })}
            </span>
            <button
              onClick={clearStarListFilter}
              className="text-xs text-pink-600 hover:text-pink-700 transition-colors"
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
