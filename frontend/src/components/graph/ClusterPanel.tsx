import React, { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Layers, Check, X, ChevronDown, ChevronUp } from 'lucide-react';
import { useGraph } from '../../contexts/GraphContext';
import { ClusterInfo } from '../../types';

// ============================================================================
// Types
// ============================================================================

interface ClusterPanelProps {
  /** Collapsed state */
  collapsed?: boolean;
  /** Toggle collapsed callback */
  onToggleCollapsed?: () => void;
  /** Additional class names */
  className?: string;
}

interface ClusterItemProps {
  cluster: ClusterInfo;
  isSelected: boolean;
  nodeCount: number;
  onToggle: () => void;
}

// ============================================================================
// ClusterItem Component
// ============================================================================

const ClusterItem: React.FC<ClusterItemProps> = ({
  cluster,
  isSelected,
  nodeCount,
  onToggle,
}) => {
  return (
    <button
      onClick={onToggle}
      className={clsx(
        'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left border',
        'hover:bg-bg-hover hover:border-border-light/70 group border-transparent dark:hover:bg-dark-bg-sidebar/70',
        isSelected && 'bg-bg-main ring-1 ring-action-primary/20 border-border-light shadow-sm dark:bg-dark-bg-main dark:border-dark-border'
      )}
    >
      {/* Color indicator */}
      <div
        className="w-3 h-3 rounded-full flex-shrink-0 ring-1 ring-black/10"
        style={{ backgroundColor: cluster.color }}
      />

      {/* Cluster info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={clsx(
            'text-sm truncate',
            isSelected ? 'font-medium text-text-main dark:text-dark-text-main' : 'text-text-muted dark:text-dark-text-main/70'
          )}>
            {cluster.name || `Cluster ${cluster.id}`}
          </span>
          <span className="text-xs text-text-dim font-mono tabular-nums dark:text-dark-text-main/60">
            {nodeCount}
          </span>
        </div>

        {/* Keywords */}
        {cluster.keywords && cluster.keywords.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {cluster.keywords.slice(0, 3).map((keyword, idx) => (
              <span
                key={idx}
                className="text-[10px] px-1.5 py-0.5 bg-bg-hover text-text-muted rounded-full dark:bg-dark-bg-sidebar dark:text-dark-text-main/70"
              >
                {keyword}
              </span>
            ))}
            {cluster.keywords.length > 3 && (
              <span className="text-[10px] text-text-dim dark:text-dark-text-main/60">
                +{cluster.keywords.length - 3}
              </span>
            )}
          </div>
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
// ClusterPanel Component
// ============================================================================

const ClusterPanel: React.FC<ClusterPanelProps> = ({
  collapsed = false,
  onToggleCollapsed,
  className,
}) => {
  const { t } = useTranslation();
  const {
    rawData,
    filteredData,
    filters,
    toggleCluster,
    clearClusterFilter,
  } = useGraph();

  // Get clusters with node counts
  const clustersWithCounts = useMemo(() => {
    if (!rawData) return [];

    // Count nodes per cluster from raw data
    const nodeCounts = new Map<number, number>();
    rawData.nodes.forEach(node => {
      if (node.cluster_id != null) {
        nodeCounts.set(node.cluster_id, (nodeCounts.get(node.cluster_id) || 0) + 1);
      }
    });

    // Sort clusters by node count (descending)
    return rawData.clusters
      .map(cluster => ({
        cluster,
        nodeCount: nodeCounts.get(cluster.id) || 0,
      }))
      .filter(c => c.nodeCount > 0)
      .sort((a, b) => b.nodeCount - a.nodeCount);
  }, [rawData]);

  const hasSelectedClusters = filters.selectedClusters.size > 0;
  const selectedCount = filters.selectedClusters.size;
  const totalClusters = clustersWithCounts.length;

  // Empty state
  if (!rawData || clustersWithCounts.length === 0) {
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
          aria-controls="cluster-panel-list"
        >
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-text-muted dark:text-dark-text-main/70" />
            <span className="text-sm font-medium text-text-main dark:text-dark-text-main">
              {t('graph.clusters')}
            </span>
            <span className="text-xs text-text-dim dark:text-dark-text-main/60">
              ({hasSelectedClusters ? `${selectedCount}/${totalClusters}` : totalClusters})
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

        {hasSelectedClusters && (
          <button
            onClick={clearClusterFilter}
            className="p-1 rounded hover:bg-bg-hover text-text-dim hover:text-text-main transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:text-dark-text-main/60 dark:hover:text-dark-text-main dark:hover:bg-dark-bg-sidebar/70"
            title={t('common.clear_filter')}
            aria-label={t('common.clear_filter')}
            type="button"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Cluster list */}
      {!collapsed && (
        <div id="cluster-panel-list" className="max-h-80 overflow-y-auto p-2 space-y-1">
          {clustersWithCounts.map(({ cluster, nodeCount }) => (
            <ClusterItem
              key={cluster.id}
              cluster={cluster}
              isSelected={filters.selectedClusters.has(cluster.id)}
              nodeCount={nodeCount}
              onToggle={() => toggleCluster(cluster.id)}
            />
          ))}
        </div>
      )}

      {/* Footer with filter info */}
      {!collapsed && hasSelectedClusters && (
        <div className="px-4 py-2 border-t border-border-light bg-bg-sidebar/50 dark:border-dark-border dark:bg-dark-bg-sidebar/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted dark:text-dark-text-main/70">
              {t('graph.showing_repos', { count: filteredData?.total_nodes || 0 })}
            </span>
            <button
              onClick={clearClusterFilter}
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

export default ClusterPanel;
