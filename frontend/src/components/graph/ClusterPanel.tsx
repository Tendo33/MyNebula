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
        'hover:bg-white hover:border-border-light/70 group border-transparent',
        isSelected && 'bg-white ring-1 ring-action-primary/20 border-border-light shadow-sm'
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
            isSelected ? 'font-medium text-text-main' : 'text-text-muted'
          )}>
            {cluster.name || `Cluster ${cluster.id}`}
          </span>
          <span className="text-xs text-text-dim font-mono tabular-nums">
            {nodeCount}
          </span>
        </div>

        {/* Keywords */}
        {cluster.keywords && cluster.keywords.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {cluster.keywords.slice(0, 3).map((keyword, idx) => (
              <span
                key={idx}
                className="text-[10px] px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded-full"
              >
                {keyword}
              </span>
            ))}
            {cluster.keywords.length > 3 && (
              <span className="text-[10px] text-text-dim">
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
          ? 'bg-text-main text-white'
          : 'bg-gray-100 text-transparent group-hover:bg-gray-200'
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
          <Layers className="w-4 h-4 text-text-muted" />
          <span className="text-sm font-medium text-text-main">
            {t('graph.clusters')}
          </span>
          <span className="text-xs text-text-dim">
            ({hasSelectedClusters ? `${selectedCount}/${totalClusters}` : totalClusters})
          </span>
        </div>

        <div className="flex items-center gap-2">
          {hasSelectedClusters && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                clearClusterFilter();
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

      {/* Cluster list */}
      {!collapsed && (
        <div className="max-h-80 overflow-y-auto p-2 space-y-1">
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
        <div className="px-4 py-2 border-t border-border-light bg-bg-sidebar/50">
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted">
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
