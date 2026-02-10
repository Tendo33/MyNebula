import React, { useMemo, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { clsx } from 'clsx';
import { Layers, Check, X, ChevronDown, ChevronUp } from 'lucide-react';
import { useGraph } from '../../contexts/GraphContext';
import { ClusterInfo } from '../../types';
import { startClustering, getSyncStatus } from '../../api/sync';

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
        'w-full flex items-center gap-3 px-3 py-2.5 rounded-md transition-all text-left',
        'hover:bg-bg-hover group',
        isSelected && 'bg-bg-hover ring-1 ring-border-light'
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
                className="text-[10px] px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded"
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
    settings,
    updateSettings,
    refreshData,
  } = useGraph();

  const [reclustering, setReclustering] = useState(false);

  const handleRecluster = useCallback(async () => {
    if (reclustering) return;

    try {
      setReclustering(true);
      const started = await startClustering(
        true,
        settings.maxClusters,
        settings.minClusters
      );

      // Poll until clustering completes, then refresh graph data
      // (Keeps this lightweight vs. the full 4-step sync flow)
      for (;;) {
        const status = await getSyncStatus(started.task_id);
        if (status.status === 'completed') break;
        if (status.status === 'failed') throw new Error(status.error_message || 'Clustering failed');
        await new Promise((r) => setTimeout(r, 1200));
      }

      await refreshData();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('Re-cluster failed:', e);
    } finally {
      setReclustering(false);
    }
  }, [reclustering, refreshData, settings.maxClusters, settings.minClusters]);

  // Get clusters with node counts
  const clustersWithCounts = useMemo(() => {
    if (!rawData) return [];

    // Count nodes per cluster from raw data
    const nodeCounts = new Map<number, number>();
    rawData.nodes.forEach(node => {
      if (node.cluster_id !== undefined) {
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
      'bg-white border border-border-light rounded-lg shadow-sm overflow-hidden',
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

      {/* Controls */}
      {!collapsed && (
        <div className="px-4 py-3 border-b border-border-light bg-bg-sidebar/40">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-muted">
                  {t('graph.max_clusters')}
                </span>
                <span className="text-xs font-mono tabular-nums text-text-dim">
                  {settings.maxClusters}
                </span>
              </div>
              <input
                type="range"
                min={2}
                max={20}
                step={1}
                value={settings.maxClusters}
                onChange={(e) => {
                  const nextMax = Number(e.target.value);
                  if (nextMax < settings.minClusters) {
                    // Only link when bounds would be invalid (max < min)
                    updateSettings({ maxClusters: nextMax, minClusters: nextMax });
                    return;
                  }
                  updateSettings({ maxClusters: nextMax });
                }}
                className="w-full mt-2"
              />

              <div className="flex items-center justify-between mt-3">
                <span className="text-xs text-text-muted">
                  {t('graph.min_clusters', { defaultValue: '最小簇数（越大越细）' })}
                </span>
                <span className="text-xs font-mono tabular-nums text-text-dim">
                  {settings.minClusters}
                </span>
              </div>
              <input
                type="range"
                min={2}
                max={20}
                step={1}
                value={settings.minClusters}
                onChange={(e) => {
                  const nextMin = Number(e.target.value);
                  if (nextMin > settings.maxClusters) {
                    // Only link when bounds would be invalid (min > max)
                    updateSettings({ minClusters: nextMin, maxClusters: nextMin });
                    return;
                  }
                  updateSettings({ minClusters: nextMin });
                }}
                className="w-full mt-2"
              />
            </div>

            <button
              onClick={handleRecluster}
              disabled={reclustering}
              className={clsx(
                'h-8 px-3 rounded-md text-xs font-medium border transition-colors flex-shrink-0',
                reclustering
                  ? 'bg-gray-100 text-gray-400 border-border-light cursor-not-allowed'
                  : 'bg-white text-text-main border-border-light hover:bg-bg-hover'
              )}
              title={t('graph.recluster_hint')}
            >
              {reclustering ? t('graph.reclustering') : t('graph.recluster')}
            </button>
          </div>
        </div>
      )}

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
