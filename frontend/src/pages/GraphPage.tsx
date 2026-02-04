import { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Sidebar } from '../components/layout/Sidebar';
import Graph2D from '../components/graph/Graph2D';
import Graph3D from '../components/graph/Graph3D';
import Timeline from '../components/graph/Timeline';
import ClusterPanel from '../components/graph/ClusterPanel';
import { SearchInput } from '../components/ui/SearchInput';
import { RepoDetailsPanel } from '../components/graph/RepoDetailsPanel';
import { startStarSync, getSyncStatus, startEmbedding, startClustering } from '../api/sync';
import { useGraph } from '../contexts/GraphContext';
import { Loader2, Grid3X3, Box, Filter, X } from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

type ViewMode = '2d' | '3d';

// ============================================================================
// Component
// ============================================================================

const GraphPage = () => {
  const { t } = useTranslation();

  // Global state
  const {
    filteredData,
    rawData,
    selectedNode,
    setSelectedNode,
    filters,
    setSearchQuery,
    clearFilters,
    loading,
    syncing,
    syncStep,
    setSyncing,
    setSyncStep,
    refreshData,
  } = useGraph();

  // Local UI state
  const [viewMode, setViewMode] = useState<ViewMode>('2d');
  const [clusterPanelCollapsed, setClusterPanelCollapsed] = useState(false);
  const [showFilters, setShowFilters] = useState(true);

  // Close details panel
  const handleCloseDetails = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  // Handle search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
  }, [setSearchQuery]);

  // Wait for async task to complete
  const waitForTaskComplete = async (taskId: number): Promise<boolean> => {
    return new Promise((resolve) => {
      const poll = async () => {
        try {
          const status = await getSyncStatus(taskId);
          if (status.status === 'completed') {
            resolve(true);
          } else if (status.status === 'failed') {
            console.error('Task failed:', status.error_message);
            resolve(false);
          } else {
            setTimeout(poll, 2000);
          }
        } catch (err) {
          console.error('Poll status error:', err);
          resolve(false);
        }
      };
      setTimeout(poll, 1000);
    });
  };

  // Handle sync stars
  const handleSyncStars = async () => {
    try {
      setSyncing(true);

      // Step 1: Sync stars
      setSyncStep(t('dashboard.sync_step_stars'));
      const starsResult = await startStarSync('incremental');
      console.log('Stars sync started:', starsResult);

      const starsSuccess = await waitForTaskComplete(starsResult.task_id);
      if (!starsSuccess) {
        throw new Error(t('errors.sync_stars_failed'));
      }

      // Step 2: Compute embeddings
      setSyncStep(t('dashboard.sync_step_embedding'));
      const embeddingResult = await startEmbedding();
      console.log('Embedding started:', embeddingResult);

      const embeddingSuccess = await waitForTaskComplete(embeddingResult.task_id);
      if (!embeddingSuccess) {
        throw new Error(t('errors.embedding_failed'));
      }

      // Step 3: Run clustering
      setSyncStep(t('dashboard.sync_step_clustering'));
      const clusterResult = await startClustering();
      console.log('Clustering started:', clusterResult);

      const clusterSuccess = await waitForTaskComplete(clusterResult.task_id);
      if (!clusterSuccess) {
        throw new Error(t('errors.clustering_failed'));
      }

      // Refresh data
      setSyncStep('');
      await refreshData();

    } catch (error) {
      console.error('Sync failed:', error);
      alert(error instanceof Error ? error.message : t('errors.sync_failed'));
    } finally {
      setSyncing(false);
      setSyncStep('');
    }
  };

  // Check if any filters are active
  const hasActiveFilters = filters.selectedClusters.size > 0 ||
                          filters.searchQuery.trim() !== '' ||
                          filters.timeRange !== null ||
                          filters.minStars > 0 ||
                          filters.languages.size > 0;

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main">
      <Sidebar />

      <main className="flex-1 ml-60 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between h-14 px-6 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 transition-all">
          <div className="flex items-center gap-3 select-none">
            <h2 className="text-base font-semibold text-text-main tracking-tight">
              {t('sidebar.graph')}
            </h2>
            <div className="h-4 w-[1px] bg-border-light mx-1" />
            <span className="text-sm text-text-muted">
              {filteredData?.total_nodes !== undefined
                ? t('dashboard.subtitle', { count: filteredData.total_nodes })
                : t('dashboard.subtitle_infinite')}
            </span>
            {rawData && filteredData && rawData.total_nodes !== filteredData.total_nodes && (
              <span className="text-xs text-text-dim">
                / {rawData.total_nodes} {t('common.total')}
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="w-56 transition-all focus-within:w-64">
              <SearchInput
                onSearch={handleSearch}
                value={filters.searchQuery}
              />
            </div>

            {/* View mode toggle */}
            <div className="flex items-center bg-bg-sidebar rounded-md p-0.5 border border-border-light">
              <button
                onClick={() => setViewMode('2d')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-all ${
                  viewMode === '2d'
                    ? 'bg-white shadow-sm text-text-main font-medium'
                    : 'text-text-muted hover:text-text-main'
                }`}
              >
                <Grid3X3 className="w-4 h-4" />
                <span>2D</span>
              </button>
              <button
                onClick={() => setViewMode('3d')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-all ${
                  viewMode === '3d'
                    ? 'bg-white shadow-sm text-text-main font-medium'
                    : 'text-text-muted hover:text-text-main'
                }`}
              >
                <Box className="w-4 h-4" />
                <span>3D</span>
              </button>
            </div>

            {/* Filter toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`h-9 px-3 rounded-md text-sm font-medium transition-all flex items-center gap-2 border ${
                showFilters
                  ? 'bg-bg-sidebar border-border-light text-text-main'
                  : 'border-transparent text-text-muted hover:text-text-main hover:bg-bg-hover'
              }`}
            >
              <Filter className="w-4 h-4" />
              {hasActiveFilters && (
                <span className="w-2 h-2 bg-action-primary rounded-full" />
              )}
            </button>

            {/* Sync button */}
            <button
              onClick={handleSyncStars}
              disabled={syncing}
              className={`h-9 px-4 rounded-md text-sm font-medium transition-all flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black/20 hover:shadow-md active:scale-95 ${
                syncing
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-black text-white hover:bg-gray-800 shadow-sm'
              }`}
            >
              {syncing && <Loader2 className="animate-spin h-4 w-4" />}
              {syncing ? (syncStep || t('dashboard.syncing')) : t('dashboard.sync_button')}
            </button>
          </div>
        </header>

        {/* Content Area */}
        <section className="flex-1 relative flex">
          {/* Filters sidebar */}
          {showFilters && (
            <aside className="w-72 border-r border-border-light bg-bg-sidebar/50 p-4 space-y-4 overflow-y-auto">
              {/* Clear all filters */}
              {hasActiveFilters && (
                <button
                  onClick={clearFilters}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm text-text-muted hover:text-text-main hover:bg-bg-hover rounded-md transition-colors"
                >
                  <X className="w-4 h-4" />
                  {t('common.clear_all_filters')}
                </button>
              )}

              {/* Cluster panel */}
              <ClusterPanel
                collapsed={clusterPanelCollapsed}
                onToggleCollapsed={() => setClusterPanelCollapsed(!clusterPanelCollapsed)}
              />

              {/* Timeline */}
              <Timeline />
            </aside>
          )}

          {/* Graph Container */}
          <div className="flex-1 relative bg-white">
            {viewMode === '2d' ? <Graph2D /> : <Graph3D />}

            {/* Details panel */}
            {selectedNode && (
              <RepoDetailsPanel node={selectedNode} onClose={handleCloseDetails} />
            )}

            {/* Loading overlay */}
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/50 z-50">
                <Loader2 className="animate-spin h-8 w-8 text-action-primary" />
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
};

export default GraphPage;
