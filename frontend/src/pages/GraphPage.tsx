import { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useSearchParams } from 'react-router-dom';
import { Sidebar } from '../components/layout/Sidebar';
import Graph2D from '../components/graph/Graph2D';
import Timeline from '../components/graph/Timeline';
import ClusterPanel from '../components/graph/ClusterPanel';
import StarListPanel from '../components/graph/StarListPanel';
import { SearchInput } from '../components/ui/SearchInput';
import { RepoDetailsPanel } from '../components/graph/RepoDetailsPanel';
import { SyncProgress, SyncStep } from '../components/ui/SyncProgress';
import { startStarSync, getSyncStatus, startEmbedding, startClustering, startSummaries } from '../api/sync';
import { useGraph } from '../contexts/GraphContext';
import { Loader2, Filter, X } from 'lucide-react';

// ============================================================================
// Component
// ============================================================================

const GraphPage = () => {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();

  // Global state
  const {
    filteredData,
    rawData,
    selectedNode,
    setSelectedNode,
    filters,
    setSearchQuery,
    clearFilters,

    syncing,
    syncStep,
    setSyncing,
    setSyncStep,
    refreshData,
  } = useGraph();

  // Handle URL parameter for node selection
  useEffect(() => {
    const nodeId = searchParams.get('node');
    if (nodeId && rawData?.nodes) {
      const node = rawData.nodes.find(n => n.id === parseInt(nodeId, 10));
      if (node) {
        setSelectedNode(node);
        // Clear the URL parameter after selecting
        searchParams.delete('node');
        setSearchParams(searchParams, { replace: true });
      }
    }
  }, [searchParams, rawData, setSelectedNode, setSearchParams]);

  // Local UI state
  const [clusterPanelCollapsed, setClusterPanelCollapsed] = useState(false);
  const [starListPanelCollapsed, setStarListPanelCollapsed] = useState(false);
  const [showFilters, setShowFilters] = useState(true);

  // Sync progress state
  const [showSyncProgress, setShowSyncProgress] = useState(false);
  const [syncSteps, setSyncSteps] = useState<SyncStep[]>([
    { id: 'stars', label: 'Fetching GitHub Stars', description: 'Syncing your starred repositories', status: 'pending' },
    { id: 'summaries', label: 'Generating AI Summaries', description: 'Creating intelligent descriptions and tags', status: 'pending' },
    { id: 'embeddings', label: 'Computing Embeddings', description: 'Building semantic representations', status: 'pending' },
    { id: 'clustering', label: 'Clustering Repositories', description: 'Organizing into knowledge groups', status: 'pending' },
  ]);

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

  // Update step status helper
  const updateStepStatus = (stepId: string, status: SyncStep['status'], error?: string) => {
    setSyncSteps(prev => prev.map(step =>
      step.id === stepId ? { ...step, status, error } : step
    ));
  };

  // Reset all steps
  const resetSteps = () => {
    setSyncSteps([
      { id: 'stars', label: 'Fetching GitHub Stars', description: 'Syncing your starred repositories', status: 'pending' },
      { id: 'summaries', label: 'Generating AI Summaries', description: 'Creating intelligent descriptions and tags', status: 'pending' },
      { id: 'embeddings', label: 'Computing Embeddings', description: 'Building semantic representations', status: 'pending' },
      { id: 'clustering', label: 'Clustering Repositories', description: 'Organizing into knowledge groups', status: 'pending' },
    ]);
  };

  // Handle sync stars with progress tracking
  const handleSyncStars = async () => {
    try {
      setSyncing(true);
      setShowSyncProgress(true);
      resetSteps();

      // Step 1: Sync stars
      updateStepStatus('stars', 'running');
      const starsResult = await startStarSync('incremental');

      const starsSuccess = await waitForTaskComplete(starsResult.task_id);
      if (!starsSuccess) {
        updateStepStatus('stars', 'failed', 'Failed to sync GitHub stars');
        throw new Error(t('errors.sync_stars_failed'));
      }
      updateStepStatus('stars', 'completed');

      // Step 2: Generate summaries (AI enhancement)
      updateStepStatus('summaries', 'running');
      try {
        const summariesResult = await startSummaries();
        const summariesSuccess = await waitForTaskComplete(summariesResult.task_id);
        if (!summariesSuccess) {
          console.warn('Summaries generation had issues, continuing...');
        }
      } catch (e) {
        console.warn('Summaries generation skipped:', e);
      }
      updateStepStatus('summaries', 'completed');

      // Step 3: Compute embeddings
      updateStepStatus('embeddings', 'running');
      const embeddingResult = await startEmbedding();

      const embeddingSuccess = await waitForTaskComplete(embeddingResult.task_id);
      if (!embeddingSuccess) {
        updateStepStatus('embeddings', 'failed', 'Failed to compute embeddings');
        throw new Error(t('errors.embedding_failed'));
      }
      updateStepStatus('embeddings', 'completed');

      // Step 4: Run clustering
      updateStepStatus('clustering', 'running');
      const clusterResult = await startClustering();

      const clusterSuccess = await waitForTaskComplete(clusterResult.task_id);
      if (!clusterSuccess) {
        updateStepStatus('clustering', 'failed', 'Failed to cluster repositories');
        throw new Error(t('errors.clustering_failed'));
      }
      updateStepStatus('clustering', 'completed');

      // Refresh data
      await refreshData();

    } catch (error) {
      console.error('Sync failed:', error);
    } finally {
      setSyncing(false);
      setSyncStep('');
    }
  };

  // Handle sync progress close
  const handleSyncProgressClose = () => {
    setShowSyncProgress(false);
    if (!syncing) {
      resetSteps();
    }
  };

  // Check if any filters are active
  const hasActiveFilters = filters.selectedClusters.size > 0 ||
                          filters.selectedStarLists.size > 0 ||
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
            <aside className="w-72 flex-shrink-0 border-r border-border-light bg-bg-sidebar p-4 space-y-4 overflow-y-auto z-30 relative">
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

              {/* User's Star Lists panel */}
              <StarListPanel
                collapsed={starListPanelCollapsed}
                onToggleCollapsed={() => setStarListPanelCollapsed(!starListPanelCollapsed)}
              />

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
          <div className="flex-1 min-w-0 relative bg-white flex flex-row overflow-hidden">
            <div className="flex-1 relative min-w-0 h-full">
              <Graph2D />
            </div>

            {/* Details panel - Sidebar style */}
            {selectedNode && (
              <RepoDetailsPanel node={selectedNode} onClose={handleCloseDetails} />
            )}

            {/* Loading overlay */}

          </div>
        </section>
      </main>

      {/* Sync Progress Modal */}
      <SyncProgress
        isOpen={showSyncProgress}
        onClose={handleSyncProgressClose}
        steps={syncSteps}
        title={t('sync.title', 'Syncing Data')}
        canClose={!syncing}
      />
    </div>
  );
};

export default GraphPage;
