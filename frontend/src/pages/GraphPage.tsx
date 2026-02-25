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
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { useGraph } from '../contexts/GraphContext';
import { Filter, X } from 'lucide-react';

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
    setSelectedClusters,
    setSearchQuery,
    clearFilters,
  } = useGraph();

  // Handle URL parameter for node selection
  useEffect(() => {
    const nodeId = searchParams.get('node');
    if (!nodeId || !rawData?.nodes) return;

    const parsedNodeId = Number.parseInt(nodeId, 10);
    if (Number.isFinite(parsedNodeId)) {
      const node = rawData.nodes.find(n => n.id === parsedNodeId);
      if (node) {
        setSelectedNode(node);
      }
    }

    // Clear URL parameter regardless of whether node exists to avoid stale links.
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete('node');
    setSearchParams(nextParams, { replace: true });
  }, [searchParams, rawData, setSelectedNode, setSearchParams]);

  // Handle URL parameter for cluster filter
  useEffect(() => {
    const clusterIdParam = searchParams.get('cluster');
    if (!clusterIdParam || !rawData) return;

    const clusterId = parseInt(clusterIdParam, 10);
    if (Number.isFinite(clusterId) && rawData.clusters.some(cluster => cluster.id === clusterId)) {
      clearFilters();
      setSelectedClusters([clusterId]);
    }

    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete('cluster');
    setSearchParams(nextParams, { replace: true });
  }, [searchParams, rawData, setSearchParams, clearFilters, setSelectedClusters]);

  // Local UI state
  const [clusterPanelCollapsed, setClusterPanelCollapsed] = useState(false);
  const [starListPanelCollapsed, setStarListPanelCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(
    typeof window !== 'undefined' ? window.innerWidth < 1024 : false
  );
  const [showFilters, setShowFilters] = useState(
    typeof window !== 'undefined' ? window.innerWidth >= 1024 : true
  );

  useEffect(() => {
    const handleResize = () => {
      const nextIsMobile = window.innerWidth < 1024;
      setIsMobile(nextIsMobile);
      setShowFilters((prev) => (nextIsMobile ? prev : true));
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Keep details panel open even if the node is filtered out (ghost mode)
  // We no longer force setSelectedNode(null) here because ghost nodes can still be selected.

  // Close details panel
  const handleCloseDetails = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  // Handle search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
  }, [setSearchQuery]);

  // Check if any filters are active
  const hasActiveFilters = filters.selectedClusters.size > 0 ||
                          filters.selectedStarLists.size > 0 ||
                          filters.searchQuery.trim() !== '' ||
                          filters.timeRange !== null ||
                          filters.minStars > 0 ||
                          filters.languages.size > 0;

  return (
    <div className="flex h-screen overflow-hidden bg-bg-main text-text-main">
      <Sidebar />

      <main className="flex-1 flex flex-col min-w-0" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        {/* Header */}
        <header className="flex items-center justify-between h-14 px-4 sm:px-6 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 transition-all">
          <div className="flex items-center gap-3 select-none">
            <h2 className="text-base font-semibold text-text-main tracking-tight">
              {t('sidebar.graph')}
            </h2>
            <div className="hidden sm:block h-4 w-[1px] bg-border-light mx-1" />
            <span className="hidden sm:inline text-sm text-text-muted">
              {filteredData?.total_nodes !== undefined
                ? t('dashboard.subtitle', { count: filteredData.total_nodes })
                : t('dashboard.subtitle_infinite')}
            </span>
            {rawData && filteredData && rawData.total_nodes !== filteredData.total_nodes && (
              <span className="hidden sm:inline text-xs text-text-dim">
                / {rawData.total_nodes} {t('common.total')}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            <LanguageSwitch />

            {/* Search */}
            <div className="w-36 sm:w-56 transition-all sm:focus-within:w-64">
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

          </div>
        </header>

        {/* Content Area */}
        <section className="flex-1 relative flex overflow-hidden">
          {isMobile && showFilters && (
            <button
              className="absolute inset-0 z-20 bg-black/25"
              onClick={() => setShowFilters(false)}
              aria-label={t('common.close')}
            />
          )}

          {/* Filters sidebar */}
          {showFilters && (
            <aside
              className={`${
                isMobile
                  ? 'absolute inset-y-0 left-0 w-[85vw] max-w-sm'
                  : 'w-72 flex-shrink-0'
              } border-r border-border-light bg-bg-sidebar/90 backdrop-blur-sm p-4 space-y-4 overflow-y-auto z-30 relative`}
            >
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
          <div className="flex-1 min-w-0 relative bg-white/80 flex flex-row overflow-hidden">
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
    </div>
  );
};

export default GraphPage;
