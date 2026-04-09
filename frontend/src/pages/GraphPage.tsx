import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
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
  const lastAppliedSearchRef = useRef<string | null>(null);
  const searchSignature = useMemo(() => searchParams.toString(), [searchParams]);
  const urlNodeId = useMemo(() => searchParams.get('node'), [searchParams]);
  const urlClusterIds = useMemo(() => searchParams.get('clusters'), [searchParams]);
  const urlClusterId = useMemo(() => searchParams.get('cluster'), [searchParams]);
  const urlLanguage = useMemo(() => searchParams.get('language'), [searchParams]);
  const urlQuery = useMemo(() => searchParams.get('q') ?? searchParams.get('tag') ?? '', [searchParams]);

  // Global state
  const {
    filteredData,
    rawData,
    loadData,
    edgesLoading,
    error,
    selectedNode,
    setSelectedNode,
    filters,
    setSelectedClusters,
    setSearchQuery,
    setSelectedLanguages,
    clearFilters,
    retryEdgeLoading,
  } = useGraph();

  // Apply URL state to graph filters/details.
  useEffect(() => {
    if (!rawData) return;
    if (lastAppliedSearchRef.current === searchSignature) {
      return;
    }
    lastAppliedSearchRef.current = searchSignature;

    if (urlNodeId) {
      const parsedNodeId = Number.parseInt(urlNodeId, 10);
      if (Number.isFinite(parsedNodeId)) {
        const node = rawData.nodes.find(n => n.id === parsedNodeId);
        if (node && selectedNode?.id !== node.id) {
          setSelectedNode(node);
        } else if (!node && selectedNode) {
          setSelectedNode(null);
        }
      } else if (selectedNode) {
        setSelectedNode(null);
      }
    } else if (selectedNode) {
      setSelectedNode(null);
    }

    const parsedClusterIds = urlClusterIds
      ? urlClusterIds
          .split(',')
          .map((value) => Number.parseInt(value, 10))
          .filter((value) => Number.isFinite(value))
      : urlClusterId
        ? [Number.parseInt(urlClusterId, 10)].filter((value) => Number.isFinite(value))
        : [];
    const validClusterIds = parsedClusterIds.filter((clusterId) =>
      rawData.clusters.some((cluster) => cluster.id === clusterId)
    );
    const currentClusters = Array.from(filters.selectedClusters).sort((left, right) => left - right);
    if (currentClusters.join(',') !== validClusterIds.join(',')) {
      setSelectedClusters(validClusterIds);
    }

    if (filters.searchQuery !== urlQuery) {
      setSearchQuery(urlQuery);
    }

    const currentLanguages = Array.from(filters.languages);
    if (urlLanguage) {
      if (currentLanguages.length !== 1 || currentLanguages[0] !== urlLanguage) {
        setSelectedLanguages([urlLanguage]);
      }
    } else if (currentLanguages.length > 0) {
      setSelectedLanguages([]);
    }
  }, [
    filters.languages,
    filters.searchQuery,
    filters.selectedClusters,
    rawData,
    searchSignature,
    selectedNode,
    setSearchQuery,
    setSelectedClusters,
    setSelectedLanguages,
    setSelectedNode,
    urlClusterId,
    urlClusterIds,
    urlLanguage,
    urlNodeId,
    urlQuery,
  ]);

  useEffect(() => {
    const nextParams = new URLSearchParams();

    if (selectedNode) {
      nextParams.set('node', String(selectedNode.id));
    }
    if (filters.selectedClusters.size === 1) {
      nextParams.set('cluster', String(Array.from(filters.selectedClusters)[0]));
    } else if (filters.selectedClusters.size > 1) {
      nextParams.set(
        'clusters',
        Array.from(filters.selectedClusters).sort((left, right) => left - right).join(',')
      );
    }
    if (filters.languages.size === 1) {
      nextParams.set('language', Array.from(filters.languages)[0]);
    }
    if (filters.searchQuery.trim()) {
      nextParams.set('q', filters.searchQuery.trim());
    }

    const nextSignature = nextParams.toString();
    if (nextSignature !== searchSignature) {
      lastAppliedSearchRef.current = nextSignature;
      setSearchParams(nextParams, { replace: true });
    }
  }, [filters.languages, filters.searchQuery, filters.selectedClusters, searchSignature, selectedNode, setSearchParams]);

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
    <div className="flex h-screen overflow-hidden bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main">
      <Sidebar />

      <main className="flex-1 flex flex-col min-w-0" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        {/* Header */}
        <header className="flex items-center justify-between h-14 px-4 sm:px-6 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 transition-all dark:bg-dark-bg-main/95 dark:border-dark-border">
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
            {edgesLoading && (
              <span className="hidden sm:inline text-xs text-text-dim">
                · {t('sync.loading', 'Loading')} edges...
              </span>
            )}
            {error && (
              <button
                onClick={() => {
                  void Promise.all([loadData(), retryEdgeLoading()]);
                }}
                className="hidden sm:inline text-xs text-red-600 hover:underline"
              >
                {t('common.retry')}
              </button>
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
              aria-expanded={showFilters}
              aria-controls="graph-filters-panel"
              aria-label={t('common.filter', 'Filters')}
              className={`h-9 px-3 rounded-md text-sm font-medium transition-all flex items-center gap-2 border ${
                showFilters
                  ? 'bg-bg-sidebar border-border-light text-text-main dark:bg-dark-bg-sidebar dark:border-dark-border dark:text-dark-text-main'
                  : 'border-transparent text-text-muted hover:text-text-main hover:bg-bg-hover dark:text-dark-text-main/70 dark:hover:text-dark-text-main dark:hover:bg-dark-bg-sidebar/70'
              } focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30`}
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
              id="graph-filters-panel"
              className={`${
                isMobile
                  ? 'absolute inset-y-0 left-0 w-[85vw] max-w-sm'
                  : 'w-72 flex-shrink-0'
              } border-r border-border-light bg-bg-sidebar/90 backdrop-blur-sm p-4 space-y-4 overflow-y-auto z-30 relative dark:bg-dark-bg-sidebar/90 dark:border-dark-border`}
            >
              {/* Clear all filters */}
              {hasActiveFilters && (
                <button
                  onClick={() => {
                    clearFilters();
                    setSelectedNode(null);
                  }}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm text-text-muted hover:text-text-main hover:bg-bg-hover rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30"
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
          <div className="flex-1 min-w-0 relative bg-bg-main/80 flex flex-row overflow-hidden dark:bg-dark-bg-main/80">
            {error && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 rounded-md bg-red-50/95 px-3 py-2 text-xs text-red-700 shadow">
                <span>{t('common.load_failed', 'Failed to load data')}</span>
                <button
                  type="button"
                  onClick={() => {
                    void Promise.all([loadData(), retryEdgeLoading()]);
                  }}
                  className="underline underline-offset-2"
                >
                  {t('common.retry')}
                </button>
              </div>
            )}
            <div className="flex-1 relative min-w-0 h-full">
              <Graph2D />
            </div>

            {/* Details panel - Sidebar style */}
            {selectedNode && (
              <RepoDetailsPanel node={selectedNode} onClose={handleCloseDetails} />
            )}

            {/* Loading overlay */}
            {!filteredData && !error && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-bg-main/80 backdrop-blur-sm dark:bg-dark-bg-main/80">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-action-primary border-t-transparent" />
                <p className="text-sm text-text-muted">{t('common.loading', 'Loading...')}</p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
};

export default GraphPage;
