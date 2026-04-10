import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSearchParams } from 'react-router-dom';
import { Filter, X } from 'lucide-react';

import { Sidebar } from '../components/layout/Sidebar';
import Graph2D from '../components/graph/Graph2D';
import Timeline from '../components/graph/Timeline';
import ClusterPanel from '../components/graph/ClusterPanel';
import StarListPanel from '../components/graph/StarListPanel';
import { SearchInput } from '../components/ui/SearchInput';
import { RepoDetailsPanel } from '../components/graph/RepoDetailsPanel';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { useGraph } from '../contexts/GraphContext';

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

  useEffect(() => {
    if (!rawData) return;
    if (lastAppliedSearchRef.current === searchSignature) return;
    lastAppliedSearchRef.current = searchSignature;

    if (urlNodeId) {
      const parsedNodeId = Number.parseInt(urlNodeId, 10);
      if (Number.isFinite(parsedNodeId)) {
        const node = rawData.nodes.find((item) => item.id === parsedNodeId);
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

  const handleCloseDetails = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
  }, [setSearchQuery]);

  const hasActiveFilters =
    filters.selectedClusters.size > 0 ||
    filters.selectedStarLists.size > 0 ||
    filters.searchQuery.trim() !== '' ||
    filters.timeRange !== null ||
    filters.minStars > 0 ||
    filters.languages.size > 0;

  return (
    <div className="flex h-screen overflow-hidden bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main">
      <Sidebar />

      <main className="flex min-w-0 flex-1 flex-col" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        <header className="sticky top-0 z-40 flex min-h-[4.5rem] items-center justify-between border-b border-border-light bg-bg-main/92 px-4 py-3 backdrop-blur-md sm:px-6 dark:border-dark-border dark:bg-dark-bg-main/92">
          <div className="flex items-center gap-3 select-none">
            <h2 className="text-base font-semibold tracking-tight text-text-main">{t('sidebar.graph')}</h2>
            <div className="mx-1 hidden h-4 w-px bg-border-light sm:block" />
            <span className="hidden text-sm text-text-muted sm:inline">
              {filteredData?.total_nodes !== undefined
                ? t('dashboard.subtitle', { count: filteredData.total_nodes })
                : t('dashboard.subtitle_infinite')}
            </span>
            {rawData && filteredData && rawData.total_nodes !== filteredData.total_nodes && (
              <span className="hidden text-xs text-text-dim sm:inline">
                / {rawData.total_nodes} {t('common.total')}
              </span>
            )}
            {edgesLoading && (
              <span className="hidden rounded-full bg-bg-sidebar/75 px-2.5 py-1 text-xs font-medium text-text-dim sm:inline dark:bg-dark-bg-sidebar/75 dark:text-dark-text-main/60">
                {t('sync.loading', 'Loading')} edges...
              </span>
            )}
            {error && (
              <button
                type="button"
                onClick={() => {
                  void Promise.all([loadData(), retryEdgeLoading()]);
                }}
                className="hidden text-xs text-red-600 hover:underline sm:inline"
              >
                {t('common.retry')}
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            <LanguageSwitch />

            <div className="w-40 transition-all sm:w-56 sm:focus-within:w-64">
              <SearchInput onSearch={handleSearch} value={filters.searchQuery} />
            </div>

            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              aria-expanded={showFilters}
              aria-controls="graph-filters-panel"
              aria-label={t('common.filter', 'Filters')}
              className={`header-action min-h-0 px-3 ${
                showFilters
                  ? 'border-border-light bg-bg-sidebar text-text-main dark:border-dark-border dark:bg-dark-bg-sidebar dark:text-dark-text-main'
                  : 'border-transparent bg-transparent text-text-muted shadow-none hover:bg-bg-hover dark:text-dark-text-main/70 dark:hover:bg-dark-bg-sidebar/70 dark:hover:text-dark-text-main'
              }`}
            >
              <Filter className="h-4 w-4" />
              {hasActiveFilters && <span className="h-2 w-2 rounded-full bg-action-primary" />}
            </button>
          </div>
        </header>

        <section className="relative flex flex-1 overflow-hidden">
          {isMobile && showFilters && (
            <button
              type="button"
              className="absolute inset-0 z-20 bg-slate-950/28 backdrop-blur-[1px]"
              onClick={() => setShowFilters(false)}
              aria-label={t('common.close')}
            />
          )}

          {showFilters && (
            <aside
              id="graph-filters-panel"
              className={`${
                isMobile ? 'absolute inset-y-0 left-0 w-[85vw] max-w-sm' : 'w-72 flex-shrink-0'
              } relative z-30 space-y-4 overflow-y-auto border-r border-border-light bg-bg-sidebar/92 p-4 backdrop-blur-md dark:border-dark-border dark:bg-dark-bg-sidebar/92`}
            >
              {hasActiveFilters && (
                <button
                  type="button"
                  onClick={() => {
                    clearFilters();
                    setSelectedNode(null);
                  }}
                  className="header-action-ghost w-full justify-center"
                >
                  <X className="h-4 w-4" />
                  {t('common.clear_all_filters')}
                </button>
              )}

              <StarListPanel
                collapsed={starListPanelCollapsed}
                onToggleCollapsed={() => setStarListPanelCollapsed(!starListPanelCollapsed)}
              />

              <ClusterPanel
                collapsed={clusterPanelCollapsed}
                onToggleCollapsed={() => setClusterPanelCollapsed(!clusterPanelCollapsed)}
              />

              <Timeline />
            </aside>
          )}

          <div className="relative flex min-w-0 flex-1 flex-row overflow-hidden bg-bg-main/80 dark:bg-dark-bg-main/80">
            {error && (
              <div className="panel-surface-strong absolute left-1/2 top-3 z-20 flex -translate-x-1/2 items-center gap-2 px-3 py-2 text-xs text-red-700">
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

            <div className="relative h-full min-w-0 flex-1">
              <Graph2D />
            </div>

            {selectedNode && <RepoDetailsPanel node={selectedNode} onClose={handleCloseDetails} />}

            {!filteredData && !error && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-bg-main/82 backdrop-blur-sm dark:bg-dark-bg-main/82">
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
