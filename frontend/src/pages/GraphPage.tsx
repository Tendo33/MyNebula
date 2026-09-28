import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSearchParams } from 'react-router-dom';
import { List, X } from 'lucide-react';
import type { PanelImperativeHandle } from 'react-resizable-panels';

import Graph2D from '../components/graph/Graph2D';
import Timeline from '../components/graph/Timeline';
import ClusterPanel from '../components/graph/ClusterPanel';
import StarListPanel from '../components/graph/StarListPanel';
import { SearchInput } from '../components/ui/SearchInput';
import { RepoDetailsPanel } from '../components/graph/RepoDetailsPanel';
import { useGraph } from '../contexts/GraphContext';
import { Alert, AlertAction, AlertDescription } from '../components/ui/alert';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '../components/ui/resizable';
import { ScrollArea } from '../components/ui/scroll-area';

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
    error,
    selectedNode,
    setSelectedNode,
    filters,
    setSelectedClusters,
    setSearchQuery,
    setSelectedLanguages,
    clearFilters,
    retryEdgeLoading,
    loadMoreEdges,
    canLoadMoreEdges,
    autoLoadHalted,
    loadedEdgePages,
    edgePageSize,
    retryNodeLoading,
    loadMoreNodes,
    canLoadMoreNodes,
    nodeAutoLoadHalted,
    loadedNodePages,
    nodePageSize,
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
        } else if (!node && selectedNode && selectedNode.id !== parsedNodeId) {
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
    if (!rawData) return;

    const parsedNodeId = urlNodeId ? Number.parseInt(urlNodeId, 10) : null;
    const requestedNode = Number.isFinite(parsedNodeId)
      ? rawData.nodes.find((item) => item.id === parsedNodeId)
      : null;
    const hasPendingNodeSync = urlNodeId
      ? requestedNode
        ? selectedNode?.id !== requestedNode.id
        : selectedNode !== null
      : selectedNode !== null;

    const requestedClusterIds = urlClusterIds
      ? urlClusterIds
          .split(',')
          .map((value) => Number.parseInt(value, 10))
          .filter((value) => Number.isFinite(value))
      : urlClusterId
        ? [Number.parseInt(urlClusterId, 10)].filter((value) => Number.isFinite(value))
        : [];
    const validClusterIds = requestedClusterIds
      .filter((clusterId) => rawData.clusters.some((cluster) => cluster.id === clusterId))
      .sort((left, right) => left - right);
    const currentClusterIds = Array.from(filters.selectedClusters).sort((left, right) => left - right);
    const hasPendingClusterSync = currentClusterIds.join(',') !== validClusterIds.join(',');

    const currentLanguages = Array.from(filters.languages);
    const hasPendingLanguageSync = urlLanguage
      ? currentLanguages.length !== 1 || currentLanguages[0] !== urlLanguage
      : currentLanguages.length > 0;

    const hasPendingSearchSync = filters.searchQuery !== urlQuery;

    if (hasPendingNodeSync || hasPendingClusterSync || hasPendingLanguageSync || hasPendingSearchSync) {
      return;
    }

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
  }, [
    filters.languages,
    filters.searchQuery,
    filters.selectedClusters,
    rawData,
    searchSignature,
    selectedNode,
    setSearchParams,
    urlClusterId,
    urlClusterIds,
    urlLanguage,
    urlNodeId,
    urlQuery,
  ]);

  const [clusterPanelCollapsed, setClusterPanelCollapsed] = useState(false);
  const [starListPanelCollapsed, setStarListPanelCollapsed] = useState(false);
  const graphPanelRef = useRef<PanelImperativeHandle | null>(null);
  const detailsPanelRef = useRef<PanelImperativeHandle | null>(null);
  const [showNodeList, setShowNodeList] = useState(() => {
    if (typeof window === 'undefined') {
      return false;
    }
    return (
      window.innerWidth < 1024 ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    );
  });

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
  const hasGraphData = (rawData?.nodes.length ?? 0) > 0;

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      if (selectedNode) {
        graphPanelRef.current?.resize('52%');
        detailsPanelRef.current?.resize('30%');
      } else {
        graphPanelRef.current?.resize('82%');
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [selectedNode]);

  return (
    <div className="relative min-h-0 flex-1">
      <ResizablePanelGroup orientation="horizontal" className="absolute inset-0">
        <ResizablePanel defaultSize="18%" minSize="12%" collapsible collapsedSize="0%">
          <aside
            id="graph-filters-panel"
            aria-label={t('common.filter')}
            className="flex h-full min-h-0 flex-col border-r bg-sidebar"
          >
            <div className="flex flex-col gap-3 border-b p-3">
              <SearchInput
                onSearch={handleSearch}
                value={filters.searchQuery}
                placeholder={t('graph.search_placeholder')}
              />
              {filteredData && filteredData.total_nodes > 0 ? (
                <p className="text-xs text-muted-foreground">
                  {t('graph.showing_repos', { count: filteredData.total_nodes })}
                </p>
              ) : null}
              {hasActiveFilters ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    clearFilters();
                    setSelectedNode(null);
                  }}
                >
                  <X data-icon="inline-start" />
                  {t('common.clear_all_filters')}
                </Button>
              ) : null}
            </div>
            <ScrollArea className="min-h-0 flex-1">
              <div className="flex flex-col gap-4 p-3">
                <StarListPanel
                  collapsed={starListPanelCollapsed}
                  onToggleCollapsed={() => setStarListPanelCollapsed(!starListPanelCollapsed)}
                />
                <ClusterPanel
                  collapsed={clusterPanelCollapsed}
                  onToggleCollapsed={() => setClusterPanelCollapsed(!clusterPanelCollapsed)}
                />
                <Timeline />
              </div>
            </ScrollArea>
          </aside>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel
          panelRef={graphPanelRef}
          defaultSize={selectedNode ? '52%' : '82%'}
          minSize="30%"
        >
          <div className="relative h-full min-h-0 bg-background">
            {showNodeList && hasGraphData && (
              <Card
                id="graph-accessible-node-list"
                aria-label={t('graph.repository_list', 'Repository list')}
                className="absolute top-3 right-3 bottom-3 z-40 flex w-[min(92%,24rem)] flex-col overflow-hidden py-0"
              >
                <div className="flex items-center justify-between border-b px-4 py-3">
                  <div>
                    <h2 className="text-sm font-semibold">{t('graph.repository_list', 'Repository list')}</h2>
                    <p className="text-xs text-muted-foreground">
                      {t('graph.repository_list_count', {
                        count: filteredData?.nodes.length ?? 0,
                        defaultValue: `${filteredData?.nodes.length ?? 0} matching repositories`,
                      })}
                    </p>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() => setShowNodeList(false)}
                    aria-label={t('common.close')}
                  >
                    <X />
                  </Button>
                </div>
                <ul className="min-h-0 flex-1 overflow-y-auto p-2">
                  {(filteredData?.nodes ?? []).slice(0, 50).map((node) => (
                    <li key={node.id}>
                      <button
                        type="button"
                        onClick={() => setSelectedNode(node)}
                        aria-current={selectedNode?.id === node.id ? 'true' : undefined}
                        className="w-full rounded-md px-3 py-2 text-left hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                      >
                        <span className="block truncate text-sm font-medium">{node.full_name}</span>
                        <span className="block truncate text-xs text-muted-foreground">
                          {node.description || node.ai_summary || t('common.no_description')}
                        </span>
                      </button>
                    </li>
                  ))}
                  {filteredData && filteredData.nodes.length > 50 && (
                    <li className="px-3 py-2 text-xs text-muted-foreground">
                      {t('graph.repository_list_limited')}
                    </li>
                  )}
                </ul>
              </Card>
            )}
            {!showNodeList && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setShowNodeList(true)}
                aria-expanded={showNodeList}
                aria-pressed={showNodeList}
                aria-controls="graph-accessible-node-list"
                disabled={!hasGraphData}
                className="absolute right-3 bottom-3 z-30"
              >
                <List data-icon="inline-start" />
                <span className="sr-only sm:not-sr-only">{t('graph.browse_as_list', 'Browse as list')}</span>
              </Button>
            )}
            {nodeAutoLoadHalted && (
              <Card className="absolute top-3 left-1/2 z-20 flex w-[min(92%,42rem)] -translate-x-1/2 flex-row items-center justify-between gap-3 px-4 py-3 text-xs text-muted-foreground">
                <span>
                  {t('graph.node_load_paused', {
                    pages: loadedNodePages,
                    nodes: loadedNodePages * nodePageSize,
                  })}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    void loadMoreNodes();
                  }}
                  disabled={!canLoadMoreNodes}
                  className="shrink-0"
                >
                  {t('graph.load_more_nodes')}
                </Button>
              </Card>
            )}
            {autoLoadHalted && !nodeAutoLoadHalted && (
              <Card className="absolute top-3 left-1/2 z-20 flex w-[min(92%,42rem)] -translate-x-1/2 flex-row items-center justify-between gap-3 px-4 py-3 text-xs text-muted-foreground">
                <span>
                  {t(
                    'graph.edge_load_paused',
                    {
                      pages: loadedEdgePages,
                      edges: loadedEdgePages * edgePageSize,
                      defaultValue: `Loaded ${loadedEdgePages} edge pages (${loadedEdgePages * edgePageSize} edge slots). Continue on demand.`,
                    }
                  )}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    void loadMoreEdges();
                  }}
                  disabled={!canLoadMoreEdges}
                  className="shrink-0"
                >
                  {t('graph.load_more_edges', 'Load more edges')}
                </Button>
              </Card>
            )}
            {error && (
              <Alert
                variant="destructive"
                className="absolute top-3 left-1/2 z-20 w-[min(92%,28rem)] -translate-x-1/2"
              >
                <AlertDescription>{t('common.load_failed_graph')}</AlertDescription>
                <AlertAction>
                  <Button
                    type="button"
                    size="sm"
                    onClick={() => {
                      void Promise.all([loadData(), retryNodeLoading(), retryEdgeLoading()]);
                    }}
                  >
                    {t('common.retry')}
                  </Button>
                </AlertAction>
              </Alert>
            )}
            <div className="h-full min-h-0">
              <Graph2D />
            </div>
          </div>
        </ResizablePanel>
        {selectedNode ? (
          <>
            <ResizableHandle withHandle />
            <ResizablePanel panelRef={detailsPanelRef} defaultSize="30%" minSize="20%">
              <section aria-label={t('common.overview')} className="h-full min-h-0">
                <RepoDetailsPanel node={selectedNode} onClose={handleCloseDetails} />
              </section>
            </ResizablePanel>
          </>
        ) : null}
      </ResizablePanelGroup>
    </div>
  );
};

export default GraphPage;
