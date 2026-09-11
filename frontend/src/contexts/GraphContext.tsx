import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { GraphData, GraphNode, TimelineData } from '../types';
import { GraphSettingsState, useGraphStore } from '../stores/graphStore';
import {
  GRAPH_DATA_QUERY_KEY,
  useGraphDataQuery,
} from '../features/graph/hooks/useGraphDataQuery';
import {
  GRAPH_EDGES_QUERY_KEY,
  useGraphEdgesInfiniteQuery,
} from '../features/graph/hooks/useGraphEdgesInfiniteQuery';
import {
  GRAPH_NODES_QUERY_KEY,
  useGraphNodesInfiniteQuery,
} from '../features/graph/hooks/useGraphNodesInfiniteQuery';
import {
  TIMELINE_QUERY_KEY,
  useTimelineQuery,
} from '../features/graph/hooks/useTimelineQuery';
import {
  buildAdjacencyIndex,
  buildGraphEdgeIndex,
  buildGraphNodeSearchIndex,
  createVisibleNodeIds,
  filterVisibleClusters,
  filterVisibleEdges,
  filterVisibleNodes,
} from './graphFiltering';

interface GraphFilters {
  selectedClusters: Set<number>;
  selectedStarLists: Set<number>;
  searchQuery: string;
  timeRange: [number, number] | null;
  minStars: number;
  languages: Set<string>;
}

export type GraphSettings = GraphSettingsState;

interface GraphState {
  rawData: GraphData | null;
  timelineData: TimelineData | null;
  selectedNode: GraphNode | null;
  filters: GraphFilters;
  settings: GraphSettings;
  loading: boolean;
  edgesLoading: boolean;
  nodesLoading: boolean;
  syncing: boolean;
  syncStep: string;
  error: string | null;
}

interface GraphContextValue extends GraphState {
  filteredData: GraphData | null;
  adjacencyIndex: Map<number, Set<number>>;
  loadData: () => Promise<void>;
  refreshData: () => Promise<void>;
  setSelectedNode: (node: GraphNode | null) => void;
  setSearchQuery: (query: string) => void;
  toggleCluster: (clusterId: number) => void;
  setSelectedClusters: (clusterIds: number[]) => void;
  clearClusterFilter: () => void;
  toggleStarList: (listId: number) => void;
  setSelectedStarLists: (listIds: number[]) => void;
  clearStarListFilter: () => void;
  setTimeRange: (range: [number, number] | null) => void;
  setMinStars: (min: number) => void;
  setSelectedLanguages: (languages: string[]) => void;
  toggleLanguage: (language: string) => void;
  clearFilters: () => void;
  updateSettings: (settings: Partial<GraphSettings>) => void;
  setSyncing: (syncing: boolean) => void;
  setSyncStep: (step: string) => void;
  retryEdgeLoading: () => Promise<void>;
  loadMoreEdges: () => Promise<void>;
  canLoadMoreEdges: boolean;
  autoLoadHalted: boolean;
  loadedEdgePages: number;
  edgePageSize: number;
  retryNodeLoading: () => Promise<void>;
  loadMoreNodes: () => Promise<void>;
  canLoadMoreNodes: boolean;
  nodeAutoLoadHalted: boolean;
  loadedNodePages: number;
  nodePageSize: number;
}

const GraphContext = createContext<GraphContextValue | null>(null);

export const GraphProvider: React.FC<{ children: React.ReactNode; enabled?: boolean }> = ({
  children,
  enabled = true,
}) => {
  const queryClient = useQueryClient();
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [localError, setLocalError] = useState<string | null>(null);

  const selectedNode = useGraphStore((state) => state.selectedNode);
  const filters = useGraphStore((state) => state.filters);
  const settings = useGraphStore((state) => state.settings);
  const syncing = useGraphStore((state) => state.syncing);
  const syncStep = useGraphStore((state) => state.syncStep);

  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const updateSettings = useGraphStore((state) => state.updateSettings);
  const setSearchQuery = useGraphStore((state) => state.setSearchQuery);
  const toggleCluster = useGraphStore((state) => state.toggleCluster);
  const setSelectedClusters = useGraphStore((state) => state.setSelectedClusters);
  const clearClusterFilter = useGraphStore((state) => state.clearClusterFilter);
  const toggleStarList = useGraphStore((state) => state.toggleStarList);
  const setSelectedStarLists = useGraphStore((state) => state.setSelectedStarLists);
  const clearStarListFilter = useGraphStore((state) => state.clearStarListFilter);
  const setTimeRange = useGraphStore((state) => state.setTimeRange);
  const setMinStars = useGraphStore((state) => state.setMinStars);
  const setSelectedLanguages = useGraphStore((state) => state.setSelectedLanguages);
  const toggleLanguage = useGraphStore((state) => state.toggleLanguage);
  const clearFilters = useGraphStore((state) => state.clearFilters);
  const setSyncing = useGraphStore((state) => state.setSyncing);
  const setSyncStep = useGraphStore((state) => state.setSyncStep);

  const graphQuery = useGraphDataQuery(refreshNonce, enabled);
  const timelineQuery = useTimelineQuery(refreshNonce, enabled);
  const graphVersion = graphQuery.data?.version ?? 'active';
  const nodesQuery = useGraphNodesInfiniteQuery({
    version: graphVersion,
    refreshNonce,
    enabled: enabled && !!graphQuery.data,
  });
  const edgesQuery = useGraphEdgesInfiniteQuery({
    version: graphVersion,
    refreshNonce,
    enabled: enabled && !!graphQuery.data,
  });
  const stagedNodes = nodesQuery.stagedNodes;
  const stagedEdges = edgesQuery.stagedEdges;
  const rawData = useMemo(() => {
    const graphPayload = graphQuery.data;
    if (!graphPayload) return null;
    return {
      ...graphPayload,
      nodes: stagedNodes,
      edges: stagedEdges,
      total_nodes: graphPayload.total_nodes,
      total_edges: graphPayload.total_edges,
    };
  }, [graphQuery.data, stagedEdges, stagedNodes]);
  const timelineData = timelineQuery.data ?? null;
  const nodeFilterIndexes = useMemo(
    () => buildGraphNodeSearchIndex(stagedNodes),
    [stagedNodes]
  );
  const edgeFilterIndexes = useMemo(() => buildGraphEdgeIndex(stagedEdges), [stagedEdges]);
  // Built once here rather than inside each consumer of `useNodeNeighbors`.
  // Keyed on edges only: adjacency does not depend on node payloads.
  const adjacencyIndex = useMemo(() => buildAdjacencyIndex(stagedEdges), [stagedEdges]);
  const graphFilterIndexes = useMemo(
    () => ({ ...nodeFilterIndexes, ...edgeFilterIndexes }),
    [edgeFilterIndexes, nodeFilterIndexes]
  );
  const loading = enabled && (graphQuery.isLoading || timelineQuery.isLoading);
  const nodesLoading = Boolean(
    graphQuery.data && (nodesQuery.isLoading || nodesQuery.isFetchingNextPage)
  );
  const edgesLoading = Boolean(
    graphQuery.data && (edgesQuery.isLoading || edgesQuery.isFetchingNextPage)
  );
  const error = useMemo(() => {
    const queryError = graphQuery.error ?? timelineQuery.error;
    if (queryError) {
      return queryError instanceof Error ? queryError.message : 'Failed to load graph data';
    }
    return nodesQuery.nodesError ?? edgesQuery.edgesError ?? localError;
  }, [
    edgesQuery.edgesError,
    graphQuery.error,
    localError,
    nodesQuery.nodesError,
    timelineQuery.error,
  ]);

  const loadData = useCallback(async () => {
    try {
      setLocalError(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: [GRAPH_DATA_QUERY_KEY] }),
        queryClient.invalidateQueries({ queryKey: [TIMELINE_QUERY_KEY] }),
        queryClient.invalidateQueries({ queryKey: [GRAPH_EDGES_QUERY_KEY] }),
        queryClient.invalidateQueries({ queryKey: [GRAPH_NODES_QUERY_KEY] }),
      ]);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Failed to load data');
    }
  }, [queryClient]);

  const refreshData = useCallback(async () => {
    try {
      setLocalError(null);
      setRefreshNonce((current) => current + 1);
      await Promise.all([graphQuery.refetch(), timelineQuery.refetch()]);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Failed to refresh data');
    }
  }, [graphQuery, timelineQuery]);

  const visibleNodes = useMemo(
    () =>
      filterVisibleNodes({
        rawData,
        timelineData,
        filters,
        indexes: graphFilterIndexes,
      }),
    [filters, graphFilterIndexes, rawData, timelineData]
  );

  const visibleNodeIds = useMemo(() => createVisibleNodeIds(visibleNodes), [visibleNodes]);

  const visibleEdges = useMemo(
    () => {
      if (!rawData) return [];
      if (visibleNodeIds.size === rawData.nodes.length) {
        return rawData.edges;
      }
      return filterVisibleEdges(rawData.edges, visibleNodeIds, graphFilterIndexes);
    },
    [graphFilterIndexes, rawData, visibleNodeIds]
  );

  const visibleClusters = useMemo(
    () => (rawData ? filterVisibleClusters(rawData.clusters, visibleNodes) : []),
    [rawData, visibleNodes]
  );

  const filteredData = useMemo(() => {
    if (!rawData) return null;

    return {
      nodes: visibleNodes,
      edges: visibleEdges,
      clusters: visibleClusters,
      star_lists: rawData.star_lists || [],
      total_nodes: visibleNodes.length,
      total_edges: visibleEdges.length,
      total_clusters: visibleClusters.length,
      total_star_lists: rawData.star_lists?.length || 0,
      version: rawData.version,
      generated_at: rawData.generated_at,
      request_id: rawData.request_id,
    };
  }, [rawData, visibleClusters, visibleEdges, visibleNodes]);

  const retryEdgeLoading = edgesQuery.retryEdgeLoading;
  const loadMoreEdges = edgesQuery.loadMoreEdges;
  const retryNodeLoading = nodesQuery.retryNodeLoading;
  const loadMoreNodes = nodesQuery.loadMoreNodes;

  const value: GraphContextValue = useMemo(() => ({
    rawData,
    timelineData,
    selectedNode,
    filters,
    settings,
    loading,
    edgesLoading,
    nodesLoading,
    syncing,
    syncStep,
    error,
    filteredData,
    adjacencyIndex,
    loadData,
    refreshData,
    setSelectedNode,
    setSearchQuery,
    toggleCluster,
    setSelectedClusters,
    clearClusterFilter,
    toggleStarList,
    setSelectedStarLists,
    clearStarListFilter,
    setTimeRange,
    setMinStars,
    setSelectedLanguages,
    toggleLanguage,
    clearFilters,
    updateSettings,
    setSyncing,
    setSyncStep,
    retryEdgeLoading,
    loadMoreEdges,
    canLoadMoreEdges: edgesQuery.canLoadMoreEdges,
    autoLoadHalted: edgesQuery.autoLoadHalted,
    loadedEdgePages: edgesQuery.loadedPages,
    edgePageSize: edgesQuery.pageSize,
    retryNodeLoading,
    loadMoreNodes,
    canLoadMoreNodes: nodesQuery.canLoadMoreNodes,
    nodeAutoLoadHalted: nodesQuery.autoLoadHalted,
    loadedNodePages: nodesQuery.loadedPages,
    nodePageSize: nodesQuery.pageSize,
  }), [
    rawData, timelineData, selectedNode, filters, settings,
    loading, edgesLoading, nodesLoading, syncing, syncStep, error, filteredData,
    adjacencyIndex,
    loadData, refreshData,
    setSelectedNode, setSearchQuery, toggleCluster, setSelectedClusters,
    clearClusterFilter, toggleStarList, setSelectedStarLists,
    clearStarListFilter, setTimeRange, setMinStars, setSelectedLanguages, toggleLanguage,
    clearFilters, updateSettings, setSyncing, setSyncStep, retryEdgeLoading,
    loadMoreEdges, edgesQuery.canLoadMoreEdges, edgesQuery.autoLoadHalted,
    edgesQuery.loadedPages, edgesQuery.pageSize,
    retryNodeLoading, loadMoreNodes, nodesQuery.canLoadMoreNodes,
    nodesQuery.autoLoadHalted, nodesQuery.loadedPages, nodesQuery.pageSize,
  ]);

  return <GraphContext.Provider value={value}>{children}</GraphContext.Provider>;
};

// eslint-disable-next-line react-refresh/only-export-components
export const useGraph = (): GraphContextValue => {
  const context = useContext(GraphContext);
  if (!context) {
    throw new Error('useGraph must be used within a GraphProvider');
  }
  return context;
};

// eslint-disable-next-line react-refresh/only-export-components
export const useNodeNeighbors = (nodeId: number | undefined): Set<number> => {
  const { adjacencyIndex } = useGraph();

  return useMemo(() => {
    if (nodeId === undefined) {
      return new Set<number>();
    }
    return new Set(adjacencyIndex.get(nodeId) ?? []);
  }, [adjacencyIndex, nodeId]);
};
