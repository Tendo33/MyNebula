import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
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
  TIMELINE_QUERY_KEY,
  useTimelineQuery,
} from '../features/graph/hooks/useTimelineQuery';
import {
  buildGraphFilterIndexes,
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
  syncing: boolean;
  syncStep: string;
  error: string | null;
}

interface GraphContextValue extends GraphState {
  filteredData: GraphData | null;
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
  toggleLanguage: (language: string) => void;
  clearFilters: () => void;
  updateSettings: (settings: Partial<GraphSettings>) => void;
  setSyncing: (syncing: boolean) => void;
  setSyncStep: (step: string) => void;
  retryEdgeLoading: () => Promise<void>;
}

const GraphContext = createContext<GraphContextValue | null>(null);

export const GraphProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = useQueryClient();
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [error, setError] = useState<string | null>(null);

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
  const toggleLanguage = useGraphStore((state) => state.toggleLanguage);
  const clearFilters = useGraphStore((state) => state.clearFilters);
  const setSyncing = useGraphStore((state) => state.setSyncing);
  const setSyncStep = useGraphStore((state) => state.setSyncStep);

  const graphQuery = useGraphDataQuery(refreshNonce);
  const timelineQuery = useTimelineQuery(refreshNonce);
  const graphVersion = graphQuery.data?.version ?? 'active';
  const edgesQuery = useGraphEdgesInfiniteQuery({
    version: graphVersion,
    refreshNonce,
    enabled: !!graphQuery.data,
  });
  const stagedEdges = edgesQuery.stagedEdges;
  const rawData = useMemo(() => {
    const graphPayload = graphQuery.data;
    if (!graphPayload) return null;
    return {
      ...graphPayload,
      edges: stagedEdges,
      total_edges: graphPayload.total_edges,
    };
  }, [graphQuery.data, stagedEdges]);
  const timelineData = timelineQuery.data ?? null;
  const graphFilterIndexes = useMemo(() => buildGraphFilterIndexes(rawData), [rawData]);
  const loading = graphQuery.isLoading || timelineQuery.isLoading;
  const edgesLoading = Boolean(
    graphQuery.data &&
      (edgesQuery.isLoading || edgesQuery.isFetchingNextPage || edgesQuery.hasNextPage)
  );

  useEffect(() => {
    if (edgesQuery.edgesError) {
      setError(edgesQuery.edgesError);
    }
  }, [edgesQuery.edgesError]);

  useEffect(() => {
    const queryError = graphQuery.error ?? timelineQuery.error;
    if (!queryError) {
      setError((current) => (current === null ? current : null));
      return;
    }
    setError(queryError instanceof Error ? queryError.message : 'Failed to load graph data');
  }, [graphQuery.error, timelineQuery.error]);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: [GRAPH_DATA_QUERY_KEY] }),
        queryClient.invalidateQueries({ queryKey: [TIMELINE_QUERY_KEY] }),
        queryClient.invalidateQueries({ queryKey: [GRAPH_EDGES_QUERY_KEY] }),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    }
  }, [queryClient]);

  const refreshData = useCallback(async () => {
    try {
      setError(null);
      setRefreshNonce((current) => current + 1);
      await Promise.all([graphQuery.refetch(), timelineQuery.refetch()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
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
    () => (rawData ? filterVisibleEdges(rawData.edges, visibleNodeIds) : []),
    [rawData, visibleNodeIds]
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

  const value: GraphContextValue = useMemo(() => ({
    rawData,
    timelineData,
    selectedNode,
    filters,
    settings,
    loading,
    edgesLoading,
    syncing,
    syncStep,
    error,
    filteredData,
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
    toggleLanguage,
    clearFilters,
    updateSettings,
    setSyncing,
    setSyncStep,
    retryEdgeLoading,
  }), [
    rawData, timelineData, selectedNode, filters, settings,
    loading, edgesLoading, syncing, syncStep, error, filteredData,
    loadData, refreshData,
    setSelectedNode, setSearchQuery, toggleCluster, setSelectedClusters,
    clearClusterFilter, toggleStarList, setSelectedStarLists,
    clearStarListFilter, setTimeRange, setMinStars, toggleLanguage,
    clearFilters, updateSettings, setSyncing, setSyncStep, retryEdgeLoading,
  ]);

  return <GraphContext.Provider value={value}>{children}</GraphContext.Provider>;
};

export const useGraph = (): GraphContextValue => {
  const context = useContext(GraphContext);
  if (!context) {
    throw new Error('useGraph must be used within a GraphProvider');
  }
  return context;
};

export const useNodeNeighbors = (nodeId: number | undefined): Set<number> => {
  const { rawData } = useGraph();
  const adjacencyIndex = useMemo(() => {
    const index = new Map<number, Set<number>>();
    if (!rawData) {
      return index;
    }
    rawData.edges.forEach((edge) => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      if (!index.has(sourceId)) {
        index.set(sourceId, new Set());
      }
      if (!index.has(targetId)) {
        index.set(targetId, new Set());
      }
      index.get(sourceId)?.add(targetId);
      index.get(targetId)?.add(sourceId);
    });
    return index;
  }, [rawData]);

  return useMemo(() => {
    if (nodeId === undefined) {
      return new Set<number>();
    }
    return new Set(adjacencyIndex.get(nodeId) ?? []);
  }, [adjacencyIndex, nodeId]);
};
