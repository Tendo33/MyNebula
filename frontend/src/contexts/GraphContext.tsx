import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { ClusterInfo, GraphData, GraphNode, TimelineData } from '../types';
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

  const loadData = async () => {
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
  };

  const refreshData = async () => {
    try {
      setError(null);
      setRefreshNonce((current) => current + 1);
      await Promise.all([graphQuery.refetch(), timelineQuery.refetch()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    }
  };

  const filteredData = useMemo(() => {
    if (!rawData) return null;

    let filteredNodes = [...rawData.nodes];

    if (filters.selectedClusters.size > 0) {
      filteredNodes = filteredNodes.filter(
        (node) => node.cluster_id != null && filters.selectedClusters.has(node.cluster_id)
      );
    }

    if (filters.selectedStarLists.size > 0) {
      filteredNodes = filteredNodes.filter(
        (node) => node.star_list_id != null && filters.selectedStarLists.has(node.star_list_id)
      );
    }

    if (filters.searchQuery.trim()) {
      const query = filters.searchQuery.toLowerCase().trim();
      const starQueryMatch = query.match(/^stars:\s*>\s*(\d+)$/);
      if (starQueryMatch) {
        const minStars = Number.parseInt(starQueryMatch[1], 10);
        filteredNodes = filteredNodes.filter((node) => node.stargazers_count > minStars);
      } else {
        filteredNodes = filteredNodes.filter(
          (node) =>
            node.name.toLowerCase().includes(query) ||
            node.full_name.toLowerCase().includes(query) ||
            (node.description?.toLowerCase().includes(query) ?? false) ||
            (node.ai_summary?.toLowerCase().includes(query) ?? false) ||
            (node.language?.toLowerCase().includes(query) ?? false) ||
            (node.ai_tags?.some((tag) => tag.toLowerCase().includes(query)) ?? false) ||
            (node.topics?.some((topic) => topic.toLowerCase().includes(query)) ?? false)
        );
      }
    }

    if (filters.minStars > 0) {
      filteredNodes = filteredNodes.filter((node) => node.stargazers_count >= filters.minStars);
    }

    if (filters.languages.size > 0) {
      filteredNodes = filteredNodes.filter(
        (node) => node.language && filters.languages.has(node.language)
      );
    }

    if (filters.timeRange && timelineData && timelineData.points.length > 0) {
      const [startIdx, endIdx] = filters.timeRange;
      const startDate = timelineData.points[startIdx]?.date;
      const endDate = timelineData.points[endIdx]?.date;
      if (startDate && endDate) {
        filteredNodes = filteredNodes.filter((node) => {
          if (!node.starred_at) return true;
          const nodeDate = node.starred_at.substring(0, 7);
          return nodeDate >= startDate && nodeDate <= endDate;
        });
      }
    }

    const visibleNodeIds = new Set(filteredNodes.map((node) => node.id));
    const filteredEdges = rawData.edges.filter((edge) => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });
    const visibleClusterIds = new Set(
      filteredNodes.map((node) => node.cluster_id).filter((id): id is number => id != null)
    );
    const filteredClusters = rawData.clusters.filter((cluster) => visibleClusterIds.has(cluster.id));

    return {
      nodes: filteredNodes,
      edges: filteredEdges,
      clusters: filteredClusters,
      star_lists: rawData.star_lists || [],
      total_nodes: filteredNodes.length,
      total_edges: filteredEdges.length,
      total_clusters: filteredClusters.length,
      total_star_lists: rawData.star_lists?.length || 0,
      version: rawData.version,
      generated_at: rawData.generated_at,
      request_id: rawData.request_id,
    };
  }, [filters, rawData, timelineData]);

  const value: GraphContextValue = {
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
    retryEdgeLoading: edgesQuery.retryEdgeLoading,
  };

  return <GraphContext.Provider value={value}>{children}</GraphContext.Provider>;
};

export const useGraph = (): GraphContextValue => {
  const context = useContext(GraphContext);
  if (!context) {
    throw new Error('useGraph must be used within a GraphProvider');
  }
  return context;
};

export const useLanguages = (): string[] => {
  const { rawData } = useGraph();
  return useMemo(() => {
    if (!rawData) return [];
    const languageCounts = new Map<string, number>();
    rawData.nodes.forEach((node) => {
      if (node.language) {
        languageCounts.set(node.language, (languageCounts.get(node.language) || 0) + 1);
      }
    });
    return Array.from(languageCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([language]) => language);
  }, [rawData]);
};

export const useCluster = (clusterId: number | null): ClusterInfo | undefined => {
  const { rawData } = useGraph();
  return useMemo(() => {
    if (!rawData || clusterId == null) return undefined;
    return rawData.clusters.find((cluster) => cluster.id === clusterId);
  }, [rawData, clusterId]);
};

export const useNodeNeighbors = (nodeId: number | undefined): Set<number> => {
  const { rawData } = useGraph();
  return useMemo(() => {
    const neighbors = new Set<number>();
    if (!rawData || nodeId === undefined) return neighbors;
    rawData.edges.forEach((edge) => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      if (sourceId === nodeId) neighbors.add(targetId);
      if (targetId === nodeId) neighbors.add(sourceId);
    });
    return neighbors;
  }, [rawData, nodeId]);
};
