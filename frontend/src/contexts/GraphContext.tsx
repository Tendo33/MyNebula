import React, { createContext, useContext, useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { GraphData, GraphNode, ClusterInfo, TimelineData } from '../types';
import { getGraphData, getTimelineData } from '../api/graph';

// ============================================================================
// Cache Utilities
// ============================================================================

const CACHE_KEY_GRAPH = 'nebula_graph_data';
const CACHE_KEY_TIMELINE = 'nebula_timeline_data';
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}

const getCachedData = <T,>(key: string): T | null => {
  try {
    const stored = localStorage.getItem(key);
    if (stored) {
      const entry: CacheEntry<T> = JSON.parse(stored);
      if (Date.now() < entry.expiresAt) {
        return entry.data;
      }
      localStorage.removeItem(key);
    }
  } catch (e) {
    console.warn('Cache read error:', e);
  }
  return null;
};

const setCachedData = <T,>(key: string, data: T, ttl = CACHE_TTL) => {
  try {
    const entry: CacheEntry<T> = {
      data,
      expiresAt: Date.now() + ttl,
    };
    localStorage.setItem(key, JSON.stringify(entry));
  } catch (e) {
    console.warn('Cache write error:', e);
  }
};

const clearCache = () => {
  try {
    localStorage.removeItem(CACHE_KEY_GRAPH);
    localStorage.removeItem(CACHE_KEY_TIMELINE);
  } catch (e) {
    console.warn('Cache clear error:', e);
  }
};

// ============================================================================
// Types
// ============================================================================

interface GraphFilters {
  /** Selected cluster IDs to show (empty = show all) */
  selectedClusters: Set<number>;
  /** Selected star list IDs to show (empty = show all) */
  selectedStarLists: Set<number>;
  /** Search query for filtering nodes */
  searchQuery: string;
  /** Time range filter [startIndex, endIndex] into timeline points */
  timeRange: [number, number] | null;
  /** Minimum stars to show */
  minStars: number;
  /** Languages to filter by */
  languages: Set<string>;
}

interface GraphState {
  /** Raw data from API */
  rawData: GraphData | null;
  /** Timeline data from API */
  timelineData: TimelineData | null;
  /** Currently selected node for details panel */
  selectedNode: GraphNode | null;
  /** Hovered node (for cross-component highlighting) */
  hoveredNode: GraphNode | null;
  /** Current filters */
  filters: GraphFilters;
  /** Loading states */
  loading: boolean;
  syncing: boolean;
  syncStep: string;
  /** Error state */
  error: string | null;
}

interface GraphContextValue extends GraphState {
  // Computed data (filtered)
  filteredData: GraphData | null;

  // Actions
  loadData: () => Promise<void>;
  refreshData: () => Promise<void>;
  setSelectedNode: (node: GraphNode | null) => void;
  setHoveredNode: (node: GraphNode | null) => void;

  // Filter actions
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

  // Sync actions
  setSyncing: (syncing: boolean) => void;
  setSyncStep: (step: string) => void;
}

// ============================================================================
// Context
// ============================================================================

const defaultFilters: GraphFilters = {
  selectedClusters: new Set(),
  selectedStarLists: new Set(),
  searchQuery: '',
  timeRange: null,
  minStars: 0,
  languages: new Set(),
};

const GraphContext = createContext<GraphContextValue | null>(null);

// ============================================================================
// Provider
// ============================================================================

export const GraphProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // State
  const [rawData, setRawData] = useState<GraphData | null>(null);
  const [timelineData, setTimelineData] = useState<TimelineData | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [filters, setFilters] = useState<GraphFilters>(defaultFilters);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncStep, setSyncStep] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Load data from API with caching
  const loadData = useCallback(async () => {
    if (loading) return;

    try {
      setLoading(true);
      setError(null);

      // Try to load from cache first for instant display
      const cachedGraph = getCachedData<GraphData>(CACHE_KEY_GRAPH);
      const cachedTimeline = getCachedData<TimelineData>(CACHE_KEY_TIMELINE);

      if (cachedGraph && cachedTimeline) {
        // Use cached data immediately
        setRawData(cachedGraph);
        setTimelineData(cachedTimeline);
        setLoading(false);

        // Fetch fresh data in background
        Promise.all([
          getGraphData({ include_edges: true, min_similarity: 0.7 }),
          getTimelineData(),
        ]).then(([graphData, timeline]) => {
          setRawData(graphData);
          setTimelineData(timeline);
          setCachedData(CACHE_KEY_GRAPH, graphData);
          setCachedData(CACHE_KEY_TIMELINE, timeline);
        }).catch(err => {
          console.warn('Background refresh failed:', err);
        });

        return;
      }

      // Fetch fresh data
      const [graphData, timeline] = await Promise.all([
        getGraphData({ include_edges: true, min_similarity: 0.7 }),
        getTimelineData(),
      ]);

      setRawData(graphData);
      setTimelineData(timeline);

      // Cache the data
      setCachedData(CACHE_KEY_GRAPH, graphData);
      setCachedData(CACHE_KEY_TIMELINE, timeline);
    } catch (err) {
      console.error('Failed to load graph data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [loading]);

  // Refresh data (force reload, clear cache)
  const refreshData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Clear cache to force fresh fetch
      clearCache();

      const [graphData, timeline] = await Promise.all([
        getGraphData({ include_edges: true, min_similarity: 0.7 }),
        getTimelineData(),
      ]);

      setRawData(graphData);
      setTimelineData(timeline);

      // Update cache with fresh data
      setCachedData(CACHE_KEY_GRAPH, graphData);
      setCachedData(CACHE_KEY_TIMELINE, timeline);
    } catch (err) {
      console.error('Failed to refresh graph data:', err);
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setLoading(false);
    }
  }, []);

  // Filtered data (memoized)
  const filteredData = useMemo(() => {
    if (!rawData) return null;

    let filteredNodes = [...rawData.nodes];

    // Filter by clusters
    if (filters.selectedClusters.size > 0) {
      filteredNodes = filteredNodes.filter(
        node => node.cluster_id !== undefined && filters.selectedClusters.has(node.cluster_id)
      );
    }

    // Filter by star lists
    if (filters.selectedStarLists.size > 0) {
      filteredNodes = filteredNodes.filter(
        node => node.star_list_id !== undefined && filters.selectedStarLists.has(node.star_list_id)
      );
    }

    // Filter by search query
    if (filters.searchQuery.trim()) {
      const query = filters.searchQuery.toLowerCase().trim();
      filteredNodes = filteredNodes.filter(
        node =>
          node.name.toLowerCase().includes(query) ||
          node.full_name.toLowerCase().includes(query) ||
          (node.description?.toLowerCase().includes(query) ?? false) ||
          (node.language?.toLowerCase().includes(query) ?? false)
      );
    }

    // Filter by minimum stars
    if (filters.minStars > 0) {
      filteredNodes = filteredNodes.filter(node => node.stargazers_count >= filters.minStars);
    }

    // Filter by languages
    if (filters.languages.size > 0) {
      filteredNodes = filteredNodes.filter(
        node => node.language && filters.languages.has(node.language)
      );
    }

    // Filter by time range
    if (filters.timeRange && timelineData && timelineData.points.length > 0) {
      const [startIdx, endIdx] = filters.timeRange;
      const startDate = timelineData.points[startIdx]?.date;
      const endDate = timelineData.points[endIdx]?.date;

      if (startDate && endDate) {
        filteredNodes = filteredNodes.filter(node => {
          if (!node.starred_at) return true; // Include nodes without starred_at
          const nodeDate = node.starred_at.substring(0, 7); // YYYY-MM
          return nodeDate >= startDate && nodeDate <= endDate;
        });
      }
    }

    // Build set of visible node IDs for edge filtering
    const visibleNodeIds = new Set(filteredNodes.map(n => n.id));

    // Filter edges to only include those between visible nodes
    const filteredEdges = rawData.edges.filter(edge => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });

    // Filter clusters to only include those with visible nodes
    const visibleClusterIds = new Set(
      filteredNodes
        .map(n => n.cluster_id)
        .filter((id): id is number => id !== undefined)
    );
    const filteredClusters = rawData.clusters.filter(c => visibleClusterIds.has(c.id));

    return {
      nodes: filteredNodes,
      edges: filteredEdges,
      clusters: filteredClusters,
      star_lists: rawData.star_lists || [],
      total_nodes: filteredNodes.length,
      total_edges: filteredEdges.length,
      total_clusters: filteredClusters.length,
      total_star_lists: rawData.star_lists?.length || 0,
    };
  }, [rawData, timelineData, filters]);

  // Filter actions
  const setSearchQuery = useCallback((query: string) => {
    setFilters(prev => ({ ...prev, searchQuery: query }));
  }, []);

  const toggleCluster = useCallback((clusterId: number) => {
    setFilters(prev => {
      const newClusters = new Set(prev.selectedClusters);
      if (newClusters.has(clusterId)) {
        newClusters.delete(clusterId);
      } else {
        newClusters.add(clusterId);
      }
      return { ...prev, selectedClusters: newClusters };
    });
  }, []);

  const setSelectedClusters = useCallback((clusterIds: number[]) => {
    setFilters(prev => ({ ...prev, selectedClusters: new Set(clusterIds) }));
  }, []);

  const clearClusterFilter = useCallback(() => {
    setFilters(prev => ({ ...prev, selectedClusters: new Set() }));
  }, []);

  const toggleStarList = useCallback((listId: number) => {
    setFilters(prev => {
      const newLists = new Set(prev.selectedStarLists);
      if (newLists.has(listId)) {
        newLists.delete(listId);
      } else {
        newLists.add(listId);
      }
      return { ...prev, selectedStarLists: newLists };
    });
  }, []);

  const setSelectedStarLists = useCallback((listIds: number[]) => {
    setFilters(prev => ({ ...prev, selectedStarLists: new Set(listIds) }));
  }, []);

  const clearStarListFilter = useCallback(() => {
    setFilters(prev => ({ ...prev, selectedStarLists: new Set() }));
  }, []);

  const setTimeRange = useCallback((range: [number, number] | null) => {
    setFilters(prev => ({ ...prev, timeRange: range }));
  }, []);

  const setMinStars = useCallback((min: number) => {
    setFilters(prev => ({ ...prev, minStars: min }));
  }, []);

  const toggleLanguage = useCallback((language: string) => {
    setFilters(prev => {
      const newLanguages = new Set(prev.languages);
      if (newLanguages.has(language)) {
        newLanguages.delete(language);
      } else {
        newLanguages.add(language);
      }
      return { ...prev, languages: newLanguages };
    });
  }, []);

  const clearFilters = useCallback(() => {
    setFilters(defaultFilters);
  }, []);

  // Auto-load data on mount
  useEffect(() => {
    loadData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Context value
  const value: GraphContextValue = {
    // State
    rawData,
    timelineData,
    selectedNode,
    hoveredNode,
    filters,
    loading,
    syncing,
    syncStep,
    error,

    // Computed
    filteredData,

    // Actions
    loadData,
    refreshData,
    setSelectedNode,
    setHoveredNode,

    // Filter actions
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

    // Sync actions
    setSyncing,
    setSyncStep,
  };

  return <GraphContext.Provider value={value}>{children}</GraphContext.Provider>;
};

// ============================================================================
// Hook
// ============================================================================

export const useGraph = (): GraphContextValue => {
  const context = useContext(GraphContext);
  if (!context) {
    throw new Error('useGraph must be used within a GraphProvider');
  }
  return context;
};

// ============================================================================
// Utility hooks
// ============================================================================

/** Get all unique languages from the graph data */
export const useLanguages = (): string[] => {
  const { rawData } = useGraph();

  return useMemo(() => {
    if (!rawData) return [];

    const languageCounts = new Map<string, number>();
    rawData.nodes.forEach(node => {
      if (node.language) {
        languageCounts.set(node.language, (languageCounts.get(node.language) || 0) + 1);
      }
    });

    // Sort by count descending
    return Array.from(languageCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([lang]) => lang);
  }, [rawData]);
};

/** Get cluster by ID */
export const useCluster = (clusterId: number | undefined): ClusterInfo | undefined => {
  const { rawData } = useGraph();

  return useMemo(() => {
    if (!rawData || clusterId === undefined) return undefined;
    return rawData.clusters.find(c => c.id === clusterId);
  }, [rawData, clusterId]);
};

/** Get neighbors of a node */
export const useNodeNeighbors = (nodeId: number | undefined): Set<number> => {
  const { rawData } = useGraph();

  return useMemo(() => {
    const neighbors = new Set<number>();
    if (!rawData || nodeId === undefined) return neighbors;

    rawData.edges.forEach(edge => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;

      if (sourceId === nodeId) neighbors.add(targetId);
      if (targetId === nodeId) neighbors.add(sourceId);
    });

    return neighbors;
  }, [rawData, nodeId]);
};
