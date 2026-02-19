import client from './client';
import { GraphData, GraphEdge, TimelineData } from '../types';

export interface GetGraphDataParams {
  /** Whether to include similarity edges between nodes */
  include_edges?: boolean;
  /** Minimum similarity threshold for edges (0.5-0.95) */
  min_similarity?: number;
}

export interface GetGraphEdgesParams {
  strategy?: 'knn';
  min_similarity?: number;
  k?: number;
  max_nodes?: number;
}

/**
 * Fetch graph data for visualization
 * @param params - Query parameters for graph data
 * @returns Complete graph data with nodes, edges, and clusters
 */
export const getGraphData = async (params?: GetGraphDataParams): Promise<GraphData> => {
  const response = await client.get<GraphData>('/graph', {
    params: {
      include_edges: params?.include_edges ?? false,
      min_similarity: params?.min_similarity ?? 0.7,
    },
  });
  return response.data;
};

/**
 * Fetch timeline data for visualization
 * @returns Timeline data grouped by month
 */
export const getTimelineData = async (): Promise<TimelineData> => {
  const response = await client.get<TimelineData>('/graph/timeline');
  return response.data;
};

/**
 * Fetch graph edges separately so nodes can render first.
 */
export const getGraphEdges = async (
  params?: GetGraphEdgesParams
): Promise<GraphEdge[]> => {
  const response = await client.get<GraphEdge[]>('/graph/edges', {
    params: {
      strategy: params?.strategy ?? 'knn',
      min_similarity: params?.min_similarity ?? 0.7,
      k: params?.k ?? 8,
      max_nodes: params?.max_nodes ?? 1000,
    },
  });
  return response.data;
};
